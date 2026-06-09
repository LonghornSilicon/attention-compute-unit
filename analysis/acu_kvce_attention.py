"""ACU x KVCE attention substitute for HuggingFace Qwen2.

Registered under the name ``acu_kvce`` in ``ALL_ATTENTION_FUNCTIONS``.
A single forward function dispatches on the module-level ``CURRENT_CONFIG``
so the harness can flip configs without rebuilding the model.

Modes:

  A          : baseline FP16 dense (fast path, no KVCE, no PC routing)
  B          : PC routing only (true K, V; per-tile FP16/INT8 SV)
  C          : KVCE only (K_hat, V_hat; FP16 SV)
  C_prenorm  : KVCE with per-vector L2 prenorm at the Q4.12 boundary
  E          : Integrated (K_hat, V_hat; PC-routed SV) - as-is, has C1
  E_prenorm  : Integrated with prenorm (isolates KVCE quantization noise
               from the C1 clipping defect)

Semantics: flash-attention-style streaming over K blocks. Online softmax
tracks (m, l, O) per query position. Per (q_block, k_block) tile, the
precision controller looks at the int8-quantized pre-softmax scores S
and decides FP16 vs INT8 for that tile's SV matmul. INT8 SV is
simulated with per-tile symmetric int8 quantization of un-normalized
exp(S - m_new) and V_block, an int32 accumulator, and per-tile rescale -
matching analysis/integration_test_kv_pc.py's INT8 path lifted into
flash-attention semantics.

GPU compute path with a CPU detour for KVCE round-trip (via
kvce_pool.kv_roundtrip).
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import numpy as np
import torch

from kvce_pool import kv_roundtrip


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
CURRENT_CONFIG: dict = {"mode": "A"}
# If None: KVCE applies on every layer (the C11 default behaviour).
# If a set of int layer indices: KVCE applies ONLY on those layers; on
# every other layer the substitute falls back to mode A (FP16 dense)
# regardless of CURRENT_CONFIG["mode"]. Used by the per-layer ablation
# in analysis/c11_per_layer_ablation.py.
KVCE_LAYERS: set | None = None
KVCE_REF_PATH: str = os.environ.get(
    "KVCE_REF", "/home/chaithu/lhs/kv-cache-engine/sw/reference_model"
)
TILE = 64
PC_THRESHOLD = 10  # max * N > 10 * sum -> FP16

# Per-layer / per-forward telemetry, reset by the harness.
CALL_STATS: dict = {
    "fwd_count": 0,
    "kvce_ms": 0.0,
    "attn_ms": 0.0,
    "pc_fp16_tiles": 0,
    "pc_total_tiles": 0,
}


def reset_stats() -> None:
    for k in CALL_STATS:
        CALL_STATS[k] = 0 if isinstance(CALL_STATS[k], int) else 0.0


def set_config(mode: str) -> None:
    if mode not in {"A", "B", "C", "C_prenorm", "E", "E_prenorm"}:
        raise ValueError(f"unknown mode {mode!r}")
    CURRENT_CONFIG["mode"] = mode


def set_kvce_layers(layers: set | list | None) -> None:
    """Restrict KVCE/PC routing to a layer subset (the rest fall back to
    config A's fast path). Pass None to apply on every layer."""
    global KVCE_LAYERS
    KVCE_LAYERS = None if layers is None else set(int(l) for l in layers)


# ---------------------------------------------------------------------------
# GQA repeat
# ---------------------------------------------------------------------------
def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, Hkv, N, D] -> [B, Hkv*n_rep, N, D]"""
    if n_rep == 1:
        return x
    B, H, N, D = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, H, n_rep, N, D)
        .reshape(B, H * n_rep, N, D)
    )


# ---------------------------------------------------------------------------
# KVCE round-trip with GPU<->CPU transfer
# ---------------------------------------------------------------------------
def _kvce_roundtrip_tensor(
    K: torch.Tensor, V: torch.Tensor, kvce_mode: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """K, V: [B, Hkv, N, D] fp16/bf16/fp32. Round-tripped through KVCE."""
    B, H, N, D = K.shape
    device, dtype = K.device, K.dtype
    K_cpu = K.detach().float().cpu().contiguous().view(-1, D).numpy()
    V_cpu = V.detach().float().cpu().contiguous().view(-1, D).numpy()
    K_hat, V_hat = kv_roundtrip(KVCE_REF_PATH, K_cpu, V_cpu, mode=kvce_mode)
    K_hat_t = torch.from_numpy(K_hat).view(B, H, N, D).to(device=device, dtype=dtype)
    V_hat_t = torch.from_numpy(V_hat).view(B, H, N, D).to(device=device, dtype=dtype)
    return K_hat_t, V_hat_t


# ---------------------------------------------------------------------------
# Attention substitute
# ---------------------------------------------------------------------------
def acu_kvce_attention(
    module,
    query: torch.Tensor,      # [B, Hq, N, D]
    key: torch.Tensor,        # [B, Hkv, N, D]
    value: torch.Tensor,      # [B, Hkv, N, D]
    attention_mask: Optional[torch.Tensor],   # [B, 1, N, N] additive or None
    scaling: float,
    dropout: float = 0.0,
    sliding_window: Optional[int] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    mode = CURRENT_CONFIG["mode"]
    base = mode.replace("_prenorm", "")
    kvce_mode = "prenorm" if mode.endswith("_prenorm") else "naive"

    # Per-layer gate: if KVCE_LAYERS restricts the set and this layer is
    # outside it, fall through to mode A (the dense FP16 fast path). The
    # PC/KVCE machinery still applies on the listed layers exactly as
    # configured. Used by the per-layer ablation.
    if KVCE_LAYERS is not None:
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is not None and int(layer_idx) not in KVCE_LAYERS:
            base = "A"

    CALL_STATS["fwd_count"] += 1
    t_attn0 = time.time()

    B, Hq, N, D = query.shape
    Hkv = key.shape[1]
    n_rep = Hq // Hkv
    orig_dtype = query.dtype
    device = query.device

    # HF passes attention_mask=None to custom attention impls for causal
    # decoder LMs (the equivalent of sdpa(is_causal=True)). We must build
    # the causal mask ourselves. Shape [1, 1, N, N], additive.
    causal_mask = torch.full(
        (N, N), float("-inf"), device=device, dtype=torch.float32
    ).triu(diagonal=1)[None, None, :, :]
    if attention_mask is not None:
        causal_mask = causal_mask + attention_mask.float()

    # ---- A: baseline FP16 dense attention via the standard path ----
    if base == "A":
        K_full = _repeat_kv(key, n_rep)
        V_full = _repeat_kv(value, n_rep)
        S = torch.matmul(query, K_full.transpose(-1, -2)) * scaling
        S = S + causal_mask.to(S.dtype)
        A = torch.softmax(S, dim=-1, dtype=torch.float32).to(orig_dtype)
        O = torch.matmul(A, V_full)
        CALL_STATS["attn_ms"] += (time.time() - t_attn0) * 1000.0
        return O.transpose(1, 2).contiguous(), None

    # ---- KVCE round-trip for C / E configs ----
    if base in ("C", "E"):
        t_kv0 = time.time()
        K_used, V_used = _kvce_roundtrip_tensor(key, value, kvce_mode=kvce_mode)
        CALL_STATS["kvce_ms"] += (time.time() - t_kv0) * 1000.0
    else:
        K_used, V_used = key, value

    K_full = _repeat_kv(K_used, n_rep)
    V_full = _repeat_kv(V_used, n_rep)
    use_pc = base in ("B", "E")

    # Promote to fp32 for the streaming loop (stability + the chip's INT8
    # accumulator is wider than fp16 anyway).
    Q_f = query.float()
    K_f = K_full.float()
    V_f = V_full.float()
    mask_f = causal_mask  # already fp32, includes the user-provided mask if any

    tile = TILE
    n_q_blocks = (N + tile - 1) // tile

    O = torch.zeros(B, Hq, N, D, device=device, dtype=torch.float32)
    m_state = torch.full((B, Hq, N), float("-inf"), device=device, dtype=torch.float32)
    l_state = torch.zeros(B, Hq, N, device=device, dtype=torch.float32)

    for qb in range(n_q_blocks):
        q_lo, q_hi = qb * tile, min((qb + 1) * tile, N)
        Q_blk = Q_f[:, :, q_lo:q_hi]   # [B, Hq, Bq, D]
        Bq = q_hi - q_lo

        # Causal: k ranges over [0, q_hi). Iterate over all full+partial
        # K blocks that can contribute.
        n_k_blocks_seen = (q_hi + tile - 1) // tile
        for kb in range(n_k_blocks_seen):
            k_lo, k_hi = kb * tile, min((kb + 1) * tile, N)
            # If the entire K block is strictly past the last query position
            # this q block sees, skip.
            if k_lo >= q_hi:
                break
            K_blk = K_f[:, :, k_lo:k_hi]   # [B, Hq, Bk, D]
            V_blk = V_f[:, :, k_lo:k_hi]
            Bk = k_hi - k_lo

            S = torch.matmul(Q_blk, K_blk.transpose(-1, -2)) * scaling  # [B, Hq, Bq, Bk]
            if mask_f is not None:
                S = S + mask_f[:, :, q_lo:q_hi, k_lo:k_hi]

            # PC decision per (B, Hq) tile, on int8-quantized S.
            if use_pc:
                S_abs_max = S.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-12)
                S_q = torch.round((S / S_abs_max) * 127.0).clamp(-127, 127)
                S_q_abs = S_q.abs()
                tile_max = S_q_abs.amax(dim=(-1, -2))    # [B, Hq]
                tile_sum = S_q_abs.sum(dim=(-1, -2))     # [B, Hq]
                tile_n = float(Bq * Bk)
                d_fp16 = (tile_max * tile_n) > (PC_THRESHOLD * tile_sum)
                CALL_STATS["pc_fp16_tiles"] += int(d_fp16.sum().item())
                CALL_STATS["pc_total_tiles"] += int(d_fp16.numel())
            else:
                d_fp16 = None

            # Online softmax across the row.
            m_block = S.amax(dim=-1)                     # [B, Hq, Bq]
            m_prev = m_state[:, :, q_lo:q_hi]
            m_new = torch.maximum(m_prev, m_block)
            P = torch.exp(S - m_new.unsqueeze(-1))       # un-normalized weights
            alpha = torch.exp(m_prev - m_new)            # rescale factor for previously accumulated state

            # SV matmul.
            if use_pc:
                # FP path
                O_fp = torch.matmul(P, V_blk)            # [B, Hq, Bq, D]
                # INT8 path: per-tile symmetric int8 quantization of P and V_blk.
                P_max = P.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-12)
                P_scale = P_max / 127.0
                P_q = torch.round(P / P_scale).clamp(-127, 127)
                V_max = V_blk.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-12)
                V_scale = V_max / 127.0
                V_q = torch.round(V_blk / V_scale).clamp(-127, 127)
                # int32 acc simulated in fp32 (per-tile scale).
                acc = torch.matmul(P_q, V_q)             # [B, Hq, Bq, D]
                combined_scale = (P_scale * V_scale).squeeze(-1).squeeze(-1)  # [B, Hq]
                O_int = acc * combined_scale[:, :, None, None]
                d_fp16_w = d_fp16[:, :, None, None].to(O_fp.dtype)
                O_block = O_fp * d_fp16_w + O_int * (1.0 - d_fp16_w)
            else:
                O_block = torch.matmul(P, V_blk)

            l_block = P.sum(dim=-1)                      # [B, Hq, Bq]
            l_state[:, :, q_lo:q_hi] = l_state[:, :, q_lo:q_hi] * alpha + l_block
            O[:, :, q_lo:q_hi] = O[:, :, q_lo:q_hi] * alpha.unsqueeze(-1) + O_block
            m_state[:, :, q_lo:q_hi] = m_new

    out = O / l_state.clamp(min=1e-12).unsqueeze(-1)
    out = out.to(orig_dtype).transpose(1, 2).contiguous()
    CALL_STATS["attn_ms"] += (time.time() - t_attn0) * 1000.0
    return out, None


def register(name: str = "acu_kvce") -> None:
    """Register the substitute under ALL_ATTENTION_FUNCTIONS so we can
    set ``model.config._attn_implementation = name``."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS.register(name, acu_kvce_attention)
