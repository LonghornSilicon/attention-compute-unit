"""Multiprocessing wrapper around the KVCE Python reference model.

The KVCE reference is per-vector and pure Python (~1.2 ms / vector on this
host). For end-to-end perplexity on Qwen2-0.5B we have 24 layers x 2
KV-heads x N tokens of compress+decompress per forward pass. A process
pool with 20 workers cuts the per-forward KV cost from ~30s to ~1.5s.

Workers are spawned (not forked) so they don't inherit CUDA state from
the parent. Each worker constructs its own KVCacheEngine(s) at init time.

Two execution modes per vector:

  * "naive"   - direct float -> Q4.12 cast at the boundary. Clips at +/-8.
                Matches what analysis/integration_test_kv_pc.py does.
                Exposes conflict C1 (raw |K| can reach ~152).
  * "prenorm" - per-vector scale so max(|v|) lands at +/-4.0 before the
                Q4.12 cast; restored at decompress. Eliminates C1.

Per-layer centroid tables (C10 work) are supported via the
`centroid_tables_path` arg to `kv_roundtrip` / `get_pool`. The JSON is
shaped {layer_idx: {"centroids": [...], "boundaries": [...]}}; layers
absent from the file fall back to the default Lloyd-Max-for-Gaussian
table. If `centroid_tables_path` is None (the C11 default), every
layer uses the chip's default table and behaviour is unchanged.

Capture mode: setting `capture` to a list-of-int layer ids in the call
enables the codec's rotation-capture hook on those layers. Captured
post-rotation coords come back from each call as a `{layer_idx: ndarray}`
dict. Used by c10_capture_per_layer.py to gather distributions for
Lloyd-Max retraining.

Both modes are bit-exact w.r.t. the KVCE reference; the only difference
is the boundary scaling.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

COORD_FRAC = 12
COORD_MAX_INT = (1 << 15) - 1
COORD_MIN_INT = -(1 << 15)
PRENORM_TARGET = 4.0  # leave headroom inside the +/-8.0 Q4.12 range

# DWB per-token tier (controller bit-class) -> KVCE codec realization.
# This is the software tier set used by the DWB-routed full test:
#   16 -> FP16 BYPASS (codec skipped, original fp value kept; lossless)
#    8 -> turbo8  (pq_bits=4 + QJL on keys)
#    4 -> turbo4  (pq_bits=3 + QJL on keys) -- the chip/RTL default
#    2 -> turbo   (pq_bits=2 + QJL on keys) -- most aggressive, software-only
# Hardware (FPGA) tiering restricts to {bypass, 8, 4} per the integration
# plan (§4): the 2-bit tier gives zero BRAM benefit. The map is monotone in
# stored bits, so the controller's {2,4,8,16} ordering is preserved.
TIER_PQBITS = {2: 2, 4: 3, 8: 4}
BYPASS_TIER = 16

# Env override so callers (or wrappers) can lower the pool size without
# editing every call site. Honoured by get_pool() / kv_roundtrip() when
# the explicit `workers` arg uses the default. Set to reduce UVM
# concurrency on GB10 (default 20 has caused page-fault crashes here).
_WORKERS_ENV = int(os.environ.get("KVCE_POOL_WORKERS", "20"))


# ---------------------------------------------------------------------------
# Worker-local KVCE engines (dict keyed by layer_idx; key -1 = default engine)
# ---------------------------------------------------------------------------
_engines: dict = {}
# Per-tier engines for per-token routing, keyed by pq_bits (default chip
# tables). Built lazily on first routed call. Distinct from _engines, which
# is keyed by layer_idx for the per-layer centroid-override path.
_tier_engines: dict = {}
_vector_dim: int | None = None
_KVCE_KLASS = None  # KVCacheEngine class (cached after first worker init)
_KVCE_INFO_KLASS = None


def _worker_init(
    kvce_ref_path: str,
    vector_dim: int,
    centroid_tables_json: Optional[str],
) -> None:
    global _engines, _tier_engines, _vector_dim, _KVCE_KLASS, _KVCE_INFO_KLASS
    sys.path.insert(0, kvce_ref_path)
    from kv_cache_engine_ref import KVCacheEngine, KVCacheEngineInfo  # noqa: E402
    _KVCE_KLASS = KVCacheEngine
    _KVCE_INFO_KLASS = KVCacheEngineInfo
    _vector_dim = vector_dim
    _tier_engines = {}

    # Default engine (key = -1) always present, used when a layer_idx is
    # not in the per-layer table.
    _engines = {-1: KVCacheEngine(KVCacheEngineInfo(vector_dim=vector_dim))}

    # Build per-layer engines if a table was provided.
    if centroid_tables_json:
        tables = json.loads(centroid_tables_json)
        for layer_str, entry in tables.items():
            L = int(layer_str)
            kw = dict(
                vector_dim=vector_dim,
                centroids=list(entry["centroids"]),
                boundaries=(list(entry["boundaries"])
                            if entry.get("boundaries") is not None else None),
            )
            # qjl_scale is optional; absent => codec falls back to
            # sqrt(pi/2) (the C11 default that regressed under retuned
            # centroids).
            if entry.get("qjl_scale") is not None:
                kw["qjl_scale"] = float(entry["qjl_scale"])
            # pq_bits is optional (Path C); absent => codec falls back to
            # chip default 3 (turbo4). When set to 4/5/6 the engine
            # auto-derives num_centroids = 1 << pq_bits and expects a
            # matching centroid table length.
            if entry.get("pq_bits") is not None:
                kw["pq_bits"] = int(entry["pq_bits"])
            info = KVCacheEngineInfo(**kw)
            _engines[L] = KVCacheEngine(info)


def _engine_for(layer_idx: int):
    """Return the engine for this layer; fall back to the default."""
    if layer_idx in _engines:
        return _engines[layer_idx]
    return _engines[-1]


def _tier_engine(pq_bits: int):
    """Return (build on first use) a default-table engine at this pq_bits, for
    per-token tier routing. Tier engines use the chip default centroid table;
    per-layer centroid overrides are not (yet) composed with per-token tiers."""
    eng = _tier_engines.get(pq_bits)
    if eng is None:
        eng = _KVCE_KLASS(_KVCE_INFO_KLASS(vector_dim=_vector_dim,
                                           pq_bits=pq_bits))
        _tier_engines[pq_bits] = eng
    return eng


def _quantize_q412(v: np.ndarray, mode: str):
    """Float vector -> (int Q4.12 list, inv_scale) at the KVCE boundary.
    Returns (None, None) for a ~zero vector (caller emits zeros)."""
    if mode == "naive":
        q = (np.round(v * (1 << COORD_FRAC))
               .clip(COORD_MIN_INT, COORD_MAX_INT).astype(np.int32).tolist())
        return q, 1.0 / (1 << COORD_FRAC)
    mx = float(np.abs(v).max())
    if mx < 1e-12:
        return None, None
    s = PRENORM_TARGET / mx
    q = (np.round(v * s * (1 << COORD_FRAC))
           .clip(COORD_MIN_INT, COORD_MAX_INT).astype(np.int32).tolist())
    return q, 1.0 / (s * (1 << COORD_FRAC))


def _roundtrip_chunk(
    payload: Tuple[str, str, int, bool, np.ndarray, Optional[np.ndarray]]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Round-trip a contiguous chunk of vectors through KVCE.

    payload = (kind, mode, layer_idx, capture, chunk, bits)
      kind       = "K" or "V"
      mode       = "naive" or "prenorm"
      layer_idx  = int; -1 picks the default engine
      capture    = bool; if True, also return the post-rotation coords
                   for these vectors (in coordinate-frame floats)
      chunk      = float32 array, shape [n_vectors, vector_dim]
      bits       = optional int array of DWB tier labels {2,4,8,16}, one per
                   vector (aligned to `chunk` rows). None => uniform routing
                   through the layer engine (the legacy per-layer path,
                   bit-exact w.r.t. before this change). When given, each
                   vector is dispatched per-token:
                     16 -> FP16 BYPASS (codec skipped; original fp kept)
                      8 -> turbo8 (pq_bits=4)   4 -> turbo4 (pq_bits=3)
                      2 -> turbo  (pq_bits=2)

    Returns (out, captured):
      out       = float32 reconstructed values, shape [n_vectors, vector_dim]
      captured  = float32 post-rotation coords, shape [n_vectors, vector_dim]
                  when capture else None (capture requires bits is None)
    """
    kind, mode, layer_idx, capture, chunk, bits = payload
    n, d = chunk.shape
    out = np.empty_like(chunk)

    # ---- Legacy uniform path (bits is None): single engine, capture OK ----
    if bits is None:
        engine = _engine_for(layer_idx)
        if capture:
            engine.enable_rotation_capture()
        for i in range(n):
            q, inv_scale = _quantize_q412(chunk[i], mode)
            if q is None:
                out[i] = 0.0
                continue
            if kind == "K":
                qhat = engine.decompress_key(engine.compress_key(q))
            else:
                qhat = engine.decompress_value(engine.compress_value(q))
            out[i] = np.asarray(qhat, dtype=np.float32) * inv_scale
        captured = None
        if capture:
            raw = engine.pop_rotation_capture()
            if raw:
                captured = (np.asarray(raw, dtype=np.float32)
                            * (1.0 / (1 << COORD_FRAC)))
        return out, captured

    # ---- Per-token routed path: dispatch each vector by its DWB tier ----
    for i in range(n):
        v = chunk[i]
        tier = int(bits[i])
        if tier == BYPASS_TIER:
            # FP16 bypass: keep the original fp value, skip Q4.12 + codec.
            out[i] = v
            continue
        engine = _tier_engine(TIER_PQBITS[tier])
        q, inv_scale = _quantize_q412(v, mode)
        if q is None:
            out[i] = 0.0
            continue
        if kind == "K":
            qhat = engine.decompress_key(engine.compress_key(q))
        else:
            qhat = engine.decompress_value(engine.compress_value(q))
        out[i] = np.asarray(qhat, dtype=np.float32) * inv_scale
    return out, None


# ---------------------------------------------------------------------------
# Pool singleton
# ---------------------------------------------------------------------------
_pool: ProcessPoolExecutor | None = None
_pool_table_signature: str | None = None  # to detect a table change


def get_pool(
    kvce_ref_path: str,
    vector_dim: int = 64,
    workers: int = -1,
    centroid_tables_path: Optional[str] = None,
) -> ProcessPoolExecutor:
    if workers == -1:
        workers = _WORKERS_ENV
    """Get the pool, creating (or recreating) it if the centroid table changes."""
    global _pool, _pool_table_signature

    # Read the centroid table JSON once in the parent so workers don't all hit
    # the file. Empty string -> "no per-layer overrides".
    tables_json = ""
    if centroid_tables_path:
        with open(centroid_tables_path) as f:
            tables_json = f.read()

    signature = f"{kvce_ref_path}|{vector_dim}|{len(tables_json)}"
    if _pool is not None and signature != _pool_table_signature:
        shutdown_pool()

    if _pool is None:
        ctx = mp.get_context("spawn")
        _pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(kvce_ref_path, vector_dim,
                      tables_json if tables_json else None),
        )
        _pool_table_signature = signature
    return _pool


def shutdown_pool() -> None:
    global _pool, _pool_table_signature
    if _pool is not None:
        _pool.shutdown(cancel_futures=True)
        _pool = None
        _pool_table_signature = None


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def kv_roundtrip(
    kvce_ref_path: str,
    K: np.ndarray,
    V: np.ndarray,
    mode: str = "naive",
    workers: int = -1,
    layer_idx: int = -1,
    centroid_tables_path: Optional[str] = None,
    capture: bool = False,
    bits: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """Round-trip K and V through KVCE.

    K, V: float32 arrays of shape [n_vectors, vector_dim].
    mode: "naive" or "prenorm".
    layer_idx: which per-layer engine to use; -1 = chip default.
    centroid_tables_path: optional path to per-layer centroid JSON. Only
        consulted at pool-init time; later calls reuse the same pool.
    capture: if True, also return post-rotation coords for K and V.
    bits: optional int array of DWB per-token tier labels {2,4,8,16}, one per
        row of K/V (same length as K.shape[0]). None => uniform per-layer
        routing (legacy, bit-exact). Per-token routing and capture are
        mutually exclusive.

    Returns (K_hat, V_hat, captured):
        K_hat, V_hat: same shape as input.
        captured: when capture=True, a dict {"K": ndarray, "V": ndarray}
                  of post-rotation coords (one row per input vector).
                  When capture=False, None.
    """
    assert K.ndim == 2 and V.ndim == 2 and K.shape[1] == V.shape[1]
    if mode not in ("naive", "prenorm"):
        raise ValueError(f"unknown mode {mode!r}")
    if bits is not None:
        if capture:
            raise ValueError("capture and per-token bits are mutually exclusive")
        bits = np.asarray(bits)
        assert bits.shape[0] == K.shape[0] == V.shape[0], \
            f"bits length {bits.shape[0]} must match K/V rows {K.shape[0]}"
    if workers == -1:
        workers = _WORKERS_ENV
    pool = get_pool(
        kvce_ref_path,
        vector_dim=K.shape[1],
        workers=workers,
        centroid_tables_path=centroid_tables_path,
    )

    def chunk_payloads(kind: str, arr: np.ndarray) -> List[Tuple]:
        n = arr.shape[0]
        if n == 0:
            return []
        n_chunks = min(workers, n)
        bounds = np.linspace(0, n, n_chunks + 1, dtype=int)
        return [(kind, mode, layer_idx, capture,
                 arr[bounds[i]:bounds[i + 1]],
                 None if bits is None else bits[bounds[i]:bounds[i + 1]])
                for i in range(n_chunks) if bounds[i + 1] > bounds[i]]

    k_payloads = chunk_payloads("K", K)
    v_payloads = chunk_payloads("V", V)
    payloads = k_payloads + v_payloads
    results = list(pool.map(_roundtrip_chunk, payloads))

    n_k = len(k_payloads)
    if n_k:
        K_hat = np.concatenate([r[0] for r in results[:n_k]], axis=0)
    else:
        K_hat = K.copy()
    if (len(results) - n_k):
        V_hat = np.concatenate([r[0] for r in results[n_k:]], axis=0)
    else:
        V_hat = V.copy()

    cap_out: Optional[Dict[str, np.ndarray]] = None
    if capture:
        k_caps = [r[1] for r in results[:n_k] if r[1] is not None]
        v_caps = [r[1] for r in results[n_k:] if r[1] is not None]
        cap_out = {
            "K": (np.concatenate(k_caps, axis=0) if k_caps
                  else np.zeros((0, K.shape[1]), dtype=np.float32)),
            "V": (np.concatenate(v_caps, axis=0) if v_caps
                  else np.zeros((0, V.shape[1]), dtype=np.float32)),
        }
    return K_hat, V_hat, cap_out
