"""ChannelQuant + APA end-to-end on Qwen (the combined-system bridge).

Reuses two *already-validated* pieces, no codec reimplementation:
  - KVCE half  : channelquant_ref fake-quant hooks on k_proj/v_proj (== c23:
                 per-channel INT4 keys grouped G=128, per-token INT4 values,
                 CQ-4+ adds the static-mask fp16 outlier lane).
  - APA half   : acu_kvce_attention mode B = flash-attention streaming with the
                 precision-controller ratio test picking INT8 vs FP16 per S·V tile.

Configs (2x3 = {APA off/on} x {no-KVCE, CQ-4, CQ-4+}), HellaSwag acc_norm vs FP16:
  fp16          A , hooks=none      apa           B , hooks=none
  cq4           A , hooks=cq4       cq4+apa       B , hooks=cq4
  cq4plus       A , hooks=cq4plus   cq4plus+apa   B , hooks=cq4plus
acu mode A = dense FP16 SV; mode B = APA PC-routed INT8/FP16 SV (never calls the
TurboQuant KVCE round-trip, so no GB10 hang path). Output JSON -> CWD.

  HF_HOME=<cache> KVCE_REF=<kvce ref> python cq_apa_e2e.py --model Qwen/Qwen2-0.5B \
      --device cuda --tag q05 --n-items 500
"""
import sys, os, json, time, argparse, math
import numpy as np, torch, torch.nn.functional as F

APA_DIR = "/home/chaithu/lhs/adaptive-precision-attention/analysis"
CQ_REF  = "/home/chaithu/lhs/channelquant/reference"
CQ_ROOT = "/home/chaithu/lhs/channelquant"
sys.path.insert(0, APA_DIR); sys.path.insert(0, CQ_REF)
import acu_kvce_attention as acu       # reuse its config/stats globals + _repeat_kv
import channelquant_ref as cq          # ChannelQuant fake-quant

G_KNEE = 128


# ---------------------------------------------------------------------------
# Mask-correct APA attention.  Same flash-streaming + per-tile INT8/FP16 S·V as
# acu_kvce mode B, but the precision-controller ratio test is computed over the
# VALID (unmasked) scores only.  acu_kvce's version feeds the causal -inf into
# S_abs_max -> inf -> nan int8 scores -> the test always falls through to INT8
# (int8_tiles==1.0); the OUTPUT is still correct (masked P=0) but the routing
# fraction is meaningless.  Fixing it gives APA's true INT8/FP16 split on Qwen.
# mode A = dense FP16 baseline; mode B = APA PC-routed SV.  KVCE is applied
# upstream by the channelquant hooks, so this never does a codec round-trip.
# ---------------------------------------------------------------------------
def cq_apa_attention(module, query, key, value, attention_mask, scaling,
                     dropout=0.0, sliding_window=None, **kwargs):
    import torch
    mode = acu.CURRENT_CONFIG["mode"]; base = mode.replace("_prenorm", "")
    acu.CALL_STATS["fwd_count"] += 1
    B, Hq, N, D = query.shape
    Hkv = key.shape[1]; n_rep = Hq // Hkv
    orig_dtype = query.dtype; device = query.device
    causal = torch.full((N, N), float("-inf"), device=device, dtype=torch.float32).triu(1)[None, None]
    if attention_mask is not None:
        causal = causal + attention_mask.float()

    K_full = acu._repeat_kv(key, n_rep); V_full = acu._repeat_kv(value, n_rep)
    if base == "A":
        S = torch.matmul(query, K_full.transpose(-1, -2)) * scaling + causal.to(query.dtype)
        A = torch.softmax(S, dim=-1, dtype=torch.float32).to(orig_dtype)
        return torch.matmul(A, V_full).transpose(1, 2).contiguous(), None

    Q_f, K_f, V_f = query.float(), K_full.float(), V_full.float()
    tile = acu.TILE; PCT = acu.PC_THRESHOLD
    O = torch.zeros(B, Hq, N, D, device=device, dtype=torch.float32)
    m_state = torch.full((B, Hq, N), float("-inf"), device=device, dtype=torch.float32)
    l_state = torch.zeros(B, Hq, N, device=device, dtype=torch.float32)
    for qb in range((N + tile - 1)//tile):
        q_lo, q_hi = qb*tile, min((qb+1)*tile, N); Bq = q_hi - q_lo
        Q_blk = Q_f[:, :, q_lo:q_hi]
        for kb in range((q_hi + tile - 1)//tile):
            k_lo, k_hi = kb*tile, min((kb+1)*tile, N)
            if k_lo >= q_hi: break
            K_blk = K_f[:, :, k_lo:k_hi]; V_blk = V_f[:, :, k_lo:k_hi]
            S = torch.matmul(Q_blk, K_blk.transpose(-1, -2)) * scaling + causal[:, :, q_lo:q_hi, k_lo:k_hi]
            # ---- PC decision over VALID scores only ----
            finite = torch.isfinite(S)
            Sv = torch.where(finite, S, torch.zeros_like(S))
            s_max = Sv.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-12)
            S_q = torch.where(finite, torch.round((Sv/s_max)*127.0).clamp(-127, 127), torch.zeros_like(S))
            s_abs = S_q.abs()
            tile_max = s_abs.amax(dim=(-1, -2))
            tile_sum = s_abs.sum(dim=(-1, -2))
            tile_n = finite.sum(dim=(-1, -2)).float().clamp(min=1)     # valid-position count
            d_fp16 = (tile_max * tile_n) > (PCT * tile_sum)
            acu.CALL_STATS["pc_fp16_tiles"] += int(d_fp16.sum().item())
            acu.CALL_STATS["pc_total_tiles"] += int(d_fp16.numel())
            # ---- online softmax ----
            m_block = S.amax(dim=-1); m_prev = m_state[:, :, q_lo:q_hi]
            m_new = torch.maximum(m_prev, m_block)
            P = torch.exp(S - m_new.unsqueeze(-1)); alpha = torch.exp(m_prev - m_new)
            # ---- per-tile INT8/FP16 S·V ----
            O_fp = torch.matmul(P, V_blk)
            P_s = (P.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-12))/127.0
            P_q = torch.round(P/P_s).clamp(-127, 127)
            V_s = (V_blk.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-12))/127.0
            V_q = torch.round(V_blk/V_s).clamp(-127, 127)
            O_int = torch.matmul(P_q, V_q) * (P_s*V_s)     # [B,Hq,Bq,D]*[B,Hq,1,1]
            w = d_fp16[:, :, None, None].to(O_fp.dtype)
            O_block = O_fp*w + O_int*(1.0 - w)
            l_block = P.sum(dim=-1)
            l_state[:, :, q_lo:q_hi] = l_state[:, :, q_lo:q_hi]*alpha + l_block
            O[:, :, q_lo:q_hi] = O[:, :, q_lo:q_hi]*alpha.unsqueeze(-1) + O_block
            m_state[:, :, q_lo:q_hi] = m_new
    out = (O / l_state.clamp(min=1e-12).unsqueeze(-1)).to(orig_dtype).transpose(1, 2).contiguous()
    return out, None


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n; c = p + z*z/(2*n)
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (round((c-h)/d, 4), round((c+h)/d, 4))


def load_mask(tag, k, n_kv, D):
    f = os.path.join(CQ_ROOT, "reference", "masks", f"{tag}_k{k}.npz")
    if not os.path.exists(f): return None
    d = np.load(f); idx = d["outlier_idx"]           # [L, n_kv, k]
    out = []
    for l in range(idx.shape[0]):
        feats = []
        for h in range(n_kv):
            feats.extend((h*D + idx[l, h]).tolist())
        out.append(torch.tensor(sorted(feats), dtype=torch.long))
    return out


def make_hooks(model, variant, G, layer_masks):
    """== c23 make_hooks: per-channel INT4 keys (+cq4+ static outlier lane),
    per-token INT4 values. variant None -> no KVCE."""
    if variant is None: return []
    Gv = None if G == "full" else G
    hooks = []
    kmods = [(n, m) for n, m in model.named_modules() if n.endswith(".k_proj")]
    for l, (name, m) in enumerate(kmods):
        if variant == "cq4":
            fn = (lambda mod, inp, out, gg=Gv: cq.fq_per_channel(out, 4, gg))
        else:
            oi = None if layer_masks is None else layer_masks[l].to(next(m.parameters()).device)
            fn = (lambda mod, inp, out, gg=Gv, ix=oi: cq.fq_per_channel_outlier(out, 4, 2, gg, ix))
        hooks.append(m.register_forward_hook(fn))
    for name, m in model.named_modules():
        if name.endswith(".v_proj"):
            hooks.append(m.register_forward_hook(lambda mod, inp, out: cq.fq_per_token(out, 4)))
    return hooks


@torch.no_grad()
def score_config(model, tok, items, acu_mode, variant, G, layer_masks):
    acu.set_config(acu_mode); acu.reset_stats()
    correct = np.zeros(len(items), dtype=np.int64)
    for i, it in enumerate(items):
        nlls, nts = [], []
        for ch in it["choices"]:
            end = tok(ch, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            if end.numel() == 0:
                nlls.append(1e9); nts.append(1); continue
            full = torch.cat([it["ctx_ids"], end]).unsqueeze(0).to(model.device)
            hooks = make_hooks(model, variant, G, layer_masks)
            try:
                logits = model(full, use_cache=False).logits[0]
            finally:
                for h in hooks: h.remove()
            cl = it["ctx_ids"].shape[0]
            lp = F.log_softmax(logits[cl-1:cl-1+end.shape[0]], dim=-1)
            nlls.append(-lp[range(end.shape[0]), end].sum().item()); nts.append(int(end.shape[0]))
        nlls = np.array(nlls); nts = np.array(nts, float)
        correct[i] = int((nlls/np.maximum(nts, 1)).argmin() == it["label"])
    st = dict(acu.CALL_STATS)
    return correct, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B")
    ap.add_argument("--n-items", type=int, default=500)
    ap.add_argument("--max-ctx-tokens", type=int, default=160)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="q05")
    ap.add_argument("--out", default="cq_apa_e2e.json")
    args = ap.parse_args()
    dtype = torch.float16 if args.device == "cuda" else torch.float32

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS.register("cq_apa", cq_apa_attention)   # mask-correct PC
    model.config._attn_implementation = "cq_apa"
    cfg = model.config
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    D = cfg.hidden_size // cfg.num_attention_heads
    layer_masks = load_mask(args.tag, 2, n_kv, D)
    print(f"[setup] {args.model}: D={D}, n_kv={n_kv}, G={G_KNEE}, mask={'loaded' if layer_masks else 'MISSING'}", flush=True)

    ds = load_dataset("Rowan/hellaswag", split="validation")
    items = []
    for ex in ds:
        ctx = (ex["ctx_a"].strip() + " " + ex["ctx_b"].strip()).strip()
        try: label = int(ex["label"])
        except (TypeError, ValueError): continue
        ids = tok(ctx, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if ids.shape[0] > args.max_ctx_tokens: continue
        items.append({"ctx_ids": ids, "choices": [" " + e.strip() for e in ex["endings"]], "label": label})
        if len(items) >= args.n_items: break
    n = len(items)

    configs = [
        ("fp16",         "A", None),       # baseline
        ("cq4",          "A", "cq4"),      # KVCE only  (== c23)
        ("cq4plus",      "A", "cq4plus"),  # KVCE only  (== c23)
        ("apa",          "B", None),       # APA only
        ("cq4_apa",      "B", "cq4"),      # combined
        ("cq4plus_apa",  "B", "cq4plus"),  # combined
    ]
    out = {"model": args.model, "task": "hellaswag", "n_items": n, "D": D,
           "n_kv_heads": n_kv, "G": G_KNEE, "PC_THRESHOLD": acu.PC_THRESHOLD,
           "note": "KVCE=channelquant hooks; APA=acu_kvce mode B PC-routed SV", "configs": {}}
    vecs = {}
    for name, amode, variant in configs:
        t0 = time.time()
        c, st = score_config(model, tok, items, amode, variant, G_KNEE, layer_masks)
        vecs[name] = c; k = int(c.sum())
        tot = st.get("pc_total_tiles", 0); fp = st.get("pc_fp16_tiles", 0)
        int8_frac = round(1 - fp/tot, 4) if tot else None
        out["configs"][name] = {"acc_norm": round(k/n, 4), "acc_norm_ci95": wilson(k, n),
                                "n_correct": k, "apa": amode == "B", "kvce": variant,
                                "int8_tile_frac": int8_frac}
        print(f"  {name:14s} acc_norm={k/n:.4f} {wilson(k,n)}  "
              f"int8_tiles={int8_frac}  ({time.time()-t0:.0f}s)", flush=True)

    fpb = out["configs"]["fp16"]["acc_norm"]
    for name in out["configs"]:
        out["configs"][name]["delta_vs_fp16"] = round(out["configs"][name]["acc_norm"] - fpb, 4)
    out["per_item_correct"] = {k: v.tolist() for k, v in vecs.items()}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[done] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
