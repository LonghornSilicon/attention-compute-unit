"""C11: HellaSwag accuracy for Qwen2-0.5B with ACU x KVCE attention.

For each HellaSwag item:
  ctx = ctx_a + " " + ctx_b
  For each of 4 candidate endings:
    Tokenize ctx + " " + ending
    Forward pass, sum NLL over the ending tokens only
  Predicted label = argmin (NLL / n_ending_tokens)

Reports per-config acc and length-normalized acc on a subset of the
validation set. Run::

    HF_HOME=/home/chaithu/lhs/.hf_cache \\
    KVCE_REF=/home/chaithu/lhs/kv-cache-engine/sw/reference_model \\
    python analysis/c11_hellaswag.py \\
        --configs A,B,C,C_prenorm,E,E_prenorm \\
        --n-items 250
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))

import acu_kvce_attention as akvce
from kvce_pool import shutdown_pool


CONFIGS_ALL = ["A", "B", "C", "C_prenorm", "E", "E_prenorm"]
DEFAULT_MODEL = "Qwen/Qwen2-0.5B"


@torch.no_grad()
def score_ending(model, tokenizer, ctx_ids: torch.Tensor, ending: str, device) -> tuple[float, int]:
    """Return (sum NLL of ending tokens, n_ending_tokens).

    Tokenization: " " + ending appended to the context. NLL is computed
    only over the ending positions (the context positions are excluded
    from the sum but still produce key-attending logits in the forward
    pass).
    """
    end_ids = tokenizer(" " + ending, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    full_ids = torch.cat([ctx_ids, end_ids], dim=0).unsqueeze(0).to(device)
    out = model(input_ids=full_ids, use_cache=False)
    n_ctx = ctx_ids.shape[0]
    n_end = end_ids.shape[0]
    # Predict ending token at position p means using logits at position p-1.
    logits = out.logits[0, n_ctx - 1 : n_ctx - 1 + n_end, :].float()  # [n_end, V]
    targets = full_ids[0, n_ctx : n_ctx + n_end]                       # [n_end]
    log_probs = torch.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)     # [n_end]
    return float(nll.sum().item()), int(n_end)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--configs", default=",".join(CONFIGS_ALL))
    ap.add_argument("--n-items", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ctx-tokens", type=int, default=160,
                    help="Skip items whose context is longer than this.")
    ap.add_argument(
        "--runs-out", default=str(REPO_ROOT / "analysis" / "c11_hellaswag_runs.jsonl")
    )
    ap.add_argument(
        "--summary-out", default=str(REPO_ROOT / "analysis" / "c11_hellaswag_summary.json")
    )
    args = ap.parse_args(argv)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in CONFIGS_ALL:
            raise SystemExit(f"unknown config {c!r}; valid: {CONFIGS_ALL}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    akvce.register("acu_kvce")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation="acu_kvce",
    )
    model.eval()
    device = next(model.parameters()).device

    print(f"[setup] loading HellaSwag validation...", flush=True)
    ds = load_dataset("Rowan/hellaswag", split="validation")

    # Build subset: take first n_items items that fit our max_ctx budget.
    items = []
    for ex in ds:
        ctx = (ex["ctx_a"].strip() + " " + ex["ctx_b"].strip()).strip()
        try:
            label = int(ex["label"])
        except (TypeError, ValueError):
            continue
        ctx_ids = tokenizer(ctx, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if ctx_ids.shape[0] > args.max_ctx_tokens:
            continue
        items.append({
            "ind": ex["ind"],
            "ctx_ids": ctx_ids,
            "endings": list(ex["endings"]),
            "label": label,
            "ctx_len": int(ctx_ids.shape[0]),
        })
        if len(items) >= args.n_items:
            break

    print(f"[setup] selected {len(items)} items "
          f"(median ctx len = {int(np.median([it['ctx_len'] for it in items]))} toks)",
          flush=True)

    runs_path = Path(args.runs_out)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model,
        "n_items": len(items),
        "seed": args.seed,
        "configs": {},
    }

    for cfg in configs:
        akvce.set_config(cfg)
        akvce.reset_stats()
        print(f"\n=== config {cfg} ===", flush=True)

        n_correct = 0
        n_correct_norm = 0
        t0 = time.time()
        per_item = []
        for idx, it in enumerate(items):
            ctx_ids = it["ctx_ids"]
            nlls, ntoks = [], []
            for end in it["endings"]:
                nll, ntok = score_ending(model, tokenizer, ctx_ids, end, device)
                nlls.append(nll)
                ntoks.append(ntok)
            nlls = np.array(nlls)
            ntoks = np.array(ntoks, dtype=float)
            pred = int(nlls.argmin())                              # unnormalized
            pred_norm = int((nlls / np.maximum(ntoks, 1)).argmin())  # length-normalized
            n_correct += int(pred == it["label"])
            n_correct_norm += int(pred_norm == it["label"])
            per_item.append({"ind": it["ind"], "label": it["label"],
                             "pred": pred, "pred_norm": pred_norm,
                             "nlls": nlls.tolist(), "ntoks": ntoks.tolist()})
            if (idx + 1) % 25 == 0 or idx + 1 == len(items):
                acc = n_correct / (idx + 1)
                acc_n = n_correct_norm / (idx + 1)
                dt = time.time() - t0
                print(f"  [{idx+1:>3}/{len(items)}] acc={acc:.3f}  acc_norm={acc_n:.3f}  "
                      f"({dt:.0f}s, {dt/(idx+1):.2f}s/item)", flush=True)

        dt_total = time.time() - t0
        acc = n_correct / len(items)
        acc_norm = n_correct_norm / len(items)
        n = len(items)
        # Wilson 95% CI for proportions
        def wilson(k, n):
            if n == 0:
                return (0.0, 0.0)
            p = k / n
            z = 1.96
            denom = 1 + z * z / n
            centre = (p + z * z / (2 * n)) / denom
            half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
            return (max(0.0, centre - half), min(1.0, centre + half))
        ci_lo, ci_hi = wilson(n_correct, n)
        ci_norm_lo, ci_norm_hi = wilson(n_correct_norm, n)

        cfg_stats = {
            "config": cfg,
            "n_items": n,
            "acc": acc,
            "acc_ci95": [ci_lo, ci_hi],
            "acc_norm": acc_norm,
            "acc_norm_ci95": [ci_norm_lo, ci_norm_hi],
            "wall_s": dt_total,
            "kvce_ms_total": akvce.CALL_STATS["kvce_ms"],
            "attn_ms_total": akvce.CALL_STATS["attn_ms"],
            "pc_fp16_tiles": akvce.CALL_STATS["pc_fp16_tiles"],
            "pc_total_tiles": akvce.CALL_STATS["pc_total_tiles"],
            "fwd_count": akvce.CALL_STATS["fwd_count"],
        }
        if cfg_stats["pc_total_tiles"] > 0:
            cfg_stats["pc_fp16_pct"] = 100.0 * cfg_stats["pc_fp16_tiles"] / cfg_stats["pc_total_tiles"]
        summary["configs"][cfg] = cfg_stats

        run_row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": args.model,
            "n_items": n,
            "seed": args.seed,
            **cfg_stats,
        }
        with runs_path.open("a") as f:
            f.write(json.dumps(run_row) + "\n")

        print(f"  -> acc={acc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]  "
              f"acc_norm={acc_norm:.3f} [{ci_norm_lo:.3f}, {ci_norm_hi:.3f}]  "
              f"wall={dt_total:.0f}s",
              flush=True)

    Path(args.summary_out).write_text(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {args.runs_out} and {args.summary_out}", flush=True)
    shutdown_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main())
