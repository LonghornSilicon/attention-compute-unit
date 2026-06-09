"""C11: WikiText-2 perplexity for Qwen2-0.5B with ACU x KVCE attention.

Sweeps the 6 configs from analysis/acu_kvce_attention.py over
non-overlapping seq_len-token chunks of WikiText-2 (test split), computes
mean per-token NLL, reports perplexity = exp(NLL).

Run::

    HF_HOME=/home/chaithu/lhs/.hf_cache \\
    KVCE_REF=/home/chaithu/lhs/kv-cache-engine/sw/reference_model \\
    python analysis/c11_wikitext_ppl.py \\
        --configs A,B,C,C_prenorm,E,E_prenorm \\
        --n-samples 4 --seq-len 512

Outputs:
    analysis/c11_wikitext_ppl_runs.jsonl   - append-only log (one row per
                                             config x run)
    analysis/c11_wikitext_ppl_summary.json - last run's aggregate
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


def load_wikitext_chunks(tokenizer, seq_len: int, n_samples: int, seed: int = 0):
    """Return up to n_samples chunks of seq_len tokens from WikiText-2-raw test."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    n_full = ids.shape[0] // seq_len
    n_full = min(n_full, n_samples) if n_samples > 0 else n_full
    # Take the first n_full non-overlapping chunks. Deterministic.
    chunks = [ids[i * seq_len:(i + 1) * seq_len] for i in range(n_full)]
    return chunks


@torch.no_grad()
def nll_for_chunk(model, input_ids: torch.Tensor, device) -> tuple[float, int]:
    """Return (sum_nll_in_nats, n_tokens) for next-token prediction over the chunk.

    Standard LM eval: shift logits/labels by 1; CE loss summed over the
    seq_len-1 prediction positions.
    """
    input_ids = input_ids.unsqueeze(0).to(device)
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[0, :-1, :].float()    # [N-1, V]
    targets = input_ids[0, 1:]                # [N-1]
    log_probs = torch.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [N-1]
    return float(nll.sum().item()), int(nll.numel())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--configs", default=",".join(CONFIGS_ALL))
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--runs-out", default=str(REPO_ROOT / "analysis" / "c11_wikitext_ppl_runs.jsonl")
    )
    ap.add_argument(
        "--summary-out", default=str(REPO_ROOT / "analysis" / "c11_wikitext_ppl_summary.json")
    )
    args = ap.parse_args(argv)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in CONFIGS_ALL:
            raise SystemExit(f"unknown config {c!r}; valid: {CONFIGS_ALL}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[setup] model={args.model}  seq_len={args.seq_len}  n_samples={args.n_samples}", flush=True)
    print(f"[setup] configs={configs}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
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

    chunks = load_wikitext_chunks(tokenizer, args.seq_len, args.n_samples, seed=args.seed)
    print(f"[setup] prepared {len(chunks)} chunks of {args.seq_len} tokens", flush=True)

    runs_path = Path(args.runs_out)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model,
        "seq_len": args.seq_len,
        "n_samples": len(chunks),
        "seed": args.seed,
        "configs": {},
    }

    for cfg in configs:
        akvce.set_config(cfg)
        akvce.reset_stats()
        print(f"\n=== config {cfg} ===", flush=True)

        per_chunk_nll = []
        per_chunk_tokens = []
        t0 = time.time()
        for i, chunk in enumerate(chunks):
            t_chunk0 = time.time()
            try:
                nll_sum, n_tok = nll_for_chunk(model, chunk, device)
            except Exception as e:
                print(f"  chunk {i}: FAILED ({type(e).__name__}: {e})", flush=True)
                nll_sum, n_tok = float("nan"), 0
            per_chunk_nll.append(nll_sum)
            per_chunk_tokens.append(n_tok)
            ppl_chunk = math.exp(nll_sum / max(n_tok, 1)) if n_tok else float("nan")
            dt = time.time() - t_chunk0
            print(
                f"  chunk {i:>3}: NLL/tok={nll_sum / max(n_tok, 1):.4f}  "
                f"ppl={ppl_chunk:.3f}  dt={dt:.1f}s",
                flush=True,
            )

        dt_total = time.time() - t0
        total_nll = float(np.nansum(per_chunk_nll))
        total_tok = int(np.nansum(per_chunk_tokens))
        per_chunk_ppl = [
            math.exp(n / max(t, 1)) for n, t in zip(per_chunk_nll, per_chunk_tokens) if t > 0
        ]
        ppl = math.exp(total_nll / max(total_tok, 1)) if total_tok else float("nan")
        median_ppl = float(np.median(per_chunk_ppl)) if per_chunk_ppl else float("nan")

        cfg_stats = {
            "config": cfg,
            "ppl_pooled": ppl,
            "ppl_median": median_ppl,
            "ppl_per_chunk": per_chunk_ppl,
            "nll_total": total_nll,
            "tokens_total": total_tok,
            "wall_s": dt_total,
            "kvce_ms_total": akvce.CALL_STATS["kvce_ms"],
            "attn_ms_total": akvce.CALL_STATS["attn_ms"],
            "pc_fp16_tiles": akvce.CALL_STATS["pc_fp16_tiles"],
            "pc_total_tiles": akvce.CALL_STATS["pc_total_tiles"],
            "fwd_count": akvce.CALL_STATS["fwd_count"],
        }
        if cfg_stats["pc_total_tiles"] > 0:
            cfg_stats["pc_fp16_pct"] = (
                100.0 * cfg_stats["pc_fp16_tiles"] / cfg_stats["pc_total_tiles"]
            )
        summary["configs"][cfg] = cfg_stats

        # Append-only log: one row per config x run.
        run_row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": args.model,
            "seq_len": args.seq_len,
            "n_samples": len(chunks),
            "seed": args.seed,
            **cfg_stats,
        }
        with runs_path.open("a") as f:
            f.write(json.dumps(run_row) + "\n")

        print(
            f"  -> ppl_pooled={ppl:.3f}  ppl_median={median_ppl:.3f}  "
            f"tokens={total_tok}  wall={dt_total:.1f}s  "
            f"kvce={cfg_stats['kvce_ms_total']/1000:.1f}s  "
            f"attn={cfg_stats['attn_ms_total']/1000:.1f}s",
            flush=True,
        )
        if cfg_stats.get("pc_fp16_pct") is not None:
            print(f"  PC FP16% = {cfg_stats['pc_fp16_pct']:.3f}%", flush=True)

    Path(args.summary_out).write_text(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {args.runs_out} and {args.summary_out}", flush=True)

    shutdown_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main())
