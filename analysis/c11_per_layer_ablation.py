"""C11 follow-on: per-layer attribution of KVCE-induced PPL loss.

For each Qwen2-0.5B layer L in 0..n_layers-1, measure:

  PPL_single[L] = PPL when KVCE is applied ONLY on layer L (rest are
                  identity / FP16 baseline). Marginal cost of compressing
                  layer L alone.

  PPL_loo[L]    = PPL when KVCE is applied on EVERY layer EXCEPT L.
                  How much PPL improves if you skip layer L.

Reference points (also measured):

  PPL_A         = baseline FP16, no KVCE anywhere
  PPL_full      = KVCE on every layer (config C_prenorm by default)

Tells you whether the C12 noise floor is concentrated in a few layers
(per-layer centroid / mode-select design) or spread (uniform turbo8
design). Feeds into kv-cache-engine/findings/centroid_design_brief.md.

Default uses ``C_prenorm`` (KVCE only, with the C1 prenorm bridge on)
because that isolates the C12 / centroid-fit contribution from the C1
clip. Run::

    HF_HOME=/home/chaithu/lhs/.hf_cache \\
    KVCE_REF=/home/chaithu/lhs/kv-cache-engine/sw/reference_model \\
    python analysis/c11_per_layer_ablation.py --n-samples 16

Outputs:
    analysis/c11_per_layer_ablation_runs.jsonl
    analysis/c11_per_layer_ablation_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))

import acu_kvce_attention as akvce
from c11_wikitext_ppl import load_wikitext_chunks, nll_for_chunk
from kvce_pool import shutdown_pool


DEFAULT_MODEL = "Qwen/Qwen2-0.5B"


def run_config(model, chunks, device, mode: str, kvce_layers, label: str) -> dict:
    akvce.set_config(mode)
    akvce.set_kvce_layers(kvce_layers)
    akvce.reset_stats()
    t0 = time.time()
    nll_total, tok_total = 0.0, 0
    for chunk in chunks:
        nll, ntok = nll_for_chunk(model, chunk, device)
        if not math.isnan(nll) and ntok > 0:
            nll_total += nll
            tok_total += ntok
    dt = time.time() - t0
    ppl = math.exp(nll_total / max(tok_total, 1)) if tok_total else float("nan")
    return {
        "label": label,
        "mode": mode,
        "kvce_layers": sorted(kvce_layers) if kvce_layers is not None else None,
        "ppl": ppl,
        "tokens": tok_total,
        "wall_s": dt,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--mode", default="C_prenorm",
                    help="Which KVCE config to use on the gated layers")
    ap.add_argument("--n-samples", type=int, default=16,
                    help="Chunks of seq-len tokens of WikiText-2. Each ablation "
                         "uses the same chunks for pairing.")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default="",
                    help="Comma-separated layer subset to ablate (e.g. "
                         "'0,4,8,12,16,20,23'). Default: every layer.")
    ap.add_argument("--skip-loo", action="store_true",
                    help="Skip the leave-one-out direction (just measure "
                         "single-layer marginal cost).")
    ap.add_argument(
        "--runs-out",
        default=str(REPO_ROOT / "analysis" / "c11_per_layer_ablation_runs.jsonl"),
    )
    ap.add_argument(
        "--summary-out",
        default=str(REPO_ROOT / "analysis" / "c11_per_layer_ablation_stats.json"),
    )
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[setup] model={args.model}  mode={args.mode}  "
          f"n_samples={args.n_samples}  seq_len={args.seq_len}", flush=True)

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
    n_layers = model.config.num_hidden_layers
    layer_set = sorted(int(s) for s in args.layers.split(",") if s) \
                if args.layers else list(range(n_layers))
    print(f"[setup] ablating layers {layer_set}", flush=True)

    chunks = load_wikitext_chunks(tokenizer, args.seq_len, args.n_samples,
                                  seed=args.seed)
    print(f"[setup] {len(chunks)} chunks prepared", flush=True)

    runs_path = Path(args.runs_out)
    runs_path.parent.mkdir(parents=True, exist_ok=True)

    all_records = []

    def log_row(rec: dict) -> None:
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}
        with runs_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    # -- Baselines -----------------------------------------------------
    print("\n[ref] baseline A (no KVCE anywhere)", flush=True)
    rec = run_config(model, chunks, device, "A", set(), label="baseline_A")
    print(f"  PPL = {rec['ppl']:.3f}   wall={rec['wall_s']:.1f}s", flush=True)
    all_records.append(rec); log_row(rec)
    ppl_A = rec["ppl"]

    print(f"\n[ref] {args.mode} on every layer", flush=True)
    rec = run_config(model, chunks, device, args.mode, None,
                     label=f"{args.mode}_full")
    print(f"  PPL = {rec['ppl']:.3f}   wall={rec['wall_s']:.1f}s", flush=True)
    all_records.append(rec); log_row(rec)
    ppl_full = rec["ppl"]

    # -- Single-layer ablation: KVCE on exactly one layer --------------
    print(f"\n[ablate] single-layer: KVCE on one layer, identity elsewhere",
          flush=True)
    single = {}
    for L in layer_set:
        rec = run_config(model, chunks, device, args.mode, {L},
                         label=f"single_L{L}")
        single[L] = rec["ppl"]
        delta = math.log(rec["ppl"] / ppl_A)
        print(f"  L{L:>2}: PPL={rec['ppl']:8.3f}  delta_nats={delta:+6.4f}  "
              f"wall={rec['wall_s']:.1f}s", flush=True)
        all_records.append(rec); log_row(rec)

    # -- Leave-one-out: KVCE everywhere except one layer ---------------
    loo = {}
    if not args.skip_loo:
        print(f"\n[ablate] leave-one-out: KVCE everywhere EXCEPT one layer",
              flush=True)
        for L in layer_set:
            others = set(range(n_layers)) - {L}
            rec = run_config(model, chunks, device, args.mode, others,
                             label=f"loo_L{L}")
            loo[L] = rec["ppl"]
            recovery = math.log(ppl_full / rec["ppl"])  # +ve = removing L helped
            print(f"  L{L:>2}: PPL={rec['ppl']:8.3f}  "
                  f"recovery_nats={recovery:+6.4f}  wall={rec['wall_s']:.1f}s",
                  flush=True)
            all_records.append(rec); log_row(rec)

    summary = {
        "model": args.model,
        "mode": args.mode,
        "n_samples": len(chunks),
        "seq_len": args.seq_len,
        "n_layers": n_layers,
        "ppl_baseline_A": ppl_A,
        "ppl_full": ppl_full,
        "single_layer_ppl": {str(k): v for k, v in single.items()},
        "loo_ppl": {str(k): v for k, v in loo.items()},
        # Convenience: PPL multiplier vs baseline for single-layer ablation
        "single_layer_factor_vs_A": {
            str(k): v / ppl_A for k, v in single.items()
        },
    }
    Path(args.summary_out).write_text(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {args.runs_out} and {args.summary_out}", flush=True)
    shutdown_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main())
