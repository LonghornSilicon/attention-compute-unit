"""Resume the per-layer ablation: load existing rows from the runs JSONL,
run any missing LOO configs, then write the final stats JSON.

Recovers from a mid-LOO crash without re-running the (cheap) baseline /
full / single-layer pieces or the LOO entries that already landed.

Run with the same env as the original::

    HF_HOME=/home/chaithu/lhs/.hf_cache \\
    KVCE_REF=/home/chaithu/lhs/kv-cache-engine/sw/reference_model \\
    python analysis/c11_per_layer_ablation_resume.py
"""

from __future__ import annotations

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

MODEL = "Qwen/Qwen2-0.5B"
MODE = "C_prenorm"
N_SAMPLES = 16
SEQ_LEN = 512
SEED = 0

RUNS_PATH = REPO_ROOT / "analysis" / "c11_per_layer_ablation_runs.jsonl"
STATS_PATH = REPO_ROOT / "analysis" / "c11_per_layer_ablation_stats.json"


def load_existing():
    """Return (baseline_A_ppl, full_ppl, {L: ppl} single, {L: ppl} loo)
    from the runs JSONL. Strategy: keep only rows whose token count matches
    the configured N_SAMPLES x (SEQ_LEN-1), and for each label take the
    LAST row (append-only file, latest write wins). This is robust to the
    smoke run and the n=16 baseline landing in the same hour bucket but
    the full+ablation rows landing later.
    """
    rows = []
    for line in RUNS_PATH.read_text().splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        rows.append(json.loads(line))
    if not rows:
        raise SystemExit("no rows in runs JSONL")
    target_tokens = (SEQ_LEN - 1) * N_SAMPLES
    kept = [r for r in rows if r["tokens"] == target_tokens]
    by_label = {}
    for r in kept:
        by_label[r["label"]] = r  # latest write wins
    print(f"[resume] {len(kept)} rows at n_samples={N_SAMPLES}, "
          f"{len(by_label)} distinct labels")

    ppl_A = None
    ppl_full = None
    single = {}
    loo = {}
    for r in by_label.values():
        lab = r["label"]
        if lab == "baseline_A":
            ppl_A = r["ppl"]
        elif lab.endswith("_full"):
            ppl_full = r["ppl"]
        elif lab.startswith("single_L"):
            L = int(lab.removeprefix("single_L"))
            single[L] = r["ppl"]
        elif lab.startswith("loo_L"):
            L = int(lab.removeprefix("loo_L"))
            loo[L] = r["ppl"]
    if ppl_A is None or ppl_full is None:
        raise SystemExit(f"missing baseline_A or _full row at "
                         f"tokens={(SEQ_LEN-1)*N_SAMPLES}")
    return ppl_A, ppl_full, single, loo


def run_loo_config(model, chunks, device, L: int, n_layers: int) -> dict:
    akvce.set_config(MODE)
    akvce.set_kvce_layers(set(range(n_layers)) - {L})
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
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label": f"loo_L{L}",
        "mode": MODE,
        "kvce_layers": sorted(set(range(n_layers)) - {L}),
        "ppl": ppl,
        "tokens": tok_total,
        "wall_s": dt,
    }


def main():
    ppl_A, ppl_full, single, loo = load_existing()
    print(f"[resume] baseline_A PPL={ppl_A:.3f}")
    print(f"[resume] full      PPL={ppl_full:.3f}")
    print(f"[resume] single-layer rows: {len(single)}/24  "
          f"({sorted(single.keys())[:5]}..{sorted(single.keys())[-1:]})")
    print(f"[resume] loo rows: {len(loo)}  already done: {sorted(loo.keys())}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    akvce.register("acu_kvce")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation="acu_kvce",
    )
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    chunks = load_wikitext_chunks(tokenizer, SEQ_LEN, N_SAMPLES, seed=SEED)
    print(f"[setup] model loaded, n_layers={n_layers}, {len(chunks)} chunks")

    missing = [L for L in range(n_layers) if L not in loo]
    print(f"[resume] LOO layers to run: {missing}")

    for L in missing:
        rec = run_loo_config(model, chunks, device, L, n_layers)
        loo[L] = rec["ppl"]
        rec_for_log = {k: v for k, v in rec.items() if k != "ts"}
        recovery = math.log(ppl_full / rec["ppl"])
        print(f"  L{L:>2}: PPL={rec['ppl']:8.3f}  "
              f"recovery_nats={recovery:+6.4f}  wall={rec['wall_s']:.1f}s",
              flush=True)
        with RUNS_PATH.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    summary = {
        "model": MODEL,
        "mode": MODE,
        "n_samples": N_SAMPLES,
        "seq_len": SEQ_LEN,
        "n_layers": n_layers,
        "ppl_baseline_A": ppl_A,
        "ppl_full": ppl_full,
        "single_layer_ppl": {str(k): v for k, v in sorted(single.items())},
        "loo_ppl": {str(k): v for k, v in sorted(loo.items())},
        "single_layer_factor_vs_A": {
            str(k): v / ppl_A for k, v in sorted(single.items())
        },
    }
    STATS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {STATS_PATH}")
    shutdown_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main())
