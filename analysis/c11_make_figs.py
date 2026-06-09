"""Figures for the C11 end-to-end accuracy report.

Reads append-only JSONL logs in analysis/ and writes:

  paper/figs/c11_ppl_by_config.pdf  - bar chart with bootstrap CI per config
  paper/figs/c11_hellaswag.pdf      - HellaSwag acc bar chart with Wilson CI

Regenerate with::

    python analysis/c11_make_figs.py

Source of truth: analysis/c11_wikitext_ppl_runs.jsonl and
                 analysis/c11_hellaswag_runs.jsonl.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_PPL = REPO_ROOT / "analysis" / "c11_wikitext_ppl_runs.jsonl"
RUNS_HS = REPO_ROOT / "analysis" / "c11_hellaswag_runs.jsonl"
FIG_DIR = REPO_ROOT / "paper" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

CONFIG_ORDER = ["A", "B", "C", "C_prenorm", "E", "E_prenorm"]
CONFIG_LABELS = {
    "A":         "A: baseline\nFP16",
    "B":         "B: PC routing\nonly",
    "C":         "C: KVCE only\n(naive Q4.12)",
    "C_prenorm": "C': KVCE only\n(prenorm)",
    "E":         "E: integrated\n(naive Q4.12)",
    "E_prenorm": "E': integrated\n(prenorm)",
}


def latest_rows(runs_path: Path) -> dict[str, dict]:
    """For each config, take the most recent row in the JSONL."""
    if not runs_path.exists():
        return {}
    rows: dict[str, dict] = {}
    with runs_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["config"]] = row
    return rows


def bootstrap_ppl_ci(per_chunk_ppl: list[float], n_boot: int = 2000, alpha: float = 0.05,
                     rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Bootstrap 95% CI for the geometric-mean perplexity over chunks
    (equivalent to pooled exp(mean log-ppl)).
    """
    if not per_chunk_ppl:
        return (float("nan"), float("nan"))
    arr = np.log(np.array(per_chunk_ppl, dtype=float))
    rng = rng or np.random.default_rng(0)
    n = arr.size
    boots = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, n)
        boots[i] = arr[s].mean()
    lo = float(np.exp(np.percentile(boots, 100 * alpha / 2)))
    hi = float(np.exp(np.percentile(boots, 100 * (1 - alpha / 2))))
    return lo, hi


def fig_ppl():
    rows = latest_rows(RUNS_PPL)
    if not rows:
        print(f"[skip] {RUNS_PPL} not found")
        return
    configs = [c for c in CONFIG_ORDER if c in rows]
    ppls = [rows[c]["ppl_pooled"] for c in configs]
    cis = [bootstrap_ppl_ci(rows[c].get("ppl_per_chunk", [])) for c in configs]
    # tokens_total counts only positions with a prediction (seq_len - 1 per chunk)
    n_chunks = max(rows[c].get("tokens_total", 0) // max(rows[c].get("seq_len", 512) - 1, 1)
                   for c in configs)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(configs))
    bars = ax.bar(x, ppls, width=0.62,
                  color=["#2b8cbe", "#4292c6", "#fdae61", "#e6f598",
                         "#d7191c", "#a6d96a"][: len(configs)],
                  edgecolor="black", linewidth=0.5)
    err_lo = np.array([p - lo for p, (lo, hi) in zip(ppls, cis)])
    err_hi = np.array([hi - p for p, (lo, hi) in zip(ppls, cis)])
    ax.errorbar(x, ppls, yerr=[err_lo, err_hi], fmt="none",
                ecolor="black", capsize=2.0, capthick=0.5, linewidth=0.5)
    for xi, p in zip(x, ppls):
        ax.text(xi, p, f" {p:,.1f}" if p < 1000 else f" {p:,.0f}",
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs])
    ax.set_ylabel("WikiText-2 perplexity (log scale)")
    ax.set_yscale("log")
    ax.grid(axis="y", which="both", alpha=0.25, linewidth=0.4)
    title = (f"Qwen2-0.5B WikiText-2 perplexity, ACU x KVCE attention substitute "
             f"({n_chunks} x 512-token chunks)")
    ax.set_title(title, fontsize=9)
    ax.text(0.99, 0.97,
            "error bars: 95 % bootstrap CI over chunks\n"
            "lower is better; log scale",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="0.7", boxstyle="round,pad=0.3"))
    fig.tight_layout()
    out = FIG_DIR / "c11_ppl_by_config.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=160)
    print(f"[ok] {out}")
    plt.close(fig)


def fig_hellaswag():
    rows = latest_rows(RUNS_HS)
    if not rows:
        print(f"[skip] {RUNS_HS} not found")
        return
    configs = [c for c in CONFIG_ORDER if c in rows]
    accs = [rows[c]["acc_norm"] for c in configs]
    cis = [rows[c].get("acc_norm_ci95", [acc, acc]) for c, acc in zip(configs, accs)]
    n_items = rows[configs[0]].get("n_items", 0)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(configs))
    ax.bar(x, accs, width=0.62,
           color=["#2b8cbe", "#4292c6", "#fdae61", "#e6f598",
                  "#d7191c", "#a6d96a"][: len(configs)],
           edgecolor="black", linewidth=0.5)
    err_lo = np.array([a - lo for a, (lo, _) in zip(accs, cis)])
    err_hi = np.array([hi - a for a, (_, hi) in zip(accs, cis)])
    ax.errorbar(x, accs, yerr=[err_lo, err_hi], fmt="none",
                ecolor="black", capsize=2.0, capthick=0.5, linewidth=0.5)
    for xi, a in zip(x, accs):
        ax.text(xi, a, f" {a:.3f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0.25, color="grey", linewidth=0.5, linestyle="--")
    ax.text(len(configs) - 0.5, 0.255, "chance (0.25)", fontsize=7, color="grey", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs])
    ax.set_ylabel("HellaSwag accuracy (length-normalized)")
    ax.set_ylim(0, max(0.55, max(accs) * 1.15))
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    ax.set_title(f"Qwen2-0.5B HellaSwag (val, n={n_items})", fontsize=9)
    ax.text(0.99, 0.97,
            "error bars: 95 % Wilson CI",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="0.7", boxstyle="round,pad=0.3"))
    fig.tight_layout()
    out = FIG_DIR / "c11_hellaswag.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=160)
    print(f"[ok] {out}")
    plt.close(fig)


def fig_per_layer_ablation():
    path = REPO_ROOT / "analysis" / "c11_per_layer_ablation_stats.json"
    if not path.exists():
        print(f"[skip] {path} not found")
        return
    data = json.loads(path.read_text())
    n_layers = int(data["n_layers"])
    ppl_A = float(data["ppl_baseline_A"])
    ppl_full = float(data["ppl_full"])
    single = {int(k): float(v) for k, v in data["single_layer_ppl"].items()}
    loo = {int(k): float(v) for k, v in data.get("loo_ppl", {}).items()}

    layers = sorted(single.keys())
    single_nats = np.array([math.log(single[L] / ppl_A) for L in layers])
    loo_nats = np.array([math.log(ppl_full / loo[L]) for L in layers]) if loo else None

    fig, axes = plt.subplots(2 if loo else 1, 1, figsize=(7.0, 5.6 if loo else 3.0),
                              sharex=True)
    if not loo:
        axes = [axes]

    # Single-layer marginal cost (lower is better)
    ax = axes[0]
    bars = ax.bar(layers, single_nats, color="#d7191c", edgecolor="black",
                  linewidth=0.4, width=0.78)
    ax.set_ylabel("log(PPL/baseline) [nats]\n(marginal cost of KVCE on this layer)")
    ax.set_title(
        f"Per-layer KVCE cost on {data['model']}  "
        f"(mode={data['mode']}, n={data['n_samples']} chunks, "
        f"PPL_A={ppl_A:.2f}, PPL_full={ppl_full:.1f})",
        fontsize=9,
    )
    ax.axhline(0, color="black", linewidth=0.4)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    for L, v in zip(layers, single_nats):
        if abs(v) > 0.5 or L in (0, n_layers - 1):
            ax.text(L, v, f"{v:+.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(layers)

    # Leave-one-out recovery (higher = removing this layer helps more)
    if loo is not None:
        ax = axes[1]
        ax.bar(layers, loo_nats, color="#2b8cbe", edgecolor="black",
               linewidth=0.4, width=0.78)
        ax.set_ylabel("log(PPL_full/PPL_loo) [nats]\n(recovery from removing layer L)")
        ax.set_xlabel("Layer index")
        ax.axhline(0, color="black", linewidth=0.4)
        ax.grid(axis="y", alpha=0.25, linewidth=0.4)
        for L, v in zip(layers, loo_nats):
            if abs(v) > 0.5 or L in (0, n_layers - 1):
                ax.text(L, v, f"{v:+.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(layers)
    else:
        ax.set_xlabel("Layer index")

    fig.tight_layout()
    out = FIG_DIR / "c11_per_layer_ablation.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=160)
    print(f"[ok] {out}")
    plt.close(fig)


if __name__ == "__main__":
    fig_ppl()
    fig_hellaswag()
    fig_per_layer_ablation()
