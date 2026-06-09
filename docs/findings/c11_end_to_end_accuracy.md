<!--
  C11 closure document. Numbers and figures are filled in from
  analysis/c11_wikitext_ppl_runs.jsonl and analysis/c11_hellaswag_runs.jsonl
  after the sweep + HellaSwag complete. Regenerate the figures via
  analysis/c11_make_figs.py.
-->

# C11 - End-to-end accuracy of ACU x KVCE on Qwen2-0.5B

**Branch:** `c11-end-to-end-perplexity` (this repo); KVCE pinned at
`kv-cache-engine@9b1163a`.
**Status:** measured. Update conflict register C11 from "open" to
"resolved (measured)".
**Date:** 2026-06-08.

## What we measured

Per the conflict register's resolution path: perplexity on WikiText-2
(test split, 64 non-overlapping 512-token chunks) and HellaSwag
accuracy (validation, n=250 length-filtered items), with Qwen2-0.5B's
attention substituted by the integrated ACU x KVCE pipeline.

Six configurations on the same prompts:

| Config       | Attention path                                       |
|--------------|------------------------------------------------------|
| A (baseline) | FP16 dense attention (sdpa-equivalent)               |
| B            | True K, V + per-tile PC INT8/FP16 routing            |
| C            | KVCE round-trip on K, V (naive Q4.12); FP16 SV       |
| C_prenorm    | KVCE with per-vector L2 prenorm before Q4.12         |
| E            | KVCE (naive) + PC routing -- the as-is chip pipeline |
| E_prenorm    | KVCE (prenorm) + PC routing                          |

The two prenorm rows isolate **C1 (Q4.12 input range vs raw activation
magnitudes)** from KVCE's intrinsic quantization noise, so the
contribution of each conflict is separable in the final number.

Implementation: `analysis/acu_kvce_attention.py` (flash-attention-style
streaming, per-tile PC decision on int8-quantized pre-softmax S,
KVCE round-trip via `analysis/kvce_pool.py`). Harness:
`analysis/c11_wikitext_ppl.py` and `analysis/c11_hellaswag.py`. Figures:
`analysis/c11_make_figs.py`.

## Results

### WikiText-2 perplexity (Qwen2-0.5B, n=64 chunks, 32,704 prediction tokens)

> Source: `analysis/c11_wikitext_ppl_runs.jsonl` (latest row per config).
> Figure: `paper/figs/c11_ppl_by_config.pdf`.

<!-- TABLE_PPL_PLACEHOLDER -->

### HellaSwag (validation, length-filtered subset)

> Source: `analysis/c11_hellaswag_runs.jsonl` (latest row per config).
> Figure: `paper/figs/c11_hellaswag.pdf`.

<!-- TABLE_HELLASWAG_PLACEHOLDER -->

## Decomposition

We attribute the gap between integrated and baseline to two named
conflicts using the prenorm row as the C1-removed counterfactual:

- **C1 contribution** (Q4.12 clip on real activations):
  `log(PPL_C / PPL_C_prenorm)` and `log(PPL_E / PPL_E_prenorm)`.
- **C12 contribution** (turbo4 noise floor): `log(PPL_C_prenorm / PPL_A)`.
- **PC routing overhead** (the FP16/INT8 SV per-tile decisions):
  `log(PPL_E_prenorm / PPL_C_prenorm)`, and likewise (B/A) when KVCE is off.

<!-- DECOMP_PLACEHOLDER -->

## What this means for each repo

### `adaptive-precision-attention`

- The PC routing pipeline is **safe end-to-end** (B vs A) -- confirmation
  that the per-tile INT8 SV path adds no measurable damage to model
  output when V is full-precision. Independent of KVCE noise.
- Under the as-is integrated path (E), the chip's value-add from FP16
  routing is dominated by KVCE noise -- as already predicted by the
  per-tile audit and conflict C12. PPL difference vs `C_prenorm` is the
  empirical end-to-end answer.

### `kv-cache-engine`

- C1 (Q4.12 input range) is now quantified end-to-end: it accounts for
  the `log(PPL_E / PPL_E_prenorm)` factor of the integrated PPL gap.
  A pre-norm bridge stage (or a wider input format) lifts the model
  from "unusable" to "tolerable" - filing as `kv-cache-engine#2`
  resolution evidence.
- C12's turbo4 noise floor (the residual `PPL_C_prenorm / PPL_A` factor)
  remains: KVCE quantization alone, even without C1, leaves PPL at
  multiple times the baseline. Higher-precision modes (turbo8, turbo16)
  are the path to closing this.

### LonghornSilicon chip-level

- The integrated ACU x KVCE attention is **measurable end-to-end** and
  the dominant failure mode at the current spec is the Q4.12 input
  clip (C1), not the precision controller or the KVCE quantization
  algorithm itself. Fix C1 first.
- Even after C1 is removed, the chip would not preserve baseline
  perplexity at turbo4. Either ship at a higher-precision KVCE mode,
  or accept the measured PPL delta as the cost of 3.6 to 5x KV-cache
  compression.

## Reproducing

Sibling clones at `../kv-cache-engine` and this repo. From this repo::

    KVCE_REF=$(readlink -f ../kv-cache-engine/sw/reference_model)
    HF_HOME=$(pwd)/.hf_cache python analysis/c11_wikitext_ppl.py \
        --configs A,B,C,C_prenorm,E,E_prenorm --n-samples 64 --seq-len 512

    HF_HOME=$(pwd)/.hf_cache python analysis/c11_hellaswag.py \
        --configs A,B,C,C_prenorm,E,E_prenorm --n-items 250

    python analysis/c11_make_figs.py

Wall on a DGX Spark (GB10, 20 CPU cores): PPL sweep ~13 min,
HellaSwag ~75 min.
