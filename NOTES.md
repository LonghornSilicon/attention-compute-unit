# Lab Notebook — ACU x KVCE end-to-end accuracy (C11)

This notebook tracks the C11 measurement campaign: end-to-end accuracy
of Qwen2-0.5B with attention substituted by the integrated ACU x KVCE
pipeline, on perplexity and (later) HellaSwag.

Branch: `c11-end-to-end-perplexity` (this repo); KVCE pinned at
`kv-cache-engine@9b1163a` (post-reconstruction-fix). Sibling clones at
`../kv-cache-engine` and this repo.

Hardware: DGX Spark, GB10 GPU, 121 GiB LPDDR5, 20 CPU cores. Ubuntu
24.04 aarch64, CUDA 13.0, torch 2.12.0+cu130, transformers 5.10.2,
numpy 2.4.6. Python venv at `/home/chaithu/lhs/.venv`.

Conventions:
- All artefacts under `analysis/c11_*` (data + scripts).
- HF cache redirected to `/home/chaithu/lhs/.hf_cache/` because
  `~/.cache` is root-owned on this box. Set `HF_HOME` for every run.
- `KVCE_REF=/home/chaithu/lhs/kv-cache-engine/sw/reference_model` for
  every run that imports the KVCE Python ref.
- Every run records: seed, command line, config hash, n samples, wall
  time, and host. Numbers in this notebook carry units; uncertainty
  expressed as median + IQR (or mean ± SEM where pairing is natural).
- Single-run improvements below ~3 sigma are reported as "noise".

---

## 2026-06-08 — Set up: cross-repo wiring works end-to-end

Cloned both LonghornSilicon repos as siblings under `/home/chaithu/lhs`
and reproduced the published `kvce-acu-audit-0.2` integration result
on 2,744 tiles of Qwen2-0.5B (seq_len=512, layers 0/4/8/12/16/20/23,
1 prose prompt). Numbers match within rounding:

| Metric                         | Audit (May 21) | This run (Jun 8) |
|--------------------------------|---------------:|-----------------:|
| n tiles                        | 2,744          | 2,744            |
| Decision agreement             | 99.82 %        | 99.78 %          |
| Median rMSE B (PC alone)       | 0.0002         | 0.0002           |
| Median rMSE C (KV + FP16 SV)   | 0.3637         | 0.3632           |
| Median rMSE D (KV + INT8 SV)   | 0.3631         | 0.3641           |
| Median rMSE E (integrated)     | 0.3631         | 0.3641           |

Bug found in the harness during setup: `analysis/integration_test_kv_pc.py`
hard-coded `KVCE_REF = Path("/home/shadeform/...")`. Replaced with
`$KVCE_REF` env var + sibling-clone fallback. Updated the audit doc's
repro section to match.

Stored artefact: `analysis/integration_test_kv_pc_stats.json` (the
post-fix 2,744-tile run on this host).

Decision: proceed to C11 (end-to-end perplexity on Qwen2-0.5B).

---

## 2026-06-08 — KVCE multiprocessing pool: 12x speedup, prenorm fix validated

Wrote `analysis/kvce_pool.py`: a `ProcessPoolExecutor` (spawn context,
20 workers) where each worker holds its own `KVCacheEngine` instance.
Exposes `kv_roundtrip(K, V, mode)` with `mode in {"naive", "prenorm"}`.

- Naive mode: direct float -> Q4.12 cast, clips at +/-8.0 (the current
  C1 defect).
- Prenorm mode: scales each vector so `max(|v|)` lands at +/-4.0
  before the Q4.12 cast; restored at decompress. Eliminates C1
  in software for the experiment.

Sanity numbers on random N(0, 1/sqrt(64)) vectors:

| Test                                    | Wall  | cos(K_hat, K) |
|-----------------------------------------|------:|--------------:|
| Warmup, 128 vecs                        | 0.17s | n/a           |
| Naive, 2,048 vecs                       | 0.20s | 0.975         |
| Naive on |K| ~ 50 (C1 stress)           | 0.08s | **0.417**     |
| Prenorm on |K| ~ 50                     | 0.08s | **0.975**     |

The naive-mode collapse to cos=0.417 when the input magnitude exits
the Q4.12 range directly demonstrates C1 in numerical form; the
prenorm restores cos to the audit's nominal 0.975. This is the
isolator we'll use in the e2e PPL sweep.

Throughput: 2,048 vectors in 0.20 s -> ~100 us / vector across the
pool. Single-thread Python ref is ~1.2 ms / vector -> ~12x speedup.
For Qwen2-0.5B end-to-end (24 layers x 2 KV-heads x 512 tokens =
24,576 KV pairs per forward) this puts the KVCE round-trip at ~2.4 s
per forward pass instead of ~30 s.

Stored artefact: `analysis/kvce_pool.py` (committed below).

Decision: build the e2e PPL harness next; KVCE is no longer the
runtime bottleneck.

---

## 2026-06-08 — Smoke test (n=2 chunks): bug found and fixed in the patched-attention path

Built `analysis/acu_kvce_attention.py` (flash-attention-style streaming
attention with per-tile PC routing and optional KVCE round-trip) and
`analysis/c11_wikitext_ppl.py` (WikiText-2 PPL sweep harness). Registered
the substitute under `ALL_ATTENTION_FUNCTIONS["acu_kvce"]` and loaded
Qwen2-0.5B with `attn_implementation="acu_kvce"`.

**Bug found:** first config-A smoke gave PPL = 1.07 — impossibly low.
HF passes `attention_mask=None` to custom attention impls for causal
decoder LMs (the contract is "you know it's causal, build the mask
yourself" — same as SDPA's `is_causal=True`). My substitute wasn't
applying a causal mask, so every position attended to future tokens
and the LM became trivially predictive. Fixed by constructing the
strict-upper-triangular `-inf` mask inside the substitute and applying
it on both the A fast path and the streaming path. After the fix,
config A reproduces PPL exactly:

| Method                                                     | chunk 0 PPL |
|------------------------------------------------------------|------------:|
| Default `attn_implementation` (sdpa) via HF labels loss    | 11.654      |
| My `acu_kvce` config A, post-fix                           | 11.654      |

Rule added: any custom HF attention impl must apply its own causal mask
when `attention_mask` is None on a causal-LM forward.

**Smoke results (n=2 chunks, seq_len=512, 1022 prediction tokens):**

| Config        | PPL (pooled) | PPL (median) | Wall  | PC FP16% |
|---------------|-------------:|-------------:|------:|---------:|
| A baseline    | 15.0         | 15.5         | 0.6s  | n/a      |
| B (PC only)   | 15.1         | 15.5         | 0.6s  | 0.00 %   |
| C (KVCE naive)| 3,998        | 4,433        | 6.5s  | n/a      |
| C_prenorm     | 545          | 695          | 5.5s  | n/a      |
| E (naive)     | 6,162        | 6,787        | 6.2s  | 0.15 %   |
| E_prenorm     | 569          | 741          | 6.0s  | 0.00 %   |

Decomposition:
- B vs A (15.1 vs 15.0): PC routing alone is **harmless**. Matches the
  per-tile finding that 99.8 % of tiles route INT8 and per-tile INT8
  noise is rMSE ~2e-4.
- C vs A (3998 vs 15): KVCE alone with naive Q4.12 is **catastrophic**
  (~265x worse PPL). Direct end-to-end confirmation of C1 (Q4.12 clip
  vs real activation magnitudes).
- C_prenorm vs C (545 vs 3998): isolating C1 alone wins ~7x. So C1 is
  a major contributor, but ~545 PPL is still ~36x baseline — KVCE's
  intrinsic `turbo4` quantization noise (the C12 noise floor) is doing
  most of the remaining damage.
- E vs C (6162 vs 3998), E_prenorm vs C_prenorm (569 vs 545): PC
  routing in the integrated path doesn't materially change the picture
  when KVCE noise dominates. Consistent with the per-tile audit's
  observation that almost no tile escapes INT8 routing under lossy V.

n=2 is way too small for a confident number; scaling to 64 chunks next
(~13 min wall on this host).

Wall-time projection (64 chunks):
- KVCE configs (C/C_prenorm/E/E_prenorm): ~3s/chunk -> ~3 min each
- Non-KVCE (A/B):                          ~0.3s/chunk -> ~20s each
- Total:                                   ~13 min

Stored artefacts:
- `analysis/c11_wikitext_ppl_runs.jsonl` (append-only, smoke rows)
- `analysis/c11_wikitext_ppl_summary.json` (last run summary)
- `analysis/acu_kvce_attention.py`, `analysis/c11_wikitext_ppl.py`

---

## 2026-06-08 — Full sweep (n=64 chunks, 32,704 prediction tokens): C11 measured

Re-ran the harness at n=64 non-overlapping 512-token chunks of WikiText-2
test. PC FP16% = 0.00 % across all PC-using configs (B, E, E_prenorm)
except E at 0.14 % — matches the per-tile audit's finding that lossy S
nudges very few tiles from INT8 to FP16. Wall: 13.7 min total.

| Config       | PPL pooled | PPL median | Wall (s) | KVCE (s) | Notes                                       |
|--------------|-----------:|-----------:|---------:|---------:|---------------------------------------------|
| A baseline   |      17.56 |      18.87 |      2.4 |      0.0 | sdpa-equivalent, ground truth               |
| B PC only    |      17.58 |      18.92 |     15.5 |      0.0 | per-tile INT8 SV on true V -- harmless      |
| C KVCE naive |   3,642.10 |   3,582.34 |    217.8 |    189.8 | as-is bridge -- C1 fires                    |
| C_prenorm    |     877.88 |   1,024.99 |    192.5 |    163.0 | C1 removed; turbo4 noise remains            |
| E integrated |   4,368.83 |   4,179.28 |    203.7 |    145.7 | as-is chip pipeline                         |
| E_prenorm    |     892.11 |   1,053.17 |    204.8 |    146.1 | chip pipeline minus C1                      |

**Decomposition (PPL factors over baseline A):**

| Source                                            | Factor      | Cost in bits/tok       |
|---------------------------------------------------|------------:|-----------------------:|
| PC routing on full-precision V (B / A)            |    **1.001x** | +0.001                |
| C12 turbo4 noise floor alone (C_prenorm / A)      |   **49.99x** | +5.64                  |
| C1 Q4.12 clip alone (C / C_prenorm)               |    **4.15x** | +2.05                  |
| C12 + PC on lossy V (E_prenorm / A)               |   **50.81x** | +5.66                  |
| C1 inside the chip pipeline (E / E_prenorm)       |    **4.90x** | +2.29                  |
| Full integrated pipeline (E / A)                  |  **248.84x** | +7.95                  |

Findings:

1. **The chip's PC routing is end-to-end safe.** B vs A is 17.577 vs
   17.557 -- a 0.001x ratio, well inside chunk-to-chunk noise (median
   chunk PPL spans 5.4 to 35.5 across the 64 chunks). PC FP16 % on B
   is 0.00 %, i.e. 100 % INT8 routing. Independent of KVCE.

2. **C1 (Q4.12 input clip on real activations) is real and large.**
   In isolation: PPL_C / PPL_C_prenorm = 4.15x (+2.05 bits/tok). In
   the integrated pipeline: PPL_E / PPL_E_prenorm = 4.90x (+2.29 bits/tok).
   The pre-norm wrapper (analysis/kvce_pool.py mode="prenorm") removes
   it entirely -- so a per-vector scale-then-quantize bridge between
   ACU's k_proj output and KVCE's Q4.12 input is sufficient to close
   C1 in software. The chip would need this as either an ACU output
   sub-block (preferred per the conflict register's resolution path)
   or a KVCE input sub-block.

3. **C12 (turbo4 noise floor) is the dominant remaining cost.** Even
   with C1 removed, KVCE round-trip alone (C_prenorm) gives PPL 878 vs
   baseline 17.6 -- a 50x perplexity gap (+5.64 bits/tok). This is
   the intrinsic cost of 3-bit PQ + 1-bit QJL on K and 3-bit PQ on V
   at this model scale. Higher-precision modes (turbo8/turbo16) would
   shrink this, at the cost of the 3.6 to 5x compression ratio.

4. **PC routing inside the integrated path is approximately free even
   under lossy V.** E_prenorm / C_prenorm = 1.016x. The "PC adds
   nothing when KVCE noise dominates" prediction from the audit holds
   at the model-output level too -- but the safety verdict is what
   matters: PC routing does not make end-to-end PPL meaningfully
   worse than the KVCE-only baseline.

PC FP16% on E in this run is 0.14 % (47 out of ~33,800 tiles). The
audit measured 0.18 % at n=2,744 tiles. Same order, same direction.

Stored artefacts:
- `analysis/c11_wikitext_ppl_runs.jsonl` (now has the n=2 smoke +
  n=64 full rows; latest row per config is the full one)
- `analysis/c11_wikitext_ppl_summary_n64.json`
- `paper/figs/c11_ppl_by_config.{pdf,png}` (regenerable via
  `python analysis/c11_make_figs.py`)

Decision: with C1 isolation result in hand, fix order is clear ->
  (1) close C1 in the spec (per-vector pre-norm bridge),
  (2) re-measure with prenorm baked in,
  (3) decide on C12 (ship turbo4 + measured PPL hit, or escalate to a
      higher-precision mode).

HellaSwag (n=250) running in background. ETA ~75 min.

---

## Claims ledger

| # | Claim | Value | n | Source data | Status |
|--:|---|---|--:|---|---|
| L1 | Integration test reproduces published numbers within rounding | median rMSE_E = 0.3641 vs audit 0.3631 | 2,744 tiles | `analysis/integration_test_kv_pc_stats.json` (Jun 8 run) | single-seed confirmed |
| L2 | KVCE naive-mode cosine drops to 0.417 when input \|max\| exits Q4.12 range | cos = 0.417 (\|K\|~50) vs 0.975 (\|K\|~1) | 1 vec | this notebook, "KVCE pool" entry | smoke-only, n=1, not enough for a paper claim |
| L3 | Prenorm restores cosine to the in-range value | cos = 0.975 at \|K\|~50 | 1 vec | same | smoke-only, n=1 |
| L4 | Qwen2-0.5B WikiText-2 baseline (sdpa-equivalent via mode A) | PPL = 17.56 | 64 chunks (32,704 toks) | `analysis/c11_wikitext_ppl_runs.jsonl` (config A, latest row) | single-seed; bootstrap CI in figure |
| L5 | PC routing alone is harmless end-to-end | B/A PPL ratio = 1.001x (+0.001 bits/tok) | 64 chunks | same JSONL (configs A, B) | single-seed |
| L6 | C1 Q4.12 clip alone costs ~4.15x PPL on real activations | PPL_C / PPL_C_prenorm = 4.15x (+2.05 bits/tok) | 64 chunks | same JSONL (configs C, C_prenorm) | single-seed |
| L7 | C12 turbo4 noise floor alone costs ~50x PPL | PPL_C_prenorm / PPL_A = 49.99x (+5.64 bits/tok) | 64 chunks | same JSONL (configs A, C_prenorm) | single-seed |
| L8 | Integrated chip pipeline costs ~249x PPL vs baseline | PPL_E / PPL_A = 248.84x (+7.95 bits/tok) | 64 chunks | same JSONL (configs A, E) | single-seed |
| L9 | Prenorm bridge removes ~80 % of integrated PPL gap | E_prenorm / E = 0.204 | 64 chunks | same JSONL (configs E, E_prenorm) | single-seed |

L2/L3 will be re-measured with proper n + CI from the per-layer
activation statistics computed inside the e2e harness.

---

## Discipline log (incidents -> rules added)

- `~/.cache` is root-owned on this DGX Spark host. Rule: every HF run
  must set `HF_HOME=/home/chaithu/lhs/.hf_cache`.
- The KVCE Python ref is per-vector and CPU-bound; never run an e2e
  forward pass through it serially. Rule: always go through
  `kvce_pool.kv_roundtrip()` for any batch of >=64 vectors.
- The `/home/shadeform/...` hard-codes are an anti-pattern. Rule: use
  `$KVCE_REF` env var with sibling-clone fallback in every script that
  imports the KVCE ref.
