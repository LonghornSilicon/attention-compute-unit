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

## 2026-06-08 — Per-layer ablation: C12 is concentrated in 3 layers

After C11 closed, the open question for the chip's PolarQuant
centroid design (C10/C12) was: is the +5.64 bits/tok C12 cost
"spread evenly across layers" (-> bump everyone to turbo8) or
"concentrated in a few" (-> hybrid per-layer mode select)? Wrote
`analysis/c11_per_layer_ablation.py` + `_resume.py` to measure both
directions:

- **Single-layer:** KVCE on exactly one layer L (identity
  elsewhere) -> marginal cost of compressing L.
- **Leave-one-out:** KVCE on every layer except L -> recovery
  from skipping L.

Config: Qwen2-0.5B, mode `C_prenorm` (C1 already off, isolates the
C12 / centroid-fit cost), n=16 chunks x 512 tokens. Wall: ~17 min
once the model+pool are warm (single-layer is ~3.4 s, LOO is ~42 s
because KVCE runs on 23 layers per chunk vs 1).

**Reference points on this sample:**
- Baseline A: PPL = 21.06
- C_prenorm full: PPL = 1191.13
- Log-gap: +4.035 nats (different chunk sample than the n=64 sweep,
  so absolute numbers differ; gap-shape is what matters here)

**The active set is 3 layers out of 24:**

| Layer | single Δlog-PPL | LOO recovery |
|------:|----------------:|-------------:|
| **L0**  | **+2.761** | **+2.869** |
| **L1**  | +0.362 | +0.653 |
| **L23** | +0.041 | +0.258 |
| L10 | +0.058 | -0.079 (sample noise) |
| L16 | +0.011 | +0.003 |
| all other 19 | within ±0.02 | within ±0.04 |

Sum of LOO recoveries for {L0, L1, L23} = +3.78 nats out of the
+4.035 full gap = **93.7 %**.

**Two methodological things this exposed:**

1. **Errors compound.** LOO recovery > single-layer marginal for
   every active layer (L0 +0.11, L1 +0.29, L23 +0.22 extra). The
   single-layer marginal under-attributes the value of fixing a
   given layer. **LOO is the right metric for sizing per-layer
   intervention wins**, not the marginal.

2. **Per-tile rMSE is a poor proxy for end-to-end PPL** past L0.
   The integration-test rMSE study flagged L4 (rMSE 3.3x L12) and
   L20/L23 (rMSE 2.2x L12) as next-worst after L0. The ablation
   says only L23 matters; L4 and L20 are neutral. L1 -- which the
   rMSE study didn't sample -- is actually the second-worst layer.
   Next iteration of `analysis/integration_test_kv_pc.py` should
   sweep all 24 layers, not the 7-layer subset.

**Implication landed in
`kv-cache-engine/findings/centroid_design_brief.md` Section 9:**
Path C (hybrid per-layer mode select) with `turbo8` on {L0, L1,
L23} and `turbo4` everywhere else. Predicted ~94 % C12 recovery at
~12 % bit-rate impact. The KVCE-side designer can start from that
write-up.

Crash-recovery notes:
- First full run died after `loo_L0` landed (~17:20). Salvaged
  baseline + C_prenorm_full + all 24 single-layer rows + LOO L0
  from the append-only `c11_per_layer_ablation_runs.jsonl`. Wrote
  `analysis/c11_per_layer_ablation_resume.py` to read the file,
  identify missing LOO layers, and run only those. Took 23 layers
  x ~43 s = ~16 min, no other work re-run.
- Rule added: every multi-hour sweep that writes an append-only
  log gets a `_resume.py` sibling. Useful twice -- on crashes and
  on incremental config additions.

Stored artefacts:
- `analysis/c11_per_layer_ablation.py` (full sweep harness)
- `analysis/c11_per_layer_ablation_resume.py` (crash/incremental
  recovery harness)
- `analysis/c11_per_layer_ablation_runs.jsonl` (append-only per-run
  rows; first 7 rows are from the n=4 smoke)
- `analysis/c11_per_layer_ablation_stats.json` (final summary; n=16
  baseline + full + 24 single + 24 LOO)
- `analysis/c11_per_layer_ablation_stats.smoke.json` (preserved
  smoke summary for reference)
- `paper/figs/c11_per_layer_ablation.{pdf,png}` (two-panel: single
  marginal cost + LOO recovery, log-PPL nats)
- `analysis/c11_per_layer_ablation.log` (combined stdout from both
  the original run and the resume)

Decision: KVCE side now has a sized, evidence-backed design path
to start the centroid pipeline from. APA side stops here on C11;
re-open if the next-model ablation shows the active set isn't
{L0, L1, L23}.

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
| L10 | L0 single-layer KVCE marginal cost (C_prenorm mode) | +2.761 nats log-PPL (PPL 333 vs 21.06) | 16 chunks | `analysis/c11_per_layer_ablation_stats.json` | single-seed |
| L11 | L0 leave-one-out KVCE recovery (cross-layer compounding visible) | +2.869 nats (PPL 67.6 vs 1191) | 16 chunks | same | single-seed |
| L12 | L1 LOO recovery (second-worst layer) | +0.653 nats (PPL 619.8 vs 1191) | 16 chunks | same | single-seed |
| L13 | L23 LOO recovery (third-worst layer) | +0.258 nats (PPL 919.9 vs 1191) | 16 chunks | same | single-seed |
| L14 | Sum of LOO recoveries for {L0, L1, L23} as fraction of full C12 gap | 3.78 / 4.035 = 93.7 % | 16 chunks | derived from same | derived |
| L15 | Mid-network layers (L2-L22 excl. L10) LOO recovery individually | within ±0.04 nats | 16 chunks each | same | single-seed |
| L16 | Path C (`turbo8`/pq4 16-lvl on {L0,L1,L23}) gives no significant HellaSwag lift over the turbo4 floor | acc_norm 0.336 [0.280, 0.397] vs 0.316 floor (baseline A 0.420; chance 0.25); pq4 PPL ~3.5× better than pq3 yet task flat | 250 items | `analysis/c12_hellaswag_pq4_n250_summary.json` | single-seed, CI-bounded; **decisive negative — closes Path C** |
| L17 | ChannelQuant CQ-4 near-lossless on Qwen2-0.5B (D=64) | acc_norm 0.4170 [0.387,0.448] vs FP16 0.4260 [0.396,0.457]; Δ=−0.009 (within CI); 4.19 bits/val | 1000 items | `analysis/c20_kvce_q05_headline.json` | single-run, CI-bounded; **indistinguishable from FP16 = near-lossless** |
| L18 | ChannelQuant CQ-4+ near-lossless on Qwen2-0.5B (D=64) | acc_norm 0.4220 [0.392,0.453]; Δ=−0.004 (within CI); 4.38 bits/val | 1000 items | `analysis/c20_kvce_q05_headline.json` | single-run, CI-bounded; indistinguishable from FP16 |
| L19 | ChannelQuant on Qwen2-1.5B (D=128); CQ-4+ outlier lane closes ~half the CQ-4 gap | CQ-4 0.5050 (Δ−0.016), CQ-4+ 0.5130 (Δ−0.008) vs FP16 0.5210; paired cq4→cq4+ +0.008, McNemar p=0.28; 4.13/4.22 bits/val | 1000 items | `analysis/c20_kvce_q15_headline.json` | single-run, CI-bounded; deltas within CI; outlier-lane effect directional, not sig at n=1000 |
| L20 | Combined APA+KVCE holds FP16 accuracy on Qwen2-0.5B | apa-only 0.4420 (=FP16 exactly); cq4+APA 0.4360 (Δ−0.006); cq4+ +APA 0.4500 (Δ+0.008) — all within CI | 500 items | `analysis/c20_cq_apa_q05.json` | single-run, CI-bounded |
| L21 | APA precision controller routes ~all tiles to INT8 on Qwen (THRESHOLD=10) | int8_tile_frac = 0.9999 (FP16 escape essentially never fires); INT8 S·V lossless (apa-only = FP16) | 500 items, all S·V tiles | `analysis/c20_cq_apa_q05.json` | structural, robust; confirms 2026-06-18 "controller is the bottleneck, not the codec" on a 2nd workload |
| L22 | ChannelQuant measured compression matches the pre-build target | 4.13–4.38 bits/value (mean K+V) vs HW_CONTRACT §6 target ~4.2/4.4; ≈3.8×/3.6× vs FP16 | derived | `eff_bits` in `analysis/c20_kvce_{q05,q15}_headline.json` | derived; matches target |

L2/L3 will be re-measured with proper n + CI from the per-layer
activation statistics computed inside the e2e harness.
L10-L15 are n=16; bootstrap CIs not computed -- the +2.87 nat L0
result is unambiguous, but the +0.04 nat individual mid-network
deltas should be treated as within noise rather than as positive
findings.

---

## 2026-06-09 -- C10 plumbing + Path A revival via QJL alpha co-tune

End-to-end wire-up of the per-layer Lloyd-Max centroid retuning
pipeline (KVCE side landed in `kv-cache-engine@2bf08b5` and the
follow-on alpha override). Five new analysis scripts, two refactors:

- `analysis/kvce_pool.py`: per-layer engine support
  (`dict[layer_idx → engine]`), capture mode, centroid_tables_path arg
  to `get_pool` / `kv_roundtrip`. Honors `qjl_scale` per layer in the
  table JSON.
- `analysis/acu_kvce_attention.py`: `set_centroid_tables(path)`,
  `set_capture_mode(on)`, `pop_capture_buffer()`. Passes
  `module.layer_idx` to the pool so per-layer engines route correctly.
- `analysis/c10_capture_per_layer.py`: capture under C_prenorm with
  KVCE on every layer (production distribution).
- `analysis/c10_capture_clean.py`: capture under mode A via forward
  hooks (clean per-layer K/V before any KVCE noise).
- `analysis/c10_retune_centroids.py`: Lloyd-Max per layer + optional
  closed-form QJL alpha calibration (`centroid_lloyd_max.calibrate_qjl_scale`).
- `analysis/c10_run_ppl.py`: baseline_A + default + retuned PPL with
  per-config summary JSON.

### First test: naive Path A regresses

Smoke (n=4) with retuned centroids + default sqrt(pi/2) alpha:

| Retune scope          | PPL    | vs default | Verdict |
|-----------------------|-------:|-----------:|---------|
| all 24 layers         | 2090.8 | +0.997 nat | regress |
| {L0, L1, L23} only    | 2359.6 | +1.118 nat | regress |
| L0 alone              | 2520.8 | +1.184 nat | regress |

L0-only regressing rules out cascade and capture-quality artefacts.
Diagnosis: per-coord MSE dropped 12% on L0 but whole-vector cos
dropped 0.0020 -- the codec's QJL residual correction uses a fixed
sqrt(pi/2) scaling that assumes Gaussian residuals; retuning shifts
the residual distribution and miscalibrates QJL. Documented in
`kv-cache-engine/findings/path_a_qjl_coupling.md`.

### Second test: alpha co-tune flips the result

KVCE side shipped `qjl_scale` override + `calibrate_qjl_scale`
closed-form MSE-min alpha. Re-ran the same captures, now emitting
per-layer alpha alongside centroids. All calibrated alphas landed at
~0.49 (vs default sqrt(pi/2) ~1.253) -- matching the analytic
MSE-min for d=64 in the calibrator's docstring.

n=16 results:

| Config                                  | PPL    | log-gap | recovered |
|-----------------------------------------|-------:|--------:|----------:|
| baseline A                              |  21.06 |       0 |        -- |
| C_prenorm default (sqrt(pi/2), default) | 1191.13 | +4.035 |        -- |
| default centroids + alpha=0.49 (diag)   |  548.63 | +3.260 | **+19.2%** |
| dirty captures retune + alpha           |  506.87 | +3.181 | **+21.2%** |
| clean captures retune + alpha           |  567.85 | +3.294 | **+18.4%** |

Two findings:

1. **Alpha dominates.** Default centroids + alpha=0.49 alone
   recovers 19.2 of the 21.2 percentage points -- ~91% of the win
   comes from fixing the QJL scaling, not from retuning centroids.
   This collapses Path A's complexity: ship chip-default centroids
   plus a single per-model-calibrated alpha; the per-layer `tuser`
   extension for centroid routing is no longer required for this
   recovery level.

2. **Clean captures lost to dirty captures (surprise).** Clean
   captures from mode A (counterfactual: what L1 would see if L0
   were FP16) underfit the deployed-distribution L1 actually sees
   (where L0 is KVCE-corrupted). Rule: capture under the
   configuration you'll deploy under, not a clean reference.

### What this changes in the chip story

- C12 noise floor at full-deploy bit budget drops from +5.64 to
  +4.81 bits/tok on Qwen2-0.5B (21.2% recovery).
- Centroid override is still useful at +2% on top of alpha, AND was
  the infrastructure that made the alpha diagnostic possible.
- Path C with `turbo8` on {L0, L1, L23} now has +19% of "alpha
  alone" priced in -- the marginal value of `turbo8` is on the
  remaining +3.18 nats, not the original +4.04.

Stored artefacts:
- `analysis/c10_*` (all scripts + JSON tables + per-config summaries)
- `analysis/c10_captures{,_clean}.npz` (capture corpora)
- `analysis/c10_ppl_runs.jsonl` (append-only per-config rows)
- `analysis/c10_ppl_summary_{dirty,defaultplus,clean}_alpha_n16.json`
- `analysis/c10_n16_sweep.log` (combined log)

Cross-ref: `kv-cache-engine/findings/path_a_qjl_coupling.md`
sections 1-6 documents the negative-result diagnosis; the
revival-status banner at top of that doc cites the n=16 numbers
above.

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
- Per-coord MSE is NOT a valid proxy for end-to-end attention quality
  in this codec. The first centroid retune passed every existing
  per-coord gate AND regressed PPL by 2.7x. Rule: any centroid-table
  change must measure whole-vector cos (or equivalent rotation-
  preserving metric) before claiming success. A regression test on
  cos is requested in `kv-cache-engine/findings/path_a_qjl_coupling.md` §5.3.
- The QJL alpha is coupled to the centroid table. Any centroid
  override requires a matching alpha override; the codec's default
  sqrt(pi/2) only holds for Gaussian residuals. Rule: emit
  `qjl_scale` alongside `centroids` in every per-layer table; never
  ship one without the other.
- Capture corpora must come from the deployed configuration, not a
  clean reference. The clean-mode-A captures lost ~3 percentage
  points of recovery to the dirty C_prenorm captures because the
  deployed L1+ distributions are post-KVCE-noise, not FP16-clean.
- E_prenorm (PC routing) + per-layer retuned-α tables hangs GB10 at
  the first KV roundtrip; reproducible across two launches with
  different worker counts and CUDA allocator settings, both ended in
  hard reboots. Neither component alone crashes. Rule: don't ship
  combined PC × per-layer-table configs until the interaction is
  diagnosed; treat the E_prenorm-with-retuned numbers as unmeasured,
  not "zero".

---

## 2026-06-11 — Path A end of line: scale + HellaSwag close the case

Full Qwen2-0.5B scale-up of the α-revival result and the first task-
level measurement of the α-tables intervention. Closes Path A as a
productive direction and pivots the chip story to Path C.

### What was run

Two phases (script:
`analysis/c10_finish_full_qwen_test.sh`,
log: `analysis/c10_full_qwen_test.log`):

1. **WikiText n=64 perplexity** — repeated C10's n=16 sweep at 4×
   the chunk count, both `C_prenorm` and `E_prenorm` modes, with
   `c10_centroid_tables_defaultplusα.json` (chip-default centroids
   + closed-form MSE-min α ≈ 0.49 per layer).
2. **HellaSwag n=250 task accuracy** — first task-level measurement
   of α-tables. Configs A (no KVCE) and C_prenorm (KVCE only). Same
   `--tables` JSON as Phase 1.

Defensive env after a GB10 crash incident (see "Incident" below):
`KVCE_POOL_WORKERS=12`, `OMP_NUM_THREADS=1`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`,
`CUDA_DEVICE_MAX_CONNECTIONS=1`.

### Result — WikiText n=64

| Config | PPL pooled | log-gap vs A [nats] | recovery |
|---|---:|---:|---:|
| baseline A | 17.557 | 0 | — |
| C_prenorm default | 877.878 | +3.912 | — |
| **C_prenorm retuned + α** | **431.911** | **+3.203** | **+0.709 nats (+18.1%)** |
| E_prenorm default | 892.111 | +3.928 | — |
| E_prenorm retuned + α | *unmeasured* | — | hangs GB10 (see incident) |

The n=16 result (+21.2%) and the n=64 result (+18.1%) overlap within
the implied sampling slop on 32 704 tokens vs 8 176; the headline
α-revival recovery on `C_prenorm` is robust at scale.

### Result — HellaSwag n=250 (95% Wilson CI)

| Config | acc | acc_norm |
|---|---:|---:|
| baseline A | 0.404 [0.345, 0.466] | 0.420 [0.360, 0.482] |
| **C_prenorm + α tables** | **0.336 [0.280, 0.397]** | **0.316 [0.262, 0.376]** |

For reference (`c11_hellaswag_summary.json`, same seed/n, default
centroids no α): `C_prenorm` acc_norm = 0.316.

**The α-tables intervention and the chip-default centroids land at
the same task-level accuracy: acc_norm 0.316 vs 0.316.** Identical
to three decimals. The +18.1% PPL recovery does not translate to a
measurable HellaSwag gain.

Stored artefacts:
- `analysis/c10_ppl_summary_C_prenorm_alpha_n64.json` — Phase 1
  summary.
- `analysis/c10_ppl_runs.jsonl` — per-config rows (including the
  `E_prenorm_default` n=64 row before the first crash).
- `analysis/c11_hellaswag_summary_alpha.json` — Phase 3 summary.
- `analysis/c11_hellaswag_runs_alpha.jsonl` — per-config row +
  Wilson CIs.
- `analysis/c10_full_qwen_test.log` — appended log across both
  attempts.

### Interpretation

- Path A revival ("default centroids + closed-form α") **diagnoses
  the QJL coupling** the n=4 negative result identified: no 2.7×
  regression, +18% PPL recovery instead.
- Path A revival **does not change the task-level outcome**.
  HellaSwag acc_norm with α tables is identical to HellaSwag
  acc_norm with no intervention. The +3.2 nats still remaining
  between `C_prenorm` and baseline are doing all the task-quality
  damage; the +0.7 nats α saved are below the model's reasoning
  noise floor.
- The chip-side C12 noise floor revision (5.64 → 4.81 bits/tok)
  from the n=16 work is **PPL-accurate but task-level meaningless**
  at the bit budget Path A operates within. The remaining +3.2 nats
  is what matters and α has no more to give it.

**Decision: Path A is closed as a stand-alone intervention.** Pivot
to Path C (`turbo8` on the active layers {L0, L1, L23}) per the
brief's §9.3. Landing doc for the pivot:
`kv-cache-engine/findings/path_c_plan.md`. The α override and the
per-layer table JSON shape stay — they're the right contract for
Path C too (Path C needs per-layer mode = `turbo4 | turbo8`, which
is a strict superset of "per-layer centroid table"). The α
calibration step extends naturally to `turbo8` tables.

### Incident: GB10 hard hang on E_prenorm × per-layer-tables

`E_prenorm` (PC tile routing on top of KVCE) with retuned-α tables
hangs the GB10 within ~15 s of the first KV roundtrip, requiring a
hard reboot. Reproduced twice (2026-06-09 and 2026-06-10) across
two launch configurations:

| Attempt | Workers | CUDA env | Outcome |
|---|---|---|---|
| 1 (06-09 17:31) | 20 | defaults | Log freezes at "E_prenorm retuned" header; 5 reboots between 23:05 and 00:28 next day. |
| 2 (06-10 12:14) | 12 | `expandable_segments:False`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `OMP=1` | Log freezes at the same point ~15 s in; box rebooted at 15:22 (3 h later, likely watchdog or manual). |

Differential — what crashes vs what doesn't:

|  | KVCE on | PC routing | per-layer tables | Result |
|---|---|---|---|---|
| baseline A | — | — | — | runs |
| C_prenorm default | yes | — | — | runs |
| C_prenorm retuned+α | yes | — | yes | runs (Phase 1 of this entry) |
| E_prenorm default | yes | yes | — | runs (Phase 1 of this entry) |
| **E_prenorm retuned+α** | yes | yes | yes | **hangs, both attempts** |

Neither PC routing alone nor per-layer tables alone is the
trigger — it's the combination on this hardware. Possibilities
worth investigating (filed in `findings/path_c_plan.md` §
"Hardware investigations needed"):

1. UVM page fault: PC routing introduces extra host-side
   tile decisions per layer; in combination with the per-worker
   per-layer engine `_engines` dict (24 engines × 12 workers = 288
   engine objects' rotation matrices + boundary arrays) the working
   set may exceed what UVM can keep resident on GB10.
2. CUDA context corruption: PC's tile-selection branch may invoke a
   kernel path the regular `C_prenorm` doesn't, and that path may
   race against the per-vector CPU→GPU `to(device=device,
   dtype=dtype)` calls the KVCE roundtrip does.
3. Driver bug specific to GB10 + driver 580.142 + CUDA 13.0 +
   torch 2.12 — would need a minimal repro to file with NVIDIA.

`CUDA_LAUNCH_BLOCKING=1` was not tried (it would have ~3× slowed
the failing path); the next attempt to characterise this should run
with it set so the failure surfaces as a Python exception with a
PC trace instead of a kernel hang.

For the lab notebook: treat `E_prenorm` results from this campaign
as **unmeasured**, not "zero". The `E_prenorm_default n=64` row in
the JSONL is real (it survived a clean fresh-process run before the
first crash); `E_prenorm_retuned_α` does not exist in any data file
and is not estimable from this campaign.

Cross-refs: `kv-cache-engine/findings/path_c_plan.md` (pivot doc,
includes the hardware investigation as a Path-C-side dependency)
and `kv-cache-engine/findings/path_a_qjl_coupling.md` (Path A
diagnosis + revival history, now superseded by the pivot doc).

## 2026-06-17 — GB10 hard hang #3: RCA. PC routing was a red herring; the parametric `pq_bits=4` codec path hangs on plain C_prenorm

Third hard-down event on this workload (prior two: 06-09, 06-10,
both `E_prenorm`). This one reproduced on **plain `C_prenorm` (no PC
routing)** during a deliberate n-ladder of the Path C smoke, which
overturns the prior entry's conclusion that PC routing is the
trigger.

### What was run (provenance)
- `c10_run_ppl.py --mode C_prenorm --tables c12_centroid_tables_pathC_L0L1L23.json`
  (the new parametric table: per-layer `pq_bits=4`, 16-level).
- Ladder: n=4 first (passed), then n=16. Both with
  `CUDA_LAUNCH_BLOCKING=1` and a `timeout --signal=KILL 600` watchdog.
- Competing load present throughout: `macro-place-challenge`
  CPU jobs running (load avg ~13–14); they were writing
  `vis/snaps_ibm13/*` up to the freeze.

### Timeline (reconstructed; kernel log NOT persisted — see gap)
| Wall | Event | Evidence |
|---|---|---|
| 19:37–19:38 | **n=4** C_prenorm pq4: baseline/default/retuned all complete, exit 0 | `c12_path_c_probe_n4_L0L1L23.log` (full), `c12_ppl_summary_L0L1L23_n4.json`, 3 jsonl rows |
| 19:42:00 | **n=16** launched (`c12_probe_start.ts`) | log file created (`> redirect`) |
| ~19:42–19:43 | **hard hang**: zero flushed output, box wedges | `c12_path_c_probe_n16_L0L1L23.log` is **0 bytes**; no n=16 jsonl row |
| 19:42→19:56 | watchdog (`SIGKILL` @600s) powerless; box down | proc stuck in uninterruptible driver sleep |
| 19:56:40 | reboot (manual/watchdog), box healthy on return | `journalctl --list-boots`; GB10 46C P0, driver 580.159.03 |

### The smoking gun
The n=16 log is **0 bytes** and no n=16 row reached
`c12_ppl_runs_probe.jsonl`. The first print in `c10_run_ppl.py` is
`[setup] ... flush=True` — it flushed to the page cache but the page
never synced before the hard reset, so it was lost. A 0-byte log +
no jsonl row = the process produced no durable output before the box
died. That is the signature of a **device-level hard hang**, not a
recoverable CUDA error: `CUDA_LAUNCH_BLOCKING=1` was set and still
yielded no traceback, because a true driver/device lockup never
returns control to the CPU thread — there is no API error to raise.
Userspace instrumentation (LAUNCH_BLOCKING, `timeout`/SIGKILL) is
structurally unable to catch or interrupt this failure mode.

### Corrected differential (refutes the prior entry)
| KVCE | PC routing | table | n | Result | Source |
|---|---|---|---|---|---|
| yes | — | old pq3 (24-layer) | 4/16/64 | **runs at all three** | `c10_ppl_runs.jsonl` (2044/8176/32704 tok rows) |
| yes | — | **new pq4 (3-layer)** | 4 | runs | this entry, 19:38 |
| yes | — | **new pq4 (3-layer)** | 16 | **HANGS** | this entry, 19:42 |
| yes | yes | old pq3 | 64 | hangs (06-09/06-10) | prior entry |

Two variables now isolated:
1. The **parametric `pq_bits=4` / 16-level codec path** is
   necessary — the old `pq_bits=3` path (no `pq_bits` field, falls
   back to 8-level default) runs clean at n up to 64.
2. **Sufficient workload** is also necessary — the same pq4 table
   survives n=4 and only hangs at n≥16. The trigger accumulates
   over KV roundtrips; it is not hit in the first few.

So PC routing was **not** the root cause — it was one of (at least)
two independent ways to push the GB10 over the same edge. Plain
C_prenorm + the larger 16-level codebook is the other. The original
"verify on plain SDPA / C_prenorm first, it's safe" guardrail is
**false** for parametric pq4 tables.

### Root cause — leading hypothesis (not yet proven)
**Unified-memory / driver resource exhaustion on GB10**, reached
once per-roundtrip GPU footprint × roundtrip count crosses an
envelope. The pq4 path uses a 16-level codebook (2× the pq3
codebook) and exercises the new parametric allocation/index path
(kvce commit `604093c`); if it allocates a per-roundtrip GPU buffer
that isn't freed, ~N roundtrips exhaust UVM → page-fault thrash →
driver hang. Consistent with the original entry's hypothesis #1 and
with the n-threshold (n=4 below it, n≥16 above). PC routing reaches
the same envelope via 288 per-worker engine objects instead.

### The logging gap (top remediation)
We **cannot confirm the mechanism** because no Xid was captured:
`journalctl -b -1 -k` → no entries; `/var/log/journal` exists but
kernel messages for boot -1 were not flushed before the crash; no
`/var/crash` dump, no `nvidia-bug-report`. So we cannot yet tell
UVM-OOM (Xid 31/13) from "GPU fell off the bus" (Xid 79) from a
kernel panic. **Before the next attempt:** enable persistent kernel
journald + an fsync'd external heartbeat, so the next occurrence
yields an Xid.

### Actions (next session)
1. Enable Xid capture (persistent kernel log + nvidia-smi `-l 1`
   logger to an fsync'd file) — without it every repro is wasted.
2. **Isolate**: never co-run the GPU probe with the macro-place CPU
   load — removes the contention confound entirely.
3. **Decisive bisection**: run n=16 with a 3-layer table forced to
   `pq_bits=3` (old path, new layer set). If it runs → the pq4/
   16-level path is the cause, not n or the 3-layer set. If it
   hangs → workload threshold dominates regardless of pq_bits.
4. Stop blind n-laddering until (1)–(3); each hang costs a reboot
   and risks co-tenant jobs.

### For the lab notebook
Path C has **no valid n≥16 measurement** on the pq4 table. The only
real Path C numbers are n=4: L0-only +18.0% gap recovery, and
`{L0,L1,L23}` +37.5% gap recovery (PPL only — HellaSwag unmeasured,
and PPL is known to overstate at this bit budget). Treat all n≥16
Path C cells as **unmeasured**, not zero.

## 2026-06-17 — Path C CLOSED: turbo8 gives no task lift. PPL lied a third time. Pivoting.

Got the one missing measurement — HellaSwag task accuracy for the
best Path C config — without touching the GB10 (CPU/fp32, the path
that hangs was never invoked).

### Result (HellaSwag, n=250, CPU/fp32, same 250 items, seed 0)
| config | acc_norm | 95% CI | vs baseline |
|---|---:|---|---:|
| baseline A (no KVCE) | **0.420** | [0.360, 0.482] | — |
| turbo8 / pq4 on {L0,L1,L23} | **0.336** | [0.280, 0.397] | −0.084 |
| turbo4 / pq3 floor (Path A) | 0.316 | [0.262, 0.376] | −0.104 |
| chance | 0.250 | — | — |

CPU control re-ran the pq3 floor under identical fp32/CPU conditions
and landed **0.308 [0.254, 0.368]** (matches the 0.316 GPU number);
CPU baseline A reproduced 0.420 exactly, so fp32-CPU ≡ fp16-GPU here
and the comparison is valid. Same-condition gap pq4−pq3 = +0.028,
CIs heavily overlapping — no lift.

### The decision and why
turbo8 moved acc_norm by **+0.020 (0.316 → 0.336), inside the CI** —
statistically indistinguishable from the floor. This is despite pq4
perplexity being **~3.5× better** than pq3 (n=4: PPL 187 vs 659).
That is the **third** time in this project PPL improved while task
accuracy did not move (Path A +18% PPL / 0% acc; bit-budget axis
here). The lesson is now firm: **the failure is not bit-count — it's
what the PQ+QJL structure discards.** It throws away information the
task needs and that perplexity cannot see. More bits on the
expensive layers was the whole Path C thesis; it is now measured and
refuted. Path C is **closed**.

### What's salvageable (not wasted)
- The codec is bit-exact-verified (K1–K18); the per-layer table
  format, α-calibration, capture pipeline, and the CPU eval harness
  all stand and transfer to whatever comes next.
- The PPL↔accuracy divergence is itself a reusable, hard-won
  finding — it tells the next scheme to be **evaluated on task
  accuracy from measurement #1**. PPL has earned zero trust here.

### Pivot under evaluation: DWB ("Don't Waste Bits!", arXiv:2604.04722)
Adaptive **per-token** importance-driven bit allocation across
{2,4,8,FP16}, controller predicts token importance (entropy /
rarity / attention-variance / confidence). Reports near-lossless
HellaSwag (41.2 vs 41.5 FP16) and +7.6 pts over static-4bit on
same-scale SmolLM-360M. It is almost a direct answer to our failure
mode (static low-bit destroys task accuracy; importance-aware
allocation + keeping some tokens FP16 recovers it).
**Open architecture question, not yet resolved:** DWB's per-token
variable bit-width does not map cleanly onto KVCE's fixed
turbo4/turbo8 hardware modes, and it needs a learned controller at
runtime. The algorithm clearly works; whether it fits *this chip's*
fixed-function contract is the real question for the pivot.

### Reproduce
`analysis/c12_hellaswag_pivot.log`,
`analysis/c12_hellaswag_pq4_n250_summary.json`,
`analysis/c12_hellaswag_pq3cpu_n250_summary.json`. Eval harness
patched to fp32 on CPU (`c11_hellaswag.py`, guarded — GPU path
unchanged).

## 2026-06-17 — TurboQuant is NOT a bust: DWB per-token routing reproduces near-FP16. Path C "pivot away" revised.

Same day as the Path C closure above — and it revises that closure's
strategic read. The codec was never the problem; **uniform usage was.**

Reproduced themoddedcube's prior DWB+TurboQuant integration
(`/home/chaithu/lhs/dont-waste-bits`, branch `turboquant-integration`)
on the current stack (transformers 5.10.2, CPU/fp32). Both hypotheses
re-confirmed:

| test | condition | acc | FP16 | avg_bits |
|---|---|---:|---:|---:|
| TQ-H2 (HellaSwag n=100) | **DWB-TurboQuant** | **42.0%** | 42.6% | 5.05 |
|  | DWB-scalar (INT2) | 40.0% | | 5.05 |
| TQ-H3 (ARC-Challenge n=100) | **DWB-TurboQuant** | **29.0%** | 35.0% | 7.72 |
|  | DWB-scalar | 25.0% | | 7.72 |

DWB-scalar reproduced to the decimal (40.0%, bit-dist {2:57.3, 4:18.9,
8:8.3, 16:15.6}); FP16 ARC exact (35.0%). DWB-TurboQuant beats scalar
by **+2pp (HellaSwag) / +4pp (ARC)** at identical bits and identical
routing — the gain is purely from quantizing the 2-bit tier with
PolarQuant instead of scalar INT2.

**Reconciliation with Path C.** Path C applied turbo *uniformly*
(every token, no importance signal, no FP16 escape) → 0.336. DWB keeps
important tokens at FP16/INT8 and routes only low-importance tokens
through TurboQuant → near-FP16. Same codec, opposite outcome. The
missing pieces were the **controller** and the **FP16 bypass**, not
more bits.

**So the pivot is NOT away from the KVCE codec** — it is to put a
per-token importance controller in front of it. Integration plan:
`kv-cache-engine/findings/dwb_turboquant_integration_plan.md`. The
make-or-break is **the full test**: DWB-routed KVCE on Qwen2-0.5B
HellaSwag (CPU), must recover 0.336 → ~0.420. Gating dependency: the
SmolLM-360M controller does not transfer; retrain for Qwen2-0.5B. For
silicon use {4,8} tiers (the 2-bit tier gives zero BRAM benefit). PPL
stays banned as a metric.

Reproduce: `dont-waste-bits/tq_repro.log`,
`dont-waste-bits/research/data/tq_h2_eval_100samp_*.json`.

## 2026-06-18 — THE FULL TEST (DWB-routed KVCE on Qwen2-0.5B): negative. Bottleneck = the controller, not the codec.

Ran the integration-plan §5.4 full test (CPU/fp32, HellaSwag n=250, same
items as the Path C baselines; harness `analysis/c13_dwb_routed_hellaswag.py`).
DWB controller (`dont-waste-bits/.../dwb_controller_qwen2-0.5b.pt`, trained
train-only, val_acc 0.42) predicts per-token tiers; routed through the real
KVCE codec via `akvce.set_token_bits`. The SmolLM TQ-H2 win does **not**
transfer to Qwen.

| variant (DWB-routed config) | DWB acc_norm | uniform turbo4 | FP16 |
|---|---:|---:|---:|
| baseline {2,4,8,16} | 0.316 | 0.352 | 0.420 |
| no-2-bit {4,8,16} (`--floor-tier 4`) | 0.360 | 0.352 | 0.420 |
| keep-2-bit, important→FP16 (`--bypass-above 8`) | 0.308 | 0.352 | 0.420 |
| no-2-bit + important→FP16 (`--floor-tier 4 --bypass-above 8`) | 0.344 | 0.352 | 0.420 |
| **ORACLE** sink+recent 50% FP16 (`--oracle-frac 0.5`, avg **10.0 bits**) | **0.348** | 0.352 | 0.420 |

All DWB variants ≈ or below uniform turbo4, all ~0.06–0.11 below FP16
(CIs ~±0.057, so DWB ≈ uniform, both ≪ FP16).

**Diagnosis by elimination:**
1. **turbo2 tier is catastrophic** — any config with ~42% of tokens at
   pq_bits=2 floors at ~0.31. (Matches the FPGA-branch 2-bit finding.)
2. **High-tier fidelity is NOT the blocker** — the last row routes the
   controller's "important" 58% of tokens to lossless FP16 (more precision
   than uniform's 0%) and *still* scores 0.344 < uniform 0.352. More
   lossless tokens producing worse accuracy ⇒ the controller protects the
   **wrong** tokens.
3. Initially looked like the controller's ranking — **but the oracle row
   refutes even that** (see UPDATE).

**UPDATE — oracle ceiling test settles it: the codec is the wall, not the
controller.** A sink+recent importance heuristic protecting 50% of tokens
at lossless FP16 (avg **10.0 bits/token**) scored **0.348 ≈ uniform turbo4**
— zero recovery. The oracle is the *ceiling* of any controller (best-case
token selection), so:
- **Retraining the controller will NOT help** — its ceiling is already at
  the floor. The pre-agreed "if routing fails, retrain" fallback is void.
- The accuracy loss is **diffuse across tokens**: the 50% left at turbo4
  cap accuracy on their own, and protecting the other 50% losslessly
  recovers nothing. Token-selection (DWB's whole premise) cannot rescue
  this — you'd need ~all tokens at FP16, i.e. no compression.
- Most likely cause: **GQA**. Qwen2-0.5B has 2 KV heads shared across all
  query heads, so KV quantization damages every query head in the group —
  structurally far more destructive than SmolLM's MHA. DWB+TurboQuant's
  success appears **MHA-specific**. Consistent with Path C (uniform KVCE
  wrecked Qwen).

**Next (strategic fork, open):**
1. One isolation run: oracle with the unprotected half at **scalar INT4**
   (SmolLM's recipe) instead of the turbo vector codec. Recovers → the KVCE
   vector codec is the problem on Qwen (fixable via scalar tiers); fails →
   GQA sensitivity, codec doesn't suit GQA models.
2. Reframe the chip to **MHA targets** where DWB+KVCE demonstrably works.
3. Conclude: DWB+KVCE does not transfer to Qwen2-0.5B; ship SmolLM, scope
   Qwen out.

**RESOLVED (2026-06-18) — it's the CODEC, not GQA, and DWB is NOT a no-go.**
Scalar-INT4 isolation (`analysis/c14_scalar_oracle_qwen.py`, n=250, no KVCE,
same items): the SAME oracle routing recovers with scalar INT4 where it
failed with the KVCE codec.

| config | acc_norm | note |
|---|---:|---|
| FP16 | 0.420 | — |
| uniform scalar INT4 | 0.360 | ≈ KVCE turbo4 (0.352) — uniform, both lossy |
| **oracle scalar INT4** (50% FP16) | **0.412** | ≈ FP16; KVCE oracle was 0.348 |

The dissociation: **identical routing + protected set; only the quantizer
for the unprotected half differs.** Scalar INT4 → 0.412 (recovers), KVCE
PolarQuant+QJL → 0.348 (no recovery). Mechanism (hypothesis): the per-head
WHT rotation + QJL **delocalizes** quantization error across the head
dimension, so protecting the important tokens doesn't catch their error;
scalar error stays token-local and IS caught. The codec's cleverness
backfires precisely when combined with per-token protection on GQA.

**Verdict: TurboQuant+ (PolarQuant+QJL) is a bust for GQA.** On SmolLM/MHA
the vector codec *beat* scalar (+2pp at 2-bit, TQ-H2); on Qwen/GQA scalar
INT4 + routing wins decisively. GQA is harder (uniform INT4 still −0.06)
but recoverable. **Path forward: keep DWB routing, drop the vector codec,
use scalar-INT tiers.** Caveat: oracle recovery is at ~10 bits avg (50%
FP16) — mechanism proven, compression-at-budget still to show with the
learned controller (next: c15, DWB learned controller + scalar tiers).

Data: `analysis/c14_scalar_oracle_qwen_summary.json` (+ `.log`).

## 2026-06-18 — KVCE revamp research batch (c16, GPU): uniform INT8 dominates; naive INT4 collapses at scale; DWB routing not worth it

Ran on the **GB10 GPU** (safe — plain HF inference + scalar hooks, never the KVCE
codec). Box stable throughout (no hang). Harness `analysis/c16_research.py`,
HellaSwag/ARC, n=250, fp16. Four runs: Qwen2-{0.5B,1.5B,7B}.

### Pareto across scale (HellaSwag acc_norm)
| config | 0.5B | 1.5B | 7B | avg bits |
|---|---:|---:|---:|---:|
| FP16 | 0.416 | 0.540 | 0.612 | 16 |
| **uniform INT8** | **0.420** | **0.524** | **0.596** | 8 |
| uniform INT4 (naive per-tensor) | 0.364 | **0.236** | **0.276** | 4 |
| oracle 50% FP16 + INT4 | 0.372 | 0.360 | 0.428 | 10 |
| oracle 65% FP16 + INT4 | 0.364 | 0.384 | 0.496 | 11.8 |
| oracle 50% FP16 + INT2 | 0.376 | 0.340 | 0.472 | 9 |
(ARC-Easy 0.5B mirrors HellaSwag: FP16 0.468, INT8 0.456, INT4 0.408, routing 0.36–0.44.)

### Three robust findings
1. **Uniform INT8 ≈ FP16 at every scale (2×, lossless, zero routing), and it
   DOMINATES every adaptive/routed config at every scale.** No routed point beats
   uniform INT8 on the acc/bits frontier — they cost more bits and score lower.
   The c15 "deployable win" (0.416 @ 7.6b) is, in this light, just matching
   uniform INT8 (0.420 @ 8b) — the controller buys nothing.
2. **Naive per-tensor INT4 COLLAPSES at scale.** 0.364 (0.5B) → 0.236 (1.5B) →
   0.276 (7B) — at/near chance (0.25) for the larger models, while INT8 stays
   fine. This is the classic **outlier-at-scale** signature: a per-tensor 4-bit
   scale can't represent growing outlier channels, crushing the rest. Exactly
   what KIVI/KVQuant-style **per-channel / outlier-aware** quant exists to fix.
   My earlier prediction (INT4 gets easier at scale) was WRONG — the opposite.
3. **Routing recovers from the INT4 collapse at scale but never beats INT8.** At
   7B, protecting 65% at FP16 lifts INT4 0.276→0.496, but that's 11.8 bits and
   still < uniform INT8's 0.596 @ 8 bits. Routing is strictly dominated.

### Rotation isolation (#2) — RETRACTION of the earlier claim
At matched 16 levels (oracle 50%): rotated-INT4 (rint4, PolarQuant b=4) = **0.408**
vs scalar-INT4 = 0.372 — i.e. **rotation did NOT hurt** (slightly higher, within
CI ±0.057). The earlier "rotation smears error, scalar wins" story (c13 KVCE
codec 0.348 vs c14 scalar 0.412) was **confounded by levels**: KVCE turbo4 is
pq_bits=3 = **8 levels**, not 16. At matched levels rotation ≈ scalar. **Soften
the rotation-mechanism claim in the turboquant-plus `docs/09` writeup** — the
clean story is "fewer effective levels + GQA," not "rotation is the culprit."

### Implication for the KVCE revamp (the big reframe)
- **2× lossless is free** at all scales via plain **uniform INT8** — ship that as
  the floor; it beats everything fancy we tried.
- **Drop DWB adaptive per-token routing** — dominated by uniform INT8 at every
  scale; the controller + per-token machinery isn't earning its complexity.
- **The TurboQuant+ vector codec** is not the answer either (no better than
  scalar at matched levels; its 8-level config loses).
- **The only credible path to 3–4×** is a **better 4-bit quantizer**:
  per-channel / outlier-aware INT4 (KIVI/KVQuant family), because the thing
  blocking 4-bit is outliers, not bit-allocation. That — not adaptive precision,
  not vector rotation — is what the revamp should target.

### Loose ends
- Minor harness bug: the final `rotation check` print in c16 raises KeyError when
  `--skip-rotation` is set (after the summary json is already written — **no data
  lost**; configs + JSONL intact). Fix before reuse.
- Next research (pre-revamp): **per-channel/outlier-aware INT4** test on 1.5B/7B —
  does it close the INT4 collapse and unlock clean 4× (or even 3-bit)? That's the
  decisive experiment for whether 3–4× is reachable at all.

Data: `analysis/c16_q05_hs_summary.json`, `c16_q05_arc_summary.json`,
`c16_q15_hs_summary.json`, `c16_q7_hs_summary.json` (+ `c16_*_runs.jsonl`,
`c16_batch.log`).

## 2026-06-19 — DECISIVE: 4× is reachable. Per-channel-key INT4 (KIVI) recovers ~FP16 at every scale; microscaling FP4 does NOT win

Quantizer sweep (`analysis/c17_quantizer_sweep.py`, GPU, HellaSwag n=250, uniform
— no routing). GPU freed overnight (other users' vLLM released); watcher
auto-launched. The collapse was 100% a quantizer-granularity problem.

| 4-bit variant | 0.5B | 1.5B | 7B | bits |
|---|---:|---:|---:|---:|
| FP16 | 0.416 | 0.540 | 0.612 | 16 |
| INT8 | 0.420 | 0.528 | 0.600 | 8 |
| naive INT4 (per-token) | 0.372 | **0.248** | **0.212** | 4 |
| **per-channel INT4** | 0.436 | **0.536** | **0.604** | 4 |
| **KIVI (per-ch K / per-tok V)** | 0.408 | **0.540** | **0.600** | 4 |
| **outlier (top-2 ch FP16 + per-ch)** | 0.428 | **0.552** | **0.616** | 4 |
| MXFP4 (microscale, block-32) | 0.336 | 0.500 | 0.288 | 4.25 |
| NVFP4 (microscale, block-16) | 0.384 | 0.412 | 0.380 | 4.5 |

### Findings
1. **4× (4-bit) IS reachable on Qwen GQA** — per-channel INT4 / KIVI / outlier all
   recover to ~FP16 at every scale, **near-lossless at 7B** (kivi 0.600,
   per-channel 0.604, outlier 0.616 vs FP16 0.612). The naive collapse
   (0.248/0.212 at 1.5B/7B) was purely per-token scale failing to contain
   per-channel **key outliers** — exactly KIVI's diagnosis. Fix = scale keys
   per-channel.
2. **Microscaling FP4 (MXFP4/NVFP4) did NOT win** — mxfp4 0.336/0.500/0.288,
   nvfp4 0.384/0.412/0.380; erratic, collapses at 7B. CONTRADICTS my earlier
   recommendation. Reason: a contiguous 16/32-elem block still lumps an outlier
   channel with its neighbors (doesn't isolate per-channel like true
   per-channel does), and the e2m1 4-bit-float grid is coarse. **Caveat:** this
   is my fake-quant *simulation*; real Blackwell NVFP4 (two-level FP8+FP32
   scales, HW rounding) may do better — but per-channel INT4 is the simpler,
   clearly-working winner regardless.
3. **INT8 ≈ FP16 at all scales** reconfirmed (the banked 2× floor).

### Revamp target (now evidence-backed — corrects the FP4 rec)
**Per-channel-key INT4 (KIVI-style: per-channel K, per-token V), optionally with
top-k outlier-channel isolation (KVQuant-style).** Clean **4×** at ~FP16 across
0.5B→7B. NOT microscaling FP4, NOT the TurboQuant vector codec, NOT DWB routing.
This is what the KVCE revamp should implement; INT8 stays as the safe 2× fallback.
Figure: `analysis/fig_quantizer_fix.png`. Data: `analysis/c17_q{05,15,7}_summary.json`.

Data: `analysis/c13_dwb_routed_hellaswag_summary.json`,
`c13_dwb_routed_floor4_summary.json`, `c13_dwb_routed_bypass8_summary.json`,
`c13_dwb_routed_floor4_bypass8_summary.json` (+ matching `.log`s).

---

## 2026-06-22 — ChannelQuant revamp de-risk: KEY outlier channels are input-independent (static-ROM mask validated)

Before scaffolding the new codec repo, ran the one blocking gate that decides
the hardware story for the CQ-4+ outlier tier: **are the high-magnitude key
channels a property of the weights (stable across inputs) or of the activation
(drift with input)?** If stable → calibrate the outlier mask offline, ship as a
per-layer ROM, no runtime argsort in silicon. If drift → runtime top-k needed
(bad for HW), ship CQ-4 without the "+".

Harness `analysis/c19_outlier_stability.py` (GPU/CPU safe, plain k_proj forward
hooks, NO KVCE codec). 8 independent HellaSwag-context batches × 16 items; per
(layer, KV-head) channel magnitude = mean_t |k[:,c]|; consensus top-2 = top-2
by magnitude summed over all batches; stability = mean over batches of
|topk(batch) ∩ consensus| / 2. Gate = mean ≥ 0.90.

| Model | mean stab | layer-0 stab | conc (median / p90) | gate |
|---|---|---|---|---|
| Qwen2-0.5B | 0.958 | **1.00** | 5.4× / 25× | PASS |
| Qwen2-1.5B | 0.986 | **1.00** | 7.8× / 12× | PASS |
| Qwen2-7B   | 0.984 | **1.00** | 8.0× / 22× | PASS |

**PASS at all three scales.** Three findings strengthen it beyond a bare pass:
(1) **Layer 0 is perfectly stable (1.00) at every scale** — the layer with ~16×
the quant noise (the one that dominated the TurboQuant+ collapse, see 2026-06-17
centroid brief) has the most extreme AND most pinned outliers; the hardest layer
is the easiest to mask. (2) **Concentration sharpens with scale** (5.4→8.0× the
median channel) — bigger models, where naive INT4 collapsed worst, have *cleaner*
outliers for the mask to catch. (3) **Stability rises with scale** (0.958→0.986),
no 7B drift.

Decision: **static outlier ROM is valid → CQ-4+ ships in v1.** This was the last
algorithmic unknown before building. The revamp is now de-risked at the software
level; remaining risks are pure HW (residual-group buffer cost, §5.1 of the
spec), retired in synth not simulation.

Spec: `/home/chaithu/lhs/channelquant_revamp_spec.md` (§7 Step 0 marked PASSED).
Data: `analysis/c19_{q05,q15,q7}_summary.json`.

---

## 2026-07-05 — C20: ChannelQuant verified end-to-end on Qwen; combined APA+KVCE holds FP16; PC routes ~all-INT8 on Qwen

The KVCE block's ChannelQuant revamp is complete and RTL-signed-off (kv-cache-engine
master, all CI gates green). Closed the loop with a **software end-to-end accuracy
verification on Qwen2**, and built the **combined APA+KVCE system** to see whether the
two blocks compose. All runs on GB10 (one H100-class GPU), venv torch 2.12 /
transformers 5.10, HellaSwag `acc_norm`, deterministic greedy NLL scoring (no inference
sampling → uncertainty is item-sampling Wilson/paired CI, not seed variance). Data +
scripts persisted in `analysis/c20_*` and `paper/figs/c20_apa_kvce_system.py`.

**Method (no channelquant edits).** KVCE half = the frozen `../channelquant`
`c23_headline.py` hooks (`fq_per_channel` INT4 keys grouped G=128, `fq_per_channel_outlier`
for CQ-4+ with the static k=2 ROM mask, `fq_per_token` INT4 values). APA half =
`acu_kvce` mode B (flash-streaming with the precision-controller ratio test picking
INT8/FP16 per S·V tile). Combined bridge is `analysis/c20_cq_apa_e2e.py` (a 2×3 grid:
{APA off/on} × {no-KVCE, CQ-4, CQ-4+}). None of this touches the retired TurboQuant
codec, so no `pq_bits=4`/`C_prenorm` GB10 hang path (see 2026-06-17 hang RCA).

**Finding 1 — ChannelQuant is near-lossless on Qwen, at both scales (n=1000).**
Qwen2-0.5B (D=64): FP16 0.4260 [0.396,0.457]; CQ-4 0.4170 (Δ−0.009); CQ-4+ 0.4220
(Δ−0.004). Qwen2-1.5B (D=128): FP16 0.5210 [0.490,0.552]; CQ-4 0.5050 (Δ−0.016);
CQ-4+ 0.5130 (Δ−0.008). Effective bits 4.13–4.38 (~3.8× vs FP16's 16). **All CQ
deltas fall inside the paired CI → statistically indistinguishable from FP16**, i.e.
the "near-lossless at ~4 bits" pre-build target (channelquant HW_CONTRACT §6:
~4.2/4.4 bits, ~3.6–3.8×) is *met*, not a measured regression. The CQ-4+ outlier lane
closes about half the CQ-4 gap at D=128 (paired cq4→cq4+ +0.008, McNemar p=0.28 —
directionally as predicted but not significant at n=1000). Data:
`analysis/c20_kvce_{q05,q15}_headline.json`.

**Finding 2 — the combined APA+KVCE system holds FP16 accuracy (n=500, Qwen2-0.5B).**
2×3 grid, `analysis/c20_cq_apa_q05.json`: fp16 0.4420; APA-only 0.4420 (= FP16
exactly); cq4+APA 0.4360 (Δ−0.006); cq4+ +APA 0.4500 (Δ+0.008). Stacking per-channel
INT4 KV *and* per-tile INT8-routed S·V shows no collapse — deltas within the n=500 CI.

**Finding 3 (anomaly, then structural result) — APA routes ~99.99% of tiles to INT8
on Qwen.** First pass reported a hard int8_frac = 1.000, which I traced to a **bug in
the existing `acu_kvce` mode-B routing**: the causal mask's −inf leaks into the PC
ratio test (`S_abs_max → inf → nan` int8 scores), so `max·N > 10·sum` never holds and
every tile falls through to INT8 — the *output* is still correct (masked P=0) but the
routing fraction is meaningless. `c20_cq_apa_e2e.py` computes the ratio test over the
**valid (unmasked)** scores; the fraction becomes real but stays **0.9999** at
THRESHOLD=10. So on Qwen the FP16 escape genuinely almost never fires and INT8-S·V is
essentially lossless (APA-only = FP16 exactly). This is a concrete second-workload
confirmation of the 2026-06-18 finding: *on this workload the controller adds little
discrimination — the codec, not the controller, is where the bits are.*

**Expectations vs measured.** KVCE matched precisely: bits/value 4.13–4.38 vs target
~4.2/4.4; "recovers ~FP16 at every scale" held; the D=128-helps-more prediction for
CQ-4+ held. APA's *accuracy* claim ("zero loss") matched exactly; APA's *routing-mix*
claim ("~79% INT8 / 21% FP16" — measured on its own benchmark) did **not** transfer to
Qwen, where it is ~100% INT8.

**Speedup (two orthogonal axes, do not multiply).** (a) KV-cache: ~3.8× smaller /
lower-bandwidth from ChannelQuant's ~4.2 bits/value vs FP16's 16, near-lossless — the
verified headline. (b) Attention S·V compute: ~all-INT8 on Qwen → roughly 2× cheaper
MACs (INT8 vs FP16) with zero accuracy loss. Different resources (memory vs compute),
so the system win is "~3.8× KV memory + INT8 attention math", not a single product.

**Hygiene caveats (honest bounds).** Scoring is deterministic given (model, item set),
so there is no inference seed; the CIs are item-sampling (Wilson) / paired (McNemar +
bootstrap). Each accuracy number is a **single item-set draw** (one n=1000 per model,
one n=500 combined) — NOT re-run on disjoint subsets, so ledger status is
"single-run, CI-bounded." The near-lossless claim is a *within-CI* result (indistinguishable
from FP16), which is exactly the target; it is not a claim that CQ measurably beats or
trails FP16 at these n. Finding 3's ~100% INT8 is structural (all tiles across items),
robust to the item draw.

Next: (optional) combined test on Qwen2-1.5B (D=128); a THRESHOLD sweep to find where
APA's FP16 escape begins to fire on Qwen (currently never at 10); land the mode-B mask
fix into `acu_kvce_attention.py` proper if the combined harness is kept.

Figure: `paper/figs/c20_apa_kvce_system.py` → `c20_apa_kvce_system.{png,pdf}`
(regenerates from `analysis/c20_kvce_q05_headline.json`).
