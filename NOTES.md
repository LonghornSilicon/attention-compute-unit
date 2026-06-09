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

## Claims ledger

| # | Claim | Value | n | Source data | Status |
|--:|---|---|--:|---|---|
| L1 | Integration test reproduces published numbers within rounding | median rMSE_E = 0.3641 vs audit 0.3631 | 2,744 tiles | `analysis/integration_test_kv_pc_stats.json` (Jun 8 run) | single-seed confirmed |
| L2 | KVCE naive-mode cosine drops to 0.417 when input \|max\| exits Q4.12 range | cos = 0.417 (\|K\|~50) vs 0.975 (\|K\|~1) | 1 vec | this notebook, "KVCE pool" entry | smoke-only, n=1, not enough for a paper claim |
| L3 | Prenorm restores cosine to the in-range value | cos = 0.975 at \|K\|~50 | 1 vec | same | smoke-only, n=1 |

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
