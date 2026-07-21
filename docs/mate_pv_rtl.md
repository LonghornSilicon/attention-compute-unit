# `mate_pv` — synthesizable INT8 P·V MAC tile (RTL)

**Status:** RTL complete, bit-exact to the reference, Yosys-clean (513 FFs, no latches).
**Home:** `rtl/mate_pv.sv` (+ `rtl/tb/tb_mate_pv.sv`, `rtl/tb/gen_mate_pv_vectors.py`).
**One line:** the token-reduction vector-MAC core of the MatE matrix engine, in
hand-written synthesizable Verilog for the Sky130 flow — the piece of the P·V datapath
that was previously **HLS-only**.

## Why this exists

The attention output stage `o[n] = Σ_t A[t]·V̂[t][n]` (P·V) was the one block with no
synthesizable RTL — the real MatE (`architecture/src/blocks/mate`) is a SystemC/Stratus
HLS project targeting Cadence N16FFC, so the cross-block cosim could not exercise the
actual accumulation between the KVE (rotated V̂) and the `wht_inverse_out` unspin. This
block fills that gap: a real INT8 MAC with the correct accumulator width, in the same
OpenLane/Sky130 flow as the other blocks, so the end-to-end cosim runs through a genuine
P·V accumulation rather than copying V̂ straight across.

## What it computes (bit-exact contract)

Signed **INT8 × INT8 → INT32**, no saturation — bit-exact to the ACU MAC-array reference
`sw/reference_model/mac_array_ref.{hpp,py}` `matmul_int8` for M=1 (one attention row):

```
o[n] = Σ_t  A[t] · V[t][n]        A,V ∈ int8   →   o ∈ int32
```

This is the ACU's **INT8 tile** (`precision_controller.d_fp16 == 0`). The FP16 tile is
tolerance-only in the reference (`rel_err < 5e-3`, see `MAC_ARRAY_DESIGN.md`) and is not
part of this integer datapath; only INT8 carries a bit-exact contract, so that is what the
RTL implements.

## Why INT32 (not INT24)

The P·V tile reduces over the **token** dimension, so the accumulator width scales with
**context length**, not hidden dim. A maximally-flat causal row of length L drives every
code to ±127, so `|acc| ≤ 127·127·L → 14+⌈log₂L⌉ bits`: INT24 overflows past ~520 tokens,
INT32 covers ~133k. (The hidden-dim reductions — W4A8 GEMM, Q·Kᵀ — fit INT24; only the
token-reduction P·V accumulator needs INT32.) See `analysis/pv_accumulator_width.py` and
the `arch.yml` accumulator rationale. The testbench's flat-attention corner (K=520, all
±127, max|acc| = 8,387,080) is the empirical proof this overflows INT24 territory and needs
the INT32 register.

## Interface (house streaming style)

One token per clock, `s_valid=1`; scalar A-code on `a_data`, the N-wide packed V-row on
`v_data`; `s_last=1` on the final token. `c_valid` pulses the cycle after `s_last` with the
N int32 results on `c_data`; accumulators auto-reset on `s_last` (same pattern as
`precision_controller`). Latency 1 cycle after `s_last`; throughput 1 token/cycle. `N` is
the parallel head-dim lane count (default 8 for the physical run; the cosim uses the full
head dim).

## Verification

- **Bit-exact:** `make sim_mate` — `mate_pv` vs `matmul_int8` on 7 rows incl. the K=520
  flat corner: **7/7, 0 errors**. Golden computed in pure Python (integer `Σ A·V`, provably
  identical to the reference's `int32` matmul) so it runs on the bare venv (no numpy).
- **Synthesis:** `yosys synth -flatten` — **513 FFs** (8 acc×32 + 8 out×32 + 1 valid), no
  latches. This is the `expected-ff-count` for the CI synthesis gate.

## Still open

- `openlane/mate_pv/` LibreLane config → GDSII on Sky130A (parity with `precision_controller`).
- Wired into the cross-block cosim (`architecture` rtl branch) as the true end-to-end
  KVE → P·V → `wht_inverse_out` chain on real-Qwen data.
- CI: a `mate_pv` block gate (synth FF-count + `make sim_mate`) alongside `precision_controller`.
