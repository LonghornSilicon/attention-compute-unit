# Adaptive Precision Attention — `compiler-integration-prep` branch

> 🪧 **You are on a feature branch.** This branch adds the software
> reference model and ISA stub a silicon-agnostic compiler can target
> *before* the chip is fabricated. For the main project (precision
> controller RTL, OpenLane Sky130 sign-off, paper, CI), switch to
> [`master`](https://github.com/LonghornSilicon/adaptive-precision-attention/tree/master).

This branch exists because we have a meeting with a compiler team that
wants to integrate their silicon-agnostic compiler with the
LonghornSilicon chip. They can't wait for tape-out (Q3/Q4 2026), so we
expose:

1. A **bit-accurate Python reference model** of the precision
   controller that produces exactly the same INT8/FP16 decision as the
   real chip will.
2. An **ISA / interface specification** describing the memory map,
   AXI-Lite registers, AXI-Stream protocols, and a C API stub. This is
   the contract their compiler targets.
3. **Self-tests** proving the Python model agrees bit-for-bit with the
   RTL testbench across all 143 replay tiles.

The compiler team writes their backend against the ISA. Their backend
emits operations the Python model can execute now, the FPGA prototype
can execute when ZCU102/104 arrives, and the chip can execute after
tape-out — same interface throughout.

---

## What's in this branch (delta from `master`)

```
adaptive-precision-attention/  (compiler-integration-prep)
├── sw/                                                ← NEW directory
│   └── reference_model/
│       ├── precision_controller_ref.py                ← bit-accurate Python model
│       └── test_precision_controller_ref.py           ← 143 replay tests + sanity checks
├── docs/
│   └── isa/                                           ← NEW directory
│       └── precision_controller_isa.md                ← ISA / interface spec
└── README.md                                          ← this file (branch-scoped)
```

Everything else (`rtl/`, `openlane/`, `paper/`, `analysis/`, the CI
pipeline) is unchanged from `master`.

---

## Quick verification — model is bit-exact against the RTL

```sh
python3 sw/reference_model/test_precision_controller_ref.py
```

Expected output:

```
Tests: 143  Pass: 143  Fail: 0  ALL TESTS PASSED
ALL SELF-TESTS PASSED
```

The first line is the exact match string the RTL testbench prints when
all 143 replay tiles agree with the integer reference. The Python
model reaches the same result through the same arithmetic
(SCORE_WIDTH-bit two's-complement abs, SUM_WIDTH accumulator,
CMP_WIDTH comparison, shift-and-add multiply by THRESHOLD).

---

## Using the model from your compiler

### High-level (stateless, one-shot)

```python
from precision_controller_ref import PrecisionController

scores = [...]   # exactly 4096 int8 values (BLOCK_M * BLOCK_N)
decision = PrecisionController.decide(scores)
# decision is True (FP16) or False (INT8)
```

### High-level (stateful, batch)

```python
pc = PrecisionController()
decisions = pc.process_tiles(many_tiles)   # list of bools, one per tile
```

### Low-level (streaming, mirrors the chip's AXI-Stream interface)

```python
pc = PrecisionController()
pc.reset()
for i, s in enumerate(scores):
    last = (i == len(scores) - 1)
    pc.tick(s_valid=True, s_data=s, s_last=last)
    if pc.d_valid:
        decision = pc.d_fp16
```

This last form is what an FPGA driver or a cycle-accurate simulator
would do — feed one beat per clock, watch for `d_valid`.

---

## ISA / interface spec

[`docs/isa/precision_controller_isa.md`](docs/isa/precision_controller_isa.md)

Covers:

- 11-register AXI-Lite memory map (control, status, INFO, decision FIFO,
  interrupt)
- Two AXI-Stream channels (`s_axis_scores`, `m_axis_decisions`) with
  protocol details
- 7 logical operations a compiler emits (`PC_QUERY`, `PC_RESET`,
  `PC_ENABLE`, `PC_PUSH_TILE`, `PC_READ`, `PC_READ_BATCH`, `PC_STATUS`)
- C API stub for the runtime driver layer
- Synthesis-time configuration table (BLOCK_M/N, SCORE_WIDTH, THRESHOLD)
- The 4-phase integration plan (Python ref → FPGA → multi-block FPGA →
  real silicon)
- Open questions for the compiler team meeting

The spec is versioned `pc-isa-0.1`. Breaking changes bump the major
version; backwards-compatible additions bump the minor.

---

## 4-phase integration plan (recap from the ISA spec)

| Phase | Timeline | Compiler targets | We provide |
|---|---|---|---|
| **0 — Python ref** | now | `sw/reference_model/precision_controller_ref.py` | This branch ✅ |
| **1 — FPGA prototype** | when ZCU102/104 arrives | AXI-Lite + AXI-Stream on the dev board | Vivado bitstream + PYNQ driver (TBD) |
| **2 — Multi-block FPGA** | after KV cache + token importance + memory controller blocks | Same AXI interface, now with neighbor blocks | Larger Vivado project (TBD) |
| **3 — Real silicon** | post-tape-out, 2027+ | Same AXI-mapped interface over PCIe | TSMC 16FFC chip (TBD) |

Phase 0 is what this branch enables. It commits both sides to a
concrete, testable interface *before silicon exists* — which is the
only way to start meaningful compiler-side work months ahead of
tape-out.

---

## When this branch merges back to master

After the compiler integration meeting and any consequent revisions to
the ISA stub, this branch's contents (the `sw/` directory, the `docs/isa/`
directory, and the model self-tests) will merge into `master`. The
branch-scoped README will be replaced with the standard master README
at that point, and a "Compiler integration" section will land in the
top-level docs.

If you're a Claude Code session arriving here from
[`docs/new_block_blueprint.md`](docs/new_block_blueprint.md) (on
master) to set up similar software for another LonghornSilicon block
(KV Cache Engine, Token Importance Unit, etc.), use this branch's
`sw/reference_model/` and `docs/isa/` directories as the template.
Same pattern, different block.

---

## Provenance and caveats

- The Python model assumes `THRESHOLD = 10` (the synthesized default).
  Other thresholds require an RTL re-synthesis; the model raises
  `NotImplementedError` if you try to instantiate with a different
  threshold. See "Open questions" at the end of the ISA spec.
- `N = BLOCK_M * BLOCK_N` must be a power of two (the SV uses
  `$clog2(N)` as a hardwired shift; non-powers-of-two would need a
  parameterized shift register, which the current implementation does
  not have).
- The C API in §5 of the ISA is a **stub**, not a working
  implementation. A real driver requires the FPGA bitstream and the
  AXI-Lite/AXI-Stream IPs to actually wire up.
- This branch does **not** modify any RTL, OpenLane config, or paper
  content. The chip-side commitments are unchanged.

---

## License / contact

Same as master. Project lives at
[`LonghornSilicon/adaptive-precision-attention`](https://github.com/LonghornSilicon/adaptive-precision-attention).
Author: Chaithu Talasila, UT Austin.
