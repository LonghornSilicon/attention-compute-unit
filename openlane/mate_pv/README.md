# Sky130 OpenLane / LibreLane flow — `mate_pv`

End-to-end open-source RTL → GDSII flow for `mate_pv`, the INT8 P·V MAC tile
(INT32 accumulator), targeting SkyWater Sky130A. Same flow and tuning as
`../precision_controller` (the signed-off reference), so the two blocks reach
GDSII the same way.

> 130nm Sky130 proxy, used for 16nm estimates — Lambda targets TSMC 16nm.

## Run it

Requires Docker (~25 GB free disk) and `pip install librelane`.

```sh
cd openlane/mate_pv
librelane --docker-no-tty --dockerized config.json
```

Flag order matters: `--docker-no-tty` must precede `--dockerized`. First
invocation downloads the Sky130A PDK (~500 MB via Ciel) and the LibreLane
Docker image (~6 GB); subsequent runs reuse both caches.

## Config

Cloned verbatim from `precision_controller/config.json` — only `DESIGN_NAME`
(`mate_pv`) and `VERILOG_FILES` (`dir::src/*.sv`) differ. `CLOCK_PERIOD` 12.5 ns
(80 MHz) start point; relax only if the SS corner cannot close. The RTL is
Verilog-2001-clean (read with `-sv`), single-file, no submodules.

`src/mate_pv.sv` is the block top (kept in sync with `rtl/mate_pv.sv`). Default
`N=8` head-dim lanes → **513 FFs** (8 acc×32 + 8 out×32 + 1 valid), matching the
CI `expected-ff-count`.

## Sign-off

The `openlane-sky130` CI gate runs this config and asserts the six physical
checks are zero (setup / hold / DRC / LVS / antenna / IR-drop), same as every
other block. `runs/` is gitignored; committed results (metrics + GDS) land under
`results/` when the flow is run.
