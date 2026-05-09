# rtl/Makefile — sim, real-data replay, and Yosys synthesis flows.
#
# Run all targets from this directory (rtl/) so $readmemh paths resolve.

PYTHON       ?= python3
RTL          := precision_controller.sv
TB_DIR       := tb
TV_DIR       := $(TB_DIR)/testvectors
GEN_SCRIPT   := ../analysis/gen_rtl_testvectors.py

SIM_OUT          := sim.out
SIM_REAL_OUT     := sim_realdata.out
SYNTH_LOG        := synth.log

.PHONY: all sim sim_realdata testvectors synth clean help

all: sim sim_realdata

help:
	@echo "Targets:"
	@echo "  sim           — directed self-checking testbench (110 tests, synthetic)"
	@echo "  sim_realdata  — replay TB on quantized score tiles + boundary cases"
	@echo "  testvectors   — regenerate scores.hex / expected.hex / labels.txt"
	@echo "  synth         — Yosys area report (FFs + NAND2-equivalent gate count)"
	@echo "  clean         — remove sim binaries, hex vectors, synth logs"

# ---------------------------------------------------------------------------
# Functional simulation (existing testbench)
# ---------------------------------------------------------------------------

sim: $(SIM_OUT)
	./$(SIM_OUT)

$(SIM_OUT): $(RTL) $(TB_DIR)/tb_precision_controller.sv
	iverilog -g2012 -o $@ $^

# ---------------------------------------------------------------------------
# Real-data replay
# ---------------------------------------------------------------------------

testvectors: $(TV_DIR)/scores.hex

$(TV_DIR)/scores.hex $(TV_DIR)/expected.hex: $(GEN_SCRIPT)
	$(PYTHON) $(GEN_SCRIPT)

sim_realdata: $(SIM_REAL_OUT) $(TV_DIR)/scores.hex $(TV_DIR)/expected.hex
	./$(SIM_REAL_OUT)

$(SIM_REAL_OUT): $(RTL) $(TB_DIR)/tb_realdata.sv
	iverilog -g2012 -o $@ $^

# ---------------------------------------------------------------------------
# Yosys synthesis (area sanity check)
# ---------------------------------------------------------------------------

synth: $(RTL) synth.ys
	yosys -q -l $(SYNTH_LOG) -s synth.ys
	@echo ""
	@echo "Full log: $(SYNTH_LOG)"

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

clean:
	rm -f $(SIM_OUT) $(SIM_REAL_OUT) $(SYNTH_LOG)
	rm -f $(TV_DIR)/scores.hex $(TV_DIR)/expected.hex $(TV_DIR)/labels.txt
