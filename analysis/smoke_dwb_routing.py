"""End-to-end smoke for DWB per-token routing through the real Qwen2-0.5B
attention path (acu_kvce_attention). Phase 1 step 3 validation — NOT the full
eval. Confirms: set_token_bits() flows controller tiers into every layer's
KVCE round-trip, an all-bypass (tier 16) run tracks FP16 closely, and a
uniform turbo run degrades as expected.

Run:
  HF_HOME=/home/chaithu/lhs/.hf_cache \
  KVCE_REF=/home/chaithu/lhs/kv-cache-engine/sw/reference_model \
  /home/chaithu/lhs/.venv/bin/python analysis/smoke_dwb_routing.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acu_kvce_attention as akvce
from kvce_pool import shutdown_pool

MODEL_ID = "Qwen/Qwen2-0.5B"


@torch.no_grad()
def last_logits(model, ids, config, bits=None):
    akvce.set_config(config)
    akvce.set_token_bits(bits)
    out = model(input_ids=ids, use_cache=False)
    akvce.set_token_bits(None)
    return out.logits[0, -1].float()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    akvce.register("acu_kvce")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="acu_kvce").eval()

    ids = tok("The capital of France is", return_tensors="pt")["input_ids"]
    N = ids.shape[1]
    print(f"  seq len N={N}", flush=True)

    fp16 = last_logits(model, ids, "A")                     # dense baseline
    uni = last_logits(model, ids, "C_prenorm", bits=None)   # uniform turbo4
    bypass = last_logits(model, ids, "C_prenorm",
                         bits=np.full(N, 16))               # all FP16 bypass
    turbo2 = last_logits(model, ids, "C_prenorm",
                         bits=np.full(N, 2))                # all turbo (pq2)
    mixed = last_logits(model, ids, "C_prenorm",
                        bits=np.array([16, 8, 4, 2] * N)[:N])

    def cos(a, b):
        return float(torch.nn.functional.cosine_similarity(a, b, dim=0))

    c_uni = cos(fp16, uni)
    c_bypass = cos(fp16, bypass)
    c_turbo2 = cos(fp16, turbo2)
    c_mixed = cos(fp16, mixed)
    print(f"  cos(FP16, uniform turbo4)   = {c_uni:.5f}", flush=True)
    print(f"  cos(FP16, all-bypass tier16)= {c_bypass:.5f}", flush=True)
    print(f"  cos(FP16, all-turbo tier2)  = {c_turbo2:.5f}", flush=True)
    print(f"  cos(FP16, mixed 16/8/4/2)   = {c_mixed:.5f}", flush=True)
    print(f"  avg bits: uniform≈4  bypass={akvce.tier_avg_bits(np.full(N,16)):.1f}"
          f"  mixed={akvce.tier_avg_bits(np.array([16,8,4,2]*N)[:N]):.2f}",
          flush=True)

    assert torch.isfinite(uni).all() and torch.isfinite(turbo2).all()
    assert c_bypass > c_uni > c_turbo2, \
        f"expected bypass>uniform>turbo2, got {c_bypass:.4f}/{c_uni:.4f}/{c_turbo2:.4f}"
    assert c_bypass > 0.999, f"FP16 bypass should ~match dense, got {c_bypass:.5f}"
    print("  ordering bypass > uniform > turbo2 and bypass≈FP16: OK", flush=True)

    shutdown_pool()
    print("DWB ROUTING SMOKE PASSED", flush=True)


if __name__ == "__main__":
    main()
