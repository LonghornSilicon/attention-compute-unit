"""Gate for DWB per-token routing wired into the KVCE harness
(findings/dwb_turboquant_integration_plan.md, Phase 1 step 3).

Properties checked against kvce_pool.kv_roundtrip:
  1. bits=None (legacy uniform) is unchanged.
  2. bits=all-4 (turbo4 via the tier engine) == bits=None (both pq3 default).
  3. bits=all-16 (FP16 bypass) reconstructs the input losslessly.
  4. Mixed routing: per-token tier actually reaches the codec — a token at
     tier 16 is exact, tiers 2/4/8 quantize, and error decreases as the tier
     (stored bits) increases.

Run:
  KVCE_REF=/home/chaithu/lhs/kv-cache-engine/sw/reference_model \
  /home/chaithu/lhs/.venv/bin/python analysis/test_per_token_routing.py
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kvce_pool import kv_roundtrip, shutdown_pool  # noqa: E402

KVCE_REF = os.environ.get(
    "KVCE_REF", "/home/chaithu/lhs/kv-cache-engine/sw/reference_model")
D = 64


def _rel(a, b):
    return np.linalg.norm(a - b, axis=1) / np.clip(
        np.linalg.norm(a, axis=1), 1e-12, None)


def main():
    rng = np.random.default_rng(0)
    n = 40
    K = rng.standard_normal((n, D)).astype(np.float32)
    V = rng.standard_normal((n, D)).astype(np.float32)

    # 1 & 2: uniform None vs all-tier-4
    Kn, Vn, _ = kv_roundtrip(KVCE_REF, K, V, mode="prenorm", bits=None)
    bits4 = np.full(n, 4)
    K4, V4, _ = kv_roundtrip(KVCE_REF, K, V, mode="prenorm", bits=bits4)
    assert np.allclose(Kn, K4, atol=1e-6), "tier-4 engine must match the default engine"
    assert np.allclose(Vn, V4, atol=1e-6)
    print(f"  uniform-None == all-tier-4: OK (max|Δ|={np.abs(Kn-K4).max():.2e})")

    # 3: bypass lossless
    bits16 = np.full(n, 16)
    K16, V16, _ = kv_roundtrip(KVCE_REF, K, V, mode="prenorm", bits=bits16)
    assert np.array_equal(K16, K), "tier-16 must be a lossless FP16 bypass"
    assert np.array_equal(V16, V)
    print("  all-tier-16 bypass lossless (K_hat == K): OK")

    # 4: error decreases monotonically with tier on the SAME vectors
    errs = {}
    for t in (2, 4, 8):
        Kt, _, _ = kv_roundtrip(KVCE_REF, K, V, mode="prenorm",
                                bits=np.full(n, t))
        errs[t] = float(_rel(K.astype(np.float64), Kt.astype(np.float64)).mean())
    errs[16] = 0.0
    print(f"  mean rel-L2 by tier: "
          f"2={errs[2]:.4f} 4={errs[4]:.4f} 8={errs[8]:.4f} 16={errs[16]:.4f}")
    assert errs[2] > errs[4] > errs[8] > errs[16], \
        f"error must decrease as tier/bits increase, got {errs}"
    print("  monotone error vs tier (2>4>8>16): OK")

    # Mixed array: alternate bypass / aggressive; bypass rows stay exact.
    mixed = np.where(np.arange(n) % 2 == 0, 16, 2)
    Km, _, _ = kv_roundtrip(KVCE_REF, K, V, mode="prenorm", bits=mixed)
    assert np.array_equal(Km[0::2], K[0::2]), "even (tier-16) rows must be exact"
    assert not np.allclose(Km[1::2], K[1::2]), "odd (tier-2) rows must be quantized"
    print("  mixed per-row routing reaches codec: OK")

    shutdown_pool()
    print("ALL PER-TOKEN ROUTING GATES PASSED")


if __name__ == "__main__":
    main()
