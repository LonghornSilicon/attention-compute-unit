"""Multiprocessing wrapper around the KVCE Python reference model.

The KVCE reference is per-vector and pure Python (~1.2 ms / vector on this
host). For end-to-end perplexity on Qwen2-0.5B we have 24 layers x 2
KV-heads x N tokens of compress+decompress per forward pass. A process
pool with 20 workers cuts the per-forward KV cost from ~30s to ~1.5s.

Workers are spawned (not forked) so they don't inherit CUDA state from
the parent. Each worker constructs its own KVCacheEngine at init time.

Two execution modes are supported per vector:

  * "naive"   - direct float -> Q4.12 cast at the boundary. Clips at +/-8.
                Matches what analysis/integration_test_kv_pc.py does.
                Exposes conflict C1 (raw |K| can reach ~152).
  * "prenorm" - per-vector scale so max(|v|) lands at +/-4.0 before the
                Q4.12 cast; restored at decompress. Eliminates C1.

Both modes are bit-exact w.r.t. the KVCE reference; the only difference
is the boundary scaling.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import numpy as np

COORD_FRAC = 12
COORD_MAX_INT = (1 << 15) - 1
COORD_MIN_INT = -(1 << 15)
PRENORM_TARGET = 4.0  # leave headroom inside the +/-8.0 Q4.12 range


# ---------------------------------------------------------------------------
# Worker-local KVCE engine
# ---------------------------------------------------------------------------
_engine = None
_vector_dim = None


def _worker_init(kvce_ref_path: str, vector_dim: int) -> None:
    global _engine, _vector_dim
    sys.path.insert(0, kvce_ref_path)
    from kv_cache_engine_ref import KVCacheEngine, KVCacheEngineInfo  # noqa: E402
    _engine = KVCacheEngine(KVCacheEngineInfo(vector_dim=vector_dim))
    _vector_dim = vector_dim


def _roundtrip_chunk(payload: Tuple[str, str, np.ndarray]) -> np.ndarray:
    """Round-trip a contiguous chunk of vectors through KVCE.

    payload = (kind, mode, chunk)
      kind   = "K" or "V"
      mode   = "naive" or "prenorm"
      chunk  = float32 array, shape [n_vectors, vector_dim]

    Returns float32 array of the same shape with the reconstructed values.
    """
    kind, mode, chunk = payload
    n, d = chunk.shape
    out = np.empty_like(chunk)
    for i in range(n):
        v = chunk[i]
        if mode == "naive":
            q = (np.round(v * (1 << COORD_FRAC))
                   .clip(COORD_MIN_INT, COORD_MAX_INT)
                   .astype(np.int32).tolist())
            inv_scale = 1.0 / (1 << COORD_FRAC)
        else:  # prenorm
            mx = float(np.abs(v).max())
            if mx < 1e-12:
                out[i] = 0.0
                continue
            s = PRENORM_TARGET / mx
            q = (np.round(v * s * (1 << COORD_FRAC))
                   .clip(COORD_MIN_INT, COORD_MAX_INT)
                   .astype(np.int32).tolist())
            inv_scale = 1.0 / (s * (1 << COORD_FRAC))

        if kind == "K":
            qhat = _engine.decompress_key(_engine.compress_key(q))
        else:
            qhat = _engine.decompress_value(_engine.compress_value(q))
        out[i] = np.asarray(qhat, dtype=np.float32) * inv_scale
    return out


# ---------------------------------------------------------------------------
# Pool singleton
# ---------------------------------------------------------------------------
_pool: ProcessPoolExecutor | None = None


def get_pool(kvce_ref_path: str, vector_dim: int = 64, workers: int = 20) -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        ctx = mp.get_context("spawn")
        _pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(kvce_ref_path, vector_dim),
        )
    return _pool


def shutdown_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown(cancel_futures=True)
        _pool = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def kv_roundtrip(
    kvce_ref_path: str,
    K: np.ndarray,
    V: np.ndarray,
    mode: str = "naive",
    workers: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Round-trip K and V through KVCE.

    K, V: float32 arrays of shape [n_vectors, vector_dim].
    mode:  "naive" or "prenorm".
    Returns (K_hat, V_hat) at the same shape and dtype.
    """
    assert K.ndim == 2 and V.ndim == 2 and K.shape[1] == V.shape[1]
    if mode not in ("naive", "prenorm"):
        raise ValueError(f"unknown mode {mode!r}")
    pool = get_pool(kvce_ref_path, vector_dim=K.shape[1], workers=workers)

    # Split each tensor into ~workers chunks for low IPC overhead.
    def chunk_payloads(kind: str, arr: np.ndarray) -> List[Tuple[str, str, np.ndarray]]:
        n = arr.shape[0]
        if n == 0:
            return []
        n_chunks = min(workers, n)
        bounds = np.linspace(0, n, n_chunks + 1, dtype=int)
        return [(kind, mode, arr[bounds[i]:bounds[i + 1]])
                for i in range(n_chunks) if bounds[i + 1] > bounds[i]]

    payloads = chunk_payloads("K", K) + chunk_payloads("V", V)
    results = list(pool.map(_roundtrip_chunk, payloads))

    n_k_payloads = sum(1 for p in payloads if p[0] == "K")
    K_hat = np.concatenate(results[:n_k_payloads], axis=0) if n_k_payloads else K.copy()
    V_hat = np.concatenate(results[n_k_payloads:], axis=0) if (len(results) - n_k_payloads) else V.copy()
    return K_hat, V_hat
