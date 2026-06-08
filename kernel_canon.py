"""
GPU canonical-form kernel for AGL₂(F₇) orbit reduction.

For each k-set S, finds min over all 2016 GL₂(F₇) matrices × k anchor
choices of sort(A·S − anchor) packed into a uint64 (6 bits per index).

Design: one warp (32 threads) per set.  Each lane processes ceil(2016/32)=63
permutations; the inner loop tries all k anchor positions for each permutation.
Warp-level min reduction via __shfl_xor_sync collapses to a single uint64.

Supports k ≤ 10  (10 × 6 = 60 bits < 64).
On T4 (40 SMs, 64 CUDA cores/SM): 200K-set chunks complete in ~0.5 s,
vs ~60 s for the equivalent CPU numpy loop.
"""

from __future__ import annotations

import numpy as np
import cupy as cp


_CUDA_CANON_SRC = r"""
#define N_PERMS  2016
#define N_POINTS   49

/*
 * canonical_forms — one warp per set, computes AGL₂(F₇) canonical form.
 *
 * sets      : (n, k) int32, each row sorted ascending
 * perms     : (2016, 49) int32 — GL₂(F₇) as permutations of {0..48}
 * rel_table : (49, 49) uint8  — rel_table[a,b] = (b−a) mod F₇²  (index 0..48)
 * out       : (n,) uint64     — caller initialises to UINT64_MAX
 */
extern "C" __global__ void canonical_forms(
    const int*           __restrict__ sets,
    const int*           __restrict__ perms,
    const unsigned char* __restrict__ rel_table,
    unsigned long long*  __restrict__ out,
    int n, int k
) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int set_idx    = global_tid >> 5;   /* one warp = one set */
    int lane       = global_tid & 31;
    if (set_idx >= n) return;

    /* Load k set indices into registers. */
    int S[12];
    for (int i = 0; i < k; i++)
        S[i] = sets[set_idx * k + i];

    unsigned long long best = 0xffffffffffffffffULL;

    /*
     * Each lane handles ceil(2016/32)=63 permutations.
     * Inner loop: try all k anchor positions for each permutation.
     * __ldg caches the (2016×49) perm table in read-only L1.
     */
    for (int pi = lane; pi < N_PERMS; pi += 32) {

        int perm[12];
        for (int i = 0; i < k; i++)
            perm[i] = __ldg(perms + pi * N_POINTS + S[i]);

        for (int anc = 0; anc < k; anc++) {
            int anchor = perm[anc];   /* translate: this point → index 0 */

            /* Translate and sort (insertion sort, cheap for k ≤ 12). */
            int t[12];
            for (int i = 0; i < k; i++)
                t[i] = (int)rel_table[anchor * N_POINTS + perm[i]];

            for (int i = 1; i < k; i++) {
                int tmp = t[i], j = i;
                while (j > 0 && t[j-1] > tmp) { t[j] = t[j-1]; j--; }
                t[j] = tmp;
            }

            /* Pack into uint64.
               k≤10: positions 0..k-1  (t[0]=0 explicit) — 10×6=60 bits max.
               k=11: skip position 0 (always 0 after translate+sort),
                     pack positions 1..10 — 10×6=60 bits, no overflow.
               Both cases fit; k≤10 format is unchanged for checkpoint compat. */
            unsigned long long packed = 0;
            int pack_from = (k <= 10) ? 0 : 1;
            for (int i = pack_from; i < k; i++)
                packed |= (unsigned long long)t[i] << (6 * (i - pack_from));

            if (packed < best) best = packed;
        }
    }

    /* Warp-level min reduction. */
    for (int delta = 16; delta > 0; delta >>= 1)
        best = min(best, __shfl_xor_sync(0xffffffff, best, delta));

    if (lane == 0)
        out[set_idx] = best;
}
"""


class CanonicalOracle:
    """GPU canonical-form engine; one instance per CUDA device."""

    _THREADS = 256   # 8 warps per block

    def __init__(
        self,
        device_id: int,
        gl2_perms: np.ndarray,   # (2016, 49) int32
        rel_table: np.ndarray,   # (49, 49) uint8
    ):
        self.device_id = device_id
        with cp.cuda.Device(device_id):
            self._perms_gpu = cp.asarray(gl2_perms.astype(np.int32))
            self._rel_gpu   = cp.asarray(rel_table.astype(np.uint8))
            module = cp.RawModule(code=_CUDA_CANON_SRC)
            self._knl = module.get_function("canonical_forms")

    def compute(self, sets: np.ndarray) -> np.ndarray:
        """
        sets : (n, k) int32, rows sorted ascending.
        Returns (n,) uint64 canonical packed values.
        k≤10: 6 bits per index, positions 0..k-1.
        k=11: positions 1..10 only (position 0 is always 0); 10×6=60 bits.
        """
        n, k = sets.shape
        threads = self._THREADS
        grid    = (n * 32 + threads - 1) // threads
        with cp.cuda.Device(self.device_id):
            sets_gpu = cp.asarray(sets.astype(np.int32))
            out_gpu  = cp.full(n, 0xFFFF_FFFF_FFFF_FFFF, dtype=cp.uint64)
            self._knl(
                (grid,), (threads,),
                (sets_gpu, self._perms_gpu, self._rel_gpu,
                 out_gpu, np.int32(n), np.int32(k)),
            )
            return cp.asnumpy(out_gpu)


def init_canon_oracle(
    device_id: int,
    gl2_perms: np.ndarray,
    rel_table: np.ndarray,
) -> CanonicalOracle:
    """Compile the kernel and return a ready-to-use CanonicalOracle."""
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    name  = props["name"].decode()
    print(f"[canon] GPU {device_id}: {name} — compiling canonical kernel … ",
          end="", flush=True)
    oracle = CanonicalOracle(device_id, gl2_perms, rel_table)
    print("done")
    return oracle
