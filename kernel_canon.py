"""
GPU canonical-form kernel for AGL₂(F₇) orbit reduction.

For each k-set S, finds min over all 2016 GL₂(F₇) matrices × k anchor
choices of sort(A·S − anchor), packed as a (lo, hi) uint64 pair.

Design: one warp (32 threads) per set.  Each lane processes ceil(2016/32)=63
permutations; the inner loop tries all k anchor positions.  A 128-bit
warp-min reduction collapses to a (lo, hi) pair per set.

Packing scheme (all formats are backward-compatible):
  k≤10 : positions 0..k-1,   6 bits each → ≤60 bits, hi=0
  k=11 : positions 1..10,    6 bits each → 60 bits  (skip pos 0, always 0), hi=0
  k=12..16 : positions 1..k-1, 6 bits each → up to 90 bits
             lo = bits 0..63, hi = bits 64..89

Python unpacking: unpack_canonical(packed, k) where packed = lo | (hi << 64).
"""

from __future__ import annotations

import numpy as np
import cupy as cp


_CUDA_CANON_SRC = r"""
#define N_PERMS  2016
#define N_POINTS   49

/*
 * canonical_forms — one warp per set, 128-bit packed canonical form.
 *
 * sets      : (n, k) int32, each row sorted ascending
 * perms     : (2016, 49) int32  — GL₂(F₇) as permutations of {0..48}
 * rel_table : (49, 49) uint8   — rel_table[a,b] = (b−a) mod F₇²
 * out_lo/hi : (n,) uint64 each — caller inits lo=UINT64_MAX, hi=UINT64_MAX
 *
 * Packing (6 bits per index, pack_from = 0 for k≤10, 1 for k≥11):
 *   bit_off = 6*(i - pack_from)
 *   bits [bit_off, bit_off+6) of the 128-bit word ← t[i]
 *   lo holds bits 0..63, hi holds bits 64..127.
 * For k≤11 hi is always 0.  For k=12..16 hi holds overflow bits.
 */
extern "C" __global__ void canonical_forms(
    const int*           __restrict__ sets,
    const int*           __restrict__ perms,
    const unsigned char* __restrict__ rel_table,
    unsigned long long*  __restrict__ out_lo,
    unsigned long long*  __restrict__ out_hi,
    int n, int k
) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int set_idx    = global_tid >> 5;
    int lane       = global_tid & 31;
    if (set_idx >= n) return;

    int S[16];
    for (int i = 0; i < k; i++)
        S[i] = sets[set_idx * k + i];

    unsigned long long best_lo = 0xffffffffffffffffULL;
    unsigned long long best_hi = 0xffffffffffffffffULL;

    for (int pi = lane; pi < N_PERMS; pi += 32) {

        int perm[16];
        for (int i = 0; i < k; i++)
            perm[i] = __ldg(perms + pi * N_POINTS + S[i]);

        for (int anc = 0; anc < k; anc++) {
            int anchor = perm[anc];

            int t[16];
            for (int i = 0; i < k; i++)
                t[i] = (int)rel_table[anchor * N_POINTS + perm[i]];

            /* insertion sort */
            for (int i = 1; i < k; i++) {
                int tmp = t[i], j = i;
                while (j > 0 && t[j-1] > tmp) { t[j] = t[j-1]; j--; }
                t[j] = tmp;
            }

            /* 128-bit pack.
               k≤10: pack positions 0..k-1 (t[0]=0 explicit).
               k≥11: skip position 0 (always 0); pack positions 1..k-1.
               Position i sits at bit_off = 6*(i - pack_from).
               Bits that fall in [0,64) go to lo; bits in [64,128) go to hi;
               a value straddling the boundary is split across both words. */
            unsigned long long packed_lo = 0, packed_hi = 0;
            int pack_from = (k <= 10) ? 0 : 1;
            for (int i = pack_from; i < k; i++) {
                int bit_off = 6 * (i - pack_from);
                unsigned long long v = (unsigned long long)t[i];
                if (bit_off + 6 <= 64) {
                    packed_lo |= v << bit_off;
                } else if (bit_off < 64) {
                    /* straddles boundary */
                    packed_lo |= v << bit_off;
                    packed_hi |= v >> (64 - bit_off);
                } else {
                    packed_hi |= v << (bit_off - 64);
                }
            }

            /* 128-bit lexicographic min (compare hi first). */
            if (packed_hi < best_hi ||
                (packed_hi == best_hi && packed_lo < best_lo)) {
                best_lo = packed_lo;
                best_hi = packed_hi;
            }
        }
    }

    /* Warp-level 128-bit min reduction. */
    for (int delta = 16; delta > 0; delta >>= 1) {
        unsigned long long cand_lo = __shfl_xor_sync(0xffffffff, best_lo, delta);
        unsigned long long cand_hi = __shfl_xor_sync(0xffffffff, best_hi, delta);
        if (cand_hi < best_hi || (cand_hi == best_hi && cand_lo < best_lo)) {
            best_lo = cand_lo;
            best_hi = cand_hi;
        }
    }

    if (lane == 0) {
        out_lo[set_idx] = best_lo;
        out_hi[set_idx] = best_hi;
    }
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
        Returns (n,) numpy object array of Python ints (canonical packed values).

        k≤10 : lo only (hi=0), same uint64 format as before.
        k=11 : 60-bit value in lo (hi=0).
        k=12..16 : split across lo + hi; Python int = lo | (hi << 64).
        """
        n, k = sets.shape
        if k > 16:
            raise ValueError("GPU canonical kernel supports k <= 16")
        threads = self._THREADS
        grid    = (n * 32 + threads - 1) // threads
        with cp.cuda.Device(self.device_id):
            sets_gpu = cp.asarray(sets.astype(np.int32))
            out_lo   = cp.full(n, 0xFFFF_FFFF_FFFF_FFFF, dtype=cp.uint64)
            out_hi   = cp.zeros(n, dtype=cp.uint64)
            self._knl(
                (grid,), (threads,),
                (sets_gpu, self._perms_gpu, self._rel_gpu,
                 out_lo, out_hi, np.int32(n), np.int32(k)),
            )
            lo = cp.asnumpy(out_lo).astype(object)   # Python ints, no truncation
            hi = cp.asnumpy(out_hi).astype(object)
            return lo + hi * (1 << 64)


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
