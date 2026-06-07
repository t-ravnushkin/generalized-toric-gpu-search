"""
Bit-parallel CUDA kernels for GF(8) toric code distance evaluation.

Core idea — replace the k*49 GF8_MUL lookups per polynomial with a
precomputed BITPAT table and 3k XOR + popcount:

  BITPAT[b][i][v] = 49-bit uint64 mask over torus points j, where
                    bit j is 1 iff (bit b of GF8_MUL[v*8+M[s[i]][j]]) == 1.

  b ∈ {0,1,2}  — bit plane of the GF(8) output
  i ∈ 0..k-1   — basis-vector index in the set S
  v ∈ 0..7     — GF(8) coefficient value

BITPAT is precomputed once per set in shared memory (cooperative, ~3k*8*49
= 11760 GF8_MUL reads for k=10).  After that, evaluating one polynomial is:

  v0 ^= BITPAT[0][i][c[i]]
  v1 ^= BITPAT[1][i][c[i]]
  v2 ^= BITPAT[2][i][c[i]]   (for each i)
  zeros = popcount(~v0 & ~v1 & ~v2 & TORUS_MASK)

i.e. 3k XOR ops on 64-bit registers + 1 __popcll, instead of k*49 table
lookups.  For k=10: 30 register ops vs ~490 memory lookups — ~14× fewer
operations per polynomial.

Combined with the block-per-set structure (same as kernel_cuda.py), the
expected total speedup over the original OpenCL kernel is 50–150× for k≥6.

Same interface as DistanceOracleCUDA so it is a drop-in replacement.
"""

from __future__ import annotations

import numpy as np
import cupy as cp

# ---------------------------------------------------------------------------
# CUDA source
# ---------------------------------------------------------------------------

_CUDA_BP_SRC = r"""
__constant__ int GF8_MUL[64] = {
    0,0,0,0,0,0,0,0,  0,1,2,3,4,5,6,7,  0,2,4,6,3,1,7,5,  0,3,6,5,7,4,1,2,
    0,4,3,7,6,2,5,1,  0,5,1,4,2,7,3,6,  0,6,7,1,5,3,2,4,  0,7,5,2,1,6,4,3
};

/* Build BITPAT cooperatively.
   bitpat must point to 3*k*8 uint64 words in shared memory.
   Call __syncthreads() after this returns. */
__device__ void build_bitpat(
    unsigned long long* __restrict__ bitpat,
    const int* __restrict__ M,
    const int* __restrict__ s_local,
    int k
) {
    int total = 3 * k * 8;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int b   = idx / (k * 8);
        int rem = idx % (k * 8);
        int i   = rem / 8;
        int v   = rem % 8;
        int row = s_local[i];
        unsigned long long mask = 0;
        for (int j = 0; j < 49; j++)
            if ((GF8_MUL[v * 8 + M[row * 49 + j]] >> b) & 1)
                mask |= (1ULL << j);
        bitpat[idx] = mask;
    }
}

/* ── Kernel 1-bp: single set, bit-parallel evaluation ────────────────────
   Grid  = ceil(total_polys / BLOCK)
   Block = 256
   Smem  = 3*k*8 uint64 (dynamic)

   All blocks work on the same set, each builds BITPAT independently, then
   every thread evaluates its polynomial via 3k XOR + popcount. */
extern "C" __global__ void eval_min_distance_single_bp(
    const int* __restrict__ M,
    const int* __restrict__ s_idx,
    int k, int tmax_zeros, int total_polys,
    int* out
) {
    extern __shared__ unsigned long long smem_ull[];
    unsigned long long* bitpat = smem_ull;   /* 3 * k * 8 entries */

    build_bitpat(bitpat, M, s_idx, k);
    __syncthreads();

    int poly_id = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (poly_id > total_polys) return;

    unsigned long long v0 = 0, v1 = 0, v2 = 0;
    int tmp = poly_id;
    for (int i = 0; i < k; i++) {
        int c = tmp & 7; tmp >>= 3;
        v0 ^= bitpat[          i * 8 + c];
        v1 ^= bitpat[    k * 8 + i * 8 + c];
        v2 ^= bitpat[2 * k * 8 + i * 8 + c];
    }

    const unsigned long long TORUS_MASK = (1ULL << 49) - 1;
    int zeros = __popcll((~v0) & (~v1) & (~v2) & TORUS_MASK);
    if (zeros > tmax_zeros) { atomicMax(out, zeros); return; }
    atomicMax(out, zeros);
}

/* ── Kernel 2-bp: batched BFS, one block per set, bit-parallel ───────────
   Grid  = num_sets
   Block = 256
   Smem  = s_cache (64 B) + bitpat (192k B) + block_max + abort (8 B)

   Dynamic shared memory layout:
     [0  .. 63]            s_cache   — 16 int32 (room for k ≤ 16 indices)
     [64 .. 64+192k-1]     bitpat    — 3*k*8 uint64  (64-byte offset ensures
                                        8-byte alignment of uint64 array)
     [64+192k .. +7]       block_max, abort_flag — 2 int32

   Polynomial evaluation collapses to:
     for i in 0..k-1:  v0 ^= bitpat[0][i][c[i]]
                        v1 ^= bitpat[1][i][c[i]]
                        v2 ^= bitpat[2][i][c[i]]
     zeros = popcount(~v0 & ~v1 & ~v2 & TORUS_MASK)
   — entirely register ops after BITPAT is built in shared memory. */
extern "C" __global__ void eval_min_distance_batch_bp(
    const int* __restrict__ M,
    const int* __restrict__ batched_s_idx,
    int k, int tmax_zeros, int num_sets,
    int* out
) {
    int set_id = blockIdx.x;
    if (set_id >= num_sets) return;

    extern __shared__ char smem_raw[];
    int*                s_cache    = (int*)smem_raw;
    unsigned long long* bitpat     = (unsigned long long*)(smem_raw + 64);
    volatile int*       block_max  = (volatile int*)(bitpat + 3 * k * 8);
    volatile int*       abort_flag = block_max + 1;

    if (threadIdx.x == 0) { *block_max = 0; *abort_flag = 0; }

    /* Load row indices into shared s_cache (thread 0..k-1 each load one). */
    if (threadIdx.x < k)
        s_cache[threadIdx.x] = batched_s_idx[set_id * k + threadIdx.x];
    __syncthreads();

    build_bitpat(bitpat, M, s_cache, k);
    __syncthreads();

    const unsigned long long TORUS_MASK = (1ULL << 49) - 1;
    int total_polys = (int)((1LL << (3 * k)) - 1);
    int thread_max  = 0;

    for (int poly_id = (int)threadIdx.x + 1;
         poly_id <= total_polys;
         poly_id += blockDim.x)
    {
        if (*abort_flag) break;

        unsigned long long v0 = 0, v1 = 0, v2 = 0;
        int tmp = poly_id;
        for (int i = 0; i < k; i++) {
            int c = tmp & 7; tmp >>= 3;
            v0 ^= bitpat[          i * 8 + c];
            v1 ^= bitpat[    k * 8 + i * 8 + c];
            v2 ^= bitpat[2 * k * 8 + i * 8 + c];
        }

        int zeros = __popcll((~v0) & (~v1) & (~v2) & TORUS_MASK);

        if (zeros > tmax_zeros) {
            atomicMax((int*)block_max, zeros);
            *abort_flag = 1;
            break;
        }
        if (zeros > thread_max) thread_max = zeros;
    }

    atomicMax((int*)block_max, thread_max);
    __syncthreads();
    if (threadIdx.x == 0) out[set_id] = *block_max;
}
"""


# ---------------------------------------------------------------------------
# DistanceOracleCUDABP — drop-in replacement for DistanceOracleCUDA
# ---------------------------------------------------------------------------

class DistanceOracleCUDABP:
    """Bit-parallel variant.  Same interface as DistanceOracleCUDA."""

    BLOCK_SIZE = 256

    def __init__(self, device_id: int, M: np.ndarray):
        self.device_id = device_id
        with cp.cuda.Device(device_id):
            self._M_gpu = cp.asarray(M.astype(np.int32))
            module = cp.RawModule(code=_CUDA_BP_SRC)
            self._knl_single = module.get_function("eval_min_distance_single_bp")
            self._knl_batch  = module.get_function("eval_min_distance_batch_bp")

    def _single_smem(self, k: int) -> int:
        return 3 * k * 8 * 8          # 3*k*8 uint64

    def _batch_smem(self, k: int) -> int:
        return 64 + 3 * k * 8 * 8 + 8  # s_cache + bitpat + block_max + abort

    def max_zeros(self, s_indices: list[int], target_distance: int) -> int:
        k = len(s_indices)
        if k == 0:
            return 0
        tmax_zeros  = 49 - target_distance
        total_polys = (8 ** k) - 1

        with cp.cuda.Device(self.device_id):
            s_gpu   = cp.array(s_indices, dtype=np.int32)
            out_gpu = cp.zeros(1, dtype=np.int32)
            block = self.BLOCK_SIZE
            grid  = (total_polys + block - 1) // block
            self._knl_single(
                (grid,), (block,),
                (self._M_gpu, s_gpu,
                 np.int32(k), np.int32(tmax_zeros), np.int32(total_polys),
                 out_gpu),
                shared_mem=self._single_smem(k),
            )
            cp.cuda.Device(self.device_id).synchronize()
            return int(out_gpu[0])

    def max_zeros_batch(
        self, batched_indices: list[list[int]], target_distance: int
    ) -> list[int]:
        num_sets = len(batched_indices)
        if num_sets == 0:
            return []
        k          = len(batched_indices[0])
        tmax_zeros = 49 - target_distance

        flat = np.array(
            [idx for s in batched_indices for idx in s], dtype=np.int32
        )

        with cp.cuda.Device(self.device_id):
            s_gpu   = cp.asarray(flat)
            out_gpu = cp.zeros(num_sets, dtype=np.int32)
            self._knl_batch(
                (num_sets,), (self.BLOCK_SIZE,),
                (self._M_gpu, s_gpu,
                 np.int32(k), np.int32(tmax_zeros), np.int32(num_sets),
                 out_gpu),
                shared_mem=self._batch_smem(k),
            )
            cp.cuda.Device(self.device_id).synchronize()
            return out_gpu.tolist()


# ---------------------------------------------------------------------------
# Initialisation helper
# ---------------------------------------------------------------------------

def init_cuda_oracles_bp(M: np.ndarray) -> tuple[list[DistanceOracleCUDABP], int]:
    """Detect all CUDA GPUs, compile the BP kernels, return (oracles, sm_count)."""
    n = cp.cuda.runtime.getDeviceCount()
    if n == 0:
        raise RuntimeError("No CUDA GPU found.")

    oracles: list[DistanceOracleCUDABP] = []
    sm_count = 40
    for i in range(n):
        props      = cp.cuda.runtime.getDeviceProperties(i)
        name       = props["name"].decode()
        sm_count_i = props.get("multiProcessorCount", 40)
        print(f"[CUDA-BP] Device {i}: {name}  ({sm_count_i} SMs)")
        if i == 0:
            sm_count = sm_count_i
        oracles.append(DistanceOracleCUDABP(i, M))

    return oracles, sm_count
