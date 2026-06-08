"""
CUDA-native kernels for GF(8) toric code distance evaluation.

Key design difference from the OpenCL kernel_batch:
  Old: one thread per set, all 8^k polynomials evaluated serially inside
       one thread → warp tail latency kills throughput at k≥6 because
       a warp of 32 threads must wait for its slowest set.

  New: one CUDA block per set, 256 threads split the polynomial space.
       All warps in the block work on the same set and exit together when
       the cooperative abort flag fires → no warp tail latency.

Expected improvement over OpenCL: 2–8× for k≥6, negligible for k≤5.
"""

from __future__ import annotations

import numpy as np
import cupy as cp  # pre-installed on Kaggle; no pip needed

# ---------------------------------------------------------------------------
# CUDA source (compiled by NVRTC via cp.RawModule)
# ---------------------------------------------------------------------------

_CUDA_SRC = r"""
/* GF(8) multiplication table: GF8_MUL[a*8+b] = gf8_mul(a,b).
   In constant memory so repeated reads from many threads are cached. */
__constant__ int GF8_MUL[64] = {
    0,0,0,0,0,0,0,0,  0,1,2,3,4,5,6,7,  0,2,4,6,3,1,7,5,  0,3,6,5,7,4,1,2,
    0,4,3,7,6,2,5,1,  0,5,1,4,2,7,3,6,  0,6,7,1,5,3,2,4,  0,7,5,2,1,6,4,3
};

/* ── Kernel 1: single set, threads split polynomial space ─────────────────
   Same interface as the OpenCL single-set kernel.
   Grid = ceil(total_polys / BLOCK), Block = BLOCK (any power-of-2). */
extern "C" __global__ void eval_min_distance_single(
    const int* __restrict__ M,
    const int* __restrict__ s_idx,
    int k, int tmax_zeros, long long total_polys,
    int* out
) {
    long long poly_id = (long long)blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (poly_id > total_polys) return;

    int coeffs[16];
    long long tmp = poly_id;
    for (int i = 0; i < k; i++) { coeffs[i] = tmp & 7; tmp >>= 3; }

    int zeros = 0;
    for (int j = 0; j < 49; j++) {
        int val = 0;
        for (int i = 0; i < k; i++)
            val ^= GF8_MUL[coeffs[i] * 8 + M[s_idx[i] * 49 + j]];
        if (val == 0 && ++zeros > tmax_zeros) {
            atomicMax(out, zeros);
            return;
        }
    }
    atomicMax(out, zeros);
}

/* ── Kernel 2: batched BFS, ONE BLOCK PER SET ────────────────────────────
   Grid  = num_sets  (one block per set — no warp tail latency between sets)
   Block = BLOCK_SIZE (256) — threads split the 8^k polynomial space

   Shared memory layout (allocated dynamically by the launcher):
     [0 .. k*49-1]  : M_cache  — k rows of M for this set
     [k*49]         : block_max — running max-zeros across all threads
     [k*49 + 1]     : abort     — set to 1 when any thread finds zeros > tmax

   When abort fires, all threads in the block see it within one outer-loop
   iteration and exit — eliminating the warp-level tail latency of the
   old "one thread per set" OpenCL design. */
extern "C" __global__ void eval_min_distance_batch(
    const int* __restrict__ M,
    const int* __restrict__ batched_s_idx,
    int k, int tmax_zeros, int num_sets,
    int* out
) {
    int set_id = blockIdx.x;
    if (set_id >= num_sets) return;

    extern __shared__ int smem[];
    int*          M_cache   = smem;                          /* k*49 ints   */
    volatile int* block_max = (volatile int*)(smem + k*49); /* 1 int       */
    volatile int* abort     = block_max + 1;                 /* 1 int       */

    if (threadIdx.x == 0) { *block_max = 0; *abort = 0; }

    /* Load this set's row indices into registers (small k). */
    int s_local[16];
    for (int i = 0; i < k; i++)
        s_local[i] = batched_s_idx[set_id * k + i];

    /* Cooperatively load the k relevant rows of M into shared memory.
       49 floats per row → at most 16*49 = 784 ints = 3136 bytes. */
    for (int i = 0; i < k; i++)
        for (int j = threadIdx.x; j < 49; j += blockDim.x)
            M_cache[i * 49 + j] = M[s_local[i] * 49 + j];
    __syncthreads();

    long long total_polys = (1LL << (3 * k)) - 1;
    int thread_max  = 0;

    for (long long poly_id = (long long)threadIdx.x + 1;
         poly_id <= total_polys;
         poly_id += blockDim.x)
    {
        if (*abort) break;  /* cooperative early exit */

        int coeffs[16];
        long long tmp = poly_id;
        for (int i = 0; i < k; i++) { coeffs[i] = tmp & 7; tmp >>= 3; }

        int zeros = 0;
        bool bad  = false;
        for (int j = 0; j < 49; j++) {
            int val = 0;
            for (int i = 0; i < k; i++)
                val ^= GF8_MUL[coeffs[i] * 8 + M_cache[i * 49 + j]];
            if (val == 0) {
                zeros++;
                if (zeros > tmax_zeros) {
                    atomicMax((int*)block_max, zeros);
                    *abort = 1;   /* signal other threads */
                    bad    = true;
                    break;
                }
            }
        }
        if (!bad && zeros > thread_max) thread_max = zeros;
    }

    /* Final reduction: each thread contributes its local max. */
    atomicMax((int*)block_max, thread_max);
    __syncthreads();

    if (threadIdx.x == 0) out[set_id] = *block_max;
}
"""


# ---------------------------------------------------------------------------
# DistanceOracleCUDA — drop-in replacement for DistanceOracle (OpenCL)
# ---------------------------------------------------------------------------

class DistanceOracleCUDA:
    """One oracle per GPU (identified by device_id).
    Same interface as DistanceOracle so champion_search.py works unchanged."""

    BLOCK_SIZE = 256  # threads per block for the batch kernel

    def __init__(self, device_id: int, M: np.ndarray):
        self.device_id = device_id
        with cp.cuda.Device(device_id):
            self._M_gpu = cp.asarray(M.astype(np.int32))
            module = cp.RawModule(code=_CUDA_SRC)
            self._knl_single = module.get_function("eval_min_distance_single")
            self._knl_batch  = module.get_function("eval_min_distance_batch")

    def max_zeros(self, s_indices: list[int], target_distance: int) -> int:
        """Exact evaluation of one set (parallelised over polynomials)."""
        k = len(s_indices)
        if k == 0:
            return 0
        tmax_zeros  = 49 - target_distance
        total_polys = (8 ** k) - 1

        with cp.cuda.Device(self.device_id):
            s_gpu   = cp.array(s_indices, dtype=np.int32)
            out_gpu = cp.zeros(1, dtype=np.int32)
            block = 256
            grid  = (total_polys + block - 1) // block
            self._knl_single(
                (grid,), (block,),
                (self._M_gpu, s_gpu,
                 np.int32(k), np.int32(tmax_zeros), np.int64(total_polys),
                 out_gpu),
            )
            cp.cuda.Device(self.device_id).synchronize()
            return int(out_gpu[0])

    def max_zeros_batch(
        self, batched_indices: list[list[int]], target_distance: int
    ) -> list[int]:
        """High-throughput batched evaluation: one CUDA block per set."""
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

            smem_bytes = (k * 49 + 2) * 4   # M_cache + block_max + abort
            self._knl_batch(
                (num_sets,), (self.BLOCK_SIZE,),
                (self._M_gpu, s_gpu,
                 np.int32(k), np.int32(tmax_zeros), np.int32(num_sets),
                 out_gpu),
                shared_mem=smem_bytes,
            )
            cp.cuda.Device(self.device_id).synchronize()
            return out_gpu.tolist()


# ---------------------------------------------------------------------------
# Initialisation helper
# ---------------------------------------------------------------------------

def init_cuda_oracles(M: np.ndarray) -> tuple[list[DistanceOracleCUDA], int]:
    """
    Detect all CUDA GPUs, compile the kernels, return (oracles, sm_count).
    sm_count is taken from device 0 and used to compute the adaptive batch.
    """
    n = cp.cuda.runtime.getDeviceCount()
    if n == 0:
        raise RuntimeError("No CUDA GPU found.")

    oracles: list[DistanceOracleCUDA] = []
    sm_count = 40  # T4 default, overridden below
    for i in range(n):
        with cp.cuda.Device(i):
            props    = cp.cuda.runtime.getDeviceProperties(i)
            name     = props["name"].decode()
            sm_count_i = props.get("multiProcessorCount", 40)
            print(f"[CUDA] Device {i}: {name}  ({sm_count_i} SMs)")
            if i == 0:
                sm_count = sm_count_i
        oracles.append(DistanceOracleCUDA(i, M))

    return oracles, sm_count
