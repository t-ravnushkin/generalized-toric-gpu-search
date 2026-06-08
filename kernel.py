import numpy as np
import pyopencl as cl

KERNEL_SRC = r"""
__constant int GF8_MUL[64] = {
    0,0,0,0,0,0,0,0,  0,1,2,3,4,5,6,7,  0,2,4,6,3,1,7,5,  0,3,6,5,7,4,1,2,
    0,4,3,7,6,2,5,1,  0,5,1,4,2,7,3,6,  0,6,7,1,5,3,2,4,  0,7,5,2,1,6,4,3
};

/* Build bit-plane masks for the current set:
   bitpat[b][i][v] is a 49-bit mask whose j-th bit is the b-th output bit of
   GF8_MUL[v * M[s[i], j]].  This turns each polynomial evaluation into
   3*k XORs plus one popcount. */
inline void build_bitpat(
    __local ulong* bitpat,
    __global const int* M,
    __local const int* s_local,
    int k
) {
    int lid = (int)get_local_id(0);
    int lsz = (int)get_local_size(0);
    int total = 3 * k * 8;

    for (int idx = lid; idx < total; idx += lsz) {
        int b = idx / (k * 8);
        int rem = idx - b * k * 8;
        int i = rem / 8;
        int v = rem - i * 8;
        int row = s_local[i];

        ulong mask = 0UL;
        for (int j = 0; j < 49; j++) {
            int val = GF8_MUL[v * 8 + M[row * 49 + j]];
            if ((val >> b) & 1)
                mask |= (1UL << j);
        }
        bitpat[idx] = mask;
    }
}

/* --- KERNEL 1: Single Set, Parallelized over Polynomials (For Exact Math) --- */
__kernel void eval_min_distance(
    __global const int* M,
    __global const int* s_idx,
    int k, int tmax_zeros,
    __global int* out
) {
    int poly_id = (int)get_global_id(0) + 1; 

    int coeffs[16];
    int tmp = poly_id;
    for (int i = 0; i < k; i++) {
        coeffs[i] = tmp & 7;
        tmp >>= 3;
    }

    int zeros = 0;
    for (int j = 0; j < 49; j++) {
        int val = 0;
        for (int i = 0; i < k; i++) {
            int m_val = M[s_idx[i] * 49 + j];
            val ^= GF8_MUL[coeffs[i] * 8 + m_val];
        }
        if (val == 0) {
            zeros++;
            if (zeros > tmax_zeros) {
                atomic_max(out, zeros);
                return;
            }
        }
    }
    atomic_max(out, zeros);
}

/* --- KERNEL 2: Batched Sets, one work-group per set, bit-parallel --- */
__kernel void eval_min_distance_batch_bp(
    __global const int* M,
    __global const int* batched_s_idx,
    int k, int tmax_zeros,
    int num_sets,
    __local ulong* bitpat,
    __local int* scratch,
    __global int* out
) {
    int set_id = (int)get_group_id(0);
    int lid = (int)get_local_id(0);
    int lsz = (int)get_local_size(0);
    if (set_id >= num_sets) return;

    int base_idx = set_id * k;

    __local int local_s[16];
    __local volatile int abort_flag;

    if (lid < k)
        local_s[lid] = batched_s_idx[base_idx + lid];
    if (lid == 0)
        abort_flag = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    build_bitpat(bitpat, M, local_s, k);
    barrier(CLK_LOCAL_MEM_FENCE);

    const ulong TORUS_MASK = (1UL << 49) - 1UL;
    long total_polys = (1L << (3 * k)) - 1L;

    int thread_max = 0;
    for (long poly_id = (long)lid + 1L;
         poly_id <= total_polys;
         poly_id += (long)lsz)
    {
        if (abort_flag) break;

        ulong v0 = 0UL, v1 = 0UL, v2 = 0UL;
        long tmp = poly_id;
        for (int i = 0; i < k; i++) {
            int c = (int)(tmp & 7L);
            tmp >>= 3;
            v0 ^= bitpat[          i * 8 + c];
            v1 ^= bitpat[    k * 8 + i * 8 + c];
            v2 ^= bitpat[2 * k * 8 + i * 8 + c];
        }

        int zeros = popcount((~v0) & (~v1) & (~v2) & TORUS_MASK);
        if (zeros > tmax_zeros) {
            thread_max = zeros;
            abort_flag = 1;
            break;
        }
        if (zeros > thread_max) thread_max = zeros;
    }

    scratch[lid] = thread_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = lsz >> 1; offset > 0; offset >>= 1) {
        if (lid < offset) {
            int other = scratch[lid + offset];
            if (other > scratch[lid])
                scratch[lid] = other;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
        out[set_id] = scratch[0];
}
"""


class DistanceOracle:
    def __init__(self, ctx: cl.Context, queue: cl.CommandQueue, M_buf: cl.Buffer):
        self.ctx = ctx
        self.queue = queue
        self.M_buf = M_buf
        prog = cl.Program(ctx, KERNEL_SRC).build()
        self._knl_single = cl.Kernel(prog, "eval_min_distance")
        self._knl_batch = cl.Kernel(prog, "eval_min_distance_batch_bp")
        self._wg_size = self._choose_work_group_size()
        # Pre-allocated GPU buffers; grown lazily, never shrunk.
        self._s_buf: cl.Buffer | None = None
        self._out_buf: cl.Buffer | None = None
        self._s_cap = 0   # capacity in int32 elements
        self._out_cap = 0

    def _choose_work_group_size(self) -> int:
        device = self.queue.device
        max_wg = int(device.max_work_group_size)
        wg = min(256, max_wg)
        # The reduction assumes a power-of-two local size.
        return 1 << (wg.bit_length() - 1)

    def _ensure_buffers(self, s_count: int, out_count: int) -> None:
        if s_count > self._s_cap:
            self._s_buf = cl.Buffer(
                self.ctx, cl.mem_flags.READ_ONLY, size=s_count * 4
            )
            self._s_cap = s_count
        if out_count > self._out_cap:
            self._out_buf = cl.Buffer(
                self.ctx, cl.mem_flags.READ_WRITE, size=out_count * 4
            )
            self._out_cap = out_count

    def max_zeros(self, s_indices: list[int], target_distance: int) -> int:
        """Original single-set exact evaluation (uses Kernel 1)"""
        k = len(s_indices)
        if k == 0:
            return 0
        tmax_zeros = 49 - target_distance
        total_polys = (8**k) - 1

        s_np = np.array(s_indices, dtype=np.int32)
        out_np = np.zeros(1, dtype=np.int32)
        s_buf = cl.Buffer(
            self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=s_np
        )
        out_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=out_np,
        )

        self._knl_single(
            self.queue,
            (total_polys,),
            None,
            self.M_buf,
            s_buf,
            np.int32(k),
            np.int32(tmax_zeros),
            out_buf,
        )
        cl.enqueue_copy(self.queue, out_np, out_buf)
        self.queue.finish()
        return int(out_np[0])

    def max_zeros_batch(
        self, batched_indices: list[list[int]], target_distance: int
    ) -> list[int]:
        """High-performance batched BFS evaluation (one work-group per set)."""
        num_sets = len(batched_indices)
        if num_sets == 0:
            return []
        k = len(batched_indices[0])
        if k > 16:
            raise ValueError("OpenCL bit-parallel kernel supports k <= 16")
        tmax_zeros = 49 - target_distance

        flat_indices = np.array(
            [idx for subset in batched_indices for idx in subset], dtype=np.int32
        )
        out_np = np.zeros(num_sets, dtype=np.int32)

        self._ensure_buffers(len(flat_indices), num_sets)
        s_buf = self._s_buf
        out_buf = self._out_buf
        assert s_buf is not None and out_buf is not None
        cl.enqueue_copy(self.queue, s_buf, flat_indices)
        cl.enqueue_copy(self.queue, out_buf, out_np)

        self._knl_batch(
            self.queue,
            (num_sets * self._wg_size,),
            (self._wg_size,),
            self.M_buf,
            s_buf,
            np.int32(k),
            np.int32(tmax_zeros),
            np.int32(num_sets),
            cl.LocalMemory(3 * k * 8 * 8),
            cl.LocalMemory(self._wg_size * 4),
            out_buf,
        )
        cl.enqueue_copy(self.queue, out_np, out_buf)
        self.queue.finish()
        return out_np.tolist()
