import numpy as np
import pyopencl as cl

KERNEL_SRC = r"""
__constant int GF8_MUL[64] = {
    0,0,0,0,0,0,0,0,  0,1,2,3,4,5,6,7,  0,2,4,6,3,1,7,5,  0,3,6,5,7,4,1,2,
    0,4,3,7,6,2,5,1,  0,5,1,4,2,7,3,6,  0,6,7,1,5,3,2,4,  0,7,5,2,1,6,4,3
};

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

/* --- KERNEL 2: Batched Sets, Parallelized over Combinations (For BFS) --- */
__kernel void eval_min_distance_batch(
    __global const int* M,
    __global const int* batched_s_idx,
    int k, int tmax_zeros,
    __global int* out
) {
    int set_id = get_global_id(0);
    int base_idx = set_id * k;
    int total_polys = (1 << (3 * k)) - 1; 

    int local_s[16];
    for(int i = 0; i < k; i++){
        local_s[i] = batched_s_idx[base_idx + i];
    }

    int max_z = 0;
    for (int poly_id = 1; poly_id <= total_polys; poly_id++) {
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
                int m_val = M[local_s[i] * 49 + j];
                val ^= GF8_MUL[coeffs[i] * 8 + m_val];
            }
            if (val == 0) {
                zeros++;
                if (zeros > tmax_zeros) {
                    out[set_id] = zeros;
                    return; 
                }
            }
        }
        if (zeros > max_z) max_z = zeros;
    }
    out[set_id] = max_z;
}
"""


class DistanceOracle:
    def __init__(self, ctx: cl.Context, queue: cl.CommandQueue, M_buf: cl.Buffer):
        self.ctx = ctx
        self.queue = queue
        self.M_buf = M_buf
        prog = cl.Program(ctx, KERNEL_SRC).build()
        self._knl_single = cl.Kernel(prog, "eval_min_distance")
        self._knl_batch = cl.Kernel(prog, "eval_min_distance_batch")
        # Pre-allocated GPU buffers; grown lazily, never shrunk.
        self._s_buf: cl.Buffer | None = None
        self._out_buf: cl.Buffer | None = None
        self._s_cap = 0   # capacity in int32 elements
        self._out_cap = 0

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
        """High-performance batched BFS evaluation (uses Kernel 2)"""
        num_sets = len(batched_indices)
        if num_sets == 0:
            return []
        k = len(batched_indices[0])
        tmax_zeros = 49 - target_distance

        flat_indices = np.array(
            [idx for subset in batched_indices for idx in subset], dtype=np.int32
        )
        out_np = np.zeros(num_sets, dtype=np.int32)
<<<<<<< HEAD

        self._ensure_buffers(len(flat_indices), num_sets)
        s_buf = self._s_buf
        out_buf = self._out_buf
        assert s_buf is not None and out_buf is not None
        # Upload inputs; zero the output region used this call.
        cl.enqueue_copy(self.queue, s_buf, flat_indices)
        cl.enqueue_copy(self.queue, out_buf, out_np)
=======
        s_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=flat_indices,
        )
        out_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=out_np,
        )
>>>>>>> main

        self._knl_batch(
            self.queue,
            (num_sets,),
            None,
            self.M_buf,
            s_buf,
            np.int32(k),
            np.int32(tmax_zeros),
            out_buf,
        )
        cl.enqueue_copy(self.queue, out_np, out_buf)
        self.queue.finish()
        return out_np.tolist()
