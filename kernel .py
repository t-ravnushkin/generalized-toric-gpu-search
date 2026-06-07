import numpy as np
import pyopencl as cl

KERNEL_SRC = r"""
__constant int GF8_MUL[64] = {
    0,0,0,0,0,0,0,0,
    0,1,2,3,4,5,6,7,
    0,2,4,6,3,1,7,5,
    0,3,6,5,7,4,1,2,
    0,4,3,7,6,2,5,1,
    0,5,1,4,2,7,3,6,
    0,6,7,1,5,3,2,4,
    0,7,5,2,1,6,4,3
};

__kernel void eval_min_distance_batch(
    __global const int* M,
    __global const int* batched_s_idx,
    int k,
    int tmax_zeros,
    __global int* out
) {
    int poly_id = (int)get_global_id(0) + 1; // 1 to 8^k - 1
    int set_id  = (int)get_global_id(1);     // index of the combination

    int base_idx = set_id * k;

    // decode base-8 coefficients
    int coeffs[16]; // Safely handle up to k=16
    int tmp = poly_id;
    for (int i = 0; i < k; i++) {
        coeffs[i] = tmp & 7;
        tmp >>= 3;
    }

    // evaluate at each of the 49 torus points
    int zeros = 0;
    for (int j = 0; j < 49; j++) {
        int val = 0;
        for (int i = 0; i < k; i++) {
            int m_val = M[batched_s_idx[base_idx + i] * 49 + j];
            val ^= GF8_MUL[coeffs[i] * 8 + m_val];
        }
        if (val == 0) {
            zeros++;
            // early abort for this specific thread
            if (zeros > tmax_zeros) {
                atomic_max(&out[set_id], zeros);
                return;
            }
        }
    }
    atomic_max(&out[set_id], zeros);
}
"""


class DistanceOracle:
    def __init__(self, ctx: cl.Context, queue: cl.CommandQueue, M_buf: cl.Buffer):
        self.ctx = ctx
        self.queue = queue
        self.M_buf = M_buf
        prog = cl.Program(ctx, KERNEL_SRC).build()
        self._knl_batch = cl.Kernel(prog, "eval_min_distance_batch")

    def max_zeros_batch(
        self, batched_indices: list[list[int]], target_distance: int
    ) -> list[int]:
        num_sets = len(batched_indices)
        if num_sets == 0:
            return []

        k = len(batched_indices[0])
        tmax_zeros = 49 - target_distance
        total_polys = (8**k) - 1

        # Flatten the batch for OpenCL: shape (num_sets * k,)
        flat_indices = np.array(
            [idx for subset in batched_indices for idx in subset], dtype=np.int32
        )
        out_np = np.zeros(num_sets, dtype=np.int32)

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

        # Launch 2D Grid: (polynomials, sets)
        self._knl_batch(
            self.queue,
            (total_polys, num_sets),
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
