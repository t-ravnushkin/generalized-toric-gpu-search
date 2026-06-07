"""
Phase 2: OpenCL kernel for exact minimum-distance evaluation.

The minimum distance of the linear code spanned by monomials S is:
    d(S) = 49 - max_zeros(S)
where max_zeros(S) = max over all non-zero polynomials f in span(S) of
    |{ j in T^2 : f(p_j) = 0 }|

Thread mapping: global_id -> polynomial index (1 .. 8^k - 1).
Each thread extracts k GF(8) coefficients in base-8 from its index,
evaluates the polynomial at all 49 torus points using the precomputed
evaluation matrix M, counts zeros, and atomically updates the global
max-zeros counter.  Early-abort fires if zeros > target_max_zeros.
"""

import numpy as np
import pyopencl as cl

# ---------------------------------------------------------------------------
# OpenCL kernel source
# ---------------------------------------------------------------------------
KERNEL_SRC = r"""
/* ---- GF(8) multiplication table, hardcoded for cache speed ---- */
__constant int GF8_MUL[64] = {
    /* row a=0 */ 0,0,0,0,0,0,0,0,
    /* row a=1 */ 0,1,2,3,4,5,6,7,
    /* row a=2 */ 0,2,4,6,3,1,7,5,
    /* row a=3 */ 0,3,6,5,7,4,1,2,
    /* row a=4 */ 0,4,3,7,6,2,5,1,
    /* row a=5 */ 0,5,1,4,2,7,3,6,
    /* row a=6 */ 0,6,7,1,5,3,2,4,
    /* row a=7 */ 0,7,5,2,1,6,4,3
};

/*
 * eval_min_distance
 *
 * M          : flat 49*49 int32 evaluation matrix (row=monomial, col=torus pt)
 * s_idx      : array of k selected row-indices into M
 * k          : number of monomials in the candidate set S_new
 * tmax_zeros : early-abort threshold  (= 49 - target_distance)
 * out        : single int, initialised to 0 by host; receives atomic_max(zeros)
 */
__kernel void eval_min_distance(
    __global const int* M,
    __global const int* s_idx,
    int k,
    int tmax_zeros,
    __global int* out
) {
    int poly_id = (int)get_global_id(0) + 1;   /* skip zero polynomial */

    /* ---- decode base-8 coefficients ---- */
    int coeffs[49];
    int tmp = poly_id;
    for (int i = 0; i < k; i++) {
        coeffs[i] = tmp & 7;
        tmp >>= 3;
    }

    /* ---- evaluate at each of the 49 torus points ---- */
    int zeros = 0;
    for (int j = 0; j < 49; j++) {
        int val = 0;
        for (int i = 0; i < k; i++) {
            int m_val = M[s_idx[i] * 49 + j];
            val ^= GF8_MUL[coeffs[i] * 8 + m_val];
        }
        if (val == 0) {
            zeros++;
            /* early abort: already worse than target */
            if (zeros > tmax_zeros) {
                atomic_max(out, zeros);
                return;
            }
        }
    }
    atomic_max(out, zeros);
}
"""

# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

class DistanceOracle:
    """
    Wraps the OpenCL kernel for minimum-distance queries.

    Parameters
    ----------
    ctx, queue : PyOpenCL context and command queue
    M_buf      : OpenCL Buffer holding the 49x49 evaluation matrix (int32)
    """

    def __init__(self, ctx: cl.Context, queue: cl.CommandQueue, M_buf: cl.Buffer):
        self.ctx    = ctx
        self.queue  = queue
        self.M_buf  = M_buf
        prog        = cl.Program(ctx, KERNEL_SRC).build()
        self._knl   = cl.Kernel(prog, "eval_min_distance")  # reuse to avoid warning

    def max_zeros(self, s_indices: list[int], target_distance: int) -> int:
        """
        Return the maximum number of zeros attained by any non-zero polynomial
        in the span of the monomials indexed by s_indices.

        Uses early-abort once a polynomial exceeds (49 - target_distance) zeros,
        so this is also useful as a go/no-go filter.

        Parameters
        ----------
        s_indices       : list of int, indices into the 49 lattice points
        target_distance : the distance threshold we are trying to meet/exceed

        Returns
        -------
        max_zeros : int
            Contract:
            * If returned value <= (49 - target_distance): exact maximum zeros;
              the code meets the target distance.
            * If returned value >  (49 - target_distance): some polynomial
              already exceeded the threshold; early-abort fired.  The returned
              value is NOT the true maximum — only signals failure.
        """
        k = len(s_indices)
        if k == 0:
            return 0

        tmax_zeros = 49 - target_distance   # early-abort threshold

        # Total non-zero polynomials: 8^k - 1
        total_polys = (8 ** k) - 1

        # Host-side arrays
        s_np  = np.array(s_indices, dtype=np.int32)
        out_np = np.zeros(1, dtype=np.int32)

        # Buffers
        s_buf = cl.Buffer(self.ctx,
                          cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                          hostbuf=s_np)
        out_buf = cl.Buffer(self.ctx,
                            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                            hostbuf=out_np)

        # Launch kernel (use cached Kernel object to avoid per-call allocation)
        self._knl(
            self.queue, (total_polys,), None,
            self.M_buf, s_buf,
            np.int32(k), np.int32(tmax_zeros),
            out_buf
        )

        # Read back result
        cl.enqueue_copy(self.queue, out_np, out_buf)
        self.queue.finish()

        return int(out_np[0])

    def min_distance(self, s_indices: list[int], target_distance: int = 1) -> int:
        """Return minimum distance of the code spanned by s_indices."""
        return 49 - self.max_zeros(s_indices, target_distance)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from precompute import init_opencl, bounding_cube_points, torus_points
    from gf8 import MUL_TABLE

    ctx, queue, M_buf, M = init_opencl()
    oracle = DistanceOracle(ctx, queue, M_buf)

    lattice = bounding_cube_points()
    torus   = torus_points()

    # ---- Test 1: single monomial S = {0} = {t^(0,0)} = constant 1 --------
    # Every polynomial is c*1 for c in GF(8)*.
    # The constant c != 0 never evaluates to 0, so max_zeros = 0, d = 49.
    mz = oracle.max_zeros([0], target_distance=49)
    print(f"Test 1  S={{t^(0,0)}}  max_zeros={mz}  d={49-mz}  (expected d=49)")
    assert mz == 0, f"Expected 0, got {mz}"

    # ---- Test 2: two monomials S = {0, 7} = {t^(0,0), t^(1,0)} -----------
    # Polynomials: a + b*x  for (a,b) != (0,0).
    # Max zeros = number of x in GF(8)* where a + b*x = 0, i.e. x = -a/b = a/b.
    # Each nonzero polynomial a + b*x has exactly 0 or 1 zeros on T^1 component,
    # but we're on T^2 so 0 or 7 zeros (one full column of the torus if zero on x-coord,
    # or spread if polynomial depends on x).
    # Actually let's just check the value is reasonable.
    mz2 = oracle.max_zeros([0, 7], target_distance=1)
    print(f"Test 2  S={{t^(0,0), t^(1,0)}}  max_zeros={mz2}  d={49-mz2}")

    # ---- Test 3: 3 monomials, verify against brute-force Python ------------
    # S = {0, 1, 7} = {t^(0,0), t^(0,1), t^(1,0)}
    s3 = [0, 1, 7]
    mz3_gpu = oracle.max_zeros(s3, target_distance=1)

    # Brute-force in Python
    def bf_max_zeros(s_idx, M):
        k = len(s_idx)
        best = 0
        for poly_id in range(1, 8**k):
            tmp = poly_id
            coeffs = []
            for _ in range(k):
                coeffs.append(tmp & 7)
                tmp >>= 3
            zeros = 0
            for j in range(49):
                val = 0
                for ci, si in zip(coeffs, s_idx):
                    val ^= MUL_TABLE[ci][M[si, j]]
                if val == 0:
                    zeros += 1
            best = max(best, zeros)
        return best

    mz3_cpu = bf_max_zeros(s3, M)
    print(f"Test 3  S={{0,1,7}}  GPU max_zeros={mz3_gpu}  CPU max_zeros={mz3_cpu}  "
          f"({'MATCH' if mz3_gpu == mz3_cpu else 'MISMATCH'})")
    assert mz3_gpu == mz3_cpu, f"GPU/CPU mismatch: {mz3_gpu} vs {mz3_cpu}"

    # ---- Test 4: early-abort fires correctly when target is too tight -------
    # target_distance=49  =>  tmax_zeros=0.
    # S={0,1,7} has true max_zeros=7, so the code fails.
    # The oracle fires early and returns SOME value > 0 (not the exact 7).
    # We only verify that the "fail" signal is correct: returned > tmax_zeros.
    tmax_tight = 49 - 49   # = 0
    mz3_tight  = oracle.max_zeros(s3, target_distance=49)
    passes_tight = mz3_tight <= tmax_tight   # should be False (code fails)
    print(f"Test 4  early-abort  returned={mz3_tight}  tmax={tmax_tight}  "
          f"fails={'YES (correct)' if not passes_tight else 'NO (wrong)'}")
    assert not passes_tight, "Code with max_zeros=7 should fail target_distance=49"

    # Loose target: target_distance=42  =>  tmax_zeros=7.
    # max_zeros=7 <= 7, so code PASSES and we get the exact value.
    tmax_loose = 49 - 42   # = 7
    mz3_loose  = oracle.max_zeros(s3, target_distance=42)
    print(f"Test 4b loose-target  returned={mz3_loose}  tmax={tmax_loose}  "
          f"exact={'YES' if mz3_loose == mz3_cpu else 'NO'}")
    assert mz3_loose == mz3_cpu, f"Expected exact {mz3_cpu}, got {mz3_loose}"

    print("\nAll kernel smoke tests passed.")
