"""
Phase 1: Precomputation Module.
Builds the torus T^2, bounding cube K_8^2, and the 49x49 evaluation matrix M.
"""

import os
import numpy as np
import pyopencl as cl

from gf8 import gf8_mul, gf8_pow, MUL_TABLE

# Suppress PyOpenCL platform selection prompt; pick Apple/Metal automatically
os.environ.setdefault("PYOPENCL_CTX", "0")
os.environ["PYOPENCL_COMPILER_OPTIONS"] = "-cl-mad-enable -cl-fast-relaxed-math"


def torus_points() -> list[tuple[int, int]]:
    """Return the 49 points of T^2 = (GF(8)*)^2 (non-zero coordinates)."""
    return [(x, y) for x in range(1, 8) for y in range(1, 8)]


def bounding_cube_points() -> list[tuple[int, int]]:
    """Return the 49 lattice points in K_8^2 = [0,6]^2."""
    return [(a, b) for a in range(7) for b in range(7)]


def eval_monomial(a: tuple[int, int], p: tuple[int, int]) -> int:
    """Evaluate monomial t^a = t1^a0 * t2^a1 at torus point p over GF(8)."""
    return gf8_mul(gf8_pow(p[0], a[0]), gf8_pow(p[1], a[1]))


def build_eval_matrix() -> np.ndarray:
    """
    Build the 49x49 evaluation matrix M (int32).
    M[i, j] = monomial a_i evaluated at torus point p_j.
    """
    lattice = bounding_cube_points()   # rows: 49 lattice points
    torus   = torus_points()           # cols: 49 torus points
    M = np.zeros((49, 49), dtype=np.int32)
    for i, a in enumerate(lattice):
        for j, p in enumerate(torus):
            M[i, j] = eval_monomial(a, p)
    return M


def init_opencl() -> tuple[cl.Context, cl.CommandQueue, cl.Buffer, np.ndarray]:
    """
    Initialise PyOpenCL context (Apple platform), upload M to GPU, return
    (ctx, queue, M_buf, M).
    """
    # Explicitly request GPU device — on Apple Silicon get_devices()[0] may be
    # the OpenCL CPU device, which would explain 0% GPU in Activity Monitor.
    platform = cl.get_platforms()[0]
    gpu_devs = platform.get_devices(device_type=cl.device_type.GPU)
    if not gpu_devs:
        raise RuntimeError(
            "No OpenCL GPU device found on the default platform. "
            "Available devices: " + str(platform.get_devices())
        )
    device = gpu_devs[0]
    ctx   = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    M = build_eval_matrix()
    M_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                      hostbuf=M)
    print(f"[precompute] OpenCL platform : {platform.name}")
    print(f"[precompute] Device          : {device.name}  (type=GPU)")
    return ctx, queue, M_buf, M


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== GF(8) multiplication table ===")
    for row in MUL_TABLE:
        print(row)

    print("\n=== Torus T^2 (first 10 points) ===")
    print(torus_points()[:10])

    print("\n=== Bounding cube K_8^2 (first 10 points) ===")
    print(bounding_cube_points()[:10])

    print("\n=== Building evaluation matrix M ===")
    M = build_eval_matrix()
    print(f"Shape: {M.shape}, dtype: {M.dtype}")
    print("M[0:5, 0:5] (rows=first 5 lattice pts, cols=first 5 torus pts):")
    print(M[:5, :5])

    # Spot-check: M[0,0] = t^(0,0) at (1,1) = 1^0 * 1^0 = 1
    assert M[0, 0] == 1, "M[0,0] should be 1"
    # M[7,0] = lattice(1,0) at torus(1,1): 1^1 * 1^0 = 1
    assert M[7, 0] == 1, "M[7,0] should be 1"
    print("\nSpot-checks passed.")

    print("\n=== Initialising OpenCL ===")
    ctx, queue, M_buf, _ = init_opencl()
    print("OpenCL init OK.")
