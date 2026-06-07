"""
Systematic champion search for GF(8) toric codes.
Optimized for Apple Silicon (M4 Pro) with Dynamic Batched OpenCL.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from precompute import init_opencl, bounding_cube_points
from kernel import DistanceOracle

ALL_INDICES = list(range(49))
TARGET_TOTAL_THREADS = 50_000_000  # Saturates M4 Pro

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def idx(a: int, b: int) -> int:
    return a * 7 + b


def append_result(record: dict, results_file: Path) -> None:
    with results_file.open("a") as f:
        f.write(json.dumps(record) + "\n")


def check_set(
    name: str,
    s: list[int],
    oracle: DistanceOracle,
    lattice: list[tuple[int, int]],
    results_file: Path,
    target_d: int = 1,
    pbar: tqdm | None = None,
) -> dict:
    """Evaluate one specific geometric set exactly (target_distance=1)."""
    out = pbar.write if pbar is not None else print
    t0 = time.perf_counter()

    # Use the new batched oracle with a batch of 1
    mz = oracle.max_zeros_batch([s], target_distance=1)[0]
    dt = (time.perf_counter() - t0) * 1e3
    d = 49 - mz

    out(f"\n[{name}]  k={len(s)}  d={d}  ({dt:.1f} ms)")

    record = {
        "name": name,
        "indices": s,
        "lattice_points": [lattice[i] for i in s],
        "k": len(s),
        "max_zeros": mz,
        "min_distance": d,
    }
    append_result(record, results_file)
    return record


# ---------------------------------------------------------------------------
# High-Performance Batched BFS
# ---------------------------------------------------------------------------


def global_batched_bfs(
    target_distance: int,
    oracle: DistanceOracle,
    lattice: list[tuple[int, int]],
    results_file: Path,
    max_k: int = 10,
):
    print(
        f"\n=== Global Batched BFS  target_distance={target_distance}  max_k={max_k} ==="
    )

    # Start at origin to eliminate 49x translation symmetry
    current_level = [[0]]
    k = 1

    while current_level and k < max_k:
        next_k = k + 1

        # 1. Canonical generation (No memory-leaking 'visited' set)
        next_candidates = []
        for S in current_level:
            last_p = S[-1]
            for p in range(last_p + 1, 49):
                next_candidates.append(S + [p])

        if not next_candidates:
            break

        total_candidates = len(next_candidates)
        threads_per_set = (8**next_k) - 1
        batch_size = max(1, TARGET_TOTAL_THREADS // threads_per_set)

        print(f"  Level {next_k}: {total_candidates} subsets. Batch size: {batch_size}")

        valid_next_level = []
        t0 = time.perf_counter()

        # 2. Process in chunks
        for i in tqdm(
            range(0, total_candidates, batch_size),
            desc=f"k={next_k}",
            dynamic_ncols=True,
        ):
            batch = next_candidates[i : i + batch_size]
            results = oracle.max_zeros_batch(batch, target_distance)

            tmax_zeros = 49 - target_distance
            for S_new, mz in zip(batch, results):
                if mz <= tmax_zeros:
                    valid_next_level.append(S_new)
                    # Save local champions directly to disk
                    append_result(
                        {
                            "name": f"bfs_k{next_k}_d{49 - mz}",
                            "indices": S_new,
                            "lattice_points": [lattice[idx] for idx in S_new],
                            "k": next_k,
                            "max_zeros": mz,
                            "min_distance": 49 - mz,
                        },
                        results_file,
                    )

        print(
            f"  k={next_k} complete: {len(valid_next_level)} passed in {time.perf_counter() - t0:.1f}s"
        )
        current_level = valid_next_level
        k = next_k


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ.setdefault("PYOPENCL_CTX", "0")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_FILE = Path(f"champions_{run_ts}.json")
    print(f"Results file: {RUN_FILE}")

    ctx, queue_cl, M_buf, _ = init_opencl()
    oracle = DistanceOracle(ctx, queue_cl, M_buf)
    lattice = bounding_cube_points()

    # --- PART A: Structured Conic Geometry ---
    print("\n" + "=" * 60 + "\nPART A — Structured candidates\n" + "=" * 60)

    circle_8 = sorted(
        [
            idx(0, 1),
            idx(0, 6),
            idx(1, 0),
            idx(2, 2),
            idx(2, 5),
            idx(5, 2),
            idx(5, 5),
            idx(6, 0),
        ]
    )
    check_set("circle_conic_k8", circle_8, oracle, lattice, RUN_FILE)

    parabola_7 = sorted([idx(t, (t * t) % 7) for t in range(7)])
    check_set("parabola_conic_k7", parabola_7, oracle, lattice, RUN_FILE)

    # --- PART B: Global BFS ---
    # Target 40 instead of 42 to catch a wider net of high-performing codes
    # without being pruned out by the Griesmer bound limits.
    print("\n" + "=" * 60 + "\nPART B — Global Batched BFS (target d=40)\n" + "=" * 60)
    t0 = time.perf_counter()
    global_batched_bfs(30, oracle, lattice, RUN_FILE, max_k=8)
    print(f"\nTotal BFS time: {time.perf_counter() - t0:.1f}s")
