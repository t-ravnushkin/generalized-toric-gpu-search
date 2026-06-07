import json
import os
import time
from pathlib import Path
from tqdm import tqdm

from precompute import init_opencl, bounding_cube_points
from kernel import DistanceOracle

MAX_K = 9
RESULTS_FILE = Path("champions_batched.json")
TARGET_TOTAL_THREADS = 50_000_000  # Will saturate the M4 Pro


def _save_champions(records: list[dict]) -> None:
    results = []
    if RESULTS_FILE.exists():
        with RESULTS_FILE.open() as f:
            results = json.load(f)
    results.extend(records)
    with RESULTS_FILE.open("w") as f:
        json.dump(results, f, indent=2)


def batched_bfs(
    target_distance: int,
    oracle: DistanceOracle,
    lattice: list[tuple[int, int]],
    max_k: int = MAX_K,
):
    print(f"=== Batched BFS  target_distance={target_distance} ===")

    # 1. Symmetry reduction: Lock the first point to index 0 (origin)
    # This reduces the initial search space by 49x
    current_level = [[0]]
    champions = []

    k = 1
    while current_level and k < max_k:
        next_k = k + 1

        # Generate all combinations perfectly uniquely WITHOUT a visited set memory sink
        next_candidates = []
        for S in current_level:
            last_p = S[-1]  # Elements are appended in sorted order
            for p in range(last_p + 1, 49):
                next_candidates.append(S + [p])

        if not next_candidates:
            break

        total_candidates = len(next_candidates)
        threads_per_set = (8**next_k) - 1

        # 2. Dynamic Sizing
        batch_size = max(1, TARGET_TOTAL_THREADS // threads_per_set)

        print(
            f"\nLevel {next_k}: {total_candidates} candidates. "
            f"Dynamic batch size: {batch_size} ({threads_per_set} threads/set)"
        )

        valid_next_level = []
        level_champions = []
        t0 = time.perf_counter()

        # 3. Process the level in chunks
        for i in tqdm(
            range(0, total_candidates, batch_size),
            desc=f"k={next_k}",
            dynamic_ncols=True,
        ):
            batch = next_candidates[i : i + batch_size]

            # Fire to GPU
            results = oracle.max_zeros_batch(batch, target_distance)

            # Analyze results
            tmax_zeros = 49 - target_distance
            for S_new, mz in zip(batch, results):
                if mz <= tmax_zeros:
                    valid_next_level.append(S_new)

                    # Log successful champion
                    level_champions.append(
                        {
                            "indices": S_new,
                            "lattice_points": [lattice[idx] for idx in S_new],
                            "k": next_k,
                            "max_zeros": mz,
                            "min_distance": 49 - mz,
                            "target_distance": target_distance,
                        }
                    )

        elapsed = time.perf_counter() - t0
        print(
            f"  Passed filter: {len(valid_next_level)} / {total_candidates} in {elapsed:.2f}s"
        )

        # Save real-time level champions
        if level_champions:
            _save_champions(level_champions)
            print(f"  >> Saved {len(level_champions)} new valid sets to JSON.")

        current_level = valid_next_level
        k = next_k


if __name__ == "__main__":
    os.environ.setdefault("PYOPENCL_CTX", "0")
    ctx, queue, M_buf, _ = init_opencl()
    oracle = DistanceOracle(ctx, queue, M_buf)
    lattice = bounding_cube_points()

    # NOTE: Set target_distance to 41 to discover the k=8 conic subsets
    # instead of pruning them early!
    batched_bfs(target_distance=41, oracle=oracle, lattice=lattice, max_k=8)
