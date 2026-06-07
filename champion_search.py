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
from tqdm.auto import tqdm

from precompute import init_opencl, bounding_cube_points
from kernel import DistanceOracle

BATCH_SIZE = 250_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def idx(a: int, b: int) -> int:
    return a * 7 + b


def append_result(record: dict, results_file: Path) -> None:
    with results_file.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_results(results_file: Path) -> list[dict]:
    records = []
    with results_file.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_latest_results() -> Path | None:
    """Return the most recently created champions_*.json file, or None."""
    files = sorted(Path(".").glob("champions_*.json"))
    return files[-1] if files else None


def load_bfs_state(results_file: Path) -> dict[int, list[list[int]]]:
    """
    Return {k: [sets that passed at level k]} for every fully-completed level.
    A level is 'complete' when a sentinel {"type":"level_complete","k":k} exists.
    """
    completed: dict[int, list[list[int]]] = {}
    sets_by_k: dict[int, list[list[int]]] = {}

    for rec in load_results(results_file):
        if rec.get("type") == "level_complete":
            k = rec["k"]
            completed[k] = sets_by_k.get(k, [])
        elif "k" in rec and "indices" in rec and rec.get("name", "").startswith("bfs_"):
            sets_by_k.setdefault(rec["k"], []).append(rec["indices"])

    return completed


def resume_level(results_file: Path) -> tuple[int, list[list[int]]]:
    """
    Return (start_k, current_level) so BFS can resume after the last
    completed level.  Falls back to (1, [[0]]) when nothing is completed yet.
    """
    state = load_bfs_state(results_file)
    if not state:
        return 1, [[0]]
    max_k = max(state.keys())
    return max_k, state[max_k]


# ---------------------------------------------------------------------------
# Exact evaluation of a single named set
# ---------------------------------------------------------------------------


def check_set(
    name: str,
    s: list[int],
    oracle: DistanceOracle,
    lattice: list[tuple[int, int]],
    results_file: Path,
    target_d: int = 1,
    already_done: set[str] | None = None,
) -> dict | None:
    if already_done is not None and name in already_done:
        print(f"[{name}] already in results — skipping")
        return None

    t0 = time.perf_counter()
    mz = oracle.max_zeros(s, target_distance=target_d)
    dt = (time.perf_counter() - t0) * 1e3
    d = 49 - mz

    print(f"[{name}]  k={len(s)}  d={d}  ({dt:.1f} ms)")

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
    resume: bool = True,
):
    print(
        f"\n=== Global Batched BFS  target_distance={target_distance}  max_k={max_k} ==="
    )

    if resume and results_file.exists():
        k, current_level = resume_level(results_file)
        if k > 1:
            print(f"  Resuming from level k={k+1} ({len(current_level)} seeds loaded)")
    else:
        k, current_level = 1, [[0]]

    while current_level and k < max_k:
        k += 1

        next_candidates = []
        for S in current_level:
            last_p = S[-1]
            for p in range(last_p + 1, 49):
                next_candidates.append(S + [p])

        if not next_candidates:
            break

        total_candidates = len(next_candidates)
        print(f"  Level {k}: {total_candidates} subsets. Batch size: {BATCH_SIZE}")

        valid_next_level = []
        t0 = time.perf_counter()
        tmax_zeros = 49 - target_distance

        for i in tqdm(
            range(0, total_candidates, BATCH_SIZE),
            desc=f"k={k}",
            dynamic_ncols=True,
        ):
            batch = next_candidates[i : i + BATCH_SIZE]
            results = oracle.max_zeros_batch(batch, target_distance)

            for S_new, mz in zip(batch, results):
                if mz <= tmax_zeros:
                    valid_next_level.append(S_new)
                    append_result(
                        {
                            "name": f"bfs_k{k}_d{49 - mz}",
                            "indices": S_new,
                            "lattice_points": [lattice[i] for i in S_new],
                            "k": k,
                            "max_zeros": mz,
                            "min_distance": 49 - mz,
                        },
                        results_file,
                    )

        # Mark level complete so the next run can resume past it.
        append_result({"type": "level_complete", "k": k}, results_file)

        print(
            f"  k={k} complete: {len(valid_next_level)} passed in {time.perf_counter() - t0:.1f}s"
        )
        current_level = valid_next_level


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(results_file: Path | None = None, resume: bool = True) -> Path:
    os.environ.setdefault("PYOPENCL_CTX", "0")

    if results_file is None:
        if resume:
            results_file = find_latest_results()
        if results_file is None:
            run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = Path(f"champions_{run_ts}.json")
    print(f"Results file: {results_file}")

    ctx, queue_cl, M_buf, _ = init_opencl()
    oracle = DistanceOracle(ctx, queue_cl, M_buf)
    lattice = bounding_cube_points()

    # Collect already-evaluated named sets so Part A is idempotent.
    already_done: set[str] = set()
    if results_file.exists():
        already_done = {
            rec["name"] for rec in load_results(results_file) if "name" in rec
        }

    print("\n" + "=" * 60 + "\nPART A — Structured candidates\n" + "=" * 60)

    circle_8 = sorted([
        idx(0, 1), idx(0, 6), idx(1, 0), idx(2, 2),
        idx(2, 5), idx(5, 2), idx(5, 5), idx(6, 0),
    ])
    check_set("circle_conic_k8", circle_8, oracle, lattice, results_file,
              already_done=already_done)

    parabola_7 = sorted([idx(t, (t * t) % 7) for t in range(7)])
    check_set("parabola_conic_k7", parabola_7, oracle, lattice, results_file,
              already_done=already_done)

    print("\n" + "=" * 60 + "\nPART B — Global Batched BFS (target d=30)\n" + "=" * 60)
    t0 = time.perf_counter()
    global_batched_bfs(30, oracle, lattice, results_file, max_k=8, resume=resume)
    print(f"\nTotal BFS time: {time.perf_counter() - t0:.1f}s")

    return results_file


if __name__ == "__main__":
    main()
