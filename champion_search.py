"""
Systematic champion search for GF(8) toric codes.
Optimized for Apple Silicon (M4 Pro) with Dynamic Batched OpenCL.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
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


def _dispatch_batch(
    oracles: list[DistanceOracle],
    batch: list[list[int]],
    target_distance: int,
) -> list[int]:
    """Split a batch across multiple DistanceOracle instances (one per GPU) using threads.
    PyOpenCL releases the GIL during kernel dispatch, so threads run truly in parallel."""
    n = len(oracles)
    if n == 1:
        return oracles[0].max_zeros_batch(batch, target_distance)

    # Divide batch as evenly as possible across GPUs.
    q, r = divmod(len(batch), n)
    sizes = [q + (1 if i < r else 0) for i in range(n)]
    chunks, start = [], 0
    for s in sizes:
        chunks.append(batch[start : start + s])
        start += s

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [
            pool.submit(oracle.max_zeros_batch, chunk, target_distance)
            for oracle, chunk in zip(oracles, chunks)
            if chunk
        ]
        results: list[int] = []
        for f in futures:
            results.extend(f.result())
    return results


def global_batched_bfs(
    target_distance: int | dict[int, int],
    oracles: DistanceOracle | list[DistanceOracle],
    lattice: list[tuple[int, int]],
    results_file: Path,
    max_k: int = 10,
    resume: bool = True,
    batch_size: int = BATCH_SIZE,
):
    """
    oracles: single DistanceOracle or list (one per GPU for multi-GPU runs).
    target_distance: fixed int, or {k: d} dict from codetables.bounds_for_n(n).
    """
    if isinstance(oracles, DistanceOracle):
        oracles = [oracles]
    n_gpus = len(oracles)

    def _target(k: int) -> int:
        if isinstance(target_distance, dict):
            return target_distance.get(k, 1)
        return target_distance

    print(f"\n=== Global Batched BFS  max_k={max_k}  gpus={n_gpus} ===")
    if isinstance(target_distance, dict):
        preview = {k: target_distance[k] for k in sorted(target_distance)[:8]}
        print(f"  Per-level targets (first 8): {preview}")
    else:
        print(f"  Fixed target_distance={target_distance}")

    if resume and results_file.exists():
        k, current_level = resume_level(results_file)
        if k > 1:
            print(f"  Resuming from level k={k+1} ({len(current_level)} seeds loaded)")
    else:
        k, current_level = 1, [[0]]

    while current_level and k < max_k:
        k += 1
        td = _target(k)

        next_candidates = []
        for S in current_level:
            last_p = S[-1]
            for p in range(last_p + 1, 49):
                next_candidates.append(S + [p])

        if not next_candidates:
            break

        total_candidates = len(next_candidates)
        print(f"  Level {k}: {total_candidates} subsets  target_d={td}  batch={batch_size}")

        valid_next_level = []
        t0 = time.perf_counter()
        tmax_zeros = 49 - td

        for i in tqdm(
            range(0, total_candidates, batch_size),
            desc=f"k={k}",
            dynamic_ncols=True,
        ):
            batch = next_candidates[i : i + batch_size]
            results = _dispatch_batch(oracles, batch, td)

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

    print("\n" + "=" * 60 + "\nPART B — Global Batched BFS (codetables targets)\n" + "=" * 60)
    from codetables import bounds_for_n
    targets = bounds_for_n(49)
    print(f"  Loaded {len(targets)} per-k bounds from codetables.de")
    t0 = time.perf_counter()
    global_batched_bfs(targets, oracle, lattice, results_file, max_k=8, resume=resume)
    print(f"\nTotal BFS time: {time.perf_counter() - t0:.1f}s")

    return results_file


if __name__ == "__main__":
    main()
