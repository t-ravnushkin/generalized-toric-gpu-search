"""
Phase 3: Host Expansion Loop — greedy BFS for champion toric codes.

A "champion" for a given target_distance d* is a set S of lattice-point
indices (into K_8^2) such that:
  1. d(S) >= d*                           (minimum distance condition)
  2. |S| is as large as possible          (maximise code dimension)

Strategy: BFS starting from a seed S_0.  At each level we try every unused
lattice point p and call the Oracle to check d(S ∪ {p}) >= d*.  Successful
expansions are enqueued for the next level.  All champions (sets that meet
the target and could not be expanded further, or are the best found so far)
are written to a JSON results file.

Pruning layers
--------------
1. Monotonicity  : if d(S) < d* then every superset also fails — skip.
2. Oracle early-abort : kernel fires atomic_max and returns as soon as a
   polynomial exceeds (49 - d*) zeros, avoiding full enumeration.
3. Visited set   : frozensets already evaluated are never re-checked.
4. k-cap         : for safety, expansions with |S| > MAX_K are not launched
   (8^k grows fast; users can raise MAX_K for deeper searches).
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path

from precompute import init_opencl, bounding_cube_points
from kernel import DistanceOracle

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ALL_LATTICE_INDICES: list[int] = list(range(49))   # indices 0..48
MAX_K = 15          # safety cap: 8^15 ≈ 3.5 × 10^13 — raise carefully
RESULTS_FILE = Path("champions.json")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_results() -> list[dict]:
    if RESULTS_FILE.exists():
        with RESULTS_FILE.open() as f:
            return json.load(f)
    return []


def _save_champion(record: dict, results: list[dict]) -> None:
    results.append(record)
    with RESULTS_FILE.open("w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------

def expand_search(
    seed: list[int],
    target_distance: int,
    oracle: DistanceOracle,
    lattice: list[tuple[int, int]],
    *,
    max_k: int = MAX_K,
    verbose: bool = True,
) -> list[frozenset[int]]:
    """
    BFS expansion from `seed`.  Returns every champion set found.

    Parameters
    ----------
    seed            : initial set of lattice-point indices
    target_distance : minimum distance threshold
    oracle          : DistanceOracle wrapping the OpenCL kernel
    lattice         : list of 49 (a0, a1) lattice-point tuples (for display)
    max_k           : hard cap on |S| before kernel launch
    verbose         : print progress

    Returns
    -------
    list of frozenset[int]  — all maximal champion sets discovered
    """
    tmax_zeros = 49 - target_distance

    # ---- Verify seed -------------------------------------------------------
    seed_fs = frozenset(seed)
    t0 = time.perf_counter()
    seed_mz = oracle.max_zeros(list(seed_fs), target_distance)
    dt = time.perf_counter() - t0
    seed_d  = 49 - seed_mz

    if verbose:
        print(f"\n[search] Seed  S={sorted(seed_fs)}  |S|={len(seed_fs)}")
        print(f"         d(S)={seed_d}  max_zeros={seed_mz}  ({dt*1e3:.1f} ms)")

    if seed_mz > tmax_zeros:
        print(f"[search] Seed fails target_distance={target_distance}. Aborting.")
        return []

    # ---- BFS queue ---------------------------------------------------------
    visited: set[frozenset[int]] = {seed_fs}
    queue: deque[frozenset[int]] = deque([seed_fs])
    champions: list[frozenset[int]] = []
    results = _load_results()

    best_k = len(seed_fs)

    while queue:
        S = queue.popleft()
        candidates = [p for p in ALL_LATTICE_INDICES if p not in S]

        if len(S) >= max_k:
            if verbose:
                print(f"[search] |S|={len(S)} reached max_k={max_k}, "
                      "not expanding further.")
            champions.append(S)
            continue

        expanded = False
        level_t0 = time.perf_counter()
        n_checked = 0

        for p in candidates:
            S_new = S | {p}

            if S_new in visited:
                continue
            visited.add(S_new)

            # Monotonicity already guaranteed because S passed its own check,
            # but explicitly guard against the degenerate case.
            mz = oracle.max_zeros(list(S_new), target_distance)
            n_checked += 1

            if mz <= tmax_zeros:                   # d(S_new) >= target_distance
                queue.append(S_new)
                if len(S_new) > best_k:
                    best_k = len(S_new)
                    record = {
                        "indices": sorted(S_new),
                        "lattice_points": [lattice[i] for i in sorted(S_new)],
                        "k": len(S_new),
                        "max_zeros": mz,
                        "min_distance": 49 - mz,
                        "target_distance": target_distance,
                    }
                    _save_champion(record, results)
                    if verbose:
                        print(f"\n*** NEW CHAMPION  |S|={len(S_new)}  "
                              f"d={49-mz}  max_zeros={mz} ***")
                        print(f"    indices: {sorted(S_new)}")
                expanded = True

        elapsed = time.perf_counter() - level_t0
        if verbose and n_checked:
            print(f"[search] |S|={len(S)+1}  checked={n_checked}  "
                  f"expanded={expanded}  ({elapsed*1e3:.0f} ms total, "
                  f"{elapsed/n_checked*1e3:.1f} ms/call)")

        if not expanded:
            champions.append(S)   # maximal set: cannot be grown further

    return champions


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    seed_indices: list[int],
    target_distance: int,
    max_k: int = MAX_K,
    verbose: bool = True,
) -> None:
    os.environ.setdefault("PYOPENCL_CTX", "0")

    print("=== Toric Code Champion Search ===")
    print(f"Target distance : {target_distance}")
    print(f"Max |S| (k cap) : {max_k}")
    print(f"Seed            : {seed_indices}")

    ctx, queue, M_buf, _ = init_opencl()
    oracle  = DistanceOracle(ctx, queue, M_buf)
    lattice = bounding_cube_points()

    t_start = time.perf_counter()
    champions = expand_search(
        seed_indices, target_distance, oracle, lattice,
        max_k=max_k, verbose=verbose,
    )
    elapsed = time.perf_counter() - t_start

    print(f"\n=== Search complete in {elapsed:.2f}s ===")
    print(f"Maximal champion sets found: {len(champions)}")

    # Re-load results written during search (real-time new-bests), then append
    # any maximal champion that wasn't already saved.
    results = _load_results()
    saved_indices = {frozenset(r["indices"]) for r in results}
    for ch in champions:
        mz = oracle.max_zeros(list(ch), target_distance=1)
        d  = 49 - mz
        print(f"  |S|={len(ch)}  d={d}  indices={sorted(ch)}")
        if frozenset(ch) not in saved_indices and d >= target_distance:
            record = {
                "indices": sorted(ch),
                "lattice_points": [lattice[i] for i in sorted(ch)],
                "k": len(ch),
                "max_zeros": mz,
                "min_distance": d,
                "target_distance": target_distance,
                "maximal": True,
            }
            _save_champion(record, results)
            saved_indices.add(frozenset(ch))
    print(f"Results written to: {RESULTS_FILE} ({len(results)} total entries)")


# ---------------------------------------------------------------------------
# CLI / test entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Phase 4 integration test: 3-point seed, target distance 40
    # S_0 = {0, 1, 7} = {t^(0,0), t^(0,1), t^(1,0)}
    SEED          = [0, 1, 7]
    TARGET_DIST   = 40
    K_CAP         = 10        # keep test run tractable (8^10 ≈ 10^9 threads/call)

    run(SEED, TARGET_DIST, max_k=K_CAP)
