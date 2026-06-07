"""
Systematic champion search for GF(8) toric codes.

Mathematical background
-----------------------
Lattice K_8^2 = [0,6]^2 is the affine plane AG(2, GF(7)).
A k-monomial set S with d(S) >= 42 must be a "cap" (no 3 collinear in GF(7)^2).
By the Singleton bound d <= 50 - k, so d=42 allows k up to 8.
A conic in PG(2, GF(7)) is an oval of 8 points (no 3 collinear).
If all 8 points land in the affine chart [0,6]^2 we get a k=8 cap.

Circle conic in AG(2, GF(7)):
  x^2 + y^2 = 1  mod 7
  Points: (0,1),(0,6),(1,0),(2,2),(2,5),(5,2),(5,5),(6,0)  — 8 affine points!
  Indices: 1, 6, 7, 16, 19, 37, 40, 42

Parabola in AG(2, GF(7)):
  y = x^2 mod 7  (includes origin)
  Points: (0,0),(1,1),(2,4),(3,2),(4,2),(5,4),(6,1)  — 7 affine points
  Indices: 0, 8, 18, 23, 30, 39, 43

If a conic gives d=42, it is a [49, 8, 42] MDS toric code over GF(8).

Output
------
Each run writes its results to  champions_YYYYMMDD_HHMMSS.json
so successive runs never overwrite each other.
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


# ---------------------------------------------------------------------------
# Bounds & persistence
# ---------------------------------------------------------------------------

def griesmer_bound(k: int, d: int, q: int = 8) -> int:
    """Griesmer lower bound on n for a linear [n, k, d]_q code."""
    total, di = 0, d
    for _ in range(k):
        total += di
        di = (di + q - 1) // q   # ceil(di / q)
    return total


def is_feasible(k: int, d: int, n: int = 49, q: int = 8) -> bool:
    return n >= griesmer_bound(k, d, q)


def append_result(record: dict, results_file: Path) -> bool:
    """
    Append one record to the run's JSON file.
    Returns False (and prints a warning) if the Griesmer bound rules it out.
    """
    k, d = record["k"], record["min_distance"]
    if not is_feasible(k, d):
        print(f"  !! [49,{k},{d}] violates Griesmer bound "
              f"(min n={griesmer_bound(k,d)}) — skipping")
        return False
    results: list[dict] = []
    if results_file.exists():
        with results_file.open() as f:
            results = json.load(f)
    results.append(record)
    with results_file.open("w") as f:
        json.dump(results, f, indent=2)
    return True


# ---------------------------------------------------------------------------
# Lattice helpers
# ---------------------------------------------------------------------------

def idx(a: int, b: int) -> int:
    return a * 7 + b


# ---------------------------------------------------------------------------
# Single-set evaluation
# ---------------------------------------------------------------------------

def check_set(
    name: str,
    s: list[int],
    oracle: DistanceOracle,
    lattice: list[tuple[int, int]],
    results_file: Path,
    target_d: int = 1,
    pbar: tqdm | None = None,
) -> dict:
    """
    Evaluate one set and save the result.

    Always calls the oracle with target_distance=1 to get the exact minimum
    distance (no early-abort distortion).  target_d is only used for the
    "CHAMPION" label in the printed output.
    """
    out = pbar.write if pbar is not None else print
    t0  = time.perf_counter()
    mz  = oracle.max_zeros(s, target_distance=1)   # exact — no early abort
    dt  = (time.perf_counter() - t0) * 1e3
    d   = 49 - mz
    k   = len(s)
    pts = [lattice[i] for i in s]
    out(f"\n[{name}]")
    out(f"  indices     : {s}")
    out(f"  lattice pts : {pts}")
    out(f"  k={k}  max_zeros={mz}  d={d}  ({dt:.1f} ms)")
    if d >= target_d:
        out(f"  *** d={d} >= {target_d} — CHAMPION ***")
    record = {
        "name": name, "indices": s, "lattice_points": pts,
        "k": k, "max_zeros": mz, "min_distance": d,
    }
    saved = append_result(record, results_file)
    if saved:
        out(f"  >> Saved to {results_file.name}")
    return record


# ---------------------------------------------------------------------------
# Global BFS (level-by-level, one tqdm bar per k-level)
# ---------------------------------------------------------------------------

def global_bfs(
    target_distance: int,
    oracle: DistanceOracle,
    lattice: list[tuple[int, int]],
    results_file: Path,
    max_k: int = 10,
) -> list[frozenset[int]]:
    """
    BFS from every single-point seed simultaneously, deduplicating with a
    global visited set.  Processes sets level-by-level so each tqdm bar has
    a meaningful total.  Saves new-best champions in real time.
    Returns all maximal sets (those that could not be expanded further).
    """
    tmax      = 49 - target_distance
    print(f"\n=== Global BFS  target_distance={target_distance}  max_k={max_k} ===")

    visited:   set[frozenset[int]] = set()
    saved_keys: set[frozenset[int]] = set()   # per-run in-memory dedup for BFS
    champions: list[frozenset[int]] = []
    best_k    = 0
    level_counts: dict[int, int] = {}

    # Level 1: all 49 single-point seeds
    current_level: list[frozenset[int]] = [frozenset([p]) for p in ALL_INDICES]
    for fs in current_level:
        visited.add(fs)
    level_counts[1] = 49
    print(f"  Level 1: 49 seeds")

    k = 1
    while current_level:
        if k >= max_k:
            champions.extend(current_level)
            print(f"  Reached max_k={max_k}, stopping expansion.")
            break

        next_level: list[frozenset[int]] = []
        n_calls = n_passes = 0
        t_level = time.perf_counter()

        bar = tqdm(
            current_level,
            desc=f"  k={k}→{k+1}",
            unit="set",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}",
        )
        for S in bar:
            s_expanded = False
            for p in sorted(p for p in ALL_INDICES if p not in S):
                S_new = S | {p}
                if S_new in visited:
                    continue
                visited.add(S_new)

                mz = oracle.max_zeros(list(sorted(S_new)), target_distance)
                n_calls += 1

                if mz <= tmax:
                    next_level.append(S_new)
                    n_passes += 1
                    s_expanded = True

                    if len(S_new) > best_k:
                        best_k = len(S_new)
                        fkey = frozenset(S_new)
                        if fkey not in saved_keys:
                            saved_keys.add(fkey)
                            rec = {
                                "name": f"bfs_k{len(S_new)}_d{49-mz}",
                                "indices": sorted(S_new),
                                "lattice_points": [lattice[i] for i in sorted(S_new)],
                                "k": len(S_new),
                                "max_zeros": mz,
                                "min_distance": 49 - mz,
                                "target_distance": target_distance,
                            }
                            append_result(rec, results_file)
                            bar.write(
                                f"\n*** NEW BEST  k={len(S_new)}  d={49-mz} ***"
                                f"  {sorted(S_new)}"
                            )

            if not s_expanded:
                champions.append(S)

            ms_per_call = (time.perf_counter() - t_level) / max(n_calls, 1) * 1e3
            bar.set_postfix(
                passes=n_passes,
                calls=n_calls,
                best_k=best_k,
                ms_call=f"{ms_per_call:.2f}",
            )
        bar.close()

        level_counts[k + 1] = len(next_level)
        elapsed = time.perf_counter() - t_level
        print(f"  Level k={k}: {len(current_level)} sets → {n_passes} passed "
              f"({n_calls} oracle calls, {elapsed:.1f}s)")

        current_level = next_level
        k += 1

    print("\n--- BFS level summary ---")
    for lv in sorted(level_counts):
        print(f"  k={lv}: {level_counts[lv]} unique sets")
    print(f"Maximal champions: {len(champions)}  |  Best k found: {best_k}")
    return champions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ.setdefault("PYOPENCL_CTX", "0")

    # Each run gets its own timestamped output file
    run_ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_FILE  = Path(f"champions_{run_ts}.json")
    print(f"Results file: {RUN_FILE}")

    ctx, queue_cl, M_buf, _ = init_opencl()
    oracle  = DistanceOracle(ctx, queue_cl, M_buf)
    lattice = bounding_cube_points()

    # =========================================================
    # Part A: Test mathematically motivated special sets
    # =========================================================
    print("\n" + "="*60)
    print("PART A — Structured candidates from conic geometry")
    print("="*60)

    # 1. Circle conic: x^2 + y^2 = 1 mod 7 — 8 affine points (an oval)
    circle_8 = sorted([idx(0,1), idx(0,6), idx(1,0), idx(2,2),
                       idx(2,5), idx(5,2), idx(5,5), idx(6,0)])
    check_set("circle_conic_k8", circle_8, oracle, lattice, RUN_FILE, target_d=42)

    # 2. Circle conic minus one point each (8 × k=7 sub-sets)
    print("\n--- Circle conic: removing one point each ---")
    with tqdm(circle_8, desc="  circle -1 pt", unit="set", dynamic_ncols=True) as bar:
        for drop in bar:
            sub = [p for p in circle_8 if p != drop]
            check_set(f"circle_k7_drop{lattice[drop]}", sub, oracle, lattice,
                      RUN_FILE, target_d=42, pbar=bar)

    # 3. Parabola y = x^2 mod 7 (7 affine points, includes origin)
    parabola_7 = sorted([idx(t, (t * t) % 7) for t in range(7)])
    check_set("parabola_conic_k7", parabola_7, oracle, lattice, RUN_FILE, target_d=42)

    # 4. Circle conic + origin (9 points, expected d < 42)
    circle_plus_origin = sorted(circle_8 + [idx(0, 0)])
    check_set("circle_plus_origin_k9", circle_plus_origin, oracle, lattice,
              RUN_FILE, target_d=40)

    # =========================================================
    # Part B: Global BFS — target d >= 42, max k = 9
    # =========================================================
    print("\n" + "="*60)
    print("PART B — Global BFS: target d=42, max_k=9")
    print("="*60)

    t0     = time.perf_counter()
    champs = global_bfs(42, oracle, lattice, RUN_FILE, max_k=9)
    print(f"\nBFS completed in {time.perf_counter()-t0:.1f}s")

    # =========================================================
    # Part C: If best d=42 code has k < 6, also search d=40 deeper
    # =========================================================
    run_results: list[dict] = []
    if RUN_FILE.exists():
        with RUN_FILE.open() as f:
            run_results = json.load(f)

    best_d42_k = max((r["k"] for r in run_results if r["min_distance"] >= 42), default=0)
    if best_d42_k < 6:
        print("\n" + "="*60)
        print("PART C — BFS: target d=40, max_k=7")
        print("="*60)
        t0 = time.perf_counter()
        global_bfs(40, oracle, lattice, RUN_FILE, max_k=7)
        print(f"\nBFS completed in {time.perf_counter()-t0:.1f}s")

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "="*60)
    print("CHAMPION SUMMARY  (this run)")
    print("="*60)
    if RUN_FILE.exists():
        with RUN_FILE.open() as f:
            run_results = json.load(f)
    by_d: dict[int, list] = {}
    for r in run_results:
        by_d.setdefault(r["min_distance"], []).append(r)
    for d_val in sorted(by_d, reverse=True):
        best_k_for_d = max(r["k"] for r in by_d[d_val])
        reps = [r for r in by_d[d_val] if r["k"] == best_k_for_d]
        print(f"  d={d_val}  best k={best_k_for_d}  "
              f"({len(reps)} set(s))  example: {reps[0]['indices']}")
    print(f"\nFull results: {RUN_FILE}")
