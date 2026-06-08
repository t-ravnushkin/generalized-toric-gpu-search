#!/usr/bin/env python3
"""
Fixed-dimension evolutionary search for GF(8) generalised toric codes.

This is a candidate factory, not a certificate engine: it evolves whole
k-point subsets of [0,6]^2 directly and scores them by sampled nonzero
coefficient vectors.  Promising candidates are rechecked with larger samples
and written as unverified JSONL records for later exact/BZ verification.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np
from tqdm.auto import tqdm

from codetables import bounds_for_n
from gf8 import MUL_TABLE
from precompute import bounding_cube_points, build_eval_matrix

N_POINTS = 49
GF8_MUL = np.array(MUL_TABLE, dtype=np.uint8)


def index_to_point(i: int) -> tuple[int, int]:
    return divmod(int(i), 7)


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def random_set(k: int, rng: random.Random) -> tuple[int, ...]:
    return tuple(sorted(rng.sample(range(N_POINTS), k)))


def parse_seed_set(text: str) -> tuple[int, ...]:
    values = tuple(sorted({int(x.strip()) for x in text.split(",") if x.strip()}))
    if any(i < 0 or i >= N_POINTS for i in values):
        raise argparse.ArgumentTypeError("seed-set indices must be in 0..48")
    return values


def make_child(
    a: tuple[int, ...],
    b: tuple[int, ...],
    k: int,
    rng: random.Random,
) -> tuple[int, ...]:
    pool = list(set(a) | set(b))
    rng.shuffle(pool)
    child = set(pool[: min(k, len(pool))])
    while len(child) < k:
        child.add(rng.randrange(N_POINTS))
    return tuple(sorted(child))


def mutate(
    s: tuple[int, ...],
    k: int,
    rng: random.Random,
    mutation_rate: float,
) -> tuple[int, ...]:
    child = set(s)
    changed = False
    for old in list(s):
        if rng.random() < mutation_rate:
            child.remove(old)
            while True:
                new = rng.randrange(N_POINTS)
                if new not in child:
                    child.add(new)
                    break
            changed = True

    if not changed and rng.random() < mutation_rate:
        old = rng.choice(tuple(child))
        child.remove(old)
        while True:
            new = rng.randrange(N_POINTS)
            if new not in child:
                child.add(new)
                break

    if len(child) != k:
        raise RuntimeError("mutation changed set size")
    return tuple(sorted(child))


def unique_fill(
    population: list[tuple[int, ...]],
    k: int,
    size: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    for s in population:
        if len(s) == k and s not in seen:
            seen.add(s)
            out.append(s)
            if len(out) == size:
                return out
    while len(out) < size:
        s = random_set(k, rng)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def tournament_pick(
    ranked: list[tuple[tuple[int, ...], int, int]],
    rng: random.Random,
    size: int = 4,
) -> tuple[int, ...]:
    sample = rng.sample(ranked, min(size, len(ranked)))
    return max(sample, key=lambda item: (item[1], -item[2]))[0]


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFF_FFFF_FFFF_FFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFF_FFFF_FFFF_FFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFF_FFFF_FFFF_FFFF
    return x ^ (x >> 31)


class DistanceOracleCPU:
    """Small sampled fallback for smoke tests when no GPU/OpenCL device exists."""

    def __init__(self, M: np.ndarray):
        self.M = M.astype(np.uint8)

    def _coeffs(self, k: int, set_id: int, work_id: int, sample_seed: int) -> list[int]:
        coeffs = []
        any_nonzero = False
        for i in range(k):
            x = (
                int(sample_seed)
                ^ (set_id * 0x9E3779B97F4A7C15)
                ^ (work_id * 0xBF58476D1CE4E5B9)
                ^ ((i + 1) * 0x94D049BB133111EB)
            )
            c = splitmix64(x) & 7
            any_nonzero = any_nonzero or c != 0
            coeffs.append(c)
        if not any_nonzero:
            coeffs[splitmix64(sample_seed ^ work_id ^ set_id) % k] = 1
        return coeffs

    def max_zeros_batch(
        self,
        batched_indices: list[list[int]],
        target_distance: int,
        sample_count: int = 0,
        sample_seed: int = 0,
    ) -> list[int]:
        if sample_count <= 0:
            raise ValueError("CPU fallback supports sampled evaluation only")
        tmax_zeros = N_POINTS - target_distance
        out: list[int] = []
        for set_id, indices in enumerate(batched_indices):
            rows = self.M[np.array(indices, dtype=np.int32)]
            best = 0
            for work_id in range(1, sample_count + 1):
                vals = np.zeros(N_POINTS, dtype=np.uint8)
                for c, row in zip(
                    self._coeffs(len(indices), set_id, work_id, sample_seed),
                    rows,
                ):
                    vals ^= GF8_MUL[c, row]
                zeros = int(np.count_nonzero(vals == 0))
                if zeros > best:
                    best = zeros
                if zeros > tmax_zeros:
                    break
            out.append(best)
        return out


def init_oracles():
    M = build_eval_matrix()

    try:
        from kernel_cuda_bp import init_cuda_oracles_bp

        oracles, sm_count = init_cuda_oracles_bp(M)
        return oracles, max(1, sm_count * 1024 * len(oracles)), "cuda-bp"
    except Exception as cuda_exc:
        print(f"[init] CUDA-BP unavailable: {cuda_exc}")

    try:
        from kernel import DistanceOracle
        from precompute import init_opencl_all

        device_triplets, _ = init_opencl_all()
        oracles = [DistanceOracle(ctx, queue, m_buf) for ctx, queue, m_buf in device_triplets]
        return oracles, max(1, 20_000 * len(oracles)), "opencl-bp"
    except Exception as opencl_all_exc:
        print(f"[init] Multi-device OpenCL unavailable: {opencl_all_exc}")

    try:
        from kernel import DistanceOracle
        from precompute import init_opencl

        ctx, queue, m_buf, _ = init_opencl(allow_cpu=True)
        return [DistanceOracle(ctx, queue, m_buf)], 2_000, "opencl-bp-cpu"
    except Exception as opencl_cpu_exc:
        print(f"[init] CPU OpenCL unavailable: {opencl_cpu_exc}")

    return [DistanceOracleCPU(M)], 256, "cpu-sampled"


def _dispatch(
    oracles,
    batch: list[tuple[int, ...]],
    target_d: int,
    sample_count: int,
    sample_seed: int,
) -> list[int]:
    if not batch:
        return []
    if len(oracles) == 1:
        return oracles[0].max_zeros_batch(
            [list(s) for s in batch],
            target_d,
            sample_count=sample_count,
            sample_seed=sample_seed,
        )

    q, r = divmod(len(batch), len(oracles))
    chunks: list[list[tuple[int, ...]]] = []
    start = 0
    for dev_id in range(len(oracles)):
        size = q + (1 if dev_id < r else 0)
        chunks.append(batch[start : start + size])
        start += size

    results: list[int] = []
    with ThreadPoolExecutor(max_workers=len(oracles)) as pool:
        futures = [
            pool.submit(
                oracle.max_zeros_batch,
                [list(s) for s in chunk],
                target_d,
                sample_count,
                sample_seed + dev_id * 1_000_003,
            )
            for dev_id, (oracle, chunk) in enumerate(zip(oracles, chunks))
            if chunk
        ]
        for future in futures:
            results.extend(future.result())
    return results


def evaluate_sets(
    oracles,
    sets: list[tuple[int, ...]],
    target_d: int,
    sample_count: int,
    sample_seed: int,
    batch_size: int,
    *,
    quiet: bool = False,
) -> list[int]:
    out: list[int] = []
    ranges = range(0, len(sets), batch_size)
    iterator = ranges if quiet else tqdm(ranges, desc="score", dynamic_ncols=True, leave=False)
    for start in iterator:
        batch = sets[start : start + batch_size]
        out.extend(_dispatch(oracles, batch, target_d, sample_count, sample_seed + start))
    return out


def recheck_candidate(
    oracles,
    candidate: tuple[int, ...],
    target_d: int,
    sample_count: int,
    rounds: int,
    seed: int,
) -> list[int]:
    distances: list[int] = []
    for r in range(rounds):
        mz = _dispatch(
            oracles,
            [candidate],
            target_d,
            sample_count,
            seed + r * 10_000_019,
        )[0]
        distances.append(N_POINTS - mz)
    return distances


def run_evolution(
    *,
    k: int,
    target_d: int,
    generations: int,
    population_size: int,
    elite_count: int,
    parent_pool: int,
    mutation_rate: float,
    sample_count: int,
    recheck_sample_count: int,
    recheck_rounds: int,
    recheck_top: int,
    recheck_every: int,
    batch_size: int | None,
    seed: int,
    seed_sets: list[tuple[int, ...]],
    results_file: Path,
    oracles_info: tuple | None = None,
    log_every: int = 1,
) -> Path:
    if not (1 <= k <= N_POINTS):
        raise ValueError("k must be in 1..49")
    if elite_count >= population_size:
        raise ValueError("elite-count must be smaller than population-size")
    if parent_pool < elite_count:
        raise ValueError("parent-pool must be at least elite-count")
    if sample_count <= 0:
        raise ValueError("sample-count must be positive for evolutionary search")

    rng = random.Random(seed)
    if oracles_info is None:
        oracles, default_batch, backend = init_oracles()
    else:
        oracles, default_batch, backend = oracles_info
    batch_size = batch_size or default_batch
    lattice = bounding_cube_points()

    population = unique_fill(
        seed_sets + [random_set(k, rng) for _ in range(population_size)],
        k,
        population_size,
        rng,
    )

    append_jsonl(
        results_file,
        {
            "type": "run_start",
            "backend": backend,
            "k": k,
            "target_d": target_d,
            "generations": generations,
            "population_size": population_size,
            "elite_count": elite_count,
            "parent_pool": parent_pool,
            "mutation_rate": mutation_rate,
            "sample_count": sample_count,
            "recheck_sample_count": recheck_sample_count,
            "recheck_rounds": recheck_rounds,
            "seed": seed,
        },
    )

    print(
        f"=== Evolutionary fixed-k search  k={k}  target_d={target_d}  "
        f"backend={backend}  pop={population_size}  sample={sample_count:,} ==="
    )
    print(f"Results -> {results_file}")

    logged: set[tuple[int, ...]] = set()
    best_overall: tuple[tuple[int, ...], int, int] | None = None

    for gen in range(1, generations + 1):
        t0 = time.perf_counter()
        mz_values = evaluate_sets(
            oracles,
            population,
            target_d,
            sample_count,
            seed + gen * 1_000_003,
            batch_size,
            quiet=True,
        )
        scored = [
            (s, N_POINTS - mz, mz)
            for s, mz in zip(population, mz_values)
        ]
        scored.sort(key=lambda item: (item[1], -item[2]), reverse=True)

        best = scored[0]
        if best_overall is None or (best[1], -best[2]) > (best_overall[1], -best_overall[2]):
            best_overall = best

        n_at_target = sum(1 for _, d, _ in scored if d >= target_d)
        avg_top = mean(d for _, d, _ in scored[: max(1, min(20, len(scored)))])
        elapsed = time.perf_counter() - t0
        show_generation = (
            gen == 1
            or gen == generations
            or (log_every > 0 and gen % log_every == 0)
        )
        if show_generation:
            print(
                f"gen {gen:4d}: best sampled d={best[1]:2d}  "
                f"top20 avg={avg_top:5.2f}  pass={n_at_target:3d}  "
                f"time={elapsed:5.1f}s"
            )

        should_recheck = (
            recheck_sample_count > 0
            and recheck_rounds > 0
            and (gen == 1 or gen % recheck_every == 0 or n_at_target > 0)
        )
        if should_recheck:
            finalists = [item for item in scored if item[1] >= target_d]
            finalists.extend(scored[:recheck_top])
            seen_finalists: set[tuple[int, ...]] = set()
            for candidate, sampled_d, sampled_mz in finalists:
                if candidate in seen_finalists or candidate in logged:
                    continue
                seen_finalists.add(candidate)
                distances = recheck_candidate(
                    oracles,
                    candidate,
                    target_d,
                    recheck_sample_count,
                    recheck_rounds,
                    seed + gen * 100_000_007,
                )
                if min(distances) >= target_d:
                    logged.add(candidate)
                    record = {
                        "type": "candidate",
                        "verified": False,
                        "name": f"evolve_k{k}_sample_d{min(distances)}",
                        "indices": list(candidate),
                        "lattice_points": [lattice[i] for i in candidate],
                        "k": k,
                        "best_known_d": target_d,
                        "sampled_min_distance": sampled_d,
                        "sampled_max_zeros": sampled_mz,
                        "recheck_min_distance": min(distances),
                        "recheck_distances": distances,
                        "sample_count": sample_count,
                        "recheck_sample_count": recheck_sample_count,
                        "recheck_rounds": recheck_rounds,
                        "generation": gen,
                    }
                    append_jsonl(results_file, record)

        ranked_parent_pool = scored[: min(parent_pool, len(scored))]
        next_population = [item[0] for item in scored[:elite_count]]

        while len(next_population) < population_size:
            if rng.random() < 0.65:
                a = tournament_pick(ranked_parent_pool, rng)
                b = tournament_pick(ranked_parent_pool, rng)
                child = make_child(a, b, k, rng)
            else:
                parent = tournament_pick(ranked_parent_pool, rng)
                child = parent
            child = mutate(child, k, rng, mutation_rate)
            next_population.append(child)

        population = unique_fill(next_population, k, population_size, rng)

    if best_overall is not None:
        append_jsonl(
            results_file,
            {
                "type": "run_complete",
                "k": k,
                "target_d": target_d,
                "best_sampled_min_distance": best_overall[1],
                "best_indices": list(best_overall[0]),
                "best_lattice_points": [index_to_point(i) for i in best_overall[0]],
                "logged_candidates": len(logged),
            },
        )

    return results_file


def default_target(k: int) -> int:
    return bounds_for_n(49, q=8, cache=True)[k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, required=True, help="target dimension")
    parser.add_argument("--target-d", type=int, help="distance threshold to hunt")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--population-size", type=int, default=300)
    parser.add_argument("--elite-count", type=int, default=30)
    parser.add_argument("--parent-pool", type=int, default=200)
    parser.add_argument("--mutation-rate", type=float, default=0.10)
    parser.add_argument("--sample-count", type=int, default=200_000)
    parser.add_argument("--recheck-sample-count", type=int, default=2_000_000)
    parser.add_argument("--recheck-rounds", type=int, default=3)
    parser.add_argument("--recheck-top", type=int, default=8)
    parser.add_argument("--recheck-every", type=int, default=10)
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="print progress every N generations; 0 prints only first and last",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--seed-set",
        action="append",
        type=parse_seed_set,
        default=[],
        help="comma-separated 0..48 indices; may be passed multiple times",
    )
    parser.add_argument("--results-file", type=Path)
    args = parser.parse_args()

    target_d = args.target_d
    if target_d is None:
        target_d = default_target(args.k)

    results_file = args.results_file
    if results_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(f"evolve_k{args.k}_{ts}.json")

    run_evolution(
        k=args.k,
        target_d=target_d,
        generations=args.generations,
        population_size=args.population_size,
        elite_count=args.elite_count,
        parent_pool=args.parent_pool,
        mutation_rate=args.mutation_rate,
        sample_count=args.sample_count,
        recheck_sample_count=args.recheck_sample_count,
        recheck_rounds=args.recheck_rounds,
        recheck_top=args.recheck_top,
        recheck_every=args.recheck_every,
        batch_size=args.batch_size,
        seed=args.seed,
        seed_sets=args.seed_set,
        results_file=results_file,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
