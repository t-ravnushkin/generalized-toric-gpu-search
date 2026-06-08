#!/usr/bin/env python3
r"""Compare CNN support-point predictions with exact distances on synthetic supports.

The preferred distance backend is the repository's OpenCL ``DistanceOracle``
(``precompute.init_opencl`` + ``kernel.DistanceOracle``).  Because exact minimum-distance evaluation is exponential in support size, the script only evaluates
supports with ``k <= --max-exact-k``.  In ``--backend auto`` mode it falls back to
an exact CPU implementation for small smoke tests when PyOpenCL or an OpenCL
platform is unavailable.

Example::

    python compare_synthetic_supports.py \
        --data champions_20260607_213729.json canon_local_20260608_171130.json \
        --n-samples 24 --k-min 2 --k-max 6 --max-exact-k 6
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from gf8 import MUL_TABLE, gf8_mul, gf8_pow
from support_point_cnn_estimator import (
    GRID,
    N_POINTS,
    RidgeCnnEstimator,
    evaluate,
    format_metrics,
    load_examples,
    synthetic_examples,
    train_estimator,
    train_validation_split,
)


@dataclass(frozen=True)
class DistanceResult:
    support: tuple[int, ...]
    true_distance: int
    max_zeros: int
    backend: str


@dataclass(frozen=True)
class ComparisonRow:
    support: tuple[int, ...]
    true_distance: int
    predicted_distance: float
    backend: str

    @property
    def k(self) -> int:
        return len(self.support)

    @property
    def error(self) -> float:
        return self.predicted_distance - self.true_distance


# ---------------------------------------------------------------------------
# Synthetic support generation


def generate_supports(n: int, k_min: int, k_max: int, seed: int) -> list[tuple[int, ...]]:
    if n <= 0:
        raise ValueError("--n-samples must be positive")
    if not (1 <= k_min <= k_max <= N_POINTS):
        raise ValueError(f"expected 1 <= k-min <= k-max <= {N_POINTS}")

    rng = random.Random(seed)
    supports: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    k_values = list(range(k_min, k_max + 1))
    attempts = 0
    while len(supports) < n:
        attempts += 1
        if attempts > n * 200:
            raise RuntimeError("could not generate enough unique synthetic supports")
        k = k_values[len(supports) % len(k_values)]
        support = tuple(sorted(rng.sample(range(N_POINTS), k)))
        if support in seen:
            continue
        seen.add(support)
        supports.append(support)
    return supports


# ---------------------------------------------------------------------------
# Exact distance backends


def _group_by_k(supports: Sequence[tuple[int, ...]]) -> dict[int, list[tuple[int, tuple[int, ...]]]]:
    grouped: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    for pos, support in enumerate(supports):
        grouped[len(support)].append((pos, support))
    return dict(grouped)


def compute_distances_opencl(
    supports: Sequence[tuple[int, ...]],
    *,
    max_exact_k: int,
    allow_cpu_opencl: bool,
) -> list[DistanceResult]:
    """Compute exact distances with the repository OpenCL oracle.

    The oracle is called with ``target_distance=1`` so successful non-degenerate
    supports are fully scanned rather than screened against a high threshold.
    """
    from precompute import init_opencl
    from kernel import DistanceOracle

    ctx, queue, m_buf, _ = init_opencl(allow_cpu=allow_cpu_opencl)
    oracle = DistanceOracle(ctx, queue, m_buf)

    out_by_pos: dict[int, DistanceResult] = {}
    for k, items in sorted(_group_by_k(supports).items()):
        if k > max_exact_k:
            print(f"Skipping {len(items)} supports with k={k} (> max_exact_k={max_exact_k}).", file=sys.stderr)
            continue
        if k > 21:
            print(f"Skipping {len(items)} supports with k={k}; exact OpenCL kernel supports k <= 21.", file=sys.stderr)
            continue
        batch = [list(support) for _, support in items]
        max_zeros = oracle.max_zeros_batch(batch, target_distance=1)
        for (pos, support), mz in zip(items, max_zeros, strict=True):
            out_by_pos[pos] = DistanceResult(support, N_POINTS - int(mz), int(mz), "opencl")

    return [out_by_pos[i] for i in sorted(out_by_pos)]


def _build_eval_matrix_cpu() -> np.ndarray:
    lattice = [(a, b) for a in range(GRID) for b in range(GRID)]
    torus = [(x, y) for x in range(1, 8) for y in range(1, 8)]
    matrix = np.zeros((N_POINTS, N_POINTS), dtype=np.uint8)
    for i, (a, b) in enumerate(lattice):
        for j, (x, y) in enumerate(torus):
            matrix[i, j] = gf8_mul(gf8_pow(x, a), gf8_pow(y, b))
    return matrix


def _coeff_chunk(start: int, stop: int, k: int) -> np.ndarray:
    nums = np.arange(start, stop, dtype=np.uint64)
    coeffs = np.empty((len(nums), k), dtype=np.uint8)
    tmp = nums.copy()
    for col in range(k):
        coeffs[:, col] = (tmp & np.uint64(7)).astype(np.uint8)
        tmp >>= np.uint64(3)
    return coeffs


def _max_zeros_cpu_exact(support: tuple[int, ...], matrix: np.ndarray, mul_table: np.ndarray, chunk_size: int) -> int:
    k = len(support)
    if k == 0:
        return 0
    rows = matrix[list(support)]
    total = 8**k
    best = 0
    for start in range(1, total, chunk_size):  # skip the all-zero coefficient vector
        stop = min(total, start + chunk_size)
        coeffs = _coeff_chunk(start, stop, k)
        values = np.zeros((stop - start, N_POINTS), dtype=np.uint8)
        for col in range(k):
            values ^= mul_table[coeffs[:, col, None], rows[col][None, :]]
        zeros = np.count_nonzero(values == 0, axis=1)
        chunk_best = int(zeros.max(initial=0))
        if chunk_best > best:
            best = chunk_best
    return best


def compute_distances_cpu_exact(
    supports: Sequence[tuple[int, ...]],
    *,
    max_exact_k: int,
    chunk_size: int,
) -> list[DistanceResult]:
    """Compute exact distances on CPU; intended only for small fallback runs."""
    matrix = _build_eval_matrix_cpu()
    mul_table = np.asarray(MUL_TABLE, dtype=np.uint8)
    out: list[DistanceResult] = []
    for support in supports:
        k = len(support)
        if k > max_exact_k:
            print(f"Skipping support {support}: k={k} > max_exact_k={max_exact_k}.", file=sys.stderr)
            continue
        mz = _max_zeros_cpu_exact(support, matrix, mul_table, chunk_size)
        out.append(DistanceResult(support, N_POINTS - mz, mz, "cpu-exact"))
    return out


def compute_distances(
    supports: Sequence[tuple[int, ...]],
    *,
    backend: str,
    max_exact_k: int,
    allow_cpu_opencl: bool,
    cpu_chunk_size: int,
) -> list[DistanceResult]:
    if backend in {"auto", "opencl"}:
        try:
            return compute_distances_opencl(
                supports,
                max_exact_k=max_exact_k,
                allow_cpu_opencl=allow_cpu_opencl,
            )
        except Exception as exc:
            if backend == "opencl":
                raise
            print(f"Warning: OpenCL distance oracle unavailable ({exc}); falling back to CPU exact.", file=sys.stderr)
    return compute_distances_cpu_exact(supports, max_exact_k=max_exact_k, chunk_size=cpu_chunk_size)


# ---------------------------------------------------------------------------
# Estimator and reporting


def build_estimator(args: argparse.Namespace) -> RidgeCnnEstimator:
    if args.model_in is not None:
        estimator = RidgeCnnEstimator.load(args.model_in)
        print(f"Loaded model: {args.model_in}")
        return estimator

    if args.smoke_train:
        examples = synthetic_examples(args.smoke_count, args.seed)
    else:
        if not args.data:
            raise SystemExit("Provide --data files, --model-in, or --smoke-train.")
        examples = load_examples(args.data, target_field=args.target_field)
    if not examples:
        raise SystemExit("No usable training examples found.")

    train, val = train_validation_split(examples, args.val_fraction, args.seed)
    estimator = train_estimator(train, ridge=args.ridge)
    print(f"Training examples: train={len(train)} validation={len(val)}")
    train_metrics = evaluate(estimator, train)
    val_metrics = evaluate(estimator, val)
    print(f"Train {format_metrics(train_metrics)}")
    if val:
        print(f"Validation {format_metrics(val_metrics)}")
    if args.model_out is not None:
        estimator.save(args.model_out)
        print(f"Saved model: {args.model_out}")
    return estimator


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarize(rows: Sequence[ComparisonRow]) -> dict[str, float]:
    y = np.asarray([row.true_distance for row in rows], dtype=np.float64)
    pred = np.asarray([row.predicted_distance for row in rows], dtype=np.float64)
    err = pred - y
    return {
        "count": float(len(rows)),
        "mae": float(np.mean(np.abs(err))) if len(rows) else float("nan"),
        "rmse": float(np.sqrt(np.mean(err * err))) if len(rows) else float("nan"),
        "bias": float(np.mean(err)) if len(rows) else float("nan"),
        "pearson": _corr(y, pred),
        "spearman": _corr(_rankdata(y), _rankdata(pred)) if len(rows) else float("nan"),
    }


def print_summary(rows: Sequence[ComparisonRow]) -> None:
    overall = summarize(rows)
    print(
        "Overall: "
        f"n={int(overall['count'])} "
        f"MAE={overall['mae']:.3f} RMSE={overall['rmse']:.3f} "
        f"bias={overall['bias']:.3f} pearson={overall['pearson']:.3f} "
        f"spearman={overall['spearman']:.3f}"
    )
    by_k: dict[int, list[ComparisonRow]] = defaultdict(list)
    for row in rows:
        by_k[row.k].append(row)
    for k in sorted(by_k):
        metrics = summarize(by_k[k])
        print(
            f"  k={k}: n={int(metrics['count'])} "
            f"MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f} bias={metrics['bias']:.3f}"
        )


def print_rows(rows: Sequence[ComparisonRow], limit: int) -> None:
    if limit <= 0:
        return
    print("\nSample comparison rows:")
    print("#  k  true_d  pred_d   error   backend    support")
    for i, row in enumerate(rows[:limit], start=1):
        support = ",".join(str(x) for x in row.support)
        print(
            f"{i:2d} {row.k:2d} {row.true_distance:7d} "
            f"{row.predicted_distance:7.3f} {row.error:8.3f} "
            f"{row.backend:9s} [{support}]"
        )


def write_jsonl(rows: Sequence[ComparisonRow], path: Path) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {
                        "indices": list(row.support),
                        "k": row.k,
                        "true_min_distance": row.true_distance,
                        "predicted_min_distance": row.predicted_distance,
                        "error": row.error,
                        "backend": row.backend,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="*", type=Path, help="training JSON/JSONL files for the CNN estimator")
    parser.add_argument("--target-field", default="min_distance", help="target field to learn from --data")
    parser.add_argument("--model-in", type=Path, help="load an existing support-point CNN .npz model")
    parser.add_argument("--model-out", type=Path, help="optional path to save a newly trained estimator")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--smoke-train", action="store_true", help="train on toy synthetic targets instead of result files")
    parser.add_argument("--smoke-count", type=int, default=96)
    parser.add_argument("--n-samples", type=int, default=24, help="number of synthetic supports to compare")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-exact-k", type=int, default=6, help="skip exact distance computation above this k")
    parser.add_argument(
        "--backend",
        choices=("auto", "opencl", "cpu-exact"),
        default="auto",
        help="distance backend; auto tries OpenCL then CPU exact fallback",
    )
    parser.add_argument("--allow-cpu-opencl", action="store_true", help="allow CPU OpenCL devices in init_opencl")
    parser.add_argument("--cpu-chunk-size", type=int, default=65536, help="coefficient-vector chunk size for CPU fallback")
    parser.add_argument("--show", type=int, default=24, help="number of comparison rows to print")
    parser.add_argument("--out", type=Path, help="optional JSONL output path for per-support comparisons")
    args = parser.parse_args(argv)

    if args.max_exact_k > args.k_max:
        print(f"Note: --max-exact-k={args.max_exact_k} but --k-max={args.k_max}; all sampled k are eligible.")

    t0 = time.perf_counter()
    estimator = build_estimator(args)
    supports = generate_supports(args.n_samples, args.k_min, args.k_max, args.seed)
    print(f"Generated {len(supports)} synthetic supports with k in [{args.k_min}, {args.k_max}].")

    distances = compute_distances(
        supports,
        backend=args.backend,
        max_exact_k=args.max_exact_k,
        allow_cpu_opencl=args.allow_cpu_opencl,
        cpu_chunk_size=args.cpu_chunk_size,
    )
    if not distances:
        raise SystemExit("No distances computed; reduce --k-max or raise --max-exact-k.")

    preds = estimator.predict_indices([item.support for item in distances])
    rows = [
        ComparisonRow(item.support, item.true_distance, float(pred), item.backend)
        for item, pred in zip(distances, preds, strict=True)
    ]
    print_summary(rows)
    print_rows(rows, args.show)
    if args.out is not None:
        write_jsonl(rows, args.out)
        print(f"Wrote comparison rows: {args.out}")
    print(f"Elapsed: {time.perf_counter() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
