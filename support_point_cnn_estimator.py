r"""Prototype support-point CNN estimator for GF(8) toric-code searches.

This module is intentionally lightweight: it uses NumPy only and implements a
small CNN-style feature extractor (fixed 3x3 convolution filters, support-mask,
row/column, and pairwise-geometry summaries) followed by standardized ridge
regression.  It is meant to rank candidate support sets/extensions before
expensive exact distance evaluation, not to certify code distances.

Examples
--------
Smoke-test the whole workflow with synthetic data::

    python support_point_cnn_estimator.py --smoke-test --epochs 1

Train/validate from existing canonical search JSONL files::

    python support_point_cnn_estimator.py --data canon_20260608_125209\(1\).json \
        --model-out support_point_cnn_model.npz

Score all one-point extensions of a base support::

    python support_point_cnn_estimator.py --model-in support_point_cnn_model.npz \
        --score-base 0,1,8,9
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

GRID = 7
N_POINTS = GRID * GRID


@dataclass(frozen=True)
class Example:
    indices: tuple[int, ...]
    target: float


def _iter_json_records(path: Path) -> Iterable[dict]:
    """Yield records from JSONL, a JSON object, or a JSON array.

    Some long-running search outputs in this repo are interrupted JSON arrays:
    they contain many complete objects followed by a truncated final object.
    For those, salvage the complete prefix instead of failing the whole run.
    """
    text = path.read_text().strip()
    if not text:
        return
    if text[0] in "[{":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            payload = None
            full_json_error = exc
        else:
            full_json_error = None
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield item
            return
        if isinstance(payload, dict):
            yield payload
            return
        if text[0] == "[":
            yielded = 0
            decoder = json.JSONDecoder()
            pos = 1
            while pos < len(text):
                while pos < len(text) and text[pos].isspace():
                    pos += 1
                if pos < len(text) and text[pos] == ",":
                    pos += 1
                    continue
                if pos < len(text) and text[pos] == "]":
                    return
                try:
                    item, pos = decoder.raw_decode(text, pos)
                except json.JSONDecodeError:
                    break
                if isinstance(item, dict):
                    yielded += 1
                    yield item
            if yielded:
                print(
                    f"Warning: salvaged {yielded} complete JSON-array records from "
                    f"{path}; ignored truncated/malformed tail ({full_json_error}).",
                    file=sys.stderr,
                )
                return

    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse {path} as JSON/JSONL at line {line_no}: {exc}") from exc


def _indices_from_record(record: dict) -> tuple[int, ...] | None:
    if "indices" in record:
        indices = record["indices"]
    elif "support" in record:
        indices = record["support"]
    elif "lattice_points" in record:
        indices = [int(a) * GRID + int(b) for a, b in record["lattice_points"]]
    else:
        return None

    try:
        out = tuple(sorted({int(i) for i in indices}))
    except (TypeError, ValueError):
        return None
    if not out or out[0] < 0 or out[-1] >= N_POINTS:
        return None
    return out


def _target_from_record(record: dict, target_field: str) -> float | None:
    if target_field in record:
        return float(record[target_field])
    if target_field == "min_distance":
        if "sampled_min_distance" in record:
            return float(record["sampled_min_distance"])
        if "max_zeros" in record:
            return float(N_POINTS - int(record["max_zeros"]))
        if "sampled_max_zeros" in record:
            return float(N_POINTS - int(record["sampled_max_zeros"]))
    return None


def load_examples(paths: Sequence[Path], target_field: str = "min_distance") -> list[Example]:
    examples: list[Example] = []
    seen: set[tuple[tuple[int, ...], float]] = set()
    for path in paths:
        for record in _iter_json_records(path):
            indices = _indices_from_record(record)
            target = _target_from_record(record, target_field)
            if indices is None or target is None or not math.isfinite(target):
                continue
            key = (indices, target)
            if key not in seen:
                seen.add(key)
                examples.append(Example(indices, target))
    return examples


def mask_from_indices(indices: Sequence[int]) -> np.ndarray:
    mask = np.zeros((GRID, GRID), dtype=np.float32)
    for idx in indices:
        mask[int(idx) // GRID, int(idx) % GRID] = 1.0
    return mask


FILTERS = np.array(
    [
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
        [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
    ],
    dtype=np.float32,
)


def _conv_valid(mask: np.ndarray, filt: np.ndarray) -> np.ndarray:
    out = np.empty((GRID - 2, GRID - 2), dtype=np.float32)
    for r in range(GRID - 2):
        for c in range(GRID - 2):
            out[r, c] = float(np.sum(mask[r : r + 3, c : c + 3] * filt))
    return out


def _basic_cnn_features(masks: np.ndarray) -> np.ndarray:
    """Original compact feature extractor, kept for loading older .npz models."""
    rows: list[list[float]] = []
    coords = np.arange(GRID, dtype=np.float32)
    rr, cc = np.meshgrid(coords, coords, indexing="ij")
    for mask in masks.astype(np.float32, copy=False):
        feats: list[float] = [1.0, float(mask.sum())]
        mass = max(float(mask.sum()), 1.0)
        feats.extend(
            [
                float((mask * rr).sum() / mass),
                float((mask * cc).sum() / mass),
                float(mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum()),
            ]
        )
        for filt in FILTERS:
            act = np.maximum(_conv_valid(mask, filt), 0.0)
            feats.extend([float(act.mean()), float(act.max()), float(act.sum())])
        rows.append(feats)
    return np.asarray(rows, dtype=np.float64)


def cnn_features(masks: np.ndarray) -> np.ndarray:
    """Return fixed CNN-style features for an array shaped (n, 7, 7).

    The first prototype used only a handful of global pooled convolution
    statistics.  On small exact-labeled datasets those features mostly recover
    support size ``k`` and ridge regression can collapse toward per-k/overall
    label means.  Keep the lightweight NumPy path, but expose more signal:
    local convolution summaries, row/column occupancy, raw support-mask pixels,
    and pairwise toroidal geometry histograms.
    """
    rows: list[list[float]] = []
    coords = np.arange(GRID, dtype=np.float32)
    rr, cc = np.meshgrid(coords, coords, indexing="ij")

    for mask in masks.astype(np.float32, copy=False):
        k = float(mask.sum())
        mass = max(k, 1.0)
        feats: list[float] = [1.0, k]
        feats.extend(
            [
                float((mask * rr).sum() / mass),
                float((mask * cc).sum() / mass),
                float(mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum()),
            ]
        )

        # Fixed 3x3 convolutional responses with global pooling.
        for filt in FILTERS:
            act = np.maximum(_conv_valid(mask, filt), 0.0)
            feats.extend([float(act.mean()), float(act.max()), float(act.sum())])

        # Marginal occupancy features.  Include both density and non-empty flags
        # so the linear model can distinguish compact vs spread supports.
        row_counts = mask.sum(axis=1)
        col_counts = mask.sum(axis=0)
        feats.extend((row_counts / mass).astype(float).tolist())
        feats.extend((col_counts / mass).astype(float).tolist())
        feats.extend((row_counts > 0).astype(float).tolist())
        feats.extend((col_counts > 0).astype(float).tolist())

        # Raw pixels are a cheap capacity boost for the tiny 7x7 domain.  Ridge
        # regularization plus feature standardization keeps this from becoming
        # an unregularized memorizer when datasets are small.
        feats.extend(mask.reshape(-1).astype(float).tolist())

        pts = np.argwhere(mask > 0.5)
        pair_count = len(pts) * (len(pts) - 1) // 2
        hist = np.zeros((4, 4), dtype=np.float64)
        manhattan: list[float] = []
        chebyshev: list[float] = []
        euclidean: list[float] = []
        same_row = same_col = same_diag = adjacent = close = 0
        for i in range(len(pts)):
            r1, c1 = (int(pts[i, 0]), int(pts[i, 1]))
            for j in range(i + 1, len(pts)):
                r2, c2 = (int(pts[j, 0]), int(pts[j, 1]))
                dr_raw = abs(r1 - r2)
                dc_raw = abs(c1 - c2)
                dr = min(dr_raw, GRID - dr_raw)
                dc = min(dc_raw, GRID - dc_raw)
                hist[dr, dc] += 1.0
                man = float(dr + dc)
                cheb = float(max(dr, dc))
                euc = float(math.sqrt(dr * dr + dc * dc))
                manhattan.append(man)
                chebyshev.append(cheb)
                euclidean.append(euc)
                same_row += int(dr == 0)
                same_col += int(dc == 0)
                same_diag += int(dr == dc)
                adjacent += int(man == 1.0)
                close += int(man <= 2.0)
        denom = float(max(pair_count, 1))
        feats.extend((hist.reshape(-1) / denom).tolist())
        for values in (manhattan, chebyshev, euclidean):
            if values:
                arr = np.asarray(values, dtype=np.float64)
                feats.extend([float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())])
            else:
                feats.extend([0.0, 0.0, 0.0, 0.0])
        feats.extend([same_row / denom, same_col / denom, same_diag / denom, adjacent / denom, close / denom])

        rows.append(feats)
    return np.asarray(rows, dtype=np.float64)


@dataclass
class RidgeCnnEstimator:
    weights: np.ndarray
    target_mean: float
    target_std: float
    ridge: float
    x_mean: np.ndarray | None = None
    x_std: np.ndarray | None = None

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        if self.x_mean is None or self.x_std is None:
            return X
        if len(self.x_mean) != X.shape[1] or len(self.x_std) != X.shape[1]:
            return X
        return (X - self.x_mean) / self.x_std

    def predict_masks(self, masks: np.ndarray) -> np.ndarray:
        X = cnn_features(masks)
        if X.shape[1] != self.weights.shape[0]:
            # Backward compatibility for models saved by the first prototype.
            X = _basic_cnn_features(masks)
        X = self._transform_features(X)
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError(f"model expects {self.weights.shape[0]} features, got {X.shape[1]}")
        y_scaled = X @ self.weights
        return y_scaled * self.target_std + self.target_mean

    def predict_indices(self, supports: Sequence[Sequence[int]]) -> np.ndarray:
        masks = np.stack([mask_from_indices(s) for s in supports])
        return self.predict_masks(masks)

    def save(self, path: Path) -> None:
        np.savez(
            path,
            weights=self.weights,
            target_mean=self.target_mean,
            target_std=self.target_std,
            ridge=self.ridge,
            x_mean=self.x_mean if self.x_mean is not None else np.array([], dtype=np.float64),
            x_std=self.x_std if self.x_std is not None else np.array([], dtype=np.float64),
        )

    @classmethod
    def load(cls, path: Path) -> "RidgeCnnEstimator":
        data = np.load(path)
        x_mean = data["x_mean"] if "x_mean" in data.files and data["x_mean"].size else None
        x_std = data["x_std"] if "x_std" in data.files and data["x_std"].size else None
        return cls(
            weights=data["weights"],
            target_mean=float(data["target_mean"]),
            target_std=float(data["target_std"]),
            ridge=float(data["ridge"]),
            x_mean=x_mean,
            x_std=x_std,
        )


def _example_balance_weights(examples: Sequence[Example], balance_by: str) -> np.ndarray | None:
    if balance_by == "none":
        return None
    if balance_by == "k":
        keys = [len(ex.indices) for ex in examples]
    elif balance_by == "target":
        keys = [round(float(ex.target), 8) for ex in examples]
    elif balance_by == "k-target":
        keys = [(len(ex.indices), round(float(ex.target), 8)) for ex in examples]
    else:
        raise ValueError(f"unknown balance mode {balance_by!r}")
    counts = Counter(keys)
    weights = np.asarray([1.0 / counts[key] for key in keys], dtype=np.float64)
    return weights / float(weights.mean())


def train_estimator(examples: Sequence[Example], ridge: float = 1e-2, balance_by: str = "none") -> RidgeCnnEstimator:
    if not examples:
        raise ValueError("no training examples found")
    masks = np.stack([mask_from_indices(ex.indices) for ex in examples])
    y = np.asarray([ex.target for ex in examples], dtype=np.float64)
    X_raw = cnn_features(masks)

    # Standardize all non-bias columns before ridge regression.  Without this,
    # large-scale count/sum features dominate the penalty and the model often
    # behaves like a low-capacity predictor of the global/per-k mean.
    x_mean = X_raw.mean(axis=0)
    x_std = X_raw.std(axis=0)
    x_mean[0] = 0.0
    x_std[0] = 1.0
    x_std[x_std == 0.0] = 1.0
    X = (X_raw - x_mean) / x_std

    sample_weight = _example_balance_weights(examples, balance_by)
    if sample_weight is None:
        target_mean = float(y.mean())
        target_std = float(y.std() or 1.0)
        y_scaled = (y - target_mean) / target_std
        X_fit = X
        y_fit = y_scaled
    else:
        wsum = float(sample_weight.sum())
        target_mean = float(np.sum(sample_weight * y) / wsum)
        target_std = float(math.sqrt(np.sum(sample_weight * (y - target_mean) ** 2) / wsum) or 1.0)
        y_scaled = (y - target_mean) / target_std
        scale = np.sqrt(sample_weight)
        X_fit = X * scale[:, None]
        y_fit = y_scaled * scale

    reg = ridge * np.eye(X.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0  # do not penalize bias
    weights = np.linalg.solve(X_fit.T @ X_fit + reg, X_fit.T @ y_fit)
    return RidgeCnnEstimator(
        weights=weights,
        target_mean=target_mean,
        target_std=target_std,
        ridge=ridge,
        x_mean=x_mean,
        x_std=x_std,
    )


def train_validation_split(
    examples: Sequence[Example],
    val_fraction: float,
    seed: int,
    stratify: bool = True,
) -> tuple[list[Example], list[Example]]:
    items = list(examples)
    rng = random.Random(seed)
    if not stratify:
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_fraction))) if len(items) > 1 else 0
        return items[n_val:], items[:n_val]

    grouped: dict[tuple[int, float], list[Example]] = defaultdict(list)
    for ex in items:
        grouped[(len(ex.indices), round(float(ex.target), 8))].append(ex)

    train: list[Example] = []
    val: list[Example] = []
    for group in grouped.values():
        rng.shuffle(group)
        if len(group) <= 1:
            train.extend(group)
            continue
        n_val = max(1, int(round(len(group) * val_fraction)))
        n_val = min(n_val, len(group) - 1)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    if not val and len(items) > 1:
        return train_validation_split(items, val_fraction, seed, stratify=False)
    return train, val


def evaluate(estimator: RidgeCnnEstimator, examples: Sequence[Example]) -> dict[str, float]:
    if not examples:
        return {"count": 0.0, "mae": float("nan"), "rmse": float("nan")}
    preds = estimator.predict_indices([ex.indices for ex in examples])
    y = np.asarray([ex.target for ex in examples], dtype=np.float64)
    err = preds - y
    return {"count": float(len(examples)), "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err * err)))}


def one_point_extensions(base: Sequence[int]) -> list[tuple[int, ...]]:
    base_set = set(int(i) for i in base)
    if any(i < 0 or i >= N_POINTS for i in base_set):
        raise ValueError("support indices must be in 0..48")
    return [tuple(sorted(base_set | {i})) for i in range(N_POINTS) if i not in base_set]


def synthetic_examples(n: int, seed: int) -> list[Example]:
    """Create deterministic toy data for CI/smoke validation."""
    rng = random.Random(seed)
    examples: list[Example] = []
    center = np.array([3.0, 3.0])
    for _ in range(n):
        k = rng.randint(2, 10)
        support = tuple(sorted(rng.sample(range(N_POINTS), k)))
        pts = np.array([[i // GRID, i % GRID] for i in support], dtype=np.float64)
        spread = float(np.mean(np.linalg.norm(pts - center, axis=1)))
        rows = len({int(p[0]) for p in pts})
        cols = len({int(p[1]) for p in pts})
        target = 49.0 - 1.6 * k + 0.8 * spread + 0.25 * (rows + cols)
        examples.append(Example(support, target))
    return examples


def _parse_support(text: str) -> tuple[int, ...]:
    if not text.strip():
        return ()
    return tuple(sorted({int(part) for part in text.split(",") if part.strip()}))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="*", type=Path, help="JSONL/JSON result files with indices and min_distance/max_zeros")
    parser.add_argument("--target-field", default="min_distance", help="record field to learn (default: min_distance)")
    parser.add_argument("--model-out", type=Path, default=Path("support_point_cnn_model.npz"))
    parser.add_argument("--model-in", type=Path, help="load an existing .npz model instead of training")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument(
        "--balance-by",
        choices=["none", "k", "target", "k-target"],
        default="none",
        help="optional inverse-frequency weighting for ridge training",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-stratify-split", action="store_true", help="use a plain random validation split")
    parser.add_argument("--smoke-test", action="store_true", help="use synthetic data so the workflow can be validated without search output")
    parser.add_argument("--smoke-count", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=1, help="accepted for compatibility; ridge fit is closed-form")
    parser.add_argument("--score-base", help="comma-separated support indices; ranks all one-point extensions")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)

    estimator: RidgeCnnEstimator
    if args.model_in:
        estimator = RidgeCnnEstimator.load(args.model_in)
        print(f"Loaded model: {args.model_in}")
    else:
        if args.smoke_test:
            examples = synthetic_examples(args.smoke_count, args.seed)
        else:
            if not args.data:
                parser.error("provide --data files, --model-in, or --smoke-test")
            examples = load_examples(args.data, target_field=args.target_field)
        if not examples:
            raise SystemExit("No usable examples found (need indices plus min_distance/max_zeros).")
        train, val = train_validation_split(examples, args.val_fraction, args.seed, stratify=not args.no_stratify_split)
        estimator = train_estimator(train, ridge=args.ridge, balance_by=args.balance_by)
        estimator.save(args.model_out)
        train_metrics = evaluate(estimator, train)
        val_metrics = evaluate(estimator, val)
        print(f"Examples: train={len(train)} validation={len(val)}")
        print(f"Train MAE={train_metrics['mae']:.3f} RMSE={train_metrics['rmse']:.3f}")
        if val:
            print(f"Validation MAE={val_metrics['mae']:.3f} RMSE={val_metrics['rmse']:.3f}")
        print(f"Saved model: {args.model_out}")

    if args.score_base is not None:
        base = _parse_support(args.score_base)
        candidates = one_point_extensions(base)
        scores = estimator.predict_indices(candidates)
        order = np.argsort(scores)[::-1][: args.top]
        print("Top one-point extensions (predicted target):")
        for rank, pos in enumerate(order, start=1):
            added = sorted(set(candidates[int(pos)]) - set(base))
            print(f"{rank:2d}. add={added[0]:2d} support={list(candidates[int(pos)])} pred={scores[int(pos)]:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
