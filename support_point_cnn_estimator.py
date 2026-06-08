r"""Prototype support-point CNN estimator for GF(8) toric-code searches.

This module is intentionally lightweight by default: the ridge backend uses
NumPy only and implements a small CNN-style feature extractor (fixed 3x3
convolution filters, support-mask, row/column, and pairwise-geometry summaries)
followed by standardized ridge regression.  For Kaggle GPU notebooks there is
also an optional PyTorch backend that trains a small learnable 7x7 CNN on CUDA
when torch + a CUDA device are available.  Both paths are meant to rank
candidate support sets/extensions before expensive exact distance evaluation,
not to certify code distances.

Examples
--------
Smoke-test the whole workflow with synthetic data::

    python support_point_cnn_estimator.py --smoke-test --epochs 1

Train/validate from existing canonical search JSON/JSONL/CSV files::

    python support_point_cnn_estimator.py --data canon_20260608_125209\(1\).json \
        --model-out support_point_cnn_model.npz

Score all one-point extensions of a base support::

    python support_point_cnn_estimator.py --model-in support_point_cnn_model.npz \
        --score-base 0,1,8,9
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

GRID = 7
N_POINTS = GRID * GRID


@dataclass(frozen=True)
class Example:
    indices: tuple[int, ...]
    target: float


def _iter_json_records(path: Path) -> Iterable[dict]:
    """Yield records from CSV, JSONL, a JSON object, or a JSON array.

    Some long-running search outputs in this repo are interrupted JSON arrays:
    they contain many complete objects followed by a truncated final object.
    For those, salvage the complete prefix instead of failing the whole run.
    """
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            yield from csv.DictReader(f)
        return

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


def _parse_sequence(value: object) -> object:
    """Parse JSON/CSV string encodings of supports or lattice-point lists."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (json.JSONDecodeError, SyntaxError, ValueError, TypeError):
            pass
    return [part.strip() for part in text.split(",") if part.strip()]


def _indices_from_record(record: dict) -> tuple[int, ...] | None:
    if "indices" in record:
        indices = _parse_sequence(record["indices"])
    elif "support" in record:
        indices = _parse_sequence(record["support"])
    elif "lattice_points" in record:
        points = _parse_sequence(record["lattice_points"])
        try:
            indices = [int(a) * GRID + int(b) for a, b in points]
        except (TypeError, ValueError):
            return None
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
    if target_field in record and record[target_field] not in (None, ""):
        return float(record[target_field])
    if target_field == "min_distance":
        for key in ("actual_min_distance_cuda_bp", "exact_min_distance", "true_distance", "sampled_min_distance"):
            if key in record and record[key] not in (None, ""):
                return float(record[key])
        if "max_zeros" in record and record["max_zeros"] not in (None, ""):
            return float(N_POINTS - int(record["max_zeros"]))
        if "sampled_max_zeros" in record and record["sampled_max_zeros"] not in (None, ""):
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


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional local/Kaggle environment
        raise RuntimeError(
            "PyTorch is required for the torch/CUDA training backend. "
            "Kaggle GPU notebooks usually include torch; locally install a CUDA-enabled "
            "PyTorch build or use --backend ridge."
        ) from exc
    return torch


def torch_cuda_available() -> bool:
    """Return True when the optional PyTorch backend can see a CUDA GPU."""
    try:
        torch = _import_torch()
    except RuntimeError:
        return False
    return bool(torch.cuda.is_available())


def _make_torch_support_point_cnn(torch: Any, channels: int = 48, hidden: int = 128, dropout: float = 0.05) -> Any:
    """Construct a tiny learnable CNN without importing torch at module import time."""

    class SupportPointTorchCNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, channels, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                torch.nn.ReLU(),
            )
            self.head = torch.nn.Sequential(
                torch.nn.Linear(channels * GRID * GRID + 1, hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, max(16, hidden // 2)),
                torch.nn.ReLU(),
                torch.nn.Linear(max(16, hidden // 2), 1),
            )

        def forward(self, x: Any) -> Any:
            z = self.conv(x).flatten(1)
            k_norm = x.sum(dim=(1, 2, 3), keepdim=False).unsqueeze(1) / float(N_POINTS)
            return self.head(torch.cat([z, k_norm], dim=1)).squeeze(1)

    return SupportPointTorchCNN()


@dataclass
class TorchCnnEstimator:
    """Optional PyTorch/CUDA support-mask CNN regressor.

    The class deliberately mirrors ``RidgeCnnEstimator``'s prediction API so the
    notebook can switch between CPU ridge and GPU torch training without
    changing downstream ranking/evaluation cells.
    """

    model: Any
    target_mean: float
    target_std: float
    device: str
    model_kwargs: dict[str, Any]
    epochs: int = 0

    def predict_masks(self, masks: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        torch = _import_torch()
        device = torch.device(self.device)
        self.model.to(device)
        self.model.eval()
        outputs: list[np.ndarray] = []
        arr = masks.astype(np.float32, copy=False)[:, None, :, :]
        with torch.no_grad():
            for start in range(0, len(arr), batch_size):
                xb = torch.from_numpy(arr[start : start + batch_size]).to(device)
                pred_scaled = self.model(xb).detach().cpu().numpy().astype(np.float64)
                outputs.append(pred_scaled * self.target_std + self.target_mean)
        return np.concatenate(outputs) if outputs else np.asarray([], dtype=np.float64)

    def predict_indices(self, supports: Sequence[Sequence[int]]) -> np.ndarray:
        masks = np.stack([mask_from_indices(s) for s in supports])
        return self.predict_masks(masks)

    def save(self, path: Path) -> None:
        torch = _import_torch()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": "TorchSupportPointCNN",
                "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                "target_mean": self.target_mean,
                "target_std": self.target_std,
                "device": self.device,
                "model_kwargs": self.model_kwargs,
                "epochs": self.epochs,
                "grid": GRID,
                "n_points": N_POINTS,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "auto") -> "TorchCnnEstimator":
        torch = _import_torch()
        chosen = _select_torch_device(torch, device)
        payload = torch.load(path, map_location=chosen)
        model_kwargs = dict(payload.get("model_kwargs", {}))
        model = _make_torch_support_point_cnn(torch, **model_kwargs)
        model.load_state_dict(payload["state_dict"])
        model.to(chosen)
        model.eval()
        return cls(
            model=model,
            target_mean=float(payload["target_mean"]),
            target_std=float(payload["target_std"]),
            device=str(chosen),
            model_kwargs=model_kwargs,
            epochs=int(payload.get("epochs", 0)),
        )


def _select_torch_device(torch: Any, device: str) -> Any:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chosen = torch.device(device)
    if chosen.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA torch backend, but torch.cuda.is_available() is false")
    return chosen


def train_torch_cnn_estimator(
    examples: Sequence[Example],
    *,
    val_examples: Sequence[Example] | None = None,
    balance_by: str = "none",
    epochs: int = 300,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 60,
    seed: int = 1,
    device: str = "auto",
    channels: int = 48,
    hidden: int = 128,
    dropout: float = 0.05,
    verbose: bool = False,
) -> tuple[TorchCnnEstimator, list[dict[str, float]]]:
    """Train a small learnable support-mask CNN with PyTorch.

    On Kaggle GPU runtimes ``device='auto'`` selects CUDA; local CPU-only
    environments remain usable for smoke tests by selecting CPU.  The returned
    estimator exposes the same ``predict_indices`` method as the ridge backend.
    """
    if not examples:
        raise ValueError("no training examples found")
    torch = _import_torch()
    chosen = _select_torch_device(torch, device)
    torch.manual_seed(seed)
    if chosen.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    masks_np = np.stack([mask_from_indices(ex.indices) for ex in examples]).astype(np.float32)[:, None, :, :]
    y_np = np.asarray([ex.target for ex in examples], dtype=np.float32)
    sample_weight = _example_balance_weights(examples, balance_by)
    if sample_weight is None:
        target_mean = float(y_np.mean())
        target_std = float(y_np.std() or 1.0)
        weight_np = np.ones_like(y_np, dtype=np.float32)
    else:
        weight_np = sample_weight.astype(np.float32)
        wsum = float(weight_np.sum())
        target_mean = float(np.sum(weight_np * y_np) / wsum)
        target_std = float(math.sqrt(np.sum(weight_np * (y_np - target_mean) ** 2) / wsum) or 1.0)
    y_scaled_np = ((y_np - target_mean) / target_std).astype(np.float32)

    X = torch.from_numpy(masks_np).to(chosen)
    y = torch.from_numpy(y_scaled_np).to(chosen)
    weights = torch.from_numpy(weight_np).to(chosen)
    model_kwargs = {"channels": int(channels), "hidden": int(hidden), "dropout": float(dropout)}
    model = _make_torch_support_point_cnn(torch, **model_kwargs).to(chosen)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if val_examples:
        val_masks_np = np.stack([mask_from_indices(ex.indices) for ex in val_examples]).astype(np.float32)[:, None, :, :]
        val_y_np = np.asarray([ex.target for ex in val_examples], dtype=np.float32)
        X_val = torch.from_numpy(val_masks_np).to(chosen)
        y_val = torch.from_numpy(((val_y_np - target_mean) / target_std).astype(np.float32)).to(chosen)
    else:
        X_val = y_val = None

    n = len(examples)
    batch_size = max(1, min(int(batch_size), n))
    patience = max(1, int(patience))
    history: list[dict[str, float]] = []
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None

    if verbose:
        dev_name = torch.cuda.get_device_name(chosen) if chosen.type == "cuda" else "CPU"
        print(f"Torch CNN training device: {chosen} ({dev_name}); train={n}; val={len(val_examples or [])}")

    for epoch in range(1, int(epochs) + 1):
        model.train()
        perm = torch.randperm(n, device=chosen)
        epoch_loss = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            pred = model(X[idx])
            loss = ((pred - y[idx]) ** 2 * weights[idx]).mean()
            loss.backward()
            optimizer.step()
            bs = int(idx.numel())
            epoch_loss += float(loss.detach().cpu()) * bs
            seen += bs
        train_mse_scaled = epoch_loss / max(seen, 1)

        model.eval()
        with torch.no_grad():
            train_pred = model(X)
            train_mae = float((torch.abs(train_pred - y) * target_std).mean().detach().cpu())
            if X_val is not None and y_val is not None:
                val_pred = model(X_val)
                val_mae = float((torch.abs(val_pred - y_val) * target_std).mean().detach().cpu())
            else:
                val_mae = train_mae
        row = {"epoch": float(epoch), "train_mse_scaled": train_mse_scaled, "train_mae": train_mae, "validation_mae": val_mae}
        history.append(row)

        score = val_mae
        if score + 1e-9 < best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}; best_validation_mae={best_score:.3f}")
            break

        if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == int(epochs)):
            print(f"epoch={epoch:4d} train_MAE={train_mae:.3f} validation_MAE={val_mae:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    estimator = TorchCnnEstimator(
        model=model,
        target_mean=target_mean,
        target_std=target_std,
        device=str(chosen),
        model_kwargs=model_kwargs,
        epochs=best_epoch,
    )
    return estimator, history


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
        return {
            "count": 0.0,
            "mse": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "acc_pm3": float("nan"),
            "actual_std": float("nan"),
            "pred_std": float("nan"),
        }
    preds = estimator.predict_indices([ex.indices for ex in examples])
    y = np.asarray([ex.target for ex in examples], dtype=np.float64)
    err = preds - y
    mse = float(np.mean(err * err))
    return {
        "count": float(len(examples)),
        "mse": mse,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(mse)),
        "bias": float(np.mean(err)),
        "acc_pm3": float(np.mean(np.abs(err) <= 3.0)),
        "actual_std": float(np.std(y)),
        "pred_std": float(np.std(preds)),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"MSE={metrics['mse']:.3f} MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f} "
        f"bias={metrics['bias']:.3f} acc_pm3={metrics['acc_pm3'] * 100.0:.2f}% "
        f"actual_std={metrics['actual_std']:.3f} pred_std={metrics['pred_std']:.3f}"
    )


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
    parser.add_argument("--data", nargs="*", type=Path, help="JSONL/JSON/CSV result files with indices and min_distance/max_zeros")
    parser.add_argument("--target-field", default="min_distance", help="record field to learn (default: min_distance)")
    parser.add_argument("--model-out", type=Path, default=Path("support_point_cnn_model.npz"))
    parser.add_argument("--model-in", type=Path, help="load an existing .npz ridge model or .pt torch model instead of training")
    parser.add_argument("--backend", choices=["ridge", "torch"], default="ridge", help="training backend: CPU ridge or optional PyTorch CNN")
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
    parser.add_argument("--epochs", type=int, default=300, help="torch epochs; ignored by the closed-form ridge backend")
    parser.add_argument("--batch-size", type=int, default=256, help="torch mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="torch AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="torch AdamW weight decay")
    parser.add_argument("--patience", type=int, default=60, help="torch early-stopping patience in epochs")
    parser.add_argument("--device", default="auto", help="torch device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--score-base", help="comma-separated support indices; ranks all one-point extensions")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)

    estimator: Any
    if args.model_in:
        if args.model_in.suffix.lower() in {".pt", ".pth"}:
            try:
                estimator = TorchCnnEstimator.load(args.model_in, device=args.device)
            except RuntimeError as exc:
                raise SystemExit(str(exc)) from exc
        else:
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
        if args.backend == "torch":
            if args.model_out == Path("support_point_cnn_model.npz"):
                args.model_out = Path("support_point_torch_cnn_model.pt")
            try:
                estimator, history = train_torch_cnn_estimator(
                    train,
                    val_examples=val,
                    balance_by=args.balance_by,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    patience=args.patience,
                    seed=args.seed,
                    device=args.device,
                    verbose=True,
                )
            except RuntimeError as exc:
                raise SystemExit(str(exc)) from exc
            if history:
                best = min(history, key=lambda row: row["validation_mae"])
                print(f"Best torch epoch={int(best['epoch'])} validation_MAE={best['validation_mae']:.3f}")
        else:
            estimator = train_estimator(train, ridge=args.ridge, balance_by=args.balance_by)
        estimator.save(args.model_out)
        train_metrics = evaluate(estimator, train)
        val_metrics = evaluate(estimator, val)
        print(f"Examples: train={len(train)} validation={len(val)}")
        print(f"Train {format_metrics(train_metrics)}")
        if val:
            print(f"Validation {format_metrics(val_metrics)}")
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
