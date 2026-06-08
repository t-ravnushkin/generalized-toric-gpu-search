# Support-point CNN estimator prototype

`support_point_cnn_estimator.py` is a lightweight pre-screening workflow for the
GF(8) toric-code search outputs.  It learns from JSON/JSONL records containing
`indices` (or `lattice_points`) plus `min_distance`/`max_zeros`, then ranks new
support sets or one-point extensions by predicted distance.

The estimator is **not a certifier**.  Use it only to prioritize candidates;
keep validating selected supports with the exact OpenCL/CUDA distance oracle.

## Dependencies

No new required dependency is needed beyond the existing project `requirements.txt`.
The prototype uses NumPy only.  It implements fixed 3x3 convolutional summaries,
row/column occupancy, raw 7x7 mask pixels, pairwise support-geometry features,
and a standardized closed-form ridge-regression head, so it runs on CPU without
PyTorch/TensorFlow.

Generated `.npz` model files are local artifacts and can be safely deleted.

## Quick validation

```bash
python support_point_cnn_estimator.py --smoke-test --epochs 1 \
  --model-out /tmp/support_point_cnn_model.npz \
  --score-base 0,1,8,9 --top 5
```

## Train from search output

```bash
python support_point_cnn_estimator.py \
  --data 'canon_20260608_125209(1).json' canon_local_20260608_171130.json \
  --model-out support_point_cnn_model.npz

# Optional for imbalanced exact-labeled datasets:
python support_point_cnn_estimator.py \
  --data cnnruns/support_point_cuda_bp_labeled.jsonl \
  --balance-by k-target \
  --model-out support_point_cnn_model.npz
```

## Avoiding constant/per-k-mean predictors

Small exact CUDA datasets can have very discrete labels: in the copied 256-row
Kaggle run, support size explained much of the target and many `(k, distance)`
buckets were rare.  The current workflow therefore uses a stratified validation
split by `(k, min_distance)` where possible, standardized features, and optional
inverse-frequency weighting (`--balance-by k`, `target`, or `k-target`).  Always
inspect label counts by `k` and prediction standard deviation before trusting a
ranking run; CNN predictions are heuristic triage scores only.

## Rank extensions with a saved model

```bash
python support_point_cnn_estimator.py \
  --model-in support_point_cnn_model.npz \
  --score-base 0,1,8,9 --top 10
```

## Compare predictions with exact synthetic supports

`compare_synthetic_supports.py` samples random support configurations, computes
exact minimum distances with the existing OpenCL oracle when available, scores
the same supports with the estimator, and reports MAE/RMSE/correlation.  Exact
evaluation is exponential, so keep `--k-max`/`--max-exact-k` modest for local
runs.

```bash
python compare_synthetic_supports.py \
  --data champions_20260607_213729.json canon_local_20260608_171130.json \
  --n-samples 24 --k-min 2 --k-max 6 --max-exact-k 6
```

In `--backend auto` mode the script falls back to a small exact CPU evaluator if
PyOpenCL or an OpenCL device is unavailable; use `--backend opencl` to require
the repository OpenCL distance oracle.
