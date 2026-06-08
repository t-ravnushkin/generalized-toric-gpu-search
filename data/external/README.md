# External support-point datasets

Place downloaded or shared support-point datasets in this folder. Its contents are ignored by Git so large or third-party data is not accidentally committed.

Supported training inputs for `support_point_cnn_estimator.py`:

- JSONL: one object per line.
- JSON array/object files.
- CSV files with a header row.

Each record must contain a support and an exact/target distance:

- support: `indices`, `support`, or `lattice_points`
  - `indices`/`support` are 0..48 lattice indices.
  - `lattice_points` are `(row, col)` pairs on the 7×7 exponent grid.
- target: `min_distance` by default. For Kaggle prediction/evaluation CSVs, `actual_min_distance_cuda_bp` is also accepted as a `min_distance` fallback.

Example:

```bash
python support_point_cnn_estimator.py \
  --data data/external/support_point_cuda_bp_labeled.jsonl \
  --balance-by k-target \
  --model-out /tmp/support_point_cnn_model.npz
```

Keep derived models, downloaded archives, and generated prediction CSV/JSONL files in this ignored folder, `/tmp`, or `cnnruns/` unless a small fixture is intentionally added for tests.
