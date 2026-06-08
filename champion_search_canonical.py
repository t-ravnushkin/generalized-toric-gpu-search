"""
Canonical-form champion search for GF(8) toric codes.

Exploits AGL₂(F₇) symmetry (order 98,784) to reduce the search space.
Unlike champion_search.py, this is COMPLETE: it explores every equivalence
class of k-subsets under the affine symmetry group, so it cannot miss
champions due to good codes lacking good sub-codes.

Why AGL₂(F₇)?
  The code alphabet is GF(8) but exponents live in Z₇² = F₇² because
  GF(8)* ≅ Z₇ (cyclic of order 7).  For A ∈ GL₂(F₇) and δ ∈ F₇²,
  C(A·S + δ) differs from C(S) only by a column permutation of the
  evaluation matrix, so min-distance is preserved.  The full symmetry
  group is AGL₂(F₇) = F₇² ⋊ GL₂(F₇), order 49 × 2016 = 98,784.

Algorithm (no threshold, complete):
  canonical_level[1] = {canonical({(0,0)})}   # 1 form
  for k = 2..max_k:
      candidates = {canonical(S ∪ {p})
                    for S in canonical_level[k-1]
                    for p ∈ F₇² \\ S}  \\ already_seen
      evaluate all candidates on GPU → max_zeros per set
      record any S with min_distance ≥ codetable_bound[k]
      canonical_level[k] = candidates        # keep ALL (completeness)
                         or prune by margin  # faster, less complete

Canonical form of S:
  min over A ∈ GL₂(F₇) of sort(A·S − min(A·S)  mod 7)
  (translation normalises minimum point to (0,0))
  Packed as uint64: sum(idx[i] * 64^i) with idx sorted, idx < 64.

Number of canonical forms vs raw subsets:
  k=7: ~630 vs 62M    k=9: ~25K vs 2.5B    k=10: ~100K vs 10B
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# AGL₂(F₇) — precompute GL₂(F₇) as permutations of 49 lattice points
# ---------------------------------------------------------------------------

def precompute_gl2_perms() -> np.ndarray:
    """
    All 2016 elements of GL₂(F₇) as permutations of the 49 lattice points
    indexed 0..48  (point (a,b) ↦ index a*7+b).
    Returns int32 array of shape (2016, 49).
    """
    pts = np.array([(i // 7, i % 7) for i in range(49)], dtype=np.int32)  # (49,2)
    perms = []
    for a, b, c, d in product(range(7), repeat=4):
        if (a * d - b * c) % 7 != 0:
            A = np.array([[a, b], [c, d]], dtype=np.int32)
            t = (pts @ A.T) % 7                         # (49,2)
            perms.append(t[:, 0] * 7 + t[:, 1])        # (49,)
    return np.array(perms, dtype=np.int32)              # (2016,49)


_GL2_PERMS: np.ndarray | None = None


def _gl2() -> np.ndarray:
    global _GL2_PERMS
    if _GL2_PERMS is None:
        t0 = time.perf_counter()
        _GL2_PERMS = precompute_gl2_perms()
        print(f"[canonical] GL₂(F₇): {len(_GL2_PERMS)} matrices "
              f"({(time.perf_counter() - t0) * 1e3:.0f} ms)")
    return _GL2_PERMS


# ---------------------------------------------------------------------------
# Canonical form — vectorised batch version
# ---------------------------------------------------------------------------

def canonical_forms_batch(
    sets_arr: np.ndarray,   # (n, k)  int32, each row sorted
    gl2_perms: np.ndarray,  # (2016, 49)  int32
) -> np.ndarray:
    """
    Vectorised canonical form for n k-sets simultaneously.

    Each canonical form is packed as a uint64:
        packed = sum(sorted_idx[i] * 64**i,  i=0..k-1)
    where sorted_idx is the sorted index tuple after applying the
    best GL₂ image + translation-to-origin normalisation.

    Unpack with unpack_canonical(packed, k).
    Returns uint64 array of shape (n,).
    """
    n, k = sets_arr.shape
    shifts = np.arange(k, dtype=np.uint64) * 6     # 6 bits per index (max 48 < 64)
    best = np.full(n, np.iinfo(np.uint64).max, dtype=np.uint64)

    for perm in gl2_perms:
        transformed = perm[sets_arr]                         # (n, k)

        # Must try ALL k translation anchors (putting each point at origin),
        # not just the lex-min point.  A non-minimum anchor can yield a
        # lex-smaller sorted tuple when other points wrap around mod 7.
        ta = (transformed // 7).astype(np.int32)             # (n, k)
        tb = (transformed %  7).astype(np.int32)

        for t in range(k):
            anc_a = ta[:, t]                                 # (n,)
            anc_b = tb[:, t]
            sa = (ta - anc_a[:, None]) % 7                  # (n, k)
            sb = (tb - anc_b[:, None]) % 7
            shifted = (sa * 7 + sb).astype(np.uint64)
            shifted = np.sort(shifted, axis=1)
            packed = (shifted << shifts[None, :]).sum(axis=1)
            np.minimum(best, packed, out=best)

    return best


def unpack_canonical(packed: int, k: int) -> tuple[int, ...]:
    """Inverse of canonical_forms_batch packing."""
    return tuple(int((packed >> (6 * i)) & 63) for i in range(k))


# ---------------------------------------------------------------------------
# File I/O helpers (same format as champion_search.py)
# ---------------------------------------------------------------------------

def append_result(record: dict, path: Path) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_latest_results() -> Path | None:
    files = sorted(Path(".").glob("canon_*.json"))
    return files[-1] if files else None


def load_canonical_state(path: Path) -> tuple[int, set[int]]:
    """
    Re-read a partial run and return (last_completed_k, packed_survivors).
    Falls back to (1, {0}) when the file is empty or missing.
    """
    survivors_by_k: dict[int, set[int]] = {}
    completed_ks: set[int] = set()

    for rec in load_results(path):
        if rec.get("type") == "level_complete":
            k = rec["k"]
            completed_ks.add(k)
        elif rec.get("type") == "survivor":
            k = rec["k"]
            survivors_by_k.setdefault(k, set()).add(rec["packed"])

    if not completed_ks:
        return 1, {0}

    last_k = max(completed_ks)
    return last_k, survivors_by_k.get(last_k, set())


# ---------------------------------------------------------------------------
# Multi-GPU dispatch
# ---------------------------------------------------------------------------

def _dispatch(oracles, batch: list[list[int]], td: int) -> list[int]:
    if not batch:
        return []
    n = len(oracles)
    if n == 1:
        return oracles[0].max_zeros_batch(batch, td)
    q, r = divmod(len(batch), n)
    sizes = [q + (1 if i < r else 0) for i in range(n)]
    chunks, start = [], 0
    for s in sizes:
        chunks.append(batch[start: start + s])
        start += s
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(o.max_zeros_batch, c, td)
                   for o, c in zip(oracles, chunks) if c]
        results = []
        for f in futures:
            results.extend(f.result())
    return results


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

_CANON_CHUNK = 200_000   # sets per numpy canonicalization chunk


def canonical_champion_search(
    oracles,
    lattice: list[tuple[int, int]],
    results_file: Path,
    targets: dict[int, int],      # {k: best_known_d} from codetables
    max_k: int = 12,
    batch_size: int = 50_000,
    prune_margin: int | None = None,
    resume: bool = True,
    k_canonical_max: int = 9,
):
    """
    Hybrid champion search: canonical forms for k ≤ k_canonical_max,
    raw sorted-tuple BFS for k > k_canonical_max.

    For small k the AGL₂(F₇) orbit reduction (×98,784) makes canonicalization
    worthwhile.  Above k_canonical_max the GL₂ loop over 2016 matrices becomes
    the CPU bottleneck while the GPU sits idle; switching to raw-tuple
    deduplication (O(1) numpy) keeps the GPU fully utilised at the cost of
    evaluating AGL-equivalent sets multiple times.

    k_canonical_max:
      k ≤ this — full GL₂ canonical forms (complete, CPU-heavy at large k).
      k > this — sorted-tuple deduplication only (GPU-bound, slight redundancy).
      Default 9 matches the inflection point where canonical-form count jumps
      from ~25K (k=9) to ~100K (k=10), making GL₂ cost dominant.

    prune_margin:
      None  — keep ALL surviving k-forms for the next level.
      int   — prune sets with min_distance < targets[k] - prune_margin.
    """
    gl2 = _gl2()
    n_gpus = len(oracles) if hasattr(oracles, "__len__") else 1
    if not isinstance(oracles, list):
        oracles = [oracles]

    print(f"\n=== Hybrid Champion Search  max_k={max_k}  gpus={n_gpus}  "
          f"k_canonical_max={k_canonical_max}  "
          f"prune={'none' if prune_margin is None else prune_margin} ===\n"
          f"    canonical (AGL₂, complete) for k≤{k_canonical_max}; "
          f"raw-BFS (GPU-bound) for k>{k_canonical_max}\n")

    # ── Resume support ────────────────────────────────────────────────────
    start_k, canonical_level = 1, {0}
    if resume and results_file.exists():
        start_k, canonical_level = load_canonical_state(results_file)
        if start_k > 1:
            print(f"  Resuming after k={start_k}  "
                  f"({len(canonical_level)} canonical survivors loaded)")

    for k in range(start_k + 1, max_k + 1):
        td = targets.get(k, 1)

        # ── Expand: generate all canonical k-forms ────────────────────────
        t0 = time.perf_counter()

        prev_list: list[tuple[int, ...]] = [
            unpack_canonical(p, k - 1) for p in canonical_level
        ]

        # Generate raw extensions (one extra point appended, then sorted)
        raw: list[tuple[int, ...]] = []
        for S in prev_list:
            s_set = set(S)
            for p in range(49):
                if p not in s_set:
                    raw.append(tuple(sorted(S + (p,))))

        # Deduplicate raw before the packing pass
        raw = list(dict.fromkeys(raw))   # preserves order, removes dupes

        # Pack in chunks.
        # k ≤ k_canonical_max: full GL₂ canonical form (complete, CPU-heavy).
        # k > k_canonical_max: direct sorted-tuple packing (O(1), GPU-bound).
        use_canonical = (k <= k_canonical_max)
        _shifts = np.arange(k, dtype=np.uint64) * 6   # shared by both paths

        seen_packed:    set[int]             = set()
        new_packed:     list[int]            = []
        new_indices:    list[tuple[int,...]] = []

        for ci in range(0, len(raw), _CANON_CHUNK):
            chunk_arr = np.array(raw[ci: ci + _CANON_CHUNK], dtype=np.int32)
            if use_canonical:
                packed = canonical_forms_batch(chunk_arr, gl2)
            else:
                packed = (chunk_arr.astype(np.uint64) << _shifts).sum(axis=1)
            for p_val in packed.tolist():
                if p_val not in seen_packed:
                    seen_packed.add(p_val)
                    new_packed.append(p_val)
                    new_indices.append(unpack_canonical(p_val, k))

        n_cands = len(new_indices)
        t_expand = time.perf_counter() - t0
        mode_label = "canonical forms" if use_canonical else "raw-BFS sets"
        print(f"  k={k}: {n_cands:,} {mode_label}  "
              f"target_d={td}  expand={t_expand:.1f}s")

        # ── Evaluate on GPU ───────────────────────────────────────────────
        t0 = time.perf_counter()
        all_mz: list[int] = []

        for bi in tqdm(range(0, n_cands, batch_size),
                       desc=f"k={k}", dynamic_ncols=True, leave=False):
            chunk = [list(new_indices[j])
                     for j in range(bi, min(bi + batch_size, n_cands))]
            all_mz.extend(_dispatch(oracles, chunk, td))

        t_eval = time.perf_counter() - t0

        # ── Record champions and survivors ────────────────────────────────
        next_level: set[int] = set()
        n_champ = 0

        for packed_j, S, mz in zip(new_packed, new_indices, all_mz):
            d = 49 - mz
            if d >= td:
                n_champ += 1
                append_result({
                    "name":         f"canon_k{k}_d{d}",
                    "indices":      list(S),
                    "lattice_points": [lattice[i] for i in S],
                    "k":            k,
                    "max_zeros":    mz,
                    "min_distance": d,
                    "best_known_d": td,
                    "new_record":   d > td,
                }, results_file)

            keep = (prune_margin is None) or (d >= td - prune_margin)
            if keep:
                next_level.add(packed_j)
                # persist so resume works
                append_result({"type": "survivor", "k": k,
                               "packed": packed_j}, results_file)

        append_result({"type": "level_complete", "k": k,
                       "n_canonical": n_cands, "n_champions": n_champ,
                       "n_survivors": len(next_level)}, results_file)

        print(f"  k={k}: {n_champ} champions (d≥{td}), "
              f"{len(next_level):,} survivors  eval={t_eval:.1f}s\n")

        if not next_level:
            print(f"  *** 0 survivors — BFS frontier exhausted at k={k} ***")
            if prune_margin is not None:
                print(f"      Try increasing prune_margin (currently {prune_margin}).")
            break

        canonical_level = next_level


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> Path:
    from precompute import build_eval_matrix, bounding_cube_points
    from codetables import bounds_for_n

    try:
        from kernel_cuda_bp import init_cuda_oracles_bp
        M = build_eval_matrix()
        oracles, sm_count = init_cuda_oracles_bp(M)
    except Exception:
        from kernel_cuda import init_cuda_oracles
        M = build_eval_matrix()
        oracles, sm_count = init_cuda_oracles(M)

    lattice  = bounding_cube_points()
    targets  = bounds_for_n(49)
    bs       = sm_count * 1024 * len(oracles)

    run_ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"canon_{run_ts}.json")
    print(f"Results → {results_file}")

    canonical_champion_search(
        oracles         = oracles,
        lattice         = lattice,
        results_file    = results_file,
        targets         = targets,
        max_k           = 12,
        batch_size      = bs,
        prune_margin    = None,
        resume          = True,
        k_canonical_max = 9,
    )
    return results_file


if __name__ == "__main__":
    main()
