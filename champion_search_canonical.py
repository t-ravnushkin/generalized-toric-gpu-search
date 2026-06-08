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
from queue import Queue
from threading import Thread

import numpy as np
from tqdm.auto import tqdm

try:
    from kernel_canon import init_canon_oracle
    _HAS_GPU_CANON = True
except Exception:
    _HAS_GPU_CANON = False


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
# Relative-position table  (49×49, fits in L1 cache)
# ---------------------------------------------------------------------------
# rel_table[a, b] = index of (point_b − point_a) mod 7 in F₇².
# Replaces the 5-op chain  (tr//7, tr%7, subtract, %7, *7+)  with one lookup.

_REL_TABLE: np.ndarray | None = None


def _rel_table() -> np.ndarray:
    global _REL_TABLE
    if _REL_TABLE is None:
        pts_a = np.arange(49, dtype=np.int32) // 7
        pts_b = np.arange(49, dtype=np.int32) % 7
        _REL_TABLE = (
            (pts_a[None, :] - pts_a[:, None]) % 7 * 7 +
            (pts_b[None, :] - pts_b[:, None]) % 7
        ).astype(np.uint8)   # (49, 49), values 0–48
    return _REL_TABLE


# ---------------------------------------------------------------------------
# Canonical form — CPU reference (kept for testing; pipeline uses GPU kernel)
# ---------------------------------------------------------------------------

def canonical_forms_batch(
    sets_arr: np.ndarray,   # (n, k)  int32, each row sorted
    gl2_perms: np.ndarray,  # (2016, 49)  int32
) -> np.ndarray:
    """CPU fallback: vectorised canonical form for n k-sets. Returns uint64 (n,).

    This reference path packs all k positions into uint64, so it is limited to
    k <= 10.  The CUDA canonical kernel supports larger 128-bit keys.
    """
    if sets_arr.shape[1] > 10:
        raise ValueError("CPU canonical fallback supports k <= 10")
    _CHUNK = 64
    n, k = sets_arr.shape
    rt   = _rel_table()
    best = np.full(n, np.iinfo(np.uint64).max, dtype=np.uint64)
    for pi in range(0, len(gl2_perms), _CHUNK):
        pc = gl2_perms[pi: pi + _CHUNK]
        tr = pc[:, sets_arr]
        for t in range(k):
            idx = rt[tr[:, :, [t]], tr]
            idx.sort(axis=2)
            packed = np.zeros((len(pc), n), dtype=np.uint64)
            for j in range(k):
                packed |= idx[:, :, j].astype(np.uint64) << np.uint64(6 * j)
            np.minimum(best, packed.min(axis=0), out=best)
    return best


class CPUCanonicalOracle:
    """Local canonical-form fallback for machines without CUDA/CuPy."""

    def __init__(self, gl2_perms: np.ndarray):
        self.gl2_perms = gl2_perms

    def compute(self, sets: np.ndarray) -> np.ndarray:
        return canonical_forms_batch(sets, self.gl2_perms).astype(object)


def unpack_canonical(packed: int, k: int) -> tuple[int, ...]:
    """Unpack a canonical packed value.
    k≤10 : 6 bits per index, positions 0..k-1  (position 0 explicit, always 0).
    k≥11 : position 0 implicit 0; positions 1..k-1 at bits 0..6(k-1)-1.
           k=11 → 60-bit int; k=12 → 66-bit Python int (lo | hi<<64).
           Raw-BFS for k≥11 uses the same skip-pos-0 convention."""
    if k <= 10:
        return tuple(int((packed >> (6 * i)) & 63) for i in range(k))
    return (0,) + tuple(int((packed >> (6 * i)) & 63) for i in range(k - 1))


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

def _call_max_zeros_batch(
    oracle,
    batch: list[list[int]],
    td: int,
    sample_count: int,
    sample_seed: int,
) -> list[int]:
    if sample_count <= 0:
        return oracle.max_zeros_batch(batch, td)
    try:
        return oracle.max_zeros_batch(
            batch, td, sample_count=sample_count, sample_seed=sample_seed
        )
    except TypeError as exc:
        raise TypeError(
            "This distance oracle does not support sampled evaluation. "
            "Use kernel_cuda_bp.py or set SCREEN_SAMPLE_COUNT=0."
        ) from exc


def _dispatch(
    oracles,
    batch: list[list[int]],
    td: int,
    sample_count: int = 0,
    sample_seed: int = 0,
) -> list[int]:
    if not batch:
        return []
    n = len(oracles)
    if n == 1:
        return _call_max_zeros_batch(oracles[0], batch, td, sample_count, sample_seed)
    q, r = divmod(len(batch), n)
    sizes = [q + (1 if i < r else 0) for i in range(n)]
    chunks, start = [], 0
    for s in sizes:
        chunks.append(batch[start: start + s])
        start += s
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [
            pool.submit(
                _call_max_zeros_batch,
                o,
                c,
                td,
                sample_count,
                sample_seed + dev_id * 1_000_003,
            )
            for dev_id, (o, c) in enumerate(zip(oracles, chunks))
            if c
        ]
        results = []
        for f in futures:
            results.extend(f.result())
    return results


# ---------------------------------------------------------------------------
# Pipelined GPU canonicalization + GPU evaluation
# ---------------------------------------------------------------------------

_CANON_CHUNK = 200_000   # sets per canonicalization chunk


def _pipelined_eval(
    raw: list[tuple[int, ...]],
    canon_oracle,          # CanonicalOracle | None  (None → raw-BFS)
    k: int,
    eval_distance: int,
    batch_size: int,
    oracles: list,
    sample_count: int = 0,
    sample_seed: int = 0,
) -> tuple[list[int], list[tuple[int, ...]], list[int]]:
    """
    Overlap GPU canonical-form computation with GPU evaluation.

    Producer: chunks raw tuples → GPU canonical kernel → CPU dedup → queue.
    Consumer: dequeues batches → evaluation oracles → collects results.

    When canon_oracle is None (k > k_canonical_max), raw sorted-tuple
    deduplication is used instead, with Python-int packing to avoid uint64
    overflow at k ≥ 11 (11×6 = 66 > 64 bits).
    """
    q: Queue = Queue(maxsize=max(2, len(oracles)))

    packed_out:  list[int]             = []
    indices_out: list[tuple[int, ...]] = []
    mz_out:      list[int]             = []

    def _producer() -> None:
        seen: set[int] = set()
        buf_p: list[int]             = []
        buf_i: list[tuple[int, ...]] = []

        for ci in range(0, len(raw), _CANON_CHUNK):
            chunk = np.array(raw[ci: ci + _CANON_CHUNK], dtype=np.int32)

            if canon_oracle is not None:
                # GPU: apply all 2016 GL₂ permutations × k anchors in parallel.
                packed_arr  = canon_oracle.compute(chunk)   # (n,) uint64
                packed_vals = packed_arr.tolist()
                index_vals  = [unpack_canonical(p, k) for p in packed_vals]
            else:
                # Raw-BFS: use skip-pos-0 for k≥11 to match GPU canonical convention
                # so unpack_canonical works uniformly across both paths.
                chunk_list  = chunk.tolist()
                if k <= 10:
                    packed_vals = [
                        sum(v << (6 * j) for j, v in enumerate(row))
                        for row in chunk_list
                    ]
                else:
                    # position 0 always 0 (implicit); pack positions 1..k-1
                    packed_vals = [
                        sum(row[j] << (6 * (j - 1)) for j in range(1, k))
                        for row in chunk_list
                    ]
                index_vals = [tuple(row) for row in chunk_list]

            for p, idx in zip(packed_vals, index_vals):
                if p not in seen:
                    seen.add(p)
                    buf_p.append(p)
                    buf_i.append(idx)
                    if len(buf_i) == batch_size:
                        q.put((buf_p[:], buf_i[:]))
                        buf_p.clear()
                        buf_i.clear()

        if buf_i:
            q.put((buf_p, buf_i))
        q.put(None)  # sentinel

    def _consumer() -> None:
        pbar = tqdm(unit=" sets", desc=f"k={k}", dynamic_ncols=True, leave=False)
        while True:
            item = q.get()
            if item is None:
                pbar.close()
                return
            bp, bi = item
            mz = _dispatch(
                oracles,
                [list(s) for s in bi],
                eval_distance,
                sample_count=sample_count,
                sample_seed=sample_seed,
            )
            packed_out.extend(bp)
            indices_out.extend(bi)
            mz_out.extend(mz)
            pbar.update(len(bi))

    prod = Thread(target=_producer, daemon=True)
    cons = Thread(target=_consumer, daemon=True)
    prod.start()
    cons.start()
    prod.join()
    cons.join()

    return packed_out, indices_out, mz_out


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------


def canonical_champion_search(
    oracles,
    lattice: list[tuple[int, int]],
    results_file: Path,
    targets: dict[int, int],      # {k: best_known_d} from codetables
    max_k: int = 12,
    batch_size: int = 50_000,
    prune_margin: int | None = None,
    resume: bool = True,
    k_canonical_max: int = 10,
    prune_eval_mode: str = "champion",
    screen_from_k: int | None = None,
    screen_sample_count: int = 0,
    screen_seed: int = 1,
    screen_candidate_log_limit: int = 200,
):
    """
    Hybrid champion search: GPU canonical forms for k ≤ k_canonical_max,
    raw sorted-tuple BFS for k > k_canonical_max.

    Canonical forms exploit AGL₂(F₇) symmetry (order 98,784) so each orbit
    is evaluated once.  The GPU kernel applies all 2016 GL₂ matrices × k
    anchors in parallel (one warp per set), keeping the GPU busy at all k.

    k_canonical_max:
      k ≤ this — GPU GL₂ canonical forms (complete; current kernel supports
                 k ≤ 16 with 128-bit packed canonical keys).
      k > this — raw sorted-tuple BFS; Python-int packing avoids uint64
                 overflow at k ≥ 11.
      Default 10 for conservative CLI runs; Kaggle notebook sets 15.

    prune_margin:
      None  — keep ALL surviving k-forms for the next level.
      int   — prune sets with min_distance < targets[k] - prune_margin.

    prune_eval_mode:
      "champion" — fast record-hunting mode.  Evaluate at targets[k], so most
                   non-champions abort early.  Survivor pruning is optimistic:
                   it never discards a set that might meet the margin cutoff,
                   but may keep false positives for later levels.
      "survivor" — exact survivor-pruning mode.  Evaluate at
                   targets[k] - prune_margin, which gives an honest frontier
                   but can be much slower for loose margins at k≥10.

    screen_from_k / screen_sample_count:
      Optional sampled screening for deep levels.  For k >= screen_from_k,
      evaluate only screen_sample_count pseudo-random nonzero coefficient
      vectors per set.  This keeps searches moving at k≥12, but distances and
      champions from screened levels are NOT certified.
    """
    if prune_eval_mode not in {"champion", "survivor"}:
        raise ValueError("prune_eval_mode must be 'champion' or 'survivor'")

    gl2    = _gl2()
    n_gpus = len(oracles) if hasattr(oracles, "__len__") else 1
    if not isinstance(oracles, list):
        oracles = [oracles]

    # Initialise canonical-form engine (compile once on CUDA; CPU fallback locally).
    canon_oracle = None
    if _HAS_GPU_CANON and k_canonical_max > 0 and hasattr(oracles[0], "device_id"):
        canon_oracle = init_canon_oracle(
            device_id = oracles[0].device_id,
            gl2_perms = gl2,
            rel_table = _rel_table(),
        )
    elif k_canonical_max > 0:
        if k_canonical_max > 10:
            print("[canon] CUDA canonical kernel unavailable; "
                  "using CPU canonical fallback for k≤10")
            k_canonical_max = 10
        canon_oracle = CPUCanonicalOracle(gl2)

    print(f"\n=== Hybrid Champion Search  max_k={max_k}  gpus={n_gpus}  "
          f"k_canonical_max={k_canonical_max}  "
          f"prune={'none' if prune_margin is None else prune_margin}  "
          f"prune_eval={prune_eval_mode} ===\n"
          f"    GPU canonical (AGL₂) for k≤{k_canonical_max}; "
          f"raw-BFS for k>{k_canonical_max}\n")
    if screen_from_k is not None and screen_sample_count > 0:
        print(f"    sampled screening for k≥{screen_from_k}: "
              f"{screen_sample_count:,} random codewords/set "
              f"(unverified distances)\n")

    # ── Resume support ────────────────────────────────────────────────────
    start_k, canonical_level = 1, {0}
    if resume and results_file.exists():
        start_k, canonical_level = load_canonical_state(results_file)
        if start_k > 1:
            print(f"  Resuming after k={start_k}  "
                  f"({len(canonical_level)} canonical survivors loaded)")

    for k in range(start_k + 1, max_k + 1):
        td = targets.get(k, 1)
        keep_distance = 1 if prune_margin is None else max(1, td - prune_margin)
        eval_distance = (
            td
            if prune_margin is None or prune_eval_mode == "champion"
            else keep_distance
        )
        sample_count = (
            screen_sample_count
            if screen_from_k is not None and k >= screen_from_k
            else 0
        )
        is_screened = sample_count > 0

        # ── Expand raw extensions ─────────────────────────────────────────
        t0 = time.perf_counter()

        prev_list: list[tuple[int, ...]] = [
            unpack_canonical(p, k - 1) for p in canonical_level
        ]

        raw: list[tuple[int, ...]] = []
        for S in prev_list:
            s_set = set(S)
            for p in range(49):
                if p not in s_set:
                    raw.append(tuple(sorted(S + (p,))))
        raw = list(dict.fromkeys(raw))

        # ── Pipeline: GPU canonicalization interleaved with GPU evaluation ──
        co         = canon_oracle if (k <= k_canonical_max) else None
        mode_label = "canonical forms" if co is not None else "raw-BFS sets"

        new_packed, new_indices, all_mz = _pipelined_eval(
            raw,
            co,
            k,
            eval_distance,
            batch_size,
            oracles,
            sample_count=sample_count,
            sample_seed=screen_seed + k * 1_000_003,
        )

        n_cands = len(new_indices)
        t_total = time.perf_counter() - t0
        print(f"  k={k}: {n_cands:,} {mode_label}  target_d={td}  "
              f"keep_d≥{keep_distance if prune_margin is not None else 'all'}  "
              f"eval_d={eval_distance}  "
              f"{'sampled=' + format(sample_count, ',') + '  ' if is_screened else ''}"
              f"time={t_total:.1f}s")

        # ── Record champions and survivors ────────────────────────────────
        next_level: set[int] = set()
        n_champ = 0
        n_screen_candidates_logged = 0

        for packed_j, S, mz in zip(new_packed, new_indices, all_mz):
            d = 49 - mz
            if d >= td and not is_screened:
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
            elif (
                d >= td
                and is_screened
                and n_screen_candidates_logged < screen_candidate_log_limit
            ):
                n_screen_candidates_logged += 1
                append_result({
                    "type":         "candidate",
                    "verified":     False,
                    "name":         f"screen_k{k}_sample_d{d}",
                    "indices":      list(S),
                    "lattice_points": [lattice[i] for i in S],
                    "k":            k,
                    "sampled_max_zeros": mz,
                    "sampled_min_distance": d,
                    "best_known_d": td,
                    "sample_count": sample_count,
                }, results_file)

            keep = (prune_margin is None) or (d >= keep_distance)
            if keep:
                next_level.add(packed_j)
                # persist so resume works
                append_result({"type": "survivor", "k": k,
                               "packed": packed_j}, results_file)

        append_result({"type": "level_complete", "k": k,
                       "n_canonical": n_cands, "n_champions": n_champ,
                       "n_survivors": len(next_level),
                       "prune_eval_mode": prune_eval_mode,
                       "eval_distance": eval_distance,
                       "keep_distance": keep_distance,
                       "screen_sample_count": sample_count,
                       "verified": not is_screened}, results_file)

        survivor_label = (
            "optimistic survivors"
            if prune_margin is not None and prune_eval_mode == "champion"
            else "survivors"
        )
        if is_screened:
            print(f"  k={k}: sampled screening only; 0 verified champions, "
                  f"{len(next_level):,} {survivor_label}, "
                  f"{n_screen_candidates_logged} candidates logged\n")
        else:
            print(f"  k={k}: {n_champ} champions (d≥{td}), "
                  f"{len(next_level):,} {survivor_label}\n")

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
        k_canonical_max = 10,
        prune_eval_mode = "champion",
        screen_from_k   = None,
        screen_sample_count = 0,
        screen_candidate_log_limit = 200,
    )
    return results_file


if __name__ == "__main__":
    main()
