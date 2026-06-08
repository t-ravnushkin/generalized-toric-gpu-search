#!/usr/bin/env python3
"""
Compare found codes against codetables.de lower bounds for GF(8), n=49.

Usage:
    python check_results.py                  # uses most recent champions_*.json/canon_*.json
    python check_results.py results.json     # specific file
    python check_results.py *.json           # multiple files merged
    python check_results.py --all-codes results.json
"""

import argparse
import json
import sys
from pathlib import Path

from codetables import bounds_for_n

N = 49  # torus size — fixed for this project


def index_to_point(i: int) -> tuple[int, int]:
    """Convert a row index 0..48 into its exponent point (a,b) in [0,6]^2."""
    return divmod(int(i), 7)


def indices_to_points(indices: list[int]) -> list[tuple[int, int]]:
    return [index_to_point(i) for i in indices]


def unpack_canonical(packed: int, k: int) -> list[int]:
    """Unpack the canonical packed representation used in canon_*.json logs."""
    packed = int(packed)
    if k <= 10:
        return [int((packed >> (6 * i)) & 63) for i in range(k)]
    return [0] + [int((packed >> (6 * i)) & 63) for i in range(k - 1)]


def record_indices(rec: dict) -> list[int] | None:
    """Return code indices from either explicit records or packed survivor rows."""
    if "indices" in rec:
        return [int(i) for i in rec["indices"]]
    if "packed" in rec and "k" in rec:
        return unpack_canonical(rec["packed"], int(rec["k"]))
    return None


def record_points(rec: dict) -> list[tuple[int, int]] | None:
    if "lattice_points" in rec:
        return [tuple(p) for p in rec["lattice_points"]]
    indices = record_indices(rec)
    if indices is None:
        return None
    return indices_to_points(indices)


def format_points(points: list[tuple[int, int]]) -> str:
    return "[" + ", ".join(f"({a},{b})" for a, b in points) + "]"


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL, a well-formed JSON array, or a truncated JSON array (partial write)."""
    text = path.read_text().strip()

    if text.startswith("["):
        # Try well-formed array first.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Truncated array: scan for complete objects using raw_decode.
        decoder = json.JSONDecoder()
        records, pos = [], 0
        while pos < len(text):
            # Skip whitespace, commas, brackets.
            while pos < len(text) and text[pos] in " \t\n\r,[":
                pos += 1
            if pos >= len(text) or text[pos] == "]":
                break
            try:
                obj, pos = decoder.raw_decode(text, pos)
                records.append(obj)
            except json.JSONDecodeError:
                break
        return records

    # JSONL format.
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def find_latest() -> Path | None:
    files = sorted(
        list(Path(".").glob("champions_*.json")) +
        list(Path(".").glob("canon_*.json")) +
        list(Path(".").glob("canon_local_*.json"))
    )
    return files[-1] if files else None


def summarise(
    records: list[dict],
    bounds: dict[int, int],
    *,
    all_codes: bool = False,
) -> None:
    # Collect our best d per k (skip sentinels and non-bfs named sets)
    best: dict[int, tuple[int, dict]] = {}   # k -> (best_d, record)
    named_non_bfs: list[dict] = []

    for rec in records:
        if rec.get("type") == "level_complete":
            continue
        k = rec.get("k")
        d = rec.get("min_distance")
        if k is None or d is None:
            continue
        name = rec.get("name", "")
        if not name.startswith("bfs_") and not name.startswith("canon_"):
            named_non_bfs.append(rec)
        if k not in best or d > best[k][0]:
            best[k] = (d, rec)

    # Add named non-bfs records (Part A) if they beat current best for that k
    for rec in named_non_bfs:
        k, d = rec["k"], rec["min_distance"]
        if k not in best or d > best[k][0]:
            best[k] = (d, rec)

    if not best:
        print("No code records found.")
        return

    # Table header
    header = f"{'k':>4}  {'our d':>6}  {'table d':>8}  {'gap':>5}  status"
    sep = "-" * len(header)
    print(header)
    print(sep)

    new_records: list[tuple[int, int, dict]] = []
    matched: list[tuple[int, int, dict]] = []

    for k in sorted(best):
        our_d, rec = best[k]
        table_d = bounds.get(k)
        if table_d is None:
            gap_str = "n/a"
            status = "no table entry"
        else:
            gap = table_d - our_d
            gap_str = f"{gap:+d}"
            if gap < 0:
                status = "*** NEW RECORD ***"
                new_records.append((k, our_d, rec))
            elif gap == 0:
                status = "MATCHES BOUND"
                matched.append((k, our_d, rec))
            elif gap <= 3:
                status = f"{gap} below"
            else:
                status = f"{gap} below"

        td_str = str(table_d) if table_d is not None else "—"
        print(f"{k:>4}  {our_d:>6}  {td_str:>8}  {gap_str:>5}  {status}")

    print(sep)

    print("\nBest code representatives:")
    for k in sorted(best):
        our_d, rec = best[k]
        indices = record_indices(rec)
        points = record_points(rec)
        print(f"  [n={N}, k={k}] d={our_d} name={rec.get('name', '—')}")
        if indices is not None:
            print(f"    indices: {indices}")
        if points is not None:
            print(f"    points : {format_points(points)}")

    # Highlight wins
    if new_records:
        print(f"\n{'='*60}")
        print(f"NEW RECORDS ({len(new_records)}):  toric codes beating best known general bound!")
        print(f"{'='*60}")
        for k, d, rec in new_records:
            td = bounds.get(k, "?")
            print(f"  [n={N}, k={k}]  d={d}  (table={td},  gap={td - d if isinstance(td, int) else '?'})")
            indices = record_indices(rec)
            points = record_points(rec)
            if indices is not None:
                print(f"  indices: {indices}")
            if points is not None:
                print(f"  points : {format_points(points)}")

    if matched:
        print(f"\nMATCHED BOUNDS ({len(matched)}):")
        for k, d, rec in matched:
            print(f"  [n={N}, k={k}]  d={d}  name={rec.get('name','—')}")
            points = record_points(rec)
            if points is not None:
                print(f"    points: {format_points(points)}")

    if not new_records and not matched:
        ks = sorted(best)
        gaps = [bounds[k] - best[k][0] for k in ks if k in bounds]
        if gaps:
            best_gap = min(gaps)
            best_k = ks[gaps.index(best_gap)]
            print(f"\nClosest to bound: k={best_k}, gap={best_gap}")

    if all_codes:
        code_records = [
            rec for rec in records
            if rec.get("type") not in ("level_complete", "survivor")
            and rec.get("k") is not None
            and rec.get("min_distance") is not None
        ]
        code_records.sort(
            key=lambda r: (int(r["k"]), -int(r["min_distance"]), r.get("name", ""))
        )
        print(f"\nAll code records ({len(code_records)}):")
        for rec in code_records:
            points = record_points(rec)
            indices = record_indices(rec)
            print(
                f"  {rec.get('name', '—')}  "
                f"k={rec.get('k')}  d={rec.get('min_distance')}"
            )
            if indices is not None:
                print(f"    indices: {indices}")
            if points is not None:
                print(f"    points : {format_points(points)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="JSONL results file(s)")
    parser.add_argument(
        "--all-codes",
        action="store_true",
        help="print every code record, not only the best representative per k",
    )
    args = parser.parse_args()

    paths: list[Path] = [Path(f) for f in args.files]
    if not paths:
        latest = find_latest()
        if latest is None:
            sys.exit("No champions_*.json found in current directory.")
        paths = [latest]

    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.exit(f"File(s) not found: {missing}")

    print(f"Loading bounds for GF(8), n={N} from codetables.de ...")
    bounds = bounds_for_n(N)

    all_records: list[dict] = []
    for p in paths:
        recs = load_jsonl(p)
        all_records.extend(recs)
        print(f"  {p}:  {sum(1 for r in recs if r.get('type') not in ('level_complete', 'survivor'))} code records")

    print()
    summarise(all_records, bounds, all_codes=args.all_codes)


if __name__ == "__main__":
    main()
