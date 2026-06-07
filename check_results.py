#!/usr/bin/env python3
"""
Compare found codes against codetables.de lower bounds for GF(8), n=49.

Usage:
    python check_results.py                  # uses most recent champions_*.json
    python check_results.py results.json     # specific file
    python check_results.py *.json           # multiple files merged
"""

import argparse
import json
import sys
from pathlib import Path

from codetables import bounds_for_n

N = 49  # torus size — fixed for this project


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
    files = sorted(Path(".").glob("champions_*.json"))
    return files[-1] if files else None


def summarise(records: list[dict], bounds: dict[int, int]) -> None:
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
        if not rec.get("name", "").startswith("bfs_"):
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

    # Highlight wins
    if new_records:
        print(f"\n{'='*60}")
        print(f"NEW RECORDS ({len(new_records)}):  toric codes beating best known general bound!")
        print(f"{'='*60}")
        for k, d, rec in new_records:
            td = bounds.get(k, "?")
            print(f"  [n={N}, k={k}]  d={d}  (table={td},  gap={td - d if isinstance(td, int) else '?'})")
            print(f"  indices: {rec['indices']}")
            if "lattice_points" in rec:
                print(f"  lattice: {rec['lattice_points']}")

    if matched:
        print(f"\nMATCHED BOUNDS ({len(matched)}):")
        for k, d, rec in matched:
            print(f"  [n={N}, k={k}]  d={d}  name={rec.get('name','—')}")

    if not new_records and not matched:
        ks = sorted(best)
        gaps = [bounds[k] - best[k][0] for k in ks if k in bounds]
        if gaps:
            best_gap = min(gaps)
            best_k = ks[gaps.index(best_gap)]
            print(f"\nClosest to bound: k={best_k}, gap={best_gap}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="JSONL results file(s)")
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
        print(f"  {p}:  {sum(1 for r in recs if r.get('type') != 'level_complete')} code records")

    print()
    summarise(all_records, bounds)


if __name__ == "__main__":
    main()
