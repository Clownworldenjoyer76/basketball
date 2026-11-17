#!/usr/bin/env python3
"""DRatings manual basketball final scores -> existing 18-field drat_raw JSON.

For games already present in raw JSON, only adds scores/status; it retains
original pregame probabilities, lines, and projections without hindsight edits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from basketball_manual_input import LEAGUE_SPORT, merge_into_files, parse_text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--league", choices=sorted(LEAGUE_SPORT), required=True)
    ap.add_argument("--raw-file", type=Path, required=True)
    ap.add_argument("--time-basis", choices=["et", "utc"], required=True)
    args = ap.parse_args(argv)
    text = args.raw_file.read_text(encoding="utf-8-sig")
    rows = parse_text(text, args.league, "final", args.time_basis)
    paths = merge_into_files(rows, args.league, "final")
    for path, added, updated in paths:
        print(f"WROTE/VERIFIED {path}: added={added} updated={updated}")
    print(f"SUCCESS: parsed={len(rows)} league={args.league} files={len(paths)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
