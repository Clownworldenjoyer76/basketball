#!/usr/bin/env python3
"""Remove clearly corrupt historical projection records without guessing values.

The sanitizer preserves every untouched CSV line byte-for-byte and removes only rows
whose projected team/total scores are far outside any plausible basketball range.
A fresh diagnostic is rewritten every run so stale warnings do not persist.
"""
from __future__ import annotations

import codecs
import csv
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("docs/win/basketball")
ROOT = BASE / "00_intake/final_combined_files"
ERROR_DIR = BASE / "errors/00_intake"
LOG_FILE = ERROR_DIR / "historical_input_sanitize.txt"
EXCLUSION_FILE = ERROR_DIR / "historical_input_exclusions.csv"

MAX_TEAM_PROJECTION = 250.0
MAX_TOTAL_PROJECTION = 500.0
DIAGNOSTIC_FIELDS = [
    "source_file", "line_number", "league", "game_date", "game_id",
    "home_team", "away_team", "home_projected_points", "away_projected_points",
    "total_projected_points", "reason",
]


def log(message: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} | {message}\n")


def number(value: str) -> float | None:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def reason_for(row: dict[str, str]) -> str | None:
    home = number(row.get("home_projected_points", ""))
    away = number(row.get("away_projected_points", ""))
    total = number(row.get("total_projected_points", ""))
    if home is not None and (home <= 0 or home > MAX_TEAM_PROJECTION):
        return f"implausible home_projected_points={row.get('home_projected_points', '')}"
    if away is not None and (away <= 0 or away > MAX_TEAM_PROJECTION):
        return f"implausible away_projected_points={row.get('away_projected_points', '')}"
    if total is not None and (total <= 0 or total > MAX_TOTAL_PROJECTION):
        return f"implausible total_projected_points={row.get('total_projected_points', '')}"
    return None


def candidate_files() -> list[Path]:
    files = sorted(ROOT.glob("*_predictions.csv"))
    combined = ROOT / "combined"
    if combined.exists():
        files.extend(sorted(combined.glob("*.csv")))
    return files


def sanitize_file(path: Path) -> list[dict[str, str]]:
    raw_bytes = path.read_bytes()
    had_bom = raw_bytes.startswith(codecs.BOM_UTF8)
    raw = raw_bytes.decode("utf-8-sig")
    lines = raw.splitlines(keepends=True)
    if not lines:
        return []

    try:
        header_values = next(csv.reader([lines[0].rstrip("\r\n")]))
    except StopIteration:
        return []
    required = {"home_projected_points", "away_projected_points", "total_projected_points"}
    if not required.issubset(set(header_values)):
        return []

    exclusions: list[dict[str, str]] = []
    kept = [lines[0]]
    for line_number, original_line in enumerate(lines[1:], start=2):
        record_text = original_line.rstrip("\r\n")
        if not record_text:
            kept.append(original_line)
            continue
        try:
            values = next(csv.reader([record_text]))
        except StopIteration:
            kept.append(original_line)
            continue
        if len(values) != len(header_values):
            kept.append(original_line)
            continue
        row = dict(zip(header_values, values))
        reason = reason_for(row)
        if reason is None:
            kept.append(original_line)
            continue
        exclusions.append({
            "source_file": str(path),
            "line_number": str(line_number),
            "league": str(row.get("league", "")),
            "game_date": str(row.get("game_date", "")),
            "game_id": str(row.get("game_id", "")),
            "home_team": str(row.get("home_team", "")),
            "away_team": str(row.get("away_team", "")),
            "home_projected_points": str(row.get("home_projected_points", "")),
            "away_projected_points": str(row.get("away_projected_points", "")),
            "total_projected_points": str(row.get("total_projected_points", "")),
            "reason": reason,
        })

    if exclusions:
        # Preserve the original BOM/line endings/content for every retained line.
        output_bytes = "".join(kept).encode("utf-8")
        if had_bom:
            output_bytes = codecs.BOM_UTF8 + output_bytes
        path.write_bytes(output_bytes)
    return exclusions


def write_exclusions(rows: list[dict[str, str]]) -> None:
    with open(EXCLUSION_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DIAGNOSTIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(
        f"=== historical input sanitizer {datetime.now(timezone.utc).isoformat()} ===\n",
        encoding="utf-8",
    )
    exclusions: list[dict[str, str]] = []
    try:
        files = candidate_files()
        for path in files:
            rows = sanitize_file(path)
            exclusions.extend(rows)
            for row in rows:
                log(
                    "EXCLUDED CORRUPT HISTORICAL ROW | "
                    f"{row['source_file']}:{row['line_number']} | {row['game_id']} | {row['reason']}"
                )
        write_exclusions(exclusions)
        log(f"Files scanned: {len(files)}")
        log(f"Corrupt rows excluded this run: {len(exclusions)}")
        log("STATUS: SUCCESS")
    except Exception as exc:
        write_exclusions(exclusions)
        log(f"FATAL: {exc}")
        log(traceback.format_exc().rstrip())
        log("STATUS: FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
