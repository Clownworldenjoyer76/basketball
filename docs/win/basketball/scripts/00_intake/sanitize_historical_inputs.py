#!/usr/bin/env python3
"""Remove clearly corrupt historical projection records without guessing values.

The sanitizer preserves every untouched CSV line byte-for-byte and removes only rows
whose projected team/total scores are far outside any plausible basketball range.
A fresh diagnostic is rewritten every run so stale warnings do not persist.

Removed rows are never lost: every run that removes rows also writes the full
original rows to a new file, historical_input_removed_rows_<UTC date/time>.csv,
in the error folder. Existing files of that name are never overwritten.

Values that are present but not numeric (for example "abc") are NOT removed;
the row is kept and a WARNING is logged.

All changed historical files are written to temporary files first and swapped
in together. If any step fails, no historical file is changed (files already
swapped are restored) and the removed-rows file of that run is discarded.
"""
from __future__ import annotations

import codecs
import csv
import io
import math
import os
import re
import tempfile
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
REMOVED_ROWS_PREFIX = "historical_input_removed_rows_"
REMOVED_ROWS_FIELDS = [
    "source_file", "line_number", "reason", "header_line", "raw_record",
]
PROJECTION_FIELDS = (
    "home_projected_points", "away_projected_points", "total_projected_points",
)
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


def split_records(text: str) -> list[tuple[int, str]]:
    """Split CSV text into records: [(first physical line number, record text)].

    Only \\r\\n, \\n and \\r end a line. A record that is inside double quotes
    continues onto the next line, so quoted multi-line fields stay in one record.
    Each record keeps its original line ending.
    """
    pieces = re.findall(r"[^\r\n]*(?:\r\n|\n|\r)|[^\r\n]+", text)
    records: list[tuple[int, str]] = []
    current: list[str] = []
    quotes = 0
    start = 1
    for number_, piece in enumerate(pieces, start=1):
        if not current:
            start = number_
        current.append(piece)
        quotes += piece.count('"')
        if quotes % 2 == 0:
            records.append((start, "".join(current)))
            current = []
            quotes = 0
    if current:
        records.append((start, "".join(current)))
    return records


def non_numeric_fields(row: dict[str, str]) -> list[tuple[str, str]]:
    """Projection fields that hold text which is not a finite number."""
    bad = []
    for field in PROJECTION_FIELDS:
        text = str(row.get(field, "") or "").strip()
        if text and number(text) is None:
            bad.append((field, text))
    return bad


def scan_file(path: Path) -> dict:
    """Read one file and decide what to remove. Writes nothing."""
    raw_bytes = path.read_bytes()
    had_bom = raw_bytes.startswith(codecs.BOM_UTF8)
    raw = raw_bytes.decode("utf-8-sig")
    result = {
        "output_bytes": None,
        "exclusions": [],
        "removed": [],
        "warnings": [],
    }
    records = split_records(raw)
    if not records:
        return result

    header_text = records[0][1].rstrip("\r\n")
    try:
        header_values = next(csv.reader(io.StringIO(header_text)))
    except StopIteration:
        return result
    required = set(PROJECTION_FIELDS)
    if not required.issubset(set(header_values)):
        return result

    kept = [records[0][1]]
    for line_number, original_record in records[1:]:
        record_text = original_record.rstrip("\r\n")
        if not record_text:
            kept.append(original_record)
            continue
        try:
            values = next(csv.reader(io.StringIO(record_text)))
        except (StopIteration, csv.Error):
            kept.append(original_record)
            continue
        if len(values) != len(header_values):
            kept.append(original_record)
            continue
        row = dict(zip(header_values, values))
        reason = reason_for(row)
        if reason is None:
            kept.append(original_record)
            for field, text in non_numeric_fields(row):
                result["warnings"].append({
                    "source_file": str(path),
                    "line_number": str(line_number),
                    "game_id": str(row.get("game_id", "")),
                    "field": field,
                    "value": text,
                })
            continue
        result["exclusions"].append({
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
        result["removed"].append({
            "source_file": str(path),
            "line_number": str(line_number),
            "reason": reason,
            "header_line": header_text,
            "raw_record": record_text,
        })

    if result["exclusions"]:
        # Preserve the original BOM/line endings/content for every retained line.
        output_bytes = "".join(kept).encode("utf-8")
        if had_bom:
            output_bytes = codecs.BOM_UTF8 + output_bytes
        result["output_bytes"] = output_bytes
    return result


def _write_temp_bytes(path: Path, data: bytes, prefix: str) -> Path:
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=prefix, suffix=path.suffix or ".tmp",
            delete=False,
        ) as f:
            tmp = Path(f.name)
            f.write(data)
        return tmp
    except BaseException:
        if tmp is not None and tmp.exists():
            tmp.unlink()
        raise


def swap_files(changes: list[tuple[Path, bytes]]) -> None:
    """All-or-nothing replacement of historical files."""
    prepared: list[tuple[Path, Path, bytes]] = []
    all_tmps: list[Path] = []
    swapped: list[tuple[Path, bytes]] = []
    try:
        for path, data in changes:
            original = path.read_bytes()
            tmp = _write_temp_bytes(path, data, ".sanitize-")
            all_tmps.append(tmp)
            prepared.append((path, tmp, original))
        for path, tmp, original in prepared:
            os.replace(tmp, path)
            swapped.append((path, original))
    except BaseException as exc:
        restore_errors = []
        for path, original in reversed(swapped):
            restore_tmp = None
            try:
                restore_tmp = _write_temp_bytes(path, original, ".sanitize-restore-")
                os.replace(restore_tmp, path)
            except Exception as restore_exc:
                restore_errors.append(f"{path}: {restore_exc}")
            finally:
                if restore_tmp is not None and restore_tmp.exists():
                    restore_tmp.unlink()
        if restore_errors:
            raise RuntimeError(
                f"File swap failed ({exc}); could not restore: "
                + "; ".join(restore_errors)
            ) from exc
        raise
    finally:
        for tmp in all_tmps:
            if tmp.exists():
                tmp.unlink()


def write_removed_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the removed-rows file; refuses to overwrite an existing file."""
    with open(path, "x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REMOVED_ROWS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_exclusions(rows: list[dict[str, str]]) -> None:
    with open(EXCLUSION_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DIAGNOSTIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    LOG_FILE.write_text(
        f"=== historical input sanitizer {started.isoformat()} ===\n",
        encoding="utf-8",
    )
    removed_path: Path | None = None
    swap_done = False
    try:
        files = candidate_files()
        scans: list[tuple[Path, dict]] = []
        exclusions: list[dict[str, str]] = []
        warnings: list[dict[str, str]] = []
        removed: list[dict[str, str]] = []

        for path in files:
            scan = scan_file(path)
            scans.append((path, scan))
            exclusions.extend(scan["exclusions"])
            warnings.extend(scan["warnings"])
            removed.extend(scan["removed"])
            for row in scan["exclusions"]:
                log(
                    "EXCLUDED CORRUPT HISTORICAL ROW | "
                    f"{row['source_file']}:{row['line_number']} | {row['game_id']} | {row['reason']}"
                )
            for item in scan["warnings"]:
                log(
                    "WARNING NON-NUMERIC VALUE KEPT | "
                    f"{item['source_file']}:{item['line_number']} | {item['game_id']} | "
                    f"{item['field']}={item['value']!r}"
                )

        # Save the full removed rows BEFORE any historical file is changed.
        if removed:
            stamp = started.strftime("%Y%m%dT%H%M%SZ")
            removed_path = ERROR_DIR / f"{REMOVED_ROWS_PREFIX}{stamp}.csv"
            write_removed_rows(removed_path, removed)
            log(f"REMOVED ROWS SAVED | {removed_path} | rows={len(removed)}")

        changes = [
            (path, scan["output_bytes"])
            for path, scan in scans
            if scan["output_bytes"] is not None
        ]
        write_exclusions(exclusions)
        swap_files(changes)
        swap_done = True

        log(f"Files scanned: {len(files)}")
        log(f"Files rewritten: {len(changes)}")
        log(f"Corrupt rows excluded this run: {len(exclusions)}")
        log(f"Non-numeric values kept (warnings): {len(warnings)}")
        log("STATUS: SUCCESS")
    except Exception as exc:
        if not swap_done:
            # No historical file was changed (a failed swap restores them).
            try:
                write_exclusions([])
            except Exception:
                pass
            if removed_path is not None and removed_path.exists():
                removed_path.unlink()
                log(
                    f"REMOVED ROWS FILE DISCARDED | {removed_path} | "
                    "no historical file was changed"
                )
        log(f"FATAL: {exc}")
        log(traceback.format_exc().rstrip())
        log("STATUS: FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
