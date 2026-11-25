#!/usr/bin/env python3
"""Consolidate sportsbook rows that represent the same game under different IDs.

Identity is league + game_date + normalized home + normalized away. Numeric ESPN
IDs are preferred over legacy/custom IDs. Distinct numeric IDs for one identity are
fatal because that indicates a genuine identity conflict rather than an alias.
Rows with unresolved placeholder teams are excluded from identity consolidation.

Sportsbook provenance fields are preserved through normalization, alias
consolidation, and file rewrites:
- sportsbook_provider
- scraped_at_utc
- provider_updated_at_utc

Merge rule for rows of one identity: the newest row (by odds_last_update, then
number of filled fields, then load order) wins for every price column
(PRICE_FIELDS), and a blank price in the newest row stays blank. Older rows
only fill blank non-price columns. Columns that are not in FIELDNAMES are kept.
All changed files are written to temporary files first and swapped in together;
if a swap fails, files already swapped are restored.
"""
from __future__ import annotations

import os
import sys
import csv
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from basketball_shared import resolve_repository_path, identity_key as shared_identity_key

BASE = Path("docs/win/basketball")
ERROR_DIR = BASE / "errors/00_intake"
LOG_FILE = ERROR_DIR / "basketball_sportsbook_alias_cleanup.txt"

LEAGUES = {
    "nba": ("NBA", BASE / "00_intake/sportsbook/nba"),
    "ncaam": ("NCAAM", BASE / "00_intake/sportsbook/ncaam"),
    "wnba": ("WNBA", BASE / "00_intake/sportsbook/wnba"),
}

FIELDNAMES = [
    "sport", "league", "game_date", "game_id", "odds_last_update",
    "sportsbook_provider", "scraped_at_utc", "provider_updated_at_utc",
    "game_time",
    "home_team", "away_team", "home_spread", "away_spread", "total",
    "home_dk_moneyline_american", "away_dk_moneyline_american",
    "home_dk_spread_american", "away_dk_spread_american",
    "dk_total_over_american", "dk_total_under_american",
    "home_dk_moneyline_decimal", "away_dk_moneyline_decimal",
    "home_dk_spread_decimal", "away_dk_spread_decimal",
    "dk_total_over_decimal", "dk_total_under_decimal",
]

# Market price/line columns. The newest row wins for these; blanks are not filled
# from older rows (an older price must never be mixed into a newer snapshot).
PRICE_FIELDS = frozenset({
    "home_spread", "away_spread", "total",
    "home_dk_moneyline_american", "away_dk_moneyline_american",
    "home_dk_spread_american", "away_dk_spread_american",
    "dk_total_over_american", "dk_total_under_american",
    "home_dk_moneyline_decimal", "away_dk_moneyline_decimal",
    "home_dk_spread_decimal", "away_dk_spread_decimal",
    "dk_total_over_decimal", "dk_total_under_decimal",
})

UNRESOLVED_TEAM_NAMES = {
    "tbd",
    "to be determined",
    "unknown",
}


def clean(value) -> str:
    return "" if value is None else str(value).strip()


def unresolved_team(value) -> bool:
    team = clean(value).casefold()
    return not team or team in UNRESOLVED_TEAM_NAMES


def log(message: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} | {message}\n")


def normalize_row(row: dict) -> dict:
    normalized = {field: clean(row.get(field)) for field in FIELDNAMES}
    # Keep columns that are not in FIELDNAMES (DictReader uses None for cells
    # beyond the header; those are ignored).
    for field, value in row.items():
        if field is None or field in normalized:
            continue
        normalized[field] = clean(value)
    return normalized


def identity_key(row: dict) -> tuple[str, str, str, str]:
    return shared_identity_key(
        row,
        clean,
        unresolved_team,
    )


def parse_timestamp(value: str) -> datetime:
    text = clean(value)
    if not text:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def nonblank_count(row: dict) -> int:
    return sum(1 for field in FIELDNAMES if clean(row.get(field)))


def merge_newest_wins(ordered_rows: list[dict]) -> dict:
    """Merge rows ordered oldest -> newest.

    Price columns come from the newest row only (blank stays blank). Every other
    column takes the newest row's value, or the next newest non-blank value.
    """
    merged = normalize_row(ordered_rows[-1])
    for older in reversed(ordered_rows[:-1]):
        for field, value in normalize_row(older).items():
            if field in PRICE_FIELDS:
                continue
            if not merged.get(field) and value:
                merged[field] = value
    return merged


def choose_canonical_id(copies: list[tuple]) -> str:
    ids = {clean(item[3].get("game_id")) for item in copies if clean(item[3].get("game_id"))}
    numeric_ids = sorted(gid for gid in ids if gid.isdigit())
    if len(numeric_ids) > 1:
        key = identity_key(copies[0][3])
        raise ValueError(
            "Conflicting numeric game_ids for the same sportsbook identity: "
            f"{key} -> {', '.join(numeric_ids)}"
        )
    if numeric_ids:
        return numeric_ids[0]

    # No numeric ESPN ID exists. Keep the ID belonging to the strongest/latest row.
    ranked = sorted(
        copies,
        key=lambda item: (
            parse_timestamp(item[3].get("odds_last_update")),
            nonblank_count(item[3]),
            item[0],
        ),
    )
    return clean(ranked[-1][3].get("game_id"))


def output_fieldnames(rows: list[dict]) -> list[str]:
    """FIELDNAMES first, then any extra columns in order of first appearance."""
    seen = set(FIELDNAMES)
    extras: list[str] = []
    for row in rows:
        for field in row:
            if field is None or field in seen:
                continue
            seen.add(field)
            extras.append(field)
    return FIELDNAMES + extras


def _write_temp(path: Path, rows: list[dict]) -> tuple[Path, Path]:
    safe_path = resolve_repository_path(
        path,
        strict=False,
    )

    safe_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", newline="", encoding="utf-8", dir=safe_path.parent,
            prefix=".alias-", suffix=".csv", delete=False,
        ) as f:
            tmp = Path(f.name)
            writer = csv.DictWriter(
                f,
                fieldnames=output_fieldnames(rows),
                extrasaction="ignore",
                restval="",
            )
            writer.writeheader()
            writer.writerows(normalize_row(row) for row in rows)
        return safe_path, tmp
    except BaseException:
        if tmp is not None and tmp.exists():
            tmp.unlink()
        raise


def _restore(safe_path: Path, original: bytes) -> None:
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=safe_path.parent, prefix=".alias-restore-",
            suffix=".csv", delete=False,
        ) as f:
            tmp = Path(f.name)
            f.write(original)
        os.replace(tmp, safe_path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def write_files(items: list[tuple[Path, list[dict]]]) -> None:
    """All-or-nothing write: temp files first, then swap; restore on failure."""
    prepared: list[tuple[Path, Path, bytes | None]] = []
    all_tmps: list[Path] = []
    swapped: list[tuple[Path, bytes | None]] = []
    try:
        for path, rows in items:
            safe_path, tmp = _write_temp(path, rows)
            all_tmps.append(tmp)
            original = safe_path.read_bytes() if safe_path.exists() else None
            prepared.append((safe_path, tmp, original))
        for safe_path, tmp, original in prepared:
            os.replace(tmp, safe_path)
            swapped.append((safe_path, original))
    except BaseException as exc:
        restore_errors = []
        for safe_path, original in reversed(swapped):
            try:
                if original is None:
                    safe_path.unlink()
                else:
                    _restore(safe_path, original)
            except Exception as restore_exc:
                restore_errors.append(f"{safe_path}: {restore_exc}")
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


def write_file(path: Path, rows: list[dict]) -> None:
    write_files([(path, rows)])


def _load_consolidation_rows():
    file_rows: dict[Path, list[dict]] = {}
    groups: dict[
        tuple[str, str, str, str],
        list[tuple[int, Path, int, dict]],
    ] = {}
    sequence = 0

    for _league, (label, folder) in LEAGUES.items():
        if not folder.exists():
            log(f"WARNING | Missing folder, skipped: {folder}")
            continue
        league_files = 0
        league_rows = 0
        for path in sorted(folder.glob(f"*_{label}_odds.csv")):
            with open(path, newline="", encoding="utf-8-sig") as source:
                rows = [
                    normalize_row(row)
                    for row in csv.DictReader(source)
                ]
            file_rows[path] = rows
            league_files += 1
            league_rows += len(rows)
            for row_index, row in enumerate(rows):
                key = identity_key(row)
                if all(key):
                    groups.setdefault(key, []).append(
                        (sequence, path, row_index, row)
                    )
                sequence += 1
        log(f"{label} | files scanned={league_files} rows scanned={league_rows}")

    return file_rows, groups


def _consolidate_identity(
    key,
    canonical_id,
    copies,
    file_rows,
    changed_paths,
    remove_indexes,
):
    ordered = sorted(
        copies,
        key=lambda item: (
            parse_timestamp(item[3].get("odds_last_update")),
            nonblank_count(item[3]),
            item[0],
        ),
    )
    consolidated = merge_newest_wins([item[3] for item in ordered])
    consolidated["game_id"] = canonical_id

    canonical_copies = [
        item
        for item in ordered
        if clean(item[3].get("game_id")) == canonical_id
    ]
    target = canonical_copies[-1] if canonical_copies else ordered[-1]
    _, target_path, target_index, _ = target
    file_rows[target_path][target_index] = consolidated
    changed_paths.add(target_path)

    aliases = 0
    duplicates = 0
    for _, path, row_index, row in copies:
        if path == target_path and row_index == target_index:
            continue
        remove_indexes.setdefault(path, set()).add(row_index)
        changed_paths.add(path)
        alias = clean(row.get("game_id"))
        if alias and alias != canonical_id:
            aliases += 1
            log(
                "ID ALIAS CONSOLIDATED | "
                f"{key[0]} {key[1]} | "
                f"{row.get('home_team')} vs {row.get('away_team')} | "
                f"{alias} -> {canonical_id}"
            )
        else:
            duplicates += 1
            log(
                "DUPLICATE ROW REMOVED | "
                f"{key[0]} {key[1]} | "
                f"{row.get('home_team')} vs {row.get('away_team')} | "
                f"game_id={alias or '(blank)'} kept={canonical_id}"
            )
    return aliases, duplicates


def consolidate() -> tuple[int, int, int, int]:
    """Return (identities consolidated, alias IDs removed, files rewritten,
    duplicate rows removed that carried the same or a blank game_id)."""
    file_rows, groups = _load_consolidation_rows()
    log(
        f"Files scanned: {len(file_rows)} | "
        f"Rows scanned: {sum(len(rows) for rows in file_rows.values())}"
    )
    if not file_rows:
        log("NO SPORTSBOOK FILES FOUND | nothing to consolidate")
    canonical_by_key = {
        key: choose_canonical_id(copies)
        for key, copies in groups.items()
        if len(copies) > 1
    }

    changed_paths: set[Path] = set()
    remove_indexes: dict[Path, set[int]] = {}
    aliases = 0
    duplicates = 0

    for key, canonical_id in canonical_by_key.items():
        identity_aliases, identity_duplicates = _consolidate_identity(
            key,
            canonical_id,
            groups[key],
            file_rows,
            changed_paths,
            remove_indexes,
        )
        aliases += identity_aliases
        duplicates += identity_duplicates

    for path, indexes in remove_indexes.items():
        file_rows[path] = [
            row
            for idx, row in enumerate(file_rows[path])
            if idx not in indexes
        ]

    ordered_paths = sorted(changed_paths)
    write_files([(path, file_rows[path]) for path in ordered_paths])
    for path in ordered_paths:
        log(f"REWROTE {path} ({len(file_rows[path])} rows)")

    return len(canonical_by_key), aliases, len(changed_paths), duplicates


def main() -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(
        f"=== sportsbook alias cleanup {datetime.now(timezone.utc).isoformat()} ===\n",
        encoding="utf-8",
    )
    try:
        identities, aliases, files, duplicates = consolidate()
        log(f"Duplicate identities consolidated: {identities}")
        log(f"Alias IDs removed: {aliases}")
        log(f"Duplicate rows removed (same or blank game_id): {duplicates}")
        log(f"Files rewritten: {files}")
        log("STATUS: SUCCESS")
    except Exception as exc:
        log(f"FATAL: {exc}")
        log(traceback.format_exc().rstrip())
        log("STATUS: FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
