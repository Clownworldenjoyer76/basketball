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
"""
from __future__ import annotations

import csv
import traceback
from datetime import datetime, timezone
from pathlib import Path

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
    return {field: clean(row.get(field)) for field in FIELDNAMES}


def identity_key(row: dict) -> tuple[str, str, str, str]:
    league = clean(row.get("league")).upper()
    game_date = clean(row.get("game_date"))
    home_team = clean(row.get("home_team"))
    away_team = clean(row.get("away_team"))

    if unresolved_team(home_team) or unresolved_team(away_team):
        return league, game_date, "", ""

    return (
        league,
        game_date,
        home_team.casefold(),
        away_team.casefold(),
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


def merge_nonblank(old_row: dict, new_row: dict) -> dict:
    merged = normalize_row(old_row)
    for field in FIELDNAMES:
        value = clean(new_row.get(field))
        if value:
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


def write_file(path: Path, rows: list[dict]) -> None:
    repo_root = Path(__file__).resolve().parents[5]
    candidate = path if path.is_absolute() else Path.cwd() / path
    safe_path = candidate.resolve()
    try:
        safe_path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {path}") from exc

    safe_path.parent.mkdir(parents=True, exist_ok=True)
    with safe_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalize_row(row) for row in rows)


def consolidate() -> tuple[int, int, int]:
    file_rows: dict[Path, list[dict]] = {}
    groups: dict[tuple[str, str, str, str], list[tuple[int, Path, int, dict]]] = {}
    sequence = 0

    for _league, (label, folder) in LEAGUES.items():
        if not folder.exists():
            continue
        for path in sorted(folder.glob(f"*_{label}_odds.csv")):
            with open(path, newline="", encoding="utf-8-sig") as f:
                rows = [normalize_row(row) for row in csv.DictReader(f)]
            file_rows[path] = rows
            for row_index, row in enumerate(rows):
                key = identity_key(row)
                if not all(key):
                    sequence += 1
                    continue
                groups.setdefault(key, []).append((sequence, path, row_index, row))
                sequence += 1

    # Validate all duplicate identities before writing anything.
    canonical_by_key: dict[tuple[str, str, str, str], str] = {}
    for key, copies in groups.items():
        if len(copies) > 1:
            canonical_by_key[key] = choose_canonical_id(copies)

    changed_paths: set[Path] = set()
    remove_indexes: dict[Path, set[int]] = {}
    identities = 0
    aliases = 0

    for key, canonical_id in canonical_by_key.items():
        copies = groups[key]
        identities += 1
        ordered = sorted(
            copies,
            key=lambda item: (
                parse_timestamp(item[3].get("odds_last_update")),
                nonblank_count(item[3]),
                item[0],
            ),
        )
        consolidated = normalize_row(ordered[0][3])
        for _, _, _, row in ordered[1:]:
            consolidated = merge_nonblank(consolidated, row)
        consolidated["game_id"] = canonical_id

        canonical_copies = [item for item in ordered if clean(item[3].get("game_id")) == canonical_id]
        target = canonical_copies[-1] if canonical_copies else ordered[-1]
        _, target_path, target_index, _ = target
        file_rows[target_path][target_index] = consolidated
        changed_paths.add(target_path)

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
                    f"{key[0]} {key[1]} | {row.get('home_team')} vs {row.get('away_team')} | "
                    f"{alias} -> {canonical_id}"
                )

    for path, indexes in remove_indexes.items():
        file_rows[path] = [row for idx, row in enumerate(file_rows[path]) if idx not in indexes]

    for path in sorted(changed_paths):
        write_file(path, file_rows[path])
        log(f"REWROTE {path} ({len(file_rows[path])} rows)")

    return identities, aliases, len(changed_paths)


def main() -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(
        f"=== sportsbook alias cleanup {datetime.now(timezone.utc).isoformat()} ===\n",
        encoding="utf-8",
    )
    try:
        identities, aliases, files = consolidate()
        log(f"Duplicate identities consolidated: {identities}")
        log(f"Alias IDs removed: {aliases}")
        log(f"Files rewritten: {files}")
        log("STATUS: SUCCESS")
    except Exception as exc:
        log(f"FATAL: {exc}")
        log(traceback.format_exc().rstrip())
        log("STATUS: FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
