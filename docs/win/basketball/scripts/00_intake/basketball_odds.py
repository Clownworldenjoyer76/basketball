#!/usr/bin/env python3
"""Season-aware launcher for the ESPN basketball odds collector.

The implementation remains in basketball_odds_core.py.

Operational season dates are loaded from:
    docs/win/basketball/config/season_dates.yaml

Normal production runs only query leagues that are in season for the current
New York date. Set BASKETBALL_FORCE_ALL_LEAGUES=1 (or true/yes/on) to run every
league explicitly.

Expected season_dates.yaml format:

nba:
  start_month: 10
  start_day: 15
  end_month: 7
  end_day: 1

ncaam:
  start_month: 10
  start_day: 31
  end_month: 7
  end_day: 1

wnba:
  start_month: 5
  start_day: 1
  end_month: 10
  end_day: 31

Both normal calendar-year windows and cross-year windows are supported.

After collection, raw sportsbook files are normalized to one row per (date,
home team, away team) matchup. This cleanup also runs when the collector
reported errors. Numeric ESPN game IDs outrank legacy IDs; legacy alias rows are
merged into the canonical row and removed. Conflicting numeric IDs for the same
matchup are a hard failure. Rows with unresolved placeholder teams (blank, TBD,
to be determined, unknown, or a name containing "/") are DELETED from the raw
sportsbook files. All files are checked before any file is rewritten, and the
alias report is written last.

The launcher writes the single final STATUS line: SUCCESS, COMPLETED WITH
ERRORS (exit code 1), or FAILED (exit code 1).
"""
from __future__ import annotations

import sys
import csv
import importlib.util
import os
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from basketball_shared import resolve_repository_path


NY = ZoneInfo("America/New_York")

CORE_PATH = Path(__file__).with_name("basketball_odds_core.py")

BASE = Path("docs/win/basketball")
SPORTSBOOK_BASE = BASE / "00_intake/sportsbook"
ERROR_DIR = BASE / "errors/00_intake"

SEASON_CONFIG = BASE / "config/season_dates.yaml"
ALIAS_REPORT = ERROR_DIR / "basketball_sportsbook_alias_cleanup.csv"

LEAGUE_LABELS = {
    "nba": "NBA",
    "ncaam": "NCAAM",
    "wnba": "WNBA",
}

ALIAS_FIELDS = [
    "league",
    "source_file",
    "game_date",
    "home_team",
    "away_team",
    "alias_game_id",
    "canonical_game_id",
]

UNRESOLVED_TEAM_NAMES = {
    "tbd",
    "to be determined",
    "unknown",
}


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def validate_month_day(
    league: str,
    label: str,
    month: int,
    day: int,
) -> None:
    """Validate a month/day pair using a leap year."""
    try:
        datetime(2000, month, day)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {league}.{label}: month={month}, day={day}"
        ) from exc


def load_season_config() -> dict[str, dict[str, int]]:
    """Load and validate operational season dates."""
    if not SEASON_CONFIG.exists():
        raise FileNotFoundError(
            f"Season config not found: {SEASON_CONFIG}"
        )

    with open(SEASON_CONFIG, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(
            f"{SEASON_CONFIG} must contain a top-level mapping"
        )

    required_fields = (
        "start_month",
        "start_day",
        "end_month",
        "end_day",
    )

    config: dict[str, dict[str, int]] = {}

    for league in LEAGUE_LABELS:
        row = raw.get(league)

        if not isinstance(row, dict):
            raise ValueError(
                f"Missing season configuration for league={league}"
            )

        values: dict[str, int] = {}

        for field in required_fields:
            if field not in row:
                raise ValueError(
                    f"Missing {league}.{field} in {SEASON_CONFIG}"
                )

            try:
                values[field] = int(row[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {league}.{field}: {row[field]!r}"
                ) from exc

        validate_month_day(
            league,
            "start",
            values["start_month"],
            values["start_day"],
        )
        validate_month_day(
            league,
            "end",
            values["end_month"],
            values["end_day"],
        )

        config[league] = values

    return config


def in_season(
    league: str,
    now: datetime,
    season_config: dict[str, dict[str, int]],
) -> bool:
    """Return True when the current date is inside the league's season."""
    league = league.lower()

    if league not in season_config:
        raise KeyError(
            f"No season configuration found for league={league}"
        )

    cfg = season_config[league]

    current_mmdd = (now.month, now.day)
    start_mmdd = (
        cfg["start_month"],
        cfg["start_day"],
    )
    end_mmdd = (
        cfg["end_month"],
        cfg["end_day"],
    )

    # Normal season window contained within one calendar year.
    # Example: May 1 through October 31.
    if start_mmdd <= end_mmdd:
        return start_mmdd <= current_mmdd <= end_mmdd

    # Cross-year season window.
    # Example: October 15 through July 1.
    return current_mmdd >= start_mmdd or current_mmdd <= end_mmdd


def load_core():
    spec = importlib.util.spec_from_file_location(
        "basketball_odds_core",
        CORE_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load odds core: {CORE_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean(value) -> str:
    return "" if value is None else str(value).strip()


def unresolved_team(value) -> bool:
    team = clean(value).casefold()

    if not team or team in UNRESOLVED_TEAM_NAMES:
        return True

    # ESPN playoff placeholders can expose unresolved matchup labels such as
    # "Dream/Mystics" before the participating team is known.
    return "/" in team


def matchup_key(row: dict) -> tuple[str, str, str]:
    game_date = clean(row.get("game_date"))
    home_team = clean(row.get("home_team"))
    away_team = clean(row.get("away_team"))

    if unresolved_team(home_team) or unresolved_team(away_team):
        return game_date, "", ""

    return (
        game_date,
        home_team.casefold(),
        away_team.casefold(),
    )


def id_rank(game_id: str) -> int:
    game_id = clean(game_id)

    if game_id.isdigit():
        return 2

    if game_id:
        return 1

    return 0


def row_score(row: dict) -> tuple[int, str]:
    # Prefer the row carrying the most usable data, then the latest
    # odds timestamp.
    populated = sum(
        1
        for key, value in row.items()
        if key != "game_id" and clean(value)
    )

    return populated, clean(row.get("odds_last_update"))


def atomic_write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict],
) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")

    try:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _partition_sportsbook_rows(rows):
    grouped: dict[
        tuple[str, str, str],
        list[tuple[int, dict]],
    ] = {}
    passthrough: list[tuple[int, dict]] = []
    placeholder_removed = 0

    for index, row in enumerate(rows):
        home_team = clean(row.get("home_team"))
        away_team = clean(row.get("away_team"))
        if unresolved_team(home_team) or unresolved_team(away_team):
            placeholder_removed += 1
            continue

        key = matchup_key(row)
        if all(key):
            grouped.setdefault(key, []).append((index, row))
        else:
            passthrough.append((index, row))

    return grouped, passthrough, placeholder_removed


def _collapse_sportsbook_group(path, league, key, items, fieldnames):
    group_rows = [row for _, row in items]
    numeric_ids = sorted({
        clean(row.get("game_id"))
        for row in group_rows
        if clean(row.get("game_id")).isdigit()
    })

    if len(numeric_ids) > 1:
        raise RuntimeError(
            f"Conflicting numeric sportsbook game IDs for "
            f"{league.upper()} {key[0]} "
            f"{group_rows[0].get('away_team')} at "
            f"{group_rows[0].get('home_team')}: "
            + ", ".join(numeric_ids)
        )

    canonical_id = (
        numeric_ids[0]
        if numeric_ids
        else max(
            (clean(row.get("game_id")) for row in group_rows),
            key=lambda gid: (id_rank(gid), gid),
            default="",
        )
    )

    _, best = max(
        items,
        key=lambda item: (
            id_rank(clean(item[1].get("game_id"))),
            row_score(item[1]),
        ),
    )
    merged = dict(best)
    merged["game_id"] = canonical_id

    for _, row in sorted(
        items,
        key=lambda item: row_score(item[1]),
        reverse=True,
    ):
        for field in fieldnames:
            if not clean(merged.get(field)) and clean(row.get(field)):
                merged[field] = row[field]

    merged["game_id"] = canonical_id

    aliases = []
    if len(items) > 1:
        for _, row in items:
            alias_id = clean(row.get("game_id"))
            if alias_id != canonical_id:
                aliases.append({
                    "league": league,
                    "source_file": str(path),
                    "game_date": clean(row.get("game_date")),
                    "home_team": clean(row.get("home_team")),
                    "away_team": clean(row.get("away_team")),
                    "alias_game_id": alias_id,
                    "canonical_game_id": canonical_id,
                })

    return (
        (min(index for index, _ in items), merged),
        max(len(items) - 1, 0),
        aliases,
    )


def plan_collapse_file(
    path: Path,
    league: str,
) -> dict | None:
    """Work out the cleaned rows for one file. Nothing is written here."""
    safe_path = resolve_repository_path(
        path,
        strict=True,
    )

    with safe_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows or not fieldnames:
        return None

    grouped, passthrough, placeholder_removed = _partition_sportsbook_rows(rows)
    output: list[tuple[int, dict]] = list(passthrough)
    aliases: list[dict] = []
    removed = 0

    for key, items in grouped.items():
        output_row, group_removed, group_aliases = _collapse_sportsbook_group(
            path,
            league,
            key,
            items,
            fieldnames,
        )
        output.append(output_row)
        removed += group_removed
        aliases.extend(group_aliases)

    output.sort(key=lambda item: item[0])
    changed = bool(removed or placeholder_removed)

    return {
        "league": league,
        "source": path,
        "path": safe_path,
        "fieldnames": fieldnames,
        "rows": [row for _, row in output] if changed else None,
        "removed": removed,
        "placeholder_removed": placeholder_removed,
        "aliases": aliases,
    }


def cleanup_sportsbook_aliases(core) -> tuple[int, int]:
    ERROR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Step 1: check every file. Nothing is written in this step, so an error
    # (such as conflicting numeric IDs) stops the run before any file changes.
    plans: list[dict] = []
    alias_rows: list[dict] = []

    for league in LEAGUE_LABELS:
        folder = SPORTSBOOK_BASE / league

        if not folder.exists():
            continue

        for path in sorted(folder.glob("*.csv")):
            plan = plan_collapse_file(
                path,
                league,
            )

            if plan is None:
                continue

            plans.append(plan)
            alias_rows.extend(plan["aliases"])

    # Step 2: rewrite the changed files.
    changed_files = 0
    removed_alias_rows = 0
    removed_placeholder_rows = 0

    for plan in plans:
        if plan["rows"] is None:
            continue

        atomic_write_csv(
            plan["path"],
            plan["fieldnames"],
            plan["rows"],
        )

        changed_files += 1
        removed_alias_rows += plan["removed"]
        removed_placeholder_rows += plan["placeholder_removed"]

        if plan["placeholder_removed"]:
            core.log(
                "SPORTSBOOK PLACEHOLDER REMOVAL | "
                f"league={plan['league'].upper()} | "
                f"file={plan['source']} | "
                f"rows={plan['placeholder_removed']}"
            )

    # Step 3: the alias report is written last.
    atomic_write_csv(
        ALIAS_REPORT,
        ALIAS_FIELDS,
        alias_rows,
    )

    total_removed_rows = (
        removed_alias_rows
        + removed_placeholder_rows
    )

    core.log(
        "SPORTSBOOK ALIAS CLEANUP | "
        f"changed_files={changed_files} | "
        f"removed_alias_rows={removed_alias_rows} | "
        f"removed_placeholder_rows={removed_placeholder_rows} | "
        f"aliases={len(alias_rows)} | "
        f"report={ALIAS_REPORT}"
    )

    return changed_files, total_removed_rows


def main() -> None:
    core = load_core()
    now = datetime.now(NY)

    force_all = truthy(
        os.getenv("BASKETBALL_FORCE_ALL_LEAGUES")
    )

    try:
        season_config = load_season_config()
    except Exception as exc:
        core.log(
            f"SEASON CONFIG FAILED: {exc}"
        )
        core.log("STATUS: FAILED")
        raise SystemExit(1) from exc

    original = dict(core.LEAGUES)

    if force_all:
        active = original
    else:
        active = {
            league: value
            for league, value in original.items()
            if in_season(
                league,
                now,
                season_config,
            )
        }

    skipped = sorted(
        set(original) - set(active)
    )

    core.LEAGUES = active

    core.log(
        "SEASON GATE | "
        f"date={now.strftime('%Y_%m_%d')} | "
        f"config={SEASON_CONFIG} | "
        f"active={','.join(active) or 'none'} | "
        f"skipped={','.join(skipped) or 'none'} | "
        f"force_all={int(force_all)}"
    )

    errors = 0

    if active:
        try:
            result = core.main()
        except Exception as exc:
            core.log(
                f"COLLECTOR CRASHED: {exc}"
            )
            core.log(
                traceback.format_exc().rstrip()
            )
            core.log("STATUS: FAILED")
            raise SystemExit(1) from exc

        errors = int(result["errors"])

    else:
        core.log(
            "SEASON GATE: no leagues in season; "
            "collector skipped"
        )

    # Cleanup runs even when the collector counted errors.
    try:
        cleanup_sportsbook_aliases(core)

    except Exception as exc:
        core.log(
            f"SPORTSBOOK ALIAS CLEANUP FAILED: {exc}"
        )
        core.log(
            traceback.format_exc().rstrip()
        )
        core.log("STATUS: FAILED")
        raise SystemExit(1) from exc

    if not active:
        core.log(
            "STATUS: SUCCESS (no leagues in season)"
        )
        return

    core.log(
        core.status_line(errors)
    )

    if errors:
        raise SystemExit(1)

    print("basketball_odds complete.")


if __name__ == "__main__":
    main()
