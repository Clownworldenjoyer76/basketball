#!/usr/bin/env python3
"""Season-aware transform launcher with final-score upsert semantics.

The parser/prediction implementation remains in transform_basketball_core.py.

Operational season dates are loaded from:
    docs/win/basketball/config/season_dates.yaml

Set BASKETBALL_FORCE_ALL_LEAGUES=1 (or true/yes/on) to run every
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

This launcher replaces the old "skip if date file exists" final-score behavior
with an idempotent upsert so late finals can be incorporated. A matched final whose
score differs from the stored score raises an error.
"""
from __future__ import annotations

import csv
import importlib.util
import re
import traceback
import unicodedata
from functools import lru_cache
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


NY = ZoneInfo("America/New_York")

CORE_PATH = Path(__file__).with_name("transform_basketball_core.py")

BASE = Path("docs/win/basketball")
SEASON_CONFIG = BASE / "config/season_dates.yaml"
ERROR_DIR = BASE / "errors/00_intake"
LOG_FILE = ERROR_DIR / "transform_basketball.txt"

SUPPORTED_LEAGUES = ("nba", "ncaam", "wnba")

FINAL_COLUMNS = [
    "sport",
    "league",
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "total",
    "home_spread",
    "away_spread",
]


def log(message: str) -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{datetime.now(timezone.utc).isoformat()} | {message}\n"
        )


def truthy(value: str | None) -> bool:
    normalized = str(
        value or ""
    ).strip().casefold()

    return normalized in {
        "1",
        "true",
        "yes",
        "on",
    }


def _season_field(
    row: dict,
    league: str,
    field: str,
) -> int:
    if field not in row:
        raise ValueError(
            f"Missing {league}.{field} "
            f"in {SEASON_CONFIG}"
        )

    try:
        return int(
            row[field]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {league}.{field}: "
            f"{row[field]!r}"
        ) from exc


def load_season_config() -> dict[str, dict[str, int]]:
    if not SEASON_CONFIG.exists():
        raise FileNotFoundError(
            f"Season config not found: "
            f"{SEASON_CONFIG}"
        )

    raw = (
        yaml.safe_load(
            SEASON_CONFIG.read_text(
                encoding="utf-8"
            )
        )
        or {}
    )

    if not isinstance(
        raw,
        dict,
    ):
        raise ValueError(
            f"{SEASON_CONFIG} must contain "
            "a top-level mapping"
        )

    fields = (
        "start_month",
        "start_day",
        "end_month",
        "end_day",
    )

    config: dict[
        str,
        dict[str, int],
    ] = {}

    for league in SUPPORTED_LEAGUES:
        row = raw.get(
            league
        )

        if not isinstance(
            row,
            dict,
        ):
            raise ValueError(
                "Missing season configuration "
                f"for league={league}"
            )

        values = {
            field: _season_field(
                row,
                league,
                field,
            )
            for field in fields
        }

        for label in (
            "start",
            "end",
        ):
            month = values[
                f"{label}_month"
            ]
            day = values[
                f"{label}_day"
            ]

            try:
                datetime(
                    2000,
                    month,
                    day,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid {league}.{label}: "
                    f"month={month}, day={day}"
                ) from exc

        config[
            league
        ] = values

    return config

def in_season(
    league: str,
    now: datetime,
    season_config: dict[str, dict[str, int]],
) -> bool:
    """Return True when the current date is inside the league's season."""
    league = league.strip().lower()

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

    # Season contained within one calendar year.
    # Example: May 1 through October 31.
    if start_mmdd <= end_mmdd:
        return start_mmdd <= current_mmdd <= end_mmdd

    # Season crosses New Year.
    # Example: October 15 through July 1.
    return current_mmdd >= start_mmdd or current_mmdd <= end_mmdd


def load_core():
    spec = importlib.util.spec_from_file_location(
        "transform_basketball_core",
        CORE_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load transform core: {CORE_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def clean(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""

    return str(v).strip()


# NBA_ALIAS_FINAL_SCORE_UPSERT_V1: match the same team across provider spellings.
@lru_cache(maxsize=1)
def _nba_final_aliases() -> dict[str, str]:
    path = BASE / "maps/team_map_nba.csv"
    result: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"league", "alias", "canonical_team"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"NBA team map has invalid columns: {path}")
        for lineno, entry in enumerate(reader, start=2):
            if clean(entry.get("league")).casefold() != "nba":
                continue
            alias = re.sub(r"[^a-z0-9]", "", clean(entry.get("alias")).casefold())
            canonical = re.sub(r"[^a-z0-9]", "", clean(entry.get("canonical_team")).casefold())
            if not alias or not canonical:
                raise ValueError(f"Missing team identity in NBA team map: {path}:{lineno}")
            for key in (alias, canonical):
                prior = result.get(key)
                if prior is not None and prior != canonical:
                    raise ValueError(f"Conflicting NBA alias {key!r} in {path}:{lineno}")
                result[key] = canonical
    if not result:
        raise ValueError(f"NBA team map contains no identities: {path}")
    return result


@lru_cache(maxsize=1)
def _wnba_final_aliases() -> dict[str, str]:
    path = BASE / "maps/team_map_wnba.csv"
    result: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"league", "alias", "canonical_team"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"WNBA team map has invalid columns: {path}")
        for lineno, entry in enumerate(reader, start=2):
            if clean(entry.get("league")).casefold() != "wnba":
                continue
            alias = re.sub(r"[^a-z0-9]", "", clean(entry.get("alias")).casefold())
            canonical = re.sub(r"[^a-z0-9]", "", clean(entry.get("canonical_team")).casefold())
            if not alias or not canonical:
                raise ValueError(f"Missing team identity in WNBA team map: {path}:{lineno}")
            for key in (alias, canonical):
                prior = result.get(key)
                if prior is not None and prior != canonical:
                    raise ValueError(f"Conflicting WNBA alias {key!r} in {path}:{lineno}")
                result[key] = canonical
    if not result:
        raise ValueError(f"WNBA team map contains no identities: {path}")
    return result


def _normalized_ncaam_final_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", clean(value)).casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", text))


@lru_cache(maxsize=1)
def _ncaam_final_aliases() -> dict[str, str]:
    """Use the validated NCAAM team identities when merging final scores."""
    path = BASE / "maps/team_map_ncaam.csv"
    result: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"league", "alias", "canonical_team"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"NCAAM team map has invalid columns: {path}")
        for lineno, entry in enumerate(reader, start=2):
            if clean(entry.get("league")).casefold() != "ncaam":
                continue
            alias = _normalized_ncaam_final_name(entry.get("alias"))
            canonical = _normalized_ncaam_final_name(entry.get("canonical_team"))
            if not alias or not canonical:
                raise ValueError(f"Missing team identity in NCAAM map: {path}:{lineno}")
            for key in (alias, canonical):
                prior = result.get(key)
                if prior is not None and prior != canonical:
                    raise ValueError(f"Conflicting NCAAM alias {key!r} in {path}:{lineno}")
                result[key] = canonical
    if not result:
        raise ValueError(f"NCAAM team map contains no identities: {path}")
    return result


def _team_key_for_final(row: dict, field: str) -> str:
    name = clean(row.get(field)).casefold()
    league = clean(row.get("league")).casefold()
    if league == "ncaam":
        canonical_key = _normalized_ncaam_final_name(name)
        return _ncaam_final_aliases().get(canonical_key, canonical_key)
    if league == "wnba":
        compact = re.sub(r"[^a-z0-9]", "", name)
        return _wnba_final_aliases().get(compact, compact)
    if league != "nba":
        return name
    compact = re.sub(r"[^a-z0-9]", "", name)
    return _nba_final_aliases().get(compact, compact)


def composite_key(row: dict) -> tuple[str, str, str]:
    return (
        clean(row.get("game_date")),
        _team_key_for_final(row, "home_team"),
        _team_key_for_final(row, "away_team"),
    )


def id_rank(game_id: str) -> int:
    gid = clean(game_id)

    if not gid:
        return 0

    if gid.isdigit():
        return 2

    return 1


def collapse_rows(
    rows: list[dict],
    merge_log: list[str] | None = None,
) -> list[dict]:
    """One row per date/home/away, preserving the strongest known game_id."""
    by_comp: dict[tuple[str, str, str], dict] = {}

    def note(message: str) -> None:
        if merge_log is not None:
            merge_log.append(message)

    for raw in rows:
        row = {
            column: raw.get(column, "")
            for column in FINAL_COLUMNS
        }

        key = composite_key(row)

        # Historical DRatings rows can reverse the ESPN home/away
        # orientation. Align all paired score/spread fields before merging.
        if (
            clean(row.get("league")).casefold() in ("ncaam", "nba", "wnba")
            and key not in by_comp
        ):
            reversed_key = (key[0], key[2], key[1])
            if reversed_key in by_comp:
                note(
                    "REALIGNED reversed home/away | "
                    f"league={clean(row.get('league'))} key={reversed_key}"
                )
                for left, right in (
                    ("home_team", "away_team"),
                    ("home_score", "away_score"),
                    ("home_spread", "away_spread"),
                ):
                    row[left], row[right] = row[right], row[left]
                key = reversed_key

        if key not in by_comp:
            by_comp[key] = row
            continue

        old = by_comp[key]

        old_gid = clean(
            old.get("game_id")
        )

        new_gid = clean(
            row.get("game_id")
        )

        note(
            "MERGE final score | "
            f"league={clean(row.get('league'))} key={key} "
            f"old_game_id={old_gid!r} new_game_id={new_gid!r}"
        )

        if (
            old_gid.isdigit()
            and new_gid.isdigit()
            and old_gid != new_gid
        ):
            raise ValueError(
                f"Conflicting numeric game_ids for final score "
                f"{key}: {old_gid} vs {new_gid}"
            )

        for field in ("home_score", "away_score"):
            previous = clean(old.get(field))
            incoming = clean(row.get(field))
            if previous and incoming:
                try:
                    same = float(previous) == float(incoming)
                except ValueError:
                    same = previous == incoming
                if not same:
                    raise ValueError(
                        f"{clean(old.get('league')).upper()} matchup-matched "
                        "final has conflicting scores: "
                        f"key={key}, field={field}, "
                        f"stored={previous}, incoming={incoming}"
                    )

        if clean(old.get("league")).casefold() == "ncaam":
            # ESPN's identified final is authoritative. A DRatings row with
            # no game ID must never overwrite it or create an alias duplicate.
            if old_gid.isdigit() and not new_gid:
                note(f"MERGE skipped incoming row without game_id | key={key}")
                continue

        # New score values are authoritative; preserve/prefer canonical ID.
        chosen_gid = (
            new_gid
            if id_rank(new_gid) > id_rank(old_gid)
            else old_gid
        )

        # Alias-based duplicates must not create a second game or silently
        # replace an official score with an inconsistent provider score.
        name_alias_duplicate = (
            clean(old.get("league")).casefold() in ("nba", "ncaam", "wnba")
            and (
                clean(old.get("home_team")).casefold()
                != clean(row.get("home_team")).casefold()
                or clean(old.get("away_team")).casefold()
                != clean(row.get("away_team")).casefold()
            )
        )
        if name_alias_duplicate:
            for score_field in ("home_score", "away_score"):
                old_score = clean(old.get(score_field))
                new_score = clean(row.get(score_field))
                if old_score and new_score:
                    try:
                        same_score = float(old_score) == float(new_score)
                    except ValueError:
                        same_score = old_score == new_score
                    if not same_score:
                        raise ValueError(
                            "NBA alias-matched final has conflicting scores: "
                            f"key={key}, field={score_field}, "
                            f"stored={old_score}, incoming={new_score}"
                        )

        merged = dict(old)

        for col in FINAL_COLUMNS:
            value = row.get(col, "")
            # Preserve the original, canonical result-file team labels on an
            # alias match; DRatings may use a different spelling for one team.
            if name_alias_duplicate and col in ("home_team", "away_team") and clean(old.get(col)):
                continue
            if clean(value) != "":
                merged[col] = value

        merged["game_id"] = chosen_gid
        by_comp[key] = merged

    return list(
        by_comp.values()
    )


def make_process_final_scores(core):
    def process_final_scores(
        df,
        files_written,
        league_key,
        stats,
    ):
        cfg = core.LEAGUE_CONFIG[league_key]
        league_label = cfg["league_label"]

        mask = (
            df["score1"].notna()
            & (
                df["score1"]
                .astype(str)
                .str.strip()
                != ""
            )
        )

        completed = df[mask].copy()

        stats["completed_games"] += len(completed)

        if completed.empty:
            core.log(
                league_key,
                f"No completed {league_label} games found in this file.",
            )
            return

        pending = []

        for date_val, group in completed.groupby("game_date"):
            path = (
                Path(cfg["final_scores_dir"])
                / f"{date_val}_final_scores_{league_label}.csv"
            )

            incoming = []

            for _, row in group.iterrows():
                context = (
                    f"game_date={date_val} "
                    f"home={row['team2']!r} away={row['team1']!r}"
                )

                away_score = int(
                    core.require_number(
                        row["score1"], "away score", context
                    )
                )

                home_score = int(
                    core.require_number(
                        row["score2"], "home score", context
                    )
                )

                total = (
                    away_score
                    + home_score
                )

                away_spread = (
                    away_score
                    - home_score
                )

                home_spread = (
                    home_score
                    - away_score
                )

                incoming.append({
                    "sport": "Basketball",
                    "league": league_label,
                    "game_id": "",
                    "game_date": date_val,
                    "home_team": row["team2"],
                    "away_team": row["team1"],
                    "home_score": home_score,
                    "away_score": away_score,
                    "total": total,
                    "home_spread": home_spread,
                    "away_spread": away_spread,
                })

            existing: list[dict] = []

            if path.exists():
                old = pd.read_csv(
                    path,
                    dtype=str,
                    keep_default_na=False,
                )

                dropped = [
                    column
                    for column in old.columns
                    if column not in FINAL_COLUMNS
                ]

                if dropped:
                    core.log(
                        league_key,
                        f"WARNING dropped columns not in FINAL_COLUMNS: "
                        f"{path} columns={dropped}",
                    )

                existing = old.to_dict(
                    "records"
                )

            before = len(existing)

            merge_log: list[str] = []

            merged = collapse_rows(
                existing + incoming,
                merge_log,
            )

            out = pd.DataFrame(
                merged,
                columns=FINAL_COLUMNS,
            )

            pending.append(
                (path, out, before, len(incoming), merge_log)
            )

        for path, out, before, incoming_rows, merge_log in pending:
            core.save(
                out,
                str(path),
                files_written,
                league_key,
            )

            stats["final_score_files_written"] += 1
            stats["final_score_rows_written"] += len(out)

            for message in merge_log:
                core.log(
                    league_key,
                    f"{message} | file={path}",
                )

            core.log(
                league_key,
                f"UPSERT final scores: {path} "
                f"existing_rows={before} "
                f"incoming_rows={incoming_rows} "
                f"result_rows={len(out)}",
            )

    return process_final_scores


def league_status_text(core, league_key: str) -> tuple[str, str]:
    text = Path(
        core.LEAGUE_CONFIG[
            league_key
        ]["log_file"]
    ).read_text(
        encoding="utf-8",
        errors="replace",
    )

    found = [
        line.split("|", 1)[-1].strip()
        for line in text.splitlines()
        if "STATUS:" in line
    ]

    return text, (found[-1] if found else "STATUS: (none found)")


def main() -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)

    LOG_FILE.write_text(
        "=== TRANSFORM BASKETBALL LAUNCHER "
        f"{datetime.now(timezone.utc).isoformat()} ===\n",
        encoding="utf-8",
    )

    try:
        core = load_core()

    except Exception as exc:
        log(f"FATAL: unable to load transform core: {exc}")
        log(traceback.format_exc().rstrip())
        log("STATUS: FAILED")
        print(f"TRANSFORM CORE LOAD FAILED: {exc}")
        raise SystemExit(1) from exc

    core.process_final_scores = make_process_final_scores(
        core
    )

    now = datetime.now(NY)

    force_all = truthy(
        os.getenv(
            "BASKETBALL_FORCE_ALL_LEAGUES"
        )
    )

    log(
        f"RUN | date_ny={now.strftime('%Y_%m_%d')} "
        f"force_all={force_all}"
    )

    try:
        season_config = load_season_config()

    except Exception as exc:
        log(f"SEASON CONFIG FAILED: {exc}")
        log(traceback.format_exc().rstrip())
        log("STATUS: FAILED")
        print(
            f"SEASON CONFIG FAILED: {exc}"
        )
        raise SystemExit(1) from exc

    log(f"SEASON CONFIG VALID | config={SEASON_CONFIG}")

    leagues = list(
        core.LEAGUE_CONFIG
    )

    if force_all:
        active = leagues
    else:
        active = [
            league
            for league in leagues
            if in_season(
                league,
                now,
                season_config,
            )
        ]

    skipped = sorted(
        set(leagues)
        - set(active)
    )

    log(f"LEAGUES | active={active} skipped={skipped}")

    error_leagues = []

    for league_key in active:
        try:
            core.process_league(
                league_key
            )

        except Exception as exc:
            error_leagues.append(league_key)
            log(f"LEAGUE FAILED | league={league_key} error={exc}")
            log(traceback.format_exc().rstrip())
            print(f"{league_key} FAILED: {exc}")

        text, status_text = league_status_text(
            core, league_key
        )

        log(f"LEAGUE RESULT | league={league_key} {status_text}")

        if (
            "STATUS: FAILED" in text
            or "STATUS: PARTIAL" in text
        ):
            if league_key not in error_leagues:
                error_leagues.append(league_key)

    for league_key in skipped:
        core.init_log(
            league_key
        )

        core.log(
            league_key,
            "SEASON GATE: "
            f"skipped on {now.strftime('%Y_%m_%d')} "
            "America/New_York; "
            f"config={SEASON_CONFIG}; "
            "set BASKETBALL_FORCE_ALL_LEAGUES=1 "
            "to override",
        )

        core.write_summary(
            league_key,
            [],
            {
                "input_files_found": 0,
                "input_files_processed": 0,
                "games_loaded": 0,
                "upcoming_games": 0,
                "completed_games": 0,
                "prediction_files_written": 0,
                "prediction_rows_written": 0,
                "final_score_files_written": 0,
                "final_score_rows_written": 0,
                "file_errors": 0,
            },
            "SUCCESS (offseason skipped)",
        )

        log(f"LEAGUE SKIPPED | league={league_key}")

    if error_leagues:
        log(f"STATUS: FAILED | leagues_with_errors={error_leagues}")
        raise SystemExit(1)

    log(f"STATUS: SUCCESS | active={active} skipped={skipped}")

    print("\nDone.")


if __name__ == "__main__":
    main()
