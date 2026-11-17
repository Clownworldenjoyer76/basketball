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
with an idempotent upsert so late finals and corrected scores can be incorporated.
"""
from __future__ import annotations

import csv
import importlib.util
import re
import unicodedata
from functools import lru_cache
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


NY = ZoneInfo("America/New_York")

CORE_PATH = Path(__file__).with_name("transform_basketball_core.py")

BASE = Path("docs/win/basketball")
SEASON_CONFIG = BASE / "config/season_dates.yaml"

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


def collapse_rows(rows: list[dict]) -> list[dict]:
    """One row per date/home/away, preserving the strongest known game_id."""
    by_comp: dict[tuple[str, str, str], dict] = {}

    for raw in rows:
        row = {
            column: raw.get(column, "")
            for column in FINAL_COLUMNS
        }

        key = composite_key(row)

        # NCAAM historical DRatings rows can reverse the ESPN home/away
        # orientation. Align all paired score/spread fields before merging.
        if clean(row.get("league")).casefold() == "ncaam" and key not in by_comp:
            reversed_key = (key[0], key[2], key[1])
            if reversed_key in by_comp:
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

        if (
            old_gid.isdigit()
            and new_gid.isdigit()
            and old_gid != new_gid
        ):
            raise ValueError(
                f"Conflicting numeric game_ids for final score "
                f"{key}: {old_gid} vs {new_gid}"
            )

        if clean(old.get("league")).casefold() == "ncaam":
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
                            "NCAAM matchup-matched final has conflicting scores: "
                            f"key={key}, field={field}, "
                            f"stored={previous}, incoming={incoming}"
                        )
            # ESPN's identified final is authoritative. A DRatings row with
            # no game ID must never overwrite it or create an alias duplicate.
            if old_gid.isdigit() and not new_gid:
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
            clean(old.get("league")).casefold() in ("nba", "ncaam")
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

        for date_val, group in completed.groupby("game_date"):
            path = (
                Path(cfg["final_scores_dir"])
                / f"{date_val}_final_scores_{league_label}.csv"
            )

            incoming = []

            for _, row in group.iterrows():
                try:
                    away_score = int(
                        float(row["score1"])
                    )

                    home_score = int(
                        float(row["score2"])
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

                except (ValueError, TypeError):
                    away_score = ""
                    home_score = ""
                    total = ""
                    away_spread = ""
                    home_spread = ""

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

                existing = old.to_dict(
                    "records"
                )

            before = len(existing)

            merged = collapse_rows(
                existing + incoming
            )

            out = pd.DataFrame(
                merged,
                columns=FINAL_COLUMNS,
            )

            core.save(
                out,
                str(path),
                files_written,
                league_key,
            )

            stats["final_score_files_written"] += 1
            stats["final_score_rows_written"] += len(out)

            core.log(
                league_key,
                f"UPSERT final scores: {path} "
                f"existing_rows={before} "
                f"incoming_rows={len(incoming)} "
                f"result_rows={len(out)}",
            )

    return process_final_scores


def main() -> None:
    core = load_core()

    core.process_final_scores = make_process_final_scores(
        core
    )

    now = datetime.now(NY)

    force_all = truthy(
        os.getenv(
            "BASKETBALL_FORCE_ALL_LEAGUES"
        )
    )

    try:
        season_config = load_season_config()

    except Exception as exc:
        print(
            f"SEASON CONFIG FAILED: {exc}"
        )
        raise SystemExit(1) from exc

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

    had_errors = False

    for league_key in active:
        core.process_league(
            league_key
        )

        text = Path(
            core.LEAGUE_CONFIG[
                league_key
            ]["log_file"]
        ).read_text(
            encoding="utf-8",
            errors="replace",
        )

        if (
            "STATUS: FAILED" in text
            or "STATUS: PARTIAL" in text
        ):
            had_errors = True

    skipped = sorted(
        set(leagues)
        - set(active)
    )

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

    if had_errors:
        raise SystemExit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
