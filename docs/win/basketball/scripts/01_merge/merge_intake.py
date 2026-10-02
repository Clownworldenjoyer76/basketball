#!/usr/bin/env python3
# docs/win/basketball/scripts/01_merge/merge_intake.py
"""Merge cleaned predictions and sportsbook data with game_id-first identity.

Full historical rebuild behavior is retained. Matching prefers canonical game_id
and uses date/home/away only as a controlled unique fallback. Current in-season
coverage gaps are fatal so a partially merged live slate cannot pass green.

Operational season dates are loaded from:
    docs/win/basketball/config/season_dates.yaml
"""
from __future__ import annotations

import csv
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from basketball_shared import resolve_repository_path


LEAGUES = ["nba", "ncaam", "wnba"]

BASE = Path("docs/win/basketball")
SEASON_CONFIG = BASE / "config/season_dates.yaml"

INTAKE_DIR = BASE / "00_intake"
PREDICTIONS_DIR = INTAKE_DIR / "predictions" / "predictions_cleaned"
SPORTSBOOK_DIR = INTAKE_DIR / "sportsbook" / "sportsbook_cleaned"
MERGE_DIR = BASE / "01_merge"
ERROR_DIR = BASE / "errors/01_merge"

ERROR_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "merge_intake.txt"
COVERAGE_FILE = ERROR_DIR / "merge_coverage.csv"

NY = ZoneInfo("America/New_York")

PROVENANCE_FIELDS = [
    "bias_applied",
    "margin_bias",
    "total_bias",
    "model_source",
    "model_version",
    "feature_version",
    "ensemble_version",
]

SPORTSBOOK_PROVENANCE_FIELDS = [
    "sportsbook_provider",
    "scraped_at_utc",
    "provider_updated_at_utc",
]

MONEYLINE_FIELDS = [
    "sport",
    "league",
    "game_id",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_prob",
    "away_prob",
    "away_projected_points",
    "home_projected_points",
    "total_projected_points",
    *PROVENANCE_FIELDS,
    *SPORTSBOOK_PROVENANCE_FIELDS,
    "total",
    "home_dk_moneyline_american",
    "away_dk_moneyline_american",
    "home_dk_moneyline_decimal",
    "away_dk_moneyline_decimal",
]

SPREAD_FIELDS = [
    "sport",
    "league",
    "game_id",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_prob",
    "away_prob",
    "away_projected_points",
    "home_projected_points",
    "total_projected_points",
    *PROVENANCE_FIELDS,
    *SPORTSBOOK_PROVENANCE_FIELDS,
    "total",
    "home_spread",
    "away_spread",
    "home_dk_spread_american",
    "away_dk_spread_american",
    "home_dk_spread_decimal",
    "away_dk_spread_decimal",
]

TOTAL_FIELDS = [
    "sport",
    "league",
    "game_id",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_prob",
    "away_prob",
    "away_projected_points",
    "home_projected_points",
    "total_projected_points",
    *PROVENANCE_FIELDS,
    *SPORTSBOOK_PROVENANCE_FIELDS,
    "total",
    "dk_total_over_american",
    "dk_total_under_american",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
]

COVERAGE_FIELDS = [
    "league",
    "game_date",
    "prediction_rows",
    "sportsbook_rows",
    "matched_rows",
    "missing_matches",
    "match_by_game_id",
    "match_by_composite",
    "identity_mismatches",
    "coverage_pct",
    "current_in_season",
    "status",
]


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


def clean(v) -> str:
    return "" if v is None else str(v).strip()


def comp_key(r: dict) -> tuple[str, str, str]:
    fields = (
        "game_date",
        "home_team",
        "away_team",
    )

    (
        game_date,
        home_team,
        away_team,
    ) = (
        clean(
            r.get(field)
        )
        for field in fields
    )

    return (
        game_date,
        home_team.casefold(),
        away_team.casefold(),
    )


def id_rank(game_id: str) -> int:
    gid = clean(game_id)

    if not gid:
        return 0

    return 2 if gid.isdigit() else 1


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def season_field_value(
    league: str,
    row: dict,
    field: str,
) -> int:
    if field not in row:
        raise ValueError(
            f"Missing {league}.{field} "
            f"in {SEASON_CONFIG}"
        )

    value = row[
        field
    ]

    try:
        return int(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"Invalid {league}.{field}: "
            f"{value!r}"
        ) from exc


def validate_season_boundary(
    league: str,
    label: str,
    values: dict[str, int],
) -> None:
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


def load_season_config() -> dict[str, dict[str, int]]:
    if not SEASON_CONFIG.is_file():
        raise FileNotFoundError(
            f"Season config not found: "
            f"{SEASON_CONFIG}"
        )

    with SEASON_CONFIG.open(
        "r",
        encoding="utf-8",
    ) as handle:
        raw = (
            yaml.safe_load(
                handle
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

    for league in LEAGUES:
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
            field: season_field_value(
                league,
                row,
                field,
            )
            for field in fields
        }

        for label in (
            "start",
            "end",
        ):
            validate_season_boundary(
                league,
                label,
                values,
            )

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

    current_mmdd = (
        now.month,
        now.day,
    )

    start_mmdd = (
        cfg["start_month"],
        cfg["start_day"],
    )

    end_mmdd = (
        cfg["end_month"],
        cfg["end_day"],
    )

    if start_mmdd <= end_mmdd:
        return start_mmdd <= current_mmdd <= end_mmdd

    return (
        current_mmdd >= start_mmdd
        or current_mmdd <= end_mmdd
    )


def load_rows(path: Path) -> list[dict]:
    safe_path = resolve_repository_path(
        path,
        strict=True,
    )

    with safe_path.open(newline="", encoding="utf-8") as f:
        result = list(csv.DictReader(f))
    return result


def wipe_outputs() -> None:
    for league in LEAGUES:
        for subdir in [
            "moneyline",
            "spread",
            "total",
        ]:
            folder = MERGE_DIR / league / subdir
            folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            for f in folder.glob("*.csv"):
                f.unlink(missing_ok=True)

    log(
        "Wiped all output folders for full replay rebuild."
    )


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict],
) -> None:
    safe_path = resolve_repository_path(
        path,
        strict=False,
    )

    safe_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with safe_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def canonicalize_book_rows(
    rows: list[dict],
    league: str,
    date: str,
) -> tuple[
    dict[str, dict],
    dict[tuple, dict],
    int,
]:
    by_comp: dict[tuple, dict] = {}
    identity_mismatches = 0

    for row in rows:
        key = comp_key(row)

        if key not in by_comp:
            by_comp[key] = row
            continue

        old = by_comp[key]

        old_id = clean(
            old.get("game_id")
        )

        new_id = clean(
            row.get("game_id")
        )

        if (
            old_id.isdigit()
            and new_id.isdigit()
            and old_id != new_id
        ):
            raise ValueError(
                f"{league.upper()} {date}: conflicting numeric "
                f"sportsbook IDs {old_id} vs {new_id} for {key}"
            )

        if (
            old_id != new_id
            and old_id
            and new_id
        ):
            identity_mismatches += 1

            log(
                f"ID ALIAS | {league.upper()} {date} | "
                f"{old_id} <-> {new_id} | "
                f"{key[1]} vs {key[2]}"
            )

        if id_rank(new_id) > id_rank(old_id):
            by_comp[key] = row

    by_id: dict[str, dict] = {}

    for row in by_comp.values():
        gid = clean(
            row.get("game_id")
        )

        if not gid:
            continue

        if (
            gid in by_id
            and comp_key(by_id[gid]) != comp_key(row)
        ):
            raise ValueError(
                f"{league.upper()} {date}: game_id {gid} "
                "maps to multiple game identities"
            )

        by_id[gid] = row

    return (
        by_id,
        by_comp,
        identity_mismatches,
    )


def canonical_game_id(
    pred: dict,
    book: dict,
) -> str:
    p = clean(
        pred.get("game_id")
    )

    b = clean(
        book.get("game_id")
    )

    if (
        p.isdigit()
        and b.isdigit()
        and p != b
    ):
        raise ValueError(
            f"Conflicting numeric game_id "
            f"prediction={p} sportsbook={b} "
            f"for {comp_key(pred)}"
        )

    return (
        b
        if id_rank(b) > id_rank(p)
        else p
    )


def build_base(
    p: dict,
    b: dict,
) -> dict:
    return {
        "sport": p.get("sport", ""),
        "league": p.get("league", ""),
        "game_id": canonical_game_id(p, b),
        "game_date": p.get("game_date", ""),
        "game_time": (
            p.get("game_time", "")
            or b.get("game_time", "")
        ),
        "home_team": p.get("home_team", ""),
        "away_team": p.get("away_team", ""),
        "home_prob": p.get("home_prob", ""),
        "away_prob": p.get("away_prob", ""),
        "away_projected_points": p.get(
            "away_projected_points",
            "",
        ),
        "home_projected_points": p.get(
            "home_projected_points",
            "",
        ),
        "total_projected_points": p.get(
            "total_projected_points",
            "",
        ),
        "bias_applied": p.get(
            "bias_applied",
            "",
        ),
        "margin_bias": p.get(
            "margin_bias",
            "",
        ),
        "total_bias": p.get(
            "total_bias",
            "",
        ),
        "model_source": p.get(
            "model_source",
            "",
        ),
        "model_version": p.get(
            "model_version",
            "",
        ),
        "feature_version": p.get(
            "feature_version",
            "",
        ),
        "ensemble_version": p.get(
            "ensemble_version",
            "",
        ),
        "sportsbook_provider": b.get(
            "sportsbook_provider",
            "",
        ),
        "scraped_at_utc": b.get(
            "scraped_at_utc",
            "",
        ),
        "provider_updated_at_utc": b.get(
            "provider_updated_at_utc",
            "",
        ),
        "total": b.get(
            "total",
            "",
        ),
    }


def main() -> None:
    with open(
        LOG_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"=== merge_intake RUN "
            f"{datetime.now().isoformat()} ===\n"
        )

    files_written = 0
    total_merged = 0
    total_missing = 0
    slates_skipped = 0
    errors = 0

    coverage_rows: list[dict] = []

    now = datetime.now(NY)
    current_date = now.strftime(
        "%Y_%m_%d"
    )

    full_rebuild = truthy(
        os.getenv(
            "BASKETBALL_FULL_REBUILD"
        )
    )

    try:
        season_config = load_season_config()

        log(
            "SEASON CONFIG | "
            f"file={SEASON_CONFIG}"
        )

        if full_rebuild:
            wipe_outputs()

        else:
            for league in LEAGUES:
                upper = league.upper()

                for market in [
                    "moneyline",
                    "spread",
                    "total",
                ]:
                    path = (
                        MERGE_DIR
                        / league
                        / market
                        / f"{current_date}_{upper}_{market}.csv"
                    )

                    path.unlink(
                        missing_ok=True
                    )

            log(
                f"Incremental mode: rebuilding only "
                f"{current_date}; historical merge outputs preserved."
            )

        for league in LEAGUES:
            league_upper = league.upper()

            pred_dir = (
                PREDICTIONS_DIR
                / league
            )

            book_dir = (
                SPORTSBOOK_DIR
                / league
            )

            if not pred_dir.exists():
                log(
                    f"PREDICTIONS DIR NOT FOUND: {pred_dir}"
                )
                continue

            if full_rebuild:
                pred_files = sorted(
                    pred_dir.glob(
                        f"*_{league_upper}_predictions.csv"
                    )
                )

            else:
                current_pred = (
                    pred_dir
                    / f"{current_date}_{league_upper}_predictions.csv"
                )

                pred_files = (
                    [current_pred]
                    if current_pred.exists()
                    else []
                )

            if not pred_files:
                log(
                    f"NO PREDICTION FILES: {pred_dir}"
                )
                continue

            for pred_file in pred_files:
                date = pred_file.stem.replace(
                    f"_{league_upper}_predictions",
                    "",
                )

                book_file = (
                    book_dir
                    / f"{date}_{league_upper}_odds.csv"
                )

                current_live = (
                    date == current_date
                    and in_season(
                        league,
                        now,
                        season_config,
                    )
                )

                pred_rows = load_rows(
                    pred_file
                )

                if not pred_rows:
                    log(
                        f"EMPTY PREDICTIONS: {pred_file} — skipping"
                    )
                    slates_skipped += 1
                    continue

                if not book_file.exists():
                    log(
                        f"NO SPORTSBOOK FILE: {book_file} — skipping"
                    )

                    slates_skipped += 1

                    coverage_rows.append({
                        "league": league_upper,
                        "game_date": date,
                        "prediction_rows": len(pred_rows),
                        "sportsbook_rows": 0,
                        "matched_rows": 0,
                        "missing_matches": len(pred_rows),
                        "match_by_game_id": 0,
                        "match_by_composite": 0,
                        "identity_mismatches": 0,
                        "coverage_pct": 0.0,
                        "current_in_season": int(current_live),
                        "status": (
                            "ERROR"
                            if current_live
                            else "SKIPPED"
                        ),
                    })

                    if current_live:
                        errors += 1

                    continue

                book_rows = load_rows(
                    book_file
                )

                if not book_rows:
                    log(
                        f"EMPTY SPORTSBOOK: {book_file} — skipping"
                    )

                    slates_skipped += 1

                    if current_live:
                        errors += 1

                    continue

                (
                    book_by_id,
                    book_by_comp,
                    identity_mismatches,
                ) = canonicalize_book_rows(
                    book_rows,
                    league,
                    date,
                )

                ml_rows: list[dict] = []
                spread_rows: list[dict] = []
                total_rows: list[dict] = []

                missing = 0
                by_id_matches = 0
                by_comp_matches = 0

                for p in pred_rows:
                    p_gid = clean(
                        p.get("game_id")
                    )

                    b = (
                        book_by_id.get(p_gid)
                        if p_gid
                        else None
                    )

                    if b is not None:
                        by_id_matches += 1

                    else:
                        b = book_by_comp.get(
                            comp_key(p)
                        )

                        if b is not None:
                            by_comp_matches += 1

                    if b is None:
                        missing += 1
                        total_missing += 1

                        log(
                            f"MISSING MATCH | "
                            f"{league_upper} {date} | "
                            f"{p.get('home_team')} vs "
                            f"{p.get('away_team')} | "
                            f"game_id={p_gid}"
                        )

                        continue

                    book_gid = clean(
                        b.get("game_id")
                    )

                    if (
                        p_gid
                        and book_gid
                        and p_gid != book_gid
                    ):
                        identity_mismatches += 1

                        log(
                            f"IDENTITY FALLBACK | "
                            f"{league_upper} {date} | "
                            f"prediction_id={p_gid} "
                            f"sportsbook_id={book_gid}"
                        )

                    base = build_base(
                        p,
                        b,
                    )

                    ml_rows.append({
                        **base,
                        "home_dk_moneyline_american": b.get(
                            "home_dk_moneyline_american",
                            "",
                        ),
                        "away_dk_moneyline_american": b.get(
                            "away_dk_moneyline_american",
                            "",
                        ),
                        "home_dk_moneyline_decimal": b.get(
                            "home_dk_moneyline_decimal",
                            "",
                        ),
                        "away_dk_moneyline_decimal": b.get(
                            "away_dk_moneyline_decimal",
                            "",
                        ),
                    })

                    spread_rows.append({
                        **base,
                        "home_spread": b.get(
                            "home_spread",
                            "",
                        ),
                        "away_spread": b.get(
                            "away_spread",
                            "",
                        ),
                        "home_dk_spread_american": b.get(
                            "home_dk_spread_american",
                            "",
                        ),
                        "away_dk_spread_american": b.get(
                            "away_dk_spread_american",
                            "",
                        ),
                        "home_dk_spread_decimal": b.get(
                            "home_dk_spread_decimal",
                            "",
                        ),
                        "away_dk_spread_decimal": b.get(
                            "away_dk_spread_decimal",
                            "",
                        ),
                    })

                    total_rows.append({
                        **base,
                        "dk_total_over_american": b.get(
                            "dk_total_over_american",
                            "",
                        ),
                        "dk_total_under_american": b.get(
                            "dk_total_under_american",
                            "",
                        ),
                        "dk_total_over_decimal": b.get(
                            "dk_total_over_decimal",
                            "",
                        ),
                        "dk_total_under_decimal": b.get(
                            "dk_total_under_decimal",
                            "",
                        ),
                    })

                matched = len(
                    ml_rows
                )

                pct = round(
                    (
                        matched
                        / len(pred_rows)
                        * 100.0
                    )
                    if pred_rows
                    else 100.0,
                    2,
                )

                status = (
                    "OK"
                    if missing == 0
                    else (
                        "ERROR"
                        if current_live
                        else "PARTIAL"
                    )
                )

                coverage_rows.append({
                    "league": league_upper,
                    "game_date": date,
                    "prediction_rows": len(pred_rows),
                    "sportsbook_rows": len(book_rows),
                    "matched_rows": matched,
                    "missing_matches": missing,
                    "match_by_game_id": by_id_matches,
                    "match_by_composite": by_comp_matches,
                    "identity_mismatches": identity_mismatches,
                    "coverage_pct": pct,
                    "current_in_season": int(current_live),
                    "status": status,
                })

                log(
                    f"COVERAGE | "
                    f"{league_upper} {date} | "
                    f"matched={matched}/{len(pred_rows)} "
                    f"({pct:.2f}%) | "
                    f"missing={missing} | "
                    f"id={by_id_matches} "
                    f"fallback={by_comp_matches}"
                )

                if current_live and missing:
                    errors += 1

                if not ml_rows:
                    log(
                        f"NO MERGED ROWS: "
                        f"{league_upper} {date} — skipping"
                    )

                    slates_skipped += 1
                    continue

                ml_path = (
                    MERGE_DIR
                    / league
                    / "moneyline"
                    / f"{date}_{league_upper}_moneyline.csv"
                )

                spread_path = (
                    MERGE_DIR
                    / league
                    / "spread"
                    / f"{date}_{league_upper}_spread.csv"
                )

                total_path = (
                    MERGE_DIR
                    / league
                    / "total"
                    / f"{date}_{league_upper}_total.csv"
                )

                write_csv(
                    ml_path,
                    MONEYLINE_FIELDS,
                    ml_rows,
                )

                write_csv(
                    spread_path,
                    SPREAD_FIELDS,
                    spread_rows,
                )

                write_csv(
                    total_path,
                    TOTAL_FIELDS,
                    total_rows,
                )

                total_merged += matched
                files_written += 3

                log(
                    f"WROTE {ml_path.name} | "
                    f"{spread_path.name} | "
                    f"{total_path.name} "
                    f"({matched} rows each)"
                )

    except Exception as exc:
        errors += 1

        log(
            f"FATAL ERROR: {exc}\n"
            f"{traceback.format_exc()}"
        )

    write_csv(
        COVERAGE_FILE,
        COVERAGE_FIELDS,
        coverage_rows,
    )

    log("--- SUMMARY ---")

    log(
        f"Mode: "
        f"{'full_rebuild' if full_rebuild else 'incremental_current_date'}"
    )

    log(
        f"Season config: {SEASON_CONFIG}"
    )

    log(
        f"Files written: {files_written}"
    )

    log(
        f"Total rows merged: {total_merged}"
    )

    log(
        f"Total missing matches: {total_missing}"
    )

    log(
        f"Slates skipped: {slates_skipped}"
    )

    log(
        f"Errors: {errors}"
    )

    log(
        f"STATUS: "
        f"{'SUCCESS' if errors == 0 else 'FAILED'}"
    )

    if errors:
        sys.exit(1)

    print(
        "merge_intake complete."
    )


if __name__ == "__main__":
    main()