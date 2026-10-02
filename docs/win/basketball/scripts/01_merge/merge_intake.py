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
    repo_root = Path(__file__).resolve().parents[5]
    candidate = path if path.is_absolute() else Path.cwd() / path
    safe_path = candidate.resolve(strict=True)
    try:
        safe_path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {path}") from exc

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
    repo_root = Path(__file__).resolve().parents[5]
    candidate = path if path.is_absolute() else Path.cwd() / path
    safe_path = candidate.resolve()
    try:
        safe_path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {path}") from exc

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
    _py_r1000_NONE = 0
    _py_r1000_RETURN = 1
    _py_r1000_BREAK = 2
    _py_r1000_CONTINUE = 3

    def _py_r1000_impl():
        b: object
        base: object
        book_by_comp: object
        book_by_id: object
        book_dir: object
        book_file: object
        book_gid: object
        book_rows: object
        by_comp_matches: object
        by_id_matches: object
        coverage_rows: object
        current_date: object
        current_live: object
        current_pred: object
        date: object
        errors: object
        exc: object
        f: object
        files_written: object
        full_rebuild: object
        identity_mismatches: object
        league: object
        league_upper: object
        market: object
        matched: object
        missing: object
        ml_path: object
        ml_rows: object
        now: object
        p: object
        p_gid: object
        path: object
        pct: object
        pred_dir: object
        pred_file: object
        pred_files: object
        pred_rows: object
        season_config: object
        slates_skipped: object
        spread_path: object
        spread_rows: object
        status: object
        total_merged: object
        total_missing: object
        total_path: object
        total_rows: object
        upper: object

        def _py_r1000_try_1():
            nonlocal b, base, book_by_comp, book_by_id, book_dir, book_file, book_gid, book_rows, by_comp_matches, by_id_matches, current_live, current_pred, date, errors, files_written, identity_mismatches, league, league_upper, market, matched, missing, ml_path, ml_rows, p, p_gid, path, pct, pred_dir, pred_file, pred_files, pred_rows, season_config, slates_skipped, spread_path, spread_rows, status, total_merged, total_missing, total_path, total_rows, upper

            def _py_r1000_else_2():
                nonlocal league, market, path, upper

                def _py_r1000_loop_3():
                    nonlocal market, path, upper
                    upper = league.upper()
                    for market in ['moneyline', 'spread', 'total']:
                        path = MERGE_DIR / league / market / f'{current_date}_{upper}_{market}.csv'
                        path.unlink(missing_ok=True)
                    return (_py_r1000_NONE, None)
                for league in LEAGUES:
                    _py_r1000_result_4 = _py_r1000_loop_3()
                    if _py_r1000_result_4[0] == _py_r1000_RETURN:
                        return _py_r1000_result_4
                    if _py_r1000_result_4[0] == _py_r1000_BREAK:
                        break
                    if _py_r1000_result_4[0] == _py_r1000_CONTINUE:
                        continue
                log(f'Incremental mode: rebuilding only {current_date}; historical merge outputs preserved.')
                return (_py_r1000_NONE, None)

            def _py_r1000_loop_6():
                nonlocal b, base, book_by_comp, book_by_id, book_dir, book_file, book_gid, book_rows, by_comp_matches, by_id_matches, current_live, current_pred, date, errors, files_written, identity_mismatches, league_upper, matched, missing, ml_path, ml_rows, p, p_gid, pct, pred_dir, pred_file, pred_files, pred_rows, slates_skipped, spread_path, spread_rows, status, total_merged, total_missing, total_path, total_rows

                def _py_r1000_else_7():
                    nonlocal current_pred, pred_files
                    current_pred = pred_dir / f'{current_date}_{league_upper}_predictions.csv'
                    pred_files = [current_pred] if current_pred.exists() else []
                    return (_py_r1000_NONE, None)

                def _py_r1000_loop_9():
                    nonlocal b, base, book_by_comp, book_by_id, book_file, book_gid, book_rows, by_comp_matches, by_id_matches, current_live, date, errors, files_written, identity_mismatches, matched, missing, ml_path, ml_rows, p, p_gid, pct, pred_rows, slates_skipped, spread_path, spread_rows, status, total_merged, total_missing, total_path, total_rows

                    def _py_r1000_if_10():
                        nonlocal errors, slates_skipped
                        log(f'NO SPORTSBOOK FILE: {book_file} — skipping')
                        slates_skipped += 1
                        coverage_rows.append({'league': league_upper, 'game_date': date, 'prediction_rows': len(pred_rows), 'sportsbook_rows': 0, 'matched_rows': 0, 'missing_matches': len(pred_rows), 'match_by_game_id': 0, 'match_by_composite': 0, 'identity_mismatches': 0, 'coverage_pct': 0.0, 'current_in_season': int(current_live), 'status': 'ERROR' if current_live else 'SKIPPED'})
                        if current_live:
                            errors += 1
                        return (_py_r1000_CONTINUE, None)

                    def _py_r1000_if_12():
                        nonlocal errors, slates_skipped
                        log(f'EMPTY SPORTSBOOK: {book_file} — skipping')
                        slates_skipped += 1
                        if current_live:
                            errors += 1
                        return (_py_r1000_CONTINUE, None)

                    def _py_r1000_loop_14():
                        nonlocal b, base, book_gid, by_comp_matches, by_id_matches, identity_mismatches, missing, p_gid, total_missing

                        def _py_r1000_else_15():
                            nonlocal b, by_comp_matches
                            b = book_by_comp.get(comp_key(p))
                            if b is not None:
                                by_comp_matches += 1
                            return (_py_r1000_NONE, None)
                        p_gid = clean(p.get('game_id'))
                        b = book_by_id.get(p_gid) if p_gid else None
                        if b is not None:
                            by_id_matches += 1
                        else:
                            _py_r1000_result_16 = _py_r1000_else_15()
                            if _py_r1000_result_16[0] != _py_r1000_NONE:
                                return _py_r1000_result_16
                        if b is None:
                            missing += 1
                            total_missing += 1
                            log(f"MISSING MATCH | {league_upper} {date} | {p.get('home_team')} vs {p.get('away_team')} | game_id={p_gid}")
                            return (_py_r1000_CONTINUE, None)
                        book_gid = clean(b.get('game_id'))
                        if p_gid and book_gid and (p_gid != book_gid):
                            identity_mismatches += 1
                            log(f'IDENTITY FALLBACK | {league_upper} {date} | prediction_id={p_gid} sportsbook_id={book_gid}')
                        base = build_base(p, b)
                        ml_rows.append({**base, 'home_dk_moneyline_american': b.get('home_dk_moneyline_american', ''), 'away_dk_moneyline_american': b.get('away_dk_moneyline_american', ''), 'home_dk_moneyline_decimal': b.get('home_dk_moneyline_decimal', ''), 'away_dk_moneyline_decimal': b.get('away_dk_moneyline_decimal', '')})
                        spread_rows.append({**base, 'home_spread': b.get('home_spread', ''), 'away_spread': b.get('away_spread', ''), 'home_dk_spread_american': b.get('home_dk_spread_american', ''), 'away_dk_spread_american': b.get('away_dk_spread_american', ''), 'home_dk_spread_decimal': b.get('home_dk_spread_decimal', ''), 'away_dk_spread_decimal': b.get('away_dk_spread_decimal', '')})
                        total_rows.append({**base, 'dk_total_over_american': b.get('dk_total_over_american', ''), 'dk_total_under_american': b.get('dk_total_under_american', ''), 'dk_total_over_decimal': b.get('dk_total_over_decimal', ''), 'dk_total_under_decimal': b.get('dk_total_under_decimal', '')})
                        return (_py_r1000_NONE, None)

                    def _py_r1000_chunk_18():
                        nonlocal book_by_comp, book_by_id, book_file, book_rows, by_comp_matches, by_id_matches, current_live, date, identity_mismatches, missing, ml_rows, pred_rows, slates_skipped, spread_rows, total_rows

                        def _py_r1000_if_19():
                            _py_r1000_result_11 = _py_r1000_if_10()
                            if _py_r1000_result_11[0] != _py_r1000_NONE:
                                return _py_r1000_result_11
                            return (_py_r1000_NONE, None)

                        def _py_r1000_if_21():
                            _py_r1000_result_13 = _py_r1000_if_12()
                            if _py_r1000_result_13[0] != _py_r1000_NONE:
                                return _py_r1000_result_13
                            return (_py_r1000_NONE, None)
                        date = pred_file.stem.replace(f'_{league_upper}_predictions', '')
                        book_file = book_dir / f'{date}_{league_upper}_odds.csv'
                        current_live = date == current_date and in_season(league, now, season_config)
                        pred_rows = load_rows(pred_file)
                        if not pred_rows:
                            log(f'EMPTY PREDICTIONS: {pred_file} — skipping')
                            slates_skipped += 1
                            return (_py_r1000_CONTINUE, None)
                        if not book_file.exists():
                            _py_r1000_result_20 = _py_r1000_if_19()
                            if _py_r1000_result_20[0] != _py_r1000_NONE:
                                return _py_r1000_result_20
                        book_rows = load_rows(book_file)
                        if not book_rows:
                            _py_r1000_result_22 = _py_r1000_if_21()
                            if _py_r1000_result_22[0] != _py_r1000_NONE:
                                return _py_r1000_result_22
                        book_by_id, book_by_comp, identity_mismatches = canonicalize_book_rows(book_rows, league, date)
                        ml_rows = []
                        spread_rows = []
                        total_rows = []
                        missing = 0
                        by_id_matches = 0
                        by_comp_matches = 0
                        return (_py_r1000_NONE, None)

                    def _py_r1000_chunk_23():
                        nonlocal matched, p, pct, status

                        def _py_r1000_loop_24():
                            _py_r1000_result_17 = _py_r1000_loop_14()
                            if _py_r1000_result_17[0] == _py_r1000_RETURN:
                                return _py_r1000_result_17
                            if _py_r1000_result_17[0] == _py_r1000_BREAK:
                                return (_py_r1000_BREAK, None)
                            if _py_r1000_result_17[0] == _py_r1000_CONTINUE:
                                return (_py_r1000_CONTINUE, None)
                            return (_py_r1000_NONE, None)
                        for p in pred_rows:
                            _py_r1000_result_25 = _py_r1000_loop_24()
                            if _py_r1000_result_25[0] == _py_r1000_RETURN:
                                return _py_r1000_result_25
                            if _py_r1000_result_25[0] == _py_r1000_BREAK:
                                break
                            if _py_r1000_result_25[0] == _py_r1000_CONTINUE:
                                continue
                        matched = len(ml_rows)
                        pct = round(matched / len(pred_rows) * 100.0 if pred_rows else 100.0, 2)
                        status = 'OK' if missing == 0 else 'ERROR' if current_live else 'PARTIAL'
                        coverage_rows.append({'league': league_upper, 'game_date': date, 'prediction_rows': len(pred_rows), 'sportsbook_rows': len(book_rows), 'matched_rows': matched, 'missing_matches': missing, 'match_by_game_id': by_id_matches, 'match_by_composite': by_comp_matches, 'identity_mismatches': identity_mismatches, 'coverage_pct': pct, 'current_in_season': int(current_live), 'status': status})
                        log(f'COVERAGE | {league_upper} {date} | matched={matched}/{len(pred_rows)} ({pct:.2f}%) | missing={missing} | id={by_id_matches} fallback={by_comp_matches}')
                        return (_py_r1000_NONE, None)

                    def _py_r1000_chunk_26():
                        nonlocal errors, files_written, ml_path, slates_skipped, spread_path, total_merged, total_path
                        if current_live and missing:
                            errors += 1
                        if not ml_rows:
                            log(f'NO MERGED ROWS: {league_upper} {date} — skipping')
                            slates_skipped += 1
                            return (_py_r1000_CONTINUE, None)
                        ml_path = MERGE_DIR / league / 'moneyline' / f'{date}_{league_upper}_moneyline.csv'
                        spread_path = MERGE_DIR / league / 'spread' / f'{date}_{league_upper}_spread.csv'
                        total_path = MERGE_DIR / league / 'total' / f'{date}_{league_upper}_total.csv'
                        write_csv(ml_path, MONEYLINE_FIELDS, ml_rows)
                        write_csv(spread_path, SPREAD_FIELDS, spread_rows)
                        write_csv(total_path, TOTAL_FIELDS, total_rows)
                        total_merged += matched
                        files_written += 3
                        log(f'WROTE {ml_path.name} | {spread_path.name} | {total_path.name} ({matched} rows each)')
                        return (_py_r1000_NONE, None)
                    for _py_r1000_block_27 in (_py_r1000_chunk_18, _py_r1000_chunk_23, _py_r1000_chunk_26):
                        _py_r1000_result_28 = _py_r1000_block_27()
                        if _py_r1000_result_28[0] != _py_r1000_NONE:
                            return _py_r1000_result_28
                    return (_py_r1000_NONE, None)
                league_upper = league.upper()
                pred_dir = PREDICTIONS_DIR / league
                book_dir = SPORTSBOOK_DIR / league
                if not pred_dir.exists():
                    log(f'PREDICTIONS DIR NOT FOUND: {pred_dir}')
                    return (_py_r1000_CONTINUE, None)
                if full_rebuild:
                    pred_files = sorted(pred_dir.glob(f'*_{league_upper}_predictions.csv'))
                else:
                    _py_r1000_result_8 = _py_r1000_else_7()
                    if _py_r1000_result_8[0] != _py_r1000_NONE:
                        return _py_r1000_result_8
                if not pred_files:
                    log(f'NO PREDICTION FILES: {pred_dir}')
                    return (_py_r1000_CONTINUE, None)
                for pred_file in pred_files:
                    _py_r1000_result_29 = _py_r1000_loop_9()
                    if _py_r1000_result_29[0] == _py_r1000_RETURN:
                        return _py_r1000_result_29
                    if _py_r1000_result_29[0] == _py_r1000_BREAK:
                        break
                    if _py_r1000_result_29[0] == _py_r1000_CONTINUE:
                        continue
                return (_py_r1000_NONE, None)
            season_config = load_season_config()
            log(f'SEASON CONFIG | file={SEASON_CONFIG}')
            if full_rebuild:
                wipe_outputs()
            else:
                _py_r1000_result_5 = _py_r1000_else_2()
                if _py_r1000_result_5[0] != _py_r1000_NONE:
                    return _py_r1000_result_5
            for league in LEAGUES:
                _py_r1000_result_30 = _py_r1000_loop_6()
                if _py_r1000_result_30[0] == _py_r1000_RETURN:
                    return _py_r1000_result_30
                if _py_r1000_result_30[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_30[0] == _py_r1000_CONTINUE:
                    continue
            return (_py_r1000_NONE, None)
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f'=== merge_intake RUN {datetime.now().isoformat()} ===\n')
        files_written = 0
        total_merged = 0
        total_missing = 0
        slates_skipped = 0
        errors = 0
        coverage_rows: list[dict] = []
        now = datetime.now(NY)
        current_date = now.strftime('%Y_%m_%d')
        full_rebuild = truthy(os.getenv('BASKETBALL_FULL_REBUILD'))
        try:
            _py_r1000_result_31 = _py_r1000_try_1()
            if _py_r1000_result_31[0] != _py_r1000_NONE:
                return _py_r1000_result_31
        except Exception as exc:
            errors += 1
            log(f'FATAL ERROR: {exc}\n{traceback.format_exc()}')
        write_csv(COVERAGE_FILE, COVERAGE_FIELDS, coverage_rows)
        log('--- SUMMARY ---')
        log(f"Mode: {('full_rebuild' if full_rebuild else 'incremental_current_date')}")
        log(f'Season config: {SEASON_CONFIG}')
        log(f'Files written: {files_written}')
        log(f'Total rows merged: {total_merged}')
        log(f'Total missing matches: {total_missing}')
        log(f'Slates skipped: {slates_skipped}')
        log(f'Errors: {errors}')
        log(f"STATUS: {('SUCCESS' if errors == 0 else 'FAILED')}")
        if errors:
            sys.exit(1)
        print('merge_intake complete.')
        return (_py_r1000_NONE, None)
    _py_r1000_outcome = _py_r1000_impl()
    if _py_r1000_outcome[0] == _py_r1000_RETURN:
        return _py_r1000_outcome[1]


if __name__ == "__main__":
    main()