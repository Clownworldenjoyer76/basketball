#!/usr/bin/env python3
# docs/win/basketball/scripts/01_merge/merge_intake.py
"""Merge predictions with available sportsbook prices; missing odds do not block predictions.

Existing market output filenames and schemas are preserved.  A game is eligible
only when it has cleaned predictions (derived from DRatings).  Missing odds stay
empty and therefore cannot become a bet.
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

PROVENANCE_FIELDS = ["bias_applied", "margin_bias", "total_bias", "model_source", "model_version", "feature_version", "ensemble_version"]
SPORTSBOOK_PROVENANCE_FIELDS = ["sportsbook_provider", "scraped_at_utc", "provider_updated_at_utc"]
COMMON_FIELDS = ["sport", "league", "game_id", "game_date", "game_time", "home_team", "away_team", "home_prob", "away_prob", "away_projected_points", "home_projected_points", "total_projected_points", *PROVENANCE_FIELDS, *SPORTSBOOK_PROVENANCE_FIELDS, "total"]
MONEYLINE_FIELDS = [*COMMON_FIELDS, "home_dk_moneyline_american", "away_dk_moneyline_american", "home_dk_moneyline_decimal", "away_dk_moneyline_decimal"]
SPREAD_FIELDS = [*COMMON_FIELDS, "home_spread", "away_spread", "home_dk_spread_american", "away_dk_spread_american", "home_dk_spread_decimal", "away_dk_spread_decimal"]
TOTAL_FIELDS = [*COMMON_FIELDS, "dk_total_over_american", "dk_total_under_american", "dk_total_over_decimal", "dk_total_under_decimal"]
COVERAGE_FIELDS = ["league", "game_date", "prediction_rows", "sportsbook_rows", "matched_rows", "missing_matches", "match_by_game_id", "match_by_composite", "identity_mismatches", "coverage_pct", "current_in_season", "status"]


def log(message: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now().isoformat()} | {message}\n")


def clean(v) -> str:
    return "" if v is None else str(v).strip()


def comp_key(row: dict) -> tuple[str, str, str]:
    return (clean(row.get("game_date")), clean(row.get("home_team")).casefold(), clean(row.get("away_team")).casefold())


def truthy(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def load_season_config() -> dict:
    with resolve_repository_path(SEASON_CONFIG, strict=True).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for league in LEAGUES:
        row = config.get(league)
        if not isinstance(row, dict):
            raise ValueError(f"Missing season config for {league}")
        for boundary in ("start", "end"):
            datetime(2000, int(row[f"{boundary}_month"]), int(row[f"{boundary}_day"]))
    return config


def in_season(league: str, now: datetime, cfg: dict) -> bool:
    row = cfg[league]
    start = (int(row["start_month"]), int(row["start_day"]))
    end = (int(row["end_month"]), int(row["end_day"]))
    today = (now.month, now.day)
    return start <= today <= end if start <= end else today >= start or today <= end


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with resolve_repository_path(path, strict=True).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    dest = resolve_repository_path(path, strict=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def wipe_outputs() -> None:
    for league in LEAGUES:
        for market in ("moneyline", "spread", "total"):
            folder = MERGE_DIR / league / market
            folder.mkdir(parents=True, exist_ok=True)
            for f in folder.glob("*.csv"):
                f.unlink(missing_ok=True)
    log("Wiped all output folders for full replay rebuild.")


def id_rank(identifier: str) -> int:
    return 0 if not identifier else 2 if identifier.isdigit() else 1


def canonicalize_book_rows(rows: list[dict], league: str, date: str) -> tuple[dict, dict, int]:
    by_comp = {}
    identity_mismatches = 0
    for row in rows:
        key = comp_key(row)
        old = by_comp.get(key)
        if old is None:
            by_comp[key] = row
            continue
        old_id = clean(old.get("game_id"))
        new_id = clean(row.get("game_id"))
        if old_id.isdigit() and new_id.isdigit() and old_id != new_id:
            raise ValueError(f"{league.upper()} {date}: conflicting sportsbook IDs {old_id} and {new_id} for {key}")
        if old_id and new_id and old_id != new_id:
            identity_mismatches += 1
            log(f"ID ALIAS | {league.upper()} {date} | {old_id} vs {new_id} | {key}")
        if id_rank(new_id) > id_rank(old_id):
            by_comp[key] = row
    by_id = {}
    for row in by_comp.values():
        identifier = clean(row.get("game_id"))
        if identifier:
            if identifier in by_id and comp_key(by_id[identifier]) != comp_key(row):
                raise ValueError(f"{league.upper()} {date}: game_id {identifier} maps to multiple games")
            by_id[identifier] = row
    return by_id, by_comp, identity_mismatches


def canonical_game_id(pred: dict, book: dict) -> str:
    a = clean(pred.get("game_id"))
    b = clean(book.get("game_id"))
    if a.isdigit() and b.isdigit() and a != b:
        raise ValueError(f"Conflicting numeric game_id prediction={a} sportsbook={b} for {comp_key(pred)}")
    return b if id_rank(b) > id_rank(a) else a


def build_base(p: dict, b: dict) -> dict:
    return {
        "sport": p.get("sport", ""), "league": p.get("league", ""),
        "game_id": canonical_game_id(p, b), "game_date": p.get("game_date", ""),
        "game_time": p.get("game_time", "") or b.get("game_time", ""),
        "home_team": p.get("home_team", ""), "away_team": p.get("away_team", ""),
        "home_prob": p.get("home_prob", ""), "away_prob": p.get("away_prob", ""),
        "away_projected_points": p.get("away_projected_points", ""),
        "home_projected_points": p.get("home_projected_points", ""),
        "total_projected_points": p.get("total_projected_points", ""),
        **{field: p.get(field, "") for field in PROVENANCE_FIELDS},
        **{field: b.get(field, "") for field in SPORTSBOOK_PROVENANCE_FIELDS},
        "total": b.get("total", ""),
    }


def main() -> None:
    LOG_FILE.write_text(f"=== merge_intake RUN {datetime.now().isoformat()} ===\n", encoding="utf-8")
    written = missing_total = processed = skipped = errors = 0
    coverage = []
    now = datetime.now(NY)
    today = now.strftime("%Y_%m_%d")
    full = truthy(os.getenv("BASKETBALL_FULL_REBUILD"))
    try:
        seasons = load_season_config()
        if full:
            wipe_outputs()
        else:
            for league in LEAGUES:
                for market in ("moneyline", "spread", "total"):
                    (MERGE_DIR / league / market / f"{today}_{league.upper()}_{market}.csv").unlink(missing_ok=True)
            log(f"Incremental mode: rebuilding only {today}; historical merge outputs preserved.")
        for league in LEAGUES:
            label = league.upper()
            pred_dir = PREDICTIONS_DIR / league
            book_dir = SPORTSBOOK_DIR / league
            files = sorted(pred_dir.glob(f"*_{label}_predictions.csv")) if full else [pred_dir / f"{today}_{label}_predictions.csv"]
            for pred_file in files:
                if not pred_file.exists():
                    continue
                date = pred_file.stem.removesuffix(f"_{label}_predictions")
                current = date == today and in_season(league, now, seasons)
                predictions = load_rows(pred_file)
                if not predictions:
                    skipped += 1
                    continue
                book_file = book_dir / f"{date}_{label}_odds.csv"
                books = load_rows(book_file)
                if not books:
                    log(f"NO SPORTSBOOK DATA | {label} {date} | retaining {len(predictions)} predictions with empty prices")
                books_by_id, books_by_comp, mismatches = canonicalize_book_rows(books, league, date)
                market_rows = {"moneyline": [], "spread": [], "total": []}
                missing = id_matches = comp_matches = 0
                for pred in predictions:
                    pred_id = clean(pred.get("game_id"))
                    book = books_by_id.get(pred_id) if pred_id else None
                    if book is not None:
                        id_matches += 1
                    else:
                        book = books_by_comp.get(comp_key(pred))
                        if book is not None:
                            comp_matches += 1
                    if book is None:
                        missing += 1
                        missing_total += 1
                        book = {}
                        log(f"NO SPORTSBOOK MATCH | {label} {date} | game_id={pred_id} | {pred.get('home_team')} vs {pred.get('away_team')} | no bet without odds")
                    elif pred_id and clean(book.get("game_id")) and pred_id != clean(book["game_id"]):
                        mismatches += 1
                        log(f"IDENTITY FALLBACK | {label} {date} | prediction_id={pred_id} sportsbook_id={book['game_id']}")
                    base = build_base(pred, book)
                    for market, fields in (("moneyline", MONEYLINE_FIELDS), ("spread", SPREAD_FIELDS), ("total", TOTAL_FIELDS)):
                        market_rows[market].append({**base, **{field: book.get(field, "") for field in fields if field not in base}})
                matched = id_matches + comp_matches
                coverage.append({"league": label, "game_date": date, "prediction_rows": len(predictions), "sportsbook_rows": len(books), "matched_rows": matched, "missing_matches": missing, "match_by_game_id": id_matches, "match_by_composite": comp_matches, "identity_mismatches": mismatches, "coverage_pct": round(100 * matched / len(predictions), 2), "current_in_season": int(current), "status": "OK" if missing == 0 else "PARTIAL_NO_ODDS"})
                for market, fields in (("moneyline", MONEYLINE_FIELDS), ("spread", SPREAD_FIELDS), ("total", TOTAL_FIELDS)):
                    write_csv(MERGE_DIR / league / market / f"{date}_{label}_{market}.csv", fields, market_rows[market])
                    written += 1
                processed += len(predictions)
                log(f"WROTE {label} {date}: {len(predictions)} eligible prediction rows, {matched} sportsbook matches, {missing} unpriced (no bet)")
    except Exception as exc:
        errors += 1
        log(f"FATAL ERROR: {exc}\n{traceback.format_exc()}")
    write_csv(COVERAGE_FILE, COVERAGE_FIELDS, coverage)
    log(f"Mode: {'full_rebuild' if full else 'incremental_current_date'}")
    log(f"Files written: {written} | predictions processed: {processed} | unpriced: {missing_total} | skipped: {skipped} | errors: {errors}")
    log(f"STATUS: {'SUCCESS' if errors == 0 else 'FAILED'}")
    if errors:
        sys.exit(1)
    print("merge_intake complete.")


if __name__ == "__main__":
    main()
