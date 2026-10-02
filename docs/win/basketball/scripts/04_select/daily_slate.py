#!/usr/bin/env python3
# docs/win/basketball/scripts/04_select/daily_slate.py
"""Build current-day per-league selected slates.

Unlike the previous implementation, this reads only the New York current-date
`daily_picks` file. Historical/season-to-date picks remain in daily_picks and are
not concatenated into an artifact named daily_slate.
"""
from __future__ import annotations

import csv
import sys
import traceback
from datetime import datetime, UTC
from pathlib import Path
from zoneinfo import ZoneInfo

SELECT_DIR = Path("docs/win/basketball/04_select")
DAILY_SLATE_DIR = SELECT_DIR / "daily_slate"
ERROR_DIR = Path("docs/win/basketball/errors/04_select")
LOG_FILE = ERROR_DIR / "daily_slate.txt"
NY = ZoneInfo("America/New_York")
LEAGUES = ["nba", "wnba", "ncaam"]

HEADERS = [
    "sport","league","game_id","game_date","game_time","home_team","away_team",
    "model_source","model_version","feature_version","ensemble_version",
    "home_prob","away_prob","away_projected_points","home_projected_points","total_projected_points",
    "bias_applied","margin_bias","total_bias","total","home_dk_moneyline_american",
    "away_dk_moneyline_american","home_dk_moneyline_decimal","away_dk_moneyline_decimal",
    "away_decimal","home_decimal","away_implied_prob","home_implied_prob","away_market_prob",
    "home_market_prob","home_model_prob","away_model_prob","away_fair","home_fair",
    "away_acceptable_decimal_moneyline","home_acceptable_decimal_moneyline",
    "away_acceptable_american_moneyline","home_acceptable_american_moneyline","home_ml_ev",
    "away_ml_ev","home_ml_edge_vs_market","away_ml_edge_vs_market","home_ml_kelly","away_ml_kelly",
    "home_ml_ev_pct","away_ml_ev_pct","home_ml_edge_vs_market_pct","away_ml_edge_vs_market_pct",
    "bet_side","bet_line","bet_odds_american","bet_ev","bet_raw_ev","bet_uncertainty_adjusted_ev",
    "bet_kelly","bet_raw_kelly","bet_model_prob","bet_adjusted_model_prob","bet_edge_vs_market",
    "bet_uncertainty_multiplier","bet_uncertainty_points","bet_signal_points",
    "bet_fractional_kelly_pct","bet_individual_capped_stake_pct","bet_requested_stake_pct",
    "bet_final_stake_pct","bet_stake_pct","exposure_rank","exposure_limited","exposure_limit_reason",
    "game_exposure_after_pct","league_day_exposure_after_pct","total_day_exposure_after_pct",
    "maximum_exposure_per_game","maximum_exposure_per_league_per_day","maximum_total_daily_exposure",
    "maximum_individual_bet_kelly_fraction","uncertainty_adjustment_method","uncertainty_adjustment_version",
    "market_type","league_lower","home_spread","away_spread",
    "home_dk_spread_american","away_dk_spread_american","home_dk_spread_decimal","away_dk_spread_decimal",
    "home_spread_model_prob","away_spread_model_prob","fair_home_spread_decimal","fair_away_spread_decimal",
    "home_acceptable_spread_decimal","away_acceptable_spread_decimal","home_acceptable_spread_american",
    "away_acceptable_spread_american","home_spread_implied_prob","away_spread_implied_prob",
    "home_spread_market_prob","away_spread_market_prob","home_spread_ev","away_spread_ev",
    "home_spread_edge_vs_market","away_spread_edge_vs_market","home_spread_kelly","away_spread_kelly",
    "home_spread_ev_pct","away_spread_ev_pct","home_spread_edge_vs_market_pct",
    "away_spread_edge_vs_market_pct","dk_total_over_american","dk_total_under_american",
    "dk_total_over_decimal","dk_total_under_decimal","over_model_prob","under_model_prob","fair_over",
    "fair_under","acceptable_over","acceptable_under","over_implied_prob","under_implied_prob",
    "over_market_prob","under_market_prob","over_ev","under_ev","over_edge_vs_market",
    "under_edge_vs_market","over_kelly","under_kelly","over_ev_pct","under_ev_pct",
    "over_edge_vs_market_pct","under_edge_vs_market_pct",
]
CRITICAL = {"game_id", "game_date", "market_type", "bet_side"}


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def log(message: str, level: str = "INFO") -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{now_iso()} | {level:<5} | {message.rstrip()}\n")


def write_output(path: Path, rows: list[dict]) -> None:
    repo_root = Path(__file__).resolve().parents[5]
    candidate = path if path.is_absolute() else Path.cwd() / path
    safe_path = candidate.resolve()
    try:
        safe_path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {path}") from exc

    safe_path.parent.mkdir(parents=True, exist_ok=True)
    with safe_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    DAILY_SLATE_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== basketball daily_slate RUN {now_iso()} ===\n")

    game_date = datetime.now(NY).strftime("%Y_%m_%d")
    errors = 0; total_rows = 0
    log(f"CURRENT GAME DATE: {game_date} America/New_York")

    for league in LEAGUES:
        source = SELECT_DIR / league / "daily_picks" / f"{game_date}_{league}_selected.csv"
        target = DAILY_SLATE_DIR / f"{league}_selected.csv"
        rows: list[dict] = []
        try:
            if source.exists():
                with open(source, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    actual = set(reader.fieldnames or [])
                    missing_critical = sorted(CRITICAL - actual)
                    if missing_critical:
                        raise ValueError(f"missing critical headers: {', '.join(missing_critical)}")
                    extra = sorted(actual - set(HEADERS))
                    if extra:
                        log(f"[{league}] extra input headers ignored: {', '.join(extra)}", "WARN")
                    for row in reader:
                        if str(row.get("game_date", "")).strip() != game_date:
                            raise ValueError(f"row game_date {row.get('game_date')!r} does not equal {game_date}")
                        rows.append({h: row.get(h, "") for h in HEADERS})
                log(f"[{league}] READ current daily picks: {source} ({len(rows)} rows)")
            else:
                log(f"[{league}] no current daily picks: {source}")
            write_output(target, rows)
            total_rows += len(rows)
            log(f"[{league}] WROTE {target} ({len(rows)} rows)")
        except Exception as exc:
            errors += 1
            log(f"[{league}] FAILED: {exc}\n{traceback.format_exc()}", "ERROR")
            # Never leave yesterday's slate in place after a failed current build.
            write_output(target, [])

    log("--- SUMMARY ---")
    log(f"Current game date: {game_date}")
    log(f"Output rows: {total_rows}")
    log(f"Errors: {errors}")
    log(f"STATUS: {'SUCCESS' if errors == 0 else 'FAILED'}")
    if errors:
        sys.exit(1)
    print("basketball daily_slate complete.")


if __name__ == "__main__":
    main()
