#!/usr/bin/env python3
# docs/win/basketball/scripts/05_final_scores/01_basketball_results_grade_core.py
#
# Grades selected bets against final scores and maintains an immutable live-pick
# snapshot for the current New York game date.
#
# Inputs:
#   docs/win/basketball/04_select/{league}/daily_picks/*.csv
#       replay/current-selection outputs; these may be rebuilt by the pipeline
#   docs/win/basketball/04_select/{league}/locked_picks/*.csv
#       immutable live selections; current-day file is created once from daily_picks
#   docs/win/basketball/05_final_scores/results/{league}/*.csv
#       final scores, joined by game_id
#
# Outputs:
#   docs/win/basketball/05_final_scores/results/{league}/graded/{LEAGUE}_final.csv
#       grading of rebuildable daily_picks (existing behavior)
#   docs/win/basketball/05_final_scores/locked_picks/{league}/{LEAGUE}_final.csv
#       rebuildable grading of immutable locked_picks
#   docs/win/basketball/errors/05_final_scores/{league}_game_id_no_match.csv
#   docs/win/basketball/errors/05_final_scores/{league}_locked_game_id_no_match.csv
#   docs/win/basketball/errors/05_final_scores/01_basketball_results_grade.txt
#
# Grading:
#   moneyline  -> winning side based on home/away score, bet_side ∈ {home, away}
#   spread     -> uses bet_line (the side's signed spread); cover iff
#                   home: (home_score + bet_line) > away_score
#                   away: (away_score + bet_line) > home_score
#   total      -> uses bet_line (the book total); over wins iff total > line, etc.
#
# Per-row P&L columns added:
#   profit_unit   = (decimal - 1) if Win else -1 if Loss else 0
#   profit_kelly  = bet_stake_pct * (decimal - 1) if Win
#                 = -bet_stake_pct                if Loss
#                 = 0                              if Push
#                 (None if bet_stake_pct is missing)

import os
import sys
import traceback
from datetime import datetime, UTC
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# =========================
# PATHS / SETTINGS
# =========================

LEAGUES = ["nba", "ncaam", "wnba"]

MODEL_METADATA_COLUMNS = [
    "model_source",
    "model_version",
    "feature_version",
    "ensemble_version",
    "bet_model_prob",
]

LOCK_TIMEZONE = ZoneInfo("America/New_York")

BASE               = Path("docs/win/basketball")
SELECT_BASE        = BASE / "04_select"
FINAL_SCORES_DIR   = BASE / "05_final_scores/results"
LOCKED_RESULTS_DIR = BASE / "05_final_scores/locked_picks"
ERROR_DIR          = BASE / "errors/05_final_scores"
LOG_FILE           = ERROR_DIR / "01_basketball_results_grade.txt"

ERROR_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOGGING
# =========================

def _now():
    return datetime.now(UTC).isoformat()


def _current_game_date() -> str:
    return datetime.now(LOCK_TIMEZONE).strftime("%Y_%m_%d")


def _init_log():
    with open(LOG_FILE, "w", encoding="utf-8") as log_handle:
        log_handle.write(f"=== 01_basketball_results_grade RUN {_now()} ===\n")


def _log(msg: str, level: str = "INFO"):
    line = f"{_now()} | {level:<5} | {msg.rstrip()}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as log_handle:
        log_handle.write(line + "\n")



def log(msg: str, level: str = "INFO"):
    _log(msg, level)

# =========================
# LOCK CURRENT PICKS
# =========================

def _exclusive_copy(source: Path, target: Path) -> bool:
    """Copy source to target without ever overwriting target.

    Returns True when target was created, False when it already existed.
    """
    data = source.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False

    try:
        with os.fdopen(fd, "wb") as file_handle:
            file_handle.write(data)
    except Exception:
        target.unlink(missing_ok=True)
        raise

    return True


def lock_current_picks(summary: dict) -> None:
    """Freeze today's selected-bet file once per league.

    daily_picks remains rebuildable. The current New York game-date file is copied
    to locked_picks only when the locked file does not already exist. An existing
    locked file is never changed. If a regenerated daily file differs from its
    lock, the difference is logged while the locked file is preserved.
    """
    game_date = _current_game_date()
    _log(f"LOCK GAME DATE   : {game_date} ({LOCK_TIMEZONE.key})")

    for league in LEAGUES:
        source = SELECT_BASE / league / "daily_picks" / f"{game_date}_{league}_selected.csv"
        target = SELECT_BASE / league / "locked_picks" / f"{game_date}_{league}_selected.csv"

        if not source.exists():
            _log(f"[{league}] no current daily pick file to lock: {source}")
            summary[f"{league}_locked_created"] = 0
            continue

        created = _exclusive_copy(source, target)
        if created:
            _log(f"[{league}] LOCKED PICKS CREATED: {target}")
            summary[f"{league}_locked_created"] = 1
            continue

        summary[f"{league}_locked_created"] = 0
        summary[f"{league}_locked_preserved"] = 1
        try:
            differs = source.read_bytes() != target.read_bytes()
        except Exception as e:
            _log(f"[{league}] locked file exists but comparison failed: {e}", "WARN")
            continue

        if differs:
            summary[f"{league}_locked_differs"] = 1
            _log(
                f"[{league}] LOCKED PICKS IMMUTABLE: regenerated daily file differs; "
                f"preserved existing {target}",
                "WARN",
            )
        else:
            _log(f"[{league}] LOCKED PICKS EXISTS: preserved immutable {target}")


# =========================
# HELPERS
# =========================


def ensure_model_metadata(df: pd.DataFrame, league: str, label: str) -> pd.DataFrame:
    """Guarantee model provenance/probability columns survive grading.

    Historical rows created before these fields existed are left blank; values are
    never inferred or backfilled. Current rows retain the exact values supplied by
    stage 04.
    """
    out = df.copy()
    missing = []
    for column in MODEL_METADATA_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
            missing.append(column)
    if missing:
        _log(
            f"[{league}] {label} missing model metadata columns; "
            f"preserving as blank: {', '.join(missing)}",
            "WARN",
        )
    return out


def american_to_decimal(odds):
    try:
        a = float(odds)
    except (TypeError, ValueError):
        return None
    if not bool(a):
        return None
    return 1 + (a / 100.0) if a > 0 else 1 + (100.0 / abs(a))


def optional_float(v):
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError, OverflowError):
        return None


def determine_outcome(row) -> str:
    market = str(row.get("market_type", "")).lower()
    side   = str(row.get("bet_side", "")).lower()

    home = optional_float(row.get("home_score"))
    away = optional_float(row.get("away_score"))
    if home is None or away is None:
        return "Unknown"

    if market == "moneyline":
        if home == away:
            return "Push"
        home_won = home > away
        if (side == "home" and home_won) or (side == "away" and not home_won):
            return "Win"
        return "Loss"

    if market == "spread":
        line = optional_float(row.get("bet_line"))
        if line is None:
            return "Unknown"
        if side == "home":
            diff = (home + line) - away
        elif side == "away":
            diff = (away + line) - home
        else:
            return "Unknown"
        if abs(diff) < 1e-9:
            return "Push"
        return "Win" if diff > 0 else "Loss"

    if market == "total":
        line = optional_float(row.get("bet_line"))
        if line is None:
            return "Unknown"
        total = home + away
        if abs(total - line) < 1e-9:
            return "Push"
        if (total > line and side == "over") or (total < line and side == "under"):
            return "Win"
        return "Loss"

    return "Unknown"


def compute_profits(row) -> tuple:
    """Returns (profit_unit, profit_kelly). profit_kelly is None if bet_stake_pct missing."""
    result = str(row.get("bet_result", "")).strip()
    odds   = row.get("bet_odds_american")
    decimal = american_to_decimal(odds)

    if result == "Push":
        return 0.0, 0.0
    if result not in ("Win", "Loss"):
        return None, None
    if decimal is None or decimal <= 1:
        return None, None

    stake = optional_float(row.get("bet_stake_pct"))

    if result == "Win":
        unit  = decimal - 1.0
        kelly = (stake * (decimal - 1.0)) if stake is not None else None
    else:
        unit  = -1.0
        kelly = -stake if stake is not None else None

    return unit, kelly


def load_picks_for_league(league: str, *, locked: bool = False) -> pd.DataFrame:
    folder_name = "locked_picks" if locked else "daily_picks"
    label = "locked picks" if locked else "picks"
    folder = SELECT_BASE / league / folder_name

    if not folder.exists():
        _log(f"[{league}] {label} folder missing: {folder}", "WARN")
        return pd.DataFrame()
    files = sorted(folder.glob("*.csv"))
    if not files:
        _log(f"[{league}] no {label} files in {folder}", "WARN")
        return pd.DataFrame()

    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            _log(f"[{league}] FAILED reading {label} file {fp.name}: {e}", "ERROR")

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    out = ensure_model_metadata(out, league, label)
    _log(f"[{league}] loaded {len(out)} {label} rows from {len(files)} files")
    return out


def load_scores_for_league(league: str) -> pd.DataFrame:
    folder = FINAL_SCORES_DIR / league
    if not folder.exists():
        _log(f"[{league}] scores folder missing: {folder}", "WARN")
        return pd.DataFrame()
    files = sorted(folder.glob("*.csv"))
    if not files:
        _log(f"[{league}] no score files in {folder}", "WARN")
        return pd.DataFrame()

    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            _log(f"[{league}] FAILED reading {fp.name}: {e}", "ERROR")

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    keep = [c for c in ["game_id", "home_score", "away_score"] if c in out.columns]
    if "game_id" not in keep:
        _log(f"[{league}] scores have no game_id column", "ERROR")
        return pd.DataFrame()

    out = out[keep].copy()
    out = out.drop_duplicates(subset=["game_id"], keep="last")
    _log(f"[{league}] loaded {len(out)} unique scored games from {len(files)} files")
    return out


def grade_league(league: str, summary: dict, *, locked: bool = False) -> None:
    upper = league.upper()
    label = "locked" if locked else "replay"
    summary_prefix = "locked_" if locked else ""

    if locked:
        out_dir = LOCKED_RESULTS_DIR / league
        unmatched_path = ERROR_DIR / f"{league}_locked_game_id_no_match.csv"
    else:
        out_dir = FINAL_SCORES_DIR / league / "graded"
        unmatched_path = ERROR_DIR / f"{league}_game_id_no_match.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.csv"):
        stale.unlink(missing_ok=True)

    picks = load_picks_for_league(league, locked=locked)
    if picks.empty:
        _log(f"[{league} {label}] no picks; skipping")
        return

    if "game_id" not in picks.columns:
        _log(f"[{league} {label}] picks missing game_id; cannot grade", "ERROR")
        summary["errors"] += 1
        return

    scores = load_scores_for_league(league)
    if scores.empty:
        _log(f"[{league} {label}] no scores; skipping")
        return

    merged = picks.merge(scores, on="game_id", how="left", suffixes=("", "_score"))

    for col in ("home_score", "away_score"):
        side = f"{col}_score"
        if side in merged.columns:
            merged[col] = merged[side].combine_first(merged.get(col))
            merged = merged.drop(columns=[side])

    matched_mask = merged["home_score"].notna() & merged["away_score"].notna()
    unmatched = merged.loc[~matched_mask].copy()
    matched   = merged.loc[ matched_mask].copy()

    if not unmatched.empty:
        no_match_cols = [c for c in [
            "game_id", "game_date", "home_team", "away_team",
            "league", "market_type", "bet_side"
        ] if c in unmatched.columns]
        unmatched_out = unmatched[no_match_cols].copy()
        unmatched_out["source"] = "locked_pick_no_score" if locked else "pick_no_score"
        unmatched_out.to_csv(unmatched_path, index=False)
        _log(
            f"[{league} {label}] {len(unmatched_out)} picks had no matching final score "
            f"-> {unmatched_path}"
        )
        summary[f"{summary_prefix}{league}_unmatched"] = len(unmatched_out)

    if matched.empty:
        _log(f"[{league} {label}] no picks matched a final score; nothing to grade")
        return

    matched["bet_result"] = matched.apply(determine_outcome, axis=1)

    profits = matched.apply(compute_profits, axis=1, result_type="expand")
    profits.columns = ["profit_unit", "profit_kelly"]
    matched = pd.concat([matched, profits], axis=1)

    # Keep distinct model/version rows distinct if multiple model cohorts ever
    # grade the same game/market/side.
    key_cols = [c for c in [
        "game_id", "market_type", "bet_side",
        "model_source", "model_version", "feature_version", "ensemble_version",
    ] if c in matched.columns]
    if key_cols:
        before = len(matched)
        matched = matched.drop_duplicates(subset=key_cols, keep="last")
        dropped = before - len(matched)
        if dropped:
            _log(f"[{league} {label}] deduped {dropped} duplicate graded rows")

    out_path = out_dir / f"{upper}_final.csv"
    matched.to_csv(out_path, index=False)

    n = len(matched)
    wins    = int((matched["bet_result"] == "Win").sum())
    losses  = int((matched["bet_result"] == "Loss").sum())
    pushes  = int((matched["bet_result"] == "Push").sum())
    unknown = int((matched["bet_result"] == "Unknown").sum())
    pnl_unit  = float(matched["profit_unit"].sum(skipna=True)) if "profit_unit" in matched.columns else 0.0
    pnl_kelly = float(matched["profit_kelly"].sum(skipna=True)) if "profit_kelly" in matched.columns else 0.0

    summary[f"{summary_prefix}{league}_graded"]    = n
    summary[f"{summary_prefix}{league}_wins"]      = wins
    summary[f"{summary_prefix}{league}_losses"]    = losses
    summary[f"{summary_prefix}{league}_pushes"]    = pushes
    summary[f"{summary_prefix}{league}_unknown"]   = unknown
    summary[f"{summary_prefix}{league}_pnl_unit"]  = round(pnl_unit, 4)
    summary[f"{summary_prefix}{league}_pnl_kelly"] = round(pnl_kelly, 4)

    _log(f"[{league} {label}] graded {n} bets -> {out_path}")
    _log(
        f"[{league} {label}] W/L/P/Unk = {wins}/{losses}/{pushes}/{unknown}  "
        f"PnL_unit={pnl_unit:+.2f}  PnL_kelly={pnl_kelly:+.4f}"
    )


# =========================
# MAIN
# =========================

def main():
    _init_log()
    _log(f"SELECT_BASE        : {SELECT_BASE}")
    _log(f"FINAL_SCORES_DIR   : {FINAL_SCORES_DIR}")
    _log(f"LOCKED_RESULTS_DIR : {LOCKED_RESULTS_DIR}")
    _log(f"ERROR_DIR          : {ERROR_DIR}")

    summary = {"errors": 0}
    try:
        try:
            lock_current_picks(summary)
        except Exception as e:
            _log(f"FATAL during current-pick lock: {e}\n{traceback.format_exc()}", "ERROR")
            summary["errors"] += 1

        for league in LEAGUES:
            try:
                grade_league(league, summary, locked=False)
            except Exception as e:
                _log(
                    f"[{league} replay] FATAL during grading: {e}\n{traceback.format_exc()}",
                    "ERROR",
                )
                summary["errors"] += 1

            try:
                grade_league(league, summary, locked=True)
            except Exception as e:
                _log(
                    f"[{league} locked] FATAL during grading: {e}\n{traceback.format_exc()}",
                    "ERROR",
                )
                summary["errors"] += 1

        _log("--- SUMMARY ---")
        for k in sorted(summary.keys()):
            _log(f"  {k:<24} : {summary[k]}")
        status = "SUCCESS" if summary["errors"] == 0 else "COMPLETED WITH ERRORS"
        _log(f"STATUS: {status}")

        print("Grading complete.")

    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
