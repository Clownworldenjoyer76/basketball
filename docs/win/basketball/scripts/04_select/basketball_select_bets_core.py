#!/usr/bin/env python3
# docs/win/basketball/scripts/04_select/basketball_select_bets.py
#
# Reads stage-3 EV/Kelly outputs and applies the per-league market EDGE from
# model_config.yaml first, then the per-league × per-market × per-side filters
# from markets.yaml. Picks bet(s) per game according to the configured
# selection_mode and pick_preference. Adds fractional-Kelly stake sizing.

import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime, UTC
from pathlib import Path

# ITEM 19 shared staking/uncertainty rules
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pandas as pd
import yaml

from staking_runtime import (
    KELLY_FRACTION,
    UNCERTAINTY_METHOD, UNCERTAINTY_VERSION,
    attach_candidate_uncertainty, requested_stake,
)

INPUT_DIR        = Path("docs/win/basketball/03_edges/ev_kelly")
SELECT_DIR       = Path("docs/win/basketball/04_select")
CONFIG_PATH      = Path("docs/win/basketball/config/markets.yaml")
MODEL_CONFIG_PATH = Path("docs/win/basketball/config/model_config.yaml")
ERROR_DIR        = Path("docs/win/basketball/errors/04_select")
LOG_FILE         = ERROR_DIR / "select_bets.txt"

SELECT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as markets_config_handle:
    CONFIG = yaml.safe_load(markets_config_handle) or {}

with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as model_config_handle:
    MODEL_CONFIG = yaml.safe_load(model_config_handle) or {}

ML_VS_SPREAD_TIEBREAK = str(CONFIG.get("ml_vs_spread_tiebreak", "ev")).strip().lower()
TIEBREAK_COL_MAP = {
    "ev":              "bet_ev",
    "kelly":           "bet_kelly",
    "edge_vs_market":  "bet_edge_vs_market",
}
if ML_VS_SPREAD_TIEBREAK not in TIEBREAK_COL_MAP:
    ML_VS_SPREAD_TIEBREAK = "ev"

LEAGUES = ["nba", "ncaam", "wnba"]
MARKETS = ["moneyline", "spread", "total"]

DEBUG_COUNTS = defaultdict(int)


# =========================
# MODEL EDGE CONFIG
# =========================

def load_model_edges() -> dict:
    leagues_cfg = MODEL_CONFIG.get("leagues")
    if not isinstance(leagues_cfg, dict):
        raise ValueError("model_config.yaml must contain a top-level 'leagues' mapping")

    edges = {}
    for league in LEAGUES:
        league_cfg = leagues_cfg.get(league)
        if not isinstance(league_cfg, dict):
            raise ValueError(f"Missing model config for league={league}")

        if str(league_cfg.get("status", "")).strip().lower() != "active":
            raise ValueError(f"League {league.upper()} is not active in model_config.yaml")

        edge_cfg = league_cfg.get("edge") or {}
        edges[league] = {}
        for market in MARKETS:
            value = edge_cfg.get(market)
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid model EDGE for league={league} market={market}: {value!r}"
                ) from exc
            edges[league][market] = value

    return edges


MODEL_EDGES = load_model_edges()


def model_edge(league: str, market: str) -> float:
    try:
        return MODEL_EDGES[league.lower()][market]
    except KeyError as exc:
        raise KeyError(f"No model EDGE: league={league!r} market={market!r}") from exc


# =========================
# LOGGING
# =========================

def _now():
    return datetime.now(UTC).isoformat()


def _log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


def _write_summary(summary: dict, per_file: list) -> None:
    lines = [
        "",
        "=" * 70,
        f"SUMMARY  {_now()}",
        "=" * 70,
        f"  files_processed       : {summary['files_processed']}",
        f"  candidate_selected    : {summary['total_candidates']}",
        f"  total_selected        : {summary['total_selected']}",
        f"  nba_bets              : {summary['nba_bets']}",
        f"  ncaam_bets            : {summary['ncaam_bets']}",
        f"  wnba_bets             : {summary['wnba_bets']}",
        f"  ml_vs_spread_dropped  : {summary['ml_vs_spread_dropped']}",
        f"  skipped               : {summary['skipped']}",
        f"  errors                : {summary['errors']}",
        f"  kelly_fraction        : {KELLY_FRACTION}",
        f"  uncertainty_method    : {UNCERTAINTY_METHOD}",
        f"  uncertainty_version   : {UNCERTAINTY_VERSION}",
        f"  ml_vs_spread_tiebreak : {ML_VS_SPREAD_TIEBREAK}",
        "",
        "--- Filter Reject Counts ---",
    ]
    for k, v in sorted(DEBUG_COUNTS.items()):
        lines.append(f"  {k:<28} : {v}")
    lines += [
        "",
        f"  {'file':<48} {'market':<10} {'league':>6} {'selected':>9} {'status':>10}",
    ]
    for pf in per_file:
        lines.append(
            f"  {pf['name']:<48} {pf['market']:<10} {pf['league']:>6} "
            f"{pf['selected']:>9} {pf['status']:>10}"
        )
    status = "SUCCESS" if summary["errors"] == 0 else "COMPLETED WITH ERRORS"
    lines += ["", f"STATUS: {status}", "=" * 70]
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================
# HELPERS
# =========================

def fv(x):
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except (TypeError, ValueError, OverflowError):
        return None


def in_any_band(value, bands):
    if value is None or bands is None:
        return False
    return any(lo <= value <= hi for lo, hi in bands)


def parse_date(s):
    try:
        return datetime.strptime(s, "%Y_%m_%d")
    except (TypeError, ValueError):
        return None


def date_ok(game_date, months, exclude_dow):
    if not months and not exclude_dow:
        return True
    dt = parse_date(game_date) if isinstance(game_date, str) else None
    if dt is None:
        return True
    if months and dt.month not in months:
        DEBUG_COUNTS["fail_month"] += 1
        return False
    if exclude_dow and dt.weekday() in exclude_dow:
        DEBUG_COUNTS["fail_dow"] += 1
        return False
    return True


def passes_filters(values: dict, scfg: dict, game_date: str) -> bool:
    if (
        "odds_bands" in scfg
        and not in_any_band(values.get("odds"), scfg["odds_bands"])
    ):
        DEBUG_COUNTS["fail_odds"] += 1
        return False
    if (
        "line_bands" in scfg
        and values.get("line") is not None
        and not in_any_band(values.get("line"), scfg["line_bands"])
    ):
        DEBUG_COUNTS["fail_line"] += 1
        return False
    if (
        "ev_bands" in scfg
        and not in_any_band(values.get("ev"), scfg["ev_bands"])
    ):
        DEBUG_COUNTS["fail_ev"] += 1
        return False
    if (
        "kelly_bands" in scfg
        and not in_any_band(values.get("kelly"), scfg["kelly_bands"])
    ):
        DEBUG_COUNTS["fail_kelly"] += 1
        return False
    if (
        "model_prob_bands" in scfg
        and not in_any_band(values.get("model_prob"), scfg["model_prob_bands"])
    ):
        DEBUG_COUNTS["fail_model_prob"] += 1
        return False
    if (
        "edge_vs_market_bands" in scfg
        and not in_any_band(
            values.get("edge_vs_market_pct"),
            scfg["edge_vs_market_bands"],
        )
    ):
        DEBUG_COUNTS["fail_edge_vs_market"] += 1
        return False
    if not date_ok(game_date, scfg.get("months", []) or [],
                   scfg.get("exclude_days_of_week", []) or []):
        return False
    return True


def passes_model_edge(ev, league: str, market: str) -> bool:
    threshold = model_edge(league, market)
    if ev is None or ev < threshold:
        DEBUG_COUNTS[f"fail_model_edge_{market}"] += 1
        return False
    return True


def pick(qualifying, preference):
    if not qualifying:
        return None
    metric = preference.get("metric", "ev")
    direction = preference.get("direction", "max")

    def key(c):
        v = c.get(metric)
        if v is None:
            return float("-inf") if direction == "max" else float("inf")
        return v

    return max(qualifying, key=key) if direction == "max" else min(qualifying, key=key)


def market_cfg(league, market_type):
    try:
        return CONFIG["markets"][league.lower()][market_type]
    except KeyError as e:
        raise KeyError(f"No config: league={league!r} market_type={market_type!r}") from e


def extract_date(filename):
    m = re.search(r"\d{4}_\d{2}_\d{2}", filename)
    return m.group(0) if m else None


def stake_pct(kelly, uncertainty_multiplier=1.0):
    # raw Kelly is a sizing input only; it is not a predicted win rate.
    _, _, requested = requested_stake(kelly, uncertainty_multiplier)
    return requested if requested > 0 else None


def clear_old_select_outputs() -> None:
    deleted_daily_picks = 0

    for league in LEAGUES:
        daily_pick_dir = SELECT_DIR / league / "daily_picks"
        if not daily_pick_dir.exists():
            continue

        for stale in daily_pick_dir.glob("*_selected.csv"):
            stale.unlink(missing_ok=True)
            deleted_daily_picks += 1
            _log(f"DELETED OLD DAILY PICK FILE: {stale}")

    DEBUG_COUNTS["deleted_old_daily_pick_files"] += deleted_daily_picks
    _log(f"Old basketball daily pick outputs deleted: daily_picks={deleted_daily_picks}")


def write_daily_pick_files(league: str, out_df: pd.DataFrame) -> None:
    daily_pick_dir = SELECT_DIR / league / "daily_picks"
    daily_pick_dir.mkdir(parents=True, exist_ok=True)

    if "game_date" not in out_df.columns:
        _log(f"Cannot write daily picks for {league}: missing game_date column", "ERROR")
        return

    for game_date, date_df in out_df.groupby("game_date", dropna=False):
        if pd.isna(game_date) or not str(game_date).strip():
            game_date = "unknown_date"
        out_path = daily_pick_dir / f"{game_date}_{league}_selected.csv"
        date_df.to_csv(out_path, index=False)
        _log(f"WROTE DAILY PICKS: {out_path} ({len(date_df)} rows)")


# =========================
# ML vs SPREAD RECONCILIATION
# =========================

def reconcile_ml_vs_spread(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0

    if "game_id" not in df.columns or "market_type" not in df.columns:
        _log("Cannot reconcile ML vs spread: missing 'game_id' or 'market_type' column",
             "WARN")
        return df, 0

    metric_col = TIEBREAK_COL_MAP.get(ML_VS_SPREAD_TIEBREAK, "bet_ev")
    if metric_col not in df.columns:
        _log(f"Cannot reconcile: tiebreak column {metric_col!r} missing", "WARN")
        return df, 0

    df = df.copy()
    df["_tiebreak_metric"] = pd.to_numeric(df[metric_col], errors="coerce")

    ml_mask     = df["market_type"] == "moneyline"
    spread_mask = df["market_type"] == "spread"

    if not ml_mask.any() or not spread_mask.any():
        df = df.drop(columns=["_tiebreak_metric"])
        return df, 0

    ml_best     = df.loc[ml_mask].groupby("game_id")["_tiebreak_metric"].max()
    spread_best = df.loc[spread_mask].groupby("game_id")["_tiebreak_metric"].max()

    conflict_games = ml_best.index.intersection(spread_best.index)
    if len(conflict_games) == 0:
        df = df.drop(columns=["_tiebreak_metric"])
        return df, 0

    drop_indices = []
    for gid in conflict_games:
        ml_val = ml_best.loc[gid]
        sp_val = spread_best.loc[gid]

        if pd.isna(ml_val) and pd.isna(sp_val):
            losing_market = "spread"
        elif pd.isna(ml_val):
            losing_market = "moneyline"
        elif pd.isna(sp_val):
            losing_market = "spread"
        else:
            losing_market = "spread" if ml_val >= sp_val else "moneyline"

        loss_mask = (df["game_id"] == gid) & (df["market_type"] == losing_market)
        drop_indices.extend(df.index[loss_mask].tolist())
        DEBUG_COUNTS[f"ml_vs_spread_dropped_{losing_market}"] += int(loss_mask.sum())

    n_dropped = len(drop_indices)
    if n_dropped:
        df = df.drop(index=drop_indices)
    df = df.drop(columns=["_tiebreak_metric"])
    return df, n_dropped


# =========================
# MARKET SIDE BUILDERS
# =========================

def build_ml_sides(row, league, game_date, cfg):
    sides = []
    for side in ("home", "away"):
        scfg = cfg[side]
        if not scfg.get("enabled", True):
            continue
        odds  = fv(row.get(f"{side}_dk_moneyline_american"))
        ev    = fv(row.get(f"{side}_ml_ev"))
        kelly = fv(row.get(f"{side}_ml_kelly"))
        mp    = fv(row.get(f"{side}_model_prob"))
        if mp is None:
            mp = fv(row.get(f"{side}_prob"))
        evm   = fv(row.get(f"{side}_ml_edge_vs_market_pct"))

        if not passes_model_edge(ev, league, "moneyline"):
            DEBUG_COUNTS["rejected_ml"] += 1
            continue

        values = {"odds": odds, "ev": ev, "kelly": kelly,
                  "model_prob": mp, "edge_vs_market_pct": evm}
        if passes_filters(values, scfg, game_date):
            sides.append({
                "side": side, "line": odds, "odds": odds,
                "ev": ev, "kelly": kelly,
                "model_prob": mp, "edge_vs_market": evm,
            })
        else:
            DEBUG_COUNTS["rejected_ml"] += 1
    return sides


def build_spread_sides(row, league, game_date, cfg):
    sides = []
    for side in ("home", "away"):
        scfg = cfg[side]
        if not scfg.get("enabled", True):
            continue

        (
            line,
            odds,
            ev,
            kelly,
            mp,
            evm,
        ) = (
            fv(
                row.get(
                    f"{side}_{suffix}"
                )
            )
            for suffix in (
                "spread",
                "dk_spread_american",
                "spread_ev",
                "spread_kelly",
                "spread_model_prob",
                "spread_edge_vs_market_pct",
            )
        )

        if not passes_model_edge(ev, league, "spread"):
            DEBUG_COUNTS["rejected_spread"] += 1
            continue

        values = {"odds": odds, "line": line, "ev": ev, "kelly": kelly,
                  "model_prob": mp, "edge_vs_market_pct": evm}
        if passes_filters(values, scfg, game_date):
            sides.append({
                "side": side, "line": line, "odds": odds,
                "ev": ev, "kelly": kelly,
                "model_prob": mp, "edge_vs_market": evm,
            })
        else:
            DEBUG_COUNTS["rejected_spread"] += 1
    return sides


def build_total_sides(row, league, game_date, cfg):
    sides = []
    line = fv(row.get("total"))
    for side in ("over", "under"):
        scfg = cfg[side]
        if not scfg.get("enabled", True):
            continue
        odds  = fv(row.get(f"dk_total_{side}_american"))
        ev    = fv(row.get(f"{side}_ev"))
        kelly = fv(row.get(f"{side}_kelly"))
        mp    = fv(row.get(f"{side}_model_prob"))
        evm   = fv(row.get(f"{side}_edge_vs_market_pct"))

        if not passes_model_edge(ev, league, "total"):
            DEBUG_COUNTS["rejected_total"] += 1
            continue

        values = {"odds": odds, "line": line, "ev": ev, "kelly": kelly,
                  "model_prob": mp, "edge_vs_market_pct": evm}
        if passes_filters(values, scfg, game_date):
            sides.append({
                "side": side, "line": line, "odds": odds,
                "ev": ev, "kelly": kelly,
                "model_prob": mp, "edge_vs_market": evm,
            })
        else:
            DEBUG_COUNTS["rejected_total"] += 1
    return sides


SIDE_BUILDERS = {
    "moneyline": build_ml_sides,
    "spread":    build_spread_sides,
    "total":     build_total_sides,
}


# =========================
# FILE PROCESSOR
# =========================

def process_file(file: Path, league: str, market_type: str):
    df = pd.read_csv(file)
    if df.empty:
        _log(f"EMPTY: {file.name}", "WARN")
        return pd.DataFrame(), 0

    cfg = market_cfg(league, market_type)
    if not cfg.get("enabled", True):
        _log(f"DISABLED in config: league={league} market={market_type}")
        return pd.DataFrame(), 0

    selection_mode = cfg.get("selection_mode", "pick_one")
    preference     = cfg.get("pick_preference", {"metric": "ev", "direction": "max"})
    builder        = SIDE_BUILDERS[market_type]
    file_date      = extract_date(file.name)

    _log(
        f"--- FILE: {file.name} league={league} market={market_type} "
        f"rows={len(df)} mode={selection_mode} model_edge={model_edge(league, market_type)}"
    )

    out_rows = []
    for _, row in df.iterrows():
        game_date = row.get("game_date") or file_date
        sides = builder(row, league, game_date, cfg)
        sides = [
            attach_candidate_uncertainty(row, market_type, side)
            for side in sides
        ]

        if not sides:
            continue

        if selection_mode == "all_qualifying":
            picks = sides
        else:
            p = pick(sides, preference)
            picks = [p] if p else []

        for sel in picks:
            DEBUG_COUNTS["selected"] += 1
            r = row.to_dict()
            r.update({
                "bet_side":           sel["side"],
                "bet_line":           sel["line"],
                "bet_odds_american":  sel["odds"],
                "bet_ev":             sel["ev"],
                "bet_raw_ev":         sel["raw_ev"],
                "bet_uncertainty_adjusted_ev": sel["uncertainty_adjusted_ev"],
                "bet_kelly":          sel["kelly"],
                "bet_raw_kelly":      sel["raw_kelly"],
                "bet_model_prob":     sel["model_prob"],
                "bet_adjusted_model_prob": sel["adjusted_model_prob"],
                "bet_edge_vs_market": sel["edge_vs_market"],
                "bet_uncertainty_multiplier": sel["uncertainty_multiplier"],
                "bet_uncertainty_points": sel["uncertainty_points"],
                "bet_signal_points": sel["signal_points"],
                "bet_requested_stake_pct": stake_pct(sel["raw_kelly"], sel["uncertainty_multiplier"]),
                "bet_stake_pct":      stake_pct(sel["raw_kelly"], sel["uncertainty_multiplier"]),
                "market_type":        market_type,
                "league_lower":       league,
                "league":             league.upper(),
                "game_date":          game_date,
            })
            out_rows.append(r)

    n = len(out_rows)
    _log(f"{file.name} | {n} selected from {len(df)} rows")
    return pd.DataFrame(out_rows), n


# =========================
# MAIN
# =========================

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== basketball select_bets RUN {_now()} ===\n")

    clear_old_select_outputs()

    summary = {
        "files_processed": 0, "total_candidates": 0, "total_selected": 0,
        "nba_bets": 0, "ncaam_bets": 0, "wnba_bets": 0,
        "ml_vs_spread_dropped": 0,
        "skipped": 0, "errors": 0,
    }
    per_file = []

    _log(f"INPUT_DIR : {INPUT_DIR}")
    _log(f"OUTPUT_DIR: {SELECT_DIR}")
    _log(f"MODEL_CONFIG: {MODEL_CONFIG_PATH}")
    _log(
        "staking.yaml: "
        f"kelly_fraction={KELLY_FRACTION} "
        f"uncertainty={UNCERTAINTY_METHOD}/{UNCERTAINTY_VERSION}; "
        "exposure-based selection filtering disabled"
    )
    _log(f"ml_vs_spread_tiebreak: {ML_VS_SPREAD_TIEBREAK}")
    for league in LEAGUES:
        _log(
            f"{league.upper()} model edges: "
            f"moneyline={model_edge(league, 'moneyline')} "
            f"spread={model_edge(league, 'spread')} "
            f"total={model_edge(league, 'total')}"
        )

    league_dfs = {lg: [] for lg in LEAGUES}

    try:
        for league in LEAGUES:
            for market in MARKETS:
                folder = INPUT_DIR / league / market
                if not folder.exists():
                    _log(f"INPUT FOLDER MISSING: {folder}", "WARN")
                    continue

                files = sorted(folder.glob("*.csv"))
                if not files:
                    _log(f"NO FILES: league={league} market={market}", "WARN")
                    continue

                for f in files:
                    pf = {"name": f.name, "market": market, "league": league.upper(),
                          "selected": 0, "status": "ok"}
                    try:
                        df, n = process_file(f, league, market)
                        pf["selected"] = n
                        summary["files_processed"] += 1
                        summary["total_candidates"] += n
                        if not df.empty:
                            league_dfs[league].append(df)
                    except KeyError as e:
                        _log(f"{f.name} CONFIG ERROR: {e}", "ERROR")
                        pf["status"] = "config_error"
                        summary["errors"] += 1
                    except Exception as e:
                        _log(f"{f.name} FAILED: {e}\n{traceback.format_exc()}", "ERROR")
                        pf["status"] = "error"
                        summary["errors"] += 1
                    per_file.append(pf)

        candidate_frames = [
            pd.concat(league_dfs[league], ignore_index=True)
            for league in LEAGUES
            if league_dfs[league]
        ]
        all_candidates = (
            pd.concat(candidate_frames, ignore_index=True)
            if candidate_frames
            else pd.DataFrame()
        )
        if not all_candidates.empty:
            all_candidates, n_dropped = reconcile_ml_vs_spread(all_candidates)
            summary["ml_vs_spread_dropped"] = n_dropped

        final_picks = all_candidates.copy()
        summary["total_candidates"] = len(all_candidates) + summary["ml_vs_spread_dropped"]
        summary["total_selected"] = len(final_picks)

        for league in LEAGUES:
            out_df = (
                final_picks[final_picks["league_lower"].astype(str).str.lower() == league].copy()
                if not final_picks.empty
                else pd.DataFrame()
            )
            summary[f"{league}_bets"] = len(out_df)
            if out_df.empty:
                _log(f"NO FINAL SELECTED ROWS FOR LEAGUE: {league}; daily pick files not written")
                continue
            write_daily_pick_files(league, out_df)

    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        summary["errors"] += 1

    _write_summary(summary, per_file)
    print("basketball select_bets complete.")


if __name__ == "__main__":
    main()
