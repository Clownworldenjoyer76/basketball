#!/usr/bin/env python3
# docs/win/basketball/scripts/05_final_scores/03_basketball_results_reports.py
#
# Reads work_{league}.csv from script 02 and produces summary/detail CSVs.
# All "by X" files share a canonical schema (see SCHEMA below). All sides files
# follow Option A naming: "_home_away_summary" for ML/spread, "_over_under_summary"
# for total.
#
# Per-market report policy for NBA, NCAAM, and WNBA:
#
# Moneyline:
#   by_ev
#   by_ev_home_away_summary
#   by_kelly
#   by_kelly_home_away_summary
#   by_odds
#   by_odds_home_away_summary
#   by_win_prob
#   by_win_prob_home_away_summary
#
# Spread:
#   by_ev
#   by_ev_home_away_summary
#   by_kelly
#   by_kelly_home_away_summary
#   by_odds
#   by_odds_home_away_summary
#   by_win_prob
#   by_win_prob_home_away_summary
#
# Total:
#   by_ev
#   by_ev_over_under_summary
#   by_kelly
#   by_kelly_over_under_summary
#   by_odds
#   by_odds_over_under_summary
#   by_side
#   by_side_over_under_summary
#   by_total_range
#   by_total_range_over_under_summary
#   by_win_prob
#   by_win_prob_over_under_summary
#
# Inputs:
#   docs/win/basketball/05_final_scores/work_nba.csv
#   docs/win/basketball/05_final_scores/work_ncaam.csv
#   docs/win/basketball/05_final_scores/work_wnba.csv
#
# Outputs (per league: nba, ncaam, wnba):
#   docs/win/basketball/05_final_scores/{league}_summary_overall.csv
#   docs/win/basketball/05_final_scores/{league}_summary_grand_total.csv
#   docs/win/basketball/05_final_scores/reports/{league}/{moneyline,spread,total,overview}/*.csv
#
# Log:
#   docs/win/basketball/errors/05_final_scores/03_basketball_results_reports.txt

from datetime import datetime, UTC
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

# =========================
# PATHS
# =========================

LEAGUES = ["nba", "ncaam", "wnba"]

BASE = Path("docs/win/basketball/05_final_scores")
REPORT_DIR = BASE / "reports"
ERROR_DIR = Path("docs/win/basketball/errors/05_final_scores")
LOG_FILE = ERROR_DIR / "03_basketball_results_reports.txt"

WORK_FILES = {
    league: BASE / f"work_{league}.csv"
    for league in LEAGUES
}

QUALITY_FILES = {
    league: BASE / f"quality_metrics_{league}.csv"
    for league in LEAGUES
}

ERROR_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# LOGGING
# =========================

RUN_STARTED = datetime.now(UTC)
RUN_COUNTS = {
    "warnings": 0,
    "errors": 0,
    "input_files": 0,
    "input_rows": 0,
    "output_files": 0,
    "output_rows": 0,
}

with open(LOG_FILE, "w", encoding="utf-8") as startup_log_handle:
    startup_log_handle.write("=== 03_basketball_results_reports ===\n")
    startup_log_handle.write(
        f"START_TIMESTAMP_UTC: {RUN_STARTED.isoformat()}\n"
    )


def log(level: str, message: str) -> None:
    timestamp = datetime.now(
        UTC
    ).isoformat()

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as log_handle:
        log_handle.write(
            f"{timestamp} | "
            f"{level} | "
            f"{message}\n"
        )


def _count_and_log(
    level: str,
    message: str,
) -> None:
    if level == "WARNING":
        RUN_COUNTS["warnings"] += 1
    elif level == "ERROR":
        RUN_COUNTS["errors"] += 1

    log(
        level,
        message,
    )


def warn(message: str) -> None:
    _count_and_log(
        "WARNING",
        message,
    )


def error(message: str) -> None:
    _count_and_log(
        "ERROR",
        message,
    )


def performance_rates(
    bets: int,
    units_flat: float,
    stake_total: float,
    units_kelly: float,
    wins: int,
    losses: int,
) -> tuple[float, float, float]:
    decided = (
        wins
        + losses
    )

    return (
        (
            units_flat / bets
            if bets > 0
            else np.nan
        ),
        (
            units_kelly / stake_total
            if stake_total > 0
            else np.nan
        ),
        (
            wins / decided
            if decided > 0
            else np.nan
        ),
    )

def log_input(
    path: Path,
    rows: int,
    exists: bool = True,
) -> None:
    RUN_COUNTS["input_files"] += 1
    RUN_COUNTS["input_rows"] += rows

    log(
        "INFO",
        (
            f"INPUT | file={path} | "
            f"exists={int(exists)} | rows={rows}"
        ),
    )


def write_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )

    rows = len(df)

    RUN_COUNTS["output_files"] += 1
    RUN_COUNTS["output_rows"] += rows

    log(
        "INFO",
        f"OUTPUT | file={path} | rows={rows}",
    )


def finish(status: str) -> None:
    ended = datetime.now(UTC)

    with open(LOG_FILE, "a", encoding="utf-8") as log_handle:
        log_handle.write(
            (
                f"INPUT_SUMMARY | "
                f"files={RUN_COUNTS['input_files']} | "
                f"rows={RUN_COUNTS['input_rows']}\n"
            )
        )

        log_handle.write(
            (
                f"OUTPUT_SUMMARY | "
                f"files={RUN_COUNTS['output_files']} | "
                f"rows={RUN_COUNTS['output_rows']}\n"
            )
        )

        log_handle.write(
            f"WARNING_COUNT: {RUN_COUNTS['warnings']}\n"
        )

        log_handle.write(
            f"ERROR_COUNT: {RUN_COUNTS['errors']}\n"
        )

        log_handle.write(
            f"END_TIMESTAMP_UTC: {ended.isoformat()}\n"
        )

        log_handle.write(
            f"STATUS: {status}\n"
        )


# =========================
# CANONICAL SCHEMA
# =========================

CANON_COLS_NO_SIDE = [
    "league",
    "market_type",
    "bucket_dimension",
    "bucket",
    "bets",
    "wins",
    "losses",
    "pushes",
    "total",
    "win_pct",
    "units_flat",
    "roi_flat",
    "units_kelly",
    "roi_kelly",
    "avg_ev",
    "avg_edge_vs_market_pp",
    "avg_kelly_pct",
    "avg_model_prob",
    "avg_odds_american",
]

CANON_COLS_WITH_SIDE = [
    "league",
    "market_type",
    "side_group",
    "bucket_dimension",
    "bucket",
    "bets",
    "wins",
    "losses",
    "pushes",
    "total",
    "win_pct",
    "units_flat",
    "roi_flat",
    "units_kelly",
    "roi_kelly",
    "avg_ev",
    "avg_edge_vs_market_pp",
    "avg_kelly_pct",
    "avg_model_prob",
    "avg_odds_american",
]


# =========================
# REPORT POLICY
# =========================

REPORT_BUCKETS = [
    ("ev", "ev_bucket"),
    ("kelly", "kelly_bucket"),
    ("odds", "odds_bucket"),
    ("win_prob", "model_prob_bucket"),
]


def allowed_market_reports(
    league: str,
    market_type: str,
) -> set[str]:
    if market_type == "moneyline":
        return {
            f"{league}_moneyline_by_ev.csv",
            f"{league}_moneyline_by_ev_home_away_summary.csv",
            f"{league}_moneyline_by_kelly.csv",
            f"{league}_moneyline_by_kelly_home_away_summary.csv",
            f"{league}_moneyline_by_odds.csv",
            f"{league}_moneyline_by_odds_home_away_summary.csv",
            f"{league}_moneyline_by_win_prob.csv",
            f"{league}_moneyline_by_win_prob_home_away_summary.csv",
        }

    if market_type == "spread":
        return {
            f"{league}_spread_by_ev.csv",
            f"{league}_spread_by_ev_home_away_summary.csv",
            f"{league}_spread_by_kelly.csv",
            f"{league}_spread_by_kelly_home_away_summary.csv",
            f"{league}_spread_by_odds.csv",
            f"{league}_spread_by_odds_home_away_summary.csv",
            f"{league}_spread_by_win_prob.csv",
            f"{league}_spread_by_win_prob_home_away_summary.csv",
        }

    if market_type == "total":
        return {
            f"{league}_total_by_ev.csv",
            f"{league}_total_by_ev_over_under_summary.csv",
            f"{league}_total_by_kelly.csv",
            f"{league}_total_by_kelly_over_under_summary.csv",
            f"{league}_total_by_odds.csv",
            f"{league}_total_by_odds_over_under_summary.csv",
            f"{league}_total_by_side.csv",
            f"{league}_total_by_side_over_under_summary.csv",
            f"{league}_total_by_total_range.csv",
            f"{league}_total_by_total_range_over_under_summary.csv",
            f"{league}_total_by_win_prob.csv",
            f"{league}_total_by_win_prob_over_under_summary.csv",
        }

    return set()


def cleanup_market_reports(
    league: str,
) -> None:
    for market_type in [
        "moneyline",
        "spread",
        "total",
    ]:
        market_dir = (
            REPORT_DIR
            / league
            / market_type
        )

        if not market_dir.exists():
            continue

        allowed_files = allowed_market_reports(
            league,
            market_type,
        )

        for path in market_dir.glob("*.csv"):
            if path.name in allowed_files:
                continue

            path.unlink()

            log(
                "INFO",
                f"REMOVED | file={path}",
            )


# =========================
# HELPERS
# =========================

def to_num(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def aggregate_block(
    df: pd.DataFrame,
    league: str,
    market_type: str | None,
    bucket_dimension: str,
    bucket_col: str,
    side_group_col: str | None = None,
) -> pd.DataFrame:
    """Build one canonical aggregation DataFrame."""

    if df.empty:
        cols = (
            CANON_COLS_WITH_SIDE
            if side_group_col
            else CANON_COLS_NO_SIDE
        )

        return pd.DataFrame(
            columns=cols
        )

    work = df.copy()

    for col in (
        "profit_unit",
        "profit_kelly",
        "bet_stake_pct",
        "bet_ev",
        "bet_edge_vs_market",
        "bet_kelly",
        "bet_model_prob",
        "bet_odds_american",
    ):
        if col in work.columns:
            work[col] = to_num(
                work[col]
            )

    if "bet_result" in work.columns:
        result = (
            work["bet_result"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
    else:
        result = pd.Series(
            [""] * len(work)
        )

    work["_is_win"] = (
        result == "win"
    ).astype(int)

    work["_is_loss"] = (
        result == "loss"
    ).astype(int)

    work["_is_push"] = (
        result == "push"
    ).astype(int)

    group_cols = [
        bucket_col
    ]

    if side_group_col:
        group_cols = [
            side_group_col,
            bucket_col,
        ]

    rows = []

    for keys, sub in work.groupby(
        group_cols,
        dropna=False,
        observed=True,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        wins = int(
            sub["_is_win"].sum()
        )

        losses = int(
            sub["_is_loss"].sum()
        )

        pushes = int(
            sub["_is_push"].sum()
        )

        bets = (
            wins
            + losses
            + pushes
        )

        total = bets

        units_flat = (
            float(
                sub["profit_unit"].sum(
                    skipna=True
                )
            )
            if "profit_unit" in sub.columns
            else 0.0
        )

        units_kelly = (
            float(
                sub["profit_kelly"].sum(
                    skipna=True
                )
            )
            if "profit_kelly" in sub.columns
            else 0.0
        )

        stake_total = (
            float(
                sub["bet_stake_pct"].sum(
                    skipna=True
                )
            )
            if "bet_stake_pct" in sub.columns
            else 0.0
        )

        (
            roi_flat,
            roi_kelly,
            win_pct,
        ) = performance_rates(
            bets,
            units_flat,
            stake_total,
            units_kelly,
            wins,
            losses,
        )

        avg_ev = (
            float(
                sub["bet_ev"].mean(
                    skipna=True
                )
            )
            if "bet_ev" in sub.columns
            else np.nan
        )

        avg_edgepp = (
            float(
                sub[
                    "bet_edge_vs_market"
                ].mean(
                    skipna=True
                )
            )
            if "bet_edge_vs_market"
            in sub.columns
            else np.nan
        )

        avg_kpct = (
            float(
                sub["bet_kelly"].mean(
                    skipna=True
                )
            )
            if "bet_kelly" in sub.columns
            else np.nan
        )

        avg_mp = (
            float(
                sub[
                    "bet_model_prob"
                ].mean(
                    skipna=True
                )
            )
            if "bet_model_prob"
            in sub.columns
            else np.nan
        )

        avg_odds = (
            float(
                sub[
                    "bet_odds_american"
                ].mean(
                    skipna=True
                )
            )
            if "bet_odds_american"
            in sub.columns
            else np.nan
        )

        if market_type is not None:
            resolved_market_type = (
                market_type
            )
        else:
            if "market_type" in sub.columns:
                market_values = (
                    sub["market_type"]
                    .astype(str)
                    .str.lower()
                    .unique()
                )
            else:
                market_values = []

            resolved_market_type = (
                market_values[0]
                if len(market_values) == 1
                else "mixed"
            )

        row = {
            "league": league,
            "market_type": resolved_market_type,
            "bucket_dimension": bucket_dimension,
            "bucket": (
                keys[-1]
                if len(keys) == 1
                else keys[1]
            ),
            "bets": bets,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "total": total,
            "win_pct": (
                round(
                    win_pct,
                    4,
                )
                if not pd.isna(win_pct)
                else np.nan
            ),
            "units_flat": round(
                units_flat,
                4,
            ),
            "roi_flat": (
                round(
                    roi_flat,
                    4,
                )
                if not pd.isna(roi_flat)
                else np.nan
            ),
            "units_kelly": round(
                units_kelly,
                6,
            ),
            "roi_kelly": (
                round(
                    roi_kelly,
                    4,
                )
                if not pd.isna(roi_kelly)
                else np.nan
            ),
            "avg_ev": (
                round(
                    avg_ev,
                    4,
                )
                if not pd.isna(avg_ev)
                else np.nan
            ),
            "avg_edge_vs_market_pp": (
                round(
                    avg_edgepp,
                    4,
                )
                if not pd.isna(avg_edgepp)
                else np.nan
            ),
            "avg_kelly_pct": (
                round(
                    avg_kpct,
                    4,
                )
                if not pd.isna(avg_kpct)
                else np.nan
            ),
            "avg_model_prob": (
                round(
                    avg_mp,
                    4,
                )
                if not pd.isna(avg_mp)
                else np.nan
            ),
            "avg_odds_american": (
                round(
                    avg_odds,
                    1,
                )
                if not pd.isna(avg_odds)
                else np.nan
            ),
        }

        if side_group_col:
            row["side_group"] = (
                keys[0]
            )

        rows.append(row)

    cols = (
        CANON_COLS_WITH_SIDE
        if side_group_col
        else CANON_COLS_NO_SIDE
    )

    out = pd.DataFrame(rows)

    if out.empty:
        return pd.DataFrame(
            columns=cols
        )

    sort_cols = [
        col
        for col in cols
        if col in (
            "side_group",
            "bucket",
        )
    ]

    out = (
        out[cols]
        .sort_values(
            by=sort_cols
        )
        .reset_index(
            drop=True
        )
    )

    return out


# =========================
# PER-MARKET REPORTS
# =========================

def side_suffix(
    market_type: str,
) -> str:
    if market_type == "total":
        return "over_under"

    return "home_away"


def write_market_reports(
    work_df: pd.DataFrame,
    league: str,
    market_type: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sub = work_df[
        work_df["market_type"]
        .astype(str)
        .str.lower()
        == market_type
    ].copy()

    if sub.empty:
        log(
            "INFO",
            (
                f"[{league} / {market_type}] "
                "no rows; skipping per-market reports"
            ),
        )
        return

    side_col = (
        "side_group"
        if "side_group" in sub.columns
        else None
    )

    suffix = side_suffix(
        market_type
    )

    # EV, Kelly, Odds, and Win Probability
    # are retained for all leagues and markets.
    for label, bucket_col in REPORT_BUCKETS:
        if bucket_col not in sub.columns:
            warn(
                (
                    f"[{league} / {market_type}] "
                    f"missing bucket column "
                    f"{bucket_col}; skipping {label}"
                )
            )
            continue

        aggregate = aggregate_block(
            sub,
            league=league,
            market_type=market_type,
            bucket_dimension=label,
            bucket_col=bucket_col,
        )

        write_csv(
            aggregate,
            out_dir
            / (
                f"{league}_{market_type}_"
                f"by_{label}.csv"
            ),
        )

        if side_col:
            side_aggregate = aggregate_block(
                sub,
                league=league,
                market_type=market_type,
                bucket_dimension=label,
                bucket_col=bucket_col,
                side_group_col=side_col,
            )

            write_csv(
                side_aggregate,
                out_dir
                / (
                    f"{league}_{market_type}_"
                    f"by_{label}_{suffix}_summary.csv"
                ),
            )

    # Explicit side reports are retained only
    # for totals.
    if (
        market_type == "total"
        and side_col
    ):
        aggregate = aggregate_block(
            sub,
            league=league,
            market_type=market_type,
            bucket_dimension="side",
            bucket_col=side_col,
        )

        write_csv(
            aggregate,
            out_dir
            / (
                f"{league}_{market_type}_"
                "by_side.csv"
            ),
        )

        side_aggregate = (
            aggregate.copy()
        )

        if not side_aggregate.empty:
            side_aggregate.insert(
                2,
                "side_group",
                side_aggregate["bucket"],
            )

            side_aggregate = (
                side_aggregate[
                    CANON_COLS_WITH_SIDE
                ]
            )
        else:
            side_aggregate = pd.DataFrame(
                columns=CANON_COLS_WITH_SIDE
            )

        write_csv(
            side_aggregate,
            out_dir
            / (
                f"{league}_{market_type}_"
                f"by_side_{suffix}_summary.csv"
            ),
        )

    # Total-range reports are retained only
    # for totals.
    if (
        market_type == "total"
        and "total_bucket" in sub.columns
    ):
        aggregate = aggregate_block(
            sub,
            league=league,
            market_type=market_type,
            bucket_dimension="total_range",
            bucket_col="total_bucket",
        )

        write_csv(
            aggregate,
            out_dir
            / (
                f"{league}_{market_type}_"
                "by_total_range.csv"
            ),
        )

        if side_col:
            side_aggregate = aggregate_block(
                sub,
                league=league,
                market_type=market_type,
                bucket_dimension="total_range",
                bucket_col="total_bucket",
                side_group_col=side_col,
            )

            write_csv(
                side_aggregate,
                out_dir
                / (
                    f"{league}_{market_type}_"
                    f"by_total_range_{suffix}_summary.csv"
                ),
            )


# =========================
# OVERVIEW
# =========================

def write_overview(
    work_df: pd.DataFrame,
    league: str,
    overview_dir: Path,
) -> None:
    overview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if work_df.empty:
        log(
            "INFO",
            (
                f"[{league}] "
                "no rows; skipping overview"
            ),
        )
        return

    by_market = []

    for market_type, sub in work_df.groupby(
        "market_type",
        dropna=False,
        observed=True,
    ):
        aggregate = aggregate_block(
            sub,
            league=league,
            market_type=str(
                market_type
            ).lower(),
            bucket_dimension="market_type",
            bucket_col="market_type",
        )

        by_market.append(
            aggregate
        )

    if by_market:
        output = pd.concat(
            by_market,
            ignore_index=True,
        )

        write_csv(
            output,
            overview_dir
            / (
                f"{league}_"
                "summary_by_market.csv"
            ),
        )

    if "side_group" in work_df.columns:
        aggregate = aggregate_block(
            work_df,
            league=league,
            market_type=None,
            bucket_dimension="side_group",
            bucket_col="side_group",
        )

        write_csv(
            aggregate,
            overview_dir
            / (
                f"{league}_"
                "summary_by_side_group.csv"
            ),
        )

    if "game_date" in work_df.columns:
        aggregate = aggregate_block(
            work_df,
            league=league,
            market_type=None,
            bucket_dimension="game_date",
            bucket_col="game_date",
        )

        write_csv(
            aggregate,
            overview_dir
            / (
                f"{league}_"
                "summary_by_date.csv"
            ),
        )

    log_cols = [
        "game_date",
        "league",
        "market_type",
        "side_group",
        "model_source",
        "model_version",
        "feature_version",
        "ensemble_version",
        "home_team",
        "away_team",
        "bet_side",
        "bet_line",
        "bet_odds_american",
        "bet_ev",
        "bet_edge_vs_market",
        "bet_kelly",
        "bet_model_prob",
        "bet_stake_pct",
        "ev_bucket",
        "edge_vs_market_bucket",
        "kelly_bucket",
        "odds_bucket",
        "model_prob_bucket",
        "spread_bucket",
        "total_bucket",
        "dow_bucket",
        "month_bucket",
        "bet_result",
        "profit_unit",
        "profit_kelly",
        "probability_outcome",
        "brier_component",
        "log_loss_component",
        "actual_margin",
        "projected_margin",
        "margin_error",
        "actual_total",
        "projected_total",
        "total_error",
        "closing_observed_at_utc",
        "scheduled_tipoff_utc",
        "minutes_before_tipoff",
        "closing_line",
        "closing_market_prob",
        "entry_market_prob",
        "clv",
        "clv_units",
        "model_vs_market_prob_pp",
        "model_vs_market_line",
    ]

    existing = [
        col
        for col in log_cols
        if col in work_df.columns
    ]

    write_csv(
        work_df[existing],
        overview_dir
        / f"{league}_bet_log.csv",
    )

    overall = build_summary_overall(
        work_df,
        league,
    )

    write_csv(
        overall,
        overview_dir
        / (
            f"{league}_"
            "summary_overall.csv"
        ),
    )


# =========================
# TOP-LEVEL SUMMARIES
# =========================

def build_summary_overall(
    work_df: pd.DataFrame,
    league: str,
) -> pd.DataFrame:
    rows = []

    if work_df.empty:
        return pd.DataFrame(
            columns=[
                "league",
                "market_type",
                "Win",
                "Loss",
                "Push",
                "Total",
                "Win_Pct",
            ]
        )

    for market_type in [
        "moneyline",
        "spread",
        "total",
    ]:
        sub = work_df[
            work_df["market_type"]
            .astype(str)
            .str.lower()
            == market_type
        ]

        if "bet_result" in sub.columns:
            result = (
                sub["bet_result"]
                .astype(str)
                .str.strip()
                .str.lower()
            )
        else:
            result = pd.Series(
                [""] * len(sub)
            )

        wins = int(
            (result == "win").sum()
        )

        losses = int(
            (result == "loss").sum()
        )

        pushes = int(
            (result == "push").sum()
        )

        total = (
            wins
            + losses
            + pushes
        )

        win_pct = (
            round(
                wins / (wins + losses),
                4,
            )
            if (wins + losses) > 0
            else np.nan
        )

        rows.append({
            "league": league.upper(),
            "market_type": market_type,
            "Win": wins,
            "Loss": losses,
            "Push": pushes,
            "Total": total,
            "Win_Pct": win_pct,
        })

    return pd.DataFrame(rows)


def build_summary_grand_total(
    work_df: pd.DataFrame,
    league: str,
) -> pd.DataFrame:
    if work_df.empty:
        return pd.DataFrame([
            {
                "league": league.upper(),
                "bets": 0,
                "wins": 0,
                "losses": 0,
                "pushes": 0,
                "total": 0,
                "win_pct": np.nan,
                "units_flat": 0.0,
                "roi_flat": np.nan,
                "units_kelly": 0.0,
                "roi_kelly": np.nan,
                "avg_ev": np.nan,
                "avg_edge_vs_market_pp": np.nan,
                "avg_kelly_pct": np.nan,
                "avg_model_prob": np.nan,
                "avg_odds_american": np.nan,
            }
        ])

    result = (
        work_df["bet_result"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    wins = int(
        (result == "win").sum()
    )

    losses = int(
        (result == "loss").sum()
    )

    pushes = int(
        (result == "push").sum()
    )

    bets = (
        wins
        + losses
        + pushes
    )

    units_flat = float(
        to_num(
            work_df.get(
                "profit_unit",
                pd.Series(dtype=float),
            )
        ).sum(
            skipna=True
        )
    )

    units_kelly = float(
        to_num(
            work_df.get(
                "profit_kelly",
                pd.Series(dtype=float),
            )
        ).sum(
            skipna=True
        )
    )

    stake_total = float(
        to_num(
            work_df.get(
                "bet_stake_pct",
                pd.Series(dtype=float),
            )
        ).sum(
            skipna=True
        )
    )

    (
        roi_flat,
        roi_kelly,
        win_pct,
    ) = performance_rates(
        bets,
        units_flat,
        stake_total,
        units_kelly,
        wins,
        losses,
    )

    avg_ev = (
        round(
            float(
                to_num(
                    work_df.get(
                        "bet_ev",
                        pd.Series(dtype=float),
                    )
                ).mean(
                    skipna=True
                )
            ),
            4,
        )
        if "bet_ev" in work_df.columns
        else np.nan
    )

    avg_edge_vs_market = (
        round(
            float(
                to_num(
                    work_df.get(
                        "bet_edge_vs_market",
                        pd.Series(dtype=float),
                    )
                ).mean(
                    skipna=True
                )
            ),
            4,
        )
        if "bet_edge_vs_market"
        in work_df.columns
        else np.nan
    )

    avg_kelly = (
        round(
            float(
                to_num(
                    work_df.get(
                        "bet_kelly",
                        pd.Series(dtype=float),
                    )
                ).mean(
                    skipna=True
                )
            ),
            4,
        )
        if "bet_kelly" in work_df.columns
        else np.nan
    )

    avg_model_prob = (
        round(
            float(
                to_num(
                    work_df.get(
                        "bet_model_prob",
                        pd.Series(dtype=float),
                    )
                ).mean(
                    skipna=True
                )
            ),
            4,
        )
        if "bet_model_prob"
        in work_df.columns
        else np.nan
    )

    avg_odds = (
        round(
            float(
                to_num(
                    work_df.get(
                        "bet_odds_american",
                        pd.Series(dtype=float),
                    )
                ).mean(
                    skipna=True
                )
            ),
            1,
        )
        if "bet_odds_american"
        in work_df.columns
        else np.nan
    )

    return pd.DataFrame([
        {
            "league": league.upper(),
            "bets": bets,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "total": bets,
            "win_pct": (
                round(
                    win_pct,
                    4,
                )
                if not pd.isna(win_pct)
                else np.nan
            ),
            "units_flat": round(
                units_flat,
                4,
            ),
            "roi_flat": (
                round(
                    roi_flat,
                    4,
                )
                if not pd.isna(roi_flat)
                else np.nan
            ),
            "units_kelly": round(
                units_kelly,
                6,
            ),
            "roi_kelly": (
                round(
                    roi_kelly,
                    4,
                )
                if not pd.isna(roi_kelly)
                else np.nan
            ),
            "avg_ev": avg_ev,
            "avg_edge_vs_market_pp": (
                avg_edge_vs_market
            ),
            "avg_kelly_pct": avg_kelly,
            "avg_model_prob": avg_model_prob,
            "avg_odds_american": avg_odds,
        }
    ])


# =========================
# MODEL QUALITY REPORTS
# =========================

def write_quality_reports(
    league: str,
) -> None:
    source = QUALITY_FILES[
        league
    ]

    quality_dir = (
        REPORT_DIR
        / league
        / "quality"
    )

    quality_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not source.exists():
        log_input(
            source,
            0,
            exists=False,
        )

        warn(
            (
                f"[{league}] "
                f"quality metrics missing: {source}"
            )
        )

        return

    try:
        quality = pd.read_csv(
            source
        )

    except Exception as exc:
        log_input(
            source,
            0,
            exists=True,
        )

        warn(
            (
                f"[{league}] unable to read "
                f"quality metrics {source}: {exc}"
            )
        )

        return

    log_input(
        source,
        len(quality),
        exists=True,
    )

    write_csv(
        quality,
        quality_dir
        / (
            f"{league}_"
            "model_quality_all.csv"
        ),
    )

    scope_files = {
        "league":
            f"{league}_model_quality_league.csv",
        "market":
            f"{league}_model_quality_by_market.csv",
        "model_source":
            f"{league}_model_quality_by_model_source.csv",
        "model_version":
            f"{league}_model_quality_by_model_version.csv",
        "market_model_version":
            (
                f"{league}_"
                "model_quality_by_market_model_version.csv"
            ),
    }

    for scope, filename in scope_files.items():
        if "scope" not in quality.columns:
            warn(
                (
                    f"[{league}] "
                    "quality metrics missing scope column"
                )
            )
            break

        subset = quality[
            quality["scope"].astype(str)
            == scope
        ].copy()

        write_csv(
            subset,
            quality_dir / filename,
        )


# =========================
# RUN
# =========================

def run_one(
    league: str,
) -> None:
    # Remove obsolete per-market CSVs before processing.
    # This also handles missing/empty work files so stale
    # reports cannot remain from an earlier run.
    cleanup_market_reports(
        league
    )

    work_path = WORK_FILES[
        league
    ]

    if not work_path.exists():
        log_input(
            work_path,
            0,
            exists=False,
        )

        warn(
            (
                f"[{league}] "
                f"missing work file: {work_path}"
            )
        )

        return

    work = pd.read_csv(
        work_path
    )

    log_input(
        work_path,
        len(work),
        exists=True,
    )

    # Publish model-quality outputs even when
    # the betting work file is empty.
    write_quality_reports(
        league
    )

    if work.empty:
        warn(
            (
                f"[{league}] "
                "empty work file; skipping betting reports"
            )
        )
        return

    if "market_type" in work.columns:
        work["market_type"] = (
            work["market_type"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    if "side_group" in work.columns:
        work["side_group"] = (
            work["side_group"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    write_csv(
        build_summary_overall(
            work,
            league,
        ),
        BASE
        / (
            f"{league}_"
            "summary_overall.csv"
        ),
    )

    write_csv(
        build_summary_grand_total(
            work,
            league,
        ),
        BASE
        / (
            f"{league}_"
            "summary_grand_total.csv"
        ),
    )

    for market_type in [
        "moneyline",
        "spread",
        "total",
    ]:
        out_dir = (
            REPORT_DIR
            / league
            / market_type
        )

        write_market_reports(
            work,
            league,
            market_type,
            out_dir,
        )

    # Enforce the allowlist again after generation.
    cleanup_market_reports(
        league
    )

    overview_dir = (
        REPORT_DIR
        / league
        / "overview"
    )

    write_overview(
        work,
        league,
        overview_dir,
    )

    log(
        "INFO",
        (
            f"[{league}] reports written under "
            f"{REPORT_DIR / league}"
        ),
    )


def run() -> None:
    BASE.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for league in LEAGUES:
        run_one(
            league
        )

    log(
        "INFO",
        "Basketball reports complete.",
    )


def main() -> None:
    status = "FAILED"

    try:
        run()
        status = "SUCCESS"

    except Exception as exc:
        error(
            (
                "Unhandled exception: "
                f"{type(exc).__name__}: {exc}"
            )
        )

        trace = traceback.format_exc()

        with open(
            LOG_FILE,
            "a",
            encoding="utf-8",
        ) as log_handle:
            log_handle.write(
                trace
            )

            if not trace.endswith("\n"):
                log_handle.write(
                    "\n"
                )

        raise

    finally:
        finish(
            status
        )


if __name__ == "__main__":
    main()
