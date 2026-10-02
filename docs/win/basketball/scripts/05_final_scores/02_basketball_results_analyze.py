#!/usr/bin/env python3
# docs/win/basketball/scripts/05_final_scores/02_basketball_results_analyze.py
#
# Reads graded bet files from script 01 and adds derived columns + bucket
# labels for downstream reporting. Uses the bet_* columns already on the row
# (bet_ev, bet_edge_vs_market, bet_kelly, bet_model_prob, bet_odds_american,
# bet_line, bet_stake_pct) — does NOT look up home/away columns.
#
# Inputs:
#   docs/win/basketball/05_final_scores/results/nba/graded/NBA_final.csv
#   docs/win/basketball/05_final_scores/results/ncaam/graded/NCAAM_final.csv
#   docs/win/basketball/05_final_scores/results/wnba/graded/WNBA_final.csv
#
# Outputs:
#   docs/win/basketball/05_final_scores/work_nba.csv
#   docs/win/basketball/05_final_scores/work_ncaam.csv
#   docs/win/basketball/05_final_scores/work_wnba.csv
#
# Log:
#   docs/win/basketball/errors/05_final_scores/02_basketball_results_analyze.txt
#
# Buckets added per row:
#   ev_bucket               (signed; pulls from bet_ev)
#   edge_vs_market_bucket   (signed; pulls from bet_edge_vs_market in pp)
#   kelly_bucket            (pulls from bet_kelly)
#   model_prob_bucket       (pulls from bet_model_prob)
#   odds_bucket             (pulls from bet_odds_american; meaningful for ML)
#   spread_bucket           (pulls from bet_line; meaningful for spread)
#   total_bucket            (pulls from bet_line; meaningful for total)
#   dow_bucket              (Mon..Sun from game_date)
#   month_bucket            (Jan..Dec from game_date)
#
# profit_unit and profit_kelly carry through unchanged from script 01.

from datetime import datetime, UTC
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

# =========================
# PATHS
# =========================

LEAGUES = ["nba", "ncaam", "wnba"]

BASE       = Path("docs/win/basketball")
INPUT_DIR  = BASE / "05_final_scores/results"
OUTPUT_DIR = BASE / "05_final_scores"
ERROR_DIR  = BASE / "errors/05_final_scores"
LOG_FILE   = ERROR_DIR / "02_basketball_results_analyze.txt"
CLOSING_DIR = BASE / "05_final_scores/closing_lines"

MIN_PROB_SAMPLE = 10
MIN_GAME_ERROR_SAMPLE = 5
MIN_CLV_SAMPLE = 5
MIN_DISAGREEMENT_SAMPLE = 5

QUALITY_COLUMNS = [
    "scope", "league", "market_type", "model_source", "model_version",
    "rows", "probability_n", "brier_score", "log_loss", "calibration_error",
    "margin_n", "margin_mae", "margin_rmse",
    "total_n", "total_mae", "total_rmse",
    "clv_n", "avg_clv", "clv_units",
    "prob_disagreement_n", "avg_model_vs_market_prob_pp",
    "mean_abs_model_vs_market_prob_pp",
    "line_disagreement_n", "avg_model_vs_market_line",
    "mean_abs_model_vs_market_line",
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
    startup_log_handle.write("=== 02_basketball_results_analyze ===\n")
    startup_log_handle.write(f"START_TIMESTAMP_UTC: {RUN_STARTED.isoformat()}\n")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def log(level: str, message: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level} | {message}\n")


def warn(message: str) -> None:
    RUN_COUNTS["warnings"] += 1
    log("WARNING", message)


def error(message: str) -> None:
    RUN_COUNTS["errors"] += 1
    log("ERROR", message)


def log_input(path: Path, rows: int, exists: bool = True) -> None:
    RUN_COUNTS["input_files"] += 1
    RUN_COUNTS["input_rows"] += rows
    log("INFO", f"INPUT | file={path} | exists={int(exists)} | rows={rows}")


def log_output(path: Path, rows: int) -> None:
    RUN_COUNTS["output_files"] += 1
    RUN_COUNTS["output_rows"] += rows
    log("INFO", f"OUTPUT | file={path} | rows={rows}")


def finish(status: str) -> None:
    ended = datetime.now(UTC)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"INPUT_SUMMARY | files={RUN_COUNTS['input_files']} | rows={RUN_COUNTS['input_rows']}\n")
        f.write(f"OUTPUT_SUMMARY | files={RUN_COUNTS['output_files']} | rows={RUN_COUNTS['output_rows']}\n")
        f.write(f"WARNING_COUNT: {RUN_COUNTS['warnings']}\n")
        f.write(f"ERROR_COUNT: {RUN_COUNTS['errors']}\n")
        f.write(f"END_TIMESTAMP_UTC: {ended.isoformat()}\n")
        f.write(f"STATUS: {status}\n")


# =========================
# HELPERS
# =========================

def to_float(value):
    try:
        if value is None or pd.isna(value):
            return pd.NA
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return pd.NA


def build_side_group(row):
    market_type = str(row.get("market_type", "")).strip().lower()
    bet_side    = str(row.get("bet_side", "")).strip().lower()

    if market_type in {"moneyline", "spread"}:
        if bet_side == "home":
            return "HOME"
        if bet_side == "away":
            return "AWAY"

    if market_type == "total":
        if bet_side == "over":
            return "OVER"
        if bet_side == "under":
            return "UNDER"

    return ""


# ---- bucket functions ----

def ev_bucket(value):
    """Signed buckets on bet_ev (decimal). Negative bucket exists because
    ev can technically be <= 0 if a row leaked through filters."""
    v = to_float(value)
    if pd.isna(v):
        return "UNBUCKETED"
    if v < 0:
        return "<0"
    if v < 0.01:
        return "0.00_to_0.0099"
    if v < 0.02:
        return "0.01_to_0.0199"
    if v < 0.03:
        return "0.02_to_0.0299"
    if v < 0.04:
        return "0.03_to_0.0399"
    if v < 0.05:
        return "0.04_to_0.0499"
    if v < 0.075:
        return "0.05_to_0.0749"
    if v < 0.10:
        return "0.075_to_0.0999"
    return "0.10_plus"


def edge_vs_market_bucket(value):
    """bet_edge_vs_market is in percentage points (model_prob - market_prob) * 100.
    Signed; negative means model thinks the bet is worse than the market does."""
    v = to_float(value)
    if pd.isna(v):
        return "UNBUCKETED"
    if v < -10:
        return "below_neg10pp"
    if v < -5:
        return "neg10_to_neg5pp"
    if v < -2:
        return "neg5_to_neg2pp"
    if v < 0:
        return "neg2_to_0pp"
    if v < 1:
        return "0_to_1pp"
    if v < 2:
        return "1_to_2pp"
    if v < 3:
        return "2_to_3pp"
    if v < 5:
        return "3_to_5pp"
    if v < 7:
        return "5_to_7pp"
    if v < 10:
        return "7_to_10pp"
    return "10pp_plus"


def kelly_bucket(value):
    v = to_float(value)
    if pd.isna(v):
        return "UNBUCKETED"
    if v < 0:
        return "<0"
    if v < 0.025:
        return "0.0_to_2.5pct"
    if v < 0.05:
        return "2.5_to_5pct"
    if v < 0.10:
        return "5_to_10pct"
    if v < 0.20:
        return "10_to_20pct"
    return "20pct_plus"


def model_prob_bucket(value):
    v = to_float(value)
    if pd.isna(v):
        return "UNBUCKETED"
    if v < 0.20:
        return "below_0.20"
    if v < 0.30:
        return "0.20_to_0.30"
    if v < 0.40:
        return "0.30_to_0.40"
    if v < 0.50:
        return "0.40_to_0.50"
    if v < 0.60:
        return "0.50_to_0.60"
    if v < 0.70:
        return "0.60_to_0.70"
    if v < 0.80:
        return "0.70_to_0.80"
    if v < 0.90:
        return "0.80_to_0.90"
    return "0.90_plus"


def odds_bucket(value):
    v = to_float(value)
    if pd.isna(v):
        return "UNBUCKETED"
    if v <= -200:
        return "minus_200_or_lower"
    if v <= -150:
        return "minus_199_to_minus_150"
    if v <= -125:
        return "minus_149_to_minus_125"
    if v <= -110:
        return "minus_124_to_minus_110"
    if v <= -101:
        return "minus_109_to_minus_101"
    if v <= 100:
        return "minus_100_to_plus_100"
    if v <= 125:
        return "plus_101_to_plus_125"
    if v <= 150:
        return "plus_126_to_plus_150"
    if v <= 200:
        return "plus_151_to_plus_200"
    return "plus_201_or_higher"


def spread_bucket(value):
    v = to_float(value)
    if pd.isna(v):
        return "UNBUCKETED"
    abs_v = abs(float(v))
    if abs_v < 1:    return "0.0_to_0.9"
    if abs_v < 2:    return "1.0_to_1.9"
    if abs_v < 3:    return "2.0_to_2.9"
    if abs_v < 4:    return "3.0_to_3.9"
    if abs_v < 5:    return "4.0_to_4.9"
    if abs_v < 6:    return "5.0_to_5.9"
    if abs_v < 7:    return "6.0_to_6.9"
    if abs_v < 8:    return "7.0_to_7.9"
    if abs_v < 9:    return "8.0_to_8.9"
    if abs_v < 10:   return "9.0_to_9.9"
    if abs_v < 12:   return "10.0_to_11.9"
    if abs_v < 15:   return "12.0_to_14.9"
    return "15.0_plus"


def total_bucket(value):
    v = to_float(value)
    if pd.isna(v):
        return "UNBUCKETED"
    start = int(float(v) // 5) * 5
    end = start + 4.9
    return f"{start}_to_{end:.1f}"


# ---- date buckets ----

DOW_NAMES = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
MONTH_NAMES = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
               "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def parse_date(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s).strip()
    if not s:
        return None
    # game_date in your pipeline is YYYY_MM_DD; tolerate a couple of common variants
    for fmt in ("%Y_%m_%d", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def dow_bucket(value):
    dt = parse_date(value)
    if dt is None:
        return "UNBUCKETED"
    return DOW_NAMES[dt.weekday()]


def month_bucket(value):
    dt = parse_date(value)
    if dt is None:
        return "UNBUCKETED"
    return MONTH_NAMES[dt.month - 1]



# =========================
# MODEL QUALITY / CLOSING LINES
# =========================

MODEL_METADATA_COLUMNS = [
    "model_source", "model_version", "feature_version",
    "ensemble_version", "bet_model_prob",
]

CLOSING_VALUE_COLUMNS = [
    "sportsbook_provider", "snapshot_file", "closing_observed_at_utc",
    "scheduled_tipoff_utc", "minutes_before_tipoff",
    "closing_home_spread", "closing_away_spread", "closing_total",
    "closing_home_ml_american", "closing_away_ml_american",
    "closing_home_spread_american", "closing_away_spread_american",
    "closing_over_american", "closing_under_american",
    "closing_home_ml_decimal", "closing_away_ml_decimal",
    "closing_home_spread_decimal", "closing_away_spread_decimal",
    "closing_over_decimal", "closing_under_decimal",
    "closing_home_market_prob", "closing_away_market_prob",
    "closing_home_spread_market_prob", "closing_away_spread_market_prob",
    "closing_over_market_prob", "closing_under_market_prob",
]


def normalize_game_id(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def load_closing_lines(league: str) -> pd.DataFrame:
    folder = CLOSING_DIR / league
    files = sorted(folder.glob("*_closing_lines.csv")) if folder.exists() else []
    if not files:
        warn(f"[{league.upper()}] no closing-line files in {folder}; CLV will be unavailable")
        return pd.DataFrame()

    frames = []
    for path in files:
        try:
            frame = pd.read_csv(path)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            warn(f"[{league.upper()}] unable to read closing-line file {path}: {exc}")

    if not frames:
        return pd.DataFrame()

    closing = pd.concat(frames, ignore_index=True)
    if "game_id" not in closing.columns or "market_type" not in closing.columns:
        warn(f"[{league.upper()}] closing-line data missing game_id/market_type")
        return pd.DataFrame()

    closing["_join_game_id"] = closing["game_id"].apply(normalize_game_id)
    closing["market_type"] = closing["market_type"].astype(str).str.strip().str.lower()
    closing = closing.drop_duplicates(["_join_game_id", "market_type"], keep="last")
    return closing


def attach_closing_lines(work: pd.DataFrame, league: str) -> pd.DataFrame:
    out = work.copy()
    for col in CLOSING_VALUE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    closing = load_closing_lines(league)
    if closing.empty or "game_id" not in out.columns or "market_type" not in out.columns:
        return out

    out["_join_game_id"] = out["game_id"].apply(normalize_game_id)
    wanted = ["_join_game_id", "market_type"] + [
        c for c in CLOSING_VALUE_COLUMNS if c in closing.columns
    ]
    right = closing[wanted].copy()

    # Remove placeholder columns before merge so the closing data can populate them.
    out = out.drop(columns=[c for c in CLOSING_VALUE_COLUMNS if c in out.columns], errors="ignore")
    out = out.merge(
        right,
        on=["_join_game_id", "market_type"],
        how="left",
        validate="many_to_one",
    )
    out = out.drop(columns=["_join_game_id"], errors="ignore")

    for col in CLOSING_VALUE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def numeric(value):
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def selected_entry_market_prob(row):
    market = str(row.get("market_type", "")).strip().lower()
    side = str(row.get("bet_side", "")).strip().lower()
    direct = None
    if market == "moneyline" and side in {"home", "away"}:
        direct = row.get(f"{side}_market_prob")
    elif market == "spread" and side in {"home", "away"}:
        direct = row.get(f"{side}_spread_market_prob")
    elif market == "total" and side in {"over", "under"}:
        direct = row.get(f"{side}_market_prob")

    value = numeric(direct)
    if not np.isnan(value):
        return value

    model_prob = numeric(row.get("bet_model_prob"))
    edge_pp = numeric(row.get("bet_edge_vs_market"))
    if not np.isnan(model_prob) and not np.isnan(edge_pp):
        return model_prob - edge_pp / 100.0
    return np.nan


def selected_closing_market_prob(row):
    market = str(row.get("market_type", "")).strip().lower()
    side = str(row.get("bet_side", "")).strip().lower()
    column = None
    if market == "moneyline":
        column = {
            "home": "closing_home_market_prob",
            "away": "closing_away_market_prob",
        }.get(side)
    elif market == "spread":
        column = {
            "home": "closing_home_spread_market_prob",
            "away": "closing_away_spread_market_prob",
        }.get(side)
    elif market == "total":
        column = {
            "over": "closing_over_market_prob",
            "under": "closing_under_market_prob",
        }.get(side)
    return numeric(row.get(column)) if column else np.nan


def selected_closing_line(row):
    market = str(row.get("market_type", "")).strip().lower()
    side = str(row.get("bet_side", "")).strip().lower()
    if market == "spread":
        if side == "home":
            return numeric(row.get("closing_home_spread"))
        if side == "away":
            return numeric(row.get("closing_away_spread"))
    elif market == "total":
        return numeric(row.get("closing_total"))
    return np.nan


def calculate_clv(row):
    market = str(row.get("market_type", "")).strip().lower()
    side = str(row.get("bet_side", "")).strip().lower()

    if market == "moneyline":
        entry = numeric(row.get("entry_market_prob"))
        close = numeric(row.get("closing_market_prob"))
        if np.isnan(entry) or np.isnan(close):
            return np.nan
        # Positive means the market moved toward the selected side after entry.
        return (close - entry) * 100.0

    bet_line = numeric(row.get("bet_line"))
    closing_line = numeric(row.get("closing_line"))
    if np.isnan(bet_line) or np.isnan(closing_line):
        return np.nan

    if market == "spread":
        # Signed side spread: -3 taken vs -4 close => +1 point CLV.
        return bet_line - closing_line
    if market == "total":
        if side == "over":
            return closing_line - bet_line
        if side == "under":
            return bet_line - closing_line
    return np.nan


def calculate_line_disagreement(row):
    market = str(row.get("market_type", "")).strip().lower()
    side = str(row.get("bet_side", "")).strip().lower()
    home_proj = numeric(row.get("home_projected_points"))
    away_proj = numeric(row.get("away_projected_points"))
    total_proj = numeric(row.get("total_projected_points"))

    if market == "spread":
        closing_home_spread = numeric(row.get("closing_home_spread"))
        if np.isnan(home_proj) or np.isnan(away_proj) or np.isnan(closing_home_spread):
            return np.nan
        home_model_minus_market = (home_proj - away_proj) + closing_home_spread
        if side == "home":
            return home_model_minus_market
        if side == "away":
            return -home_model_minus_market

    if market == "total":
        closing_total = numeric(row.get("closing_total"))
        if np.isnan(total_proj) or np.isnan(closing_total):
            return np.nan
        diff = total_proj - closing_total
        if side == "over":
            return diff
        if side == "under":
            return -diff

    return np.nan


def add_quality_columns(work: pd.DataFrame) -> pd.DataFrame:
    out = work.copy()

    for col in MODEL_METADATA_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    for col in [
        "home_score", "away_score", "home_projected_points",
        "away_projected_points", "total_projected_points",
        "bet_model_prob", "bet_edge_vs_market", "bet_line",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    result = out.get("bet_result", pd.Series(index=out.index, dtype=object))
    result = result.astype(str).str.strip().str.lower()
    out["probability_outcome"] = np.where(
        result.eq("win"), 1.0,
        np.where(result.eq("loss"), 0.0, np.nan),
    )

    p = pd.to_numeric(out["bet_model_prob"], errors="coerce")
    y = pd.to_numeric(out["probability_outcome"], errors="coerce")
    valid_prob = p.between(0, 1, inclusive="both") & y.isin([0.0, 1.0])
    out["brier_component"] = np.where(valid_prob, (p - y) ** 2, np.nan)
    p_clip = p.clip(1e-15, 1 - 1e-15)
    out["log_loss_component"] = np.where(
        valid_prob,
        -(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip)),
        np.nan,
    )

    home_score = pd.to_numeric(out.get("home_score"), errors="coerce")
    away_score = pd.to_numeric(out.get("away_score"), errors="coerce")
    home_proj = pd.to_numeric(out.get("home_projected_points"), errors="coerce")
    away_proj = pd.to_numeric(out.get("away_projected_points"), errors="coerce")
    total_proj = pd.to_numeric(out.get("total_projected_points"), errors="coerce")

    out["actual_margin"] = home_score - away_score
    out["projected_margin"] = home_proj - away_proj
    out["margin_error"] = out["projected_margin"] - out["actual_margin"]
    out["margin_abs_error"] = out["margin_error"].abs()
    out["margin_squared_error"] = out["margin_error"] ** 2

    out["actual_total"] = home_score + away_score
    out["projected_total"] = total_proj
    out["total_error"] = out["projected_total"] - out["actual_total"]
    out["total_abs_error"] = out["total_error"].abs()
    out["total_squared_error"] = out["total_error"] ** 2

    out["entry_market_prob"] = out.apply(selected_entry_market_prob, axis=1)
    out["closing_market_prob"] = out.apply(selected_closing_market_prob, axis=1)
    out["closing_line"] = out.apply(selected_closing_line, axis=1)
    out["clv"] = out.apply(calculate_clv, axis=1)
    out["clv_units"] = np.where(
        out["market_type"].eq("moneyline"),
        "probability_points",
        np.where(out["market_type"].isin(["spread", "total"]), "points", ""),
    )

    out["model_vs_market_prob_pp"] = (
        pd.to_numeric(out["bet_model_prob"], errors="coerce")
        - pd.to_numeric(out["closing_market_prob"], errors="coerce")
    ) * 100.0
    out["abs_model_vs_market_prob_pp"] = out["model_vs_market_prob_pp"].abs()

    out["model_vs_market_line"] = out.apply(calculate_line_disagreement, axis=1)
    out["abs_model_vs_market_line"] = pd.to_numeric(
        out["model_vs_market_line"], errors="coerce"
    ).abs()

    return out


def calibration_error(sub: pd.DataFrame) -> float:
    p = pd.to_numeric(sub.get("bet_model_prob"), errors="coerce")
    y = pd.to_numeric(sub.get("probability_outcome"), errors="coerce")
    valid = p.between(0, 1, inclusive="both") & y.isin([0.0, 1.0])
    p = p[valid]
    y = y[valid]
    if len(p) < MIN_PROB_SAMPLE:
        return np.nan

    bins = pd.cut(
        p,
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
        right=True,
    )
    frame = pd.DataFrame({"p": p, "y": y, "bin": bins})
    total = len(frame)
    ece = 0.0
    for _, bucket in frame.groupby("bin", observed=True):
        if bucket.empty:
            continue
        ece += (len(bucket) / total) * abs(bucket["p"].mean() - bucket["y"].mean())
    return float(ece)


def game_level_rows(sub: pd.DataFrame) -> pd.DataFrame:
    keys = [
        c for c in [
            "game_id", "model_source", "model_version",
            "feature_version", "ensemble_version",
        ] if c in sub.columns
    ]
    if not keys:
        return sub
    return sub.drop_duplicates(keys, keep="last")


def quality_row(
    sub: pd.DataFrame,
    *,
    scope: str,
    league: str,
    market_type: str = "ALL",
    model_source: str = "ALL",
    model_version: str = "ALL",
) -> dict:
    p = pd.to_numeric(sub.get("bet_model_prob"), errors="coerce")
    y = pd.to_numeric(sub.get("probability_outcome"), errors="coerce")
    prob_valid = p.between(0, 1, inclusive="both") & y.isin([0.0, 1.0])
    prob_n = int(prob_valid.sum())

    brier = float(pd.to_numeric(sub.get("brier_component"), errors="coerce").mean()) \
        if prob_n >= MIN_PROB_SAMPLE else np.nan
    logloss = float(pd.to_numeric(sub.get("log_loss_component"), errors="coerce").mean()) \
        if prob_n >= MIN_PROB_SAMPLE else np.nan
    cal = calibration_error(sub)

    games = game_level_rows(sub)
    margin_values = pd.to_numeric(games.get("margin_error"), errors="coerce").dropna()
    total_values = pd.to_numeric(games.get("total_error"), errors="coerce").dropna()
    margin_n = len(margin_values)
    total_n = len(total_values)

    margin_mae = float(margin_values.abs().mean()) if margin_n >= MIN_GAME_ERROR_SAMPLE else np.nan
    margin_rmse = float(np.sqrt((margin_values ** 2).mean())) if margin_n >= MIN_GAME_ERROR_SAMPLE else np.nan
    total_mae = float(total_values.abs().mean()) if total_n >= MIN_GAME_ERROR_SAMPLE else np.nan
    total_rmse = float(np.sqrt((total_values ** 2).mean())) if total_n >= MIN_GAME_ERROR_SAMPLE else np.nan

    clv = pd.to_numeric(sub.get("clv"), errors="coerce").dropna()
    clv_n = len(clv)
    clv_units_values = sorted({
        str(x) for x in sub.loc[pd.to_numeric(sub.get("clv"), errors="coerce").notna(), "clv_units"].dropna()
        if str(x).strip()
    }) if "clv_units" in sub.columns else []
    clv_units = clv_units_values[0] if len(clv_units_values) == 1 else ("mixed" if clv_units_values else "")
    avg_clv = (
        float(clv.mean())
        if clv_n >= MIN_CLV_SAMPLE and clv_units != "mixed"
        else np.nan
    )

    prob_dis = pd.to_numeric(sub.get("model_vs_market_prob_pp"), errors="coerce").dropna()
    prob_dis_n = len(prob_dis)
    avg_prob_dis = float(prob_dis.mean()) if prob_dis_n >= MIN_DISAGREEMENT_SAMPLE else np.nan
    abs_prob_dis = float(prob_dis.abs().mean()) if prob_dis_n >= MIN_DISAGREEMENT_SAMPLE else np.nan

    line_dis = pd.to_numeric(sub.get("model_vs_market_line"), errors="coerce").dropna()
    line_dis_n = len(line_dis)
    market_values = {
        str(x).lower()
        for x in sub.loc[pd.to_numeric(sub.get("model_vs_market_line"), errors="coerce").notna(), "market_type"].dropna()
    } if "market_type" in sub.columns else set()
    line_mixed = len(market_values) > 1
    avg_line_dis = (
        float(line_dis.mean())
        if line_dis_n >= MIN_DISAGREEMENT_SAMPLE and not line_mixed
        else np.nan
    )
    abs_line_dis = (
        float(line_dis.abs().mean())
        if line_dis_n >= MIN_DISAGREEMENT_SAMPLE and not line_mixed
        else np.nan
    )

    return {
        "scope": scope,
        "league": league.upper(),
        "market_type": market_type,
        "model_source": model_source,
        "model_version": model_version,
        "rows": len(sub),
        "probability_n": prob_n,
        "brier_score": brier,
        "log_loss": logloss,
        "calibration_error": cal,
        "margin_n": margin_n,
        "margin_mae": margin_mae,
        "margin_rmse": margin_rmse,
        "total_n": total_n,
        "total_mae": total_mae,
        "total_rmse": total_rmse,
        "clv_n": clv_n,
        "avg_clv": avg_clv,
        "clv_units": clv_units,
        "prob_disagreement_n": prob_dis_n,
        "avg_model_vs_market_prob_pp": avg_prob_dis,
        "mean_abs_model_vs_market_prob_pp": abs_prob_dis,
        "line_disagreement_n": line_dis_n,
        "avg_model_vs_market_line": avg_line_dis,
        "mean_abs_model_vs_market_line": abs_line_dis,
    }


def build_quality_metrics(work: pd.DataFrame, league: str) -> pd.DataFrame:
    if work.empty:
        return pd.DataFrame(columns=QUALITY_COLUMNS)

    frame = work.copy()
    frame["_model_source"] = (
        frame.get("model_source", pd.Series(index=frame.index, dtype=object))
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "unknown")
    )
    frame["_model_version"] = (
        frame.get("model_version", pd.Series(index=frame.index, dtype=object))
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "unknown")
    )

    rows = [
        quality_row(frame, scope="league", league=league)
    ]

    if "market_type" in frame.columns:
        for market, sub in frame.groupby("market_type", dropna=False, observed=True):
            rows.append(
                quality_row(
                    sub,
                    scope="market",
                    league=league,
                    market_type=str(market).lower(),
                )
            )

    for source, sub in frame.groupby("_model_source", dropna=False, observed=True):
        rows.append(
            quality_row(
                sub,
                scope="model_source",
                league=league,
                model_source=str(source),
            )
        )

    for (source, version), sub in frame.groupby(
        ["_model_source", "_model_version"], dropna=False, observed=True
    ):
        rows.append(
            quality_row(
                sub,
                scope="model_version",
                league=league,
                model_source=str(source),
                model_version=str(version),
            )
        )

    if "market_type" in frame.columns:
        for (market, source, version), sub in frame.groupby(
            ["market_type", "_model_source", "_model_version"],
            dropna=False,
            observed=True,
        ):
            rows.append(
                quality_row(
                    sub,
                    scope="market_model_version",
                    league=league,
                    market_type=str(market).lower(),
                    model_source=str(source),
                    model_version=str(version),
                )
            )

    out = pd.DataFrame(rows, columns=QUALITY_COLUMNS)
    return out



# =========================
# PREPARE
# =========================

def prepare(df: pd.DataFrame, league_label: str) -> pd.DataFrame:
    work = df.copy()

    # Normalize string columns
    if "market_type" in work.columns:
        work["market_type"] = work["market_type"].astype(str).str.strip().str.lower()
    if "bet_side" in work.columns:
        work["bet_side"] = work["bet_side"].astype(str).str.strip().str.lower()
    if "bet_result" in work.columns:
        work["bet_result"] = work["bet_result"].astype(str).str.strip().str.title()

    # Tag with league
    work["market"] = league_label

    # Preserve model provenance explicitly. Historical rows remain blank rather
    # than being assigned invented versions.
    for column in MODEL_METADATA_COLUMNS:
        if column not in work.columns:
            work[column] = pd.NA

    # Side grouping (HOME/AWAY/OVER/UNDER)
    work["side_group"] = work.apply(build_side_group, axis=1)

    # Source columns (use bet_* directly — already on the row from stage 04)
    if "bet_ev" not in work.columns:
        work["bet_ev"] = pd.NA
    if "bet_edge_vs_market" not in work.columns:
        work["bet_edge_vs_market"] = pd.NA
    if "bet_kelly" not in work.columns:
        work["bet_kelly"] = pd.NA
    if "bet_model_prob" not in work.columns:
        work["bet_model_prob"] = pd.NA
    if "bet_odds_american" not in work.columns:
        work["bet_odds_american"] = pd.NA
    if "bet_line" not in work.columns:
        work["bet_line"] = pd.NA
    if "bet_stake_pct" not in work.columns:
        work["bet_stake_pct"] = pd.NA

    # profit_unit and profit_kelly should already be on the row from script 01;
    # if missing, leave them missing rather than try to recompute here.
    if "profit_unit" not in work.columns:
        work["profit_unit"] = pd.NA
    if "profit_kelly" not in work.columns:
        work["profit_kelly"] = pd.NA

    # Bucket columns
    work["ev_bucket"]              = work["bet_ev"].apply(ev_bucket)
    work["edge_vs_market_bucket"]  = work["bet_edge_vs_market"].apply(edge_vs_market_bucket)
    work["kelly_bucket"]           = work["bet_kelly"].apply(kelly_bucket)
    work["model_prob_bucket"]      = work["bet_model_prob"].apply(model_prob_bucket)

    # odds_bucket only applies cleanly to moneyline rows — but compute for all
    # so the column exists; rows where it doesn't apply will be UNBUCKETED.
    work["odds_bucket"] = work["bet_odds_american"].apply(odds_bucket)

    # spread_bucket only meaningful for spread rows; total_bucket only for total rows.
    # Compute selectively so unrelated markets get UNBUCKETED rather than nonsense.
    def spread_for_row(row):
        if str(row.get("market_type", "")).lower() == "spread":
            return spread_bucket(row.get("bet_line"))
        return "UNBUCKETED"

    def total_for_row(row):
        if str(row.get("market_type", "")).lower() == "total":
            return total_bucket(row.get("bet_line"))
        return "UNBUCKETED"

    work["spread_bucket"] = work.apply(spread_for_row, axis=1)
    work["total_bucket"]  = work.apply(total_for_row,  axis=1)

    # Date buckets from game_date
    if "game_date" in work.columns:
        work["dow_bucket"]   = work["game_date"].apply(dow_bucket)
        work["month_bucket"] = work["game_date"].apply(month_bucket)
    else:
        work["dow_bucket"]   = "UNBUCKETED"
        work["month_bucket"] = "UNBUCKETED"

    return work


# =========================
# RUN
# =========================

def run_one(league: str) -> None:
    upper = league.upper()
    in_path  = INPUT_DIR / league / "graded" / f"{upper}_final.csv"
    out_path = OUTPUT_DIR / f"work_{league}.csv"
    quality_path = OUTPUT_DIR / f"quality_metrics_{league}.csv"

    if not in_path.exists():
        log_input(in_path, 0, exists=False)
        warn(f"[{upper}] input missing: {in_path} — skipping")
        pd.DataFrame(columns=QUALITY_COLUMNS).to_csv(quality_path, index=False)
        log_output(quality_path, 0)
        return

    df = pd.read_csv(in_path)
    log_input(in_path, len(df), exists=True)

    if df.empty:
        warn(f"[{upper}] input is empty: {in_path} — writing empty work/quality files")
        df.to_csv(out_path, index=False)
        log_output(out_path, 0)
        pd.DataFrame(columns=QUALITY_COLUMNS).to_csv(quality_path, index=False)
        log_output(quality_path, 0)
        return

    work = prepare(df, upper)
    work = attach_closing_lines(work, league)
    work = add_quality_columns(work)

    work.to_csv(out_path, index=False)
    log_output(out_path, len(work))

    quality = build_quality_metrics(work, league)
    quality.to_csv(quality_path, index=False)
    log_output(quality_path, len(quality))

    log("INFO", f"[{upper}] wrote {len(work)} rows -> {out_path}")
    log("INFO", f"[{upper}] wrote {len(quality)} quality rows -> {quality_path}")


def run() -> None:
    # Wipe basketball intermediate outputs before regenerating, matching the
    # previous behavior but after the run log has been initialized.
    for league in LEAGUES:
        for out_path in [
            OUTPUT_DIR / f"work_{league}.csv",
            OUTPUT_DIR / f"quality_metrics_{league}.csv",
        ]:
            if out_path.exists():
                out_path.unlink()
                log("INFO", f"REMOVED_STALE_OUTPUT | file={out_path}")

    for league in LEAGUES:
        run_one(league)
    log("INFO", "Analyze complete.")


def main() -> None:
    status = "FAILED"
    try:
        run()
        status = "SUCCESS"
    except Exception as exc:
        error(f"Unhandled exception: {type(exc).__name__}: {exc}")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(traceback.format_exc())
            if not traceback.format_exc().endswith("\n"):
                f.write("\n")
        raise
    finally:
        finish(status)


if __name__ == "__main__":
    main()
