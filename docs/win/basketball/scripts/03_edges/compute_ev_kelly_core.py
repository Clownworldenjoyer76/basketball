#!/usr/bin/env python3
# docs/win/basketball/scripts/03_edges/compute_ev_kelly_core.py
#
# Reads merguiced files (per market, per league) and adds EV, edge-vs-market,
# and Kelly columns using the model/market probabilities ALREADY computed
# upstream in build_juice_files.py.
#
# Inputs:
#   docs/win/basketball/01_merge/01_merguiced/{league}/{market}/*.csv
#       where league in {nba, ncaam, wnba}
#       and market in {moneyline, spread, total}
#
# Outputs:
#   docs/win/basketball/03_edges/ev_kelly/{league}/{market}/*.csv
#
# Key design points:
# - Spread cover probability is NOT recomputed. It comes from the
#   home_spread_model_prob / away_spread_model_prob columns in the merguiced
#   file, which were computed in build_juice_files.py with the correct
#   sign convention. Recomputing here historically reintroduced a sign bug.
# - Total over/under probability is NOT round-tripped through fair_over.
#   It comes from over_model_prob / under_model_prob directly.
# - Book decimal odds are recomputed EXACTLY from the American odds columns
#   (rounded once to DECIMAL_PLACES) and written back into the existing
#   *_decimal columns. The sportsbook decimal columns in the input are cut off
#   at 2 decimals and are not used. EV, Kelly and the uncertainty adjustment all
#   use the same exact value, so staking_runtime's EV consistency check matches.
# - EV at offered price = (model_prob * book_decimal) - 1
# - Edge vs devigged market = model_prob - market_prob
# - Kelly fraction = (b*p - q)/b where b=d-1, q=1-p, clipped to >= 0.
#   Result is 0 (never NaN) when EV <= 0 or when odds/probability are missing
#   or invalid (decimal odds <= 1 give blank EV and Kelly 0).
# - Outputs are built in a temporary folder and copied over the real outputs
#   only after the build; a fatal error leaves existing outputs untouched.
# - All three leagues (NBA, NCAAM, WNBA) supported uniformly.

import shutil
import sys
import tempfile
import traceback
from datetime import datetime, UTC
from pathlib import Path

# ITEM 19 shared staking/uncertainty rules
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# =========================
# PATHS
# =========================

# Repository root, derived from this file's location (docs/win/basketball/scripts/03_edges/).
ROOT = SCRIPTS_DIR.parents[3]
BASE = ROOT / "docs/win/basketball"

INPUT_DIR  = BASE / "01_merge/01_merguiced"
OUTPUT_DIR = BASE / "03_edges/ev_kelly"
ERROR_DIR  = BASE / "errors/03_edges"
LOG_FILE   = ERROR_DIR / "compute_ev_kelly.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LEAGUES = ["nba", "ncaam", "wnba"]
MARKETS = ["moneyline", "spread", "total"]

# Exact decimal odds are rounded once to this many places.
DECIMAL_PLACES = 6

# Temporary build folder; set by main(). process_one writes here when set.
STAGE_DIR = None


# =========================
# LOGGING
# =========================

def _now():
    return datetime.now(UTC).isoformat()


def _log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


# Imports that can fail are done after the log exists so a failure is recorded.
try:
    import numpy as np
    import pandas as pd

    from staking_runtime import add_uncertainty_adjusted_ev
except Exception as _import_exc:
    with open(LOG_FILE, "w", encoding="utf-8") as _f:
        _f.write(f"=== compute_ev_kelly RUN {_now()} ===\n")
    _log(f"FATAL: import failed: {_import_exc}\n{traceback.format_exc()}", "ERROR")
    _log("STATUS: FAILED")
    raise


class SchemaError(ValueError):
    """Input file is missing required columns."""


def _write_summary(summary: dict, per_file: list, fatal: bool = False) -> None:
    lines = [
        "",
        "=" * 70,
        f"SUMMARY  {_now()}",
        "=" * 70,
        f"  files_processed   : {summary['files_processed']}",
        f"  rows_processed    : {summary['rows_processed']}",
        f"  moneyline_files   : {summary['moneyline_files']}",
        f"  spread_files      : {summary['spread_files']}",
        f"  total_files       : {summary['total_files']}",
        f"  skipped           : {summary['skipped']}",
        f"  schema_errors     : {summary['schema_errors']}",
        f"  errors            : {summary['errors']}",
        "",
        f"  {'file':<48} {'league':>6} {'market':<10} {'rows':>6} {'status':>10}",
    ]
    for pf in per_file:
        lines.append(
            f"  {pf['name']:<48} {pf['league']:>6} {pf['market']:<10} "
            f"{pf['rows']:>6} {pf['status']:>10}"
        )
    has_errors = summary["errors"] != 0 or summary["schema_errors"] != 0
    if fatal:
        status = "FAILED"
    elif has_errors:
        status = "COMPLETED WITH ERRORS"
    else:
        status = "SUCCESS"
    if not fatal and not has_errors and summary["files_processed"] == 0 and summary["skipped"] == 0:
        lines += ["", "  NOTE: no input files were processed (no games, or no input files found)."]
    lines += ["", f"STATUS: {status}", "=" * 70]
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================
# HELPERS
# =========================

def to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def validate_columns(df, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")


def exact_decimal_from_american(american):
    """
    American odds -> exact decimal odds, rounded once to DECIMAL_PLACES.
    Positive: 1 + odds/100.  Negative: 1 + 100/|odds|.
    Blank, zero or non-finite odds -> NaN.
    """
    odds = pd.to_numeric(american, errors="coerce")
    valid = np.isfinite(odds) & (odds != 0)
    dec = pd.Series(np.nan, index=odds.index, dtype=float)
    pos = valid & (odds > 0)
    neg = valid & (odds < 0)
    dec[pos] = 1.0 + odds[pos] / 100.0
    dec[neg] = 1.0 + 100.0 / odds[neg].abs()
    return dec.round(DECIMAL_PLACES)


def _valid_decimal(book_decimal):
    """Decimal odds as numbers; anything not finite or <= 1 becomes NaN."""
    d = pd.to_numeric(book_decimal, errors="coerce")
    return d.where(np.isfinite(d) & (d > 1))


def compute_ev(model_prob, book_decimal):
    """EV per $1 staked at the book's offered decimal odds. NaN if odds are invalid."""
    return (model_prob * _valid_decimal(book_decimal)) - 1


def compute_kelly(model_prob, book_decimal):
    """
    Kelly fraction for a binary bet: f = (b*p - q) / b
    where b = decimal - 1, p = model_prob, q = 1 - p.
    Returns 0 (not NaN) when EV <= 0 and when odds or probability are
    missing or invalid, so "no bet" is always 0.
    """
    d = _valid_decimal(book_decimal)
    b = d - 1
    q = 1 - model_prob
    k = ((b * model_prob) - q) / b
    return k.clip(lower=0).fillna(0)

def atomic_write_csv(df, path):
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


# =========================
# MARKET PROCESSORS
# =========================

def process_moneyline(df: pd.DataFrame, league: str | None = None) -> pd.DataFrame:
    """
    Required columns from merguiced ML file:
      home_model_prob, away_model_prob          (model)
      home_market_prob, away_market_prob        (devigged market)
      home_dk_moneyline_american, away_dk_moneyline_american  (book, source of exact decimals)
      home_dk_moneyline_decimal, away_dk_moneyline_decimal    (book; overwritten with exact values)
    """
    required = [
        "home_model_prob", "away_model_prob",
        "home_market_prob", "away_market_prob",
        "home_dk_moneyline_american", "away_dk_moneyline_american",
        "home_dk_moneyline_decimal", "away_dk_moneyline_decimal",
    ]
    validate_columns(df, required)
    df = to_numeric(df, [
        "home_model_prob", "away_model_prob",
        "home_market_prob", "away_market_prob",
    ])

    # Exact book odds from American odds (replaces the cut-off decimal columns)
    df["home_dk_moneyline_decimal"] = exact_decimal_from_american(df["home_dk_moneyline_american"])
    df["away_dk_moneyline_decimal"] = exact_decimal_from_american(df["away_dk_moneyline_american"])

    # EV at offered price
    df["home_ml_ev"] = compute_ev(df["home_model_prob"], df["home_dk_moneyline_decimal"])
    df["away_ml_ev"] = compute_ev(df["away_model_prob"], df["away_dk_moneyline_decimal"])

    # Edge vs devigged market
    df["home_ml_edge_vs_market"] = df["home_model_prob"] - df["home_market_prob"]
    df["away_ml_edge_vs_market"] = df["away_model_prob"] - df["away_market_prob"]

    # Kelly fraction (0 if EV <= 0 or inputs missing/invalid)
    df["home_ml_kelly"] = compute_kelly(df["home_model_prob"], df["home_dk_moneyline_decimal"])
    df["away_ml_kelly"] = compute_kelly(df["away_model_prob"], df["away_dk_moneyline_decimal"])

    df = add_uncertainty_adjusted_ev(
        df, "moneyline",
        [
            ("home_ml", "home_model_prob", "home_market_prob", "home_dk_moneyline_decimal", "home_ml_kelly"),
            ("away_ml", "away_model_prob", "away_market_prob", "away_dk_moneyline_decimal", "away_ml_kelly"),
        ],
        league,
    )

    # Convenience: percent versions
    df["home_ml_ev_pct"]              = df["home_ml_ev"] * 100
    df["away_ml_ev_pct"]              = df["away_ml_ev"] * 100
    df["home_ml_edge_vs_market_pct"]  = df["home_ml_edge_vs_market"] * 100
    df["away_ml_edge_vs_market_pct"]  = df["away_ml_edge_vs_market"] * 100

    return df


def process_spread(df: pd.DataFrame, league: str | None = None) -> pd.DataFrame:
    """
    Required columns from merguiced spread file:
      home_spread_model_prob, away_spread_model_prob       (model — sign fix applied upstream)
      home_spread_market_prob, away_spread_market_prob     (devigged market)
      home_dk_spread_american, away_dk_spread_american     (book, source of exact decimals)
      home_dk_spread_decimal, away_dk_spread_decimal       (book; overwritten with exact values)
    """
    required = [
        "home_spread_model_prob", "away_spread_model_prob",
        "home_spread_market_prob", "away_spread_market_prob",
        "home_dk_spread_american", "away_dk_spread_american",
        "home_dk_spread_decimal", "away_dk_spread_decimal",
    ]
    validate_columns(df, required)
    df = to_numeric(df, [
        "home_spread_model_prob", "away_spread_model_prob",
        "home_spread_market_prob", "away_spread_market_prob",
    ])

    df["home_dk_spread_decimal"] = exact_decimal_from_american(df["home_dk_spread_american"])
    df["away_dk_spread_decimal"] = exact_decimal_from_american(df["away_dk_spread_american"])

    df["home_spread_ev"] = compute_ev(df["home_spread_model_prob"], df["home_dk_spread_decimal"])
    df["away_spread_ev"] = compute_ev(df["away_spread_model_prob"], df["away_dk_spread_decimal"])

    df["home_spread_edge_vs_market"] = df["home_spread_model_prob"] - df["home_spread_market_prob"]
    df["away_spread_edge_vs_market"] = df["away_spread_model_prob"] - df["away_spread_market_prob"]

    df["home_spread_kelly"] = compute_kelly(df["home_spread_model_prob"], df["home_dk_spread_decimal"])
    df["away_spread_kelly"] = compute_kelly(df["away_spread_model_prob"], df["away_dk_spread_decimal"])

    df = add_uncertainty_adjusted_ev(
        df, "spread",
        [
            ("home_spread", "home_spread_model_prob", "home_spread_market_prob", "home_dk_spread_decimal", "home_spread_kelly"),
            ("away_spread", "away_spread_model_prob", "away_spread_market_prob", "away_dk_spread_decimal", "away_spread_kelly"),
        ],
        league,
    )

    df["home_spread_ev_pct"]              = df["home_spread_ev"] * 100
    df["away_spread_ev_pct"]              = df["away_spread_ev"] * 100
    df["home_spread_edge_vs_market_pct"]  = df["home_spread_edge_vs_market"] * 100
    df["away_spread_edge_vs_market_pct"]  = df["away_spread_edge_vs_market"] * 100

    return df


def process_total(df: pd.DataFrame, league: str | None = None) -> pd.DataFrame:
    """
    Required columns from merguiced total file:
      over_model_prob, under_model_prob              (model — Poisson removed upstream)
      over_market_prob, under_market_prob            (devigged market)
      dk_total_over_american, dk_total_under_american  (book, source of exact decimals)
      dk_total_over_decimal, dk_total_under_decimal    (book; overwritten with exact values)
    """
    required = [
        "over_model_prob", "under_model_prob",
        "over_market_prob", "under_market_prob",
        "dk_total_over_american", "dk_total_under_american",
        "dk_total_over_decimal", "dk_total_under_decimal",
    ]
    validate_columns(df, required)
    df = to_numeric(df, [
        "over_model_prob", "under_model_prob",
        "over_market_prob", "under_market_prob",
    ])

    df["dk_total_over_decimal"]  = exact_decimal_from_american(df["dk_total_over_american"])
    df["dk_total_under_decimal"] = exact_decimal_from_american(df["dk_total_under_american"])

    df["over_ev"]  = compute_ev(df["over_model_prob"],  df["dk_total_over_decimal"])
    df["under_ev"] = compute_ev(df["under_model_prob"], df["dk_total_under_decimal"])

    df["over_edge_vs_market"]  = df["over_model_prob"]  - df["over_market_prob"]
    df["under_edge_vs_market"] = df["under_model_prob"] - df["under_market_prob"]

    df["over_kelly"]  = compute_kelly(df["over_model_prob"],  df["dk_total_over_decimal"])
    df["under_kelly"] = compute_kelly(df["under_model_prob"], df["dk_total_under_decimal"])

    df = add_uncertainty_adjusted_ev(
        df, "total",
        [
            ("over", "over_model_prob", "over_market_prob", "dk_total_over_decimal", "over_kelly"),
            ("under", "under_model_prob", "under_market_prob", "dk_total_under_decimal", "under_kelly"),
        ],
        league,
    )

    df["over_ev_pct"]               = df["over_ev"]  * 100
    df["under_ev_pct"]              = df["under_ev"] * 100
    df["over_edge_vs_market_pct"]   = df["over_edge_vs_market"]  * 100
    df["under_edge_vs_market_pct"]  = df["under_edge_vs_market"] * 100

    return df


MARKET_PROCESSORS = {
    "moneyline": process_moneyline,
    "spread":    process_spread,
    "total":     process_total,
}


# =========================
# MAIN LOOP
# =========================

def process_one(path: Path, league: str, market: str,
                summary: dict, per_file: list) -> None:
    pf = {"name": path.name, "league": league.upper(), "market": market,
          "rows": 0, "status": "ok"}

    try:
        df = pd.read_csv(path)
        if df.empty:
            _log(f"{path.name} empty — skipping")
            pf["status"] = "empty"
            summary["skipped"] += 1
            per_file.append(pf)
            return

        pf["rows"] = len(df)

        df = MARKET_PROCESSORS[market](df, league)

        out_dir = (STAGE_DIR or OUTPUT_DIR) / league / market
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / path.name
        atomic_write_csv(df, out_path)

        summary["rows_processed"]       += len(df)
        summary["files_processed"]      += 1
        summary[f"{market}_files"]      += 1
        _log(f"BUILT: {path.name} ({len(df)} rows)")

    except SchemaError as e:
        _log(f"{path.name} schema error: {e}", "ERROR")
        pf["status"] = "schema_error"
        summary["schema_errors"] += 1
    except Exception as e:
        _log(f"{path.name} FAILED: {e}\n{traceback.format_exc()}", "ERROR")
        pf["status"] = "error"
        summary["errors"] += 1

    per_file.append(pf)


def discover_input_files(
    league: str,
    market: str,
) -> list[Path]:
    folder = (
        INPUT_DIR
        / league
        / market
    )

    if folder.is_dir():
        files = sorted(
            folder.glob("*.csv")
        )
        if files:
            return files

        warning = (
            "NO FILES: "
            f"league={league} "
            f"market={market}"
        )
    else:
        warning = (
            f"INPUT FOLDER MISSING: {folder}"
        )

    _log(
        warning,
        "WARN",
    )
    return []


def publish(stage: Path, summary: dict) -> None:
    """
    Copy the built files over the real outputs.
    No errors: old output files are removed first (stale files cleared).
    Errors: old files are NOT removed; built files overwrite matching names only.
    """
    clean_build = summary["errors"] == 0 and summary["schema_errors"] == 0
    for league in LEAGUES:
        for market in MARKETS:
            folder = OUTPUT_DIR / league / market
            folder.mkdir(parents=True, exist_ok=True)
            if clean_build:
                for stale in folder.glob("*.csv"):
                    stale.unlink(missing_ok=True)
    count = 0
    for src in sorted(stage.rglob("*.csv")):
        dest = OUTPUT_DIR / src.relative_to(stage)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        count += 1
    if clean_build:
        _log(f"Replaced old outputs with {count} new file(s).")
    else:
        _log(f"Errors occurred: existing outputs NOT wiped; {count} built file(s) copied over matching names.", "WARN")


def main():
    global STAGE_DIR

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== compute_ev_kelly RUN {_now()} ===\n")

    summary = {
        "files_processed": 0, "rows_processed": 0,
        "moneyline_files": 0, "spread_files": 0, "total_files": 0,
        "skipped": 0, "schema_errors": 0, "errors": 0,
    }
    per_file = []

    _log(f"INPUT_DIR : {INPUT_DIR}")
    _log(f"OUTPUT_DIR: {OUTPUT_DIR}")

    STAGE_DIR = Path(tempfile.mkdtemp(prefix="compute_ev_kelly_"))
    try:
        for league in LEAGUES:
            for market in MARKETS:
                for path in discover_input_files(
                    league,
                    market,
                ):
                    process_one(
                        path,
                        league,
                        market,
                        summary,
                        per_file,
                    )
        publish(STAGE_DIR, summary)

    except Exception as e:
        _log(f"FATAL: {e}\n{traceback.format_exc()}", "ERROR")
        _log("Existing outputs were not replaced unless the error happened while copying.", "ERROR")
        _write_summary(summary, per_file, fatal=True)
        sys.exit(1)
    finally:
        shutil.rmtree(STAGE_DIR, ignore_errors=True)
        STAGE_DIR = None

    _write_summary(summary, per_file)
    if summary["errors"] or summary["schema_errors"]:
        print("compute_ev_kelly finished with errors.")
        sys.exit(1)
    print("compute_ev_kelly complete.")


if __name__ == "__main__":
    main()
