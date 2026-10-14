#!/usr/bin/env python3
# docs/win/basketball/scripts/05_final_scores/04_basketball_results_dashboard.py
#
# Rebuilds the dashboard data read live by the React pages:
#   frontend/data/dashboards/basketball.json
#   frontend/data/dashboards/nba.json
#   frontend/data/dashboards/ncaam.json
#   frontend/data/dashboards/wnba.json
# Each file is {"generated_at": <UTC ISO>, "data": <season payload>}.
#
# Current-season inputs:
#   docs/win/basketball/05_final_scores/{league}_summary_grand_total.csv
#   docs/win/basketball/05_final_scores/{league}_summary_overall.csv
#   docs/win/basketball/05_final_scores/reports/{league}/...
#
# Optional historical-season inputs:
#   docs/win/basketball/05_final_scores/seasons/<season>/
#
# Each historical season directory should mirror the current Stage 05 layout.

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import json
import math
import os
import traceback

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

LEAGUES = ("nba", "ncaam", "wnba")
MARKETS = ("moneyline", "spread", "total")

LEAGUE_DISPLAY = {
    "all": "All",
    "nba": "NBA",
    "ncaam": "NCAAM",
    "wnba": "WNBA",
}

BASE = Path("docs/win/basketball/05_final_scores")
REPORTS = BASE / "reports"
SEASONS_ROOT = BASE / "seasons"

DASHBOARD_DIR = Path("frontend/data/dashboards")
MASTER_OUTPUT = DASHBOARD_DIR / "basketball.json"
LEAGUE_OUTPUTS = {
    "nba": DASHBOARD_DIR / "nba.json",
    "ncaam": DASHBOARD_DIR / "ncaam.json",
    "wnba": DASHBOARD_DIR / "wnba.json",
}

ERROR_DIR = Path("docs/win/basketball/errors/05_final_scores")
LOG_FILE = ERROR_DIR / "04_basketball_results_dashboard.txt"

ERROR_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

RUN_STARTED = datetime.now(UTC)
RUN_COUNTS = {
    "warnings": 0,
    "errors": 0,
    "input_files": 0,
    "input_rows": 0,
    "output_files": 0,
    "output_rows": 0,
}
INPUT_FILES_SEEN: set[str] = set()


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def reset_log() -> None:
    LOG_FILE.write_text(
        "=== 04_basketball_results_dashboard ===\n"
        f"START_TIMESTAMP_UTC: {RUN_STARTED.isoformat()}\n",
        encoding="utf-8",
    )


def log(level: str, message: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"{now_utc()} | {level} | {message}\n")


def warn(message: str) -> None:
    RUN_COUNTS["warnings"] += 1
    log("WARNING", message)


def error(message: str) -> None:
    RUN_COUNTS["errors"] += 1
    log("ERROR", message)


def log_input(path: Path, rows: int, exists: bool) -> None:
    key = str(path)
    if key in INPUT_FILES_SEEN:
        return

    INPUT_FILES_SEEN.add(key)
    RUN_COUNTS["input_files"] += 1
    RUN_COUNTS["input_rows"] += rows
    log("INFO", f"INPUT | file={path} | exists={int(exists)} | rows={rows}")


def log_output(path: Path, rows: int, bytes_written: int) -> None:
    RUN_COUNTS["output_files"] += 1
    RUN_COUNTS["output_rows"] += rows
    log("INFO", f"OUTPUT | file={path} | rows={rows} | bytes={bytes_written}")


def finish(status: str) -> None:
    ended = datetime.now(UTC)

    summary = (
        f"INPUT_SUMMARY | files={RUN_COUNTS['input_files']} | rows={RUN_COUNTS['input_rows']}",
        f"OUTPUT_SUMMARY | files={RUN_COUNTS['output_files']} | rows={RUN_COUNTS['output_rows']}",
        f"WARNING_COUNT: {RUN_COUNTS['warnings']}",
        f"ERROR_COUNT: {RUN_COUNTS['errors']}",
        f"END_TIMESTAMP_UTC: {ended.isoformat()}",
        f"STATUS: {status}",
    )

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.writelines(
            f"{line}\n"
            for line in summary
        )


# ============================================================
# DATA HELPERS
# ============================================================


def safe_read(path: Path, *, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        log_input(path, 0, False)
        if required:
            warn(f"Required dashboard input missing: {path}")
        else:
            log("INFO", f"Optional dashboard input missing: {path}")
        return pd.DataFrame()

    try:
        frame = pd.read_csv(path)
        log_input(path, len(frame), True)
        return frame
    except Exception as exc:
        log_input(path, 0, True)
        warn(f"Unable to read {path}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def clean_scalar(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def df_to_records(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []

    return [
        {key: clean_scalar(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def first_row(frame: pd.DataFrame) -> dict:
    rows = df_to_records(frame.head(1))
    return rows[0] if rows else {}


def first_nonblank(row: dict, keys: tuple[str, ...]):
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def normalize_side_rows(frame: pd.DataFrame) -> list[dict]:
    records = df_to_records(frame)

    for row in records:
        if first_nonblank(row, ("bucket",)) is not None:
            continue

        value = first_nonblank(row, ("side_group", "side", "variable"))
        if value is not None:
            row["bucket"] = value

    return records


def add_league(rows: list[dict], league: str) -> list[dict]:
    display = LEAGUE_DISPLAY[league]
    return [{"league": display, **row} for row in rows]


def to_number(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


# ============================================================
# SEASONS
# ============================================================


def season_label(name: str) -> str:
    return name.replace("_", "-").strip()


def discover_seasons() -> list[tuple[str, str, Path]]:
    found: list[tuple[str, str, Path]] = [("current", "Current", BASE)]

    if not SEASONS_ROOT.exists():
        return found

    historical = sorted(
        [path for path in SEASONS_ROOT.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )

    for path in historical:
        found.append((path.name, season_label(path.name), path))

    return found


# ============================================================
# LEAGUE DATA COLLECTION
# ============================================================


def collect_market_data(league: str, root: Path, market: str) -> dict:
    market_dir = root / "reports" / league / market
    market_data = {"by": {}, "by_side": {}}

    dimensions = (
        "ev",
        "kelly",
        "odds",
        "win_prob",
        "edge_vs_market",
        "dow",
        "month",
    )

    for dimension in dimensions:
        overall = market_dir / f"{league}_{market}_by_{dimension}.csv"
        side_suffix = "over_under" if market == "total" else "home_away"
        side = market_dir / f"{league}_{market}_by_{dimension}_{side_suffix}_summary.csv"

        market_data["by"][dimension] = df_to_records(safe_read(overall))
        market_data["by_side"][dimension] = normalize_side_rows(safe_read(side))

    if market in ("spread", "total"):
        side_suffix = "over_under" if market == "total" else "home_away"
        market_data["by"]["side"] = df_to_records(
            safe_read(market_dir / f"{league}_{market}_by_side.csv")
        )
        market_data["by_side"]["side"] = normalize_side_rows(
            safe_read(
                market_dir / f"{league}_{market}_by_side_{side_suffix}_summary.csv"
            )
        )

    if market == "total":
        market_data["by"]["total_range"] = df_to_records(
            safe_read(market_dir / f"{league}_{market}_by_total_range.csv")
        )
        market_data["by_side"]["total_range"] = normalize_side_rows(
            safe_read(
                market_dir / f"{league}_{market}_by_total_range_over_under_summary.csv"
            )
        )

    return market_data


def collect_league_data(league: str, root: Path) -> dict:
    is_current = root == BASE
    report_root = root / "reports" / league

    return {
        "league": league.upper(),
        "display": LEAGUE_DISPLAY[league],
        "grand_total": first_row(
            safe_read(
                root / f"{league}_summary_grand_total.csv",
                required=is_current,
            )
        ),
        "by_market_summary": df_to_records(
            safe_read(
                root / f"{league}_summary_overall.csv",
                required=is_current,
            )
        ),
        "quality": df_to_records(
            safe_read(report_root / "quality" / f"{league}_model_quality_all.csv")
        ),
        "overview": {
            "by_market": df_to_records(
                safe_read(report_root / "overview" / f"{league}_summary_by_market.csv")
            ),
            "by_side_group": normalize_side_rows(
                safe_read(
                    report_root
                    / "overview"
                    / f"{league}_summary_by_side_group.csv"
                )
            ),
            "by_date": df_to_records(
                safe_read(report_root / "overview" / f"{league}_summary_by_date.csv")
            ),
        },
        "markets": {
            market: collect_market_data(league, root, market)
            for market in MARKETS
        },
    }


# ============================================================
# ALL-LEAGUES AGGREGATION
# ============================================================


def collect_all_data(payloads: dict[str, dict]) -> dict:
    grand_rows = [
        payload.get("grand_total") or {}
        for payload in payloads.values()
        if payload.get("grand_total")
    ]

    bets = int(sum(to_number(row.get("bets")) for row in grand_rows))
    wins = int(sum(to_number(row.get("wins")) for row in grand_rows))
    losses = int(sum(to_number(row.get("losses")) for row in grand_rows))
    pushes = int(sum(to_number(row.get("pushes")) for row in grand_rows))
    units_flat = sum(to_number(row.get("units_flat")) for row in grand_rows)
    units_kelly = sum(to_number(row.get("units_kelly")) for row in grand_rows)

    grand_total = {
        "league": "ALL",
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_pct": wins / (wins + losses) if (wins + losses) else None,
        "units_flat": units_flat if grand_rows else None,
        "roi_flat": units_flat / bets if bets else None,
        "units_kelly": units_kelly if grand_rows else None,
        "roi_kelly": None,
    }

    by_market_summary: list[dict] = []
    quality: list[dict] = []
    overview = {
        "by_market": [],
        "by_side_group": [],
        "by_date": [],
    }
    markets = {
        market: {"by": {}, "by_side": {}}
        for market in MARKETS
    }

    for league, payload in payloads.items():
        by_market_summary.extend(
            add_league(payload.get("by_market_summary") or [], league)
        )
        quality.extend(add_league(payload.get("quality") or [], league))

        league_overview = payload.get("overview") or {}
        for key in overview:
            overview[key].extend(
                add_league(league_overview.get(key) or [], league)
            )

        league_markets = payload.get("markets") or {}
        for market in MARKETS:
            market_payload = league_markets.get(market) or {"by": {}, "by_side": {}}

            for view in ("by", "by_side"):
                for dimension, rows in (market_payload.get(view) or {}).items():
                    markets[market][view].setdefault(dimension, [])
                    markets[market][view][dimension].extend(
                        add_league(rows or [], league)
                    )

    return {
        "league": "ALL",
        "display": "All",
        "is_all": True,
        "grand_total": grand_total,
        "by_market_summary": by_market_summary,
        "quality": quality,
        "overview": overview,
        "markets": markets,
    }


# ============================================================
# PAYLOAD + JSON OUTPUT
# ============================================================


def build_payload(leagues: tuple[str, ...], *, include_all: bool) -> dict[str, dict]:
    payload: dict[str, dict] = {}

    for season_key, label, root in discover_seasons():
        league_payloads = {
            league: collect_league_data(league, root)
            for league in leagues
        }

        page_leagues = dict(league_payloads)
        if include_all:
            page_leagues = {
                "all": collect_all_data(league_payloads),
                **league_payloads,
            }

        payload[season_key] = {
            "label": label,
            "leagues": page_leagues,
        }

    return payload


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, data: dict) -> None:
    document = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data": json_safe(data),
    }
    content = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(content, encoding="utf-8")
    os.replace(staging, path)
    log_output(path, 1, len(content.encode("utf-8")))
    log("INFO", f"dashboard data -> {path}")


def run() -> None:
    write_json(MASTER_OUTPUT, build_payload(LEAGUES, include_all=True))

    for league in LEAGUES:
        write_json(LEAGUE_OUTPUTS[league], build_payload((league,), include_all=False))


def main() -> None:
    reset_log()
    status = "FAILED"

    try:
        run()
        status = "SUCCESS"
    except Exception as exc:
        error(f"Unhandled exception: {type(exc).__name__}: {exc}")
        trace = traceback.format_exc()
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(trace)
            if not trace.endswith("\n"):
                handle.write("\n")
        raise
    finally:
        finish(status)


if __name__ == "__main__":
    main()
