#!/usr/bin/env python3
"""Current-day basketball pipeline validation using DRatings as the eligible slate.

ESPN schedules are diagnostic only. Unpriced DRatings games are a no-bet case,
not a fatal pipeline error. Real duplicate identity, lost predictions and stage
failures remain fatal. Writes the existing JSON and text report locations.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from datetime import datetime, UTC
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

BASE = Path("docs/win/basketball")
ERRORS = BASE / "errors"
OUTPUT = BASE / "pipeline_health.json"
LOG = ERRORS / "pipeline_health.txt"
SEASON_CONFIG = BASE / "config/season_dates.yaml"
SDV_SEASONS_CONFIG = BASE / "config/sdv_seasons.yaml"
SDV_STORAGE_CONFIG = BASE / "config/sdv_storage.yaml"
SDV_MODEL_CONFIG = BASE / "config/sdv_model.yaml"
MODEL_CONFIG = BASE / "config/model_config.yaml"
WNBA_DRIFT_REPORT = ERRORS / "99_validation/wnba_bias_drift.csv"
SDV_FEATURE_ROOT = BASE / "00_intake/sdv/features/current"
SDV_PREDICTION_ROOT = BASE / "00_intake/predictions_sdv"
ENSEMBLE_PREDICTION_ROOT = BASE / "00_intake/predictions_ensemble"
CLEANED_PREDICTION_ROOT = BASE / "00_intake/predictions/predictions_cleaned"
RAW_DRATINGS_ROOT = BASE / "00_intake/predictions"
NY = ZoneInfo("America/New_York")
LEAGUES = ("nba", "ncaam", "wnba")
LABEL = {lg: lg.upper() for lg in LEAGUES}
STAGE_LOGS = [
    ERRORS / "00_intake/basketball_odds.txt",
    ERRORS / "00_intake/basketball_drat_scraper.txt",
    ERRORS / "00_intake/transform_basketball_nba.txt",
    ERRORS / "00_intake/transform_basketball_ncaam.txt",
    ERRORS / "00_intake/transform_basketball_wnba.txt",
    ERRORS / "00_intake/basketball_daily_games.txt",
    ERRORS / "00_intake/basketball_game_id.txt",
    ERRORS / "00_intake/sdv_predict.txt",
    ERRORS / "00_intake/basketball_model_ensemble.txt",
    ERRORS / "00_intake/clean_basketball_inputs.txt",
    ERRORS / "01_merge/merge_intake.txt",
    ERRORS / "01_merge/build_juice_files.txt",
    ERRORS / "03_edges/compute_ev_kelly.txt",
    ERRORS / "04_select/select_bets.txt",
    ERRORS / "04_select/daily_slate.txt",
    ERRORS / "05_final_scores/01_basketball_results_grade.txt",
]
BAD_STATUS = ("STATUS: FAILED", "STATUS: PARTIAL", "STATUS: COMPLETED WITH ERRORS")


def clean(value) -> str:
    return "" if value is None else str(value).strip()


def clean_id(value) -> str:
    val = clean(value)
    try:
        n = float(val)
        return str(int(n)) if math.isfinite(n) and n.is_integer() else val
    except (TypeError, ValueError, OverflowError):
        return val


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def row_ids(rows: list[dict]) -> tuple[set[str], list[str], list[str]]:
    ids, blanks, duplicates = set(), [], set()
    for index, row in enumerate(rows, start=1):
        game_id = clean_id(row.get("game_id"))
        if not game_id:
            blanks.append(f"row={index}")
        elif game_id in ids:
            duplicates.add(game_id)
        else:
            ids.add(game_id)
    return ids, blanks, sorted(duplicates)


def comp(row: dict) -> tuple:
    return (clean(row.get("game_date")), clean(row.get("home_team")).casefold(), clean(row.get("away_team")).casefold())


def duplicate_integrity(rows: list[dict]):
    by_composite = {}
    by_id = {}
    duplicate_composites, conflicting_ids = set(), set()
    for row in rows:
        key, gid = comp(row), clean_id(row.get("game_id"))
        if all(key):
            if key in by_composite:
                duplicate_composites.add("|".join(key))
            by_composite[key] = row
        if gid:
            if gid in by_id and comp(by_id[gid]) != key:
                conflicting_ids.add(gid)
            by_id[gid] = row
    return sorted(duplicate_composites), sorted(conflicting_ids)


def load_season_config() -> dict:
    if not SEASON_CONFIG.is_file():
        raise FileNotFoundError(SEASON_CONFIG)
    with SEASON_CONFIG.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("Season configuration must be a mapping")
    values = {}
    for league in LEAGUES:
        raw = config.get(league)
        if not isinstance(raw, dict):
            raise ValueError(f"Missing season configuration for {league}")
        row = {name: int(raw[name]) for name in ("start_month", "start_day", "end_month", "end_day")}
        for key in ("start", "end"):
            datetime(2000, row[f"{key}_month"], row[f"{key}_day"])
        values[league] = row
    return values


def in_season(league: str, now: datetime, config: dict) -> bool:
    row = config[league]
    start, end = (row["start_month"], row["start_day"]), (row["end_month"], row["end_day"])
    today = (now.month, now.day)
    return start <= today <= end if start <= end else today >= start or today <= end


def stage_health():
    stages, fatals = [], []
    for path in STAGE_LOGS:
        status = None
        if path.exists():
            lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if "STATUS:" in line]
            status = lines[-1] if lines else None
        stages.append({"path": str(path), "exists": path.exists(), "status": status})
        if status and any(bad in status for bad in BAD_STATUS):
            fatals.append(f"stage failure: {path} -> {status}")
    return stages, fatals


def current_league_health(league: str, now: datetime, config: dict):
    label = LABEL[league]
    date = now.strftime("%Y_%m_%d")
    active = in_season(league, now, config)
    daily_path = BASE / "daily_games" / league / f"{date}_{label}.csv"
    dratings_path = RAW_DRATINGS_ROOT / league / f"{date}_{label}_predictions.csv"
    pred_path = CLEANED_PREDICTION_ROOT / league / f"{date}_{label}_predictions.csv"
    book_path = BASE / "00_intake/sportsbook/sportsbook_cleaned" / league / f"{date}_{label}_odds.csv"
    merge_path = BASE / "01_merge" / league / "moneyline" / f"{date}_{label}_moneyline.csv"
    picks_path = BASE / "04_select" / league / "daily_picks" / f"{date}_{league}_selected.csv"
    locked_path = BASE / "04_select" / league / "locked_picks" / f"{date}_{league}_selected.csv"
    daily = read_rows(daily_path)
    dratings = read_rows(dratings_path)
    preds = read_rows(pred_path)
    books = read_rows(book_path)
    merged = read_rows(merge_path)
    picks = read_rows(picks_path)
    locked = read_rows(locked_path)
    daily_ids, daily_blank, daily_dup = row_ids(daily)
    dratings_ids, dratings_blank, dratings_dup = row_ids(dratings)
    pred_ids, pred_blank, pred_dup = row_ids(preds)
    book_ids, _, _ = row_ids(books)
    merge_ids, merged_blank, merged_dup = row_ids(merged)
    missing_predicted = sorted(dratings_ids - pred_ids)
    missing_merged = sorted(pred_ids - merge_ids)
    unpriced = sorted(pred_ids - book_ids)
    scheduled_unmodeled = sorted(daily_ids - dratings_ids)
    duplicate_comp, conflicting_ids = duplicate_integrity(daily)
    pred_dup_comp, pred_conflicting_ids = duplicate_integrity(preds)
    fatals = []
    if active:
        if duplicate_comp or conflicting_ids or pred_dup_comp or pred_conflicting_ids:
            fatals.append(f"{label}: duplicate/conflicting current game identity")
        if daily_blank:
            fatals.append(f"{label}: {len(daily_blank)} current daily games missing game_id")
        if dratings_dup:
            fatals.append(f"{label}: duplicate raw DRatings game IDs: {dratings_dup}")
        # Unmatched DRatings rows have predictions but no canonical ESPN ID.
        # They cannot be joined or bet on; report them instead of blocking
        # correctly identified DRatings games.
        if preds and (pred_blank or pred_dup):
            fatals.append(f"{label}: downstream prediction game ID integrity failure")
        if merged and (merged_blank or merged_dup):
            fatals.append(f"{label}: merged game ID integrity failure")
        if missing_predicted:
            fatals.append(f"{label}: {len(missing_predicted)} DRatings games missing cleaned predictions")
        if missing_merged:
            fatals.append(f"{label}: {len(missing_merged)} predicted games missing merge output")
    item = {
        "in_season": active,
        "season_config": dict(config[league]),
        "paths": {"daily_games": str(daily_path), "dratings": str(dratings_path), "predictions": str(pred_path), "sportsbook": str(book_path), "merged": str(merge_path), "selected": str(picks_path), "locked": str(locked_path)},
        "counts": {"scheduled_games": len(daily_ids), "dratings_games": len(dratings_ids), "prediction_games": len(pred_ids), "sportsbook_games": len(book_ids), "merged_games": len(merge_ids), "selected_bets": len(picks), "locked_bets": len(locked)},
        "identity": {"daily_duplicate_composites": duplicate_comp, "daily_conflicting_game_ids": conflicting_ids, "prediction_duplicate_composites": pred_dup_comp, "prediction_conflicting_game_ids": pred_conflicting_ids, "blank_daily_game_ids": len(daily_blank), "blank_dratings_game_ids": len(dratings_blank), "blank_prediction_game_ids": len(pred_blank)},
        "coverage": {"scheduled_missing_predictions": [str(i) for i in scheduled_unmodeled], "scheduled_missing_sportsbook": [str(i) for i in unpriced], "predictions_not_merged": [str(i) for i in missing_merged], "dratings_missing_cleaned": missing_predicted},
        "warnings": [f"{label}: {len(scheduled_unmodeled)} ESPN games omitted (no DRatings prediction)" if scheduled_unmodeled else "", f"{label}: {len(dratings_blank)} DRatings prediction rows have no canonical game ID; preserved in raw source but cannot be merged or bet on until mapped" if dratings_blank else "", f"{label}: {len(unpriced)} predicted games without sportsbook data (no bet)" if unpriced else ""],
    }
    item["warnings"] = [v for v in item["warnings"] if v]
    return item, fatals


def sdv_health(now: datetime, seasons: dict, leagues: dict):
    """Check current prediction lineage, not unmatched ESPN fixtures."""
    date = now.strftime("%Y_%m_%d")
    errors, warnings, current = [], [], {}
    model_cfg = {}
    production_source = "ensemble"
    try:
        with MODEL_CONFIG.open(encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        errors.append(f"Model config cannot be read: {exc}")
    if isinstance(model_cfg, dict):
        configured = model_cfg.get("production_prediction_source")
        if isinstance(configured, str) and configured.strip().lower() in {"dratings", "sdv", "ensemble"}:
            production_source = configured.strip().lower()
    for league in LEAGUES:
        label = LABEL[league]
        active = in_season(league, now, seasons)
        eligible = row_ids(read_rows(RAW_DRATINGS_ROOT / league / f"{date}_{label}_predictions.csv"))[0]
        sdv_path = SDV_PREDICTION_ROOT / league / f"{date}_{label}_predictions.csv"
        ensemble_path = ENSEMBLE_PREDICTION_ROOT / league / f"{date}_{label}_predictions.csv"
        cleaned_path = CLEANED_PREDICTION_ROOT / league / f"{date}_{label}_predictions.csv"
        merged_path = BASE / "01_merge" / league / "moneyline" / f"{date}_{label}_moneyline.csv"
        scheduled = row_ids(read_rows(BASE / "daily_games" / league / f"{date}_{label}.csv"))[0]
        sdv_ids = row_ids(read_rows(sdv_path))[0]
        ensemble_ids, ensemble_blank, ensemble_duplicate = row_ids(read_rows(ensemble_path))
        clean_ids = row_ids(read_rows(cleaned_path))[0]
        merged_ids = row_ids(read_rows(merged_path))[0]
        ensemble_missing = sorted(eligible - ensemble_ids)
        cleaned_missing = sorted(eligible - clean_ids)
        if active and eligible:
            if not sdv_path.exists():
                errors.append(f"{label}: missing current SDV predictions: {sdv_path}")
            if not ensemble_path.exists() and production_source == "ensemble":
                errors.append(f"{label}: missing current ensemble predictions: {ensemble_path}")
            if production_source == "ensemble" and ensemble_missing:
                errors.append(f"{label}: DRatings game IDs missing from ensemble: {ensemble_missing}")
            if cleaned_missing:
                errors.append(f"{label}: DRatings game IDs missing downstream: {cleaned_missing}")
            if ensemble_blank or ensemble_duplicate:
                errors.append(f"{label}: current ensemble game IDs are blank/duplicated")
        current[league] = {
            "in_season": active,
            "scheduled_game_count": len(scheduled),
            "eligible_dratings_game_count": len(eligible),
            "ensemble_enabled": production_source == "ensemble",
            "configured_production_source": production_source,
            "feature": {"exists": (SDV_FEATURE_ROOT / league / f"{date}_{label}_features.parquet").exists(), "row_count": None},
            "sdv_predictions": {"exists": sdv_path.exists(), "row_count": len(sdv_ids), "game_ids": sorted(sdv_ids)},
            "ensemble_predictions": {"exists": ensemble_path.exists(), "row_count": len(ensemble_ids), "game_ids": sorted(ensemble_ids), "missing_scheduled_game_ids": ensemble_missing},
            "production_source": {"cleaned_row_count": len(clean_ids), "merged_row_count": len(merged_ids), "source_confirmed_downstream": bool(eligible and eligible <= clean_ids and eligible <= merged_ids), "cleaned_missing_dratings_game_ids": cleaned_missing},
        }
        if len(scheduled - eligible):
            warnings.append(f"{label}: {len(scheduled-eligible)} scheduled games have no DRatings prediction; ignored")
    return {"configured_production_source": production_source, "current": current, "warnings": warnings, "historical_manifests": {"skipped": True, "reason": "Current-slate health check does not revalidate historical SDV manifests", "valid_count": 0, "required_count": 0}, "validation_scope": "current_dratings_slate"}, errors


def wnba_bias_drift():
    return {"windows": {str(n): {"games": 0, "margin_mean_residual": None, "total_mean_residual": None, "warning": False} for n in (25, 50, 100)}, "warnings": [], "report_path": str(WNBA_DRIFT_REPORT)}


def main() -> None:
    now = datetime.now(NY)
    errors = []
    try:
        seasons = load_season_config()
    except Exception as exc:
        seasons = {league: {"start_month": 1, "start_day": 1, "end_month": 12, "end_day": 31} for league in LEAGUES}
        errors.append(f"season config failure: {exc}")
    stages, stage_errors = stage_health()
    errors.extend(stage_errors)
    leagues = {}
    for league in LEAGUES:
        info, fatal = current_league_health(league, now, seasons)
        leagues[league] = info
        errors.extend(fatal)
    sdv, sdv_errors = sdv_health(now, seasons, leagues)
    errors.extend(sdv_errors)
    errors = list(dict.fromkeys(errors))
    now_utc = datetime.now(UTC).isoformat()
    drift = wnba_bias_drift()
    payload = {"schema_version": 1, "generated_at_utc": now_utc, "game_date_new_york": now.strftime("%Y_%m_%d"), "season_config_path": str(SEASON_CONFIG), "status": "failed" if errors else "healthy", "fatal_errors": errors, "stage_status": stages, "leagues": leagues, "sdv_health": sdv, "wnba_bias_drift": drift}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [f"=== basketball pipeline health {now_utc} ===", f"game_date_new_york: {payload['game_date_new_york']}", f"season_config: {SEASON_CONFIG}", f"status: {payload['status']}"]
    for league, item in leagues.items():
        c = item["counts"]
        lines.append(f"{league.upper()}: in_season={item['in_season']} scheduled={c['scheduled_games']} dratings={c['dratings_games']} predictions={c['prediction_games']} sportsbook={c['sportsbook_games']} merged={c['merged_games']} selected={c['selected_bets']}")
        for warning in item["warnings"]:
            lines.append(f"WARN: {warning}")
    for warning in sdv["warnings"]:
        lines.append(f"WARN: {warning}")
    for error in errors:
        lines.append(f"FATAL: {error}")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
