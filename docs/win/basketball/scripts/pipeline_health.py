#!/usr/bin/env python3
# docs/win/basketball/scripts/pipeline_health.py
"""Current-run basketball pipeline health contract.

Writes:
    docs/win/basketball/pipeline_health.json
    docs/win/basketball/errors/pipeline_health.txt

Exits non-zero on production-critical current-run integrity failures.

Operational season dates:
    docs/win/basketball/config/season_dates.yaml

SDV / ensemble health:
    docs/win/basketball/config/sdv_seasons.yaml
    docs/win/basketball/config/sdv_storage.yaml
    docs/win/basketball/config/sdv_model.yaml
    docs/win/basketball/config/model_config.yaml

Historical/offseason mismatch noise is reported but does not make an otherwise
valid live run fail. Static configuration, required historical manifests, model
artifacts, and enabled ensemble weights are production-critical.
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


# =============================================================================
# PATHS / SETTINGS
# =============================================================================

BASE = Path("docs/win/basketball")
ERRORS = BASE / "errors"
OUTPUT = BASE / "pipeline_health.json"
LOG = ERRORS / "pipeline_health.txt"

SEASON_CONFIG = BASE / "config/season_dates.yaml"
SDV_SEASONS_CONFIG = BASE / "config/sdv_seasons.yaml"
SDV_STORAGE_CONFIG = BASE / "config/sdv_storage.yaml"
SDV_MODEL_CONFIG = BASE / "config/sdv_model.yaml"
MODEL_CONFIG = BASE / "config/model_config.yaml"

SDV_HISTORY_ROOT = BASE / "00_intake/sdv/history"
SDV_FEATURE_ROOT = BASE / "00_intake/sdv/features/current"
SDV_PREDICTION_ROOT = BASE / "00_intake/predictions_sdv"
ENSEMBLE_PREDICTION_ROOT = BASE / "00_intake/predictions_ensemble"
CLEANED_PREDICTION_ROOT = BASE / "00_intake/predictions/predictions_cleaned"
SDV_MODEL_ROOT = BASE / "models/sdv"
ENSEMBLE_MODEL_ROOT = BASE / "models/ensemble"

PRODUCTION_ROOTS = {
    "dratings": BASE / "00_intake/predictions",
    "sdv": SDV_PREDICTION_ROOT,
    "ensemble": ENSEMBLE_PREDICTION_ROOT,
}

REQUIRED_SDV_MODEL_FILES = (
    "margin_model.json",
    "total_model.json",
    "feature_schema.json",
    "metadata.json",
)

WNBA_DRIFT_REPORT = ERRORS / "99_validation/wnba_bias_drift.csv"

NY = ZoneInfo("America/New_York")

LEAGUES = ["nba", "ncaam", "wnba"]

LABEL = {
    "nba": "NBA",
    "ncaam": "NCAAM",
    "wnba": "WNBA",
}

DRIFT_WINDOWS = [25, 50, 100]
DRIFT_WARN = float(os.getenv("WNBA_BIAS_DRIFT_WARN", "2.0"))

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

BAD_STATUS = (
    "STATUS: FAILED",
    "STATUS: PARTIAL",
    "STATUS: COMPLETED WITH ERRORS",
)


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def clean(value) -> str:
    return "" if value is None else str(value).strip()


def clean_id(value) -> str:
    text = clean(value)
    number = fnum(
        text
    )

    if (
        number is not None
        and number.is_integer()
    ):
        return str(
            int(number)
        )

    return text

def fnum(value):
    try:
        if value is None or clean(value) == "":
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def bool_value(value) -> bool | None:
    if isinstance(value, bool):
        return value

    text = clean(value).lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return None


def comp(row: dict) -> tuple[str, str, str]:
    return (
        clean(row.get("game_date")),
        clean(row.get("home_team")).casefold(),
        clean(row.get("away_team")).casefold(),
    )


def format_ids(values) -> str:
    return json.dumps(
        sorted(
            {
                clean_id(value)
                for value in values
                if clean_id(value)
            }
        )
    )


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with path.open(newline="", encoding="utf-8-sig") as handle:
        result = list(csv.DictReader(handle))
    return result


def read_yaml_mapping(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a top-level mapping")

    return payload


def read_json_mapping(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")

    return payload


def unique_by_comp(rows: list[dict]) -> dict[tuple, dict]:
    out = {}
    for row in rows:
        key = comp(row)
        if all(key):
            out[key] = row
    return out


def row_game_ids(rows: list[dict]) -> tuple[set[str], list[str], list[str]]:
    ids: set[str] = set()
    blank_rows: list[str] = []
    duplicate_ids: set[str] = set()

    for index, row in enumerate(rows, start=1):
        gid = clean_id(row.get("game_id"))

        if not gid:
            blank_rows.append(f"row={index}")
            continue

        if gid in ids:
            duplicate_ids.add(gid)

        ids.add(gid)

    return ids, blank_rows, sorted(duplicate_ids)


def duplicate_integrity(rows: list[dict]) -> tuple[list[str], list[str]]:
    duplicate_composites = []
    duplicate_ids = []

    seen_comp = {}
    seen_id = {}

    for row in rows:
        key = comp(row)
        gid = clean_id(row.get("game_id"))

        if all(key):
            if key in seen_comp:
                duplicate_composites.append("|".join(key))
            else:
                seen_comp[key] = row

        if gid:
            prior = seen_id.get(gid)
            if prior is not None and comp(prior) != key:
                duplicate_ids.append(gid)
            else:
                seen_id[gid] = row

    return sorted(set(duplicate_composites)), sorted(set(duplicate_ids))


def last_status(path: Path) -> str | None:
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8", errors="replace")

    statuses = [
        line.strip()
        for line in text.splitlines()
        if "STATUS:" in line
    ]

    return statuses[-1] if statuses else None


def stage_health() -> tuple[list[dict], list[str]]:
    rows = []
    fatals = []

    for path in STAGE_LOGS:
        status = last_status(path)

        rows.append({
            "path": str(path),
            "exists": path.exists(),
            "status": status,
        })

        if status and any(bad in status for bad in BAD_STATUS):
            fatals.append(f"stage failure: {path} -> {status}")

    return rows, fatals


# =============================================================================
# OPERATIONAL SEASON HEALTH
# =============================================================================

def validate_month_day(league: str, label: str, month: int, day: int) -> None:
    try:
        datetime(2000, month, day)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {league}.{label}: month={month}, day={day}"
        ) from exc


def parse_season_config_row(
    league: str,
    row,
    required_fields: tuple[str, ...],
) -> dict[str, int]:
    if not isinstance(
        row,
        dict,
    ):
        raise ValueError(
            "Missing season configuration "
            f"for league={league}"
        )

    missing_field = next(
        (
            field
            for field in required_fields
            if field not in row
        ),
        None,
    )

    if missing_field is not None:
        raise ValueError(
            f"Missing {league}.{missing_field} "
            f"in {SEASON_CONFIG}"
        )

    def parsed_value(
        field: str,
    ) -> int:
        raw_value = row[
            field
        ]

        try:
            return int(
                raw_value
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Invalid {league}.{field}: "
                f"{raw_value!r}"
            ) from exc

    values = {
        field: parsed_value(
            field
        )
        for field in required_fields
    }

    for label in (
        "start",
        "end",
    ):
        validate_month_day(
            league,
            label,
            values[
                f"{label}_month"
            ],
            values[
                f"{label}_day"
            ],
        )

    return values


def load_season_config() -> dict[str, dict[str, int]]:
    if not SEASON_CONFIG.exists():
        raise FileNotFoundError(
            f"Season config not found: "
            f"{SEASON_CONFIG}"
        )

    raw = read_yaml_mapping(
        SEASON_CONFIG
    )

    required_fields = (
        "start_month",
        "start_day",
        "end_month",
        "end_day",
    )

    return {
        league: parse_season_config_row(
            league,
            raw.get(
                league
            ),
            required_fields,
        )
        for league in LEAGUES
    }


def in_season(
    league: str,
    now: datetime,
    season_config: dict[str, dict[str, int]],
) -> bool:
    league = league.strip().lower()

    if league not in season_config:
        raise KeyError(f"No season configuration found for league={league}")

    cfg = season_config[league]
    current_mmdd = (now.month, now.day)
    start_mmdd = (cfg["start_month"], cfg["start_day"])
    end_mmdd = (cfg["end_month"], cfg["end_day"])

    if start_mmdd <= end_mmdd:
        return start_mmdd <= current_mmdd <= end_mmdd

    return current_mmdd >= start_mmdd or current_mmdd <= end_mmdd


# =============================================================================
# EXISTING CURRENT PIPELINE HEALTH
# =============================================================================

def current_league_health(
    league: str,
    now: datetime,
    season_config: dict[str, dict[str, int]],
) -> tuple[dict, list[str]]:
    upper = LABEL[league]
    date = now.strftime("%Y_%m_%d")
    active = in_season(league, now, season_config)

    daily_path = BASE / f"daily_games/{league}/{date}_{upper}.csv"
    pred_path = (
        BASE
        / "00_intake/predictions/predictions_cleaned"
        / league
        / f"{date}_{upper}_predictions.csv"
    )
    book_path = (
        BASE
        / "00_intake/sportsbook/sportsbook_cleaned"
        / league
        / f"{date}_{upper}_odds.csv"
    )
    merge_path = (
        BASE
        / "01_merge"
        / league
        / "moneyline"
        / f"{date}_{upper}_moneyline.csv"
    )
    picks_path = (
        BASE
        / "04_select"
        / league
        / "daily_picks"
        / f"{date}_{league}_selected.csv"
    )
    locked_path = (
        BASE
        / "04_select"
        / league
        / "locked_picks"
        / f"{date}_{league}_selected.csv"
    )

    daily = read_rows(daily_path)
    preds = read_rows(pred_path)
    books = read_rows(book_path)
    merged = read_rows(merge_path)
    picks = read_rows(picks_path)
    locked = read_rows(locked_path)

    daily_map = unique_by_comp(daily)
    pred_map = unique_by_comp(preds)
    book_map = unique_by_comp(books)
    merge_map = unique_by_comp(merged)

    dup_comp, dup_ids = duplicate_integrity(daily)
    pred_dup_comp, pred_dup_ids = duplicate_integrity(preds)

    missing_predictions = sorted(
        "|".join(key)
        for key in set(daily_map) - set(pred_map)
    )
    missing_sportsbook = sorted(
        "|".join(key)
        for key in set(daily_map) - set(book_map)
    )
    predicted_not_merged = sorted(
        "|".join(key)
        for key in set(pred_map) - set(merge_map)
    )

    blank_pred_ids = sum(
        1 for row in preds if not clean_id(row.get("game_id"))
    )
    blank_daily_ids = sum(
        1 for row in daily if not clean_id(row.get("game_id"))
    )

    item = {
        "in_season": active,
        "season_config": dict(season_config[league]),
        "paths": {
            "daily_games": str(daily_path),
            "predictions": str(pred_path),
            "sportsbook": str(book_path),
            "merged": str(merge_path),
            "selected": str(picks_path),
            "locked": str(locked_path),
        },
        "counts": {
            "scheduled_games": len(daily_map),
            "prediction_games": len(pred_map),
            "sportsbook_games": len(book_map),
            "merged_games": len(merge_map),
            "selected_bets": len(picks),
            "locked_bets": len(locked),
        },
        "identity": {
            "daily_duplicate_composites": dup_comp,
            "daily_conflicting_game_ids": dup_ids,
            "prediction_duplicate_composites": pred_dup_comp,
            "prediction_conflicting_game_ids": pred_dup_ids,
            "blank_daily_game_ids": blank_daily_ids,
            "blank_prediction_game_ids": blank_pred_ids,
        },
        "coverage": {
            "scheduled_missing_predictions": missing_predictions,
            "scheduled_missing_sportsbook": missing_sportsbook,
            "predictions_not_merged": predicted_not_merged,
        },
    }

    fatals: list[str] = []

    if active:
        if dup_comp or dup_ids or pred_dup_comp or pred_dup_ids:
            fatals.append(
                f"{upper}: duplicate/conflicting current game identity"
            )

        if blank_daily_ids:
            fatals.append(
                f"{upper}: {blank_daily_ids} current daily games missing game_id"
            )

        if preds and blank_pred_ids:
            fatals.append(
                f"{upper}: {blank_pred_ids} current predictions missing game_id"
            )

        if daily_map and missing_predictions:
            fatals.append(
                f"{upper}: {len(missing_predictions)} scheduled games "
                "missing predictions"
            )

        if daily_map and missing_sportsbook:
            fatals.append(
                f"{upper}: {len(missing_sportsbook)} scheduled games "
                "missing sportsbook rows"
            )

        if pred_map and predicted_not_merged:
            fatals.append(
                f"{upper}: {len(predicted_not_merged)} prediction games did not merge"
            )

    return item, fatals


# =============================================================================
# SDV CONFIG VALIDATION
# =============================================================================

def validate_sdv_seasons_config() -> tuple[
    dict[str, dict[int, int]],
    dict,
    list[str],
]:
    report = {
        "path": str(SDV_SEASONS_CONFIG),
        "exists": SDV_SEASONS_CONFIG.exists(),
        "valid": False,
        "leagues": {},
        "errors": [],
    }
    fatals: list[str] = []
    mappings: dict[str, dict[int, int]] = {}

    try:
        cfg = read_yaml_mapping(SDV_SEASONS_CONFIG)
    except Exception as exc:
        message = f"SDV seasons config failure: {SDV_SEASONS_CONFIG} -> {exc}"
        report["errors"].append(message)
        fatals.append(message)
        return mappings, report, fatals

    if cfg.get("schema_version") != 1:
        report["errors"].append(
            f"{SDV_SEASONS_CONFIG}: schema_version must be 1"
        )

    if clean(cfg.get("mapping_policy")).lower() != "explicit_only":
        report["errors"].append(
            f"{SDV_SEASONS_CONFIG}: mapping_policy must be explicit_only"
        )

    leagues_cfg = cfg.get("leagues")
    if not isinstance(leagues_cfg, dict):
        report["errors"].append(
            f"{SDV_SEASONS_CONFIG}: missing leagues mapping"
        )
        leagues_cfg = {}

    for league in LEAGUES:
        league_errors: list[str] = []
        raw = leagues_cfg.get(league)
        parsed: dict[int, int] = {}

        if not isinstance(raw, dict):
            league_errors.append(
                f"{SDV_SEASONS_CONFIG}: missing leagues.{league}"
            )
        else:
            raw_mappings = raw.get("mappings")
            if not isinstance(raw_mappings, dict) or not raw_mappings:
                league_errors.append(
                    f"{SDV_SEASONS_CONFIG}: {league}.mappings "
                    "must be a non-empty mapping"
                )
            else:
                for internal, sdv in raw_mappings.items():
                    try:
                        internal_i = int(internal)
                        sdv_i = int(sdv)
                    except (TypeError, ValueError):
                        league_errors.append(
                            f"{SDV_SEASONS_CONFIG}: {league} invalid mapping "
                            f"{internal!r} -> {sdv!r}"
                        )
                        continue

                    if internal_i < 1900 or sdv_i < 1900:
                        league_errors.append(
                            f"{SDV_SEASONS_CONFIG}: {league} invalid season "
                            f"mapping {internal_i} -> {sdv_i}"
                        )
                        continue

                    parsed[internal_i] = sdv_i

        mappings[league] = parsed
        report["leagues"][league] = {
            "valid": not league_errors,
            "mappings": {
                str(key): value
                for key, value in sorted(parsed.items())
            },
            "errors": league_errors,
        }
        report["errors"].extend(league_errors)

    report["valid"] = not report["errors"]
    fatals.extend(report["errors"])
    return mappings, report, fatals


def validate_sdv_storage_config(
    season_mappings: dict[str, dict[int, int]],
) -> tuple[dict, dict, list[str]]:
    report = {
        "path": str(SDV_STORAGE_CONFIG),
        "exists": SDV_STORAGE_CONFIG.exists(),
        "valid": False,
        "expected_version": None,
        "history_root": None,
        "format": None,
        "tables": [],
        "historical_internal_seasons": {},
        "errors": [],
    }
    fatals: list[str] = []

    try:
        cfg = read_yaml_mapping(SDV_STORAGE_CONFIG)
    except Exception as exc:
        message = f"SDV storage config failure: {SDV_STORAGE_CONFIG} -> {exc}"
        report["errors"].append(message)
        fatals.append(message)
        return {}, report, fatals

    if cfg.get("schema_version") != 1:
        report["errors"].append(
            f"{SDV_STORAGE_CONFIG}: schema_version must be 1"
        )

    sportsdataverse = cfg.get("sportsdataverse")
    if not isinstance(sportsdataverse, dict):
        sportsdataverse = {}

    expected_version = clean(sportsdataverse.get("expected_version"))
    report["expected_version"] = expected_version or None

    if not expected_version:
        report["errors"].append(
            f"{SDV_STORAGE_CONFIG}: sportsdataverse.expected_version is blank"
        )

    storage = cfg.get("storage")
    if not isinstance(storage, dict):
        storage = {}
        report["errors"].append(
            f"{SDV_STORAGE_CONFIG}: missing storage mapping"
        )

    root = clean(storage.get("root"))
    storage_format = clean(storage.get("format")).lower()

    report["history_root"] = root or None
    report["format"] = storage_format or None

    if os.path.normcase(os.path.normpath(root)) != os.path.normcase(os.path.normpath(str(SDV_HISTORY_ROOT))):
        report["errors"].append(
            f"{SDV_STORAGE_CONFIG}: storage.root mismatch "
            f"expected={SDV_HISTORY_ROOT} actual={root!r}"
        )

    if storage_format != "parquet":
        report["errors"].append(
            f"{SDV_STORAGE_CONFIG}: storage.format must be parquet"
        )

    raw_tables = cfg.get("tables")
    if not isinstance(raw_tables, list) or not raw_tables:
        raw_tables = []
        report["errors"].append(
            f"{SDV_STORAGE_CONFIG}: tables must be a non-empty list"
        )

    tables = [clean(value) for value in raw_tables if clean(value)]
    report["tables"] = tables

    if len(tables) != len(set(tables)):
        report["errors"].append(
            f"{SDV_STORAGE_CONFIG}: tables contains duplicates"
        )

    historical = cfg.get("historical_internal_seasons")
    if not isinstance(historical, dict):
        historical = {}
        report["errors"].append(
            f"{SDV_STORAGE_CONFIG}: missing historical_internal_seasons mapping"
        )

    for league in LEAGUES:
        raw_seasons = historical.get(league)

        if not isinstance(raw_seasons, list) or not raw_seasons:
            report["historical_internal_seasons"][league] = []
            report["errors"].append(
                f"{SDV_STORAGE_CONFIG}: historical_internal_seasons.{league} "
                "must be a non-empty list"
            )
            continue

        seasons: list[int] = []

        for value in raw_seasons:
            try:
                season = int(value)
            except (TypeError, ValueError):
                report["errors"].append(
                    f"{SDV_STORAGE_CONFIG}: {league} invalid historical "
                    f"season {value!r}"
                )
                continue

            seasons.append(season)

            if season not in season_mappings.get(league, {}):
                report["errors"].append(
                    f"{SDV_STORAGE_CONFIG}: {league} historical season={season} "
                    f"has no mapping in {SDV_SEASONS_CONFIG}"
                )

        if len(seasons) != len(set(seasons)):
            report["errors"].append(
                f"{SDV_STORAGE_CONFIG}: historical_internal_seasons.{league} "
                "contains duplicates"
            )

        report["historical_internal_seasons"][league] = sorted(set(seasons))

    report["valid"] = not report["errors"]
    fatals.extend(report["errors"])
    return cfg, report, fatals


def validate_sdv_model_config() -> tuple[dict, dict, list[str]]:
    report = {
        "path": str(SDV_MODEL_CONFIG),
        "exists": SDV_MODEL_CONFIG.exists(),
        "valid": False,
        "feature_version": None,
        "model_version": None,
        "errors": [],
    }
    fatals: list[str] = []

    try:
        cfg = read_yaml_mapping(SDV_MODEL_CONFIG)
    except Exception as exc:
        message = f"SDV model config failure: {SDV_MODEL_CONFIG} -> {exc}"
        report["errors"].append(message)
        fatals.append(message)
        return {}, report, fatals

    advanced_features = cfg.get("advanced_features")
    advanced_features = (
        advanced_features
        if isinstance(advanced_features, dict)
        else {}
    )
    candidate_feature_version = clean(cfg.get("feature_version"))
    feature_version = (
        clean(advanced_features.get("production_feature_version"))
        or candidate_feature_version
    )
    training = cfg.get("training")
    training = training if isinstance(training, dict) else {}
    model_version = clean(training.get("model_version"))

    report["feature_version"] = feature_version or None
    report["model_version"] = model_version or None

    if not feature_version:
        report["errors"].append(
            f"{SDV_MODEL_CONFIG}: feature_version is blank"
        )

    if not model_version:
        report["errors"].append(
            f"{SDV_MODEL_CONFIG}: training.model_version is blank"
        )

    artifacts = cfg.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, dict) else {}

    if os.path.normcase(os.path.normpath(clean(artifacts.get("root")))) != os.path.normcase(os.path.normpath(str(SDV_MODEL_ROOT))):
        report["errors"].append(
            f"{SDV_MODEL_CONFIG}: artifacts.root mismatch "
            f"expected={SDV_MODEL_ROOT} actual={artifacts.get('root')!r}"
        )

    if not bool_value(artifacts.get("require_feature_version_match")):
        report["errors"].append(
            f"{SDV_MODEL_CONFIG}: artifacts.require_feature_version_match "
            "must be true"
        )

    report["valid"] = not report["errors"]
    fatals.extend(report["errors"])
    return cfg, report, fatals


def validate_model_config() -> tuple[dict, str | None, dict, list[str]]:
    report = {
        "path": str(MODEL_CONFIG),
        "exists": MODEL_CONFIG.exists(),
        "valid": False,
        "configured_source": None,
        "config_key": None,
        "errors": [],
    }
    fatals: list[str] = []

    try:
        cfg = read_yaml_mapping(MODEL_CONFIG)
    except Exception as exc:
        message = f"model config failure: {MODEL_CONFIG} -> {exc}"
        report["errors"].append(message)
        fatals.append(message)
        return {}, None, report, fatals

    source = clean(cfg.get("production_prediction_source")).lower()
    config_key = "production_prediction_source"

    if not source:
        source = clean(cfg.get("model_source")).lower()
        config_key = "model_source"

    report["configured_source"] = source or None
    report["config_key"] = config_key

    if source not in PRODUCTION_ROOTS:
        report["errors"].append(
            f"{MODEL_CONFIG}: {config_key} must be one of "
            f"{sorted(PRODUCTION_ROOTS)} actual={source!r}"
        )
        source = None

    leagues_cfg = cfg.get("leagues")
    if not isinstance(leagues_cfg, dict):
        report["errors"].append(
            f"{MODEL_CONFIG}: missing leagues mapping"
        )

    report["valid"] = not report["errors"]
    fatals.extend(report["errors"])
    return cfg, source, report, fatals


def explicit_ensemble_flag(
    config: dict,
) -> bool | None:
    direct = bool_value(
        config.get(
            "ensemble_enabled"
        )
    )

    if direct is not None:
        return direct

    nested = config.get(
        "ensemble"
    )

    if isinstance(
        nested,
        dict,
    ):
        return bool_value(
            nested.get(
                "enabled"
            )
        )

    return None


def ensemble_enabled_for_league(
    model_cfg: dict,
    league: str,
    production_source: str | None,
) -> bool:
    if production_source == "ensemble":
        return True

    enabled = explicit_ensemble_flag(
        model_cfg
    )

    if enabled is not None:
        return enabled

    leagues_cfg = model_cfg.get(
        "leagues"
    )

    if not isinstance(
        leagues_cfg,
        dict,
    ):
        return False

    league_cfg = leagues_cfg.get(
        league
    )

    if not isinstance(
        league_cfg,
        dict,
    ):
        return False

    enabled = explicit_ensemble_flag(
        league_cfg
    )

    return (
        enabled
        if enabled is not None
        else False
    )

# =============================================================================
# SDV HISTORICAL MANIFEST HEALTH
# =============================================================================

def validate_historical_manifests(
    season_mappings: dict[str, dict[int, int]],
    storage_report: dict,
) -> tuple[dict, list[str]]:
    report = {
        "required_count": 0,
        "valid_count": 0,
        "manifests": [],
    }
    fatals: list[str] = []

    expected_version = clean(storage_report.get("expected_version"))
    expected_format = clean(storage_report.get("format"))
    required_tables = list(storage_report.get("tables", []))
    historical = storage_report.get("historical_internal_seasons", {})

    for league in LEAGUES:
        for season in historical.get(league, []):
            report["required_count"] += 1
            path = SDV_HISTORY_ROOT / league / str(season) / "manifest.json"

            item = {
                "league": league,
                "internal_season": season,
                "path": str(path),
                "exists": path.exists(),
                "valid": False,
                "errors": [],
            }

            if not path.exists():
                message = f"missing required SDV historical manifest: {path}"
                item["errors"].append(message)
                fatals.append(message)
                report["manifests"].append(item)
                continue

            try:
                manifest = read_json_mapping(path)
            except Exception as exc:
                message = f"invalid SDV historical manifest: {path} -> {exc}"
                item["errors"].append(message)
                fatals.append(message)
                report["manifests"].append(item)
                continue

            if clean(manifest.get("league")).lower() != league:
                item["errors"].append(
                    f"{path}: league mismatch expected={league.upper()} "
                    f"actual={manifest.get('league')!r}"
                )

            try:
                internal_season = int(manifest.get("internal_season"))
            except (TypeError, ValueError):
                internal_season = None

            if internal_season != season:
                item["errors"].append(
                    f"{path}: internal_season mismatch expected={season} "
                    f"actual={manifest.get('internal_season')!r}"
                )

            expected_sdv = season_mappings.get(league, {}).get(season)

            try:
                actual_sdv = int(manifest.get("sdv_season"))
            except (TypeError, ValueError):
                actual_sdv = None

            if expected_sdv is not None and actual_sdv != expected_sdv:
                item["errors"].append(
                    f"{path}: sdv_season mismatch expected={expected_sdv} "
                    f"actual={actual_sdv}"
                )

            actual_version = clean(manifest.get("sportsdataverse_version"))
            if expected_version and actual_version != expected_version:
                item["errors"].append(
                    f"{path}: sportsdataverse_version mismatch "
                    f"expected={expected_version!r} actual={actual_version!r}"
                )

            actual_format = clean(manifest.get("storage_format")).lower()
            if expected_format and actual_format != expected_format:
                item["errors"].append(
                    f"{path}: storage_format mismatch "
                    f"expected={expected_format!r} actual={actual_format!r}"
                )

            tables = manifest.get("tables")
            if not isinstance(tables, dict):
                item["errors"].append(f"{path}: missing tables mapping")
                tables = {}

            for table in required_tables:
                table_info = tables.get(table)

                if not isinstance(table_info, dict):
                    item["errors"].append(
                        f"{path}: missing required table entry={table}"
                    )
                    continue

                rows = table_info.get("rows")
                row_count = None

                if rows is not None:
                    try:
                        row_count = int(rows)
                    except (TypeError, ValueError):
                        item["errors"].append(
                            f"{path}: table={table} rows is not an integer: "
                            f"{rows!r}"
                        )
                    else:
                        if row_count < 0:
                            item["errors"].append(
                                f"{path}: table={table} rows={row_count} "
                                "cannot be negative"
                            )

                status = clean(table_info.get("status")).lower()
                filename = clean(table_info.get("filename"))
                needs_file = (
                    (row_count is not None and row_count > 0)
                    or status
                    in {
                        "written",
                        "ready",
                        "success",
                        "existing",
                        "existing_not_rebuilt",
                    }
                )

                if needs_file:
                    table_path = path.parent / (filename or f"{table}.parquet")
                    if not table_path.exists():
                        item["errors"].append(
                            f"missing SDV historical table file: {table_path} "
                            f"(manifest={path}, table={table})"
                        )

            if item["errors"]:
                fatals.extend(item["errors"])
            else:
                item["valid"] = True
                report["valid_count"] += 1

            report["manifests"].append(item)

    return report, fatals


# =============================================================================
# SDV MODEL / ENSEMBLE ARTIFACT HEALTH
# =============================================================================

def validate_sdv_models(
    expected_feature_version: str,
    expected_model_version: str,
) -> tuple[dict, dict[str, dict], list[str]]:
    report = {}
    contexts: dict[str, dict] = {}
    fatals: list[str] = []

    for league in LEAGUES:
        folder = SDV_MODEL_ROOT / league
        item = {
            "directory": str(folder),
            "valid": False,
            "feature_version": None,
            "model_version": None,
            "files": {},
            "errors": [],
        }
        payloads: dict[str, dict] = {}

        for filename in REQUIRED_SDV_MODEL_FILES:
            path = folder / filename
            file_item = {
                "path": str(path),
                "exists": path.exists(),
                "valid_json": False,
            }
            item["files"][filename] = file_item

            if not path.exists():
                item["errors"].append(f"missing SDV model file: {path}")
                continue

            try:
                payloads[filename] = read_json_mapping(path)
                file_item["valid_json"] = True
            except Exception as exc:
                item["errors"].append(
                    f"invalid SDV model JSON: {path} -> {exc}"
                )

        metadata = payloads.get("metadata.json", {})
        metadata_feature = clean(metadata.get("feature_version"))
        metadata_model = clean(metadata.get("model_version"))

        item["feature_version"] = metadata_feature or None
        item["model_version"] = metadata_model or None

        if not metadata_feature:
            item["errors"].append(
                f"{folder / 'metadata.json'}: feature_version is blank"
            )

        if not metadata_model:
            item["errors"].append(
                f"{folder / 'metadata.json'}: model_version is blank"
            )

        if clean(metadata.get("league")).lower() not in {"", league}:
            item["errors"].append(
                f"{folder / 'metadata.json'}: league mismatch "
                f"expected={league.upper()} actual={metadata.get('league')!r}"
            )

        artifact_files = metadata.get("artifact_files")
        if isinstance(artifact_files, list):
            missing_declared = sorted(
                set(REQUIRED_SDV_MODEL_FILES)
                - {clean(value) for value in artifact_files}
            )
            for filename in missing_declared:
                item["errors"].append(
                    f"{folder / 'metadata.json'}: artifact_files missing "
                    f"{filename}"
                )

        for filename, payload in payloads.items():
            path = folder / filename
            feature_version = clean(payload.get("feature_version"))
            model_version = clean(payload.get("model_version"))

            if not feature_version:
                item["errors"].append(
                    f"{path}: feature_version is blank"
                )
            elif (
                expected_feature_version
                and feature_version != expected_feature_version
            ):
                item["errors"].append(
                    f"{path}: feature_version mismatch "
                    f"configured={expected_feature_version!r} "
                    f"artifact={feature_version!r}"
                )

            if (
                model_version
                and expected_model_version
                and model_version != expected_model_version
            ):
                item["errors"].append(
                    f"{path}: model_version mismatch "
                    f"configured={expected_model_version!r} "
                    f"artifact={model_version!r}"
                )

        contexts[league] = {
            "feature_version": metadata_feature or expected_feature_version,
            "model_version": metadata_model or expected_model_version,
        }

        if item["errors"]:
            fatals.extend(item["errors"])
        else:
            item["valid"] = True

        report[league] = item

    return report, contexts, fatals


def validate_ensemble_weights(
    model_cfg: dict,
    production_source: str | None,
    sdv_contexts: dict[str, dict],
) -> tuple[dict, list[str], list[str]]:
    report = {}
    fatals: list[str] = []
    warnings: list[str] = []

    for league in LEAGUES:
        enabled = ensemble_enabled_for_league(
            model_cfg,
            league,
            production_source,
        )
        path = ENSEMBLE_MODEL_ROOT / league / "weights.json"

        item = {
            "path": str(path),
            "enabled": enabled,
            "exists": path.exists(),
            "valid": False,
            "ensemble_version": None,
            "errors": [],
        }

        if not path.exists():
            message = f"missing ensemble weights file: {path}"
            item["errors"].append(message)
            (fatals if enabled else warnings).append(message)
            report[league] = item
            continue

        try:
            payload = read_json_mapping(path)
        except Exception as exc:
            message = f"invalid ensemble weights JSON: {path} -> {exc}"
            item["errors"].append(message)
            (fatals if enabled else warnings).append(message)
            report[league] = item
            continue

        ensemble_version = clean(payload.get("ensemble_version"))
        item["ensemble_version"] = ensemble_version or None

        if not ensemble_version:
            item["errors"].append(f"{path}: ensemble_version is blank")

        if clean(payload.get("league")).lower() not in {"", league}:
            item["errors"].append(
                f"{path}: league mismatch expected={league.upper()} "
                f"actual={payload.get('league')!r}"
            )

        for market in ("margin", "total", "moneyline"):
            section = payload.get(market)

            if not isinstance(section, dict):
                item["errors"].append(
                    f"{path}: missing {market} weights mapping"
                )
                continue

            dratings_weight = fnum(section.get("dratings_weight"))
            sdv_weight = fnum(section.get("sdv_weight"))

            if dratings_weight is None or sdv_weight is None:
                item["errors"].append(
                    f"{path}: {market} weights must be numeric"
                )
                continue

            if not (0 <= dratings_weight <= 1 and 0 <= sdv_weight <= 1):
                item["errors"].append(
                    f"{path}: {market} weights must be between 0 and 1 "
                    f"dratings={dratings_weight} sdv={sdv_weight}"
                )

            if abs((dratings_weight + sdv_weight) - 1.0) > 1e-9:
                item["errors"].append(
                    f"{path}: {market} weights must sum to 1.0 "
                    f"actual={dratings_weight + sdv_weight}"
                )

        components = payload.get("components")
        sdv_component = (
            components.get("sdv")
            if isinstance(components, dict)
            else None
        )

        if not isinstance(sdv_component, dict):
            item["errors"].append(f"{path}: components.sdv is missing")
        else:
            expected_feature = clean(
                sdv_contexts.get(league, {}).get("feature_version")
            )
            expected_model = clean(
                sdv_contexts.get(league, {}).get("model_version")
            )
            weight_feature = clean(sdv_component.get("feature_version"))
            weight_model = clean(sdv_component.get("model_version"))

            if expected_feature and weight_feature != expected_feature:
                item["errors"].append(
                    f"{path}: components.sdv.feature_version mismatch "
                    f"model={expected_feature!r} weights={weight_feature!r}"
                )

            if expected_model and weight_model != expected_model:
                item["errors"].append(
                    f"{path}: components.sdv.model_version mismatch "
                    f"model={expected_model!r} weights={weight_model!r}"
                )

        if item["errors"]:
            target = fatals if enabled else warnings
            target.extend(item["errors"])
        else:
            item["valid"] = True

        report[league] = item

    return report, fatals, warnings


# =============================================================================
# CURRENT SDV FEATURE / PREDICTION HEALTH
# =============================================================================

def read_current_feature_file(path: Path) -> dict:
    try:
        import polars as pl
    except Exception as exc:
        raise RuntimeError(
            f"polars is required to validate current SDV feature parquet: {path}"
        ) from exc

    try:
        frame = pl.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"unable to read parquet {path}: {exc}") from exc

    required = {"game_id", "feature_version"}
    missing_columns = sorted(required - set(frame.columns))

    if missing_columns:
        raise RuntimeError(
            f"{path}: missing columns {missing_columns}"
        )

    rows = frame.select(["game_id", "feature_version"]).to_dicts()
    ids: set[str] = set()
    duplicates: set[str] = set()
    blanks: list[str] = []
    versions: dict[str, str] = {}

    for index, row in enumerate(rows, start=1):
        gid = clean_id(row.get("game_id"))

        if not gid:
            blanks.append(f"row={index}")
            continue

        if gid in ids:
            duplicates.add(gid)

        ids.add(gid)
        versions[gid] = clean(row.get("feature_version"))

    return {
        "row_count": frame.height,
        "game_ids": ids,
        "blank_game_id_rows": blanks,
        "duplicate_game_ids": sorted(duplicates),
        "feature_version_by_game_id": versions,
    }


def mismatch_source_ids(
    rows: list[dict],
    expected_source: str,
) -> list[str]:
    bad = []

    for row in rows:
        gid = clean_id(row.get("game_id"))
        if gid and clean(row.get("model_source")).lower() != expected_source:
            bad.append(gid)

    return sorted(set(bad))


def mismatch_version_ids(
    rows: list[dict],
    column: str,
    expected: str,
) -> list[str]:
    if not expected:
        return []

    return sorted(
        {
            clean_id(row.get("game_id"))
            for row in rows
            if clean_id(row.get("game_id"))
            and clean(row.get(column)) != expected
        }
    )


def current_sdv_league_health(
    league: str,
    now: datetime,
    season_config: dict[str, dict[str, int]],
    model_cfg: dict,
    production_source: str | None,
    sdv_context: dict,
    ensemble_context: dict,
) -> tuple[dict, list[str]]:
    upper = LABEL[league]
    date = now.strftime("%Y_%m_%d")
    active = in_season(league, now, season_config)
    ensemble_enabled = ensemble_enabled_for_league(
        model_cfg,
        league,
        production_source,
    )

    daily_path = BASE / "daily_games" / league / f"{date}_{upper}.csv"
    feature_path = (
        SDV_FEATURE_ROOT
        / league
        / f"{date}_{upper}_features.parquet"
    )
    sdv_pred_path = (
        SDV_PREDICTION_ROOT
        / league
        / f"{date}_{upper}_predictions.csv"
    )
    ensemble_pred_path = (
        ENSEMBLE_PREDICTION_ROOT
        / league
        / f"{date}_{upper}_predictions.csv"
    )
    cleaned_path = (
        CLEANED_PREDICTION_ROOT
        / league
        / f"{date}_{upper}_predictions.csv"
    )
    merge_path = (
        BASE
        / "01_merge"
        / league
        / "moneyline"
        / f"{date}_{upper}_moneyline.csv"
    )
    source_path = (
        PRODUCTION_ROOTS[production_source]
        / league
        / f"{date}_{upper}_predictions.csv"
        if production_source in PRODUCTION_ROOTS
        else None
    )

    daily_rows = read_rows(daily_path)
    scheduled_ids, scheduled_blanks, scheduled_dupes = row_game_ids(daily_rows)
    issues: list[str] = []

    item = {
        "in_season": active,
        "production_critical": active,
        "ensemble_enabled": ensemble_enabled,
        "configured_production_source": production_source,
        "scheduled_game_count": len(scheduled_ids),
        "scheduled_game_ids": sorted(scheduled_ids),
        "paths": {
            "daily_games": str(daily_path),
            "current_features": str(feature_path),
            "sdv_predictions": str(sdv_pred_path),
            "ensemble_predictions": str(ensemble_pred_path),
            "configured_source_predictions": (
                str(source_path) if source_path else None
            ),
            "cleaned_predictions": str(cleaned_path),
            "merged_moneyline": str(merge_path),
        },
        "feature": {
            "exists": feature_path.exists(),
            "row_count": None,
            "game_ids": [],
            "missing_scheduled_game_ids": [],
            "unexpected_game_ids": [],
            "blank_game_id_rows": [],
            "duplicate_game_ids": [],
            "feature_version_mismatch_game_ids": [],
        },
        "sdv_predictions": {
            "exists": sdv_pred_path.exists(),
            "row_count": 0,
            "game_ids": [],
            "missing_scheduled_game_ids": [],
            "unexpected_game_ids": [],
            "model_source_mismatch_game_ids": [],
            "feature_version_mismatch_game_ids": [],
            "model_version_mismatch_game_ids": [],
        },
        "ensemble_predictions": {
            "required": ensemble_enabled,
            "exists": ensemble_pred_path.exists(),
            "row_count": 0,
            "game_ids": [],
            "missing_scheduled_game_ids": [],
            "unexpected_game_ids": [],
            "model_source_mismatch_game_ids": [],
            "feature_version_mismatch_game_ids": [],
            "ensemble_version_mismatch_game_ids": [],
        },
        "production_source": {
            "raw_exists": source_path.exists() if source_path else False,
            "raw_row_count": 0,
            "raw_game_ids": [],
            "raw_missing_scheduled_game_ids": [],
            "cleaned_exists": cleaned_path.exists(),
            "cleaned_row_count": 0,
            "cleaned_game_ids": [],
            "cleaned_missing_scheduled_game_ids": [],
            "cleaned_model_source_mismatch_game_ids": [],
            "merged_exists": merge_path.exists(),
            "merged_row_count": 0,
            "merged_model_source_mismatch_game_ids": [],
            "source_confirmed_downstream": False,
        },
        "issues": issues,
    }

    if scheduled_blanks:
        issues.append(
            f"{upper}: current daily file {daily_path} has blank game_id "
            f"rows={scheduled_blanks}"
        )

    if scheduled_dupes:
        issues.append(
            f"{upper}: current daily file {daily_path} has duplicate "
            f"game_ids={format_ids(scheduled_dupes)}"
        )

    # sdv_predict intentionally skips true zero-game slates.
    if scheduled_ids:
        # ---------------------------------------------------------------------
        # CURRENT SDV FEATURE FILE
        # ---------------------------------------------------------------------

        if not feature_path.exists():
            issues.append(
                f"{upper}: missing current SDV feature file: {feature_path}"
            )
        else:
            try:
                feature_info = read_current_feature_file(feature_path)
            except Exception as exc:
                issues.append(
                    f"{upper}: unable to validate current SDV feature file "
                    f"{feature_path}: {exc}"
                )
            else:
                feature_ids = feature_info["game_ids"]
                missing = sorted(scheduled_ids - feature_ids)
                unexpected = sorted(feature_ids - scheduled_ids)
                expected_feature = clean(sdv_context.get("feature_version"))

                version_bad = sorted(
                    gid
                    for gid, actual
                    in feature_info["feature_version_by_game_id"].items()
                    if expected_feature and actual != expected_feature
                )

                item["feature"].update({
                    "row_count": feature_info["row_count"],
                    "game_ids": sorted(feature_ids),
                    "missing_scheduled_game_ids": missing,
                    "unexpected_game_ids": unexpected,
                    "blank_game_id_rows": (
                        feature_info["blank_game_id_rows"]
                    ),
                    "duplicate_game_ids": (
                        feature_info["duplicate_game_ids"]
                    ),
                    "feature_version_mismatch_game_ids": version_bad,
                })

                if feature_info["row_count"] != len(scheduled_ids):
                    issues.append(
                        f"{upper}: current SDV feature row count mismatch "
                        f"file={feature_path} expected={len(scheduled_ids)} "
                        f"actual={feature_info['row_count']}"
                    )

                if feature_info["blank_game_id_rows"]:
                    issues.append(
                        f"{upper}: current SDV feature file={feature_path} "
                        "has blank game_id rows="
                        f"{feature_info['blank_game_id_rows']}"
                    )

                if feature_info["duplicate_game_ids"]:
                    issues.append(
                        f"{upper}: current SDV feature file={feature_path} "
                        "has duplicate game_ids="
                        f"{format_ids(feature_info['duplicate_game_ids'])}"
                    )

                if missing:
                    issues.append(
                        f"{upper}: scheduled game_ids missing from current SDV "
                        f"features {feature_path}: {format_ids(missing)}"
                    )

                if unexpected:
                    issues.append(
                        f"{upper}: unexpected game_ids in current SDV features "
                        f"{feature_path}: {format_ids(unexpected)}"
                    )

                if version_bad:
                    issues.append(
                        f"{upper}: feature/model feature_version mismatch "
                        f"file={feature_path} "
                        f"model_feature_version={expected_feature!r} "
                        f"game_ids={format_ids(version_bad)}"
                    )

        # ---------------------------------------------------------------------
        # CURRENT SDV PREDICTIONS
        # ---------------------------------------------------------------------

        if not sdv_pred_path.exists():
            issues.append(
                f"{upper}: missing current SDV prediction file: {sdv_pred_path}"
            )
        else:
            sdv_rows = read_rows(sdv_pred_path)
            sdv_ids, sdv_blanks, sdv_dupes = row_game_ids(sdv_rows)

            missing = sorted(scheduled_ids - sdv_ids)
            unexpected = sorted(sdv_ids - scheduled_ids)

            expected_feature = clean(sdv_context.get("feature_version"))
            expected_model = clean(sdv_context.get("model_version"))

            source_bad = mismatch_source_ids(
                sdv_rows,
                "sdv",
            )

            feature_bad = mismatch_version_ids(
                sdv_rows,
                "feature_version",
                expected_feature,
            )

            model_bad = mismatch_version_ids(
                sdv_rows,
                "model_version",
                expected_model,
            )

            item["sdv_predictions"].update({
                "row_count": len(sdv_rows),
                "game_ids": sorted(sdv_ids),
                "missing_scheduled_game_ids": missing,
                "unexpected_game_ids": unexpected,
                "model_source_mismatch_game_ids": source_bad,
                "feature_version_mismatch_game_ids": feature_bad,
                "model_version_mismatch_game_ids": model_bad,
            })

            if len(sdv_rows) != len(scheduled_ids):
                issues.append(
                    f"{upper}: current SDV prediction row count mismatch "
                    f"file={sdv_pred_path} expected={len(scheduled_ids)} "
                    f"actual={len(sdv_rows)}"
                )

            if sdv_blanks:
                issues.append(
                    f"{upper}: current SDV prediction file {sdv_pred_path} "
                    f"has blank game_id rows={sdv_blanks}"
                )

            if sdv_dupes:
                issues.append(
                    f"{upper}: current SDV prediction file {sdv_pred_path} "
                    f"has duplicate game_ids={format_ids(sdv_dupes)}"
                )

            if missing:
                issues.append(
                    f"{upper}: scheduled game_ids missing from SDV predictions "
                    f"{sdv_pred_path}: {format_ids(missing)}"
                )

            if unexpected:
                issues.append(
                    f"{upper}: unexpected game_ids in SDV predictions "
                    f"{sdv_pred_path}: {format_ids(unexpected)}"
                )

            if source_bad:
                issues.append(
                    f"{upper}: SDV prediction model_source mismatch "
                    f"file={sdv_pred_path} expected='sdv' "
                    f"game_ids={format_ids(source_bad)}"
                )

            if feature_bad:
                issues.append(
                    f"{upper}: SDV prediction feature_version mismatch "
                    f"file={sdv_pred_path} expected={expected_feature!r} "
                    f"game_ids={format_ids(feature_bad)}"
                )

            if model_bad:
                issues.append(
                    f"{upper}: SDV prediction model_version mismatch "
                    f"file={sdv_pred_path} expected={expected_model!r} "
                    f"game_ids={format_ids(model_bad)}"
                )

        # ---------------------------------------------------------------------
        # CURRENT ENSEMBLE PREDICTIONS
        # ---------------------------------------------------------------------

        if ensemble_enabled:
            if not ensemble_pred_path.exists():
                issues.append(
                    f"{upper}: ensemble is enabled but current ensemble "
                    f"prediction file is missing: {ensemble_pred_path}"
                )
            else:
                ensemble_rows = read_rows(ensemble_pred_path)

                (
                    ensemble_ids,
                    ensemble_blanks,
                    ensemble_dupes,
                ) = row_game_ids(ensemble_rows)

                missing = sorted(scheduled_ids - ensemble_ids)
                unexpected = sorted(ensemble_ids - scheduled_ids)

                expected_feature = clean(
                    sdv_context.get("feature_version")
                )
                expected_ensemble = clean(
                    ensemble_context.get("ensemble_version")
                )

                source_bad = mismatch_source_ids(
                    ensemble_rows,
                    "ensemble",
                )

                feature_bad = mismatch_version_ids(
                    ensemble_rows,
                    "feature_version",
                    expected_feature,
                )

                ensemble_bad = mismatch_version_ids(
                    ensemble_rows,
                    "ensemble_version",
                    expected_ensemble,
                )

                item["ensemble_predictions"].update({
                    "row_count": len(ensemble_rows),
                    "game_ids": sorted(ensemble_ids),
                    "missing_scheduled_game_ids": missing,
                    "unexpected_game_ids": unexpected,
                    "model_source_mismatch_game_ids": source_bad,
                    "feature_version_mismatch_game_ids": feature_bad,
                    "ensemble_version_mismatch_game_ids": ensemble_bad,
                })

                if len(ensemble_rows) != len(scheduled_ids):
                    issues.append(
                        f"{upper}: current ensemble prediction row count "
                        f"mismatch file={ensemble_pred_path} "
                        f"expected={len(scheduled_ids)} "
                        f"actual={len(ensemble_rows)}"
                    )

                if ensemble_blanks:
                    issues.append(
                        f"{upper}: current ensemble prediction file "
                        f"{ensemble_pred_path} has blank game_id "
                        f"rows={ensemble_blanks}"
                    )

                if ensemble_dupes:
                    issues.append(
                        f"{upper}: current ensemble prediction file "
                        f"{ensemble_pred_path} has duplicate game_ids="
                        f"{format_ids(ensemble_dupes)}"
                    )

                if missing:
                    issues.append(
                        f"{upper}: scheduled game_ids missing from ensemble "
                        f"predictions {ensemble_pred_path}: "
                        f"{format_ids(missing)}"
                    )

                if unexpected:
                    issues.append(
                        f"{upper}: unexpected game_ids in ensemble predictions "
                        f"{ensemble_pred_path}: {format_ids(unexpected)}"
                    )

                if source_bad:
                    issues.append(
                        f"{upper}: ensemble prediction model_source mismatch "
                        f"file={ensemble_pred_path} expected='ensemble' "
                        f"game_ids={format_ids(source_bad)}"
                    )

                if feature_bad:
                    issues.append(
                        f"{upper}: ensemble prediction feature_version mismatch "
                        f"file={ensemble_pred_path} "
                        f"expected={expected_feature!r} "
                        f"game_ids={format_ids(feature_bad)}"
                    )

                if ensemble_bad:
                    issues.append(
                        f"{upper}: ensemble prediction ensemble_version mismatch "
                        f"file={ensemble_pred_path} "
                        f"expected={expected_ensemble!r} "
                        f"game_ids={format_ids(ensemble_bad)}"
                    )

        # ---------------------------------------------------------------------
        # CONFIGURED PRODUCTION SOURCE -> DOWNSTREAM CLEANED SLATE
        # ---------------------------------------------------------------------

        source_ids: set[str] = set()

        if source_path is None:
            issues.append(
                f"{upper}: configured production prediction source is invalid: "
                f"{production_source!r}"
            )

        elif not source_path.exists():
            issues.append(
                f"{upper}: configured production source prediction file is "
                f"missing: source={production_source} path={source_path}"
            )

        else:
            source_rows = read_rows(source_path)
            source_ids, source_blanks, source_dupes = row_game_ids(source_rows)

            source_missing = sorted(
                scheduled_ids - source_ids
            )

            item["production_source"]["raw_row_count"] = len(source_rows)
            item["production_source"]["raw_game_ids"] = sorted(source_ids)
            item["production_source"][
                "raw_missing_scheduled_game_ids"
            ] = source_missing

            if source_blanks:
                issues.append(
                    f"{upper}: configured production source file "
                    f"{source_path} has blank game_id rows={source_blanks}"
                )

            if source_dupes:
                issues.append(
                    f"{upper}: configured production source file "
                    f"{source_path} has duplicate game_ids="
                    f"{format_ids(source_dupes)}"
                )

            if source_missing:
                issues.append(
                    f"{upper}: configured production source="
                    f"{production_source} is missing scheduled game_ids in "
                    f"{source_path}: {format_ids(source_missing)}"
                )

        if not cleaned_path.exists():
            issues.append(
                f"{upper}: downstream cleaned prediction slate is missing: "
                f"{cleaned_path}"
            )

        else:
            cleaned_rows = read_rows(cleaned_path)

            (
                cleaned_ids,
                cleaned_blanks,
                cleaned_dupes,
            ) = row_game_ids(cleaned_rows)

            cleaned_missing = sorted(
                scheduled_ids - cleaned_ids
            )

            source_bad = mismatch_source_ids(
                cleaned_rows,
                production_source or "",
            )

            item["production_source"]["cleaned_row_count"] = len(
                cleaned_rows
            )
            item["production_source"]["cleaned_game_ids"] = sorted(
                cleaned_ids
            )
            item["production_source"][
                "cleaned_missing_scheduled_game_ids"
            ] = cleaned_missing
            item["production_source"][
                "cleaned_model_source_mismatch_game_ids"
            ] = source_bad

            if cleaned_blanks:
                issues.append(
                    f"{upper}: downstream cleaned slate {cleaned_path} has "
                    f"blank game_id rows={cleaned_blanks}"
                )

            if cleaned_dupes:
                issues.append(
                    f"{upper}: downstream cleaned slate {cleaned_path} has "
                    f"duplicate game_ids={format_ids(cleaned_dupes)}"
                )

            if cleaned_missing:
                issues.append(
                    f"{upper}: downstream cleaned prediction slate "
                    f"{cleaned_path} is missing scheduled game_ids="
                    f"{format_ids(cleaned_missing)}"
                )

            if source_bad:
                issues.append(
                    f"{upper}: model_config source={production_source!r} did "
                    f"not produce downstream cleaned slate {cleaned_path}; "
                    f"model_source mismatch game_ids={format_ids(source_bad)}"
                )

            if (
                source_ids
                and scheduled_ids <= source_ids
                and scheduled_ids <= cleaned_ids
                and not source_bad
            ):
                item["production_source"][
                    "source_confirmed_downstream"
                ] = True

        # merge_intake is the first downstream consumer after cleaned inputs.
        if merge_path.exists():
            merged_rows = read_rows(merge_path)

            merged_source_bad = mismatch_source_ids(
                merged_rows,
                production_source or "",
            )

            item["production_source"]["merged_row_count"] = len(
                merged_rows
            )
            item["production_source"][
                "merged_model_source_mismatch_game_ids"
            ] = merged_source_bad

            if merged_source_bad:
                issues.append(
                    f"{upper}: merged downstream slate={merge_path} does not "
                    f"match model_config source={production_source!r}; "
                    f"model_source mismatch game_ids="
                    f"{format_ids(merged_source_bad)}"
                )

    critical_failures = list(issues) if active else []

    item["critical_failures"] = critical_failures
    item["valid"] = not issues

    return item, critical_failures


# =============================================================================
# COMBINED SDV HEALTH
# =============================================================================

def sdv_health(
    now: datetime,
    season_config: dict[str, dict[str, int]],
) -> tuple[dict, list[str]]:
    fatals: list[str] = []
    warnings: list[str] = []

    season_mappings, seasons_report, errors = (
        validate_sdv_seasons_config()
    )
    fatals.extend(errors)

    _, storage_report, errors = validate_sdv_storage_config(
        season_mappings
    )
    fatals.extend(errors)

    _, sdv_model_report, errors = validate_sdv_model_config()
    fatals.extend(errors)

    (
        model_cfg,
        production_source,
        model_config_report,
        errors,
    ) = validate_model_config()
    fatals.extend(errors)

    expected_feature = clean(
        sdv_model_report.get("feature_version")
    )
    expected_model = clean(
        sdv_model_report.get("model_version")
    )

    (
        model_report,
        sdv_contexts,
        errors,
    ) = validate_sdv_models(
        expected_feature,
        expected_model,
    )
    fatals.extend(errors)

    (
        ensemble_report,
        errors,
        noncritical,
    ) = validate_ensemble_weights(
        model_cfg,
        production_source,
        sdv_contexts,
    )
    fatals.extend(errors)
    warnings.extend(noncritical)

    if (
        storage_report.get("historical_internal_seasons")
        and season_mappings
    ):
        (
            manifest_report,
            errors,
        ) = validate_historical_manifests(
            season_mappings,
            storage_report,
        )
        fatals.extend(errors)

    else:
        manifest_report = {
            "required_count": 0,
            "valid_count": 0,
            "manifests": [],
            "skipped": True,
            "reason": (
                "SDV season/storage config was not valid enough to resolve "
                "required manifests"
            ),
        }

    current = {}

    for league in LEAGUES:
        item, errors = current_sdv_league_health(
            league,
            now,
            season_config,
            model_cfg,
            production_source,
            sdv_contexts.get(league, {}),
            ensemble_report.get(league, {}),
        )

        current[league] = item
        fatals.extend(errors)

        if (
            item.get("issues")
            and not item.get("production_critical")
        ):
            warnings.extend(item["issues"])

    fatals = list(dict.fromkeys(fatals))

    warnings = [
        value
        for value in dict.fromkeys(warnings)
        if value not in fatals
    ]

    return {
        "configured_production_source": production_source,
        "configs": {
            "sdv_seasons": seasons_report,
            "sdv_storage": storage_report,
            "sdv_model": sdv_model_report,
            "model_config": model_config_report,
        },
        "historical_manifests": manifest_report,
        "model_artifacts": model_report,
        "ensemble_weights": ensemble_report,
        "current": current,
        "warnings": warnings,
    }, fatals


# =============================================================================
# WNBA BIAS DRIFT
# =============================================================================

def load_all_csv(
    folder: Path,
    pattern: str = "*.csv",
) -> list[dict]:
    rows = []

    if not folder.exists():
        return rows

    for path in sorted(folder.glob(pattern)):
        try:
            rows.extend(read_rows(path))
        except (OSError, UnicodeError, csv.Error):
            continue

    return rows


def wnba_bias_drift() -> dict:
    preds = load_all_csv(
        BASE / "00_intake/predictions/predictions_cleaned/wnba",
        "*_WNBA_predictions.csv",
    )

    finals = load_all_csv(
        BASE / "05_final_scores/results/wnba",
        "*_final_scores_WNBA.csv",
    )

    finals_by_id = {}
    finals_by_comp = {}

    for row in finals:
        if (
            fnum(row.get("home_score")) is None
            or fnum(row.get("away_score")) is None
        ):
            continue

        gid = clean_id(row.get("game_id"))
        key = comp(row)

        if gid:
            finals_by_id[gid] = row

        if all(key):
            finals_by_comp[key] = row

    residuals = []

    for prediction in preds:
        gid = clean_id(prediction.get("game_id"))

        final = finals_by_id.get(gid) if gid else None

        if final is None:
            final = finals_by_comp.get(
                comp(prediction)
            )

        if final is None:
            continue

        hp = fnum(
            prediction.get("home_projected_points")
        )
        ap = fnum(
            prediction.get("away_projected_points")
        )
        tp = fnum(
            prediction.get("total_projected_points")
        )
        hs = fnum(
            final.get("home_score")
        )
        aws = fnum(
            final.get("away_score")
        )

        if None in (
            hp,
            ap,
            tp,
            hs,
            aws,
        ):
            continue

        residuals.append({
            "game_date": clean(
                prediction.get("game_date")
            ),
            "game_id": (
                gid
                or clean_id(
                    final.get("game_id")
                )
            ),
            "margin_residual_projected_minus_actual": (
                (hp - ap)
                - (hs - aws)
            ),
            "total_residual_projected_minus_actual": (
                tp
                - (hs + aws)
            ),
        })

    residuals.sort(
        key=lambda sort_row: (
            sort_row["game_date"],
            sort_row["game_id"],
        )
    )

    windows = {}
    warnings = []

    for n in DRIFT_WINDOWS:
        sample = residuals[-n:]

        if not sample:
            windows[str(n)] = {
                "games": 0,
                "margin_mean_residual": None,
                "total_mean_residual": None,
                "warning": False,
            }
            continue

        margin = (
            sum(
                row[
                    "margin_residual_projected_minus_actual"
                ]
                for row in sample
            )
            / len(sample)
        )

        total = (
            sum(
                row[
                    "total_residual_projected_minus_actual"
                ]
                for row in sample
            )
            / len(sample)
        )

        warn = (
            len(sample) >= min(n, 25)
            and (
                abs(margin) >= DRIFT_WARN
                or abs(total) >= DRIFT_WARN
            )
        )

        windows[str(n)] = {
            "games": len(sample),
            "margin_mean_residual": round(
                margin,
                4,
            ),
            "total_mean_residual": round(
                total,
                4,
            ),
            "warning": warn,
        }

        if warn:
            warnings.append(
                f"WNBA {len(sample)}-game residual drift exceeds "
                f"{DRIFT_WARN}: margin={margin:.3f}, "
                f"total={total:.3f}"
            )

    return {
        "definition": "adjusted_projected_minus_actual",
        "warning_threshold_abs_points": DRIFT_WARN,
        "matched_games": len(residuals),
        "windows": windows,
        "warnings": warnings,
    }


def write_wnba_drift_report(
    drift: dict,
    generated_at_utc: str,
) -> None:
    WNBA_DRIFT_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "generated_at_utc",
        "matched_games",
        "window_games",
        "sample_games",
        "margin_mean_residual",
        "total_mean_residual",
        "threshold_points",
        "warning",
    ]

    tmp = WNBA_DRIFT_REPORT.with_suffix(
        WNBA_DRIFT_REPORT.suffix
        + ".tmp"
    )

    with tmp.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for n in DRIFT_WINDOWS:
            values = (
                drift
                .get(
                    "windows",
                    {},
                )
                .get(
                    str(n),
                    {},
                )
            )

            writer.writerow({
                "generated_at_utc": (
                    generated_at_utc
                ),
                "matched_games": (
                    drift.get(
                        "matched_games",
                        0,
                    )
                ),
                "window_games": n,
                "sample_games": (
                    values.get(
                        "games",
                        0,
                    )
                ),
                "margin_mean_residual": (
                    values.get(
                        "margin_mean_residual"
                    )
                ),
                "total_mean_residual": (
                    values.get(
                        "total_mean_residual"
                    )
                ),
                "threshold_points": (
                    drift.get(
                        "warning_threshold_abs_points"
                    )
                ),
                "warning": bool(
                    values.get(
                        "warning",
                        False,
                    )
                ),
            })

    tmp.replace(
        WNBA_DRIFT_REPORT
    )


# =============================================================================
# OUTPUT / MAIN
# =============================================================================

def write_failure_for_season_config(
    now: datetime,
    exc: Exception,
) -> None:
    generated_at_utc = datetime.now(
        UTC
    ).isoformat()

    fatal = (
        f"season config failure: {exc}"
    )

    payload = {
        "schema_version": 1,
        "generated_at_utc": (
            generated_at_utc
        ),
        "game_date_new_york": (
            now.strftime(
                "%Y_%m_%d"
            )
        ),
        "season_config_path": str(
            SEASON_CONFIG
        ),
        "status": "failed",
        "fatal_errors": [
            fatal
        ],
        "stage_status": [],
        "leagues": {},
        "sdv_health": {},
        "wnba_bias_drift": {},
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        (
            "=== basketball pipeline "
            f"health {generated_at_utc} ==="
        ),
        (
            "game_date_new_york: "
            f"{payload['game_date_new_york']}"
        ),
        f"season_config: {SEASON_CONFIG}",
        "status: failed",
        f"FATAL: {fatal}",
    ]

    LOG.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(
        "\n".join(lines)
    )


def main() -> None:
    now = datetime.now(NY)

    try:
        season_config = load_season_config()

    except Exception as exc:
        write_failure_for_season_config(
            now,
            exc,
        )
        sys.exit(1)

    stages, fatals = stage_health()

    leagues = {}

    for league in LEAGUES:
        item, errors = current_league_health(
            league,
            now,
            season_config,
        )

        leagues[league] = item
        fatals.extend(errors)

    sdv, sdv_fatals = sdv_health(
        now,
        season_config,
    )

    fatals.extend(
        sdv_fatals
    )

    fatals = list(
        dict.fromkeys(
            fatals
        )
    )

    drift = wnba_bias_drift()

    generated_at_utc = datetime.now(
        UTC
    ).isoformat()

    write_wnba_drift_report(
        drift,
        generated_at_utc,
    )

    drift[
        "report_path"
    ] = str(
        WNBA_DRIFT_REPORT
    )

    payload = {
        "schema_version": 1,
        "generated_at_utc": (
            generated_at_utc
        ),
        "game_date_new_york": (
            now.strftime(
                "%Y_%m_%d"
            )
        ),
        "season_config_path": str(
            SEASON_CONFIG
        ),
        "status": (
            "failed"
            if fatals
            else "healthy"
        ),
        "fatal_errors": fatals,
        "stage_status": stages,
        "leagues": leagues,
        "sdv_health": sdv,
        "wnba_bias_drift": drift,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        (
            "=== basketball pipeline "
            f"health {generated_at_utc} ==="
        ),
        (
            "game_date_new_york: "
            f"{payload['game_date_new_york']}"
        ),
        f"season_config: {SEASON_CONFIG}",
        (
            "sdv_seasons_config: "
            f"{SDV_SEASONS_CONFIG}"
        ),
        (
            "sdv_storage_config: "
            f"{SDV_STORAGE_CONFIG}"
        ),
        (
            "model_config: "
            f"{MODEL_CONFIG}"
        ),
        (
            "production_prediction_source: "
            f"{sdv.get('configured_production_source')}"
        ),
        f"status: {payload['status']}",
    ]

    for league, item in leagues.items():
        counts = item["counts"]
        season = item["season_config"]

        lines.append(
            f"{league.upper()}: "
            f"in_season={item['in_season']} "
            f"season="
            f"{season['start_month']:02d}/"
            f"{season['start_day']:02d}-"
            f"{season['end_month']:02d}/"
            f"{season['end_day']:02d} "
            f"scheduled="
            f"{counts['scheduled_games']} "
            f"predictions="
            f"{counts['prediction_games']} "
            f"sportsbook="
            f"{counts['sportsbook_games']} "
            f"merged="
            f"{counts['merged_games']} "
            f"selected="
            f"{counts['selected_bets']} "
            f"locked="
            f"{counts['locked_bets']}"
        )

    manifests = sdv.get(
        "historical_manifests",
        {},
    )

    lines.append(
        "SDV historical manifests: "
        f"valid="
        f"{manifests.get('valid_count', 0)}/"
        f"{manifests.get('required_count', 0)}"
    )

    for league in LEAGUES:
        current = (
            sdv
            .get(
                "current",
                {},
            )
            .get(
                league,
                {},
            )
        )

        feature = current.get(
            "feature",
            {},
        )

        sdv_preds = current.get(
            "sdv_predictions",
            {},
        )

        ensemble_preds = current.get(
            "ensemble_predictions",
            {},
        )

        source = current.get(
            "production_source",
            {},
        )

        lines.append(
            f"SDV {league.upper()}: "
            f"in_season="
            f"{current.get('in_season')} "
            f"scheduled="
            f"{current.get('scheduled_game_count', 0)} "
            f"feature_rows="
            f"{feature.get('row_count')} "
            f"sdv_prediction_rows="
            f"{sdv_preds.get('row_count', 0)} "
            f"ensemble_enabled="
            f"{current.get('ensemble_enabled')} "
            f"ensemble_prediction_rows="
            f"{ensemble_preds.get('row_count', 0)} "
            f"configured_source="
            f"{current.get('configured_production_source')} "
            f"source_confirmed_downstream="
            f"{source.get('source_confirmed_downstream')}"
        )

    for warning in sdv.get(
        "warnings",
        [],
    ):
        lines.append(
            f"WARN: {warning}"
        )

    lines.append(
        f"WNBA drift report: "
        f"{WNBA_DRIFT_REPORT}"
    )

    for n, values in drift[
        "windows"
    ].items():
        lines.append(
            f"WNBA drift {n}: "
            f"games="
            f"{values['games']} "
            f"margin="
            f"{values['margin_mean_residual']} "
            f"total="
            f"{values['total_mean_residual']} "
            f"warning="
            f"{values['warning']}"
        )

    for warning in drift[
        "warnings"
    ]:
        lines.append(
            f"WARN: {warning}"
        )

    for error in fatals:
        lines.append(
            f"FATAL: {error}"
        )

    LOG.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(
        "\n".join(lines)
    )

    if fatals:
        sys.exit(1)


if __name__ == "__main__":
    main()
