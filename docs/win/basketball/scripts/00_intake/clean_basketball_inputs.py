#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/clean_basketball_inputs.py
#
# Reads originals from:
#   docs/win/basketball/00_intake/predictions/{league}/
#   docs/win/basketball/00_intake/sportsbook/{league}/
#
# Writes cleaned copies to:
#   docs/win/basketball/00_intake/predictions/predictions_cleaned/{league}/
#   docs/win/basketball/00_intake/sportsbook/sportsbook_cleaned/{league}/
#
# Originals are never mutated. Filenames are preserved.
# Permanent bias rules come from model_config.yaml. Dynamic rolling/regime-aware
# bias values come from rolling_bias_state.yaml. Past cleaned prediction files are preserved so
# today's rolling bias is never retroactively applied to historical predictions.

import csv
import re
import traceback
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import yaml

# =========================
# PATHS
# =========================

PREDICTION_ROOTS = {
    "dratings": Path(
        "docs/win/basketball/00_intake/predictions"
    ),
    "sdv": Path(
        "docs/win/basketball/00_intake/predictions_sdv"
    ),
    "ensemble": Path(
        "docs/win/basketball/00_intake/predictions_ensemble"
    ),
}

SPORTSBOOK_DIRS = {
    "NBA": Path("docs/win/basketball/00_intake/sportsbook/nba"),
    "NCAAM": Path("docs/win/basketball/00_intake/sportsbook/ncaam"),
    "WNBA": Path("docs/win/basketball/00_intake/sportsbook/wnba"),
}

CLEANED_PREDICTION_DIRS = {
    "NBA": Path("docs/win/basketball/00_intake/predictions/predictions_cleaned/nba"),
    "NCAAM": Path("docs/win/basketball/00_intake/predictions/predictions_cleaned/ncaam"),
    "WNBA": Path("docs/win/basketball/00_intake/predictions/predictions_cleaned/wnba"),
}

CLEANED_SPORTSBOOK_DIRS = {
    "NBA": Path("docs/win/basketball/00_intake/sportsbook/sportsbook_cleaned/nba"),
    "NCAAM": Path("docs/win/basketball/00_intake/sportsbook/sportsbook_cleaned/ncaam"),
    "WNBA": Path("docs/win/basketball/00_intake/sportsbook/sportsbook_cleaned/wnba"),
}

CONFIG_PATH = Path("docs/win/basketball/config/model_config.yaml")
BIAS_STATE_PATH = Path("docs/win/basketball/config/rolling_bias_state.yaml")

ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)

MASTER_LOG_FILE = ERROR_DIR / "clean_basketball_inputs.txt"

LEAGUE_LOG_FILES = {
    "NBA": ERROR_DIR / "clean_basketball_inputs_nba.txt",
    "NCAAM": ERROR_DIR / "clean_basketball_inputs_ncaam.txt",
    "WNBA": ERROR_DIR / "clean_basketball_inputs_wnba.txt",
}

NY_TZ = ZoneInfo("America/New_York")

# =========================
# SETTINGS
# =========================

ODDS_CHECKS = [
    ("home_dk_moneyline_decimal", "away_dk_moneyline_decimal", "ML"),
    ("dk_total_over_decimal",     "dk_total_under_decimal",    "TOTAL"),
    ("home_dk_spread_decimal",    "away_dk_spread_decimal",    "SPREAD"),
]

MARKET_FIELDS = {
    "ML": [
        "home_dk_moneyline_american",
        "away_dk_moneyline_american",
        "home_dk_moneyline_decimal",
        "away_dk_moneyline_decimal",
        "home_decimal",
        "away_decimal",
    ],
    "TOTAL": [
        "total",
        "dk_total_over_american",
        "dk_total_under_american",
        "dk_total_over_decimal",
        "dk_total_under_decimal",
    ],
    "SPREAD": [
        "home_spread",
        "away_spread",
        "home_dk_spread_american",
        "away_dk_spread_american",
        "home_dk_spread_decimal",
        "away_dk_spread_decimal",
    ],
}

SPREAD_OUTLIER_MAX = 25.0
TOTAL_OUTLIER_MAX = 40.0

BIAS_FLAG_COLUMN = "bias_applied"
BIAS_FLAG_VALUE = "1"
MARGIN_BIAS_COLUMN = "margin_bias"
TOTAL_BIAS_COLUMN = "total_bias"

# =========================
# LOGGING
# =========================

def init_logs() -> None:
    run_line = f"=== clean_basketball_inputs RUN {datetime.now().isoformat()} ===\n"

    with open(MASTER_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(run_line)

    for league, path in LEAGUE_LOG_FILES.items():
        with open(path, "w", encoding="utf-8") as f:
            f.write(run_line)
            f.write(f"=== LEAGUE LOG: {league} ===\n")


def log_master(msg: str) -> None:
    with open(MASTER_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


def log_league(league: str, msg: str) -> None:
    path = LEAGUE_LOG_FILES.get(league)
    if path is None:
        log_master(f"{league} | {msg}")
        return

    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


# =========================
# HELPERS
# =========================

def to_float(value):
    try:
        if value is None:
            return None
        value = str(value).strip()
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def odds_malformed_reason(raw_a, raw_b):
    raw_a_str = "" if raw_a is None else str(raw_a).strip()
    raw_b_str = "" if raw_b is None else str(raw_b).strip()

    if raw_a_str == "" and raw_b_str == "":
        return "BOTH_BLANK"

    if raw_a_str == "" or raw_b_str == "":
        return "MISSING_PAIRED_ODDS"

    dec_a = to_float(raw_a)
    dec_b = to_float(raw_b)

    if dec_a is None or dec_b is None:
        return "NON_NUMERIC"

    if dec_a <= 1 or dec_b <= 1:
        return "DECIMAL_LESS_THAN_OR_EQUAL_1"

    return None


def row_key(row):
    game_id = str(row.get("game_id", "")).strip()
    if game_id:
        return game_id

    return "|".join([
        str(row.get("game_date", "")).strip(),
        str(row.get("home_team", "")).strip(),
        str(row.get("away_team", "")).strip(),
    ])


def read_csv(path: Path):
    repo_root = Path(__file__).resolve().parents[5]
    candidate = path if path.is_absolute() else Path.cwd() / path
    safe_path = candidate.resolve(strict=True)
    try:
        safe_path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {path}") from exc

    with safe_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return fieldnames, rows


def write_csv(path: Path, fieldnames, rows):
    repo_root = Path(__file__).resolve().parents[5]
    candidate = path if path.is_absolute() else Path.cwd() / path
    safe_path = candidate.resolve()
    try:
        safe_path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {path}") from exc

    safe_path.parent.mkdir(parents=True, exist_ok=True)
    with safe_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def csv_files(folder: Path, league: str):
    if not folder.exists():
        log_league(league, f"WARN | Missing folder: {folder}")
        return []
    return sorted(folder.rglob("*.csv"))


def book_home_spread_to_model_margin(home_spread):
    if home_spread is None:
        return None
    return -home_spread


def blank_market_fields(row, fieldnames, market):
    blanked = []

    for col in MARKET_FIELDS.get(market, []):
        if col in fieldnames:
            old = row.get(col, "")
            if str(old).strip() != "":
                blanked.append(col)
            row[col] = ""

    return blanked


def add_market_action(actions, league, path, key, market, reason):
    actions.setdefault(league, {})
    actions[league].setdefault(path, {})
    actions[league][path].setdefault(key, {})
    actions[league][path][key][market] = reason


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return data


def production_prediction_source() -> str:
    model_cfg = load_yaml(
        CONFIG_PATH
    )

    source = str(
        model_cfg.get(
            "production_prediction_source",
            "",
        )
    ).strip().lower()

    if source not in PREDICTION_ROOTS:
        raise ValueError(
            "model_config.yaml "
            "production_prediction_source "
            "must be one of: "
            "dratings, sdv, ensemble"
        )

    return source


def prediction_dirs_for_source(
    source: str,
) -> dict[str, Path]:
    root = PREDICTION_ROOTS[
        source
    ]

    return {
        "NBA": root / "nba",
        "NCAAM": root / "ncaam",
        "WNBA": root / "wnba",
    }


def resolve_bias_values() -> dict:
    model_cfg = load_yaml(CONFIG_PATH)
    state_cfg = load_yaml(BIAS_STATE_PATH)

    leagues_cfg = model_cfg.get("leagues")
    state_leagues = state_cfg.get("leagues")
    if not isinstance(leagues_cfg, dict):
        raise ValueError("model_config.yaml missing top-level leagues mapping")
    if not isinstance(state_leagues, dict):
        raise ValueError("rolling_bias_state.yaml missing top-level leagues mapping")

    resolved = {}

    for league in ("NBA", "NCAAM", "WNBA"):
        key = league.lower()
        league_cfg = leagues_cfg.get(key)
        if not isinstance(league_cfg, dict):
            raise ValueError(f"Missing model config for {league}")
        if str(league_cfg.get("status", "")).strip().lower() != "active":
            raise ValueError(f"{league} is not active in model_config.yaml")

        bias_cfg = league_cfg.get("bias") or {}
        state_league = state_leagues.get(key) or {}
        league_values = {}

        for component in ("margin", "total"):
            rule = bias_cfg.get(component)
            if not isinstance(rule, dict):
                raise ValueError(f"{league} bias.{component} must be configured")

            method = str(rule.get("method", "")).strip().lower()

            if method == "fixed":
                value = to_float(rule.get("value"))
                if value is None:
                    raise ValueError(f"{league} fixed {component} bias requires numeric value")

                window_games = None
                windows_games = None

            elif method in {"rolling", "regime_aware"}:
                state_name = f"{component}_bias"
                state_component = state_league.get(state_name)

                if not isinstance(state_component, dict):
                    raise ValueError(
                        f"{league} missing {state_name} in rolling_bias_state.yaml"
                    )

                if str(state_component.get("status", "")).strip().lower() != "ready":
                    raise ValueError(
                        f"{league} {state_name} is not ready in rolling_bias_state.yaml"
                    )

                state_method = str(
                    state_component.get("method", "")
                ).strip().lower()

                if state_method != method:
                    raise ValueError(
                        f"{league} {state_name} state method mismatch: "
                        f"config={method} state={state_method}"
                    )

                if method == "rolling":
                    configured_window = rule.get("window_games")
                    state_window = state_component.get("window_games")

                    if (
                        configured_window is not None
                        and state_window is not None
                        and int(configured_window) != int(state_window)
                    ):
                        raise ValueError(
                            f"{league} {component} rolling window mismatch: "
                            f"config={configured_window} state={state_window}"
                        )

                    window_games = (
                        int(configured_window)
                        if configured_window is not None
                        else None
                    )
                    windows_games = None

                else:
                    configured_windows = rule.get("windows_games")
                    state_windows = state_component.get("windows_games")

                    if not isinstance(configured_windows, list) or not configured_windows:
                        raise ValueError(
                            f"{league} regime_aware {component} bias requires "
                            f"windows_games in model_config.yaml"
                        )

                    if not isinstance(state_windows, list) or not state_windows:
                        raise ValueError(
                            f"{league} {state_name} missing windows_games "
                            f"in rolling_bias_state.yaml"
                        )

                    configured_windows_int = [
                        int(window)
                        for window in configured_windows
                    ]
                    state_windows_int = [
                        int(window)
                        for window in state_windows
                    ]

                    if configured_windows_int != state_windows_int:
                        raise ValueError(
                            f"{league} {component} regime-aware windows mismatch: "
                            f"config={configured_windows_int} "
                            f"state={state_windows_int}"
                        )

                    configured_weights = rule.get("weights")
                    state_weights = state_component.get("weights")

                    if (
                        not isinstance(configured_weights, list)
                        or len(configured_weights) != len(configured_windows_int)
                    ):
                        raise ValueError(
                            f"{league} regime_aware {component} bias requires "
                            f"one configured weight per window"
                        )

                    if (
                        not isinstance(state_weights, list)
                        or len(state_weights) != len(configured_windows_int)
                    ):
                        raise ValueError(
                            f"{league} {state_name} missing valid weights "
                            f"in rolling_bias_state.yaml"
                        )

                    config_weight_values = [
                        float(weight)
                        for weight in configured_weights
                    ]
                    weight_sum = sum(config_weight_values)

                    if weight_sum <= 0:
                        raise ValueError(
                            f"{league} regime_aware {component} weights "
                            f"must sum to > 0"
                        )

                    normalized_config_weights = [
                        weight / weight_sum
                        for weight in config_weight_values
                    ]
                    state_weight_values = [
                        float(weight)
                        for weight in state_weights
                    ]

                    if any(
                        abs(configured - state) > 1e-6
                        for configured, state
                        in zip(
                            normalized_config_weights,
                            state_weight_values,
                        )
                    ):
                        raise ValueError(
                            f"{league} {component} regime-aware weight mismatch: "
                            f"config={normalized_config_weights} "
                            f"state={state_weight_values}"
                        )

                    configured_shrink = to_float(
                        rule.get("sign_conflict_shrink")
                    )
                    state_shrink = to_float(
                        state_component.get("sign_conflict_shrink")
                    )

                    if configured_shrink is None or state_shrink is None:
                        raise ValueError(
                            f"{league} regime_aware {component} bias requires "
                            f"sign_conflict_shrink in config and state"
                        )

                    if abs(configured_shrink - state_shrink) > 1e-6:
                        raise ValueError(
                            f"{league} {component} regime-aware shrink mismatch: "
                            f"config={configured_shrink} state={state_shrink}"
                        )

                    window_games = max(
                        configured_windows_int
                    )
                    windows_games = (
                        configured_windows_int
                    )

                value = to_float(state_component.get("value"))
                if value is None:
                    raise ValueError(
                        f"{league} {state_name} has no numeric value"
                    )

            elif method == "none":
                value = 0.0
                window_games = None
                windows_games = None

            else:
                raise ValueError(
                    f"Unsupported {league} bias.{component}.method={method!r}"
                )

            league_values[component] = {
                "method": method,
                "value": float(value),
                "window_games": window_games,
                "windows_games": windows_games,
            }

        resolved[league] = league_values

    return resolved


def prediction_file_date(path: Path):
    match = re.match(r"^(\d{4})_(\d{2})_(\d{2})_", path.name)
    if not match:
        return None
    try:
        return datetime(
            int(match.group(1)), int(match.group(2)), int(match.group(3))
        ).date()
    except ValueError:
        return None


# =========================
# LOAD ALL ORIGINALS INTO MEMORY
# =========================

def load_all(dir_map):
    loaded = {}

    for league, folder in dir_map.items():
        loaded[league] = {}

        for path in csv_files(folder, league):
            fieldnames, rows = read_csv(path)
            loaded[league][path] = [fieldnames, rows]

        log_league(
            league,
            f"LOADED | folder={folder} files={len(loaded[league])}"
        )

    return loaded


# =========================
# STEP 1: BLANK MALFORMED MARKET ODDS ONLY
# =========================

def blank_bad_market_odds(book_files):
    blanked_by_market = {
        league: {market: 0 for _, _, market in ODDS_CHECKS}
        for league in book_files
    }

    for league, files in book_files.items():
        for path, data in files.items():
            fieldnames, rows = data
            fieldset = set(fieldnames)

            file_blanked = 0

            for row in rows:
                key = row_key(row)

                for col_a, col_b, market in ODDS_CHECKS:
                    if col_a not in fieldset or col_b not in fieldset:
                        continue

                    raw_a = row.get(col_a)
                    raw_b = row.get(col_b)
                    reason = odds_malformed_reason(raw_a, raw_b)

                    if reason is None:
                        continue

                    blanked_cols = blank_market_fields(row, fieldnames, market)
                    blanked_by_market[league][market] += 1
                    file_blanked += 1

                    log_league(
                        league,
                        f"BLANK_BAD_{market}_MALFORMED | {path} | {key} | "
                        f"reason={reason} a={raw_a!r} b={raw_b!r} "
                        f"blanked_cols={blanked_cols}"
                    )

            if file_blanked:
                log_league(
                    league,
                    f"MARKET_BLANKED | {path} | bad_market_odds_blank_events={file_blanked}"
                )

    return blanked_by_market


# =========================
# STEP 2: BLANK BAD MARKET LINES ONLY
# =========================

def blank_bad_market_lines(book_files):
    blanked_by_market = {
        league: {"TOTAL": 0, "SPREAD": 0}
        for league in book_files
    }

    for league, files in book_files.items():
        for path, data in files.items():
            fieldnames, rows = data
            fieldset = set(fieldnames)

            file_blanked = 0

            for row in rows:
                key = row_key(row)

                if "total" in fieldset:
                    book_total = to_float(row.get("total"))

                    if book_total is not None and book_total <= 0:
                        blanked_cols = blank_market_fields(row, fieldnames, "TOTAL")
                        blanked_by_market[league]["TOTAL"] += 1
                        file_blanked += 1

                        log_league(
                            league,
                            f"BLANK_BAD_TOTAL_LINE | {path} | {key} | "
                            f"reason=TOTAL_LESS_THAN_OR_EQUAL_0 total={book_total} "
                            f"blanked_cols={blanked_cols}"
                        )

                if "home_spread" in fieldset and "away_spread" in fieldset:
                    home_spread = to_float(row.get("home_spread"))
                    away_spread = to_float(row.get("away_spread"))

                    if home_spread is None and away_spread is None:
                        continue

                    if home_spread is None or away_spread is None:
                        blanked_cols = blank_market_fields(row, fieldnames, "SPREAD")
                        blanked_by_market[league]["SPREAD"] += 1
                        file_blanked += 1

                        log_league(
                            league,
                            f"BLANK_BAD_SPREAD_LINE | {path} | {key} | "
                            f"reason=MISSING_PAIRED_SPREAD_LINE "
                            f"home_spread={row.get('home_spread')!r} away_spread={row.get('away_spread')!r} "
                            f"blanked_cols={blanked_cols}"
                        )

            if file_blanked:
                log_league(
                    league,
                    f"MARKET_BLANKED | {path} | bad_market_line_blank_events={file_blanked}"
                )

    return blanked_by_market


# =========================
# STEP 3: BUILD INDEXES
# =========================

def build_pred_index(pred_files):
    index = {}

    for league, files in pred_files.items():
        for path, (fieldnames, rows) in files.items():
            for row in rows:
                key = row_key(row)
                if key:
                    index[key] = {
                        "league": league,
                        "path": path,
                        "row": row,
                    }

    return index


def build_book_index(book_files):
    index = {}

    for league, files in book_files.items():
        for path, (fieldnames, rows) in files.items():
            for row in rows:
                key = row_key(row)
                if key:
                    index.setdefault(key, []).append({
                        "league": league,
                        "path": path,
                        "fieldnames": fieldnames,
                        "row": row,
                    })

    return index


# =========================
# STEP 4: FIND MODEL VS BOOK MARKET OUTLIERS
# =========================

def find_market_outlier_actions(pred_index, book_index):
    actions = {}

    outlier_counts = {
        "NBA": {"spread": 0, "total": 0},
        "NCAAM": {"spread": 0, "total": 0},
        "WNBA": {"spread": 0, "total": 0},
    }

    for key, pred_item in pred_index.items():
        if key not in book_index:
            continue

        pred_league = pred_item["league"]
        pred = pred_item["row"]

        home_proj = to_float(pred.get("home_projected_points"))
        away_proj = to_float(pred.get("away_projected_points"))
        model_total = to_float(pred.get("total_projected_points"))

        for book_item in book_index[key]:
            book = book_item["row"]
            book_path = book_item["path"]

            book_home_spread = to_float(book.get("home_spread"))
            book_total = to_float(book.get("total"))

            if home_proj is not None and away_proj is not None and book_home_spread is not None:
                model_spread = home_proj - away_proj
                book_model_margin = book_home_spread_to_model_margin(book_home_spread)
                spread_diff = abs(model_spread - book_model_margin)

                if spread_diff > SPREAD_OUTLIER_MAX:
                    add_market_action(
                        actions,
                        pred_league,
                        book_path,
                        key,
                        "SPREAD",
                        f"SPREAD_OUTLIER_DIFF_{round(spread_diff, 4)}",
                    )

                    outlier_counts[pred_league]["spread"] += 1

                    log_league(
                        pred_league,
                        f"BLANK_OUTLIER_SPREAD | {key} | {book_path} | "
                        f"model_spread_home_minus_away={round(model_spread, 4)} "
                        f"book_home_spread={book_home_spread} "
                        f"book_spread_converted_to_home_margin={round(book_model_margin, 4)} "
                        f"diff={round(spread_diff, 4)}"
                    )

            if model_total is not None and book_total is not None:
                total_diff = abs(model_total - book_total)

                if total_diff > TOTAL_OUTLIER_MAX:
                    add_market_action(
                        actions,
                        pred_league,
                        book_path,
                        key,
                        "TOTAL",
                        f"TOTAL_OUTLIER_DIFF_{round(total_diff, 4)}",
                    )

                    outlier_counts[pred_league]["total"] += 1

                    log_league(
                        pred_league,
                        f"BLANK_OUTLIER_TOTAL | {key} | {book_path} | "
                        f"model_total={model_total} book_total={book_total} "
                        f"diff={round(total_diff, 4)}"
                    )

    for league, counts in outlier_counts.items():
        log_league(
            league,
            f"OUTLIER SUMMARY | spread_outliers={counts['spread']} total_outliers={counts['total']}"
        )

    return actions, outlier_counts


# =========================
# STEP 5: APPLY MARKET OUTLIER BLANKS
# =========================

def apply_market_outlier_actions(book_files, actions):
    blanked_by_league = {league: 0 for league in book_files}

    for league, files in book_files.items():
        league_actions = actions.get(league, {})

        for path, data in files.items():
            path_actions = league_actions.get(path, {})
            if not path_actions:
                continue

            fieldnames, rows = data
            file_blanked = 0

            for row in rows:
                key = row_key(row)
                market_actions = path_actions.get(key, {})

                if not market_actions:
                    continue

                for market, reason in market_actions.items():
                    blanked_cols = blank_market_fields(row, fieldnames, market)
                    blanked_by_league[league] += 1
                    file_blanked += 1

                    log_league(
                        league,
                        f"MARKET_OUTLIER_BLANKED | SPORTSBOOK | {path} | {key} | "
                        f"market={market} reason={reason} blanked_cols={blanked_cols}"
                    )

            if file_blanked:
                log_league(
                    league,
                    f"MARKET_BLANKED | SPORTSBOOK | {path} | outlier_market_blank_events={file_blanked}"
                )

    return blanked_by_league


# =========================
# STEP 6A: PRESERVE HISTORICAL CLEANED PREDICTIONS
# =========================

def preserve_past_cleaned_predictions(pred_files):
    today = datetime.now(NY_TZ).date()
    preserved = {league: 0 for league in pred_files}
    skipped_missing = {league: 0 for league in pred_files}

    for league, files in pred_files.items():
        cleaned_root = CLEANED_PREDICTION_DIRS[league]

        for raw_path in list(files.keys()):
            file_date = prediction_file_date(raw_path)
            if file_date is None or file_date >= today:
                continue

            cleaned_path = cleaned_root / raw_path.name
            if not cleaned_path.exists():
                del files[raw_path]
                skipped_missing[league] += 1
                log_league(
                    league,
                    f"WARN | Historical raw prediction has no existing cleaned copy; "
                    f"not applying current bias retroactively: {raw_path}"
                )
                continue

            fieldnames, rows = read_csv(cleaned_path)
            files[raw_path] = [fieldnames, rows]
            preserved[league] += 1
            log_league(
                league,
                f"PRESERVED HISTORICAL CLEANED PREDICTION | {cleaned_path}"
            )

    return preserved, skipped_missing


# =========================
# STEP 6B: APPLY PREDICTION BIASES
# =========================

def apply_prediction_biases(pred_files, bias_values):
    stats = {
        league: {
            "files_with_biased_rows": 0,
            "rows_adjusted": 0,
            "rows_skipped_already_flagged": 0,
        }
        for league in pred_files
    }

    required = {
        "home_projected_points",
        "away_projected_points",
        "total_projected_points",
    }

    for league, files in pred_files.items():
        margin_info = bias_values[league]["margin"]
        total_info = bias_values[league]["total"]
        margin_bias = margin_info["value"]
        total_bias = total_info["value"]
        margin_half = margin_bias / 2.0
        total_half = total_bias / 2.0

        for path, data in files.items():
            fieldnames, rows = data

            if not required.issubset(set(fieldnames)):
                log_league(league, f"WARN | Missing prediction columns, skipped: {path}")
                continue

            fieldnames = list(fieldnames)
            for col in (BIAS_FLAG_COLUMN, MARGIN_BIAS_COLUMN, TOTAL_BIAS_COLUMN):
                if col not in fieldnames:
                    fieldnames.append(col)
            data[0] = fieldnames

            file_adjusted = 0
            file_skipped = 0

            for row in rows:
                if str(row.get(BIAS_FLAG_COLUMN, "")).strip() == BIAS_FLAG_VALUE:
                    if (
                        str(row.get(MARGIN_BIAS_COLUMN, "")).strip() == ""
                        and margin_info["method"] in ("fixed", "none")
                    ):
                        row[MARGIN_BIAS_COLUMN] = f"{margin_bias:.3f}"

                    if (
                        str(row.get(TOTAL_BIAS_COLUMN, "")).strip() == ""
                        and total_info["method"] in ("fixed", "none")
                    ):
                        row[TOTAL_BIAS_COLUMN] = f"{total_bias:.3f}"

                    file_skipped += 1
                    stats[league]["rows_skipped_already_flagged"] += 1
                    continue

                home = to_float(row.get("home_projected_points"))
                away = to_float(row.get("away_projected_points"))
                total = to_float(row.get("total_projected_points"))

                if home is None or away is None or total is None:
                    continue

                new_home = round(home - margin_half - total_half, 2)
                new_away = round(away + margin_half - total_half, 2)
                new_total = round(total - total_bias, 2)

                row["home_projected_points"] = f"{new_home:.2f}"
                row["away_projected_points"] = f"{new_away:.2f}"
                row["total_projected_points"] = f"{new_total:.2f}"
                row[BIAS_FLAG_COLUMN] = BIAS_FLAG_VALUE
                row[MARGIN_BIAS_COLUMN] = f"{margin_bias:.3f}"
                row[TOTAL_BIAS_COLUMN] = f"{total_bias:.3f}"

                file_adjusted += 1
                stats[league]["rows_adjusted"] += 1

            if file_adjusted:
                stats[league]["files_with_biased_rows"] += 1
                log_league(
                    league,
                    f"BIASED | {path} | league={league} "
                    f"margin_method={margin_info['method']} margin_bias={margin_bias} "
                    f"total_method={total_info['method']} total_bias={total_bias} "
                    f"adjusted={file_adjusted} skipped_already_flagged={file_skipped}"
                )

    return stats


# =========================
# STEP 7: WRITE CLEANED FILES
# =========================

def write_cleaned(loaded_files, cleaned_dirs, label):
    stats = {
        league: {
            "files_written": 0,
            "rows_written": 0,
        }
        for league in loaded_files
    }

    for league, files in loaded_files.items():
        cleaned_root = cleaned_dirs[league]

        for path, (fieldnames, rows) in files.items():
            new_path = cleaned_root / path.name
            write_csv(new_path, fieldnames, rows)

            stats[league]["files_written"] += 1
            stats[league]["rows_written"] += len(rows)

            log_league(
                league,
                f"WROTE | {label} | {new_path} | rows={len(rows)}"
            )

    return stats


# =========================
# SUMMARY
# =========================

def sum_nested(stats, league, key, default=0):
    return stats.get(league, {}).get(key, default)


def summary_counts(
    league,
    bad_odds_blanked,
    bad_lines_blanked,
    outlier_counts,
):
    odds = bad_odds_blanked.get(
        league,
        {},
    )
    lines = bad_lines_blanked.get(
        league,
        {},
    )
    outliers = outlier_counts.get(
        league,
        {},
    )

    bad_ml = odds.get(
        "ML",
        0,
    )
    bad_total = odds.get(
        "TOTAL",
        0,
    )
    bad_spread = odds.get(
        "SPREAD",
        0,
    )
    bad_total_lines = lines.get(
        "TOTAL",
        0,
    )
    bad_spread_lines = lines.get(
        "SPREAD",
        0,
    )

    return (
        bad_ml,
        bad_total,
        bad_spread,
        bad_ml
        + bad_total
        + bad_spread,
        bad_total_lines,
        bad_spread_lines,
        bad_total_lines
        + bad_spread_lines,
        outliers.get(
            "spread",
            0,
        ),
        outliers.get(
            "total",
            0,
        ),
    )

def write_league_summaries(
    pred_files,
    book_files,
    bad_odds_blanked,
    bad_lines_blanked,
    outlier_counts,
    sportsbook_outlier_market_blanked,
    bias_stats,
    pred_write_stats,
    book_write_stats,
    preserved_historical,
    skipped_historical_missing,
):
    for league in ("NBA", "NCAAM", "WNBA"):
        (
            bad_ml,
            bad_total,
            bad_spread,
            bad_odds_total_all,
            bad_total_lines,
            bad_spread_lines,
            bad_lines_total_all,
            spread_outliers,
            total_outliers,
        ) = summary_counts(
            league,
            bad_odds_blanked,
            bad_lines_blanked,
            outlier_counts,
        )

        log_league(league, "")
        log_league(league, "============================================================")
        log_league(league, "SUMMARY")
        log_league(league, "============================================================")
        log_league(league, f"prediction_files_loaded              : {len(pred_files.get(league, {}))}")
        log_league(league, f"sportsbook_files_loaded              : {len(book_files.get(league, {}))}")
        log_league(league, f"bad_ml_market_rows_blanked           : {bad_ml}")
        log_league(league, f"bad_total_market_rows_blanked        : {bad_total}")
        log_league(league, f"bad_spread_market_rows_blanked       : {bad_spread}")
        log_league(league, f"bad_odds_market_rows_blanked_total   : {bad_odds_total_all}")
        log_league(league, f"bad_total_line_rows_blanked          : {bad_total_lines}")
        log_league(league, f"bad_spread_line_rows_blanked         : {bad_spread_lines}")
        log_league(league, f"bad_line_rows_blanked_total          : {bad_lines_total_all}")
        log_league(league, f"spread_outlier_events_found          : {spread_outliers}")
        log_league(league, f"total_outlier_events_found           : {total_outliers}")
        log_league(league, f"sportsbook_outlier_market_rows_blanked: {sportsbook_outlier_market_blanked.get(league, 0)}")
        log_league(league, f"historical_cleaned_files_preserved   : {preserved_historical.get(league, 0)}")
        log_league(league, f"historical_raw_missing_cleaned_skipped: {skipped_historical_missing.get(league, 0)}")
        log_league(league, "prediction_rows_removed              : 0")
        log_league(league, "sportsbook_rows_removed              : 0")
        log_league(league, f"bias_files_with_adjusted_rows        : {sum_nested(bias_stats, league, 'files_with_biased_rows')}")
        log_league(league, f"bias_rows_adjusted                   : {sum_nested(bias_stats, league, 'rows_adjusted')}")
        log_league(league, f"bias_rows_skipped_already_flagged    : {sum_nested(bias_stats, league, 'rows_skipped_already_flagged')}")
        log_league(league, f"prediction_files_written             : {sum_nested(pred_write_stats, league, 'files_written')}")
        log_league(league, f"prediction_rows_written              : {sum_nested(pred_write_stats, league, 'rows_written')}")
        log_league(league, f"sportsbook_files_written             : {sum_nested(book_write_stats, league, 'files_written')}")
        log_league(league, f"sportsbook_rows_written              : {sum_nested(book_write_stats, league, 'rows_written')}")
        log_league(league, "STATUS: SUCCESS")
        log_league(league, "============================================================")


def write_master_summary(
    pred_files,
    book_files,
    bad_odds_blanked,
    bad_lines_blanked,
    outlier_counts,
    sportsbook_outlier_market_blanked,
    bias_stats,
    pred_write_stats,
    book_write_stats,
    preserved_historical,
    skipped_historical_missing,
):
    log_master("")
    log_master("============================================================")
    log_master("SUMMARY")
    log_master("============================================================")

    for league in ("NBA", "NCAAM", "WNBA"):
        (
            bad_ml,
            bad_total,
            bad_spread,
            bad_odds_total_all,
            bad_total_lines,
            bad_spread_lines,
            bad_lines_total_all,
            spread_outliers,
            total_outliers,
        ) = summary_counts(
            league,
            bad_odds_blanked,
            bad_lines_blanked,
            outlier_counts,
        )

        log_master("")
        log_master(f"--- {league} ---")
        log_master(f"prediction_files_loaded              : {len(pred_files.get(league, {}))}")
        log_master(f"sportsbook_files_loaded              : {len(book_files.get(league, {}))}")
        log_master(f"bad_ml_market_rows_blanked           : {bad_ml}")
        log_master(f"bad_total_market_rows_blanked        : {bad_total}")
        log_master(f"bad_spread_market_rows_blanked       : {bad_spread}")
        log_master(f"bad_odds_market_rows_blanked_total   : {bad_odds_total_all}")
        log_master(f"bad_total_line_rows_blanked          : {bad_total_lines}")
        log_master(f"bad_spread_line_rows_blanked         : {bad_spread_lines}")
        log_master(f"bad_line_rows_blanked_total          : {bad_lines_total_all}")
        log_master(f"spread_outlier_events_found          : {spread_outliers}")
        log_master(f"total_outlier_events_found           : {total_outliers}")
        log_master(f"sportsbook_outlier_market_rows_blanked: {sportsbook_outlier_market_blanked.get(league, 0)}")
        log_master(f"historical_cleaned_files_preserved   : {preserved_historical.get(league, 0)}")
        log_master(f"historical_raw_missing_cleaned_skipped: {skipped_historical_missing.get(league, 0)}")
        log_master("prediction_rows_removed              : 0")
        log_master("sportsbook_rows_removed              : 0")
        log_master(f"bias_files_with_adjusted_rows        : {sum_nested(bias_stats, league, 'files_with_biased_rows')}")
        log_master(f"bias_rows_adjusted                   : {sum_nested(bias_stats, league, 'rows_adjusted')}")
        log_master(f"prediction_files_written             : {sum_nested(pred_write_stats, league, 'files_written')}")
        log_master(f"prediction_rows_written              : {sum_nested(pred_write_stats, league, 'rows_written')}")
        log_master(f"sportsbook_files_written             : {sum_nested(book_write_stats, league, 'files_written')}")
        log_master(f"sportsbook_rows_written              : {sum_nested(book_write_stats, league, 'rows_written')}")
        log_master(f"league_detail_log                    : {LEAGUE_LOG_FILES[league]}")

    log_master("")
    log_master("STATUS: SUCCESS")
    log_master("============================================================")


# =========================
# MAIN
# =========================

def main():
    init_logs()

    log_master("INFO | Starting basketball input cleanup")
    log_master(f"INFO | Model config: {CONFIG_PATH}")
    log_master(f"INFO | Dynamic bias state: {BIAS_STATE_PATH}")
    log_master("INFO | Odds cleanup active: blanks only the affected market; does not drop entire games")
    log_master("INFO | Missing market odds are treated as unavailable market, not bad game")
    log_master("INFO | Hold-based odds filtering removed")
    log_master("INFO | Spread outlier fix active: sportsbook home_spread is converted to model-margin convention using -home_spread")
    log_master("INFO | Outlier cleanup active: blanks only affected sportsbook market fields; prediction rows are retained")
    log_master("INFO | Historical cleaned predictions are preserved; current dynamic bias is only applied to current/future raw predictions")
    log_master(f"INFO | League logs: {LEAGUE_LOG_FILES}")

    for league in ("NBA", "NCAAM", "WNBA"):
        log_league(league, "INFO | Starting league cleanup")
        log_league(league, "INFO | Odds cleanup active: blanks only the affected market; does not drop entire games")
        log_league(league, "INFO | Missing market odds are treated as unavailable market, not bad game")
        log_league(league, "INFO | Hold-based odds filtering removed")
        log_league(league, "INFO | Spread outlier fix active: compare model home-away margin to -book_home_spread")
        log_league(league, "INFO | Outlier cleanup active: blanks only affected sportsbook market fields; prediction rows are retained")

    production_source = (
        production_prediction_source()
    )

    prediction_dirs = (
        prediction_dirs_for_source(
            production_source
        )
    )

    log_master(
        "INFO | Production prediction source: "
        f"{production_source}"
    )

    bias_values = resolve_bias_values()
    for league in ("NBA", "NCAAM", "WNBA"):
        log_league(
            league,
            f"BIAS CONFIG | margin_method={bias_values[league]['margin']['method']} "
            f"margin_value={bias_values[league]['margin']['value']} "
            f"total_method={bias_values[league]['total']['method']} "
            f"total_value={bias_values[league]['total']['value']}"
        )

    pred_files = load_all(prediction_dirs)
    book_files = load_all(SPORTSBOOK_DIRS)

    pred_loaded = sum(len(v) for v in pred_files.values())
    book_loaded = sum(len(v) for v in book_files.values())
    log_master(f"LOADED | predictions_files={pred_loaded} sportsbook_files={book_loaded}")

    bad_odds_blanked = blank_bad_market_odds(book_files)
    bad_lines_blanked = blank_bad_market_lines(book_files)

    pred_index = build_pred_index(pred_files)
    book_index = build_book_index(book_files)

    market_outlier_actions, outlier_counts = find_market_outlier_actions(pred_index, book_index)
    sportsbook_outlier_market_blanked = apply_market_outlier_actions(book_files, market_outlier_actions)

    preserved_historical, skipped_historical_missing = preserve_past_cleaned_predictions(pred_files)
    bias_stats = apply_prediction_biases(pred_files, bias_values)

    pred_write_stats = write_cleaned(
        pred_files,
        CLEANED_PREDICTION_DIRS,
        "PREDICTIONS",
    )

    book_write_stats = write_cleaned(
        book_files,
        CLEANED_SPORTSBOOK_DIRS,
        "SPORTSBOOK",
    )

    write_league_summaries(
        pred_files=pred_files,
        book_files=book_files,
        bad_odds_blanked=bad_odds_blanked,
        bad_lines_blanked=bad_lines_blanked,
        outlier_counts=outlier_counts,
        sportsbook_outlier_market_blanked=sportsbook_outlier_market_blanked,
        bias_stats=bias_stats,
        pred_write_stats=pred_write_stats,
        book_write_stats=book_write_stats,
        preserved_historical=preserved_historical,
        skipped_historical_missing=skipped_historical_missing,
    )

    write_master_summary(
        pred_files=pred_files,
        book_files=book_files,
        bad_odds_blanked=bad_odds_blanked,
        bad_lines_blanked=bad_lines_blanked,
        outlier_counts=outlier_counts,
        sportsbook_outlier_market_blanked=sportsbook_outlier_market_blanked,
        bias_stats=bias_stats,
        pred_write_stats=pred_write_stats,
        book_write_stats=book_write_stats,
        preserved_historical=preserved_historical,
        skipped_historical_missing=skipped_historical_missing,
    )

    print("STATUS: SUCCESS")
    print(f"log_file_master                  : {MASTER_LOG_FILE}")
    print(f"log_file_nba                     : {LEAGUE_LOG_FILES['NBA']}")
    print(f"log_file_ncaam                   : {LEAGUE_LOG_FILES['NCAAM']}")
    print(f"log_file_wnba                    : {LEAGUE_LOG_FILES['WNBA']}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            log_master("STATUS: FAILED")
            log_master(traceback.format_exc())

            for error_league in ("NBA", "NCAAM", "WNBA"):
                log_league(error_league, "STATUS: FAILED")
                log_league(error_league, traceback.format_exc())
        except OSError:
            pass

        print("STATUS: FAILED")
        print(f"See log: {MASTER_LOG_FILE}")
        raise
