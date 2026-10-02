#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import math
import shutil
import sys
import tempfile
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# ITEM 19 shared staking/uncertainty rules
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from typing import Any

import pandas as pd
import yaml
from scipy.stats import norm

from staking_runtime import (
    add_uncertainty_adjusted_ev, attach_candidate_uncertainty,
    requested_stake,
)

LEAGUES = ["nba", "ncaam", "wnba"]
MARKETS = ["moneyline", "spread", "total"]

TIEBREAK_COL_MAP = {
    "ev": "bet_ev",
    "kelly": "bet_kelly",
    "edge_vs_market": "bet_edge_vs_market",
}

BASKETBALL_ROOT = Path("docs/win/basketball")
DEFAULT_BACKTEST_DIR = BASKETBALL_ROOT / "backtest"
DEFAULT_MODEL_CONFIG = BASKETBALL_ROOT / "config" / "model_config.yaml"
DEFAULT_MARKETS_CONFIG = BASKETBALL_ROOT / "config" / "markets.yaml"
DEFAULT_STAKING_CONFIG = BASKETBALL_ROOT / "config" / "staking.yaml"

MODEL_SOURCES = ("dratings", "sdv", "ensemble")

PRODUCTION_BUILD_JUICE = (
    BASKETBALL_ROOT
    / "scripts"
    / "01_merge"
    / "build_juice_files_core.py"
)

PRODUCTION_EV_KELLY = (
    BASKETBALL_ROOT
    / "scripts"
    / "03_edges"
    / "compute_ev_kelly_core.py"
)

PRODUCTION_SELECT = (
    BASKETBALL_ROOT
    / "scripts"
    / "04_select"
    / "basketball_select_bets_core.py"
)

LEGACY_HISTORICAL_BIAS = {
    ("nba", 2025): {
        "margin": 0.4,
        "total": 0.4,
    },
    ("ncaam", 2025): {
        "margin": 0.6,
        "total": 1.2,
    },
    ("wnba", 2025): {
        "margin": 0.5,
        "total": 0.0,
    },
}

RESULT_COLUMNS = [
    "home_score",
    "away_score",
    "actual_total",
    "actual_home_spread",
    "actual_away_spread",
]

REQUIRED_INPUT_COLUMNS = {
    "game_date",
    "game_id",
    "home_team",
    "away_team",
    "home_spread",
    "away_spread",
    "total",
    "home_dk_moneyline_american",
    "away_dk_moneyline_american",
    "home_dk_spread_american",
    "away_dk_spread_american",
    "dk_total_over_american",
    "dk_total_under_american",
    "home_dk_moneyline_decimal",
    "away_dk_moneyline_decimal",
    "home_dk_spread_decimal",
    "away_dk_spread_decimal",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
    "home_prob",
    "away_prob",
    "home_projected_points",
    "away_projected_points",
    "total_projected_points",
    "home_score",
    "away_score",
}

DEBUG_COUNTS: Counter = Counter()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def sanitize_run_name(value: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in "-_." else "_"
        for ch in value.strip()
    ).strip("._")

    if not cleaned:
        raise ValueError("run name becomes empty after sanitization")

    return cleaned


def ensure_mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")

    return value


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be numeric; got {value!r}"
        ) from exc

    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")

    return number


def fv(value: Any) -> float | None:
    try:
        if (
            value is None
            or pd.isna(value)
            or str(value).strip() == ""
        ):
            return None

        number = float(value)

        if not math.isfinite(number):
            return None

        return number

    except (TypeError, ValueError, OverflowError):
        return None


def sha256_file(path: Path) -> str:
    safe_path = path.resolve(strict=True)
    if (
        not safe_path.is_file()
        or safe_path.suffix.lower() not in {".csv", ".yaml", ".yml"}
    ):
        raise ValueError(f"Unsupported file path for hashing: {path}")

    h = hashlib.sha256()

    with safe_path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing YAML file: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f"YAML root must be a mapping: {path}"
        )

    return data


def atomic_write_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
    )

    tmp.replace(path)


def clear_directory_contents(
    path: Path,
) -> None:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink(
                missing_ok=True
            )


def copy_tree_contents(
    src: Path,
    dst: Path,
) -> None:
    if not src.exists():
        return

    dst.mkdir(
        parents=True,
        exist_ok=True,
    )

    for child in src.iterdir():
        target = dst / child.name

        if child.is_dir():
            shutil.copytree(
                child,
                target,
                dirs_exist_ok=True,
            )
        else:
            shutil.copy2(
                child,
                target,
            )


class RunLogger:
    def __init__(self, path: Path):
        self.path = path

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            (
                "=== basketball_backtest RUN "
                f"{now_utc()} ===\n"
            ),
            encoding="utf-8",
        )

    def log(
        self,
        msg: str,
        level: str = "INFO",
    ) -> None:
        line = (
            f"{now_utc()} | "
            f"{level:<5} | "
            f"{msg.rstrip()}"
        )

        print(
            line,
            flush=True,
        )

        with open(
            self.path,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(line + "\n")


def resolve_model_source(
    model_cfg: dict,
    requested: str | None,
) -> str:
    if requested is not None:
        source = str(
            requested
        ).strip().lower()
    else:
        source = str(
            model_cfg.get(
                "production_prediction_source",
                "",
            )
        ).strip().lower()

    if source not in MODEL_SOURCES:
        raise ValueError(
            "model_source must be one of "
            "dratings, sdv, ensemble; "
            f"got {source!r}"
        )

    return source


def apply_model_source(
    df: pd.DataFrame,
    model_source: str,
    path: Path,
) -> pd.DataFrame:
    out = df.copy()

    if "model_source" in out.columns:
        seen = {
            str(v).strip().lower()
            for v in out[
                "model_source"
            ].dropna().tolist()
            if str(v).strip()
        }

        if seen and seen != {model_source}:
            raise ValueError(
                f"{path.name} model_source "
                f"values {sorted(seen)} "
                "do not match requested "
                f"model_source={model_source}"
            )

    out["model_source"] = model_source

    return out


def season_from_input_filename(
    path: Path,
    league: str,
) -> int:
    stem = path.stem.strip()
    expected_suffix = "_" + league.upper()

    if not stem.upper().endswith(
        expected_suffix
    ):
        raise ValueError(
            f"{path.name} does not match "
            f"expected historical filename "
            f"<season>_{league.upper()}.csv"
        )

    season_text = stem[
        : -len(expected_suffix)
    ]

    if (
        len(season_text) != 4
        or not season_text.isdigit()
    ):
        raise ValueError(
            f"{path.name} does not contain "
            "a valid four-digit internal "
            "season in its filename"
        )

    return int(season_text)


def normalize_production_bias_rule(
    model_cfg: dict,
    league: str,
    component: str,
) -> dict:
    league_cfg = ensure_mapping(
        ensure_mapping(
            model_cfg.get("leagues"),
            "model_config.leagues",
        ).get(league),
        f"model_config.leagues.{league}",
    )

    bias_cfg = ensure_mapping(
        league_cfg.get("bias") or {},
        f"{league}.bias",
    )

    rule = ensure_mapping(
        bias_cfg.get(component),
        f"{league}.bias.{component}",
    )

    method = str(
        rule.get(
            "method",
            "",
        )
    ).strip().lower()

    if method not in {
        "rolling",
        "regime_aware",
        "fixed",
        "none",
    }:
        raise ValueError(
            f"Unsupported {league} "
            f"{component} bias "
            f"method={method!r}"
        )

    normalized = {
        "method": method,
    }

    if method == "fixed":
        normalized["value"] = require_number(
            rule.get("value"),
            f"{league}.bias.{component}.value",
        )

    elif method == "rolling":
        window = int(
            require_number(
                rule.get("window_games"),
                (
                    f"{league}.bias."
                    f"{component}.window_games"
                ),
            )
        )

        if window <= 0:
            raise ValueError(
                f"{league}.bias."
                f"{component}.window_games "
                "must be > 0"
            )

        normalized[
            "window_games"
        ] = window

    elif method == "regime_aware":
        raw_windows = rule.get(
            "windows_games"
        )

        raw_weights = rule.get(
            "weights"
        )

        if (
            not isinstance(
                raw_windows,
                list,
            )
            or not raw_windows
        ):
            raise ValueError(
                f"{league}.bias."
                f"{component}."
                "windows_games must be "
                "a non-empty list"
            )

        if (
            not isinstance(
                raw_weights,
                list,
            )
            or len(raw_weights)
            != len(raw_windows)
        ):
            raise ValueError(
                f"{league}.bias."
                f"{component}.weights "
                "must match windows_games"
            )

        windows = [
            int(
                require_number(
                    v,
                    (
                        f"{league}.bias."
                        f"{component}."
                        "windows_games"
                    ),
                )
            )
            for v in raw_windows
        ]

        weights = [
            require_number(
                v,
                (
                    f"{league}.bias."
                    f"{component}.weights"
                ),
            )
            for v in raw_weights
        ]

        if any(
            w <= 0
            for w in windows
        ):
            raise ValueError(
                f"{league}.bias."
                f"{component}."
                "windows_games must "
                "all be > 0"
            )

        if (
            any(
                w < 0
                for w in weights
            )
            or sum(weights) <= 0
        ):
            raise ValueError(
                f"{league}.bias."
                f"{component}.weights "
                "must be >= 0 and "
                "sum to > 0"
            )

        total_weight = sum(weights)

        weights = [
            w / total_weight
            for w in weights
        ]

        shrink = require_number(
            rule.get(
                "sign_conflict_shrink"
            ),
            (
                f"{league}.bias."
                f"{component}."
                "sign_conflict_shrink"
            ),
        )

        if not 0 <= shrink <= 1:
            raise ValueError(
                f"{league}.bias."
                f"{component}."
                "sign_conflict_shrink "
                "must be between 0 and 1"
            )

        normalized.update({
            "windows_games": windows,
            "weights": weights,
            "sign_conflict_shrink": shrink,
        })

    return normalized


def production_bias_from_errors(
    errors: list[float],
    rule: dict,
) -> float | None:
    method = rule["method"]

    if method == "none":
        return 0.0

    if method == "fixed":
        return round(
            float(rule["value"]),
            3,
        )

    if method == "rolling":
        window = int(
            rule["window_games"]
        )

        if len(errors) < window:
            return None

        return round(
            float(
                sum(
                    errors[-window:]
                )
                / window
            ),
            3,
        )

    if method == "regime_aware":
        windows = [
            int(v)
            for v in rule[
                "windows_games"
            ]
        ]

        if len(errors) < max(windows):
            return None

        means = {
            window: float(
                sum(
                    errors[-window:]
                )
                / window
            )
            for window in windows
        }

        weighted = sum(
            float(weight)
            * means[int(window)]
            for window, weight
            in zip(
                windows,
                rule["weights"],
            )
        )

        positive_present = any(
            value > 1e-12
            for value
            in means.values()
        )

        negative_present = any(
            value < -1e-12
            for value
            in means.values()
        )

        if (
            positive_present
            and negative_present
        ):
            weighted *= float(
                rule[
                    "sign_conflict_shrink"
                ]
            )

        return round(
            float(weighted),
            3,
        )

    raise ValueError(
        "Unsupported production "
        f"bias method={method!r}"
    )


def reverse_stored_bias_to_raw(
    row: pd.Series,
    league: str,
    internal_season: int,
) -> tuple[
    float,
    float,
    float,
]:
    home = fv(
        row.get(
            "home_projected_points"
        )
    )

    away = fv(
        row.get(
            "away_projected_points"
        )
    )

    total = fv(
        row.get(
            "total_projected_points"
        )
    )

    if (
        home is None
        or away is None
        or total is None
    ):
        raise ValueError(
            f"game_id={row.get('game_id')} "
            "has invalid projected points"
        )

    bias_flag = fv(
        row.get("bias_applied")
    )

    if (
        bias_flag is None
        or float(bias_flag) == 0.0
    ):
        return (
            home,
            away,
            total,
        )

    if float(bias_flag) != 1.0:
        raise ValueError(
            f"game_id={row.get('game_id')} "
            "has invalid bias_applied="
            f"{row.get('bias_applied')!r}"
        )

    margin_bias = fv(
        row.get("margin_bias")
    )

    total_bias = fv(
        row.get("total_bias")
    )

    if (
        margin_bias is None
        or total_bias is None
    ):
        legacy = LEGACY_HISTORICAL_BIAS.get(
            (
                league,
                internal_season,
            )
        )

        if legacy is None:
            raise ValueError(
                f"game_id={row.get('game_id')} "
                f"in internal season "
                f"{internal_season} has "
                "bias_applied=1 but no "
                "per-game margin_bias/"
                "total_bias and no exact "
                "legacy fallback for "
                f"{league.upper()} "
                f"season {internal_season}"
            )

        margin_bias = float(
            legacy["margin"]
        )

        total_bias = float(
            legacy["total"]
        )

    raw_home = (
        home
        + margin_bias / 2.0
        + total_bias / 2.0
    )

    raw_away = (
        away
        - margin_bias / 2.0
        + total_bias / 2.0
    )

    raw_total = (
        total
        + total_bias
    )

    return (
        raw_home,
        raw_away,
        raw_total,
    )


def apply_point_in_time_production_bias(
    df: pd.DataFrame,
    league: str,
    internal_season: int,
    model_cfg: dict,
    history_state: (
        dict[
            str,
            list[float],
        ]
        | None
    ) = None,
) -> pd.DataFrame:
    out = df.copy()

    out["_sort_date"] = pd.to_datetime(
        out[
            "game_date"
        ].astype(str).str.replace(
            "_",
            "-",
            regex=False,
        ),
        errors="coerce",
    )

    if out["_sort_date"].isna().any():
        raise ValueError(
            "Historical input contains "
            "invalid game_date values"
        )

    out = (
        out.sort_values(
            [
                "_sort_date",
                "game_id",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    margin_rule = (
        normalize_production_bias_rule(
            model_cfg,
            league,
            "margin",
        )
    )

    total_rule = (
        normalize_production_bias_rule(
            model_cfg,
            league,
            "total",
        )
    )

    if history_state is None:
        history_state = {
            "margin_errors": [],
            "total_errors": [],
        }

    margin_errors = history_state.setdefault(
        "margin_errors",
        [],
    )

    total_errors = history_state.setdefault(
        "total_errors",
        [],
    )

    input_valid = []
    ready = []
    applied_margin = []
    applied_total = []
    adjusted_home = []
    adjusted_away = []
    adjusted_total = []

    for _, row in out.iterrows():
        stored_home = fv(
            row.get(
                "home_projected_points"
            )
        )

        stored_away = fv(
            row.get(
                "away_projected_points"
            )
        )

        stored_total = fv(
            row.get(
                "total_projected_points"
            )
        )

        home_score = fv(
            row.get(
                "home_score"
            )
        )

        away_score = fv(
            row.get(
                "away_score"
            )
        )

        complete = (
            stored_home is not None
            and stored_away is not None
            and stored_total is not None
            and home_score is not None
            and away_score is not None
        )

        if not complete:
            input_valid.append(False)
            ready.append(False)
            applied_margin.append(None)
            applied_total.append(None)
            adjusted_home.append(float("nan"))
            adjusted_away.append(float("nan"))
            adjusted_total.append(float("nan"))
            continue

        input_valid.append(True)

        (
            raw_home,
            raw_away,
            raw_total,
        ) = reverse_stored_bias_to_raw(
            row,
            league,
            internal_season,
        )

        margin_bias = (
            production_bias_from_errors(
                margin_errors,
                margin_rule,
            )
        )

        total_bias = (
            production_bias_from_errors(
                total_errors,
                total_rule,
            )
        )

        is_ready = (
            margin_bias is not None
            and total_bias is not None
        )

        ready.append(is_ready)
        applied_margin.append(
            margin_bias
        )
        applied_total.append(
            total_bias
        )

        if is_ready:
            adjusted_home.append(
                raw_home
                - margin_bias / 2.0
                - total_bias / 2.0
            )

            adjusted_away.append(
                raw_away
                + margin_bias / 2.0
                - total_bias / 2.0
            )

            adjusted_total.append(
                raw_total
                - total_bias
            )
        else:
            adjusted_home.append(
                float("nan")
            )
            adjusted_away.append(
                float("nan")
            )
            adjusted_total.append(
                float("nan")
            )

        margin_errors.append(
            raw_home
            - raw_away
            - (
                home_score
                - away_score
            )
        )

        total_errors.append(
            raw_total
            - (
                home_score
                + away_score
            )
        )

    out[
        "home_projected_points"
    ] = adjusted_home

    out[
        "away_projected_points"
    ] = adjusted_away

    out[
        "total_projected_points"
    ] = adjusted_total

    out["margin_bias"] = applied_margin
    out["total_bias"] = applied_total

    out["bias_applied"] = [
        1 if value else 0
        for value in ready
    ]

    out[
        "_production_input_valid"
    ] = input_valid

    out[
        "_production_bias_ready"
    ] = ready

    out[
        "internal_season"
    ] = internal_season

    return out.drop(
        columns=[
            "_sort_date"
        ]
    )


def apply_production_selection_policy(
    test_cfg: dict,
    production_cfg: dict,
) -> dict:
    out = copy.deepcopy(test_cfg)

    out.setdefault(
        "markets",
        {},
    )

    prod_markets = ensure_mapping(
        production_cfg.get("markets"),
        "markets.yaml markets",
    )

    out["ml_vs_spread_tiebreak"] = str(
        production_cfg.get(
            "ml_vs_spread_tiebreak",
            "ev",
        )
    ).strip().lower()

    if out["ml_vs_spread_tiebreak"] not in TIEBREAK_COL_MAP:
        out["ml_vs_spread_tiebreak"] = "ev"

    for league in LEAGUES:
        out["markets"].setdefault(
            league,
            {},
        )

        prod_league = ensure_mapping(
            prod_markets.get(league),
            f"markets.{league}",
        )

        for market in MARKETS:
            out[
                "markets"
            ][
                league
            ].setdefault(
                market,
                {},
            )

            prod_market = ensure_mapping(
                prod_league.get(market),
                (
                    f"markets.{league}."
                    f"{market}"
                ),
            )

            out[
                "markets"
            ][
                league
            ][
                market
            ][
                "selection_mode"
            ] = prod_market.get(
                "selection_mode",
                "pick_one",
            )

            out[
                "markets"
            ][
                league
            ][
                market
            ][
                "pick_preference"
            ] = copy.deepcopy(
                prod_market.get(
                    "pick_preference"
                )
                or {
                    "metric": "ev",
                    "direction": "max",
                }
            )

    return out


def load_module_from_path(
    name: str,
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            "Missing production parity "
            f"module: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Unable to load production "
            f"parity module: {path}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(module)

    return module


def calibration_cfg(
    league_cfg: dict,
    market: str,
    side: str,
) -> dict:
    cfg = (
        (
            (
                league_cfg.get(
                    "calibration"
                )
                or {}
            )
            .get(market)
            or {}
        )
        .get(side)
        or {
            "method": "none"
        }
    )

    if isinstance(cfg, str):
        cfg = {
            "method": cfg
        }

    if not isinstance(cfg, dict):
        raise ValueError(
            f"calibration.{market}."
            f"{side} must be a mapping"
        )

    return cfg


def complementary_calibration_cfg(
    league_cfg: dict,
    market: str,
    first_side: str,
    second_side: str,
) -> dict:
    market_cfg = (
        (
            league_cfg.get(
                "calibration"
            )
            or {}
        )
        .get(market)
        or {}
    )

    if not isinstance(
        market_cfg,
        dict,
    ):
        raise ValueError(
            f"calibration.{market} "
            "must be a mapping"
        )

    canonical_side = str(
        market_cfg.get(
            "canonical_side",
            first_side,
        )
    ).strip().lower()

    if canonical_side not in {
        first_side,
        second_side,
    }:
        raise ValueError(
            f"calibration.{market}."
            "canonical_side must be "
            f"{first_side!r} or "
            f"{second_side!r}"
        )

    cfg = (
        market_cfg.get(
            canonical_side
        )
        or {
            "method": "none"
        }
    )

    if isinstance(cfg, str):
        cfg = {
            "method": cfg
        }

    if not isinstance(cfg, dict):
        raise ValueError(
            f"calibration.{market}."
            f"{canonical_side} "
            "must be a mapping"
        )

    opposite_side = (
        second_side
        if canonical_side
        == first_side
        else first_side
    )

    opposite_cfg = (
        market_cfg.get(
            opposite_side
        )
    )

    if (
        opposite_cfg not in (
            None,
            {},
            "none",
            "raw",
        )
        and isinstance(
            opposite_cfg,
            dict,
        )
    ):
        opposite_method = str(
            opposite_cfg.get(
                "method",
                "none",
            )
        ).strip().lower()

        if opposite_method not in {
            "none",
            "raw",
            "",
        }:
            raise ValueError(
                f"calibration.{market}."
                f"{opposite_side} must "
                "not define an independent "
                "calibration when "
                "complementary calibration "
                "is enabled"
            )

    return {
        "canonical_side": canonical_side,
        "config": cfg,
    }


def build_league_settings(
    model_cfg: dict,
) -> dict:
    leagues_cfg = ensure_mapping(
        model_cfg.get("leagues"),
        "model_config.leagues",
    )

    settings = {}

    for league in LEAGUES:
        league_cfg = ensure_mapping(
            leagues_cfg.get(league),
            (
                "model_config.leagues."
                f"{league}"
            ),
        )

        if str(
            league_cfg.get(
                "status",
                "",
            )
        ).strip().lower() != "active":
            raise ValueError(
                f"League {league.upper()} "
                "is not active in "
                "model_config.yaml"
            )

        edge_cfg = ensure_mapping(
            league_cfg.get("edge") or {},
            f"{league}.edge",
        )

        std_cfg = ensure_mapping(
            league_cfg.get("std") or {},
            f"{league}.std",
        )

        spread_std_cfg = ensure_mapping(
            std_cfg.get("spread") or {},
            f"{league}.std.spread",
        )

        total_std_cfg = ensure_mapping(
            std_cfg.get("total") or {},
            f"{league}.std.total",
        )

        if str(
            spread_std_cfg.get(
                "mode",
                "",
            )
        ).strip().lower() != "fixed":
            raise ValueError(
                f"{league.upper()} "
                "spread STD mode "
                "must be fixed"
            )

        if str(
            total_std_cfg.get(
                "mode",
                "",
            )
        ).strip().lower() != "fixed":
            raise ValueError(
                f"{league.upper()} "
                "total STD mode "
                "must be fixed"
            )

        settings[league] = {
            "ML_EDGE": require_number(
                edge_cfg.get(
                    "moneyline"
                ),
                (
                    f"{league}.edge."
                    "moneyline"
                ),
            ),
            "SPREAD_EDGE": require_number(
                edge_cfg.get(
                    "spread"
                ),
                (
                    f"{league}.edge."
                    "spread"
                ),
            ),
            "TOTAL_EDGE": require_number(
                edge_cfg.get(
                    "total"
                ),
                (
                    f"{league}.edge."
                    "total"
                ),
            ),
            "SPREAD_STD": require_number(
                spread_std_cfg.get(
                    "value"
                ),
                (
                    f"{league}.std."
                    "spread.value"
                ),
            ),
            "TOTAL_STD": require_number(
                total_std_cfg.get(
                    "value"
                ),
                (
                    f"{league}.std."
                    "total.value"
                ),
            ),
            "CALIBRATION": {
                "moneyline": {
                    "home": calibration_cfg(
                        league_cfg,
                        "moneyline",
                        "home",
                    ),
                    "away": calibration_cfg(
                        league_cfg,
                        "moneyline",
                        "away",
                    ),
                },
                "spread": (
                    complementary_calibration_cfg(
                        league_cfg,
                        "spread",
                        "home",
                        "away",
                    )
                ),
                "total": (
                    complementary_calibration_cfg(
                        league_cfg,
                        "total",
                        "over",
                        "under",
                    )
                ),
            },
        }

    return settings


def apply_calibration(
    p: Any,
    cfg: dict,
) -> float | str:
    if (
        p is None
        or pd.isna(p)
    ):
        return ""

    try:
        p = float(p)
    except (
        TypeError,
        ValueError,
    ):
        return ""

    method = str(
        (
            cfg
            or {}
        ).get(
            "method",
            "none",
        )
    ).strip().lower()

    if method in {
        "none",
        "raw",
        "",
    }:
        return p

    if method == "beta":
        p = min(
            max(
                p,
                1e-12,
            ),
            1.0 - 1e-12,
        )

        z = (
            require_number(
                cfg.get("intercept"),
                "beta.intercept",
            )
            + require_number(
                cfg.get("coef_log_p"),
                "beta.coef_log_p",
            )
            * math.log(p)
            + require_number(
                cfg.get("coef_log_1mp"),
                "beta.coef_log_1mp",
            )
            * math.log(
                1.0 - p
            )
        )

        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (
                1.0 + ez
            )

        ez = math.exp(z)

        return ez / (
            1.0 + ez
        )

    raise ValueError(
        "Unsupported calibration "
        f"method: {method!r}"
    )


def apply_complementary_calibration(
    raw_first: Any,
    raw_second: Any,
    calibration: dict,
    first_side: str,
    second_side: str,
) -> tuple[
    float | str,
    float | str,
]:
    canonical_side = str(
        calibration[
            "canonical_side"
        ]
    ).strip().lower()

    raw_canonical = (
        raw_first
        if canonical_side == first_side
        else raw_second
    )

    calibrated = apply_calibration(
        raw_canonical,
        calibration["config"],
    )

    if (
        calibrated == ""
        or pd.isna(calibrated)
    ):
        return (
            "",
            "",
        )

    p_canonical = clamp_probability(
        float(calibrated)
    )

    p_opposite = (
        1.0 - p_canonical
    )

    if canonical_side == first_side:
        return (
            p_canonical,
            p_opposite,
        )

    if canonical_side == second_side:
        return (
            p_opposite,
            p_canonical,
        )

    raise ValueError(
        "Unsupported canonical side "
        f"{canonical_side!r}; expected "
        f"{first_side!r} or "
        f"{second_side!r}"
    )


def american_to_decimal(
    odds: Any,
) -> float | str:
    if (
        odds is None
        or pd.isna(odds)
        or str(odds).strip() == ""
    ):
        return ""

    try:
        a = float(odds)
    except (
        TypeError,
        ValueError,
    ):
        return ""

    if not bool(a):
        return ""

    return (
        1.0 + a / 100.0
        if a > 0
        else 1.0
        + 100.0 / abs(a)
    )


def american_to_decimal_or_none(
    odds: Any,
) -> float | None:
    value = american_to_decimal(
        odds
    )

    return (
        None
        if value == ""
        else float(value)
    )


def to_american(
    decimal_value: Any,
) -> str:
    if (
        decimal_value is None
        or decimal_value == ""
        or pd.isna(decimal_value)
    ):
        return ""

    try:
        dec = float(
            decimal_value
        )
    except (
        TypeError,
        ValueError,
    ):
        return ""

    if dec <= 1:
        return ""

    return (
        f"+{int((dec - 1) * 100)}"
        if dec >= 2
        else f"-{int(100 / (dec - 1))}"
    )


def clamp_probability(
    p: Any,
) -> float:
    return min(
        max(
            float(p),
            0.01,
        ),
        0.99,
    )


def safe_implied_prob(
    decimal_value: Any,
) -> float | str:
    if (
        decimal_value is None
        or decimal_value == ""
        or pd.isna(decimal_value)
    ):
        return ""

    try:
        d = float(
            decimal_value
        )
    except (
        TypeError,
        ValueError,
    ):
        return ""

    return (
        ""
        if d <= 0
        else 1.0 / d
    )


def devig_pair(
    p_a: Any,
    p_b: Any,
) -> tuple[
    float | str,
    float | str,
]:
    if (
        p_a == ""
        or p_b == ""
        or pd.isna(p_a)
        or pd.isna(p_b)
    ):
        return (
            "",
            "",
        )

    try:
        a = float(p_a)
        b = float(p_b)
    except (
        TypeError,
        ValueError,
    ):
        return (
            "",
            "",
        )

    total = a + b

    if total <= 0:
        return (
            "",
            "",
        )

    return (
        a / total,
        b / total,
    )


def validate_historical_input(
    df: pd.DataFrame,
    path: Path,
    expected_league: str,
) -> None:
    missing = sorted(
        REQUIRED_INPUT_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{path.name} missing "
            "required columns: "
            f"{missing}"
        )

    if df.empty:
        raise ValueError(
            f"{path.name} is empty"
        )

    blank_ids = (
        df["game_id"].isna()
        | (
            df[
                "game_id"
            ]
            .astype(str)
            .str.strip()
            == ""
        )
    )

    if blank_ids.any():
        raise ValueError(
            f"{path.name} contains "
            "blank game_id values"
        )

    if "league" in df.columns:
        seen = {
            str(x).strip().lower()
            for x
            in df[
                "league"
            ].dropna().unique()
            if str(x).strip()
        }

        if (
            seen
            and seen != {
                expected_league
            }
        ):
            raise ValueError(
                f"{path.name} league "
                f"values {sorted(seen)} "
                "do not match expected "
                f"{expected_league}"
            )


def split_features_and_scores(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    score_cols = [
        c
        for c in RESULT_COLUMNS
        if c in df.columns
    ]

    scores = (
        df[
            [
                "game_id",
                *score_cols,
            ]
        ]
        .copy()
        .drop_duplicates(
            subset=[
                "game_id"
            ],
            keep="last",
        )
    )

    features = df.drop(
        columns=score_cols,
        errors="ignore",
    ).copy()

    return (
        features,
        scores,
    )


def process_moneyline_juice(
    df: pd.DataFrame,
    settings: dict,
) -> pd.DataFrame:
    out = df.copy()

    edge = settings["ML_EDGE"]

    cal = settings[
        "CALIBRATION"
    ][
        "moneyline"
    ]

    out["away_decimal"] = out[
        "away_dk_moneyline_american"
    ].apply(
        american_to_decimal
    )

    out["home_decimal"] = out[
        "home_dk_moneyline_american"
    ].apply(
        american_to_decimal
    )

    out["away_implied_prob"] = out[
        "away_decimal"
    ].apply(
        safe_implied_prob
    )

    out["home_implied_prob"] = out[
        "home_decimal"
    ].apply(
        safe_implied_prob
    )

    pairs = out.apply(
        lambda r: devig_pair(
            r["away_implied_prob"],
            r["home_implied_prob"],
        ),
        axis=1,
    )

    out["away_market_prob"] = (
        pairs.apply(
            lambda x: x[0]
        )
    )

    out["home_market_prob"] = (
        pairs.apply(
            lambda x: x[1]
        )
    )

    out["home_model_prob"] = (
        pd.to_numeric(
            out["home_prob"],
            errors="coerce",
        ).apply(
            lambda p: apply_calibration(
                p,
                cal["home"],
            )
        )
    )

    out["away_model_prob"] = (
        pd.to_numeric(
            out["away_prob"],
            errors="coerce",
        ).apply(
            lambda p: apply_calibration(
                p,
                cal["away"],
            )
        )
    )

    out["away_fair"] = out[
        "away_model_prob"
    ].apply(
        lambda x: (
            1.0 / float(x)
            if (
                x != ""
                and pd.notna(x)
                and float(x) > 0
            )
            else ""
        )
    )

    out["home_fair"] = out[
        "home_model_prob"
    ].apply(
        lambda x: (
            1.0 / float(x)
            if (
                x != ""
                and pd.notna(x)
                and float(x) > 0
            )
            else ""
        )
    )

    out[
        "away_acceptable_decimal_moneyline"
    ] = out["away_fair"].apply(
        lambda x: (
            float(x) * (1.0 + edge)
            if x != ""
            else ""
        )
    )

    out[
        "home_acceptable_decimal_moneyline"
    ] = out["home_fair"].apply(
        lambda x: (
            float(x) * (1.0 + edge)
            if x != ""
            else ""
        )
    )

    out[
        "away_acceptable_american_moneyline"
    ] = out[
        "away_acceptable_decimal_moneyline"
    ].apply(to_american)

    out[
        "home_acceptable_american_moneyline"
    ] = out[
        "home_acceptable_decimal_moneyline"
    ].apply(to_american)

    return out


def process_total_juice(
    df: pd.DataFrame,
    settings: dict,
) -> pd.DataFrame:
    out = df.copy()

    edge = settings["TOTAL_EDGE"]
    std = settings["TOTAL_STD"]

    cal = settings[
        "CALIBRATION"
    ][
        "total"
    ]

    vals = {
        k: []
        for k in [
            "over_model_prob",
            "under_model_prob",
            "fair_over",
            "fair_under",
            "acceptable_over",
            "acceptable_under",
        ]
    }

    for _, row in out.iterrows():
        line = fv(
            row.get("total")
        )

        mean = fv(
            row.get(
                "total_projected_points"
            )
        )

        if (
            line is None
            or mean is None
        ):
            for key in vals:
                vals[key].append("")

            continue

        raw_under = clamp_probability(
            norm.cdf(
                (
                    line
                    - mean
                )
                / std
            )
        )

        raw_over = (
            1.0 - raw_under
        )

        (
            p_over,
            p_under,
        ) = apply_complementary_calibration(
            raw_over,
            raw_under,
            cal,
            "over",
            "under",
        )

        if (
            p_over == ""
            or p_under == ""
        ):
            for key in vals:
                vals[key].append("")

            continue

        p_over = float(p_over)
        p_under = float(p_under)

        if not math.isclose(
            p_over + p_under,
            1.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Total probabilities are "
                "not complementary: "
                f"over={p_over}, "
                f"under={p_under}"
            )

        fair_over = (
            1.0 / p_over
        )

        fair_under = (
            1.0 / p_under
        )

        vals[
            "over_model_prob"
        ].append(p_over)

        vals[
            "under_model_prob"
        ].append(p_under)

        vals[
            "fair_over"
        ].append(fair_over)

        vals[
            "fair_under"
        ].append(fair_under)

        vals[
            "acceptable_over"
        ].append(
            fair_over
            * (
                1.0 + edge
            )
        )

        vals[
            "acceptable_under"
        ].append(
            fair_under
            * (
                1.0 + edge
            )
        )

    for key, value in vals.items():
        out[key] = value

    out["over_implied_prob"] = out[
        "dk_total_over_decimal"
    ].apply(
        safe_implied_prob
    )

    out["under_implied_prob"] = out[
        "dk_total_under_decimal"
    ].apply(
        safe_implied_prob
    )

    pairs = out.apply(
        lambda r: devig_pair(
            r["over_implied_prob"],
            r["under_implied_prob"],
        ),
        axis=1,
    )

    out["over_market_prob"] = (
        pairs.apply(
            lambda x: x[0]
        )
    )

    out["under_market_prob"] = (
        pairs.apply(
            lambda x: x[1]
        )
    )

    return out


def process_spread_juice(
    df: pd.DataFrame,
    settings: dict,
) -> pd.DataFrame:
    out = df.copy()

    edge = settings["SPREAD_EDGE"]
    std = settings["SPREAD_STD"]

    cal = settings[
        "CALIBRATION"
    ][
        "spread"
    ]

    vals = {
        k: []
        for k in [
            "home_spread_model_prob",
            "away_spread_model_prob",
            "fair_home_spread_decimal",
            "fair_away_spread_decimal",
            "home_acceptable_spread_decimal",
            "away_acceptable_spread_decimal",
        ]
    }

    for _, row in out.iterrows():
        hp = fv(
            row.get(
                "home_projected_points"
            )
        )

        ap = fv(
            row.get(
                "away_projected_points"
            )
        )

        line = fv(
            row.get(
                "home_spread"
            )
        )

        if (
            hp is None
            or ap is None
            or line is None
        ):
            for key in vals:
                vals[key].append("")

            continue

        raw_home = clamp_probability(
            1.0
            - norm.cdf(
                -line,
                loc=(
                    hp - ap
                ),
                scale=std,
            )
        )

        raw_away = (
            1.0 - raw_home
        )

        (
            p_home,
            p_away,
        ) = apply_complementary_calibration(
            raw_home,
            raw_away,
            cal,
            "home",
            "away",
        )

        if (
            p_home == ""
            or p_away == ""
        ):
            for key in vals:
                vals[key].append("")

            continue

        p_home = float(p_home)
        p_away = float(p_away)

        if not math.isclose(
            p_home + p_away,
            1.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Spread probabilities are "
                "not complementary: "
                f"home={p_home}, "
                f"away={p_away}"
            )

        fair_home = (
            1.0 / p_home
        )

        fair_away = (
            1.0 / p_away
        )

        vals[
            "home_spread_model_prob"
        ].append(p_home)

        vals[
            "away_spread_model_prob"
        ].append(p_away)

        vals[
            "fair_home_spread_decimal"
        ].append(fair_home)

        vals[
            "fair_away_spread_decimal"
        ].append(fair_away)

        vals[
            "home_acceptable_spread_decimal"
        ].append(
            fair_home
            * (
                1.0 + edge
            )
        )

        vals[
            "away_acceptable_spread_decimal"
        ].append(
            fair_away
            * (
                1.0 + edge
            )
        )

    for key, value in vals.items():
        out[key] = value

    out[
        "home_acceptable_spread_american"
    ] = out[
        "home_acceptable_spread_decimal"
    ].apply(to_american)

    out[
        "away_acceptable_spread_american"
    ] = out[
        "away_acceptable_spread_decimal"
    ].apply(to_american)

    out[
        "home_spread_implied_prob"
    ] = out[
        "home_dk_spread_decimal"
    ].apply(
        safe_implied_prob
    )

    out[
        "away_spread_implied_prob"
    ] = out[
        "away_dk_spread_decimal"
    ].apply(
        safe_implied_prob
    )

    pairs = out.apply(
        lambda r: devig_pair(
            r[
                "home_spread_implied_prob"
            ],
            r[
                "away_spread_implied_prob"
            ],
        ),
        axis=1,
    )

    out[
        "home_spread_market_prob"
    ] = pairs.apply(
        lambda x: x[0]
    )

    out[
        "away_spread_market_prob"
    ] = pairs.apply(
        lambda x: x[1]
    )

    return out


def compute_ev(
    model_prob: Any,
    book_decimal: Any,
) -> float | None:
    p = fv(model_prob)
    d = fv(book_decimal)

    if (
        p is None
        or d is None
    ):
        return None

    return (
        p * d
        - 1.0
    )


def compute_kelly(
    model_prob: Any,
    book_decimal: Any,
) -> float | None:
    p = fv(model_prob)
    d = fv(book_decimal)

    if (
        p is None
        or d is None
        or d <= 1.0
    ):
        return None

    b = d - 1.0

    k = (
        (
            b * p
        )
        - (
            1.0 - p
        )
    ) / b

    if not math.isfinite(k):
        return None

    return max(
        k,
        0.0,
    )


def process_moneyline_ev(
    df: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()

    for side in (
        "home",
        "away",
    ):
        out[
            f"{side}_ml_ev"
        ] = out.apply(
            lambda r, s=side: compute_ev(
                r.get(
                    f"{s}_model_prob"
                ),
                r.get(
                    f"{s}_dk_moneyline_decimal"
                ),
            ),
            axis=1,
        )

        out[
            f"{side}_ml_edge_vs_market"
        ] = (
            pd.to_numeric(
                out[
                    f"{side}_model_prob"
                ],
                errors="coerce",
            )
            - pd.to_numeric(
                out[
                    f"{side}_market_prob"
                ],
                errors="coerce",
            )
        )

        out[
            f"{side}_ml_kelly"
        ] = out.apply(
            lambda r, s=side: compute_kelly(
                r.get(
                    f"{s}_model_prob"
                ),
                r.get(
                    f"{s}_dk_moneyline_decimal"
                ),
            ),
            axis=1,
        )

        out[
            f"{side}_ml_ev_pct"
        ] = (
            out[
                f"{side}_ml_ev"
            ]
            * 100.0
        )

        out[
            f"{side}_ml_edge_vs_market_pct"
        ] = (
            out[
                f"{side}_ml_edge_vs_market"
            ]
            * 100.0
        )

    out = add_uncertainty_adjusted_ev(
        out, "moneyline",
        [
            ("home_ml", "home_model_prob", "home_market_prob", "home_dk_moneyline_decimal", "home_ml_kelly"),
            ("away_ml", "away_model_prob", "away_market_prob", "away_dk_moneyline_decimal", "away_ml_kelly"),
        ],
    )
    return out


def process_spread_ev(
    df: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()

    for side in (
        "home",
        "away",
    ):
        out[
            f"{side}_spread_ev"
        ] = out.apply(
            lambda r, s=side: compute_ev(
                r.get(
                    f"{s}_spread_model_prob"
                ),
                r.get(
                    f"{s}_dk_spread_decimal"
                ),
            ),
            axis=1,
        )

        out[
            f"{side}_spread_edge_vs_market"
        ] = (
            pd.to_numeric(
                out[
                    f"{side}_spread_model_prob"
                ],
                errors="coerce",
            )
            - pd.to_numeric(
                out[
                    f"{side}_spread_market_prob"
                ],
                errors="coerce",
            )
        )

        out[
            f"{side}_spread_kelly"
        ] = out.apply(
            lambda r, s=side: compute_kelly(
                r.get(
                    f"{s}_spread_model_prob"
                ),
                r.get(
                    f"{s}_dk_spread_decimal"
                ),
            ),
            axis=1,
        )

        out[
            f"{side}_spread_ev_pct"
        ] = (
            out[
                f"{side}_spread_ev"
            ]
            * 100.0
        )

        out[
            f"{side}_spread_edge_vs_market_pct"
        ] = (
            out[
                f"{side}_spread_edge_vs_market"
            ]
            * 100.0
        )

    out = add_uncertainty_adjusted_ev(
        out, "spread",
        [
            ("home_spread", "home_spread_model_prob", "home_spread_market_prob", "home_dk_spread_decimal", "home_spread_kelly"),
            ("away_spread", "away_spread_model_prob", "away_spread_market_prob", "away_dk_spread_decimal", "away_spread_kelly"),
        ],
    )
    return out


def process_total_ev(
    df: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()

    for side in (
        "over",
        "under",
    ):
        out[
            f"{side}_ev"
        ] = out.apply(
            lambda r, s=side: compute_ev(
                r.get(
                    f"{s}_model_prob"
                ),
                r.get(
                    f"dk_total_{s}_decimal"
                ),
            ),
            axis=1,
        )

        out[
            f"{side}_edge_vs_market"
        ] = (
            pd.to_numeric(
                out[
                    f"{side}_model_prob"
                ],
                errors="coerce",
            )
            - pd.to_numeric(
                out[
                    f"{side}_market_prob"
                ],
                errors="coerce",
            )
        )

        out[
            f"{side}_kelly"
        ] = out.apply(
            lambda r, s=side: compute_kelly(
                r.get(
                    f"{s}_model_prob"
                ),
                r.get(
                    f"dk_total_{s}_decimal"
                ),
            ),
            axis=1,
        )

        out[
            f"{side}_ev_pct"
        ] = (
            out[
                f"{side}_ev"
            ]
            * 100.0
        )

        out[
            f"{side}_edge_vs_market_pct"
        ] = (
            out[
                f"{side}_edge_vs_market"
            ]
            * 100.0
        )

    out = add_uncertainty_adjusted_ev(
        out, "total",
        [
            ("over", "over_model_prob", "over_market_prob", "dk_total_over_decimal", "over_kelly"),
            ("under", "under_model_prob", "under_market_prob", "dk_total_under_decimal", "under_kelly"),
        ],
    )
    return out


def in_any_band(
    value: float | None,
    bands: Any,
) -> bool:
    if (
        value is None
        or bands is None
    ):
        return False

    try:
        return any(
            float(lo)
            <= value
            <= float(hi)
            for lo, hi in bands
        )
    except (TypeError, ValueError, OverflowError):
        return False


def parse_game_date(
    value: Any,
) -> datetime | None:
    if (
        value is None
        or pd.isna(value)
    ):
        return None

    text = str(value).strip()

    for fmt in (
        "%Y_%m_%d",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(
                text,
                fmt,
            )
        except ValueError:
            pass

    return None


def date_ok(
    game_date: Any,
    months: list,
    exclude_dow: list,
) -> bool:
    if (
        not months
        and not exclude_dow
    ):
        return True

    dt = parse_game_date(
        game_date
    )

    if dt is None:
        return True

    if (
        months
        and dt.month not in months
    ):
        DEBUG_COUNTS[
            "fail_month"
        ] += 1

        return False

    if (
        exclude_dow
        and dt.weekday()
        in exclude_dow
    ):
        DEBUG_COUNTS[
            "fail_dow"
        ] += 1

        return False

    return True


def passes_filters(
    values: dict,
    side_cfg: dict,
    game_date: Any,
) -> bool:
    if (
        "odds_bands"
        in side_cfg
        and not in_any_band(
            values.get("odds"),
            side_cfg[
                "odds_bands"
            ],
        )
    ):
        DEBUG_COUNTS[
            "fail_odds"
        ] += 1
        return False

    if (
        "line_bands"
        in side_cfg
        and values.get("line")
        is not None
        and not in_any_band(
            values.get("line"),
            side_cfg[
                "line_bands"
            ],
        )
    ):
        DEBUG_COUNTS[
            "fail_line"
        ] += 1
        return False

    if (
        "ev_bands"
        in side_cfg
        and not in_any_band(
            values.get("ev"),
            side_cfg[
                "ev_bands"
            ],
        )
    ):
        DEBUG_COUNTS[
            "fail_ev"
        ] += 1
        return False

    if (
        "kelly_bands"
        in side_cfg
        and not in_any_band(
            values.get("kelly"),
            side_cfg[
                "kelly_bands"
            ],
        )
    ):
        DEBUG_COUNTS[
            "fail_kelly"
        ] += 1
        return False

    if (
        "model_prob_bands"
        in side_cfg
        and not in_any_band(
            values.get(
                "model_prob"
            ),
            side_cfg[
                "model_prob_bands"
            ],
        )
    ):
        DEBUG_COUNTS[
            "fail_model_prob"
        ] += 1
        return False

    if (
        "edge_vs_market_bands"
        in side_cfg
        and not in_any_band(
            values.get(
                "edge_vs_market_pct"
            ),
            side_cfg[
                "edge_vs_market_bands"
            ],
        )
    ):
        DEBUG_COUNTS[
            "fail_edge_vs_market"
        ] += 1
        return False

    return date_ok(
        game_date,
        side_cfg.get(
            "months",
            [],
        )
        or [],
        side_cfg.get(
            "exclude_days_of_week",
            [],
        )
        or [],
    )


def model_edge_threshold(
    settings: dict,
    league: str,
    market: str,
) -> float:
    return float(
        settings[
            league
        ][
            {
                "moneyline": "ML_EDGE",
                "spread": "SPREAD_EDGE",
                "total": "TOTAL_EDGE",
            }[
                market
            ]
        ]
    )


def passes_model_edge(
    ev: float | None,
    settings: dict,
    league: str,
    market: str,
) -> bool:
    if (
        ev is None
        or ev
        < model_edge_threshold(
            settings,
            league,
            market,
        )
    ):
        DEBUG_COUNTS[
            (
                "fail_model_edge_"
                f"{market}"
            )
        ] += 1

        return False

    return True


def pick_one(
    qualifying: list[dict],
    preference: dict,
) -> dict | None:
    if not qualifying:
        return None

    metric = preference.get(
        "metric",
        "ev",
    )

    direction = preference.get(
        "direction",
        "max",
    )

    def key(candidate):
        value = candidate.get(
            metric
        )

        if value is None:
            return (
                float("-inf")
                if direction == "max"
                else float("inf")
            )

        return float(value)

    if direction == "max":
        return max(
            qualifying,
            key=key,
        )

    return min(
        qualifying,
        key=key,
    )


def stake_pct(
    kelly: float | None,
    uncertainty_multiplier: float | None = 1.0,
) -> float | None:
    if kelly is None or kelly <= 0:
        return None
    _, _, requested = requested_stake(kelly, uncertainty_multiplier)
    return requested if requested > 0 else None


def market_config(
    filter_cfg: dict,
    league: str,
    market: str,
) -> dict:
    try:
        cfg = filter_cfg[
            "markets"
        ][
            league
        ][
            market
        ]
    except KeyError as exc:
        raise KeyError(
            "No test config for "
            f"league={league} "
            f"market={market}"
        ) from exc

    return ensure_mapping(
        cfg,
        (
            f"markets.{league}."
            f"{market}"
        ),
    )



def reconcile_ml_vs_spread(
    df: pd.DataFrame,
    filter_cfg: dict,
) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0

    if (
        "game_id" not in df.columns
        or "market_type" not in df.columns
    ):
        return df, 0

    tiebreak = str(
        filter_cfg.get(
            "ml_vs_spread_tiebreak",
            "ev",
        )
    ).strip().lower()

    if tiebreak not in TIEBREAK_COL_MAP:
        tiebreak = "ev"

    metric_col = TIEBREAK_COL_MAP[tiebreak]

    if metric_col not in df.columns:
        return df, 0

    out = df.copy()
    out["_tiebreak_metric"] = pd.to_numeric(
        out[metric_col],
        errors="coerce",
    )

    ml_mask = (
        out["market_type"]
        .astype(str)
        .str.lower()
        == "moneyline"
    )
    spread_mask = (
        out["market_type"]
        .astype(str)
        .str.lower()
        == "spread"
    )

    if (
        not ml_mask.any()
        or not spread_mask.any()
    ):
        return out.drop(
            columns=["_tiebreak_metric"],
        ), 0

    ml_best = (
        out.loc[ml_mask]
        .groupby("game_id")[
            "_tiebreak_metric"
        ]
        .max()
    )

    spread_best = (
        out.loc[spread_mask]
        .groupby("game_id")[
            "_tiebreak_metric"
        ]
        .max()
    )

    conflict_games = (
        ml_best.index.intersection(
            spread_best.index
        )
    )

    if len(conflict_games) == 0:
        return out.drop(
            columns=["_tiebreak_metric"],
        ), 0

    drop_indices = []

    for game_id in conflict_games:
        ml_value = ml_best.loc[game_id]
        spread_value = spread_best.loc[
            game_id
        ]

        if (
            pd.isna(ml_value)
            and pd.isna(spread_value)
        ):
            losing_market = "spread"
        elif pd.isna(ml_value):
            losing_market = "moneyline"
        elif pd.isna(spread_value):
            losing_market = "spread"
        else:
            losing_market = (
                "spread"
                if ml_value >= spread_value
                else "moneyline"
            )

        loss_mask = (
            out["game_id"].eq(game_id)
            & (
                out["market_type"]
                .astype(str)
                .str.lower()
                == losing_market
            )
        )

        drop_indices.extend(
            out.index[
                loss_mask
            ].tolist()
        )

        DEBUG_COUNTS[
            (
                "ml_vs_spread_dropped_"
                f"{losing_market}"
            )
        ] += int(
            loss_mask.sum()
        )

    dropped = len(drop_indices)

    if dropped:
        out = out.drop(
            index=drop_indices
        )

    out = out.drop(
        columns=["_tiebreak_metric"],
    )

    return (
        out.reset_index(drop=True),
        dropped,
    )


def build_moneyline_sides(
    row,
    league,
    game_date,
    cfg,
    settings,
):
    sides = []

    for side in (
        "home",
        "away",
    ):
        scfg = ensure_mapping(
            cfg.get(side),
            (
                f"markets.{league}."
                f"moneyline.{side}"
            ),
        )

        if not scfg.get(
            "enabled",
            True,
        ):
            continue

        odds = fv(
            row.get(
                f"{side}_dk_moneyline_american"
            )
        )

        ev = fv(
            row.get(
                f"{side}_ml_ev"
            )
        )

        kelly = fv(
            row.get(
                f"{side}_ml_kelly"
            )
        )

        mp = fv(
            row.get(
                f"{side}_model_prob"
            )
        )

        if mp is None:
            mp = fv(
                row.get(
                    f"{side}_prob"
                )
            )

        evm = fv(
            row.get(
                f"{side}_ml_edge_vs_market_pct"
            )
        )

        if not passes_model_edge(
            ev,
            settings,
            league,
            "moneyline",
        ):
            DEBUG_COUNTS[
                "rejected_ml"
            ] += 1
            continue

        vals = {
            "odds": odds,
            "ev": ev,
            "kelly": kelly,
            "model_prob": mp,
            "edge_vs_market_pct": evm,
        }

        if passes_filters(
            vals,
            scfg,
            game_date,
        ):
            sides.append({
                "side": side,
                "line": odds,
                "odds": odds,
                "ev": ev,
                "kelly": kelly,
                "model_prob": mp,
                "edge_vs_market": evm,
            })
        else:
            DEBUG_COUNTS[
                "rejected_ml"
            ] += 1

    return sides


def build_spread_sides(
    row,
    league,
    game_date,
    cfg,
    settings,
):
    sides = []

    for side in (
        "home",
        "away",
    ):
        scfg = ensure_mapping(
            cfg.get(side),
            (
                f"markets.{league}."
                f"spread.{side}"
            ),
        )

        if not scfg.get(
            "enabled",
            True,
        ):
            continue

        line = fv(
            row.get(
                f"{side}_spread"
            )
        )

        odds = fv(
            row.get(
                f"{side}_dk_spread_american"
            )
        )

        ev = fv(
            row.get(
                f"{side}_spread_ev"
            )
        )

        kelly = fv(
            row.get(
                f"{side}_spread_kelly"
            )
        )

        mp = fv(
            row.get(
                f"{side}_spread_model_prob"
            )
        )

        evm = fv(
            row.get(
                f"{side}_spread_edge_vs_market_pct"
            )
        )

        if not passes_model_edge(
            ev,
            settings,
            league,
            "spread",
        ):
            DEBUG_COUNTS[
                "rejected_spread"
            ] += 1
            continue

        vals = {
            "odds": odds,
            "line": line,
            "ev": ev,
            "kelly": kelly,
            "model_prob": mp,
            "edge_vs_market_pct": evm,
        }

        if passes_filters(
            vals,
            scfg,
            game_date,
        ):
            sides.append({
                "side": side,
                "line": line,
                "odds": odds,
                "ev": ev,
                "kelly": kelly,
                "model_prob": mp,
                "edge_vs_market": evm,
            })
        else:
            DEBUG_COUNTS[
                "rejected_spread"
            ] += 1

    return sides


def build_total_sides(
    row,
    league,
    game_date,
    cfg,
    settings,
):
    sides = []

    line = fv(
        row.get("total")
    )

    for side in (
        "over",
        "under",
    ):
        scfg = ensure_mapping(
            cfg.get(side),
            (
                f"markets.{league}."
                f"total.{side}"
            ),
        )

        if not scfg.get(
            "enabled",
            True,
        ):
            continue

        odds = fv(
            row.get(
                f"dk_total_{side}_american"
            )
        )

        ev = fv(
            row.get(
                f"{side}_ev"
            )
        )

        kelly = fv(
            row.get(
                f"{side}_kelly"
            )
        )

        mp = fv(
            row.get(
                f"{side}_model_prob"
            )
        )

        evm = fv(
            row.get(
                f"{side}_edge_vs_market_pct"
            )
        )

        if not passes_model_edge(
            ev,
            settings,
            league,
            "total",
        ):
            DEBUG_COUNTS[
                "rejected_total"
            ] += 1
            continue

        vals = {
            "odds": odds,
            "line": line,
            "ev": ev,
            "kelly": kelly,
            "model_prob": mp,
            "edge_vs_market_pct": evm,
        }

        if passes_filters(
            vals,
            scfg,
            game_date,
        ):
            sides.append({
                "side": side,
                "line": line,
                "odds": odds,
                "ev": ev,
                "kelly": kelly,
                "model_prob": mp,
                "edge_vs_market": evm,
            })
        else:
            DEBUG_COUNTS[
                "rejected_total"
            ] += 1

    return sides


SIDE_BUILDERS = {
    "moneyline": build_moneyline_sides,
    "spread": build_spread_sides,
    "total": build_total_sides,
}


def select_bets_for_market(
    df,
    league,
    market,
    filter_cfg,
    settings,
):
    cfg = market_config(
        filter_cfg,
        league,
        market,
    )

    if not cfg.get(
        "enabled",
        True,
    ):
        return pd.DataFrame()

    mode = str(
        cfg.get(
            "selection_mode",
            "pick_one",
        )
    ).strip().lower()

    preference = (
        cfg.get(
            "pick_preference"
        )
        or {
            "metric": "ev",
            "direction": "max",
        }
    )

    out_rows = []

    for _, row in df.iterrows():
        game_date = row.get(
            "game_date"
        )

        sides = SIDE_BUILDERS[
            market
        ](
            row,
            league,
            game_date,
            cfg,
            settings,
        )
        sides = [attach_candidate_uncertainty(row, market, side) for side in sides]

        if not sides:
            continue

        if mode == "all_qualifying":
            picks = sides
        else:
            chosen = pick_one(
                sides,
                preference,
            )

            picks = (
                [chosen]
                if chosen
                else []
            )

        for sel in picks:
            DEBUG_COUNTS[
                "selected"
            ] += 1

            result = row.to_dict()

            result.update({
                "bet_side": sel["side"],
                "bet_line": sel["line"],
                "bet_odds_american": sel[
                    "odds"
                ],
                "bet_ev": sel["ev"],
                "bet_raw_ev": sel["raw_ev"],
                "bet_uncertainty_adjusted_ev": sel["uncertainty_adjusted_ev"],
                "bet_kelly": sel["kelly"],
                "bet_raw_kelly": sel["raw_kelly"],
                "bet_model_prob": sel["model_prob"],
                "bet_adjusted_model_prob": sel["adjusted_model_prob"],
                "bet_edge_vs_market": sel["edge_vs_market"],
                "bet_uncertainty_multiplier": sel["uncertainty_multiplier"],
                "bet_uncertainty_points": sel["uncertainty_points"],
                "bet_signal_points": sel["signal_points"],
                "bet_requested_stake_pct": stake_pct(
                    sel["raw_kelly"], sel["uncertainty_multiplier"]
                ),
                "bet_stake_pct": stake_pct(
                    sel["raw_kelly"], sel["uncertainty_multiplier"]
                ),
                "market_type": market,
                "league_lower": league,
                "league": league.upper(),
                "game_date": game_date,
            })

            out_rows.append(result)

    return pd.DataFrame(out_rows)


def determine_outcome(row) -> str:
    market = str(
        row.get(
            "market_type",
            "",
        )
    ).lower()

    side = str(
        row.get(
            "bet_side",
            "",
        )
    ).lower()

    home = fv(
        row.get("home_score")
    )

    away = fv(
        row.get("away_score")
    )

    if (
        home is None
        or away is None
    ):
        return "Unknown"

    if market == "moneyline":
        if home == away:
            return "Push"

        home_won = home > away

        return (
            "Win"
            if (
                side == "home"
                and home_won
            )
            or (
                side == "away"
                and not home_won
            )
            else "Loss"
        )

    if market == "spread":
        line = fv(
            row.get("bet_line")
        )

        if line is None:
            return "Unknown"

        if side == "home":
            diff = (
                home
                + line
                - away
            )
        elif side == "away":
            diff = (
                away
                + line
                - home
            )
        else:
            return "Unknown"

        if abs(diff) < 1e-9:
            return "Push"

        return (
            "Win"
            if diff > 0
            else "Loss"
        )

    if market == "total":
        line = fv(
            row.get("bet_line")
        )

        if line is None:
            return "Unknown"

        actual_total = (
            home + away
        )

        if abs(
            actual_total - line
        ) < 1e-9:
            return "Push"

        return (
            "Win"
            if (
                actual_total > line
                and side == "over"
            )
            or (
                actual_total < line
                and side == "under"
            )
            else "Loss"
        )

    return "Unknown"


def compute_profits(row):
    result = str(
        row.get(
            "bet_result",
            "",
        )
    ).strip()

    decimal = (
        american_to_decimal_or_none(
            row.get(
                "bet_odds_american"
            )
        )
    )

    if result == "Push":
        return (
            0.0,
            0.0,
        )

    if (
        result
        not in (
            "Win",
            "Loss",
        )
        or decimal is None
        or decimal <= 1
    ):
        return (
            None,
            None,
        )

    stake = fv(
        row.get(
            "bet_stake_pct"
        )
    )

    if result == "Win":
        return (
            decimal - 1.0,
            (
                stake
                * (
                    decimal - 1.0
                )
                if stake is not None
                else None
            ),
        )

    return (
        -1.0,
        (
            -stake
            if stake is not None
            else None
        ),
    )


def grade_selections(
    selections: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    if selections.empty:
        return pd.DataFrame()

    merged = selections.merge(
        scores,
        on="game_id",
        how="left",
        suffixes=(
            "",
            "_score",
        ),
    )

    for col in RESULT_COLUMNS:
        score_col = (
            f"{col}_score"
        )

        if score_col in merged.columns:
            merged[col] = (
                merged[
                    score_col
                ].combine_first(
                    merged[col]
                )
                if col in merged.columns
                else merged[
                    score_col
                ]
            )

            merged = merged.drop(
                columns=[
                    score_col
                ]
            )

    merged["bet_result"] = (
        merged.apply(
            determine_outcome,
            axis=1,
        )
    )

    profits = merged.apply(
        compute_profits,
        axis=1,
        result_type="expand",
    )

    profits.columns = [
        "profit_unit",
        "profit_kelly",
    ]

    merged = pd.concat(
        [
            merged,
            profits,
        ],
        axis=1,
    )

    keys = [
        c
        for c in [
            "source_file",
            "game_id",
            "market_type",
            "bet_side",
        ]
        if c in merged.columns
    ]

    if keys:
        return merged.drop_duplicates(
            subset=keys,
            keep="last",
        )

    return merged


def summarize_group(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    cols = [
        *group_cols,
        "bets",
        "wins",
        "losses",
        "pushes",
        "unknown",
        "win_rate",
        "profit_units",
        "roi_units",
        "profit_kelly",
        "kelly_staked",
        "roi_kelly",
        "avg_ev",
        "avg_kelly",
        "avg_model_prob",
        "avg_edge_vs_market",
        "avg_odds_american",
    ]

    if df.empty:
        return pd.DataFrame(
            columns=cols
        )

    records = []

    grouped = (
        df.groupby(
            group_cols,
            dropna=False,
            sort=True,
        )
        if group_cols
        else [
            (
                (),
                df,
            )
        ]
    )

    for keys, group in grouped:
        if (
            group_cols
            and not isinstance(
                keys,
                tuple,
            )
        ):
            keys = (keys,)
        elif not group_cols:
            keys = ()

        wins = int(
            (
                group["bet_result"]
                == "Win"
            ).sum()
        )

        losses = int(
            (
                group["bet_result"]
                == "Loss"
            ).sum()
        )

        pushes = int(
            (
                group["bet_result"]
                == "Push"
            ).sum()
        )

        unknown = int(
            (
                group["bet_result"]
                == "Unknown"
            ).sum()
        )

        bets = (
            wins
            + losses
            + pushes
            + unknown
        )

        decisions = (
            wins + losses
        )

        graded_stakes = (
            wins
            + losses
            + pushes
        )

        profit_units = float(
            pd.to_numeric(
                group[
                    "profit_unit"
                ],
                errors="coerce",
            ).sum(
                skipna=True
            )
        )

        profit_kelly = float(
            pd.to_numeric(
                group[
                    "profit_kelly"
                ],
                errors="coerce",
            ).sum(
                skipna=True
            )
        )

        kelly_staked = float(
            pd.to_numeric(
                group[
                    "bet_stake_pct"
                ],
                errors="coerce",
            )
            .fillna(0.0)
            .sum()
        )

        record = dict(
            zip(
                group_cols,
                keys,
            )
        )

        record.update({
            "bets": bets,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "unknown": unknown,
            "win_rate": (
                wins / decisions
                if decisions
                else None
            ),
            "profit_units": profit_units,
            "roi_units": (
                profit_units
                / graded_stakes
                if graded_stakes
                else None
            ),
            "profit_kelly": profit_kelly,
            "kelly_staked": kelly_staked,
            "roi_kelly": (
                profit_kelly
                / kelly_staked
                if kelly_staked > 0
                else None
            ),
            "avg_ev": pd.to_numeric(
                group["bet_ev"],
                errors="coerce",
            ).mean(),
            "avg_kelly": pd.to_numeric(
                group["bet_kelly"],
                errors="coerce",
            ).mean(),
            "avg_model_prob": pd.to_numeric(
                group[
                    "bet_model_prob"
                ],
                errors="coerce",
            ).mean(),
            "avg_edge_vs_market": (
                pd.to_numeric(
                    group[
                        "bet_edge_vs_market"
                    ],
                    errors="coerce",
                ).mean()
            ),
            "avg_odds_american": (
                pd.to_numeric(
                    group[
                        "bet_odds_american"
                    ],
                    errors="coerce",
                ).mean()
            ),
        })

        records.append(record)

    return pd.DataFrame(
        records,
        columns=cols,
    )


def build_reports(
    graded: pd.DataFrame,
    reports_dir: Path,
) -> dict[
    str,
    pd.DataFrame,
]:
    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports = {
        "overall": summarize_group(
            graded,
            [],
        ),
        "by_source": summarize_group(
            graded,
            [
                "source_file",
                "league",
            ],
        ),
        "by_league": summarize_group(
            graded,
            [
                "league"
            ],
        ),
        "by_market": summarize_group(
            graded,
            [
                "league",
                "market_type",
            ],
        ),
        "by_market_side": summarize_group(
            graded,
            [
                "league",
                "market_type",
                "bet_side",
            ],
        ),
    }

    names = {
        "overall": "overall.csv",
        "by_source": (
            "performance_by_source.csv"
        ),
        "by_league": (
            "performance_by_league.csv"
        ),
        "by_market": (
            "performance_by_market.csv"
        ),
        "by_market_side": (
            "performance_by_market_side.csv"
        ),
    }

    for name, report in reports.items():
        atomic_write_csv(
            report,
            reports_dir
            / names[name],
        )

    atomic_write_csv(
        pd.DataFrame([
            {
                "reason": key,
                "count": value,
            }
            for key, value
            in sorted(
                DEBUG_COUNTS.items()
            )
        ]),
        reports_dir
        / "filter_counts.csv",
    )

    return reports


def collect_config_warnings(
    filter_cfg: dict,
) -> list[str]:
    warnings = []

    supported = {
        "ev",
        "kelly",
        "model_prob",
        "edge_vs_market",
    }

    markets = (
        filter_cfg.get("markets")
        or {}
    )

    for league in LEAGUES:
        for market in MARKETS:
            mcfg = (
                (
                    markets.get(league)
                    or {}
                )
                .get(market)
                or {}
            )

            metric = str(
                (
                    mcfg.get(
                        "pick_preference"
                    )
                    or {}
                )
                .get(
                    "metric",
                    "ev",
                )
            ).strip()

            if metric not in supported:
                warnings.append(
                    f"markets.{league}."
                    f"{market}."
                    "pick_preference.metric="
                    f"{metric!r} is not a "
                    "production candidate key"
                )

    return warnings


def write_manifest(
    path,
    run_id,
    model_config_path,
    filter_config_path,
    production_markets_path,
    model_source,
    input_files,
    settings,
    config_warnings,
    total_rows,
    total_selected,
    total_graded,
):
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_utc": now_utc(),
        "backtest_method": (
            "frozen_historical_predictions_"
            "current_downstream_logic"
        ),
        "historical_bias_handling": (
            "reverse_stored_bias_then_"
            "point_in_time_replay_of_"
            "model_config_bias"
        ),
        "season_identity": (
            "internal_season_from_"
            "historical_input_filename"
        ),
        "model_source": model_source,
        "outcome_leakage_prevention": (
            "final_score_columns_removed_"
            "before_selection_and_rejoined_"
            "by_game_id_after_selection"
        ),
        "model_config": {
            "path": str(
                model_config_path
            ),
            "sha256": sha256_file(
                model_config_path
            ),
        },
        "filter_config": {
            "path": str(
                filter_config_path
            ),
            "sha256": sha256_file(
                filter_config_path
            ),
        },
        "production_markets_config": {
            "path": str(
                production_markets_path
            ),
            "sha256": sha256_file(
                production_markets_path
            ),
        },
        "staking_config": {
            "path": str(DEFAULT_STAKING_CONFIG),
            "sha256": sha256_file(DEFAULT_STAKING_CONFIG),
        },
        "input_files": [
            {
                "path": str(p),
                "sha256": sha256_file(p),
                "size_bytes": (
                    p.stat().st_size
                ),
            }
            for p in input_files
        ],
        "model_settings": settings,
        "config_warnings": (
            config_warnings
        ),
        "counts": {
            "historical_rows": total_rows,
            "selected_bets": total_selected,
            "graded_bets": total_graded,
        },
        "filter_counts": dict(
            sorted(
                DEBUG_COUNTS.items()
            )
        ),
    }

    safe_path = path.resolve()
    if safe_path.name != "run_manifest.yaml":
        raise ValueError(f"Unexpected manifest path: {path}")

    safe_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with safe_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            manifest,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def append_run_index(
    index_path,
    run_id,
    overall,
):
    row = {
        "run_id": run_id,
        "generated_at_utc": now_utc(),
        "bets": 0,
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "unknown": 0,
        "win_rate": None,
        "profit_units": 0.0,
        "roi_units": None,
        "profit_kelly": 0.0,
        "roi_kelly": None,
    }

    if not overall.empty:
        first = overall.iloc[0]

        for key in list(row):
            if (
                key
                not in {
                    "run_id",
                    "generated_at_utc",
                }
                and key in first.index
            ):
                row[key] = first[key]

    new = pd.DataFrame([row])

    if index_path.exists():
        combined = pd.concat(
            [
                pd.read_csv(index_path),
                new,
            ],
            ignore_index=True,
        )
    else:
        combined = new

    atomic_write_csv(
        combined.drop_duplicates(
            subset=[
                "run_id"
            ],
            keep="last",
        ),
        index_path,
    )


def run_production_parity_test(
    feature_df: pd.DataFrame,
    league: str,
    settings: dict,
    production_filter_cfg: dict,
    parity_rows: int,
    logger: RunLogger,
) -> None:
    import numpy as np

    sample = (
        feature_df
        .head(
            max(
                1,
                int(parity_rows),
            )
        )
        .copy()
        .reset_index(drop=True)
    )

    if sample.empty:
        raise AssertionError(
            f"PARITY FAILED {league}: "
            "no rows available"
        )

    prod_juice = load_module_from_path(
        (
            "parity_build_juice_"
            f"{league}"
        ),
        PRODUCTION_BUILD_JUICE,
    )

    prod_ev = load_module_from_path(
        (
            "parity_ev_kelly_"
            f"{league}"
        ),
        PRODUCTION_EV_KELLY,
    )

    prod_select = load_module_from_path(
        (
            "parity_select_"
            f"{league}"
        ),
        PRODUCTION_SELECT,
    )

    league_upper = league.upper()

    prod_settings = (
        prod_juice.LEAGUE_SETTINGS[
            league_upper
        ]
    )

    for key in (
        "ML_EDGE",
        "SPREAD_EDGE",
        "TOTAL_EDGE",
        "SPREAD_STD",
        "TOTAL_STD",
    ):
        if not math.isclose(
            float(
                settings[
                    league
                ][key]
            ),
            float(
                prod_settings[key]
            ),
            abs_tol=1e-12,
        ):
            raise AssertionError(
                f"PARITY FAILED {league}: "
                f"setting {key} "
                "backtest="
                f"{settings[league][key]} "
                "production="
                f"{prod_settings[key]}"
            )

    back_frames = {
        "moneyline": (
            process_moneyline_ev(
                process_moneyline_juice(
                    sample,
                    settings[league],
                )
            )
        ),
        "spread": (
            process_spread_ev(
                process_spread_juice(
                    sample,
                    settings[league],
                )
            )
        ),
        "total": (
            process_total_ev(
                process_total_juice(
                    sample,
                    settings[league],
                )
            )
        ),
    }

    with tempfile.TemporaryDirectory(
        prefix="basketball_parity_",
    ) as tmp:
        prod_juice.OUTPUT_DIR = (
            Path(tmp)
            / "juice"
        )

        for market_dir in MARKETS:
            (
                prod_juice.OUTPUT_DIR
                / league
                / market_dir
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

        back_selected_parts = []
        prod_selected_parts = []

        for market in MARKETS:
            if market == "moneyline":
                (
                    out_path,
                    _,
                ) = prod_juice.process_moneyline(
                    sample.copy(),
                    "2000_01_01",
                    league_upper,
                    prod_settings,
                    league,
                )

                juice_df = pd.read_csv(
                    out_path,
                    dtype={
                        "game_id": str
                    },
                )

                prod_frame = (
                    prod_ev.process_moneyline(
                        juice_df
                    )
                )

                prob_cols = [
                    "home_model_prob",
                    "away_model_prob",
                ]

            elif market == "spread":
                (
                    out_path,
                    _,
                ) = prod_juice.process_spread(
                    sample.copy(),
                    "2000_01_01",
                    league_upper,
                    prod_settings,
                    league,
                )

                juice_df = pd.read_csv(
                    out_path,
                    dtype={
                        "game_id": str
                    },
                )

                prod_frame = (
                    prod_ev.process_spread(
                        juice_df
                    )
                )

                prob_cols = [
                    "home_spread_model_prob",
                    "away_spread_model_prob",
                ]

            else:
                (
                    out_path,
                    _,
                ) = prod_juice.process_totals(
                    sample.copy(),
                    "2000_01_01",
                    league_upper,
                    prod_settings,
                    league,
                )

                juice_df = pd.read_csv(
                    out_path,
                    dtype={
                        "game_id": str
                    },
                )

                prod_frame = (
                    prod_ev.process_total(
                        juice_df
                    )
                )

                prob_cols = [
                    "over_model_prob",
                    "under_model_prob",
                ]

            back = back_frames[market]

            for col in prob_cols:
                a = pd.to_numeric(
                    back[col],
                    errors="coerce",
                ).to_numpy(float)

                b = pd.to_numeric(
                    prod_frame[col],
                    errors="coerce",
                ).to_numpy(float)

                if not np.allclose(
                    a,
                    b,
                    rtol=0.0,
                    atol=1e-12,
                    equal_nan=True,
                ):
                    raise AssertionError(
                        "PARITY FAILED "
                        f"{league}.{market}."
                        f"{col}"
                    )

            back_selected = (
                select_bets_for_market(
                    back,
                    league,
                    market,
                    production_filter_cfg,
                    settings,
                )
            )

            prod_cfg = (
                prod_select.market_cfg(
                    league,
                    market,
                )
            )

            prod_rows = []

            for _, row in prod_frame.iterrows():
                game_date = row.get(
                    "game_date"
                )

                sides = (
                    prod_select
                    .SIDE_BUILDERS[
                        market
                    ](
                        row,
                        league,
                        game_date,
                        prod_cfg,
                    )
                )

                if not sides:
                    continue

                mode = prod_cfg.get(
                    "selection_mode",
                    "pick_one",
                )

                preference = (
                    prod_cfg.get(
                        "pick_preference",
                        {
                            "metric": "ev",
                            "direction": "max",
                        },
                    )
                )

                if mode == "all_qualifying":
                    picks = sides
                else:
                    picks = [
                        prod_select.pick(
                            sides,
                            preference,
                        )
                    ]

                for sel in picks:
                    if sel is None:
                        continue

                    prod_rows.append({
                        "game_id": row.get(
                            "game_id",
                            "",
                        ),
                        "market_type": market,
                        "bet_side": sel[
                            "side"
                        ],
                        "bet_ev": sel.get(
                            "ev"
                        ),
                        "bet_kelly": sel.get(
                            "kelly"
                        ),
                        "bet_edge_vs_market": sel.get(
                            "edge_vs_market"
                        ),
                    })

            prod_selected = pd.DataFrame(
                prod_rows
            )

            def keys(
                frame: pd.DataFrame,
            ):
                if frame.empty:
                    return set()

                return {
                    (
                        str(
                            r.get(
                                "game_id",
                                "",
                            )
                        ).strip(),
                        str(
                            r.get(
                                "market_type",
                                "",
                            )
                        ).lower(),
                        str(
                            r.get(
                                "bet_side",
                                "",
                            )
                        ).lower(),
                    )
                    for _, r
                    in frame.iterrows()
                }

            back_keys = keys(
                back_selected
            )

            prod_keys = keys(
                prod_selected
            )

            if back_keys != prod_keys:
                raise AssertionError(
                    "PARITY FAILED "
                    f"{league}.{market}."
                    "selection | "
                    f"backtest={sorted(back_keys)} | "
                    f"production={sorted(prod_keys)}"
                )

            if not back_selected.empty:
                back_selected_parts.append(
                    back_selected
                )

            if not prod_selected.empty:
                prod_selected_parts.append(
                    prod_selected
                )

        back_combined = (
            pd.concat(
                back_selected_parts,
                ignore_index=True,
            )
            if back_selected_parts
            else pd.DataFrame()
        )

        prod_combined = (
            pd.concat(
                prod_selected_parts,
                ignore_index=True,
            )
            if prod_selected_parts
            else pd.DataFrame()
        )

        back_reconciled, _ = (
            reconcile_ml_vs_spread(
                back_combined,
                production_filter_cfg,
            )
        )

        prod_reconciled, _ = (
            reconcile_ml_vs_spread(
                prod_combined,
                production_filter_cfg,
            )
        )

        if keys(back_reconciled) != keys(
            prod_reconciled
        ):
            raise AssertionError(
                "PARITY FAILED "
                f"{league}.ml_vs_spread."
                "reconciliation | "
                "backtest="
                f"{sorted(keys(back_reconciled))} | "
                "production="
                f"{sorted(keys(prod_reconciled))}"
            )

    logger.log(
        "PARITY PASS | "
        f"{league.upper()} | "
        f"rows={len(sample)}"
    )


def process_historical_file(
    path,
    league,
    settings,
    filter_cfg,
    production_filter_cfg,
    model_cfg,
    model_source,
    history_state,
    working_dir,
    selections_dir,
    graded_dir,
    parity_rows,
    logger,
):
    source_file = path.stem

    internal_season = (
        season_from_input_filename(
            path,
            league,
        )
    )

    logger.log(
        f"[{league.upper()}] "
        f"reading {path} | "
        f"internal_season="
        f"{internal_season}"
    )

    raw = pd.read_csv(
        path,
        dtype={
            "game_id": str,
            "game_date": str,
        },
    )

    validate_historical_input(
        raw,
        path,
        league,
    )

    raw = apply_model_source(
        raw,
        model_source,
        path,
    )

    raw = (
        apply_point_in_time_production_bias(
            raw,
            league,
            internal_season,
            model_cfg,
            history_state,
        )
    )

    incomplete = int(
        (
            ~raw[
                "_production_input_valid"
            ]
        ).sum()
    )

    if incomplete:
        logger.log(
            f"[{league.upper()}] "
            f"{source_file}: "
            f"skipping {incomplete} "
            "incomplete historical rows "
            "with missing projection or "
            "final-score values",
            "WARN",
        )

    warmup = int(
        (
            raw[
                "_production_input_valid"
            ]
            & (
                ~raw[
                    "_production_bias_ready"
                ]
            )
        ).sum()
    )

    if warmup:
        logger.log(
            f"[{league.upper()}] "
            f"{source_file}: "
            f"skipping {warmup} "
            "early rows because production "
            "rolling/regime-aware bias "
            "does not yet have its full "
            "lookback",
            "WARN",
        )

    raw = (
        raw[
            raw[
                "_production_input_valid"
            ]
            & raw[
                "_production_bias_ready"
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    if raw.empty:
        raise ValueError(
            f"{path.name} has no usable rows "
            "with production bias ready"
        )

    raw = raw.drop(
        columns=[
            "_production_input_valid",
            "_production_bias_ready",
        ],
        errors="ignore",
    )

    raw[
        "source_file"
    ] = source_file

    (
        feature_df,
        scores,
    ) = split_features_and_scores(
        raw
    )

    feature_df[
        "source_file"
    ] = source_file

    scores[
        "source_file"
    ] = source_file

    run_production_parity_test(
        feature_df,
        league,
        settings,
        production_filter_cfg,
        parity_rows,
        logger,
    )

    frames = {
        "moneyline": (
            process_moneyline_ev(
                process_moneyline_juice(
                    feature_df,
                    settings[league],
                )
            )
        ),
        "spread": (
            process_spread_ev(
                process_spread_juice(
                    feature_df,
                    settings[league],
                )
            )
        ),
        "total": (
            process_total_ev(
                process_total_juice(
                    feature_df,
                    settings[league],
                )
            )
        ),
    }

    selected_parts = []

    for market, market_df in (
        frames.items()
    ):
        atomic_write_csv(
            market_df,
            (
                working_dir
                / league
                / market
                / (
                    f"{source_file}_"
                    f"{market}.csv"
                )
            ),
        )

        selected = (
            select_bets_for_market(
                market_df,
                league,
                market,
                filter_cfg,
                settings,
            )
        )

        if not selected.empty:
            selected_parts.append(
                selected
            )

        logger.log(
            f"[{league.upper()}] "
            f"{source_file} "
            f"{market}: "
            f"rows={len(market_df)} "
            f"selected={len(selected)}"
        )

    if selected_parts:
        selections = pd.concat(
            selected_parts,
            ignore_index=True,
        )
        selections, ml_vs_spread_dropped = (
            reconcile_ml_vs_spread(
                selections,
                filter_cfg,
            )
        )
        logger.log(
            f"[{league.upper()}] "
            f"{source_file}: "
            "ml_vs_spread_dropped="
            f"{ml_vs_spread_dropped}"
        )
    else:
        selections = pd.DataFrame()

    atomic_write_csv(
        selections,
        (
            selections_dir
            / league
            / (
                f"{source_file}_"
                "selected.csv"
            )
        ),
    )

    if selections.empty:
        graded = pd.DataFrame()
    else:
        score_cols = [
            c
            for c in scores.columns
            if c != "source_file"
        ]

        graded = grade_selections(
            selections,
            scores[score_cols],
        )

    atomic_write_csv(
        graded,
        (
            graded_dir
            / league
            / (
                f"{source_file}_"
                "graded.csv"
            )
        ),
    )

    logger.log(
        f"[{league.upper()}] "
        f"{source_file}: "
        f"internal_season="
        f"{internal_season} "
        f"historical_rows={len(raw)} "
        f"selected={len(selections)} "
        f"graded={len(graded)}"
    )

    return (
        selections,
        graded,
        len(raw),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay combined historical "
            "basketball files through "
            "current downstream model "
            "and selection logic."
        )
    )

    parser.add_argument(
        "--backtest-dir",
        default=str(
            DEFAULT_BACKTEST_DIR
        ),
    )

    parser.add_argument(
        "--model-config",
        default=str(
            DEFAULT_MODEL_CONFIG
        ),
    )

    parser.add_argument(
        "--markets-config",
        default=str(
            DEFAULT_MARKETS_CONFIG
        ),
    )

    parser.add_argument(
        "--model-source",
        choices=MODEL_SOURCES,
        default=None,
    )

    parser.add_argument(
        "--parity-rows",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--run-name",
        default=None,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    backtest_dir = Path(
        args.backtest_dir
    )

    input_dir = (
        backtest_dir
        / "input"
    )

    configs_dir = (
        backtest_dir
        / "configs"
    )

    working_dir = (
        backtest_dir
        / "working"
    )

    selections_dir = (
        backtest_dir
        / "selections"
    )

    graded_dir = (
        backtest_dir
        / "graded"
    )

    reports_dir = (
        backtest_dir
        / "reports"
    )

    runs_dir = (
        backtest_dir
        / "runs"
    )

    model_config_path = Path(
        args.model_config
    )

    production_markets_path = Path(
        args.markets_config
    )

    filter_config_path = (
        configs_dir
        / "markets_test.yaml"
    )

    for folder in (
        input_dir,
        configs_dir,
        working_dir,
        selections_dir,
        graded_dir,
        reports_dir,
        runs_dir,
    ):
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    run_id = (
        sanitize_run_name(
            args.run_name
        )
        if args.run_name
        else timestamp_id()
    )

    run_dir = (
        runs_dir
        / run_id
    )

    if run_dir.exists():
        raise FileExistsError(
            "Run snapshot already exists: "
            f"{run_dir}"
        )

    clear_directory_contents(
        working_dir
    )

    clear_directory_contents(
        selections_dir
    )

    clear_directory_contents(
        graded_dir
    )

    clear_directory_contents(
        reports_dir
    )

    logger = RunLogger(
        reports_dir
        / "basketball_backtest.txt"
    )

    logger.log(
        f"run_id={run_id}"
    )

    logger.log(
        f"backtest_dir={backtest_dir}"
    )

    logger.log(
        "model_config="
        f"{model_config_path}"
    )

    logger.log(
        "filter_config="
        f"{filter_config_path}"
    )

    logger.log(
        "production_markets_config="
        f"{production_markets_path}"
    )
    logger.log(f"staking_config={DEFAULT_STAKING_CONFIG}")

    model_cfg = read_yaml(
        model_config_path
    )

    filter_cfg = read_yaml(
        filter_config_path
    )

    production_markets_cfg = (
        read_yaml(
            production_markets_path
        )
    )

    model_source = resolve_model_source(
        model_cfg,
        args.model_source,
    )

    logger.log(
        f"model_source={model_source}"
    )

    production_filter_cfg = (
        apply_production_selection_policy(
            production_markets_cfg,
            production_markets_cfg,
        )
    )

    filter_cfg = (
        apply_production_selection_policy(
            filter_cfg,
            production_markets_cfg,
        )
    )

    settings = build_league_settings(
        model_cfg
    )

    config_warnings = (
        collect_config_warnings(
            filter_cfg
        )
    )

    for warning in config_warnings:
        logger.log(
            warning,
            "WARN",
        )

    input_files = []

    for league in LEAGUES:
        files = sorted(
            input_dir.glob(
                f"*_{league.upper()}.csv"
            )
        )

        if not files:
            raise FileNotFoundError(
                "No historical input files "
                f"found for {league.upper()} "
                f"in {input_dir}"
            )

        for path in files:
            season_from_input_filename(
                path,
                league,
            )

        input_files.extend(files)

    all_selections = []
    all_graded = []
    total_rows = 0

    for league in LEAGUES:
        history_state = {
            "margin_errors": [],
            "total_errors": [],
        }

        paths = sorted(
            input_dir.glob(
                f"*_{league.upper()}.csv"
            ),
            key=lambda p, bound_league=league: (
                season_from_input_filename(
                    p,
                    bound_league,
                ),
                p.name,
            ),
        )

        for path in paths:
            (
                selections,
                graded,
                row_count,
            ) = process_historical_file(
                path,
                league,
                settings,
                filter_cfg,
                production_filter_cfg,
                model_cfg,
                model_source,
                history_state,
                working_dir,
                selections_dir,
                graded_dir,
                args.parity_rows,
                logger,
            )

            total_rows += row_count

            if not selections.empty:
                all_selections.append(
                    selections
                )

            if not graded.empty:
                all_graded.append(
                    graded
                )

    if all_selections:
        combined_selected = pd.concat(
            all_selections,
            ignore_index=True,
        )
    else:
        combined_selected = (
            pd.DataFrame()
        )

    if all_graded:
        combined_graded = pd.concat(
            all_graded,
            ignore_index=True,
        )
    else:
        combined_graded = (
            pd.DataFrame()
        )

    # Keep every selection produced by the configured selection rules.
    # Stake suggestions are informational and never remove a selected row.
    atomic_write_csv(
        combined_selected,
        selections_dir
        / "all_selected.csv",
    )

    atomic_write_csv(
        combined_graded,
        graded_dir
        / "all_graded.csv",
    )
    logger.log(f"selection rows retained without exposure filtering: {len(combined_selected)}")

    reports = build_reports(
        combined_graded,
        reports_dir,
    )

    write_manifest(
        reports_dir
        / "run_manifest.yaml",
        run_id,
        model_config_path,
        filter_config_path,
        production_markets_path,
        model_source,
        input_files,
        settings,
        config_warnings,
        total_rows,
        len(combined_selected),
        len(combined_graded),
    )

    logger.log(
        "--- FINAL SUMMARY ---"
    )

    logger.log(
        f"historical_rows={total_rows}"
    )

    logger.log(
        "selected_bets="
        f"{len(combined_selected)}"
    )

    logger.log(
        "graded_bets="
        f"{len(combined_graded)}"
    )

    if not reports[
        "overall"
    ].empty:
        row = reports[
            "overall"
        ].iloc[0]

        roi = row["roi_units"]

        if pd.notna(roi):
            logger.log(
                "W/L/P/U="
                f"{int(row['wins'])}/"
                f"{int(row['losses'])}/"
                f"{int(row['pushes'])}/"
                f"{int(row['unknown'])} "
                "profit_units="
                f"{float(row['profit_units']):+.4f} "
                "roi_units="
                f"{float(roi):+.4%}"
            )
        else:
            logger.log(
                "W/L/P/U="
                f"{int(row['wins'])}/"
                f"{int(row['losses'])}/"
                f"{int(row['pushes'])}/"
                f"{int(row['unknown'])} "
                "profit_units="
                f"{float(row['profit_units']):+.4f} "
                "roi_units=N/A"
            )

    logger.log(
        f"run_snapshot={run_dir}"
    )

    logger.log(
        "STATUS: SUCCESS"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    shutil.copy2(
        filter_config_path,
        run_dir
        / "markets_test.yaml",
    )

    shutil.copy2(
        production_markets_path,
        run_dir
        / "markets.yaml",
    )

    shutil.copy2(
        model_config_path,
        run_dir
        / "model_config.yaml",
    )
    shutil.copy2(
        DEFAULT_STAKING_CONFIG,
        run_dir
        / "staking.yaml",
    )

    copy_tree_contents(
        reports_dir,
        run_dir
        / "reports",
    )

    copy_tree_contents(
        selections_dir,
        run_dir
        / "selections",
    )

    copy_tree_contents(
        graded_dir,
        run_dir
        / "graded",
    )

    append_run_index(
        runs_dir
        / "index.csv",
        run_id,
        reports["overall"],
    )

    print(
        "basketball_backtest complete."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as main_exc:
        print(
            f"STATUS: FAILED | {main_exc}",
            file=sys.stderr,
        )

        traceback.print_exc()

        sys.exit(1)
