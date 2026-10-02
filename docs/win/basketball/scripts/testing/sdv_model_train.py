#!/usr/bin/env python3
# docs/win/basketball/scripts/testing/sdv_model_train.py
"""Train and evaluate SDV Model V1 without touching production DRatings models."""
from __future__ import annotations

import argparse
import json
import math
import re
import traceback
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from scipy import sparse
from scipy.sparse.linalg import lsqr


BASE = Path("docs/win/basketball")

SDV_CONFIG_PATH = (
    BASE
    / "config/sdv_model.yaml"
)

MODEL_CONFIG_PATH = (
    BASE
    / "config/model_config.yaml"
)

LOG_PATH = (
    BASE
    / "errors/99_validation/"
    "sdv_model_v1_training.txt"
)

LEAGUE_LABELS = {
    "nba": "NBA",
    "ncaam": "NCAAM",
    "wnba": "WNBA",
}

FORBIDDEN_MODEL_INPUTS = {
    "actual_home_points",
    "actual_away_points",
    "actual_margin",
    "actual_total",
    "home_score",
    "away_score",
    "home_winner",
    "away_winner",
    "team_score",
    "opponent_team_score",
    "final_score",
    "result",
}

MISSING_CATEGORY = "__MISSING__"
OTHER_CATEGORY = "__OTHER__"

PROBABILITY_EPSILON = 1e-12


def clean(
    value: Any,
) -> str:
    if value is None:
        return ""

    return str(
        value
    ).strip()


def clean_id(
    value: Any,
) -> str:
    text = clean(
        value
    )

    number = to_float(
        text
    )

    if (
        number is None
        or not number.is_integer()
    ):
        return text

    return str(
        int(
            number
        )
    )

def to_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return float(
            value
        )

    text = clean(
        value
    )

    if not text:
        return None

    try:
        result = float(
            text
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        result
    ):
        return None

    return result


def parse_feature_date(
    value: Any,
) -> date | None:
    text = clean(
        value
    )

    if not text:
        return None

    normalized = (
        text[:10]
        .replace(
            "_",
            "-",
        )
    )

    try:
        return date.fromisoformat(
            normalized
        )

    except ValueError:
        return None


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def log(
    message: str,
) -> None:
    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with LOG_PATH.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            f"{utc_now()} | "
            f"{message}\n"
        )


def read_yaml(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    payload = (
        yaml.safe_load(
            path.read_text(
                encoding="utf-8"
            )
        )
        or {}
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "YAML root must be a mapping: "
            f"{path}"
        )

    return payload


def required_mapping(
    parent: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    value = parent.get(
        key
    )

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            "Missing configuration mapping: "
            f"{key}"
        )

    return value


def validate_sdv_config(
    cfg: dict[str, Any],
) -> None:
    if int(
        cfg.get(
            "schema_version",
            0,
        )
    ) != 1:
        raise ValueError(
            "sdv_model.yaml schema_version "
            "must be 1"
        )

    if not clean(
        cfg.get(
            "feature_version"
        )
    ):
        raise ValueError(
            "sdv_model.yaml feature_version "
            "is blank"
        )

    paths = required_mapping(
        cfg,
        "paths",
    )

    for key in (
        "history_input_root",
        "history_output_root",
    ):
        if not clean(
            paths.get(
                key
            )
        ):
            raise ValueError(
                f"paths.{key} is blank"
            )

    inputs = required_mapping(
        cfg,
        "model_inputs",
    )

    numeric = inputs.get(
        "numeric"
    )

    categorical = inputs.get(
        "categorical"
    )

    if (
        not isinstance(
            numeric,
            list,
        )
        or not numeric
    ):
        raise ValueError(
            "model_inputs.numeric must "
            "be a non-empty list"
        )

    if not isinstance(
        categorical,
        list,
    ):
        raise ValueError(
            "model_inputs.categorical "
            "must be a list"
        )

    configured_inputs = {
        clean(
            value
        )
        for value
        in (
            numeric
            + categorical
        )
    }

    forbidden = sorted(
        configured_inputs
        & FORBIDDEN_MODEL_INPUTS
    )

    if forbidden:
        raise RuntimeError(
            "Target/final-result leakage "
            "detected in model_inputs: "
            f"{forbidden}"
        )

    training = required_mapping(
        cfg,
        "training",
    )

    if clean(
        training.get(
            "model_type"
        )
    ) != "ridge_linear":
        raise ValueError(
            "training.model_type must "
            "be ridge_linear"
        )

    if float(
        training.get(
            "ridge_alpha",
            -1,
        )
    ) < 0:
        raise ValueError(
            "training.ridge_alpha "
            "must be >= 0"
        )

    lockbox = required_mapping(
        training,
        "lockbox",
    )

    if clean(
        lockbox.get(
            "policy"
        )
    ) != "latest_internal_season":
        raise ValueError(
            "training.lockbox.policy must be "
            "latest_internal_season"
        )

    if bool(
        lockbox.get(
            "read_during_training",
            True,
        )
    ):
        raise ValueError(
            "training.lockbox."
            "read_during_training must be false"
        )

    if bool(
        lockbox.get(
            "use_for_residual_distribution",
            True,
        )
    ):
        raise ValueError(
            "training.lockbox."
            "use_for_residual_distribution "
            "must be false"
        )

    oos = required_mapping(
        training,
        "oos",
    )

    if clean(
        oos.get(
            "method"
        )
    ) != "expanding_window":
        raise ValueError(
            "training.oos.method must be "
            "expanding_window"
        )

    folds = int(
        oos.get(
            "folds",
            0,
        )
    )

    if folds <= 0:
        raise ValueError(
            "training.oos.folds "
            "must be positive"
        )

    minimum_fraction = float(
        oos.get(
            "minimum_training_date_fraction",
            0,
        )
    )

    if not (
        0
        < minimum_fraction
        < 1
    ):
        raise ValueError(
            "training.oos."
            "minimum_training_date_fraction "
            "must be between 0 and 1"
        )

    residual = required_mapping(
        training,
        "residual_distribution",
    )

    if clean(
        residual.get(
            "family"
        )
    ) != "normal":
        raise ValueError(
            "training.residual_distribution."
            "family must be normal"
        )

    if clean(
        residual.get(
            "source"
        )
    ) != "expanding_window_oos_only":
        raise ValueError(
            "training.residual_distribution."
            "source must be "
            "expanding_window_oos_only"
        )

    production = required_mapping(
        training,
        "production",
    )

    if bool(
        production.get(
            "write_models",
            True,
        )
    ):
        raise ValueError(
            "training.production."
            "write_models must be false"
        )

    if bool(
        production.get(
            "overwrite_dratings",
            True,
        )
    ):
        raise ValueError(
            "training.production."
            "overwrite_dratings must be false"
        )


def model_input_names(
    cfg: dict[str, Any],
) -> tuple[
    list[str],
    list[str],
]:
    inputs = required_mapping(
        cfg,
        "model_inputs",
    )

    numeric = [
        clean(
            value
        )
        for value
        in inputs[
            "numeric"
        ]
    ]

    categorical = [
        clean(
            value
        )
        for value
        in inputs.get(
            "categorical",
            []
        )
    ]

    return (
        numeric,
        categorical,
    )


def validate_calibration_config(
    cfg: dict[str, Any],
    league: str,
) -> dict[str, Any]:
    leagues = required_mapping(
        cfg,
        "leagues",
    )

    league_cfg = required_mapping(
        leagues,
        league,
    )

    calibration = required_mapping(
        league_cfg,
        "calibration",
    )

    moneyline = required_mapping(
        calibration,
        "moneyline",
    )

    home_ml = required_mapping(
        moneyline,
        "home",
    )

    away_ml = required_mapping(
        moneyline,
        "away",
    )

    if (
        clean(
            home_ml.get(
                "method"
            )
        )
        != "none"
        or clean(
            away_ml.get(
                "method"
            )
        )
        != "none"
    ):
        raise RuntimeError(
            f"{league}: moneyline calibration "
            "would break required complementary "
            "probabilities. HOME and AWAY must "
            "both remain method=none for SDV V1."
        )

    spread = required_mapping(
        calibration,
        "spread",
    )

    if clean(
        spread.get(
            "canonical_side"
        )
    ) != "home":
        raise RuntimeError(
            f"{league}: spread canonical_side "
            "must be home"
        )

    spread_home = required_mapping(
        spread,
        "home",
    )

    if clean(
        spread_home.get(
            "method"
        )
    ) not in {
        "none",
        "beta",
    }:
        raise RuntimeError(
            f"{league}: unsupported spread "
            "calibration method"
        )

    spread_away = spread.get(
        "away"
    )

    if isinstance(
        spread_away,
        dict,
    ) and clean(
        spread_away.get(
            "method"
        )
    ) not in {
        "",
        "none",
    }:
        raise RuntimeError(
            f"{league}: AWAY spread cannot be "
            "calibrated independently"
        )

    total = required_mapping(
        calibration,
        "total",
    )

    if clean(
        total.get(
            "canonical_side"
        )
    ) != "over":
        raise RuntimeError(
            f"{league}: total canonical_side "
            "must be over"
        )

    total_over = required_mapping(
        total,
        "over",
    )

    if clean(
        total_over.get(
            "method"
        )
    ) not in {
        "none",
        "beta",
    }:
        raise RuntimeError(
            f"{league}: unsupported total "
            "calibration method"
        )

    total_under = total.get(
        "under"
    )

    if isinstance(
        total_under,
        dict,
    ) and clean(
        total_under.get(
            "method"
        )
    ) not in {
        "",
        "none",
    }:
        raise RuntimeError(
            f"{league}: UNDER cannot be "
            "calibrated independently"
        )

    return calibration


def discover_feature_seasons(
    root: Path,
    league: str,
) -> list[int]:
    label = LEAGUE_LABELS[
        league
    ]

    league_root = (
        root
        / league
    )

    if not league_root.exists():
        raise FileNotFoundError(
            league_root
        )

    pattern = re.compile(
        rf"^(\d{{4}})_{label}_features\.parquet$"
    )

    seasons: list[int] = []

    for path in league_root.iterdir():
        if not path.is_file():
            continue

        match = pattern.match(
            path.name
        )

        if match:
            seasons.append(
                int(
                    match.group(
                        1
                    )
                )
            )

    seasons = sorted(
        set(
            seasons
        )
    )

    if len(
        seasons
    ) < 2:
        raise RuntimeError(
            f"{league}: at least two historical "
            "feature seasons are required"
        )

    return seasons


def load_season_rows(
    cfg: dict[str, Any],
    league: str,
    season: int,
) -> list[
    dict[str, Any]
]:
    paths = required_mapping(
        cfg,
        "paths",
    )

    history_input_root = Path(
        clean(
            paths[
                "history_input_root"
            ]
        )
    )

    history_feature_root = Path(
        clean(
            paths[
                "history_output_root"
            ]
        )
    )

    label = LEAGUE_LABELS[
        league
    ]

    feature_path = (
        history_feature_root
        / league
        / (
            f"{season}_"
            f"{label}_features.parquet"
        )
    )

    games_path = (
        history_input_root
        / league
        / str(
            season
        )
        / "games.parquet"
    )

    if not feature_path.exists():
        raise FileNotFoundError(
            feature_path
        )

    if not games_path.exists():
        raise FileNotFoundError(
            games_path
        )

    numeric_inputs, categorical_inputs = (
        model_input_names(
            cfg
        )
    )

    required_feature_columns = (
        {
            "game_id",
            "game_date",
            "internal_season",
            "feature_version",
        }
        | set(
            numeric_inputs
        )
        | set(
            categorical_inputs
        )
    )

    feature_frame = pl.read_parquet(
        feature_path
    )

    missing = sorted(
        required_feature_columns
        - set(
            feature_frame.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{feature_path}: missing "
            f"feature columns={missing}"
        )

    advanced_features = cfg.get("advanced_features")
    advanced_features = (
        advanced_features
        if isinstance(advanced_features, dict)
        else {}
    )
    expected_feature_version = (
        clean(advanced_features.get("production_feature_version"))
        or clean(cfg["feature_version"])
    )

    versions = {
        clean(
            value
        )
        for value
        in feature_frame[
            "feature_version"
        ].to_list()
        if clean(
            value
        )
    }

    if versions != {
        expected_feature_version
    }:
        raise RuntimeError(
            f"{feature_path}: feature_version "
            f"expected={expected_feature_version} "
            f"actual={sorted(versions)}"
        )

    games_frame = pl.read_parquet(
        games_path,
        columns=[
            "game_id",
            "home_score",
            "away_score",
        ],
    )

    labels: dict[
        str,
        tuple[
            float,
            float,
        ],
    ] = {}

    for game in games_frame.to_dicts():
        game_id = clean_id(
            game.get(
                "game_id"
            )
        )

        home_points = to_float(
            game.get(
                "home_score"
            )
        )

        away_points = to_float(
            game.get(
                "away_score"
            )
        )

        if (
            not game_id
            or home_points is None
            or away_points is None
        ):
            continue

        labels[
            game_id
        ] = (
            home_points,
            away_points,
        )

    rows: list[
        dict[str, Any]
    ] = []

    for feature_row in feature_frame.to_dicts():
        game_id = clean_id(
            feature_row.get(
                "game_id"
            )
        )

        target_date = parse_feature_date(
            feature_row.get(
                "game_date"
            )
        )

        scores = labels.get(
            game_id
        )

        if (
            not game_id
            or target_date is None
            or scores is None
        ):
            continue

        (
            home_points,
            away_points,
        ) = scores

        row = dict(
            feature_row
        )

        row[
            "game_id"
        ] = game_id

        row[
            "_target_date"
        ] = target_date

        row[
            "_internal_season"
        ] = season

        row[
            "_actual_home_points"
        ] = home_points

        row[
            "_actual_away_points"
        ] = away_points

        row[
            "_target_margin"
        ] = (
            home_points
            - away_points
        )

        row[
            "_target_total"
        ] = (
            home_points
            + away_points
        )

        rows.append(
            row
        )

    if not rows:
        raise RuntimeError(
            f"{league} {season}: no completed "
            "feature/score rows matched"
        )

    rows.sort(
        key=lambda sort_row: (
            sort_row[
                "_target_date"
            ],
            clean(
                sort_row.get(
                    "game_date_time_utc"
                )
            ),
            sort_row[
                "game_id"
            ],
        )
    )

    return rows


@dataclass
class SparseFeatureEncoder:
    numeric_names: list[str]
    categorical_names: list[str]
    numeric_medians: dict[str, float]
    numeric_means: dict[str, float]
    numeric_stds: dict[str, float]
    categorical_levels: dict[
        str,
        list[str],
    ]
    categorical_maps: dict[
        str,
        dict[
            str,
            int,
        ],
    ]
    feature_names: list[str]

    @classmethod
    def fit(
        cls,
        rows: list[
            dict[str, Any]
        ],
        numeric_names: list[str],
        categorical_names: list[str],
    ) -> "SparseFeatureEncoder":
        if not rows:
            raise ValueError(
                "Cannot fit encoder with zero rows"
            )

        numeric_medians: dict[
            str,
            float,
        ] = {}

        numeric_means: dict[
            str,
            float,
        ] = {}

        numeric_stds: dict[
            str,
            float,
        ] = {}

        for name in numeric_names:
            values = np.asarray(
                [
                    (
                        value
                        if (
                            value := to_float(
                                row.get(
                                    name
                                )
                            )
                        )
                        is not None
                        else np.nan
                    )
                    for row
                    in rows
                ],
                dtype=float,
            )

            finite = values[
                np.isfinite(
                    values
                )
            ]

            if finite.size:
                median = float(
                    np.median(
                        finite
                    )
                )
            else:
                median = 0.0

            imputed = np.where(
                np.isfinite(
                    values
                ),
                values,
                median,
            )

            mean_value = float(
                np.mean(
                    imputed
                )
            )

            std_value = float(
                np.std(
                    imputed
                )
            )

            if (
                not math.isfinite(
                    std_value
                )
                or std_value
                < 1e-12
            ):
                std_value = 1.0

            numeric_medians[
                name
            ] = median

            numeric_means[
                name
            ] = mean_value

            numeric_stds[
                name
            ] = std_value

        categorical_levels: dict[
            str,
            list[str],
        ] = {}

        categorical_maps: dict[
            str,
            dict[
                str,
                int,
            ],
        ] = {}

        feature_names = [
            "intercept",
            *numeric_names,
        ]

        offset = (
            1
            + len(
                numeric_names
            )
        )

        for name in categorical_names:
            observed = {
                clean(
                    row.get(
                        name
                    )
                )
                or MISSING_CATEGORY
                for row
                in rows
            }

            observed.discard(
                OTHER_CATEGORY
            )

            levels = sorted(
                observed
            )

            levels.append(
                OTHER_CATEGORY
            )

            mapping = {
                level: (
                    offset
                    + index
                )
                for (
                    index,
                    level,
                )
                in enumerate(
                    levels
                )
            }

            categorical_levels[
                name
            ] = levels

            categorical_maps[
                name
            ] = mapping

            feature_names.extend(
                [
                    f"{name}={level}"
                    for level
                    in levels
                ]
            )

            offset += len(
                levels
            )

        return cls(
            numeric_names=list(
                numeric_names
            ),
            categorical_names=list(
                categorical_names
            ),
            numeric_medians=(
                numeric_medians
            ),
            numeric_means=(
                numeric_means
            ),
            numeric_stds=(
                numeric_stds
            ),
            categorical_levels=(
                categorical_levels
            ),
            categorical_maps=(
                categorical_maps
            ),
            feature_names=(
                feature_names
            ),
        )

    def transform(
        self,
        rows: list[
            dict[str, Any]
        ],
    ) -> sparse.csr_matrix:
        n_rows = len(
            rows
        )

        if n_rows == 0:
            return sparse.csr_matrix(
                (
                    0,
                    len(
                        self.feature_names
                    ),
                ),
                dtype=float,
            )

        matrix_rows: list[int] = []
        matrix_columns: list[int] = []
        matrix_values: list[float] = []

        for row_index in range(
            n_rows
        ):
            matrix_rows.append(
                row_index
            )

            matrix_columns.append(
                0
            )

            matrix_values.append(
                1.0
            )

        for (
            numeric_index,
            name,
        ) in enumerate(
            self.numeric_names,
            start=1,
        ):
            median = (
                self.numeric_medians[
                    name
                ]
            )

            mean_value = (
                self.numeric_means[
                    name
                ]
            )

            std_value = (
                self.numeric_stds[
                    name
                ]
            )

            for (
                row_index,
                row,
            ) in enumerate(
                rows
            ):
                raw_value = to_float(
                    row.get(
                        name
                    )
                )

                value = (
                    raw_value
                    if raw_value
                    is not None
                    else median
                )

                standardized = (
                    value
                    - mean_value
                ) / std_value

                if standardized != 0:
                    matrix_rows.append(
                        row_index
                    )

                    matrix_columns.append(
                        numeric_index
                    )

                    matrix_values.append(
                        standardized
                    )

        for name in self.categorical_names:
            mapping = (
                self.categorical_maps[
                    name
                ]
            )

            other_column = mapping[
                OTHER_CATEGORY
            ]

            for (
                row_index,
                row,
            ) in enumerate(
                rows
            ):
                category = (
                    clean(
                        row.get(
                            name
                        )
                    )
                    or MISSING_CATEGORY
                )

                column = mapping.get(
                    category,
                    other_column,
                )

                matrix_rows.append(
                    row_index
                )

                matrix_columns.append(
                    column
                )

                matrix_values.append(
                    1.0
                )

        return sparse.csr_matrix(
            (
                matrix_values,
                (
                    matrix_rows,
                    matrix_columns,
                ),
            ),
            shape=(
                n_rows,
                len(
                    self.feature_names
                ),
            ),
            dtype=float,
        )


def target_array(
    rows: list[
        dict[str, Any]
    ],
    key: str,
) -> np.ndarray:
    values = np.asarray(
        [
            float(
                row[
                    key
                ]
            )
            for row
            in rows
        ],
        dtype=float,
    )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise RuntimeError(
            f"Non-finite target values: {key}"
        )

    return values


def fit_ridge(
    matrix: sparse.csr_matrix,
    target: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if matrix.shape[
        0
    ] != target.shape[
        0
    ]:
        raise ValueError(
            "Ridge matrix/target "
            "row mismatch"
        )

    if matrix.shape[
        0
    ] == 0:
        raise ValueError(
            "Cannot fit ridge with zero rows"
        )

    n_columns = matrix.shape[
        1
    ]

    penalty_values = np.ones(
        n_columns,
        dtype=float,
    )

    penalty_values[
        0
    ] = 0.0

    penalty = sparse.diags(
        np.sqrt(
            alpha
        )
        * penalty_values,
        format="csr",
    )

    augmented_matrix = sparse.vstack(
        [
            matrix,
            penalty,
        ],
        format="csr",
    )

    augmented_target = np.concatenate(
        [
            target,
            np.zeros(
                n_columns,
                dtype=float,
            ),
        ]
    )

    result = lsqr(
        augmented_matrix,
        augmented_target,
        atol=1e-8,
        btol=1e-8,
        iter_lim=max(
            1000,
            n_columns * 5,
        ),
    )

    coefficients = np.asarray(
        result[
            0
        ],
        dtype=float,
    )

    if not np.all(
        np.isfinite(
            coefficients
        )
    ):
        raise RuntimeError(
            "Non-finite ridge coefficients"
        )

    return coefficients


def predict(
    matrix: sparse.csr_matrix,
    coefficients: np.ndarray,
) -> np.ndarray:
    result = matrix @ coefficients

    predictions = np.asarray(
        result,
        dtype=float,
    ).reshape(
        -1
    )

    if not np.all(
        np.isfinite(
            predictions
        )
    ):
        raise RuntimeError(
            "Non-finite model predictions"
        )

    return predictions


def fit_target_pair(
    fit_matrix: sparse.csr_matrix,
    fit_rows: list[dict[str, Any]],
    prediction_matrix: sparse.csr_matrix,
    ridge_alpha: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    margin_target = target_array(
        fit_rows,
        "_target_margin",
    )
    total_target = target_array(
        fit_rows,
        "_target_total",
    )

    margin_coefficients = fit_ridge(
        fit_matrix,
        margin_target,
        ridge_alpha,
    )
    total_coefficients = fit_ridge(
        fit_matrix,
        total_target,
        ridge_alpha,
    )

    return (
        margin_target,
        total_target,
        margin_coefficients,
        total_coefficients,
        predict(
            prediction_matrix,
            margin_coefficients,
        ),
        predict(
            prediction_matrix,
            total_coefficients,
        ),
    )

def expanding_date_folds(
    rows: list[
        dict[str, Any]
    ],
    folds: int,
    minimum_training_fraction: float,
) -> list[
    tuple[
        list[
            dict[str, Any]
        ],
        list[
            dict[str, Any]
        ],
    ]
]:
    unique_dates = sorted(
        {
            row[
                "_target_date"
            ]
            for row
            in rows
        }
    )

    if len(
        unique_dates
    ) < 4:
        raise RuntimeError(
            "Not enough unique dates "
            "for expanding OOS folds"
        )

    first_validation_index = max(
        1,
        int(
            math.ceil(
                len(
                    unique_dates
                )
                * minimum_training_fraction
            )
        ),
    )

    if (
        first_validation_index
        >= len(
            unique_dates
        )
    ):
        raise RuntimeError(
            "OOS minimum training fraction "
            "leaves no validation dates"
        )

    validation_dates = unique_dates[
        first_validation_index:
    ]

    effective_folds = min(
        folds,
        len(
            validation_dates
        ),
    )

    date_chunks = np.array_split(
        np.asarray(
            validation_dates,
            dtype=object,
        ),
        effective_folds,
    )

    result: list[
        tuple[
            list[
                dict[str, Any]
            ],
            list[
                dict[str, Any]
            ],
        ]
    ] = []

    for chunk in date_chunks:
        if len(
            chunk
        ) == 0:
            continue

        chunk_dates = set(
            chunk.tolist()
        )

        validation_start = min(
            chunk_dates
        )

        train_rows = [
            row
            for row
            in rows
            if row[
                "_target_date"
            ]
            < validation_start
        ]

        validation_rows = [
            row
            for row
            in rows
            if row[
                "_target_date"
            ]
            in chunk_dates
        ]

        if (
            not train_rows
            or not validation_rows
        ):
            continue

        if max(
            row[
                "_target_date"
            ]
            for row
            in train_rows
        ) >= min(
            row[
                "_target_date"
            ]
            for row
            in validation_rows
        ):
            raise RuntimeError(
                "Chronological OOS split failure"
            )

        result.append(
            (
                train_rows,
                validation_rows,
            )
        )

    if not result:
        raise RuntimeError(
            "No valid expanding OOS folds "
            "were created"
        )

    return result


def residual_statistics(
    residuals: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(
        residuals,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size < 2:
        raise RuntimeError(
            "At least two OOS residuals "
            "are required"
        )

    std = float(
        np.std(
            values,
            ddof=1,
        )
    )

    if (
        not math.isfinite(
            std
        )
        or std <= 0
    ):
        raise RuntimeError(
            "Residual standard deviation "
            "must be positive"
        )

    return {
        "n": int(
            values.size
        ),
        "mean": float(
            np.mean(
                values
            )
        ),
        "std": std,
        "mae": float(
            np.mean(
                np.abs(
                    values
                )
            )
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    np.square(
                        values
                    )
                )
            )
        ),
        "p05": float(
            np.quantile(
                values,
                0.05,
            )
        ),
        "p50": float(
            np.quantile(
                values,
                0.50,
            )
        ),
        "p95": float(
            np.quantile(
                values,
                0.95,
            )
        ),
    }


def normal_cdf(
    value: float,
) -> float:
    probability = (
        0.5
        * (
            1.0
            + math.erf(
                value
                / math.sqrt(
                    2.0
                )
            )
        )
    )

    return min(
        max(
            probability,
            PROBABILITY_EPSILON,
        ),
        1.0
        - PROBABILITY_EPSILON,
    )


def beta_calibration(
    probability: float,
    config: dict[str, Any],
) -> float:
    method = clean(
        config.get(
            "method"
        )
    )

    if method == "none":
        return probability

    if method != "beta":
        raise ValueError(
            "Unsupported calibration method: "
            f"{method}"
        )

    p = min(
        max(
            probability,
            PROBABILITY_EPSILON,
        ),
        1.0
        - PROBABILITY_EPSILON,
    )

    intercept = float(
        config[
            "intercept"
        ]
    )

    coef_log_p = float(
        config[
            "coef_log_p"
        ]
    )

    coef_log_1mp = float(
        config[
            "coef_log_1mp"
        ]
    )

    linear = (
        intercept
        + (
            coef_log_p
            * math.log(
                p
            )
        )
        + (
            coef_log_1mp
            * math.log(
                1.0
                - p
            )
        )
    )

    if linear >= 0:
        calibrated = (
            1.0
            / (
                1.0
                + math.exp(
                    -linear
                )
            )
        )

    else:
        exp_value = math.exp(
            linear
        )

        calibrated = (
            exp_value
            / (
                1.0
                + exp_value
            )
        )

    return min(
        max(
            calibrated,
            PROBABILITY_EPSILON,
        ),
        1.0
        - PROBABILITY_EPSILON,
    )


def derive_probabilities(
    expected_margin: float,
    expected_total: float,
    margin_residual: dict[str, Any],
    total_residual: dict[str, Any],
    calibration: dict[str, Any],
    home_spread_line: float | None = None,
    total_line: float | None = None,
) -> dict[str, float]:
    margin_mean = float(
        margin_residual[
            "mean"
        ]
    )

    margin_std = float(
        margin_residual[
            "std"
        ]
    )

    total_mean = float(
        total_residual[
            "mean"
        ]
    )

    total_std = float(
        total_residual[
            "std"
        ]
    )

    adjusted_margin = (
        expected_margin
        + margin_mean
    )

    adjusted_total = (
        expected_total
        + total_mean
    )

    raw_home_ml = normal_cdf(
        adjusted_margin
        / margin_std
    )

    home_ml_prob = raw_home_ml
    away_ml_prob = (
        1.0
        - home_ml_prob
    )

    result = {
        "home_ml_prob": (
            home_ml_prob
        ),
        "away_ml_prob": (
            away_ml_prob
        ),
    }

    if home_spread_line is not None:
        raw_home_spread = normal_cdf(
            (
                adjusted_margin
                + home_spread_line
            )
            / margin_std
        )

        spread_cfg = required_mapping(
            calibration,
            "spread",
        )

        home_cfg = required_mapping(
            spread_cfg,
            "home",
        )

        home_spread_prob = (
            beta_calibration(
                raw_home_spread,
                home_cfg,
            )
        )

        away_spread_prob = (
            1.0
            - home_spread_prob
        )

        result[
            "home_spread_prob"
        ] = home_spread_prob

        result[
            "away_spread_prob"
        ] = away_spread_prob

    if total_line is not None:
        raw_over = normal_cdf(
            (
                adjusted_total
                - total_line
            )
            / total_std
        )

        total_cfg = required_mapping(
            calibration,
            "total",
        )

        over_cfg = required_mapping(
            total_cfg,
            "over",
        )

        over_prob = beta_calibration(
            raw_over,
            over_cfg,
        )

        under_prob = (
            1.0
            - over_prob
        )

        result[
            "over_prob"
        ] = over_prob

        result[
            "under_prob"
        ] = under_prob

    return result


def validate_probability_engine(
    margin_residual: dict[str, Any],
    total_residual: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    maximum_error = 0.0
    tests = 0

    for expected_margin in (
        -20.0,
        -7.0,
        0.0,
        7.0,
        20.0,
    ):
        for expected_total in (
            120.0,
            160.0,
            200.0,
        ):
            for spread_line in (
                -10.5,
                -3.5,
                0.0,
                3.5,
                10.5,
            ):
                for total_line in (
                    120.5,
                    160.5,
                    200.5,
                ):
                    probabilities = (
                        derive_probabilities(
                            expected_margin,
                            expected_total,
                            margin_residual,
                            total_residual,
                            calibration,
                            home_spread_line=(
                                spread_line
                            ),
                            total_line=(
                                total_line
                            ),
                        )
                    )

                    errors = [
                        abs(
                            (
                                probabilities[
                                    "home_ml_prob"
                                ]
                                + probabilities[
                                    "away_ml_prob"
                                ]
                            )
                            - 1.0
                        ),
                        abs(
                            (
                                probabilities[
                                    "home_spread_prob"
                                ]
                                + probabilities[
                                    "away_spread_prob"
                                ]
                            )
                            - 1.0
                        ),
                        abs(
                            (
                                probabilities[
                                    "over_prob"
                                ]
                                + probabilities[
                                    "under_prob"
                                ]
                            )
                            - 1.0
                        ),
                    ]

                    maximum_error = max(
                        maximum_error,
                        *errors,
                    )

                    tests += 1

    if maximum_error > 1e-12:
        raise RuntimeError(
            "Complementary probability "
            "validation failed: "
            f"max_error={maximum_error}"
        )

    return {
        "tests": tests,
        "maximum_complement_error": (
            maximum_error
        ),
        "home_ml_plus_away_ml": (
            "exact complement"
        ),
        "home_spread_plus_away_spread": (
            "exact complement"
        ),
        "over_plus_under": (
            "exact complement"
        ),
    }


def run_oos(
    rows: list[
        dict[str, Any]
    ],
    numeric_inputs: list[str],
    categorical_inputs: list[str],
    ridge_alpha: float,
    folds: int,
    minimum_training_fraction: float,
) -> list[
    dict[str, Any]
]:
    splits = expanding_date_folds(
        rows,
        folds,
        minimum_training_fraction,
    )

    predictions: list[
        dict[str, Any]
    ] = []

    seen_game_ids: set[
        str
    ] = set()

    for (
        fold_index,
        (
            train_rows,
            validation_rows,
        ),
    ) in enumerate(
        splits,
        start=1,
    ):
        train_ids = {
            row[
                "game_id"
            ]
            for row
            in train_rows
        }

        validation_ids = {
            row[
                "game_id"
            ]
            for row
            in validation_rows
        }

        overlap = (
            train_ids
            & validation_ids
        )

        if overlap:
            raise RuntimeError(
                "OOS game leakage: "
                f"{sorted(overlap)[:10]}"
            )

        encoder = SparseFeatureEncoder.fit(
            train_rows,
            numeric_inputs,
            categorical_inputs,
        )

        train_matrix = encoder.transform(
            train_rows
        )

        validation_matrix = (
            encoder.transform(
                validation_rows
            )
        )

        (
            _,
            _,
            _,
            _,
            predicted_margin,
            predicted_total,
        ) = fit_target_pair(
            train_matrix,
            train_rows,
            validation_matrix,
            ridge_alpha,
        )

        for (
            row_index,
            row,
        ) in enumerate(
            validation_rows
        ):
            game_id = row[
                "game_id"
            ]

            if game_id in seen_game_ids:
                raise RuntimeError(
                    "Duplicate OOS prediction "
                    f"for game_id={game_id}"
                )

            seen_game_ids.add(
                game_id
            )

            actual_margin = float(
                row[
                    "_target_margin"
                ]
            )

            actual_total = float(
                row[
                    "_target_total"
                ]
            )

            margin_prediction = float(
                predicted_margin[
                    row_index
                ]
            )

            total_prediction = float(
                predicted_total[
                    row_index
                ]
            )

            predictions.append(
                {
                    "fold": fold_index,
                    "game_id": game_id,
                    "game_date": (
                        row[
                            "_target_date"
                        ].isoformat()
                    ),
                    "internal_season": int(
                        row[
                            "_internal_season"
                        ]
                    ),
                    "actual_home_points": float(
                        row[
                            "_actual_home_points"
                        ]
                    ),
                    "actual_away_points": float(
                        row[
                            "_actual_away_points"
                        ]
                    ),
                    "actual_margin": (
                        actual_margin
                    ),
                    "expected_margin": (
                        margin_prediction
                    ),
                    "margin_residual": (
                        actual_margin
                        - margin_prediction
                    ),
                    "actual_total": (
                        actual_total
                    ),
                    "expected_total": (
                        total_prediction
                    ),
                    "total_residual": (
                        actual_total
                        - total_prediction
                    ),
                }
            )

    predictions.sort(
        key=lambda row: (
            row[
                "game_date"
            ],
            row[
                "game_id"
            ],
        )
    )

    if not predictions:
        raise RuntimeError(
            "OOS prediction set is empty"
        )

    return predictions


def training_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    residual = (
        actual
        - predicted
    )

    return {
        "mae": float(
            np.mean(
                np.abs(
                    residual
                )
            )
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    np.square(
                        residual
                    )
                )
            )
        ),
        "mean_residual": float(
            np.mean(
                residual
            )
        ),
    }


def coefficient_rows(
    encoder: SparseFeatureEncoder,
    margin_coefficients: np.ndarray,
    total_coefficients: np.ndarray,
) -> list[
    dict[str, Any]
]:
    rows: list[
        dict[str, Any]
    ] = []

    for (
        index,
        feature_name,
    ) in enumerate(
        encoder.feature_names
    ):
        rows.append(
            {
                "target": "margin",
                "feature": feature_name,
                "coefficient": float(
                    margin_coefficients[
                        index
                    ]
                ),
            }
        )

        rows.append(
            {
                "target": "total",
                "feature": feature_name,
                "coefficient": float(
                    total_coefficients[
                        index
                    ]
                ),
            }
        )

    return rows


def validate_ncaam_venue_model(
    rows: list[
        dict[str, Any]
    ],
    numeric_inputs: list[str],
    categorical_inputs: list[str],
    encoder: SparseFeatureEncoder,
    margin_coefficients: np.ndarray,
    total_coefficients: np.ndarray,
) -> dict[str, Any]:
    required_numeric = {
        "is_neutral_site",
        "home_court_indicator",
    }

    missing_numeric = sorted(
        required_numeric
        - set(
            numeric_inputs
        )
    )

    if missing_numeric:
        raise RuntimeError(
            "NCAAM venue validation failed: "
            "required numeric model inputs "
            f"missing={missing_numeric}"
        )

    if "venue_id" not in categorical_inputs:
        raise RuntimeError(
            "NCAAM venue validation failed: "
            "venue_id is not a categorical "
            "model input"
        )

    neutral_values = {
        to_float(
            row.get(
                "is_neutral_site"
            )
        )
        for row
        in rows
    }

    neutral_values.discard(
        None
    )

    home_court_values = {
        to_float(
            row.get(
                "home_court_indicator"
            )
        )
        for row
        in rows
    }

    home_court_values.discard(
        None
    )

    venue_values = {
        clean(
            row.get(
                "venue_id"
            )
        )
        for row
        in rows
        if clean(
            row.get(
                "venue_id"
            )
        )
    }

    if len(
        neutral_values
    ) < 2:
        raise RuntimeError(
            "NCAAM venue validation failed: "
            "is_neutral_site has insufficient "
            "training variation"
        )

    if len(
        home_court_values
    ) < 2:
        raise RuntimeError(
            "NCAAM venue validation failed: "
            "home_court_indicator has "
            "insufficient training variation"
        )

    if len(
        venue_values
    ) < 2:
        raise RuntimeError(
            "NCAAM venue validation failed: "
            "venue_id has insufficient "
            "training variation"
        )

    feature_index = {
        name: index
        for (
            index,
            name,
        )
        in enumerate(
            encoder.feature_names
        )
    }

    neutral_index = feature_index[
        "is_neutral_site"
    ]

    home_court_index = feature_index[
        "home_court_indicator"
    ]

    venue_indexes = [
        index
        for (
            index,
            feature_name,
        )
        in enumerate(
            encoder.feature_names
        )
        if feature_name.startswith(
            "venue_id="
        )
    ]

    if not venue_indexes:
        raise RuntimeError(
            "NCAAM venue validation failed: "
            "no learned venue_id columns"
        )

    venue_margin_norm = float(
        np.linalg.norm(
            margin_coefficients[
                venue_indexes
            ]
        )
    )

    venue_total_norm = float(
        np.linalg.norm(
            total_coefficients[
                venue_indexes
            ]
        )
    )

    return {
        "status": "PASS",
        "is_neutral_site_is_model_input": (
            True
        ),
        "home_court_indicator_is_model_input": (
            True
        ),
        "venue_id_is_model_input": True,
        "is_neutral_site_training_values": (
            sorted(
                float(
                    value
                )
                for value
                in neutral_values
            )
        ),
        "home_court_indicator_training_values": (
            sorted(
                float(
                    value
                )
                for value
                in home_court_values
            )
        ),
        "venue_id_training_levels": len(
            venue_values
        ),
        "learned_venue_design_columns": len(
            venue_indexes
        ),
        "margin_is_neutral_site_coefficient": (
            float(
                margin_coefficients[
                    neutral_index
                ]
            )
        ),
        "margin_home_court_indicator_coefficient": (
            float(
                margin_coefficients[
                    home_court_index
                ]
            )
        ),
        "total_is_neutral_site_coefficient": (
            float(
                total_coefficients[
                    neutral_index
                ]
            )
        ),
        "total_home_court_indicator_coefficient": (
            float(
                total_coefficients[
                    home_court_index
                ]
            )
        ),
        "margin_venue_coefficient_l2_norm": (
            venue_margin_norm
        ),
        "total_venue_coefficient_l2_norm": (
            venue_total_norm
        ),
        "effects_estimated_from_training_data": (
            True
        ),
        "manual_fixed_home_court_points": (
            False
        ),
        "fixed_home_court_adjustment_applied": (
            False
        ),
    }


def write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        f"{path}.tmp"
    )

    try:
        tmp.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        tmp.replace(
            path
        )

    finally:
        if tmp.exists():
            tmp.unlink()


def write_parquet_atomic(
    path: Path,
    rows: list[
        dict[str, Any]
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        f"{path}.tmp"
    )

    try:
        pl.DataFrame(
            rows,
            infer_schema_length=None,
            strict=False,
        ).write_parquet(
            tmp,
            compression="zstd",
        )

        tmp.replace(
            path
        )

    finally:
        if tmp.exists():
            tmp.unlink()


def write_csv_atomic(
    path: Path,
    rows: list[
        dict[str, Any]
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        f"{path}.tmp"
    )

    try:
        pl.DataFrame(
            rows,
            infer_schema_length=None,
            strict=False,
        ).write_csv(
            tmp
        )

        tmp.replace(
            path
        )

    finally:
        if tmp.exists():
            tmp.unlink()


def train_league(
    sdv_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    league: str,
) -> dict[str, Any]:
    label = LEAGUE_LABELS[
        league
    ]

    paths = required_mapping(
        sdv_cfg,
        "paths",
    )

    feature_root = Path(
        clean(
            paths[
                "history_output_root"
            ]
        )
    )

    training_cfg = required_mapping(
        sdv_cfg,
        "training",
    )

    oos_cfg = required_mapping(
        training_cfg,
        "oos",
    )

    evaluation_root = Path(
        clean(
            training_cfg[
                "evaluation_output_root"
            ]
        )
    )

    output_root = (
        evaluation_root
        / league
    )

    seasons = discover_feature_seasons(
        feature_root,
        league,
    )

    lockbox_season = max(
        seasons
    )

    development_seasons = [
        season
        for season
        in seasons
        if season
        < lockbox_season
    ]

    if not development_seasons:
        raise RuntimeError(
            f"{league}: no development seasons "
            "remain before lockbox "
            f"{lockbox_season}"
        )

    log(
        f"{label} | LOCKBOX RESERVED | "
        f"season={lockbox_season} | "
        "file_not_read=true"
    )

    development_rows: list[
        dict[str, Any]
    ] = []

    season_counts: dict[
        str,
        int,
    ] = {}

    for season in development_seasons:
        season_rows = load_season_rows(
            sdv_cfg,
            league,
            season,
        )

        development_rows.extend(
            season_rows
        )

        season_counts[
            str(
                season
            )
        ] = len(
            season_rows
        )

    development_rows.sort(
        key=lambda row: (
            row[
                "_target_date"
            ],
            clean(
                row.get(
                    "game_date_time_utc"
                )
            ),
            row[
                "game_id"
            ],
        )
    )

    game_ids = [
        row[
            "game_id"
        ]
        for row
        in development_rows
    ]

    if len(
        game_ids
    ) != len(
        set(
            game_ids
        )
    ):
        raise RuntimeError(
            f"{league}: duplicate game_id "
            "across development seasons"
        )

    numeric_inputs, categorical_inputs = (
        model_input_names(
            sdv_cfg
        )
    )

    calibration = (
        validate_calibration_config(
            model_cfg,
            league,
        )
    )

    ridge_alpha = float(
        training_cfg[
            "ridge_alpha"
        ]
    )

    oos_predictions = run_oos(
        development_rows,
        numeric_inputs,
        categorical_inputs,
        ridge_alpha,
        int(
            oos_cfg[
                "folds"
            ]
        ),
        float(
            oos_cfg[
                "minimum_training_date_fraction"
            ]
        ),
    )

    margin_oos_residuals = np.asarray(
        [
            float(
                row[
                    "margin_residual"
                ]
            )
            for row
            in oos_predictions
        ],
        dtype=float,
    )

    total_oos_residuals = np.asarray(
        [
            float(
                row[
                    "total_residual"
                ]
            )
            for row
            in oos_predictions
        ],
        dtype=float,
    )

    margin_residual_stats = (
        residual_statistics(
            margin_oos_residuals
        )
    )

    total_residual_stats = (
        residual_statistics(
            total_oos_residuals
        )
    )

    for row in oos_predictions:
        probabilities = derive_probabilities(
            float(
                row[
                    "expected_margin"
                ]
            ),
            float(
                row[
                    "expected_total"
                ]
            ),
            margin_residual_stats,
            total_residual_stats,
            calibration,
        )

        row.update(
            probabilities
        )

    probability_validation = (
        validate_probability_engine(
            margin_residual_stats,
            total_residual_stats,
            calibration,
        )
    )

    final_encoder = (
        SparseFeatureEncoder.fit(
            development_rows,
            numeric_inputs,
            categorical_inputs,
        )
    )

    final_matrix = (
        final_encoder.transform(
            development_rows
        )
    )

    (
        actual_margin,
        actual_total,
        margin_coefficients,
        total_coefficients,
        development_margin_prediction,
        development_total_prediction,
    ) = fit_target_pair(
        final_matrix,
        development_rows,
        final_matrix,
        ridge_alpha,
    )

    margin_training_metrics = (
        training_metrics(
            actual_margin,
            development_margin_prediction,
        )
    )

    total_training_metrics = (
        training_metrics(
            actual_total,
            development_total_prediction,
        )
    )

    coefficients = coefficient_rows(
        final_encoder,
        margin_coefficients,
        total_coefficients,
    )

    if league == "ncaam":
        venue_validation = (
            validate_ncaam_venue_model(
                development_rows,
                numeric_inputs,
                categorical_inputs,
                final_encoder,
                margin_coefficients,
                total_coefficients,
            )
        )

    else:
        venue_validation = {
            "status": "NOT_APPLICABLE",
            "league": label,
        }

    report = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "league": label,
        "model_version": clean(
            training_cfg[
                "model_version"
            ]
        ),
        "feature_version": clean(
            sdv_cfg[
                "feature_version"
            ]
        ),
        "model_type": "ridge_linear",
        "ridge_alpha": ridge_alpha,
        "targets": {
            "margin": (
                "actual_home_points - "
                "actual_away_points"
            ),
            "total": (
                "actual_home_points + "
                "actual_away_points"
            ),
        },
        "development": {
            "seasons": (
                development_seasons
            ),
            "rows": len(
                development_rows
            ),
            "rows_by_season": (
                season_counts
            ),
        },
        "lockbox": {
            "season": lockbox_season,
            "reserved": True,
            "read_during_training": False,
            "used_for_model_fit": False,
            "used_for_residual_distribution": (
                False
            ),
            "used_for_model_selection": False,
        },
        "models": {
            "margin": {
                "single_model": True,
                "contradictory_home_away_models": (
                    False
                ),
                "development_fit_metrics": (
                    margin_training_metrics
                ),
            },
            "total": {
                "single_model": True,
                "contradictory_over_under_models": (
                    False
                ),
                "development_fit_metrics": (
                    total_training_metrics
                ),
            },
        },
        "oos": {
            "method": "expanding_window",
            "configured_folds": int(
                oos_cfg[
                    "folds"
                ]
            ),
            "minimum_training_date_fraction": (
                float(
                    oos_cfg[
                        "minimum_training_date_fraction"
                    ]
                )
            ),
            "prediction_rows": len(
                oos_predictions
            ),
        },
        "residual_distributions": {
            "source": (
                "expanding_window_oos_only"
            ),
            "lockbox_used": False,
            "family": "normal",
            "margin": (
                margin_residual_stats
            ),
            "total": (
                total_residual_stats
            ),
        },
        "probability_engine": {
            "moneyline": (
                "derived from expected margin "
                "and OOS margin residual "
                "distribution"
            ),
            "spread": (
                "derived from expected margin "
                "+ home spread line and OOS "
                "margin residual distribution"
            ),
            "total": (
                "derived from expected total "
                "+ total line and OOS total "
                "residual distribution"
            ),
            "validation": (
                probability_validation
            ),
        },
        "calibration_compatibility": {
            "status": "PASS",
            "moneyline": (
                "HOME/AWAY calibration methods "
                "are none; raw probabilities "
                "remain complementary"
            ),
            "spread": (
                "HOME is canonical calibrated "
                "side; AWAY is 1-HOME"
            ),
            "total": (
                "OVER is canonical calibrated "
                "side; UNDER is 1-OVER"
            ),
        },
        "design_matrix": {
            "rows": int(
                final_matrix.shape[
                    0
                ]
            ),
            "columns": int(
                final_matrix.shape[
                    1
                ]
            ),
            "numeric_inputs": (
                numeric_inputs
            ),
            "categorical_inputs": (
                categorical_inputs
            ),
        },
        "ncaam_venue_home_court_validation": (
            venue_validation
        ),
        "production_safety": {
            "production_model_written": False,
            "dratings_model_overwritten": False,
            "models_directory_touched": False,
        },
        "status": "PASS",
    }

    report_path = (
        output_root
        / (
            f"{label}_"
            "sdv_model_v1_training_report.json"
        )
    )

    oos_path = (
        output_root
        / (
            f"{label}_"
            "sdv_model_v1_oos_predictions.parquet"
        )
    )

    coefficients_path = (
        output_root
        / (
            f"{label}_"
            "sdv_model_v1_evaluation_"
            "coefficients.csv"
        )
    )

    write_json_atomic(
        report_path,
        report,
    )

    write_parquet_atomic(
        oos_path,
        oos_predictions,
    )

    write_csv_atomic(
        coefficients_path,
        coefficients,
    )

    log(
        f"{label} | PASS | "
        f"development_rows="
        f"{len(development_rows)} | "
        f"oos_rows="
        f"{len(oos_predictions)} | "
        f"lockbox={lockbox_season} | "
        "lockbox_read=false | "
        f"report={report_path}"
    )

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate SDV Model V1 "
            "without touching production models."
        )
    )

    parser.add_argument(
        "--league",
        action="append",
        choices=sorted(
            LEAGUE_LABELS
        ),
        help=(
            "League to train. May be repeated. "
            "Defaults to all leagues."
        ),
    )

    parser.add_argument(
        "--sdv-config",
        type=Path,
        default=SDV_CONFIG_PATH,
    )

    parser.add_argument(
        "--model-config",
        type=Path,
        default=MODEL_CONFIG_PATH,
    )

    return parser


def main() -> int:
    args = (
        build_parser()
        .parse_args()
    )

    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_PATH.write_text(
        (
            "=== SDV MODEL V1 TRAINING "
            f"{utc_now()} ===\n"
        ),
        encoding="utf-8",
    )

    try:
        sdv_cfg = read_yaml(
            args.sdv_config
        )

        model_cfg = read_yaml(
            args.model_config
        )

        validate_sdv_config(
            sdv_cfg
        )

        leagues = (
            args.league
            or list(
                LEAGUE_LABELS
            )
        )

        reports: list[
            dict[str, Any]
        ] = []

        for league in leagues:
            reports.append(
                train_league(
                    sdv_cfg,
                    model_cfg,
                    league,
                )
            )

        failed = [
            report
            for report
            in reports
            if report.get(
                "status"
            )
            != "PASS"
        ]

        if failed:
            raise RuntimeError(
                "One or more league "
                "training reports failed"
            )

        log(
            "STATUS: SUCCESS | "
            f"leagues={len(reports)}"
        )

        print(
            "SDV Model V1 training complete: "
            "SUCCESS. "
            f"leagues={len(reports)}"
        )

        return 0

    except Exception as exc:
        log(
            f"FATAL: {exc}"
        )

        log(
            traceback
            .format_exc()
            .rstrip()
        )

        log(
            "STATUS: FAILED"
        )

        print(
            "SDV Model V1 training "
            f"FAILED: {exc}"
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )