#!/usr/bin/env python3
# docs/win/basketball/scripts/testing/sdv_model_persist.py
"""Persist/version SDV Model V1 inference artifacts without touching DRatings."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import sdv_model_train as trainer


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
    "sdv_model_v1_artifacts.txt"
)

LEAGUE_LABELS = {
    "nba": "NBA",
    "ncaam": "NCAAM",
    "wnba": "WNBA",
}

REQUIRED_ARTIFACT_FILES = (
    "margin_model.json",
    "total_model.json",
    "feature_schema.json",
    "metadata.json",
)


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
            f"{utc_now()} | {message}\n"
        )


def sha256_file(
    path: Path,
) -> str:
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def json_read(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "JSON root must be object: "
            f"{path}"
        )

    return payload


def artifact_config(
    sdv_cfg: dict[str, Any],
) -> dict[str, Any]:
    section = trainer.required_mapping(
        sdv_cfg,
        "artifacts",
    )

    if int(
        section.get(
            "schema_version",
            0,
        )
    ) != 1:
        raise ValueError(
            "artifacts.schema_version "
            "must be 1"
        )

    if not bool(
        section.get(
            "enabled",
            False,
        )
    ):
        raise ValueError(
            "artifacts.enabled must be true"
        )

    root = trainer.clean(
        section.get(
            "root"
        )
    )

    if not root:
        raise ValueError(
            "artifacts.root is blank"
        )

    expected_root = (
        "docs/win/basketball/models/sdv"
    )

    if (
        Path(root).as_posix().rstrip("/")
        != expected_root
    ):
        raise ValueError(
            "artifacts.root must be "
            f"{expected_root}"
        )

    if bool(
        section.get(
            "overwrite_dratings",
            True,
        )
    ):
        raise ValueError(
            "artifacts.overwrite_dratings "
            "must be false"
        )

    if not bool(
        section.get(
            "require_feature_version_match",
            False,
        )
    ):
        raise ValueError(
            "artifacts."
            "require_feature_version_match "
            "must be true"
        )

    if not bool(
        section.get(
            "require_model_version_match",
            False,
        )
    ):
        raise ValueError(
            "artifacts."
            "require_model_version_match "
            "must be true"
        )

    storage_config = trainer.clean(
        section.get(
            "sportsdataverse_config"
        )
    )

    if not storage_config:
        raise ValueError(
            "artifacts."
            "sportsdataverse_config "
            "is blank"
        )

    return section


def sportsdataverse_version(
    artifact_cfg: dict[str, Any],
) -> tuple[
    str,
    str | None,
]:
    storage_path = Path(
        trainer.clean(
            artifact_cfg[
                "sportsdataverse_config"
            ]
        )
    )

    storage_cfg = trainer.read_yaml(
        storage_path
    )

    sdv_section = trainer.required_mapping(
        storage_cfg,
        "sportsdataverse",
    )

    expected = trainer.clean(
        sdv_section.get(
            "expected_version"
        )
    )

    if not expected:
        raise ValueError(
            "sportsdataverse.expected_version "
            "is blank"
        )

    try:
        installed = (
            importlib.metadata.version(
                "sportsdataverse"
            )
        )

    except (
        importlib.metadata.PackageNotFoundError,
        ValueError,
    ):
        installed = None

    if (
        installed is not None
        and installed != expected
    ):
        raise RuntimeError(
            "SportsDataVerse version mismatch: "
            f"configured={expected} "
            f"installed={installed}"
        )

    return (
        expected,
        installed,
    )


def development_rows(
    sdv_cfg: dict[str, Any],
    league: str,
) -> tuple[
    list[int],
    int,
    list[dict[str, Any]],
]:
    paths = trainer.required_mapping(
        sdv_cfg,
        "paths",
    )

    feature_root = Path(
        trainer.clean(
            paths[
                "history_output_root"
            ]
        )
    )

    seasons = (
        trainer.discover_feature_seasons(
            feature_root,
            league,
        )
    )

    lockbox_season = max(
        seasons
    )

    training_seasons = [
        season
        for season
        in seasons
        if season < lockbox_season
    ]

    if not training_seasons:
        raise RuntimeError(
            f"{league}: no training seasons "
            f"before lockbox={lockbox_season}"
        )

    rows: list[
        dict[str, Any]
    ] = []

    for season in training_seasons:
        rows.extend(
            trainer.load_season_rows(
                sdv_cfg,
                league,
                season,
            )
        )

    rows.sort(
        key=lambda row: (
            row[
                "_target_date"
            ],
            trainer.clean(
                row.get(
                    "game_date_time_utc"
                )
            ),
            row[
                "game_id"
            ],
        )
    )

    if not rows:
        raise RuntimeError(
            f"{league}: zero training rows"
        )

    game_ids = [
        row[
            "game_id"
        ]
        for row
        in rows
    ]

    if (
        len(game_ids)
        != len(
            set(
                game_ids
            )
        )
    ):
        raise RuntimeError(
            f"{league}: duplicate game_id "
            "in training rows"
        )

    return (
        training_seasons,
        lockbox_season,
        rows,
    )


def fit_artifact_model(
    sdv_cfg: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[
    trainer.SparseFeatureEncoder,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    dict[str, Any],
    list[str],
    list[str],
]:
    training_cfg = trainer.required_mapping(
        sdv_cfg,
        "training",
    )

    oos_cfg = trainer.required_mapping(
        training_cfg,
        "oos",
    )

    (
        numeric_inputs,
        categorical_inputs,
    ) = trainer.model_input_names(
        sdv_cfg
    )

    ridge_alpha = float(
        training_cfg[
            "ridge_alpha"
        ]
    )

    oos_predictions = trainer.run_oos(
        rows,
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

    margin_residuals = np.asarray(
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

    total_residuals = np.asarray(
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
        trainer.residual_statistics(
            margin_residuals
        )
    )

    total_residual_stats = (
        trainer.residual_statistics(
            total_residuals
        )
    )

    encoder = (
        trainer.SparseFeatureEncoder.fit(
            rows,
            numeric_inputs,
            categorical_inputs,
        )
    )

    matrix = encoder.transform(
        rows
    )

    actual_margin = trainer.target_array(
        rows,
        "_target_margin",
    )

    actual_total = trainer.target_array(
        rows,
        "_target_total",
    )

    margin_coefficients = (
        trainer.fit_ridge(
            matrix,
            actual_margin,
            ridge_alpha,
        )
    )

    total_coefficients = (
        trainer.fit_ridge(
            matrix,
            actual_total,
            ridge_alpha,
        )
    )

    if (
        len(
            margin_coefficients
        )
        != len(
            encoder.feature_names
        )
    ):
        raise RuntimeError(
            "Margin coefficient count "
            "does not match encoded schema"
        )

    if (
        len(
            total_coefficients
        )
        != len(
            encoder.feature_names
        )
    ):
        raise RuntimeError(
            "Total coefficient count "
            "does not match encoded schema"
        )

    return (
        encoder,
        margin_coefficients,
        total_coefficients,
        margin_residual_stats,
        total_residual_stats,
        numeric_inputs,
        categorical_inputs,
    )


def encoder_payload(
    encoder: trainer.SparseFeatureEncoder,
) -> dict[str, Any]:
    numeric_scaling: list[
        dict[str, Any]
    ] = []

    for (
        position,
        name,
    ) in enumerate(
        encoder.numeric_names
    ):
        median = float(
            encoder.numeric_medians[
                name
            ]
        )

        mean_value = float(
            encoder.numeric_means[
                name
            ]
        )

        std_value = float(
            encoder.numeric_stds[
                name
            ]
        )

        if not all(
            math.isfinite(value)
            for value
            in (
                median,
                mean_value,
                std_value,
            )
        ):
            raise RuntimeError(
                "Non-finite numeric scaling "
                f"for {name}"
            )

        if std_value <= 0:
            raise RuntimeError(
                "Invalid numeric std "
                f"for {name}: {std_value}"
            )

        numeric_scaling.append(
            {
                "position": position,
                "name": name,
                "missing_fill_value": median,
                "mean": mean_value,
                "std": std_value,
                "transform": (
                    "(value - mean) / std"
                ),
            }
        )

    categorical_encoding: list[
        dict[str, Any]
    ] = []

    for (
        position,
        name,
    ) in enumerate(
        encoder.categorical_names
    ):
        levels = list(
            encoder.categorical_levels[
                name
            ]
        )

        mapping = {
            level: int(
                column
            )
            for (
                level,
                column,
            ) in encoder.categorical_maps[
                name
            ].items()
        }

        categorical_encoding.append(
            {
                "position": position,
                "name": name,
                "missing_token": (
                    trainer.MISSING_CATEGORY
                ),
                "unknown_token": (
                    trainer.OTHER_CATEGORY
                ),
                "levels": levels,
                "encoded_index_by_level": (
                    mapping
                ),
            }
        )

    return {
        "intercept": {
            "encoded_index": 0,
            "value": 1.0,
            "ridge_penalized": False,
        },
        "numeric_feature_order": list(
            encoder.numeric_names
        ),
        "categorical_feature_order": list(
            encoder.categorical_names
        ),
        "numeric_scaling": (
            numeric_scaling
        ),
        "categorical_encoding": (
            categorical_encoding
        ),
        "encoded_feature_order": list(
            encoder.feature_names
        ),
        "encoded_feature_count": len(
            encoder.feature_names
        ),
    }


def coefficient_payload(
    encoder: trainer.SparseFeatureEncoder,
    coefficients: np.ndarray,
) -> list[dict[str, Any]]:
    result: list[
        dict[str, Any]
    ] = []

    for (
        position,
        name,
    ) in enumerate(
        encoder.feature_names
    ):
        value = float(
            coefficients[
                position
            ]
        )

        if not math.isfinite(
            value
        ):
            raise RuntimeError(
                "Non-finite coefficient "
                f"position={position} "
                f"feature={name}"
            )

        result.append(
            {
                "position": position,
                "feature": name,
                "value": value,
            }
        )

    return result


def feature_schema_payload(
    league: str,
    model_version: str,
    feature_version: str,
    encoder: trainer.SparseFeatureEncoder,
) -> dict[str, Any]:
    raw_features: list[
        dict[str, Any]
    ] = []

    position = 0

    for name in encoder.numeric_names:
        raw_features.append(
            {
                "position": position,
                "name": name,
                "kind": "numeric",
                "expected_dtype": "float64",
                "missing_policy": (
                    "impute_training_median"
                ),
            }
        )

        position += 1

    for name in encoder.categorical_names:
        raw_features.append(
            {
                "position": position,
                "name": name,
                "kind": "categorical",
                "expected_dtype": "utf8",
                "missing_policy": (
                    trainer.MISSING_CATEGORY
                ),
                "unknown_policy": (
                    trainer.OTHER_CATEGORY
                ),
            }
        )

        position += 1

    encoded_features = [
        {
            "position": position,
            "name": name,
            "expected_dtype": "float64",
        }
        for (
            position,
            name,
        )
        in enumerate(
            encoder.feature_names
        )
    ]

    return {
        "schema_version": 1,
        "artifact_type": "feature_schema",
        "league": LEAGUE_LABELS[
            league
        ],
        "model_version": model_version,
        "feature_version": feature_version,
        "raw_feature_count": len(
            raw_features
        ),
        "raw_feature_order": [
            item[
                "name"
            ]
            for item
            in raw_features
        ],
        "raw_features": raw_features,
        "encoded_feature_count": len(
            encoded_features
        ),
        "encoded_feature_order": list(
            encoder.feature_names
        ),
        "encoded_features": (
            encoded_features
        ),
    }


def model_payload(
    *,
    league: str,
    model_version: str,
    feature_version: str,
    target_name: str,
    target_definition: str,
    ridge_alpha: float,
    encoder: trainer.SparseFeatureEncoder,
    coefficients: np.ndarray,
    residual_stats: dict[str, Any],
    probability_definition: dict[str, Any],
    created_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": (
            f"{target_name}_model"
        ),
        "league": LEAGUE_LABELS[
            league
        ],
        "model_version": model_version,
        "feature_version": feature_version,
        "created_at_utc": created_at,
        "model_type": "ridge_linear",
        "target_name": target_name,
        "target_definition": (
            target_definition
        ),
        "ridge_alpha": ridge_alpha,
        "coefficient_count": len(
            coefficients
        ),
        "encoder": encoder_payload(
            encoder
        ),
        "coefficients": (
            coefficient_payload(
                encoder,
                coefficients,
            )
        ),
        "residual_distribution": {
            "family": "normal",
            "source": (
                "expanding_window_oos_only"
            ),
            **residual_stats,
        },
        "probability_definition": (
            probability_definition
        ),
    }


def reproducibility_id(
    *,
    league: str,
    training_seasons: list[int],
    model_version: str,
    feature_version: str,
    training_script_sha: str,
    persistence_script_sha: str,
    sdv_config_sha: str,
    model_config_sha: str,
    storage_config_sha: str,
) -> str:
    payload = {
        "league": league,
        "training_seasons": (
            training_seasons
        ),
        "model_version": model_version,
        "feature_version": (
            feature_version
        ),
        "training_script_sha256": (
            training_script_sha
        ),
        "persistence_script_sha256": (
            persistence_script_sha
        ),
        "sdv_config_sha256": (
            sdv_config_sha
        ),
        "model_config_sha256": (
            model_config_sha
        ),
        "storage_config_sha256": (
            storage_config_sha
        ),
    }

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )

    return sha256_text(
        canonical
    )


def metadata_payload(
    *,
    league: str,
    model_version: str,
    feature_version: str,
    sportsdataverse_expected: str,
    sportsdataverse_installed: str | None,
    training_seasons: list[int],
    lockbox_season: int,
    rows: list[dict[str, Any]],
    target_definitions: dict[str, str],
    training_script_path: Path,
    persistence_script_path: Path,
    storage_config_path: Path,
    sdv_config_path: Path,
    model_config_path: Path,
    created_at: str,
) -> dict[str, Any]:
    first_date = min(
        row[
            "_target_date"
        ]
        for row
        in rows
    )

    last_date = max(
        row[
            "_target_date"
        ]
        for row
        in rows
    )

    training_script_sha = (
        sha256_file(
            training_script_path
        )
    )

    persistence_script_sha = (
        sha256_file(
            persistence_script_path
        )
    )

    sdv_config_sha = (
        sha256_file(
            sdv_config_path
        )
    )

    model_config_sha = (
        sha256_file(
            model_config_path
        )
    )

    storage_config_sha = (
        sha256_file(
            storage_config_path
        )
    )

    repro_id = reproducibility_id(
        league=league,
        training_seasons=training_seasons,
        model_version=model_version,
        feature_version=feature_version,
        training_script_sha=(
            training_script_sha
        ),
        persistence_script_sha=(
            persistence_script_sha
        ),
        sdv_config_sha=(
            sdv_config_sha
        ),
        model_config_sha=(
            model_config_sha
        ),
        storage_config_sha=(
            storage_config_sha
        ),
    )

    return {
        "schema_version": 1,
        "artifact_type": "metadata",
        "league": LEAGUE_LABELS[
            league
        ],
        "model_version": model_version,
        "sportsdataverse_version": (
            sportsdataverse_expected
        ),
        "sportsdataverse_installed_version": (
            sportsdataverse_installed
        ),
        "feature_version": feature_version,
        "training_leagues": [
            LEAGUE_LABELS[
                league
            ]
        ],
        "training_internal_seasons": (
            training_seasons
        ),
        "reserved_lockbox_internal_season": (
            lockbox_season
        ),
        "lockbox_read_during_fit": False,
        "first_training_game_date": (
            first_date.isoformat()
        ),
        "last_training_game_date": (
            last_date.isoformat()
        ),
        "training_row_count": len(
            rows
        ),
        "target_definition": (
            target_definitions
        ),
        "training_script": {
            "path": (
                "docs/win/basketball/"
                "scripts/testing/"
                "sdv_model_train.py"
            ),
            "sha256": (
                training_script_sha
            ),
        },
        "artifact_persistence_script": {
            "path": (
                "docs/win/basketball/"
                "scripts/testing/"
                "sdv_model_persist.py"
            ),
            "sha256": (
                persistence_script_sha
            ),
        },
        "configuration_hashes": {
            "sdv_model_yaml_sha256": (
                sdv_config_sha
            ),
            "model_config_yaml_sha256": (
                model_config_sha
            ),
            "sdv_storage_yaml_sha256": (
                storage_config_sha
            ),
        },
        "reproducibility_id": (
            repro_id
        ),
        "creation_timestamp_utc": (
            created_at
        ),
        "artifact_files": list(
            REQUIRED_ARTIFACT_FILES
        ),
        "production_safety": {
            "dratings_overwritten": False,
            "dratings_model_directory_touched": (
                False
            ),
            "sdv_evaluation_candidate_only": (
                True
            ),
        },
        "version_enforcement_contract": {
            "predictor_must_require_exact_model_version_match": (
                True
            ),
            "predictor_must_require_exact_feature_version_match": (
                True
            ),
            "predictor_must_refuse_on_mismatch": (
                True
            ),
        },
    }


def validate_written_bundle(
    league_root: Path,
) -> None:
    paths = {
        name: (
            league_root
            / name
        )
        for name
        in REQUIRED_ARTIFACT_FILES
    }

    missing = [
        str(path)
        for path
        in paths.values()
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing model artifacts: "
            f"{missing}"
        )

    margin = json_read(
        paths[
            "margin_model.json"
        ]
    )

    total = json_read(
        paths[
            "total_model.json"
        ]
    )

    schema = json_read(
        paths[
            "feature_schema.json"
        ]
    )

    metadata = json_read(
        paths[
            "metadata.json"
        ]
    )

    feature_versions = {
        trainer.clean(
            margin.get(
                "feature_version"
            )
        ),
        trainer.clean(
            total.get(
                "feature_version"
            )
        ),
        trainer.clean(
            schema.get(
                "feature_version"
            )
        ),
        trainer.clean(
            metadata.get(
                "feature_version"
            )
        ),
    }

    if (
        len(feature_versions) != 1
        or "" in feature_versions
    ):
        raise RuntimeError(
            "Artifact feature versions "
            "do not match: "
            f"{feature_versions}"
        )

    model_versions = {
        trainer.clean(
            margin.get(
                "model_version"
            )
        ),
        trainer.clean(
            total.get(
                "model_version"
            )
        ),
        trainer.clean(
            schema.get(
                "model_version"
            )
        ),
        trainer.clean(
            metadata.get(
                "model_version"
            )
        ),
    }

    if (
        len(model_versions) != 1
        or "" in model_versions
    ):
        raise RuntimeError(
            "Artifact model versions "
            "do not match: "
            f"{model_versions}"
        )

    encoded_order = schema.get(
        "encoded_feature_order"
    )

    if not isinstance(
        encoded_order,
        list,
    ):
        raise RuntimeError(
            "feature_schema encoded order "
            "is invalid"
        )

    for (
        name,
        model,
    ) in (
        (
            "margin",
            margin,
        ),
        (
            "total",
            total,
        ),
    ):
        encoder = model.get(
            "encoder"
        )

        if not isinstance(
            encoder,
            dict,
        ):
            raise RuntimeError(
                f"{name}: encoder missing"
            )

        model_order = encoder.get(
            "encoded_feature_order"
        )

        if model_order != encoded_order:
            raise RuntimeError(
                f"{name}: encoded feature "
                "order does not match schema"
            )

        coefficients = model.get(
            "coefficients"
        )

        if not isinstance(
            coefficients,
            list,
        ):
            raise RuntimeError(
                f"{name}: coefficients invalid"
            )

        if (
            len(coefficients)
            != len(
                encoded_order
            )
        ):
            raise RuntimeError(
                f"{name}: coefficient count "
                "does not match schema"
            )

        coefficient_names = [
            item.get(
                "feature"
            )
            for item
            in coefficients
        ]

        if coefficient_names != encoded_order:
            raise RuntimeError(
                f"{name}: coefficient feature "
                "order does not match schema"
            )

        residual = model.get(
            "residual_distribution"
        )

        if not isinstance(
            residual,
            dict,
        ):
            raise RuntimeError(
                f"{name}: residual "
                "distribution missing"
            )

        std = trainer.to_float(
            residual.get(
                "std"
            )
        )

        if (
            std is None
            or std <= 0
        ):
            raise RuntimeError(
                f"{name}: residual std "
                "must be positive"
            )

    contract = metadata.get(
        "version_enforcement_contract"
    )

    if not isinstance(
        contract,
        dict,
    ):
        raise RuntimeError(
            "metadata version enforcement "
            "contract missing"
        )

    if not all(
        bool(
            contract.get(
                key,
                False,
            )
        )
        for key
        in (
            "predictor_must_require_exact_model_version_match",
            "predictor_must_require_exact_feature_version_match",
            "predictor_must_refuse_on_mismatch",
        )
    ):
        raise RuntimeError(
            "metadata version enforcement "
            "contract is incomplete"
        )


def persist_league(
    *,
    sdv_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    artifact_cfg: dict[str, Any],
    league: str,
    sportsdataverse_expected: str,
    sportsdataverse_installed: str | None,
) -> dict[str, Any]:
    label = LEAGUE_LABELS[
        league
    ]

    training_cfg = trainer.required_mapping(
        sdv_cfg,
        "training",
    )

    model_version = trainer.clean(
        training_cfg.get(
            "model_version"
        )
    )

    feature_version = trainer.clean(
        sdv_cfg.get(
            "feature_version"
        )
    )

    if not model_version:
        raise ValueError(
            "training.model_version is blank"
        )

    if not feature_version:
        raise ValueError(
            "feature_version is blank"
        )

    target_cfg = trainer.required_mapping(
        training_cfg,
        "targets",
    )

    margin_target = trainer.clean(
        target_cfg.get(
            "margin"
        )
    )

    total_target = trainer.clean(
        target_cfg.get(
            "total"
        )
    )

    if not margin_target:
        raise ValueError(
            "training.targets.margin "
            "is blank"
        )

    if not total_target:
        raise ValueError(
            "training.targets.total "
            "is blank"
        )

    trainer.validate_calibration_config(
        model_cfg,
        league,
    )

    (
        training_seasons,
        lockbox_season,
        rows,
    ) = development_rows(
        sdv_cfg,
        league,
    )

    (
        encoder,
        margin_coefficients,
        total_coefficients,
        margin_residual_stats,
        total_residual_stats,
        numeric_inputs,
        categorical_inputs,
    ) = fit_artifact_model(
        sdv_cfg,
        rows,
    )

    configured_numeric = list(
        trainer.model_input_names(
            sdv_cfg
        )[
            0
        ]
    )

    configured_categorical = list(
        trainer.model_input_names(
            sdv_cfg
        )[
            1
        ]
    )

    if numeric_inputs != configured_numeric:
        raise RuntimeError(
            f"{label}: numeric feature "
            "order changed during fitting"
        )

    if (
        categorical_inputs
        != configured_categorical
    ):
        raise RuntimeError(
            f"{label}: categorical feature "
            "order changed during fitting"
        )

    if league == "ncaam":
        trainer.validate_ncaam_venue_model(
            rows,
            numeric_inputs,
            categorical_inputs,
            encoder,
            margin_coefficients,
            total_coefficients,
        )

    ridge_alpha = float(
        training_cfg[
            "ridge_alpha"
        ]
    )

    created_at = utc_now()

    probability_cfg = trainer.required_mapping(
        training_cfg,
        "probability",
    )

    margin_probability = {
        "moneyline": (
            trainer.required_mapping(
                probability_cfg,
                "moneyline",
            )
        ),
        "spread": (
            trainer.required_mapping(
                probability_cfg,
                "spread",
            )
        ),
    }

    total_probability = {
        "total": (
            trainer.required_mapping(
                probability_cfg,
                "total",
            )
        )
    }

    margin_payload = model_payload(
        league=league,
        model_version=model_version,
        feature_version=feature_version,
        target_name="margin",
        target_definition=margin_target,
        ridge_alpha=ridge_alpha,
        encoder=encoder,
        coefficients=(
            margin_coefficients
        ),
        residual_stats=(
            margin_residual_stats
        ),
        probability_definition=(
            margin_probability
        ),
        created_at=created_at,
    )

    total_payload = model_payload(
        league=league,
        model_version=model_version,
        feature_version=feature_version,
        target_name="total",
        target_definition=total_target,
        ridge_alpha=ridge_alpha,
        encoder=encoder,
        coefficients=(
            total_coefficients
        ),
        residual_stats=(
            total_residual_stats
        ),
        probability_definition=(
            total_probability
        ),
        created_at=created_at,
    )

    schema_payload = (
        feature_schema_payload(
            league,
            model_version,
            feature_version,
            encoder,
        )
    )

    storage_config_path = Path(
        trainer.clean(
            artifact_cfg[
                "sportsdataverse_config"
            ]
        )
    )

    training_script_path = Path(
        trainer.__file__
    ).resolve()

    persistence_script_path = Path(
        __file__
    ).resolve()

    metadata = metadata_payload(
        league=league,
        model_version=model_version,
        feature_version=feature_version,
        sportsdataverse_expected=(
            sportsdataverse_expected
        ),
        sportsdataverse_installed=(
            sportsdataverse_installed
        ),
        training_seasons=(
            training_seasons
        ),
        lockbox_season=(
            lockbox_season
        ),
        rows=rows,
        target_definitions={
            "margin": margin_target,
            "total": total_target,
        },
        training_script_path=(
            training_script_path
        ),
        persistence_script_path=(
            persistence_script_path
        ),
        storage_config_path=(
            storage_config_path
        ),
        sdv_config_path=(
            SDV_CONFIG_PATH
        ),
        model_config_path=(
            MODEL_CONFIG_PATH
        ),
        created_at=created_at,
    )

    root = Path(
        trainer.clean(
            artifact_cfg[
                "root"
            ]
        )
    )

    league_root = (
        root
        / league
    )

    league_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.write_json_atomic(
        league_root
        / "margin_model.json",
        margin_payload,
    )

    trainer.write_json_atomic(
        league_root
        / "total_model.json",
        total_payload,
    )

    trainer.write_json_atomic(
        league_root
        / "feature_schema.json",
        schema_payload,
    )

    trainer.write_json_atomic(
        league_root
        / "metadata.json",
        metadata,
    )

    validate_written_bundle(
        league_root
    )

    log(
        f"{label} | PASS | "
        f"model_version={model_version} | "
        f"feature_version={feature_version} | "
        f"training_seasons={training_seasons} | "
        f"lockbox={lockbox_season} | "
        f"training_rows={len(rows)} | "
        f"files=4 | "
        f"root={league_root}"
    )

    return {
        "league": label,
        "status": "PASS",
        "model_version": model_version,
        "feature_version": feature_version,
        "training_seasons": (
            training_seasons
        ),
        "lockbox_season": (
            lockbox_season
        ),
        "training_rows": len(
            rows
        ),
        "artifact_root": str(
            league_root
        ),
        "files": list(
            REQUIRED_ARTIFACT_FILES
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Persist/version SDV Model V1 "
            "evaluation artifacts."
        )
    )

    parser.add_argument(
        "--league",
        action="append",
        choices=sorted(
            LEAGUE_LABELS
        ),
        help=(
            "League to persist. "
            "May be repeated. "
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
            "=== SDV MODEL V1 ARTIFACT "
            f"PERSISTENCE {utc_now()} ===\n"
        ),
        encoding="utf-8",
    )

    try:
        sdv_cfg = trainer.read_yaml(
            args.sdv_config
        )

        model_cfg = trainer.read_yaml(
            args.model_config
        )

        trainer.validate_sdv_config(
            sdv_cfg
        )

        artifact_cfg = artifact_config(
            sdv_cfg
        )

        (
            sdv_expected,
            sdv_installed,
        ) = sportsdataverse_version(
            artifact_cfg
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
                persist_league(
                    sdv_cfg=sdv_cfg,
                    model_cfg=model_cfg,
                    artifact_cfg=(
                        artifact_cfg
                    ),
                    league=league,
                    sportsdataverse_expected=(
                        sdv_expected
                    ),
                    sportsdataverse_installed=(
                        sdv_installed
                    ),
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
                "One or more artifact "
                "bundles failed validation"
            )

        total_files = (
            len(reports)
            * len(
                REQUIRED_ARTIFACT_FILES
            )
        )

        log(
            "STATUS: SUCCESS | "
            f"leagues={len(reports)} | "
            f"files={total_files}"
        )

        print(
            "SDV Model V1 artifact persistence "
            "complete: SUCCESS. "
            f"leagues={len(reports)} "
            f"files={total_files}"
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
            "SDV Model V1 artifact persistence "
            f"FAILED: {exc}"
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )