#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/sdv_feature_generation.py
"""Build validated V1 SDV features plus isolated Item 20 advanced candidates."""
from __future__ import annotations

import argparse
import bisect
import csv
import math
import re
import traceback
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
import yaml


BASE = Path("docs/win/basketball")
CONFIG_PATH = BASE / "config/sdv_model.yaml"
ERROR_DIR = BASE / "errors/00_intake"
LOG_FILE = ERROR_DIR / "sdv_feature_generation.txt"
NY_TZ = ZoneInfo("America/New_York")

LEAGUE_LABELS = {
    "nba": "NBA",
    "ncaam": "NCAAM",
    "wnba": "WNBA",
}

PLAYER_STAT_COLUMNS = {
    "points": ("points",),
    "rebounds": ("rebounds",),
    "assists": ("assists",),
    "steals": ("steals",),
    "blocks": ("blocks",),
    "turnovers": ("turnovers",),
    "minutes": ("minutes",),
}

WINDOWED_MODEL_METRICS = (
    "adj_off_eff",
    "adj_def_eff",
    "pace",
    "efg_pct",
    "tov_rate",
    "orb_rate",
    "ft_rate",
    "recent_margin",
    "recent_net_eff",
)


class LeakageError(RuntimeError):
    """Raised when a target game appears in a feature source slice."""


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def clean_id(value: Any) -> str:
    text = clean(
        value
    )
    number = to_float(
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

def to_float(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return float(value)

    text = clean(value)
    if not text:
        return None

    try:
        number = float(text)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value

    text = clean(value).lower()

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n"}:
        return False

    return None


def log(message: str) -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)

    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{datetime.now(timezone.utc).isoformat()} | "
            f"{message}\n"
        )


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    payload = (
        yaml.safe_load(
            path.read_text(
                encoding="utf-8"
            )
        )
        or {}
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"YAML root must be a mapping: {path}"
        )

    return payload


def required_mapping(
    parent: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    value = parent.get(key)

    if not isinstance(value, dict):
        raise ValueError(
            f"sdv_model.yaml missing mapping: {key}"
        )

    return value


def configured_paths(
    cfg: dict[str, Any],
) -> dict[str, Path]:
    section = required_mapping(
        cfg,
        "paths",
    )

    required = (
        "history_input_root",
        "history_output_root",
        "current_output_root",
        "canonical_current_root",
    )

    result: dict[str, Path] = {}

    for key in required:
        value = clean(
            section.get(key)
        )

        if not value:
            raise ValueError(
                f"sdv_model.yaml paths.{key} is blank"
            )

        result[key] = Path(value)

    return result


def validate_config(cfg: dict[str, Any]) -> None:
    _py_r1000_NONE = 0
    _py_r1000_RETURN = 1
    _py_r1000_BREAK = 2
    _py_r1000_CONTINUE = 3

    def _py_r1000_impl():
        nonlocal cfg
        fallback: object
        key: object
        max_poss: object
        min_poss: object
        normalized: object
        player_strength: object
        possession: object
        shrinkage: object
        stat: object
        team_windows: object
        validation: object
        weights: object
        windows: object

        def _py_r1000_loop_1():
            if int(windows.get(key, 0)) <= 0:
                raise ValueError(f'feature_windows.{key} must be positive')
            return (_py_r1000_NONE, None)

        def _py_r1000_loop_3():
            if float(shrinkage.get(key, -1)) < 0:
                raise ValueError(f'shrinkage.{key} must be >= 0')
            return (_py_r1000_NONE, None)

        def _py_r1000_loop_5():
            if clean(stat) not in PLAYER_STAT_COLUMNS:
                raise ValueError(f'Unsupported player_strength weight: {stat}')
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_7():
            nonlocal normalized, team_windows, windows
            if int(cfg.get('schema_version', 0)) != 1:
                raise ValueError('sdv_model.yaml schema_version must be 1')
            if not clean(cfg.get('feature_version')):
                raise ValueError('sdv_model.yaml feature_version is blank')
            windows = required_mapping(cfg, 'feature_windows')
            team_windows = windows.get('team_games')
            if not isinstance(team_windows, list) or not team_windows:
                raise ValueError('feature_windows.team_games must be a non-empty list')
            normalized = [int(value) for value in team_windows]
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_8():
            nonlocal key, shrinkage

            def _py_r1000_loop_9():
                _py_r1000_result_2 = _py_r1000_loop_1()
                if _py_r1000_result_2[0] == _py_r1000_RETURN:
                    return _py_r1000_result_2
                if _py_r1000_result_2[0] == _py_r1000_BREAK:
                    return (_py_r1000_BREAK, None)
                if _py_r1000_result_2[0] == _py_r1000_CONTINUE:
                    return (_py_r1000_CONTINUE, None)
                return (_py_r1000_NONE, None)
            if any((value <= 0 for value in normalized)) or len(set(normalized)) != len(normalized):
                raise ValueError('feature_windows.team_games must contain unique positive integers')
            for key in ('opponent_rating_games', 'player_team_games', 'player_top_n', 'venue_history_games'):
                _py_r1000_result_10 = _py_r1000_loop_9()
                if _py_r1000_result_10[0] == _py_r1000_RETURN:
                    return _py_r1000_result_10
                if _py_r1000_result_10[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_10[0] == _py_r1000_CONTINUE:
                    continue
            shrinkage = required_mapping(cfg, 'shrinkage')
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_11():
            nonlocal fallback, key, max_poss, min_poss, player_strength, possession, validation, weights

            def _py_r1000_loop_12():
                _py_r1000_result_4 = _py_r1000_loop_3()
                if _py_r1000_result_4[0] == _py_r1000_RETURN:
                    return _py_r1000_result_4
                if _py_r1000_result_4[0] == _py_r1000_BREAK:
                    return (_py_r1000_BREAK, None)
                if _py_r1000_result_4[0] == _py_r1000_CONTINUE:
                    return (_py_r1000_CONTINUE, None)
                return (_py_r1000_NONE, None)
            for key in ('team_pseudo_games', 'opponent_pseudo_games', 'player_pseudo_games'):
                _py_r1000_result_13 = _py_r1000_loop_12()
                if _py_r1000_result_13[0] == _py_r1000_RETURN:
                    return _py_r1000_result_13
                if _py_r1000_result_13[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_13[0] == _py_r1000_CONTINUE:
                    continue
            possession = required_mapping(cfg, 'possessions')
            fallback = required_mapping(possession, 'fallback_estimate')
            validation = required_mapping(possession, 'validation')
            if float(fallback.get('free_throw_coefficient', -1)) < 0:
                raise ValueError('possessions.fallback_estimate.free_throw_coefficient must be >= 0')
            min_poss = float(validation.get('min_team_possessions', 0))
            max_poss = float(validation.get('max_team_possessions', 0))
            if min_poss <= 0 or max_poss <= min_poss:
                raise ValueError('possessions.validation min/max team possessions are invalid')
            player_strength = required_mapping(cfg, 'player_strength')
            weights = player_strength.get('weights')
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_14():
            nonlocal stat

            def _py_r1000_loop_15():
                _py_r1000_result_6 = _py_r1000_loop_5()
                if _py_r1000_result_6[0] == _py_r1000_RETURN:
                    return _py_r1000_result_6
                if _py_r1000_result_6[0] == _py_r1000_BREAK:
                    return (_py_r1000_BREAK, None)
                if _py_r1000_result_6[0] == _py_r1000_CONTINUE:
                    return (_py_r1000_CONTINUE, None)
                return (_py_r1000_NONE, None)
            if not isinstance(weights, dict) or not weights:
                raise ValueError('player_strength.weights must be a non-empty mapping')
            for stat in weights:
                _py_r1000_result_16 = _py_r1000_loop_15()
                if _py_r1000_result_16[0] == _py_r1000_RETURN:
                    return _py_r1000_result_16
                if _py_r1000_result_16[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_16[0] == _py_r1000_CONTINUE:
                    continue
            required_mapping(cfg, 'formulas')
            required_mapping(cfg, 'point_in_time')
            required_mapping(cfg, 'venue_context')
            required_mapping(cfg, 'model_inputs')
            configured_paths(cfg)
            return (_py_r1000_NONE, None)
        for _py_r1000_block_17 in (_py_r1000_chunk_7, _py_r1000_chunk_8, _py_r1000_chunk_11, _py_r1000_chunk_14):
            _py_r1000_result_18 = _py_r1000_block_17()
            if _py_r1000_result_18[0] != _py_r1000_NONE:
                return _py_r1000_result_18
        return (_py_r1000_NONE, None)
    _py_r1000_outcome = _py_r1000_impl()
    if _py_r1000_outcome[0] == _py_r1000_RETURN:
        return _py_r1000_outcome[1]


def parse_date(
    value: Any,
) -> date | None:
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    text = clean(value)

    if not text:
        return None

    text = (
        text[:10]
        .replace(
            "_",
            "-",
        )
    )

    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def parse_datetime(
    value: Any,
) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = clean(value)

        if not text:
            return None

        try:
            parsed = datetime.fromisoformat(
                text.replace(
                    "Z",
                    "+00:00",
                )
            )
        except ValueError:
            return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed.astimezone(
        timezone.utc
    )


def canonical_datetime(
    game_date: Any,
    game_time: Any,
) -> datetime | None:
    parsed_date = parse_date(
        game_date
    )

    time_text = clean(
        game_time
    )

    if (
        parsed_date is None
        or not time_text
    ):
        return None

    for fmt in (
        "%I:%M %p",
        "%H:%M",
        "%I:%M:%S %p",
        "%H:%M:%S",
    ):
        try:
            parsed_time = datetime.strptime(
                time_text,
                fmt,
            ).time()

            local = datetime.combine(
                parsed_date,
                parsed_time,
                tzinfo=NY_TZ,
            )

            return local.astimezone(
                timezone.utc
            )

        except ValueError:
            continue

    return None


def normalize_game_date(
    value: Any,
) -> str:
    parsed = parse_date(
        value
    )

    if parsed is None:
        return ""

    return parsed.strftime(
        "%Y_%m_%d"
    )


def normalize_neutral(
    value: Any,
) -> int | None:
    parsed = normalize_bool(
        value
    )

    if parsed is None:
        return None

    return (
        1
        if parsed
        else 0
    )


def first_value(
    row: dict[str, Any],
    columns: tuple[str, ...],
) -> Any:
    for column in columns:
        if (
            column in row
            and clean(
                row.get(column)
            )
            != ""
        ):
            return row.get(column)

    return None


def row_game_id(
    row: dict[str, Any],
) -> str:
    return clean_id(
        row.get("game_id")
        or row.get("id")
    )


def row_team_id(
    row: dict[str, Any],
) -> str:
    return clean_id(
        row.get("team_id")
    )


def row_game_datetime(
    row: dict[str, Any],
) -> datetime | None:
    parsed = parse_datetime(
        row.get(
            "game_date_time"
        )
    )

    if parsed is not None:
        return parsed

    for column in (
        "date",
        "start_date",
    ):
        text = clean(
            row.get(column)
        )

        if (
            not text
            or not (
                "T" in text
                or re.search(
                    r"\d{1,2}:\d{2}",
                    text,
                )
            )
        ):
            continue

        parsed = parse_datetime(
            text
        )

        if parsed is not None:
            return parsed

    return None


def row_game_date(
    row: dict[str, Any],
) -> date | None:
    for column in (
        "game_date",
        "game_date_time",
        "date",
        "start_date",
    ):
        parsed = parse_date(
            row.get(column)
        )

        if parsed is not None:
            return parsed

    return None


def target_time(
    row: dict[str, Any],
    *,
    canonical: bool,
) -> tuple[
    datetime | None,
    date,
]:
    target_date = row_game_date(
        row
    )

    if target_date is None:
        raise ValueError(
            "Target game_id="
            f"{row_game_id(row)} "
            "has no parseable game date"
        )

    if canonical:
        target_dt = canonical_datetime(
            row.get(
                "game_date"
            ),
            row.get(
                "game_time"
            ),
        )
    else:
        target_dt = row_game_datetime(
            row
        )

    return (
        target_dt,
        target_date,
    )


def source_is_prior(
    source: dict[str, Any],
    target_dt: datetime | None,
    target_date: date,
) -> bool:
    source_date = source.get(
        "_date"
    )

    source_dt = source.get(
        "_dt"
    )

    if not isinstance(
        source_date,
        date,
    ):
        return False

    if source_date < target_date:
        return True

    if source_date > target_date:
        return False

    if (
        target_dt is not None
        and isinstance(
            source_dt,
            datetime,
        )
    ):
        return source_dt < target_dt

    return False


def assert_target_absent(
    rows: list[
        dict[str, Any]
    ],
    target_game_id: str,
    source_name: str,
) -> None:
    count = sum(
        1
        for row
        in rows
        if row_game_id(row)
        == target_game_id
    )

    if count:
        raise LeakageError(
            "LEAKAGE CHECK FAILED | "
            f"target_game_id={target_game_id} "
            f"source={source_name} "
            f"rows={count}"
        )


def history_seasons(
    root: Path,
    league: str,
) -> list[int]:
    league_root = (
        root
        / league
    )

    if not league_root.exists():
        raise FileNotFoundError(
            league_root
        )

    seasons = sorted(
        int(path.name)
        for path
        in league_root.iterdir()
        if (
            path.is_dir()
            and path.name.isdigit()
        )
    )

    if not seasons:
        raise RuntimeError(
            "No historical seasons found "
            f"for {league}: {league_root}"
        )

    return seasons


def read_history_table(
    root: Path,
    league: str,
    seasons: list[int],
    table: str,
    *,
    required: bool = True,
) -> list[
    dict[str, Any]
]:
    rows: list[
        dict[str, Any]
    ] = []

    for season in seasons:
        path = (
            root
            / league
            / str(season)
            / f"{table}.parquet"
        )

        if not path.exists():
            if required:
                raise FileNotFoundError(
                    path
                )

            continue

        rows.extend(
            pl.read_parquet(
                path
            ).to_dicts()
        )

    return rows


def build_game_context(
    rows: list[
        dict[str, Any]
    ],
) -> dict[
    str,
    dict[str, Any],
]:
    context: dict[
        str,
        dict[str, Any],
    ] = {}

    for raw in rows:
        game_id = row_game_id(
            raw
        )

        if not game_id:
            continue

        game_date = row_game_date(
            raw
        )

        if game_date is None:
            continue

        context[
            game_id
        ] = {
            **raw,
            "game_id": game_id,
            "_date": game_date,
            "_dt": row_game_datetime(
                raw
            ),
            "home_team_id": clean_id(
                raw.get(
                    "home_team_id"
                )
                or raw.get(
                    "home_id"
                )
            ),
            "away_team_id": clean_id(
                raw.get(
                    "away_team_id"
                )
                or raw.get(
                    "away_id"
                )
            ),
        }

    return context


def source_sort_key(
    row: dict[str, Any],
) -> tuple[
    date,
    datetime,
]:
    source_date = row.get(
        "_date"
    )

    source_dt = row.get(
        "_dt"
    )

    return (
        source_date
        if isinstance(
            source_date,
            date,
        )
        else date.min,
        source_dt
        if isinstance(
            source_dt,
            datetime,
        )
        else datetime.min.replace(
            tzinfo=timezone.utc
        ),
    )


def prior_rows(
    rows: list[
        dict[str, Any]
    ],
    target_dt: datetime | None,
    target_date: date,
) -> list[
    dict[str, Any]
]:
    return [
        row
        for row
        in rows
        if source_is_prior(
            row,
            target_dt,
            target_date,
        )
    ]


def mean(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return (
        sum(values)
        / len(values)
    )


def shrunk_mean(
    recent_values: list[float],
    baseline_values: list[float],
    pseudo_games: float,
) -> float | None:
    recent_mean = mean(
        recent_values
    )

    if recent_mean is None:
        return None

    baseline = mean(
        baseline_values
    )

    if (
        baseline is None
        or pseudo_games <= 0
    ):
        return recent_mean

    n = float(
        len(
            recent_values
        )
    )

    return (
        (
            n
            * recent_mean
        )
        + (
            pseudo_games
            * baseline
        )
    ) / (
        n
        + pseudo_games
    )


def shrink_to_value(
    values: list[float],
    baseline: float | None,
    pseudo_games: float,
) -> float | None:
    value_mean = mean(
        values
    )

    if value_mean is None:
        return baseline

    if (
        baseline is None
        or pseudo_games <= 0
    ):
        return value_mean

    n = float(
        len(values)
    )

    return (
        (
            n
            * value_mean
        )
        + (
            pseudo_games
            * baseline
        )
    ) / (
        n
        + pseudo_games
    )


def safe_ratio(
    numerator: float | None,
    denominator: float | None,
) -> float | None:
    if (
        numerator is None
        or denominator is None
        or denominator <= 0
    ):
        return None

    return (
        numerator
        / denominator
    )


def possession_validation(
    cfg: dict[str, Any],
) -> tuple[
    float,
    float,
]:
    validation = required_mapping(
        required_mapping(
            cfg,
            "possessions",
        ),
        "validation",
    )

    return (
        float(
            validation[
                "min_team_possessions"
            ]
        ),
        float(
            validation[
                "max_team_possessions"
            ]
        ),
    )


def valid_team_possessions(
    value: float | None,
    cfg: dict[str, Any],
) -> bool:
    if (
        value is None
        or not math.isfinite(value)
    ):
        return False

    minimum, maximum = (
        possession_validation(
            cfg
        )
    )

    return (
        minimum
        <= value
        <= maximum
    )


def read_possession_counts(
    root: Path,
    league: str,
    seasons: list[int],
) -> dict[
    str,
    float,
]:
    result: dict[
        str,
        float,
    ] = {}

    for season in seasons:
        path = (
            root
            / league
            / str(season)
            / "possessions.parquet"
        )

        if not path.exists():
            continue

        game_frame = pl.read_parquet(
            path,
            columns=[
                "game_id"
            ],
        )

        if game_frame.is_empty():
            continue

        has_count_flag = True

        try:
            flag_frame = pl.read_parquet(
                path,
                columns=[
                    "count_as_possession"
                ],
            )

        except pl.exceptions.ColumnNotFoundError:
            has_count_flag = False
            frame = game_frame

        else:
            frame = (
                game_frame
                .with_columns(
                    flag_frame[
                        "count_as_possession"
                    ]
                )
            )

        total_counts: dict[
            str,
            int,
        ] = defaultdict(int)

        true_counts: dict[
            str,
            int,
        ] = defaultdict(int)

        flag_seen: dict[
            str,
            bool,
        ] = defaultdict(bool)

        if has_count_flag:
            for (
                game_id_raw,
                flag_raw,
            ) in frame.iter_rows():
                game_id = clean_id(
                    game_id_raw
                )

                if not game_id:
                    continue

                total_counts[
                    game_id
                ] += 1

                parsed_flag = normalize_bool(
                    flag_raw
                )

                if parsed_flag is not None:
                    flag_seen[
                        game_id
                    ] = True

                    if parsed_flag:
                        true_counts[
                            game_id
                        ] += 1

        else:
            for (
                game_id_raw,
            ) in frame.iter_rows():
                game_id = clean_id(
                    game_id_raw
                )

                if game_id:
                    total_counts[
                        game_id
                    ] += 1

        for (
            game_id,
            total,
        ) in total_counts.items():
            counted = (
                true_counts[
                    game_id
                ]
                if flag_seen[
                    game_id
                ]
                else total
            )

            if counted > 0:
                result[
                    game_id
                ] = (
                    counted
                    / 2.0
                )

    return result


def simple_possession_estimate(
    row: dict[str, Any],
    free_throw_coefficient: float,
) -> float | None:
    fga = to_float(
        row.get(
            "field_goals_attempted"
        )
    )

    orb = to_float(
        row.get(
            "offensive_rebounds"
        )
    )

    turnovers = to_float(
        first_value(
            row,
            (
                "total_turnovers",
                "turnovers",
            ),
        )
    )

    fta = to_float(
        row.get(
            "free_throws_attempted"
        )
    )

    if None in (
        fga,
        orb,
        turnovers,
        fta,
    ):
        return None

    value = (
        fga
        - orb
        + turnovers
        + (
            free_throw_coefficient
            * fta
        )
    )

    if value <= 0:
        return None

    return value


def game_possessions(
    game_id: str,
    team_row: dict[str, Any],
    opponent_row: dict[str, Any],
    sdv_counts: dict[
        str,
        float,
    ],
    cfg: dict[str, Any],
) -> tuple[
    float | None,
    str,
]:
    possession_cfg = required_mapping(
        cfg,
        "possessions",
    )

    if bool(
        possession_cfg.get(
            "prefer_sdv",
            True,
        )
    ):
        sdv_value = sdv_counts.get(
            game_id
        )

        if valid_team_possessions(
            sdv_value,
            cfg,
        ):
            return (
                sdv_value,
                "sdv_possessions",
            )

    fallback = required_mapping(
        possession_cfg,
        "fallback_estimate",
    )

    coefficient = float(
        fallback[
            "free_throw_coefficient"
        ]
    )

    team_estimate = (
        simple_possession_estimate(
            team_row,
            coefficient,
        )
    )

    opponent_estimate = (
        simple_possession_estimate(
            opponent_row,
            coefficient,
        )
    )

    if (
        team_estimate is None
        or opponent_estimate is None
    ):
        return (
            None,
            "unavailable",
        )

    estimate = (
        team_estimate
        + opponent_estimate
    ) / 2.0

    if not valid_team_possessions(
        estimate,
        cfg,
    ):
        return (
            None,
            "unavailable",
        )

    return (
        estimate,
        "boxscore_estimate",
    )


def prepare_team_game_rows(rows: list[dict[str, Any]], game_context: dict[str, dict[str, Any]], sdv_possessions: dict[str, float], cfg: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    _py_r1000_NONE = 0
    _py_r1000_RETURN = 1
    _py_r1000_BREAK = 2
    _py_r1000_CONTINUE = 3

    def _py_r1000_impl():
        nonlocal cfg, game_context, rows, sdv_possessions
        all_rows: object
        away_id: object
        by_team: object
        candidates: object
        context: object
        def_eff: object
        efg_pct: object
        fga: object
        fgm: object
        ft_rate: object
        fta: object
        game_id: object
        game_rows: object
        grouped: object
        home_id: object
        margin: object
        net_eff: object
        off_eff: object
        opponent_drb: object
        opponent_id: object
        opponent_row: object
        orb: object
        orb_rate: object
        points_against: object
        points_for: object
        possession_method: object
        possessions: object
        prepared: object
        raw: object
        row_by_team: object
        team_id: object
        team_rows: object
        three_made: object
        tov_rate: object
        turnovers: object

        def _py_r1000_loop_1():
            nonlocal game_id, team_id
            game_id = row_game_id(raw)
            team_id = row_team_id(raw)
            if game_id and team_id:
                grouped[game_id].append(raw)
            return (_py_r1000_NONE, None)

        def _py_r1000_loop_3():
            nonlocal away_id, candidates, context, def_eff, efg_pct, fga, fgm, ft_rate, fta, home_id, margin, net_eff, off_eff, opponent_drb, opponent_id, opponent_row, orb, orb_rate, points_against, points_for, possession_method, possessions, prepared, raw, row_by_team, team_id, three_made, tov_rate, turnovers

            def _py_r1000_loop_4():
                nonlocal candidates, def_eff, efg_pct, fga, fgm, ft_rate, fta, margin, net_eff, off_eff, opponent_drb, opponent_id, opponent_row, orb, orb_rate, points_against, points_for, possession_method, possessions, prepared, team_id, three_made, tov_rate, turnovers

                def _py_r1000_else_5():
                    nonlocal opponent_id
                    if team_id == away_id:
                        opponent_id = home_id
                    else:
                        opponent_id = clean_id(raw.get('opponent_team_id'))
                    return (_py_r1000_NONE, None)

                def _py_r1000_if_7():
                    nonlocal candidates, opponent_row
                    candidates = [row for row in game_rows if row_team_id(row) != team_id]
                    opponent_row = _py_r1000_expr_1()
                    return (_py_r1000_NONE, None)
                team_id = row_team_id(raw)
                if not team_id:
                    return (_py_r1000_CONTINUE, None)
                if team_id == home_id:
                    opponent_id = away_id
                else:
                    _py_r1000_result_6 = _py_r1000_else_5()
                    if _py_r1000_result_6[0] != _py_r1000_NONE:
                        return _py_r1000_result_6
                opponent_row = row_by_team.get(opponent_id)
                if opponent_row is None:
                    _py_r1000_result_8 = _py_r1000_if_7()
                    if _py_r1000_result_8[0] != _py_r1000_NONE:
                        return _py_r1000_result_8
                if opponent_row is None:
                    return (_py_r1000_CONTINUE, None)
                possessions, possession_method = game_possessions(game_id, raw, opponent_row, sdv_possessions, cfg)
                points_for = to_float(raw.get('team_score'))
                points_against = to_float(_py_r1000_expr_2())
                fgm = to_float(raw.get('field_goals_made'))
                fga = to_float(raw.get('field_goals_attempted'))
                three_made = to_float(raw.get('three_point_field_goals_made'))
                turnovers = to_float(first_value(raw, ('total_turnovers', 'turnovers')))
                orb = to_float(raw.get('offensive_rebounds'))
                opponent_drb = to_float(opponent_row.get('defensive_rebounds'))
                fta = to_float(raw.get('free_throws_attempted'))
                off_eff = _py_r1000_expr_3()
                def_eff = _py_r1000_expr_4()
                efg_pct = _py_r1000_expr_5()
                tov_rate = safe_ratio(turnovers, possessions)
                orb_rate = _py_r1000_expr_6()
                ft_rate = safe_ratio(fta, fga)
                margin = _py_r1000_expr_7()
                net_eff = _py_r1000_expr_8()
                prepared = {**raw, 'game_id': game_id, 'team_id': team_id, 'opponent_team_id': opponent_id, '_date': context['_date'], '_dt': context['_dt'], '_possessions': possessions, '_possession_method': possession_method, '_raw_off_eff': off_eff, '_raw_def_eff': def_eff, '_raw_pace': possessions, '_raw_efg_pct': efg_pct, '_raw_tov_rate': tov_rate, '_raw_orb_rate': orb_rate, '_raw_ft_rate': ft_rate, '_raw_margin': margin, '_raw_net_eff': net_eff}
                by_team[team_id].append(prepared)
                all_rows.append(prepared)
                return (_py_r1000_NONE, None)
            context = game_context.get(game_id)
            if context is None:
                return (_py_r1000_CONTINUE, None)
            row_by_team = {row_team_id(row): row for row in game_rows if row_team_id(row)}
            home_id = clean_id(context.get('home_team_id'))
            away_id = clean_id(context.get('away_team_id'))
            for raw in game_rows:
                _py_r1000_result_9 = _py_r1000_loop_4()
                if _py_r1000_result_9[0] == _py_r1000_RETURN:
                    return _py_r1000_result_9
                if _py_r1000_result_9[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_9[0] == _py_r1000_CONTINUE:
                    continue
            return (_py_r1000_NONE, None)

        def _py_r1000_expr_1():
            return candidates[0] if len(candidates) == 1 else None

        def _py_r1000_expr_2():
            return raw.get('opponent_team_score') if clean(raw.get('opponent_team_score')) != '' else opponent_row.get('team_score')

        def _py_r1000_expr_3():
            return 100.0 * points_for / possessions if points_for is not None and possessions is not None and (possessions > 0) else None

        def _py_r1000_expr_4():
            return 100.0 * points_against / possessions if points_against is not None and possessions is not None and (possessions > 0) else None

        def _py_r1000_expr_5():
            return (fgm + 0.5 * three_made) / fga if fgm is not None and three_made is not None and (fga is not None) and (fga > 0) else None

        def _py_r1000_expr_6():
            return orb / (orb + opponent_drb) if orb is not None and opponent_drb is not None and (orb + opponent_drb > 0) else None

        def _py_r1000_expr_7():
            return points_for - points_against if points_for is not None and points_against is not None else None

        def _py_r1000_expr_8():
            return off_eff - def_eff if off_eff is not None and def_eff is not None else None
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for raw in rows:
            _py_r1000_result_2 = _py_r1000_loop_1()
            if _py_r1000_result_2[0] == _py_r1000_RETURN:
                return _py_r1000_result_2
            if _py_r1000_result_2[0] == _py_r1000_BREAK:
                break
            if _py_r1000_result_2[0] == _py_r1000_CONTINUE:
                continue
        by_team: dict[str, list[dict[str, Any]]] = defaultdict(list)
        all_rows: list[dict[str, Any]] = []
        for game_id, game_rows in grouped.items():
            _py_r1000_result_10 = _py_r1000_loop_3()
            if _py_r1000_result_10[0] == _py_r1000_RETURN:
                return _py_r1000_result_10
            if _py_r1000_result_10[0] == _py_r1000_BREAK:
                break
            if _py_r1000_result_10[0] == _py_r1000_CONTINUE:
                continue
        for team_rows in by_team.values():
            team_rows.sort(key=source_sort_key)
        all_rows.sort(key=source_sort_key)
        return (_py_r1000_RETURN, (dict(by_team), all_rows))
    _py_r1000_outcome = _py_r1000_impl()
    if _py_r1000_outcome[0] == _py_r1000_RETURN:
        return _py_r1000_outcome[1]


class LeagueEfficiencyIndex:
    def __init__(
        self,
        rows: list[
            dict[str, Any]
        ],
    ) -> None:
        self.by_date: dict[
            date,
            list[
                dict[str, Any]
            ],
        ] = defaultdict(list)

        for row in rows:
            row_date = row.get(
                "_date"
            )

            if (
                isinstance(
                    row_date,
                    date,
                )
                and row.get(
                    "_raw_off_eff"
                )
                is not None
            ):
                self.by_date[
                    row_date
                ].append(
                    row
                )

        self.dates = sorted(
            self.by_date
        )

        self.cumulative_sum: list[
            float
        ] = []

        self.cumulative_count: list[
            int
        ] = []

        running_sum = 0.0
        running_count = 0

        for row_date in self.dates:
            values = [
                float(
                    row[
                        "_raw_off_eff"
                    ]
                )
                for row
                in self.by_date[
                    row_date
                ]
                if row.get(
                    "_raw_off_eff"
                )
                is not None
            ]

            running_sum += sum(
                values
            )

            running_count += len(
                values
            )

            self.cumulative_sum.append(
                running_sum
            )

            self.cumulative_count.append(
                running_count
            )

    def prior_mean(
        self,
        target_game_id: str,
        target_dt: datetime | None,
        target_date: date,
    ) -> float | None:
        position = (
            bisect.bisect_left(
                self.dates,
                target_date,
            )
            - 1
        )

        total = (
            self.cumulative_sum[
                position
            ]
            if position >= 0
            else 0.0
        )

        count = (
            self.cumulative_count[
                position
            ]
            if position >= 0
            else 0
        )

        if target_dt is not None:
            same_day = [
                row
                for row
                in self.by_date.get(
                    target_date,
                    [],
                )
                if (
                    isinstance(
                        row.get(
                            "_dt"
                        ),
                        datetime,
                    )
                    and row[
                        "_dt"
                    ]
                    < target_dt
                )
            ]

            assert_target_absent(
                same_day,
                target_game_id,
                "league_efficiency_same_day",
            )

            values = [
                float(
                    row[
                        "_raw_off_eff"
                    ]
                )
                for row
                in same_day
                if row.get(
                    "_raw_off_eff"
                )
                is not None
            ]

            total += sum(
                values
            )

            count += len(
                values
            )

        if not count:
            return None

        return (
            total
            / count
        )


def prepare_player_index(
    rows: list[
        dict[str, Any]
    ],
    game_context: dict[
        str,
        dict[str, Any],
    ],
) -> dict[
    str,
    list[
        dict[str, Any]
    ],
]:
    grouped: dict[
        tuple[
            str,
            str,
        ],
        dict[str, Any],
    ] = {}

    for raw in rows:
        team_id = row_team_id(
            raw
        )

        game_id = row_game_id(
            raw
        )

        context = game_context.get(
            game_id
        )

        if (
            not team_id
            or not game_id
            or context is None
        ):
            continue

        key = (
            team_id,
            game_id,
        )

        group = grouped.get(
            key
        )

        if group is None:
            group = {
                "team_id": team_id,
                "game_id": game_id,
                "_date": context[
                    "_date"
                ],
                "_dt": context[
                    "_dt"
                ],
                "players": [],
            }

            grouped[
                key
            ] = group

        group[
            "players"
        ].append(
            raw
        )

    index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = defaultdict(list)

    for group in grouped.values():
        index[
            group[
                "team_id"
            ]
        ].append(
            group
        )

    for games in index.values():
        games.sort(
            key=source_sort_key
        )

    return dict(
        index
    )


def build_home_venue_index(
    game_context: dict[
        str,
        dict[str, Any],
    ],
) -> dict[
    str,
    list[
        dict[str, Any]
    ],
]:
    index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = defaultdict(list)

    for row in game_context.values():
        home_team_id = clean_id(
            row.get(
                "home_team_id"
            )
        )

        if home_team_id:
            index[
                home_team_id
            ].append(
                row
            )

    for games in index.values():
        games.sort(
            key=source_sort_key
        )

    return dict(
        index
    )


def metric_values(
    rows: list[
        dict[str, Any]
    ],
    field: str,
) -> list[float]:
    return [
        float(
            row[
                field
            ]
        )
        for row
        in rows
        if row.get(
            field
        )
        is not None
    ]


def opponent_rating(
    opponent_id: str,
    field: str,
    target_game_id: str,
    exclude_game_id: str,
    target_dt: datetime | None,
    target_date: date,
    team_index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    opponent_rating_games: int,
    league_baseline: float | None,
    opponent_pseudo_games: float,
) -> float | None:
    eligible = prior_rows(
        team_index.get(
            opponent_id,
            [],
        ),
        target_dt,
        target_date,
    )

    assert_target_absent(
        eligible,
        target_game_id,
        f"opponent_team_game:{opponent_id}",
    )

    eligible = [
        row
        for row
        in eligible
        if row_game_id(row)
        != exclude_game_id
    ]

    recent = eligible[
        -opponent_rating_games:
    ]

    values = metric_values(
        recent,
        field,
    )

    return shrink_to_value(
        values,
        league_baseline,
        opponent_pseudo_games,
    )


def adjusted_efficiency_values(
    rows: list[
        dict[str, Any]
    ],
    target_game_id: str,
    target_dt: datetime | None,
    target_date: date,
    team_index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    opponent_rating_games: int,
    league_baseline: float | None,
    opponent_pseudo_games: float,
) -> tuple[
    list[float],
    list[float],
]:
    adjusted_offense: list[
        float
    ] = []

    adjusted_defense: list[
        float
    ] = []

    for row in rows:
        opponent_id = clean_id(
            row.get(
                "opponent_team_id"
            )
        )

        raw_off = to_float(
            row.get(
                "_raw_off_eff"
            )
        )

        raw_def = to_float(
            row.get(
                "_raw_def_eff"
            )
        )

        if not opponent_id:
            continue

        source_game_id = row_game_id(
            row
        )

        opponent_def = opponent_rating(
            opponent_id,
            "_raw_def_eff",
            target_game_id,
            source_game_id,
            target_dt,
            target_date,
            team_index,
            opponent_rating_games,
            league_baseline,
            opponent_pseudo_games,
        )

        opponent_off = opponent_rating(
            opponent_id,
            "_raw_off_eff",
            target_game_id,
            source_game_id,
            target_dt,
            target_date,
            team_index,
            opponent_rating_games,
            league_baseline,
            opponent_pseudo_games,
        )

        if (
            raw_off is not None
            and opponent_def is not None
            and league_baseline is not None
        ):
            adjusted_offense.append(
                raw_off
                - (
                    opponent_def
                    - league_baseline
                )
            )

        if (
            raw_def is not None
            and opponent_off is not None
            and league_baseline is not None
        ):
            adjusted_defense.append(
                raw_def
                - (
                    opponent_off
                    - league_baseline
                )
            )

    return (
        adjusted_offense,
        adjusted_defense,
    )


def team_features(
    team_id: str,
    target_game_id: str,
    target_dt: datetime | None,
    target_date: date,
    team_index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    league_efficiency_index: LeagueEfficiencyIndex,
    windows: list[int],
    opponent_rating_games: int,
    team_pseudo_games: float,
    opponent_pseudo_games: float,
    side: str,
) -> dict[str, Any]:
    eligible = prior_rows(
        team_index.get(
            team_id,
            [],
        ),
        target_dt,
        target_date,
    )

    assert_target_absent(
        eligible,
        target_game_id,
        f"team_game:{side}",
    )

    features: dict[
        str,
        Any,
    ] = {
        f"{side}_games_prior": len(
            eligible
        ),
        f"{side}_rest_days": None,
    }

    if eligible:
        last_date = eligible[
            -1
        ].get(
            "_date"
        )

        if isinstance(
            last_date,
            date,
        ):
            calendar_gap = (
                target_date
                - last_date
            ).days

            features[
                f"{side}_rest_days"
            ] = max(
                calendar_gap - 1,
                0,
            )

    league_baseline = (
        league_efficiency_index
        .prior_mean(
            target_game_id,
            target_dt,
            target_date,
        )
    )

    raw_field_map = {
        "pace": "_raw_pace",
        "efg_pct": "_raw_efg_pct",
        "tov_rate": "_raw_tov_rate",
        "orb_rate": "_raw_orb_rate",
        "ft_rate": "_raw_ft_rate",
        "recent_margin": "_raw_margin",
        "recent_net_eff": "_raw_net_eff",
    }

    for window in windows:
        recent = eligible[
            -window:
        ]

        features[
            f"{side}_games_used_{window}"
        ] = len(
            recent
        )

        (
            adjusted_off,
            adjusted_def,
        ) = adjusted_efficiency_values(
            recent,
            target_game_id,
            target_dt,
            target_date,
            team_index,
            opponent_rating_games,
            league_baseline,
            opponent_pseudo_games,
        )

        features[
            f"{side}_adj_off_eff_{window}"
        ] = shrink_to_value(
            adjusted_off,
            league_baseline,
            team_pseudo_games,
        )

        features[
            f"{side}_adj_def_eff_{window}"
        ] = shrink_to_value(
            adjusted_def,
            league_baseline,
            team_pseudo_games,
        )

        for (
            feature_name,
            raw_field,
        ) in raw_field_map.items():
            recent_values = metric_values(
                recent,
                raw_field,
            )

            baseline_values = metric_values(
                eligible,
                raw_field,
            )

            features[
                f"{side}_{feature_name}_{window}"
            ] = shrunk_mean(
                recent_values,
                baseline_values,
                team_pseudo_games,
            )

        if recent:
            features[
                f"{side}_sdv_possession_share_{window}"
            ] = (
                sum(
                    1
                    for row
                    in recent
                    if row.get(
                        "_possession_method"
                    )
                    == "sdv_possessions"
                )
                / len(recent)
            )

        else:
            features[
                f"{side}_sdv_possession_share_{window}"
            ] = None

    return features


def player_contribution(
    row: dict[str, Any],
    weights: dict[
        str,
        float,
    ],
) -> float:
    total = 0.0

    for (
        stat,
        weight,
    ) in weights.items():
        if stat == "minutes":
            continue

        columns = PLAYER_STAT_COLUMNS[
            stat
        ]

        value = (
            to_float(
                first_value(
                    row,
                    columns,
                )
            )
            or 0.0
        )

        total += (
            float(weight)
            * value
        )

    return total


def player_features(team_id: str, target_game_id: str, target_dt: datetime | None, target_date: date, index: dict[str, list[dict[str, Any]]], team_game_window: int, top_n: int, pseudo_games: float, weights: dict[str, float], side: str) -> dict[str, Any]:
    _py_r1000_NONE = 0
    _py_r1000_RETURN = 1
    _py_r1000_BREAK = 2
    _py_r1000_CONTINUE = 3

    def _py_r1000_impl():
        nonlocal index, pseudo_games, side, target_date, target_dt, target_game_id, team_game_window, team_id, top_n, weights
        all_contributions: object
        by_player: object
        contributions: object
        eligible_games: object
        games: object
        minutes: object
        player_id: object
        raw_strength: object
        recent_games: object
        recent_rows: object
        row: object
        rows: object
        strength: object
        summaries: object
        team_player_baseline: object
        top: object
        value: object

        def _py_r1000_loop_1():
            nonlocal player_id
            player_id = clean_id(row.get('player_id') or row.get('athlete_id'))
            if player_id:
                by_player[player_id].append(row)
            return (_py_r1000_NONE, None)

        def _py_r1000_loop_3():
            nonlocal contributions, games, minutes, raw_strength, row, strength, value

            def _py_r1000_loop_4():
                nonlocal value
                value = to_float(first_value(row, PLAYER_STAT_COLUMNS['minutes']))
                if value is not None:
                    minutes.append(value)
                return (_py_r1000_NONE, None)
            contributions = [player_contribution(row, weights) for row in rows]
            minutes = []
            for row in rows:
                _py_r1000_result_5 = _py_r1000_loop_4()
                if _py_r1000_result_5[0] == _py_r1000_RETURN:
                    return _py_r1000_result_5
                if _py_r1000_result_5[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_5[0] == _py_r1000_CONTINUE:
                    continue
            raw_strength = mean(contributions) or 0.0
            games = len(rows)
            if team_player_baseline is not None and pseudo_games > 0:
                strength = (games * raw_strength + pseudo_games * team_player_baseline) / (games + pseudo_games)
            else:
                strength = raw_strength
            summaries.append({'strength': strength, 'minutes': mean(minutes) or 0.0, 'recent_minutes': sum(minutes)})
            return (_py_r1000_NONE, None)

        def _py_r1000_expr_1():
            return sum((item['strength'] for item in top)) if top else None

        def _py_r1000_expr_2():
            return sum((item['minutes'] for item in top)) if top else None
        eligible_games = prior_rows(index.get(team_id, []), target_dt, target_date)
        assert_target_absent(eligible_games, target_game_id, f'player_game:{side}')
        recent_games = eligible_games[-team_game_window:]
        recent_rows = [player for game in recent_games for player in game['players']]
        assert_target_absent(recent_rows, target_game_id, f'player_game_rows:{side}')
        by_player: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in recent_rows:
            _py_r1000_result_2 = _py_r1000_loop_1()
            if _py_r1000_result_2[0] == _py_r1000_RETURN:
                return _py_r1000_result_2
            if _py_r1000_result_2[0] == _py_r1000_BREAK:
                break
            if _py_r1000_result_2[0] == _py_r1000_CONTINUE:
                continue
        all_contributions = [player_contribution(row, weights) for row in recent_rows]
        team_player_baseline = mean(all_contributions)
        summaries: list[dict[str, float]] = []
        for rows in by_player.values():
            _py_r1000_result_6 = _py_r1000_loop_3()
            if _py_r1000_result_6[0] == _py_r1000_RETURN:
                return _py_r1000_result_6
            if _py_r1000_result_6[0] == _py_r1000_BREAK:
                break
            if _py_r1000_result_6[0] == _py_r1000_CONTINUE:
                continue
        summaries.sort(key=lambda item: (item['recent_minutes'], item['strength']), reverse=True)
        top = summaries[:top_n]
        return (_py_r1000_RETURN, {f'{side}_player_games_used': len(recent_games), f'{side}_player_recent_count': len(summaries), f'{side}_player_strength': _py_r1000_expr_1(), f'{side}_player_minutes': _py_r1000_expr_2()})
    _py_r1000_outcome = _py_r1000_impl()
    if _py_r1000_outcome[0] == _py_r1000_RETURN:
        return _py_r1000_outcome[1]


def team_court_indicator(
    target: dict[str, Any],
    team_id: str,
    direct_team_venue_id: Any,
    target_game_id: str,
    target_dt: datetime | None,
    target_date: date,
    home_venue_index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    venue_history_games: int,
) -> int | None:
    neutral = normalize_neutral(
        target.get(
            "neutral_site"
        )
    )

    if neutral == 1:
        return 0

    if neutral is None:
        return None

    venue_id = clean_id(
        target.get(
            "venue_id"
        )
    )

    if not venue_id:
        return None

    direct_venue_id = clean_id(
        direct_team_venue_id
    )

    if direct_venue_id:
        return (
            1
            if venue_id
            == direct_venue_id
            else 0
        )

    eligible = prior_rows(
        home_venue_index.get(
            team_id,
            [],
        ),
        target_dt,
        target_date,
    )

    assert_target_absent(
        eligible,
        target_game_id,
        f"games:venue_history:{team_id}",
    )

    eligible = eligible[
        -venue_history_games:
    ]

    known_home_venues = {
        clean_id(
            row.get(
                "venue_id"
            )
        )
        for row
        in eligible
        if (
            normalize_neutral(
                row.get(
                    "neutral_site"
                )
            )
            == 0
            and clean_id(
                row.get(
                    "venue_id"
                )
            )
        )
    }

    if not known_home_venues:
        return None

    return (
        1
        if venue_id
        in known_home_venues
        else 0
    )


def venue_name(
    target: dict[str, Any],
) -> str:
    text = clean(
        target.get(
            "venue_name"
        )
        or target.get(
            "venue_full_name"
        )
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def difference(
    home: Any,
    away: Any,
) -> float | None:
    home_value = to_float(
        home
    )

    away_value = to_float(
        away
    )

    if (
        home_value is None
        or away_value is None
    ):
        return None

    return (
        home_value
        - away_value
    )


def add_differentials(
    result: dict[str, Any],
    windows: list[int],
) -> None:
    result[
        "diff_rest_days"
    ] = difference(
        result.get(
            "home_rest_days"
        ),
        result.get(
            "away_rest_days"
        ),
    )

    result[
        "diff_court_indicator"
    ] = difference(
        result.get(
            "home_court_indicator"
        ),
        result.get(
            "away_court_indicator"
        ),
    )

    result[
        "diff_player_strength"
    ] = difference(
        result.get(
            "home_player_strength"
        ),
        result.get(
            "away_player_strength"
        ),
    )

    result[
        "diff_player_minutes"
    ] = difference(
        result.get(
            "home_player_minutes"
        ),
        result.get(
            "away_player_minutes"
        ),
    )

    for window in windows:
        for metric in WINDOWED_MODEL_METRICS:
            result[
                f"diff_{metric}_{window}"
            ] = difference(
                result.get(
                    f"home_{metric}_{window}"
                ),
                result.get(
                    f"away_{metric}_{window}"
                ),
            )

        result[
            f"diff_sdv_possession_share_{window}"
        ] = difference(
            result.get(
                f"home_sdv_possession_share_{window}"
            ),
            result.get(
                f"away_sdv_possession_share_{window}"
            ),
        )


# ITEM 20 ADVANCED SDV CANDIDATE FEATURES
_ADVANCED_CANDIDATE_STATE = {"enabled": False}
ADVANCED_INDEXES: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}

ADVANCED_FAMILY_METRICS = {
    "lineup": (
        "lineup_units_used",
        "lineup_players_used",
        "lineup_top_unit_share",
        "lineup_concentration",
    ),
    "possession": (
        "possession_transition_share",
        "possession_transition_ppp",
        "possession_avg_duration_seconds",
        "possession_early_offense_share",
        "possession_ppp",
    ),
    "shot": (
        "shot_rim_rate",
        "shot_rim_fg_pct",
        "shot_three_rate",
        "shot_three_fg_pct",
        "shot_high_value_rate",
        "shot_avg_distance",
    ),
}


def set_advanced_candidate_mode(enabled: bool) -> None:
    _ADVANCED_CANDIDATE_STATE["enabled"] = bool(enabled)


def production_feature_version(cfg: dict[str, Any]) -> str:
    advanced = cfg.get("advanced_features")
    if isinstance(advanced, dict):
        value = clean(advanced.get("production_feature_version"))
        if value:
            return value
    return clean(cfg.get("feature_version"))


def active_feature_version(cfg: dict[str, Any]) -> str:
    if _ADVANCED_CANDIDATE_STATE["enabled"]:
        return clean(cfg.get("feature_version"))
    return production_feature_version(cfg)


def advanced_history_output_root(cfg: dict[str, Any]) -> Path:
    advanced = required_mapping(cfg, "advanced_features")
    value = clean(advanced.get("candidate_history_output_root"))
    if not value:
        raise ValueError("advanced_features.candidate_history_output_root is blank")
    return Path(value)


def advanced_current_output_root(cfg: dict[str, Any]) -> Path:
    advanced = required_mapping(cfg, "advanced_features")
    value = clean(advanced.get("candidate_current_output_root"))
    if not value:
        raise ValueError("advanced_features.candidate_current_output_root is blank")
    return Path(value)


def advanced_windows(cfg: dict[str, Any]) -> list[int]:
    advanced = required_mapping(cfg, "advanced_features")
    raw = advanced.get("windows")
    if not isinstance(raw, list) or not raw:
        return sorted(int(x) for x in required_mapping(cfg, "feature_windows")["team_games"])
    return sorted({int(x) for x in raw if int(x) > 0})


def advanced_team_id(row: dict[str, Any]) -> str:
    for column in (
        "team_id", "offense_team_id", "offensive_team_id", "possession_team_id",
        "shooting_team_id", "shooter_team_id", "lineup_team_id",
    ):
        value = clean_id(row.get(column))
        if value:
            return value
    return ""


def first_numeric(row: dict[str, Any], columns: tuple[str, ...]) -> float | None:
    for column in columns:
        value = to_float(row.get(column))
        if value is not None:
            return value
    return None


def first_text(row: dict[str, Any], columns: tuple[str, ...]) -> str:
    for column in columns:
        value = clean(row.get(column))
        if value:
            return value
    return ""


def bool_from_columns(row: dict[str, Any], columns: tuple[str, ...]) -> bool | None:
    for column in columns:
        if column not in row:
            continue
        parsed = normalize_bool(row.get(column))
        if parsed is not None:
            return parsed
    return None


def prepared_advanced_rows(
    history_root: Path,
    league: str,
    seasons: list[int],
    table: str,
    game_context: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for season in seasons:
        path = history_root / league / str(season) / f"{table}.parquet"
        if not path.exists():
            log(f"ADVANCED SOURCE MISSING | league={LEAGUE_LABELS[league]} season={season} table={table} path={path}")
            continue
        frame = pl.read_parquet(path)
        if frame.is_empty():
            log(f"ADVANCED SOURCE EMPTY | league={LEAGUE_LABELS[league]} season={season} table={table} path={path}")
            continue
        for raw in frame.to_dicts():
            game_id = row_game_id(raw)
            team_id = advanced_team_id(raw)
            context = game_context.get(game_id)
            if not game_id or not team_id or context is None:
                continue
            rows.append({
                **raw,
                "game_id": game_id,
                "team_id": team_id,
                "_date": context["_date"],
                "_dt": context["_dt"],
            })
        log(f"ADVANCED SOURCE READY | league={LEAGUE_LABELS[league]} season={season} table={table} rows={frame.height} path={path}")
    return rows


def group_advanced_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        team_id = row_team_id(row)
        game_id = row_game_id(row)
        if team_id and game_id:
            grouped[(team_id, game_id)].append(row)
    return dict(grouped)


def index_game_summaries(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        team_id = row_team_id(row)
        if team_id:
            index[team_id].append(row)
    for team_rows in index.values():
        team_rows.sort(key=source_sort_key)
    return dict(index)


def lineup_key(row: dict[str, Any]) -> str:
    return first_text(row, (
        "lineup_id", "lineup", "lineup_name", "unit_id", "group_id",
        "players", "player_ids", "athlete_ids", "lineup_players",
    ))


def player_ids_from_lineup_row(row: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for column in ("player_id", "athlete_id"):
        value = clean_id(row.get(column))
        if value:
            result.add(value)
    for column in ("player_ids", "athlete_ids", "players", "lineup_players"):
        value = row.get(column)
        if isinstance(value, (list, tuple, set)):
            for item in value:
                identifier = clean_id(item)
                if identifier:
                    result.add(identifier)
    return result


def lineup_weight(row: dict[str, Any]) -> float | None:
    seconds = first_numeric(row, (
        "seconds", "duration_seconds", "seconds_played", "stint_seconds",
    ))
    if seconds is not None and seconds >= 0:
        return seconds
    minutes = first_numeric(row, ("minutes", "minutes_played", "stint_minutes"))
    if minutes is not None and minutes >= 0:
        return minutes * 60.0
    possessions = first_numeric(row, ("possessions", "possession_count"))
    if possessions is not None and possessions >= 0:
        return possessions
    return None


def build_lineup_index(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    for (team_id, game_id), game_rows in group_advanced_rows(rows).items():
        lineup_weights: dict[str, float] = defaultdict(float)
        lineup_keys: set[str] = set()
        player_ids: set[str] = set()
        context = game_rows[0]
        for row in game_rows:
            key = lineup_key(row)
            if key:
                lineup_keys.add(key)
            player_ids.update(player_ids_from_lineup_row(row))
            weight = lineup_weight(row)
            if key and weight is not None:
                lineup_weights[key] += weight
        total_weight = sum(lineup_weights.values())
        shares = [value / total_weight for value in lineup_weights.values()] if total_weight > 0 else []
        summaries.append({
            "game_id": game_id,
            "team_id": team_id,
            "_date": context["_date"],
            "_dt": context["_dt"],
            "lineup_units_used": float(len(lineup_keys)) if lineup_keys else None,
            "lineup_players_used": float(len(player_ids)) if player_ids else None,
            "lineup_top_unit_share": max(shares) if shares else None,
            "lineup_concentration": sum(x * x for x in shares) if shares else None,
        })
    return index_game_summaries(summaries)


def possession_counted(row: dict[str, Any]) -> bool:
    if "count_as_possession" not in row:
        return True
    parsed = normalize_bool(row.get("count_as_possession"))
    return True if parsed is None else parsed


def possession_transition(row: dict[str, Any]) -> bool | None:
    explicit = bool_from_columns(row, (
        "is_transition", "transition", "is_fast_break", "fast_break",
    ))
    if explicit is not None:
        return explicit
    text = first_text(row, ("possession_type", "play_type", "type"))
    if not text:
        return None
    normalized = text.lower()
    return ("transition" in normalized) or ("fast break" in normalized) or ("fastbreak" in normalized)


def parse_duration_seconds(value: Any) -> float | None:
    direct = to_float(value)
    if direct is not None and direct >= 0:
        return direct
    text = clean(value)
    match = re.fullmatch(r"(\d+):(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1)) * 60.0 + float(match.group(2))
    return None


def possession_duration(row: dict[str, Any]) -> float | None:
    for column in ("duration_seconds", "possession_seconds", "seconds", "duration"):
        if column in row:
            value = parse_duration_seconds(row.get(column))
            if value is not None:
                return value
    return None


def build_possession_index(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    _py_r1000_NONE = 0
    _py_r1000_RETURN = 1
    _py_r1000_BREAK = 2
    _py_r1000_CONTINUE = 3

    def _py_r1000_impl():
        nonlocal rows
        context: object
        duration: object
        durations: object
        game_id: object
        game_rows: object
        points: object
        pts: object
        row: object
        summaries: object
        team_id: object
        transition: object
        transition_points: object
        transitions: object
        valid: object

        def _py_r1000_loop_1():
            nonlocal context, duration, durations, points, pts, row, transition, transition_points, transitions, valid

            def _py_r1000_loop_2():
                nonlocal duration, pts, transition

                def _py_r1000_if_3():
                    transitions.append(transition)
                    if transition and pts is not None:
                        transition_points.append(pts)
                    return (_py_r1000_NONE, None)
                transition = possession_transition(row)
                pts = first_numeric(row, ('possession_points', 'points_scored', 'points', 'score_value'))
                duration = possession_duration(row)
                if transition is not None:
                    _py_r1000_result_4 = _py_r1000_if_3()
                    if _py_r1000_result_4[0] != _py_r1000_NONE:
                        return _py_r1000_result_4
                if pts is not None:
                    points.append(pts)
                if duration is not None:
                    durations.append(duration)
                return (_py_r1000_NONE, None)
            valid = [row for row in game_rows if possession_counted(row)]
            if not valid:
                return (_py_r1000_CONTINUE, None)
            transitions = []
            transition_points = []
            points = []
            durations = []
            context = valid[0]
            for row in valid:
                _py_r1000_result_5 = _py_r1000_loop_2()
                if _py_r1000_result_5[0] == _py_r1000_RETURN:
                    return _py_r1000_result_5
                if _py_r1000_result_5[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_5[0] == _py_r1000_CONTINUE:
                    continue
            summaries.append({'game_id': game_id, 'team_id': team_id, '_date': context['_date'], '_dt': context['_dt'], 'possession_transition_share': _py_r1000_expr_1(), 'possession_transition_ppp': _py_r1000_expr_2(), 'possession_avg_duration_seconds': mean(durations), 'possession_early_offense_share': _py_r1000_expr_3(), 'possession_ppp': _py_r1000_expr_4()})
            return (_py_r1000_NONE, None)

        def _py_r1000_expr_1():
            return sum((1 for x in transitions if x)) / len(transitions) if transitions else None

        def _py_r1000_expr_2():
            return sum(transition_points) / sum((1 for x in transitions if x)) if transitions and any(transitions) and transition_points else None

        def _py_r1000_expr_3():
            return sum((1 for x in durations if x <= 7.0)) / len(durations) if durations else None

        def _py_r1000_expr_4():
            return sum(points) / len(valid) if points else None
        summaries: list[dict[str, Any]] = []
        for (team_id, game_id), game_rows in group_advanced_rows(rows).items():
            _py_r1000_result_6 = _py_r1000_loop_1()
            if _py_r1000_result_6[0] == _py_r1000_RETURN:
                return _py_r1000_result_6
            if _py_r1000_result_6[0] == _py_r1000_BREAK:
                break
            if _py_r1000_result_6[0] == _py_r1000_CONTINUE:
                continue
        return (_py_r1000_RETURN, index_game_summaries(summaries))
    _py_r1000_outcome = _py_r1000_impl()
    if _py_r1000_outcome[0] == _py_r1000_RETURN:
        return _py_r1000_outcome[1]


def shot_made(row: dict[str, Any]) -> bool | None:
    return bool_from_columns(row, ("made", "shot_made", "is_made", "made_shot"))


def shot_three(row: dict[str, Any]) -> bool | None:
    explicit = bool_from_columns(row, ("is_three", "three_point", "is_three_point"))
    if explicit is not None:
        return explicit
    value = first_numeric(row, ("shot_value", "point_value", "points_possible"))
    if value is not None:
        return value == 3.0
    text = first_text(row, ("shot_type", "type", "shot_type_text"))
    if text:
        normalized = text.lower()
        if "three" in normalized or "3pt" in normalized or "3-point" in normalized or "3 point" in normalized:
            return True
        if "2pt" in normalized or "2-point" in normalized or "2 point" in normalized:
            return False
    return None


def shot_distance(row: dict[str, Any]) -> float | None:
    return first_numeric(row, ("shot_distance", "distance", "distance_feet", "shot_distance_feet"))


def shot_rim(row: dict[str, Any]) -> bool | None:
    explicit = bool_from_columns(row, ("is_rim", "at_rim", "rim"))
    if explicit is not None:
        return explicit
    zone = first_text(row, ("shot_zone", "zone", "shot_zone_basic", "shot_area"))
    if zone:
        normalized = zone.lower()
        if "restricted" in normalized or "at rim" in normalized or "rim" in normalized:
            return True
        if any(token in normalized for token in ("paint", "mid", "three", "3pt", "corner", "arc")):
            return False
    distance = shot_distance(row)
    if distance is not None:
        return distance <= 4.0
    return None


def build_shot_index(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    _py_r1000_NONE = 0
    _py_r1000_RETURN = 1
    _py_r1000_BREAK = 2
    _py_r1000_CONTINUE = 3

    def _py_r1000_impl():
        nonlocal rows
        context: object
        distance: object
        distances: object
        game_id: object
        game_rows: object
        high_value: object
        made: object
        rim: object
        rim_flags: object
        rim_makes: object
        row: object
        summaries: object
        team_id: object
        three: object
        three_flags: object
        three_makes: object

        def _py_r1000_loop_1():
            nonlocal context, distance, distances, high_value, made, rim, rim_flags, rim_makes, row, three, three_flags, three_makes

            def _py_r1000_loop_2():
                nonlocal distance, made, rim, three

                def _py_r1000_if_3():
                    rim_flags.append(rim)
                    if rim and made is not None:
                        rim_makes.append(made)
                    return (_py_r1000_NONE, None)

                def _py_r1000_if_5():
                    three_flags.append(three)
                    if three and made is not None:
                        three_makes.append(made)
                    return (_py_r1000_NONE, None)

                def _py_r1000_if_7():
                    high_value.append(rim or three)
                    return (_py_r1000_NONE, None)
                rim = shot_rim(row)
                three = shot_three(row)
                made = shot_made(row)
                distance = shot_distance(row)
                if rim is not None:
                    _py_r1000_result_4 = _py_r1000_if_3()
                    if _py_r1000_result_4[0] != _py_r1000_NONE:
                        return _py_r1000_result_4
                if three is not None:
                    _py_r1000_result_6 = _py_r1000_if_5()
                    if _py_r1000_result_6[0] != _py_r1000_NONE:
                        return _py_r1000_result_6
                if rim is not None and three is not None:
                    _py_r1000_result_8 = _py_r1000_if_7()
                    if _py_r1000_result_8[0] != _py_r1000_NONE:
                        return _py_r1000_result_8
                if distance is not None:
                    distances.append(distance)
                return (_py_r1000_NONE, None)
            rim_flags = []
            three_flags = []
            rim_makes = []
            three_makes = []
            high_value = []
            distances = []
            context = game_rows[0]
            for row in game_rows:
                _py_r1000_result_9 = _py_r1000_loop_2()
                if _py_r1000_result_9[0] == _py_r1000_RETURN:
                    return _py_r1000_result_9
                if _py_r1000_result_9[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_9[0] == _py_r1000_CONTINUE:
                    continue
            summaries.append({'game_id': game_id, 'team_id': team_id, '_date': context['_date'], '_dt': context['_dt'], 'shot_rim_rate': _py_r1000_expr_1(), 'shot_rim_fg_pct': _py_r1000_expr_2(), 'shot_three_rate': _py_r1000_expr_3(), 'shot_three_fg_pct': _py_r1000_expr_4(), 'shot_high_value_rate': _py_r1000_expr_5(), 'shot_avg_distance': mean(distances)})
            return (_py_r1000_NONE, None)

        def _py_r1000_expr_1():
            return sum((1 for x in rim_flags if x)) / len(rim_flags) if rim_flags else None

        def _py_r1000_expr_2():
            return sum((1 for x in rim_makes if x)) / len(rim_makes) if rim_makes else None

        def _py_r1000_expr_3():
            return sum((1 for x in three_flags if x)) / len(three_flags) if three_flags else None

        def _py_r1000_expr_4():
            return sum((1 for x in three_makes if x)) / len(three_makes) if three_makes else None

        def _py_r1000_expr_5():
            return sum((1 for x in high_value if x)) / len(high_value) if high_value else None
        summaries: list[dict[str, Any]] = []
        for (team_id, game_id), game_rows in group_advanced_rows(rows).items():
            _py_r1000_result_10 = _py_r1000_loop_1()
            if _py_r1000_result_10[0] == _py_r1000_RETURN:
                return _py_r1000_result_10
            if _py_r1000_result_10[0] == _py_r1000_BREAK:
                break
            if _py_r1000_result_10[0] == _py_r1000_CONTINUE:
                continue
        return (_py_r1000_RETURN, index_game_summaries(summaries))
    _py_r1000_outcome = _py_r1000_impl()
    if _py_r1000_outcome[0] == _py_r1000_RETURN:
        return _py_r1000_outcome[1]


def build_advanced_indexes(
    history_root: Path,
    league: str,
    seasons: list[int],
    game_context: dict[str, dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    lineups = prepared_advanced_rows(history_root, league, seasons, "lineups", game_context)
    possessions = prepared_advanced_rows(history_root, league, seasons, "possessions", game_context)
    shots = prepared_advanced_rows(history_root, league, seasons, "shots", game_context)
    return {
        "lineup": build_lineup_index(lineups),
        "possession": build_possession_index(possessions),
        "shot": build_shot_index(shots),
    }


def advanced_side_features(
    family: str,
    team_id: str,
    target_game_id: str,
    target_dt: datetime | None,
    target_date: date,
    family_index: dict[str, list[dict[str, Any]]],
    windows: list[int],
    side: str,
) -> dict[str, Any]:
    eligible = prior_rows(family_index.get(team_id, []), target_dt, target_date)
    assert_target_absent(eligible, target_game_id, f"advanced:{family}:{side}")
    result: dict[str, Any] = {}
    for window in windows:
        recent = eligible[-window:]
        for metric in ADVANCED_FAMILY_METRICS[family]:
            values = [float(row[metric]) for row in recent if row.get(metric) is not None]
            result[f"{side}_{metric}_{window}"] = mean(values)
    return result


def advanced_features_for_target(
    league: str,
    home_team_id: str,
    away_team_id: str,
    target_game_id: str,
    target_dt: datetime | None,
    target_date: date,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    indexes = ADVANCED_INDEXES.get(league, {})
    windows = advanced_windows(cfg)
    result: dict[str, Any] = {}
    for family in ("lineup", "possession", "shot"):
        family_index = indexes.get(family, {})
        result.update(advanced_side_features(family, home_team_id, target_game_id, target_dt, target_date, family_index, windows, "home"))
        result.update(advanced_side_features(family, away_team_id, target_game_id, target_dt, target_date, family_index, windows, "away"))
        for window in windows:
            for metric in ADVANCED_FAMILY_METRICS[family]:
                result[f"diff_{metric}_{window}"] = difference(
                    result.get(f"home_{metric}_{window}"),
                    result.get(f"away_{metric}_{window}"),
                )
    return result


def feature_row(
    target: dict[str, Any],
    *,
    canonical: bool,
    league: str,
    cfg: dict[str, Any],
    feature_version: str,
    generated_at: str,
    team_index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    player_index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    home_venue_index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    league_efficiency_index: LeagueEfficiencyIndex,
    team_windows: list[int],
    opponent_rating_games: int,
    player_game_window: int,
    player_top_n: int,
    venue_history_games: int,
    team_pseudo_games: float,
    opponent_pseudo_games: float,
    player_pseudo_games: float,
    player_weights: dict[
        str,
        float,
    ],
) -> dict[str, Any]:
    target_game_id = row_game_id(
        target
    )

    if not target_game_id:
        raise ValueError(
            "Target row has blank game_id"
        )

    (
        target_dt,
        target_date,
    ) = target_time(
        target,
        canonical=canonical,
    )

    home_team_id = clean_id(
        target.get(
            "home_team_id"
        )
        or target.get(
            "home_id"
        )
    )

    away_team_id = clean_id(
        target.get(
            "away_team_id"
        )
        or target.get(
            "away_id"
        )
    )

    if (
        not home_team_id
        or not away_team_id
    ):
        raise ValueError(
            "Target game_id="
            f"{target_game_id} "
            "has blank home/away team id"
        )

    is_neutral = normalize_neutral(
        target.get(
            "neutral_site"
        )
    )

    home_court = team_court_indicator(
        target,
        home_team_id,
        target.get(
            "home_venue_id"
        ),
        target_game_id,
        target_dt,
        target_date,
        home_venue_index,
        venue_history_games,
    )

    away_court = team_court_indicator(
        target,
        away_team_id,
        target.get(
            "away_venue_id"
        ),
        target_game_id,
        target_dt,
        target_date,
        home_venue_index,
        venue_history_games,
    )

    if (
        is_neutral == 1
        and (
            home_court != 0
            or away_court != 0
        )
    ):
        raise RuntimeError(
            "HOME COURT ASSERTION FAILED | "
            f"target_game_id={target_game_id} "
            "neutral_site=1 "
            f"home_court_indicator={home_court} "
            f"away_court_indicator={away_court}"
        )

    result: dict[
        str,
        Any,
    ] = {
        "league": LEAGUE_LABELS[
            league
        ],
        "internal_season": int(
            clean_id(
                target.get(
                    "internal_season"
                )
            )
            or 0
        ),
        "sdv_season": int(
            clean_id(
                target.get(
                    "sdv_season"
                )
                or target.get(
                    "season"
                )
            )
            or 0
        ),
        "game_id": target_game_id,
        "game_date": (
            target_date.strftime(
                "%Y_%m_%d"
            )
        ),
        "game_date_time_utc": (
            target_dt.isoformat()
            if target_dt
            else None
        ),
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "is_neutral_site": is_neutral,
        "home_court_indicator": home_court,
        "away_court_indicator": away_court,
        "venue_id": clean_id(
            target.get(
                "venue_id"
            )
        ),
        "venue_name": venue_name(
            target
        ),
        "feature_version": feature_version,
        "feature_generated_at_utc": generated_at,
    }

    result.update(
        team_features(
            home_team_id,
            target_game_id,
            target_dt,
            target_date,
            team_index,
            league_efficiency_index,
            team_windows,
            opponent_rating_games,
            team_pseudo_games,
            opponent_pseudo_games,
            "home",
        )
    )

    result.update(
        team_features(
            away_team_id,
            target_game_id,
            target_dt,
            target_date,
            team_index,
            league_efficiency_index,
            team_windows,
            opponent_rating_games,
            team_pseudo_games,
            opponent_pseudo_games,
            "away",
        )
    )

    result.update(
        player_features(
            home_team_id,
            target_game_id,
            target_dt,
            target_date,
            player_index,
            player_game_window,
            player_top_n,
            player_pseudo_games,
            player_weights,
            "home",
        )
    )

    result.update(
        player_features(
            away_team_id,
            target_game_id,
            target_dt,
            target_date,
            player_index,
            player_game_window,
            player_top_n,
            player_pseudo_games,
            player_weights,
            "away",
        )
    )

    add_differentials(
        result,
        team_windows,
    )

    if _ADVANCED_CANDIDATE_STATE["enabled"]:
        result.update(
            advanced_features_for_target(
                league,
                home_team_id,
                away_team_id,
                target_game_id,
                target_dt,
                target_date,
                cfg,
            )
        )

    return result


def validate_model_inputs(
    rows: list[
        dict[str, Any]
    ],
    cfg: dict[str, Any],
) -> None:
    if not rows:
        raise RuntimeError(
            "Feature generation produced zero rows"
        )

    inputs = required_mapping(
        cfg,
        "model_inputs",
    )

    configured: set[str] = set()

    for key in (
        "categorical",
        "numeric",
    ):
        values = inputs.get(
            key,
            [],
        )

        if not isinstance(
            values,
            list,
        ):
            raise ValueError(
                f"model_inputs.{key} must be a list"
            )

        configured.update(
            clean(value)
            for value
            in values
            if clean(value)
        )

    missing = sorted(
        configured
        - set(
            rows[0]
        )
    )

    if missing:
        raise RuntimeError(
            "Configured model inputs "
            "missing from feature output: "
            f"{missing}"
        )


def write_features(
    path: Path,
    rows: list[
        dict[str, Any]
    ],
    cfg: dict[str, Any],
) -> None:
    validate_model_inputs(
        rows,
        cfg,
    )

    game_ids = [
        clean_id(
            row.get(
                "game_id"
            )
        )
        for row
        in rows
    ]

    if any(
        not game_id
        for game_id
        in game_ids
    ):
        raise RuntimeError(
            "Feature output has blank game_id: "
            f"{path}"
        )

    if (
        len(game_ids)
        != len(
            set(game_ids)
        )
    ):
        raise RuntimeError(
            "Feature output has duplicate game_id: "
            f"{path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        f"{path}.tmp"
    )

    if tmp.exists():
        tmp.unlink()

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


def canonical_target_rows(
    path: Path,
    internal_season: int,
) -> list[
    dict[str, Any]
]:
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        rows = [
            dict(row)
            for row
            in csv.DictReader(
                handle
            )
        ]

    for row in rows:
        row[
            "internal_season"
        ] = (
            clean_id(
                row.get(
                    "internal_season"
                )
            )
            or str(
                internal_season
            )
        )

    return rows


def historical_target_rows(
    root: Path,
    league: str,
    internal_season: int,
) -> list[
    dict[str, Any]
]:
    path = (
        root
        / league
        / str(
            internal_season
        )
        / "games.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    return (
        pl.read_parquet(
            path
        )
        .to_dicts()
    )


def generation_settings(
    cfg: dict[str, Any],
) -> tuple[
    list[int],
    int,
    int,
    int,
    int,
    float,
    float,
    float,
    dict[
        str,
        float,
    ],
]:
    windows_cfg = required_mapping(
        cfg,
        "feature_windows",
    )

    shrinkage = required_mapping(
        cfg,
        "shrinkage",
    )

    player_cfg = required_mapping(
        cfg,
        "player_strength",
    )

    weights_raw = player_cfg[
        "weights"
    ]

    weights = {
        clean(key): float(value)
        for (
            key,
            value,
        )
        in weights_raw.items()
    }

    return (
        sorted(
            int(value)
            for value
            in windows_cfg[
                "team_games"
            ]
        ),
        int(
            windows_cfg[
                "opponent_rating_games"
            ]
        ),
        int(
            windows_cfg[
                "player_team_games"
            ]
        ),
        int(
            windows_cfg[
                "player_top_n"
            ]
        ),
        int(
            windows_cfg[
                "venue_history_games"
            ]
        ),
        float(
            shrinkage[
                "team_pseudo_games"
            ]
        ),
        float(
            shrinkage[
                "opponent_pseudo_games"
            ]
        ),
        float(
            shrinkage[
                "player_pseudo_games"
            ]
        ),
        weights,
    )


def build_indexes(
    history_root: Path,
    league: str,
    cfg: dict[str, Any],
) -> tuple[
    list[int],
    dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    LeagueEfficiencyIndex,
]:
    seasons = history_seasons(
        history_root,
        league,
    )

    games = read_history_table(
        history_root,
        league,
        seasons,
        "games",
    )

    team_game = read_history_table(
        history_root,
        league,
        seasons,
        "team_game",
    )

    player_game = read_history_table(
        history_root,
        league,
        seasons,
        "player_game",
    )

    game_context = build_game_context(
        games
    )

    sdv_possessions = (
        read_possession_counts(
            history_root,
            league,
            seasons,
        )
    )

    (
        team_index,
        all_team_rows,
    ) = prepare_team_game_rows(
        team_game,
        game_context,
        sdv_possessions,
        cfg,
    )

    player_index = prepare_player_index(
        player_game,
        game_context,
    )

    home_venue_index = (
        build_home_venue_index(
            game_context
        )
    )

    league_efficiency_index = (
        LeagueEfficiencyIndex(
            all_team_rows
        )
    )

    if _ADVANCED_CANDIDATE_STATE["enabled"]:
        ADVANCED_INDEXES[league] = build_advanced_indexes(
            history_root,
            league,
            seasons,
            game_context,
        )
    else:
        ADVANCED_INDEXES.pop(league, None)

    return (
        seasons,
        team_index,
        player_index,
        home_venue_index,
        league_efficiency_index,
    )


def generate_rows(
    targets: list[
        dict[str, Any]
    ],
    *,
    canonical: bool,
    league: str,
    cfg: dict[str, Any],
    team_index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    player_index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    home_venue_index: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
    league_efficiency_index: LeagueEfficiencyIndex,
    generated_at: str,
) -> list[
    dict[str, Any]
]:
    (
        team_windows,
        opponent_rating_games,
        player_game_window,
        player_top_n,
        venue_history_games,
        team_pseudo_games,
        opponent_pseudo_games,
        player_pseudo_games,
        player_weights,
    ) = generation_settings(
        cfg
    )

    feature_version = active_feature_version(cfg)

    rows = [
        feature_row(
            target,
            canonical=canonical,
            league=league,
            cfg=cfg,
            feature_version=feature_version,
            generated_at=generated_at,
            team_index=team_index,
            player_index=player_index,
            home_venue_index=home_venue_index,
            league_efficiency_index=(
                league_efficiency_index
            ),
            team_windows=team_windows,
            opponent_rating_games=(
                opponent_rating_games
            ),
            player_game_window=(
                player_game_window
            ),
            player_top_n=player_top_n,
            venue_history_games=(
                venue_history_games
            ),
            team_pseudo_games=(
                team_pseudo_games
            ),
            opponent_pseudo_games=(
                opponent_pseudo_games
            ),
            player_pseudo_games=(
                player_pseudo_games
            ),
            player_weights=player_weights,
        )
        for target
        in targets
    ]

    rows.sort(
        key=lambda row: (
            row[
                "game_date"
            ],
            row[
                "game_date_time_utc"
            ]
            or "",
            row[
                "game_id"
            ],
        )
    )

    return rows


def build_historical(
    cfg: dict[str, Any],
    league: str,
    internal_seasons: list[int] | None,
) -> list[Path]:
    paths = configured_paths(
        cfg
    )

    history_root = paths[
        "history_input_root"
    ]

    (
        available,
        team_index,
        player_index,
        home_venue_index,
        league_efficiency_index,
    ) = build_indexes(
        history_root,
        league,
        cfg,
    )

    selected = sorted(
        set(
            internal_seasons
            or available
        )
    )

    invalid = sorted(
        set(selected)
        - set(available)
    )

    if invalid:
        raise ValueError(
            "Historical seasons unavailable "
            f"for {league}: {invalid}"
        )

    generated_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    outputs: list[
        Path
    ] = []

    label = LEAGUE_LABELS[
        league
    ]

    for season in selected:
        targets = historical_target_rows(
            history_root,
            league,
            season,
        )

        rows = generate_rows(
            targets,
            canonical=False,
            league=league,
            cfg=cfg,
            team_index=team_index,
            player_index=player_index,
            home_venue_index=(
                home_venue_index
            ),
            league_efficiency_index=(
                league_efficiency_index
            ),
            generated_at=(
                generated_at
            ),
        )

        output_root = (
            advanced_history_output_root(cfg)
            if _ADVANCED_CANDIDATE_STATE["enabled"]
            else paths["history_output_root"]
        )

        output = (
            output_root
            / league
            / (
                f"{season}_"
                f"{label}_features.parquet"
            )
        )

        write_features(
            output,
            rows,
            cfg,
        )

        outputs.append(
            output
        )

        log(
            "HISTORY READY | "
            f"league={label} "
            f"internal_season={season} "
            f"rows={len(rows)} "
            f"path={output}"
        )

    return outputs


def build_current(
    cfg: dict[str, Any],
    league: str,
    internal_season: int,
    game_dates: list[str] | None,
) -> list[Path]:
    paths = configured_paths(
        cfg
    )

    history_root = paths[
        "history_input_root"
    ]

    (
        _,
        team_index,
        player_index,
        home_venue_index,
        league_efficiency_index,
    ) = build_indexes(
        history_root,
        league,
        cfg,
    )

    label = LEAGUE_LABELS[
        league
    ]

    canonical_path = (
        paths[
            "canonical_current_root"
        ]
        / league
        / (
            f"{internal_season}_"
            f"{label}_games.csv"
        )
    )

    targets = canonical_target_rows(
        canonical_path,
        internal_season,
    )

    requested_dates = {
        normalize_game_date(
            value
        )
        for value
        in (
            game_dates
            or []
        )
    }

    requested_dates.discard(
        ""
    )

    if requested_dates:
        targets = [
            row
            for row
            in targets
            if normalize_game_date(
                row.get(
                    "game_date"
                )
            )
            in requested_dates
        ]

    if not targets:
        raise RuntimeError(
            "No current/upcoming target games "
            f"selected from {canonical_path}"
        )

    grouped: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = defaultdict(list)

    for row in targets:
        game_date = normalize_game_date(
            row.get(
                "game_date"
            )
        )

        if not game_date:
            raise ValueError(
                "Canonical target game_id="
                f"{row_game_id(row)} "
                "has invalid game_date"
            )

        grouped[
            game_date
        ].append(
            row
        )

    generated_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    outputs: list[
        Path
    ] = []

    for game_date in sorted(
        grouped
    ):
        rows = generate_rows(
            grouped[
                game_date
            ],
            canonical=True,
            league=league,
            cfg=cfg,
            team_index=team_index,
            player_index=player_index,
            home_venue_index=(
                home_venue_index
            ),
            league_efficiency_index=(
                league_efficiency_index
            ),
            generated_at=(
                generated_at
            ),
        )

        output_root = (
            advanced_current_output_root(cfg)
            if _ADVANCED_CANDIDATE_STATE["enabled"]
            else paths["current_output_root"]
        )

        output = (
            output_root
            / league
            / (
                f"{game_date}_"
                f"{label}_features.parquet"
            )
        )

        write_features(
            output,
            rows,
            cfg,
        )

        outputs.append(
            output
        )

        log(
            "CURRENT READY | "
            f"league={label} "
            f"internal_season={internal_season} "
            f"game_date={game_date} "
            f"rows={len(rows)} "
            f"path={output}"
        )

    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build strict point-in-time "
            "SDV Model V1 basketball features."
        )
    )

    parser.add_argument(
        "--mode",
        choices=(
            "history",
            "current",
        ),
        required=True,
    )

    parser.add_argument(
        "--league",
        action="append",
        choices=sorted(
            LEAGUE_LABELS
        ),
    )

    parser.add_argument(
        "--internal-season",
        action="append",
        type=int,
    )

    parser.add_argument(
        "--game-date",
        action="append",
        help=(
            "Current mode only; "
            "YYYY_MM_DD or YYYY-MM-DD. "
            "May be repeated."
        ),
    )

    parser.add_argument(
        "--advanced-candidate",
        action="store_true",
        help=(
            "Generate isolated Item 20 candidate features. "
            "Default remains the validated V1 feature pipeline."
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
    )

    return parser


def main() -> int:
    args = (
        build_parser()
        .parse_args()
    )

    ERROR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_FILE.write_text(
        (
            "=== SDV MODEL V1 FEATURE GENERATION "
            f"{datetime.now(timezone.utc).isoformat()} "
            "===\n"
        ),
        encoding="utf-8",
    )

    try:
        cfg = read_yaml(
            args.config
        )

        validate_config(
            cfg
        )

        set_advanced_candidate_mode(
            args.advanced_candidate
        )

        leagues = (
            args.league
            or list(
                LEAGUE_LABELS
            )
        )

        outputs: list[
            Path
        ] = []

        if args.mode == "history":
            if args.game_date:
                raise ValueError(
                    "--game-date is only valid "
                    "with --mode current"
                )

            for league in leagues:
                outputs.extend(
                    build_historical(
                        cfg,
                        league,
                        args.internal_season,
                    )
                )

        else:
            if len(leagues) != 1:
                raise ValueError(
                    "--mode current requires "
                    "exactly one --league"
                )

            if (
                not args.internal_season
                or len(
                    args.internal_season
                )
                != 1
            ):
                raise ValueError(
                    "--mode current requires "
                    "exactly one --internal-season"
                )

            outputs.extend(
                build_current(
                    cfg,
                    leagues[0],
                    args.internal_season[0],
                    args.game_date,
                )
            )

        log(
            "STATUS: SUCCESS | "
            f"files={len(outputs)} "
            "feature_version="
            f"{cfg['feature_version']}"
        )

        print(
            "SDV Model V1 feature generation "
            "complete: SUCCESS. "
            f"files={len(outputs)}"
        )

        return 0

    except LeakageError as exc:
        log(
            f"LEAKAGE FAILURE: {exc}"
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
            "SDV Model V1 feature generation "
            f"FAILED: {exc}"
        )

        return 2

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
            "SDV Model V1 feature generation "
            f"FAILED: {exc}"
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
