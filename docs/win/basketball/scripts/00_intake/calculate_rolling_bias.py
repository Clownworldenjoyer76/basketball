#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/calculate_rolling_bias.py
#
# Calculates current margin and total bias values from completed basketball games.
#
# Reads permanent rules from:
#   docs/win/basketball/config/model_config.yaml
#
# Reads historical completed games from:
#   docs/win/basketball/00_intake/final_combined_files/combined/{season}_{LEAGUE}.csv
#
# Reads current-season RAW predictions from:
#   docs/win/basketball/00_intake/predictions/{league}/{date}_{LEAGUE}_predictions.csv
#
# Reads current-season final scores from:
#   docs/win/basketball/05_final_scores/results/{league}/{date}_final_scores_{LEAGUE}.csv
#
# Writes:
#   docs/win/basketball/config/rolling_bias_state.yaml
#
# Important rules:
# - Current prediction files are RAW / PRE-BIAS and are never reversed here.
# - Historical combined projections must honor bias_applied strictly:
#     0 -> already raw
#     1 -> reverse using exact per-game margin_bias + total_bias when available;
#          otherwise use the known legacy 2025 fallback only.
#     anything else -> invalid historical row.
# - Operational season boundaries are read from:
#   docs/win/basketball/config/season_dates.yaml
# - Dates outside a league's configured season window are offseason.
# - Rolling/regime-aware windows cross season boundaries and require the full configured window.
# - Projection error sign is projected_minus_actual.

from __future__ import annotations

import csv
import hashlib
import math
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import yaml


# ============================================================================
# PATHS / CONSTANTS
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()

_EXPECTED_REPO_ROOT = (
    SCRIPT_PATH.parents[5]
    if len(SCRIPT_PATH.parents) > 5
    else Path.cwd()
)

_CWD_REPO_ROOT = Path.cwd().resolve()

if (
    _EXPECTED_REPO_ROOT
    / "docs/win/basketball/config/model_config.yaml"
).exists():
    REPO_ROOT = _EXPECTED_REPO_ROOT

elif (
    _CWD_REPO_ROOT
    / "docs/win/basketball/config/model_config.yaml"
).exists():
    REPO_ROOT = _CWD_REPO_ROOT

else:
    REPO_ROOT = _EXPECTED_REPO_ROOT


CONFIG_PATH = (
    REPO_ROOT
    / "docs/win/basketball/config/model_config.yaml"
)

SEASON_CONFIG_PATH = (
    REPO_ROOT
    / "docs/win/basketball/config/season_dates.yaml"
)

STATE_PATH = (
    REPO_ROOT
    / "docs/win/basketball/config/rolling_bias_state.yaml"
)

HISTORICAL_DIR = (
    REPO_ROOT
    / "docs/win/basketball/00_intake/final_combined_files/combined"
)

PREDICTION_ROOTS = {
    "dratings": (
        REPO_ROOT
        / "docs/win/basketball/00_intake/predictions"
    ),
    "sdv": (
        REPO_ROOT
        / "docs/win/basketball/00_intake/predictions_sdv"
    ),
    "ensemble": (
        REPO_ROOT
        / "docs/win/basketball/00_intake/predictions_ensemble"
    ),
}

FINAL_SCORES_ROOT = (
    REPO_ROOT
    / "docs/win/basketball/05_final_scores/results"
)

ERROR_DIR = (
    REPO_ROOT
    / "docs/win/basketball/errors/00_intake"
)

LOG_PATH = (
    ERROR_DIR
    / "calculate_rolling_bias.txt"
)

NY_TZ = ZoneInfo("America/New_York")

SUPPORTED_LEAGUES = (
    "nba",
    "ncaam",
    "wnba",
)


# These values are ONLY a reversal fallback for legacy 2025 historical files.
LEGACY_HISTORICAL_BIAS: dict[
    tuple[str, int],
    dict[str, float],
] = {
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


PREDICTION_REQUIRED = {
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_projected_points",
    "away_projected_points",
}


FINAL_REQUIRED = {
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
}


HISTORICAL_REQUIRED = {
    "game_date",
    "home_team",
    "away_team",
    "home_projected_points",
    "away_projected_points",
    "home_score",
    "away_score",
    "bias_applied",
}


WARNING_HISTORY_FIELDS = (
    "historical_incomplete_rows",
    "historical_rows_invalid_date",
    "historical_invalid_bias_flag_rows",
    "prediction_rows_invalid_date",
    "final_rows_invalid_date",
    "conflicting_prediction_game_ids",
    "conflicting_prediction_composites",
    "conflicting_final_game_ids",
    "conflicting_final_composites",
    "game_id_identity_mismatches",
    "ambiguous_prediction_matches",
    "true_unmatched_current_finals",
    "invalid_current_matches",
)


# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass(frozen=True)
class CompletedGame:
    league: str
    game_id: str
    game_date: str
    game_time: str
    home_team: str
    away_team: str
    home_projected_points: float
    away_projected_points: float
    total_projected_points: float
    home_score: float
    away_score: float
    source: str
    source_priority: int

    @property
    def projected_margin(self) -> float:
        return (
            self.home_projected_points
            - self.away_projected_points
        )

    @property
    def actual_margin(self) -> float:
        return (
            self.home_score
            - self.away_score
        )

    @property
    def margin_error(self) -> float:
        return (
            self.projected_margin
            - self.actual_margin
        )

    @property
    def actual_total(self) -> float:
        return (
            self.home_score
            + self.away_score
        )

    @property
    def total_error(self) -> float:
        return (
            self.total_projected_points
            - self.actual_total
        )

    @property
    def composite(self) -> str:
        return composite_key(
            self.game_date,
            self.home_team,
            self.away_team,
        )

    @property
    def sort_key(self) -> tuple:
        return (
            parse_game_datetime(
                self.game_date,
                self.game_time,
            ),
            normalize_text(
                self.home_team
            ),
            normalize_text(
                self.away_team
            ),
            canonical_game_id(
                self.game_id
            ),
            self.source,
        )


# ============================================================================
# LOGGING / PROVENANCE
# ============================================================================

def utc_now_iso() -> str:
    return (
        datetime
        .now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def local_today() -> date:
    return datetime.now(
        NY_TZ
    ).date()


def repo_relative(
    path: Path,
) -> str:
    try:
        return (
            path
            .resolve()
            .relative_to(
                REPO_ROOT.resolve()
            )
            .as_posix()
        )

    except ValueError:
        return str(
            path.resolve()
        )


def script_sha256() -> str:
    digest = hashlib.sha256()

    with open(
        SCRIPT_PATH,
        "rb",
    ) as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def init_log() -> None:
    ERROR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        LOG_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"=== calculate_rolling_bias RUN "
            f"{utc_now_iso()} ===\n"
        )

        f.write(
            f"REPO_ROOT={REPO_ROOT}\n"
        )

        f.write(
            f"CONFIG_PATH={CONFIG_PATH}\n"
        )

        f.write(
            f"STATE_PATH={STATE_PATH}\n"
        )

        f.write(
            f"LOCAL_DATE="
            f"{local_today().isoformat()}\n"
        )


def log(
    message: str,
    level: str = "INFO",
) -> None:
    line = (
        f"{utc_now_iso()} | "
        f"{level:<5} | "
        f"{message}"
    )

    print(
        line,
        flush=True,
    )

    with open(
        LOG_PATH,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            line + "\n"
        )


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def normalize_text(
    value: Any,
) -> str:
    text = (
        ""
        if value is None
        else str(value)
    )

    return re.sub(
        r"\s+",
        " ",
        text.strip().lower(),
    )


def canonical_game_id(
    value: Any,
) -> str:
    text = (
        ""
        if value is None
        else str(value).strip()
    )

    if re.fullmatch(
        r"\d+\.0",
        text,
    ):
        return text[:-2]

    return text


def normalize_date(
    value: Any,
) -> str:
    text = (
        ""
        if value is None
        else str(value).strip()
    )

    if not text:
        return ""

    text = (
        text
        .replace("/", "-")
        .replace("_", "-")
    )

    for fmt in (
        "%Y-%m-%d",
        "%m-%d-%Y",
        "%m-%d-%y",
    ):
        try:
            return datetime.strptime(
                text,
                fmt,
            ).strftime(
                "%Y-%m-%d"
            )

        except ValueError:
            pass

    return ""


def parse_date(
    value: Any,
) -> date | None:
    normalized = normalize_date(
        value
    )

    if not normalized:
        return None

    try:
        return datetime.strptime(
            normalized,
            "%Y-%m-%d",
        ).date()

    except ValueError:
        return None


def composite_key(
    game_date: Any,
    home_team: Any,
    away_team: Any,
) -> str:
    date_value = normalize_date(
        game_date
    )

    home = normalize_text(
        home_team
    )

    away = normalize_text(
        away_team
    )

    if (
        not date_value
        or not home
        or not away
    ):
        return ""

    return (
        f"{date_value}|"
        f"{home}|"
        f"{away}"
    )


def to_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    text = (
        str(value)
        .strip()
        .replace(",", "")
    )

    if not text:
        return None

    try:
        number = float(
            text
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        number
    ):
        return None

    return number


def parse_bias_flag(
    value: Any,
) -> int | None:
    """
    Strict numeric interpretation.

    Valid:
        0
        0.0
        1
        1.0

    Invalid:
        blank
        null
        true
        false
        yes
        no
        any numeric value other than 0 or 1
    """

    if value is None:
        return None

    text = str(
        value
    ).strip()

    if not text:
        return None

    # Reject boolean-like/non-numeric strings.
    if not re.fullmatch(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)",
        text,
    ):
        return None

    try:
        number = float(
            text
        )

    except ValueError:
        return None

    if not math.isfinite(
        number
    ):
        return None

    if number == 0.0:
        return 0

    if number == 1.0:
        return 1

    return None


def parse_game_datetime(
    game_date: Any,
    game_time: Any = "",
) -> datetime:
    date_text = normalize_date(
        game_date
    )

    try:
        base = datetime.strptime(
            date_text,
            "%Y-%m-%d",
        )

    except ValueError:
        return datetime.min

    time_text = (
        ""
        if game_time is None
        else str(game_time).strip()
    )

    if not time_text:
        return base

    cleaned = re.sub(
        r"\s+",
        " ",
        time_text.upper(),
    ).strip()

    cleaned = re.sub(
        r"\s+(ET|EST|EDT)$",
        "",
        cleaned,
    ).strip()

    for fmt in (
        "%I:%M %p",
        "%I:%M:%S %p",
        "%I %p",
        "%H:%M",
        "%H:%M:%S",
    ):
        try:
            parsed_time = (
                datetime
                .strptime(
                    cleaned,
                    fmt,
                )
                .time()
            )

            return datetime.combine(
                base.date(),
                parsed_time,
            )

        except ValueError:
            pass

    # Invalid or absent game_time does not invalidate
    # an otherwise valid completed game.
    return base


def read_csv_rows(
    path: Path,
) -> tuple[
    list[str],
    list[dict[str, str]],
]:
    repo_root = Path(__file__).resolve().parents[5]
    candidate = path if path.is_absolute() else Path.cwd() / path
    safe_path = candidate.resolve(strict=True)
    try:
        safe_path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {path}") from exc

    with safe_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        reader = csv.DictReader(
            f
        )

        result = (
            reader.fieldnames
            or [],
            list(reader),
        )
    return result


def require_columns(
    path: Path,
    fieldnames: Iterable[str],
    required: set[str],
) -> None:
    missing = sorted(
        required
        - set(fieldnames)
    )

    if missing:
        raise ValueError(
            f"{repo_relative(path)} "
            f"is missing required columns: "
            f"{', '.join(missing)}"
        )


def league_upper(
    league: str,
) -> str:
    return (
        league
        .strip()
        .upper()
    )


def positive_int_or_none(
    value: Any,
    label: str,
) -> int | None:
    if value in (
        None,
        "",
    ):
        return None

    if isinstance(
        value,
        bool,
    ):
        raise ValueError(
            f"{label} must be "
            f"an integer, not boolean"
        )

    try:
        number = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"{label} must be "
            f"an integer; got "
            f"{value!r}"
        ) from exc

    if (
        not math.isfinite(
            number
        )
        or not number.is_integer()
    ):
        raise ValueError(
            f"{label} must be "
            f"an integer; got "
            f"{value!r}"
        )

    return int(
        number
    )


def normalized_prediction_total(
    row: dict[str, str],
) -> float | None:
    total = to_float(
        row.get(
            "total_projected_points"
        )
    )

    if total is not None:
        return total

    home = to_float(
        row.get(
            "home_projected_points"
        )
    )

    away = to_float(
        row.get(
            "away_projected_points"
        )
    )

    if (
        home is None
        or away is None
    ):
        return None

    return (
        home
        + away
    )


def prediction_signature(
    row: dict[str, str],
) -> tuple:
    return (
        canonical_game_id(
            row.get(
                "game_id"
            )
        ),
        normalize_date(
            row.get(
                "game_date"
            )
        ),
        normalize_text(
            row.get(
                "game_time"
            )
        ),
        normalize_text(
            row.get(
                "home_team"
            )
        ),
        normalize_text(
            row.get(
                "away_team"
            )
        ),
        to_float(
            row.get(
                "home_projected_points"
            )
        ),
        to_float(
            row.get(
                "away_projected_points"
            )
        ),
        normalized_prediction_total(
            row
        ),
    )


def final_identity_score_signature(
    row: dict[str, str],
) -> tuple:
    return (
        normalize_date(
            row.get(
                "game_date"
            )
        ),
        normalize_text(
            row.get(
                "home_team"
            )
        ),
        normalize_text(
            row.get(
                "away_team"
            )
        ),
        to_float(
            row.get(
                "home_score"
            )
        ),
        to_float(
            row.get(
                "away_score"
            )
        ),
    )


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def direct_csv_files(
    folder: Path,
) -> list[Path]:
    if not folder.exists():
        return []

    return sorted(
        p
        for p in folder.iterdir()
        if (
            p.is_file()
            and p.suffix.lower()
            == ".csv"
        )
    )


def historical_files_for_league(
    league: str,
) -> list[
    tuple[int, Path]
]:
    pattern = re.compile(
        rf"^(\d{{4}})_"
        rf"{re.escape(league_upper(league))}"
        rf"\.csv$",
        re.IGNORECASE,
    )

    matches: list[
        tuple[int, Path]
    ] = []

    for path in direct_csv_files(
        HISTORICAL_DIR
    ):
        match = pattern.fullmatch(
            path.name
        )

        if not match:
            continue

        matches.append(
            (
                int(
                    match.group(1)
                ),
                path,
            )
        )

    return sorted(
        matches,
        key=lambda item: (
            item[0],
            item[1]
            .name
            .lower(),
        ),
    )


def prediction_files_for_league(
    league: str,
) -> list[Path]:
    cfg = load_model_config()

    source = production_prediction_source(
        cfg
    )

    folder = (
        PREDICTION_ROOTS[
            source
        ]
        / league.lower()
    )

    pattern = re.compile(
        rf"^\d{{4}}_"
        rf"\d{{2}}_"
        rf"\d{{2}}_"
        rf"{re.escape(league_upper(league))}"
        rf"_predictions\.csv$",
        re.IGNORECASE,
    )

    return [
        path
        for path in direct_csv_files(
            folder
        )
        if pattern.fullmatch(
            path.name
        )
    ]


def final_files_for_league(
    league: str,
) -> list[Path]:
    folder = (
        FINAL_SCORES_ROOT
        / league.lower()
    )

    pattern = re.compile(
        rf"^\d{{4}}_"
        rf"\d{{2}}_"
        rf"\d{{2}}_"
        rf"final_scores_"
        rf"{re.escape(league_upper(league))}"
        rf"\.csv$",
        re.IGNORECASE,
    )

    return [
        path
        for path in direct_csv_files(
            folder
        )
        if pattern.fullmatch(
            path.name
        )
    ]


# ============================================================================
# SEASON RULES
# ============================================================================

_SEASON_CONFIG_CACHE = {"value": None}


def load_season_config() -> dict[str, dict[str, int]]:
    cached = _SEASON_CONFIG_CACHE["value"]
    if cached is not None:
        return cached

    if not SEASON_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing season config: "
            f"{SEASON_CONFIG_PATH}"
        )

    with open(
        SEASON_CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        raw = (
            yaml.safe_load(f)
            or {}
        )

    if not isinstance(
        raw,
        dict,
    ):
        raise ValueError(
            f"{SEASON_CONFIG_PATH} must "
            f"contain a top-level mapping"
        )

    required_fields = (
        "start_month",
        "start_day",
        "end_month",
        "end_day",
    )

    config: dict[
        str,
        dict[str, int],
    ] = {}

    for league in SUPPORTED_LEAGUES:
        row = raw.get(
            league
        )

        if not isinstance(
            row,
            dict,
        ):
            raise ValueError(
                f"Missing season configuration "
                f"for league={league}"
            )

        values: dict[
            str,
            int,
        ] = {}

        for field in required_fields:
            if field not in row:
                raise ValueError(
                    f"Missing {league}.{field} "
                    f"in {SEASON_CONFIG_PATH}"
                )

            try:
                values[
                    field
                ] = int(
                    row[
                        field
                    ]
                )

            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    f"Invalid {league}.{field}: "
                    f"{row[field]!r}"
                ) from exc

        try:
            datetime(
                2000,
                values[
                    "start_month"
                ],
                values[
                    "start_day"
                ],
            )

            datetime(
                2000,
                values[
                    "end_month"
                ],
                values[
                    "end_day"
                ],
            )

        except ValueError as exc:
            raise ValueError(
                f"Invalid season dates "
                f"for league={league}"
            ) from exc

        config[
            league
        ] = values

    _SEASON_CONFIG_CACHE["value"] = config

    return config


def season_for_game_date(
    league: str,
    game_date: Any,
) -> int | None:
    d = parse_date(
        game_date
    )

    if d is None:
        return None

    key = (
        league
        .strip()
        .lower()
    )

    if (
        key
        not in SUPPORTED_LEAGUES
    ):
        raise ValueError(
            f"Unsupported league "
            f"for season classification: "
            f"{league}"
        )

    season_config = (
        load_season_config()
    )

    cfg = season_config[
        key
    ]

    month_day = (
        d.month,
        d.day,
    )

    start = (
        cfg[
            "start_month"
        ],
        cfg[
            "start_day"
        ],
    )

    end = (
        cfg[
            "end_month"
        ],
        cfg[
            "end_day"
        ],
    )

    if start <= end:
        if (
            start
            <= month_day
            <= end
        ):
            return d.year

        return None

    if month_day >= start:
        return d.year

    if month_day <= end:
        return (
            d.year
            - 1
        )

    return None


def current_season_for_league(
    league: str,
    today: date | None = None,
) -> int | None:
    reference = (
        today
        or local_today()
    )

    return season_for_game_date(
        league,
        reference.isoformat(),
    )


def season_status_for_league(
    current_season: int | None,
) -> str:
    return (
        "in_season"
        if current_season is not None
        else "offseason"
    )


# ============================================================================
# CONFIG
# ============================================================================

def load_model_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing model config: "
            f"{CONFIG_PATH}"
        )

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        cfg = (
            yaml.safe_load(f)
            or {}
        )

    if not isinstance(
        cfg.get("leagues"),
        dict,
    ):
        raise ValueError(
            f"{CONFIG_PATH} must "
            f"contain a top-level "
            f"'leagues' mapping"
        )

    return cfg


def production_prediction_source(
    cfg: dict[str, Any],
) -> str:
    source = str(
        cfg.get(
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


def resolve_bias_rule(league_cfg: dict[str, Any], component: str) -> dict[str, Any]:
    _py_r1000_NONE = 0
    _py_r1000_RETURN = 1
    _py_r1000_BREAK = 2
    _py_r1000_CONTINUE = 3

    def _py_r1000_impl():
        nonlocal component, league_cfg
        bias_cfg: object
        index: object
        method: object
        method_raw: object
        parsed_window: object
        raw_weight: object
        raw_weights: object
        raw_window: object
        raw_windows: object
        rule: object
        shrink_raw: object
        sign_conflict_shrink: object
        value: object
        value_raw: object
        weight: object
        weight_sum: object
        weights: object
        window: object
        windows: object

        def _py_r1000_if_1():
            nonlocal index, parsed_window, raw_weight, raw_weights, raw_window, raw_windows, shrink_raw, sign_conflict_shrink, weight, weight_sum, weights, windows

            def _py_r1000_loop_2():
                nonlocal parsed_window
                parsed_window = positive_int_or_none(raw_window, f'bias.{component}.windows_games[{index}]')
                if parsed_window is None or parsed_window <= 0:
                    raise ValueError(f'bias.{component}.windows_games[{index}] must be > 0')
                windows.append(parsed_window)
                return (_py_r1000_NONE, None)

            def _py_r1000_loop_4():
                nonlocal weight
                weight = to_float(raw_weight)
                if weight is None or weight < 0:
                    raise ValueError(f'bias.{component}.weights[{index}] must be a finite number >= 0')
                weights.append(float(weight))
                return (_py_r1000_NONE, None)

            def _py_r1000_chunk_6():
                nonlocal index, raw_weights, raw_window, raw_windows, windows

                def _py_r1000_loop_7():
                    _py_r1000_result_3 = _py_r1000_loop_2()
                    if _py_r1000_result_3[0] == _py_r1000_RETURN:
                        return _py_r1000_result_3
                    if _py_r1000_result_3[0] == _py_r1000_BREAK:
                        return (_py_r1000_BREAK, None)
                    if _py_r1000_result_3[0] == _py_r1000_CONTINUE:
                        return (_py_r1000_CONTINUE, None)
                    return (_py_r1000_NONE, None)
                raw_windows = rule.get('windows_games')
                raw_weights = rule.get('weights')
                if not isinstance(raw_windows, list) or not raw_windows:
                    raise ValueError(f"bias.{component}.windows_games must be a non-empty list for method='regime_aware'")
                windows = []
                for index, raw_window in enumerate(raw_windows):
                    _py_r1000_result_8 = _py_r1000_loop_7()
                    if _py_r1000_result_8[0] == _py_r1000_RETURN:
                        return _py_r1000_result_8
                    if _py_r1000_result_8[0] == _py_r1000_BREAK:
                        break
                    if _py_r1000_result_8[0] == _py_r1000_CONTINUE:
                        continue
                if len(set(windows)) != len(windows):
                    raise ValueError(f'bias.{component}.windows_games must contain unique windows')
                return (_py_r1000_NONE, None)

            def _py_r1000_chunk_9():
                nonlocal index, raw_weight, weight_sum, weights

                def _py_r1000_loop_10():
                    _py_r1000_result_5 = _py_r1000_loop_4()
                    if _py_r1000_result_5[0] == _py_r1000_RETURN:
                        return _py_r1000_result_5
                    if _py_r1000_result_5[0] == _py_r1000_BREAK:
                        return (_py_r1000_BREAK, None)
                    if _py_r1000_result_5[0] == _py_r1000_CONTINUE:
                        return (_py_r1000_CONTINUE, None)
                    return (_py_r1000_NONE, None)
                if windows != sorted(windows):
                    raise ValueError(f'bias.{component}.windows_games must be sorted ascending')
                if not isinstance(raw_weights, list) or len(raw_weights) != len(windows):
                    raise ValueError(f'bias.{component}.weights must contain exactly one weight for each configured window')
                weights = []
                for index, raw_weight in enumerate(raw_weights):
                    _py_r1000_result_11 = _py_r1000_loop_10()
                    if _py_r1000_result_11[0] == _py_r1000_RETURN:
                        return _py_r1000_result_11
                    if _py_r1000_result_11[0] == _py_r1000_BREAK:
                        break
                    if _py_r1000_result_11[0] == _py_r1000_CONTINUE:
                        continue
                weight_sum = sum(weights)
                return (_py_r1000_NONE, None)

            def _py_r1000_chunk_12():
                nonlocal shrink_raw, sign_conflict_shrink, weights
                if weight_sum <= 0:
                    raise ValueError(f'bias.{component}.weights must sum to > 0')
                weights = [weight / weight_sum for weight in weights]
                shrink_raw = rule.get('sign_conflict_shrink')
                sign_conflict_shrink = to_float(shrink_raw)
                if sign_conflict_shrink is None or sign_conflict_shrink < 0 or sign_conflict_shrink > 1:
                    raise ValueError(f'bias.{component}.sign_conflict_shrink must be between 0 and 1')
                if window is not None:
                    raise ValueError(f"bias.{component}.window_games must be null/omitted for method='regime_aware'")
                return (_py_r1000_NONE, None)
            for _py_r1000_block_13 in (_py_r1000_chunk_6, _py_r1000_chunk_9, _py_r1000_chunk_12):
                _py_r1000_result_14 = _py_r1000_block_13()
                if _py_r1000_result_14[0] != _py_r1000_NONE:
                    return _py_r1000_result_14
            return (_py_r1000_NONE, None)

        def _py_r1000_if_16():
            nonlocal value
            value = to_float(value_raw)
            if value is None:
                raise ValueError(f'bias.{component}.value must be numeric; got {value_raw!r}')
            return (_py_r1000_NONE, None)
        bias_cfg = league_cfg.get('bias') or {}
        rule = bias_cfg.get(component)
        if rule is None:
            return (_py_r1000_RETURN, {'method': None, 'window_games': None, 'windows_games': None, 'weights': None, 'sign_conflict_shrink': None, 'value': None})
        if not isinstance(rule, dict):
            raise ValueError(f'bias.{component} must be a mapping or null')
        method_raw = rule.get('method')
        method = None if method_raw is None else str(method_raw).strip().lower()
        if method in {'', 'null'}:
            method = None
        window = positive_int_or_none(rule.get('window_games'), f'bias.{component}.window_games')
        windows: list[int] | None = None
        weights: list[float] | None = None
        sign_conflict_shrink: float | None = None
        if method == 'regime_aware':
            _py_r1000_result_15 = _py_r1000_if_1()
            if _py_r1000_result_15[0] != _py_r1000_NONE:
                return _py_r1000_result_15
        value_raw = rule.get('value')
        value = None
        if value_raw not in (None, ''):
            _py_r1000_result_17 = _py_r1000_if_16()
            if _py_r1000_result_17[0] != _py_r1000_NONE:
                return _py_r1000_result_17
        if method in {'rolling', 'regime_aware', 'none'} and value is not None:
            raise ValueError(f'bias.{component}.value must be null for method={method!r}')
        if method == 'fixed' and window is not None:
            raise ValueError(f"bias.{component}.window_games must be null/omitted for method='fixed'")
        return (_py_r1000_RETURN, {'method': method, 'window_games': window, 'windows_games': windows, 'weights': weights, 'sign_conflict_shrink': sign_conflict_shrink, 'value': value})
    _py_r1000_outcome = _py_r1000_impl()
    if _py_r1000_outcome[0] == _py_r1000_RETURN:
        return _py_r1000_outcome[1]


# ============================================================================
# HISTORICAL COMPLETED GAMES
# ============================================================================

def historical_stats_template() -> dict[
    str,
    int,
]:
    return {
        "historical_files_scanned": 0,
        "historical_rows": 0,
        "historical_usable_games": 0,
        "historical_incomplete_rows": 0,
        "historical_rows_invalid_date": 0,
        "historical_invalid_bias_flag_rows": 0,
        "historical_rows_bias_reversed": 0,
        "historical_rows_bias_reversed_per_game": 0,
        "historical_rows_bias_reversed_legacy": 0,
        "historical_rows_raw_unadjusted": 0,
        "historical_unreversible_bias_rows": 0,
    }


def reverse_adjusted_projection(
    adjusted_home: float,
    adjusted_away: float,
    adjusted_total: float,
    margin_bias: float,
    total_bias: float,
) -> tuple[
    float,
    float,
    float,
]:
    raw_home = (
        adjusted_home
        + (
            margin_bias
            / 2.0
        )
        + (
            total_bias
            / 2.0
        )
    )

    raw_away = (
        adjusted_away
        - (
            margin_bias
            / 2.0
        )
        + (
            total_bias
            / 2.0
        )
    )

    raw_total = (
        adjusted_total
        + total_bias
    )

    return (
        raw_home,
        raw_away,
        raw_total,
    )


def load_historical_completed_games(league: str) -> tuple[list[CompletedGame], dict[str, int], list[str]]:
    _py_r1000_NONE = 0
    _py_r1000_RETURN = 1
    _py_r1000_BREAK = 2
    _py_r1000_CONTINUE = 3

    def _py_r1000_impl():
        nonlocal league
        away_proj: object
        away_score: object
        away_team: object
        bias_flag: object
        fatal_errors: object
        fieldnames: object
        game_date: object
        games: object
        home_proj: object
        home_score: object
        home_team: object
        legacy: object
        message: object
        path: object
        per_game_margin: object
        per_game_total: object
        raw_away: object
        raw_home: object
        raw_total: object
        reversal_margin: object
        reversal_total: object
        row: object
        row_number: object
        rows: object
        season: object
        stats: object
        total_proj: object

        def _py_r1000_loop_1():
            nonlocal away_proj, away_score, away_team, bias_flag, fieldnames, game_date, home_proj, home_score, home_team, legacy, message, per_game_margin, per_game_total, raw_away, raw_home, raw_total, reversal_margin, reversal_total, row, row_number, rows, total_proj

            def _py_r1000_loop_2():
                nonlocal away_proj, away_score, away_team, bias_flag, game_date, home_proj, home_score, home_team, legacy, message, per_game_margin, per_game_total, raw_away, raw_home, raw_total, reversal_margin, reversal_total, total_proj

                def _py_r1000_else_3():
                    nonlocal legacy, message, per_game_margin, per_game_total, raw_away, raw_home, raw_total, reversal_margin, reversal_total

                    def _py_r1000_else_4():
                        nonlocal legacy, message, reversal_margin, reversal_total
                        legacy = LEGACY_HISTORICAL_BIAS.get((league.lower(), season))
                        if legacy is None:
                            stats['historical_unreversible_bias_rows'] += 1
                            message = f'{league_upper(league)} historical {path.name} row {row_number} has bias_applied=1 but does not contain valid per-game margin_bias and total_bias, and no legacy fallback exists for this league/season'
                            fatal_errors.append(message)
                            log(message, 'ERROR')
                            return (_py_r1000_CONTINUE, None)
                        reversal_margin = float(legacy['margin'])
                        reversal_total = float(legacy['total'])
                        stats['historical_rows_bias_reversed_legacy'] += 1
                        return (_py_r1000_NONE, None)
                    per_game_margin = to_float(row.get('margin_bias'))
                    per_game_total = to_float(row.get('total_bias'))
                    if per_game_margin is not None and per_game_total is not None:
                        reversal_margin = per_game_margin
                        reversal_total = per_game_total
                        stats['historical_rows_bias_reversed_per_game'] += 1
                    else:
                        _py_r1000_result_5 = _py_r1000_else_4()
                        if _py_r1000_result_5[0] != _py_r1000_NONE:
                            return _py_r1000_result_5
                    raw_home, raw_away, raw_total = reverse_adjusted_projection(home_proj, away_proj, total_proj, reversal_margin, reversal_total)
                    stats['historical_rows_bias_reversed'] += 1
                    return (_py_r1000_NONE, None)

                def _py_r1000_chunk_7():
                    nonlocal away_proj, away_score, away_team, game_date, home_proj, home_score, home_team, total_proj
                    stats['historical_rows'] += 1
                    game_date = normalize_date(row.get('game_date'))
                    if not game_date:
                        stats['historical_rows_invalid_date'] += 1
                        return (_py_r1000_CONTINUE, None)
                    home_team = str(row.get('home_team') or '').strip()
                    away_team = str(row.get('away_team') or '').strip()
                    home_proj = to_float(row.get('home_projected_points'))
                    away_proj = to_float(row.get('away_projected_points'))
                    total_proj = to_float(row.get('total_projected_points'))
                    home_score = to_float(row.get('home_score'))
                    away_score = to_float(row.get('away_score'))
                    if total_proj is None and home_proj is not None and (away_proj is not None):
                        total_proj = home_proj + away_proj
                    return (_py_r1000_NONE, None)

                def _py_r1000_chunk_8():
                    nonlocal bias_flag
                    if not home_team or not away_team or home_proj is None or (away_proj is None) or (total_proj is None) or (home_score is None) or (away_score is None):
                        stats['historical_incomplete_rows'] += 1
                        return (_py_r1000_CONTINUE, None)
                    bias_flag = parse_bias_flag(row.get('bias_applied'))
                    return (_py_r1000_NONE, None)

                def _py_r1000_chunk_9():
                    nonlocal raw_away, raw_home, raw_total

                    def _py_r1000_else_10():
                        _py_r1000_result_6 = _py_r1000_else_3()
                        if _py_r1000_result_6[0] != _py_r1000_NONE:
                            return _py_r1000_result_6
                        return (_py_r1000_NONE, None)
                    if bias_flag is None:
                        stats['historical_invalid_bias_flag_rows'] += 1
                        log(f"{league_upper(league)} | HISTORICAL INVALID BIAS FLAG | file={path.name} row={row_number} value={row.get('bias_applied')!r}", 'WARN')
                        return (_py_r1000_CONTINUE, None)
                    raw_home = home_proj
                    raw_away = away_proj
                    raw_total = total_proj
                    if bias_flag == 0:
                        stats['historical_rows_raw_unadjusted'] += 1
                    else:
                        _py_r1000_result_11 = _py_r1000_else_10()
                        if _py_r1000_result_11[0] != _py_r1000_NONE:
                            return _py_r1000_result_11
                    games.append(CompletedGame(league=league, game_id=canonical_game_id(row.get('game_id')), game_date=game_date, game_time=str(row.get('game_time') or '').strip(), home_team=home_team, away_team=away_team, home_projected_points=raw_home, away_projected_points=raw_away, total_projected_points=raw_total, home_score=home_score, away_score=away_score, source=repo_relative(path), source_priority=1))
                    stats['historical_usable_games'] += 1
                    return (_py_r1000_NONE, None)
                for _py_r1000_block_12 in (_py_r1000_chunk_7, _py_r1000_chunk_8, _py_r1000_chunk_9):
                    _py_r1000_result_13 = _py_r1000_block_12()
                    if _py_r1000_result_13[0] != _py_r1000_NONE:
                        return _py_r1000_result_13
                return (_py_r1000_NONE, None)
            stats['historical_files_scanned'] += 1
            fieldnames, rows = read_csv_rows(path)
            require_columns(path, fieldnames, HISTORICAL_REQUIRED)
            for row_number, row in enumerate(rows, start=2):
                _py_r1000_result_14 = _py_r1000_loop_2()
                if _py_r1000_result_14[0] == _py_r1000_RETURN:
                    return _py_r1000_result_14
                if _py_r1000_result_14[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_14[0] == _py_r1000_CONTINUE:
                    continue
            log(f'{league_upper(league)} | HISTORICAL | {path.name} | rows={len(rows)}')
            return (_py_r1000_NONE, None)
        games: list[CompletedGame] = []
        stats = historical_stats_template()
        fatal_errors: list[str] = []
        for season, path in historical_files_for_league(league):
            _py_r1000_result_15 = _py_r1000_loop_1()
            if _py_r1000_result_15[0] == _py_r1000_RETURN:
                return _py_r1000_result_15
            if _py_r1000_result_15[0] == _py_r1000_BREAK:
                break
            if _py_r1000_result_15[0] == _py_r1000_CONTINUE:
                continue
        return (_py_r1000_RETURN, (games, stats, fatal_errors))
    _py_r1000_outcome = _py_r1000_impl()
    if _py_r1000_outcome[0] == _py_r1000_RETURN:
        return _py_r1000_outcome[1]


# ============================================================================
# CURRENT-SEASON INPUT LOADERS
# ============================================================================

def load_current_prediction_rows(
    league: str,
    current_season: int | None,
) -> tuple[
    list[dict[str, str]],
    dict[str, int],
]:
    accepted: list[
        dict[str, str]
    ] = []

    stats = {
        "prediction_files_scanned": 0,
        "prediction_rows_scanned": 0,
        "prediction_rows_current_season": 0,
        "prediction_rows_ignored_not_current_season": 0,
        "prediction_rows_invalid_date": 0,
    }

    for path in prediction_files_for_league(
        league
    ):
        stats[
            "prediction_files_scanned"
        ] += 1

        (
            fieldnames,
            rows,
        ) = read_csv_rows(
            path
        )

        require_columns(
            path,
            fieldnames,
            PREDICTION_REQUIRED,
        )

        for (
            row_number,
            row,
        ) in enumerate(
            rows,
            start=2,
        ):
            stats[
                "prediction_rows_scanned"
            ] += 1

            game_date = normalize_date(
                row.get(
                    "game_date"
                )
            )

            if not game_date:
                stats[
                    "prediction_rows_invalid_date"
                ] += 1

                continue

            row_season = (
                season_for_game_date(
                    league,
                    game_date,
                )
            )

            if (
                current_season is None
                or row_season
                != current_season
            ):
                stats[
                    "prediction_rows_ignored_not_current_season"
                ] += 1

                continue

            copy = dict(
                row
            )

            copy[
                "game_date"
            ] = game_date

            copy[
                "_source_file"
            ] = repo_relative(
                path
            )

            copy[
                "_source_row"
            ] = str(
                row_number
            )

            accepted.append(
                copy
            )

            stats[
                "prediction_rows_current_season"
            ] += 1

    return (
        accepted,
        stats,
    )


def load_current_final_rows(
    league: str,
    current_season: int | None,
) -> tuple[
    list[dict[str, str]],
    dict[str, int],
]:
    accepted: list[
        dict[str, str]
    ] = []

    stats = {
        "final_files_scanned": 0,
        "final_rows_scanned": 0,
        "final_rows_current_season": 0,
        "final_rows_ignored_not_current_season": 0,
        "final_rows_invalid_date": 0,
    }

    for path in final_files_for_league(
        league
    ):
        stats[
            "final_files_scanned"
        ] += 1

        (
            fieldnames,
            rows,
        ) = read_csv_rows(
            path
        )

        require_columns(
            path,
            fieldnames,
            FINAL_REQUIRED,
        )

        for (
            row_number,
            row,
        ) in enumerate(
            rows,
            start=2,
        ):
            stats[
                "final_rows_scanned"
            ] += 1

            game_date = normalize_date(
                row.get(
                    "game_date"
                )
            )

            if not game_date:
                stats[
                    "final_rows_invalid_date"
                ] += 1

                continue

            row_season = (
                season_for_game_date(
                    league,
                    game_date,
                )
            )

            if (
                current_season is None
                or row_season
                != current_season
            ):
                stats[
                    "final_rows_ignored_not_current_season"
                ] += 1

                continue

            copy = dict(
                row
            )

            copy[
                "game_date"
            ] = game_date

            copy[
                "_source_file"
            ] = repo_relative(
                path
            )

            copy[
                "_source_row"
            ] = str(
                row_number
            )

            copy[
                "_uid"
            ] = str(
                len(
                    accepted
                )
            )

            accepted.append(
                copy
            )

            stats[
                "final_rows_current_season"
            ] += 1

    return (
        accepted,
        stats,
    )


# ============================================================================
# PREDICTION DUPLICATE HANDLING
# ============================================================================

def build_prediction_index_for_key(
    rows: list[
        dict[str, str]
    ],
    key_getter,
) -> tuple[
    dict[
        str,
        dict[str, str],
    ],
    set[str],
    int,
    int,
]:
    groups: dict[
        str,
        list[
            dict[str, str]
        ],
    ] = {}

    for row in rows:
        key = key_getter(
            row
        )

        if key:
            groups.setdefault(
                key,
                [],
            ).append(
                row
            )

    index: dict[
        str,
        dict[str, str],
    ] = {}

    ambiguous: set[
        str
    ] = set()

    identical_extra_rows = 0
    conflicting_keys = 0

    for (
        key,
        group,
    ) in groups.items():
        if len(
            group
        ) == 1:
            index[
                key
            ] = group[0]

            continue

        signatures = {
            prediction_signature(
                row
            )
            for row in group
        }

        if len(
            signatures
        ) == 1:
            identical_extra_rows += (
                len(group)
                - 1
            )

            index[
                key
            ] = sorted(
                group,
                key=lambda sort_row: (
                    sort_row.get(
                        "_source_file",
                        "",
                    ),
                    sort_row.get(
                        "_source_row",
                        "",
                    ),
                ),
            )[0]

        else:
            conflicting_keys += 1

            ambiguous.add(
                key
            )

    return (
        index,
        ambiguous,
        identical_extra_rows,
        conflicting_keys,
    )


def build_prediction_indexes(
    rows: list[
        dict[str, str]
    ],
) -> tuple[
    dict[
        str,
        dict[str, str],
    ],
    dict[
        str,
        dict[str, str],
    ],
    set[str],
    set[str],
    dict[str, int],
]:
    (
        by_id,
        ambiguous_ids,
        duplicate_ids,
        conflicting_ids,
    ) = build_prediction_index_for_key(
        rows,
        lambda row: canonical_game_id(
            row.get(
                "game_id"
            )
        ),
    )

    (
        by_composite,
        ambiguous_composites,
        duplicate_composites,
        conflicting_composites,
    ) = build_prediction_index_for_key(
        rows,
        lambda row: composite_key(
            row.get(
                "game_date"
            ),
            row.get(
                "home_team"
            ),
            row.get(
                "away_team"
            ),
        ),
    )

    stats = {
        "prediction_rows_missing_game_id": sum(
            1
            for row in rows
            if not canonical_game_id(
                row.get(
                    "game_id"
                )
            )
        ),
        "prediction_rows_missing_composite": sum(
            1
            for row in rows
            if not composite_key(
                row.get(
                    "game_date"
                ),
                row.get(
                    "home_team"
                ),
                row.get(
                    "away_team"
                ),
            )
        ),
        "duplicate_prediction_game_ids": duplicate_ids,
        "duplicate_prediction_composites": duplicate_composites,
        "conflicting_prediction_game_ids": conflicting_ids,
        "conflicting_prediction_composites": conflicting_composites,
    }

    return (
        by_id,
        by_composite,
        ambiguous_ids,
        ambiguous_composites,
        stats,
    )


# ============================================================================
# FINAL-SCORE DUPLICATE HANDLING
# ============================================================================

def deduplicate_final_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    _py_r1000_NONE = 0
    _py_r1000_RETURN = 1
    _py_r1000_BREAK = 2
    _py_r1000_CONTINUE = 3

    def _py_r1000_impl():
        nonlocal rows
        comp: object
        composite_groups: object
        deduped: object
        excluded: object
        gid: object
        group: object
        grouped_remaining: object
        id_groups: object
        key: object
        preferred: object
        remaining: object
        row: object
        score_signatures: object
        signatures: object
        stats: object

        def _py_r1000_loop_1():
            nonlocal gid
            gid = canonical_game_id(row.get('game_id'))
            if gid:
                id_groups.setdefault(gid, []).append(row)
            return (_py_r1000_NONE, None)

        def _py_r1000_loop_3():
            nonlocal signatures

            def _py_r1000_else_4():
                stats['conflicting_final_game_ids'] += 1
                excluded.update((row['_uid'] for row in group))
                log(f'FINAL CONFLICTING GAME_ID | game_id={gid} rows={len(group)}', 'WARN')
                return (_py_r1000_NONE, None)
            if len(group) <= 1:
                return (_py_r1000_CONTINUE, None)
            signatures = {final_identity_score_signature(row) for row in group}
            if len(signatures) == 1:
                stats['duplicate_final_game_ids'] += len(group) - 1
            else:
                _py_r1000_result_5 = _py_r1000_else_4()
                if _py_r1000_result_5[0] != _py_r1000_NONE:
                    return _py_r1000_result_5
            return (_py_r1000_NONE, None)

        def _py_r1000_loop_7():
            nonlocal comp
            comp = composite_key(row.get('game_date'), row.get('home_team'), row.get('away_team'))
            if comp:
                composite_groups.setdefault(comp, []).append(row)
            return (_py_r1000_NONE, None)

        def _py_r1000_loop_9():
            nonlocal score_signatures

            def _py_r1000_else_10():
                stats['conflicting_final_composites'] += 1
                excluded.update((row['_uid'] for row in group))
                log(f'FINAL CONFLICTING COMPOSITE | key={comp} rows={len(group)}', 'WARN')
                return (_py_r1000_NONE, None)
            if len(group) <= 1:
                return (_py_r1000_CONTINUE, None)
            score_signatures = {(to_float(row.get('home_score')), to_float(row.get('away_score'))) for row in group}
            if len(score_signatures) == 1:
                stats['duplicate_final_composites'] += len(group) - 1
            else:
                _py_r1000_result_11 = _py_r1000_else_10()
                if _py_r1000_result_11[0] != _py_r1000_NONE:
                    return _py_r1000_result_11
            return (_py_r1000_NONE, None)

        def _py_r1000_loop_13():
            nonlocal comp, gid, key

            def _py_r1000_else_14():
                nonlocal key
                if gid:
                    key = ('game_id', gid)
                else:
                    key = ('row', row['_uid'])
                return (_py_r1000_NONE, None)
            comp = composite_key(row.get('game_date'), row.get('home_team'), row.get('away_team'))
            gid = canonical_game_id(row.get('game_id'))
            if comp:
                key = ('composite', comp)
            else:
                _py_r1000_result_15 = _py_r1000_else_14()
                if _py_r1000_result_15[0] != _py_r1000_NONE:
                    return _py_r1000_result_15
            grouped_remaining.setdefault(key, []).append(row)
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_17():
            nonlocal excluded, id_groups, row, stats

            def _py_r1000_loop_18():
                _py_r1000_result_2 = _py_r1000_loop_1()
                if _py_r1000_result_2[0] == _py_r1000_RETURN:
                    return _py_r1000_result_2
                if _py_r1000_result_2[0] == _py_r1000_BREAK:
                    return (_py_r1000_BREAK, None)
                if _py_r1000_result_2[0] == _py_r1000_CONTINUE:
                    return (_py_r1000_CONTINUE, None)
                return (_py_r1000_NONE, None)
            stats = {'duplicate_final_game_ids': 0, 'duplicate_final_composites': 0, 'conflicting_final_game_ids': 0, 'conflicting_final_composites': 0, 'final_duplicate_rows_removed': 0, 'final_conflicting_rows_excluded': 0}
            excluded = set()
            id_groups = {}
            for row in rows:
                _py_r1000_result_19 = _py_r1000_loop_18()
                if _py_r1000_result_19[0] == _py_r1000_RETURN:
                    return _py_r1000_result_19
                if _py_r1000_result_19[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_19[0] == _py_r1000_CONTINUE:
                    continue
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_20():
            nonlocal composite_groups, gid, group

            def _py_r1000_loop_21():
                _py_r1000_result_6 = _py_r1000_loop_3()
                if _py_r1000_result_6[0] == _py_r1000_RETURN:
                    return _py_r1000_result_6
                if _py_r1000_result_6[0] == _py_r1000_BREAK:
                    return (_py_r1000_BREAK, None)
                if _py_r1000_result_6[0] == _py_r1000_CONTINUE:
                    return (_py_r1000_CONTINUE, None)
                return (_py_r1000_NONE, None)
            for gid, group in id_groups.items():
                _py_r1000_result_22 = _py_r1000_loop_21()
                if _py_r1000_result_22[0] == _py_r1000_RETURN:
                    return _py_r1000_result_22
                if _py_r1000_result_22[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_22[0] == _py_r1000_CONTINUE:
                    continue
            composite_groups = {}
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_23():
            nonlocal row

            def _py_r1000_loop_24():
                _py_r1000_result_8 = _py_r1000_loop_7()
                if _py_r1000_result_8[0] == _py_r1000_RETURN:
                    return _py_r1000_result_8
                if _py_r1000_result_8[0] == _py_r1000_BREAK:
                    return (_py_r1000_BREAK, None)
                if _py_r1000_result_8[0] == _py_r1000_CONTINUE:
                    return (_py_r1000_CONTINUE, None)
                return (_py_r1000_NONE, None)
            for row in rows:
                _py_r1000_result_25 = _py_r1000_loop_24()
                if _py_r1000_result_25[0] == _py_r1000_RETURN:
                    return _py_r1000_result_25
                if _py_r1000_result_25[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_25[0] == _py_r1000_CONTINUE:
                    continue
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_26():
            nonlocal comp, group, grouped_remaining, remaining

            def _py_r1000_loop_27():
                _py_r1000_result_12 = _py_r1000_loop_9()
                if _py_r1000_result_12[0] == _py_r1000_RETURN:
                    return _py_r1000_result_12
                if _py_r1000_result_12[0] == _py_r1000_BREAK:
                    return (_py_r1000_BREAK, None)
                if _py_r1000_result_12[0] == _py_r1000_CONTINUE:
                    return (_py_r1000_CONTINUE, None)
                return (_py_r1000_NONE, None)
            for comp, group in composite_groups.items():
                _py_r1000_result_28 = _py_r1000_loop_27()
                if _py_r1000_result_28[0] == _py_r1000_RETURN:
                    return _py_r1000_result_28
                if _py_r1000_result_28[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_28[0] == _py_r1000_CONTINUE:
                    continue
            stats['final_conflicting_rows_excluded'] = len(excluded)
            remaining = [row for row in rows if row['_uid'] not in excluded]
            grouped_remaining = {}
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_29():
            nonlocal deduped, group, preferred, row

            def _py_r1000_loop_30():
                _py_r1000_result_16 = _py_r1000_loop_13()
                if _py_r1000_result_16[0] == _py_r1000_RETURN:
                    return _py_r1000_result_16
                if _py_r1000_result_16[0] == _py_r1000_BREAK:
                    return (_py_r1000_BREAK, None)
                if _py_r1000_result_16[0] == _py_r1000_CONTINUE:
                    return (_py_r1000_CONTINUE, None)
                return (_py_r1000_NONE, None)
            for row in remaining:
                _py_r1000_result_31 = _py_r1000_loop_30()
                if _py_r1000_result_31[0] == _py_r1000_RETURN:
                    return _py_r1000_result_31
                if _py_r1000_result_31[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_31[0] == _py_r1000_CONTINUE:
                    continue
            deduped = []
            for group in grouped_remaining.values():
                preferred = sorted(group, key=lambda sort_row: (0 if canonical_game_id(sort_row.get('game_id')) else 1, sort_row.get('_source_file', ''), sort_row.get('_source_row', '')))[0]
                deduped.append(preferred)
                stats['final_duplicate_rows_removed'] += len(group) - 1
            deduped.sort(key=lambda sort_row: (parse_game_datetime(sort_row.get('game_date'), ''), normalize_text(sort_row.get('home_team')), normalize_text(sort_row.get('away_team')), canonical_game_id(sort_row.get('game_id'))))
            return (_py_r1000_RETURN, (deduped, stats))
        for _py_r1000_block_32 in (_py_r1000_chunk_17, _py_r1000_chunk_20, _py_r1000_chunk_23, _py_r1000_chunk_26, _py_r1000_chunk_29):
            _py_r1000_result_33 = _py_r1000_block_32()
            if _py_r1000_result_33[0] != _py_r1000_NONE:
                return _py_r1000_result_33
        return (_py_r1000_NONE, None)
    _py_r1000_outcome = _py_r1000_impl()
    if _py_r1000_outcome[0] == _py_r1000_RETURN:
        return _py_r1000_outcome[1]


# ============================================================================
# CURRENT PREDICTION / FINAL MATCHING
# ============================================================================

def historical_coverage_sets(
    historical_games: list[
        CompletedGame
    ],
) -> tuple[
    set[str],
    set[str],
]:
    game_ids = {
        canonical_game_id(
            game.game_id
        )
        for game in historical_games
        if canonical_game_id(
            game.game_id
        )
    }

    composites = {
        game.composite
        for game in historical_games
        if game.composite
    }

    return (
        game_ids,
        composites,
    )


def identities_agree(
    prediction: dict[str, str],
    final: dict[str, str],
) -> bool | None:
    pred_date = normalize_date(
        prediction.get(
            "game_date"
        )
    )

    final_date = normalize_date(
        final.get(
            "game_date"
        )
    )

    pred_home = normalize_text(
        prediction.get(
            "home_team"
        )
    )

    final_home = normalize_text(
        final.get(
            "home_team"
        )
    )

    pred_away = normalize_text(
        prediction.get(
            "away_team"
        )
    )

    final_away = normalize_text(
        final.get(
            "away_team"
        )
    )

    values = (
        pred_date,
        final_date,
        pred_home,
        final_home,
        pred_away,
        final_away,
    )

    if any(
        not value
        for value in values
    ):
        return None

    return (
        pred_date
        == final_date
        and pred_home
        == final_home
        and pred_away
        == final_away
    )


def load_current_completed_games(league: str, current_season: int | None, historical_games: list[CompletedGame]) -> tuple[list[CompletedGame], dict[str, Any]]:
    _py_r1000_NONE = 0
    _py_r1000_RETURN = 1
    _py_r1000_BREAK = 2
    _py_r1000_CONTINUE = 3

    def _py_r1000_impl():
        nonlocal current_season, historical_games, league
        ambiguous_composites: object
        ambiguous_ids: object
        away_proj: object
        away_score: object
        away_team: object
        blocked_by_ambiguity: object
        candidate: object
        comp: object
        final: object
        final_duplicate_stats: object
        final_load_stats: object
        finals: object
        finals_raw: object
        game_date: object
        games: object
        gid: object
        historical_composites: object
        historical_ids: object
        home_proj: object
        home_score: object
        home_team: object
        identity_result: object
        match_method: object
        pred_by_composite: object
        pred_by_id: object
        prediction: object
        prediction_duplicate_stats: object
        prediction_load_stats: object
        predictions: object
        stats: object
        total_proj: object

        def _py_r1000_loop_1():
            nonlocal away_proj, away_score, away_team, blocked_by_ambiguity, candidate, comp, game_date, gid, home_proj, home_score, home_team, identity_result, match_method, prediction, total_proj

            def _py_r1000_if_2():
                nonlocal candidate, identity_result, match_method, prediction
                candidate = pred_by_id[gid]
                identity_result = identities_agree(candidate, final)
                if identity_result is None:
                    stats['invalid_current_matches'] += 1
                    return (_py_r1000_CONTINUE, None)
                if not identity_result:
                    stats['game_id_identity_mismatches'] += 1
                    log(f"{league_upper(league)} | GAME_ID IDENTITY MISMATCH | game_id={gid} | final={final.get('game_date')} {final.get('home_team')} vs {final.get('away_team')} | prediction={candidate.get('game_date')} {candidate.get('home_team')} vs {candidate.get('away_team')}", 'WARN')
                    return (_py_r1000_CONTINUE, None)
                prediction = candidate
                match_method = 'game_id'
                return (_py_r1000_NONE, None)

            def _py_r1000_else_4():
                nonlocal blocked_by_ambiguity
                if gid and gid in ambiguous_ids:
                    blocked_by_ambiguity = True
                return (_py_r1000_NONE, None)

            def _py_r1000_if_6():
                nonlocal blocked_by_ambiguity, match_method, prediction

                def _py_r1000_else_7():
                    nonlocal blocked_by_ambiguity
                    if comp in ambiguous_composites:
                        blocked_by_ambiguity = True
                    return (_py_r1000_NONE, None)
                if comp in pred_by_composite:
                    prediction = pred_by_composite[comp]
                    match_method = 'composite'
                else:
                    _py_r1000_result_8 = _py_r1000_else_7()
                    if _py_r1000_result_8[0] != _py_r1000_NONE:
                        return _py_r1000_result_8
                return (_py_r1000_NONE, None)

            def _py_r1000_if_10():
                if blocked_by_ambiguity:
                    stats['ambiguous_prediction_matches'] += 1
                else:
                    stats['true_unmatched_current_finals'] += 1
                return (_py_r1000_CONTINUE, None)

            def _py_r1000_chunk_12():
                nonlocal away_score, blocked_by_ambiguity, comp, gid, home_score, match_method, prediction
                gid = canonical_game_id(final.get('game_id'))
                comp = composite_key(final.get('game_date'), final.get('home_team'), final.get('away_team'))
                if gid and gid in historical_ids or (comp and comp in historical_composites):
                    stats['finals_already_covered_by_historical'] += 1
                    return (_py_r1000_CONTINUE, None)
                home_score = to_float(final.get('home_score'))
                away_score = to_float(final.get('away_score'))
                if home_score is None or away_score is None:
                    stats['invalid_current_matches'] += 1
                    return (_py_r1000_CONTINUE, None)
                prediction = None
                match_method = None
                blocked_by_ambiguity = False
                return (_py_r1000_NONE, None)

            def _py_r1000_chunk_13():

                def _py_r1000_if_14():
                    _py_r1000_result_3 = _py_r1000_if_2()
                    if _py_r1000_result_3[0] != _py_r1000_NONE:
                        return _py_r1000_result_3
                    return (_py_r1000_NONE, None)

                def _py_r1000_else_16():
                    _py_r1000_result_5 = _py_r1000_else_4()
                    if _py_r1000_result_5[0] != _py_r1000_NONE:
                        return _py_r1000_result_5
                    return (_py_r1000_NONE, None)

                def _py_r1000_if_18():
                    _py_r1000_result_9 = _py_r1000_if_6()
                    if _py_r1000_result_9[0] != _py_r1000_NONE:
                        return _py_r1000_result_9
                    return (_py_r1000_NONE, None)
                if gid and gid in pred_by_id:
                    _py_r1000_result_15 = _py_r1000_if_14()
                    if _py_r1000_result_15[0] != _py_r1000_NONE:
                        return _py_r1000_result_15
                else:
                    _py_r1000_result_17 = _py_r1000_else_16()
                    if _py_r1000_result_17[0] != _py_r1000_NONE:
                        return _py_r1000_result_17
                if prediction is None and comp:
                    _py_r1000_result_19 = _py_r1000_if_18()
                    if _py_r1000_result_19[0] != _py_r1000_NONE:
                        return _py_r1000_result_19
                return (_py_r1000_NONE, None)

            def _py_r1000_chunk_20():
                nonlocal away_proj, away_team, game_date, home_proj, home_team, total_proj

                def _py_r1000_if_21():
                    _py_r1000_result_11 = _py_r1000_if_10()
                    if _py_r1000_result_11[0] != _py_r1000_NONE:
                        return _py_r1000_result_11
                    return (_py_r1000_NONE, None)
                if prediction is None:
                    _py_r1000_result_22 = _py_r1000_if_21()
                    if _py_r1000_result_22[0] != _py_r1000_NONE:
                        return _py_r1000_result_22
                home_proj = to_float(prediction.get('home_projected_points'))
                away_proj = to_float(prediction.get('away_projected_points'))
                total_proj = normalized_prediction_total(prediction)
                if home_proj is None or away_proj is None or total_proj is None:
                    stats['invalid_current_matches'] += 1
                    return (_py_r1000_CONTINUE, None)
                game_date = normalize_date(final.get('game_date'))
                home_team = str(final.get('home_team') or '').strip()
                away_team = str(final.get('away_team') or '').strip()
                return (_py_r1000_NONE, None)

            def _py_r1000_chunk_23():
                if not game_date or not home_team or (not away_team):
                    stats['invalid_current_matches'] += 1
                    return (_py_r1000_CONTINUE, None)
                games.append(CompletedGame(league=league, game_id=gid or canonical_game_id(prediction.get('game_id')), game_date=game_date, game_time=str(prediction.get('game_time') or '').strip(), home_team=home_team, away_team=away_team, home_projected_points=home_proj, away_projected_points=away_proj, total_projected_points=total_proj, home_score=home_score, away_score=away_score, source=f"{prediction.get('_source_file', '')} + {final.get('_source_file', '')}", source_priority=2))
                if match_method == 'game_id':
                    stats['matched_by_game_id'] += 1
                else:
                    stats['matched_by_composite'] += 1
                return (_py_r1000_NONE, None)
            for _py_r1000_block_24 in (_py_r1000_chunk_12, _py_r1000_chunk_13, _py_r1000_chunk_20, _py_r1000_chunk_23):
                _py_r1000_result_25 = _py_r1000_block_24()
                if _py_r1000_result_25[0] != _py_r1000_NONE:
                    return _py_r1000_result_25
            return (_py_r1000_NONE, None)
        predictions, prediction_load_stats = load_current_prediction_rows(league, current_season)
        finals_raw, final_load_stats = load_current_final_rows(league, current_season)
        finals, final_duplicate_stats = deduplicate_final_rows(finals_raw)
        pred_by_id, pred_by_composite, ambiguous_ids, ambiguous_composites, prediction_duplicate_stats = build_prediction_indexes(predictions)
        historical_ids, historical_composites = historical_coverage_sets(historical_games)
        stats: dict[str, Any] = {'current_season': current_season, 'season_status': season_status_for_league(current_season), **prediction_load_stats, **final_load_stats, **prediction_duplicate_stats, **final_duplicate_stats, 'finals_already_covered_by_historical': 0, 'matched_by_game_id': 0, 'matched_by_composite': 0, 'game_id_identity_mismatches': 0, 'ambiguous_prediction_matches': 0, 'true_unmatched_current_finals': 0, 'invalid_current_matches': 0, 'current_matched_games': 0}
        games: list[CompletedGame] = []
        for final in finals:
            _py_r1000_result_26 = _py_r1000_loop_1()
            if _py_r1000_result_26[0] == _py_r1000_RETURN:
                return _py_r1000_result_26
            if _py_r1000_result_26[0] == _py_r1000_BREAK:
                break
            if _py_r1000_result_26[0] == _py_r1000_CONTINUE:
                continue
        stats['current_matched_games'] = len(games)
        return (_py_r1000_RETURN, (games, stats))
    _py_r1000_outcome = _py_r1000_impl()
    if _py_r1000_outcome[0] == _py_r1000_RETURN:
        return _py_r1000_outcome[1]


# ============================================================================
# UNIFIED COMPLETED HISTORY
# ============================================================================

def deduplicate_completed_games(
    games: list[
        CompletedGame
    ],
) -> tuple[
    list[
        CompletedGame
    ],
    int,
]:
    if not games:
        return (
            [],
            0,
        )

    parent = list(
        range(
            len(games)
        )
    )

    def find(
        i: int,
    ) -> int:
        while (
            parent[i]
            != i
        ):
            parent[i] = (
                parent[
                    parent[i]
                ]
            )

            i = parent[i]

        return i

    def union(
        a: int,
        b: int,
    ) -> None:
        root_a = find(
            a
        )

        root_b = find(
            b
        )

        if (
            root_a
            != root_b
        ):
            parent[
                root_b
            ] = root_a

    first_by_id: dict[
        str,
        int,
    ] = {}

    first_by_composite: dict[
        str,
        int,
    ] = {}

    for (
        index,
        game,
    ) in enumerate(
        games
    ):
        gid = canonical_game_id(
            game.game_id
        )

        comp = (
            game.composite
        )

        if gid:
            if (
                gid
                in first_by_id
            ):
                union(
                    index,
                    first_by_id[
                        gid
                    ],
                )

            else:
                first_by_id[
                    gid
                ] = index

        if comp:
            if (
                comp
                in first_by_composite
            ):
                union(
                    index,
                    first_by_composite[
                        comp
                    ],
                )

            else:
                first_by_composite[
                    comp
                ] = index

    groups: dict[
        int,
        list[
            CompletedGame
        ],
    ] = {}

    for (
        index,
        game,
    ) in enumerate(
        games
    ):
        groups.setdefault(
            find(
                index
            ),
            [],
        ).append(
            game
        )

    chosen: list[
        CompletedGame
    ] = []

    for group in groups.values():
        # Current RAW prediction + final-score reconstruction
        # outranks historical combined coverage.
        #
        # Remaining ties use deterministic chronological/source
        # ordering.
        winner = sorted(
            group,
            key=lambda sort_game: (
                sort_game.source_priority,
                sort_game.sort_key,
            ),
        )[-1]

        chosen.append(
            winner
        )

    chosen.sort(
        key=lambda sort_game: (
            sort_game.sort_key
        )
    )

    duplicates_removed = (
        len(games)
        - len(chosen)
    )

    return (
        chosen,
        duplicates_removed,
    )


def build_completed_history(
    league: str,
) -> tuple[
    list[
        CompletedGame
    ],
    dict[str, Any],
    list[str],
]:
    current_season = (
        current_season_for_league(
            league
        )
    )

    (
        historical,
        historical_stats,
        fatal_errors,
    ) = load_historical_completed_games(
        league
    )

    (
        current,
        current_stats,
    ) = load_current_completed_games(
        league,
        current_season,
        historical,
    )

    (
        combined,
        duplicates_removed,
    ) = deduplicate_completed_games(
        historical
        + current
    )

    meta: dict[
        str,
        Any,
    ] = {
        **historical_stats,
        **current_stats,
        "duplicates_removed_from_completed_history": duplicates_removed,
        "unique_completed_games": len(
            combined
        ),
        "first_game_date": (
            combined[0].game_date
            if combined
            else None
        ),
        "last_game_date": (
            combined[-1].game_date
            if combined
            else None
        ),
    }

    log(
        (
            f"{league_upper(league)} | "
            f"HISTORY | "
            f"historical={len(historical)} "
            f"current={len(current)} "
            f"duplicate_completed="
            f"{duplicates_removed} "
            f"unique={len(combined)} "
            f"range="
            f"{meta['first_game_date']}.."
            f"{meta['last_game_date']}"
        )
    )

    return (
        combined,
        meta,
        fatal_errors,
    )


# ============================================================================
# BIAS CALCULATION
# ============================================================================

def component_stub(
    status: str,
    method: str | None,
    window: int | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "method": method,
        "value": None,
        "window_games": window,
        "games_used": 0,
        "first_game_date": None,
        "last_game_date": None,
    }


def calculate_component_bias(league: str, component: str, rule: dict[str, Any], history: list[CompletedGame]) -> dict[str, Any]:
    _py_r1000_NONE = 0
    _py_r1000_RETURN = 1
    _py_r1000_BREAK = 2
    _py_r1000_CONTINUE = 3

    def _py_r1000_impl():
        nonlocal component, history, league, rule
        effective_value: object
        errors: object
        largest_selected: object
        largest_window: object
        method: object
        negative_present: object
        positive_present: object
        result: object
        selected: object
        shrink: object
        sign_conflict: object
        value: object
        weighted_unshrunk: object
        weights: object
        window: object
        window_int: object
        window_means: object
        windows: object

        def _py_r1000_if_1():
            nonlocal result, value
            value = rule.get('value')
            if value is None:
                raise ValueError(f'{league_upper(league)} {component} fixed bias requires bias.{component}.value')
            result = component_stub('ready', 'fixed', None)
            result['value'] = round(float(value), 3)
            return (_py_r1000_RETURN, result)

        def _py_r1000_if_3():
            nonlocal effective_value, errors, largest_selected, largest_window, negative_present, positive_present, selected, shrink, sign_conflict, weighted_unshrunk, weights, window, window_int, window_means, windows

            def _py_r1000_loop_4():
                nonlocal errors, selected, window_int

                def _py_r1000_if_5():
                    nonlocal errors
                    errors = [game.margin_error for game in selected]
                    return (_py_r1000_NONE, None)

                def _py_r1000_else_7():
                    nonlocal errors

                    def _py_r1000_if_8():
                        nonlocal errors
                        errors = [game.total_error for game in selected]
                        return (_py_r1000_NONE, None)
                    if component == 'total':
                        _py_r1000_result_9 = _py_r1000_if_8()
                        if _py_r1000_result_9[0] != _py_r1000_NONE:
                            return _py_r1000_result_9
                    else:
                        raise ValueError(f'Unsupported bias component: {component}')
                    return (_py_r1000_NONE, None)
                window_int = int(window)
                selected = history[-window_int:]
                if component == 'margin':
                    _py_r1000_result_6 = _py_r1000_if_5()
                    if _py_r1000_result_6[0] != _py_r1000_NONE:
                        return _py_r1000_result_6
                else:
                    _py_r1000_result_10 = _py_r1000_else_7()
                    if _py_r1000_result_10[0] != _py_r1000_NONE:
                        return _py_r1000_result_10
                window_means[window_int] = sum(errors) / len(errors)
                return (_py_r1000_NONE, None)

            def _py_r1000_chunk_12():
                nonlocal largest_window, shrink, weights, window_means, windows
                windows = rule.get('windows_games')
                weights = rule.get('weights')
                shrink = rule.get('sign_conflict_shrink')
                if not isinstance(windows, list) or not windows:
                    raise ValueError(f'{league_upper(league)} {component} regime_aware bias requires windows_games')
                if not isinstance(weights, list) or len(weights) != len(windows):
                    raise ValueError(f'{league_upper(league)} {component} regime_aware bias requires one weight per window')
                if shrink is None:
                    raise ValueError(f'{league_upper(league)} {component} regime_aware bias requires sign_conflict_shrink')
                largest_window = max((int(window) for window in windows))
                if len(history) < largest_window:
                    raise ValueError(f'{league_upper(league)} {component} regime_aware bias requires {largest_window} completed games; only {len(history)} unique completed games are available')
                window_means = {}
                return (_py_r1000_NONE, None)

            def _py_r1000_chunk_13():
                nonlocal negative_present, positive_present, weighted_unshrunk, window

                def _py_r1000_loop_14():
                    _py_r1000_result_11 = _py_r1000_loop_4()
                    if _py_r1000_result_11[0] == _py_r1000_RETURN:
                        return _py_r1000_result_11
                    if _py_r1000_result_11[0] == _py_r1000_BREAK:
                        return (_py_r1000_BREAK, None)
                    if _py_r1000_result_11[0] == _py_r1000_CONTINUE:
                        return (_py_r1000_CONTINUE, None)
                    return (_py_r1000_NONE, None)
                for window in windows:
                    _py_r1000_result_15 = _py_r1000_loop_14()
                    if _py_r1000_result_15[0] == _py_r1000_RETURN:
                        return _py_r1000_result_15
                    if _py_r1000_result_15[0] == _py_r1000_BREAK:
                        break
                    if _py_r1000_result_15[0] == _py_r1000_CONTINUE:
                        continue
                weighted_unshrunk = sum((float(weight) * window_means[int(window)] for window, weight in zip(windows, weights)))
                positive_present = any((value > 1e-12 for value in window_means.values()))
                negative_present = any((value < -1e-12 for value in window_means.values()))
                return (_py_r1000_NONE, None)

            def _py_r1000_chunk_16():
                nonlocal effective_value, largest_selected, sign_conflict
                sign_conflict = positive_present and negative_present
                effective_value = weighted_unshrunk * float(shrink) if sign_conflict else weighted_unshrunk
                largest_selected = history[-largest_window:]
                return (_py_r1000_RETURN, {'status': 'ready', 'method': 'regime_aware', 'value': round(effective_value, 3), 'window_games': largest_window, 'windows_games': [int(window) for window in windows], 'weights': [round(float(weight), 6) for weight in weights], 'window_mean_residuals': {str(int(window)): round(window_means[int(window)], 4) for window in windows}, 'unshrunk_weighted_value': round(weighted_unshrunk, 4), 'sign_conflict': sign_conflict, 'sign_conflict_shrink': round(float(shrink), 6), 'regime_status': 'sign_conflict_shrunk' if sign_conflict else 'aligned', 'games_used': largest_window, 'first_game_date': largest_selected[0].game_date, 'last_game_date': largest_selected[-1].game_date, 'mean_error_definition': 'projected_minus_actual'})
            for _py_r1000_block_17 in (_py_r1000_chunk_12, _py_r1000_chunk_13, _py_r1000_chunk_16):
                _py_r1000_result_18 = _py_r1000_block_17()
                if _py_r1000_result_18[0] != _py_r1000_NONE:
                    return _py_r1000_result_18
            return (_py_r1000_NONE, None)

        def _py_r1000_if_20():
            nonlocal errors
            errors = [game.margin_error for game in selected]
            return (_py_r1000_NONE, None)

        def _py_r1000_else_22():
            nonlocal errors

            def _py_r1000_if_23():
                nonlocal errors
                errors = [game.total_error for game in selected]
                return (_py_r1000_NONE, None)
            if component == 'total':
                _py_r1000_result_24 = _py_r1000_if_23()
                if _py_r1000_result_24[0] != _py_r1000_NONE:
                    return _py_r1000_result_24
            else:
                raise ValueError(f'Unsupported bias component: {component}')
            return (_py_r1000_NONE, None)
        method = rule.get('method')
        if method is None:
            return (_py_r1000_RETURN, component_stub('skipped_no_rule', None, None))
        if method == 'none':
            result = component_stub('disabled', 'none', None)
            result['value'] = 0.0
            return (_py_r1000_RETURN, result)
        if method == 'fixed':
            _py_r1000_result_2 = _py_r1000_if_1()
            if _py_r1000_result_2[0] != _py_r1000_NONE:
                return _py_r1000_result_2
        if method == 'regime_aware':
            _py_r1000_result_19 = _py_r1000_if_3()
            if _py_r1000_result_19[0] != _py_r1000_NONE:
                return _py_r1000_result_19
        if method != 'rolling':
            raise ValueError(f'Unsupported {league_upper(league)} {component} bias method {method!r}; supported methods are rolling, regime_aware, fixed, none, or null/missing')
        window = rule.get('window_games')
        if window is None or window <= 0:
            raise ValueError(f'{league_upper(league)} {component} rolling bias requires window_games > 0')
        if len(history) < window:
            raise ValueError(f'{league_upper(league)} {component} rolling bias requires {window} completed games; only {len(history)} unique completed games are available')
        selected = history[-window:]
        if component == 'margin':
            _py_r1000_result_21 = _py_r1000_if_20()
            if _py_r1000_result_21[0] != _py_r1000_NONE:
                return _py_r1000_result_21
        else:
            _py_r1000_result_25 = _py_r1000_else_22()
            if _py_r1000_result_25[0] != _py_r1000_NONE:
                return _py_r1000_result_25
        value = sum(errors) / len(errors)
        return (_py_r1000_RETURN, {'status': 'ready', 'method': 'rolling', 'value': round(value, 3), 'window_games': int(window), 'games_used': len(selected), 'first_game_date': selected[0].game_date, 'last_game_date': selected[-1].game_date, 'mean_error_definition': 'projected_minus_actual'})
    _py_r1000_outcome = _py_r1000_impl()
    if _py_r1000_outcome[0] == _py_r1000_RETURN:
        return _py_r1000_outcome[1]


def calculate_component_safely(
    league: str,
    component: str,
    rule: dict[str, Any],
    history: list[
        CompletedGame
    ],
) -> tuple[
    dict[str, Any],
    bool,
]:
    try:
        return (
            calculate_component_bias(
                league,
                component,
                rule,
                history,
            ),
            True,
        )

    except Exception as exc:
        error_window = rule.get(
            "window_games"
        )

        if (
            error_window is None
            and rule.get(
                "method"
            )
            == "regime_aware"
            and isinstance(
                rule.get(
                    "windows_games"
                ),
                list,
            )
            and rule.get(
                "windows_games"
            )
        ):
            error_window = max(
                int(
                    window
                )
                for window
                in rule[
                    "windows_games"
                ]
            )

        result = component_stub(
            "error",
            rule.get(
                "method"
            ),
            error_window,
        )

        result[
            "error"
        ] = str(
            exc
        )

        return (
            result,
            False,
        )


# ============================================================================
# LEAGUE STATUS / OUTPUT
# ============================================================================

def warning_count_from_history(
    meta: dict[str, Any],
) -> int:
    return sum(
        int(
            meta.get(
                field,
                0,
            )
            or 0
        )
        for field
        in WARNING_HISTORY_FIELDS
    )


def skipped_league_state(
    config_status: str | None,
) -> dict[str, Any]:
    return {
        "status": (
            "skipped_no_bias_rules"
        ),
        "config_status": (
            config_status
        ),
        "margin_bias": component_stub(
            "skipped_no_rule",
            None,
            None,
        ),
        "total_bias": component_stub(
            "skipped_no_rule",
            None,
            None,
        ),
    }


def unsafe_history_component(
    rule: dict,
) -> dict:
    method = rule.get(
        "method"
    )

    window = rule.get(
        "window_games"
    )

    if (
        window is None
        and method
        == "regime_aware"
    ):
        windows = (
            rule.get(
                "windows_games"
            )
            or []
        )

        window = (
            max(
                windows
            )
            if windows
            else None
        )

    return {
        **component_stub(
            "error",
            method,
            window,
        ),
        "error": (
            "Unsafe historical "
            "adjusted rows could "
            "not be reversed"
        ),
    }

def process_league(league: str, league_cfg: dict[str, Any]) -> tuple[dict[str, Any], bool, bool]:
    _py_r1000_NONE = 0
    _py_r1000_RETURN = 1
    _py_r1000_BREAK = 2
    _py_r1000_CONTINUE = 3

    def _py_r1000_impl():
        nonlocal league, league_cfg
        _: object
        config_status: object
        config_status_raw: object
        configured_component_errors: object
        error_text: object
        exc: object
        fatal_history_errors: object
        history: object
        history_meta: object
        history_required: object
        league_state: object
        league_status: object
        margin: object
        margin_ok: object
        margin_rule: object
        total: object
        total_ok: object
        total_rule: object
        warning_count: object

        def _py_r1000_if_1():
            nonlocal _, error_text, margin, total
            error_text = '; '.join(fatal_history_errors[:5])
            if len(fatal_history_errors) > 5:
                error_text += f'; and {len(fatal_history_errors) - 5} more'
            if margin_rule.get('method') in {'rolling', 'regime_aware'}:
                margin = unsafe_history_component(margin_rule)
            else:
                margin, _ = calculate_component_safely(league, 'margin', margin_rule, history)
            if total_rule.get('method') in {'rolling', 'regime_aware'}:
                total = unsafe_history_component(total_rule)
            else:
                total, _ = calculate_component_safely(league, 'total', total_rule, history)
            return (_py_r1000_RETURN, ({'status': 'error', 'config_status': config_status, 'error': error_text, 'margin_bias': margin, 'total_bias': total, 'history': history_meta}, False, False))
        config_status_raw = league_cfg.get('status')
        config_status = None if config_status_raw is None else str(config_status_raw).strip().lower()
        try:
            margin_rule = resolve_bias_rule(league_cfg, 'margin')
            total_rule = resolve_bias_rule(league_cfg, 'total')
        except Exception as exc:
            return (_py_r1000_RETURN, ({'status': 'error', 'config_status': config_status, 'error': str(exc)}, False, False))
        if margin_rule.get('method') is None and total_rule.get('method') is None:
            log(f'{league_upper(league)} | SKIPPED | no configured bias rules')
            return (_py_r1000_RETURN, (skipped_league_state(config_status), True, False))
        try:
            history, history_meta, fatal_history_errors = build_completed_history(league)
        except Exception as exc:
            log(f'{league_upper(league)} | HISTORY BUILD FAILED | {exc}', 'ERROR')
            return (_py_r1000_RETURN, ({'status': 'error', 'config_status': config_status, 'error': str(exc), 'margin_bias': component_stub('error', margin_rule.get('method'), margin_rule.get('window_games')), 'total_bias': component_stub('error', total_rule.get('method'), total_rule.get('window_games'))}, False, False))
        history_required = any((rule.get('method') in {'rolling', 'regime_aware'} for rule in (margin_rule, total_rule)))
        if history_required and fatal_history_errors:
            _py_r1000_result_2 = _py_r1000_if_1()
            if _py_r1000_result_2[0] != _py_r1000_NONE:
                return _py_r1000_result_2
        margin, margin_ok = calculate_component_safely(league, 'margin', margin_rule, history)
        total, total_ok = calculate_component_safely(league, 'total', total_rule, history)
        configured_component_errors = margin_rule.get('method') is not None and (not margin_ok) or (total_rule.get('method') is not None and (not total_ok))
        if configured_component_errors:
            league_state = {'status': 'error', 'config_status': config_status, 'margin_bias': margin, 'total_bias': total, 'history': history_meta}
            log(f'{league_upper(league)} | ERROR | component calculation failed', 'ERROR')
            return (_py_r1000_RETURN, (league_state, False, False))
        warning_count = warning_count_from_history(history_meta)
        league_status = 'ready_with_warnings' if warning_count else 'ready'
        league_state = {'status': league_status, 'config_status': config_status, 'margin_bias': margin, 'total_bias': total, 'history': history_meta}
        log(f"{league_upper(league)} | {league_status.upper()} | margin={margin.get('value')} total={total.get('value')} warnings={warning_count}")
        return (_py_r1000_RETURN, (league_state, True, bool(warning_count)))
    _py_r1000_outcome = _py_r1000_impl()
    if _py_r1000_outcome[0] == _py_r1000_RETURN:
        return _py_r1000_outcome[1]


# ============================================================================
# STATE OUTPUT
# ============================================================================

def write_state(
    state: dict[str, Any],
) -> None:
    STATE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = (
        STATE_PATH
        .with_suffix(
            STATE_PATH.suffix
            + ".tmp"
        )
    )

    with open(
        tmp_path,
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            state,
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )

    tmp_path.replace(
        STATE_PATH
    )

    log(
        (
            f"STATE WRITTEN | "
            f"{repo_relative(STATE_PATH)}"
        )
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    init_log()

    try:
        cfg = load_model_config()

        prediction_source = (
            production_prediction_source(
                cfg
            )
        )

        log(
            "PRODUCTION PREDICTION SOURCE | "
            f"source={prediction_source} | "
            f"root="
            f"{repo_relative(PREDICTION_ROOTS[prediction_source])}"
        )

        config_leagues = (
            cfg.get(
                "leagues",
                {},
            )
        )

        state: dict[
            str,
            Any,
        ] = {
            "schema_version": 1,
            "generated_at_utc": (
                utc_now_iso()
            ),
            "source_config": (
                repo_relative(
                    CONFIG_PATH
                )
            ),
            "script_path": (
                repo_relative(
                    SCRIPT_PATH
                )
            ),
            "script_sha256": (
                script_sha256()
            ),
            "leagues": {},
        }

        failures = 0
        warnings = 0

        # --------------------------------------------------------
        # SUPPORTED LEAGUES
        # --------------------------------------------------------

        for league in SUPPORTED_LEAGUES:
            league_cfg = (
                config_leagues.get(
                    league
                )
                or {}
            )

            if not isinstance(
                league_cfg,
                dict,
            ):
                state[
                    "leagues"
                ][
                    league
                ] = {
                    "status": "error",
                    "error": (
                        f"League config "
                        f"for {league} "
                        f"must be a mapping"
                    ),
                }

                failures += 1

                continue

            (
                league_state,
                ok,
                warned,
            ) = process_league(
                league,
                league_cfg,
            )

            state[
                "leagues"
            ][
                league
            ] = league_state

            failures += int(
                not ok
            )

            warnings += int(
                warned
            )

        # --------------------------------------------------------
        # UNKNOWN LEAGUE KEYS
        # --------------------------------------------------------------------------

        for raw_key in config_leagues:
            league_key = (
                str(
                    raw_key
                )
                .strip()
                .lower()
            )

            if (
                league_key
                not in SUPPORTED_LEAGUES
            ):
                state[
                    "leagues"
                ][
                    league_key
                ] = {
                    "status": (
                        "skipped_unsupported_league"
                    ),
                    "error": (
                        f"Unsupported league key: "
                        f"{raw_key}"
                    ),
                }

        # --------------------------------------------------------
        # RUN STATUS
        # --------------------------------------------------------

        if failures:
            state[
                "run_status"
            ] = (
                "completed_with_errors"
            )

        elif warnings:
            state[
                "run_status"
            ] = (
                "success_with_warnings"
            )

        else:
            state[
                "run_status"
            ] = "success"

        write_state(
            state
        )

        log(
            (
                f"RUN STATUS | "
                f"{state['run_status']}"
            )
        )

        return (
            1
            if failures
            else 0
        )

    except Exception as exc:
        log(
            (
                f"FATAL ERROR | "
                f"{exc}\n"
                f"{traceback.format_exc()}"
            ),
            "ERROR",
        )

        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )
