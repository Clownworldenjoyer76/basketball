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


def resolve_bias_rule(
    league_cfg: dict[str, Any],
    component: str,
) -> dict[str, Any]:
    bias_cfg = (
        league_cfg.get(
            "bias"
        )
        or {}
    )

    rule = bias_cfg.get(
        component
    )

    if rule is None:
        return {
            "method": None,
            "window_games": None,
            "windows_games": None,
            "weights": None,
            "sign_conflict_shrink": None,
            "value": None,
        }

    if not isinstance(
        rule,
        dict,
    ):
        raise ValueError(
            f"bias.{component} "
            f"must be a mapping "
            f"or null"
        )

    method_raw = rule.get(
        "method"
    )

    method = (
        None
        if method_raw is None
        else str(
            method_raw
        ).strip().lower()
    )

    if method in {
        "",
        "null",
    }:
        method = None

    window = positive_int_or_none(
        rule.get(
            "window_games"
        ),
        (
            f"bias.{component}."
            f"window_games"
        ),
    )

    windows: list[int] | None = None
    weights: list[float] | None = None
    sign_conflict_shrink: float | None = None

    if method == "regime_aware":
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
                f"bias.{component}.windows_games "
                f"must be a non-empty list for "
                f"method='regime_aware'"
            )

        windows = []

        for index, raw_window in enumerate(
            raw_windows
        ):
            parsed_window = positive_int_or_none(
                raw_window,
                (
                    f"bias.{component}."
                    f"windows_games[{index}]"
                ),
            )

            if (
                parsed_window is None
                or parsed_window <= 0
            ):
                raise ValueError(
                    f"bias.{component}."
                    f"windows_games[{index}] "
                    f"must be > 0"
                )

            windows.append(
                parsed_window
            )

        if len(
            set(
                windows
            )
        ) != len(
            windows
        ):
            raise ValueError(
                f"bias.{component}.windows_games "
                f"must contain unique windows"
            )

        if windows != sorted(
            windows
        ):
            raise ValueError(
                f"bias.{component}.windows_games "
                f"must be sorted ascending"
            )

        if (
            not isinstance(
                raw_weights,
                list,
            )
            or len(
                raw_weights
            ) != len(
                windows
            )
        ):
            raise ValueError(
                f"bias.{component}.weights "
                f"must contain exactly one weight "
                f"for each configured window"
            )

        weights = []

        for index, raw_weight in enumerate(
            raw_weights
        ):
            weight = to_float(
                raw_weight
            )

            if (
                weight is None
                or weight < 0
            ):
                raise ValueError(
                    f"bias.{component}."
                    f"weights[{index}] "
                    f"must be a finite number >= 0"
                )

            weights.append(
                float(
                    weight
                )
            )

        weight_sum = sum(
            weights
        )

        if weight_sum <= 0:
            raise ValueError(
                f"bias.{component}.weights "
                f"must sum to > 0"
            )

        weights = [
            weight
            / weight_sum
            for weight
            in weights
        ]

        shrink_raw = rule.get(
            "sign_conflict_shrink"
        )

        sign_conflict_shrink = to_float(
            shrink_raw
        )

        if (
            sign_conflict_shrink is None
            or sign_conflict_shrink < 0
            or sign_conflict_shrink > 1
        ):
            raise ValueError(
                f"bias.{component}."
                f"sign_conflict_shrink "
                f"must be between 0 and 1"
            )

        if window is not None:
            raise ValueError(
                f"bias.{component}.window_games "
                f"must be null/omitted for "
                f"method='regime_aware'"
            )

    value_raw = rule.get(
        "value"
    )

    value = None

    if value_raw not in (
        None,
        "",
    ):
        value = to_float(
            value_raw
        )

        if value is None:
            raise ValueError(
                f"bias.{component}.value "
                f"must be numeric; "
                f"got {value_raw!r}"
            )

    if (
        method
        in {
            "rolling",
            "regime_aware",
            "none",
        }
        and value is not None
    ):
        raise ValueError(
            f"bias.{component}.value "
            f"must be null for "
            f"method={method!r}"
        )

    if (
        method == "fixed"
        and window is not None
    ):
        raise ValueError(
            f"bias.{component}.window_games "
            f"must be null/omitted for "
            f"method='fixed'"
        )

    return {
        "method": method,
        "window_games": window,
        "windows_games": windows,
        "weights": weights,
        "sign_conflict_shrink": (
            sign_conflict_shrink
        ),
        "value": value,
    }


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


def load_historical_completed_games(
    league: str,
) -> tuple[
    list[CompletedGame],
    dict[str, int],
    list[str],
]:
    games: list[
        CompletedGame
    ] = []

    stats = (
        historical_stats_template()
    )

    fatal_errors: list[
        str
    ] = []

    for (
        season,
        path,
    ) in historical_files_for_league(
        league
    ):
        stats[
            "historical_files_scanned"
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
            HISTORICAL_REQUIRED,
        )

        for (
            row_number,
            row,
        ) in enumerate(
            rows,
            start=2,
        ):
            stats[
                "historical_rows"
            ] += 1

            game_date = normalize_date(
                row.get(
                    "game_date"
                )
            )

            if not game_date:
                stats[
                    "historical_rows_invalid_date"
                ] += 1

                continue

            home_team = str(
                row.get(
                    "home_team"
                )
                or ""
            ).strip()

            away_team = str(
                row.get(
                    "away_team"
                )
                or ""
            ).strip()

            home_proj = to_float(
                row.get(
                    "home_projected_points"
                )
            )

            away_proj = to_float(
                row.get(
                    "away_projected_points"
                )
            )

            total_proj = to_float(
                row.get(
                    "total_projected_points"
                )
            )

            home_score = to_float(
                row.get(
                    "home_score"
                )
            )

            away_score = to_float(
                row.get(
                    "away_score"
                )
            )

            if (
                total_proj is None
                and home_proj is not None
                and away_proj is not None
            ):
                total_proj = (
                    home_proj
                    + away_proj
                )

            if (
                not home_team
                or not away_team
                or home_proj is None
                or away_proj is None
                or total_proj is None
                or home_score is None
                or away_score is None
            ):
                stats[
                    "historical_incomplete_rows"
                ] += 1

                continue

            bias_flag = parse_bias_flag(
                row.get(
                    "bias_applied"
                )
            )

            if bias_flag is None:
                stats[
                    "historical_invalid_bias_flag_rows"
                ] += 1

                log(
                    (
                        f"{league_upper(league)} | "
                        f"HISTORICAL INVALID BIAS FLAG | "
                        f"file={path.name} "
                        f"row={row_number} "
                        f"value="
                        f"{row.get('bias_applied')!r}"
                    ),
                    "WARN",
                )

                continue

            raw_home = home_proj
            raw_away = away_proj
            raw_total = total_proj

            if bias_flag == 0:
                stats[
                    "historical_rows_raw_unadjusted"
                ] += 1

            else:
                per_game_margin = to_float(
                    row.get(
                        "margin_bias"
                    )
                )

                per_game_total = to_float(
                    row.get(
                        "total_bias"
                    )
                )

                if (
                    per_game_margin is not None
                    and per_game_total is not None
                ):
                    reversal_margin = (
                        per_game_margin
                    )

                    reversal_total = (
                        per_game_total
                    )

                    stats[
                        "historical_rows_bias_reversed_per_game"
                    ] += 1

                else:
                    legacy = (
                        LEGACY_HISTORICAL_BIAS
                        .get(
                            (
                                league.lower(),
                                season,
                            )
                        )
                    )

                    if legacy is None:
                        stats[
                            "historical_unreversible_bias_rows"
                        ] += 1

                        message = (
                            f"{league_upper(league)} "
                            f"historical "
                            f"{path.name} "
                            f"row {row_number} "
                            f"has bias_applied=1 "
                            f"but does not contain "
                            f"valid per-game margin_bias "
                            f"and total_bias, and no "
                            f"legacy fallback exists "
                            f"for this league/season"
                        )

                        fatal_errors.append(
                            message
                        )

                        log(
                            message,
                            "ERROR",
                        )

                        continue

                    reversal_margin = float(
                        legacy[
                            "margin"
                        ]
                    )

                    reversal_total = float(
                        legacy[
                            "total"
                        ]
                    )

                    stats[
                        "historical_rows_bias_reversed_legacy"
                    ] += 1

                (
                    raw_home,
                    raw_away,
                    raw_total,
                ) = reverse_adjusted_projection(
                    home_proj,
                    away_proj,
                    total_proj,
                    reversal_margin,
                    reversal_total,
                )

                stats[
                    "historical_rows_bias_reversed"
                ] += 1

            games.append(
                CompletedGame(
                    league=league,
                    game_id=canonical_game_id(
                        row.get(
                            "game_id"
                        )
                    ),
                    game_date=game_date,
                    game_time=str(
                        row.get(
                            "game_time"
                        )
                        or ""
                    ).strip(),
                    home_team=home_team,
                    away_team=away_team,
                    home_projected_points=raw_home,
                    away_projected_points=raw_away,
                    total_projected_points=raw_total,
                    home_score=home_score,
                    away_score=away_score,
                    source=repo_relative(
                        path
                    ),
                    source_priority=1,
                )
            )

            stats[
                "historical_usable_games"
            ] += 1

        log(
            (
                f"{league_upper(league)} | "
                f"HISTORICAL | "
                f"{path.name} | "
                f"rows={len(rows)}"
            )
        )

    return (
        games,
        stats,
        fatal_errors,
    )


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

def deduplicate_final_rows(
    rows: list[
        dict[str, str]
    ],
) -> tuple[
    list[
        dict[str, str]
    ],
    dict[str, int],
]:
    stats = {
        "duplicate_final_game_ids": 0,
        "duplicate_final_composites": 0,
        "conflicting_final_game_ids": 0,
        "conflicting_final_composites": 0,
        "final_duplicate_rows_removed": 0,
        "final_conflicting_rows_excluded": 0,
    }

    excluded: set[
        str
    ] = set()

    # ------------------------------------------------------------
    # GAME_ID DUPLICATES
    # ------------------------------------------------------------

    id_groups: dict[
        str,
        list[
            dict[str, str]
        ],
    ] = {}

    for row in rows:
        gid = canonical_game_id(
            row.get(
                "game_id"
            )
        )

        if gid:
            id_groups.setdefault(
                gid,
                [],
            ).append(
                row
            )

    for (
        gid,
        group,
    ) in id_groups.items():
        if len(
            group
        ) <= 1:
            continue

        signatures = {
            final_identity_score_signature(
                row
            )
            for row in group
        }

        if len(
            signatures
        ) == 1:
            stats[
                "duplicate_final_game_ids"
            ] += (
                len(group)
                - 1
            )

        else:
            stats[
                "conflicting_final_game_ids"
            ] += 1

            excluded.update(
                row[
                    "_uid"
                ]
                for row in group
            )

            log(
                (
                    f"FINAL CONFLICTING GAME_ID | "
                    f"game_id={gid} "
                    f"rows={len(group)}"
                ),
                "WARN",
            )

    # ------------------------------------------------------------
    # COMPOSITE DUPLICATES
    # ------------------------------------------------------------

    composite_groups: dict[
        str,
        list[
            dict[str, str]
        ],
    ] = {}

    for row in rows:
        comp = composite_key(
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

        if comp:
            composite_groups.setdefault(
                comp,
                [],
            ).append(
                row
            )

    for (
        comp,
        group,
    ) in composite_groups.items():
        if len(
            group
        ) <= 1:
            continue

        score_signatures = {
            (
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
            for row in group
        }

        if len(
            score_signatures
        ) == 1:
            stats[
                "duplicate_final_composites"
            ] += (
                len(group)
                - 1
            )

        else:
            stats[
                "conflicting_final_composites"
            ] += 1

            excluded.update(
                row[
                    "_uid"
                ]
                for row in group
            )

            log(
                (
                    f"FINAL CONFLICTING COMPOSITE | "
                    f"key={comp} "
                    f"rows={len(group)}"
                ),
                "WARN",
            )

    stats[
        "final_conflicting_rows_excluded"
    ] = len(
        excluded
    )

    remaining = [
        row
        for row in rows
        if row[
            "_uid"
        ] not in excluded
    ]

    # ------------------------------------------------------------
    # REMOVE IDENTICAL DUPLICATES
    # ------------------------------------------------------------

    grouped_remaining: dict[
        tuple[str, str],
        list[
            dict[str, str]
        ],
    ] = {}

    for row in remaining:
        comp = composite_key(
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

        gid = canonical_game_id(
            row.get(
                "game_id"
            )
        )

        if comp:
            key = (
                "composite",
                comp,
            )

        elif gid:
            key = (
                "game_id",
                gid,
            )

        else:
            key = (
                "row",
                row[
                    "_uid"
                ],
            )

        grouped_remaining.setdefault(
            key,
            [],
        ).append(
            row
        )

    deduped: list[
        dict[str, str]
    ] = []

    for group in grouped_remaining.values():
        preferred = sorted(
            group,
            key=lambda sort_row: (
                (
                    0
                    if canonical_game_id(
                        sort_row.get(
                            "game_id"
                        )
                    )
                    else 1
                ),
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

        deduped.append(
            preferred
        )

        stats[
            "final_duplicate_rows_removed"
        ] += (
            len(group)
            - 1
        )

    deduped.sort(
        key=lambda sort_row: (
            parse_game_datetime(
                sort_row.get(
                    "game_date"
                ),
                "",
            ),
            normalize_text(
                sort_row.get(
                    "home_team"
                )
            ),
            normalize_text(
                sort_row.get(
                    "away_team"
                )
            ),
            canonical_game_id(
                sort_row.get(
                    "game_id"
                )
            ),
        )
    )

    return (
        deduped,
        stats,
    )


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


def load_current_completed_games(
    league: str,
    current_season: int | None,
    historical_games: list[
        CompletedGame
    ],
) -> tuple[
    list[
        CompletedGame
    ],
    dict[str, Any],
]:
    (
        predictions,
        prediction_load_stats,
    ) = load_current_prediction_rows(
        league,
        current_season,
    )

    (
        finals_raw,
        final_load_stats,
    ) = load_current_final_rows(
        league,
        current_season,
    )

    (
        finals,
        final_duplicate_stats,
    ) = deduplicate_final_rows(
        finals_raw
    )

    (
        pred_by_id,
        pred_by_composite,
        ambiguous_ids,
        ambiguous_composites,
        prediction_duplicate_stats,
    ) = build_prediction_indexes(
        predictions
    )

    (
        historical_ids,
        historical_composites,
    ) = historical_coverage_sets(
        historical_games
    )

    stats: dict[
        str,
        Any,
    ] = {
        "current_season": current_season,
        "season_status": season_status_for_league(
            current_season,
        ),
        **prediction_load_stats,
        **final_load_stats,
        **prediction_duplicate_stats,
        **final_duplicate_stats,
        "finals_already_covered_by_historical": 0,
        "matched_by_game_id": 0,
        "matched_by_composite": 0,
        "game_id_identity_mismatches": 0,
        "ambiguous_prediction_matches": 0,
        "true_unmatched_current_finals": 0,
        "invalid_current_matches": 0,
        "current_matched_games": 0,
    }

    games: list[
        CompletedGame
    ] = []

    for final in finals:
        gid = canonical_game_id(
            final.get(
                "game_id"
            )
        )

        comp = composite_key(
            final.get(
                "game_date"
            ),
            final.get(
                "home_team"
            ),
            final.get(
                "away_team"
            ),
        )

        # --------------------------------------------------------
        # HISTORICAL COVERAGE
        # --------------------------------------------------------

        if (
            (
                gid
                and gid
                in historical_ids
            )
            or (
                comp
                and comp
                in historical_composites
            )
        ):
            stats[
                "finals_already_covered_by_historical"
            ] += 1

            continue

        # --------------------------------------------------------
        # VALID FINAL SCORE
        # --------------------------------------------------------

        home_score = to_float(
            final.get(
                "home_score"
            )
        )

        away_score = to_float(
            final.get(
                "away_score"
            )
        )

        if (
            home_score is None
            or away_score is None
        ):
            stats[
                "invalid_current_matches"
            ] += 1

            continue

        prediction: (
            dict[str, str]
            | None
        ) = None

        match_method: (
            str
            | None
        ) = None

        blocked_by_ambiguity = False

        # --------------------------------------------------------
        # PRIMARY MATCH: GAME_ID
        # --------------------------------------------------------

        if (
            gid
            and gid in pred_by_id
        ):
            candidate = (
                pred_by_id[
                    gid
                ]
            )

            identity_result = (
                identities_agree(
                    candidate,
                    final,
                )
            )

            if identity_result is None:
                stats[
                    "invalid_current_matches"
                ] += 1

                continue

            if not identity_result:
                stats[
                    "game_id_identity_mismatches"
                ] += 1

                log(
                    (
                        f"{league_upper(league)} | "
                        f"GAME_ID IDENTITY MISMATCH | "
                        f"game_id={gid} | "
                        f"final="
                        f"{final.get('game_date')} "
                        f"{final.get('home_team')} vs "
                        f"{final.get('away_team')} | "
                        f"prediction="
                        f"{candidate.get('game_date')} "
                        f"{candidate.get('home_team')} vs "
                        f"{candidate.get('away_team')}"
                    ),
                    "WARN",
                )

                # Important:
                # If the ID actually matched but the identities conflict,
                # do not silently fall back to composite.
                continue

            prediction = candidate
            match_method = (
                "game_id"
            )

        elif (
            gid
            and gid
            in ambiguous_ids
        ):
            blocked_by_ambiguity = (
                True
            )

        # --------------------------------------------------------
        # FALLBACK MATCH: DATE + HOME + AWAY
        # --------------------------------------------------------

        if (
            prediction is None
            and comp
        ):
            if (
                comp
                in pred_by_composite
            ):
                prediction = (
                    pred_by_composite[
                        comp
                    ]
                )

                match_method = (
                    "composite"
                )

            elif (
                comp
                in ambiguous_composites
            ):
                blocked_by_ambiguity = (
                    True
                )

        # --------------------------------------------------------
        # NO USABLE MATCH
        # --------------------------------------------------------

        if prediction is None:
            if blocked_by_ambiguity:
                stats[
                    "ambiguous_prediction_matches"
                ] += 1

            else:
                stats[
                    "true_unmatched_current_finals"
                ] += 1

            continue

        # --------------------------------------------------------
        # RAW PROJECTION VALUES
        # --------------------------------------------------------

        home_proj = to_float(
            prediction.get(
                "home_projected_points"
            )
        )

        away_proj = to_float(
            prediction.get(
                "away_projected_points"
            )
        )

        total_proj = (
            normalized_prediction_total(
                prediction
            )
        )

        if (
            home_proj is None
            or away_proj is None
            or total_proj is None
        ):
            stats[
                "invalid_current_matches"
            ] += 1

            continue

        game_date = normalize_date(
            final.get(
                "game_date"
            )
        )

        home_team = str(
            final.get(
                "home_team"
            )
            or ""
        ).strip()

        away_team = str(
            final.get(
                "away_team"
            )
            or ""
        ).strip()

        if (
            not game_date
            or not home_team
            or not away_team
        ):
            stats[
                "invalid_current_matches"
            ] += 1

            continue

        games.append(
            CompletedGame(
                league=league,
                game_id=(
                    gid
                    or canonical_game_id(
                        prediction.get(
                            "game_id"
                        )
                    )
                ),
                game_date=game_date,
                game_time=str(
                    prediction.get(
                        "game_time"
                    )
                    or ""
                ).strip(),
                home_team=home_team,
                away_team=away_team,
                home_projected_points=home_proj,
                away_projected_points=away_proj,
                total_projected_points=total_proj,
                home_score=home_score,
                away_score=away_score,
                source=(
                    f"{prediction.get('_source_file', '')}"
                    f" + "
                    f"{final.get('_source_file', '')}"
                ),
                source_priority=2,
            )
        )

        if (
            match_method
            == "game_id"
        ):
            stats[
                "matched_by_game_id"
            ] += 1

        else:
            stats[
                "matched_by_composite"
            ] += 1

    stats[
        "current_matched_games"
    ] = len(
        games
    )

    return (
        games,
        stats,
    )


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


def calculate_component_bias(
    league: str,
    component: str,
    rule: dict[str, Any],
    history: list[
        CompletedGame
    ],
) -> dict[str, Any]:
    method = rule.get(
        "method"
    )

    # ------------------------------------------------------------
    # NULL / MISSING
    # ------------------------------------------------------------

    if method is None:
        return component_stub(
            "skipped_no_rule",
            None,
            None,
        )

    # ------------------------------------------------------------
    # NONE
    # ------------------------------------------------------------

    if method == "none":
        result = component_stub(
            "disabled",
            "none",
            None,
        )

        # Explicitly disabled.
        # Zero is stored so downstream consumers can safely interpret
        # the component as applying no adjustment.
        result[
            "value"
        ] = 0.0

        return result

    # ------------------------------------------------------------
    # FIXED
    # ------------------------------------------------------------

    if method == "fixed":
        value = rule.get(
            "value"
        )

        if value is None:
            raise ValueError(
                f"{league_upper(league)} "
                f"{component} fixed bias "
                f"requires "
                f"bias.{component}.value"
            )

        result = component_stub(
            "ready",
            "fixed",
            None,
        )

        result[
            "value"
        ] = round(
            float(value),
            3,
        )

        return result

    # ------------------------------------------------------------
    # REGIME-AWARE MULTI-WINDOW
    # ------------------------------------------------------------

    if method == "regime_aware":
        windows = rule.get(
            "windows_games"
        )

        weights = rule.get(
            "weights"
        )

        shrink = rule.get(
            "sign_conflict_shrink"
        )

        if (
            not isinstance(
                windows,
                list,
            )
            or not windows
        ):
            raise ValueError(
                f"{league_upper(league)} "
                f"{component} regime_aware bias "
                f"requires windows_games"
            )

        if (
            not isinstance(
                weights,
                list,
            )
            or len(
                weights
            ) != len(
                windows
            )
        ):
            raise ValueError(
                f"{league_upper(league)} "
                f"{component} regime_aware bias "
                f"requires one weight per window"
            )

        if shrink is None:
            raise ValueError(
                f"{league_upper(league)} "
                f"{component} regime_aware bias "
                f"requires sign_conflict_shrink"
            )

        largest_window = max(
            int(window)
            for window
            in windows
        )

        if (
            len(history)
            < largest_window
        ):
            raise ValueError(
                f"{league_upper(league)} "
                f"{component} regime_aware bias "
                f"requires {largest_window} "
                f"completed games; only "
                f"{len(history)} unique "
                f"completed games are "
                f"available"
            )

        window_means: dict[
            int,
            float,
        ] = {}

        for window in windows:
            window_int = int(
                window
            )

            selected = history[
                -window_int:
            ]

            if component == "margin":
                errors = [
                    game.margin_error
                    for game in selected
                ]

            elif component == "total":
                errors = [
                    game.total_error
                    for game in selected
                ]

            else:
                raise ValueError(
                    f"Unsupported bias component: "
                    f"{component}"
                )

            window_means[
                window_int
            ] = (
                sum(
                    errors
                )
                / len(
                    errors
                )
            )

        weighted_unshrunk = sum(
            float(weight)
            * window_means[
                int(window)
            ]
            for (
                window,
                weight,
            )
            in zip(
                windows,
                weights,
            )
        )

        positive_present = any(
            value > 1e-12
            for value
            in window_means.values()
        )

        negative_present = any(
            value < -1e-12
            for value
            in window_means.values()
        )

        sign_conflict = (
            positive_present
            and negative_present
        )

        effective_value = (
            weighted_unshrunk
            * float(
                shrink
            )
            if sign_conflict
            else weighted_unshrunk
        )

        largest_selected = history[
            -largest_window:
        ]

        return {
            "status": "ready",
            "method": (
                "regime_aware"
            ),
            "value": round(
                effective_value,
                3,
            ),
            # window_games remains populated with the largest lookback for
            # compatibility with consumers that expect a scalar window field.
            "window_games": (
                largest_window
            ),
            "windows_games": [
                int(
                    window
                )
                for window
                in windows
            ],
            "weights": [
                round(
                    float(
                        weight
                    ),
                    6,
                )
                for weight
                in weights
            ],
            "window_mean_residuals": {
                str(
                    int(
                        window
                    )
                ): round(
                    window_means[
                        int(
                            window
                        )
                    ],
                    4,
                )
                for window
                in windows
            },
            "unshrunk_weighted_value": round(
                weighted_unshrunk,
                4,
            ),
            "sign_conflict": (
                sign_conflict
            ),
            "sign_conflict_shrink": round(
                float(
                    shrink
                ),
                6,
            ),
            "regime_status": (
                "sign_conflict_shrunk"
                if sign_conflict
                else "aligned"
            ),
            "games_used": (
                largest_window
            ),
            "first_game_date": (
                largest_selected[0]
                .game_date
            ),
            "last_game_date": (
                largest_selected[-1]
                .game_date
            ),
            "mean_error_definition": (
                "projected_minus_actual"
            ),
        }

    # ------------------------------------------------------------
    # ROLLING
    # ------------------------------------------------------------

    if method != "rolling":
        raise ValueError(
            f"Unsupported "
            f"{league_upper(league)} "
            f"{component} bias method "
            f"{method!r}; supported "
            f"methods are rolling, "
            f"regime_aware, fixed, "
            f"none, or null/missing"
        )

    window = rule.get(
        "window_games"
    )

    if (
        window is None
        or window <= 0
    ):
        raise ValueError(
            f"{league_upper(league)} "
            f"{component} rolling bias "
            f"requires window_games > 0"
        )

    if (
        len(history)
        < window
    ):
        raise ValueError(
            f"{league_upper(league)} "
            f"{component} rolling bias "
            f"requires {window} "
            f"completed games; only "
            f"{len(history)} unique "
            f"completed games are "
            f"available"
        )

    selected = (
        history[
            -window:
        ]
    )

    if component == "margin":
        errors = [
            game.margin_error
            for game in selected
        ]

    elif component == "total":
        errors = [
            game.total_error
            for game in selected
        ]

    else:
        raise ValueError(
            f"Unsupported bias component: "
            f"{component}"
        )

    value = (
        sum(errors)
        / len(errors)
    )

    return {
        "status": "ready",
        "method": "rolling",
        "value": round(
            value,
            3,
        ),
        "window_games": int(
            window
        ),
        "games_used": len(
            selected
        ),
        "first_game_date": (
            selected[0]
            .game_date
        ),
        "last_game_date": (
            selected[-1]
            .game_date
        ),
        "mean_error_definition": (
            "projected_minus_actual"
        ),
    }


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

def process_league(
    league: str,
    league_cfg: dict[str, Any],
) -> tuple[
    dict[str, Any],
    bool,
    bool,
]:
    config_status_raw = (
        league_cfg.get(
            "status"
        )
    )

    config_status = (
        None
        if config_status_raw is None
        else str(
            config_status_raw
        ).strip().lower()
    )

    # ------------------------------------------------------------
    # LOAD BIAS RULES
    # ------------------------------------------------------------

    try:
        margin_rule = (
            resolve_bias_rule(
                league_cfg,
                "margin",
            )
        )

        total_rule = (
            resolve_bias_rule(
                league_cfg,
                "total",
            )
        )

    except Exception as exc:
        return (
            {
                "status": "error",
                "config_status": (
                    config_status
                ),
                "error": str(
                    exc
                ),
            },
            False,
            False,
        )

    # ------------------------------------------------------------
    # NO CONFIGURED BIAS RULES
    # ------------------------------------------------------------

    if (
        margin_rule.get(
            "method"
        ) is None
        and total_rule.get(
            "method"
        ) is None
    ):
        log(
            (
                f"{league_upper(league)} | "
                f"SKIPPED | "
                f"no configured bias rules"
            )
        )

        return (
            skipped_league_state(
                config_status
            ),
            True,
            False,
        )

    # ------------------------------------------------------------
    # BUILD HISTORY
    # ------------------------------------------------------------

    try:
        (
            history,
            history_meta,
            fatal_history_errors,
        ) = build_completed_history(
            league
        )

    except Exception as exc:
        log(
            (
                f"{league_upper(league)} | "
                f"HISTORY BUILD FAILED | "
                f"{exc}"
            ),
            "ERROR",
        )

        return (
            {
                "status": "error",
                "config_status": (
                    config_status
                ),
                "error": str(
                    exc
                ),
                "margin_bias": (
                    component_stub(
                        "error",
                        margin_rule.get(
                            "method"
                        ),
                        margin_rule.get(
                            "window_games"
                        ),
                    )
                ),
                "total_bias": (
                    component_stub(
                        "error",
                        total_rule.get(
                            "method"
                        ),
                        total_rule.get(
                            "window_games"
                        ),
                    )
                ),
            },
            False,
            False,
        )

    # ------------------------------------------------------------
    # UNSAFE HISTORICAL ROWS
    # ------------------------------------------------------------

    history_required = any(
        rule.get(
            "method"
        )
        in {
            "rolling",
            "regime_aware",
        }
        for rule in (
            margin_rule,
            total_rule,
        )
    )

    if (
        history_required
        and fatal_history_errors
    ):
        error_text = "; ".join(
            fatal_history_errors[
                :5
            ]
        )

        if (
            len(
                fatal_history_errors
            )
            > 5
        ):
            error_text += (
                f"; and "
                f"{len(fatal_history_errors) - 5} "
                f"more"
            )

        if (
            margin_rule.get(
                "method"
            )
            in {
                "rolling",
                "regime_aware",
            }
        ):
            margin = unsafe_history_component(
                margin_rule
            )

        else:
            (
                margin,
                _,
            ) = calculate_component_safely(
                league,
                "margin",
                margin_rule,
                history,
            )

        if (
            total_rule.get(
                "method"
            )
            in {
                "rolling",
                "regime_aware",
            }
        ):
            total = unsafe_history_component(
                total_rule
            )

        else:
            (
                total,
                _,
            ) = calculate_component_safely(
                league,
                "total",
                total_rule,
                history,
            )

        return (
            {
                "status": "error",
                "config_status": (
                    config_status
                ),
                "error": (
                    error_text
                ),
                "margin_bias": (
                    margin
                ),
                "total_bias": (
                    total
                ),
                "history": (
                    history_meta
                ),
            },
            False,
            False,
        )

    # ------------------------------------------------------------
    # CALCULATE COMPONENTS
    # ------------------------------------------------------------

    (
        margin,
        margin_ok,
    ) = calculate_component_safely(
        league,
        "margin",
        margin_rule,
        history,
    )

    (
        total,
        total_ok,
    ) = calculate_component_safely(
        league,
        "total",
        total_rule,
        history,
    )

    configured_component_errors = (
        (
            margin_rule.get(
                "method"
            ) is not None
            and not margin_ok
        )
        or
        (
            total_rule.get(
                "method"
            ) is not None
            and not total_ok
        )
    )

    if configured_component_errors:
        league_state = {
            "status": "error",
            "config_status": (
                config_status
            ),
            "margin_bias": (
                margin
            ),
            "total_bias": (
                total
            ),
            "history": (
                history_meta
            ),
        }

        log(
            (
                f"{league_upper(league)} | "
                f"ERROR | component "
                f"calculation failed"
            ),
            "ERROR",
        )

        return (
            league_state,
            False,
            False,
        )

    # ------------------------------------------------------------
    # WARNING STATUS
    # ------------------------------------------------------------

    warning_count = (
        warning_count_from_history(
            history_meta
        )
    )

    league_status = (
        "ready_with_warnings"
        if warning_count
        else "ready"
    )

    league_state = {
        "status": (
            league_status
        ),
        "config_status": (
            config_status
        ),
        "margin_bias": (
            margin
        ),
        "total_bias": (
            total
        ),
        "history": (
            history_meta
        ),
    }

    log(
        (
            f"{league_upper(league)} | "
            f"{league_status.upper()} | "
            f"margin="
            f"{margin.get('value')} "
            f"total="
            f"{total.get('value')} "
            f"warnings="
            f"{warning_count}"
        )
    )

    return (
        league_state,
        True,
        bool(
            warning_count
        ),
    )


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
