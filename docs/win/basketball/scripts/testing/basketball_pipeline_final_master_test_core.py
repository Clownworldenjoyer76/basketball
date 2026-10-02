from __future__ import annotations
'''
FINAL BASKETBALL PIPELINE VALIDATION / OPTIMIZATION TEST
========================================================

Production-parity historical validation for NBA / NCAAM / WNBA.

Supports:
- model_source=dratings
- model_source=sdv
- model_source=ensemble
- production rolling/regime-aware bias
- exact WNBA regime-aware total bias
- production complementary spread/total calibration
- production selection_mode and pick_preference
- exact production-vs-backtest parity checks
'''
import argparse
import hashlib
import importlib.util
import json
import math
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except Exception as exc:
    raise SystemExit(
        'This script requires PyYAML. Install with: pip install pyyaml'
    ) from exc

try:
    from scipy.optimize import minimize
    from scipy.stats import norm
except Exception as exc:
    raise SystemExit(
        'This script requires scipy. Install with: pip install scipy'
    ) from exc

try:
    from sklearn.isotonic import IsotonicRegression
except Exception as exc:
    raise SystemExit(
        'This script requires scikit-learn. Install with: pip install scikit-learn'
    ) from exc


warnings.filterwarnings(
    'ignore',
    category=RuntimeWarning,
)
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
)


LEAGUE = 'NBA'

INPUT_FILE = Path(
    r'C:\basketball\nba\2025\_NBA.csv'
)

SEASON_LABEL = '2025'

MARKETS_FILE = Path(
    'docs/win/basketball/config/markets.yaml'
)

MODEL_SOURCES = (
    'dratings',
    'sdv',
    'ensemble',
)

DEFAULT_MODEL_SOURCE = 'dratings'

PRODUCTION_BUILD_JUICE = Path(
    'docs/win/basketball/scripts/01_merge/build_juice_files_core.py'
)

PRODUCTION_EV_KELLY = Path(
    'docs/win/basketball/scripts/03_edges/compute_ev_kelly_core.py'
)

PRODUCTION_SELECT = Path(
    'docs/win/basketball/scripts/04_select/basketball_select_bets_core.py'
)

LEGACY_HISTORICAL_BIAS = {
    ('NBA', 2025): {
        'margin': 0.4,
        'total': 0.4,
    },
    ('NCAAM', 2025): {
        'margin': 0.6,
        'total': 1.2,
    },
    ('WNBA', 2025): {
        'margin': 0.5,
        'total': 0.0,
    },
}

CURRENT_SETTINGS = {
    'NBA': {
        'MARGIN_BIAS': 0.4,
        'TOTAL_BIAS': 0.4,
        'SPREAD_STD': 14.1837,
        'TOTAL_STD': 19.1345,
        'ML_EDGE': 0.020,
        'SPREAD_EDGE': 0.050,
        'TOTAL_EDGE': 0.030,
    },
    'NCAAM': {
        'MARGIN_BIAS': 0.6,
        'TOTAL_BIAS': 1.2,
        'SPREAD_STD': 11.2375,
        'TOTAL_STD': 17.7495,
        'ML_EDGE': 0.015,
        'SPREAD_EDGE': 0.100,
        'TOTAL_EDGE': 0.030,
    },
    'WNBA': {
        'MARGIN_BIAS': 0.5,
        'TOTAL_BIAS': 0.0,
        'SPREAD_STD': 13.0424,
        'TOTAL_STD': 16.6675,
        'ML_EDGE': 0.005,
        'SPREAD_EDGE': 0.050,
        'TOTAL_EDGE': 0.030,
    },
}


def _complete_direct_settings() -> None:
    for cfg in CURRENT_SETTINGS.values():
        cfg.setdefault(
            'MODEL_SOURCE',
            DEFAULT_MODEL_SOURCE,
        )
        cfg.setdefault(
            'MARGIN_BIAS_RULE',
            {
                'method': 'fixed',
                'value': cfg['MARGIN_BIAS'],
            },
        )
        cfg.setdefault(
            'TOTAL_BIAS_RULE',
            {
                'method': 'fixed',
                'value': cfg['TOTAL_BIAS'],
            },
        )
        cfg.setdefault(
            'CALIBRATION',
            {
                'moneyline': {
                    'home': {
                        'method': 'none',
                    },
                    'away': {
                        'method': 'none',
                    },
                },
                'spread': {
                    'canonical_side': 'home',
                    'config': {
                        'method': 'none',
                    },
                },
                'total': {
                    'canonical_side': 'over',
                    'config': {
                        'method': 'none',
                    },
                },
            },
        )


_complete_direct_settings()


RANDOM_SEED = 20260813
LOCKBOX_FRACTION = 0.15
TARGET_OUTER_FOLDS = 10
MIN_OUTER_TEST_GAMES = 25
MIN_TRAIN_FRACTION = 0.35
STRESS_REPS = 10_000
SEASON_BLOCK_DAYS = 14
TOP_CONFIGS_TO_STRESS = 20

ROLLING_WINDOWS = [
    50,
    100,
    250,
    500,
]

STD_MODES = [
    'fixed',
    'q2',
    'q3',
    'q4',
]

STD_SHRINKAGE_GAMES = 50.0
MIN_STD = 2.0
MAX_STD = 50.0

CALIBRATION_METHODS = [
    'raw',
    'intercept_only',
    'temperature',
    'platt',
    'beta',
    'isotonic',
]

MIN_CALIBRATION_GAMES = 50

EDGE_GRID = np.round(
    np.arange(
        0.0,
        0.2501,
        0.005,
    ),
    3,
)

SPLIT_EDGE_GRID = np.round(
    np.arange(
        0.0,
        0.2001,
        0.010,
    ),
    3,
)

MIN_CALIBRATION_FOLD_WIN_RATE = 0.60
MIN_ADAPTIVE_STD_REL_NLL_IMPROVEMENT = 0.001
MIN_SPLIT_EDGE_DEV_PROFIT_IMPROVEMENT = 0.05
MIN_STRESS_POSITIVE_ROI_PROBABILITY = 0.60
EPS = 1e-10


def now_seconds() -> float:
    return time.perf_counter()


def progress(
    msg: str,
) -> None:
    print(
        msg,
        flush=True,
    )


def safe_num(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series,
        errors='coerce',
    )


def clip_prob(
    p: Any,
) -> np.ndarray:
    return np.clip(
        np.asarray(
            p,
            dtype=float,
        ),
        0.01,
        0.99,
    )


def sigmoid(
    x: Any,
) -> np.ndarray:
    x = np.clip(
        np.asarray(
            x,
            dtype=float,
        ),
        -35.0,
        35.0,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(
                -x
            )
        )
    )


def logit(
    p: Any,
) -> np.ndarray:
    p = np.clip(
        np.asarray(
            p,
            dtype=float,
        ),
        EPS,
        1.0 - EPS,
    )

    return np.log(
        p
        / (
            1.0 - p
        )
    )


def brier_score(
    p: Any,
    y: Any,
) -> float:
    p = np.asarray(
        p,
        dtype=float,
    )
    y = np.asarray(
        y,
        dtype=float,
    )

    mask = (
        np.isfinite(p)
        & np.isfinite(y)
    )

    if not mask.any():
        return np.nan

    return float(
        np.mean(
            (
                p[mask]
                - y[mask]
            )
            ** 2
        )
    )


def binary_log_loss(
    p: Any,
    y: Any,
) -> float:
    p = np.asarray(
        p,
        dtype=float,
    )
    y = np.asarray(
        y,
        dtype=float,
    )

    mask = (
        np.isfinite(p)
        & np.isfinite(y)
    )

    if not mask.any():
        return np.nan

    q = np.clip(
        p[mask],
        EPS,
        1.0 - EPS,
    )

    yy = y[mask]

    return float(
        -np.mean(
            yy
            * np.log(q)
            + (
                1.0 - yy
            )
            * np.log(
                1.0 - q
            )
        )
    )


def rmse(
    x: Any,
) -> float:
    x = np.asarray(
        x,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    return (
        float(
            np.sqrt(
                np.mean(
                    x * x
                )
            )
        )
        if len(x)
        else np.nan
    )


def mae(
    x: Any,
) -> float:
    x = np.asarray(
        x,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    return (
        float(
            np.mean(
                np.abs(x)
            )
        )
        if len(x)
        else np.nan
    )


def percentile(
    values: Any,
    q: float,
) -> float:
    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    return (
        float(
            np.percentile(
                x,
                q,
            )
        )
        if len(x)
        else np.nan
    )


def devig_pair(
    decimal_a: Any,
    decimal_b: Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    a = np.asarray(
        decimal_a,
        dtype=float,
    )
    b = np.asarray(
        decimal_b,
        dtype=float,
    )

    pa = np.where(
        a > 1.0,
        1.0 / a,
        np.nan,
    )

    pb = np.where(
        b > 1.0,
        1.0 / b,
        np.nan,
    )

    s = pa + pb

    return (
        pa / s,
        pb / s,
    )


def unit_profit_from_result(
    result: Any,
    decimal_odds: Any,
) -> np.ndarray:
    r = np.asarray(
        result,
        dtype=float,
    )

    o = np.asarray(
        decimal_odds,
        dtype=float,
    )

    out = np.full(
        len(r),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(o)
        & (
            o > 1.0
        )
    )

    out[
        valid
        & (
            r == 1.0
        )
    ] = (
        o[
            valid
            & (
                r == 1.0
            )
        ]
        - 1.0
    )

    out[
        valid
        & (
            r == 0.0
        )
    ] = -1.0

    out[
        valid
        & (
            r == 0.5
        )
    ] = 0.0

    return out


def normal_residual_nll(
    error: Any,
    sigma: Any,
) -> np.ndarray:
    e = np.asarray(
        error,
        dtype=float,
    )

    s = np.asarray(
        sigma,
        dtype=float,
    )

    s = np.clip(
        s,
        MIN_STD,
        MAX_STD,
    )

    return (
        np.log(s)
        + 0.5
        * (
            e / s
        )
        ** 2
    )


def save_csv(
    df: pd.DataFrame,
    path: Path,
) -> Path:
    df.to_csv(
        path,
        index=False,
    )

    return path


def json_compact(
    obj: Any,
) -> str:
    return json.dumps(
        obj,
        separators=(
            ',',
            ':',
        ),
        sort_keys=True,
    )


def format_float(
    v: Any,
    digits: int = 5,
) -> str:
    try:
        if (
            v is None
            or not np.isfinite(
                float(v)
            )
        ):
            return ''

        return (
            f'{float(v):.{digits}f}'
        )

    except (TypeError, ValueError, OverflowError):
        return str(v)


def make_prefix(
    league: str,
    season: str,
) -> str:
    return (
        f'{league}_{season}_'
        'FINAL_MASTER'
    )


def season_from_input_filename(
    input_file: Path,
    league: str,
) -> int | None:
    stem = input_file.stem.strip()
    suffix = '_' + league.upper()

    if not stem.upper().endswith(
        suffix
    ):
        return None

    season_text = stem[
        :-len(suffix)
    ]

    if (
        len(season_text) != 4
        or not season_text.isdigit()
    ):
        return None

    return int(
        season_text
    )


def resolve_internal_season(
    input_file: Path,
    league: str,
    requested: str | int | None,
) -> int:
    filename_season = (
        season_from_input_filename(
            input_file,
            league,
        )
    )

    if (
        requested is not None
        and str(
            requested
        ).strip()
    ):
        season_text = str(
            requested
        ).strip()

        if (
            len(season_text) != 4
            or not season_text.isdigit()
        ):
            raise ValueError(
                '--season must be a '
                'four-digit internal '
                'basketball season'
            )

        internal_season = int(
            season_text
        )

        if (
            filename_season
            is not None
            and filename_season
            != internal_season
        ):
            raise ValueError(
                f'--season={internal_season} '
                'does not match input-file '
                'internal season='
                f'{filename_season} from '
                f'{input_file.name}'
            )

        return internal_season

    if filename_season is not None:
        return filename_season

    raise ValueError(
        'Internal basketball season '
        'could not be resolved. '
        'Supply --season YYYY or use '
        'an input filename like '
        f'2025_{league.upper()}.csv'
    )


@dataclass(
    frozen=True
)
class MarketSelectionPolicy:
    selection_mode: str
    preference_metric: str
    preference_direction: str


def resolve_markets_file(
    markets_file: Path,
) -> Path:
    candidates: list[
        Path
    ] = []

    if markets_file.is_absolute():
        candidates.append(
            markets_file
        )

    else:
        candidates.append(
            Path.cwd()
            / markets_file
        )

        script_path = Path(
            __file__
        ).resolve()

        for parent in [
            script_path.parent,
            *script_path.parents,
        ]:
            candidates.append(
                parent
                / markets_file
            )

            if parent.name == 'basketball':
                candidates.append(
                    parent
                    / 'config'
                    / 'markets.yaml'
                )

    seen: set[str] = set()

    for candidate in candidates:
        key = str(
            candidate.resolve(
                strict=False
            )
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        if candidate.exists():
            return candidate.resolve()

    tried = '\n  - '.join(
        str(x)
        for x in candidates
    )

    raise FileNotFoundError(
        'Could not find production '
        'market-selection file '
        'docs/win/basketball/config/'
        'markets.yaml. Tried:\n  - '
        + tried
    )


def load_market_selection_policies(
    markets_file: Path,
    league: str,
) -> dict[
    str,
    MarketSelectionPolicy,
]:
    path = resolve_markets_file(
        markets_file
    )

    payload = (
        yaml.safe_load(
            path.read_text(
                encoding='utf-8'
            )
        )
        or {}
    )

    league_key = league.lower()

    league_cfg = (
        (
            payload.get(
                'markets'
            )
            or {}
        )
        .get(
            league_key
        )
    )

    if not isinstance(
        league_cfg,
        dict,
    ):
        raise ValueError(
            f'{path}: missing '
            f'markets.{league_key} '
            'configuration for '
            f'league={league}'
        )

    valid_modes = {
        'pick_one',
        'all_qualifying',
    }

    valid_metrics = {
        'ev',
        'kelly',
        'model_prob',
        'edge_vs_market',
    }

    valid_directions = {
        'max',
        'min',
    }

    policies: dict[
        str,
        MarketSelectionPolicy,
    ] = {}

    for market in [
        'moneyline',
        'spread',
        'total',
    ]:
        market_cfg = league_cfg.get(
            market
        )

        if not isinstance(
            market_cfg,
            dict,
        ):
            raise ValueError(
                f'{path}: missing '
                f'markets.{league_key}.'
                f'{market}'
            )

        selection_mode = str(
            market_cfg.get(
                'selection_mode',
                'pick_one',
            )
        ).strip().lower()

        pref = (
            market_cfg.get(
                'pick_preference'
            )
            or {
                'metric': 'ev',
                'direction': 'max',
            }
        )

        metric = str(
            pref.get(
                'metric',
                'ev',
            )
        ).strip().lower()

        direction = str(
            pref.get(
                'direction',
                'max',
            )
        ).strip().lower()

        if selection_mode not in valid_modes:
            raise ValueError(
                f'{path}: markets.'
                f'{league_key}.{market}.'
                'selection_mode='
                f'{selection_mode!r}; '
                'expected one of '
                f'{sorted(valid_modes)}'
            )

        if metric not in valid_metrics:
            raise ValueError(
                f'{path}: markets.'
                f'{league_key}.{market}.'
                'pick_preference.metric='
                f'{metric!r}; expected '
                'one of '
                f'{sorted(valid_metrics)}'
            )

        if direction not in valid_directions:
            raise ValueError(
                f'{path}: markets.'
                f'{league_key}.{market}.'
                'pick_preference.direction='
                f'{direction!r}; expected '
                'one of '
                f'{sorted(valid_directions)}'
            )

        policies[
            market
        ] = MarketSelectionPolicy(
            selection_mode=selection_mode,
            preference_metric=metric,
            preference_direction=direction,
        )

    return policies


def market_selection_policy_table(
    policies: dict[
        str,
        MarketSelectionPolicy,
    ],
    markets_file: Path,
) -> pd.DataFrame:
    return pd.DataFrame([
        {
            'markets_file': str(
                markets_file
            ),
            'market': market,
            'selection_mode': (
                policy.selection_mode
            ),
            'pick_preference_metric': (
                policy.preference_metric
            ),
            'pick_preference_direction': (
                policy.preference_direction
            ),
        }
        for (
            market,
            policy,
        )
        in policies.items()
    ])


def resolve_model_source(
    requested: str | None,
    settings: dict[
        str,
        Any,
    ],
) -> str:
    source = str(
        requested
        or settings.get(
            'MODEL_SOURCE'
        )
        or DEFAULT_MODEL_SOURCE
    ).strip().lower()

    if source not in MODEL_SOURCES:
        raise ValueError(
            'model_source must be one '
            'of dratings, sdv, ensemble; '
            f'got {source!r}'
        )

    return source


def normalize_production_bias_rule(
    rule: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    method = str(
        (
            rule
            or {}
        ).get(
            'method',
            '',
        )
    ).strip().lower()

    if method not in {
        'rolling',
        'regime_aware',
        'fixed',
        'none',
    }:
        raise ValueError(
            'Unsupported production '
            f'bias method={method!r}'
        )

    out: dict[
        str,
        Any,
    ] = {
        'method': method
    }

    if method == 'fixed':
        out[
            'value'
        ] = float(
            rule[
                'value'
            ]
        )

    elif method == 'rolling':
        window = int(
            rule[
                'window_games'
            ]
        )

        if window <= 0:
            raise ValueError(
                'rolling bias window_games '
                'must be > 0'
            )

        out[
            'window_games'
        ] = window

    elif method == 'regime_aware':
        windows = [
            int(v)
            for v
            in (
                rule.get(
                    'windows_games'
                )
                or []
            )
        ]

        weights = [
            float(v)
            for v
            in (
                rule.get(
                    'weights'
                )
                or []
            )
        ]

        if (
            not windows
            or len(windows)
            != len(weights)
        ):
            raise ValueError(
                'regime_aware requires '
                'matching windows_games '
                'and weights'
            )

        if any(
            v <= 0
            for v in windows
        ):
            raise ValueError(
                'regime_aware windows '
                'must be > 0'
            )

        if (
            any(
                v < 0
                for v in weights
            )
            or sum(weights)
            <= 0
        ):
            raise ValueError(
                'regime_aware weights '
                'must be >= 0 and sum '
                'to > 0'
            )

        total = sum(
            weights
        )

        weights = [
            v / total
            for v in weights
        ]

        shrink = float(
            rule[
                'sign_conflict_shrink'
            ]
        )

        if not (
            0
            <= shrink
            <= 1
        ):
            raise ValueError(
                'sign_conflict_shrink '
                'must be between 0 and 1'
            )

        out.update({
            'windows_games': windows,
            'weights': weights,
            'sign_conflict_shrink': (
                shrink
            ),
        })

    return out


def production_bias_from_errors(
    errors: list[
        float
    ],
    rule: dict[
        str,
        Any,
    ],
) -> float | None:
    rule = (
        normalize_production_bias_rule(
            rule
        )
    )

    method = rule[
        'method'
    ]

    if method == 'none':
        return 0.0

    if method == 'fixed':
        return round(
            float(
                rule[
                    'value'
                ]
            ),
            3,
        )

    if method == 'rolling':
        window = int(
            rule[
                'window_games'
            ]
        )

        if len(
            errors
        ) < window:
            return None

        return round(
            float(
                sum(
                    errors[
                        -window:
                    ]
                )
                / window
            ),
            3,
        )

    windows = [
        int(v)
        for v
        in rule[
            'windows_games'
        ]
    ]

    if len(
        errors
    ) < max(
        windows
    ):
        return None

    means = {
        window: float(
            sum(
                errors[
                    -window:
                ]
            )
            / window
        )
        for window
        in windows
    }

    weighted = sum(
        float(
            weight
        )
        * means[
            int(
                window
            )
        ]
        for (
            window,
            weight,
        )
        in zip(
            windows,
            rule[
                'weights'
            ],
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
                'sign_conflict_shrink'
            ]
        )

    return round(
        float(
            weighted
        ),
        3,
    )


def production_bias_values_for_targets(
    prior: pd.DataFrame,
    target: pd.DataFrame,
    market: str,
    rule: dict[
        str,
        Any,
    ],
) -> np.ndarray:
    error_col = (
        'required_margin_bias'
        if market == 'spread'
        else 'required_total_bias'
    )

    errors = (
        pd.to_numeric(
            prior.get(
                error_col,
                pd.Series(
                    dtype=float
                ),
            ),
            errors='coerce',
        )
        .dropna()
        .astype(float)
        .tolist()
    )

    values: list[
        float
    ] = []

    for _, row in target.iterrows():
        value = (
            production_bias_from_errors(
                errors,
                rule,
            )
        )

        values.append(
            np.nan
            if value is None
            else float(
                value
            )
        )

        current_error = safe_num(
            pd.Series([
                row.get(
                    error_col
                )
            ])
        ).iloc[0]

        if pd.notna(
            current_error
        ):
            errors.append(
                float(
                    current_error
                )
            )

    return np.asarray(
        values,
        dtype=float,
    )


def apply_production_calibration_scalar(
    value: Any,
    cfg: dict[
        str,
        Any,
    ],
) -> float:
    p = float(
        value
    )

    if not np.isfinite(
        p
    ):
        return np.nan

    method = str(
        (
            cfg
            or {}
        ).get(
            'method',
            'none',
        )
    ).strip().lower()

    if method in {
        'none',
        'raw',
        '',
    }:
        return p

    if method == 'beta':
        p = min(
            max(
                p,
                1e-12,
            ),
            1.0 - 1e-12,
        )

        z = (
            float(
                cfg[
                    'intercept'
                ]
            )
            + float(
                cfg[
                    'coef_log_p'
                ]
            )
            * math.log(
                p
            )
            + float(
                cfg[
                    'coef_log_1mp'
                ]
            )
            * math.log(
                1.0 - p
            )
        )

        if z >= 0:
            ez = math.exp(
                -z
            )

            return (
                1.0
                / (
                    1.0
                    + ez
                )
            )

        ez = math.exp(
            z
        )

        return (
            ez
            / (
                1.0
                + ez
            )
        )

    raise ValueError(
        'Unsupported production '
        'calibration method='
        f'{method!r}'
    )


def apply_production_independent_calibration(
    values: Any,
    cfg: dict[
        str,
        Any,
    ],
) -> np.ndarray:
    arr = np.asarray(
        values,
        dtype=float,
    )

    return np.asarray([
        (
            apply_production_calibration_scalar(
                v,
                cfg,
            )
            if np.isfinite(
                v
            )
            else np.nan
        )
        for v in arr
    ], dtype=float)


def apply_production_complementary_calibration(
    raw_first: Any,
    raw_second: Any,
    calibration: dict[
        str,
        Any,
    ],
    first_side: str,
    second_side: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    first = np.asarray(
        raw_first,
        dtype=float,
    )

    second = np.asarray(
        raw_second,
        dtype=float,
    )

    canonical = str(
        calibration[
            'canonical_side'
        ]
    ).strip().lower()

    cfg = calibration[
        'config'
    ]

    raw = (
        first
        if canonical == first_side
        else second
    )

    calibrated = (
        apply_production_independent_calibration(
            raw,
            cfg,
        )
    )

    calibrated = clip_prob(
        calibrated
    )

    opposite = (
        1.0 - calibrated
    )

    if canonical == first_side:
        return (
            calibrated,
            opposite,
        )

    if canonical == second_side:
        return (
            opposite,
            calibrated,
        )

    raise ValueError(
        'Invalid canonical side='
        f'{canonical!r}'
    )


def reverse_bias_row_to_raw(
    row: pd.Series,
    league: str,
    internal_season: int,
) -> tuple[
    float,
    float,
    float,
]:
    home = float(
        row[
            'home_projected_points'
        ]
    )

    away = float(
        row[
            'away_projected_points'
        ]
    )

    total = float(
        row[
            'total_projected_points'
        ]
    )

    if not (
        np.isfinite(
            home
        )
        and np.isfinite(
            away
        )
        and np.isfinite(
            total
        )
    ):
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    flag = row.get(
        'bias_applied',
        0,
    )

    try:
        flag = float(
            flag
        )

    except (TypeError, ValueError, OverflowError):
        flag = 0.0

    if flag == 0.0:
        return (
            home,
            away,
            total,
        )

    if flag != 1.0:
        raise ValueError(
            f"game_id={row.get('game_id')} "
            'invalid bias_applied='
            f'{flag!r}'
        )

    margin_bias = row.get(
        'margin_bias',
        np.nan,
    )

    total_bias = row.get(
        'total_bias',
        np.nan,
    )

    margin_bias = (
        float(
            margin_bias
        )
        if pd.notna(
            margin_bias
        )
        else np.nan
    )

    total_bias = (
        float(
            total_bias
        )
        if pd.notna(
            total_bias
        )
        else np.nan
    )

    if (
        not np.isfinite(
            margin_bias
        )
        or not np.isfinite(
            total_bias
        )
    ):
        legacy = (
            LEGACY_HISTORICAL_BIAS
            .get(
                (
                    league,
                    int(
                        internal_season
                    ),
                )
            )
        )

        if legacy is None:
            raise ValueError(
                f"game_id={row.get('game_id')} "
                'in internal season '
                f'{internal_season} has '
                'bias_applied=1 but '
                'no per-game '
                'margin_bias/total_bias '
                'and no exact legacy '
                'fallback for '
                f'{league} season '
                f'{internal_season}'
            )

        margin_bias = float(
            legacy[
                'margin'
            ]
        )

        total_bias = float(
            legacy[
                'total'
            ]
        )

    return (
        home
        + margin_bias
        / 2.0
        + total_bias
        / 2.0,
        away
        - margin_bias
        / 2.0
        + total_bias
        / 2.0,
        total
        + total_bias,
    )


def load_module_from_path(
    name: str,
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            'Missing production parity '
            f'module: {path}'
        )

    spec = (
        importlib.util
        .spec_from_file_location(
            name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            'Unable to load production '
            f'parity module: {path}'
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def load_data(
    input_file: Path,
    league: str,
    model_source: str,
    internal_season: int,
) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(
            f'Input file not found: '
            f'{input_file}'
        )

    df = pd.read_csv(
        input_file
    )

    if 'model_source' in df.columns:
        seen_sources = {
            str(v).strip().lower()
            for v
            in df[
                'model_source'
            ].dropna().tolist()
            if str(v).strip()
        }

        if (
            seen_sources
            and seen_sources
            != {
                model_source
            }
        ):
            raise ValueError(
                'Input model_source values '
                f'{sorted(seen_sources)} '
                'do not match requested '
                f'model_source={model_source}'
            )

    df[
        'model_source'
    ] = model_source

    required = [
        'game_date',
        'game_id',
        'home_team',
        'away_team',
        'home_spread',
        'away_spread',
        'total',
        'home_dk_moneyline_decimal',
        'away_dk_moneyline_decimal',
        'home_dk_spread_decimal',
        'away_dk_spread_decimal',
        'dk_total_over_decimal',
        'dk_total_under_decimal',
        'home_prob',
        'away_prob',
        'home_projected_points',
        'away_projected_points',
        'total_projected_points',
        'home_score',
        'away_score',
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            'Missing required columns: '
            + ', '.join(
                missing
            )
        )

    if 'bias_applied' not in df.columns:
        df[
            'bias_applied'
        ] = 0

    numeric = [
        'home_spread',
        'away_spread',
        'total',
        'home_dk_moneyline_decimal',
        'away_dk_moneyline_decimal',
        'home_dk_spread_decimal',
        'away_dk_spread_decimal',
        'dk_total_over_decimal',
        'dk_total_under_decimal',
        'home_prob',
        'away_prob',
        'home_projected_points',
        'away_projected_points',
        'total_projected_points',
        'home_score',
        'away_score',
        'bias_applied',
    ]

    for optional_bias_col in (
        'margin_bias',
        'total_bias',
    ):
        if optional_bias_col in df.columns:
            numeric.append(
                optional_bias_col
            )

    for c in numeric:
        df[
            c
        ] = safe_num(
            df[
                c
            ]
        )

    complete_cols = [
        'home_projected_points',
        'away_projected_points',
        'total_projected_points',
        'home_score',
        'away_score',
    ]

    complete_mask = np.ones(
        len(df),
        dtype=bool,
    )

    for c in complete_cols:
        complete_mask &= np.isfinite(
            df[
                c
            ].to_numpy(
                float
            )
        )

    incomplete_rows = int(
        (
            ~complete_mask
        ).sum()
    )

    if incomplete_rows:
        progress(
            'Skipping '
            f'{incomplete_rows} '
            'incomplete historical rows '
            'with missing projection or '
            'final-score values'
        )

        df = (
            df.loc[
                complete_mask
            ]
            .copy()
        )

    df[
        '_date'
    ] = pd.to_datetime(
        df[
            'game_date'
        ]
        .astype(str)
        .str.replace(
            '_',
            '-',
            regex=False,
        ),
        errors='coerce',
    )

    df = (
        df[
            df[
                '_date'
            ].notna()
        ]
        .copy()
    )

    df = (
        df.sort_values(
            [
                '_date',
                'game_id',
            ],
            kind='stable',
        )
        .reset_index(
            drop=True
        )
    )

    df[
        '_row_id'
    ] = np.arange(
        len(df),
        dtype=int,
    )

    df[
        '_week'
    ] = (
        df[
            '_date'
        ]
        .dt.to_period(
            'W-SUN'
        )
        .astype(str)
    )

    df[
        '_month'
    ] = (
        df[
            '_date'
        ]
        .dt.to_period(
            'M'
        )
        .astype(str)
    )

    df[
        'actual_margin'
    ] = (
        df[
            'home_score'
        ]
        - df[
            'away_score'
        ]
    )

    df[
        'actual_total_calc'
    ] = (
        df[
            'home_score'
        ]
        + df[
            'away_score'
        ]
    )

    raw_triplets = [
        reverse_bias_row_to_raw(
            row,
            league,
            internal_season,
        )
        for (
            _,
            row,
        )
        in df.iterrows()
    ]

    df[
        'raw_home_projected'
    ] = [
        v[0]
        for v in raw_triplets
    ]

    df[
        'raw_away_projected'
    ] = [
        v[1]
        for v in raw_triplets
    ]

    df[
        'raw_total'
    ] = [
        v[2]
        for v in raw_triplets
    ]

    df[
        'raw_margin'
    ] = (
        df[
            'raw_home_projected'
        ]
        - df[
            'raw_away_projected'
        ]
    )

    df[
        'required_margin_bias'
    ] = (
        df[
            'raw_margin'
        ]
        - df[
            'actual_margin'
        ]
    )

    df[
        'required_total_bias'
    ] = (
        df[
            'raw_total'
        ]
        - df[
            'actual_total_calc'
        ]
    )

    df[
        'home_win_result'
    ] = np.where(
        df[
            'actual_margin'
        ] > 0,
        1.0,
        np.where(
            df[
                'actual_margin'
            ] < 0,
            0.0,
            0.5,
        ),
    )

    spread_value = (
        df[
            'actual_margin'
        ]
        + df[
            'home_spread'
        ]
    )

    df[
        'home_spread_result'
    ] = np.where(
        spread_value > 0,
        1.0,
        np.where(
            spread_value < 0,
            0.0,
            0.5,
        ),
    )

    total_value = (
        df[
            'actual_total_calc'
        ]
        - df[
            'total'
        ]
    )

    df[
        'over_result'
    ] = np.where(
        total_value > 0,
        1.0,
        np.where(
            total_value < 0,
            0.0,
            0.5,
        ),
    )

    df[
        'home_win_y'
    ] = df[
        'home_win_result'
    ].replace({
        0.5: np.nan
    })

    df[
        'away_win_y'
    ] = (
        1.0
        - df[
            'home_win_y'
        ]
    )

    df[
        'home_spread_y'
    ] = df[
        'home_spread_result'
    ].replace({
        0.5: np.nan
    })

    df[
        'away_spread_y'
    ] = (
        1.0
        - df[
            'home_spread_y'
        ]
    )

    df[
        'over_y'
    ] = df[
        'over_result'
    ].replace({
        0.5: np.nan
    })

    df[
        'under_y'
    ] = (
        1.0
        - df[
            'over_y'
        ]
    )

    h, a = devig_pair(
        df[
            'home_dk_moneyline_decimal'
        ],
        df[
            'away_dk_moneyline_decimal'
        ],
    )

    df[
        'market_home_ml_prob'
    ] = h

    df[
        'market_away_ml_prob'
    ] = a

    h, a = devig_pair(
        df[
            'home_dk_spread_decimal'
        ],
        df[
            'away_dk_spread_decimal'
        ],
    )

    df[
        'market_home_spread_prob'
    ] = h

    df[
        'market_away_spread_prob'
    ] = a

    o, u = devig_pair(
        df[
            'dk_total_over_decimal'
        ],
        df[
            'dk_total_under_decimal'
        ],
    )

    df[
        'market_over_prob'
    ] = o

    df[
        'market_under_prob'
    ] = u

    if 'league' in df.columns:
        league_values = (
            df[
                'league'
            ]
            .astype(str)
            .str.upper()
            .dropna()
            .unique()
            .tolist()
        )

        if (
            league_values
            and league
            not in league_values
        ):
            print(
                'WARNING: configured '
                f'league={league}, '
                'file league values='
                f'{league_values}'
            )

    return df


def split_development_lockbox(
    df: pd.DataFrame,
    fraction: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    n = len(
        df
    )

    raw_cut = int(
        math.floor(
            n
            * (
                1.0
                - fraction
            )
        )
    )

    raw_cut = min(
        max(
            raw_cut,
            1,
        ),
        n - 1,
    )

    cut_date = df.iloc[
        raw_cut
    ][
        '_date'
    ]

    first_lock = int(
        df.index[
            df[
                '_date'
            ]
            >= cut_date
        ][0]
    )

    if first_lock < int(
        n
        * 0.70
    ):
        first_lock = raw_cut

    dev = (
        df.iloc[
            :first_lock
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    lockbox = (
        df.iloc[
            first_lock:
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return (
        dev,
        lockbox,
    )


def make_outer_folds(
    dev: pd.DataFrame,
    target_folds: int,
) -> list[
    tuple[
        int,
        np.ndarray,
        np.ndarray,
    ]
]:
    n = len(
        dev
    )

    if n < 120:
        raise ValueError(
            'Not enough development '
            'games for final test: '
            f'{n}'
        )

    min_train = max(
        80,
        int(
            round(
                n
                * MIN_TRAIN_FRACTION
            )
        ),
    )

    remaining = (
        n - min_train
    )

    folds_target = min(
        target_folds,
        max(
            4,
            remaining
            // MIN_OUTER_TEST_GAMES,
        ),
    )

    test_target = max(
        MIN_OUTER_TEST_GAMES,
        remaining
        // folds_target,
    )

    folds: list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
        ]
    ] = []

    start = min_train
    fold_id = 1

    if start < n:
        d = dev.iloc[
            start
        ][
            '_date'
        ]

        while (
            start > 0
            and dev.iloc[
                start - 1
            ][
                '_date'
            ]
            == d
        ):
            start -= 1

    while (
        start < n
        and fold_id
        <= target_folds + 2
    ):
        end = min(
            start
            + test_target,
            n,
        )

        if end < n:
            d = dev.iloc[
                end - 1
            ][
                '_date'
            ]

            while (
                end < n
                and dev.iloc[
                    end
                ][
                    '_date'
                ]
                == d
            ):
                end += 1

        if (
            end - start
            < MIN_OUTER_TEST_GAMES
            and folds
        ):
            (
                prev_id,
                tr,
                te,
            ) = folds[
                -1
            ]

            folds[
                -1
            ] = (
                prev_id,
                tr,
                np.arange(
                    te[0],
                    n,
                ),
            )

            break

        train_idx = np.arange(
            0,
            start,
        )

        test_idx = np.arange(
            start,
            end,
        )

        folds.append(
            (
                fold_id,
                train_idx,
                test_idx,
            )
        )

        fold_id += 1
        start = end

    return folds


def all_oos_indices(
    folds: list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
        ]
    ],
) -> np.ndarray:
    return np.concatenate([
        test
        for (
            _,
            _,
            test,
        )
        in folds
    ])


def bias_strategy_names(
    n_dev: int,
) -> list[str]:
    names = [
        'none',
        'fixed',
    ]

    for w in ROLLING_WINDOWS:
        if w <= max(
            50,
            n_dev,
        ):
            names.append(
                f'rolling_{w}'
            )

    return names


def estimate_bias(
    train: pd.DataFrame,
    market: str,
    strategy: str,
) -> float:
    col = (
        'required_margin_bias'
        if market == 'spread'
        else 'required_total_bias'
    )

    vals = (
        pd.to_numeric(
            train[
                col
            ],
            errors='coerce',
        )
        .dropna()
    )

    if (
        vals.empty
        or strategy == 'none'
    ):
        return 0.0

    if strategy == 'fixed':
        return float(
            vals.mean()
        )

    if strategy.startswith(
        'rolling_'
    ):
        w = int(
            strategy.split(
                '_'
            )[1]
        )

        return float(
            vals.tail(
                w
            ).mean()
        )

    raise ValueError(
        f'Unknown bias strategy: '
        f'{strategy}'
    )


def evaluate_bias_strategies(
    dev: pd.DataFrame,
    folds: list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
        ]
    ],
    market: str,
    strategies: list[str],
    current_bias: float,
    current_rule: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    raw_col = (
        'raw_margin'
        if market == 'spread'
        else 'raw_total'
    )

    actual_col = (
        'actual_margin'
        if market == 'spread'
        else 'actual_total_calc'
    )

    details = []

    candidates = (
        strategies
        + [
            'current_pipeline'
        ]
    )

    for (
        fold_id,
        train_idx,
        test_idx,
    ) in folds:
        train = dev.iloc[
            train_idx
        ]

        test = dev.iloc[
            test_idx
        ]

        for strategy in candidates:
            if (
                strategy
                == 'current_pipeline'
                and current_rule
                is not None
            ):
                bias_values = (
                    production_bias_values_for_targets(
                        train,
                        test,
                        market,
                        current_rule,
                    )
                )

            elif (
                strategy
                == 'current_pipeline'
            ):
                bias_values = np.repeat(
                    float(
                        current_bias
                    ),
                    len(
                        test
                    ),
                )

            else:
                bias_values = np.repeat(
                    estimate_bias(
                        train,
                        market,
                        strategy,
                    ),
                    len(
                        test
                    ),
                )

            pred = (
                test[
                    raw_col
                ].to_numpy(
                    float
                )
                - bias_values
            )

            actual = test[
                actual_col
            ].to_numpy(
                float
            )

            err = (
                actual
                - pred
            )

            for (
                row_id,
                week,
                date,
                e,
                bias_used,
            ) in zip(
                test[
                    '_row_id'
                ],
                test[
                    '_week'
                ],
                test[
                    '_date'
                ],
                err,
                bias_values,
            ):
                if (
                    np.isfinite(
                        e
                    )
                    and np.isfinite(
                        bias_used
                    )
                ):
                    details.append({
                        'market': market,
                        'fold_id': fold_id,
                        'strategy': strategy,
                        'bias_used': float(
                            bias_used
                        ),
                        'row_id': int(
                            row_id
                        ),
                        'week': str(
                            week
                        ),
                        'date': date,
                        'error': float(
                            e
                        ),
                    })

    detail_df = pd.DataFrame(
        details
    )

    summary_rows = []

    for (
        strategy,
        g,
    ) in detail_df.groupby(
        'strategy',
        sort=False,
    ):
        fold_metrics = (
            g.groupby(
                'fold_id'
            )
            .agg(
                fold_rmse=(
                    'error',
                    lambda x: float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    x
                                )
                            )
                        )
                    ),
                ),
                fold_mae=(
                    'error',
                    lambda x: float(
                        np.mean(
                            np.abs(
                                x
                            )
                        )
                    ),
                ),
            )
        )

        summary_rows.append({
            'market': market,
            'strategy': strategy,
            'oos_games': len(
                g
            ),
            'oos_rmse': rmse(
                g[
                    'error'
                ]
            ),
            'oos_mae': mae(
                g[
                    'error'
                ]
            ),
            'oos_mean_error': float(
                g[
                    'error'
                ].mean()
            ),
            'median_fold_rmse': float(
                fold_metrics[
                    'fold_rmse'
                ].median()
            ),
            'positive_error_folds': int(
                (
                    g.groupby(
                        'fold_id'
                    )[
                        'error'
                    ].mean()
                    > 0
                ).sum()
            ),
            'folds': int(
                g[
                    'fold_id'
                ].nunique()
            ),
        })

    summary_df = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            [
                'oos_rmse',
                'oos_mae',
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        detail_df,
        summary_df,
    )


def stress_bias_strategies(
    detail: pd.DataFrame,
    strategies: list[str],
    reps: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    x = (
        detail[
            detail[
                'strategy'
            ].isin(
                strategies
            )
        ]
        .copy()
    )

    weeks = sorted(
        x[
            'week'
        ].unique().tolist()
    )

    if not weeks:
        return pd.DataFrame()

    strat_index = {
        s: i
        for (
            i,
            s,
        )
        in enumerate(
            strategies
        )
    }

    week_index = {
        w: i
        for (
            i,
            w,
        )
        in enumerate(
            weeks
        )
    }

    sse = np.zeros(
        (
            len(
                weeks
            ),
            len(
                strategies
            ),
        ),
        dtype=float,
    )

    cnt = np.zeros(
        (
            len(
                weeks
            ),
            len(
                strategies
            ),
        ),
        dtype=float,
    )

    for (
        week,
        strategy,
    ), g in x.groupby(
        [
            'week',
            'strategy',
        ]
    ):
        wi = week_index[
            week
        ]

        si = strat_index[
            strategy
        ]

        e = g[
            'error'
        ].to_numpy(
            float
        )

        sse[
            wi,
            si
        ] = np.nansum(
            e * e
        )

        cnt[
            wi,
            si
        ] = np.sum(
            np.isfinite(
                e
            )
        )

    draw_counts = rng.multinomial(
        len(
            weeks
        ),
        np.repeat(
            1.0
            / len(
                weeks
            ),
            len(
                weeks
            ),
        ),
        size=reps,
    )

    total_sse = (
        draw_counts
        @ sse
    )

    total_cnt = (
        draw_counts
        @ cnt
    )

    rmses = np.sqrt(
        np.divide(
            total_sse,
            total_cnt,
            out=np.full_like(
                total_sse,
                np.nan,
            ),
            where=(
                total_cnt
                > 0
            ),
        )
    )

    winners = np.nanargmin(
        rmses,
        axis=1,
    )

    rows = []

    for (
        s,
        si,
    ) in strat_index.items():
        vals = rmses[
            :,
            si,
        ]

        rows.append({
            'strategy': s,
            'selection_frequency': float(
                np.mean(
                    winners
                    == si
                )
            ),
            'bootstrap_rmse_median': percentile(
                vals,
                50,
            ),
            'bootstrap_rmse_2_5': percentile(
                vals,
                2.5,
            ),
            'bootstrap_rmse_97_5': percentile(
                vals,
                97.5,
            ),
        })

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            'selection_frequency',
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


@dataclass
class StdModel:
    mode: str
    global_sigma: float
    edges: list[float]
    sigmas: list[float]
    counts: list[int]


def _std_feature(
    df: pd.DataFrame,
    market: str,
) -> np.ndarray:
    if market == 'spread':
        return np.abs(
            df[
                'home_spread'
            ].to_numpy(
                float
            )
        )

    return df[
        'total'
    ].to_numpy(
        float
    )


def _residual_after_bias(
    df: pd.DataFrame,
    market: str,
    bias: float,
) -> np.ndarray:
    if market == 'spread':
        pred = (
            df[
                'raw_margin'
            ].to_numpy(
                float
            )
            - bias
        )

        actual = df[
            'actual_margin'
        ].to_numpy(
            float
        )

    else:
        pred = (
            df[
                'raw_total'
            ].to_numpy(
                float
            )
            - bias
        )

        actual = df[
            'actual_total_calc'
        ].to_numpy(
            float
        )

    return (
        actual
        - pred
    )


def fit_std_model(
    train: pd.DataFrame,
    market: str,
    bias: float,
    mode: str,
) -> StdModel:
    err = _residual_after_bias(
        train,
        market,
        bias,
    )

    feature = _std_feature(
        train,
        market,
    )

    mask = np.isfinite(
        err
    )

    global_var = (
        float(
            np.mean(
                err[
                    mask
                ]
                ** 2
            )
        )
        if mask.any()
        else 100.0
    )

    global_sigma = float(
        np.clip(
            np.sqrt(
                max(
                    global_var,
                    1e-8,
                )
            ),
            MIN_STD,
            MAX_STD,
        )
    )

    if mode == 'fixed':
        return StdModel(
            mode=mode,
            global_sigma=global_sigma,
            edges=[],
            sigmas=[
                global_sigma
            ],
            counts=[
                int(
                    mask.sum()
                )
            ],
        )

    q = int(
        mode[1:]
    )

    valid = (
        np.isfinite(
            feature
        )
        & np.isfinite(
            err
        )
    )

    if valid.sum() < max(
        q * 30,
        80,
    ):
        return StdModel(
            mode='fixed',
            global_sigma=global_sigma,
            edges=[],
            sigmas=[
                global_sigma
            ],
            counts=[
                int(
                    valid.sum()
                )
            ],
        )

    probs = (
        np.arange(
            1,
            q,
        )
        / q
    )

    cuts = np.unique(
        np.quantile(
            feature[
                valid
            ],
            probs,
        )
    )

    if len(
        cuts
    ) != q - 1:
        return StdModel(
            mode='fixed',
            global_sigma=global_sigma,
            edges=[],
            sigmas=[
                global_sigma
            ],
            counts=[
                int(
                    valid.sum()
                )
            ],
        )

    bin_id = np.digitize(
        feature,
        cuts,
        right=True,
    )

    sigmas: list[
        float
    ] = []

    counts: list[
        int
    ] = []

    for b in range(
        q
    ):
        m = (
            valid
            & (
                bin_id == b
            )
        )

        n = int(
            m.sum()
        )

        counts.append(
            n
        )

        if n == 0:
            sigmas.append(
                global_sigma
            )
            continue

        bin_var = float(
            np.mean(
                err[
                    m
                ]
                ** 2
            )
        )

        shrunk_var = (
            (
                n
                * bin_var
            )
            + (
                STD_SHRINKAGE_GAMES
                * global_var
            )
        ) / (
            n
            + STD_SHRINKAGE_GAMES
        )

        sigmas.append(
            float(
                np.clip(
                    np.sqrt(
                        max(
                            shrunk_var,
                            1e-8,
                        )
                    ),
                    MIN_STD,
                    MAX_STD,
                )
            )
        )

    return StdModel(
        mode=mode,
        global_sigma=global_sigma,
        edges=[
            float(x)
            for x
            in cuts
        ],
        sigmas=sigmas,
        counts=counts,
    )


def apply_std_model(
    model: StdModel,
    df: pd.DataFrame,
    market: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    feature = _std_feature(
        df,
        market,
    )

    if (
        model.mode == 'fixed'
        or not model.edges
    ):
        return (
            np.repeat(
                model.global_sigma,
                len(
                    df
                ),
            ),
            np.zeros(
                len(
                    df
                ),
                dtype=int,
            ),
        )

    bins = np.digitize(
        feature,
        np.asarray(
            model.edges,
            dtype=float,
        ),
        right=True,
    )

    sigma = np.asarray([
        model.sigmas[
            min(
                max(
                    int(b),
                    0,
                ),
                len(
                    model.sigmas
                )
                - 1,
            )
        ]
        for b
        in bins
    ], dtype=float)

    return (
        sigma,
        bins,
    )


def home_spread_probability(
    mean_margin: Any,
    home_spread: Any,
    sigma: Any,
) -> np.ndarray:
    mean_margin = np.asarray(
        mean_margin,
        dtype=float,
    )

    home_spread = np.asarray(
        home_spread,
        dtype=float,
    )

    sigma = np.asarray(
        sigma,
        dtype=float,
    )

    return clip_prob(
        1.0
        - norm.cdf(
            -home_spread,
            loc=mean_margin,
            scale=sigma,
        )
    )


def over_probability(
    mean_total: Any,
    book_total: Any,
    sigma: Any,
) -> np.ndarray:
    mean_total = np.asarray(
        mean_total,
        dtype=float,
    )

    book_total = np.asarray(
        book_total,
        dtype=float,
    )

    sigma = np.asarray(
        sigma,
        dtype=float,
    )

    return clip_prob(
        1.0
        - norm.cdf(
            book_total,
            loc=mean_total,
            scale=sigma,
        )
    )


def fit_base_model(
    train: pd.DataFrame,
    market: str,
    bias_strategy: str,
    std_mode: str,
) -> dict[
    str,
    Any,
]:
    if market == 'moneyline':
        return {
            'market': market
        }

    bias = estimate_bias(
        train,
        market,
        bias_strategy,
    )

    std_model = fit_std_model(
        train,
        market,
        bias,
        std_mode,
    )

    return {
        'market': market,
        'bias_strategy': bias_strategy,
        'bias': float(
            bias
        ),
        'std_model': std_model,
    }


def apply_base_model(
    model: dict[
        str,
        Any,
    ],
    df: pd.DataFrame,
) -> dict[
    str,
    np.ndarray,
]:
    market = model[
        'market'
    ]

    if market == 'moneyline':
        return {
            'side1_prob': clip_prob(
                df[
                    'home_prob'
                ].to_numpy(
                    float
                )
            ),
            'side2_prob': clip_prob(
                df[
                    'away_prob'
                ].to_numpy(
                    float
                )
            ),
            'mean': np.full(
                len(
                    df
                ),
                np.nan,
            ),
            'sigma': np.full(
                len(
                    df
                ),
                np.nan,
            ),
            'range_bin': np.full(
                len(
                    df
                ),
                -1,
                dtype=int,
            ),
        }

    bias = float(
        model[
            'bias'
        ]
    )

    std_model: StdModel = (
        model[
            'std_model'
        ]
    )

    sigma, bins = (
        apply_std_model(
            std_model,
            df,
            market,
        )
    )

    if market == 'spread':
        mean = (
            df[
                'raw_margin'
            ].to_numpy(
                float
            )
            - bias
        )

        p1 = (
            home_spread_probability(
                mean,
                df[
                    'home_spread'
                ].to_numpy(
                    float
                ),
                sigma,
            )
        )

    else:
        mean = (
            df[
                'raw_total'
            ].to_numpy(
                float
            )
            - bias
        )

        p1 = over_probability(
            mean,
            df[
                'total'
            ].to_numpy(
                float
            ),
            sigma,
        )

    return {
        'side1_prob': p1,
        'side2_prob': (
            1.0 - p1
        ),
        'mean': mean,
        'sigma': sigma,
        'range_bin': bins,
    }


def evaluate_std_modes(
    dev: pd.DataFrame,
    folds: list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
        ]
    ],
    market: str,
    bias_strategy: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    details = []

    for (
        fold_id,
        train_idx,
        test_idx,
    ) in folds:
        train = dev.iloc[
            train_idx
        ]

        test = dev.iloc[
            test_idx
        ]

        bias = estimate_bias(
            train,
            market,
            bias_strategy,
        )

        err = _residual_after_bias(
            test,
            market,
            bias,
        )

        for mode in STD_MODES:
            sm = fit_std_model(
                train,
                market,
                bias,
                mode,
            )

            sigma, bins = (
                apply_std_model(
                    sm,
                    test,
                    market,
                )
            )

            if market == 'spread':
                mean = (
                    test[
                        'raw_margin'
                    ].to_numpy(
                        float
                    )
                    - bias
                )

                p = (
                    home_spread_probability(
                        mean,
                        test[
                            'home_spread'
                        ].to_numpy(
                            float
                        ),
                        sigma,
                    )
                )

                y = test[
                    'home_spread_y'
                ].to_numpy(
                    float
                )

            else:
                mean = (
                    test[
                        'raw_total'
                    ].to_numpy(
                        float
                    )
                    - bias
                )

                p = over_probability(
                    mean,
                    test[
                        'total'
                    ].to_numpy(
                        float
                    ),
                    sigma,
                )

                y = test[
                    'over_y'
                ].to_numpy(
                    float
                )

            nll = (
                normal_residual_nll(
                    err,
                    sigma,
                )
            )

            prob_ll = np.full(
                len(
                    test
                ),
                np.nan,
            )

            valid = np.isfinite(
                y
            )

            if valid.any():
                yy = y[
                    valid
                ]

                pp = np.clip(
                    p[
                        valid
                    ],
                    EPS,
                    1.0 - EPS,
                )

                prob_ll[
                    valid
                ] = -(
                    yy
                    * np.log(
                        pp
                    )
                    + (
                        1.0
                        - yy
                    )
                    * np.log(
                        1.0
                        - pp
                    )
                )

            for i in range(
                len(
                    test
                )
            ):
                if (
                    np.isfinite(
                        err[i]
                    )
                    and np.isfinite(
                        sigma[i]
                    )
                ):
                    details.append({
                        'market': market,
                        'fold_id': fold_id,
                        'mode': mode,
                        'row_id': int(
                            test.iloc[
                                i
                            ][
                                '_row_id'
                            ]
                        ),
                        'week': str(
                            test.iloc[
                                i
                            ][
                                '_week'
                            ]
                        ),
                        'date': (
                            test.iloc[
                                i
                            ][
                                '_date'
                            ]
                        ),
                        'bias_used': bias,
                        'range_bin': int(
                            bins[
                                i
                            ]
                        ),
                        'sigma_used': float(
                            sigma[
                                i
                            ]
                        ),
                        'residual_error': float(
                            err[
                                i
                            ]
                        ),
                        'residual_nll': float(
                            nll[
                                i
                            ]
                        ),
                        'prob_log_loss': (
                            float(
                                prob_ll[
                                    i
                                ]
                            )
                            if np.isfinite(
                                prob_ll[
                                    i
                                ]
                            )
                            else np.nan
                        ),
                    })

    detail_df = pd.DataFrame(
        details
    )

    summary_rows = []

    for (
        mode,
        g,
    ) in detail_df.groupby(
        'mode',
        sort=False,
    ):

        summary_rows.append({
            'market': market,
            'bias_strategy': bias_strategy,
            'std_mode': mode,
            'oos_games': len(
                g
            ),
            'mean_residual_nll': float(
                g[
                    'residual_nll'
                ].mean()
            ),
            'mean_probability_log_loss': float(
                g[
                    'prob_log_loss'
                ].mean()
            ),
            'oos_residual_rmse': rmse(
                g[
                    'residual_error'
                ]
            ),
            'median_sigma_used': float(
                g[
                    'sigma_used'
                ].median()
            ),
            'folds_winning_residual_nll': 0,
            'folds': int(
                g[
                    'fold_id'
                ].nunique()
            ),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    pivot = (
        detail_df
        .groupby(
            [
                'fold_id',
                'mode',
            ]
        )[
            'residual_nll'
        ]
        .mean()
        .unstack(
            'mode'
        )
    )

    if not pivot.empty:
        winners = pivot.idxmin(
            axis=1
        )

        for mode in summary[
            'std_mode'
        ]:
            summary.loc[
                summary[
                    'std_mode'
                ] == mode,
                'folds_winning_residual_nll',
            ] = int(
                (
                    winners
                    == mode
                ).sum()
            )

    summary = (
        summary
        .sort_values(
            [
                'mean_residual_nll',
                'mean_probability_log_loss',
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        detail_df,
        summary,
    )


def stress_std_modes(
    detail: pd.DataFrame,
    modes: list[str],
    reps: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    x = (
        detail[
            detail[
                'mode'
            ].isin(
                modes
            )
        ]
        .copy()
    )

    weeks = sorted(
        x[
            'week'
        ].unique().tolist()
    )

    if not weeks:
        return pd.DataFrame()

    mi = {
        m: i
        for (
            i,
            m,
        )
        in enumerate(
            modes
        )
    }

    wi = {
        w: i
        for (
            i,
            w,
        )
        in enumerate(
            weeks
        )
    }

    sum_nll = np.zeros(
        (
            len(
                weeks
            ),
            len(
                modes
            ),
        )
    )

    cnt = np.zeros(
        (
            len(
                weeks
            ),
            len(
                modes
            ),
        )
    )

    for (
        week,
        mode,
    ), g in x.groupby(
        [
            'week',
            'mode',
        ]
    ):
        i, j = (
            wi[
                week
            ],
            mi[
                mode
            ],
        )

        vals = (
            g[
                'residual_nll'
            ].to_numpy(
                float
            )
        )

        sum_nll[
            i,
            j
        ] = np.nansum(
            vals
        )

        cnt[
            i,
            j
        ] = np.sum(
            np.isfinite(
                vals
            )
        )

    draws = rng.multinomial(
        len(
            weeks
        ),
        np.repeat(
            1
            / len(
                weeks
            ),
            len(
                weeks
            ),
        ),
        size=reps,
    )

    numer = (
        draws
        @ sum_nll
    )

    denom = (
        draws
        @ cnt
    )

    means = np.divide(
        numer,
        denom,
        out=np.full_like(
            numer,
            np.nan,
        ),
        where=(
            denom > 0
        ),
    )

    winners = np.nanargmin(
        means,
        axis=1,
    )

    rows = []

    for (
        mode,
        j,
    ) in mi.items():
        vals = means[
            :,
            j,
        ]

        rows.append({
            'std_mode': mode,
            'selection_frequency': float(
                np.mean(
                    winners
                    == j
                )
            ),
            'bootstrap_mean_nll_median': percentile(
                vals,
                50,
            ),
            'bootstrap_mean_nll_2_5': percentile(
                vals,
                2.5,
            ),
            'bootstrap_mean_nll_97_5': percentile(
                vals,
                97.5,
            ),
        })

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            'selection_frequency',
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def _clean_py(
    p: Any,
    y: Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    pp = np.asarray(
        p,
        dtype=float,
    )

    yy = np.asarray(
        y,
        dtype=float,
    )

    mask = (
        np.isfinite(
            pp
        )
        & np.isfinite(
            yy
        )
    )

    return (
        np.clip(
            pp[
                mask
            ],
            0.01,
            0.99,
        ),
        yy[
            mask
        ],
    )


def fit_calibrator(
    p: Any,
    y: Any,
    method: str,
) -> dict[
    str,
    Any,
]:
    pp, yy = _clean_py(
        p,
        y,
    )

    if (
        method == 'raw'
        or len(
            pp
        ) < MIN_CALIBRATION_GAMES
        or len(
            np.unique(
                yy
            )
        ) < 2
    ):
        return {
            'method': 'raw'
        }

    if method == 'isotonic':
        try:
            iso = IsotonicRegression(
                y_min=0.01,
                y_max=0.99,
                out_of_bounds='clip',
            )

            iso.fit(
                pp,
                yy,
            )

            return {
                'method': 'isotonic',
                'model': iso,
                'x_thresholds': [
                    float(x)
                    for x
                    in iso.X_thresholds_
                ],
                'y_thresholds': [
                    float(x)
                    for x
                    in iso.y_thresholds_
                ],
            }

        except (TypeError, ValueError):
            return {
                'method': 'raw'
            }

    x = logit(
        pp
    )

    def nll_from_q(
        q: np.ndarray,
    ) -> float:
        q = np.clip(
            q,
            EPS,
            1.0 - EPS,
        )

        return float(
            -np.sum(
                (
                    yy
                    * np.log(
                        q
                    )
                )
                + (
                    (
                        1.0
                        - yy
                    )
                    * np.log(
                        1.0
                        - q
                    )
                )
            )
        )

    if method == 'intercept_only':
        res = minimize(
            lambda b: nll_from_q(
                sigmoid(
                    b[0]
                    + x
                )
            ),
            np.array([
                0.0
            ]),
            method='L-BFGS-B',
            bounds=[
                (
                    -5.0,
                    5.0,
                )
            ],
        )

        return {
            'method': method,
            'intercept': float(
                res.x[0]
            ),
            'slope': 1.0,
        }

    if method == 'temperature':
        res = minimize(
            lambda b: nll_from_q(
                sigmoid(
                    b[0]
                    * x
                )
            ),
            np.array([
                1.0
            ]),
            method='L-BFGS-B',
            bounds=[
                (
                    0.05,
                    5.0,
                )
            ],
        )

        return {
            'method': method,
            'intercept': 0.0,
            'slope': float(
                res.x[0]
            ),
        }

    if method == 'platt':
        res = minimize(
            lambda b: nll_from_q(
                sigmoid(
                    b[0]
                    + b[1]
                    * x
                )
            ),
            np.array([
                0.0,
                1.0,
            ]),
            method='L-BFGS-B',
            bounds=[
                (
                    -5.0,
                    5.0,
                ),
                (
                    0.05,
                    5.0,
                ),
            ],
        )

        return {
            'method': method,
            'intercept': float(
                res.x[0]
            ),
            'slope': float(
                res.x[1]
            ),
        }

    if method == 'beta':
        lp = np.log(
            np.clip(
                pp,
                EPS,
                1.0,
            )
        )

        lq = np.log(
            np.clip(
                1.0 - pp,
                EPS,
                1.0,
            )
        )

        res = minimize(
            lambda b: nll_from_q(
                sigmoid(
                    b[0]
                    + b[1]
                    * lp
                    + b[2]
                    * lq
                )
            ),
            np.array([
                0.0,
                1.0,
                -1.0,
            ]),
            method='L-BFGS-B',
            bounds=[
                (
                    -10.0,
                    10.0,
                ),
                (
                    -10.0,
                    10.0,
                ),
                (
                    -10.0,
                    10.0,
                ),
            ],
        )

        return {
            'method': method,
            'intercept': float(
                res.x[0]
            ),
            'coef_log_p': float(
                res.x[1]
            ),
            'coef_log_1mp': float(
                res.x[2]
            ),
        }

    raise ValueError(
        f'Unknown calibration method: '
        f'{method}'
    )


def apply_calibrator(
    model: dict[
        str,
        Any,
    ],
    p: Any,
) -> np.ndarray:
    pp = clip_prob(
        p
    )

    method = model.get(
        'method',
        'raw',
    )

    if method == 'raw':
        return pp

    if method in {
        'intercept_only',
        'temperature',
        'platt',
    }:
        return clip_prob(
            sigmoid(
                float(
                    model[
                        'intercept'
                    ]
                )
                + float(
                    model[
                        'slope'
                    ]
                )
                * logit(
                    pp
                )
            )
        )

    if method == 'beta':
        lp = np.log(
            np.clip(
                pp,
                EPS,
                1.0,
            )
        )

        lq = np.log(
            np.clip(
                1.0 - pp,
                EPS,
                1.0,
            )
        )

        z = (
            float(
                model[
                    'intercept'
                ]
            )
            + float(
                model[
                    'coef_log_p'
                ]
            )
            * lp
            + float(
                model[
                    'coef_log_1mp'
                ]
            )
            * lq
        )

        return clip_prob(
            sigmoid(
                z
            )
        )

    if method == 'isotonic':
        iso = model.get(
            'model'
        )

        if iso is not None:
            return clip_prob(
                iso.predict(
                    pp
                )
            )

        xs = np.asarray(
            model.get(
                'x_thresholds',
                [],
            ),
            dtype=float,
        )

        ys = np.asarray(
            model.get(
                'y_thresholds',
                [],
            ),
            dtype=float,
        )

        if len(
            xs
        ) >= 2:
            return clip_prob(
                np.interp(
                    pp,
                    xs,
                    ys,
                    left=ys[0],
                    right=ys[-1],
                )
            )

        return pp

    return pp


def calibrator_formula(
    model: dict[
        str,
        Any,
    ],
) -> str:
    method = model.get(
        'method',
        'raw',
    )

    if method == 'raw':
        return 'NO ADJUSTMENT'

    if method == 'complement':
        return (
            'p_derived = 1 - p_canonical'
        )

    if method in {
        'intercept_only',
        'temperature',
        'platt',
    }:
        return (
            'p_adj = logistic('
            f"{model.get('intercept', 0):.8f} + "
            f"{model.get('slope', 1):.8f} * "
            'logit(p_raw))'
        )

    if method == 'beta':
        return (
            'p_adj = logistic('
            f"{model['intercept']:.8f} + "
            f"{model['coef_log_p']:.8f}"
            '*ln(p_raw) + '
            f"{model['coef_log_1mp']:.8f}"
            '*ln(1-p_raw))'
        )

    if method == 'isotonic':
        return (
            'ISOTONIC LOOKUP TABLE - '
            'see calibration knots CSV'
        )

    return method


def market_sides(
    market: str,
) -> tuple[
    str,
    str,
]:
    if market == 'moneyline':
        return (
            'HOME',
            'AWAY',
        )

    if market == 'spread':
        return (
            'HOME',
            'AWAY',
        )

    return (
        'OVER',
        'UNDER',
    )


def uses_complementary_calibration(
    market: str,
) -> bool:
    return market in {
        'spread',
        'total',
    }


def side_outcome_columns(
    market: str,
) -> tuple[
    str,
    str,
]:
    if market == 'moneyline':
        return (
            'home_win_y',
            'away_win_y',
        )

    if market == 'spread':
        return (
            'home_spread_y',
            'away_spread_y',
        )

    return (
        'over_y',
        'under_y',
    )


def side_result_arrays(
    df: pd.DataFrame,
    market: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    if market == 'moneyline':
        r1 = df[
            'home_win_result'
        ].to_numpy(
            float
        )

    elif market == 'spread':
        r1 = df[
            'home_spread_result'
        ].to_numpy(
            float
        )

    else:
        r1 = df[
            'over_result'
        ].to_numpy(
            float
        )

    r2 = np.where(
        r1 == 0.5,
        0.5,
        1.0 - r1,
    )

    return (
        r1,
        r2,
    )


def side_odds_arrays(
    df: pd.DataFrame,
    market: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    if market == 'moneyline':
        return (
            df[
                'home_dk_moneyline_decimal'
            ].to_numpy(
                float
            ),
            df[
                'away_dk_moneyline_decimal'
            ].to_numpy(
                float
            ),
        )

    if market == 'spread':
        return (
            df[
                'home_dk_spread_decimal'
            ].to_numpy(
                float
            ),
            df[
                'away_dk_spread_decimal'
            ].to_numpy(
                float
            ),
        )

    return (
        df[
            'dk_total_over_decimal'
        ].to_numpy(
            float
        ),
        df[
            'dk_total_under_decimal'
        ].to_numpy(
            float
        ),
    )


def oos_meta(
    dev: pd.DataFrame,
    folds: list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
        ]
    ],
) -> pd.DataFrame:
    rows = []

    for (
        fold_id,
        _,
        test_idx,
    ) in folds:
        t = dev.iloc[
            test_idx
        ].copy()

        t[
            'fold_id'
        ] = fold_id

        rows.append(
            t
        )

    out = pd.concat(
        rows,
        ignore_index=True,
    )

    out[
        'oos_pos'
    ] = np.arange(
        len(
            out
        ),
        dtype=int,
    )

    return out


def build_oos_prediction_cache(
    dev: pd.DataFrame,
    folds: list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
        ]
    ],
    meta: pd.DataFrame,
    market: str,
    bias_strategy: str = 'none',
    std_mode: str = 'fixed',
) -> dict[
    str,
    Any,
]:
    n = len(
        meta
    )

    side1_methods = {
        m: np.full(
            n,
            np.nan,
        )
        for m in CALIBRATION_METHODS
    }

    side2_methods = {
        m: np.full(
            n,
            np.nan,
        )
        for m in CALIBRATION_METHODS
    }

    raw_side1 = np.full(
        n,
        np.nan,
    )

    raw_side2 = np.full(
        n,
        np.nan,
    )

    means = np.full(
        n,
        np.nan,
    )

    sigmas = np.full(
        n,
        np.nan,
    )

    bins = np.full(
        n,
        -1,
        dtype=int,
    )

    biases = np.full(
        n,
        np.nan,
    )

    fold_params = []

    oos_idx = all_oos_indices(
        folds
    )

    pos_map = {
        int(
            idx
        ): pos
        for (
            pos,
            idx,
        )
        in enumerate(
            oos_idx
        )
    }

    (
        y1_col,
        y2_col,
    ) = side_outcome_columns(
        market
    )

    for (
        fold_id,
        train_idx,
        test_idx,
    ) in folds:
        train = dev.iloc[
            train_idx
        ]

        test = dev.iloc[
            test_idx
        ]

        model = fit_base_model(
            train,
            market,
            bias_strategy,
            std_mode,
        )

        train_base = (
            apply_base_model(
                model,
                train,
            )
        )

        test_base = (
            apply_base_model(
                model,
                test,
            )
        )

        positions = np.asarray([
            pos_map[
                int(i)
            ]
            for i
            in test_idx
        ], dtype=int)

        raw_side1[
            positions
        ] = test_base[
            'side1_prob'
        ]

        raw_side2[
            positions
        ] = test_base[
            'side2_prob'
        ]

        means[
            positions
        ] = test_base[
            'mean'
        ]

        sigmas[
            positions
        ] = test_base[
            'sigma'
        ]

        bins[
            positions
        ] = test_base[
            'range_bin'
        ]

        if market != 'moneyline':
            biases[
                positions
            ] = float(
                model[
                    'bias'
                ]
            )

        for method in CALIBRATION_METHODS:
            cal1 = fit_calibrator(
                train_base[
                    'side1_prob'
                ],
                train[
                    y1_col
                ].to_numpy(
                    float
                ),
                method,
            )

            p1 = apply_calibrator(
                cal1,
                test_base[
                    'side1_prob'
                ],
            )

            side1_methods[
                method
            ][
                positions
            ] = p1

            if uses_complementary_calibration(
                market
            ):
                side2_methods[
                    method
                ][
                    positions
                ] = (
                    1.0
                    - p1
                )

            else:
                cal2 = fit_calibrator(
                    train_base[
                        'side2_prob'
                    ],
                    train[
                        y2_col
                    ].to_numpy(
                        float
                    ),
                    method,
                )

                side2_methods[
                    method
                ][
                    positions
                ] = apply_calibrator(
                    cal2,
                    test_base[
                        'side2_prob'
                    ],
                )

        if market == 'moneyline':
            fold_params.append({
                'market': market,
                'fold_id': fold_id,
                'train_games': len(
                    train
                ),
                'test_games': len(
                    test
                ),
                'bias_strategy': 'NA',
                'std_mode': 'NA',
                'bias': np.nan,
                'global_sigma': np.nan,
                'std_edges': '',
                'std_sigmas': '',
                'calibration_architecture': (
                    'independent_sides'
                ),
            })

        else:
            sm: StdModel = (
                model[
                    'std_model'
                ]
            )

            fold_params.append({
                'market': market,
                'fold_id': fold_id,
                'train_games': len(
                    train
                ),
                'test_games': len(
                    test
                ),
                'bias_strategy': bias_strategy,
                'std_mode': std_mode,
                'bias': float(
                    model[
                        'bias'
                    ]
                ),
                'global_sigma': (
                    sm.global_sigma
                ),
                'std_edges': json_compact(
                    sm.edges
                ),
                'std_sigmas': json_compact(
                    sm.sigmas
                ),
                'calibration_architecture': (
                    'canonical_side_plus_complement'
                ),
            })

    return {
        'market': market,
        'bias_strategy': bias_strategy,
        'std_mode': std_mode,
        'side1': side1_methods,
        'side2': side2_methods,
        'raw_side1': raw_side1,
        'raw_side2': raw_side2,
        'mean': means,
        'sigma': sigmas,
        'range_bin': bins,
        'bias': biases,
        'fold_params': pd.DataFrame(
            fold_params
        ),
    }


def raw_kelly_fraction(
    probability: Any,
    decimal_odds: Any,
) -> np.ndarray:
    p = np.asarray(
        probability,
        dtype=float,
    )

    o = np.asarray(
        decimal_odds,
        dtype=float,
    )

    b = (
        o
        - 1.0
    )

    q = (
        1.0
        - p
    )

    out = np.full(
        len(
            p
        ),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(
            p
        )
        & np.isfinite(
            o
        )
        & (
            o > 1.0
        )
        & (
            b > 0.0
        )
    )

    out[
        valid
    ] = np.maximum(
        (
            (
                b[
                    valid
                ]
                * p[
                    valid
                ]
            )
            - q[
                valid
            ]
        )
        / b[
            valid
        ],
        0.0,
    )

    return out


def market_opportunities(
    meta: pd.DataFrame,
    market: str,
    p1: Any,
    p2: Any,
) -> pd.DataFrame:
    p1 = np.asarray(
        p1,
        dtype=float,
    )

    p2 = np.asarray(
        p2,
        dtype=float,
    )

    (
        o1,
        o2,
    ) = side_odds_arrays(
        meta,
        market,
    )

    (
        r1,
        r2,
    ) = side_result_arrays(
        meta,
        market,
    )

    (
        side1,
        side2,
    ) = market_sides(
        market
    )

    ev1 = (
        p1
        * o1
        - 1.0
    )

    ev2 = (
        p2
        * o2
        - 1.0
    )

    profit1 = (
        unit_profit_from_result(
            r1,
            o1,
        )
    )

    profit2 = (
        unit_profit_from_result(
            r2,
            o2,
        )
    )

    valid1 = (
        np.isfinite(
            ev1
        )
        & np.isfinite(
            profit1
        )
        & np.isfinite(
            o1
        )
        & (
            o1 > 1.0
        )
    )

    valid2 = (
        np.isfinite(
            ev2
        )
        & np.isfinite(
            profit2
        )
        & np.isfinite(
            o2
        )
        & (
            o2 > 1.0
        )
    )

    (
        market_p1,
        market_p2,
    ) = devig_pair(
        o1,
        o2,
    )

    edge_market1 = (
        p1
        - market_p1
    ) * 100.0

    edge_market2 = (
        p2
        - market_p2
    ) * 100.0

    kelly1 = (
        raw_kelly_fraction(
            p1,
            o1,
        )
    )

    kelly2 = (
        raw_kelly_fraction(
            p2,
            o2,
        )
    )

    return pd.DataFrame({
        'row_id': meta[
            '_row_id'
        ].to_numpy(
            int
        ),
        'game_id': meta[
            'game_id'
        ].astype(
            str
        ).to_numpy(),
        'date': meta[
            '_date'
        ].to_numpy(),
        'week': meta[
            '_week'
        ].astype(
            str
        ).to_numpy(),
        'fold_id': (
            meta[
                'fold_id'
            ].to_numpy(
                int
            )
            if 'fold_id'
            in meta.columns
            else np.zeros(
                len(
                    meta
                ),
                dtype=int,
            )
        ),
        'market': market,
        'side1_name': side1,
        'side2_name': side2,
        'side1_valid': valid1,
        'side2_valid': valid2,
        'side1_ev': ev1,
        'side2_ev': ev2,
        'side1_profit': profit1,
        'side2_profit': profit2,
        'side1_prob': p1,
        'side2_prob': p2,
        'side1_odds': o1,
        'side2_odds': o2,
        'side1_result': r1,
        'side2_result': r2,
        'side1_market_prob': market_p1,
        'side2_market_prob': market_p2,
        'side1_edge_vs_market': edge_market1,
        'side2_edge_vs_market': edge_market2,
        'side1_kelly': kelly1,
        'side2_kelly': kelly2,
    })


def _preference_arrays(
    opps: pd.DataFrame,
    policy: MarketSelectionPolicy,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    metric = (
        policy.preference_metric
    )

    if metric == 'ev':
        return (
            opps[
                'side1_ev'
            ].to_numpy(
                float
            ),
            opps[
                'side2_ev'
            ].to_numpy(
                float
            ),
        )

    if metric == 'kelly':
        return (
            opps[
                'side1_kelly'
            ].to_numpy(
                float
            ),
            opps[
                'side2_kelly'
            ].to_numpy(
                float
            ),
        )

    if metric == 'model_prob':
        return (
            opps[
                'side1_prob'
            ].to_numpy(
                float
            ),
            opps[
                'side2_prob'
            ].to_numpy(
                float
            ),
        )

    if metric == 'edge_vs_market':
        return (
            opps[
                'side1_edge_vs_market'
            ].to_numpy(
                float
            ),
            opps[
                'side2_edge_vs_market'
            ].to_numpy(
                float
            ),
        )

    raise ValueError(
        'Unsupported pick_preference '
        f'metric: {metric}'
    )


def select_opportunities(
    opps: pd.DataFrame,
    policy: MarketSelectionPolicy,
    edge_mode: str,
    shared_edge: float | None = None,
    edge_side1: float | None = None,
    edge_side2: float | None = None,
) -> pd.DataFrame:
    out = opps.copy()

    ev1 = out[
        'side1_ev'
    ].to_numpy(
        float
    )

    ev2 = out[
        'side2_ev'
    ].to_numpy(
        float
    )

    valid1 = out[
        'side1_valid'
    ].to_numpy(
        bool
    )

    valid2 = out[
        'side2_valid'
    ].to_numpy(
        bool
    )

    if edge_mode == 'shared':
        if shared_edge is None:
            raise ValueError(
                'shared_edge is required '
                "when edge_mode='shared'"
            )

        pass1 = (
            valid1
            & (
                ev1
                >= float(
                    shared_edge
                )
            )
        )

        pass2 = (
            valid2
            & (
                ev2
                >= float(
                    shared_edge
                )
            )
        )

    elif edge_mode == 'split':
        if (
            edge_side1 is None
            or edge_side2 is None
        ):
            raise ValueError(
                'edge_side1 and edge_side2 '
                'are required when '
                "edge_mode='split'"
            )

        pass1 = (
            valid1
            & (
                ev1
                >= float(
                    edge_side1
                )
            )
        )

        pass2 = (
            valid2
            & (
                ev2
                >= float(
                    edge_side2
                )
            )
        )

    else:
        raise ValueError(
            f'Unknown edge_mode: '
            f'{edge_mode}'
        )

    if (
        policy.selection_mode
        == 'all_qualifying'
    ):
        choose1 = pass1.copy()
        choose2 = pass2.copy()

    elif (
        policy.selection_mode
        == 'pick_one'
    ):
        (
            pref1,
            pref2,
        ) = _preference_arrays(
            out,
            policy,
        )

        if (
            policy.preference_direction
            == 'max'
        ):
            cmp1 = np.where(
                np.isfinite(
                    pref1
                ),
                pref1,
                -np.inf,
            )

            cmp2 = np.where(
                np.isfinite(
                    pref2
                ),
                pref2,
                -np.inf,
            )

            choose1 = (
                pass1
                & (
                    ~pass2
                    | (
                        cmp1
                        >= cmp2
                    )
                )
            )

            choose2 = (
                pass2
                & (
                    ~pass1
                    | (
                        cmp2
                        > cmp1
                    )
                )
            )

        elif (
            policy.preference_direction
            == 'min'
        ):
            cmp1 = np.where(
                np.isfinite(
                    pref1
                ),
                pref1,
                np.inf,
            )

            cmp2 = np.where(
                np.isfinite(
                    pref2
                ),
                pref2,
                np.inf,
            )

            choose1 = (
                pass1
                & (
                    ~pass2
                    | (
                        cmp1
                        <= cmp2
                    )
                )
            )

            choose2 = (
                pass2
                & (
                    ~pass1
                    | (
                        cmp2
                        < cmp1
                    )
                )
            )

        else:
            raise ValueError(
                'Unsupported '
                'pick_preference direction: '
                f'{policy.preference_direction}'
            )

    else:
        raise ValueError(
            'Unsupported selection_mode: '
            f'{policy.selection_mode}'
        )

    selected_bets = (
        choose1.astype(
            int
        )
        + choose2.astype(
            int
        )
    )

    pr1 = out[
        'side1_profit'
    ].to_numpy(
        float
    )

    pr2 = out[
        'side2_profit'
    ].to_numpy(
        float
    )

    total_profit = (
        np.where(
            choose1,
            np.nan_to_num(
                pr1,
                nan=0.0,
            ),
            0.0,
        )
        + np.where(
            choose2,
            np.nan_to_num(
                pr2,
                nan=0.0,
            ),
            0.0,
        )
    )

    p1 = out[
        'side1_prob'
    ].to_numpy(
        float
    )

    p2 = out[
        'side2_prob'
    ].to_numpy(
        float
    )

    o1 = out[
        'side1_odds'
    ].to_numpy(
        float
    )

    o2 = out[
        'side2_odds'
    ].to_numpy(
        float
    )

    r1 = out[
        'side1_result'
    ].to_numpy(
        float
    )

    r2 = out[
        'side2_result'
    ].to_numpy(
        float
    )

    selected_side = np.full(
        len(
            out
        ),
        '',
        dtype=object,
    )

    only1 = (
        choose1
        & ~choose2
    )

    only2 = (
        choose2
        & ~choose1
    )

    both = (
        choose1
        & choose2
    )

    selected_side[
        only1
    ] = (
        str(
            out[
                'side1_name'
            ].iloc[0]
        )
        if len(
            out
        )
        else ''
    )

    selected_side[
        only2
    ] = (
        str(
            out[
                'side2_name'
            ].iloc[0]
        )
        if len(
            out
        )
        else ''
    )

    if both.any():
        s1 = str(
            out[
                'side1_name'
            ].iloc[0]
        )

        s2 = str(
            out[
                'side2_name'
            ].iloc[0]
        )

        selected_side[
            both
        ] = (
            f'{s1}|{s2}'
        )

    one = (
        selected_bets
        == 1
    )

    selected_ev = np.where(
        only1,
        ev1,
        np.where(
            only2,
            ev2,
            np.nan,
        ),
    )

    selected_prob = np.where(
        only1,
        p1,
        np.where(
            only2,
            p2,
            np.nan,
        ),
    )

    selected_odds = np.where(
        only1,
        o1,
        np.where(
            only2,
            o2,
            np.nan,
        ),
    )

    selected_result = np.where(
        only1,
        r1,
        np.where(
            only2,
            r2,
            np.nan,
        ),
    )

    out[
        'selection_mode'
    ] = (
        policy.selection_mode
    )

    out[
        'pick_preference_metric'
    ] = (
        policy.preference_metric
    )

    out[
        'pick_preference_direction'
    ] = (
        policy.preference_direction
    )

    out[
        'edge_mode'
    ] = edge_mode

    out[
        'side1_pass_edge'
    ] = pass1

    out[
        'side2_pass_edge'
    ] = pass2

    out[
        'side1_selected'
    ] = choose1

    out[
        'side2_selected'
    ] = choose2

    out[
        'selected_bets'
    ] = selected_bets

    out[
        'selected'
    ] = (
        selected_bets
        > 0
    )

    out[
        'selected_side'
    ] = selected_side

    out[
        'selected_ev'
    ] = selected_ev

    out[
        'probability'
    ] = selected_prob

    out[
        'decimal_odds'
    ] = selected_odds

    out[
        'result'
    ] = selected_result

    out[
        'unit_profit'
    ] = total_profit

    out[
        'selected_ev_sum'
    ] = (
        np.where(
            choose1,
            np.nan_to_num(
                ev1,
                nan=0.0,
            ),
            0.0,
        )
        + np.where(
            choose2,
            np.nan_to_num(
                ev2,
                nan=0.0,
            ),
            0.0,
        )
    )

    out[
        'single_selection_detail_available'
    ] = one

    return out


def edge_scan(
    opps: pd.DataFrame,
    grid: np.ndarray,
    min_bets: int,
    policy: MarketSelectionPolicy,
) -> pd.DataFrame:
    rows = []

    for edge in grid:
        selected = select_opportunities(
            opps,
            policy,
            edge_mode='shared',
            shared_edge=float(
                edge
            ),
        )

        bets_by_game = selected[
            'selected_bets'
        ].to_numpy(
            int
        )

        profit_by_game = selected[
            'unit_profit'
        ].to_numpy(
            float
        )

        fold = selected[
            'fold_id'
        ].to_numpy(
            int
        )

        n = int(
            np.sum(
                bets_by_game
            )
        )

        if n:
            p = float(
                np.nansum(
                    profit_by_game
                )
            )

            roi = (
                p / n
            )

            fold_frame = pd.DataFrame({
                'fold_id': fold,
                'profit': profit_by_game,
                'bets': bets_by_game,
            })

            fold_frame = (
                fold_frame[
                    fold_frame[
                        'bets'
                    ] > 0
                ]
            )

            fold_profits = (
                fold_frame
                .groupby(
                    'fold_id'
                )[
                    'profit'
                ]
                .sum()
            )

            positive_fold_rate = (
                float(
                    (
                        fold_profits
                        > 0
                    ).mean()
                )
                if len(
                    fold_profits
                )
                else np.nan
            )

        else:
            (
                p,
                roi,
                positive_fold_rate,
            ) = (
                0.0,
                np.nan,
                np.nan,
            )

        rows.append({
            'edge': float(
                edge
            ),
            'selection_mode': (
                policy.selection_mode
            ),
            'pick_preference_metric': (
                policy.preference_metric
            ),
            'pick_preference_direction': (
                policy.preference_direction
            ),
            'bets': n,
            'profit_units': p,
            'roi': roi,
            'positive_fold_rate': (
                positive_fold_rate
            ),
            'eligible_min_bets': (
                n >= min_bets
            ),
        })

    return pd.DataFrame(
        rows
    )


def choose_edge(
    scan: pd.DataFrame,
    min_bets: int,
) -> pd.Series:
    eligible = (
        scan[
            scan[
                'bets'
            ] >= min_bets
        ]
        .copy()
    )

    if eligible.empty:
        eligible = (
            scan[
                scan[
                    'bets'
                ] > 0
            ]
            .copy()
        )

    if eligible.empty:
        return scan.iloc[
            0
        ]

    eligible = (
        eligible
        .sort_values(
            [
                'profit_units',
                'positive_fold_rate',
                'bets',
                'edge',
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
    )

    return eligible.iloc[
        0
    ]


def probability_pair_score(
    meta: pd.DataFrame,
    market: str,
    p1: Any,
    p2: Any,
) -> tuple[
    float,
    float,
]:
    (
        y1_col,
        y2_col,
    ) = side_outcome_columns(
        market
    )

    y1 = meta[
        y1_col
    ].to_numpy(
        float
    )

    y2 = meta[
        y2_col
    ].to_numpy(
        float
    )

    ll1 = binary_log_loss(
        p1,
        y1,
    )

    ll2 = binary_log_loss(
        p2,
        y2,
    )

    br1 = brier_score(
        p1,
        y1,
    )

    br2 = brier_score(
        p2,
        y2,
    )

    return (
        float(
            np.nanmean([
                ll1,
                ll2,
            ])
        ),
        float(
            np.nanmean([
                br1,
                br2,
            ])
        ),
    )


def calibration_acceptance_for_cache(
    meta: pd.DataFrame,
    market: str,
    cache: dict[
        str,
        Any,
    ],
) -> pd.DataFrame:
    s1, s2 = market_sides(
        market
    )

    (
        y1_col,
        y2_col,
    ) = side_outcome_columns(
        market
    )

    if uses_complementary_calibration(
        market
    ):
        y1 = meta[
            y1_col
        ].to_numpy(
            float
        )

        y2 = meta[
            y2_col
        ].to_numpy(
            float
        )

        raw_p1 = cache[
            'side1'
        ][
            'raw'
        ]

        raw_p2 = cache[
            'side2'
        ][
            'raw'
        ]

        (
            raw_ll,
            raw_br,
        ) = probability_pair_score(
            meta,
            market,
            raw_p1,
            raw_p2,
        )

        rows = []

        for method in CALIBRATION_METHODS:
            p1 = cache[
                'side1'
            ][
                method
            ]

            p2 = cache[
                'side2'
            ][
                method
            ]

            (
                ll,
                br,
            ) = probability_pair_score(
                meta,
                market,
                p1,
                p2,
            )

            folds = 0
            wins_ll = 0
            wins_br = 0

            for (
                _,
                idxs,
            ) in meta.groupby(
                'fold_id'
            ).groups.items():
                idx = np.asarray(
                    list(
                        idxs
                    ),
                    dtype=int,
                )

                mll = float(
                    np.nanmean([
                        binary_log_loss(
                            p1[
                                idx
                            ],
                            y1[
                                idx
                            ],
                        ),
                        binary_log_loss(
                            p2[
                                idx
                            ],
                            y2[
                                idx
                            ],
                        ),
                    ])
                )

                rll = float(
                    np.nanmean([
                        binary_log_loss(
                            raw_p1[
                                idx
                            ],
                            y1[
                                idx
                            ],
                        ),
                        binary_log_loss(
                            raw_p2[
                                idx
                            ],
                            y2[
                                idx
                            ],
                        ),
                    ])
                )

                mbr = float(
                    np.nanmean([
                        brier_score(
                            p1[
                                idx
                            ],
                            y1[
                                idx
                            ],
                        ),
                        brier_score(
                            p2[
                                idx
                            ],
                            y2[
                                idx
                            ],
                        ),
                    ])
                )

                rbr = float(
                    np.nanmean([
                        brier_score(
                            raw_p1[
                                idx
                            ],
                            y1[
                                idx
                            ],
                        ),
                        brier_score(
                            raw_p2[
                                idx
                            ],
                            y2[
                                idx
                            ],
                        ),
                    ])
                )

                if (
                    np.isfinite(
                        mll
                    )
                    and np.isfinite(
                        rll
                    )
                    and np.isfinite(
                        mbr
                    )
                    and np.isfinite(
                        rbr
                    )
                ):
                    folds += 1
                    wins_ll += int(
                        mll < rll
                    )
                    wins_br += int(
                        mbr < rbr
                    )

            win_ll_rate = (
                wins_ll / folds
                if folds
                else np.nan
            )

            win_br_rate = (
                wins_br / folds
                if folds
                else np.nan
            )

            allowed = (
                method == 'raw'
                or (
                    np.isfinite(
                        ll
                    )
                    and np.isfinite(
                        br
                    )
                    and ll
                    < raw_ll
                    and br
                    < raw_br
                    and win_ll_rate
                    >= MIN_CALIBRATION_FOLD_WIN_RATE
                    and win_br_rate
                    >= MIN_CALIBRATION_FOLD_WIN_RATE
                )
            )

            rows.append({
                'market': market,
                'side': s1,
                'calibration_role': 'canonical',
                'method': method,
                'oos_log_loss': ll,
                'raw_oos_log_loss': raw_ll,
                'oos_brier': br,
                'raw_oos_brier': raw_br,
                'fold_win_rate_log_loss': (
                    win_ll_rate
                ),
                'fold_win_rate_brier': (
                    win_br_rate
                ),
                'allowed_in_joint_optimization': bool(
                    allowed
                ),
            })

            rows.append({
                'market': market,
                'side': s2,
                'calibration_role': 'derived_complement',
                'method': 'complement',
                'canonical_method': method,
                'oos_log_loss': ll,
                'raw_oos_log_loss': raw_ll,
                'oos_brier': br,
                'raw_oos_brier': raw_br,
                'fold_win_rate_log_loss': (
                    win_ll_rate
                ),
                'fold_win_rate_brier': (
                    win_br_rate
                ),
                'allowed_in_joint_optimization': bool(
                    allowed
                ),
            })

        return pd.DataFrame(
            rows
        )

    rows = []

    for (
        side,
        ycol,
        pmap,
    ) in [
        (
            s1,
            y1_col,
            cache[
                'side1'
            ],
        ),
        (
            s2,
            y2_col,
            cache[
                'side2'
            ],
        ),
    ]:
        y = meta[
            ycol
        ].to_numpy(
            float
        )

        raw_p = pmap[
            'raw'
        ]

        raw_ll = binary_log_loss(
            raw_p,
            y,
        )

        raw_br = brier_score(
            raw_p,
            y,
        )

        for method in CALIBRATION_METHODS:
            p = pmap[
                method
            ]

            ll = binary_log_loss(
                p,
                y,
            )

            br = brier_score(
                p,
                y,
            )

            folds = 0
            wins_ll = 0
            wins_br = 0

            for (
                _,
                idxs,
            ) in meta.groupby(
                'fold_id'
            ).groups.items():
                idx = np.asarray(
                    list(
                        idxs
                    ),
                    dtype=int,
                )

                mll = binary_log_loss(
                    p[
                        idx
                    ],
                    y[
                        idx
                    ],
                )

                rll = binary_log_loss(
                    raw_p[
                        idx
                    ],
                    y[
                        idx
                    ],
                )

                mbr = brier_score(
                    p[
                        idx
                    ],
                    y[
                        idx
                    ],
                )

                rbr = brier_score(
                    raw_p[
                        idx
                    ],
                    y[
                        idx
                    ],
                )

                if (
                    np.isfinite(
                        mll
                    )
                    and np.isfinite(
                        rll
                    )
                    and np.isfinite(
                        mbr
                    )
                    and np.isfinite(
                        rbr
                    )
                ):
                    folds += 1
                    wins_ll += int(
                        mll < rll
                    )
                    wins_br += int(
                        mbr < rbr
                    )

            win_ll_rate = (
                wins_ll / folds
                if folds
                else np.nan
            )

            win_br_rate = (
                wins_br / folds
                if folds
                else np.nan
            )

            allowed = (
                method == 'raw'
                or (
                    np.isfinite(
                        ll
                    )
                    and np.isfinite(
                        br
                    )
                    and ll
                    < raw_ll
                    and br
                    < raw_br
                    and win_ll_rate
                    >= MIN_CALIBRATION_FOLD_WIN_RATE
                    and win_br_rate
                    >= MIN_CALIBRATION_FOLD_WIN_RATE
                )
            )

            rows.append({
                'market': market,
                'side': side,
                'calibration_role': 'independent',
                'method': method,
                'oos_log_loss': ll,
                'raw_oos_log_loss': raw_ll,
                'oos_brier': br,
                'raw_oos_brier': raw_br,
                'fold_win_rate_log_loss': (
                    win_ll_rate
                ),
                'fold_win_rate_brier': (
                    win_br_rate
                ),
                'allowed_in_joint_optimization': bool(
                    allowed
                ),
            })

    return pd.DataFrame(
        rows
    )


def oos_std_acceptance_for_caches(
    meta: pd.DataFrame,
    market: str,
    caches: dict[
        str,
        dict[
            str,
            Any,
        ],
    ],
) -> pd.DataFrame:
    if market == 'moneyline':
        return pd.DataFrame([
            {
                'market': market,
                'bias_strategy': 'NA',
                'std_mode': 'NA',
                'oos_residual_nll': np.nan,
                'fixed_oos_residual_nll': np.nan,
                'relative_improvement_vs_fixed': np.nan,
                'allowed_in_joint_optimization': True,
            }
        ])

    actual = (
        meta[
            'actual_margin'
        ].to_numpy(
            float
        )
        if market == 'spread'
        else meta[
            'actual_total_calc'
        ].to_numpy(
            float
        )
    )

    rows = []
    grouped = {}

    for cache in caches.values():
        bias = cache[
            'bias_strategy'
        ]

        mode = cache[
            'std_mode'
        ]

        err = (
            actual
            - cache[
                'mean'
            ]
        )

        nll = float(
            np.nanmean(
                normal_residual_nll(
                    err,
                    cache[
                        'sigma'
                    ],
                )
            )
        )

        grouped[
            (
                bias,
                mode,
            )
        ] = nll

    for (
        bias,
        mode,
    ), nll in grouped.items():
        fixed_nll = grouped.get(
            (
                bias,
                'fixed',
            ),
            np.nan,
        )

        rel = (
            (
                fixed_nll
                - nll
            )
            / abs(
                fixed_nll
            )
            if (
                np.isfinite(
                    fixed_nll
                )
                and fixed_nll != 0
            )
            else np.nan
        )

        allowed = (
            mode == 'fixed'
            or (
                np.isfinite(
                    rel
                )
                and rel
                >= MIN_ADAPTIVE_STD_REL_NLL_IMPROVEMENT
            )
        )

        rows.append({
            'market': market,
            'bias_strategy': bias,
            'std_mode': mode,
            'oos_residual_nll': nll,
            'fixed_oos_residual_nll': fixed_nll,
            'relative_improvement_vs_fixed': rel,
            'allowed_in_joint_optimization': bool(
                allowed
            ),
        })

    return pd.DataFrame(
        rows
    )


def min_oos_bets(
    n_oos: int,
) -> int:
    return int(
        max(
            30,
            min(
                150,
                round(
                    n_oos
                    * 0.08
                ),
            ),
        )
    )


def evaluate_joint_configs_for_market(
    dev: pd.DataFrame,
    folds: list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
        ]
    ],
    meta: pd.DataFrame,
    market: str,
    bias_strategies: list[str],
    std_modes: list[str],
    selection_policy: MarketSelectionPolicy,
    output_dir: Path,
    prefix: str,
) -> tuple[
    pd.DataFrame,
    dict[
        str,
        dict[
            str,
            Any,
        ],
    ],
    list[
        Path
    ],
]:
    files: list[
        Path
    ] = []

    caches: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    if market == 'moneyline':
        base_configs = [
            (
                'NA',
                'NA',
            )
        ]

    else:
        base_configs = [
            (
                b,
                s,
            )
            for b in bias_strategies
            for s in std_modes
        ]

    progress(
        f'  {market}: building OOS '
        'probability cache for '
        f'{len(base_configs)} '
        'base configurations...'
    )

    fold_param_frames = []

    for (
        bias_strategy,
        std_mode,
    ) in base_configs:
        key = (
            f'{bias_strategy}|'
            f'{std_mode}'
        )

        cache = (
            build_oos_prediction_cache(
                dev,
                folds,
                meta,
                market,
                bias_strategy=(
                    'none'
                    if market
                    == 'moneyline'
                    else bias_strategy
                ),
                std_mode=(
                    'fixed'
                    if market
                    == 'moneyline'
                    else std_mode
                ),
            )
        )

        caches[
            key
        ] = cache

        fold_param_frames.append(
            cache[
                'fold_params'
            ]
        )

    fold_params_all = (
        pd.concat(
            fold_param_frames,
            ignore_index=True,
        )
        .drop_duplicates()
    )

    files.append(
        save_csv(
            fold_params_all,
            output_dir
            / (
                f'{prefix}_04_'
                f'{market}_'
                'base_fold_parameters.csv'
            ),
        )
    )

    std_accept = (
        oos_std_acceptance_for_caches(
            meta,
            market,
            caches,
        )
    )

    files.append(
        save_csv(
            std_accept,
            output_dir
            / (
                f'{prefix}_04_'
                f'{market}_'
                'std_joint_acceptance.csv'
            ),
        )
    )

    std_allowed = {
        (
            str(
                r[
                    'bias_strategy'
                ]
            ),
            str(
                r[
                    'std_mode'
                ]
            ),
        ): bool(
            r[
                'allowed_in_joint_optimization'
            ]
        )
        for (
            _,
            r,
        )
        in std_accept.iterrows()
    }

    cal_accept_frames = []

    allowed_cal: dict[
        tuple[
            str,
            str,
        ],
        dict[
            str,
            list[str],
        ],
    ] = {}

    for (
        bias_strategy,
        std_mode,
    ) in base_configs:
        key = (
            f'{bias_strategy}|'
            f'{std_mode}'
        )

        cache = caches[
            key
        ]

        ca = (
            calibration_acceptance_for_cache(
                meta,
                market,
                cache,
            )
        )

        ca[
            'bias_strategy'
        ] = bias_strategy

        ca[
            'std_mode'
        ] = std_mode

        ca[
            'cache_key'
        ] = key

        cal_accept_frames.append(
            ca
        )

        if uses_complementary_calibration(
            market
        ):
            canonical_side = (
                market_sides(
                    market
                )[0]
            )

            methods = ca[
                (
                    ca[
                        'side'
                    ]
                    == canonical_side
                )
                & (
                    ca[
                        'calibration_role'
                    ]
                    == 'canonical'
                )
                & (
                    ca[
                        'allowed_in_joint_optimization'
                    ]
                )
            ][
                'method'
            ].tolist()

            if 'raw' not in methods:
                methods = [
                    'raw',
                    *methods,
                ]

            allowed_cal[
                (
                    bias_strategy,
                    std_mode,
                )
            ] = {
                canonical_side: methods
            }

        else:
            side_map = {}

            for side in market_sides(
                market
            ):
                methods = ca[
                    (
                        ca[
                            'side'
                        ]
                        == side
                    )
                    & (
                        ca[
                            'allowed_in_joint_optimization'
                        ]
                    )
                ][
                    'method'
                ].tolist()

                if 'raw' not in methods:
                    methods = [
                        'raw',
                        *methods,
                    ]

                side_map[
                    side
                ] = methods

            allowed_cal[
                (
                    bias_strategy,
                    std_mode,
                )
            ] = side_map

    cal_accept_all = pd.concat(
        cal_accept_frames,
        ignore_index=True,
    )

    files.append(
        save_csv(
            cal_accept_all,
            output_dir
            / (
                f'{prefix}_04_'
                f'{market}_'
                'calibration_joint_acceptance.csv'
            ),
        )
    )

    minimum_bets = min_oos_bets(
        len(
            meta
        )
    )

    rows = []

    (
        side1_name,
        side2_name,
    ) = market_sides(
        market
    )

    progress(
        f'  {market}: evaluating '
        'OOS-approved calibration '
        'configurations + EDGE...'
    )

    for (
        bias_strategy,
        std_mode,
    ) in base_configs:
        if not std_allowed.get(
            (
                bias_strategy,
                std_mode,
            ),
            True,
        ):
            continue

        key = (
            f'{bias_strategy}|'
            f'{std_mode}'
        )

        cache = caches[
            key
        ]

        side_map = allowed_cal[
            (
                bias_strategy,
                std_mode,
            )
        ]

        if uses_complementary_calibration(
            market
        ):
            for cal1 in side_map[
                side1_name
            ]:
                p1 = cache[
                    'side1'
                ][
                    cal1
                ]

                p2 = cache[
                    'side2'
                ][
                    cal1
                ]

                (
                    ll,
                    br,
                ) = probability_pair_score(
                    meta,
                    market,
                    p1,
                    p2,
                )

                opps = market_opportunities(
                    meta,
                    market,
                    p1,
                    p2,
                )

                scan = edge_scan(
                    opps,
                    EDGE_GRID,
                    minimum_bets,
                    selection_policy,
                )

                best = choose_edge(
                    scan,
                    minimum_bets,
                )

                rows.append({
                    'market': market,
                    'calibration_architecture': (
                        'canonical_side_plus_complement'
                    ),
                    'selection_mode': (
                        selection_policy.selection_mode
                    ),
                    'pick_preference_metric': (
                        selection_policy.preference_metric
                    ),
                    'pick_preference_direction': (
                        selection_policy.preference_direction
                    ),
                    'bias_strategy': bias_strategy,
                    'std_mode': std_mode,
                    f'calibration_{side1_name.lower()}': cal1,
                    f'calibration_{side2_name.lower()}': 'complement',
                    'oos_probability_log_loss': ll,
                    'oos_probability_brier': br,
                    'selected_edge': float(
                        best[
                            'edge'
                        ]
                    ),
                    'oos_bets': int(
                        best[
                            'bets'
                        ]
                    ),
                    'oos_profit_units': float(
                        best[
                            'profit_units'
                        ]
                    ),
                    'oos_roi': (
                        float(
                            best[
                                'roi'
                            ]
                        )
                        if np.isfinite(
                            best[
                                'roi'
                            ]
                        )
                        else np.nan
                    ),
                    'positive_fold_rate': (
                        float(
                            best[
                                'positive_fold_rate'
                            ]
                        )
                        if np.isfinite(
                            best[
                                'positive_fold_rate'
                            ]
                        )
                        else np.nan
                    ),
                    'minimum_bets_required': minimum_bets,
                    'cache_key': key,
                })

        else:
            for cal1 in side_map[
                side1_name
            ]:
                p1 = cache[
                    'side1'
                ][
                    cal1
                ]

                for cal2 in side_map[
                    side2_name
                ]:
                    p2 = cache[
                        'side2'
                    ][
                        cal2
                    ]

                    (
                        ll,
                        br,
                    ) = probability_pair_score(
                        meta,
                        market,
                        p1,
                        p2,
                    )

                    opps = market_opportunities(
                        meta,
                        market,
                        p1,
                        p2,
                    )

                    scan = edge_scan(
                        opps,
                        EDGE_GRID,
                        minimum_bets,
                        selection_policy,
                    )

                    best = choose_edge(
                        scan,
                        minimum_bets,
                    )

                    rows.append({
                        'market': market,
                        'calibration_architecture': (
                            'independent_sides'
                        ),
                        'selection_mode': (
                            selection_policy.selection_mode
                        ),
                        'pick_preference_metric': (
                            selection_policy.preference_metric
                        ),
                        'pick_preference_direction': (
                            selection_policy.preference_direction
                        ),
                        'bias_strategy': bias_strategy,
                        'std_mode': std_mode,
                        f'calibration_{side1_name.lower()}': cal1,
                        f'calibration_{side2_name.lower()}': cal2,
                        'oos_probability_log_loss': ll,
                        'oos_probability_brier': br,
                        'selected_edge': float(
                            best[
                                'edge'
                            ]
                        ),
                        'oos_bets': int(
                            best[
                                'bets'
                            ]
                        ),
                        'oos_profit_units': float(
                            best[
                                'profit_units'
                            ]
                        ),
                        'oos_roi': (
                            float(
                                best[
                                    'roi'
                                ]
                            )
                            if np.isfinite(
                                best[
                                    'roi'
                                ]
                            )
                            else np.nan
                        ),
                        'positive_fold_rate': (
                            float(
                                best[
                                    'positive_fold_rate'
                                ]
                            )
                            if np.isfinite(
                                best[
                                    'positive_fold_rate'
                                ]
                            )
                            else np.nan
                        ),
                        'minimum_bets_required': minimum_bets,
                        'cache_key': key,
                    })

    ranking = pd.DataFrame(
        rows
    )

    if ranking.empty:
        raise RuntimeError(
            'No joint configurations '
            'survived OOS gates for '
            f'market={market}'
        )

    if uses_complementary_calibration(
        market
    ):
        raw_mask = (
            ranking[
                f'calibration_{side1_name.lower()}'
            ]
            == 'raw'
        )

    else:
        raw_mask = (
            (
                ranking[
                    f'calibration_{side1_name.lower()}'
                ]
                == 'raw'
            )
            & (
                ranking[
                    f'calibration_{side2_name.lower()}'
                ]
                == 'raw'
            )
        )

    raw_best_ll = (
        float(
            ranking.loc[
                raw_mask,
                'oos_probability_log_loss',
            ].min()
        )
        if raw_mask.any()
        else np.nan
    )

    ranking[
        'probability_sanity_pass'
    ] = (
        ranking[
            'oos_probability_log_loss'
        ]
        <= raw_best_ll
        + 0.01
        if np.isfinite(
            raw_best_ll
        )
        else True
    )

    ranking[
        'stability_pass'
    ] = (
        ranking[
            'positive_fold_rate'
        ]
        .fillna(
            0
        )
        >= 0.50
    )

    ranking[
        'bets_pass'
    ] = (
        ranking[
            'oos_bets'
        ]
        >= minimum_bets
    )

    ranking[
        'joint_candidate_pass'
    ] = (
        ranking[
            'probability_sanity_pass'
        ]
        & ranking[
            'bets_pass'
        ]
        & ranking[
            'stability_pass'
        ]
    )

    ranking = (
        ranking
        .sort_values(
            [
                'joint_candidate_pass',
                'oos_profit_units',
                'positive_fold_rate',
                'oos_probability_log_loss',
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    ranking[
        'rank'
    ] = np.arange(
        1,
        len(
            ranking
        )
        + 1,
    )

    files.append(
        save_csv(
            ranking,
            output_dir
            / (
                f'{prefix}_04_'
                'joint_rankings_'
                f'{market}.csv'
            ),
        )
    )

    return (
        ranking,
        caches,
        files,
    )


def season_block_id(
    dates: pd.Series,
) -> np.ndarray:
    d = pd.to_datetime(
        dates
    )

    origin = d.min()

    return (
        (
            d
            - origin
        )
        .dt.days
        // SEASON_BLOCK_DAYS
    ).to_numpy(
        int
    )


def config_opportunities(
    meta: pd.DataFrame,
    market: str,
    config_row: pd.Series,
    caches: dict[
        str,
        dict[
            str,
            Any,
        ],
    ],
) -> pd.DataFrame:
    cache = caches[
        str(
            config_row[
                'cache_key'
            ]
        )
    ]

    (
        s1,
        s2,
    ) = market_sides(
        market
    )

    cal1 = str(
        config_row[
            f'calibration_{s1.lower()}'
        ]
    )

    if uses_complementary_calibration(
        market
    ):
        p1 = cache[
            'side1'
        ][
            cal1
        ]

        p2 = cache[
            'side2'
        ][
            cal1
        ]

    else:
        cal2 = str(
            config_row[
                f'calibration_{s2.lower()}'
            ]
        )

        p1 = cache[
            'side1'
        ][
            cal1
        ]

        p2 = cache[
            'side2'
        ][
            cal2
        ]

    return market_opportunities(
        meta,
        market,
        p1,
        p2,
    )


def stress_top_joint_configs(
    meta: pd.DataFrame,
    market: str,
    ranking: pd.DataFrame,
    caches: dict[
        str,
        dict[
            str,
            Any,
        ],
    ],
    selection_policy: MarketSelectionPolicy,
    reps: int,
    rng: np.random.Generator,
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    eligible_ranking = (
        ranking[
            ranking[
                'joint_candidate_pass'
            ]
        ]
        .copy()
    )

    if eligible_ranking.empty:
        eligible_ranking = (
            ranking.copy()
        )

    top = (
        eligible_ranking
        .head(
            min(
                TOP_CONFIGS_TO_STRESS,
                len(
                    eligible_ranking
                ),
            )
        )
        .copy()
        .reset_index(
            drop=True
        )
    )

    blocks = season_block_id(
        meta[
            '_date'
        ]
    )

    unique_blocks = np.unique(
        blocks
    )

    block_count = len(
        unique_blocks
    )

    config_count = len(
        top
    )

    block_profit = np.zeros(
        (
            block_count,
            config_count,
        ),
        dtype=float,
    )

    block_bets = np.zeros(
        (
            block_count,
            config_count,
        ),
        dtype=float,
    )

    for (
        k,
        row,
    ) in top.iterrows():
        opps = config_opportunities(
            meta,
            market,
            row,
            caches,
        )

        edge = float(
            row[
                'selected_edge'
            ]
        )

        selected_opps = (
            select_opportunities(
                opps,
                selection_policy,
                edge_mode='shared',
                shared_edge=edge,
            )
        )

        profits = selected_opps[
            'unit_profit'
        ].to_numpy(
            float
        )

        bets = selected_opps[
            'selected_bets'
        ].to_numpy(
            float
        )

        for (
            bi,
            block,
        ) in enumerate(
            unique_blocks
        ):
            m = (
                blocks
                == block
            )

            block_profit[
                bi,
                k
            ] = np.nansum(
                profits[
                    m
                ]
            )

            block_bets[
                bi,
                k
            ] = np.sum(
                bets[
                    m
                ]
            )

    draws = rng.multinomial(
        block_count,
        np.repeat(
            1.0
            / block_count,
            block_count,
        ),
        size=reps,
    )

    scenario_profit = (
        draws
        @ block_profit
    )

    scenario_bets = (
        draws
        @ block_bets
    )

    scenario_roi = np.divide(
        scenario_profit,
        scenario_bets,
        out=np.full_like(
            scenario_profit,
            np.nan,
        ),
        where=(
            scenario_bets
            > 0
        ),
    )

    winners = np.nanargmax(
        scenario_profit,
        axis=1,
    )

    stress_rows = []

    for (
        k,
        row,
    ) in top.iterrows():
        stress_rows.append({
            'rank_before_stress': int(
                row[
                    'rank'
                ]
            ),
            'market': market,
            'selection_mode': (
                selection_policy.selection_mode
            ),
            'pick_preference_metric': (
                selection_policy.preference_metric
            ),
            'pick_preference_direction': (
                selection_policy.preference_direction
            ),
            'bias_strategy': row[
                'bias_strategy'
            ],
            'std_mode': row[
                'std_mode'
            ],
            'calibration_side1': row[
                f'calibration_{market_sides(market)[0].lower()}'
            ],
            'calibration_side2': row[
                f'calibration_{market_sides(market)[1].lower()}'
            ],
            'edge': float(
                row[
                    'selected_edge'
                ]
            ),
            'selection_frequency_as_best_profit': float(
                np.mean(
                    winners
                    == k
                )
            ),
            'positive_roi_probability': float(
                np.nanmean(
                    scenario_roi[
                        :,
                        k
                    ]
                    > 0
                )
            ),
            'bootstrap_profit_median': percentile(
                scenario_profit[
                    :,
                    k
                ],
                50,
            ),
            'bootstrap_profit_2_5': percentile(
                scenario_profit[
                    :,
                    k
                ],
                2.5,
            ),
            'bootstrap_profit_97_5': percentile(
                scenario_profit[
                    :,
                    k
                ],
                97.5,
            ),
            'bootstrap_roi_median': percentile(
                scenario_roi[
                    :,
                    k
                ],
                50,
            ),
            'bootstrap_roi_2_5': percentile(
                scenario_roi[
                    :,
                    k
                ],
                2.5,
            ),
            'bootstrap_roi_97_5': percentile(
                scenario_roi[
                    :,
                    k
                ],
                97.5,
            ),
        })

    stress = pd.DataFrame(
        stress_rows
    )

    eligible = (
        stress[
            stress[
                'positive_roi_probability'
            ]
            >= MIN_STRESS_POSITIVE_ROI_PROBABILITY
        ]
        .copy()
    )

    if eligible.empty:
        eligible = stress.copy()

    chosen_stress = (
        eligible
        .sort_values(
            [
                'bootstrap_profit_median',
                'selection_frequency_as_best_profit',
                'positive_roi_probability',
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .iloc[0]
    )

    chosen_rank = int(
        chosen_stress[
            'rank_before_stress'
        ]
    )

    chosen = (
        ranking[
            ranking[
                'rank'
            ]
            == chosen_rank
        ]
        .iloc[0]
    )

    return (
        stress
        .sort_values(
            'bootstrap_profit_median',
            ascending=False,
        )
        .reset_index(
            drop=True
        ),
        chosen,
    )


def calibration_method_summary(
    meta: pd.DataFrame,
    market: str,
    cache: dict[
        str,
        Any,
    ],
) -> pd.DataFrame:
    (
        s1,
        s2,
    ) = market_sides(
        market
    )

    (
        y1_col,
        y2_col,
    ) = side_outcome_columns(
        market
    )

    rows = []

    if uses_complementary_calibration(
        market
    ):
        y1 = meta[
            y1_col
        ].to_numpy(
            float
        )

        y2 = meta[
            y2_col
        ].to_numpy(
            float
        )

        raw_p1 = cache[
            'side1'
        ][
            'raw'
        ]

        raw_p2 = cache[
            'side2'
        ][
            'raw'
        ]

        (
            raw_ll,
            raw_br,
        ) = probability_pair_score(
            meta,
            market,
            raw_p1,
            raw_p2,
        )

        for method in CALIBRATION_METHODS:
            p1 = cache[
                'side1'
            ][
                method
            ]

            p2 = cache[
                'side2'
            ][
                method
            ]

            (
                ll,
                br,
            ) = probability_pair_score(
                meta,
                market,
                p1,
                p2,
            )

            fold_wins_ll = 0
            fold_wins_br = 0
            folds = 0

            for (
                _,
                gidx,
            ) in meta.groupby(
                'fold_id'
            ).groups.items():
                idx = np.asarray(
                    list(
                        gidx
                    ),
                    dtype=int,
                )

                mll = float(
                    np.nanmean([
                        binary_log_loss(
                            p1[
                                idx
                            ],
                            y1[
                                idx
                            ],
                        ),
                        binary_log_loss(
                            p2[
                                idx
                            ],
                            y2[
                                idx
                            ],
                        ),
                    ])
                )

                rll = float(
                    np.nanmean([
                        binary_log_loss(
                            raw_p1[
                                idx
                            ],
                            y1[
                                idx
                            ],
                        ),
                        binary_log_loss(
                            raw_p2[
                                idx
                            ],
                            y2[
                                idx
                            ],
                        ),
                    ])
                )

                mbr = float(
                    np.nanmean([
                        brier_score(
                            p1[
                                idx
                            ],
                            y1[
                                idx
                            ],
                        ),
                        brier_score(
                            p2[
                                idx
                            ],
                            y2[
                                idx
                            ],
                        ),
                    ])
                )

                rbr = float(
                    np.nanmean([
                        brier_score(
                            raw_p1[
                                idx
                            ],
                            y1[
                                idx
                            ],
                        ),
                        brier_score(
                            raw_p2[
                                idx
                            ],
                            y2[
                                idx
                            ],
                        ),
                    ])
                )

                if (
                    np.isfinite(
                        mll
                    )
                    and np.isfinite(
                        rll
                    )
                ):
                    folds += 1

                    fold_wins_ll += int(
                        mll < rll
                    )

                    fold_wins_br += int(
                        mbr < rbr
                    )

            rows.append({
                'market': market,
                'side': s1,
                'calibration_role': 'canonical',
                'derived_side': s2,
                'method': method,
                'oos_games': int(
                    np.isfinite(
                        y1
                    ).sum()
                ),
                'oos_log_loss': ll,
                'oos_brier': br,
                'log_loss_change_vs_raw': (
                    ll
                    - raw_ll
                ),
                'brier_change_vs_raw': (
                    br
                    - raw_br
                ),
                'fold_win_rate_log_loss': (
                    fold_wins_ll
                    / folds
                    if folds
                    else np.nan
                ),
                'fold_win_rate_brier': (
                    fold_wins_br
                    / folds
                    if folds
                    else np.nan
                ),
            })

        return pd.DataFrame(
            rows
        )

    for (
        side,
        ycol,
        pmap,
    ) in [
        (
            s1,
            y1_col,
            cache[
                'side1'
            ],
        ),
        (
            s2,
            y2_col,
            cache[
                'side2'
            ],
        ),
    ]:
        y = meta[
            ycol
        ].to_numpy(
            float
        )

        raw_ll = binary_log_loss(
            pmap[
                'raw'
            ],
            y,
        )

        raw_br = brier_score(
            pmap[
                'raw'
            ],
            y,
        )

        for method in CALIBRATION_METHODS:
            p = pmap[
                method
            ]

            fold_wins_ll = 0
            fold_wins_br = 0
            folds = 0

            for (
                _,
                gidx,
            ) in meta.groupby(
                'fold_id'
            ).groups.items():
                idx = np.asarray(
                    list(
                        gidx
                    ),
                    dtype=int,
                )

                ll = binary_log_loss(
                    p[
                        idx
                    ],
                    y[
                        idx
                    ],
                )

                ll_raw = binary_log_loss(
                    pmap[
                        'raw'
                    ][
                        idx
                    ],
                    y[
                        idx
                    ],
                )

                br = brier_score(
                    p[
                        idx
                    ],
                    y[
                        idx
                    ],
                )

                br_raw = brier_score(
                    pmap[
                        'raw'
                    ][
                        idx
                    ],
                    y[
                        idx
                    ],
                )

                if (
                    np.isfinite(
                        ll
                    )
                    and np.isfinite(
                        ll_raw
                    )
                ):
                    folds += 1
                    fold_wins_ll += int(
                        ll < ll_raw
                    )
                    fold_wins_br += int(
                        br < br_raw
                    )

            rows.append({
                'market': market,
                'side': side,
                'calibration_role': 'independent',
                'derived_side': '',
                'method': method,
                'oos_games': int(
                    np.isfinite(
                        y
                    ).sum()
                ),
                'oos_log_loss': binary_log_loss(
                    p,
                    y,
                ),
                'oos_brier': brier_score(
                    p,
                    y,
                ),
                'log_loss_change_vs_raw': (
                    binary_log_loss(
                        p,
                        y,
                    )
                    - raw_ll
                ),
                'brier_change_vs_raw': (
                    brier_score(
                        p,
                        y,
                    )
                    - raw_br
                ),
                'fold_win_rate_log_loss': (
                    fold_wins_ll
                    / folds
                    if folds
                    else np.nan
                ),
                'fold_win_rate_brier': (
                    fold_wins_br
                    / folds
                    if folds
                    else np.nan
                ),
            })

    return pd.DataFrame(
        rows
    )


def _selected_keys_from_production(
    df: pd.DataFrame,
    league: str,
    market: str,
    prod_select,
) -> set[
    tuple[
        str,
        str,
    ]
]:
    cfg = prod_select.market_cfg(
        league.lower(),
        market,
    )

    rows: set[
        tuple[
            str,
            str,
        ]
    ] = set()

    for (
        _,
        row,
    ) in df.iterrows():
        game_date = row.get(
            'game_date'
        )

        sides = (
            prod_select
            .SIDE_BUILDERS[
                market
            ](
                row,
                league.lower(),
                game_date,
                cfg,
            )
        )

        if not sides:
            continue

        mode = str(
            cfg.get(
                'selection_mode',
                'pick_one',
            )
        ).strip().lower()

        pref = (
            cfg.get(
                'pick_preference'
            )
            or {
                'metric': 'ev',
                'direction': 'max',
            }
        )

        picks = (
            sides
            if mode == 'all_qualifying'
            else [
                prod_select.pick(
                    sides,
                    pref,
                )
            ]
        )

        for sel in picks:
            if sel is not None:
                rows.add(
                    (
                        str(
                            row.get(
                                'game_id',
                                '',
                            )
                        ).strip(),
                        str(
                            sel[
                                'side'
                            ]
                        ).lower(),
                    )
                )

    return rows


def _selected_keys_from_core(
    opps: pd.DataFrame,
) -> set[
    tuple[
        str,
        str,
    ]
]:
    rows: set[
        tuple[
            str,
            str,
        ]
    ] = set()

    for (
        _,
        row,
    ) in opps.iterrows():
        gid = str(
            row.get(
                'game_id',
                '',
            )
        ).strip()

        if bool(
            row.get(
                'side1_selected',
                False,
            )
        ):
            rows.add(
                (
                    gid,
                    str(
                        row.get(
                            'side1_name',
                            '',
                        )
                    ).lower(),
                )
            )

        if bool(
            row.get(
                'side2_selected',
                False,
            )
        ):
            rows.add(
                (
                    gid,
                    str(
                        row.get(
                            'side2_name',
                            '',
                        )
                    ).lower(),
                )
            )

    return rows


def run_production_parity_test(
    full_df: pd.DataFrame,
    league: str,
    settings: dict[
        str,
        Any,
    ],
    selection_policies: dict[
        str,
        MarketSelectionPolicy,
    ],
    rows: int,
) -> None:
    n = max(
        1,
        int(
            rows
        ),
    )

    if len(
        full_df
    ) <= n:
        raise AssertionError(
            'PARITY FAILED: not enough '
            'history to create prior + '
            'target rows'
        )

    target = (
        full_df.tail(
            n
        )
        .copy()
        .reset_index(
            drop=True
        )
    )

    prior = (
        full_df.iloc[
            :-n
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    margin_bias = (
        production_bias_values_for_targets(
            prior,
            target,
            'spread',
            settings[
                'MARGIN_BIAS_RULE'
            ],
        )
    )

    total_bias = (
        production_bias_values_for_targets(
            prior,
            target,
            'total',
            settings[
                'TOTAL_BIAS_RULE'
            ],
        )
    )

    if (
        np.isnan(
            margin_bias
        ).any()
        or np.isnan(
            total_bias
        ).any()
    ):
        raise AssertionError(
            'PARITY FAILED: production '
            'bias not ready for parity '
            'target rows'
        )

    cleaned = target.copy()

    cleaned[
        'home_projected_points'
    ] = (
        cleaned[
            'raw_home_projected'
        ].to_numpy(
            float
        )
        - margin_bias
        / 2.0
        - total_bias
        / 2.0
    )

    cleaned[
        'away_projected_points'
    ] = (
        cleaned[
            'raw_away_projected'
        ].to_numpy(
            float
        )
        + margin_bias
        / 2.0
        - total_bias
        / 2.0
    )

    cleaned[
        'total_projected_points'
    ] = (
        cleaned[
            'raw_total'
        ].to_numpy(
            float
        )
        - total_bias
    )

    cleaned[
        'margin_bias'
    ] = margin_bias

    cleaned[
        'total_bias'
    ] = total_bias

    cleaned[
        'bias_applied'
    ] = 1

    prod_juice = load_module_from_path(
        (
            'master_parity_juice_'
            f'{league}'
        ),
        PRODUCTION_BUILD_JUICE,
    )

    prod_ev = load_module_from_path(
        (
            'master_parity_ev_'
            f'{league}'
        ),
        PRODUCTION_EV_KELLY,
    )

    prod_select = load_module_from_path(
        (
            'master_parity_select_'
            f'{league}'
        ),
        PRODUCTION_SELECT,
    )

    prod_settings = (
        prod_juice
        .LEAGUE_SETTINGS[
            league.upper()
        ]
    )

    with tempfile.TemporaryDirectory(
        prefix='master_parity_',
    ) as tmp:
        prod_juice.OUTPUT_DIR = (
            Path(
                tmp
            )
            / 'juice'
        )

        for market_dir in [
            'moneyline',
            'spread',
            'total',
        ]:
            (
                prod_juice.OUTPUT_DIR
                / league.lower()
                / market_dir
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

        for market in [
            'moneyline',
            'spread',
            'total',
        ]:
            (
                core_p1,
                core_p2,
            ) = current_pipeline_probabilities(
                target,
                market,
                settings,
                prior_history=prior,
            )

            if market == 'moneyline':
                (
                    out_path,
                    _,
                ) = (
                    prod_juice
                    .process_moneyline(
                        cleaned.copy(),
                        '2000_01_01',
                        league.upper(),
                        prod_settings,
                        league.lower(),
                    )
                )

                juice_df = pd.read_csv(
                    out_path,
                    dtype={
                        'game_id': str
                    },
                )

                prod_df = (
                    prod_ev
                    .process_moneyline(
                        juice_df
                    )
                )

                prod_p1 = pd.to_numeric(
                    prod_df[
                        'home_model_prob'
                    ],
                    errors='coerce',
                ).to_numpy(
                    float
                )

                prod_p2 = pd.to_numeric(
                    prod_df[
                        'away_model_prob'
                    ],
                    errors='coerce',
                ).to_numpy(
                    float
                )

            elif market == 'spread':
                (
                    out_path,
                    _,
                ) = (
                    prod_juice
                    .process_spread(
                        cleaned.copy(),
                        '2000_01_01',
                        league.upper(),
                        prod_settings,
                        league.lower(),
                    )
                )

                juice_df = pd.read_csv(
                    out_path,
                    dtype={
                        'game_id': str
                    },
                )

                prod_df = (
                    prod_ev
                    .process_spread(
                        juice_df
                    )
                )

                prod_p1 = pd.to_numeric(
                    prod_df[
                        'home_spread_model_prob'
                    ],
                    errors='coerce',
                ).to_numpy(
                    float
                )

                prod_p2 = pd.to_numeric(
                    prod_df[
                        'away_spread_model_prob'
                    ],
                    errors='coerce',
                ).to_numpy(
                    float
                )

            else:
                (
                    out_path,
                    _,
                ) = (
                    prod_juice
                    .process_totals(
                        cleaned.copy(),
                        '2000_01_01',
                        league.upper(),
                        prod_settings,
                        league.lower(),
                    )
                )

                juice_df = pd.read_csv(
                    out_path,
                    dtype={
                        'game_id': str
                    },
                )

                prod_df = (
                    prod_ev
                    .process_total(
                        juice_df
                    )
                )

                prod_p1 = pd.to_numeric(
                    prod_df[
                        'over_model_prob'
                    ],
                    errors='coerce',
                ).to_numpy(
                    float
                )

                prod_p2 = pd.to_numeric(
                    prod_df[
                        'under_model_prob'
                    ],
                    errors='coerce',
                ).to_numpy(
                    float
                )

            if not np.allclose(
                core_p1,
                prod_p1,
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            ):
                raise AssertionError(
                    'PARITY FAILED '
                    f'{league}.{market}.'
                    'side1 probabilities'
                )

            if not np.allclose(
                core_p2,
                prod_p2,
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            ):
                raise AssertionError(
                    'PARITY FAILED '
                    f'{league}.{market}.'
                    'side2 probabilities'
                )

            meta = target.copy()

            meta[
                'fold_id'
            ] = 0

            core_opps = (
                edge_mode_opportunities(
                    meta,
                    market,
                    core_p1,
                    core_p2,
                    policy=(
                        selection_policies[
                            market
                        ]
                    ),
                    edge_mode='shared',
                    shared_edge=current_edge_for_market(
                        market,
                        settings,
                    ),
                )
            )

            prod_keys = (
                _selected_keys_from_production(
                    prod_df,
                    league,
                    market,
                    prod_select,
                )
            )

            core_keys = (
                _selected_keys_from_core(
                    core_opps
                )
            )

            if prod_keys != core_keys:
                raise AssertionError(
                    'PARITY FAILED '
                    f'{league}.{market}.'
                    'selection: master='
                    f'{sorted(core_keys)} '
                    'production='
                    f'{sorted(prod_keys)}'
                )

    progress(
        'Production parity PASS: '
        f'{league} rows={len(target)}'
    )


def current_pipeline_probabilities(
    df: pd.DataFrame,
    market: str,
    settings: dict[
        str,
        Any,
    ],
    prior_history: (
        pd.DataFrame
        | None
    ) = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    cal = settings[
        'CALIBRATION'
    ]

    if market == 'moneyline':
        home = (
            apply_production_independent_calibration(
                pd.to_numeric(
                    df[
                        'home_prob'
                    ],
                    errors='coerce',
                ).to_numpy(
                    float
                ),
                cal[
                    'moneyline'
                ][
                    'home'
                ],
            )
        )

        away = (
            apply_production_independent_calibration(
                pd.to_numeric(
                    df[
                        'away_prob'
                    ],
                    errors='coerce',
                ).to_numpy(
                    float
                ),
                cal[
                    'moneyline'
                ][
                    'away'
                ],
            )
        )

        return (
            home,
            away,
        )

    prior = (
        prior_history
        if prior_history
        is not None
        else pd.DataFrame()
    )

    if market == 'spread':
        bias = (
            production_bias_values_for_targets(
                prior,
                df,
                'spread',
                settings[
                    'MARGIN_BIAS_RULE'
                ],
            )
        )

        if np.isnan(
            bias
        ).any():
            raise ValueError(
                'Production margin bias '
                'is not ready for all '
                'requested rows'
            )

        mean = (
            df[
                'raw_margin'
            ].to_numpy(
                float
            )
            - bias
        )

        sigma = np.repeat(
            float(
                settings[
                    'SPREAD_STD'
                ]
            ),
            len(
                df
            ),
        )

        raw_home = (
            home_spread_probability(
                mean,
                df[
                    'home_spread'
                ].to_numpy(
                    float
                ),
                sigma,
            )
        )

        raw_away = (
            1.0
            - raw_home
        )

        return (
            apply_production_complementary_calibration(
                raw_home,
                raw_away,
                cal[
                    'spread'
                ],
                'home',
                'away',
            )
        )

    bias = production_bias_values_for_targets(
        prior,
        df,
        'total',
        settings[
            'TOTAL_BIAS_RULE'
        ],
    )

    if np.isnan(
        bias
    ).any():
        raise ValueError(
            'Production total bias '
            'is not ready for all '
            'requested rows'
        )

    mean = (
        df[
            'raw_total'
        ].to_numpy(
            float
        )
        - bias
    )

    sigma = np.repeat(
        float(
            settings[
                'TOTAL_STD'
            ]
        ),
        len(
            df
        ),
    )

    raw_over = over_probability(
        mean,
        df[
            'total'
        ].to_numpy(
            float
        ),
        sigma,
    )

    raw_under = (
        1.0
        - raw_over
    )

    return (
        apply_production_complementary_calibration(
            raw_over,
            raw_under,
            cal[
                'total'
            ],
            'over',
            'under',
        )
    )


def current_edge_for_market(
    market: str,
    settings: dict[
        str,
        float,
    ],
) -> float:
    return {
        'moneyline': settings[
            'ML_EDGE'
        ],
        'spread': settings[
            'SPREAD_EDGE'
        ],
        'total': settings[
            'TOTAL_EDGE'
        ],
    }[
        market
    ]


def selected_config_components(
    row: pd.Series,
    market: str,
) -> tuple[
    str,
    str,
    str,
    str,
    float,
]:
    (
        s1,
        s2,
    ) = market_sides(
        market
    )

    cal1 = str(
        row[
            f'calibration_{s1.lower()}'
        ]
    )

    cal2 = str(
        row[
            f'calibration_{s2.lower()}'
        ]
    )

    if (
        uses_complementary_calibration(
            market
        )
        and cal2 != 'complement'
    ):
        raise ValueError(
            f'{market} must use '
            'complementary calibration; '
            f'found {s2} '
            f'calibration={cal2!r}'
        )

    return (
        str(
            row[
                'bias_strategy'
            ]
        ),
        str(
            row[
                'std_mode'
            ]
        ),
        cal1,
        cal2,
        float(
            row[
                'selected_edge'
            ]
        ),
    )


def fit_selected_config(
    train: pd.DataFrame,
    market: str,
    chosen: pd.Series,
) -> dict[
    str,
    Any,
]:
    (
        bias_strategy,
        std_mode,
        cal1_method,
        cal2_method,
        edge,
    ) = selected_config_components(
        chosen,
        market,
    )

    model = fit_base_model(
        train,
        market,
        bias_strategy=(
            'none'
            if market == 'moneyline'
            else bias_strategy
        ),
        std_mode=(
            'fixed'
            if market == 'moneyline'
            else std_mode
        ),
    )

    base_train = apply_base_model(
        model,
        train,
    )

    (
        y1_col,
        y2_col,
    ) = side_outcome_columns(
        market
    )

    cal1 = fit_calibrator(
        base_train[
            'side1_prob'
        ],
        train[
            y1_col
        ].to_numpy(
            float
        ),
        cal1_method,
    )

    if uses_complementary_calibration(
        market
    ):
        cal2 = {
            'method': 'complement',
            'source_side': (
                market_sides(
                    market
                )[0]
            ),
        }

    else:
        cal2 = fit_calibrator(
            base_train[
                'side2_prob'
            ],
            train[
                y2_col
            ].to_numpy(
                float
            ),
            cal2_method,
        )

    return {
        'market': market,
        'base_model': model,
        'cal1': cal1,
        'cal2': cal2,
        'cal1_method': cal1_method,
        'cal2_method': cal2_method,
        'edge': edge,
        'complementary_calibration': (
            uses_complementary_calibration(
                market
            )
        ),
    }


def apply_selected_config(
    fitted: dict[
        str,
        Any,
    ],
    df: pd.DataFrame,
) -> dict[
    str,
    Any,
]:
    base = apply_base_model(
        fitted[
            'base_model'
        ],
        df,
    )

    p1 = apply_calibrator(
        fitted[
            'cal1'
        ],
        base[
            'side1_prob'
        ],
    )

    if fitted.get(
        'complementary_calibration',
        False,
    ):
        p2 = (
            1.0
            - p1
        )

    else:
        p2 = apply_calibrator(
            fitted[
                'cal2'
            ],
            base[
                'side2_prob'
            ],
        )

    if (
        fitted.get(
            'complementary_calibration',
            False,
        )
        and not np.allclose(
            p1 + p2,
            1.0,
            atol=1e-12,
            equal_nan=True,
        )
    ):
        raise ValueError(
            f"{fitted['market']} "
            'calibrated probabilities '
            'are not complementary'
        )

    return {
        **base,
        'p1_raw': base[
            'side1_prob'
        ],
        'p2_raw': base[
            'side2_prob'
        ],
        'p1': p1,
        'p2': p2,
    }


def edge_mode_opportunities(
    meta: pd.DataFrame,
    market: str,
    p1: Any,
    p2: Any,
    policy: MarketSelectionPolicy,
    edge_mode: str,
    shared_edge: float | None = None,
    edge_side1: float | None = None,
    edge_side2: float | None = None,
) -> pd.DataFrame:
    base = market_opportunities(
        meta,
        market,
        p1,
        p2,
    )

    return select_opportunities(
        base,
        policy,
        edge_mode=edge_mode,
        shared_edge=shared_edge,
        edge_side1=edge_side1,
        edge_side2=edge_side2,
    )


def betting_summary_from_opportunities(
    opps: pd.DataFrame,
) -> tuple[
    int,
    float,
    float,
]:
    if 'selected_bets' not in opps.columns:
        raise ValueError(
            'Opportunity frame is '
            'missing selected_bets'
        )

    bets = int(
        np.nansum(
            opps[
                'selected_bets'
            ].to_numpy(
                float
            )
        )
    )

    if not bets:
        return (
            0,
            0.0,
            np.nan,
        )

    profit = float(
        np.nansum(
            opps[
                'unit_profit'
            ].to_numpy(
                float
            )
        )
    )

    roi = (
        profit
        / bets
    )

    return (
        bets,
        profit,
        roi,
    )


def evaluate_frozen_candidate_on_lockbox(
    dev: pd.DataFrame,
    lockbox: pd.DataFrame,
    market: str,
    frozen: dict[
        str,
        Any,
    ],
    settings: dict[
        str,
        float,
    ],
    selection_policy: MarketSelectionPolicy,
) -> tuple[
    pd.DataFrame,
    dict[
        str,
        Any,
    ],
    dict[
        str,
        Any,
    ],
]:
    chosen: pd.Series = (
        frozen[
            'chosen'
        ]
    )

    fitted = fit_selected_config(
        dev,
        market,
        chosen,
    )

    pred = apply_selected_config(
        fitted,
        lockbox,
    )

    meta = lockbox.copy()

    meta[
        'fold_id'
    ] = 999

    candidate_opps = (
        edge_mode_opportunities(
            meta,
            market,
            pred[
                'p1'
            ],
            pred[
                'p2'
            ],
            policy=selection_policy,
            edge_mode=str(
                frozen[
                    'edge_mode'
                ]
            ),
            shared_edge=frozen.get(
                'shared_edge'
            ),
            edge_side1=frozen.get(
                'edge_side1'
            ),
            edge_side2=frozen.get(
                'edge_side2'
            ),
        )
    )

    (
        cand_bets,
        cand_profit,
        cand_roi,
    ) = betting_summary_from_opportunities(
        candidate_opps
    )

    (
        cur_p1,
        cur_p2,
    ) = current_pipeline_probabilities(
        lockbox,
        market,
        settings,
        prior_history=dev,
    )

    current_opps = (
        edge_mode_opportunities(
            meta,
            market,
            cur_p1,
            cur_p2,
            policy=selection_policy,
            edge_mode='shared',
            shared_edge=current_edge_for_market(
                market,
                settings,
            ),
        )
    )

    (
        cur_bets,
        cur_profit,
        cur_roi,
    ) = betting_summary_from_opportunities(
        current_opps
    )

    (
        y1_col,
        y2_col,
    ) = side_outcome_columns(
        market
    )

    y1 = lockbox[
        y1_col
    ].to_numpy(
        float
    )

    y2 = lockbox[
        y2_col
    ].to_numpy(
        float
    )

    candidate_ll = float(
        np.nanmean([
            binary_log_loss(
                pred[
                    'p1'
                ],
                y1,
            ),
            binary_log_loss(
                pred[
                    'p2'
                ],
                y2,
            ),
        ])
    )

    candidate_br = float(
        np.nanmean([
            brier_score(
                pred[
                    'p1'
                ],
                y1,
            ),
            brier_score(
                pred[
                    'p2'
                ],
                y2,
            ),
        ])
    )

    current_ll = float(
        np.nanmean([
            binary_log_loss(
                cur_p1,
                y1,
            ),
            binary_log_loss(
                cur_p2,
                y2,
            ),
        ])
    )

    current_br = float(
        np.nanmean([
            brier_score(
                cur_p1,
                y1,
            ),
            brier_score(
                cur_p2,
                y2,
            ),
        ])
    )

    rows = [
        {
            'market': market,
            'system': 'frozen_candidate',
            'selection_mode': (
                selection_policy.selection_mode
            ),
            'pick_preference_metric': (
                selection_policy.preference_metric
            ),
            'pick_preference_direction': (
                selection_policy.preference_direction
            ),
            'edge_mode': frozen[
                'edge_mode'
            ],
            'edge': (
                frozen.get(
                    'shared_edge',
                    np.nan,
                )
                if frozen[
                    'edge_mode'
                ] == 'shared'
                else np.nan
            ),
            'edge_side1': frozen.get(
                'edge_side1',
                np.nan,
            ),
            'edge_side2': frozen.get(
                'edge_side2',
                np.nan,
            ),
            'lockbox_games': len(
                lockbox
            ),
            'prob_log_loss': candidate_ll,
            'prob_brier': candidate_br,
            'bets': cand_bets,
            'profit_units': cand_profit,
            'roi': cand_roi,
        },
        {
            'market': market,
            'system': 'current_pipeline',
            'selection_mode': (
                selection_policy.selection_mode
            ),
            'pick_preference_metric': (
                selection_policy.preference_metric
            ),
            'pick_preference_direction': (
                selection_policy.preference_direction
            ),
            'edge_mode': 'shared',
            'edge': current_edge_for_market(
                market,
                settings,
            ),
            'edge_side1': np.nan,
            'edge_side2': np.nan,
            'lockbox_games': len(
                lockbox
            ),
            'prob_log_loss': current_ll,
            'prob_brier': current_br,
            'bets': cur_bets,
            'profit_units': cur_profit,
            'roi': cur_roi,
        },
        {
            'market': market,
            'system': 'frozen_base_without_calibration',
            'selection_mode': (
                selection_policy.selection_mode
            ),
            'pick_preference_metric': (
                selection_policy.preference_metric
            ),
            'pick_preference_direction': (
                selection_policy.preference_direction
            ),
            'edge_mode': frozen[
                'edge_mode'
            ],
            'edge': np.nan,
            'edge_side1': np.nan,
            'edge_side2': np.nan,
            'lockbox_games': len(
                lockbox
            ),
            'prob_log_loss': float(
                np.nanmean([
                    binary_log_loss(
                        pred[
                            'p1_raw'
                        ],
                        y1,
                    ),
                    binary_log_loss(
                        pred[
                            'p2_raw'
                        ],
                        y2,
                    ),
                ])
            ),
            'prob_brier': float(
                np.nanmean([
                    brier_score(
                        pred[
                            'p1_raw'
                        ],
                        y1,
                    ),
                    brier_score(
                        pred[
                            'p2_raw'
                        ],
                        y2,
                    ),
                ])
            ),
            'bets': np.nan,
            'profit_units': np.nan,
            'roi': np.nan,
        },
    ]

    lock_detail = (
        candidate_opps.copy()
    )

    lock_detail[
        'current_selected'
    ] = current_opps[
        'selected'
    ].to_numpy(
        bool
    )

    lock_detail[
        'current_selected_bets'
    ] = current_opps[
        'selected_bets'
    ].to_numpy(
        int
    )

    lock_detail[
        'current_side1_selected'
    ] = current_opps[
        'side1_selected'
    ].to_numpy(
        bool
    )

    lock_detail[
        'current_side2_selected'
    ] = current_opps[
        'side2_selected'
    ].to_numpy(
        bool
    )

    lock_detail[
        'current_selected_side'
    ] = current_opps[
        'selected_side'
    ].to_numpy(
        object
    )

    lock_detail[
        'current_selected_ev'
    ] = current_opps[
        'selected_ev'
    ].to_numpy(
        float
    )

    lock_detail[
        'current_selected_ev_sum'
    ] = current_opps[
        'selected_ev_sum'
    ].to_numpy(
        float
    )

    lock_detail[
        'current_unit_profit'
    ] = current_opps[
        'unit_profit'
    ].to_numpy(
        float
    )

    lock_detail[
        'selected_p1'
    ] = pred[
        'p1'
    ]

    lock_detail[
        'selected_p2'
    ] = pred[
        'p2'
    ]

    lock_detail[
        'selected_p1_raw'
    ] = pred[
        'p1_raw'
    ]

    lock_detail[
        'selected_p2_raw'
    ] = pred[
        'p2_raw'
    ]

    lock_detail[
        'selected_mean'
    ] = pred[
        'mean'
    ]

    lock_detail[
        'selected_sigma'
    ] = pred[
        'sigma'
    ]

    lock_detail[
        'selected_range_bin'
    ] = pred[
        'range_bin'
    ]

    lock_detail[
        'side1_y'
    ] = y1

    lock_detail[
        'side2_y'
    ] = y2

    return (
        pd.DataFrame(
            rows
        ),
        fitted,
        {
            'prediction': pred,
            'detail': lock_detail,
        },
    )


def lockbox_market_validation(
    lockbox_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for market in [
        'moneyline',
        'spread',
        'total',
    ]:
        cand = lockbox_summary[
            (
                lockbox_summary[
                    'market'
                ]
                == market
            )
            & (
                lockbox_summary[
                    'system'
                ]
                == 'frozen_candidate'
            )
        ].iloc[0]

        cur = lockbox_summary[
            (
                lockbox_summary[
                    'market'
                ]
                == market
            )
            & (
                lockbox_summary[
                    'system'
                ]
                == 'current_pipeline'
            )
        ].iloc[0]

        probability_log_loss_pass = (
            np.isfinite(
                float(
                    cand[
                        'prob_log_loss'
                    ]
                )
            )
            and np.isfinite(
                float(
                    cur[
                        'prob_log_loss'
                    ]
                )
            )
            and float(
                cand[
                    'prob_log_loss'
                ]
            )
            <= float(
                cur[
                    'prob_log_loss'
                ]
            )
            + 1e-12
        )

        probability_brier_pass = (
            np.isfinite(
                float(
                    cand[
                        'prob_brier'
                    ]
                )
            )
            and np.isfinite(
                float(
                    cur[
                        'prob_brier'
                    ]
                )
            )
            and float(
                cand[
                    'prob_brier'
                ]
            )
            <= float(
                cur[
                    'prob_brier'
                ]
            )
            + 1e-12
        )

        profit_pass = (
            np.isfinite(
                float(
                    cand[
                        'profit_units'
                    ]
                )
            )
            and np.isfinite(
                float(
                    cur[
                        'profit_units'
                    ]
                )
            )
            and float(
                cand[
                    'profit_units'
                ]
            )
            >= float(
                cur[
                    'profit_units'
                ]
            )
            - 1e-12
        )

        roi_pass = (
            np.isfinite(
                float(
                    cand[
                        'roi'
                    ]
                )
            )
            and np.isfinite(
                float(
                    cur[
                        'roi'
                    ]
                )
            )
            and float(
                cand[
                    'roi'
                ]
            )
            >= float(
                cur[
                    'roi'
                ]
            )
            - 1e-12
        )

        market_validated = bool(
            probability_log_loss_pass
            and probability_brier_pass
            and profit_pass
            and roi_pass
        )

        rows.append({
            'market': market,
            'candidate_bets': int(
                cand[
                    'bets'
                ]
            ),
            'current_bets': int(
                cur[
                    'bets'
                ]
            ),
            'candidate_profit_units': float(
                cand[
                    'profit_units'
                ]
            ),
            'current_profit_units': float(
                cur[
                    'profit_units'
                ]
            ),
            'candidate_roi': float(
                cand[
                    'roi'
                ]
            ),
            'current_roi': float(
                cur[
                    'roi'
                ]
            ),
            'candidate_prob_log_loss': float(
                cand[
                    'prob_log_loss'
                ]
            ),
            'current_prob_log_loss': float(
                cur[
                    'prob_log_loss'
                ]
            ),
            'candidate_prob_brier': float(
                cand[
                    'prob_brier'
                ]
            ),
            'current_prob_brier': float(
                cur[
                    'prob_brier'
                ]
            ),
            'probability_log_loss_pass': bool(
                probability_log_loss_pass
            ),
            'probability_brier_pass': bool(
                probability_brier_pass
            ),
            'profit_pass': bool(
                profit_pass
            ),
            'roi_pass': bool(
                roi_pass
            ),
            'market_validated': (
                market_validated
            ),
        })

    return pd.DataFrame(
        rows
    )


def split_edge_scan(
    meta: pd.DataFrame,
    market: str,
    p1: Any,
    p2: Any,
    grid: np.ndarray,
    min_bets: int,
    selection_policy: MarketSelectionPolicy,
) -> pd.DataFrame:
    base = market_opportunities(
        meta,
        market,
        p1,
        p2,
    )

    rows = []

    for e1 in grid:
        for e2 in grid:
            selected = select_opportunities(
                base,
                selection_policy,
                edge_mode='split',
                edge_side1=float(
                    e1
                ),
                edge_side2=float(
                    e2
                ),
            )

            bets_by_game = selected[
                'selected_bets'
            ].to_numpy(
                int
            )

            profit_by_game = selected[
                'unit_profit'
            ].to_numpy(
                float
            )

            fold = selected[
                'fold_id'
            ].to_numpy(
                int
            )

            n = int(
                np.sum(
                    bets_by_game
                )
            )

            if n:
                profit = float(
                    np.nansum(
                        profit_by_game
                    )
                )

                roi = (
                    profit / n
                )

                fold_frame = pd.DataFrame({
                    'fold_id': fold,
                    'profit': profit_by_game,
                    'bets': bets_by_game,
                })

                fold_frame = (
                    fold_frame[
                        fold_frame[
                            'bets'
                        ] > 0
                    ]
                )

                fold_profits = (
                    fold_frame
                    .groupby(
                        'fold_id'
                    )[
                        'profit'
                    ]
                    .sum()
                )

                pfr = (
                    float(
                        (
                            fold_profits
                            > 0
                        ).mean()
                    )
                    if len(
                        fold_profits
                    )
                    else np.nan
                )

            else:
                (
                    profit,
                    roi,
                    pfr,
                ) = (
                    0.0,
                    np.nan,
                    np.nan,
                )

            rows.append({
                'edge_side1': float(
                    e1
                ),
                'edge_side2': float(
                    e2
                ),
                'selection_mode': (
                    selection_policy.selection_mode
                ),
                'pick_preference_metric': (
                    selection_policy.preference_metric
                ),
                'pick_preference_direction': (
                    selection_policy.preference_direction
                ),
                'bets': n,
                'profit_units': profit,
                'roi': roi,
                'positive_fold_rate': pfr,
                'eligible_min_bets': (
                    n
                    >= min_bets
                ),
            })

    return pd.DataFrame(
        rows
    )


def choose_split_edge(
    scan: pd.DataFrame,
    min_bets: int,
) -> pd.Series:
    e = (
        scan[
            scan[
                'bets'
            ] >= min_bets
        ]
        .copy()
    )

    if e.empty:
        e = (
            scan[
                scan[
                    'bets'
                ] > 0
            ]
            .copy()
        )

    if e.empty:
        return scan.iloc[
            0
        ]

    return (
        e.sort_values(
            [
                'profit_units',
                'positive_fold_rate',
                'bets',
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .iloc[0]
    )


def build_frozen_candidates(
    meta: pd.DataFrame,
    chosen_by_market: dict[
        str,
        pd.Series,
    ],
    caches_by_market: dict[
        str,
        dict[
            str,
            dict[
                str,
                Any,
            ],
        ],
    ],
    selection_policies: dict[
        str,
        MarketSelectionPolicy,
    ],
    output_dir: Path,
    prefix: str,
) -> tuple[
    dict[
        str,
        dict[
            str,
            Any,
        ],
    ],
    pd.DataFrame,
    list[
        Path
    ],
]:
    frozen: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    split_rows = []

    files: list[
        Path
    ] = []

    ml = chosen_by_market[
        'moneyline'
    ]

    frozen[
        'moneyline'
    ] = {
        'market': 'moneyline',
        'chosen': ml,
        'edge_mode': 'shared',
        'shared_edge': float(
            ml[
                'selected_edge'
            ]
        ),
        'edge_side1': None,
        'edge_side2': None,
    }

    for market in [
        'spread',
        'total',
    ]:
        chosen = chosen_by_market[
            market
        ]

        cache = caches_by_market[
            market
        ][
            str(
                chosen[
                    'cache_key'
                ]
            )
        ]

        (
            s1,
            s2,
        ) = market_sides(
            market
        )

        cal1 = str(
            chosen[
                f'calibration_{s1.lower()}'
            ]
        )

        p1 = cache[
            'side1'
        ][
            cal1
        ]

        if uses_complementary_calibration(
            market
        ):
            p2 = cache[
                'side2'
            ][
                cal1
            ]

        else:
            cal2 = str(
                chosen[
                    f'calibration_{s2.lower()}'
                ]
            )

            p2 = cache[
                'side2'
            ][
                cal2
            ]

        if not np.allclose(
            p1 + p2,
            1.0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(
                f'{market} frozen candidate '
                'probabilities are not '
                'complementary'
            )

        minbets = min_oos_bets(
            len(
                meta
            )
        )

        selection_policy = (
            selection_policies[
                market
            ]
        )

        shared_edge = float(
            chosen[
                'selected_edge'
            ]
        )

        shared_opps = (
            edge_mode_opportunities(
                meta,
                market,
                p1,
                p2,
                policy=selection_policy,
                edge_mode='shared',
                shared_edge=shared_edge,
            )
        )

        (
            shared_bets,
            shared_profit,
            shared_roi,
        ) = betting_summary_from_opportunities(
            shared_opps
        )

        split_scan_df = (
            split_edge_scan(
                meta,
                market,
                p1,
                p2,
                SPLIT_EDGE_GRID,
                minbets,
                selection_policy,
            )
        )

        split_best = choose_split_edge(
            split_scan_df,
            minbets,
        )

        split_profit = float(
            split_best[
                'profit_units'
            ]
        )

        split_bets = int(
            split_best[
                'bets'
            ]
        )

        split_roi = (
            float(
                split_best[
                    'roi'
                ]
            )
            if np.isfinite(
                split_best[
                    'roi'
                ]
            )
            else np.nan
        )

        split_pfr = (
            float(
                split_best[
                    'positive_fold_rate'
                ]
            )
            if np.isfinite(
                split_best[
                    'positive_fold_rate'
                ]
            )
            else np.nan
        )

        improvement = (
            split_profit
            - shared_profit
        ) / max(
            abs(
                shared_profit
            ),
            1.0,
        )

        dev_supports_split = bool(
            improvement
            >= MIN_SPLIT_EDGE_DEV_PROFIT_IMPROVEMENT
            and split_bets
            >= minbets
            and np.isfinite(
                split_pfr
            )
            and split_pfr
            >= 0.50
        )

        edge_mode = (
            'split'
            if dev_supports_split
            else 'shared'
        )

        frozen[
            market
        ] = {
            'market': market,
            'chosen': chosen,
            'edge_mode': edge_mode,
            'shared_edge': (
                shared_edge
                if edge_mode == 'shared'
                else None
            ),
            'edge_side1': (
                float(
                    split_best[
                        'edge_side1'
                    ]
                )
                if edge_mode == 'split'
                else None
            ),
            'edge_side2': (
                float(
                    split_best[
                        'edge_side2'
                    ]
                )
                if edge_mode == 'split'
                else None
            ),
        }

        split_rows.append({
            'market': market,
            'selection_mode': (
                selection_policy.selection_mode
            ),
            'pick_preference_metric': (
                selection_policy.preference_metric
            ),
            'pick_preference_direction': (
                selection_policy.preference_direction
            ),
            'calibration_architecture': (
                'canonical_side_plus_complement'
            ),
            'side1': s1,
            'side2': s2,
            'shared_edge': shared_edge,
            'dev_shared_bets': shared_bets,
            'dev_shared_profit': shared_profit,
            'dev_shared_roi': shared_roi,
            'split_edge_side1': float(
                split_best[
                    'edge_side1'
                ]
            ),
            'split_edge_side2': float(
                split_best[
                    'edge_side2'
                ]
            ),
            'dev_split_bets': split_bets,
            'dev_split_profit': split_profit,
            'dev_split_roi': split_roi,
            'dev_split_positive_fold_rate': split_pfr,
            'dev_split_profit_relative_improvement': improvement,
            'minimum_bets_required': minbets,
            'dev_supports_split': dev_supports_split,
            'frozen_edge_mode': edge_mode,
            'frozen_shared_edge': (
                shared_edge
                if edge_mode == 'shared'
                else np.nan
            ),
            'frozen_edge_side1': (
                float(
                    split_best[
                        'edge_side1'
                    ]
                )
                if edge_mode == 'split'
                else np.nan
            ),
            'frozen_edge_side2': (
                float(
                    split_best[
                        'edge_side2'
                    ]
                )
                if edge_mode == 'split'
                else np.nan
            ),
        })

        files.append(
            save_csv(
                split_scan_df,
                output_dir
                / (
                    f'{prefix}_06_'
                    f'{market}_'
                    'split_edge_scan.csv'
                ),
            )
        )

    split_summary = pd.DataFrame(
        split_rows
    )

    files.append(
        save_csv(
            split_summary,
            output_dir
            / (
                f'{prefix}_06_'
                'shared_vs_split_edges.csv'
            ),
        )
    )

    return (
        frozen,
        split_summary,
        files,
    )


def spread_size_bucket(
    abs_spread: pd.Series,
) -> pd.Series:
    return pd.cut(
        abs_spread,
        bins=[
            -np.inf,
            3.5,
            7.5,
            11.5,
            np.inf,
        ],
        labels=[
            '0_to_3.5',
            'over_3.5_to_7.5',
            'over_7.5_to_11.5',
            'over_11.5',
        ],
    )


def disagreement_bucket(
    abs_diff: pd.Series,
) -> pd.Series:
    return pd.cut(
        abs_diff,
        bins=[
            -np.inf,
            1,
            2,
            3,
            5,
            8,
            np.inf,
        ],
        labels=[
            'under_1',
            '1_to_2',
            '2_to_3',
            '3_to_5',
            '5_to_8',
            '8_plus',
        ],
    )


def segment_betting_summary(
    meta: pd.DataFrame,
    market: str,
    p1: np.ndarray,
    p2: np.ndarray,
    mean: np.ndarray,
    selection_policy: MarketSelectionPolicy,
    edge_mode: str,
    shared_edge: float | None = None,
    edge_side1: float | None = None,
    edge_side2: float | None = None,
) -> pd.DataFrame:
    (
        s1,
        s2,
    ) = market_sides(
        market
    )

    (
        o1,
        o2,
    ) = side_odds_arrays(
        meta,
        market,
    )

    (
        r1,
        r2,
    ) = side_result_arrays(
        meta,
        market,
    )

    (
        y1_col,
        y2_col,
    ) = side_outcome_columns(
        market
    )

    y1, y2 = (
        meta[
            y1_col
        ].to_numpy(
            float
        ),
        meta[
            y2_col
        ].to_numpy(
            float
        ),
    )

    ev1, ev2 = (
        p1
        * o1
        - 1.0,
        p2
        * o2
        - 1.0,
    )

    pr1, pr2 = (
        unit_profit_from_result(
            r1,
            o1,
        ),
        unit_profit_from_result(
            r2,
            o2,
        ),
    )

    selected_opps = (
        edge_mode_opportunities(
            meta,
            market,
            p1,
            p2,
            policy=selection_policy,
            edge_mode=edge_mode,
            shared_edge=shared_edge,
            edge_side1=edge_side1,
            edge_side2=edge_side2,
        )
    )

    if edge_mode == 'shared':
        if shared_edge is None:
            raise ValueError(
                'shared_edge required '
                'for segment analysis '
                'with shared edge'
            )

        thresholds = {
            s1: float(
                shared_edge
            ),
            s2: float(
                shared_edge
            ),
        }

    elif edge_mode == 'split':
        if (
            edge_side1 is None
            or edge_side2 is None
        ):
            raise ValueError(
                'side edges required '
                'for segment analysis '
                'with split edges'
            )

        thresholds = {
            s1: float(
                edge_side1
            ),
            s2: float(
                edge_side2
            ),
        }

    else:
        raise ValueError(
            f'Unknown edge_mode: '
            f'{edge_mode}'
        )

    temp_rows = []

    for (
        side,
        pp,
        yy,
        ev,
        pr,
        selected_mask,
    ) in [
        (
            s1,
            p1,
            y1,
            ev1,
            pr1,
            selected_opps[
                'side1_selected'
            ].to_numpy(
                bool
            ),
        ),
        (
            s2,
            p2,
            y2,
            ev2,
            pr2,
            selected_opps[
                'side2_selected'
            ].to_numpy(
                bool
            ),
        ),
    ]:
        threshold = (
            thresholds[
                side
            ]
        )

        t = pd.DataFrame({
            'side': side,
            'p': pp,
            'y': yy,
            'ev': ev,
            'profit': pr,
            'home_spread': meta[
                'home_spread'
            ].to_numpy(
                float
            ),
            'book_total': meta[
                'total'
            ].to_numpy(
                float
            ),
            'model_mean': mean,
            'selected_by_production_policy': selected_mask,
        })

        if market == 'spread':
            t[
                'segment'
            ] = spread_size_bucket(
                np.abs(
                    t[
                        'home_spread'
                    ]
                )
            )

        else:
            try:
                t[
                    'total_range'
                ] = pd.qcut(
                    t[
                        'book_total'
                    ],
                    q=4,
                    duplicates='drop',
                )

            except ValueError:
                t[
                    'total_range'
                ] = 'all'

            diff = np.abs(
                t[
                    'model_mean'
                ]
                - t[
                    'book_total'
                ]
            )

            t[
                'disagreement'
            ] = disagreement_bucket(
                diff
            )

        if market == 'spread':
            groups = [
                (
                    'spread_size',
                    t[
                        'segment'
                    ],
                )
            ]

        else:
            groups = [
                (
                    'sportsbook_total_range',
                    t[
                        'total_range'
                    ],
                ),
                (
                    'model_book_disagreement',
                    t[
                        'disagreement'
                    ],
                ),
            ]

        for (
            segment_type,
            segment_series,
        ) in groups:
            t2 = t.copy()

            t2[
                'segment'
            ] = (
                segment_series
                .astype(str)
            )

            for (
                seg,
                g,
            ) in t2.groupby(
                'segment',
                dropna=False,
            ):
                betmask = (
                    np.isfinite(
                        g[
                            'profit'
                        ]
                    )
                    & g[
                        'selected_by_production_policy'
                    ].astype(
                        bool
                    )
                )

                temp_rows.append({
                    'market': market,
                    'side': side,
                    'selection_mode': (
                        selection_policy.selection_mode
                    ),
                    'pick_preference_metric': (
                        selection_policy.preference_metric
                    ),
                    'pick_preference_direction': (
                        selection_policy.preference_direction
                    ),
                    'edge_mode': edge_mode,
                    'side_edge_threshold': threshold,
                    'segment_type': segment_type,
                    'segment': str(
                        seg
                    ),
                    'games': len(
                        g
                    ),
                    'mean_probability': float(
                        np.nanmean(
                            g[
                                'p'
                            ]
                        )
                    ),
                    'actual_rate': float(
                        np.nanmean(
                            g[
                                'y'
                            ]
                        )
                    ),
                    'brier': brier_score(
                        g[
                            'p'
                        ],
                        g[
                            'y'
                        ],
                    ),
                    'log_loss': binary_log_loss(
                        g[
                            'p'
                        ],
                        g[
                            'y'
                        ],
                    ),
                    'bets_at_frozen_edge': int(
                        betmask.sum()
                    ),
                    'profit_at_frozen_edge': float(
                        np.nansum(
                            g.loc[
                                betmask,
                                'profit',
                            ]
                        )
                    ),
                    'roi_at_frozen_edge': (
                        float(
                            np.nanmean(
                                g.loc[
                                    betmask,
                                    'profit',
                                ]
                            )
                        )
                        if betmask.any()
                        else np.nan
                    ),
                })

    return pd.DataFrame(
        temp_rows
    )


def production_parameter_tables(
    full_df: pd.DataFrame,
    frozen_candidates: dict[
        str,
        dict[
            str,
            Any,
        ],
    ],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    action_rows = []
    std_rows = []
    cal_rows = []

    for (
        market,
        frozen,
    ) in frozen_candidates.items():
        chosen: pd.Series = (
            frozen[
                'chosen'
            ]
        )

        fitted = fit_selected_config(
            full_df,
            market,
            chosen,
        )

        (
            s1,
            s2,
        ) = market_sides(
            market
        )

        if market != 'moneyline':
            base = fitted[
                'base_model'
            ]

            sm: StdModel = (
                base[
                    'std_model'
                ]
            )

            action_rows.append({
                'market': market,
                'component': 'BIAS',
                'side': '',
                'mode': base[
                    'bias_strategy'
                ],
                'production_value': float(
                    base[
                        'bias'
                    ]
                ),
                'notes': (
                    'Full-history refit of '
                    'the frozen development-'
                    'selected bias method. '
                    'If rolling mode is '
                    'validated, recompute the '
                    'trailing estimate as new '
                    'results arrive.'
                ),
            })

            action_rows.append({
                'market': market,
                'component': 'STD_MODE',
                'side': '',
                'mode': sm.mode,
                'production_value': (
                    sm.global_sigma
                    if sm.mode == 'fixed'
                    else np.nan
                ),
                'notes': (
                    'One fixed STD is used '
                    'for all games.'
                    if sm.mode == 'fixed'
                    else (
                        'Adaptive STD uses the '
                        'bins in '
                        'FINAL_STD_RANGES.csv.'
                    )
                ),
            })

            if sm.mode == 'fixed':
                std_rows.append({
                    'market': market,
                    'std_mode': sm.mode,
                    'bin': 0,
                    'feature_low_exclusive': -np.inf,
                    'feature_high_inclusive': np.inf,
                    'sigma': sm.global_sigma,
                    'training_games': sm.counts[
                        0
                    ],
                })

            else:
                edges = (
                    [
                        -np.inf
                    ]
                    + sm.edges
                    + [
                        np.inf
                    ]
                )

                for (
                    i,
                    sigma,
                ) in enumerate(
                    sm.sigmas
                ):
                    std_rows.append({
                        'market': market,
                        'std_mode': sm.mode,
                        'bin': i,
                        'feature_low_exclusive': edges[
                            i
                        ],
                        'feature_high_inclusive': edges[
                            i + 1
                        ],
                        'sigma': sigma,
                        'training_games': sm.counts[
                            i
                        ],
                    })

        if (
            frozen[
                'edge_mode'
            ]
            == 'shared'
        ):
            action_rows.append({
                'market': market,
                'component': 'EDGE',
                'side': '',
                'mode': 'shared',
                'production_value': float(
                    frozen[
                        'shared_edge'
                    ]
                ),
                'notes': (
                    'Frozen on development '
                    'OOS data; never retuned '
                    'on lockbox.'
                ),
            })

        else:
            action_rows.append({
                'market': market,
                'component': 'EDGE_MODE',
                'side': '',
                'mode': 'separate_by_side',
                'production_value': np.nan,
                'notes': (
                    'Split EDGE mode selected '
                    'and frozen on development '
                    'OOS data only.'
                ),
            })

            action_rows.append({
                'market': market,
                'component': 'EDGE',
                'side': s1,
                'mode': 'side_specific',
                'production_value': float(
                    frozen[
                        'edge_side1'
                    ]
                ),
                'notes': (
                    'Frozen on development '
                    'OOS data; never retuned '
                    'on lockbox.'
                ),
            })

            action_rows.append({
                'market': market,
                'component': 'EDGE',
                'side': s2,
                'mode': 'side_specific',
                'production_value': float(
                    frozen[
                        'edge_side2'
                    ]
                ),
                'notes': (
                    'Frozen on development '
                    'OOS data; never retuned '
                    'on lockbox.'
                ),
            })

        if uses_complementary_calibration(
            market
        ):
            canonical = fitted[
                'cal1'
            ]

            cal_rows.append({
                'market': market,
                'side': s1,
                'calibration_role': 'canonical',
                'method': canonical.get(
                    'method',
                    'raw',
                ),
                'formula': calibrator_formula(
                    canonical
                ),
                'intercept': canonical.get(
                    'intercept',
                    np.nan,
                ),
                'slope': canonical.get(
                    'slope',
                    np.nan,
                ),
                'coef_log_p': canonical.get(
                    'coef_log_p',
                    np.nan,
                ),
                'coef_log_1mp': canonical.get(
                    'coef_log_1mp',
                    np.nan,
                ),
                'isotonic_points': len(
                    canonical.get(
                        'x_thresholds',
                        [],
                    )
                ),
            })

            cal_rows.append({
                'market': market,
                'side': s2,
                'calibration_role': 'derived_complement',
                'method': 'complement',
                'formula': (
                    'p_derived = '
                    '1 - p_canonical'
                ),
                'intercept': np.nan,
                'slope': np.nan,
                'coef_log_p': np.nan,
                'coef_log_1mp': np.nan,
                'isotonic_points': 0,
            })

        else:
            for (
                side,
                cal,
            ) in [
                (
                    s1,
                    fitted[
                        'cal1'
                    ],
                ),
                (
                    s2,
                    fitted[
                        'cal2'
                    ],
                ),
            ]:
                cal_rows.append({
                    'market': market,
                    'side': side,
                    'calibration_role': 'independent',
                    'method': cal.get(
                        'method',
                        'raw',
                    ),
                    'formula': calibrator_formula(
                        cal
                    ),
                    'intercept': cal.get(
                        'intercept',
                        np.nan,
                    ),
                    'slope': cal.get(
                        'slope',
                        np.nan,
                    ),
                    'coef_log_p': cal.get(
                        'coef_log_p',
                        np.nan,
                    ),
                    'coef_log_1mp': cal.get(
                        'coef_log_1mp',
                        np.nan,
                    ),
                    'isotonic_points': len(
                        cal.get(
                            'x_thresholds',
                            [],
                        )
                    ),
                })

    return (
        pd.DataFrame(
            action_rows
        ),
        pd.DataFrame(
            std_rows
        ),
        pd.DataFrame(
            cal_rows
        ),
    )


def build_final_recommendations(
    settings: dict[
        str,
        float,
    ],
    frozen_candidates: dict[
        str,
        dict[
            str,
            Any,
        ],
    ],
    lockbox_decisions: pd.DataFrame,
    production_actions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    decisions = (
        lockbox_decisions
        .set_index(
            'market'
        )
    )

    def evidence_for(
        market_name: str,
    ) -> str:
        d = decisions.loc[
            market_name
        ]

        return (
            'WHOLE FROZEN MARKET: '
            f"bets {int(d['candidate_bets'])} "
            f"vs {int(d['current_bets'])}; "
            'profit '
            f"{d['candidate_profit_units']:.4f} "
            'vs '
            f"{d['current_profit_units']:.4f}; "
            'ROI '
            f"{d['candidate_roi']:.5f} "
            'vs '
            f"{d['current_roi']:.5f}; "
            'logloss '
            f"{d['candidate_prob_log_loss']:.5f} "
            'vs '
            f"{d['current_prob_log_loss']:.5f}; "
            'Brier '
            f"{d['candidate_prob_brier']:.5f} "
            'vs '
            f"{d['current_prob_brier']:.5f}; "
            'validated='
            f"{bool(d['market_validated'])}"
        )

    def market_pass(
        market_name: str,
    ) -> bool:
        return bool(
            decisions.loc[
                market_name,
                'market_validated',
            ]
        )

    market = 'moneyline'

    frozen = frozen_candidates[
        market
    ]

    chosen: pd.Series = frozen[
        'chosen'
    ]

    passed = market_pass(
        market
    )

    status_change = (
        'FROZEN_MARKET_CHANGE_VALIDATED'
        if passed
        else 'KEEP_CURRENT_MARKET_LOCKBOX_FAILED'
    )

    status_same = (
        'NO_CHANGE_WITHIN_VALIDATED_MARKET'
        if passed
        else 'KEEP_CURRENT_MARKET_LOCKBOX_FAILED'
    )

    rows.append({
        'market': market,
        'setting': 'ML_EDGE',
        'current_value': settings[
            'ML_EDGE'
        ],
        'development_selected': float(
            frozen[
                'shared_edge'
            ]
        ),
        'production_refit': float(
            frozen[
                'shared_edge'
            ]
        ),
        'final_recommendation': (
            float(
                frozen[
                    'shared_edge'
                ]
            )
            if passed
            else settings[
                'ML_EDGE'
            ]
        ),
        'status': (
            status_change
            if float(
                frozen[
                    'shared_edge'
                ]
            ) != settings[
                'ML_EDGE'
            ]
            else status_same
        ),
        'requires_pipeline_code_change': False,
        'evidence': evidence_for(
            market
        ),
    })

    for side in market_sides(
        market
    ):
        method = str(
            chosen[
                f'calibration_{side.lower()}'
            ]
        )

        changed = (
            method != 'raw'
        )

        rows.append({
            'market': market,
            'setting': (
                f'{side}_CALIBRATION'
            ),
            'current_value': 'raw',
            'development_selected': method,
            'production_refit': method,
            'final_recommendation': (
                method
                if passed
                else 'raw'
            ),
            'status': (
                status_change
                if changed
                and passed
                else (
                    status_same
                    if not changed
                    and passed
                    else 'KEEP_RAW_MARKET_LOCKBOX_FAILED'
                )
            ),
            'requires_pipeline_code_change': bool(
                changed
                and passed
            ),
            'evidence': evidence_for(
                market
            ),
        })

    for (
        market,
        bias_setting,
        std_setting,
        shared_edge_setting,
    ) in [
        (
            'spread',
            'MARGIN_BIAS',
            'SPREAD_STD',
            'SPREAD_EDGE',
        ),
        (
            'total',
            'TOTAL_BIAS',
            'TOTAL_STD',
            'TOTAL_EDGE',
        ),
    ]:
        frozen = frozen_candidates[
            market
        ]

        chosen = frozen[
            'chosen'
        ]

        passed = market_pass(
            market
        )

        status_change = (
            'FROZEN_MARKET_CHANGE_VALIDATED'
            if passed
            else 'KEEP_CURRENT_MARKET_LOCKBOX_FAILED'
        )

        status_same = (
            'NO_CHANGE_WITHIN_VALIDATED_MARKET'
            if passed
            else 'KEEP_CURRENT_MARKET_LOCKBOX_FAILED'
        )

        pa = (
            production_actions[
                production_actions[
                    'market'
                ]
                == market
            ]
        )

        bias_row = pa[
            pa[
                'component'
            ]
            == 'BIAS'
        ].iloc[0]

        std_row = pa[
            pa[
                'component'
            ]
            == 'STD_MODE'
        ].iloc[0]

        bias_mode = str(
            chosen[
                'bias_strategy'
            ]
        )

        prod_bias = float(
            bias_row[
                'production_value'
            ]
        )

        rows.append({
            'market': market,
            'setting': bias_setting,
            'current_value': settings[
                bias_setting
            ],
            'development_selected': bias_mode,
            'production_refit': prod_bias,
            'final_recommendation': (
                prod_bias
                if passed
                else settings[
                    bias_setting
                ]
            ),
            'status': status_change,
            'requires_pipeline_code_change': bool(
                passed
                and bias_mode.startswith(
                    'rolling_'
                )
            ),
            'evidence': evidence_for(
                market
            ),
        })

        std_mode = str(
            chosen[
                'std_mode'
            ]
        )

        prod_std_raw = (
            pd.to_numeric(
                pd.Series([
                    std_row[
                        'production_value'
                    ]
                ]),
                errors='coerce',
            )
            .iloc[0]
        )

        prod_std = (
            float(
                prod_std_raw
            )
            if np.isfinite(
                prod_std_raw
            )
            else np.nan
        )

        if std_mode == 'fixed':
            production_refit_std: Any = (
                prod_std
            )

            final_std: Any = (
                prod_std
                if passed
                else settings[
                    std_setting
                ]
            )

        else:
            production_refit_std = (
                f'{std_mode}: see '
                'FINAL_STD_RANGES.csv'
            )

            final_std = (
                production_refit_std
                if passed
                else settings[
                    std_setting
                ]
            )

        rows.append({
            'market': market,
            'setting': std_setting,
            'current_value': settings[
                std_setting
            ],
            'development_selected': std_mode,
            'production_refit': production_refit_std,
            'final_recommendation': final_std,
            'status': status_change,
            'requires_pipeline_code_change': bool(
                passed
                and std_mode
                != 'fixed'
            ),
            'evidence': evidence_for(
                market
            ),
        })

        (
            s1,
            s2,
        ) = market_sides(
            market
        )

        canonical_method = str(
            chosen[
                f'calibration_{s1.lower()}'
            ]
        )

        canonical_changed = (
            canonical_method
            != 'raw'
        )

        rows.append({
            'market': market,
            'setting': (
                f'{s1}_CALIBRATION'
            ),
            'current_value': 'raw',
            'development_selected': canonical_method,
            'production_refit': canonical_method,
            'final_recommendation': (
                canonical_method
                if passed
                else 'raw'
            ),
            'status': (
                status_change
                if canonical_changed
                and passed
                else (
                    status_same
                    if not canonical_changed
                    and passed
                    else 'KEEP_RAW_MARKET_LOCKBOX_FAILED'
                )
            ),
            'requires_pipeline_code_change': bool(
                canonical_changed
                and passed
            ),
            'evidence': evidence_for(
                market
            ),
        })

        rows.append({
            'market': market,
            'setting': (
                f'{s2}_CALIBRATION'
            ),
            'current_value': 'derived_complement',
            'development_selected': 'complement',
            'production_refit': (
                'p_derived = '
                '1 - p_canonical'
            ),
            'final_recommendation': (
                'p_derived = '
                '1 - p_canonical'
                if passed
                else 'derived_complement'
            ),
            'status': (
                status_same
                if passed
                else 'KEEP_CURRENT_MARKET_LOCKBOX_FAILED'
            ),
            'requires_pipeline_code_change': False,
            'evidence': evidence_for(
                market
            ),
        })

        if (
            frozen[
                'edge_mode'
            ]
            == 'shared'
        ):
            dev_edge = float(
                frozen[
                    'shared_edge'
                ]
            )

            rows.append({
                'market': market,
                'setting': shared_edge_setting,
                'current_value': settings[
                    shared_edge_setting
                ],
                'development_selected': dev_edge,
                'production_refit': dev_edge,
                'final_recommendation': (
                    dev_edge
                    if passed
                    else settings[
                        shared_edge_setting
                    ]
                ),
                'status': (
                    status_change
                    if dev_edge
                    != settings[
                        shared_edge_setting
                    ]
                    else status_same
                ),
                'requires_pipeline_code_change': False,
                'evidence': evidence_for(
                    market
                ),
            })

        else:
            e1 = float(
                frozen[
                    'edge_side1'
                ]
            )

            e2 = float(
                frozen[
                    'edge_side2'
                ]
            )

            rows.append({
                'market': market,
                'setting': (
                    f'{market.upper()}_EDGE_MODE'
                ),
                'current_value': 'shared',
                'development_selected': 'separate_by_side',
                'production_refit': 'separate_by_side',
                'final_recommendation': (
                    'separate_by_side'
                    if passed
                    else 'shared'
                ),
                'status': status_change,
                'requires_pipeline_code_change': bool(
                    passed
                ),
                'evidence': evidence_for(
                    market
                ),
            })

            for (
                side,
                edge_value,
            ) in [
                (
                    s1,
                    e1,
                ),
                (
                    s2,
                    e2,
                ),
            ]:
                rows.append({
                    'market': market,
                    'setting': (
                        f'{side}_EDGE'
                    ),
                    'current_value': settings[
                        shared_edge_setting
                    ],
                    'development_selected': edge_value,
                    'production_refit': edge_value,
                    'final_recommendation': (
                        edge_value
                        if passed
                        else settings[
                            shared_edge_setting
                        ]
                    ),
                    'status': status_change,
                    'requires_pipeline_code_change': bool(
                        passed
                    ),
                    'evidence': evidence_for(
                        market
                    ),
                })

    return pd.DataFrame(
        rows
    )


def isotonic_knots_table(
    full_df: pd.DataFrame,
    chosen_by_market: dict[
        str,
        pd.Series,
    ],
) -> pd.DataFrame:
    rows = []

    for (
        market,
        chosen,
    ) in chosen_by_market.items():
        fitted = fit_selected_config(
            full_df,
            market,
            chosen,
        )

        (
            s1,
            s2,
        ) = market_sides(
            market
        )

        calibrators = [
            (
                s1,
                fitted[
                    'cal1'
                ],
            )
        ]

        if not uses_complementary_calibration(
            market
        ):
            calibrators.append(
                (
                    s2,
                    fitted[
                        'cal2'
                    ],
                )
            )

        for (
            side,
            cal,
        ) in calibrators:
            if cal.get(
                'method'
            ) != 'isotonic':
                continue

            xs = cal.get(
                'x_thresholds',
                [],
            )

            ys = cal.get(
                'y_thresholds',
                [],
            )

            for (
                i,
                (
                    x,
                    y,
                ),
            ) in enumerate(
                zip(
                    xs,
                    ys,
                )
            ):
                rows.append({
                    'market': market,
                    'side': side,
                    'knot': i,
                    'raw_probability': x,
                    'adjusted_probability': y,
                })

    return pd.DataFrame(
        rows
    )


def dataframe_text(
    df: pd.DataFrame,
    max_rows: int = 100,
) -> str:
    if (
        df is None
        or df.empty
    ):
        return 'No rows.'

    x = df.head(
        max_rows
    ).copy()

    return x.to_string(
        index=False,
        float_format=(
            lambda v:
            f'{v:.5f}'
        ),
    )


def write_report(
    path: Path,
    league: str,
    input_file: Path,
    markets_file: Path,
    selection_policies: dict[
        str,
        MarketSelectionPolicy,
    ],
    full_df: pd.DataFrame,
    dev: pd.DataFrame,
    lockbox: pd.DataFrame,
    folds: list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
        ]
    ],
    bias_summaries: dict[
        str,
        pd.DataFrame,
    ],
    bias_stress: dict[
        str,
        pd.DataFrame,
    ],
    std_summaries: dict[
        str,
        pd.DataFrame,
    ],
    std_stress: dict[
        str,
        pd.DataFrame,
    ],
    cal_summaries: dict[
        str,
        pd.DataFrame,
    ],
    frozen_candidates: dict[
        str,
        dict[
            str,
            Any,
        ],
    ],
    stress_by_market: dict[
        str,
        pd.DataFrame,
    ],
    lockbox_summary: pd.DataFrame,
    lockbox_decisions: pd.DataFrame,
    split_edge_summary: pd.DataFrame,
    final_recommendations: pd.DataFrame,
    production_actions: pd.DataFrame,
    production_std: pd.DataFrame,
    production_cal: pd.DataFrame,
    stress_reps: int,
    output_files: list[
        Path
    ],
) -> None:
    lines: list[
        str
    ] = []

    line = (
        '='
        * 110
    )

    lines += [
        line,
        (
            f'{league} FINAL MASTER '
            'BASKETBALL PIPELINE TEST'
        ),
        line,
        f'Input: {input_file}',
        (
            'Production market-selection '
            f'config: {markets_file}'
        ),
        f'Full rows: {len(full_df):,}',
        (
            'Full date range: '
            f"{full_df['_date'].min().date()} "
            'through '
            f"{full_df['_date'].max().date()}"
        ),
        f'Development rows: {len(dev):,}',
        (
            'Untouched lockbox rows: '
            f'{len(lockbox):,}'
        ),
        (
            'Chronological development '
            f'OOS folds: {len(folds)}'
        ),
        (
            'Stress scenarios: '
            f'{stress_reps:,}'
        ),
        '',
        'CORRECTED VALIDATION DESIGN:',
        (
            '- Development/OOS data selects '
            'BIAS, STD, calibration, EDGE '
            'mode, and EDGE value(s).'
        ),
        (
            '- One complete candidate is '
            'frozen for each market BEFORE '
            'the lockbox is touched.'
        ),
        (
            '- The lockbox evaluates each '
            'complete frozen candidate '
            'exactly once against the '
            'complete current pipeline.'
        ),
        (
            '- No component is accepted/'
            'rejected/swapped after seeing '
            'lockbox results.'
        ),
        (
            '- A market change passes only '
            'if lockbox log loss, Brier, '
            'total profit, and ROI are all '
            'no worse than current.'
        ),
        '',
        (
            'EDGE uses the real pipeline '
            'formula: probability * '
            'decimal_odds - 1.'
        ),
        (
            'Side selection is read from '
            'docs/win/basketball/config/'
            'markets.yaml.'
        ),
        (
            'all_qualifying keeps every side '
            'clearing the frozen EDGE policy.'
        ),
        (
            'pick_one keeps one qualifying '
            'side using that market\'s '
            'configured pick_preference.'
        ),
        (
            'Stored predictions with '
            'bias_applied=1 are first '
            'reversed back to pre-bias '
            'model values.'
        ),
        '',
        'PRODUCTION MARKET-SELECTION POLICY:',
        dataframe_text(
            market_selection_policy_table(
                selection_policies,
                markets_file,
            )
        ),
    ]

    lines += [
        '',
        line,
        (
            '1. DEVELOPMENT-FROZEN COMPLETE '
            'MARKET CANDIDATES'
        ),
        line,
    ]

    frozen_rows = []

    for market in [
        'moneyline',
        'spread',
        'total',
    ]:
        f = frozen_candidates[
            market
        ]

        r: pd.Series = f[
            'chosen'
        ]

        (
            s1,
            s2,
        ) = market_sides(
            market
        )

        policy = (
            selection_policies[
                market
            ]
        )

        row = {
            'market': market,
            'selection_mode': (
                policy.selection_mode
            ),
            'pick_preference_metric': (
                policy.preference_metric
            ),
            'pick_preference_direction': (
                policy.preference_direction
            ),
            'bias_strategy': r[
                'bias_strategy'
            ],
            'std_mode': r[
                'std_mode'
            ],
            f'cal_{s1}': r[
                f'calibration_{s1.lower()}'
            ],
            f'cal_{s2}': r[
                f'calibration_{s2.lower()}'
            ],
            'edge_mode': f[
                'edge_mode'
            ],
            'shared_edge': f.get(
                'shared_edge'
            ),
            f'{s1}_edge': f.get(
                'edge_side1'
            ),
            f'{s2}_edge': f.get(
                'edge_side2'
            ),
            'joint_dev_oos_bets': r[
                'oos_bets'
            ],
            'joint_dev_oos_profit': r[
                'oos_profit_units'
            ],
            'joint_dev_oos_roi': r[
                'oos_roi'
            ],
            'positive_fold_rate': r[
                'positive_fold_rate'
            ],
        }

        frozen_rows.append(
            row
        )

    lines.append(
        dataframe_text(
            pd.DataFrame(
                frozen_rows
            )
        )
    )

    lines += [
        '',
        line,
        (
            '2. UNTOUCHED LOCKBOX: FROZEN '
            'CANDIDATE VS CURRENT PIPELINE'
        ),
        line,
        dataframe_text(
            lockbox_summary
        ),
        '',
        line,
        '3. WHOLE-MARKET LOCKBOX PASS / FAIL',
        line,
        dataframe_text(
            lockbox_decisions
        ),
        '',
        line,
        '4. BIAS: NONE VS FIXED VS ROLLING',
        line,
    ]

    for market in [
        'spread',
        'total',
    ]:
        lines += [
            (
                f'\n{market.upper()} '
                'BIAS OOS SUMMARY'
            ),
            dataframe_text(
                bias_summaries[
                    market
                ]
            ),
            (
                f'\n{market.upper()} '
                'BIAS STRESS'
            ),
            dataframe_text(
                bias_stress[
                    market
                ]
            ),
        ]

    lines += [
        '',
        line,
        (
            '5. STD: ONE FIXED VALUE VS '
            'RANGE-SPECIFIC VALUES'
        ),
        line,
    ]

    for market in [
        'spread',
        'total',
    ]:
        lines += [
            (
                f'\n{market.upper()} '
                'STD OOS SUMMARY'
            ),
            dataframe_text(
                std_summaries[
                    market
                ]
            ),
            (
                f'\n{market.upper()} '
                'STD STRESS'
            ),
            dataframe_text(
                std_stress[
                    market
                ]
            ),
        ]

    lines += [
        '',
        line,
        (
            '6. CALIBRATION METHODS '
            '(DEVELOPMENT OOS ONLY)'
        ),
        line,
    ]

    for market in [
        'moneyline',
        'spread',
        'total',
    ]:
        lines += [
            f'\n{market.upper()}',
            dataframe_text(
                cal_summaries[
                    market
                ]
            ),
        ]

    lines += [
        '',
        line,
        '7. JOINT CONFIGURATION STRESS TESTS',
        line,
    ]

    for market in [
        'moneyline',
        'spread',
        'total',
    ]:
        lines += [
            (
                f'\n{market.upper()} '
                'TOP CONFIGURATIONS'
            ),
            dataframe_text(
                stress_by_market[
                    market
                ],
                25,
            ),
        ]

    lines += [
        '',
        line,
        (
            '8. SHARED VS SIDE-SPECIFIC EDGE '
            '(DEVELOPMENT SELECTION ONLY)'
        ),
        line,
        (
            'The frozen_edge_mode column is '
            'selected without using lockbox '
            'results.'
        ),
        dataframe_text(
            split_edge_summary
        ),
        '',
        line,
        '9. FINAL ACTIONABLE RECOMMENDATIONS',
        line,
        (
            'FINAL_ACTIONS follows the whole-'
            'market decision. A failed market '
            'keeps the complete current pipeline.'
        ),
        (
            'A passed market keeps the complete '
            'frozen candidate; no post-lockbox '
            'component swapping is allowed.'
        ),
        '',
        dataframe_text(
            final_recommendations,
            100,
        ),
        '',
        'PRODUCTION REFIT CANDIDATES / METHODS',
        (
            'Full-history refits are shown '
            'for audit. Use FINAL_ACTIONS.csv '
            'for actual implementation '
            'decisions.'
        ),
        dataframe_text(
            production_actions
        ),
        '',
        'STD RANGE TABLE',
        dataframe_text(
            production_std
        ),
        '',
        'CALIBRATION FORMULAS',
        dataframe_text(
            production_cal
        ),
        '',
        line,
        '10. OUTPUT FILES',
        line,
    ]

    for p in output_files:
        lines.append(
            str(
                p
            )
        )

    path.write_text(
        '\n'.join(
            lines
        )
        + '\n',
        encoding='utf-8',
    )





# =============================================================================
# STEP 18: WNBA MARKET-BAND REVALIDATION FROM FROZEN STEP 15 PREDICTIONS
# =============================================================================

STEP18_WNBA_ROOT = Path(
    'docs/win/basketball/backtest/model_validation/sdv_ensemble/wnba'
)
STEP18_WNBA_FROZEN = (
    STEP18_WNBA_ROOT / 'sdv' / '2025_common_games.csv'
)
WNBA_MARKET_BANDS_REPORT = (
    STEP18_WNBA_ROOT / 'WNBA_MARKET_BANDS_REPORT.txt'
)


def _step18_norm_team(value: Any) -> str:
    text = '' if value is None else str(value)
    return ' '.join(
        ''.join(
            ch.lower() if ch.isalnum() else ' '
            for ch in text
        ).split()
    )


def _step18_norm_date(value: Any) -> str:
    text = '' if value is None else str(value).strip()
    text = text[:10].replace('_', '-').replace('/', '-')
    try:
        return pd.Timestamp(text).strftime('%Y-%m-%d')
    except (TypeError, ValueError, OverflowError):
        return ''


def _step18_match_key(row: pd.Series) -> str:
    return '|'.join([
        _step18_norm_date(row.get('game_date')),
        _step18_norm_team(row.get('home_team')),
        _step18_norm_team(row.get('away_team')),
    ])


def _step18_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _step18_build_source_frame(
    input_file: Path,
    source: str,
    internal_season: int,
    temp_dir: Path,
) -> pd.DataFrame:
    if source not in MODEL_SOURCES:
        raise ValueError(f'Unsupported Step 18 model source={source!r}')

    if internal_season != 2025:
        raise ValueError(
            'Step 18 WNBA market-band validation requires internal season 2025.'
        )

    if not STEP18_WNBA_FROZEN.exists():
        raise FileNotFoundError(
            f'Missing frozen Step 15 WNBA predictions: {STEP18_WNBA_FROZEN}'
        )

    base = pd.read_csv(input_file)
    frozen = pd.read_csv(STEP18_WNBA_FROZEN)

    required_base = {
        'game_date', 'game_id', 'home_team', 'away_team',
        'home_score', 'away_score',
        'home_spread', 'away_spread', 'total',
        'home_dk_moneyline_decimal', 'away_dk_moneyline_decimal',
        'home_dk_spread_decimal', 'away_dk_spread_decimal',
        'dk_total_over_decimal', 'dk_total_under_decimal',
    }
    missing_base = sorted(required_base - set(base.columns))
    if missing_base:
        raise ValueError(
            f'{input_file}: missing Step 18 sportsbook/result columns={missing_base}'
        )

    required_frozen = {
        'game_date', 'home_team', 'away_team',
        'home_score', 'away_score',
        'dratings_margin', 'dratings_total',
        'sdv_margin', 'sdv_total',
        'ensemble_margin', 'ensemble_total',
    }
    missing_frozen = sorted(required_frozen - set(frozen.columns))
    if missing_frozen:
        raise ValueError(
            f'{STEP18_WNBA_FROZEN}: missing frozen columns={missing_frozen}'
        )

    base = base.copy()
    frozen = frozen.copy()
    base['_step18_key'] = base.apply(_step18_match_key, axis=1)
    frozen['_step18_key'] = frozen.apply(_step18_match_key, axis=1)

    for label, frame in (('base', base), ('frozen', frozen)):
        blank = frame['_step18_key'].str.startswith('|') | frame['_step18_key'].str.endswith('|')
        if bool(blank.any()):
            keys = frame.loc[blank, '_step18_key'].tolist()
            raise RuntimeError(
                f'Step 18 {label} data contains incomplete match keys={keys}'
            )

        dupes = sorted(
            frame.loc[
                frame['_step18_key'].duplicated(keep=False),
                '_step18_key',
            ].unique().tolist()
        )
        if dupes:
            raise RuntimeError(
                f'Step 18 {label} data contains duplicate game keys={dupes}'
            )

    frozen_keep = [
        '_step18_key',
        'home_score', 'away_score',
        'dratings_margin', 'dratings_total',
        'sdv_margin', 'sdv_total',
        'ensemble_margin', 'ensemble_total',
    ]
    merged = base.merge(
        frozen[frozen_keep],
        on='_step18_key',
        how='inner',
        validate='one_to_one',
        suffixes=('', '_step15'),
    )

    missing_base_keys = sorted(set(base['_step18_key']) - set(merged['_step18_key']))
    missing_frozen_keys = sorted(
        set(frozen['_step18_key']) - set(merged['_step18_key'])
    )
    if missing_base_keys or missing_frozen_keys:
        raise RuntimeError(
            'Step 18 frozen/base coverage mismatch. '
            f'base_without_step15={missing_base_keys} '
            f'step15_without_base={missing_frozen_keys}'
        )

    if len(merged) != len(base) or len(merged) != len(frozen):
        raise RuntimeError(
            'Step 18 requires the same complete WNBA common sample. '
            f'base={len(base)} frozen={len(frozen)} matched={len(merged)}'
        )

    for score in ('home_score', 'away_score'):
        left = pd.to_numeric(merged[score], errors='coerce')
        right = pd.to_numeric(merged[f'{score}_step15'], errors='coerce')
        bad = ~np.isclose(left, right, equal_nan=False)
        if bool(bad.any()):
            keys = merged.loc[bad, '_step18_key'].tolist()
            raise RuntimeError(
                f'Step 18 score mismatch for {score}; game_keys={keys}'
            )

    margin_col = {
        'dratings': 'dratings_margin',
        'sdv': 'sdv_margin',
        'ensemble': 'ensemble_margin',
    }[source]
    total_col = {
        'dratings': 'dratings_total',
        'sdv': 'sdv_total',
        'ensemble': 'ensemble_total',
    }[source]

    margin = pd.to_numeric(merged[margin_col], errors='coerce')
    projected_total = pd.to_numeric(merged[total_col], errors='coerce')

    bad_projection = ~(np.isfinite(margin) & np.isfinite(projected_total))
    if bool(bad_projection.any()):
        keys = merged.loc[bad_projection, '_step18_key'].tolist()
        raise RuntimeError(
            f'{source}: non-finite frozen Step 15 projections game_keys={keys}'
        )

    merged['home_projected_points'] = (projected_total + margin) / 2.0
    merged['away_projected_points'] = (projected_total - margin) / 2.0
    merged['total_projected_points'] = projected_total

    # Required by the existing master-test loader. Step 18 probabilities are
    # rebuilt chronologically from the frozen source margins, so these values
    # are deliberately not used by the Step 18 path.
    merged['home_prob'] = 0.5
    merged['away_prob'] = 0.5

    # Step 15 projections are treated as the frozen source output; no legacy
    # historical bias is reversed or re-applied in this Step 18 path.
    merged['bias_applied'] = 0
    merged['model_source'] = source

    if source == 'dratings':
        merged['model_version'] = 'step15_frozen_dratings_2025'
        merged['feature_version'] = ''
        merged['ensemble_version'] = ''
    elif source == 'sdv':
        merged['model_version'] = 'sdv_model_v1'
        merged['feature_version'] = 'sdv_model_v1_pit_v2'
        merged['ensemble_version'] = ''
    else:
        merged['model_version'] = 'dratings_sdv_ensemble_v1_50_50'
        merged['feature_version'] = 'sdv_model_v1_pit_v2'
        merged['ensemble_version'] = 'dratings_sdv_ensemble_v1_50_50'

    drop_cols = [
        c for c in merged.columns
        if c.endswith('_step15') or c == '_step18_key'
    ]
    merged = merged.drop(columns=drop_cols)

    temp_file = temp_dir / f'2025_WNBA_{source}_STEP18.csv'
    merged.to_csv(temp_file, index=False)

    return load_data(
        temp_file,
        'WNBA',
        source,
        internal_season,
    )


def _step18_residual_parameters(
    train: pd.DataFrame,
    market: str,
) -> tuple[float, float]:
    if market in {'moneyline', 'spread'}:
        prediction = pd.to_numeric(train['raw_margin'], errors='coerce')
        actual = pd.to_numeric(train['actual_margin'], errors='coerce')
    elif market == 'total':
        prediction = pd.to_numeric(train['raw_total'], errors='coerce')
        actual = pd.to_numeric(train['actual_total_calc'], errors='coerce')
    else:
        raise ValueError(f'Unknown Step 18 market={market!r}')

    residual = (actual - prediction).to_numpy(float)
    residual = residual[np.isfinite(residual)]
    if len(residual) < 30:
        raise RuntimeError(
            f'Step 18 {market}: insufficient training residuals={len(residual)}'
        )

    mean = float(np.mean(residual))
    std = float(np.std(residual, ddof=1))
    std = float(np.clip(std, MIN_STD, MAX_STD))
    return mean, std


def _step18_probabilities(
    train: pd.DataFrame,
    target: pd.DataFrame,
    market: str,
) -> tuple[np.ndarray, np.ndarray]:
    residual_mean, residual_std = _step18_residual_parameters(train, market)

    if market == 'moneyline':
        mean_margin = target['raw_margin'].to_numpy(float) + residual_mean
        home = clip_prob(norm.cdf(mean_margin / residual_std))
        return home, 1.0 - home

    if market == 'spread':
        mean_margin = target['raw_margin'].to_numpy(float) + residual_mean
        sigma = np.repeat(residual_std, len(target))
        home = home_spread_probability(
            mean_margin,
            target['home_spread'].to_numpy(float),
            sigma,
        )
        return home, 1.0 - home

    mean_total = target['raw_total'].to_numpy(float) + residual_mean
    sigma = np.repeat(residual_std, len(target))
    over = over_probability(
        mean_total,
        target['total'].to_numpy(float),
        sigma,
    )
    return over, 1.0 - over


def _step18_decimal_to_american(value: Any) -> float:
    try:
        decimal = float(value)
    except (TypeError, ValueError, OverflowError):
        return np.nan
    if not np.isfinite(decimal) or decimal <= 1.0:
        return np.nan
    if decimal >= 2.0:
        return 100.0 * (decimal - 1.0)
    return -100.0 / (decimal - 1.0)


def _step18_flatten_bets(
    meta: pd.DataFrame,
    selected: pd.DataFrame,
    market: str,
    model_source: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    side1, side2 = market_sides(market)

    for position in range(len(selected)):
        src = selected.iloc[position]
        base = meta.iloc[position]

        for side_index, side in ((1, side1), (2, side2)):
            if not bool(src[f'side{side_index}_selected']):
                continue

            line = np.nan
            if market == 'spread':
                line = float(
                    base['home_spread']
                    if side_index == 1
                    else base['away_spread']
                )
            elif market == 'total':
                line = float(base['total'])

            decimal_odds = float(src[f'side{side_index}_odds'])
            rows.append({
                'model_source': model_source,
                'market': market,
                'side': side,
                'game_id': str(src['game_id']),
                'date': src['date'],
                'fold_id': int(src['fold_id']),
                'line': line,
                'odds': _step18_decimal_to_american(decimal_odds),
                'decimal_odds': decimal_odds,
                'ev': float(src[f'side{side_index}_ev']),
                'kelly': float(src[f'side{side_index}_kelly']),
                'model_prob': float(src[f'side{side_index}_prob']),
                'edge_vs_market': float(
                    src[f'side{side_index}_edge_vs_market']
                ),
                'result': float(src[f'side{side_index}_result']),
                'profit': float(src[f'side{side_index}_profit']),
            })

    return pd.DataFrame(rows)


def _step18_development_oos_bets(
    dev: pd.DataFrame,
    folds: list[tuple[int, np.ndarray, np.ndarray]],
    settings: dict[str, Any],
    selection_policies: dict[str, MarketSelectionPolicy],
    model_source: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for fold_id, train_idx, test_idx in folds:
        train = dev.iloc[train_idx]
        test = dev.iloc[test_idx].copy()
        test['fold_id'] = fold_id

        for market in ('moneyline', 'spread', 'total'):
            p1, p2 = _step18_probabilities(train, test, market)
            base = market_opportunities(test, market, p1, p2)
            selected = select_opportunities(
                base,
                selection_policies[market],
                edge_mode='shared',
                shared_edge=current_edge_for_market(market, settings),
            )
            flat = _step18_flatten_bets(
                test,
                selected,
                market,
                model_source,
            )
            if not flat.empty:
                frames.append(flat)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _step18_lockbox_bets(
    dev: pd.DataFrame,
    lockbox: pd.DataFrame,
    settings: dict[str, Any],
    selection_policies: dict[str, MarketSelectionPolicy],
    model_source: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for market in ('moneyline', 'spread', 'total'):
        p1, p2 = _step18_probabilities(dev, lockbox, market)
        meta = lockbox.copy()
        meta['fold_id'] = 999
        base = market_opportunities(meta, market, p1, p2)
        selected = select_opportunities(
            base,
            selection_policies[market],
            edge_mode='shared',
            shared_edge=current_edge_for_market(market, settings),
        )
        flat = _step18_flatten_bets(
            meta,
            selected,
            market,
            model_source,
        )
        if not flat.empty:
            frames.append(flat)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _step18_bet_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            'bets': 0,
            'profit_units': 0.0,
            'roi': np.nan,
            'positive_fold_rate': np.nan,
            'folds': 0,
        }

    profit = float(pd.to_numeric(df['profit'], errors='coerce').sum())
    bets = int(len(df))
    fold_profit = df.groupby('fold_id')['profit'].sum()
    return {
        'bets': bets,
        'profit_units': profit,
        'roi': profit / bets if bets else np.nan,
        'positive_fold_rate': (
            float((fold_profit > 0).mean())
            if len(fold_profit)
            else np.nan
        ),
        'folds': int(len(fold_profit)),
    }


def _step18_snap_band_value(metric: str, value: float) -> float:
    if metric == 'odds':
        return float(round(value / 25.0) * 25.0)
    if metric == 'line':
        return float(round(value * 2.0) / 2.0)
    if metric in {'ev', 'kelly'}:
        return float(round(value, 2))
    if metric == 'model_prob':
        return float(round(value / 0.025) * 0.025)
    if metric == 'edge_vs_market':
        return float(round(value))
    return float(value)


def _step18_candidate_intervals(
    df: pd.DataFrame,
    metric: str,
) -> list[tuple[float, float]]:
    values = pd.to_numeric(df[metric], errors='coerce').dropna()
    if len(values) < 12:
        return []

    quantiles = np.unique(np.quantile(
        values.to_numpy(float),
        [
            0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.33,
            0.50, 0.67, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
        ],
    ))
    cuts = sorted({
        _step18_snap_band_value(metric, float(v))
        for v in quantiles
    })
    if len(cuts) < 2:
        return []

    intervals: set[tuple[float, float]] = set()
    full_lo, full_hi = cuts[0], cuts[-1]

    for cut in cuts[1:-1]:
        intervals.add((cut, full_hi))
        intervals.add((full_lo, cut))

    for i in range(len(cuts) - 1):
        for j in range(i + 1, len(cuts)):
            lo, hi = cuts[i], cuts[j]
            if lo < hi:
                intervals.add((lo, hi))

    intervals.discard((full_lo, full_hi))
    return sorted(intervals)


def _step18_apply_bands(
    df: pd.DataFrame,
    bands: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    if df.empty or not bands:
        return df.copy()

    mask = np.ones(len(df), dtype=bool)
    for metric, (lo, hi) in bands.items():
        values = pd.to_numeric(df[metric], errors='coerce').to_numpy(float)
        mask &= (
            np.isfinite(values)
            & (values >= lo)
            & (values <= hi)
        )
    return df.loc[mask].copy()


def _step18_discover_side_bands(side_df: pd.DataFrame) -> dict[str, Any]:
    baseline = _step18_bet_metrics(side_df)
    bands: dict[str, tuple[float, float]] = {}

    if baseline['bets'] < 12:
        return {
            'bands': bands,
            'baseline': baseline,
            'development': baseline,
            'status': 'INSUFFICIENT_DEVELOPMENT_BETS',
        }

    dimensions = [
        'odds',
        'ev',
        'kelly',
        'model_prob',
        'edge_vs_market',
    ]
    if side_df['market'].iloc[0] in {'spread', 'total'}:
        dimensions.insert(1, 'line')

    current_metrics = baseline
    used: set[str] = set()

    # At most two independent filter dimensions may be frozen.
    for _ in range(2):
        best: tuple[Any, ...] | None = None

        for metric in dimensions:
            if metric in used:
                continue

            for lo, hi in _step18_candidate_intervals(side_df, metric):
                candidate_bands = dict(bands)
                candidate_bands[metric] = (lo, hi)
                candidate = _step18_apply_bands(side_df, candidate_bands)
                metrics = _step18_bet_metrics(candidate)

                min_bets = max(
                    12,
                    int(math.ceil(0.35 * baseline['bets'])),
                )
                if metrics['bets'] < min_bets:
                    continue

                minimum_folds = min(
                    3,
                    max(1, baseline['folds']),
                )
                if metrics['folds'] < minimum_folds:
                    continue

                current_roi = current_metrics['roi']
                roi_improvement = (
                    metrics['roi'] - current_roi
                    if (
                        np.isfinite(metrics['roi'])
                        and np.isfinite(current_roi)
                    )
                    else -np.inf
                )
                profit_improvement = (
                    metrics['profit_units']
                    - current_metrics['profit_units']
                )
                stable = (
                    not np.isfinite(metrics['positive_fold_rate'])
                    or metrics['positive_fold_rate'] >= 0.50
                )

                if (
                    profit_improvement < 0.50
                    or roi_improvement < 0.01
                    or not stable
                ):
                    continue

                key = (
                    profit_improvement,
                    (
                        metrics['positive_fold_rate']
                        if np.isfinite(metrics['positive_fold_rate'])
                        else -1.0
                    ),
                    roi_improvement,
                    metrics['bets'],
                )
                if best is None or key > best[0]:
                    best = (
                        key,
                        metric,
                        lo,
                        hi,
                        metrics,
                    )

        if best is None:
            break

        _, metric, lo, hi, current_metrics = best
        bands[metric] = (float(lo), float(hi))
        used.add(metric)

    return {
        'bands': bands,
        'baseline': baseline,
        'development': current_metrics,
        'status': (
            'CANDIDATE_FOUND'
            if bands
            else 'NO_CHANGE_CANDIDATE'
        ),
    }


def _step18_format_bands(
    bands: dict[str, tuple[float, float]],
) -> str:
    if not bands:
        return 'NO_CHANGE'
    return '; '.join(
        f'{metric}=[{lo:g},{hi:g}]'
        for metric, (lo, hi) in bands.items()
    )


def _step18_report_text(
    input_file: Path,
    markets_file: Path,
    split_summary: pd.DataFrame,
    candidates: pd.DataFrame,
    comparison: pd.DataFrame,
    markets_sha256: str,
    frozen_sha256: str,
) -> str:
    line = '=' * 100
    carried = comparison[
        (comparison['model_source'] == 'ensemble')
        & (comparison['lockbox_status'] == 'VALIDATED')
    ].copy()

    parts = [
        'WNBA MARKET BANDS REPORT',
        line,
        f'Historical sportsbook/results input: {input_file}',
        f'Frozen Step 15 projections: {STEP18_WNBA_FROZEN}',
        f'Frozen Step 15 sha256: {frozen_sha256}',
        f'Markets config (read only): {markets_file}',
        f'markets.yaml sha256: {markets_sha256}',
        'markets.yaml modified: NO',
        '',
        'VALIDATION CONTRACT',
        line,
        'Model sources: dratings, sdv, ensemble.',
        'All three sources use the identical chronological 85% development / 15% lockbox game split.',
        'The frozen Step 15 margin/total projections are never retrained or recomputed.',
        'Model probabilities are generated from each source frozen projection using residual mean/std fitted only on the applicable development training history.',
        'Candidate band discovery uses ENSEMBLE development/OOS bets only.',
        'DRatings and SDV are comparison-only and cannot choose or change candidate edges.',
        'Candidate bands are frozen before any lockbox evaluation.',
        'The untouched lockbox is evaluated once after candidates are frozen.',
        'No lockbox result is used to change a candidate.',
        '',
        'SPLIT / SOURCE COVERAGE',
        line,
        split_summary.to_string(index=False),
        '',
        'DEVELOPMENT/OOS CANDIDATES (ENSEMBLE ONLY)',
        line,
        candidates.to_string(index=False),
        '',
        'UNTOUCHED LOCKBOX: DRATINGS vs SDV vs ENSEMBLE',
        line,
        comparison.to_string(index=False),
        '',
        'CARRY FORWARD TO ITEM 21',
        line,
    ]

    if carried.empty:
        parts.append(
            'No WNBA market-band changes validated on the untouched ensemble lockbox.'
        )
    else:
        for _, row in carried.iterrows():
            parts.append(
                f"{row['market']} {row['side']}: {row['candidate_bands']}"
            )

    parts.extend([
        '',
        'Do not modify markets.yaml during this item.',
        '',
    ])
    return '\n'.join(parts)


def run_wnba_market_band_validation(
    input_file: Path,
    internal_season: int,
    settings: dict[str, Any],
    selection_policies: dict[str, MarketSelectionPolicy],
    markets_file: Path,
    output_path: Path,
) -> Path:
    if internal_season != 2025:
        raise ValueError(
            '--wnba-market-bands requires --season 2025'
        )

    before_hash = _step18_sha256(markets_file)
    frozen_hash = _step18_sha256(STEP18_WNBA_FROZEN)

    source_dev: dict[str, pd.DataFrame] = {}
    source_lockbox: dict[str, pd.DataFrame] = {}
    split_rows: list[dict[str, Any]] = []

    expected_dev_ids: list[str] | None = None
    expected_lock_ids: list[str] | None = None

    with tempfile.TemporaryDirectory(prefix='wnba_step18_') as temp_name:
        temp_dir = Path(temp_name)

        for source in MODEL_SOURCES:
            frame = _step18_build_source_frame(
                input_file,
                source,
                internal_season,
                temp_dir,
            )
            dev, lockbox = split_development_lockbox(
                frame,
                LOCKBOX_FRACTION,
            )

            dev_ids = dev['game_id'].astype(str).tolist()
            lock_ids = lockbox['game_id'].astype(str).tolist()

            if expected_dev_ids is None:
                expected_dev_ids = dev_ids
                expected_lock_ids = lock_ids
            elif (
                dev_ids != expected_dev_ids
                or lock_ids != expected_lock_ids
            ):
                raise RuntimeError(
                    f'{source}: development/lockbox game_id split '
                    'differs from the other model sources'
                )

            source_dev[source] = dev
            source_lockbox[source] = lockbox

            split_rows.append({
                'model_source': source,
                'full_games': len(frame),
                'development_games': len(dev),
                'lockbox_games': len(lockbox),
                'development_start': str(dev['_date'].min().date()),
                'development_end': str(dev['_date'].max().date()),
                'lockbox_start': str(lockbox['_date'].min().date()),
                'lockbox_end': str(lockbox['_date'].max().date()),
            })

    # One identical chronological fold layout for all three sources.
    folds = make_outer_folds(
        source_dev['ensemble'],
        TARGET_OUTER_FOLDS,
    )

    ensemble_dev_bets = _step18_development_oos_bets(
        source_dev['ensemble'],
        folds,
        settings,
        selection_policies,
        'ensemble',
    )
    if ensemble_dev_bets.empty:
        raise RuntimeError(
            'Ensemble WNBA development/OOS produced zero qualifying bets'
        )

    candidate_records: list[dict[str, Any]] = []
    frozen_bands: dict[
        tuple[str, str],
        dict[str, tuple[float, float]],
    ] = {}

    for market in ('moneyline', 'spread', 'total'):
        for side in market_sides(market):
            subset = ensemble_dev_bets[
                (ensemble_dev_bets['market'] == market)
                & (ensemble_dev_bets['side'] == side)
            ].copy()

            discovered = _step18_discover_side_bands(subset)
            bands = discovered['bands']
            frozen_bands[(market, side)] = bands
            baseline = discovered['baseline']
            candidate = discovered['development']

            candidate_records.append({
                'market': market,
                'side': side,
                'status': discovered['status'],
                'candidate_bands': _step18_format_bands(bands),
                'dev_baseline_bets': baseline['bets'],
                'dev_baseline_profit': baseline['profit_units'],
                'dev_baseline_roi': baseline['roi'],
                'dev_candidate_bets': candidate['bets'],
                'dev_candidate_profit': candidate['profit_units'],
                'dev_candidate_roi': candidate['roi'],
                'dev_candidate_positive_fold_rate': (
                    candidate['positive_fold_rate']
                ),
            })

    candidates = pd.DataFrame(candidate_records)

    comparison_records: list[dict[str, Any]] = []

    # Lockbox is touched only after every ensemble candidate is frozen.
    for source in MODEL_SOURCES:
        lock_bets = _step18_lockbox_bets(
            source_dev[source],
            source_lockbox[source],
            settings,
            selection_policies,
            source,
        )

        for market in ('moneyline', 'spread', 'total'):
            for side in market_sides(market):
                baseline_df = lock_bets[
                    (lock_bets['market'] == market)
                    & (lock_bets['side'] == side)
                ].copy()

                bands = frozen_bands[(market, side)]
                candidate_df = _step18_apply_bands(
                    baseline_df,
                    bands,
                )

                baseline = _step18_bet_metrics(baseline_df)
                candidate = _step18_bet_metrics(candidate_df)

                if not bands:
                    status = 'NO_CHANGE'
                elif candidate['bets'] < 5:
                    status = 'INSUFFICIENT_LOCKBOX_BETS'
                elif (
                    candidate['profit_units'] > 0
                    and candidate['profit_units'] >= baseline['profit_units']
                    and np.isfinite(candidate['roi'])
                    and (
                        not np.isfinite(baseline['roi'])
                        or candidate['roi'] >= baseline['roi']
                    )
                ):
                    status = 'VALIDATED'
                else:
                    status = 'REJECTED'

                comparison_records.append({
                    'model_source': source,
                    'market': market,
                    'side': side,
                    'candidate_bands': _step18_format_bands(bands),
                    'lockbox_status': status,
                    'baseline_bets': baseline['bets'],
                    'baseline_profit': baseline['profit_units'],
                    'baseline_roi': baseline['roi'],
                    'candidate_bets': candidate['bets'],
                    'candidate_profit': candidate['profit_units'],
                    'candidate_roi': candidate['roi'],
                })

    comparison = pd.DataFrame(comparison_records)
    split_summary = pd.DataFrame(split_rows)

    after_hash = _step18_sha256(markets_file)
    if after_hash != before_hash:
        raise RuntimeError(
            'Refusing Step 18 result: markets.yaml changed during validation: '
            f'{markets_file}'
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _step18_report_text(
            input_file,
            markets_file,
            split_summary,
            candidates,
            comparison,
            before_hash,
            frozen_hash,
        ),
        encoding='utf-8',
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Final master basketball '
            'pipeline historical validation test'
        )
    )

    parser.add_argument(
        '--league',
        default=LEAGUE,
        choices=[
            'NBA',
            'NCAAM',
            'WNBA',
        ],
    )

    parser.add_argument(
        '--input',
        default=str(
            INPUT_FILE
        ),
    )

    parser.add_argument(
        '--season',
        default=None,
        help=(
            'Internal basketball season '
            'start year. If omitted, it is '
            'read from an input filename '
            'like 2025_NBA.csv.'
        ),
    )

    parser.add_argument(
        '--model-source',
        choices=MODEL_SOURCES,
        default=None,
    )

    parser.add_argument(
        '--parity-rows',
        type=int,
        default=25,
    )

    parser.add_argument(
        '--markets-file',
        default=str(
            MARKETS_FILE
        ),
        help=(
            'Production selection config: '
            'docs/win/basketball/config/'
            'markets.yaml'
        ),
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help=(
            'Same logic, fewer bootstrap '
            'scenarios for code validation'
        ),
    )

    parser.add_argument(
        '--wnba-market-bands',
        action='store_true',
        help=(
            'Run Step 18 WNBA market-band revalidation using frozen '
            'Step 15 dratings/SDV/ensemble 2025 projections.'
        ),
    )

    parser.add_argument(
        '--market-bands-output',
        default=str(WNBA_MARKET_BANDS_REPORT),
        help='Step 18 WNBA market-band report path.',
    )

    args = parser.parse_args()

    league = (
        args.league.upper()
    )

    input_file = Path(
        args.input
    )

    settings = CURRENT_SETTINGS[
        league
    ]

    model_source = resolve_model_source(
        args.model_source,
        settings,
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    markets_file = (
        resolve_markets_file(
            Path(
                args.markets_file
            )
        )
    )

    selection_policies = (
        load_market_selection_policies(
            markets_file,
            league,
        )
    )

    stress_reps = 250 if args.quick else STRESS_REPS

    t0 = now_seconds()

    internal_season = (
        resolve_internal_season(
            input_file,
            league,
            args.season,
        )
    )

    season = str(
        internal_season
    )

    if args.wnba_market_bands:
        if league != 'WNBA':
            raise ValueError(
                '--wnba-market-bands requires --league WNBA'
            )

        report_path = run_wnba_market_band_validation(
            input_file=input_file,
            internal_season=internal_season,
            settings=settings,
            selection_policies=selection_policies,
            markets_file=markets_file,
            output_path=Path(args.market_bands_output),
        )

        progress(
            'WNBA market-band validation complete: '
            f'{report_path}'
        )
        return

    progress(
        f'Loading {league} data: '
        f'{input_file} | '
        f'internal_season='
        f'{internal_season}'
    )

    full_df = load_data(
        input_file,
        league,
        model_source,
        internal_season,
    )

    output_dir = (
        input_file.parent
    )

    prefix = make_prefix(
        league,
        season,
    )

    run_production_parity_test(
        full_df,
        league,
        settings,
        selection_policies,
        args.parity_rows,
    )

    (
        dev,
        lockbox,
    ) = split_development_lockbox(
        full_df,
        LOCKBOX_FRACTION,
    )

    folds = make_outer_folds(
        dev,
        TARGET_OUTER_FOLDS,
    )

    meta = oos_meta(
        dev,
        folds,
    )

    strategies = bias_strategy_names(
        len(
            dev
        )
    )

    progress(
        f'Rows={len(full_df):,}; '
        f'development={len(dev):,}; '
        'untouched lockbox='
        f'{len(lockbox):,}; '
        f'OOS folds={len(folds)}; '
        f'stress reps={stress_reps:,}'
    )

    progress(
        f'Model source: {model_source}'
    )

    progress(
        'Production market-selection '
        f'config: {markets_file}'
    )

    for market in [
        'moneyline',
        'spread',
        'total',
    ]:
        policy = selection_policies[
            market
        ]

        progress(
            f'  {market}: '
            'selection_mode='
            f'{policy.selection_mode}; '
            'pick_preference='
            f'{policy.preference_metric}/'
            f'{policy.preference_direction}'
        )

    output_files: list[
        Path
    ] = []

    selection_policy_path = (
        output_dir
        / (
            f'{prefix}_00_'
            'MARKET_SELECTION_POLICY.csv'
        )
    )

    output_files.append(
        save_csv(
            market_selection_policy_table(
                selection_policies,
                markets_file,
            ),
            selection_policy_path,
        )
    )

    progress(
        '[1/9] Bias: none vs fixed vs rolling...'
    )

    bias_details: dict[
        str,
        pd.DataFrame,
    ] = {}

    bias_summaries: dict[
        str,
        pd.DataFrame,
    ] = {}

    bias_stress: dict[
        str,
        pd.DataFrame,
    ] = {}

    selected_bias_strategy: dict[
        str,
        str,
    ] = {}

    for (
        market,
        current_bias,
    ) in [
        (
            'spread',
            settings[
                'MARGIN_BIAS'
            ],
        ),
        (
            'total',
            settings[
                'TOTAL_BIAS'
            ],
        ),
    ]:
        current_rule = (
            settings[
                'MARGIN_BIAS_RULE'
            ]
            if market == 'spread'
            else settings[
                'TOTAL_BIAS_RULE'
            ]
        )

        (
            detail,
            summary,
        ) = evaluate_bias_strategies(
            dev,
            folds,
            market,
            strategies,
            current_bias,
            current_rule=current_rule,
        )

        stress = stress_bias_strategies(
            detail,
            strategies,
            stress_reps,
            rng,
        )

        bias_details[
            market
        ] = detail

        bias_summaries[
            market
        ] = summary

        bias_stress[
            market
        ] = stress

        merged = (
            summary[
                summary[
                    'strategy'
                ].isin(
                    strategies
                )
            ]
            .merge(
                stress,
                on='strategy',
                how='left',
            )
        )

        selected = (
            merged
            .sort_values(
                [
                    'selection_frequency',
                    'oos_rmse',
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .iloc[0]
        )

        selected_bias_strategy[
            market
        ] = str(
            selected[
                'strategy'
            ]
        )

        output_files.append(
            save_csv(
                summary,
                output_dir
                / (
                    f'{prefix}_01_'
                    f'{market}_'
                    'bias_strategy_oos.csv'
                ),
            )
        )

        output_files.append(
            save_csv(
                stress,
                output_dir
                / (
                    f'{prefix}_01_'
                    f'{market}_'
                    'bias_strategy_stress.csv'
                ),
            )
        )

        output_files.append(
            save_csv(
                detail,
                output_dir
                / (
                    f'{prefix}_01_'
                    f'{market}_'
                    'bias_oos_detail.csv'
                ),
            )
        )

    progress(
        '[2/9] STD: fixed vs '
        'sportsbook-range-specific...'
    )

    std_details: dict[
        str,
        pd.DataFrame,
    ] = {}

    std_summaries: dict[
        str,
        pd.DataFrame,
    ] = {}

    std_stress: dict[
        str,
        pd.DataFrame,
    ] = {}

    selected_std_mode: dict[
        str,
        str,
    ] = {}

    for market in [
        'spread',
        'total',
    ]:
        (
            detail,
            summary,
        ) = evaluate_std_modes(
            dev,
            folds,
            market,
            selected_bias_strategy[
                market
            ],
        )

        stress = stress_std_modes(
            detail,
            STD_MODES,
            stress_reps,
            rng,
        )

        std_details[
            market
        ] = detail

        std_summaries[
            market
        ] = summary

        std_stress[
            market
        ] = stress

        fixed_nll = float(
            summary.loc[
                summary[
                    'std_mode'
                ]
                == 'fixed',
                'mean_residual_nll',
            ].iloc[0]
        )

        best = summary.iloc[
            0
        ]

        best_mode = str(
            best[
                'std_mode'
            ]
        )

        rel_improvement = (
            (
                fixed_nll
                - float(
                    best[
                        'mean_residual_nll'
                    ]
                )
            )
            / abs(
                fixed_nll
            )
            if fixed_nll
            else 0.0
        )

        if (
            best_mode != 'fixed'
            and rel_improvement
            < MIN_ADAPTIVE_STD_REL_NLL_IMPROVEMENT
        ):
            best_mode = 'fixed'

        selected_std_mode[
            market
        ] = best_mode

        output_files.append(
            save_csv(
                summary,
                output_dir
                / (
                    f'{prefix}_02_'
                    f'{market}_'
                    'std_mode_oos.csv'
                ),
            )
        )

        output_files.append(
            save_csv(
                stress,
                output_dir
                / (
                    f'{prefix}_02_'
                    f'{market}_'
                    'std_mode_stress.csv'
                ),
            )
        )

        output_files.append(
            save_csv(
                detail,
                output_dir
                / (
                    f'{prefix}_02_'
                    f'{market}_'
                    'std_oos_detail.csv'
                ),
            )
        )

    progress(
        '[3/9] Joint optimization: '
        'BIAS + STD + calibration + '
        'shared EDGE...'
    )

    rankings: dict[
        str,
        pd.DataFrame,
    ] = {}

    caches_by_market: dict[
        str,
        dict[
            str,
            dict[
                str,
                Any,
            ],
        ],
    ] = {}

    stress_by_market: dict[
        str,
        pd.DataFrame,
    ] = {}

    chosen_by_market: dict[
        str,
        pd.Series,
    ] = {}

    for market in [
        'moneyline',
        'spread',
        'total',
    ]:
        if market == 'moneyline':
            bset, sset = (
                [
                    'NA'
                ],
                [
                    'NA'
                ],
            )

        else:
            bset, sset = (
                strategies,
                STD_MODES,
            )

        (
            ranking,
            caches,
            files,
        ) = evaluate_joint_configs_for_market(
            dev,
            folds,
            meta,
            market,
            bset,
            sset,
            selection_policies[
                market
            ],
            output_dir,
            prefix,
        )

        rankings[
            market
        ] = ranking

        caches_by_market[
            market
        ] = caches

        output_files.extend(
            files
        )

        (
            stress,
            chosen,
        ) = stress_top_joint_configs(
            meta,
            market,
            ranking,
            caches,
            selection_policies[
                market
            ],
            stress_reps,
            rng,
        )

        stress_by_market[
            market
        ] = stress

        chosen_by_market[
            market
        ] = chosen

        output_files.append(
            save_csv(
                stress,
                output_dir
                / (
                    f'{prefix}_04_'
                    'joint_stress_'
                    f'{market}.csv'
                ),
            )
        )

    progress(
        '[4/9] Calibration: '
        'development OOS comparison '
        'by side...'
    )

    cal_summaries: dict[
        str,
        pd.DataFrame,
    ] = {}

    for market in [
        'moneyline',
        'spread',
        'total',
    ]:
        chosen = chosen_by_market[
            market
        ]

        cache = caches_by_market[
            market
        ][
            str(
                chosen[
                    'cache_key'
                ]
            )
        ]

        cal_summary = (
            calibration_method_summary(
                meta,
                market,
                cache,
            )
        )

        cal_summaries[
            market
        ] = cal_summary

        output_files.append(
            save_csv(
                cal_summary,
                output_dir
                / (
                    f'{prefix}_03_'
                    'calibration_oos_'
                    f'{market}.csv'
                ),
            )
        )

    progress(
        '[5/9] Freezing complete market '
        'candidates before lockbox...'
    )

    (
        frozen_candidates,
        split_edge_summary,
        freeze_files,
    ) = build_frozen_candidates(
        meta,
        chosen_by_market,
        caches_by_market,
        selection_policies,
        output_dir,
        prefix,
    )

    output_files.extend(
        freeze_files
    )

    for market in [
        'spread',
        'total',
    ]:
        frozen = frozen_candidates[
            market
        ]

        chosen = frozen[
            'chosen'
        ]

        cache = caches_by_market[
            market
        ][
            str(
                chosen[
                    'cache_key'
                ]
            )
        ]

        (
            s1,
            s2,
        ) = market_sides(
            market
        )

        cal1 = str(
            chosen[
                f'calibration_{s1.lower()}'
            ]
        )

        p1 = cache[
            'side1'
        ][
            cal1
        ]

        if uses_complementary_calibration(
            market
        ):
            p2 = cache[
                'side2'
            ][
                cal1
            ]

        else:
            cal2 = str(
                chosen[
                    f'calibration_{s2.lower()}'
                ]
            )

            p2 = cache[
                'side2'
            ][
                cal2
            ]

        seg = segment_betting_summary(
            meta,
            market,
            p1,
            p2,
            cache[
                'mean'
            ],
            selection_policy=(
                selection_policies[
                    market
                ]
            ),
            edge_mode=str(
                frozen[
                    'edge_mode'
                ]
            ),
            shared_edge=frozen.get(
                'shared_edge'
            ),
            edge_side1=frozen.get(
                'edge_side1'
            ),
            edge_side2=frozen.get(
                'edge_side2'
            ),
        )

        output_files.append(
            save_csv(
                seg,
                output_dir
                / (
                    f'{prefix}_07_'
                    f'{market}_'
                    'segment_analysis.csv'
                ),
            )
        )

    progress(
        '[6/9] One-time untouched '
        'lockbox validation of frozen '
        'candidates...'
    )

    lock_rows = []

    fitted_dev: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    lock_payloads: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    for market in [
        'moneyline',
        'spread',
        'total',
    ]:
        (
            summary,
            fitted,
            payload,
        ) = evaluate_frozen_candidate_on_lockbox(
            dev,
            lockbox,
            market,
            frozen_candidates[
                market
            ],
            settings,
            selection_policies[
                market
            ],
        )

        lock_rows.append(
            summary
        )

        fitted_dev[
            market
        ] = fitted

        lock_payloads[
            market
        ] = payload

        output_files.append(
            save_csv(
                payload[
                    'detail'
                ],
                output_dir
                / (
                    f'{prefix}_05_'
                    'lockbox_detail_'
                    f'{market}.csv'
                ),
            )
        )

    lockbox_summary = pd.concat(
        lock_rows,
        ignore_index=True,
    )

    lockbox_decisions = (
        lockbox_market_validation(
            lockbox_summary
        )
    )

    output_files.append(
        save_csv(
            lockbox_summary,
            output_dir
            / (
                f'{prefix}_05_'
                'lockbox_validation.csv'
            ),
        )
    )

    output_files.append(
        save_csv(
            lockbox_decisions,
            output_dir
            / (
                f'{prefix}_05_'
                'lockbox_market_decision.csv'
            ),
        )
    )

    progress(
        '[7/9] Full-history refit of '
        'frozen methods and final '
        'decisions...'
    )

    (
        production_actions,
        production_std,
        production_cal,
    ) = production_parameter_tables(
        full_df,
        frozen_candidates,
    )

    isotonic_knots = (
        isotonic_knots_table(
            full_df,
            chosen_by_market,
        )
    )

    decision_map = (
        lockbox_decisions
        .set_index(
            'market'
        )
    )

    for col in [
        'candidate_bets',
        'current_bets',
        'candidate_profit_units',
        'current_profit_units',
        'candidate_roi',
        'current_roi',
        'candidate_prob_log_loss',
        'current_prob_log_loss',
        'candidate_prob_brier',
        'current_prob_brier',
        'market_validated',
    ]:
        production_actions[
            f'lockbox_{col}'
        ] = (
            production_actions[
                'market'
            ].map(
                (
                    lambda m, c=col:
                    decision_map.loc[
                        m,
                        c,
                    ]
                )
            )
        )

    final_recommendations = (
        build_final_recommendations(
            settings,
            frozen_candidates,
            lockbox_decisions,
            production_actions,
        )
    )

    output_files.append(
        save_csv(
            production_actions,
            output_dir
            / (
                f'{prefix}_'
                'PRODUCTION_REFIT_CANDIDATES.csv'
            ),
        )
    )

    output_files.append(
        save_csv(
            production_std,
            output_dir
            / (
                f'{prefix}_'
                'FINAL_STD_RANGES.csv'
            ),
        )
    )

    output_files.append(
        save_csv(
            production_cal,
            output_dir
            / (
                f'{prefix}_'
                'FINAL_CALIBRATION_FORMULAS.csv'
            ),
        )
    )

    output_files.append(
        save_csv(
            final_recommendations,
            output_dir
            / (
                f'{prefix}_'
                'FINAL_ACTIONS.csv'
            ),
        )
    )

    if not isotonic_knots.empty:
        output_files.append(
            save_csv(
                isotonic_knots,
                output_dir
                / (
                    f'{prefix}_'
                    'FINAL_ISOTONIC_KNOTS.csv'
                ),
            )
        )

    progress(
        '[8/9] Writing final report...'
    )

    report_path = (
        output_dir
        / (
            f'{prefix}_REPORT.txt'
        )
    )

    write_report(
        report_path,
        league,
        input_file,
        markets_file,
        selection_policies,
        full_df,
        dev,
        lockbox,
        folds,
        bias_summaries,
        bias_stress,
        std_summaries,
        std_stress,
        cal_summaries,
        frozen_candidates,
        stress_by_market,
        lockbox_summary,
        lockbox_decisions,
        split_edge_summary,
        final_recommendations,
        production_actions,
        production_std,
        production_cal,
        stress_reps,
        output_files,
    )

    output_files.append(
        report_path
    )

    elapsed = (
        now_seconds()
        - t0
    )

    progress(
        '[9/9] Complete.'
    )

    progress(
        f'Report: {report_path}'
    )

    progress(
        'Lockbox decisions: '
        f"{output_dir / f'{prefix}_05_lockbox_market_decision.csv'}"
    )

    progress(
        'Final actions: '
        f"{output_dir / f'{prefix}_FINAL_ACTIONS.csv'}"
    )

    progress(
        f'Runtime: '
        f'{elapsed / 60.0:.2f} minutes'
    )


if __name__ == '__main__':
    main()
