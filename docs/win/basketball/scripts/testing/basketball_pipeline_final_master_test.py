#!/usr/bin/env python3
"""Production-settings launcher for the historical master validation test.

The full validation engine remains in basketball_pipeline_final_master_test_core.py.
This launcher replaces its stale hardcoded CURRENT_SETTINGS with values loaded from
model_config.yaml plus rolling_bias_state.yaml at execution time.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import yaml


ROOT = Path('docs/win/basketball')

MODEL_CONFIG = (
    ROOT
    / 'config/model_config.yaml'
)

ROLLING_STATE = (
    ROOT
    / 'config/rolling_bias_state.yaml'
)

CORE_PATH = (
    Path(__file__)
    .with_name(
        'basketball_pipeline_final_master_test_core.py'
    )
)

MODEL_SOURCES = (
    'dratings',
    'sdv',
    'ensemble',
)


def sha256(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_yaml(
    path: Path,
) -> dict:
    repo_root = Path(__file__).resolve().parents[5]
    candidate = path if path.is_absolute() else Path.cwd() / path
    safe_path = candidate.resolve(strict=True)
    try:
        safe_path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f'Path escapes repository root: {path}') from exc

    with safe_path.open(
        'r',
        encoding='utf-8',
    ) as f:
        result = (
            yaml.safe_load(f)
            or {}
        )
    return result


def resolve_model_source(
    cfg: dict,
    requested: str | None,
) -> str:
    source = (
        str(
            requested
        ).strip().lower()
        if requested is not None
        else str(
            cfg.get(
                'production_prediction_source',
                '',
            )
        ).strip().lower()
    )

    if source not in MODEL_SOURCES:
        raise ValueError(
            'model_source must be one of '
            'dratings, sdv, ensemble; '
            f'got {source!r}'
        )

    return source


def calibration_settings(
    league_cfg: dict,
) -> dict:
    cal = (
        league_cfg.get(
            'calibration'
        )
        or {}
    )

    def independent(
        market: str,
        side: str,
    ) -> dict:
        cfg = (
            (
                cal.get(
                    market
                )
                or {}
            )
            .get(
                side
            )
            or {
                'method': 'none'
            }
        )

        return (
            {
                'method': cfg
            }
            if isinstance(
                cfg,
                str,
            )
            else dict(
                cfg
            )
        )

    def complementary(
        market: str,
        first: str,
        second: str,
    ) -> dict:
        market_cfg = (
            cal.get(
                market
            )
            or {}
        )

        canonical = str(
            market_cfg.get(
                'canonical_side',
                first,
            )
        ).strip().lower()

        if canonical not in {
            first,
            second,
        }:
            raise ValueError(
                f'calibration.{market}.'
                'canonical_side='
                f'{canonical!r} is invalid'
            )

        cfg = (
            market_cfg.get(
                canonical
            )
            or {
                'method': 'none'
            }
        )

        if isinstance(
            cfg,
            str,
        ):
            cfg = {
                'method': cfg
            }

        return {
            'canonical_side': (
                canonical
            ),
            'config': dict(
                cfg
            ),
        }

    return {
        'moneyline': {
            'home': independent(
                'moneyline',
                'home',
            ),
            'away': independent(
                'moneyline',
                'away',
            ),
        },
        'spread': complementary(
            'spread',
            'home',
            'away',
        ),
        'total': complementary(
            'total',
            'over',
            'under',
        ),
    }


def bias_value(
    league: str,
    kind: str,
    cfg: dict,
    state: dict,
) -> float:
    bc = cfg[
        'leagues'
    ][
        league
    ][
        'bias'
    ][
        kind
    ]

    method = str(
        bc.get(
            'method',
            '',
        )
    ).lower()

    if method == 'fixed':
        return float(
            bc[
                'value'
            ]
        )

    state_key = (
        'margin_bias'
        if kind == 'margin'
        else 'total_bias'
    )

    value = (
        (
            (
                (
                    state.get(
                        'leagues'
                    )
                    or {}
                )
                .get(
                    league
                )
                or {}
            )
            .get(
                state_key
            )
            or {}
        )
        .get(
            'value'
        )
    )

    if value is None:
        raise ValueError(
            'No current rolling '
            f'{kind} bias in '
            f'{ROLLING_STATE} '
            f'for {league}'
        )

    return float(
        value
    )


def production_settings(
    model_source: str,
) -> dict[
    str,
    dict[
        str,
        object,
    ],
]:
    cfg = load_yaml(
        MODEL_CONFIG
    )

    state = load_yaml(
        ROLLING_STATE
    )

    out = {}

    for league in (
        'nba',
        'ncaam',
        'wnba',
    ):
        lc = cfg[
            'leagues'
        ][
            league
        ]

        spread_std = lc[
            'std'
        ][
            'spread'
        ]

        total_std = lc[
            'std'
        ][
            'total'
        ]

        if (
            str(
                spread_std.get(
                    'mode',
                    '',
                )
            ).lower()
            != 'fixed'
            or str(
                total_std.get(
                    'mode',
                    '',
                )
            ).lower()
            != 'fixed'
        ):
            raise ValueError(
                'Master test compatibility '
                'currently requires fixed '
                'STD modes; '
                f'{league} has spread='
                f"{spread_std.get('mode')} "
                'total='
                f"{total_std.get('mode')}"
            )

        out[
            league.upper()
        ] = {
            'MARGIN_BIAS': bias_value(
                league,
                'margin',
                cfg,
                state,
            ),
            'TOTAL_BIAS': bias_value(
                league,
                'total',
                cfg,
                state,
            ),
            'SPREAD_STD': float(
                spread_std[
                    'value'
                ]
            ),
            'TOTAL_STD': float(
                total_std[
                    'value'
                ]
            ),
            'ML_EDGE': float(
                lc[
                    'edge'
                ][
                    'moneyline'
                ]
            ),
            'SPREAD_EDGE': float(
                lc[
                    'edge'
                ][
                    'spread'
                ]
            ),
            'TOTAL_EDGE': float(
                lc[
                    'edge'
                ][
                    'total'
                ]
            ),
            'MARGIN_BIAS_RULE': dict(
                lc[
                    'bias'
                ][
                    'margin'
                ]
            ),
            'TOTAL_BIAS_RULE': dict(
                lc[
                    'bias'
                ][
                    'total'
                ]
            ),
            'CALIBRATION': (
                calibration_settings(
                    lc
                )
            ),
            'MODEL_SOURCE': model_source,
        }

    return out


def load_core():
    spec = (
        importlib.util
        .spec_from_file_location(
            'basketball_pipeline_final_master_test_core',
            CORE_PATH,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            'Unable to load '
            'master-test core: '
            f'{CORE_PATH}'
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    sys.modules[
        spec.name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def maybe_write_provenance(
    settings: dict,
    model_source: str,
) -> None:
    # Step 18 contract: the validation run itself writes only
    # WNBA_MARKET_BANDS_REPORT.txt.
    if '--wnba-market-bands' in sys.argv:
        return

    parser = argparse.ArgumentParser(
        add_help=False
    )

    parser.add_argument(
        '--input'
    )

    parser.add_argument(
        '--league'
    )

    parser.add_argument(
        '--season'
    )

    parser.add_argument(
        '--model-source',
        choices=MODEL_SOURCES,
    )

    args, _ = (
        parser.parse_known_args()
    )

    if not args.input:
        return

    path = Path(
        args.input
    )

    if not path.exists():
        return

    league = (
        args.league
        or 'NBA'
    ).upper()

    output = (
        path.parent
        / (
            f'{league}_'
            f"{args.season or 'current'}_"
            'MASTER_SETTINGS_PROVENANCE.json'
        )
    )

    payload = {
        'model_config': str(
            MODEL_CONFIG
        ),
        'model_config_sha256': sha256(
            MODEL_CONFIG
        ),
        'rolling_bias_state': str(
            ROLLING_STATE
        ),
        'rolling_bias_state_sha256': sha256(
            ROLLING_STATE
        ),
        'model_source': model_source,
        'settings': settings,
    }

    output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + '\n',
        encoding='utf-8',
    )


def main() -> None:
    cfg = load_yaml(
        MODEL_CONFIG
    )

    parser = argparse.ArgumentParser(
        add_help=False
    )

    parser.add_argument(
        '--model-source',
        choices=MODEL_SOURCES,
    )

    args, _ = (
        parser.parse_known_args()
    )

    model_source = (
        resolve_model_source(
            cfg,
            args.model_source,
        )
    )

    settings = production_settings(
        model_source
    )

    core = load_core()

    core.CURRENT_SETTINGS = (
        settings
    )

    core.DEFAULT_MODEL_SOURCE = (
        model_source
    )

    print(
        'Loaded production settings '
        f'from {MODEL_CONFIG} '
        f'sha256={sha256(MODEL_CONFIG)}',
        flush=True,
    )

    print(
        'Loaded rolling state '
        f'from {ROLLING_STATE} '
        f'sha256={sha256(ROLLING_STATE)}',
        flush=True,
    )

    print(
        f'Model source: {model_source}',
        flush=True,
    )

    maybe_write_provenance(
        settings,
        model_source,
    )

    core.main()


if __name__ == '__main__':
    main()