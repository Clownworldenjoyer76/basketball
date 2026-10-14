#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

BASKETBALL_ROOT = Path(__file__).resolve().parents[1]
STAKING_CONFIG_PATH = BASKETBALL_ROOT / 'config' / 'staking.yaml'
MODEL_CONFIG_PATH = BASKETBALL_ROOT / 'config' / 'model_config.yaml'
LEAGUES = ('nba', 'ncaam', 'wnba')
MARKETS = ('moneyline', 'spread', 'total')


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f'YAML root must be a mapping: {path}')
    return payload


STAKING_CONFIG = _read_yaml(STAKING_CONFIG_PATH)
MODEL_CONFIG = _read_yaml(MODEL_CONFIG_PATH)

KELLY_CFG = STAKING_CONFIG.get('kelly') or {}
UNCERTAINTY_CFG = STAKING_CONFIG.get('uncertainty_adjustment') or {}

KELLY_FRACTION = float(KELLY_CFG['fractional_multiplier'])
UNCERTAINTY_METHOD = str(UNCERTAINTY_CFG.get('method', '')).strip()
UNCERTAINTY_VERSION = str(UNCERTAINTY_CFG.get('version', '')).strip()
UNCERTAINTY_SOURCE = str(UNCERTAINTY_CFG.get('uncertainty_source', '')).strip()
STAKE_SCALING = str(UNCERTAINTY_CFG.get('stake_scaling', '')).strip()

if not math.isfinite(KELLY_FRACTION) or KELLY_FRACTION < 0 or KELLY_FRACTION > 1:
    raise ValueError('staking.yaml kelly.fractional_multiplier must be between 0 and 1')
if UNCERTAINTY_METHOD != 'signal_to_noise_market_shrink':
    raise ValueError(f'Unsupported uncertainty method={UNCERTAINTY_METHOD!r}')
if STAKE_SCALING not in {'uncertainty_multiplier', 'none', 'raw'}:
    raise ValueError(f'Unsupported uncertainty stake_scaling={STAKE_SCALING!r}')


def fv(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value) or str(value).strip() == '':
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _league_key(df: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit:
        league = str(explicit).strip().lower()
    else:
        values = []
        for column in ('league_lower', 'league'):
            if column in df.columns:
                values.extend(
                    str(v).strip().lower()
                    for v in df[column].dropna().unique().tolist()
                    if str(v).strip()
                )
        unique = sorted(set(values))
        if len(unique) != 1:
            raise ValueError(f'Cannot resolve one league for staking uncertainty; values={unique}')
        league = unique[0]
    if league not in LEAGUES:
        raise ValueError(f'Unsupported league={league!r}')
    return league


def _oos_uncertainty_points(league: str, market: str) -> float:
    sources = UNCERTAINTY_CFG.get('market_std_source') or {}
    std_market = str(sources.get(market, '')).strip().lower()
    if std_market not in {'spread', 'total'}:
        raise ValueError(f'Invalid market_std_source.{market}={std_market!r}')
    try:
        value = float(MODEL_CONFIG['leagues'][league]['std'][std_market]['value'])
    except Exception as exc:
        raise ValueError(
            f'Missing OOS uncertainty: model_config.leagues.{league}.std.{std_market}.value'
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'Invalid OOS uncertainty {league}.{std_market}={value!r}')
    return value


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors='coerce')


def _signal_points(df: pd.DataFrame, market: str) -> pd.Series:
    margin = _numeric(df, 'home_projected_points') - _numeric(df, 'away_projected_points')
    if market == 'moneyline':
        return margin.abs()
    if market == 'spread':
        return (margin + _numeric(df, 'home_spread')).abs()
    if market == 'total':
        return (_numeric(df, 'total_projected_points') - _numeric(df, 'total')).abs()
    raise ValueError(f'Unsupported market={market!r}')


def add_uncertainty_adjusted_ev(
    df: pd.DataFrame,
    market: str,
    side_specs: list[tuple[str, str, str, str, str]],
    league: str | None = None,
) -> pd.DataFrame:
    out = df
    league_key = _league_key(out, league)
    uncertainty_points = _oos_uncertainty_points(league_key, market)
    signal = _signal_points(out, market)
    denom = np.sqrt(np.square(signal.to_numpy(float)) + uncertainty_points ** 2)
    multiplier = np.divide(
        signal.to_numpy(float),
        denom,
        out=np.zeros(len(out), dtype=float),
        where=np.isfinite(denom) & (denom > 0),
    )
    if not bool(UNCERTAINTY_CFG.get('enabled', True)):
        multiplier = np.ones(len(out), dtype=float)

    for prefix, raw_prob_col, market_prob_col, decimal_col, raw_kelly_col in side_specs:
        raw_prob = _numeric(out, raw_prob_col)
        market_prob = _numeric(out, market_prob_col)
        decimal = _numeric(out, decimal_col)

        if league_key == 'nba' and market == 'moneyline':
            adjusted_prob = raw_prob.clip(0.001, 0.999)
        else:
            adjusted_prob = (
                market_prob + multiplier * (raw_prob - market_prob)
            ).clip(0.001, 0.999)

        adjusted_ev = adjusted_prob * decimal - 1.0
        raw_ev_col = f'{prefix}_ev'
        out[f'{prefix}_raw_ev'] = out[raw_ev_col]
        out[f'{prefix}_raw_kelly'] = out[raw_kelly_col]
        out[f'{prefix}_uncertainty_adjusted_ev'] = adjusted_ev
        out[f'{prefix}_adjusted_model_prob'] = adjusted_prob
        out[f'{prefix}_uncertainty_multiplier'] = multiplier
        out[f'{prefix}_uncertainty_points'] = uncertainty_points
        out[f'{prefix}_signal_points'] = signal

    out['uncertainty_adjustment_method'] = UNCERTAINTY_METHOD
    out['uncertainty_adjustment_version'] = UNCERTAINTY_VERSION
    out['uncertainty_source'] = UNCERTAINTY_SOURCE
    return out


def attach_candidate_uncertainty(row: Any, market: str, candidate: dict) -> dict | None:
    """Use ONE probability basis for qualification, ranking, EV, and Kelly.

    Precomputed adjusted probabilities are required, not silently replaced by raw
    probabilities. Raw values remain available as diagnostics only. A None result
    is an invalid, non-bettable candidate.
    """
    side = candidate['side']
    prefix = f'{side}_ml' if market == 'moneyline' else f'{side}_spread' if market == 'spread' else side
    decimal_col = (f'{side}_dk_moneyline_decimal' if market == 'moneyline'
                   else f'{side}_dk_spread_decimal' if market == 'spread'
                   else f'dk_total_{side}_decimal')
    market_col = f'{side}_market_prob' if market == 'moneyline' else (
        f'{side}_spread_market_prob' if market == 'spread' else f'{side}_market_prob')

    p = fv(row.get(f'{prefix}_adjusted_model_prob'))
    decimal = fv(row.get(decimal_col))
    market_p = fv(row.get(market_col))
    raw_p = fv(candidate.get('model_prob'))
    raw_ev = fv(candidate.get('ev'))
    raw_kelly = fv(candidate.get('kelly'))
    provided_adj_ev = fv(row.get(f'{prefix}_uncertainty_adjusted_ev'))
    american = fv(candidate.get('odds'))
    wager_line = fv(candidate.get('line'))
    if (american is None or american == 0 or (market != 'moneyline' and wager_line is None) or
        p is None or not 0 < p < 1 or decimal is None or decimal <= 1 or
        market_p is None or not 0 < market_p < 1 or raw_p is None or
        not 0 <= raw_p <= 1 or raw_ev is None or raw_kelly is None or
        provided_adj_ev is None):
        return None

    adj_ev = p * decimal - 1.0
    if not math.isclose(provided_adj_ev, adj_ev, abs_tol=1e-7, rel_tol=1e-7):
        # Reject stale/inconsistent stage-3 columns rather than bet on mixed inputs.
        return None
    adj_kelly = max(0.0, adj_ev / (decimal - 1.0))
    out = dict(candidate)
    out.update({
        'raw_ev': raw_ev,
        'raw_kelly': raw_kelly,
        'raw_model_prob': raw_p,
        'raw_edge_vs_market': candidate.get('edge_vs_market'),
        'ev': adj_ev,
        'kelly': adj_kelly,
        'model_prob': p,
        'edge_vs_market': 100.0 * (p - market_p),
        'uncertainty_adjusted_ev': adj_ev,
        'adjusted_model_prob': p,
        'uncertainty_multiplier': fv(row.get(f'{prefix}_uncertainty_multiplier')),
        'uncertainty_points': fv(row.get(f'{prefix}_uncertainty_points')),
        'signal_points': fv(row.get(f'{prefix}_signal_points')),
        'probability_basis': 'uncertainty_adjusted',
    })
    return out


def enforce_exposure_limits(df: pd.DataFrame) -> pd.DataFrame:
    """Greedily allocate one bankroll across a slate, never exceeding any cap.

    Ordered by adjusted EV per dollar; correlation beyond same-game caps is not
    estimated. Requests with missing game identities or invalid amounts fail closed.
    """
    if df.empty:
        return df.copy()
    limits = STAKING_CONFIG.get('exposure_limits') or {}
    names = ('maximum_individual_bet_kelly_fraction', 'maximum_exposure_per_game',
             'maximum_exposure_per_league_per_day', 'maximum_total_daily_exposure')
    cap = {}
    for name in names:
        value = fv(limits.get(name))
        if value is None or not 0 <= value <= 1:
            raise ValueError(f'Invalid staking exposure limit: {name}')
        cap[name] = value

    out = df.copy()
    required = ('game_date', 'league_lower', 'game_id', 'bet_requested_stake_pct', 'bet_ev')
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f'Missing stake allocation fields: {missing}')
    out['_alloc_order'] = range(len(out))
    out['_alloc_ev'] = pd.to_numeric(out['bet_ev'], errors='coerce')
    out = out.sort_values(['_alloc_ev', '_alloc_order'], ascending=[False, True], kind='stable')
    game_use, league_use, date_use = {}, {}, {}
    stakes, notes = {}, {}
    for index, r in out.iterrows():
        date = str(r['game_date']).strip() if pd.notna(r['game_date']) else ''
        league = str(r['league_lower']).strip().lower() if pd.notna(r['league_lower']) else ''
        gid = str(r['game_id']).strip() if pd.notna(r['game_id']) else ''
        requested = fv(r['bet_requested_stake_pct'])
        ev = fv(r['bet_ev'])
        if not date or date.lower() in {'none', 'nan', 'unknown_date'} or \
           league not in LEAGUES or not gid or gid.lower() in {'none', 'nan'} or \
           requested is None or requested <= 0 or ev is None or ev <= 0:
            stakes[index], notes[index] = 0.0, 'invalid_or_nonpositive'
            continue
        game_key = (date, league, gid)
        league_key = (date, league)
        amounts = [
            requested,
            cap['maximum_individual_bet_kelly_fraction'],
            cap['maximum_exposure_per_game'] - game_use.get(game_key, 0.0),
            cap['maximum_exposure_per_league_per_day'] - league_use.get(league_key, 0.0),
            cap['maximum_total_daily_exposure'] - date_use.get(date, 0.0),
        ]
        stake = max(0.0, min(amounts))
        if stake < 1e-12:
            stakes[index], notes[index] = 0.0, 'exhausted_exposure'
            continue
        stakes[index] = stake
        notes[index] = 'capped' if stake < requested - 1e-12 else 'full_requested'
        game_use[game_key] = game_use.get(game_key, 0.0) + stake
        league_use[league_key] = league_use.get(league_key, 0.0) + stake
        date_use[date] = date_use.get(date, 0.0) + stake
    out['bet_stake_pct'] = out.index.map(stakes.get).astype(float)
    out['bet_stake_allocation'] = out.index.map(notes.get)
    out = out[out['bet_stake_pct'] > 0].sort_values('_alloc_order')
    return out.drop(columns=['_alloc_order', '_alloc_ev'])


def requested_stake(raw_kelly: Any, uncertainty_multiplier: Any = 1.0) -> tuple[float, float, float]:
    """Return fractional-Kelly stake suggestion without filtering any bet.

    The second tuple value is retained for compatibility with existing callers;
    it is identical to the uncapped fractional-Kelly value.
    """
    kelly = fv(raw_kelly)
    multiplier = fv(uncertainty_multiplier)
    if kelly is None or kelly <= 0:
        return 0.0, 0.0, 0.0
    if multiplier is None:
        multiplier = 1.0
    multiplier = min(max(multiplier, 0.0), 1.0)
    fractional = max(kelly * KELLY_FRACTION, 0.0)
    requested = fractional * multiplier if STAKE_SCALING == 'uncertainty_multiplier' else fractional
    return fractional, fractional, requested
