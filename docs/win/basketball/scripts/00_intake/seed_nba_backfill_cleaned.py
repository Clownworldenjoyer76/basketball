#!/usr/bin/env python3
"""Seed cleaned NBA opening-backfill predictions without rewriting Oct 20 forecasts.

Oct 20: preserve existing historical DRatings forecasts, probabilities, projections,
         and bias; only reconcile blank game IDs from canonical final-score fixtures.
Oct 21-22: create missing historical cleaned ensemble files, with zero retrospective
           bias, from the ensemble source files produced earlier in the workflow.
All inputs are retrospective evidence; no pre-tipoff timestamp is invented.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

BASE = Path('docs/win/basketball')
ROOT = BASE / '00_intake'
OCT20 = '2026_10_20'
EXPECTED = {
    '2026_10_20': {'401909088', '401909089', '401909090'},
    '2026_10_21': {
        '401909834', '401909835', '401909836', '401909091',
        '401909837', '401909838', '401909839', '401909840',
        '401909092', '401909841', '401909842',
    },
    '2026_10_22': {'401909093', '401909094'},
}


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise RuntimeError(f'NBA BACKFILL missing required file: {path}')
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        if not headers:
            raise RuntimeError(f'NBA BACKFILL blank CSV headers: {path}')
        rows = list(reader)
    return headers, rows


def clean(value: object) -> str:
    return ' '.join(str(value or '').strip().split()).casefold()


def matchup(row: dict[str, str]) -> tuple[str, str, str]:
    return (clean(row.get('game_date')), clean(row.get('home_team')), clean(row.get('away_team')))


def validate_ids(rows: list[dict[str, str]], expected: set[str], path: Path) -> None:
    ids = [str(r.get('game_id') or '').strip() for r in rows]
    if len(ids) != len(expected) or len(ids) != len(set(ids)) or set(ids) != expected:
        raise RuntimeError(
            f'NBA BACKFILL incomplete game IDs: {path}; '
            f'expected={sorted(expected)}, found={sorted(ids)}'
        )


def oct20_reconciliation() -> tuple[Path, list[str], list[dict[str, str]], bool]:
    """Do not replace original Oct 20 forecasts or change any forecast value."""
    name = f'{OCT20}_NBA_predictions.csv'
    raw_path = ROOT / 'predictions/nba' / name
    dst = ROOT / 'predictions/predictions_cleaned/nba' / name
    scores_path = BASE / '05_final_scores/results/nba' / f'{OCT20}_final_scores_NBA.csv'
    _, finals = read(scores_path)
    _, original = read(raw_path)
    headers, cleaned = read(dst)

    if len(finals) != 3 or len(original) != 3 or len(cleaned) != 3:
        raise RuntimeError('NBA BACKFILL Oct 20 must have exactly 3 games in finals/raw/cleaned')
    key_to_id: dict[tuple[str, str, str], str] = {}
    for result in finals:
        key = matchup(result)
        gid = str(result.get('game_id') or '').strip()
        if key in key_to_id or gid not in EXPECTED[OCT20]:
            raise RuntimeError(f'NBA BACKFILL Oct 20 invalid final score identity: {result}')
        key_to_id[key] = gid
    if set(key_to_id.values()) != EXPECTED[OCT20]:
        raise RuntimeError('NBA BACKFILL Oct 20 canonical game IDs do not match expected IDs')

    orig_by_key = {matchup(r): r for r in original}
    if len(orig_by_key) != 3 or set(orig_by_key) != set(key_to_id):
        raise RuntimeError('NBA BACKFILL Oct 20 original forecasts do not match the final-score fixtures')
    if len({matchup(r) for r in cleaned}) != 3 or {matchup(r) for r in cleaned} != set(key_to_id):
        raise RuntimeError('NBA BACKFILL Oct 20 cleaned forecasts do not match the original fixtures')
    if 'game_id' not in headers:
        headers.append('game_id')

    updated = False
    for row in cleaned:
        key = matchup(row)
        original_row = orig_by_key[key]
        for name in ('home_prob', 'away_prob'):
            try:
                same = abs(float(row[name]) - float(original_row[name])) <= 1e-9
            except (KeyError, TypeError, ValueError):
                same = False
            if not same:
                raise RuntimeError(
                    f'NBA BACKFILL Oct 20 original {name} must not be changed: {key}'
                )
        for name in ('home_projected_points', 'away_projected_points', 'total_projected_points'):
            try:
                float(row[name])
                float(original_row[name])
            except (KeyError, ValueError, TypeError):
                raise RuntimeError(f'NBA BACKFILL Oct 20 missing projected points: {key}')
        existing = str(row.get('game_id') or '').strip()
        gid = key_to_id[key]
        if existing and existing != gid:
            raise RuntimeError(f'NBA BACKFILL Oct 20 conflicting game ID: {existing} vs {gid}')
        if existing != gid:
            row['game_id'] = gid
            updated = True

    validate_ids(cleaned, EXPECTED[OCT20], dst)
    return dst, headers, cleaned, updated


def remaining_ensemble_seeds() -> list[tuple[Path, list[str], list[dict[str, str]]]]:
    ready = []
    for day in ('2026_10_21', '2026_10_22'):
        name = f'{day}_NBA_predictions.csv'
        src = ROOT / 'predictions_ensemble/nba' / name
        dst = ROOT / 'predictions/predictions_cleaned/nba' / name
        headers, rows = read(src)
        validate_ids(rows, EXPECTED[day], src)
        for row in rows:
            for field in ('home_prob', 'away_prob', 'home_projected_points', 'away_projected_points', 'total_projected_points'):
                try:
                    float(row[field])
                except (KeyError, ValueError, TypeError):
                    raise RuntimeError(f'NBA BACKFILL invalid ensemble {field}: {src}')
        if dst.is_file():
            _, existing = read(dst)
            validate_ids(existing, EXPECTED[day], dst)
            print(f'NBA BACKFILL existing cleaned preserved: {dst}')
            continue
        for field in ('bias_applied', 'margin_bias', 'total_bias'):
            if field not in headers:
                headers.append(field)
        for row in rows:
            row['bias_applied'] = '1'
            row['margin_bias'] = '0.000'
            row['total_bias'] = '0.000'
        ready.append((dst, headers, rows))
    return ready


def write_atomic(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.csv.tmp')
    try:
        with tmp.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> None:
    if os.getenv('BASKETBALL_FULL_REBUILD', '').strip().casefold() not in ('1', 'true', 'yes', 'on'):
        print('NBA BACKFILL: skipped (not full rebuild)')
        return
    # Fail closed: validate every date before writing any output.
    legacy_dst, legacy_headers, legacy_rows, legacy_changed = oct20_reconciliation()
    ready = remaining_ensemble_seeds()
    if legacy_changed:
        write_atomic(legacy_dst, legacy_headers, legacy_rows)
        print(f'NBA BACKFILL Oct 20 original cleaned preserved; game IDs restored: {legacy_dst}')
    else:
        print(f'NBA BACKFILL Oct 20 original cleaned and game IDs already preserved: {legacy_dst}')
    for dst, headers, rows in ready:
        write_atomic(dst, headers, rows)
        print(f'NBA BACKFILL historical cleaned ensemble seeded: {dst} ({len(rows)})')
    print('NBA BACKFILL HISTORICAL CLEANED INPUTS: PASS')


if __name__ == '__main__':
    main()
