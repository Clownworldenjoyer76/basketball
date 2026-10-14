#!/usr/bin/env python3
"""Remove unusable, unscheduled current-day sportsbook placeholders after cleaning.

Preserves all priced rows, rows on the active daily slate, and all historical files.
Runs once per workflow before merge. Never manufactures odds or games.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path('docs/win/basketball')
LEAGUES = ('nba', 'ncaam', 'wnba')
NY = ZoneInfo('America/New_York')
# Keep any row containing a usable price or market line, even if not on the
# schedule: unmatched genuine quotes require investigation, not silent deletion.
MARKET_FIELDS = (
    'home_dk_moneyline_american', 'away_dk_moneyline_american',
    'home_dk_spread_american', 'away_dk_spread_american',
    'dk_total_over_american', 'dk_total_under_american',
    'home_dk_moneyline_decimal', 'away_dk_moneyline_decimal',
    'home_dk_spread_decimal', 'away_dk_spread_decimal',
    'dk_total_over_decimal', 'dk_total_under_decimal',
    'home_spread', 'away_spread', 'total',
)


def clean(v):
    return '' if v is None else str(v).strip()


def key(row):
    return (clean(row.get('game_date')).replace('-', '_'),
            clean(row.get('game_id')),
            clean(row.get('home_team')).casefold(),
            clean(row.get('away_team')).casefold())


def has_market_information(row):
    return any(clean(row.get(field)).casefold() not in
               ('', 'nan', 'null', 'none', 'n/a') for field in MARKET_FIELDS)


def read_rows(path):
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)) or not all(
            field in fields for field in ('game_date', 'home_team', 'away_team', 'game_id')
        ):
            raise ValueError(f'Invalid sportsbook/daily CSV header: {path}')
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f'Malformed sportsbook/daily CSV row: {path}')
    return fields, rows


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cleanup(root, game_date, *, dry_run=False, now_date=None):
    """Safety boundary: never rewrite a past or future date's sportsbook file."""
    now_date = now_date or datetime.now(NY).strftime('%Y_%m_%d')
    if game_date != now_date:
        raise ValueError(f'Cleanup limited to current New York date {now_date}; got {game_date}')
    report = {'date': game_date, 'dry_run': dry_run, 'leagues': {},
              'removed_rows_total': 0, 'retained_unmatched_priced_rows': 0,
              'warning': 'Only unpriced, non-scheduled current-day placeholders are removed. Source sportsbook archives and historical files are not touched.'}
    staged = []
    for league in LEAGUES:
        label = league.upper()
        scheduled_path = root / BASE / 'daily_games' / league / f'{game_date}_{label}.csv'
        clean_path = root / BASE / '00_intake/sportsbook/sportsbook_cleaned' / league / f'{game_date}_{label}_odds.csv'
        if scheduled_path.is_symlink() or clean_path.is_symlink():
            raise ValueError('Symlinked input refused')
        if scheduled_path.is_file():
            _, scheduled = read_rows(scheduled_path)
            scheduled_keys = {key(r) for r in scheduled if clean(r.get('game_date')).replace('-', '_') == game_date}
            scheduled_pairs = {(k[0], k[2], k[3]) for k in scheduled_keys}
        else:
            scheduled_keys, scheduled_pairs = set(), set()
        info = {'active_schedule_rows': len(scheduled_keys), 'input_rows': 0,
                'removed_rows': 0, 'retained_rows': 0,
                'retained_unmatched_priced_rows': 0,
                'removed': [], 'path': clean_path.relative_to(root).as_posix()}
        if not clean_path.is_file():
            report['leagues'][league] = info
            continue
        fields, rows = read_rows(clean_path)
        info['input_rows'] = len(rows)
        original_digest = sha(clean_path)
        retained = []
        for row in rows:
            same_date = clean(row.get('game_date')).replace('-', '_') == game_date
            row_key = key(row)
            on_schedule = (row_key in scheduled_keys or
                           (row_key[0], row_key[2], row_key[3]) in scheduled_pairs)
            priced = has_market_information(row)
            if same_date and not on_schedule and not priced:
                info['removed_rows'] += 1
                info['removed'].append({'game_key': list(row_key),
                    'reason': 'unpriced_current_day_fixture_not_on_active_daily_slate',
                    'row_sha256': hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()})
            else:
                retained.append(row)
                if same_date and not on_schedule and priced:
                    info['retained_unmatched_priced_rows'] += 1
        info['retained_rows'] = len(retained)
        info['original_file_sha256'] = original_digest
        report['removed_rows_total'] += info['removed_rows']
        report['retained_unmatched_priced_rows'] += info['retained_unmatched_priced_rows']
        report['leagues'][league] = info
        if info['removed_rows'] and not dry_run:
            staged.append((clean_path, fields, retained, original_digest))
    # Do not change ANY file if one league's input cannot be safely parsed.
    for path, fields, rows, digest in staged:
        if sha(path) != digest:
            raise RuntimeError(f'Input file changed during cleanup: {path}')
    for path, fields, rows, digest in staged:
        tmp = None
        try:
            with tempfile.NamedTemporaryFile('w', newline='', encoding='utf-8',
                                             dir=path.parent, prefix='.cleanup-',
                                             suffix='.csv', delete=False) as f:
                tmp = Path(f.name)
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(tmp, path)
        finally:
            if tmp is not None and tmp.exists():
                tmp.unlink()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository', type=Path, default=Path('.'))
    p.add_argument('--date', default=datetime.now(NY).strftime('%Y_%m_%d'))
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    root = args.repository.resolve()
    if not (root / BASE / 'scripts').is_dir():
        raise ValueError('Not a basketball repository')
    datetime.strptime(args.date, '%Y_%m_%d')
    report = cleanup(root, args.date, dry_run=args.dry_run)
    # Emit a diagnostics file to the artifact directory with run-specific identity.
    if not args.dry_run:
        run_id = os.getenv('GITHUB_RUN_ID')
        attempt = os.getenv('GITHUB_RUN_ATTEMPT', '1')
        if run_id and re.fullmatch(r'[A-Za-z0-9_-]{1,100}', run_id + '_' + attempt):
            path = root / BASE / 'pit_audit/runs' / (run_id + '_' + attempt) / 'sportsbook_cleanup.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf-8', newline='\n') as f:
                json.dump(report, f, sort_keys=True, indent=2)
                f.write('\n')
    print(f"SPORTSBOOK PLACEHOLDER CLEANUP: removed={report['removed_rows_total']} "
          f"unmatched_priced_retained={report['retained_unmatched_priced_rows']} "
          f"date={args.date}; historical data unchanged")


if __name__ == '__main__':
    main()
