#!/usr/bin/env python3
"""Evidence-only, fail-closed basketball point-in-time audit. No betting decisions are altered.

Capture is run after model generation; selection is checked after bet selection.
Proof is limited to in-run immutable evidence; provider availability and actual tipoff
cannot be independently certified from this repository alone.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo('America/New_York')
LEAGUES = ('nba', 'ncaam', 'wnba')
MARKET = {'moneyline': 'moneyline', 'spread': 'spread', 'total': 'total'}


def now_utc():
    return datetime.now(timezone.utc)


def utc_iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def stamp(value):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
        return dt.astimezone(timezone.utc) if dt.tzinfo else None
    except ValueError:
        return None


def tipoff(row):
    """Recorded New York local scheduled tipoff, not independently verified actual start."""
    date = str(row.get('game_date', '')).strip().replace('-', '_')
    time = str(row.get('game_time', '')).strip().upper()
    try:
        day = datetime.strptime(date, '%Y_%m_%d').date()
    except ValueError:
        return None
    for fmt in ('%I:%M %p', '%H:%M', '%I %p'):
        try:
            local_time = datetime.strptime(time, fmt).time()
            naive = datetime.combine(day, local_time)
            aware = naive.replace(tzinfo=NY)
            # Reject nonexistent and ambiguous DST local times; never guess.
            if aware.astimezone(timezone.utc).astimezone(NY).replace(tzinfo=None) != naive:
                return None
            if aware.utcoffset() != naive.replace(tzinfo=NY, fold=1).utcoffset():
                return None
            return aware.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for piece in iter(lambda: f.read(1 << 20), b''):
            h.update(piece)
    return h.hexdigest()


def row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def read_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def game_key(row, league):
    return (league, str(row.get('game_date', '')).replace('-', '_').lower(), str(row.get('game_id', '')).strip(), str(row.get('home_team', '')).strip().casefold(), str(row.get('away_team', '')).strip().casefold())


def load_active_schedule(root, game_date):
    """Read the canonical daily output, not archived prediction/odds files.

    Missing daily files represent zero confirmed scheduled games for this date.
    This is an evidence classification, not independent confirmation of actual tipoff.
    """
    index = {}
    source = {}
    for league in LEAGUES:
        path = (root / 'docs/win/basketball/daily_games' / league /
                f'{game_date}_{league.upper()}.csv')
        if path.is_file():
            rows = read_csv(path)
            keys = {game_key(r, league) for r in rows
                    if str(r.get('game_date', '')).replace('-', '_') == game_date}
            digest = sha(path)
            source[league] = {'path': path.relative_to(root).as_posix(),
                              'sha256': digest, 'active_games': len(keys)}
        else:
            keys = set()
            source[league] = {'path': None, 'sha256': None, 'active_games': 0}
        index[league] = keys
    return index, source


def classify_schedule(entry, league, schedule_index, schedule_sources):
    """Keep rejected evidence visible while preventing it from certifying bets."""
    source = schedule_sources[league]
    active = tuple(entry['game_key']) in schedule_index[league]
    entry['schedule_membership'] = 'ACTIVE_DAILY_SLATE' if active else 'ORPHAN_NOT_ON_DAILY_SLATE'
    entry['schedule_source_file'] = source['path']
    entry['schedule_source_sha256'] = source['sha256']
    if not active:
        entry['reasons'].append('game_not_on_active_daily_schedule')
    return entry


def numeric_equal(a, b, tol=1e-6):
    try:
        return abs(float(a) - float(b)) <= tol
    except (ValueError, TypeError):
        return False


def record(row, league, path, file_digest, kind, observed):
    game_start = tipoff(row)
    issued = stamp(row.get('prediction_generated_at_utc') if kind == 'forecast' else row.get('scraped_at_utc'))
    if kind == 'odds':
        provider_time = stamp(row.get('provider_updated_at_utc'))
    else:
        provider_time = None
    reasons = []
    if not game_start:
        reasons.append('scheduled_tipoff_missing_or_ambiguous')
    if not issued:
        reasons.append('event_timestamp_missing_or_invalid')
    if issued and issued > observed:
        reasons.append('event_timestamp_after_observation')
    if game_start and issued and issued >= game_start:
        reasons.append('timestamp_not_before_recorded_tipoff')
    if game_start and observed >= game_start:
        reasons.append('capture_not_before_recorded_tipoff')
    if provider_time and issued and provider_time > issued:
        reasons.append('provider_update_after_quote_capture')
    # A file's own declared source timestamps are not independently authenticated.
    return {
        'league': league, 'game_key': list(game_key(row, league)),
        'path': path, 'file_sha256': file_digest, 'row_sha256': row_hash(row),
        'kind': kind, 'observed_at_utc': utc_iso(observed),
        'declared_at_utc': utc_iso(issued) if issued else None,
        'recorded_tipoff_utc': utc_iso(game_start) if game_start else None,
        'reasons': reasons, 'fields': {
            k: row.get(k, '') for k in (
                'model_source', 'model_version', 'feature_version',
                'prediction_generated_at_utc', 'sportsbook_provider',
                'scraped_at_utc', 'provider_updated_at_utc',
                'game_date', 'game_id', 'game_time', 'home_team', 'away_team',
                'home_prob', 'away_prob', 'home_spread', 'away_spread', 'total',
                'home_dk_moneyline_american', 'away_dk_moneyline_american',
                'home_dk_spread_american', 'away_dk_spread_american',
                'dk_total_over_american', 'dk_total_under_american'
            )
        }
    }


def collect(root, game_date, observed, schedule_index=None, schedule_sources=None):
    if schedule_index is None or schedule_sources is None:
        schedule_index, schedule_sources = load_active_schedule(root, game_date)
    base = root / 'docs/win/basketball/00_intake'
    forecasts, quotes = [], []
    inputs = []
    for league in LEAGUES:
        label = league.upper()
        for dirname in ('predictions_ensemble', 'predictions_sdv', 'predictions'):
            p = base / dirname / league / f'{game_date}_{label}_predictions.csv'
            if p.exists():
                inputs.append((p, league, 'forecast'))
        snapshot_dir = base / 'sportsbook_snapshots' / league
        if snapshot_dir.is_dir():
            # Every archived snapshot may quote future games; do not assume its filename date equals the game date.
            for p in sorted(snapshot_dir.glob('*_odds_snapshot.csv')):
                inputs.append((p, league, 'odds'))
    for p, league, kind in inputs:
        digest = sha(p)
        relative = p.relative_to(root).as_posix()
        for row in read_csv(p):
            if str(row.get('game_date', '')).replace('-', '_') != game_date:
                continue
            entry = classify_schedule(
                record(row, league, relative, digest, kind, observed),
                league, schedule_index, schedule_sources,
            )
            if kind == 'forecast':
                forecasts.append(entry)
            else:
                # Archive snapshots must have a collector-issued filename matching row timestamp.
                m = re.match(r'^(\d{8}T\d{6}\d{6}Z)_', p.name)
                date_value = stamp(row.get('scraped_at_utc'))
                if not m or not date_value:
                    entry['reasons'].append('snapshot_filename_capture_unverified')
                else:
                    filename_time = datetime.strptime(m.group(1), '%Y%m%dT%H%M%S%fZ').replace(tzinfo=timezone.utc)
                    if abs((filename_time - date_value).total_seconds()) > 30:
                        entry['reasons'].append('snapshot_filename_capture_mismatch')
                quotes.append(entry)
    return forecasts, quotes


def offered_fields(market, side):
    if market == 'moneyline' and side in ('home', 'away'):
        return f'{side}_dk_moneyline_american', None
    if market == 'spread' and side in ('home', 'away'):
        return f'{side}_dk_spread_american', f'{side}_spread'
    if market == 'total' and side in ('over', 'under'):
        return f'dk_total_{side}_american', 'total'
    return None, None


def evaluate(selection, league, forecasts, quotes, root):
    key = list(game_key(selection, league))
    matching_forecasts = [x for x in forecasts if x['game_key'] == key]
    matching_quotes = [x for x in quotes if x['game_key'] == key]
    model_source = str(selection.get('model_source', '')).strip()
    model_version = str(selection.get('model_version', '')).strip()
    matching_forecasts = [x for x in matching_forecasts
                          if x['fields'].get('model_source', '') == model_source
                          and x['fields'].get('model_version', '') == model_version
                          and numeric_equal(x['fields'].get('home_prob'), selection.get('home_prob'), 1e-7)]
    # Accept exactly one matching source row; duplicated artifacts are not independent proof.
    eligible_forecasts = [x for x in matching_forecasts if not x['reasons']]
    market = str(selection.get('market_type', '')).strip().lower()
    side = str(selection.get('bet_side', '')).strip().lower()
    price_field, line_field = offered_fields(market, side)
    quote_time = stamp(selection.get('scraped_at_utc'))
    eligible_quotes = []
    for x in matching_quotes:
        f = x['fields']
        if x['reasons'] or not price_field:
            continue
        if f.get('sportsbook_provider') != selection.get('sportsbook_provider'):
            continue
        if not quote_time or stamp(f.get('scraped_at_utc')) != quote_time:
            continue
        if not numeric_equal(f.get(price_field), selection.get('bet_odds_american')):
            continue
        if line_field and not numeric_equal(f.get(line_field), selection.get('bet_line')):
            continue
        eligible_quotes.append(x)
    reasons = []
    if not price_field:
        reasons.append('unsupported_market_or_side')
    if not eligible_forecasts:
        reasons.append('no_matching_pre_tipoff_forecast_with_version_probability')
    if not eligible_quotes:
        reasons.append('no_matching_pre_tipoff_immutable_quote_with_price_line_timestamp')
    # Selected row may carry a different scheduled tipoff than source evidence.
    selected_tip = tipoff(selection)
    if selected_tip is None:
        reasons.append('selected_tipoff_missing_or_ambiguous')
    else:
        if any(x['recorded_tipoff_utc'] != utc_iso(selected_tip) for x in eligible_quotes + eligible_forecasts):
            reasons.append('recorded_tipoff_disagreement')
    # Neither source evidence nor the active-slate proof may change after capture.
    candidates = eligible_forecasts + eligible_quotes
    for x in candidates:
        file_path = root / x['path']
        if not file_path.exists() or sha(file_path) != x['file_sha256']:
            reasons.append('source_file_changed_after_capture')
            break
        schedule_file = x.get('schedule_source_file')
        schedule_hash = x.get('schedule_source_sha256')
        if (not schedule_file or not schedule_hash or
                not (root / schedule_file).is_file() or
                sha(root / schedule_file) != schedule_hash):
            reasons.append('active_daily_schedule_changed_after_capture')
            break
    chosen_forecast = eligible_forecasts[0] if eligible_forecasts else None
    chosen_quote = eligible_quotes[0] if eligible_quotes else None
    return {
        'league': league, 'game_key': key, 'market': market, 'side': side,
        'selected_price_american': selection.get('bet_odds_american'),
        'selected_line': selection.get('bet_line'),
        'selected_row_sha256': row_hash(selection),
        'status': 'PROVISIONAL_PRETIPOFF_MATCH' if not reasons else 'NOT_VERIFIED',
        'reasons': sorted(set(reasons)),
        'forecast_row_sha256': chosen_forecast['row_sha256'] if chosen_forecast else None,
        'forecast_file_sha256': chosen_forecast['file_sha256'] if chosen_forecast else None,
        'forecast_model_version': model_version,
        'quote_row_sha256': chosen_quote['row_sha256'] if chosen_quote else None,
        'quote_file_sha256': chosen_quote['file_sha256'] if chosen_quote else None,
        'forecast_generated_at_utc': chosen_forecast['declared_at_utc'] if chosen_forecast else None,
        'quote_scraped_at_utc': chosen_quote['declared_at_utc'] if chosen_quote else None,
        'recorded_tipoff_utc': utc_iso(selected_tip) if selected_tip else None,
    }


def write_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, sort_keys=True, indent=2)
        f.write('\n')


def run(args):
    root = args.repository.resolve()
    if not (root / 'docs/win/basketball/scripts').is_dir():
        raise ValueError('Not a basketball repository')
    date = args.date or datetime.now(NY).strftime('%Y_%m_%d')
    datetime.strptime(date, '%Y_%m_%d')
    run_id = args.run_id or ('local_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,110}', run_id):
        raise ValueError('Invalid run id')
    folder = root / 'docs/win/basketball/pit_audit/runs' / run_id
    phase = args.phase
    if phase == 'capture':
        observed = now_utc()
        schedule_index, schedule_sources = load_active_schedule(root, date)
        forecasts, quotes = collect(root, date, observed, schedule_index, schedule_sources)
        active_forecasts = sum(x['schedule_membership'] == 'ACTIVE_DAILY_SLATE' for x in forecasts)
        active_quotes = sum(x['schedule_membership'] == 'ACTIVE_DAILY_SLATE' for x in quotes)
        data = {'schema_version': 'basketball_pit_v1', 'phase': 'capture',
                'run_id': run_id, 'date': date, 'observed_at_utc': utc_iso(observed),
                'active_schedule': schedule_sources,
                'forecast_rows': forecasts, 'quote_rows': quotes,
                'active_forecast_rows': active_forecasts,
                'orphan_forecast_rows': len(forecasts) - active_forecasts,
                'active_quote_rows': active_quotes,
                'orphan_quote_rows': len(quotes) - active_quotes,
                'warning': 'Schedule membership is based on a captured daily slate, not independent actual tipoff or provider proof. Orphaned records are retained for audit but cannot verify wagers.'}
        write_once(folder / 'capture.json', data)
        print(f'PIT CAPTURE: active_forecasts={active_forecasts} orphan_forecasts={len(forecasts)-active_forecasts} active_quotes={active_quotes} orphan_quotes={len(quotes)-active_quotes} file={folder / "capture.json"}')
    elif phase == 'selection':
        input_path = folder / 'capture.json'
        with input_path.open('r', encoding='utf-8') as f:
            captured = json.load(f)
        if captured.get('run_id') != run_id or captured.get('date') != date or captured.get('schema_version') != 'basketball_pit_v1':
            raise ValueError('Capture manifest does not match run identity/date/schema')
        outcomes = []
        for league in LEAGUES:
            file = root / 'docs/win/basketball/04_select' / league / 'daily_picks' / f'{date}_{league}_selected.csv'
            if not file.exists():
                continue
            file_hash = sha(file)
            for sel in read_csv(file):
                if str(sel.get('game_date', '')).replace('-', '_') != date:
                    continue
                result = evaluate(sel, league, captured['forecast_rows'], captured['quote_rows'], root)
                result['selected_file'] = file.relative_to(root).as_posix()
                result['selected_file_sha256'] = file_hash
                outcomes.append(result)
        passed = sum(x['status'] == 'PROVISIONAL_PRETIPOFF_MATCH' for x in outcomes)
        output = {'schema_version': 'basketball_pit_v1', 'phase': 'selection',
                  'run_id': run_id, 'date': date, 'evaluated_at_utc': utc_iso(now_utc()),
                  'selected_rows': len(outcomes), 'provisional_matches': passed,
                  'not_verified': len(outcomes)-passed, 'selections': outcomes,
                  'warning': 'Provisional status checks recorded scheduled tipoff, not independent actual tipoff or sportsbook execution.'}
        write_once(folder / 'selection.json', output)
        print(f'PIT SELECTION AUDIT: {passed}/{len(outcomes)} provisional matches; no picks changed.')
    else:
        raise ValueError('Unknown phase')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository', type=Path, default=Path('.'))
    p.add_argument('--phase', choices=('capture', 'selection'), required=True)
    p.add_argument('--date', default=None)
    p.add_argument('--run-id', default=os.getenv('GITHUB_RUN_ID', 'local') + '_' + os.getenv('GITHUB_RUN_ATTEMPT', '1') if os.getenv('GITHUB_RUN_ID') else None)
    run(p.parse_args())


if __name__ == '__main__':
    main()
