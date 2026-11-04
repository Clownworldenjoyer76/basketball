#!/usr/bin/env python3
"""Assign schedule game IDs to basketball predictions and final scores.

NCAAM teams are matched through the repository's validated alias map. Matches
must have the same normalized date, the same home/away orientation, and one
unambiguous schedule game ID. No game ID is invented from a team name.
"""

import csv
import re
import sys
import traceback
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from basketball_shared import resolve_repository_path

LEAGUES = {
    'nba': {
        'league_label': 'NBA',
        'daily_games_dir': Path('docs/win/basketball/daily_games/nba'),
        'predictions_dir': Path('docs/win/basketball/00_intake/predictions/nba'),
        'final_scores_dir': Path('docs/win/basketball/05_final_scores/results/nba'),
    },
    'ncaam': {
        'league_label': 'NCAAM',
        'daily_games_dir': Path('docs/win/basketball/daily_games/ncaam'),
        'predictions_dir': Path('docs/win/basketball/00_intake/predictions/ncaam'),
        'final_scores_dir': Path('docs/win/basketball/05_final_scores/results/ncaam'),
    },
    'wnba': {
        'league_label': 'WNBA',
        'daily_games_dir': Path('docs/win/basketball/daily_games/wnba'),
        'predictions_dir': Path('docs/win/basketball/00_intake/predictions/wnba'),
        'final_scores_dir': Path('docs/win/basketball/05_final_scores/results/wnba'),
    },
}

NCAAM_MAP_FILE = Path('docs/win/basketball/maps/team_map_ncaam.csv')
ERROR_DIR = Path('docs/win/basketball/errors/00_intake')
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / 'basketball_game_id.txt'

with open(LOG_FILE, 'w', encoding='utf-8') as startup_log_handle:
    startup_log_handle.write(f'=== basketball_game_id RUN {datetime.now().isoformat()} ===\n')

_ALIAS_MAPS = {}
ERRORS = []


def log(msg: str) -> None:
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f'{datetime.now().isoformat()} | {msg}\n')


def clean_value(val):
    if val is None:
        return ''
    value = str(val).strip()
    return '' if value.casefold() in {'nan', 'none', 'null', 'nat'} else value


def normalize_date(value):
    """Normalize known dates only. Never infer a day from an invalid date."""
    raw = clean_value(value)
    for fmt in ('%Y_%m_%d', '%Y-%m-%d', '%Y/%m/%d'):
        try:
            return datetime.strptime(raw, fmt).strftime('%Y_%m_%d')
        except ValueError:
            continue
    return ''


def normalize_team(value):
    """Normalize case, accents, spacing and punctuation (not team identities)."""
    value = unicodedata.normalize('NFKD', clean_value(value)).casefold()
    value = ''.join(c for c in value if not unicodedata.combining(c))
    return ' '.join(re.findall(r'[a-z0-9]+', value))


def load_team_alias_map(league_label):
    if league_label != 'NCAAM':
        return {}
    if league_label in _ALIAS_MAPS:
        return _ALIAS_MAPS[league_label]

    safe_path = resolve_repository_path(NCAAM_MAP_FILE, strict=True)
    mapping = {}
    with safe_path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'league', 'alias', 'canonical_team'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f'Invalid NCAAM team map headers: {safe_path}')
        for number, row in enumerate(reader, start=2):
            if clean_value(row.get('league')).upper() != 'NCAAM':
                continue
            alias = normalize_team(row.get('alias'))
            canonical = normalize_team(row.get('canonical_team'))
            if not alias or not canonical:
                raise ValueError(f'Empty NCAAM alias/canonical team at {safe_path}:{number}')
            for name in (alias, canonical):
                existing = mapping.get(name)
                if existing is not None and existing != canonical:
                    raise ValueError(
                        f'Conflicting NCAAM team alias {name!r}: '
                        f'{existing!r} vs {canonical!r} at {safe_path}:{number}'
                    )
                mapping[name] = canonical
    if not mapping:
        raise ValueError(f'No NCAAM aliases loaded from {safe_path}')
    _ALIAS_MAPS[league_label] = mapping
    log(f'NCAAM validated alias keys loaded: {len(mapping)} from {safe_path}')
    return mapping


def canonical_team(team, league_label):
    if league_label != 'NCAAM':
        return clean_value(team).lower()
    name = normalize_team(team)
    return load_team_alias_map(league_label).get(name, name)


def make_key(game_date, home_team, away_team, league_label=None):
    date = normalize_date(game_date)
    home = canonical_team(home_team, league_label)
    away = canonical_team(away_team, league_label)
    if not date or not home or not away or home == away:
        return None
    return date, home, away


def ensure_fieldnames(fieldnames, wanted):
    fieldnames = list(fieldnames or [])
    for col in wanted:
        if col not in fieldnames:
            fieldnames.append(col)
    return fieldnames


def read_csv_rows(path: Path):
    safe_path = resolve_repository_path(path, strict=True)
    with safe_path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def write_csv_rows(path: Path, fieldnames, rows):
    safe_path = resolve_repository_path(path, strict=False)
    with safe_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _issue_row(league_label, path, row, reason, *, game_id=None):
    return {
        'league': league_label,
        'file': str(path),
        'source_file': str(path),
        'game_date': clean_value(row.get('game_date')),
        'home_team': clean_value(row.get('home_team')),
        'away_team': clean_value(row.get('away_team')),
        'game_id': clean_value(row.get('game_id')) if game_id is None else game_id,
        'reason': reason,
    }


def load_daily_games_for_league(daily_games_dir: Path, league_label: str):
    daily_map = {}
    daily_keys_by_date = defaultdict(set)
    duplicate_rows = []
    files_found = rows_loaded = 0
    entries = defaultdict(list)

    if not daily_games_dir.exists():
        log(f'DAILY GAMES DIR NOT FOUND: {daily_games_dir}')
        return daily_map, daily_keys_by_date, duplicate_rows, files_found, rows_loaded

    for csv_path in sorted(daily_games_dir.glob('*.csv')):
        files_found += 1
        try:
            rows, _ = read_csv_rows(csv_path)
            for row in rows:
                rows_loaded += 1
                key = make_key(
                    row.get('game_date'), row.get('home_team'),
                    row.get('away_team'), league_label,
                )
                gid = clean_value(row.get('game_id'))
                if key is None or not gid:
                    duplicate_rows.append(_issue_row(
                        league_label, csv_path, row, 'invalid_identity_or_missing_schedule_id'))
                    continue
                entries[key].append({
                    'game_id': gid,
                    'game_time': clean_value(row.get('game_time')),
                    'source_file': str(csv_path),
                    'game_date': key[0],
                    'home_team': clean_value(row.get('home_team')),
                    'away_team': clean_value(row.get('away_team')),
                })
        except Exception as exc:
            msg = f'ERROR loading daily games file {csv_path}: {exc}\n{traceback.format_exc()}'
            log(msg)
            ERRORS.append(msg)

    id_to_keys = defaultdict(set)
    for key, values in entries.items():
        for item in values:
            id_to_keys[item['game_id']].add(key)

    for key, values in entries.items():
        ids = {item['game_id'] for item in values}
        collision = len(ids) != 1 or any(len(id_to_keys[gid]) != 1 for gid in ids)
        if collision:
            reason = 'ambiguous_schedule_ids_or_game_id_reused'
            for item in values:
                duplicate_rows.append({
                    'league': league_label,
                    'game_date': key[0],
                    'home_team': item['home_team'],
                    'away_team': item['away_team'],
                    'game_id': item['game_id'],
                    'source_file': item['source_file'],
                    'reason': reason,
                })
            log(f'AMBIGUOUS SCHEDULE KEY SKIPPED: {league_label} {key} ids={sorted(ids)}')
            continue

        if len(values) > 1:
            # Multiple exports of the same date/matchup/ID are one game.
            log(f'DUPLICATE CONSISTENT SCHEDULE ROWS: {league_label} {key} count={len(values)}')

        # Never allow a blank time from a duplicate export to replace a known time.
        times = {item['game_time'] for item in values if item['game_time']}
        if len(times) > 1:
            log(f'SCHEDULE TIME CONFLICT (ID STILL UNIQUE): {league_label} {key} times={sorted(times)}')
        meta = dict(values[0])
        meta['game_time'] = sorted(times)[0] if len(times) == 1 else ''
        daily_map[key] = meta
        daily_keys_by_date[key[0]].add(key)

    return daily_map, daily_keys_by_date, duplicate_rows, files_found, rows_loaded


def track_duplicate_row(row, seen_keys, duplicate_rows, league_label, csv_path):
    key = make_key(
        row.get('game_date'), row.get('home_team'),
        row.get('away_team'), league_label,
    )
    if key is not None and key in seen_keys:
        duplicate_rows.append(_issue_row(
            league_label, csv_path, row, 'duplicate_source_matchup'))
    if key is not None:
        seen_keys.add(key)
    return key


def _match_reason(row, daily_map, league_label):
    key = make_key(
        row.get('game_date'), row.get('home_team'),
        row.get('away_team'), league_label,
    )
    if key is None:
        return None, 'invalid_date_or_team'
    if key in daily_map:
        return key, None
    reverse_key = key[0], key[2], key[1]
    if reverse_key in daily_map:
        # Assigning an ID without swapping side-specific probabilities/points
        # would link the wrong team to the model's home/away predictions.
        return None, 'reversed_home_away_not_auto_assigned'
    return None, 'no_validated_matchup_alias_or_schedule_game'


def update_predictions(predictions_dir: Path, daily_map: dict, league_label: str):
    files_processed = rows_processed = rows_updated = 0
    matched_keys = set()
    prediction_no_match = []
    duplicate_prediction_rows = []

    if not predictions_dir.exists():
        log(f'PREDICTIONS DIR NOT FOUND: {predictions_dir}')
        return (files_processed, rows_processed, rows_updated, matched_keys,
                prediction_no_match, duplicate_prediction_rows)

    for csv_path in sorted(predictions_dir.glob('*.csv')):
        try:
            files_processed += 1
            rows, fieldnames = read_csv_rows(csv_path)
            fieldnames = ensure_fieldnames(fieldnames, ['game_id', 'game_time'])
            keys = [make_key(row.get('game_date'), row.get('home_team'),
                             row.get('away_team'), league_label) for row in rows]
            counts = Counter(k for k in keys if k is not None)
            seen_keys = set()
            modified = False

            for row, candidate_key in zip(rows, keys):
                rows_processed += 1
                track_duplicate_row(row, seen_keys, duplicate_prediction_rows,
                                    league_label, csv_path)
                if candidate_key is not None and counts[candidate_key] > 1:
                    prediction_no_match.append(_issue_row(
                        league_label, csv_path, row, 'ambiguous_duplicate_prediction_matchup'))
                    continue
                key, reason = _match_reason(row, daily_map, league_label)
                if key is None:
                    prediction_no_match.append(_issue_row(league_label, csv_path, row, reason))
                    continue

                source = daily_map[key]
                matched_keys.add(key)
                source_gid = source['game_id']
                source_time = source['game_time']
                if clean_value(row.get('game_id')) != source_gid:
                    previous = clean_value(row.get('game_id'))
                    if previous:
                        log(f'CORRECTED PREDICTION GAME ID: {csv_path} {key} {previous} -> {source_gid}')
                    row['game_id'] = source_gid
                    rows_updated += 1
                    modified = True
                if source_time and clean_value(row.get('game_time')) != source_time:
                    row['game_time'] = source_time
                    rows_updated += 1
                    modified = True

            if modified:
                write_csv_rows(csv_path, fieldnames, rows)
                log(f'UPDATED PREDICTIONS: {csv_path}')
            else:
                log(f'NO CHANGES PREDICTIONS: {csv_path}')
        except Exception as exc:
            msg = f'ERROR updating predictions file {csv_path}: {exc}\n{traceback.format_exc()}'
            log(msg)
            ERRORS.append(msg)

    return (files_processed, rows_processed, rows_updated, matched_keys,
            prediction_no_match, duplicate_prediction_rows)


def update_final_scores(final_scores_dir: Path, daily_map: dict, league_label: str):
    files_processed = rows_processed = rows_updated = 0
    duplicate_final_score_rows = []
    if not final_scores_dir.exists():
        log(f'FINAL SCORES DIR NOT FOUND: {final_scores_dir}')
        return files_processed, rows_processed, rows_updated, duplicate_final_score_rows

    for csv_path in sorted(final_scores_dir.glob('*.csv')):
        try:
            files_processed += 1
            rows, fieldnames = read_csv_rows(csv_path)
            fieldnames = ensure_fieldnames(fieldnames, ['game_id'])
            keys = [make_key(row.get('game_date'), row.get('home_team'),
                             row.get('away_team'), league_label) for row in rows]
            counts = Counter(k for k in keys if k is not None)
            seen_keys = set()
            modified = False
            for row, candidate_key in zip(rows, keys):
                rows_processed += 1
                track_duplicate_row(row, seen_keys, duplicate_final_score_rows,
                                    league_label, csv_path)
                if candidate_key is not None and counts[candidate_key] > 1:
                    continue
                key, reason = _match_reason(row, daily_map, league_label)
                if key is None:
                    if reason == 'reversed_home_away_not_auto_assigned':
                        log(f'REVERSED FINAL SCORE MATCHUP SKIPPED: {csv_path}: '
                            f'{row.get("home_team")} vs {row.get("away_team")}')
                    continue
                source_gid = daily_map[key]['game_id']
                if clean_value(row.get('game_id')) != source_gid:
                    previous = clean_value(row.get('game_id'))
                    if previous:
                        log(f'CORRECTED FINAL SCORE GAME ID: {csv_path} {key} {previous} -> {source_gid}')
                    row['game_id'] = source_gid
                    rows_updated += 1
                    modified = True
            if modified:
                write_csv_rows(csv_path, fieldnames, rows)
                log(f'UPDATED FINAL SCORES: {csv_path}')
            else:
                log(f'NO CHANGES FINAL SCORES: {csv_path}')
        except Exception as exc:
            msg = f'ERROR updating final scores file {csv_path}: {exc}\n{traceback.format_exc()}'
            log(msg)
            ERRORS.append(msg)
    return files_processed, rows_processed, rows_updated, duplicate_final_score_rows


def main():
    totals = defaultdict(int)
    sportsbook_no_prediction_match = []
    prediction_no_sportsbook_match = []
    duplicate_key_rows = []
    ERRORS.clear()

    try:
        for cfg in LEAGUES.values():
            league_label = cfg['league_label']
            log(f'--- LEAGUE {league_label} ---')
            if league_label == 'NCAAM':
                load_team_alias_map(league_label)

            (daily_map, _, daily_duplicates, daily_files, daily_rows) = (
                load_daily_games_for_league(cfg['daily_games_dir'], league_label))
            totals['daily_files'] += daily_files
            totals['daily_rows'] += daily_rows
            duplicate_key_rows.extend(daily_duplicates)

            (prediction_files, prediction_rows, prediction_updates,
             matched_keys, prediction_mismatches, prediction_duplicates) = (
                update_predictions(cfg['predictions_dir'], daily_map, league_label))
            totals['prediction_files'] += prediction_files
            totals['prediction_rows'] += prediction_rows
            totals['prediction_updates'] += prediction_updates
            prediction_no_sportsbook_match.extend(prediction_mismatches)
            duplicate_key_rows.extend(prediction_duplicates)

            (score_files, score_rows, score_updates, score_duplicates) = (
                update_final_scores(cfg['final_scores_dir'], daily_map, league_label))
            totals['score_files'] += score_files
            totals['score_rows'] += score_rows
            totals['score_updates'] += score_updates
            duplicate_key_rows.extend(score_duplicates)

            for key in sorted(set(daily_map) - matched_keys):
                meta = daily_map[key]
                sportsbook_no_prediction_match.append({
                    'league': league_label,
                    'source_file': meta['source_file'],
                    'game_date': key[0],
                    'home_team': meta['home_team'],
                    'away_team': meta['away_team'],
                    'game_id': meta['game_id'],
                })

        log('--- SUMMARY ---')
        log(f"Daily games files found: {totals['daily_files']}")
        log(f"Daily games rows loaded: {totals['daily_rows']}")
        log(f"Prediction files processed: {totals['prediction_files']}")
        log(f"Prediction rows processed: {totals['prediction_rows']}")
        log(f"Prediction field updates made: {totals['prediction_updates']}")
        log(f"Final score files processed: {totals['score_files']}")
        log(f"Final score rows processed: {totals['score_rows']}")
        log(f"Final score field updates made: {totals['score_updates']}")
        log(f'Sportsbook rows with no prediction match: {len(sportsbook_no_prediction_match)}')
        log(f'Prediction rows with no sportsbook match: {len(prediction_no_sportsbook_match)}')
        log(f'Duplicate key rows: {len(duplicate_key_rows)}')
        log('--- SMALL MISMATCH REPORT ---')

        log('Sportsbook rows with no prediction match:')
        for row in sportsbook_no_prediction_match[:25]:
            log(f"  {row['league']} | {row['game_date']} | {row['home_team']} | "
                f"{row['away_team']} | game_id={row['game_id']} | source={row['source_file']}")
        if not sportsbook_no_prediction_match:
            log('  none')
        elif len(sportsbook_no_prediction_match) > 25:
            log(f'  ... and {len(sportsbook_no_prediction_match) - 25} more')

        log('Prediction rows with no sportsbook match:')
        for row in prediction_no_sportsbook_match[:25]:
            log(f"  {row['league']} | {row['game_date']} | {row['home_team']} | "
                f"{row['away_team']} | reason={row['reason']} | file={row['file']}")
        if not prediction_no_sportsbook_match:
            log('  none')
        elif len(prediction_no_sportsbook_match) > 25:
            log(f'  ... and {len(prediction_no_sportsbook_match) - 25} more')

        log('Duplicate key rows:')
        for row in duplicate_key_rows[:25]:
            log(f"  {row['league']} | {row['game_date']} | {row['home_team']} | "
                f"{row['away_team']} | reason={row['reason']} | source={row['source_file']}")
        if not duplicate_key_rows:
            log('  none')
        elif len(duplicate_key_rows) > 25:
            log(f'  ... and {len(duplicate_key_rows) - 25} more')

        if ERRORS:
            raise RuntimeError(f'{len(ERRORS)} file processing error(s); see {LOG_FILE}')
        log('STATUS: SUCCESS')
        print('Basketball game_id injection complete.')
    except Exception as exc:
        log(f'FATAL ERROR: {exc}\n{traceback.format_exc()}')
        log('STATUS: FAILED')
        raise


if __name__ == '__main__':
    main()
