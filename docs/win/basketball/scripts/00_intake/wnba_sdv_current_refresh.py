#!/usr/bin/env python3
"""Fail-closed WNBA 2026 scoreboard + completed boxscore refresh.

Called from local sdv_canonical_games / sdv_predict before WNBA forecasts.
Existing 2026 tables are never fetched from year-level truncated SDV data.
Maintains pregame-only availability via completed status + 8h tipoff lag.
2027+ is intentionally out of scope and must be configured separately.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
from wnba_2026_espn_parser import TEAM_NAMES, get_json, iso_date, norm, scoreboard_event, summary_tables

YEAR = 2026
LOOKBACK = 21
LOOKAHEAD = 14
MIN_TIPOFF_LAG_HOURS = 8
FIELDS = ["sport", "league", "internal_season", "sdv_season", "game_date", "game_time",
          "home_team", "away_team", "game_id", "home_team_id", "away_team_id", "neutral_site",
          "venue_id", "venue_full_name", "venue_name", "status", "source", "fetched_at_utc"]


def fullpath(repo_root):
    root = Path(repo_root).resolve()
    if not (root / 'docs/win/basketball/config/sdv_model.yaml').is_file():
        raise RuntimeError('Expected repository root missing: ' + str(root))
    return root


def read_canonical(path):
    with path.open('r', encoding='utf-8-sig', newline='') as fp:
        rd = csv.DictReader(fp)
        if set(rd.fieldnames or []) != set(FIELDS):
            raise RuntimeError('Canonical field names differ; refuse update')
        rows = {r['game_id']: r for r in rd if r.get('game_id')}
    return rows


def fetch_window(now, lookback=LOOKBACK, lookahead=LOOKAHEAD):
    from zoneinfo import ZoneInfo
    et_today = now.astimezone(ZoneInfo('America/New_York')).date()
    result = {}
    for delta in range(-lookback, lookahead + 1):
        day = et_today + timedelta(days=delta)
        if day.year != YEAR: continue
        data = get_json('https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates='
                        + day.strftime('%Y%m%d') + '&limit=500')
        if not isinstance(data.get('events'), list):
            raise RuntimeError('ESPN scoreboard response missing events on ' + str(day))
        for ev in data['events']:
            eid = str(ev.get('id') or '')
            if not eid: raise RuntimeError('Blank ESPN event id')
            if str((ev.get('season') or {}).get('type')) not in ('2','3'):
                continue
            competition = (ev.get('competitions') or [None])[0]
            if not isinstance(competition, dict): raise RuntimeError('Bad scoreboard competition '+eid)
            sides = {c.get('homeAway'): c for c in competition.get('competitors') or []}
            if set(sides) != {'home', 'away'}: raise RuntimeError('Incomplete teams on event '+eid)
            home = sides['home'].get('team') or {}
            away = sides['away'].get('team') or {}
            if norm(home.get('displayName')) not in TEAM_NAMES or norm(away.get('displayName')) not in TEAM_NAMES:
                continue    # explicitly exclude nonclub, All-Star, international exhibitions
            game_day = iso_date(ev['date'])
            if not game_day.startswith(str(YEAR) + '_'): continue
            prev = result.get(eid)
            if prev is not None and prev['date'] != ev['date']:
                raise RuntimeError('Conflicting event time from ESPN '+eid)
            result[eid] = ev
    return result


def event_canonical(ev, now):
    comp = ev['competitions'][0]
    sides = {c['homeAway']:c for c in comp['competitors']}
    utc_start = datetime.fromisoformat(ev['date'].replace('Z', '+00:00'))
    from zoneinfo import ZoneInfo
    local = utc_start.astimezone(ZoneInfo('America/New_York'))
    venue = comp.get('venue') or {}
    status = (ev.get('status') or {}).get('type') or {}
    return {
        'sport':'basketball','league':'WNBA','internal_season':'2026','sdv_season':'2026',
        'game_date':local.strftime('%Y_%m_%d'),'game_time':local.strftime('%I:%M %p'),
        'home_team':sides['home']['team']['displayName'],
        'away_team':sides['away']['team']['displayName'],
        'game_id':str(ev['id']),
        'home_team_id':str(sides['home']['team']['id']),
        'away_team_id':str(sides['away']['team']['id']),
        'neutral_site':'true' if comp.get('neutralSite') else 'false',
        'venue_id':str(venue.get('id') or ''), 'venue_full_name':str(venue.get('fullName') or ''),
        'venue_name':str(venue.get('fullName') or ''),
        'status':str(status.get('description') or status.get('name') or ''),
        'source':'espn_daily_wnba_verified', 'fetched_at_utc':now.isoformat(),
    }


def fit_schema(rows, template):
    if not rows: return template
    columns = template.schema
    df = pl.DataFrame(rows, infer_schema_length=None, strict=False)
    for c, ty in columns.items():
        if c not in df.columns: df = df.with_columns(pl.lit(None, dtype=ty).alias(c))
        else: df = df.with_columns(pl.col(c).cast(ty, strict=True))
    extra = set(df.columns)-set(columns)
    if extra: raise RuntimeError('ESPN output unexpected new columns: '+str(sorted(extra)))
    df = df.select(list(columns))
    return pl.concat([template,df], how='vertical')


def check_history(games, team, players):
    ids=[str(x) for x in games['game_id'].to_list()]
    if len(ids) < 350 or len(ids)!=len(set(ids)):
        raise RuntimeError('Missing/duplicate historical WNBA games')
    gids=set(ids)
    from collections import Counter
    teams=Counter(str(x) for x in team['game_id'].to_list())
    if set(teams)!=gids or any(v!=2 for v in teams.values()):
        raise RuntimeError('History must have two team rows per game')
    pgids={str(x) for x in players['game_id'].to_list()}
    if pgids!=gids: raise RuntimeError('History player coverage differs from games')
    if any(str(x).startswith('WNBA_OFFICIAL_UNMAPPED') for x in players['player_id'].to_list()):
        raise RuntimeError('Unresolved player aliases')
    if '401857114' not in gids: raise RuntimeError('Official August 4 dispute resolution missing')
    aug=games.filter(pl.col('game_id').cast(pl.Utf8)=='401857114')
    if aug.height!=1 or int(aug['home_score'][0])!=92 or int(aug['away_score'][0])!=81:
        raise RuntimeError('Official August 4 score guard failed')


def atomic_update(file_contents):
    """Rollback on any mid-transaction write failure. Store backup copies alongside originals."""
    txid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup_root=Path.home()/'Downloads'/'WNBA_SDV_Daily_Backups'
    backup_root.mkdir(parents=True,exist_ok=True)
    backed=[]; staged=[]; installed=[]
    try:
        for path, writer in file_contents:
            tmp = path.with_name(path.name+'.wnba-staged-'+txid)
            writer(tmp)
            staged.append((path,tmp))
        for path, tmp in staged:
            backup=backup_root/(path.name+'.wnba-before-'+txid)
            shutil.copy2(path,backup)
            backed.append((path,backup))
        for path,tmp in staged:
            os.replace(tmp,path)
            installed.append(path)
        # keep transaction backup files for post-incident restoration
    except BaseException:
        for path, backup in reversed(backed):
            if path in installed and backup.exists():
                shutil.copy2(backup,path)
        raise
    finally:
        for path,tmp in staged:
            if tmp.exists():tmp.unlink()


def refresh(repo_root=None, *, dry_run=False, lookback=LOOKBACK, lookahead=LOOKAHEAD):
    """Refresh 2026 WNBA statistics and canonical schedule; fail closed."""
    root = fullpath(repo_root or Path.cwd())
    hist=root/'docs/win/basketball/00_intake/sdv/history/wnba/2026'
    canonical=root/'docs/win/basketball/00_intake/sdv/canonical_games/wnba/2026_WNBA_games.csv'
    if not hist.is_dir() or not canonical.is_file():
        raise RuntimeError('2026 history not installed; deploy reviewed 350-game baseline first')
    stamp=Path.home()/'Downloads'/'WNBA_SDV_Daily_Backups'/'LAST_SUCCESSFUL_REFRESH.json'
    now=datetime.now(timezone.utc)
    if not dry_run and stamp.exists():
        try:
            prior=datetime.fromisoformat(json.loads(stamp.read_text(encoding='utf8'))['completed_at_utc'])
            if now-prior < timedelta(minutes=40):return {'status':'recently_refreshed'}
        except (ValueError,KeyError,OSError): pass
    paths=[hist/(name+'.parquet') for name in ('games','team_game','player_game')]
    games, team, players=[pl.read_parquet(path) for path in paths]
    check_history(games,team,players)
    old_ids={str(x) for x in games['game_id'].to_list()}
    old_scores={str(r['game_id']):(int(r['home_score']),int(r['away_score'])) for r in games.select(['game_id','home_score','away_score']).to_dicts()}
    canon=read_canonical(canonical)
    events=fetch_window(now,lookback,lookahead)
    from zoneinfo import ZoneInfo
    et_today=now.astimezone(ZoneInfo('America/New_York')).date()
    start=(et_today-timedelta(days=lookback)).strftime('%Y_%m_%d')
    end=(et_today+timedelta(days=lookahead)).strftime('%Y_%m_%d')
    known_in_window={str(x['game_id']) for x in games.select(['game_id','game_date']).to_dicts()
                     if start<=str(x['game_date'])<=end}
    unreturned=known_in_window-set(events)
    if unreturned:
        raise RuntimeError('ESPN date-scoped scoreboard omitted staged completed games: '+repr(sorted(unreturned)[:15]))
    added_games=[];added_team=[];added_players=[];new_rows={}
    canonical_changed=False
    for eid,ev in sorted(events.items()):
        row=event_canonical(ev,now)
        prev=canon.get(eid)
        if prev and (prev['home_team_id']!=row['home_team_id'] or prev['away_team_id']!=row['away_team_id'] or prev['game_date']!=row['game_date']):
            raise RuntimeError('Canonical game identity conflict: '+eid)
        new_rows[eid]=row
        if not prev or any(prev.get(k)!=row.get(k) for k in
            ('game_date','game_time','home_team_id','away_team_id','status')):
            canonical_changed=True
        completed=(ev.get('status') or {}).get('type',{}).get('completed') is True
        if not completed: continue
        parsed=scoreboard_event(ev,[])
        if not parsed:raise RuntimeError('Completed scoreboard game could not parse: '+eid)
        newscore=(parsed['home_score'],parsed['away_score'])
        if eid in old_ids:
            if old_scores[eid]!=newscore:raise RuntimeError('Historical score changed; quarantine and investigate: '+eid)
            continue
        utc_start=datetime.fromisoformat(ev['date'].replace('Z','+00:00'))
        if now-utc_start<timedelta(hours=MIN_TIPOFF_LAG_HOURS):
            continue  # do not treat potentially unpublished boxscores as available
        raw=get_json('https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary?event='+eid)
        flags=[]
        parsed_rows=summary_tables(raw,parsed,flags)
        if parsed_rows is None or any(f.get('severity')=='error' for f in flags):
            raise RuntimeError('Incomplete/disputed new WNBA boxscore: '+eid+' '+str(flags[:3]))
        gr, trs, prs=parsed_rows
        if len(trs)!=2 or len(prs)<10:raise RuntimeError('Invalid new team/player groups '+eid)
        added_games.append(gr);added_team.extend(trs);added_players.extend(prs)
    if added_games:
        games=fit_schema(added_games,games)
        team=fit_schema(added_team,team)
        players=fit_schema(added_players,players)
    check_history(games,team,players)
    canon.update(new_rows)
    gids=[r['game_id'] for r in canon.values()]
    if len(gids)!=len(set(gids)) or len(canon)<350:raise RuntimeError('Canonical schedule truncated')
    if not old_ids.issubset(set(str(x) for x in games['game_id'].to_list())):
        raise RuntimeError('Historical game loss')
    result={'status':'dry_run' if dry_run else 'updated','completed_history':games.height,
            'added_history':len(added_games),'canonical_games':len(canon),'daily_events':len(events),
            'lookback_days':lookback,'lookahead_days':lookahead,
            'as_of_utc':now.isoformat(),'season_limit':'2026 ONLY'}
    if dry_run:return result
    if len(added_games) or canonical_changed:
        def save_csv(path):
            with path.open('w',newline='',encoding='utf-8') as fp:
                writer=csv.DictWriter(fp, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(sorted(canon.values(),key=lambda x:(x['game_date'],x['game_time'],x['game_id'])))
        payloads=[]
        if added_games:
            payloads.extend((path,lambda p,table=t:table.write_parquet(p,compression='zstd'))
                            for path,t in zip(paths,[games,team,players]))
        payloads.append((canonical,save_csv))
        atomic_update(payloads)
    stamp.parent.mkdir(parents=True,exist_ok=True)
    stamp.write_text(json.dumps({'completed_at_utc':now.isoformat(),**result},indent=2),encoding='utf-8')
    return result


def cli():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,default=Path.cwd())
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--lookback',type=int,default=LOOKBACK)
    ap.add_argument('--lookahead',type=int,default=LOOKAHEAD)
    args=ap.parse_args()
    if args.lookback<1 or args.lookahead<0:ap.error('Invalid date window')
    print(json.dumps(refresh(args.repo,dry_run=args.dry_run,lookback=args.lookback,lookahead=args.lookahead),indent=2))

if __name__=='__main__':cli()
