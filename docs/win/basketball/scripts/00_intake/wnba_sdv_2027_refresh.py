#!/usr/bin/env python3
"""2027 WNBA daily ESPN refresh, fail-closed and separate from the verified 2026 updater.

Only 2027 regular/playoff WNBA club games. Reads completed 2026 Parquet as schema
and 2027 backfill context. Writes 2027 tables, 2027 canonical CSV, and external
backup/status records upon explicit non-dry-run invocation. No GitHub operations.

2027 roster expansions are NOT auto-accepted: unknown team identities cause an
error until separately reviewed in the ESPN parser's allowlist.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
from wnba_2026_espn_parser import TEAM_NAMES, norm, get_json, iso_date, scoreboard_event, summary_tables
from wnba_sdv_current_refresh import FIELDS, fit_schema, fullpath

YEAR=2027
NY=ZoneInfo('America/New_York')
START_MONTH=4
LOOKBACK=21
LOOKAHEAD=14
MIN_TIPOFF_LAG_HOURS=8
FULL_SCAN_INTERVAL_DAYS=7


def backup_dir():
    return Path.home() / 'Downloads' / 'WNBA_SDV_Daily_Backups'


def fetch_2027(now: datetime, start: date, end: date):
    result={}
    if start>end:
        return result
    for offset in range((end-start).days+1):
        day=start+timedelta(days=offset)
        if day.year != YEAR:
            continue
        url=('https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates='
             +day.strftime('%Y%m%d')+'&limit=500')
        payload=get_json(url)
        if not isinstance(payload,dict) or not isinstance(payload.get('events'),list):
            raise RuntimeError('ESPN missing WNBA events array for '+str(day))
        for event in payload['events']:
            if not isinstance(event,dict):
                raise RuntimeError('ESPN event not object for '+str(day))
            game_id=str(event.get('id') or '')
            if not game_id:
                raise RuntimeError('Blank ESPN event ID for '+str(day))
            if str((event.get('season') or {}).get('type')) not in ('2','3'):
                continue
            comp=(event.get('competitions') or [None])[0]
            if not isinstance(comp,dict):
                raise RuntimeError('Missing ESPN WNBA competition for '+game_id)
            sides={x.get('homeAway'):x for x in comp.get('competitors') or [] if isinstance(x,dict)}
            if set(sides)!={'home','away'}:
                raise RuntimeError('Missing ESPN WNBA home/away for '+game_id)
            home=sides['home'].get('team') or {}
            away=sides['away'].get('team') or {}
            # WNBA All-Star is sometimes incorrectly marked season_type=2.
            if norm(home.get('displayName')) not in TEAM_NAMES or norm(away.get('displayName')) not in TEAM_NAMES:
                if 'all-star' in str(home.get('displayName','')).lower() or 'all-star' in str(away.get('displayName','')).lower() or 'team ' in str(home.get('displayName','')).lower() or 'team ' in str(away.get('displayName','')).lower():
                    continue
                raise RuntimeError('2027 club not in reviewed WNBA allowlist: '+str(home.get('displayName'))+' / '+str(away.get('displayName'))+' event='+game_id)
            if not iso_date(event['date']).startswith('2027_'):
                continue
            prev=result.get(game_id)
            if prev is not None and prev['date']!=event['date']:
                raise RuntimeError('ESPN duplicate event with conflicting start '+game_id)
            result[game_id]=event
    return result


def canonical_row(event: dict, now: datetime):
    comp=event['competitions'][0]
    sides={x['homeAway']:x for x in comp['competitors']}
    tip=datetime.fromisoformat(event['date'].replace('Z','+00:00')).astimezone(NY)
    venue=comp.get('venue') or {}
    status=(event.get('status') or {}).get('type') or {}
    return {
        'sport':'basketball','league':'WNBA','internal_season':'2027','sdv_season':'2027',
        'game_date':tip.strftime('%Y_%m_%d'),'game_time':tip.strftime('%I:%M %p'),
        'home_team':str(sides['home']['team']['displayName']),
        'away_team':str(sides['away']['team']['displayName']),
        'game_id':str(event['id']),
        'home_team_id':str(sides['home']['team']['id']),
        'away_team_id':str(sides['away']['team']['id']),
        'neutral_site':'true' if comp.get('neutralSite') else 'false',
        'venue_id':str(venue.get('id') or ''),
        'venue_full_name':str(venue.get('fullName') or ''),
        'venue_name':str(venue.get('fullName') or ''),
        'status':str(status.get('description') or status.get('name') or ''),
        'source':'espn_daily_wnba_2027_verified', 'fetched_at_utc':now.isoformat(),
    }


def read_canonical(path):
    if not path.is_file():
        return {}
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise RuntimeError('2027 canonical CSV schema changed; refuse update')
        rows={}
        for r in reader:
            eid=r.get('game_id','')
            if not eid or eid in rows or not str(r.get('game_date','')).startswith('2027_'):
                raise RuntimeError('2027 canonical identity duplicated or wrong year')
            rows[eid]=r
        return rows


def validate_history(games,team,players):
    ids=[str(x) for x in games['game_id'].to_list()]
    if len(ids)!=len(set(ids)):
        raise RuntimeError('2027 duplicate game IDs')
    if any(not str(x).startswith('2027_') for x in games['game_date'].to_list()):
        raise RuntimeError('2027 history contains non-2027 game date')
    gids=set(ids)
    from collections import Counter
    counted=Counter(str(x) for x in team['game_id'].to_list())
    if set(counted)!=gids or any(n!=2 for n in counted.values()):
        raise RuntimeError('2027 team statistics not exactly 2 per completed game')
    p_ids={str(x) for x in players['game_id'].to_list()}
    if p_ids!=gids:
        raise RuntimeError('2027 player rows missing for completed games')
    if any(str(x).startswith('WNBA_OFFICIAL_UNMAPPED') for x in players['player_id'].to_list()):
        raise RuntimeError('Unresolved 2027 player ID')


def atomic_update(entries):
    """Atomic per-file replace; whole batch restored if a replacement fails.

    Existing files get durable copies under Downloads, newly created ones are
    deleted if any step fails. Never create transient/backup copies inside GitHub.
    """
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backups=backup_dir() / ('2027_TRANSACTION_'+stamp)
    backups.mkdir(parents=True,exist_ok=False)
    staged=[]; before=[]; installed=[]
    try:
        for target, writer in entries:
            target.parent.mkdir(parents=True,exist_ok=True)
            tmp=target.with_name(target.name+'.wnba_2027_staged_'+stamp)
            writer(tmp)
            staged.append((target,tmp))
        for idx,(target,_) in enumerate(staged):
            existed=target.is_file()
            bp=backups / (str(idx)+'_'+target.name)
            if existed:shutil.copy2(target,bp)
            before.append((target,existed,bp))
        for target,tmp in staged:
            os.replace(tmp,target)
            installed.append(target)
    except BaseException:
        for target,existed,bp in reversed(before):
            if target in installed:
                if existed:shutil.copy2(bp,target)
                else:target.unlink(missing_ok=True)
        raise
    finally:
        for _,tmp in staged:
            tmp.unlink(missing_ok=True)
    return str(backups)


def refresh(repo_root=None, *, dry_run=False, lookback=LOOKBACK, lookahead=LOOKAHEAD, now=None, full_scan=False):
    root=fullpath(repo_root or Path.cwd())
    now=now or datetime.now(timezone.utc)
    if now.tzinfo is None:raise ValueError('Clock must include UTC offset')
    today=now.astimezone(NY).date()
    if today.year != YEAR:
        # Explicitly do not prebuild 2027 in 2026 or run it during 2028.
        return {'status':'outside_2027','current_date':str(today),'repo_modified':False}
    if lookback<1 or lookahead<0 or lookahead>31:
        raise ValueError('Invalid scoreboard range')
    prior_dir=root/'docs/win/basketball/00_intake/sdv/history/wnba/2026'
    output_dir=root/'docs/win/basketball/00_intake/sdv/history/wnba/2027'
    can=root/'docs/win/basketball/00_intake/sdv/canonical_games/wnba/2027_WNBA_games.csv'
    templates=[prior_dir/(name+'.parquet') for name in ('games','team_game','player_game')]
    for t in templates:
        if not t.is_file():raise RuntimeError('Verified 2026 history required before 2027')
    existing=[output_dir/(name+'.parquet') for name in ('games','team_game','player_game')]
    existing_flags=[x.is_file() for x in existing]
    if any(existing_flags) and not all(existing_flags):
        raise RuntimeError('Partial 2027 history partition; refuse update')
    tables=[pl.read_parquet(x) for x in (existing if all(existing_flags) else templates)]
    if not all(existing_flags): tables=[t.head(0) for t in tables]
    games,teams,players=tables
    validate_history(games,teams,players)
    old_ids={str(x) for x in games['game_id'].to_list()}
    old_scores={str(r['game_id']):(int(r['home_score']),int(r['away_score'])) for r in games.select(['game_id','home_score','away_score']).to_dicts()}
    canon=read_canonical(can)
    start=date(YEAR,START_MONTH,1)
    end=min(today+timedelta(days=lookahead),date(YEAR,12,31))
    if end < start:
        return {'status':'preseason_no_window','repo_modified':False,'history_games':games.height}
    full_stamp=backup_dir()/'2027_LAST_FULL_SCAN.json'
    if not full_scan and not can.exists():full_scan=True
    if not full_scan and full_stamp.is_file():
        try:
            payload=json.loads(full_stamp.read_text(encoding='utf8'))
            dt=datetime.fromisoformat(payload['finished_at_utc'])
            if now-dt >= timedelta(days=FULL_SCAN_INTERVAL_DAYS):full_scan=True
        except (OSError,KeyError,TypeError,ValueError):full_scan=True
    if not full_scan and not full_stamp.exists():full_scan=True
    first=start if full_scan else max(start,today-timedelta(days=lookback))
    events=fetch_2027(now,first,end)
    known_in_window={str(r['game_id']) for r in games.select(['game_id','game_date']).to_dicts() if first.strftime('%Y_%m_%d') <= str(r['game_date']) <= end.strftime('%Y_%m_%d')}
    missing=known_in_window-set(events)
    if missing:raise RuntimeError('ESPN 2027 scoreboard omitted staged completed games: '+repr(sorted(missing)[:15]))
    new_game=[];new_team=[];new_player=[];canon_changed=False
    for gid,ev in sorted(events.items()):
        r=canonical_row(ev,now)
        prev=canon.get(gid)
        if prev and any(prev[k]!=r[k] for k in ('home_team_id','away_team_id','game_date')):
            raise RuntimeError('2027 canonical identity conflict '+gid)
        if not prev or any(prev.get(k)!=r.get(k) for k in ('game_date','game_time','home_team_id','away_team_id','status')):
            canon_changed=True
        canon[gid]=r
        if (ev.get('status') or {}).get('type',{}).get('completed') is not True:
            continue
        parsed=scoreboard_event(ev,[])
        if not parsed:raise RuntimeError('Cannot parse completed ESPN scoreboard game '+gid)
        score=(parsed['home_score'],parsed['away_score'])
        if gid in old_ids:
            if score!=old_scores[gid]:raise RuntimeError('2027 completed game score changed '+gid)
            continue
        tip=datetime.fromisoformat(ev['date'].replace('Z','+00:00'))
        if now-tip < timedelta(hours=MIN_TIPOFF_LAG_HOURS):
            continue
        raw=get_json('https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary?event='+gid)
        issues=[]
        parsed_rows=summary_tables(raw,parsed,issues)
        if parsed_rows is None or any(x.get('severity')=='error' for x in issues):
            raise RuntimeError('Invalid 2027 ESPN boxscore '+gid+' '+repr(issues[:3]))
        gr,trs,prs=parsed_rows
        if len(trs)!=2 or len(prs)<10:
            raise RuntimeError('Insufficient 2027 boxscore rows '+gid)
        new_game.append(gr);new_team.extend(trs);new_player.extend(prs)
    if new_game:
        games=fit_schema(new_game,games)
        teams=fit_schema(new_team,teams)
        players=fit_schema(new_player,players)
    validate_history(games,teams,players)
    if any(gid not in canon for gid in games['game_id'].to_list()):
        raise RuntimeError('Historical game absent from 2027 canonical')
    result={'status':'dry_run' if dry_run else 'refreshed','season':YEAR,
            'completed_history':games.height,'added_history':len(new_game),
            'team_rows':teams.height,'player_rows':players.height,
            'canonical_games':len(canon),'api_events':len(events),
            'scan_start':str(first),'scan_end':str(end),
            'full_scan':bool(full_scan),'repo_modified':False,
            'source_publication_timestamps_verified':False}
    if dry_run:return result
    # Never emit an empty 2027 partition without a dated WNBA club fixture.
    if not canon:return result
    entries=[]
    if new_game or not all(existing_flags):
        entries.extend((p,lambda dest,table=t:table.write_parquet(dest,compression='zstd'))
                       for p,t in zip(existing,[games,teams,players]))
    if canon_changed or not can.is_file():
        def write_csv(dest):
            with dest.open('w',newline='',encoding='utf-8') as fp:
                writer=csv.DictWriter(fp,fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(sorted(canon.values(),key=lambda r:(r['game_date'],r['game_time'],r['game_id'])))
        entries.append((can,write_csv))
    if entries:
        result['backup_directory']=atomic_update(entries)
        result['repo_modified']=True
    if full_scan:
        full_stamp.parent.mkdir(parents=True,exist_ok=True)
        full_stamp.write_text(json.dumps({'finished_at_utc':now.isoformat(),'games':games.height},indent=2),encoding='utf-8')
    return result


def cli():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,default=Path.cwd())
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--full-scan',action='store_true')
    ns=ap.parse_args()
    print(json.dumps(refresh(ns.repo,dry_run=ns.dry_run,full_scan=ns.full_scan),indent=2))

if __name__=='__main__':cli()
