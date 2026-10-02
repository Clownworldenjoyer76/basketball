#!/usr/bin/env python3
"""Fetch completed NBA/NCAAM/WNBA final scores from ESPN.

Default behavior fetches the current New York date plus the previous 7 dates so
a transient missed workflow run does not permanently create a grading hole.

Examples:
    python docs/win/basketball/scripts/05_final_scores/00_basketball_final_scores.py
    python docs/win/basketball/scripts/05_final_scores/00_basketball_final_scores.py --date 2026_08_20
    python docs/win/basketball/scripts/05_final_scores/00_basketball_final_scores.py --start-date 2026_08_20 --end-date 2026_08_22
"""
from __future__ import annotations

import argparse
import csv
import http.client
import json
import sys
import urllib.parse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

BASE = Path("docs/win/basketball")
RESULTS_BASE = BASE / "05_final_scores/results"
ERROR_DIR = BASE / "errors/05_final_scores"
LOG_FILE = ERROR_DIR / "00_basketball_final_scores.txt"
NY = ZoneInfo("America/New_York")

LEAGUES = {
    "nba": {
        "espn_slug": "nba",
        "label": "NBA",
    },
    "ncaam": {
        "espn_slug": "mens-college-basketball",
        "label": "NCAAM",
    },
    "wnba": {
        "espn_slug": "wnba",
        "label": "WNBA",
    },
}

OUTPUT_COLUMNS = [
    "sport",
    "league",
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "total",
    "home_spread",
    "away_spread",
]


def now_utc() -> str:
    return datetime.now(tz=ZoneInfo("UTC")).isoformat()


def log(message: str, level: str = "INFO") -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{now_utc()} | {level:<5} | {message.rstrip()}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_date_value(value: str) -> date:
    text = str(value).strip()
    for fmt in ("%Y_%m_%d", "%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(
        f"Invalid date {value!r}; expected YYYY_MM_DD or YYYY-MM-DD"
    )


def date_range(start: date, end: date):
    if end < start:
        raise ValueError("end date is before start date")
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def score_value(value: Any) -> int | None:
    if isinstance(value, dict):
        value = value.get("displayValue") or value.get("value")
    try:
        return int(float(str(value).strip()))
    except (ValueError, OverflowError):
        return None


def fetch_json(url: str) -> dict:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(f"Only HTTPS URLs are permitted: {url}")

    target = parsed.path or "/"
    if parsed.query:
        target = f"{target}?{parsed.query}"

    connection = http.client.HTTPSConnection(
        parsed.hostname,
        parsed.port or 443,
        timeout=30,
    )
    try:
        connection.request(
            "GET",
            target,
            headers={
                "User-Agent": "Mozilla/5.0 basketball-final-scores/1.0",
                "Accept": "application/json",
            },
        )
        response = connection.getresponse()
        payload = response.read()
        if not 200 <= response.status < 300:
            raise RuntimeError(f"HTTP {response.status} fetching {url}")
        return json.loads(payload.decode("utf-8"))
    except (OSError, http.client.HTTPException) as exc:
        raise RuntimeError(f"Network error fetching {url}: {exc}") from exc
    finally:
        connection.close()


def scoreboard_url(espn_slug: str, game_date: date) -> str:
    query = urllib.parse.urlencode(
        {
            "dates": game_date.strftime("%Y%m%d"),
            "limit": 500,
        }
    )
    return (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/"
        f"{espn_slug}/scoreboard?{query}"
    )


def parse_completed_games(payload: dict, league_key: str, game_date: date) -> list[dict]:
    _py_r1000_NONE = 0
    _py_r1000_RETURN = 1
    _py_r1000_BREAK = 2
    _py_r1000_CONTINUE = 3

    def _py_r1000_impl():
        nonlocal game_date, league_key, payload
        away: object
        away_score: object
        away_team: object
        competition: object
        competitions: object
        competitor: object
        completed: object
        event: object
        game_id: object
        home: object
        home_away: object
        home_score: object
        home_team: object
        league_label: object
        margin: object
        rows: object
        state: object
        status: object
        status_type: object

        def _py_r1000_loop_1():
            nonlocal away, away_score, away_team, competition, competitions, competitor, completed, game_id, home, home_away, home_score, home_team, margin, state, status, status_type

            def _py_r1000_loop_2():
                nonlocal away, home, home_away

                def _py_r1000_else_3():
                    nonlocal away
                    if home_away == 'away':
                        away = competitor
                    return (_py_r1000_NONE, None)
                home_away = str(competitor.get('homeAway') or '').strip().lower()
                if home_away == 'home':
                    home = competitor
                else:
                    _py_r1000_result_4 = _py_r1000_else_3()
                    if _py_r1000_result_4[0] != _py_r1000_NONE:
                        return _py_r1000_result_4
                return (_py_r1000_NONE, None)

            def _py_r1000_chunk_6():
                nonlocal competition, competitions, completed, state, status, status_type
                competitions = event.get('competitions') or []
                if not competitions:
                    return (_py_r1000_CONTINUE, None)
                competition = competitions[0]
                status = competition.get('status') or event.get('status') or {}
                status_type = status.get('type') or {}
                completed = bool(status_type.get('completed'))
                state = str(status_type.get('state') or '').strip().lower()
                return (_py_r1000_NONE, None)

            def _py_r1000_chunk_7():
                nonlocal away, competitor, home

                def _py_r1000_loop_8():
                    _py_r1000_result_5 = _py_r1000_loop_2()
                    if _py_r1000_result_5[0] == _py_r1000_RETURN:
                        return _py_r1000_result_5
                    if _py_r1000_result_5[0] == _py_r1000_BREAK:
                        return (_py_r1000_BREAK, None)
                    if _py_r1000_result_5[0] == _py_r1000_CONTINUE:
                        return (_py_r1000_CONTINUE, None)
                    return (_py_r1000_NONE, None)
                if not completed and state != 'post':
                    return (_py_r1000_CONTINUE, None)
                home = None
                away = None
                for competitor in competition.get('competitors') or []:
                    _py_r1000_result_9 = _py_r1000_loop_8()
                    if _py_r1000_result_9[0] == _py_r1000_RETURN:
                        return _py_r1000_result_9
                    if _py_r1000_result_9[0] == _py_r1000_BREAK:
                        break
                    if _py_r1000_result_9[0] == _py_r1000_CONTINUE:
                        continue
                return (_py_r1000_NONE, None)

            def _py_r1000_chunk_10():
                nonlocal away_score, home_score
                if home is None or away is None:
                    return (_py_r1000_CONTINUE, None)
                home_score = score_value(home.get('score'))
                away_score = score_value(away.get('score'))
                if home_score is None or away_score is None:
                    return (_py_r1000_CONTINUE, None)
                return (_py_r1000_NONE, None)

            def _py_r1000_chunk_11():
                nonlocal home_team
                home_team = (home.get('team') or {}).get('displayName') or (home.get('team') or {}).get('shortDisplayName') or ''
                return (_py_r1000_NONE, None)

            def _py_r1000_chunk_12():
                nonlocal away_team, game_id
                away_team = (away.get('team') or {}).get('displayName') or (away.get('team') or {}).get('shortDisplayName') or ''
                game_id = str(event.get('id') or competition.get('id') or '').strip()
                return (_py_r1000_NONE, None)

            def _py_r1000_chunk_13():
                nonlocal margin
                if not game_id or not str(home_team).strip() or (not str(away_team).strip()):
                    return (_py_r1000_CONTINUE, None)
                margin = home_score - away_score
                rows.append({'sport': 'Basketball', 'league': league_label, 'game_id': game_id, 'game_date': game_date.strftime('%Y_%m_%d'), 'home_team': str(home_team).strip(), 'away_team': str(away_team).strip(), 'home_score': home_score, 'away_score': away_score, 'total': home_score + away_score, 'home_spread': margin, 'away_spread': -margin})
                return (_py_r1000_NONE, None)
            for _py_r1000_block_14 in (_py_r1000_chunk_6, _py_r1000_chunk_7, _py_r1000_chunk_10, _py_r1000_chunk_11, _py_r1000_chunk_12, _py_r1000_chunk_13):
                _py_r1000_result_15 = _py_r1000_block_14()
                if _py_r1000_result_15[0] != _py_r1000_NONE:
                    return _py_r1000_result_15
            return (_py_r1000_NONE, None)
        league_label = LEAGUES[league_key]['label']
        rows: list[dict] = []
        for event in payload.get('events') or []:
            _py_r1000_result_16 = _py_r1000_loop_1()
            if _py_r1000_result_16[0] == _py_r1000_RETURN:
                return _py_r1000_result_16
            if _py_r1000_result_16[0] == _py_r1000_BREAK:
                break
            if _py_r1000_result_16[0] == _py_r1000_CONTINUE:
                continue
        return (_py_r1000_RETURN, rows)
    _py_r1000_outcome = _py_r1000_impl()
    if _py_r1000_outcome[0] == _py_r1000_RETURN:
        return _py_r1000_outcome[1]


def composite(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("game_date") or "").strip().replace("-", "_"),
        " ".join(str(row.get("home_team") or "").split()).casefold(),
        " ".join(str(row.get("away_team") or "").split()).casefold(),
    )


def score_signature(row: dict) -> tuple[str, str]:
    return (
        str(row.get("home_score") or "").strip(),
        str(row.get("away_score") or "").strip(),
    )


def read_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        result = list(csv.DictReader(f))
    return result


def merge_rows(existing: list[dict], fetched: list[dict], label: str) -> list[dict]:
    merged = [dict(row) for row in existing]
    by_id: dict[str, tuple[tuple[str, str, str], tuple[str, str]]] = {}
    by_comp: dict[tuple[str, str, str], tuple[str, str]] = {}

    def register(row: dict) -> None:
        registered_gid = str(row.get("game_id") or "").strip()
        registered_key = composite(row)
        registered_scores = score_signature(row)

        if not registered_gid or not all(registered_key) or not all(registered_scores):
            return

        registered_prior_id = by_id.get(registered_gid)
        if registered_prior_id and (registered_prior_id[0] != registered_key or registered_prior_id[1] != registered_scores):
            raise RuntimeError(f"{label}: conflicting final score for game_id {registered_gid}")

        registered_prior_comp = by_comp.get(registered_key)
        if registered_prior_comp and registered_prior_comp != registered_scores:
            raise RuntimeError(f"{label}: conflicting final score for {registered_key}")

        by_id[registered_gid] = (registered_key, registered_scores)
        by_comp[registered_key] = registered_scores

    for row in merged:
        register(row)

    for row in fetched:
        gid = str(row.get("game_id") or "").strip()
        key = composite(row)
        scores = score_signature(row)

        prior_id = by_id.get(gid)
        if prior_id:
            if prior_id[0] != key or prior_id[1] != scores:
                raise RuntimeError(f"{label}: fetched score conflicts for game_id {gid}")
            continue

        prior_comp = by_comp.get(key)
        if prior_comp and prior_comp != scores:
            raise RuntimeError(f"{label}: fetched score conflicts for {key}")

        merged.append(dict(row))
        register(row)

    return merged


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")

    with open(temp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})

    temp.replace(path)


def fetch_one(league_key: str, game_date: date) -> tuple[int, Path | None]:
    cfg = LEAGUES[league_key]
    payload = fetch_json(scoreboard_url(cfg["espn_slug"], game_date))
    fetched = parse_completed_games(payload, league_key, game_date)

    if not fetched:
        log(f"[{cfg['label']}] {game_date}: no completed games")
        return 0, None

    output = (
        RESULTS_BASE
        / league_key
        / f"{game_date.strftime('%Y_%m_%d')}_final_scores_{cfg['label']}.csv"
    )
    existing = read_existing(output)
    merged = merge_rows(existing, fetched, f"{cfg['label']} {game_date}")
    write_rows(output, merged)

    log(
        f"[{cfg['label']}] {game_date}: fetched={len(fetched)} "
        f"stored={len(merged)} -> {output}"
    )
    return len(fetched), output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch completed basketball final scores from ESPN."
    )
    parser.add_argument(
        "--date",
        type=parse_date_value,
        default=None,
        help="Fetch exactly one date.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_date_value,
        default=None,
        help="Inclusive start date.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date_value,
        default=None,
        help="Inclusive end date.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Default days before today to fetch when no explicit date is supplied.",
    )
    parser.add_argument(
        "--league",
        choices=tuple(LEAGUES),
        action="append",
        default=None,
        help="Limit to one or more leagues; may be repeated.",
    )
    return parser.parse_args()


def resolve_dates(args) -> list[date]:
    if args.date is not None:
        if args.start_date is not None or args.end_date is not None:
            raise ValueError("--date cannot be combined with --start-date/--end-date")
        return [args.date]

    if args.start_date is not None or args.end_date is not None:
        if args.start_date is None or args.end_date is None:
            raise ValueError("--start-date and --end-date must be supplied together")
        return list(date_range(args.start_date, args.end_date))

    if args.lookback_days < 0:
        raise ValueError("--lookback-days must be >= 0")

    today = datetime.now(NY).date()
    start = today - timedelta(days=args.lookback_days)
    return list(date_range(start, today))


def main() -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(
        f"=== 00_basketball_final_scores RUN {now_utc()} ===\n",
        encoding="utf-8",
    )

    args = parse_args()
    dates = resolve_dates(args)
    leagues = args.league or list(LEAGUES)

    failures: list[str] = []
    total_completed = 0

    for game_date in dates:
        for league_key in leagues:
            try:
                count, _ = fetch_one(league_key, game_date)
                total_completed += count
            except Exception as exc:
                message = f"[{LEAGUES[league_key]['label']}] {game_date}: {exc}"
                log(message, "ERROR")
                failures.append(message)

    log(
        f"completed_games_fetched={total_completed} "
        f"dates={len(dates)} leagues={len(leagues)}"
    )

    if failures:
        log(f"STATUS: FAILED | failures={len(failures)}", "ERROR")
        raise RuntimeError("\n".join(failures))

    log("STATUS: SUCCESS")


if __name__ == "__main__":
    try:
        main()
    except Exception as main_exc:
        print(f"STATUS: FAILED | {main_exc}", file=sys.stderr)
        sys.exit(1)
