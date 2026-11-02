#!/usr/bin/env python3
"""ESPN DraftKings odds collector with full NCAAM Division I scoreboard coverage.

Only actual DraftKings observations populate betting fields. Missing sportsbook
quotes remain empty, and historical rows are retained when newer prices are absent.
"""
from __future__ import annotations
import csv
import http.client
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from basketball_shared import resolve_repository_path

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")
SNAPSHOT_ROOT = Path("docs/win/basketball/00_intake/sportsbook_snapshots")
LEAGUES = {
    "nba": {"label": "NBA", "espn_slug": "nba", "output_dir": Path("docs/win/basketball/00_intake/sportsbook/nba")},
    "ncaam": {"label": "NCAAM", "espn_slug": "mens-college-basketball", "output_dir": Path("docs/win/basketball/00_intake/sportsbook/ncaam")},
    "wnba": {"label": "WNBA", "espn_slug": "wnba", "output_dir": Path("docs/win/basketball/00_intake/sportsbook/wnba")},
}
ODDS_FIELDS = ["home_spread", "away_spread", "total", "home_dk_moneyline_american", "away_dk_moneyline_american", "home_dk_spread_american", "away_dk_spread_american", "dk_total_over_american", "dk_total_under_american", "home_dk_moneyline_decimal", "away_dk_moneyline_decimal", "home_dk_spread_decimal", "away_dk_spread_decimal", "dk_total_over_decimal", "dk_total_under_decimal"]
FIELDNAMES = ["sport", "league", "game_date", "game_id", "odds_last_update", "sportsbook_provider", "scraped_at_utc", "provider_updated_at_utc", "game_time", "home_team", "away_team", *ODDS_FIELDS]
ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "basketball_odds.txt"
LOG_FILE.write_text(f"=== basketball_odds RUN {datetime.now().isoformat()} ===\n", encoding="utf-8")
for league, cfg in LEAGUES.items():
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    cfg["snapshot_dir"] = SNAPSHOT_ROOT / league
    cfg["snapshot_dir"].mkdir(parents=True, exist_ok=True)

DRAFTKINGS_PROVIDER_NAME = "DraftKings"
DRAFTKINGS_PROVIDER_IDS = {"41"}
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/140.0.0.0 Safari/537.36"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.5


def log(message: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {message}\n")


def get_json(url: str) -> dict:
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise ValueError(f"Only HTTPS URLs permitted: {url}")
    target = parts.path or "/"
    if parts.query:
        target += "?" + parts.query
    last = None
    for attempt in range(MAX_RETRIES):
        conn = http.client.HTTPSConnection(parts.hostname, parts.port or 443, timeout=REQUEST_TIMEOUT)
        try:
            conn.request("GET", target, headers={"User-Agent": USER_AGENT, "Accept": "application/json,text/plain,*/*", "Accept-Language": "en-US,en;q=0.9"})
            response = conn.getresponse()
            payload = response.read()
            if not 200 <= response.status < 300:
                raise RuntimeError(f"HTTP {response.status} fetching {url}")
            value = json.loads(payload.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object from {url}")
            return value
        except (OSError, TimeoutError, ValueError, RuntimeError, json.JSONDecodeError) as e:
            last = e
            if attempt + 1 < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
        finally:
            conn.close()
    raise RuntimeError(f"Unable to fetch ESPN: {url}: {last}") from last


def clean_text(value) -> str:
    return "" if value is None else str(value).strip()


def is_blank(value) -> bool:
    return value is None or clean_text(value).casefold() in {"", "none", "null", "nan"}


def clean_number(value) -> str:
    if is_blank(value):
        return ""
    try:
        number = float(value)
        return "" if not math.isfinite(number) else clean_text(value)
    except (TypeError, ValueError):
        return ""


def nested_get(value: dict | None, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def parse_espn_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_update_timestamp(value) -> datetime:
    try:
        parsed = parse_espn_datetime(clean_text(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC_TZ)
        return parsed.astimezone(UTC_TZ)
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=UTC_TZ)


def normalized_row(row: dict) -> dict:
    return {field: clean_text(row.get(field)) for field in FIELDNAMES}


def is_draftkings_odds(odds: dict | None) -> bool:
    if not isinstance(odds, dict):
        return False
    provider = odds.get("provider") or {}
    if not isinstance(provider, dict):
        return False
    name = "".join(c for c in clean_text(provider.get("name")).casefold() if c.isalnum())
    return name == "draftkings" or clean_text(provider.get("id")) in DRAFTKINGS_PROVIDER_IDS


def provider_name_from_odds(odds: dict | None) -> str:
    return clean_text(nested_get(odds, "provider", "name"))


def provider_updated_at_utc(odds: dict | None) -> str:
    odds = odds or {}
    for value in (odds.get("lastUpdated"), odds.get("lastUpdatedDate"), odds.get("updatedAt"), odds.get("updateDate"), odds.get("lastModified"), nested_get(odds, "current", "lastUpdated"), nested_get(odds, "current", "updatedAt")):
        if not is_blank(value):
            parsed = parse_update_timestamp(value)
            if parsed.year > 1900:
                return parsed.isoformat().replace("+00:00", "Z")
    return ""


def scoreboard_url(espn_slug: str, date_yyyymmdd: str) -> str:
    if espn_slug == "mens-college-basketball":
        return ("https://site.api.espn.com/apis/site/v2/sports/basketball/"
                f"mens-college-basketball/scoreboard?dates={date_yyyymmdd}&groups=50&limit=1000")
    return f"https://cdn.espn.com/core/{espn_slug}/scoreboard?xhr=1&date={date_yyyymmdd}"


def fetch_scoreboard(espn_slug: str, date_yyyymmdd: str) -> list[dict]:
    payload = get_json(scoreboard_url(espn_slug, date_yyyymmdd))
    if espn_slug == "mens-college-basketball":
        events = payload.get("events")
        if isinstance(events, list) and len(events) >= 1000:
            raise RuntimeError("NCAAM scoreboard hit limit; refusing truncated schedule")
    else:
        events = nested_get(payload, "content", "sbData", "events")
    if not isinstance(events, list):
        raise RuntimeError(f"Unexpected ESPN scoreboard structure {espn_slug} {date_yyyymmdd}")
    return events


def core_odds_url(espn_slug: str, event_id: str, competition_id: str) -> str:
    return f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/{espn_slug}/events/{event_id}/competitions/{competition_id}/odds"


def fetch_current_odds(espn_slug: str, event_id: str, competition_id: str) -> dict | None:
    response = get_json(core_odds_url(espn_slug, event_id, competition_id))
    items = response.get("items") or []
    for item in items:
        if is_draftkings_odds(item):
            return item
    providers = [provider_name_from_odds(item) for item in items if isinstance(item, dict)]
    log(f"NO DRAFTKINGS PROVIDER: event_id={event_id} providers={providers}")
    return None


def get_competition(event: dict) -> dict | None:
    competitions = event.get("competitions") or []
    return competitions[0] if isinstance(competitions, list) and competitions else None


def resolve_competition_id(competition: dict, event_id: str) -> str:
    return clean_text(competition.get("id")) or event_id


def get_competitors(competition: dict):
    home = away = None
    for c in competition.get("competitors") or []:
        if c.get("homeAway") == "home":
            home = c
        if c.get("homeAway") == "away":
            away = c
    return home, away


def team_name(competitor: dict | None) -> str:
    team = (competitor or {}).get("team") or {}
    return clean_text(team.get("displayName") or team.get("shortDisplayName") or team.get("name"))


def event_is_completed(event: dict, competition: dict) -> bool:
    val = nested_get(event, "status", "type", "completed")
    if val is None:
        val = nested_get(competition, "status", "type", "completed")
    return val is True


def has_odds_values(row: dict) -> bool:
    return any(not is_blank(row.get(field)) for field in ODDS_FIELDS)


def build_row(league_label: str, event: dict, competition: dict, odds: dict | None, run_timestamp: str) -> dict | None:
    event_id = clean_text(event.get("id"))
    event_date = clean_text(event.get("date"))
    if not event_id or not event_date:
        return None
    dt = parse_espn_datetime(event_date).astimezone(NY_TZ)
    home, away = get_competitors(competition)
    if odds is not None and not is_draftkings_odds(odds):
        raise ValueError("Refusing to put non-DraftKings prices into DraftKings fields")
    o = odds or {}
    home_spread = nested_get(o, "homeTeamOdds", "current", "pointSpread", "american")
    away_spread = nested_get(o, "awayTeamOdds", "current", "pointSpread", "american")
    total = nested_get(o, "current", "total", "american")
    if is_blank(home_spread):
        home_spread = o.get("spread")
    if is_blank(away_spread) and not is_blank(home_spread):
        try:
            away_spread = -float(home_spread)
        except (TypeError, ValueError):
            pass
    if is_blank(total):
        total = o.get("overUnder")
    row = {
        "sport": "Basketball", "league": league_label, "game_date": dt.strftime("%Y_%m_%d"),
        "game_id": event_id, "odds_last_update": "", "sportsbook_provider": "", "scraped_at_utc": "",
        "provider_updated_at_utc": "", "game_time": dt.strftime("%I:%M %p"),
        "home_team": team_name(home), "away_team": team_name(away),
        "home_spread": clean_number(home_spread), "away_spread": clean_number(away_spread),
        "total": clean_number(total),
    }
    paths = {
        "home_dk_moneyline_american": ("homeTeamOdds", "current", "moneyLine", "american"),
        "away_dk_moneyline_american": ("awayTeamOdds", "current", "moneyLine", "american"),
        "home_dk_spread_american": ("homeTeamOdds", "current", "spread", "american"),
        "away_dk_spread_american": ("awayTeamOdds", "current", "spread", "american"),
        "dk_total_over_american": ("current", "over", "american"),
        "dk_total_under_american": ("current", "under", "american"),
        "home_dk_moneyline_decimal": ("homeTeamOdds", "current", "moneyLine", "decimal"),
        "away_dk_moneyline_decimal": ("awayTeamOdds", "current", "moneyLine", "decimal"),
        "home_dk_spread_decimal": ("homeTeamOdds", "current", "spread", "decimal"),
        "away_dk_spread_decimal": ("awayTeamOdds", "current", "spread", "decimal"),
        "dk_total_over_decimal": ("current", "over", "decimal"),
        "dk_total_under_decimal": ("current", "under", "decimal"),
    }
    fallbacks = {
        "home_dk_moneyline_american": ("homeTeamOdds", "moneyLine"),
        "away_dk_moneyline_american": ("awayTeamOdds", "moneyLine"),
        "home_dk_spread_american": ("homeTeamOdds", "spreadOdds"),
        "away_dk_spread_american": ("awayTeamOdds", "spreadOdds"),
        "dk_total_over_american": ("overOdds",),
        "dk_total_under_american": ("underOdds",),
    }
    for field, path in paths.items():
        val = nested_get(o, *path)
        if is_blank(val) and field in fallbacks:
            val = nested_get(o, *fallbacks[field])
        row[field] = clean_number(val)
    if has_odds_values(row):
        row["sportsbook_provider"] = provider_name_from_odds(odds) or DRAFTKINGS_PROVIDER_NAME
        row["scraped_at_utc"] = run_timestamp
        row["provider_updated_at_utc"] = provider_updated_at_utc(odds)
        row["odds_last_update"] = row["provider_updated_at_utc"] or run_timestamp
    return normalized_row(row)


def merge_nonblank(old: dict, new: dict) -> dict:
    out = normalized_row(old)
    for field, value in normalized_row(new).items():
        if not is_blank(value):
            out[field] = value
    return out


def write_file(path: Path, rows: list[dict]) -> int:
    p = resolve_repository_path(path, strict=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized_row(r) for r in rows)
    return len(rows)


def load_existing_files(output_dir: Path, league_label: str) -> dict[Path, list[dict]]:
    result = {}
    for path in output_dir.glob(f"*_{league_label}_odds.csv"):
        with path.open(newline="", encoding="utf-8") as f:
            result[path] = [normalized_row(r) for r in csv.DictReader(f)]
    return result


def main() -> None:
    run_dt = datetime.now(NY_TZ)
    full_rebuild = str(os.getenv("BASKETBALL_FULL_REBUILD", "")).strip().casefold() in {"1", "true", "yes", "on"}
    end_dt = (run_dt + timedelta(days=7) if full_rebuild else run_dt.replace(hour=23, minute=59, second=59, microsecond=999999))
    timestamp = datetime.now(UTC_TZ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    snapshot_id = datetime.now(UTC_TZ).strftime("%Y%m%dT%H%M%S%fZ")
    total_events = total_priced = total_errors = 0
    for league, config in LEAGUES.items():
        label, slug, output_dir = config["label"], config["espn_slug"], config["output_dir"]
        snapshot_dir = config.get("snapshot_dir") or SNAPSHOT_ROOT / league
        existing = load_existing_files(output_dir, label)
        index = {}
        for path, rows in existing.items():
            for row in rows:
                gid = clean_text(row.get("game_id"))
                if gid:
                    old = index.get(gid)
                    if old is None or parse_update_timestamp(row.get("odds_last_update")) > parse_update_timestamp(old[1].get("odds_last_update")):
                        index[gid] = (path, row)
        modified = set()
        snapshots = []
        day = run_dt.date()
        while day <= end_dt.date():
            date_code = day.strftime("%Y%m%d")
            try:
                events = fetch_scoreboard(slug, date_code)
            except Exception as exc:
                total_errors += 1
                log(f"ERROR fetching scoreboard: {label} {date_code} {exc}")
                day += timedelta(days=1)
                continue
            total_events += len(events)
            log(f"SCOREBOARD: {label} {date_code} events={len(events)}")
            for event in events:
                event_id = clean_text(event.get("id"))
                competition = get_competition(event)
                if not event_id or competition is None:
                    total_errors += 1
                    log(f"ERROR malformed event {label}: {event_id}")
                    continue
                try:
                    event_dt = parse_espn_datetime(event.get("date", "")).astimezone(NY_TZ)
                except (ValueError, TypeError) as exc:
                    total_errors += 1
                    log(f"ERROR event time {label} {event_id}: {exc}")
                    continue
                if event_dt < run_dt or event_dt > end_dt or event_is_completed(event, competition):
                    continue
                try:
                    odds = fetch_current_odds(slug, event_id, resolve_competition_id(competition, event_id))
                    row = build_row(label, event, competition, odds, timestamp)
                    if row is None:
                        raise ValueError("Missing event identifier/date")
                except Exception as exc:
                    total_errors += 1
                    log(f"ERROR building odds {label} {event_id}: {exc}\n{traceback.format_exc()}")
                    continue
                if not has_odds_values(row):
                    log(f"NO DRAFTKINGS ODDS: {label} {event_id} {event.get('name', '')}")
                else:
                    total_priced += 1
                    snapshots.append(row)
                prior = index.get(event_id)
                if prior:
                    if has_odds_values(row):
                        path, old = prior
                        updated = merge_nonblank(old, row)
                        if updated != old:
                            old.update(updated)
                            modified.add(path)
                    continue
                path = output_dir / f"{row['game_date']}_{label}_odds.csv"
                existing.setdefault(path, []).append(row)
                index[event_id] = (path, row)
                modified.add(path)
            day += timedelta(days=1)
        for path in sorted(modified):
            write_file(path, existing[path])
            log(f"WROTE {path} ({len(existing[path])} rows)")
        if snapshots:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            path = snapshot_dir / f"{snapshot_id}_{label}_odds_snapshot.csv"
            with path.open("x", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(normalized_row(row) for row in snapshots)
            log(f"WROTE IMMUTABLE SNAPSHOT {path} ({len(snapshots)} observations)")
    log(f"Events found: {total_events} | priced: {total_priced} | errors: {total_errors}")
    log("STATUS: SUCCESS" if total_errors == 0 else "STATUS: COMPLETED WITH ERRORS")
    if total_errors:
        sys.exit(1)
    print("basketball_odds complete.")


if __name__ == "__main__":
    main()
