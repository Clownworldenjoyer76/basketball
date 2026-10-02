#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/basketball_odds_core.py

import csv
import http.client
import json
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

SNAPSHOT_ROOT = Path(
    "docs/win/basketball/00_intake/sportsbook_snapshots"
)

LEAGUES = {
    "nba": {
        "label": "NBA",
        "espn_slug": "nba",
        "output_dir": Path(
            "docs/win/basketball/00_intake/sportsbook/nba"
        ),
    },
    "wnba": {
        "label": "WNBA",
        "espn_slug": "wnba",
        "output_dir": Path(
            "docs/win/basketball/00_intake/sportsbook/wnba"
        ),
    },
    "ncaam": {
        "label": "NCAAM",
        "espn_slug": "mens-college-basketball",
        "output_dir": Path(
            "docs/win/basketball/00_intake/sportsbook/ncaam"
        ),
    },
}

FIELDNAMES = [
    "sport",
    "league",
    "game_date",
    "game_id",
    "odds_last_update",
    "sportsbook_provider",
    "scraped_at_utc",
    "provider_updated_at_utc",
    "game_time",
    "home_team",
    "away_team",
    "home_spread",
    "away_spread",
    "total",
    "home_dk_moneyline_american",
    "away_dk_moneyline_american",
    "home_dk_spread_american",
    "away_dk_spread_american",
    "dk_total_over_american",
    "dk_total_under_american",
    "home_dk_moneyline_decimal",
    "away_dk_moneyline_decimal",
    "home_dk_spread_decimal",
    "away_dk_spread_decimal",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
]

ODDS_FIELDS = [
    "home_spread",
    "away_spread",
    "total",
    "home_dk_moneyline_american",
    "away_dk_moneyline_american",
    "home_dk_spread_american",
    "away_dk_spread_american",
    "dk_total_over_american",
    "dk_total_under_american",
    "home_dk_moneyline_decimal",
    "away_dk_moneyline_decimal",
    "home_dk_spread_decimal",
    "away_dk_spread_decimal",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
]

DRAFTKINGS_PROVIDER_NAME = "DraftKings"
DRAFTKINGS_PROVIDER_IDS = {"41"}

ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "basketball_odds.txt"

for league, league_cfg in LEAGUES.items():
    league_cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    league_cfg["snapshot_dir"] = SNAPSHOT_ROOT / league
    league_cfg["snapshot_dir"].mkdir(parents=True, exist_ok=True)

with open(LOG_FILE, "w", encoding="utf-8") as startup_log_handle:
    startup_log_handle.write(f"=== basketball_odds RUN {datetime.now().isoformat()} ===\n")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36"
)

REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.5


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


def get_json(url: str) -> dict:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(f"Only HTTPS URLs are permitted: {url}")

    target = parsed.path or "/"
    if parsed.query:
        target = f"{target}?{parsed.query}"

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        connection = http.client.HTTPSConnection(
            parsed.hostname,
            parsed.port or 443,
            timeout=REQUEST_TIMEOUT,
        )
        try:
            connection.request(
                "GET",
                target,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json,text/plain,*/*",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            response = connection.getresponse()
            payload = response.read()

            if not 200 <= response.status < 300:
                raise RuntimeError(
                    f"HTTP {response.status} fetching {url}"
                )

            return json.loads(payload.decode("utf-8"))

        except (
            OSError,
            TimeoutError,
            http.client.HTTPException,
            json.JSONDecodeError,
            RuntimeError,
        ) as exc:
            last_error = exc
            log(
                f"HTTP attempt {attempt}/{MAX_RETRIES} failed: "
                f"{url} | {type(exc).__name__}: {exc}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * attempt)
        finally:
            connection.close()

    raise RuntimeError(f"Failed to fetch ESPN JSON: {url}") from last_error


def nested_get(obj, *keys):
    current = obj

    for key in keys:
        if not isinstance(current, dict):
            return None

        current = current.get(key)

        if current is None:
            return None

    return current


def is_blank(value) -> bool:
    return value is None or str(value).strip() == ""


def clean_text(value) -> str:
    return "" if value is None else str(value).strip()


def clean_number(value) -> str:
    if is_blank(value):
        return ""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return clean_text(value)

    if number.is_integer():
        return str(int(number))

    return format(number, ".15g")


def normalize_row(row: dict) -> dict:
    return {
        field: clean_text(row.get(field))
        for field in FIELDNAMES
    }


def parse_espn_datetime(value: str) -> datetime:
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )


def parse_update_timestamp(value) -> datetime:
    if is_blank(value):
        return datetime.min.replace(tzinfo=UTC_TZ)

    try:
        parsed = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC_TZ)

        return parsed.astimezone(UTC_TZ)

    except ValueError:
        return datetime.min.replace(tzinfo=UTC_TZ)


def normalize_utc_timestamp(value) -> str:
    if is_blank(value):
        return ""

    try:
        parsed = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        return ""

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC_TZ)

    return parsed.astimezone(UTC_TZ).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def provider_name_from_odds(
    odds: dict | None,
) -> str:
    if not isinstance(odds, dict):
        return ""

    provider = odds.get("provider") or {}

    if not isinstance(provider, dict):
        return ""

    return clean_text(
        provider.get("name")
    )


def is_draftkings_odds(
    odds: dict | None,
) -> bool:
    if not isinstance(odds, dict):
        return False

    provider = odds.get("provider") or {}

    if not isinstance(provider, dict):
        return False

    provider_name = clean_text(
        provider.get("name")
    ).casefold()

    provider_id = clean_text(
        provider.get("id")
    )

    return (
        provider_name == "draftkings"
        or provider_id in DRAFTKINGS_PROVIDER_IDS
    )


def provider_updated_at_utc(
    odds: dict | None,
) -> str:
    """
    Return an ESPN-supplied odds update timestamp when one is present.

    ESPN payloads are not guaranteed to expose a provider update time, so
    only explicit update/modified timestamp fields are considered. The field
    remains blank when ESPN does not provide one.
    """
    if not isinstance(odds, dict):
        return ""

    candidates = [
        odds.get("lastUpdated"),
        odds.get("lastUpdatedDate"),
        odds.get("updatedAt"),
        odds.get("updateDate"),
        odds.get("lastModified"),
        nested_get(odds, "current", "lastUpdated"),
        nested_get(odds, "current", "lastUpdatedDate"),
        nested_get(odds, "current", "updatedAt"),
        nested_get(odds, "current", "updateDate"),
        nested_get(odds, "current", "lastModified"),
    ]

    for value in candidates:
        normalized = normalize_utc_timestamp(
            value
        )

        if normalized:
            return normalized

    return ""


def scoreboard_url(
    espn_slug: str,
    date_yyyymmdd: str,
) -> str:
    return (
        f"https://cdn.espn.com/core/{espn_slug}/scoreboard"
        f"?xhr=1&date={date_yyyymmdd}"
    )


def fetch_scoreboard(
    espn_slug: str,
    date_yyyymmdd: str,
) -> list[dict]:
    payload = get_json(
        scoreboard_url(
            espn_slug,
            date_yyyymmdd,
        )
    )

    events = (
        ((payload.get("content") or {})
         .get("sbData") or {})
        .get("events")
        or []
    )

    if not isinstance(events, list):
        raise RuntimeError(
            f"Unexpected ESPN scoreboard structure for "
            f"{espn_slug} {date_yyyymmdd}"
        )

    return events


def core_odds_url(
    espn_slug: str,
    event_id: str,
    competition_id: str,
) -> str:
    return (
        "https://sports.core.api.espn.com/v2/"
        "sports/basketball/"
        f"leagues/{espn_slug}/"
        f"events/{event_id}/"
        f"competitions/{competition_id}/odds"
    )


def fetch_current_odds(
    espn_slug: str,
    event_id: str,
    competition_id: str,
) -> dict | None:
    """
    Return DraftKings odds only.

    A non-DraftKings provider must never be written into *_dk_* columns.
    If DraftKings is not present in ESPN's provider list, return None.
    """
    payload = get_json(
        core_odds_url(
            espn_slug,
            event_id,
            competition_id,
        )
    )

    items = payload.get("items") or []

    if not isinstance(items, list) or not items:
        return None

    available_providers = []

    for item in items:
        if not isinstance(item, dict):
            continue

        provider_name = provider_name_from_odds(
            item
        )

        if provider_name:
            available_providers.append(
                provider_name
            )

        if is_draftkings_odds(item):
            return item

    log(
        f"NO DRAFTKINGS PROVIDER: "
        f"event_id={event_id} "
        f"available_providers="
        f"{available_providers or ['unknown']}"
    )

    return None


def get_competition(
    event: dict,
) -> dict | None:
    competitions = event.get("competitions") or []

    if (
        not isinstance(competitions, list)
        or not competitions
    ):
        return None

    return competitions[0]


def resolve_competition_id(
    competition: dict,
    event_id: str,
) -> str:
    value = competition.get("id")

    if value in (None, ""):
        return event_id

    return str(value)


def get_competitors(
    competition: dict,
) -> tuple[dict | None, dict | None]:
    home = None
    away = None

    for competitor in (
        competition.get("competitors") or []
    ):
        home_away = clean_text(
            competitor.get("homeAway")
        ).lower()

        if home_away == "home":
            home = competitor

        elif home_away == "away":
            away = competitor

    return home, away


def team_name(
    competitor: dict | None,
) -> str:
    if not isinstance(competitor, dict):
        return ""

    team = competitor.get("team") or {}

    return clean_text(
        team.get("displayName")
        or team.get("shortDisplayName")
        or team.get("name")
    )


def team_name_from_odds(
    odds: dict | None,
    side: str,
) -> str:
    return clean_text(
        nested_get(
            odds,
            f"{side}TeamOdds",
            "team",
            "displayName",
        )
    )


def event_is_completed(
    event: dict,
    competition: dict,
) -> bool:
    completed = nested_get(
        event,
        "status",
        "type",
        "completed",
    )

    if completed is None:
        completed = nested_get(
            competition,
            "status",
            "type",
            "completed",
        )

    return completed is True


def has_odds_values(
    row: dict,
) -> bool:
    return any(
        not is_blank(row.get(field))
        for field in ODDS_FIELDS
    )


def build_row(
    league_label: str,
    event: dict,
    competition: dict,
    odds: dict | None,
    run_timestamp: str,
) -> dict | None:
    event_id = clean_text(
        event.get("id")
    )

    event_date_raw = clean_text(
        event.get("date")
    )

    if not event_id or not event_date_raw:
        return None

    event_dt_ny = (
        parse_espn_datetime(
            event_date_raw
        )
        .astimezone(NY_TZ)
    )

    game_date = event_dt_ny.strftime(
        "%Y_%m_%d"
    )

    game_time = event_dt_ny.strftime(
        "%I:%M %p"
    )

    home_competitor, away_competitor = (
        get_competitors(
            competition
        )
    )

    home_team = (
        team_name(home_competitor)
        or team_name_from_odds(
            odds,
            "home",
        )
    )

    away_team = (
        team_name(away_competitor)
        or team_name_from_odds(
            odds,
            "away",
        )
    )

    if odds is not None and not is_draftkings_odds(
        odds
    ):
        raise ValueError(
            "Non-DraftKings odds reached build_row; "
            "refusing to write them into *_dk_* columns"
        )

    sportsbook_provider = (
        provider_name_from_odds(
            odds
        )
        if odds is not None
        else ""
    )

    provider_update = (
        provider_updated_at_utc(
            odds
        )
        if odds is not None
        else ""
    )

    odds = odds or {}

    home_spread = nested_get(
        odds,
        "homeTeamOdds",
        "current",
        "pointSpread",
        "american",
    )

    away_spread = nested_get(
        odds,
        "awayTeamOdds",
        "current",
        "pointSpread",
        "american",
    )

    total = nested_get(
        odds,
        "current",
        "total",
        "american",
    )

    home_ml_american = nested_get(
        odds,
        "homeTeamOdds",
        "current",
        "moneyLine",
        "american",
    )

    away_ml_american = nested_get(
        odds,
        "awayTeamOdds",
        "current",
        "moneyLine",
        "american",
    )

    home_spread_american = nested_get(
        odds,
        "homeTeamOdds",
        "current",
        "spread",
        "american",
    )

    away_spread_american = nested_get(
        odds,
        "awayTeamOdds",
        "current",
        "spread",
        "american",
    )

    over_american = nested_get(
        odds,
        "current",
        "over",
        "american",
    )

    under_american = nested_get(
        odds,
        "current",
        "under",
        "american",
    )

    home_ml_decimal = nested_get(
        odds,
        "homeTeamOdds",
        "current",
        "moneyLine",
        "decimal",
    )

    away_ml_decimal = nested_get(
        odds,
        "awayTeamOdds",
        "current",
        "moneyLine",
        "decimal",
    )

    home_spread_decimal = nested_get(
        odds,
        "homeTeamOdds",
        "current",
        "spread",
        "decimal",
    )

    away_spread_decimal = nested_get(
        odds,
        "awayTeamOdds",
        "current",
        "spread",
        "decimal",
    )

    over_decimal = nested_get(
        odds,
        "current",
        "over",
        "decimal",
    )

    under_decimal = nested_get(
        odds,
        "current",
        "under",
        "decimal",
    )

    if is_blank(home_spread):
        home_spread = odds.get("spread")

    if (
        is_blank(away_spread)
        and not is_blank(home_spread)
    ):
        try:
            away_spread = -float(
                home_spread
            )
        except (TypeError, ValueError):
            pass

    if is_blank(total):
        total = odds.get(
            "overUnder"
        )

    if is_blank(home_ml_american):
        home_ml_american = nested_get(
            odds,
            "homeTeamOdds",
            "moneyLine",
        )

    if is_blank(away_ml_american):
        away_ml_american = nested_get(
            odds,
            "awayTeamOdds",
            "moneyLine",
        )

    if is_blank(home_spread_american):
        home_spread_american = nested_get(
            odds,
            "homeTeamOdds",
            "spreadOdds",
        )

    if is_blank(away_spread_american):
        away_spread_american = nested_get(
            odds,
            "awayTeamOdds",
            "spreadOdds",
        )

    if is_blank(over_american):
        over_american = odds.get(
            "overOdds"
        )

    if is_blank(under_american):
        under_american = odds.get(
            "underOdds"
        )

    row = {
        "sport": "Basketball",
        "league": league_label,
        "game_date": game_date,
        "game_id": event_id,
        "odds_last_update": "",
        "sportsbook_provider": "",
        "scraped_at_utc": "",
        "provider_updated_at_utc": "",
        "game_time": game_time,
        "home_team": home_team,
        "away_team": away_team,
        "home_spread": clean_number(
            home_spread
        ),
        "away_spread": clean_number(
            away_spread
        ),
        "total": clean_number(
            total
        ),
        "home_dk_moneyline_american":
            clean_number(
                home_ml_american
            ),
        "away_dk_moneyline_american":
            clean_number(
                away_ml_american
            ),
        "home_dk_spread_american":
            clean_number(
                home_spread_american
            ),
        "away_dk_spread_american":
            clean_number(
                away_spread_american
            ),
        "dk_total_over_american":
            clean_number(
                over_american
            ),
        "dk_total_under_american":
            clean_number(
                under_american
            ),
        "home_dk_moneyline_decimal":
            clean_number(
                home_ml_decimal
            ),
        "away_dk_moneyline_decimal":
            clean_number(
                away_ml_decimal
            ),
        "home_dk_spread_decimal":
            clean_number(
                home_spread_decimal
            ),
        "away_dk_spread_decimal":
            clean_number(
                away_spread_decimal
            ),
        "dk_total_over_decimal":
            clean_number(
                over_decimal
            ),
        "dk_total_under_decimal":
            clean_number(
                under_decimal
            ),
    }

    if has_odds_values(row):
        row["sportsbook_provider"] = (
            sportsbook_provider
            or DRAFTKINGS_PROVIDER_NAME
        )

        row["scraped_at_utc"] = (
            run_timestamp
        )

        row["provider_updated_at_utc"] = (
            provider_update
        )

        # Legacy compatibility: downstream code already uses odds_last_update
        # to choose the newest row. Prefer ESPN's provider timestamp when
        # available; otherwise use the actual scrape time.
        row["odds_last_update"] = (
            provider_update
            or run_timestamp
        )

    return row


def merge_nonblank(
    old_row: dict,
    new_row: dict,
) -> dict:
    merged = normalize_row(
        old_row
    )

    for field in FIELDNAMES:
        if not is_blank(
            new_row.get(field)
        ):
            merged[field] = clean_text(
                new_row[field]
            )

    return merged


def consolidate_duplicates(
    file_rows: dict[Path, list[dict]],
) -> set[Path]:
    changed_paths = set()
    occurrences = {}
    sequence = 0

    for path, rows in file_rows.items():
        for row in rows:
            game_id = clean_text(
                row.get("game_id")
            )

            if not game_id:
                continue

            occurrences.setdefault(
                game_id,
                [],
            ).append(
                (
                    parse_update_timestamp(
                        row.get(
                            "odds_last_update"
                        )
                    ),
                    sequence,
                    path,
                    row,
                )
            )

            sequence += 1

    remove_ids_by_path = {}

    for game_id, copies in (
        occurrences.items()
    ):
        if len(copies) < 2:
            continue

        copies.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        consolidated = normalize_row(
            copies[0][3]
        )

        for _, _, _, row in copies[1:]:
            consolidated = merge_nonblank(
                consolidated,
                row,
            )

        (
            _,
            _,
            target_path,
            target_row,
        ) = copies[-1]

        target_row.clear()
        target_row.update(
            consolidated
        )

        changed_paths.add(
            target_path
        )

        for _, _, path, row in copies[:-1]:
            remove_ids_by_path.setdefault(
                path,
                set(),
            ).add(
                id(row)
            )

            changed_paths.add(
                path
            )

        log(
            f"CONSOLIDATED DUPLICATES: "
            f"game_id={game_id} "
            f"copies={len(copies)}"
        )

    for path, remove_ids in (
        remove_ids_by_path.items()
    ):
        file_rows[path] = [
            row
            for row in file_rows[path]
            if id(row) not in remove_ids
        ]

    return changed_paths


def load_existing_files(
    output_dir: Path,
    league_label: str,
) -> tuple[
    dict[Path, list[dict]],
    set[Path],
]:
    file_rows = {}

    for path in sorted(
        output_dir.glob(
            f"*_{league_label}_odds.csv"
        )
    ):
        with open(
            path,
            newline="",
            encoding="utf-8",
        ) as csvfile:
            file_rows[path] = [
                normalize_row(row)
                for row in csv.DictReader(
                    csvfile
                )
            ]

    changed_paths = (
        consolidate_duplicates(
            file_rows
        )
    )

    return (
        file_rows,
        changed_paths,
    )


def build_existing_index(
    file_rows: dict[Path, list[dict]],
) -> dict[
    str,
    tuple[Path, dict],
]:
    index = {}

    for path, rows in (
        file_rows.items()
    ):
        for row in rows:
            game_id = clean_text(
                row.get("game_id")
            )

            if game_id:
                index[game_id] = (
                    path,
                    row,
                )

    return index


def write_file(
    path: Path,
    rows: list[dict],
) -> int:
    repo_root = Path(__file__).resolve().parents[5]
    candidate = path if path.is_absolute() else Path.cwd() / path
    safe_path = candidate.resolve()
    try:
        safe_path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {path}") from exc

    safe_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with safe_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=FIELDNAMES,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            normalize_row(row)
            for row in rows
        )

    return len(rows)


def write_snapshot_file(
    snapshot_dir: Path,
    league_label: str,
    snapshot_id: str,
    rows: list[dict],
) -> tuple[Path | None, int]:
    """
    Write one immutable odds snapshot for this league/run.

    Existing snapshot files are never opened for update, merge, consolidation,
    deletion, or overwrite. Exclusive-create mode makes a filename collision
    fatal instead of silently replacing historical observations.
    """
    if not rows:
        return None, 0

    snapshot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        snapshot_dir
        / f"{snapshot_id}_{league_label}_odds_snapshot.csv"
    )

    with open(
        path,
        "x",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=FIELDNAMES,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            normalize_row(row)
            for row in rows
        )

    return path, len(rows)


def scoreboard_dates(
    start_dt: datetime,
    end_dt: datetime,
):
    current_date = (
        start_dt.date()
    )

    end_date = (
        end_dt.date()
    )

    while current_date <= end_date:
        yield current_date

        current_date += timedelta(
            days=1
        )


def main():
    run_dt = datetime.now(
        NY_TZ
    )

    end_dt = run_dt + timedelta(
        days=7
    )

    run_utc_dt = datetime.now(
        UTC_TZ
    )

    run_timestamp = run_utc_dt.strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )

    snapshot_id = run_utc_dt.strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    files_written = []
    snapshots_written = []
    total_snapshot_rows = 0
    total_events_found = 0
    total_new_games = 0
    total_updated_games = 0
    total_completed_skipped = 0
    total_outside_window_skipped = 0
    total_no_dk_odds = 0
    total_errors = 0

    try:
        for cfg in LEAGUES.values():
            league_label = (
                cfg["label"]
            )

            espn_slug = (
                cfg["espn_slug"]
            )

            output_dir = (
                cfg["output_dir"]
            )

            snapshot_dir = (
                cfg["snapshot_dir"]
            )

            snapshot_rows = []

            (
                file_rows,
                changed_paths,
            ) = load_existing_files(
                output_dir,
                league_label,
            )

            existing_index = (
                build_existing_index(
                    file_rows
                )
            )

            for fetch_date in (
                scoreboard_dates(
                    run_dt,
                    end_dt,
                )
            ):
                date_yyyymmdd = (
                    fetch_date.strftime(
                        "%Y%m%d"
                    )
                )

                log(
                    f"FETCH SCOREBOARD: "
                    f"{league_label} "
                    f"{date_yyyymmdd}"
                )

                try:
                    events = fetch_scoreboard(
                        espn_slug,
                        date_yyyymmdd,
                    )

                except Exception as exc:
                    total_errors += 1

                    log(
                        f"ERROR fetching scoreboard "
                        f"{league_label} "
                        f"{date_yyyymmdd}: "
                        f"{exc}\n"
                        f"{traceback.format_exc()}"
                    )

                    continue

                total_events_found += len(
                    events
                )

                for event in events:
                    event_id = clean_text(
                        event.get("id")
                    )

                    event_name = clean_text(
                        event.get("name")
                    )

                    event_date_raw = (
                        clean_text(
                            event.get("date")
                        )
                    )

                    competition = (
                        get_competition(
                            event
                        )
                    )

                    if (
                        not event_id
                        or not event_date_raw
                        or competition is None
                    ):
                        total_errors += 1

                        log(
                            f"SKIP malformed event: "
                            f"{league_label} "
                            f"{event_name or event_id}"
                        )

                        continue

                    try:
                        event_dt_ny = (
                            parse_espn_datetime(
                                event_date_raw
                            )
                            .astimezone(
                                NY_TZ
                            )
                        )

                    except ValueError as exc:
                        total_errors += 1

                        log(
                            f"SKIP bad event date: "
                            f"{league_label} "
                            f"{event_id} "
                            f"{event_date_raw} | "
                            f"{exc}"
                        )

                        continue

                    if (
                        event_dt_ny < run_dt
                        or event_dt_ny > end_dt
                    ):
                        total_outside_window_skipped += 1
                        continue

                    if event_is_completed(
                        event,
                        competition,
                    ):
                        total_completed_skipped += 1

                        log(
                            f"SKIP COMPLETED: "
                            f"{league_label} "
                            f"{event_id} "
                            f"{event_name}"
                        )

                        continue

                    comp_id = (
                        resolve_competition_id(
                            competition,
                            event_id,
                        )
                    )

                    odds = None

                    try:
                        odds = fetch_current_odds(
                            espn_slug,
                            event_id,
                            comp_id,
                        )

                    except Exception as exc:
                        total_errors += 1

                        log(
                            f"ERROR fetching odds: "
                            f"{league_label} "
                            f"{event_id} "
                            f"{event_name}: "
                            f"{exc}"
                        )

                    if odds is None:
                        total_no_dk_odds += 1

                        log(
                            f"NO DRAFTKINGS ODDS: "
                            f"{league_label} "
                            f"{event_id} "
                            f"{event_name}"
                        )

                    try:
                        new_row = build_row(
                            league_label,
                            event,
                            competition,
                            odds,
                            run_timestamp,
                        )

                    except Exception as exc:
                        total_errors += 1

                        log(
                            f"ERROR building row: "
                            f"{league_label} "
                            f"{event_id} "
                            f"{event_name}: "
                            f"{exc}\n"
                            f"{traceback.format_exc()}"
                        )

                        continue

                    if new_row is None:
                        total_errors += 1
                        continue

                    if has_odds_values(
                        new_row
                    ):
                        snapshot_rows.append(
                            normalize_row(
                                new_row
                            )
                        )

                    existing = (
                        existing_index.get(
                            event_id
                        )
                    )

                    if existing is not None:
                        (
                            existing_path,
                            existing_row,
                        ) = existing

                        if has_odds_values(
                            new_row
                        ):
                            merged = (
                                merge_nonblank(
                                    existing_row,
                                    new_row,
                                )
                            )

                            if (
                                merged
                                != existing_row
                            ):
                                existing_row.clear()

                                existing_row.update(
                                    merged
                                )

                                changed_paths.add(
                                    existing_path
                                )

                                total_updated_games += 1

                        continue

                    out_path = (
                        output_dir
                        / (
                            f"{new_row['game_date']}"
                            f"_{league_label}"
                            f"_odds.csv"
                        )
                    )

                    file_rows.setdefault(
                        out_path,
                        [],
                    ).append(
                        new_row
                    )

                    existing_index[
                        event_id
                    ] = (
                        out_path,
                        new_row,
                    )

                    changed_paths.add(
                        out_path
                    )

                    total_new_games += 1

            snapshot_path, snapshot_count = (
                write_snapshot_file(
                    snapshot_dir,
                    league_label,
                    snapshot_id,
                    snapshot_rows,
                )
            )

            if snapshot_path is not None:
                snapshots_written.append(
                    (
                        str(snapshot_path),
                        snapshot_count,
                    )
                )

                total_snapshot_rows += (
                    snapshot_count
                )

                log(
                    f"WROTE IMMUTABLE SNAPSHOT "
                    f"{snapshot_path} "
                    f"({snapshot_count} observations)"
                )

            for path in sorted(
                changed_paths
            ):
                row_count = (
                    write_file(
                        path,
                        file_rows.get(
                            path,
                            [],
                        ),
                    )
                )

                files_written.append(
                    (
                        str(path),
                        row_count,
                    )
                )

                log(
                    f"WROTE {path} "
                    f"({row_count} games)"
                )

        log("--- SUMMARY ---")
        log(
            f"Run window start: "
            f"{run_dt.isoformat()}"
        )
        log(
            f"Run window end: "
            f"{end_dt.isoformat()}"
        )
        log(
            f"Events found: "
            f"{total_events_found}"
        )
        log(
            f"New games added: "
            f"{total_new_games}"
        )
        log(
            f"Existing games updated: "
            f"{total_updated_games}"
        )
        log(
            f"Completed games skipped: "
            f"{total_completed_skipped}"
        )
        log(
            f"Outside-window events skipped: "
            f"{total_outside_window_skipped}"
        )
        log(
            f"Games without DraftKings odds: "
            f"{total_no_dk_odds}"
        )
        log(
            f"Latest-state files written: "
            f"{len(files_written)}"
        )
        log(
            f"Immutable snapshot files written: "
            f"{len(snapshots_written)}"
        )
        log(
            f"Immutable snapshot observations: "
            f"{total_snapshot_rows}"
        )
        log(
            f"Errors: "
            f"{total_errors}"
        )

        for path, count in files_written:
            log(
                f"LATEST FILE: {path} "
                f"({count} games)"
            )

        for path, count in snapshots_written:
            log(
                f"SNAPSHOT FILE: {path} "
                f"({count} observations)"
            )

        log("STATUS: SUCCESS")

        print(
            "Basketball odds complete."
        )

        print(
            f"Window: "
            f"{run_dt.isoformat()} "
            f"through "
            f"{end_dt.isoformat()}"
        )

        print(
            f"New games added: "
            f"{total_new_games}"
        )

        print(
            f"Existing games updated: "
            f"{total_updated_games}"
        )

        print(
            f"Latest-state files written: "
            f"{len(files_written)}"
        )

        print(
            f"Immutable snapshot files written: "
            f"{len(snapshots_written)}"
        )

        print(
            f"Immutable snapshot observations: "
            f"{total_snapshot_rows}"
        )

        if total_errors:
            print(
                f"Errors: "
                f"{total_errors}"
            )

            print(
                f"See log: "
                f"{LOG_FILE}"
            )

    except Exception as exc:
        log(
            f"FATAL ERROR: "
            f"{exc}\n"
            f"{traceback.format_exc()}"
        )

        log(
            "STATUS: FAILED"
        )

        raise


if __name__ == "__main__":
    main()
