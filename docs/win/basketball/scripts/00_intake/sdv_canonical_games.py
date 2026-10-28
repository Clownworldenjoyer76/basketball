
#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/sdv_canonical_games.py
"""
Build canonical basketball schedules backed by SportsDataVerse/ESPN.

NBA:
    Fetch the mapped ESPN season.
    Supplement current dates using ESPN's daily scoreboard.
    Validate current-day coverage and game identities.
    Preserve valid previously collected games.

NCAAM/WNBA:
    Retain the existing SportsDataVerse season retrieval.

Output:
    docs/win/basketball/00_intake/sdv/canonical_games/
        {league}/{internal_season}_{LEAGUE}_games.csv

The canonical ESPN game_id remains the primary downstream identity.
No sportsbook prices are synthesized or modified.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
import traceback

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from sdv_season_mapping import sdv_season_id


BASE = Path("docs/win/basketball")
CONFIG_PATH = BASE / "config/sdv_seasons.yaml"
OUTPUT_ROOT = BASE / "00_intake/sdv/canonical_games"
ERROR_DIR = BASE / "errors/00_intake"
LOG_FILE = ERROR_DIR / "sdv_canonical_games.txt"

NY_TZ = ZoneInfo("America/New_York")

ESPN_NBA_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/basketball/nba/scoreboard"
)

NBA_LOOKBACK_DAYS = 2
NBA_LOOKAHEAD_DAYS = 7
HTTP_TIMEOUT = 30
HTTP_ATTEMPTS = 3

TEAM_MAPS = {
    "nba": BASE / "maps/team_map_nba.csv",
    "ncaam": BASE / "maps/team_map_ncaam.csv",
    "wnba": BASE / "maps/team_map_wnba.csv",
}

LEAGUES = {
    "nba": {
        "label": "NBA",
        "sport": "basketball",
        "limit": 5000,
    },
    "ncaam": {
        "label": "NCAAM",
        "sport": "basketball",
        "limit": 15000,
    },
    "wnba": {
        "label": "WNBA",
        "sport": "basketball",
        "limit": 2000,
    },
}

FIELDNAMES = [
    "sport",
    "league",
    "internal_season",
    "sdv_season",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "game_id",
    "home_team_id",
    "away_team_id",
    "neutral_site",
    "venue_id",
    "venue_full_name",
    "venue_name",
    "status",
    "source",
    "fetched_at_utc",
]

REQUIRED_SDV_COLUMNS = {
    "game_id",
    "date",
    "home_display_name",
    "away_display_name",
    "season",
}

NCAAM_REQUIRED_VENUE_COLUMNS = {
    "neutral_site",
    "venue_id",
    "venue_full_name",
    "home_id",
    "away_id",
}


def clean(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def clean_id(value: Any) -> str:
    text = clean(value)

    if not text:
        return ""

    try:
        number = float(text)

        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass

    return text


def log(message: str) -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)

    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{datetime.now(timezone.utc).isoformat()} | "
            f"{message}\n"
        )


def current_internal_season(
    league: str,
    now: datetime | None = None,
) -> int:
    if league not in LEAGUES:
        raise ValueError(f"Unsupported league: {league}")

    local_now = now or datetime.now(NY_TZ)

    if local_now.tzinfo is None:
        local_now = local_now.replace(tzinfo=NY_TZ)

    local_now = local_now.astimezone(NY_TZ)

    if league in {"nba", "ncaam"}:
        return (
            local_now.year
            if local_now.month >= 10
            else local_now.year - 1
        )

    return local_now.year


def load_team_map(league: str) -> dict[str, str]:
    path = TEAM_MAPS[league]
    mapping: dict[str, str] = {}

    if not path.exists():
        log(
            f"TEAM MAP MISSING: {path}; "
            "SDV names will be used unchanged"
        )
        return mapping

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        for row in csv.DictReader(handle):
            row_league = clean(
                row.get("league")
            ).lower()

            alias = clean(
                row.get("alias")
            ).casefold()

            canonical = clean(
                row.get("canonical_team")
            )

            if (
                row_league == league
                and alias
                and canonical
            ):
                mapping[alias] = canonical

    return mapping


def normalize_team_name(
    value: Any,
    team_map: dict[str, str],
) -> str:
    text = clean(value)

    if not text:
        return ""

    return team_map.get(text.casefold(), text)


def normalize_venue_name(value: Any) -> str:
    return " ".join(clean(value).split())


def parse_event_datetime(value: Any) -> datetime:
    text = clean(value)

    if not text:
        raise ValueError("Schedule event has blank date")

    parsed = pd.to_datetime(
        text,
        utc=True,
        errors="raise",
    )

    return (
        parsed.to_pydatetime()
        .astimezone(NY_TZ)
    )


def bool_text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    if isinstance(value, bool):
        return "true" if value else "false"

    text = str(value).strip().lower()

    if text in {"true", "1", "yes"}:
        return "true"

    if text in {"false", "0", "no"}:
        return "false"

    return text


def fetch_schedule(
    league: str,
    sdv_season: int,
    limit: int,
) -> pd.DataFrame:
    if league == "nba":
        from sportsdataverse.nba import espn_nba_schedule

        df = espn_nba_schedule(
            dates=sdv_season,
            season_type=None,
            limit=limit,
            return_as_pandas=True,
        )

    elif league == "ncaam":
        from sportsdataverse.mbb import espn_mbb_schedule

        df = espn_mbb_schedule(
            dates=sdv_season,
            groups=50,
            season_type=None,
            limit=limit,
            return_as_pandas=True,
        )

    elif league == "wnba":
        from sportsdataverse.wnba import espn_wnba_schedule

        df = espn_wnba_schedule(
            dates=sdv_season,
            season_type=None,
            limit=limit,
            return_as_pandas=True,
        )

    else:
        raise ValueError(f"Unsupported league: {league}")

    if df is None:
        return pd.DataFrame()

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    return df


def validate_source_schema(
    df: pd.DataFrame,
    league: str,
    limit: int,
) -> None:
    missing = sorted(
        REQUIRED_SDV_COLUMNS - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{league}: SDV schedule schema missing "
            f"required columns: {missing}"
        )

    if league == "ncaam":
        missing_venue = sorted(
            NCAAM_REQUIRED_VENUE_COLUMNS
            - set(df.columns)
        )

        if missing_venue:
            raise ValueError(
                "ncaam: missing required venue/context "
                f"columns: {missing_venue}"
            )

    if len(df) >= limit:
        raise ValueError(
            f"{league}: returned {len(df)} records "
            f"at configured limit={limit}; "
            "possible truncated response"
        )


def source_team_name(
    row: pd.Series,
    side: str,
) -> str:
    for column in (
        f"{side}_display_name",
        f"{side}_short_display_name",
        f"{side}_location",
        f"{side}_name",
    ):
        value = clean(row.get(column))

        if value:
            return value

    return ""


def canonicalize_schedule(
    df: pd.DataFrame,
    *,
    league: str,
    internal_season: int,
    sdv_season: int,
    fetched_at_utc: str,
) -> list[dict[str, str]]:
    cfg = LEAGUES[league]
    team_map = load_team_map(league)

    rows: list[dict[str, str]] = []
    seen_ids = {}
    seen_identities = {}

    for _, source in df.iterrows():
        source_season = clean_id(
            source.get("season")
        )

        if (
            source_season
            and int(source_season) != int(sdv_season)
        ):
            raise ValueError(
                f"{league}: SDV returned "
                f"season={source_season}; "
                f"expected={sdv_season}"
            )

        game_id = clean_id(
            source.get("game_id")
            or source.get("id")
        )

        if not game_id:
            raise ValueError(
                f"{league}: blank game_id"
            )

        event_value = source.get("date")

        if not clean(event_value):
            event_value = source.get("start_date")

        event_dt = parse_event_datetime(
            event_value
        )

        home_team = normalize_team_name(
            source_team_name(source, "home"),
            team_map,
        )

        away_team = normalize_team_name(
            source_team_name(source, "away"),
            team_map,
        )

        if not home_team or not away_team:
            raise ValueError(
                f"{league}: game_id={game_id} "
                "has blank home/away team"
            )

        game_date = event_dt.strftime("%Y_%m_%d")

        identity = (
            game_date,
            home_team.casefold(),
            away_team.casefold(),
        )

        if (
            game_id in seen_ids
            and seen_ids[game_id] != identity
        ):
            raise ValueError(
                f"{league}: game_id={game_id} "
                "has conflicting identities"
            )

        if (
            identity in seen_identities
            and seen_identities[identity] != game_id
        ):
            raise ValueError(
                f"{league}: game identity {identity} "
                "has conflicting game IDs"
            )

        seen_ids[game_id] = identity
        seen_identities[identity] = game_id

        venue_full_name = clean(
            source.get("venue_full_name")
        )

        rows.append({
            "sport": cfg["sport"],
            "league": cfg["label"],
            "internal_season": str(internal_season),
            "sdv_season": str(sdv_season),
            "game_date": game_date,
            "game_time": event_dt.strftime("%I:%M %p"),
            "home_team": home_team,
            "away_team": away_team,
            "game_id": game_id,
            "home_team_id": clean_id(
                source.get("home_id")
            ),
            "away_team_id": clean_id(
                source.get("away_id")
            ),
            "neutral_site": bool_text(
                source.get("neutral_site")
            ),
            "venue_id": clean_id(
                source.get("venue_id")
            ),
            "venue_full_name": venue_full_name,
            "venue_name": normalize_venue_name(
                venue_full_name
            ),
            "status": clean(
                source.get("status_type_description")
            ),
            "source": "sportsdataverse_espn",
            "fetched_at_utc": fetched_at_utc,
        })

    return sort_rows(rows)


def sort_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            row["game_date"],
            row["game_time"],
            row["home_team"],
            row["away_team"],
            row["game_id"],
        ),
    )


def output_path(
    league: str,
    internal_season: int,
) -> Path:
    label = LEAGUES[league]["label"]

    return (
        OUTPUT_ROOT
        / league
        / f"{internal_season}_{label}_games.csv"
    )


def write_atomic(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:
        with tmp.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=FIELDNAMES,
            )
            writer.writeheader()
            writer.writerows(rows)

        os.replace(tmp, path)

    finally:
        if tmp.exists():
            tmp.unlink()


def read_cached_rows(
    path: Path,
    *,
    league: str,
    internal_season: int,
    sdv_season: int,
) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        if not set(FIELDNAMES).issubset(
            set(reader.fieldnames or [])
        ):
            raise ValueError(
                f"Invalid cached canonical headers: {path}"
            )

        rows = list(reader)

    for row in rows:
        if (
            clean(row["league"]).upper()
            != LEAGUES[league]["label"]
        ):
            raise ValueError(
                f"Cached league mismatch: {path}"
            )

        if clean_id(
            row["internal_season"]
        ) != str(internal_season):
            raise ValueError(
                f"Cached internal season mismatch: {path}"
            )

        if clean_id(
            row["sdv_season"]
        ) != str(sdv_season):
            raise ValueError(
                f"Cached SDV season mismatch: {path}"
            )

        validate_canonical_row(row)

    validate_canonical_identities(rows)

    return rows


def existing_file_is_valid(
    path: Path,
    *,
    league: str,
    internal_season: int,
    sdv_season: int,
) -> bool:
    if not path.exists():
        return False

    try:
        read_cached_rows(
            path,
            league=league,
            internal_season=internal_season,
            sdv_season=sdv_season,
        )
        return True
    except (
        OSError,
        UnicodeError,
        csv.Error,
        ValueError,
    ):
        return False


def validate_canonical_row(
    row: dict[str, str],
) -> None:
    required = (
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "league",
    )

    missing = [
        name
        for name in required
        if not clean(row.get(name))
    ]

    if missing:
        raise ValueError(
            f"Invalid canonical row; missing {missing}"
        )

    datetime.strptime(
        row["game_date"],
        "%Y_%m_%d",
    )


def identity_key(
    row: dict[str, str],
) -> tuple[str, str, str]:
    return (
        clean(row["game_date"]),
        clean(row["home_team"]).casefold(),
        clean(row["away_team"]).casefold(),
    )


def validate_canonical_identities(
    rows: list[dict[str, str]],
) -> None:
    ids = {}
    identities = {}

    for row in rows:
        validate_canonical_row(row)

        game_id = clean_id(row["game_id"])
        identity = identity_key(row)

        if (
            game_id in ids
            and ids[game_id] != identity
        ):
            raise ValueError(
                f"Game ID conflict: {game_id}"
            )

        if (
            identity in identities
            and identities[identity] != game_id
        ):
            raise ValueError(
                f"Game identity conflict: {identity}"
            )

        ids[game_id] = identity
        identities[identity] = game_id


def merge_canonical_rows(
    previous: list[dict[str, str]],
    fresh: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Preserve verified prior dates while allowing freshly
    retrieved records to update existing games.

    Never allow one game ID to change identity, or one
    identity to acquire a different game ID.
    """
    validate_canonical_identities(previous)
    validate_canonical_identities(fresh)

    by_id = {
        clean_id(row["game_id"]): row.copy()
        for row in previous
    }

    for row in fresh:
        game_id = clean_id(row["game_id"])

        if game_id in by_id:
            if (
                identity_key(by_id[game_id])
                != identity_key(row)
            ):
                raise ValueError(
                    f"Fresh schedule conflicts with "
                    f"cached game_id={game_id}"
                )

        by_id[game_id] = row.copy()

    result = list(by_id.values())
    validate_canonical_identities(result)

    return sort_rows(result)


def request_espn_nba_day(
    date_value: datetime,
) -> list[dict[str, Any]]:
    """
    Fetch an explicit ESPN NBA calendar date.

    A failed HTTP request is never treated as a valid
    empty slate.
    """
    date_string = date_value.strftime("%Y%m%d")
    last_error = None

    for attempt in range(1, HTTP_ATTEMPTS + 1):
        try:
            response = requests.get(
                ESPN_NBA_SCOREBOARD,
                params={
                    "dates": date_string,
                    "limit": 500,
                },
                timeout=HTTP_TIMEOUT,
                headers={
                    "Accept": "application/json",
                    "User-Agent": (
                        "bet-tracker-canonical-schedule/1.0"
                    ),
                },
            )

            response.raise_for_status()
            payload = response.json()

            if not isinstance(payload, dict):
                raise ValueError(
                    "ESPN scoreboard response is not "
                    "a JSON object"
                )

            if payload.get("code") in (400, 404, 500):
                raise ValueError(
                    f"ESPN error response: "
                    f"{payload.get('code')}"
                )

            events = payload.get("events")

            if not isinstance(events, list):
                raise ValueError(
                    "ESPN scoreboard response has "
                    "no valid events array"
                )

            log(
                "NBA DAILY FETCH | "
                f"date={date_string} "
                f"events={len(events)}"
            )

            return events

        except (
            requests.RequestException,
            ValueError,
        ) as exc:
            last_error = exc

            log(
                "NBA DAILY RETRY | "
                f"date={date_string} "
                f"attempt={attempt}/{HTTP_ATTEMPTS} "
                f"error={exc}"
            )

            if attempt < HTTP_ATTEMPTS:
                time.sleep(attempt * 2)

    raise RuntimeError(
        f"NBA daily scoreboard unavailable for "
        f"{date_string}: {last_error}"
    )


def nba_scoreboard_row(
    event: dict[str, Any],
    *,
    internal_season: int,
    sdv_season: int,
    fetched_at_utc: str,
    team_map: dict[str, str],
) -> dict[str, str]:
    game_id = clean_id(event.get("id"))

    if not game_id:
        raise ValueError(
            "NBA daily event has blank game ID"
        )

    competitions = event.get("competitions")

    if (
        not isinstance(competitions, list)
        or not competitions
    ):
        raise ValueError(
            f"NBA game_id={game_id}: "
            "missing competitions"
        )

    competition = competitions[0]

    if not isinstance(competition, dict):
        raise ValueError(
            f"NBA game_id={game_id}: "
            "invalid competition"
        )

    competitors = competition.get("competitors")

    if not isinstance(competitors, list):
        raise ValueError(
            f"NBA game_id={game_id}: "
            "invalid competitors"
        )

    teams = {}

    for competitor in competitors:
        if not isinstance(competitor, dict):
            continue

        side = clean(
            competitor.get("homeAway")
        ).lower()

        if side not in {"home", "away"}:
            continue

        team = competitor.get("team") or {}

        if not isinstance(team, dict):
            continue

        teams[side] = {
            "name": normalize_team_name(
                team.get("displayName")
                or team.get("name"),
                team_map,
            ),
            "id": clean_id(team.get("id")),
        }

    if set(teams) != {"home", "away"}:
        raise ValueError(
            f"NBA game_id={game_id}: "
            "missing home/away teams"
        )

    if (
        not teams["home"]["name"]
        or not teams["away"]["name"]
    ):
        raise ValueError(
            f"NBA game_id={game_id}: "
            "blank team name"
        )

    event_dt = parse_event_datetime(
        competition.get("date")
        or event.get("date")
    )

    venue = competition.get("venue") or {}

    if not isinstance(venue, dict):
        venue = {}

    venue_name = clean(
        venue.get("fullName")
        or venue.get("name")
    )

    status_object = event.get("status") or {}

    if not isinstance(status_object, dict):
        status_object = {}

    status_type = (
        status_object.get("type") or {}
    )

    if not isinstance(status_type, dict):
        status_type = {}

    status = clean(
        status_type.get("description")
        or status_type.get("name")
    )

    return {
        "sport": "basketball",
        "league": "NBA",
        "internal_season": str(internal_season),
        "sdv_season": str(sdv_season),
        "game_date": event_dt.strftime("%Y_%m_%d"),
        "game_time": event_dt.strftime("%I:%M %p"),
        "home_team": teams["home"]["name"],
        "away_team": teams["away"]["name"],
        "game_id": game_id,
        "home_team_id": teams["home"]["id"],
        "away_team_id": teams["away"]["id"],
        "neutral_site": bool_text(
            competition.get("neutralSite")
        ),
        "venue_id": clean_id(venue.get("id")),
        "venue_full_name": venue_name,
        "venue_name": normalize_venue_name(
            venue_name
        ),
        "status": status,
        "source": "espn_nba_daily_scoreboard",
        "fetched_at_utc": fetched_at_utc,
    }


def fetch_nba_current_window(
    *,
    internal_season: int,
    sdv_season: int,
    fetched_at_utc: str,
) -> tuple[list[dict[str, str]], set[str]]:
    """
    Supplement incomplete season responses.

    Dates are queried separately, because the annual
    request previously returned January games while
    omitting live October fixtures.

    The scoreboard is the authoritative source for
    the specific days successfully queried.
    """
    today = datetime.now(NY_TZ).date()
    team_map = load_team_map("nba")

    rows = []
    queried_dates: set[str] = set()

    start = today - timedelta(
        days=NBA_LOOKBACK_DAYS
    )

    total_days = (
        NBA_LOOKBACK_DAYS
        + NBA_LOOKAHEAD_DAYS
        + 1
    )

    for offset in range(total_days):
        day = start + timedelta(days=offset)

        # Only include dates from the requested season.
        # NBA seasons use the October start year.
        season_start = (
            day.year
            if day.month >= 10
            else day.year - 1
        )

        if season_start != internal_season:
            continue

        day_datetime = datetime.combine(
            day,
            datetime.min.time(),
            tzinfo=NY_TZ,
        )

        events = request_espn_nba_day(
            day_datetime
        )

        queried_dates.add(
            day.strftime("%Y_%m_%d")
        )

        for event in events:
            if not isinstance(event, dict):
                raise ValueError(
                    "Invalid NBA scoreboard event"
                )

            row = nba_scoreboard_row(
                event,
                internal_season=internal_season,
                sdv_season=sdv_season,
                fetched_at_utc=fetched_at_utc,
                team_map=team_map,
            )

            # ESPN may return an event whose ET game date
            # is adjacent to the requested UTC calendar date.
            # Only retain events in the mapped NBA season.
            event_day = datetime.strptime(
                row["game_date"],
                "%Y_%m_%d",
            )

            event_internal = (
                event_day.year
                if event_day.month >= 10
                else event_day.year - 1
            )

            if event_internal != internal_season:
                continue

            rows.append(row)

    validate_canonical_identities(rows)

    log(
        "NBA CURRENT WINDOW COMPLETE | "
        f"dates={len(queried_dates)} "
        f"games={len(rows)}"
    )

    return sort_rows(rows), queried_dates


def validate_nba_current_coverage(
    rows: list[dict[str, str]],
    daily_rows: list[dict[str, str]],
    queried_dates: set[str],
) -> None:
    """
    Require every game returned by the current ESPN
    scoreboard to be represented in the canonical output.

    Also require the current date to have been checked.

    A day with zero ESPN events is valid. An unrequested
    day or lost event is not.
    """
    today = datetime.now(NY_TZ).strftime(
        "%Y_%m_%d"
    )

    if today not in queried_dates:
        raise ValueError(
            "NBA current date was not queried "
            "from ESPN"
        )

    expected = {
        clean_id(row["game_id"]): identity_key(row)
        for row in daily_rows
    }

    actual = {
        clean_id(row["game_id"]): identity_key(row)
        for row in rows
    }

    missing = sorted(
        game_id
        for game_id, identity in expected.items()
        if actual.get(game_id) != identity
    )

    if missing:
        raise ValueError(
            "NBA current schedule lost ESPN "
            f"scoreboard games: {missing}"
        )

    log(
        "NBA CURRENT COVERAGE VERIFIED | "
        f"today={today} "
        f"scoreboard_games={len(expected)} "
        f"canonical_games={len(actual)}"
    )


def build_league(
    league: str,
    *,
    internal_season: int | None = None,
    allow_cached_on_fetch_error: bool = True,
) -> Path:
    if league not in LEAGUES:
        raise ValueError(
            f"Unsupported league: {league}"
        )

    internal = (
        int(internal_season)
        if internal_season is not None
        else current_internal_season(league)
    )

    mapped_sdv_season = sdv_season_id(
        league,
        internal,
        config_path=CONFIG_PATH,
    )

    config = LEAGUES[league]
    path = output_path(league, internal)

    fetched_at_utc = datetime.now(
        timezone.utc
    ).isoformat()

    log(
        "FETCH START | "
        f"league={config['label']} "
        f"internal_season={internal} "
        f"sdv_season={mapped_sdv_season}"
    )

    # WNBA_2026_LOCAL_REFRESH_HOOK_V1
    if league == "wnba" and int(internal) == 2026:
        from wnba_sdv_current_refresh import refresh as _wnba_refresh
        _wnba_refresh(Path.cwd())  # completed stats + daily schedule; abort on errors

    # WNBA_2027_SEASON_REFRESH_HOOK_V1
    if league == "wnba" and int(internal) == 2027:
        from wnba_sdv_2027_refresh import refresh as _wnba_2027_refresh
        _wnba_2027_refresh(Path.cwd())
        _wnba_2027_canonical = output_path("wnba", 2027)
        if _wnba_2027_canonical.is_file():
            return _wnba_2027_canonical  # avoids truncated season-wide SDV fetch
        raise RuntimeError("WNBA 2027 scoreboard has no regular/playoff schedule yet")

    # A corrupted existing cache must not be used as a
    # fallback or merged into fresh data.
    previous = []

    if path.exists():
        try:
            previous = read_cached_rows(
                path,
                league=league,
                internal_season=internal,
                sdv_season=mapped_sdv_season,
            )
        except Exception as exc:
            log(
                "CACHE INVALID | "
                f"league={config['label']} "
                f"path={path} error={exc}"
            )

    current_nba = (
        league == "nba"
        and internal == current_internal_season("nba")
    )

    try:
        df = fetch_schedule(
            league,
            mapped_sdv_season,
            config["limit"],
        )

        validate_source_schema(
            df,
            league,
            config["limit"],
        )

        fresh = canonicalize_schedule(
            df,
            league=league,
            internal_season=internal,
            sdv_season=mapped_sdv_season,
            fetched_at_utc=fetched_at_utc,
        )

        log(
            "SDV SEASON RESULT | "
            f"league={config['label']} "
            f"rows={len(fresh)}"
        )

        # Current NBA season: supplement the incomplete
        # season-level response with date-specific events.
        daily_rows = []
        queried_dates: set[str] = set()

        if current_nba:
            daily_rows, queried_dates = (
                fetch_nba_current_window(
                    internal_season=internal,
                    sdv_season=mapped_sdv_season,
                    fetched_at_utc=fetched_at_utc,
                )
            )

            fresh = merge_canonical_rows(
                fresh,
                daily_rows,
            )

        # Preserve previously verified canonical rows.
        # Newer data takes precedence for matching IDs.
        result = merge_canonical_rows(
            previous,
            fresh,
        )

        if current_nba:
            validate_nba_current_coverage(
                result,
                daily_rows,
                queried_dates,
            )

        if not result:
            raise ValueError(
                f"{league}: empty canonical schedule"
            )

        validate_canonical_identities(result)

        # This is the only point where the season
        # canonical CSV is replaced.
        write_atomic(path, result)

        log(
            "FETCH SUCCESS | "
            f"league={config['label']} "
            f"rows={len(result)} "
            f"previous={len(previous)} "
            f"fresh={len(fresh)} "
            f"path={path}"
        )

        return path

    except Exception as exc:
        log(
            "FETCH FAILURE | "
            f"league={config['label']} "
            f"error={exc}"
        )

        # Never fall back to the previously incomplete
        # current NBA schedule. It was the original
        # failure condition.
        if current_nba:
            raise

        if (
            allow_cached_on_fetch_error
            and existing_file_is_valid(
                path,
                league=league,
                internal_season=internal,
                sdv_season=mapped_sdv_season,
            )
        ):
            log(
                "FETCH WARNING | "
                f"league={config['label']} "
                f"using validated cached schedule "
                f"{path}"
            )

            return path

        raise


def build_current_canonical_games(
    leagues: list[str] | None = None,
) -> list[Path]:
    selected = leagues or [
        "nba",
        "ncaam",
        "wnba",
    ]

    ERROR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_FILE.write_text(
        "=== SDV CANONICAL GAMES "
        f"{datetime.now(timezone.utc).isoformat()} "
        "===\n",
        encoding="utf-8",
    )

    paths = []

    for league in selected:
        paths.append(
            build_league(league)
        )

    log(
        f"STATUS: SUCCESS | files={len(paths)}"
    )

    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build SportsDataVerse/ESPN canonical "
            "basketball game schedules."
        )
    )

    parser.add_argument(
        "--league",
        choices=sorted(LEAGUES),
        action="append",
        help=(
            "League to build; may be repeated. "
            "Default: NBA, NCAAM, WNBA."
        ),
    )

    parser.add_argument(
        "--internal-season",
        type=int,
        help=(
            "Explicit internal season. "
            "Requires exactly one --league."
        ),
    )

    parser.add_argument(
        "--no-cache-fallback",
        action="store_true",
        help=(
            "Do not use cached NCAAM/WNBA "
            "schedules when refresh fails."
        ),
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    selected = args.league or [
        "nba",
        "ncaam",
        "wnba",
    ]

    ERROR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_FILE.write_text(
        "=== SDV CANONICAL GAMES "
        f"{datetime.now(timezone.utc).isoformat()} "
        "===\n",
        encoding="utf-8",
    )

    try:
        if args.internal_season is not None:
            if len(selected) != 1:
                raise ValueError(
                    "--internal-season requires "
                    "exactly one --league"
                )

            build_league(
                selected[0],
                internal_season=args.internal_season,
                allow_cached_on_fetch_error=(
                    not args.no_cache_fallback
                ),
            )

        else:
            for league in selected:
                build_league(
                    league,
                    allow_cached_on_fetch_error=(
                        not args.no_cache_fallback
                    ),
                )

        log("STATUS: SUCCESS")

        print(
            "SDV canonical basketball "
            "games complete."
        )

    except Exception as exc:
        log(f"FATAL: {exc}")
        log(traceback.format_exc().rstrip())
        log("STATUS: FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
