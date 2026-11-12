
#!/usr/bin/env python3
"""Fetch completed NBA/NCAAM/WNBA final scores from ESPN.

Default: current New York date plus the previous seven days.

NBA team-name comparisons use:
    docs/win/basketball/maps/team_map_nba.csv

Aliases are resolved transitively. Placeholder names such as
TBC and TBD are supported without changing actual team identities.

Previously stored final scores and team names are preserved.
Genuine game-ID, matchup, and final-score conflicts are fatal.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import sys
import urllib.parse

from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


BASE = Path("docs/win/basketball")
RESULTS_BASE = BASE / "05_final_scores/results"
ERROR_DIR = BASE / "errors/05_final_scores"
LOG_FILE = ERROR_DIR / "00_basketball_final_scores.txt"
NBA_TEAM_MAP = BASE / "maps/team_map_nba.csv"
NY = ZoneInfo("America/New_York")
NCAAM_SCOREBOARD_LIMIT = 1000
LOCKED_PICKS_ROOT = BASE / "04_select"
CANONICAL_GAMES_ROOT = BASE / "00_intake/sdv/canonical_games"

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
    return datetime.now(
        tz=ZoneInfo("UTC")
    ).isoformat()


def log(message: str, level: str = "INFO") -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)

    line = (
        f"{now_utc()} | {level:<5} | "
        f"{message.rstrip()}"
    )
    print(line, flush=True)

    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def comparison_text(value: Any) -> str:
    return normalize_text(value).casefold()


@lru_cache(maxsize=1)
def load_nba_team_map() -> dict[str, str]:
    """Load and validate the existing NBA team-map CSV.

    Explicit alias mappings take precedence over implicit
    canonical-name self mappings.

    This prevents the erroneous conflict:
        TBC -> TBD
        Winner of EC -> TBC

    Alias chains are resolved to their final destination.

    Contradictory explicit aliases and circular mappings
    remain fatal.
    """
    if not NBA_TEAM_MAP.is_file():
        raise FileNotFoundError(
            f"NBA team map not found: {NBA_TEAM_MAP}"
        )

    explicit: dict[str, str] = {}
    canonical_names: set[str] = set()

    with NBA_TEAM_MAP.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "league",
            "alias",
            "canonical_team",
        }

        if not required.issubset(
            set(reader.fieldnames or [])
        ):
            raise ValueError(
                "NBA team map missing required columns: "
                f"{sorted(required)}"
            )

        for line_number, row in enumerate(
            reader, start=2
        ):
            if comparison_text(row.get("league")) != "nba":
                continue

            alias = comparison_text(row.get("alias"))
            canonical = comparison_text(
                row.get("canonical_team")
            )

            if not alias or not canonical:
                raise ValueError(
                    "Blank NBA alias/canonical name "
                    f"on line {line_number}"
                )

            previous = explicit.get(alias)

            if previous is not None and previous != canonical:
                raise ValueError(
                    "Conflicting NBA team-map alias "
                    f"on line {line_number}: "
                    f"{alias!r} maps to both "
                    f"{previous!r} and {canonical!r}"
                )

            explicit[alias] = canonical
            canonical_names.add(canonical)

    if not explicit:
        raise ValueError(
            "NBA team map contains no NBA mappings: "
            f"{NBA_TEAM_MAP}"
        )

    resolved: dict[str, str] = {}

    def resolve(name: str, visiting: set[str]) -> str:
        if name in resolved:
            return resolved[name]

        if name in visiting:
            raise ValueError(
                "Circular NBA team-map aliases involving "
                f"{name!r}"
            )

        destination = explicit.get(name)

        if destination is None or destination == name:
            resolved[name] = name
            return name

        visiting.add(name)
        try:
            terminal = resolve(destination, visiting)
        finally:
            visiting.remove(name)

        resolved[name] = terminal
        return terminal

    for alias in explicit:
        resolve(alias, set())

    for canonical in canonical_names:
        if canonical not in resolved:
            resolved[canonical] = canonical

    for alias, target in explicit.items():
        if resolved[alias] != resolved[target]:
            raise ValueError(
                "Unresolved NBA team-map conflict: "
                f"{alias!r} -> {target!r}"
            )

    log(
        "NBA TEAM MAP VALIDATED | "
        f"path={NBA_TEAM_MAP} "
        f"explicit_aliases={len(explicit)} "
        f"resolved_names={len(resolved)}"
    )

    return resolved


def normalize_team(value: Any, league: str) -> str:
    name = comparison_text(value)

    if not name:
        return ""

    if comparison_text(league) == "nba":
        return load_nba_team_map().get(name, name)

    return name


def parse_date_value(value: str) -> date:
    text = str(value).strip()

    for fmt in (
        "%Y_%m_%d",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y%m%d",
    ):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass

    raise argparse.ArgumentTypeError(
        f"Invalid date {value!r}; expected "
        "YYYY_MM_DD or YYYY-MM-DD"
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
        value = (
            value.get("displayValue")
            or value.get("value")
        )

    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError, OverflowError):
        return None


def fetch_json(url: str) -> dict:
    parsed = urllib.parse.urlsplit(url)

    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(
            f"Only HTTPS URLs are permitted: {url}"
        )

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
                "User-Agent": (
                    "Mozilla/5.0 "
                    "basketball-final-scores/1.0"
                ),
                "Accept": "application/json",
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
        http.client.HTTPException,
    ) as exc:
        raise RuntimeError(
            f"Network error fetching {url}: {exc}"
        ) from exc

    finally:
        connection.close()


def scoreboard_url(
    espn_slug: str,
    game_date: date,
) -> str:
    # NCAAM must use the same Division I group and limit as the canonical
    # schedule fetcher. Without groups=50 ESPN may return an incomplete slate.
    params = {
        "dates": game_date.strftime("%Y%m%d"),
        "limit": NCAAM_SCOREBOARD_LIMIT if espn_slug == "mens-college-basketball" else 500,
    }
    if espn_slug == "mens-college-basketball":
        params["groups"] = 50
    query = urllib.parse.urlencode(params)

    return (
        "https://site.api.espn.com/apis/site/v2/"
        "sports/basketball/"
        f"{espn_slug}/scoreboard?{query}"
    )


def parse_completed_games(
    payload: dict,
    league_key: str,
    game_date: date,
) -> list[dict]:
    league_label = LEAGUES[league_key]["label"]
    rows: list[dict] = []

    for event in payload.get("events") or []:
        competitions = event.get("competitions") or []

        if not competitions:
            continue

        competition = competitions[0]

        status = (
            competition.get("status")
            or event.get("status")
            or {}
        )
        status_type = status.get("type") or {}

        completed = bool(status_type.get("completed"))
        state = str(
            status_type.get("state") or ""
        ).strip().lower()

        if not completed and state != "post":
            continue

        home = None
        away = None

        for competitor in (
            competition.get("competitors") or []
        ):
            home_away = str(
                competitor.get("homeAway") or ""
            ).strip().lower()

            if home_away == "home":
                home = competitor
            elif home_away == "away":
                away = competitor

        if home is None or away is None:
            continue

        home_score = score_value(home.get("score"))
        away_score = score_value(away.get("score"))

        if home_score is None or away_score is None:
            continue

        home_team = (
            (home.get("team") or {}).get("displayName")
            or (home.get("team") or {}).get(
                "shortDisplayName"
            )
            or ""
        )

        away_team = (
            (away.get("team") or {}).get("displayName")
            or (away.get("team") or {}).get(
                "shortDisplayName"
            )
            or ""
        )

        game_id = str(
            event.get("id")
            or competition.get("id")
            or ""
        ).strip()

        if (
            not game_id
            or not str(home_team).strip()
            or not str(away_team).strip()
        ):
            continue

        margin = home_score - away_score

        rows.append({
            "sport": "Basketball",
            "league": league_label,
            "game_id": game_id,
            "game_date": game_date.strftime("%Y_%m_%d"),
            "home_team": str(home_team).strip(),
            "away_team": str(away_team).strip(),
            "home_score": home_score,
            "away_score": away_score,
            "total": home_score + away_score,
            "home_spread": margin,
            "away_spread": -margin,
        })

    return rows


def composite(row: dict) -> tuple[str, str, str]:
    league = str(row.get("league") or "")
    return (
        str(row.get("game_date") or "").strip().replace("-", "_"),
        normalize_team(row.get("home_team"), league),
        normalize_team(row.get("away_team"), league),
    )


def score_signature(row: dict) -> tuple[str, str]:
    return (
        str(row.get("home_score") or "").strip(),
        str(row.get("away_score") or "").strip(),
    )


def result_signature(row: dict) -> tuple[str, ...]:
    return tuple(
        str(row.get(field, "")).strip()
        for field in (
            "home_score",
            "away_score",
            "total",
            "home_spread",
            "away_spread",
        )
    )


def read_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        if not set(OUTPUT_COLUMNS).issubset(
            set(reader.fieldnames or [])
        ):
            raise ValueError(
                "Existing final-score CSV missing "
                f"required columns: {path}"
            )

        return list(reader)


def conflict_details(
    previous: dict,
    incoming: dict,
) -> str:
    fields = (
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "total",
        "home_spread",
        "away_spread",
    )

    previous_values = {
        field: str(
            previous.get(field, "")
        ).strip()
        for field in fields
    }

    incoming_values = {
        field: str(
            incoming.get(field, "")
        ).strip()
        for field in fields
    }

    changes = {
        field: {
            "stored": previous_values[field],
            "incoming": incoming_values[field],
        }
        for field in fields
        if previous_values[field] != incoming_values[field]
    }

    return (
        "stored="
        + json.dumps(
            previous_values,
            ensure_ascii=False,
        )
        + "; incoming="
        + json.dumps(
            incoming_values,
            ensure_ascii=False,
        )
        + "; differences="
        + json.dumps(
            changes,
            ensure_ascii=False,
        )
    )


def merge_rows(
    existing: list[dict], fetched: list[dict], label: str,
    output_path: Path | None = None,
) -> list[dict]:
    """Preserve stored finals and team names when ESPN revises a display name.

    Matching game IDs may have one renamed team only when the date, all
    result numbers, and the other team's identity agree. Changes to scores,
    dates, both teams, or home/away orientation are fatal. Matching team
    identities can also promote a previously blank game ID.
    """
    merged: list[dict] = []
    by_id: dict[str, dict] = {}
    by_comp: dict[tuple[str, str, str], dict] = {}
    source = f" file={output_path};" if output_path is not None else ""

    def signature(row: dict) -> tuple[str, ...]:
        return tuple(str(row.get(field, "")).strip() for field in (
            "home_score", "away_score", "total", "home_spread", "away_spread"
        ))

    def insert(row: dict, *, fetched_row: bool) -> None:
        gid = str(row.get("game_id") or "").strip()
        key = composite(row)
        scores = signature(row)
        if not all(key) or not all(scores[:2]):
            raise RuntimeError(f"{label}: incomplete final-score record;{source} {row}")
        if fetched_row and not gid:
            raise RuntimeError(f"{label}: incoming game_id is blank;{source} {row}")

        same_id = by_id.get(gid) if gid else None
        if same_id is not None:
            original_key = composite(same_id)
            if original_key == key and signature(same_id) == scores:
                return

            one_team_unchanged = (
                (original_key[1] == key[1] and original_key[2] != key[2])
                or (original_key[2] == key[2] and original_key[1] != key[1])
            )
            if (
                fetched_row
                and original_key[0] == key[0]
                and signature(same_id) == scores
                and one_team_unchanged
            ):
                log(
                    f"{label}: ESPN team name changed for game_id {gid}; "
                    f"preserved stored teams; stored="
                    f"{same_id.get('home_team')!r}/{same_id.get('away_team')!r}; "
                    f"incoming={row.get('home_team')!r}/{row.get('away_team')!r}",
                    "WARN",
                )
                return

            raise RuntimeError(
                f"{label}: conflicting final for game_id {gid};{source} "
                f"stored={same_id} incoming={row}"
            )

        previous = by_comp.get(key)
        if previous is not None:
            old_id = str(previous.get("game_id") or "").strip()
            if signature(previous) != scores:
                raise RuntimeError(
                    f"{label}: conflicting final for {key};{source} "
                    f"stored={previous} incoming={row}"
                )
            if old_id and gid and old_id != gid:
                raise RuntimeError(
                    f"{label}: conflicting game IDs for {key};{source} "
                    f"stored={old_id} incoming={gid}"
                )
            if not old_id and gid:
                previous["game_id"] = gid
                by_id[gid] = previous
            return

        saved = dict(row)
        merged.append(saved)
        by_comp[key] = saved
        if gid:
            by_id[gid] = saved

    for row in existing:
        insert(row, fetched_row=False)
    for row in fetched:
        insert(row, fetched_row=True)
    return merged


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(path.suffix + ".tmp")

    try:
        with temp.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=OUTPUT_COLUMNS,
            )
            writer.writeheader()

            for row in rows:
                writer.writerow({
                    column: row.get(column, "")
                    for column in OUTPUT_COLUMNS
                })

        temp.replace(path)

    finally:
        if temp.exists():
            temp.unlink()


def _locked_final_ids(league_key: str, game_date: date) -> set[str]:
    """Get final ESPN game IDs for this date's immutable locked selections.

    Only the canonical ESPN schedule is allowed to establish that a locked
    game is complete. A past date alone is not completion evidence.
    """
    date_key = game_date.strftime("%Y_%m_%d")
    picks_file = (
        LOCKED_PICKS_ROOT / league_key / "locked_picks"
        / f"{date_key}_{league_key}_selected.csv"
    )
    if not picks_file.exists():
        return set()

    with picks_file.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"game_id", "game_date"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Locked pick file missing {required}: {picks_file}")
        ids = {
            str(row.get("game_id") or "").strip()
            for row in reader
            if str(row.get("game_date") or "").strip().replace("-", "_") == date_key
            and str(row.get("game_id") or "").strip()
        }
    if not ids:
        return set()

    canonical_dir = CANONICAL_GAMES_ROOT / league_key
    if not canonical_dir.is_dir():
        raise FileNotFoundError(
            f"Canonical ESPN schedule unavailable for locked picks: {canonical_dir}"
        )
    files = sorted(canonical_dir.glob(f"*_{LEAGUES[league_key]['label']}_games.csv"))
    if not files:
        raise FileNotFoundError(f"No canonical ESPN schedule files: {canonical_dir}")

    completed = set()
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                gid = str(row.get("game_id") or "").strip()
                if gid not in ids:
                    continue
                if str(row.get("game_date") or "").strip().replace("-", "_") != date_key:
                    continue
                status = str(row.get("status") or "").strip().casefold()
                if status.startswith("final") or status in {"complete", "completed"}:
                    completed.add(gid)
    return completed


def _summary_final(league_key: str, game_date: date, game_id: str) -> dict:
    """Fetch one confirmed-complete ESPN game omitted by the date scoreboard.

    Fail rather than fabricate a result if the ESPN summary is incomplete or
    returns an unrelated game. This fallback only runs for locked selections.
    """
    slug = LEAGUES[league_key]["espn_slug"]
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/"
        f"{slug}/summary?event={urllib.parse.quote(game_id, safe='')}"
    )
    payload = fetch_json(url)
    header = payload.get("header") if isinstance(payload, dict) else None
    if not isinstance(header, dict):
        raise RuntimeError(f"{league_key} {game_id}: invalid ESPN summary header")
    competitions = header.get("competitions")
    if not isinstance(competitions, list) or not competitions:
        raise RuntimeError(f"{league_key} {game_id}: ESPN summary has no competition")

    competition = competitions[0]
    returned_id = str(header.get("id") or competition.get("id") or "").strip()
    if returned_id != game_id:
        raise RuntimeError(
            f"{league_key} {game_id}: ESPN summary returned game_id={returned_id!r}"
        )
    event = {"id": returned_id, "competitions": [competition]}
    rows = parse_completed_games({"events": [event]}, league_key, game_date)
    if len(rows) != 1 or rows[0]["game_id"] != game_id:
        raise RuntimeError(
            f"{league_key} {game_id}: ESPN summary lacks a verified final score"
        )
    return rows[0]


def _recover_missing_locked_finals(
    league_key: str,
    game_date: date,
    fetched: list[dict],
    existing: list[dict],
) -> list[dict]:
    """Recover ESPN-confirmed locked finals omitted by scoreboard filtering."""
    required = _locked_final_ids(league_key, game_date)
    found = {
        str(row.get("game_id") or "").strip()
        for row in [*fetched, *existing]
    }
    missing = sorted(required - found)
    if missing:
        log(
            f"[{LEAGUES[league_key]['label']}] {game_date}: "
            f"scoreboard omitted {len(missing)} confirmed final locked games: {missing}",
            "WARN",
        )
    for game_id in missing:
        fetched.append(_summary_final(league_key, game_date, game_id))
        log(
            f"[{LEAGUES[league_key]['label']}] {game_date}: "
            f"recovered locked final game_id={game_id} via ESPN summary"
        )
    return fetched


def fetch_one(
    league_key: str,
    game_date: date,
) -> tuple[int, Path | None]:
    cfg = LEAGUES[league_key]

    payload = fetch_json(
        scoreboard_url(
            cfg["espn_slug"],
            game_date,
        )
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise RuntimeError(f"{cfg['label']} {game_date}: invalid ESPN scoreboard events")
    if league_key == "ncaam" and len(payload["events"]) >= NCAAM_SCOREBOARD_LIMIT:
        raise RuntimeError(
            f"NCAAM {game_date}: scoreboard reached the configured limit "
            f"({NCAAM_SCOREBOARD_LIMIT}); possible truncated slate"
        )
    log(
        f"[{cfg['label']}] {game_date}: "
        f"scoreboard_events={len(payload['events'])}"
    )

    fetched = parse_completed_games(
        payload,
        league_key,
        game_date,
    )

    output = (
        RESULTS_BASE
        / league_key
        / (
            f"{game_date.strftime('%Y_%m_%d')}"
            f"_final_scores_{cfg['label']}.csv"
        )
    )

    existing = read_existing(output)
    # Independently verify locked, completed games; even a full-size or
    # successful ESPN scoreboard request can omit a Division I matchup.
    if league_key == "ncaam":
        fetched = _recover_missing_locked_finals(
            league_key, game_date, fetched, existing
        )

    if not fetched:
        log(f"[{cfg['label']}] {game_date}: no completed games")
        return 0, None

    merged = merge_rows(
        existing,
        fetched,
        f"{cfg['label']} {game_date}",
        output,
    )

    write_rows(output, merged)

    log(
        f"[{cfg['label']}] {game_date}: "
        f"fetched={len(fetched)} "
        f"stored={len(merged)} -> {output}"
    )

    return len(fetched), output


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch completed basketball "
            "final scores from ESPN."
        )
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
        help=(
            "Default days before today to fetch "
            "when no explicit date is supplied."
        ),
    )

    parser.add_argument(
        "--league",
        choices=tuple(LEAGUES),
        action="append",
        default=None,
        help=(
            "Limit to one or more leagues; "
            "may be repeated."
        ),
    )

    return parser.parse_args()


def resolve_dates(args) -> list[date]:
    if args.date is not None:
        if (
            args.start_date is not None
            or args.end_date is not None
        ):
            raise ValueError(
                "--date cannot be combined with "
                "--start-date/--end-date"
            )

        return [args.date]

    if (
        args.start_date is not None
        or args.end_date is not None
    ):
        if (
            args.start_date is None
            or args.end_date is None
        ):
            raise ValueError(
                "--start-date and --end-date "
                "must be supplied together"
            )

        return list(
            date_range(
                args.start_date,
                args.end_date,
            )
        )

    if args.lookback_days < 0:
        raise ValueError(
            "--lookback-days must be >= 0"
        )

    today = datetime.now(NY).date()
    start = today - timedelta(
        days=args.lookback_days
    )

    return list(date_range(start, today))


def main() -> None:
    ERROR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_FILE.write_text(
        "=== 00_basketball_final_scores RUN "
        f"{now_utc()} ===\n",
        encoding="utf-8",
    )

    args = parse_args()

    dates = resolve_dates(args)
    leagues = args.league or list(LEAGUES)

    if "nba" in leagues:
        aliases = load_nba_team_map()

        log(
            "NBA TEAM MAP LOADED | "
            f"path={NBA_TEAM_MAP} "
            f"aliases={len(aliases)}"
        )

    failures: list[str] = []
    total_completed = 0

    for game_date in dates:
        for league_key in leagues:
            try:
                count, _ = fetch_one(
                    league_key,
                    game_date,
                )

                total_completed += count

            except Exception as exc:
                message = (
                    f"[{LEAGUES[league_key]['label']}] "
                    f"{game_date}: {exc}"
                )

                log(message, "ERROR")
                failures.append(message)

    log(
        f"completed_games_fetched={total_completed} "
        f"dates={len(dates)} "
        f"leagues={len(leagues)}"
    )

    if failures:
        log(
            "STATUS: FAILED | "
            f"failures={len(failures)}",
            "ERROR",
        )

        raise RuntimeError("\n".join(failures))

    log("STATUS: SUCCESS")


if __name__ == "__main__":
    try:
        main()

    except Exception as main_exc:
        print(
            f"STATUS: FAILED | {main_exc}",
            file=sys.stderr,
        )
        sys.exit(1)
