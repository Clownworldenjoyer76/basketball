
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

    # Resolve explicit mappings first.
    # An alias can legitimately appear as another row's
    # canonical target, as with TBC and TBD.
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

    # Canonical names without their own explicit alias
    # map to themselves.
    for canonical in canonical_names:
        if canonical not in resolved:
            resolved[canonical] = canonical

    # Ensure the same canonical identity is never
    # ambiguously represented by a contradictory explicit
    # mapping. Self mappings and alias chains are valid.
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
    query = urllib.parse.urlencode({
        "dates": game_date.strftime("%Y%m%d"),
        "limit": 500,
    })

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
    """Preserve finals, merge equivalent NBA aliases, and promote blank IDs.

    A matching (date, canonical home, canonical away) with the same
    result is one game, even if an older stored row has no game_id.
    Conflicting results, matchups or nonblank game IDs remain fatal.
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
        if same_id is not None and (
            composite(same_id) != key or signature(same_id) != scores
        ):
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

    fetched = parse_completed_games(
        payload,
        league_key,
        game_date,
    )

    if not fetched:
        log(
            f"[{cfg['label']}] {game_date}: "
            "no completed games"
        )
        return 0, None

    output = (
        RESULTS_BASE
        / league_key
        / (
            f"{game_date.strftime('%Y_%m_%d')}"
            f"_final_scores_{cfg['label']}.csv"
        )
    )

    existing = read_existing(output)

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
