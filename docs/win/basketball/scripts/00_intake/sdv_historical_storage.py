#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/sdv_historical_storage.py
"""Build SportsDataVerse historical basketball storage for configured seasons.

NCAAM possessions/lineups are derived locally from SportsDataVerse's published
stats.ncaa.org NCAA MBB PBP parquet. This avoids the Akamai-protected live NCAA
fetch while still running the required SportsDataVerse possession/lineup
transforms on NCAA PBP (never ESPN PBP).

Individual NCAAM games missing from the published NCAA source are logged with
their exact ESPN game_id and NCAA contest id when available. They are skipped;
the entire season is not discarded because SportsDataVerse's published NCAA
dataset can legitimately omit a small number of canonical ESPN games.

No zero-row placeholder rows are ever created.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
import importlib
import importlib.metadata
import io
import json
import re
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

from sdv_season_mapping import sdv_season_id


def optional_callable_attr(
    module: Any,
    name: str,
) -> Callable[..., Any] | None:
    candidate = getattr(
        module,
        name,
        None,
    )

    return (
        candidate
        if callable(candidate)
        else None
    )


BASE = Path("docs/win/basketball")
CONFIG_PATH = BASE / "config/sdv_storage.yaml"
SEASON_CONFIG_PATH = BASE / "config/sdv_seasons.yaml"
ERROR_DIR = BASE / "errors/00_intake"
LOG_FILE = ERROR_DIR / "sdv_historical_storage.txt"

LEAGUES = ("nba", "ncaam", "wnba")

TABLES = (
    "games",
    "team_game",
    "player_game",
    "rosters",
    "pbp",
    "possessions",
    "lineups",
    "shots",
)

LOADERS: dict[str, dict[str, tuple[str, str] | None]] = {
    "nba": {
        "games": ("sportsdataverse.nba", "load_nba_schedule"),
        "team_game": ("sportsdataverse.nba", "load_nba_team_boxscore"),
        "player_game": ("sportsdataverse.nba", "load_nba_player_boxscore"),
        "rosters": ("sportsdataverse.nba", "load_nba_rosters"),
        "pbp": ("sportsdataverse.nba", "load_nba_pbp"),
        "possessions": ("sportsdataverse.nba", "load_nba_stats_possessions"),
        "lineups": ("sportsdataverse.nba", "load_nba_stats_game_lineups"),
        "shots": ("sportsdataverse.nba", "load_nba_shots"),
    },
    "ncaam": {
        "games": ("sportsdataverse.mbb", "load_mbb_schedule"),
        "team_game": ("sportsdataverse.mbb", "load_mbb_team_boxscore"),
        "player_game": ("sportsdataverse.mbb", "load_mbb_player_boxscore"),
        "rosters": ("sportsdataverse.mbb", "load_mbb_rosters"),
        "pbp": ("sportsdataverse.mbb", "load_mbb_pbp"),
        "possessions": None,
        "lineups": None,
        "shots": ("sportsdataverse.mbb", "load_mbb_shots"),
    },
    "wnba": {
        "games": ("sportsdataverse.wnba", "load_wnba_schedule"),
        "team_game": ("sportsdataverse.wnba", "load_wnba_team_boxscore"),
        "player_game": ("sportsdataverse.wnba", "load_wnba_player_boxscore"),
        "rosters": ("sportsdataverse.wnba", "load_wnba_rosters"),
        "pbp": ("sportsdataverse.wnba", "load_wnba_pbp"),
        "possessions": ("sportsdataverse.wnba", "load_wnba_stats_possessions"),
        "lineups": ("sportsdataverse.wnba", "load_wnba_stats_game_lineups"),
        "shots": ("sportsdataverse.wnba", "load_wnba_shots"),
    },
}


LOADER_FALLBACKS: dict[
    tuple[str, str],
    tuple[str, str, str],
] = {
    ("nba", "rosters"): (
        "sportsdataverse.nba",
        "load_nba_player_core",
        "sdv",
    ),
    ("ncaam", "rosters"): (
        "sportsdataverse.mbb",
        "load_mbb_player_core",
        "sdv",
    ),
    ("wnba", "rosters"): (
        "sportsdataverse.wnba",
        "load_wnba_player_core",
        "sdv",
    ),
}

RELEASE_FALLBACKS = {
    ("nba", "possessions"): (
        "sportsdataverse.nba.load_nba_stats_possessions",
        "https://github.com/sportsdataverse/"
        "sportsdataverse-data/releases/download/"
        "nba_stats_possessions/"
        "nba_possessions_{year}.parquet",
    ),
    ("nba", "lineups"): (
        "sportsdataverse.nba.load_nba_stats_game_lineups",
        "https://github.com/sportsdataverse/"
        "sportsdataverse-data/releases/download/"
        "nba_stats_game_lineups/"
        "nba_lineups_{year}.parquet",
    ),
    ("ncaam", "shots"): (
        "sportsdataverse.mbb.load_mbb_shots",
        "https://github.com/sportsdataverse/"
        "sportsdataverse-data/releases/download/"
        "espn_mens_college_basketball_shots/"
        "shots_{year}.parquet",
    ),
    ("wnba", "shots"): (
        "sportsdataverse.wnba.load_wnba_shots",
        "https://github.com/sportsdataverse/"
        "sportsdataverse-data/releases/download/"
        "espn_wnba_shots/"
        "shots_{year}.parquet",
    ),
    ("wnba", "possessions"): (
        "sportsdataverse.wnba.load_wnba_stats_possessions",
        "https://github.com/sportsdataverse/"
        "sportsdataverse-data/releases/download/"
        "wnba_stats_possessions/"
        "wnba_possessions_{year}.parquet",
    ),
    ("wnba", "lineups"): (
        "sportsdataverse.wnba.load_wnba_stats_game_lineups",
        "https://github.com/sportsdataverse/"
        "sportsdataverse-data/releases/download/"
        "wnba_stats_game_lineups/"
        "wnba_lineups_{year}.parquet",
    ),
}

PRO_SCHEDULE_CROSSWALKS = {
    "nba": {
        "module": "sportsdataverse.nba",
        "loader": "load_nba_schedule_crosswalk",
        "native_game_id": "nba_game_id",
        "fallback_url": (
            "https://github.com/sportsdataverse/"
            "sportsdataverse-data/releases/download/"
            "nba_crosswalk/"
            "nba_schedule_crosswalk_{season}.parquet"
        ),
        "minimum_season": 2026,
    },
    "wnba": {
        "module": "sportsdataverse.wnba",
        "loader": "load_wnba_schedule_crosswalk",
        "native_game_id": "wnba_game_id",
        "fallback_url": (
            "https://github.com/sportsdataverse/"
            "sportsdataverse-data/releases/download/"
            "wnba_crosswalk/"
            "wnba_schedule_crosswalk_{season}.parquet"
        ),
        "minimum_season": 2026,
    },
}

NBA_STATS_SCHEDULE_FALLBACK_URL = (
    "https://github.com/sportsdataverse/"
    "sportsdataverse-data/releases/download/"
    "nba_stats_schedules/"
    "nba_schedule_{season}.parquet"
)

WNBA_STATS_SCHEDULE_FALLBACK_URL = (
    "https://github.com/sportsdataverse/"
    "sportsdataverse-data/releases/download/"
    "wnba_stats_schedules/"
    "wnba_schedule_{season}.parquet"
)

NCAAM_NCAA_PBP_URL = (
    "https://raw.githubusercontent.com/"
    "sportsdataverse/ncaa-mbb-hoops-data/main/"
    "mbb/pbp/parquet/"
    "ncaa_mbb_pbp_{season}.parquet"
)

NCAAM_GAME_XWALK_URL = (
    "https://raw.githubusercontent.com/"
    "sportsdataverse/ncaa-mbb-hoops-raw/main/"
    "mbb/xwalk/espn_game_id/{season}.json"
)

ALIASES = {
    "games": {
        "game_id": ("game_id", "id"),
        "game_date": ("game_date", "date"),
        "home_team_id": ("home_team_id", "home_id"),
        "away_team_id": ("away_team_id", "away_id"),
    },
    "team_game": {
        "game_id": ("game_id",),
        "team_id": ("team_id",),
    },
    "player_game": {
        "game_id": ("game_id",),
        "player_id": ("player_id", "athlete_id"),
        "team_id": ("team_id",),
    },
    "rosters": {
        "player_id": ("player_id", "athlete_id"),
        "team_id": ("team_id", "current_team_id"),
    },
    "pbp": {
        "game_id": ("game_id",),
        "play_id": ("play_id", "id"),
    },
    "possessions": {
        "game_id": ("game_id",),
        "team_id": ("team_id",),
    },
    "lineups": {
        "game_id": ("game_id",),
        "team_id": ("team_id",),
    },
    "shots": {
        "game_id": ("game_id",),
        "shot_id": ("shot_id", "id"),
        "player_id": ("player_id", "athlete_id", "person_id"),
        "team_id": ("team_id",),
    },
}


def pl():
    return importlib.import_module("polars")


def log(message: str) -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            f"{datetime.now(timezone.utc).isoformat()} | "
            f"{message}\n"
        )


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def snake(value: str) -> str:
    text = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1_\2",
        clean(value),
    )

    text = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        text,
    )

    return re.sub(
        r"_+",
        "_",
        text,
    ).strip("_").lower()


def to_pl(frame):
    pl_module = pl()

    if frame is None:
        return pl_module.DataFrame()

    if isinstance(frame, pl_module.DataFrame):
        return frame

    if hasattr(frame, "collect"):
        collected = frame.collect()

        if isinstance(collected, pl_module.DataFrame):
            return collected

    if hasattr(frame, "to_pandas"):
        return pl_module.from_pandas(
            frame.to_pandas()
        )

    return pl_module.DataFrame(frame)


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    cfg = (
        yaml.safe_load(
            path.read_text(
                encoding="utf-8",
            )
        )
        or {}
    )

    if cfg.get("schema_version") != 1:
        raise ValueError(
            "sdv_storage.yaml schema_version must be 1"
        )

    if (
        clean(
            cfg.get(
                "storage",
                {},
            ).get(
                "format"
            )
        ).lower()
        != "parquet"
    ):
        raise ValueError(
            "storage.format must be parquet"
        )

    configured_tables = [
        clean(x).lower()
        for x in cfg.get(
            "tables",
            [],
        )
    ]

    if configured_tables != list(TABLES):
        raise ValueError(
            "tables must exactly equal "
            f"{list(TABLES)}"
        )

    snapshots = cfg.get(
        "sportsbook_snapshots",
        {},
    )

    if snapshots.get(
        "reuse_existing"
    ) is not True:
        raise ValueError(
            "sportsbook_snapshots."
            "reuse_existing must be true"
        )

    if snapshots.get(
        "duplicate_into_sdv_storage"
    ) is not False:
        raise ValueError(
            "sportsbook snapshots must not "
            "be copied into SDV history"
        )

    seasons = cfg.get(
        "historical_internal_seasons",
        {},
    )

    for league in LEAGUES:
        values = seasons.get(league)

        if (
            not isinstance(values, list)
            or not values
        ):
            raise ValueError(
                "historical_internal_seasons."
                f"{league} must be non-empty"
            )

    return cfg


def verify_version(
    cfg: dict[str, Any],
) -> str:
    expected = clean(
        cfg[
            "sportsdataverse"
        ][
            "expected_version"
        ]
    )

    installed = (
        importlib.metadata.version(
            "sportsdataverse"
        )
    )

    if installed != expected:
        raise RuntimeError(
            "sportsdataverse version mismatch: "
            f"installed={installed}, "
            f"expected={expected}"
        )

    return installed


def storage_root(
    cfg: dict[str, Any],
) -> Path:
    return Path(
        clean(
            cfg[
                "storage"
            ][
                "root"
            ]
        )
    )


def compression(
    cfg: dict[str, Any],
) -> str:
    return clean(
        cfg[
            "storage"
        ].get(
            "compression"
        )
        or "zstd"
    ).lower()


def partition(
    root: Path,
    league: str,
    season: int,
) -> Path:
    return (
        root
        / league
        / str(season)
    )


def table_path(
    root: Path,
    league: str,
    season: int,
    table: str,
) -> Path:
    return (
        partition(
            root,
            league,
            season,
        )
        / f"{table}.parquet"
    )


def release_loader_season(
    league: str,
    table: str,
    internal_season: int,
    sdv_season: int,
) -> int:
    if (
        league == "nba"
        and table
        in {
            "possessions",
            "lineups",
        }
    ):
        return int(
            internal_season
        )

    return int(
        sdv_season
    )


def loader_fallback_season(
    mode: str,
    internal_season: int,
    sdv_season: int,
) -> int:
    if mode == "internal":
        return int(
            internal_season
        )

    if mode == "sdv":
        return int(
            sdv_season
        )

    raise ValueError(
        "Unsupported loader fallback "
        f"season mode: {mode}"
    )


def fallback_asset_year(
    league: str,
    table: str,
    internal_season: int,
    sdv_season: int,
) -> int:
    if (
        league == "nba"
        and table
        in {
            "possessions",
            "lineups",
        }
    ):
        return (
            int(
                internal_season
            )
            + 1
        )

    return int(
        sdv_season
    )


def http_get(
    url: str,
    *,
    timeout: int = 180,
) -> requests.Response:
    response = requests.get(
        url,
        timeout=timeout,
    )

    if response.status_code == 404:
        raise RuntimeError(
            "SportsDataVerse source "
            f"not published: {url}"
        )

    response.raise_for_status()

    return response


def release_fallback(
    league: str,
    table: str,
    internal_season: int,
    sdv_season: int,
):
    pl_module = pl()

    source, template = (
        RELEASE_FALLBACKS[
            (
                league,
                table,
            )
        ]
    )

    year = fallback_asset_year(
        league,
        table,
        internal_season,
        sdv_season,
    )

    url = template.format(
        year=year
    )

    log(
        "RELEASE FALLBACK | "
        f"league={league} "
        f"table={table} "
        f"asset_year={year} "
        f"url={url}"
    )

    response = http_get(url)

    return (
        pl_module.read_parquet(
            io.BytesIO(
                response.content
            )
        ),
        (
            f"{source} "
            "[release fallback]"
        ),
    )


def normalize_date_key(
    value: Any,
) -> str:
    text = clean(value)

    if not text:
        return ""

    return text[:10]


def team_variants(
    value: Any,
) -> set[str]:
    text = clean(
        value
    ).lower()

    if not text:
        return set()

    tokens = re.findall(
        r"[a-z0-9]+",
        text,
    )

    if not tokens:
        return set()

    variants = {
        "".join(tokens),
        tokens[-1],
    }

    if len(tokens) >= 2:
        variants.add(
            "".join(
                tokens[-2:]
            )
        )

    return {
        value
        for value
        in variants
        if value
    }


def normalize_frame_columns(
    frame,
):
    df = to_pl(frame)

    old_columns = list(
        df.columns
    )

    new_columns = [
        snake(column)
        for column
        in old_columns
    ]

    if (
        len(new_columns)
        != len(set(new_columns))
    ):
        raise RuntimeError(
            "column collision after "
            "normalization"
        )

    if old_columns != new_columns:
        df = df.rename(
            dict(
                zip(
                    old_columns,
                    new_columns,
                )
            )
        )

    return df


def normalize_legacy_schedule_date(value: Any) -> str:
    """Return YYYY-MM-DD for schedule dates without assuming one source format."""
    text = clean(value)

    if not text:
        return ""

    iso_match = re.match(
        r"^(\d{4}-\d{2}-\d{2})",
        text,
    )

    if iso_match:
        return iso_match.group(1)

    for date_format in (
        "%b %d, %Y",
        "%B %d, %Y",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(
                text,
                date_format,
            ).date().isoformat()
        except ValueError:
            continue

    return ""


def load_nba_stats_schedule(
    internal_season: int,
    sdv_season: int,
):
    """Load the published NBA Stats schedule used for legacy ID matching.

    The SportsDataVerse loader uses the NBA season start year, while the
    published release asset used as fallback is keyed by the season end year.
    """
    pl_module = pl()

    module_name = (
        "sportsdataverse.nba"
    )

    loader_name = (
        "load_nba_stats_schedules"
    )

    module = importlib.import_module(
        module_name
    )

    loader = optional_callable_attr(
        module,
        loader_name,
    )

    if loader is not None:
        try:
            frame = loader(
                seasons=[
                    int(
                        internal_season
                    )
                ],
                return_as_pandas=False,
            )

            df = normalize_frame_columns(
                frame
            )

            if df.height:
                return (
                    df,
                    (
                        f"{module_name}."
                        f"{loader_name}"
                    ),
                )

            loader_error = RuntimeError(
                "loader returned zero rows"
            )

        except Exception as exc:
            loader_error = exc

        log(
            "NBA STATS SCHEDULE LOADER FAILED | "
            f"loader_season={internal_season} "
            f"fallback_asset_season={sdv_season} "
            f"error={loader_error} "
            "action=release_fallback"
        )

    else:
        loader_error = RuntimeError(
            "SportsDataVerse loader missing: "
            f"{module_name}.{loader_name}"
        )

        log(
            "NBA STATS SCHEDULE LOADER FAILED | "
            f"loader_season={internal_season} "
            f"fallback_asset_season={sdv_season} "
            f"error={loader_error} "
            "action=release_fallback"
        )

    url = (
        NBA_STATS_SCHEDULE_FALLBACK_URL.format(
            season=int(
                sdv_season
            )
        )
    )

    try:
        response = http_get(
            url
        )

        df = normalize_frame_columns(
            pl_module.read_parquet(
                io.BytesIO(
                    response.content
                )
            )
        )

    except Exception as exc:
        raise RuntimeError(
            "NBA Stats schedule loader and published "
            "release fallback both failed; "
            f"loader_season={internal_season}; "
            f"fallback_asset_season={sdv_season}; "
            f"loader_error={loader_error}; "
            f"fallback_error={exc}"
        ) from exc

    if df.height == 0:
        raise RuntimeError(
            "NBA Stats schedule published release "
            "returned zero rows; "
            f"loader_season={internal_season}; "
            f"fallback_asset_season={sdv_season}; "
            f"loader_error={loader_error}; "
            f"url={url}"
        )

    return (
        df,
        (
            f"{module_name}."
            f"{loader_name} "
            "[release fallback]"
        ),
    )


def build_nba_legacy_schedule_crosswalk(cfg: dict[str, Any], internal_season: int, sdv_season: int):
    """Map historical NBA Stats game IDs to canonical ESPN game IDs.

    SportsDataVerse 0.0.75 does not support load_nba_schedule_crosswalk
    before season 2026. Historical NBA Stats schedule data can arrive as
    either team rows (one row per team) or game rows (explicit home/away
    columns).

    NBA Stats team-row ``matchup`` values are not reliable for determining
    home/away. Some games contain ``@`` on both team rows. For every team-row
    game, this function therefore ignores ``matchup`` entirely, requires
    exactly two unique teams, and matches the game to the canonical ESPN
    schedule by game date plus the unordered pair of teams. The canonical ESPN
    schedule supplies the final home/away orientation. Scores are used only as
    a deterministic disambiguator if more than one canonical candidate remains.
    """
    _py_r1000_NONE = 0
    _py_r1000_RETURN = 1
    _py_r1000_BREAK = 2
    _py_r1000_CONTINUE = 3

    def _py_r1000_impl():
        nonlocal cfg, internal_season, sdv_season
        ambiguous: object
        away_name_columns: object
        away_pts: object
        away_score: object
        away_variants: object
        bad_date_ids: object
        by_date: object
        candidate: object
        candidate_variants: object
        candidates: object
        canonical_bad_dates: object
        canonical_columns: object
        chosen: object
        columns: object
        conflict_found: object
        conflicting_duplicate_ids: object
        date_key: object
        duplicate_espn: object
        duplicate_found: object
        duplicate_native: object
        espn_game_id: object
        exact_duplicate_rows: object
        existing: object
        existing_side: object
        game: object
        game_id: object
        game_identity: object
        game_row_required: object
        games: object
        games_file: object
        has_game_rows: object
        has_team_rows: object
        home_name_columns: object
        home_pts: object
        home_score: object
        home_variants: object
        incomplete_ids: object
        mappings: object
        match_records: object
        maybe_float: object
        method: object
        missing_games: object
        native_game_id: object
        optional_column: object
        orientation_match: object
        orientations: object
        pl_module: object
        raw_game: object
        raw_games: object
        record: object
        required_games: object
        root: object
        row: object
        schema_used: object
        score_column: object
        score_matches: object
        selected_columns: object
        side: object
        side_identity: object
        side_payload: object
        side_team_identity: object
        side_variants: object
        stats_away_variants: object
        stats_games: object
        stats_home_variants: object
        stats_schedule: object
        stats_source: object
        team_a_pts: object
        team_b_pts: object
        team_row_required: object
        teams: object
        unmatched: object
        unordered_orientations: object
        unordered_team_row_games: object
        xwalk: object

        def _py_r1000_if_1():
            nonlocal date_key, exact_duplicate_rows, existing, game, native_game_id, optional_column, row, schema_used, selected_columns

            def _py_r1000_loop_2():
                if optional_column in columns:
                    selected_columns.append(optional_column)
                return (_py_r1000_NONE, None)

            def _py_r1000_loop_4():
                nonlocal date_key, exact_duplicate_rows, existing, game, native_game_id
                native_game_id = clean(row.get('game_id'))
                if not native_game_id:
                    return (_py_r1000_CONTINUE, None)
                date_key = normalize_legacy_schedule_date(row.get('game_date'))
                if not date_key:
                    bad_date_ids.add(native_game_id)
                    return (_py_r1000_CONTINUE, None)
                game = {'game_date_key': date_key, 'match_mode': 'explicit_home_away', 'home': side_payload(team_name=row.get('home_team_name'), team_abbreviation=row.get('home_team_abbreviation'), pts=row.get('home_pts')), 'away': side_payload(team_name=row.get('away_team_name'), team_abbreviation=row.get('away_team_abbreviation'), pts=row.get('away_pts')), 'team_a': None, 'team_b': None}
                if not side_variants(game['home']) or not side_variants(game['away']):
                    incomplete_ids.add(native_game_id)
                    return (_py_r1000_CONTINUE, None)
                existing = stats_games.get(native_game_id)
                if existing is None:
                    stats_games[native_game_id] = game
                    return (_py_r1000_CONTINUE, None)
                if game_identity(existing) == game_identity(game):
                    exact_duplicate_rows += 1
                else:
                    conflicting_duplicate_ids.add(native_game_id)
                return (_py_r1000_NONE, None)
            schema_used = 'game_rows'
            selected_columns = ['game_id', 'game_date', 'home_team_name', 'home_team_abbreviation', 'away_team_name', 'away_team_abbreviation']
            for optional_column in ('home_pts', 'away_pts'):
                _py_r1000_result_3 = _py_r1000_loop_2()
                if _py_r1000_result_3[0] == _py_r1000_RETURN:
                    return _py_r1000_result_3
                if _py_r1000_result_3[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_3[0] == _py_r1000_CONTINUE:
                    continue
            for row in stats_schedule.select(selected_columns).iter_rows(named=True):
                _py_r1000_result_5 = _py_r1000_loop_4()
                if _py_r1000_result_5[0] == _py_r1000_RETURN:
                    return _py_r1000_result_5
                if _py_r1000_result_5[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_5[0] == _py_r1000_CONTINUE:
                    continue
            return (_py_r1000_NONE, None)

        def _py_r1000_else_7():
            nonlocal conflict_found, date_key, duplicate_found, exact_duplicate_rows, existing_side, native_game_id, raw_game, raw_games, row, schema_used, selected_columns, side, teams, unordered_team_row_games

            def _py_r1000_loop_8():
                nonlocal conflict_found, date_key, duplicate_found, exact_duplicate_rows, existing_side, native_game_id, raw_game, side

                def _py_r1000_loop_9():
                    nonlocal conflict_found, duplicate_found
                    if side_team_identity(existing_side) != side_team_identity(side):
                        return (_py_r1000_CONTINUE, None)
                    if side_identity(existing_side) == side_identity(side):
                        duplicate_found = True
                    else:
                        conflict_found = True
                    return (_py_r1000_BREAK, None)
                native_game_id = clean(row.get('game_id'))
                if not native_game_id:
                    return (_py_r1000_CONTINUE, None)
                date_key = normalize_legacy_schedule_date(row.get('game_date'))
                if not date_key:
                    bad_date_ids.add(native_game_id)
                    return (_py_r1000_CONTINUE, None)
                side = side_payload(team_name=row.get('team_name'), team_abbreviation=row.get('team_abbreviation'), pts=row.get('pts'))
                if not side_variants(side):
                    incomplete_ids.add(native_game_id)
                    return (_py_r1000_CONTINUE, None)
                raw_game = raw_games.setdefault(native_game_id, {'game_date_key': date_key, 'teams': []})
                if raw_game['game_date_key'] != date_key:
                    conflicting_duplicate_ids.add(native_game_id)
                    return (_py_r1000_CONTINUE, None)
                duplicate_found = False
                conflict_found = False
                for existing_side in raw_game['teams']:
                    _py_r1000_result_10 = _py_r1000_loop_9()
                    if _py_r1000_result_10[0] == _py_r1000_RETURN:
                        return _py_r1000_result_10
                    if _py_r1000_result_10[0] == _py_r1000_BREAK:
                        break
                    if _py_r1000_result_10[0] == _py_r1000_CONTINUE:
                        continue
                if duplicate_found:
                    exact_duplicate_rows += 1
                    return (_py_r1000_CONTINUE, None)
                if conflict_found:
                    conflicting_duplicate_ids.add(native_game_id)
                    return (_py_r1000_CONTINUE, None)
                raw_game['teams'].append(side)
                return (_py_r1000_NONE, None)

            def _py_r1000_loop_12():
                nonlocal teams, unordered_team_row_games
                if native_game_id in conflicting_duplicate_ids:
                    return (_py_r1000_CONTINUE, None)
                teams = raw_game['teams']
                if len(teams) != 2:
                    incomplete_ids.add(native_game_id)
                    return (_py_r1000_CONTINUE, None)
                if side_team_identity(teams[0]) == side_team_identity(teams[1]):
                    incomplete_ids.add(native_game_id)
                    return (_py_r1000_CONTINUE, None)
                stats_games[native_game_id] = {'game_date_key': raw_game['game_date_key'], 'match_mode': 'unordered_teams', 'home': None, 'away': None, 'team_a': teams[0], 'team_b': teams[1]}
                unordered_team_row_games += 1
                return (_py_r1000_NONE, None)
            schema_used = 'team_rows'
            selected_columns = ['game_id', 'game_date', 'team_name', 'team_abbreviation']
            if 'pts' in columns:
                selected_columns.append('pts')
            raw_games = {}
            for row in stats_schedule.select(selected_columns).iter_rows(named=True):
                _py_r1000_result_11 = _py_r1000_loop_8()
                if _py_r1000_result_11[0] == _py_r1000_RETURN:
                    return _py_r1000_result_11
                if _py_r1000_result_11[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_11[0] == _py_r1000_CONTINUE:
                    continue
            for native_game_id, raw_game in raw_games.items():
                _py_r1000_result_13 = _py_r1000_loop_12()
                if _py_r1000_result_13[0] == _py_r1000_RETURN:
                    return _py_r1000_result_13
                if _py_r1000_result_13[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_13[0] == _py_r1000_CONTINUE:
                    continue
            return (_py_r1000_NONE, None)

        def _py_r1000_loop_15():
            if score_column in games.columns:
                canonical_columns.append(score_column)
            return (_py_r1000_NONE, None)

        def _py_r1000_loop_17():
            nonlocal date_key, espn_game_id

            def _py_r1000_if_18():
                if espn_game_id:
                    canonical_bad_dates.append(espn_game_id)
                return (_py_r1000_CONTINUE, None)
            espn_game_id = clean(row.get('game_id'))
            date_key = normalize_legacy_schedule_date(row.get('game_date'))
            if not date_key:
                _py_r1000_result_19 = _py_r1000_if_18()
                if _py_r1000_result_19[0] != _py_r1000_NONE:
                    return _py_r1000_result_19
            by_date.setdefault(date_key, []).append(row)
            return (_py_r1000_NONE, None)

        def _py_r1000_loop_21():
            nonlocal away_pts, away_score, away_variants, candidate, candidates, chosen, espn_game_id, game, home_pts, home_score, home_variants, match_records, method, orientation_match, orientations, record, score_matches, stats_away_variants, stats_home_variants, team_a_pts, team_b_pts

            def _py_r1000_if_22():
                nonlocal candidate, orientations

                def _py_r1000_loop_23():
                    nonlocal orientations
                    orientations = unordered_orientations(game['team_a'], game['team_b'], candidate)
                    if orientations:
                        match_records.append({'candidate': candidate, 'orientations': orientations})
                    return (_py_r1000_NONE, None)
                for candidate in candidates:
                    _py_r1000_result_24 = _py_r1000_loop_23()
                    if _py_r1000_result_24[0] == _py_r1000_RETURN:
                        return _py_r1000_result_24
                    if _py_r1000_result_24[0] == _py_r1000_BREAK:
                        break
                    if _py_r1000_result_24[0] == _py_r1000_CONTINUE:
                        continue
                return (_py_r1000_NONE, None)

            def _py_r1000_else_26():
                nonlocal away_variants, candidate, home_variants, stats_away_variants, stats_home_variants

                def _py_r1000_loop_27():
                    nonlocal away_variants, home_variants
                    home_variants = candidate_variants(candidate, home_name_columns)
                    away_variants = candidate_variants(candidate, away_name_columns)
                    if stats_home_variants & home_variants and stats_away_variants & away_variants:
                        match_records.append({'candidate': candidate, 'orientations': {'home_away'}})
                    return (_py_r1000_NONE, None)
                stats_home_variants = side_variants(game['home'])
                stats_away_variants = side_variants(game['away'])
                for candidate in candidates:
                    _py_r1000_result_28 = _py_r1000_loop_27()
                    if _py_r1000_result_28[0] == _py_r1000_RETURN:
                        return _py_r1000_result_28
                    if _py_r1000_result_28[0] == _py_r1000_BREAK:
                        break
                    if _py_r1000_result_28[0] == _py_r1000_CONTINUE:
                        continue
                return (_py_r1000_NONE, None)

            def _py_r1000_if_30():
                nonlocal chosen, method
                chosen = match_records[0]['candidate']
                if game['match_mode'] == 'unordered_teams':
                    method = 'date_unordered_team_pair_canonical_home_away'
                else:
                    method = 'date_home_away_team'
                return (_py_r1000_NONE, None)

            def _py_r1000_else_32():
                nonlocal away_pts, away_score, candidate, chosen, home_pts, home_score, method, orientation_match, record, score_matches, team_a_pts, team_b_pts

                def _py_r1000_if_33():
                    nonlocal away_pts, away_score, candidate, chosen, home_pts, home_score, method, orientation_match, record, score_matches, team_a_pts, team_b_pts

                    def _py_r1000_loop_34():
                        nonlocal away_pts, away_score, candidate, home_pts, home_score, orientation_match, team_a_pts, team_b_pts

                        def _py_r1000_if_35():
                            nonlocal orientation_match, team_a_pts, team_b_pts
                            team_a_pts = game['team_a'].get('pts')
                            team_b_pts = game['team_b'].get('pts')
                            if team_a_pts is None or team_b_pts is None:
                                return (_py_r1000_CONTINUE, None)
                            orientation_match = False
                            if 'a_home' in record['orientations'] and home_score == team_a_pts and (away_score == team_b_pts):
                                orientation_match = True
                            if 'a_away' in record['orientations'] and home_score == team_b_pts and (away_score == team_a_pts):
                                orientation_match = True
                            if orientation_match:
                                score_matches.append(record)
                            return (_py_r1000_NONE, None)

                        def _py_r1000_else_37():
                            nonlocal away_pts, home_pts
                            home_pts = game['home'].get('pts')
                            away_pts = game['away'].get('pts')
                            if home_pts is not None and away_pts is not None and (home_score == home_pts) and (away_score == away_pts):
                                score_matches.append(record)
                            return (_py_r1000_NONE, None)
                        candidate = record['candidate']
                        try:
                            home_score = float(candidate.get('home_score'))
                            away_score = float(candidate.get('away_score'))
                        except (TypeError, ValueError):
                            return (_py_r1000_CONTINUE, None)
                        if game['match_mode'] == 'unordered_teams':
                            _py_r1000_result_36 = _py_r1000_if_35()
                            if _py_r1000_result_36[0] != _py_r1000_NONE:
                                return _py_r1000_result_36
                        else:
                            _py_r1000_result_38 = _py_r1000_else_37()
                            if _py_r1000_result_38[0] != _py_r1000_NONE:
                                return _py_r1000_result_38
                        return (_py_r1000_NONE, None)

                    def _py_r1000_if_40():
                        nonlocal chosen, method
                        chosen = score_matches[0]['candidate']
                        if game['match_mode'] == 'unordered_teams':
                            method = 'date_unordered_team_pair_score_canonical_home_away'
                        else:
                            method = 'date_home_away_team_score'
                        return (_py_r1000_NONE, None)
                    score_matches = []
                    for record in match_records:
                        _py_r1000_result_39 = _py_r1000_loop_34()
                        if _py_r1000_result_39[0] == _py_r1000_RETURN:
                            return _py_r1000_result_39
                        if _py_r1000_result_39[0] == _py_r1000_BREAK:
                            break
                        if _py_r1000_result_39[0] == _py_r1000_CONTINUE:
                            continue
                    if len(score_matches) == 1:
                        _py_r1000_result_41 = _py_r1000_if_40()
                        if _py_r1000_result_41[0] != _py_r1000_NONE:
                            return _py_r1000_result_41
                    return (_py_r1000_NONE, None)
                if len(match_records) > 1:
                    _py_r1000_result_42 = _py_r1000_if_33()
                    if _py_r1000_result_42[0] != _py_r1000_NONE:
                        return _py_r1000_result_42
                return (_py_r1000_NONE, None)

            def _py_r1000_if_44():
                if len(match_records) > 1:
                    ambiguous.append(native_game_id)
                else:
                    unmatched.append(native_game_id)
                return (_py_r1000_CONTINUE, None)
            game = stats_games[native_game_id]
            candidates = by_date.get(game['game_date_key'], [])
            match_records = []
            if game['match_mode'] == 'unordered_teams':
                _py_r1000_result_25 = _py_r1000_if_22()
                if _py_r1000_result_25[0] != _py_r1000_NONE:
                    return _py_r1000_result_25
            else:
                _py_r1000_result_29 = _py_r1000_else_26()
                if _py_r1000_result_29[0] != _py_r1000_NONE:
                    return _py_r1000_result_29
            chosen = None
            method = ''
            if len(match_records) == 1:
                _py_r1000_result_31 = _py_r1000_if_30()
                if _py_r1000_result_31[0] != _py_r1000_NONE:
                    return _py_r1000_result_31
            else:
                _py_r1000_result_43 = _py_r1000_else_32()
                if _py_r1000_result_43[0] != _py_r1000_NONE:
                    return _py_r1000_result_43
            if chosen is None:
                _py_r1000_result_45 = _py_r1000_if_44()
                if _py_r1000_result_45[0] != _py_r1000_NONE:
                    return _py_r1000_result_45
            espn_game_id = clean(chosen.get('game_id'))
            if not espn_game_id:
                unmatched.append(native_game_id)
                return (_py_r1000_CONTINUE, None)
            mappings.append({'nba_game_id': native_game_id, 'espn_game_id': espn_game_id, 'match_method': method})
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_47():
            nonlocal bad_date_ids, columns, conflicting_duplicate_ids, exact_duplicate_rows, game_identity, game_row_required, has_game_rows, has_team_rows, incomplete_ids, maybe_float, pl_module, side_identity, side_payload, side_team_identity, side_variants, stats_games, stats_schedule, stats_source, team_row_required, unordered_team_row_games

            def _py_r1000_if_48():
                _py_r1000_result_6 = _py_r1000_if_1()
                if _py_r1000_result_6[0] != _py_r1000_NONE:
                    return _py_r1000_result_6
                return (_py_r1000_NONE, None)

            def _py_r1000_else_50():
                _py_r1000_result_14 = _py_r1000_else_7()
                if _py_r1000_result_14[0] != _py_r1000_NONE:
                    return _py_r1000_result_14
                return (_py_r1000_NONE, None)
            pl_module = pl()
            stats_schedule, stats_source = load_nba_stats_schedule(internal_season, sdv_season)
            columns = set(stats_schedule.columns)
            team_row_required = {'game_id', 'game_date', 'team_name', 'team_abbreviation'}
            game_row_required = {'game_id', 'game_date', 'home_team_name', 'home_team_abbreviation', 'away_team_name', 'away_team_abbreviation'}
            has_team_rows = team_row_required <= columns
            has_game_rows = game_row_required <= columns
            if not (has_team_rows or has_game_rows):
                raise RuntimeError(f'NBA Stats schedule has unsupported schema; columns={stats_schedule.columns}')
            stats_games = {}
            bad_date_ids = set()
            conflicting_duplicate_ids = set()
            incomplete_ids = set()
            exact_duplicate_rows = 0
            unordered_team_row_games = 0

            def maybe_float(value: Any) -> float | None:
                if value is None:
                    return None
                text = clean(value)
                if not text:
                    return None
                try:
                    return float(text)
                except (TypeError, ValueError):
                    return None

            def side_payload(*, team_name: Any, team_abbreviation: Any, pts: Any) -> dict[str, Any]:
                return {'team_name': clean(team_name), 'team_abbreviation': clean(team_abbreviation), 'pts': maybe_float(pts)}

            def side_identity(side_data: dict[str, Any] | None) -> tuple[Any, ...] | None:
                if side_data is None:
                    return None
                return (clean(side_data.get('team_name')).lower(), clean(side_data.get('team_abbreviation')).lower(), side_data.get('pts'))

            def side_team_identity(side_data: dict[str, Any] | None) -> tuple[str, str] | None:
                if side_data is None:
                    return None
                return (clean(side_data.get('team_name')).lower(), clean(side_data.get('team_abbreviation')).lower())

            def side_variants(side_data: dict[str, Any] | None) -> set[str]:
                if side_data is None:
                    return set()
                return team_variants(side_data.get('team_name')) | team_variants(side_data.get('team_abbreviation'))

            def game_identity(game_data: dict[str, Any]) -> tuple[Any, ...]:
                return (clean(game_data.get('game_date_key')), clean(game_data.get('match_mode')), side_identity(game_data.get('home')), side_identity(game_data.get('away')), side_identity(game_data.get('team_a')), side_identity(game_data.get('team_b')))
            if has_game_rows:
                _py_r1000_result_49 = _py_r1000_if_48()
                if _py_r1000_result_49[0] != _py_r1000_NONE:
                    return _py_r1000_result_49
            else:
                _py_r1000_result_51 = _py_r1000_else_50()
                if _py_r1000_result_51[0] != _py_r1000_NONE:
                    return _py_r1000_result_51
            if bad_date_ids:
                raise RuntimeError(f'NBA Stats schedule has unparseable game_date values for game_ids={sorted(bad_date_ids)[:20]}')
            if conflicting_duplicate_ids:
                raise RuntimeError(f'NBA Stats schedule has conflicting duplicate team rows for game_ids={sorted(conflicting_duplicate_ids)[:20]}')
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_52():
            nonlocal games, games_file, home_name_columns, missing_games, required_games, root
            if incomplete_ids:
                raise RuntimeError(f'NBA Stats schedule cannot identify exactly two unique teams for game_ids={sorted(incomplete_ids)[:20]}')
            if not stats_games:
                raise RuntimeError(f'NBA Stats schedule produced zero usable games for SDV season={sdv_season}')
            log(f'NBA STATS SCHEDULE SCHEMA | internal={internal_season} sdv={sdv_season} schema={schema_used} rows={stats_schedule.height} games={len(stats_games)} unordered_team_row_games={unordered_team_row_games} exact_duplicate_rows_ignored={exact_duplicate_rows}')
            root = storage_root(cfg)
            games_file = table_path(root, 'nba', internal_season, 'games')
            if not games_file.exists():
                raise RuntimeError(f'NBA canonical games file missing: {games_file}')
            games = pl_module.read_parquet(games_file)
            required_games = {'game_id', 'game_date'}
            missing_games = sorted(required_games - set(games.columns))
            if missing_games:
                raise RuntimeError(f'NBA games.parquet missing columns={missing_games}')
            home_name_columns = [column for column in ('home_display_name', 'home_name', 'home_short_display_name', 'home_location', 'home_abbreviation') if column in games.columns]
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_53():
            nonlocal away_name_columns, canonical_columns
            away_name_columns = [column for column in ('away_display_name', 'away_name', 'away_short_display_name', 'away_location', 'away_abbreviation') if column in games.columns]
            if not home_name_columns or not away_name_columns:
                raise RuntimeError('NBA games.parquet does not expose usable home/away team-name columns')
            canonical_columns = ['game_id', 'game_date', *home_name_columns, *away_name_columns]
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_54():
            nonlocal by_date, canonical_bad_dates, score_column

            def _py_r1000_loop_55():
                _py_r1000_result_16 = _py_r1000_loop_15()
                if _py_r1000_result_16[0] == _py_r1000_RETURN:
                    return _py_r1000_result_16
                if _py_r1000_result_16[0] == _py_r1000_BREAK:
                    return (_py_r1000_BREAK, None)
                if _py_r1000_result_16[0] == _py_r1000_CONTINUE:
                    return (_py_r1000_CONTINUE, None)
                return (_py_r1000_NONE, None)
            for score_column in ('home_score', 'away_score'):
                _py_r1000_result_56 = _py_r1000_loop_55()
                if _py_r1000_result_56[0] == _py_r1000_RETURN:
                    return _py_r1000_result_56
                if _py_r1000_result_56[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_56[0] == _py_r1000_CONTINUE:
                    continue
            by_date = {}
            canonical_bad_dates = []
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_57():
            nonlocal ambiguous, candidate_variants, mappings, row, unmatched, unordered_orientations

            def _py_r1000_loop_58():
                _py_r1000_result_20 = _py_r1000_loop_17()
                if _py_r1000_result_20[0] == _py_r1000_RETURN:
                    return _py_r1000_result_20
                if _py_r1000_result_20[0] == _py_r1000_BREAK:
                    return (_py_r1000_BREAK, None)
                if _py_r1000_result_20[0] == _py_r1000_CONTINUE:
                    return (_py_r1000_CONTINUE, None)
                return (_py_r1000_NONE, None)
            for row in games.select(canonical_columns).iter_rows(named=True):
                _py_r1000_result_59 = _py_r1000_loop_58()
                if _py_r1000_result_59[0] == _py_r1000_RETURN:
                    return _py_r1000_result_59
                if _py_r1000_result_59[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_59[0] == _py_r1000_CONTINUE:
                    continue
            if canonical_bad_dates:
                raise RuntimeError(f'NBA games.parquet has unparseable game_date values for game_ids={sorted(canonical_bad_dates)[:20]}')

            def candidate_variants(candidate_row: dict[str, Any], name_columns: list[str]) -> set[str]:
                variants: set[str] = set()
                for column in name_columns:
                    variants.update(team_variants(candidate_row.get(column)))
                return variants

            def unordered_orientations(team_a: dict[str, Any], team_b: dict[str, Any], candidate_row: dict[str, Any]) -> set[str]:
                team_a_variants = side_variants(team_a)
                team_b_variants = side_variants(team_b)
                candidate_home_variants = candidate_variants(candidate_row, home_name_columns)
                candidate_away_variants = candidate_variants(candidate_row, away_name_columns)
                matched_orientations: set[str] = set()
                if team_a_variants & candidate_home_variants and team_b_variants & candidate_away_variants:
                    matched_orientations.add('a_home')
                if team_a_variants & candidate_away_variants and team_b_variants & candidate_home_variants:
                    matched_orientations.add('a_away')
                return matched_orientations
            mappings = []
            unmatched = []
            ambiguous = []
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_60():
            nonlocal duplicate_espn, duplicate_native, native_game_id, xwalk

            def _py_r1000_loop_61():
                _py_r1000_result_46 = _py_r1000_loop_21()
                if _py_r1000_result_46[0] == _py_r1000_RETURN:
                    return _py_r1000_result_46
                if _py_r1000_result_46[0] == _py_r1000_BREAK:
                    return (_py_r1000_BREAK, None)
                if _py_r1000_result_46[0] == _py_r1000_CONTINUE:
                    return (_py_r1000_CONTINUE, None)
                return (_py_r1000_NONE, None)
            for native_game_id in sorted(stats_games):
                _py_r1000_result_62 = _py_r1000_loop_61()
                if _py_r1000_result_62[0] == _py_r1000_RETURN:
                    return _py_r1000_result_62
                if _py_r1000_result_62[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_62[0] == _py_r1000_CONTINUE:
                    continue
            if not mappings:
                raise RuntimeError(f'NBA historical Stats->ESPN crosswalk produced zero mapped games for internal={internal_season} sdv={sdv_season}')
            xwalk = pl_module.DataFrame(mappings)
            duplicate_native = xwalk.group_by('nba_game_id').agg(pl_module.col('espn_game_id').n_unique().alias('n')).filter(pl_module.col('n') > 1)
            duplicate_espn = xwalk.group_by('espn_game_id').agg(pl_module.col('nba_game_id').n_unique().alias('n')).filter(pl_module.col('n') > 1)
            if duplicate_native.height or duplicate_espn.height:
                raise RuntimeError('NBA historical Stats->ESPN crosswalk is not one-to-one')
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_63():
            nonlocal game_id
            for game_id in sorted(unmatched):
                log(f'NBA GAME UNMAPPED | nba_game_id={game_id} reason=no_unique_date_unordered_team_match')
            for game_id in sorted(ambiguous):
                log(f'NBA GAME AMBIGUOUS | nba_game_id={game_id} reason=multiple_date_unordered_team_matches')
            log(f'NBA LEGACY GAME CROSSWALK | internal={internal_season} sdv={sdv_season} stats_games={len(stats_games)} unordered_team_row_games={unordered_team_row_games} mapped={xwalk.height} unmatched={len(unmatched)} ambiguous={len(ambiguous)} source={stats_source}')
            return (_py_r1000_RETURN, (xwalk.select('nba_game_id', 'espn_game_id'), f'{stats_source} -> deterministic game_date/unordered-team-pair match with canonical ESPN home-away orientation', 'nba_game_id'))
        for _py_r1000_block_64 in (_py_r1000_chunk_47, _py_r1000_chunk_52, _py_r1000_chunk_53, _py_r1000_chunk_54, _py_r1000_chunk_57, _py_r1000_chunk_60, _py_r1000_chunk_63):
            _py_r1000_result_65 = _py_r1000_block_64()
            if _py_r1000_result_65[0] != _py_r1000_NONE:
                return _py_r1000_result_65
        return (_py_r1000_NONE, None)
    _py_r1000_outcome = _py_r1000_impl()
    if _py_r1000_outcome[0] == _py_r1000_RETURN:
        return _py_r1000_outcome[1]


def load_wnba_stats_schedule(
    sdv_season: int,
):
    pl_module = pl()

    module_name = (
        "sportsdataverse.wnba"
    )

    loader_name = (
        "load_wnba_stats_schedules"
    )

    module = importlib.import_module(
        module_name
    )

    loader = optional_callable_attr(
        module,
        loader_name,
    )

    loader_error = None

    if loader is not None:
        try:
            frame = loader(
                seasons=[
                    int(
                        sdv_season
                    )
                ],
                return_as_pandas=False,
            )

            df = normalize_frame_columns(
                frame
            )

            if df.height:
                return (
                    df,
                    (
                        f"{module_name}."
                        f"{loader_name}"
                    ),
                )

        except Exception as exc:
            loader_error = exc

            log(
                "WNBA STATS SCHEDULE LOADER FAILED | "
                f"season={sdv_season} "
                f"error={exc} "
                "action=release_fallback"
            )

    url = (
        WNBA_STATS_SCHEDULE_FALLBACK_URL.format(
            season=int(
                sdv_season
            )
        )
    )

    response = http_get(url)

    df = normalize_frame_columns(
        pl_module.read_parquet(
            io.BytesIO(
                response.content
            )
        )
    )

    if df.height == 0:
        detail = (
            f"; loader_error={loader_error}"
            if loader_error
            else ""
        )

        raise RuntimeError(
            "WNBA Stats schedule "
            "returned zero rows for "
            f"season={sdv_season}"
            f"{detail}"
        )

    return (
        df,
        (
            f"{module_name}."
            f"{loader_name} "
            "[release fallback]"
        ),
    )


def build_wnba_legacy_schedule_crosswalk(cfg: dict[str, Any], internal_season: int, sdv_season: int):
    _py_r1000_NONE = 0
    _py_r1000_RETURN = 1
    _py_r1000_BREAK = 2
    _py_r1000_CONTINUE = 3

    def _py_r1000_impl():
        nonlocal cfg, internal_season, sdv_season
        ambiguous: object
        away_name_columns: object
        away_score: object
        away_variants: object
        by_date: object
        candidate: object
        candidates: object
        chosen: object
        column: object
        date_key: object
        duplicate_espn: object
        duplicate_native: object
        expressions: object
        game_id: object
        game_rows: object
        games: object
        games_file: object
        grouped: object
        home_name_columns: object
        home_score: object
        home_variants: object
        ids: object
        incomplete: object
        mappings: object
        method: object
        missing: object
        missing_games: object
        name_matches: object
        native_game_id: object
        pl_module: object
        required: object
        required_games: object
        root: object
        row: object
        score_matches: object
        stats_away_variants: object
        stats_home_variants: object
        stats_schedule: object
        stats_source: object
        unclassified: object
        unmatched: object
        xwalk: object

        def _py_r1000_loop_1():
            nonlocal date_key
            date_key = normalize_date_key(row.get('game_date'))
            if not date_key:
                return (_py_r1000_CONTINUE, None)
            by_date.setdefault(date_key, []).append(row)
            return (_py_r1000_NONE, None)

        def _py_r1000_loop_3():
            nonlocal away_score, away_variants, candidate, candidates, chosen, column, date_key, home_score, home_variants, method, name_matches, native_game_id, score_matches, stats_away_variants, stats_home_variants

            def _py_r1000_loop_4():
                nonlocal away_variants, column, home_variants
                home_variants = set()
                for column in home_name_columns:
                    home_variants.update(team_variants(candidate.get(column)))
                away_variants = set()
                for column in away_name_columns:
                    away_variants.update(team_variants(candidate.get(column)))
                if stats_home_variants & home_variants and stats_away_variants & away_variants:
                    name_matches.append(candidate)
                return (_py_r1000_NONE, None)

            def _py_r1000_else_6():
                nonlocal candidates
                if len(name_matches) > 1:
                    candidates = name_matches
                return (_py_r1000_NONE, None)

            def _py_r1000_if_8():
                nonlocal away_score, candidate, chosen, home_score, method, score_matches

                def _py_r1000_loop_9():
                    nonlocal away_score, home_score
                    try:
                        home_score = float(clean(candidate.get('home_score')))
                        away_score = float(clean(candidate.get('away_score')))
                    except (TypeError, ValueError):
                        return (_py_r1000_CONTINUE, None)
                    if home_score == float(row['home_pts']) and away_score == float(row['away_pts']):
                        score_matches.append(candidate)
                    return (_py_r1000_NONE, None)

                def _py_r1000_else_11():
                    if len(score_matches) > 1:
                        ambiguous.append(native_game_id)
                        return (_py_r1000_CONTINUE, None)
                    return (_py_r1000_NONE, None)
                score_matches = []
                for candidate in candidates:
                    _py_r1000_result_10 = _py_r1000_loop_9()
                    if _py_r1000_result_10[0] == _py_r1000_RETURN:
                        return _py_r1000_result_10
                    if _py_r1000_result_10[0] == _py_r1000_BREAK:
                        break
                    if _py_r1000_result_10[0] == _py_r1000_CONTINUE:
                        continue
                if len(score_matches) == 1:
                    chosen = score_matches[0]
                    method = 'date_home_away_score'
                else:
                    _py_r1000_result_12 = _py_r1000_else_11()
                    if _py_r1000_result_12[0] != _py_r1000_NONE:
                        return _py_r1000_result_12
                return (_py_r1000_NONE, None)

            def _py_r1000_if_14():
                if len(name_matches) > 1:
                    ambiguous.append(native_game_id)
                else:
                    unmatched.append(native_game_id)
                return (_py_r1000_CONTINUE, None)
            native_game_id = clean(row.get('game_id'))
            date_key = clean(row.get('game_date_key'))
            candidates = by_date.get(date_key, [])
            stats_home_variants = set()
            stats_home_variants.update(team_variants(row.get('home_team_name')))
            stats_home_variants.update(team_variants(row.get('home_team_abbreviation')))
            stats_away_variants = set()
            stats_away_variants.update(team_variants(row.get('away_team_name')))
            stats_away_variants.update(team_variants(row.get('away_team_abbreviation')))
            name_matches = []
            for candidate in candidates:
                _py_r1000_result_5 = _py_r1000_loop_4()
                if _py_r1000_result_5[0] == _py_r1000_RETURN:
                    return _py_r1000_result_5
                if _py_r1000_result_5[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_5[0] == _py_r1000_CONTINUE:
                    continue
            chosen = None
            method = ''
            if len(name_matches) == 1:
                chosen = name_matches[0]
                method = 'date_home_away_team'
            else:
                _py_r1000_result_7 = _py_r1000_else_6()
                if _py_r1000_result_7[0] != _py_r1000_NONE:
                    return _py_r1000_result_7
            if chosen is None and row.get('home_pts') is not None and (row.get('away_pts') is not None):
                _py_r1000_result_13 = _py_r1000_if_8()
                if _py_r1000_result_13[0] != _py_r1000_NONE:
                    return _py_r1000_result_13
            if chosen is None:
                _py_r1000_result_15 = _py_r1000_if_14()
                if _py_r1000_result_15[0] != _py_r1000_NONE:
                    return _py_r1000_result_15
            mappings.append({'wnba_game_id': native_game_id, 'espn_game_id': clean(chosen.get('game_id')), 'match_method': method})
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_17():
            nonlocal expressions, games, games_file, grouped, ids, incomplete, missing, missing_games, pl_module, required, required_games, root, stats_schedule, stats_source, unclassified
            pl_module = pl()
            stats_schedule, stats_source = load_wnba_stats_schedule(sdv_season)
            required = {'game_id', 'game_date', 'team_name', 'team_abbreviation', 'matchup'}
            missing = sorted(required - set(stats_schedule.columns))
            if missing:
                raise RuntimeError(f'WNBA Stats schedule missing columns={missing}; columns={stats_schedule.columns}')
            expressions = [pl_module.col('game_id').cast(pl_module.Utf8, strict=False).str.strip_chars().alias('game_id'), pl_module.col('game_date').cast(pl_module.Utf8, strict=False).str.slice(0, 10).alias('game_date_key'), pl_module.col('team_name').cast(pl_module.Utf8, strict=False).alias('team_name'), pl_module.col('team_abbreviation').cast(pl_module.Utf8, strict=False).alias('team_abbreviation'), pl_module.col('matchup').cast(pl_module.Utf8, strict=False).alias('matchup')]
            if 'pts' in stats_schedule.columns:
                expressions.append(pl_module.col('pts').cast(pl_module.Float64, strict=False).alias('pts'))
            else:
                expressions.append(pl_module.lit(None, dtype=pl_module.Float64).alias('pts'))
            stats_schedule = stats_schedule.select(expressions).filter(pl_module.col('game_id').is_not_null() & (pl_module.col('game_id') != '')).with_columns(pl_module.when(pl_module.col('matchup').str.contains('@', literal=True)).then(pl_module.lit('away')).when(pl_module.col('matchup').str.to_lowercase().str.contains('vs', literal=True)).then(pl_module.lit('home')).otherwise(pl_module.lit(None, dtype=pl_module.Utf8)).alias('home_away'))
            unclassified = stats_schedule.filter(pl_module.col('home_away').is_null()).select('game_id').unique()
            if unclassified.height:
                ids = unclassified.get_column('game_id').to_list()[:20]
                raise RuntimeError(f'WNBA Stats schedule has unrecognized matchup format for game_ids={ids}')
            grouped = stats_schedule.group_by('game_id', maintain_order=False).agg(pl_module.col('game_date_key').drop_nulls().first().alias('game_date_key'), pl_module.col('team_name').filter(pl_module.col('home_away') == 'home').first().alias('home_team_name'), pl_module.col('team_abbreviation').filter(pl_module.col('home_away') == 'home').first().alias('home_team_abbreviation'), pl_module.col('pts').filter(pl_module.col('home_away') == 'home').first().alias('home_pts'), pl_module.col('team_name').filter(pl_module.col('home_away') == 'away').first().alias('away_team_name'), pl_module.col('team_abbreviation').filter(pl_module.col('home_away') == 'away').first().alias('away_team_abbreviation'), pl_module.col('pts').filter(pl_module.col('home_away') == 'away').first().alias('away_pts'))
            incomplete = grouped.filter(pl_module.col('game_date_key').is_null() | pl_module.col('home_team_name').is_null() | pl_module.col('away_team_name').is_null())
            if incomplete.height:
                ids = incomplete.get_column('game_id').to_list()[:20]
                raise RuntimeError(f'WNBA Stats schedule cannot identify both teams for game_ids={ids}')
            root = storage_root(cfg)
            games_file = table_path(root, 'wnba', internal_season, 'games')
            if not games_file.exists():
                raise RuntimeError(f'WNBA canonical games file missing: {games_file}')
            games = pl_module.read_parquet(games_file)
            required_games = {'game_id', 'game_date'}
            missing_games = sorted(required_games - set(games.columns))
            if missing_games:
                raise RuntimeError(f'WNBA games.parquet missing columns={missing_games}')
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_18():
            nonlocal away_name_columns, home_name_columns
            home_name_columns = [column for column in ('home_display_name', 'home_name', 'home_short_display_name', 'home_location', 'home_abbreviation') if column in games.columns]
            away_name_columns = [column for column in ('away_display_name', 'away_name', 'away_short_display_name', 'away_location', 'away_abbreviation') if column in games.columns]
            if not home_name_columns or not away_name_columns:
                raise RuntimeError('WNBA games.parquet does not expose usable home/away team-name columns')
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_19():
            nonlocal ambiguous, by_date, game_rows, mappings, row, unmatched

            def _py_r1000_loop_20():
                _py_r1000_result_2 = _py_r1000_loop_1()
                if _py_r1000_result_2[0] == _py_r1000_RETURN:
                    return _py_r1000_result_2
                if _py_r1000_result_2[0] == _py_r1000_BREAK:
                    return (_py_r1000_BREAK, None)
                if _py_r1000_result_2[0] == _py_r1000_CONTINUE:
                    return (_py_r1000_CONTINUE, None)
                return (_py_r1000_NONE, None)
            game_rows = games.select(['game_id', 'game_date', *home_name_columns, *away_name_columns, *[column for column in ('home_score', 'away_score') if column in games.columns]]).iter_rows(named=True)
            by_date = {}
            for row in game_rows:
                _py_r1000_result_21 = _py_r1000_loop_20()
                if _py_r1000_result_21[0] == _py_r1000_RETURN:
                    return _py_r1000_result_21
                if _py_r1000_result_21[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_21[0] == _py_r1000_CONTINUE:
                    continue
            mappings = []
            unmatched = []
            ambiguous = []
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_22():
            nonlocal duplicate_espn, duplicate_native, row, xwalk

            def _py_r1000_loop_23():
                _py_r1000_result_16 = _py_r1000_loop_3()
                if _py_r1000_result_16[0] == _py_r1000_RETURN:
                    return _py_r1000_result_16
                if _py_r1000_result_16[0] == _py_r1000_BREAK:
                    return (_py_r1000_BREAK, None)
                if _py_r1000_result_16[0] == _py_r1000_CONTINUE:
                    return (_py_r1000_CONTINUE, None)
                return (_py_r1000_NONE, None)
            for row in grouped.iter_rows(named=True):
                _py_r1000_result_24 = _py_r1000_loop_23()
                if _py_r1000_result_24[0] == _py_r1000_RETURN:
                    return _py_r1000_result_24
                if _py_r1000_result_24[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_24[0] == _py_r1000_CONTINUE:
                    continue
            if not mappings:
                raise RuntimeError('WNBA Stats->ESPN crosswalk produced zero mapped games')
            xwalk = pl_module.DataFrame(mappings)
            duplicate_native = xwalk.group_by('wnba_game_id').agg(pl_module.col('espn_game_id').n_unique().alias('n')).filter(pl_module.col('n') > 1)
            duplicate_espn = xwalk.group_by('espn_game_id').agg(pl_module.col('wnba_game_id').n_unique().alias('n')).filter(pl_module.col('n') > 1)
            if duplicate_native.height or duplicate_espn.height:
                raise RuntimeError('WNBA Stats->ESPN crosswalk is not one-to-one')
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_25():
            nonlocal game_id
            for game_id in sorted(unmatched):
                log(f'WNBA GAME UNMAPPED | wnba_game_id={game_id} reason=no_unique_date_team_or_score_match')
            for game_id in sorted(ambiguous):
                log(f'WNBA GAME AMBIGUOUS | wnba_game_id={game_id} reason=multiple_canonical_matches')
            log(f'WNBA LEGACY GAME CROSSWALK | internal={internal_season} sdv={sdv_season} stats_games={grouped.height} mapped={xwalk.height} unmatched={len(unmatched)} ambiguous={len(ambiguous)} source={stats_source}')
            return (_py_r1000_RETURN, (xwalk.select('wnba_game_id', 'espn_game_id'), f'{stats_source} -> deterministic game_date/home-away team/score match', 'wnba_game_id'))
        for _py_r1000_block_26 in (_py_r1000_chunk_17, _py_r1000_chunk_18, _py_r1000_chunk_19, _py_r1000_chunk_22, _py_r1000_chunk_25):
            _py_r1000_result_27 = _py_r1000_block_26()
            if _py_r1000_result_27[0] != _py_r1000_NONE:
                return _py_r1000_result_27
        return (_py_r1000_NONE, None)
    _py_r1000_outcome = _py_r1000_impl()
    if _py_r1000_outcome[0] == _py_r1000_RETURN:
        return _py_r1000_outcome[1]


def load_pro_schedule_crosswalk(
    cfg: dict[str, Any],
    league: str,
    internal_season: int,
    sdv_season: int,
):
    if league not in PRO_SCHEDULE_CROSSWALKS:
        raise ValueError(
            "No pro schedule crosswalk "
            f"configured for league={league}"
        )

    spec = (
        PRO_SCHEDULE_CROSSWALKS[
            league
        ]
    )

    minimum_season = int(
        spec[
            "minimum_season"
        ]
    )

    if (
        int(
            sdv_season
        )
        < minimum_season
    ):
        if league == "nba":
            return (
                build_nba_legacy_schedule_crosswalk(
                    cfg,
                    internal_season,
                    sdv_season,
                )
            )

        if league == "wnba":
            return (
                build_wnba_legacy_schedule_crosswalk(
                    cfg,
                    internal_season,
                    sdv_season,
                )
            )

        raise RuntimeError(
            "No historical schedule crosswalk strategy "
            f"configured for league={league} "
            f"sdv_season={sdv_season}"
        )

    pl_module = pl()

    module_name = clean(
        spec[
            "module"
        ]
    )

    loader_name = clean(
        spec[
            "loader"
        ]
    )

    native_game_id = clean(
        spec[
            "native_game_id"
        ]
    )

    module = importlib.import_module(
        module_name
    )

    loader = optional_callable_attr(
        module,
        loader_name,
    )

    loader_error = None

    if loader is not None:
        try:
            frame = loader(
                seasons=[
                    int(
                        sdv_season
                    )
                ],
                return_as_pandas=False,
            )

            xwalk = normalize_frame_columns(
                frame
            )

            if xwalk.height:
                source = (
                    f"{module_name}."
                    f"{loader_name}"
                )

                return (
                    prepare_pro_crosswalk(
                        xwalk,
                        native_game_id,
                        league,
                    ),
                    source,
                    native_game_id,
                )

        except Exception as exc:
            loader_error = exc

            log(
                "PRO SCHEDULE CROSSWALK LOADER FAILED | "
                f"league={league} "
                f"sdv={sdv_season} "
                f"error={exc} "
                "action=release_fallback"
            )

    url = clean(
        spec[
            "fallback_url"
        ]
    ).format(
        season=int(
            sdv_season
        )
    )

    try:
        response = http_get(
            url
        )

    except Exception as exc:
        raise RuntimeError(
            f"{league} schedule crosswalk "
            "loader and release fallback "
            "both failed; "
            f"loader_error={loader_error}; "
            f"fallback_error={exc}"
        ) from exc

    xwalk = normalize_frame_columns(
        pl_module.read_parquet(
            io.BytesIO(
                response.content
            )
        )
    )

    return (
        prepare_pro_crosswalk(
            xwalk,
            native_game_id,
            league,
        ),
        (
            f"{module_name}."
            f"{loader_name} "
            "[release fallback]"
        ),
        native_game_id,
    )


def prepare_pro_crosswalk(
    xwalk,
    native_game_id: str,
    league: str,
):
    pl_module = pl()

    required = {
        native_game_id,
        "espn_game_id",
    }

    missing = sorted(
        required
        - set(
            xwalk.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{league} schedule crosswalk "
            f"missing columns={missing}; "
            f"columns={xwalk.columns}"
        )

    xwalk = (
        xwalk
        .select(
            pl_module.col(
                native_game_id
            )
            .cast(
                pl_module.Utf8,
                strict=False,
            )
            .str.strip_chars()
            .alias(
                native_game_id
            ),
            pl_module.col(
                "espn_game_id"
            )
            .cast(
                pl_module.Utf8,
                strict=False,
            )
            .str.strip_chars()
            .alias(
                "espn_game_id"
            ),
        )
        .filter(
            pl_module.col(
                native_game_id
            ).is_not_null()
            & (
                pl_module.col(
                    native_game_id
                )
                != ""
            )
            & pl_module.col(
                "espn_game_id"
            ).is_not_null()
            & (
                pl_module.col(
                    "espn_game_id"
                )
                != ""
            )
        )
    )

    if xwalk.height == 0:
        raise RuntimeError(
            f"{league} schedule crosswalk "
            "has zero usable mapped rows"
        )

    conflicts = (
        xwalk
        .group_by(
            native_game_id
        )
        .agg(
            pl_module.col(
                "espn_game_id"
            )
            .n_unique()
            .alias(
                "n"
            )
        )
        .filter(
            pl_module.col(
                "n"
            )
            > 1
        )
    )

    if conflicts.height:
        ids = (
            conflicts
            .get_column(
                native_game_id
            )
            .to_list()[:20]
        )

        raise RuntimeError(
            f"{league} schedule crosswalk "
            "maps one Stats game id "
            "to multiple ESPN ids: "
            f"{ids}"
        )

    return (
        xwalk.unique(
            subset=[
                native_game_id
            ],
            keep="first",
        )
    )


def canonicalize_pro_game_ids(
    cfg: dict[str, Any],
    df,
    *,
    league: str,
    internal_season: int,
    sdv_season: int,
    table: str,
):
    if (
        league
        not in {
            "nba",
            "wnba",
        }
        or table
        not in {
            "possessions",
            "lineups",
        }
    ):
        return (
            df,
            {},
        )

    pl_module = pl()

    if "game_id" not in df.columns:
        raise RuntimeError(
            f"{league}.{table}: "
            "Stats release is missing "
            "game_id"
        )

    root = storage_root(cfg)

    games_file = table_path(
        root,
        league,
        internal_season,
        "games",
    )

    if not games_file.exists():
        raise RuntimeError(
            f"{league}.{table}: "
            "canonical games.parquet "
            f"missing: {games_file}"
        )

    games = pl_module.read_parquet(
        games_file
    )

    if "game_id" not in games.columns:
        raise RuntimeError(
            f"{league} games.parquet "
            "missing game_id"
        )

    canonical_ids = (
        games
        .select(
            pl_module.col(
                "game_id"
            )
            .cast(
                pl_module.Utf8,
                strict=False,
            )
            .str.strip_chars()
            .alias(
                "game_id"
            )
        )
        .filter(
            pl_module.col(
                "game_id"
            ).is_not_null()
            & (
                pl_module.col(
                    "game_id"
                )
                != ""
            )
        )
        .unique()
    )

    if canonical_ids.height == 0:
        raise RuntimeError(
            f"{league} games.parquet "
            "contains zero canonical game ids"
        )

    canonical_id_set = set(
        canonical_ids
        .get_column(
            "game_id"
        )
        .to_list()
    )

    (
        xwalk,
        xwalk_source,
        native_game_id,
    ) = load_pro_schedule_crosswalk(
        cfg,
        league,
        internal_season,
        sdv_season,
    )

    original_rows = int(
        df.height
    )

    source_ids = (
        df
        .select(
            pl_module.col(
                "game_id"
            )
            .cast(
                pl_module.Utf8,
                strict=False,
            )
            .str.strip_chars()
            .alias(
                native_game_id
            )
        )
        .filter(
            pl_module.col(
                native_game_id
            ).is_not_null()
            & (
                pl_module.col(
                    native_game_id
                )
                != ""
            )
        )
        .unique()
    )

    source_game_count = int(
        source_ids.height
    )

    if native_game_id in df.columns:
        df = df.drop(
            native_game_id
        )

    df = df.with_columns(
        pl_module.col(
            "game_id"
        )
        .cast(
            pl_module.Utf8,
            strict=False,
        )
        .str.strip_chars()
        .alias(
            native_game_id
        )
    )

    df = df.join(
        xwalk,
        on=native_game_id,
        how="left",
        maintain_order="left",
    )

    unmapped_source_ids = (
        df
        .filter(
            pl_module.col(
                "espn_game_id"
            ).is_null()
            | (
                pl_module.col(
                    "espn_game_id"
                )
                == ""
            )
        )
        .select(
            native_game_id
        )
        .drop_nulls()
        .unique()
        .get_column(
            native_game_id
        )
        .to_list()
    )

    outside_schedule_source_ids = (
        df
        .filter(
            pl_module.col(
                "espn_game_id"
            ).is_not_null()
            & (
                pl_module.col(
                    "espn_game_id"
                )
                != ""
            )
            & (
                ~pl_module.col(
                    "espn_game_id"
                )
                .is_in(
                    sorted(
                        canonical_id_set
                    )
                )
            )
        )
        .select(
            native_game_id
        )
        .drop_nulls()
        .unique()
        .get_column(
            native_game_id
        )
        .to_list()
    )

    df = (
        df
        .filter(
            pl_module.col(
                "espn_game_id"
            )
            .is_in(
                sorted(
                    canonical_id_set
                )
            )
        )
        .drop(
            "game_id"
        )
        .rename(
            {
                "espn_game_id": (
                    "game_id"
                )
            }
        )
    )

    if df.height == 0:
        raise RuntimeError(
            f"{league}.{table}: "
            "zero rows remain after "
            "Stats-to-ESPN game-id crosswalk"
        )

    mapped_game_count = int(
        df
        .select(
            "game_id"
        )
        .unique()
        .height
    )

    dropped_rows = (
        original_rows
        - int(
            df.height
        )
    )

    log(
        "PRO GAME ID CROSSWALK | "
        f"league={league} "
        f"internal={internal_season} "
        f"sdv={sdv_season} "
        f"table={table} "
        f"source_games={source_game_count} "
        f"canonical_games={len(canonical_id_set)} "
        f"kept_games={mapped_game_count} "
        f"unmapped_source_games={len(unmapped_source_ids)} "
        f"outside_schedule_games="
        f"{len(outside_schedule_source_ids)} "
        f"rows_before={original_rows} "
        f"rows_after={df.height} "
        f"dropped_rows={dropped_rows} "
        f"crosswalk={xwalk_source}"
    )

    for game_id in sorted(
        set(
            unmapped_source_ids
        )
    ):
        log(
            "PRO GAME ID UNMAPPED | "
            f"league={league} "
            f"table={table} "
            f"{native_game_id}={game_id}"
        )

    for game_id in sorted(
        set(
            outside_schedule_source_ids
        )
    ):
        log(
            "PRO GAME OUTSIDE CANONICAL SCHEDULE | "
            f"league={league} "
            f"table={table} "
            f"{native_game_id}={game_id} "
            "action=filtered"
        )

    extra = {
        "game_id_namespace": "espn",
        "source_game_id_column": native_game_id,
        "game_id_crosswalk_source": xwalk_source,
        "source_unique_games": source_game_count,
        "canonical_games": len(
            canonical_id_set
        ),
        "crosswalk_kept_games": mapped_game_count,
        "crosswalk_unmapped_source_games": len(
            unmapped_source_ids
        ),
        "crosswalk_outside_schedule_games": len(
            outside_schedule_source_ids
        ),
        "rows_before_game_id_crosswalk": original_rows,
        "rows_after_game_id_crosswalk": int(
            df.height
        ),
        "rows_filtered_by_game_id_crosswalk": dropped_rows,
    }

    return (
        df,
        extra,
    )


def call_loader(
    league: str,
    table: str,
    internal_season: int,
    sdv_season: int,
):
    spec = LOADERS[
        league
    ][
        table
    ]

    if spec is None:
        raise RuntimeError(
            f"{league}.{table} "
            "uses dedicated derived-table logic"
        )

    (
        module_name,
        function_name,
    ) = spec

    loader_season = (
        release_loader_season(
            league,
            table,
            internal_season,
            sdv_season,
        )
    )

    fallback_key = (
        league,
        table,
    )

    if (
        league == "wnba"
        and table
        in {
            "possessions",
            "lineups",
        }
        and int(
            sdv_season
        )
        < 2026
    ):
        (
            frame,
            source,
        ) = release_fallback(
            league,
            table,
            internal_season,
            sdv_season,
        )

        return (
            frame,
            source,
            loader_season,
        )

    module = importlib.import_module(
        module_name
    )

    loader = optional_callable_attr(
        module,
        function_name,
    )

    if loader is not None:
        try:
            return (
                loader(
                    seasons=[
                        loader_season
                    ],
                    return_as_pandas=False,
                ),
                (
                    f"{module_name}."
                    f"{function_name}"
                ),
                loader_season,
            )

        except Exception as exc:
            primary_error = exc

    else:
        primary_error = RuntimeError(
            "SportsDataVerse loader missing: "
            f"{module_name}.{function_name}"
        )

    if (
        fallback_key
        in LOADER_FALLBACKS
    ):
        action = "loader_fallback"

    elif (
        fallback_key
        in RELEASE_FALLBACKS
    ):
        action = "release_fallback"

    else:
        raise primary_error

    if loader is not None:
        log(
            "RELEASE LOADER FAILED | "
            f"league={league} "
            f"table={table} "
            f"loader={module_name}.{function_name} "
            f"loader_season={loader_season} "
            f"error={primary_error} "
            f"action={action}"
        )

    if (
        fallback_key
        in LOADER_FALLBACKS
    ):
        (
            fallback_module_name,
            fallback_function_name,
            fallback_season_mode,
        ) = LOADER_FALLBACKS[
            fallback_key
        ]

        fallback_season = (
            loader_fallback_season(
                fallback_season_mode,
                internal_season,
                sdv_season,
            )
        )

        fallback_module = (
            importlib.import_module(
                fallback_module_name
            )
        )

        fallback_loader = optional_callable_attr(
            fallback_module,
            fallback_function_name,
        )

        if fallback_loader is None:
            fallback_error = RuntimeError(
                "SportsDataVerse fallback "
                "loader missing: "
                f"{fallback_module_name}."
                f"{fallback_function_name}"
            )

        else:
            try:
                frame = fallback_loader(
                    seasons=[
                        fallback_season
                    ],
                    return_as_pandas=False,
                )

                source = (
                    f"{fallback_module_name}."
                    f"{fallback_function_name} "
                    f"[{table} fallback]"
                )

                log(
                    "LOADER FALLBACK COMPLETE | "
                    f"league={league} "
                    f"table={table} "
                    f"fallback_loader="
                    f"{fallback_module_name}."
                    f"{fallback_function_name} "
                    f"fallback_season={fallback_season}"
                )

                return (
                    frame,
                    source,
                    fallback_season,
                )

            except Exception as exc:
                fallback_error = exc

        log(
            "LOADER FALLBACK FAILED | "
            f"league={league} "
            f"table={table} "
            f"fallback_loader="
            f"{fallback_module_name}."
            f"{fallback_function_name} "
            f"fallback_season={fallback_season} "
            f"error={fallback_error}"
        )

        if (
            fallback_key
            not in RELEASE_FALLBACKS
        ):
            raise RuntimeError(
                f"{league}.{table}: "
                "primary and fallback loaders "
                "both failed; "
                f"primary_error={primary_error}; "
                f"fallback_error={fallback_error}"
            ) from fallback_error

    (
        frame,
        source,
    ) = release_fallback(
        league,
        table,
        internal_season,
        sdv_season,
    )

    return (
        frame,
        source,
        loader_season,
    )



def normalize(frame, *, table: str, league: str, internal_season: int, sdv_season: int, source: str, ingested_at_utc: str):
    _py_r1000_NONE = 0
    _py_r1000_RETURN = 1
    _py_r1000_BREAK = 2
    _py_r1000_CONTINUE = 3

    def _py_r1000_impl():
        nonlocal frame, ingested_at_utc, internal_season, league, sdv_season, source, table
        candidates: object
        canonical: object
        df: object
        front: object
        id_columns: object
        new_columns: object
        old_columns: object
        pl_module: object
        source_column: object

        def _py_r1000_loop_1():
            nonlocal df, source_column
            if canonical in df.columns:
                return (_py_r1000_CONTINUE, None)
            source_column = next((candidate for candidate in candidates if candidate in df.columns), None)
            if source_column is not None:
                df = df.with_columns(pl_module.col(source_column).alias(canonical))
            return (_py_r1000_NONE, None)

        def _py_r1000_if_3():
            nonlocal df
            df = df.with_columns([pl_module.col(column).cast(pl_module.Utf8, strict=False).alias(column) for column in id_columns])
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_5():
            nonlocal candidates, canonical, df, new_columns, old_columns, pl_module

            def _py_r1000_loop_6():
                _py_r1000_result_2 = _py_r1000_loop_1()
                if _py_r1000_result_2[0] == _py_r1000_RETURN:
                    return _py_r1000_result_2
                if _py_r1000_result_2[0] == _py_r1000_BREAK:
                    return (_py_r1000_BREAK, None)
                if _py_r1000_result_2[0] == _py_r1000_CONTINUE:
                    return (_py_r1000_CONTINUE, None)
                return (_py_r1000_NONE, None)
            pl_module = pl()
            df = to_pl(frame)
            old_columns = list(df.columns)
            new_columns = [snake(column) for column in old_columns]
            if len(new_columns) != len(set(new_columns)):
                raise ValueError(f'column collision after normalization in {table}')
            if old_columns != new_columns:
                df = df.rename(dict(zip(old_columns, new_columns)))
            for canonical, candidates in ALIASES.get(table, {}).items():
                _py_r1000_result_7 = _py_r1000_loop_6()
                if _py_r1000_result_7[0] == _py_r1000_RETURN:
                    return _py_r1000_result_7
                if _py_r1000_result_7[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_7[0] == _py_r1000_CONTINUE:
                    continue
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_8():
            nonlocal df, front, id_columns

            def _py_r1000_if_9():
                _py_r1000_result_4 = _py_r1000_if_3()
                if _py_r1000_result_4[0] != _py_r1000_NONE:
                    return _py_r1000_result_4
                return (_py_r1000_NONE, None)
            id_columns = [column for column in df.columns if column.endswith('_id') or column in {'game_id', 'play_id', 'shot_id'}]
            if id_columns:
                _py_r1000_result_10 = _py_r1000_if_9()
                if _py_r1000_result_10[0] != _py_r1000_NONE:
                    return _py_r1000_result_10
            if df.height == 0:
                raise RuntimeError(f'{league}.{table}: loader returned zero rows')
            df = df.with_columns(pl_module.lit(league.upper()).alias('league'), pl_module.lit(int(internal_season)).cast(pl_module.Int32).alias('internal_season'), pl_module.lit(int(sdv_season)).cast(pl_module.Int32).alias('sdv_season'), pl_module.lit(source).alias('source_loader'), pl_module.lit(ingested_at_utc).alias('ingested_at_utc'))
            front = ['league', 'internal_season', 'sdv_season', 'source_loader', 'ingested_at_utc']
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_11():
            return (_py_r1000_RETURN, df.select(front + [column for column in df.columns if column not in front]))
        for _py_r1000_block_12 in (_py_r1000_chunk_5, _py_r1000_chunk_8, _py_r1000_chunk_11):
            _py_r1000_result_13 = _py_r1000_block_12()
            if _py_r1000_result_13[0] != _py_r1000_NONE:
                return _py_r1000_result_13
        return (_py_r1000_NONE, None)
    _py_r1000_outcome = _py_r1000_impl()
    if _py_r1000_outcome[0] == _py_r1000_RETURN:
        return _py_r1000_outcome[1]


def write_parquet(
    df,
    path: Path,
    codec: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    if tmp.exists():
        tmp.unlink()

    df.write_parquet(
        tmp,
        compression=codec,
        statistics=True,
    )

    tmp.replace(path)


def manifest_entry(
    df,
    path: Path,
    source: str,
    status: str = "ready",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "status": status,
        "rows": int(
            df.height
        ),
        "columns": list(
            df.columns
        ),
        "source_loader": source,
        "source_function": source,
        "filename": path.name,
        "file": str(path),
    }

    if extra:
        entry.update(extra)

    return entry


def build_release_table(
    cfg: dict[str, Any],
    league: str,
    internal_season: int,
    sdv_season: int,
    table: str,
    ingested_at_utc: str,
    force: bool,
) -> dict[str, Any]:
    root = storage_root(cfg)

    output = table_path(
        root,
        league,
        internal_season,
        table,
    )

    spec = LOADERS[
        league
    ][
        table
    ]

    if spec is None:
        raise RuntimeError(
            f"{league}.{table} "
            "has no release-loader spec"
        )

    intended = (
        f"{spec[0]}."
        f"{spec[1]}"
    )

    if (
        output.exists()
        and not force
    ):
        df = pl().read_parquet(
            output
        )

        return manifest_entry(
            df,
            output,
            intended,
            "existing_not_rebuilt",
        )

    actual_loader_season = (
        release_loader_season(
            league,
            table,
            internal_season,
            sdv_season,
        )
    )

    log(
        "LOAD START | "
        f"league={league} "
        f"internal={internal_season} "
        f"sdv={sdv_season} "
        f"loader_season={actual_loader_season} "
        f"table={table} "
        f"loader={intended}"
    )

    (
        frame,
        source,
        _,
    ) = call_loader(
        league,
        table,
        internal_season,
        sdv_season,
    )

    df = normalize(
        frame,
        table=table,
        league=league,
        internal_season=internal_season,
        sdv_season=sdv_season,
        source=source,
        ingested_at_utc=ingested_at_utc,
    )

    game_id_extra: dict[str, Any] = {}

    if (
        league
        in {
            "nba",
            "wnba",
        }
        and table
        in {
            "possessions",
            "lineups",
        }
    ):
        (
            df,
            game_id_extra,
        ) = canonicalize_pro_game_ids(
            cfg,
            df,
            league=league,
            internal_season=internal_season,
            sdv_season=sdv_season,
            table=table,
        )

    if (
        table != "rosters"
        and "game_id"
        not in df.columns
    ):
        raise RuntimeError(
            f"{league}.{table}: "
            "normalized table missing game_id"
        )

    write_parquet(
        df,
        output,
        compression(cfg),
    )

    log(
        "LOAD COMPLETE | "
        f"league={league} "
        f"internal={internal_season} "
        f"table={table} "
        f"rows={df.height} "
        "status=ready "
        f"file={output}"
    )

    return manifest_entry(
        df,
        output,
        source,
        extra=(
            game_id_extra
            or None
        ),
    )


def download_parquet(
    url: str,
):
    pl_module = pl()

    temp_path: Path | None = None

    try:
        with requests.get(
            url,
            stream=True,
            timeout=300,
        ) as response:
            if response.status_code == 404:
                raise RuntimeError(
                    "SportsDataVerse source "
                    f"not published: {url}"
                )

            response.raise_for_status()

            with tempfile.NamedTemporaryFile(
                prefix="sdv_ncaa_",
                suffix=".parquet",
                delete=False,
            ) as handle:
                temp_path = Path(
                    handle.name
                )

                for chunk in response.iter_content(
                    chunk_size=(
                        1024
                        * 1024
                    )
                ):
                    if chunk:
                        handle.write(chunk)

        return pl_module.read_parquet(
            temp_path
        )

    finally:
        if (
            temp_path is not None
            and temp_path.exists()
        ):
            temp_path.unlink()


def load_ncaam_game_crosswalk(
    sdv_season: int,
):
    pl_module = pl()

    url = (
        NCAAM_GAME_XWALK_URL.format(
            season=sdv_season
        )
    )

    response = http_get(url)

    payload = response.json()

    if not isinstance(
        payload,
        list,
    ):
        raise RuntimeError(
            "Invalid NCAA ESPN game "
            f"crosswalk payload: {url}"
        )

    rows: list[
        dict[str, Any]
    ] = []

    for item in payload:
        if not isinstance(
            item,
            dict,
        ):
            continue

        contest_id = clean(
            item.get(
                "contest_id"
            )
        )

        espn_game_id = clean(
            item.get(
                "espn_game_id"
            )
        )

        if (
            not contest_id
            or not espn_game_id
        ):
            continue

        rows.append(
            {
                "ncaa_game_id": contest_id,
                "game_id": espn_game_id,
                "ncaa_espn_match_method": (
                    clean(
                        item.get(
                            "match_method"
                        )
                    )
                    or None
                ),
            }
        )

    if not rows:
        raise RuntimeError(
            "NCAA ESPN game crosswalk "
            "has zero mapped rows for "
            f"season={sdv_season}"
        )

    xwalk = (
        pl_module.DataFrame(rows)
        .unique(
            subset=[
                "ncaa_game_id"
            ],
            keep="first",
        )
    )

    duplicate_espn = (
        xwalk
        .group_by(
            "game_id"
        )
        .agg(
            pl_module.len().alias(
                "n"
            )
        )
        .filter(
            pl_module.col(
                "n"
            )
            > 1
        )
    )

    if duplicate_espn.height:
        ids = (
            duplicate_espn
            .get_column(
                "game_id"
            )
            .to_list()[:20]
        )

        raise RuntimeError(
            "NCAA crosswalk has "
            "duplicate ESPN game ids: "
            f"{ids}"
        )

    return (
        xwalk,
        url,
    )


def prepare_ncaa_pbp_for_transform(
    game_rows,
):
    pl_module = pl()

    if (
        "contest_id"
        not in game_rows.columns
    ):
        raise RuntimeError(
            "Published NCAA PBP "
            "is missing contest_id"
        )

    df = game_rows

    if "game_id" in df.columns:
        df = df.drop(
            "game_id"
        )

    return df.with_columns(
        pl_module.col(
            "contest_id"
        )
        .cast(
            pl_module.Utf8,
            strict=False,
        )
        .alias(
            "game_id"
        )
    )


def stamp_ncaam_derived_ids(
    frame,
    *,
    espn_game_id: str,
    ncaa_game_id: str,
):
    pl_module = pl()

    df = to_pl(frame)

    if df.height == 0:
        return df

    if "game_id" in df.columns:
        df = df.drop(
            "game_id"
        )

    if "ncaa_game_id" in df.columns:
        df = df.drop(
            "ncaa_game_id"
        )

    return df.with_columns(
        pl_module.lit(
            espn_game_id
        )
        .cast(
            pl_module.Utf8
        )
        .alias(
            "game_id"
        ),
        pl_module.lit(
            ncaa_game_id
        )
        .cast(
            pl_module.Utf8
        )
        .alias(
            "ncaa_game_id"
        ),
    )


def build_ncaam_derived_tables(cfg: dict[str, Any], internal_season: int, sdv_season: int, ingested_at_utc: str, force: bool) -> dict[str, dict[str, Any]]:
    _py_r1000_NONE = 0
    _py_r1000_RETURN = 1
    _py_r1000_BREAK = 2
    _py_r1000_CONTINUE = 3

    def _py_r1000_impl():
        nonlocal cfg, force, ingested_at_utc, internal_season, sdv_season
        canonical_game_count: object
        espn_game_id: object
        espn_game_ids: object
        exc: object
        function_name: object
        game_id: object
        game_identity: object
        game_partitions: object
        game_pbp: object
        games: object
        games_file: object
        lineup_coverage: object
        lineup_extra: object
        lineup_failed_games: object
        lineup_frames: object
        lineup_game_count: object
        lineup_skipped_ids: object
        lineup_source: object
        lineups: object
        mapped_espn_ids: object
        mapped_ncaa_ids: object
        mapping: object
        mbb: object
        missing_espn_ids: object
        missing_pbp_games: object
        ncaa_game_id: object
        ncaa_pbp: object
        ncaa_pbp_url: object
        out_lineups: object
        out_possessions: object
        pl_module: object
        possession_coverage: object
        possession_extra: object
        possession_failed_games: object
        possession_frames: object
        possession_game_count: object
        possession_skipped_ids: object
        possession_source: object
        possessions: object
        required_functions: object
        root: object
        transform_pbp: object
        xwalk: object
        xwalk_rows: object
        xwalk_url: object
        zero_lineup_games: object
        zero_possession_games: object

        def _py_r1000_loop_1():
            if not hasattr(mbb, function_name):
                raise RuntimeError(f'SportsDataVerse function missing: sportsdataverse.mbb.{function_name}')
            return (_py_r1000_NONE, None)

        def _py_r1000_loop_3():
            nonlocal espn_game_id, exc, game_identity, game_pbp, lineup_game_count, lineups, ncaa_game_id, possession_game_count, possessions, transform_pbp

            def _py_r1000_try_4():
                nonlocal possession_game_count, possessions
                possessions = mbb.ncaa_mbb_possessions(transform_pbp, simple=False, fix_cross_game_leak=True, return_as_pandas=False)
                possessions = stamp_ncaam_derived_ids(possessions, espn_game_id=espn_game_id, ncaa_game_id=ncaa_game_id)
                if possessions.height == 0:
                    zero_possession_games.append(game_identity)
                    log(f'NCAAM GAME ZERO POSSESSIONS | game_id={espn_game_id} ncaa_game_id={ncaa_game_id} action=skipped')
                else:
                    possession_frames.append(possessions)
                    possession_game_count += 1
                return (_py_r1000_NONE, None)

            def _py_r1000_try_6():
                nonlocal lineup_game_count, lineups
                lineups = mbb.ncaa_mbb_lineups(transform_pbp, include_transition=False, fix_tip_in=True, return_as_pandas=False)
                lineups = stamp_ncaam_derived_ids(lineups, espn_game_id=espn_game_id, ncaa_game_id=ncaa_game_id)
                if lineups.height == 0:
                    zero_lineup_games.append(game_identity)
                    log(f'NCAAM GAME ZERO LINEUPS | game_id={espn_game_id} ncaa_game_id={ncaa_game_id} action=skipped')
                else:
                    lineup_frames.append(lineups)
                    lineup_game_count += 1
                return (_py_r1000_NONE, None)
            espn_game_id = clean(mapping['game_id'])
            ncaa_game_id = clean(mapping['ncaa_game_id'])
            game_identity = {'game_id': espn_game_id, 'ncaa_game_id': ncaa_game_id}
            game_pbp = game_partitions.get(ncaa_game_id)
            if game_pbp is None or game_pbp.height == 0:
                missing_pbp_games.append(game_identity)
                log(f'NCAAM GAME NO NCAA PBP | game_id={espn_game_id} ncaa_game_id={ncaa_game_id} action=skipped')
                return (_py_r1000_CONTINUE, None)
            try:
                transform_pbp = prepare_ncaa_pbp_for_transform(game_pbp)
            except Exception as exc:
                possession_failed_games.append({**game_identity, 'error': str(exc)})
                lineup_failed_games.append({**game_identity, 'error': str(exc)})
                log(f'NCAAM GAME PBP PREP FAILED | game_id={espn_game_id} ncaa_game_id={ncaa_game_id} error={exc} action=skipped')
                return (_py_r1000_CONTINUE, None)
            try:
                _py_r1000_result_5 = _py_r1000_try_4()
                if _py_r1000_result_5[0] != _py_r1000_NONE:
                    return _py_r1000_result_5
            except Exception as exc:
                possession_failed_games.append({**game_identity, 'error': str(exc)})
                log(f'NCAAM GAME POSSESSIONS FAILED | game_id={espn_game_id} ncaa_game_id={ncaa_game_id} error={exc} action=skipped')
            try:
                _py_r1000_result_7 = _py_r1000_try_6()
                if _py_r1000_result_7[0] != _py_r1000_NONE:
                    return _py_r1000_result_7
            except Exception as exc:
                lineup_failed_games.append({**game_identity, 'error': str(exc)})
                log(f'NCAAM GAME LINEUPS FAILED | game_id={espn_game_id} ncaa_game_id={ncaa_game_id} error={exc} action=skipped')
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_9():
            nonlocal canonical_game_count, espn_game_ids, games, games_file, lineup_source, mapped_espn_ids, missing_espn_ids, out_lineups, out_possessions, pl_module, possession_source, root, xwalk, xwalk_url
            pl_module = pl()
            root = storage_root(cfg)
            out_possessions = table_path(root, 'ncaam', internal_season, 'possessions')
            out_lineups = table_path(root, 'ncaam', internal_season, 'lineups')
            possession_source = 'sportsdataverse/ncaa-mbb-hoops-data:stats.ncaa.org PBP -> sportsdataverse.mbb.ncaa_mbb_possessions'
            lineup_source = 'sportsdataverse/ncaa-mbb-hoops-data:stats.ncaa.org PBP -> sportsdataverse.mbb.ncaa_mbb_lineups'
            if out_possessions.exists() and out_lineups.exists() and (not force):
                return (_py_r1000_RETURN, {'possessions': manifest_entry(pl_module.read_parquet(out_possessions), out_possessions, possession_source, 'existing_not_rebuilt'), 'lineups': manifest_entry(pl_module.read_parquet(out_lineups), out_lineups, lineup_source, 'existing_not_rebuilt')})
            games_file = table_path(root, 'ncaam', internal_season, 'games')
            if not games_file.exists():
                raise RuntimeError(f'NCAAM games parquet missing: {games_file}')
            games = pl_module.read_parquet(games_file)
            if 'game_id' not in games.columns:
                raise RuntimeError('NCAAM games.parquet missing game_id')
            espn_game_ids = set(games.get_column('game_id').cast(pl_module.Utf8, strict=False).drop_nulls().to_list())
            if not espn_game_ids:
                raise RuntimeError('NCAAM games.parquet contains zero usable game_id values')
            canonical_game_count = len(espn_game_ids)
            xwalk, xwalk_url = load_ncaam_game_crosswalk(sdv_season)
            xwalk = xwalk.filter(pl_module.col('game_id').is_in(sorted(espn_game_ids)))
            if xwalk.height == 0:
                raise RuntimeError(f'No NCAA contest ids map to NCAAM games for SDV season={sdv_season}')
            mapped_espn_ids = set(xwalk.get_column('game_id').to_list())
            missing_espn_ids = sorted(espn_game_ids - mapped_espn_ids)
            log(f'NCAAM CROSSWALK | internal={internal_season} sdv={sdv_season} espn_games={canonical_game_count} mapped={len(mapped_espn_ids)} unmapped={len(missing_espn_ids)} source={xwalk_url}')
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_10():
            nonlocal game_id, game_partitions, mapped_ncaa_ids, mbb, ncaa_pbp, ncaa_pbp_url, required_functions
            for game_id in missing_espn_ids:
                log(f'NCAAM GAME UNMAPPED | game_id={game_id} ncaa_game_id=UNRESOLVED')
            ncaa_pbp_url = NCAAM_NCAA_PBP_URL.format(season=sdv_season)
            log(f'NCAAM NCAA PBP DOWNLOAD START | internal={internal_season} sdv={sdv_season} url={ncaa_pbp_url}')
            ncaa_pbp = download_parquet(ncaa_pbp_url)
            if ncaa_pbp.height == 0:
                raise RuntimeError(f'Published NCAA PBP returned zero rows: {ncaa_pbp_url}')
            if 'contest_id' not in ncaa_pbp.columns:
                raise RuntimeError(f'Published NCAA PBP missing contest_id; columns={ncaa_pbp.columns}')
            ncaa_pbp = ncaa_pbp.with_columns(pl_module.col('contest_id').cast(pl_module.Utf8, strict=False).alias('contest_id'))
            mapped_ncaa_ids = set(xwalk.get_column('ncaa_game_id').to_list())
            ncaa_pbp = ncaa_pbp.filter(pl_module.col('contest_id').is_in(sorted(mapped_ncaa_ids)))
            if ncaa_pbp.height == 0:
                raise RuntimeError('Published NCAA PBP has zero rows for mapped NCAAM games')
            game_partitions = {clean(part.get_column('contest_id')[0]): part for part in ncaa_pbp.partition_by('contest_id', maintain_order=False) if part.height}
            log(f'NCAAM NCAA PBP DOWNLOAD COMPLETE | rows={ncaa_pbp.height} games={len(game_partitions)}')
            mbb = importlib.import_module('sportsdataverse.mbb')
            required_functions = ('ncaa_mbb_possessions', 'ncaa_mbb_lineups')
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_11():
            nonlocal function_name, lineup_failed_games, lineup_frames, lineup_game_count, missing_pbp_games, possession_failed_games, possession_frames, possession_game_count, xwalk_rows, zero_lineup_games, zero_possession_games

            def _py_r1000_loop_12():
                _py_r1000_result_2 = _py_r1000_loop_1()
                if _py_r1000_result_2[0] == _py_r1000_RETURN:
                    return _py_r1000_result_2
                if _py_r1000_result_2[0] == _py_r1000_BREAK:
                    return (_py_r1000_BREAK, None)
                if _py_r1000_result_2[0] == _py_r1000_CONTINUE:
                    return (_py_r1000_CONTINUE, None)
                return (_py_r1000_NONE, None)
            for function_name in required_functions:
                _py_r1000_result_13 = _py_r1000_loop_12()
                if _py_r1000_result_13[0] == _py_r1000_RETURN:
                    return _py_r1000_result_13
                if _py_r1000_result_13[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_13[0] == _py_r1000_CONTINUE:
                    continue
            possession_frames = []
            lineup_frames = []
            missing_pbp_games = []
            possession_failed_games = []
            lineup_failed_games = []
            zero_possession_games = []
            zero_lineup_games = []
            possession_game_count = 0
            lineup_game_count = 0
            xwalk_rows = xwalk.sort('game_id').iter_rows(named=True)
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_14():
            nonlocal lineups, mapping, possessions

            def _py_r1000_loop_15():
                _py_r1000_result_8 = _py_r1000_loop_3()
                if _py_r1000_result_8[0] == _py_r1000_RETURN:
                    return _py_r1000_result_8
                if _py_r1000_result_8[0] == _py_r1000_BREAK:
                    return (_py_r1000_BREAK, None)
                if _py_r1000_result_8[0] == _py_r1000_CONTINUE:
                    return (_py_r1000_CONTINUE, None)
                return (_py_r1000_NONE, None)
            for mapping in xwalk_rows:
                _py_r1000_result_16 = _py_r1000_loop_15()
                if _py_r1000_result_16[0] == _py_r1000_RETURN:
                    return _py_r1000_result_16
                if _py_r1000_result_16[0] == _py_r1000_BREAK:
                    break
                if _py_r1000_result_16[0] == _py_r1000_CONTINUE:
                    continue
            if not possession_frames:
                raise RuntimeError('NCAAM possessions produced zero usable season rows')
            if not lineup_frames:
                raise RuntimeError('NCAAM lineups produced zero usable season rows')
            possessions = pl_module.concat(possession_frames, how='diagonal_relaxed')
            lineups = pl_module.concat(lineup_frames, how='diagonal_relaxed')
            possessions = normalize(possessions, table='possessions', league='ncaam', internal_season=internal_season, sdv_season=sdv_season, source=possession_source, ingested_at_utc=ingested_at_utc)
            lineups = normalize(lineups, table='lineups', league='ncaam', internal_season=internal_season, sdv_season=sdv_season, source=lineup_source, ingested_at_utc=ingested_at_utc)
            write_parquet(possessions, out_possessions, compression(cfg))
            write_parquet(lineups, out_lineups, compression(cfg))
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_17():
            nonlocal lineup_skipped_ids, possession_coverage, possession_skipped_ids
            possession_skipped_ids = sorted({*missing_espn_ids, *[item['game_id'] for item in missing_pbp_games], *[item['game_id'] for item in possession_failed_games], *[item['game_id'] for item in zero_possession_games]})
            lineup_skipped_ids = sorted({*missing_espn_ids, *[item['game_id'] for item in missing_pbp_games], *[item['game_id'] for item in lineup_failed_games], *[item['game_id'] for item in zero_lineup_games]})
            possession_coverage = 'complete' if not possession_skipped_ids else 'partial_source_coverage'
            return (_py_r1000_NONE, None)

        def _py_r1000_chunk_18():
            nonlocal lineup_coverage, lineup_extra, possession_extra
            lineup_coverage = 'complete' if not lineup_skipped_ids else 'partial_source_coverage'
            possession_extra = {'coverage_status': possession_coverage, 'canonical_games': canonical_game_count, 'crosswalk_mapped_games': int(xwalk.height), 'transformed_games': possession_game_count, 'unmapped_games': len(missing_espn_ids), 'missing_pbp_games': len(missing_pbp_games), 'transform_failed_games': len(possession_failed_games), 'zero_output_games': len(zero_possession_games), 'skipped_game_ids': possession_skipped_ids, 'source_pbp_url': ncaa_pbp_url, 'crosswalk_url': xwalk_url}
            lineup_extra = {'coverage_status': lineup_coverage, 'canonical_games': canonical_game_count, 'crosswalk_mapped_games': int(xwalk.height), 'transformed_games': lineup_game_count, 'unmapped_games': len(missing_espn_ids), 'missing_pbp_games': len(missing_pbp_games), 'transform_failed_games': len(lineup_failed_games), 'zero_output_games': len(zero_lineup_games), 'skipped_game_ids': lineup_skipped_ids, 'source_pbp_url': ncaa_pbp_url, 'crosswalk_url': xwalk_url}
            log(f'LOAD COMPLETE | league=ncaam internal={internal_season} table=possessions rows={possessions.height} games={possession_game_count} coverage={possession_coverage} skipped_games={len(possession_skipped_ids)} status=ready file={out_possessions}')
            log(f'LOAD COMPLETE | league=ncaam internal={internal_season} table=lineups rows={lineups.height} games={lineup_game_count} coverage={lineup_coverage} skipped_games={len(lineup_skipped_ids)} status=ready file={out_lineups}')
            log(f'NCAAM DERIVED COMPLETE | canonical_games={canonical_game_count} crosswalk_mapped={xwalk.height} unmapped={len(missing_espn_ids)} published_pbp_games={len(game_partitions)} missing_pbp={len(missing_pbp_games)} possession_games={possession_game_count} possession_failures={len(possession_failed_games)} possession_zero={len(zero_possession_games)} lineup_games={lineup_game_count} lineup_failures={len(lineup_failed_games)} lineup_zero={len(zero_lineup_games)}')
            return (_py_r1000_RETURN, {'possessions': manifest_entry(possessions, out_possessions, possession_source, 'ready', possession_extra), 'lineups': manifest_entry(lineups, out_lineups, lineup_source, 'ready', lineup_extra)})
        for _py_r1000_block_19 in (_py_r1000_chunk_9, _py_r1000_chunk_10, _py_r1000_chunk_11, _py_r1000_chunk_14, _py_r1000_chunk_17, _py_r1000_chunk_18):
            _py_r1000_result_20 = _py_r1000_block_19()
            if _py_r1000_result_20[0] != _py_r1000_NONE:
                return _py_r1000_result_20
        return (_py_r1000_NONE, None)
    _py_r1000_outcome = _py_r1000_impl()
    if _py_r1000_outcome[0] == _py_r1000_RETURN:
        return _py_r1000_outcome[1]


def build_season(
    cfg: dict[str, Any],
    league: str,
    internal_season: int,
    force: bool,
) -> Path:
    root = storage_root(cfg)

    sdv_season = int(
        sdv_season_id(
            league,
            internal_season,
            config_path=(
                SEASON_CONFIG_PATH
            ),
        )
    )

    ingested_at_utc = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    manifest: dict[
        str,
        Any,
    ] = {
        "schema_version": 1,
        "generated_at_utc": ingested_at_utc,
        "sportsdataverse_version": clean(
            cfg[
                "sportsdataverse"
            ][
                "expected_version"
            ]
        ),
        "league": league.upper(),
        "internal_season": internal_season,
        "sdv_season": sdv_season,
        "storage_format": "parquet",
        "compression": compression(cfg),
        "sportsbook_snapshots": {
            "path": clean(
                cfg[
                    "sportsbook_snapshots"
                ][
                    "root"
                ]
            ),
            "reused_external_to_sdv_storage": True,
            "copied_into_sdv_storage": False,
        },
        "tables": {},
    }

    for table in TABLES:
        if (
            league == "ncaam"
            and table
            in {
                "possessions",
                "lineups",
            }
        ):
            continue

        manifest[
            "tables"
        ][
            table
        ] = build_release_table(
            cfg,
            league,
            internal_season,
            sdv_season,
            table,
            ingested_at_utc,
            force,
        )

    if league == "ncaam":
        manifest[
            "tables"
        ].update(
            build_ncaam_derived_tables(
                cfg,
                internal_season,
                sdv_season,
                ingested_at_utc,
                force,
            )
        )

    manifest[
        "tables"
    ] = {
        table: (
            manifest[
                "tables"
            ][
                table
            ]
        )
        for table
        in TABLES
    }

    output = (
        partition(
            root,
            league,
            internal_season,
        )
        / "manifest.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = output.with_suffix(
        ".json.tmp"
    )

    tmp.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    tmp.replace(output)

    log(
        "MANIFEST | "
        f"league={league} "
        f"internal={internal_season} "
        f"path={output}"
    )

    return output


def resolve_jobs(
    cfg: dict[str, Any],
    only_leagues: list[str] | None,
    only_seasons: list[int] | None,
) -> list[
    tuple[
        str,
        int,
    ]
]:
    selected_leagues = (
        only_leagues
        or list(
            LEAGUES
        )
    )

    selected_seasons = set(
        only_seasons
        or []
    )

    configured = cfg[
        "historical_internal_seasons"
    ]

    result: list[
        tuple[
            str,
            int,
        ]
    ] = []

    for league in selected_leagues:
        if league not in LEAGUES:
            raise ValueError(
                "Unsupported league: "
                f"{league}"
            )

        for season in sorted(
            {
                int(value)
                for value
                in configured[
                    league
                ]
            }
        ):
            if (
                selected_seasons
                and season
                not in selected_seasons
            ):
                continue

            result.append(
                (
                    league,
                    season,
                )
            )

    if (
        only_seasons
        and not result
    ):
        raise ValueError(
            "Requested internal season "
            "is not configured as historical"
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--league",
        action="append",
        choices=LEAGUES,
    )

    parser.add_argument(
        "--internal-season",
        action="append",
        type=int,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    parser.add_argument(
        "--validate-config",
        action="store_true",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
    )

    args = parser.parse_args()

    ERROR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_FILE.write_text(
        (
            "=== SDV HISTORICAL STORAGE "
            f"{datetime.now(timezone.utc).isoformat()} ===\n"
        ),
        encoding="utf-8",
    )

    try:
        cfg = load_config(
            args.config
        )

        version = verify_version(
            cfg
        )

        work = resolve_jobs(
            cfg,
            args.league,
            args.internal_season,
        )

        mappings = {
            f"{league}:{season}": int(
                sdv_season_id(
                    league,
                    season,
                    config_path=(
                        SEASON_CONFIG_PATH
                    ),
                )
            )
            for (
                league,
                season,
            )
            in work
        }

        log(
            "CONFIG VALID | "
            f"sportsdataverse={version} "
            f"jobs={work} "
            f"mappings={mappings}"
        )

        if args.validate_config:
            print(
                "SDV historical storage "
                "config valid."
            )
            return

        manifests = [
            build_season(
                cfg,
                league,
                season,
                args.force,
            )
            for (
                league,
                season,
            )
            in work
        ]

        log(
            "STATUS: SUCCESS | "
            f"manifests={len(manifests)}"
        )

        print(
            "SDV historical basketball "
            "storage complete."
        )

    except Exception as exc:
        log(
            f"FATAL: {exc}"
        )

        log(
            traceback
            .format_exc()
            .rstrip()
        )

        log(
            "STATUS: FAILED"
        )

        print(
            "SDV historical storage failed: "
            f"{exc}"
        )

        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
