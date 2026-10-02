#!/usr/bin/env python3
# docs/win/basketball/scripts/testing/sdv_storage_test.py
"""Validate SportsDataVerse historical basketball storage."""
from __future__ import annotations

import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml

BASE = Path("docs/win/basketball")
STORAGE_CONFIG = BASE / "config/sdv_storage.yaml"
SEASON_CONFIG = BASE / "config/sdv_seasons.yaml"
VALIDATION_LOG = (
    BASE
    / "errors/99_validation/sdv_storage_validation.txt"
)

EXPECTED_SDV_VERSION = "0.0.75"

LEAGUES = (
    "nba",
    "ncaam",
    "wnba",
)

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

REQUIRED_FILES = (
    *(
        f"{table}.parquet"
        for table in TABLES
    ),
    "manifest.json",
)

METADATA_COLUMNS = (
    "league",
    "internal_season",
    "sdv_season",
    "source_loader",
    "ingested_at_utc",
)

REQUIRED_COLUMNS = {
    "games": (
        "game_id",
        "game_date",
        "home_team_id",
        "away_team_id",
    ),
    "team_game": (
        "game_id",
        "team_id",
    ),
    "player_game": (
        "game_id",
        "player_id",
        "team_id",
    ),
    "pbp": (
        "game_id",
    ),
    "possessions": (
        "game_id",
    ),
    "shots": (
        "game_id",
    ),
}

NON_NULL_REQUIRED_COLUMNS = {
    "games": (
        "game_id",
        "game_date",
        "home_team_id",
        "away_team_id",
    ),
    "team_game": (
        "game_id",
        "team_id",
    ),
    "player_game": (
        "game_id",
        "team_id",
    ),
    "pbp": (
        "game_id",
    ),
    "possessions": (
        "game_id",
    ),
    "shots": (
        "game_id",
    ),
}

NCAAM_GAMES_VENUE_COLUMNS = (
    "neutral_site",
    "venue_id",
    "venue_full_name",
    "venue_name",
    "home_team_id",
    "away_team_id",
)

NCAAM_GAMES_SOURCE_TEAM_ID_COLUMNS = (
    "home_id",
    "away_id",
)

GAME_KEY_TABLES = (
    "team_game",
    "player_game",
    "pbp",
    "possessions",
    "lineups",
    "shots",
)

VALID_MANIFEST_STATUSES = {
    "ready",
    "existing_not_rebuilt",
}

NCAAM_LINEUP_IDENTITY = (
    "game_id",
    "team",
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
)

PRO_LINEUP_IDENTITY = (
    "game_id",
    "action_number",
    "period",
    "home_player_1",
    "home_player_2",
    "home_player_3",
    "home_player_4",
    "home_player_5",
    "away_player_1",
    "away_player_2",
    "away_player_3",
    "away_player_4",
    "away_player_5",
)

TRUNCATION_BOOLEAN_KEYS = {
    "truncated",
    "truncation_guard_hit",
    "limit_reached",
    "hit_limit",
    "stopped_at_limit",
}

GAME_LIMIT_KEYS = {
    "max_games",
    "game_limit",
    "season_game_limit",
    "max_games_per_season",
}

ROW_LIMIT_KEYS = {
    "max_rows",
    "row_limit",
    "truncation_limit",
    "loader_limit",
}


class Validation:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.seasons_checked = 0
        self.tables_checked = 0

    def emit(
        self,
        text: str,
    ) -> None:
        self.lines.append(text)

    def passed(
        self,
        text: str,
    ) -> None:
        self.emit(
            f"PASS | {text}"
        )

    def warn(
        self,
        text: str,
    ) -> None:
        self.warnings.append(text)
        self.emit(
            f"WARN | {text}"
        )

    def fail(
        self,
        text: str,
    ) -> None:
        self.errors.append(text)
        self.emit(
            f"FAIL | {text}"
        )

    def write_log(self) -> None:
        VALIDATION_LOG.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        VALIDATION_LOG.write_text(
            "\n".join(
                self.lines
            )
            + "\n",
            encoding="utf-8",
        )


def clean(
    value: Any,
) -> str:
    if value is None:
        return ""

    return str(value).strip()


def read_yaml(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    payload = (
        yaml.safe_load(
            path.read_text(
                encoding="utf-8"
            )
        )
        or {}
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "YAML root must be a mapping: "
            f"{path}"
        )

    return payload


def storage_root(
    cfg: dict[str, Any],
) -> Path:
    storage = cfg.get(
        "storage"
    )

    if not isinstance(
        storage,
        dict,
    ):
        raise ValueError(
            "storage section missing "
            "from sdv_storage.yaml"
        )

    root = clean(
        storage.get("root")
    )

    if not root:
        raise ValueError(
            "storage.root is empty"
        )

    return Path(root)


def configured_jobs(
    cfg: dict[str, Any],
) -> list[
    tuple[
        str,
        int,
    ]
]:
    section = cfg.get(
        "historical_internal_seasons"
    )

    if not isinstance(
        section,
        dict,
    ):
        raise ValueError(
            "historical_internal_seasons "
            "must be a mapping"
        )

    jobs: list[
        tuple[
            str,
            int,
        ]
    ] = []

    for league in LEAGUES:
        seasons = section.get(
            league
        )

        if (
            not isinstance(
                seasons,
                list,
            )
            or not seasons
        ):
            raise ValueError(
                "historical_internal_seasons."
                f"{league} must be non-empty"
            )

        for season in seasons:
            jobs.append(
                (
                    league,
                    int(season),
                )
            )

    return jobs


def mapped_sdv_season(
    cfg: dict[str, Any],
    league: str,
    internal_season: int,
) -> int:
    leagues = cfg.get(
        "leagues"
    )

    if (
        not isinstance(
            leagues,
            dict,
        )
        or league not in leagues
    ):
        raise KeyError(
            "Missing league mapping: "
            f"{league}"
        )

    league_cfg = leagues[
        league
    ]

    if not isinstance(
        league_cfg,
        dict,
    ):
        raise ValueError(
            "Invalid league mapping "
            f"section: {league}"
        )

    mappings = league_cfg.get(
        "mappings"
    )

    if not isinstance(
        mappings,
        dict,
    ):
        raise KeyError(
            f"Missing mappings: {league}"
        )

    value = mappings.get(
        internal_season
    )

    if value is None:
        value = mappings.get(
            str(internal_season)
        )

    if value is None:
        raise KeyError(
            "Missing mapping: "
            f"{league}:{internal_season}"
        )

    return int(value)


def parquet_info(
    path: Path,
) -> tuple[
    int,
    list[str],
]:
    scan = pl.scan_parquet(path)

    columns = (
        scan.collect_schema()
        .names()
    )

    rows = int(
        scan.select(
            pl.len().alias(
                "rows"
            )
        )
        .collect()
        .item()
    )

    return (
        rows,
        columns,
    )


def metadata_stats(
    path: Path,
) -> dict[
    str,
    dict[str, Any],
]:
    scan = pl.scan_parquet(path)

    available = set(
        scan.collect_schema()
        .names()
    )

    present = [
        column
        for column in METADATA_COLUMNS
        if column in available
    ]

    if not present:
        return {}

    expressions: list[
        pl.Expr
    ] = []

    for column in present:
        expressions.extend(
            [
                pl.col(column)
                .null_count()
                .alias(
                    f"{column}__nulls"
                ),
                pl.col(column)
                .drop_nulls()
                .n_unique()
                .alias(
                    f"{column}__unique"
                ),
                pl.col(column)
                .drop_nulls()
                .first()
                .alias(
                    f"{column}__first"
                ),
            ]
        )

    row = (
        scan.select(expressions)
        .collect()
        .to_dicts()[0]
    )

    return {
        column: {
            "nulls": int(
                row[
                    f"{column}__nulls"
                ]
            ),
            "unique": int(
                row[
                    f"{column}__unique"
                ]
            ),
            "first": row[
                f"{column}__first"
            ],
        }
        for column in present
    }


def null_blank_counts(
    path: Path,
    columns: tuple[str, ...],
) -> dict[
    str,
    tuple[
        int,
        int,
    ],
]:
    scan = pl.scan_parquet(path)

    available = set(
        scan.collect_schema()
        .names()
    )

    present = [
        column
        for column in columns
        if column in available
    ]

    if not present:
        return {}

    expressions: list[
        pl.Expr
    ] = []

    for column in present:
        expressions.extend(
            [
                pl.col(column)
                .null_count()
                .alias(
                    f"{column}__nulls"
                ),
                pl.col(column)
                .cast(
                    pl.String,
                    strict=False,
                )
                .str.strip_chars()
                .eq("")
                .fill_null(False)
                .sum()
                .alias(
                    f"{column}__blanks"
                ),
            ]
        )

    row = (
        scan.select(expressions)
        .collect()
        .to_dicts()[0]
    )

    return {
        column: (
            int(
                row[
                    f"{column}__nulls"
                ]
            ),
            int(
                row[
                    f"{column}__blanks"
                ]
            ),
        )
        for column in present
    }


def unique_game_ids(
    path: Path,
) -> set[str]:
    frame = (
        pl.scan_parquet(path)
        .select(
            pl.col("game_id")
            .cast(
                pl.String,
                strict=False,
            )
            .str.strip_chars()
            .alias("game_id")
        )
        .filter(
            pl.col("game_id").is_not_null()
            & (
                pl.col("game_id")
                != ""
            )
        )
        .unique()
        .collect()
    )

    return set(
        frame.get_column(
            "game_id"
        ).to_list()
    )


def validate_metadata(
    validation: Validation,
    path: Path,
    league: str,
    internal_season: int,
    sdv_season: int,
) -> str:
    stats = metadata_stats(path)

    missing = [
        column
        for column in METADATA_COLUMNS
        if column not in stats
    ]

    if missing:
        validation.fail(
            f"{path} missing metadata "
            f"columns {missing}"
        )
        return ""

    expected = {
        "league": league.upper(),
        "internal_season": (
            internal_season
        ),
        "sdv_season": sdv_season,
    }

    for (
        column,
        info,
    ) in stats.items():
        if info["nulls"]:
            validation.fail(
                f"{path} metadata "
                f"{column} has "
                f"{info['nulls']} null rows"
            )

        if info["unique"] != 1:
            validation.fail(
                f"{path} metadata "
                f"{column} has "
                f"{info['unique']} distinct "
                "non-null values"
            )

    for (
        column,
        expected_value,
    ) in expected.items():
        actual = stats[
            column
        ]["first"]

        if column in {
            "internal_season",
            "sdv_season",
        }:
            try:
                actual = int(actual)

            except (
                TypeError,
                ValueError,
            ):
                pass

        else:
            actual = clean(actual)

        if actual != expected_value:
            validation.fail(
                f"{path} metadata "
                f"{column}={actual!r}; "
                f"expected={expected_value!r}"
            )

    for column in (
        "source_loader",
        "ingested_at_utc",
    ):
        if not clean(
            stats[column]["first"]
        ):
            validation.fail(
                f"{path} metadata "
                f"{column} is blank"
            )

    return clean(
        stats[
            "source_loader"
        ]["first"]
    )


def validate_required_keys(
    validation: Validation,
    path: Path,
    table: str,
    columns: list[str],
) -> None:
    required = REQUIRED_COLUMNS.get(
        table,
        (),
    )

    missing = [
        column
        for column in required
        if column not in columns
    ]

    if missing:
        validation.fail(
            f"{path} missing required "
            f"columns {missing}"
        )
        return

    non_null_required = (
        NON_NULL_REQUIRED_COLUMNS.get(
            table,
            (),
        )
    )

    counts = null_blank_counts(
        path,
        non_null_required,
    )

    for (
        column,
        (
            nulls,
            blanks,
        ),
    ) in counts.items():
        if nulls or blanks:
            validation.fail(
                f"{path} required key "
                f"{column} has "
                f"nulls={nulls} "
                f"blanks={blanks}"
            )

    if (
        table == "player_game"
        and "player_id" in columns
    ):
        player_id_counts = (
            null_blank_counts(
                path,
                ("player_id",),
            )
        )

        (
            player_nulls,
            player_blanks,
        ) = player_id_counts.get(
            "player_id",
            (
                0,
                0,
            ),
        )

        if (
            player_nulls
            or player_blanks
        ):
            validation.emit(
                "INFO | "
                f"{path} player_id "
                "nullable source rows "
                f"nulls={player_nulls} "
                f"blanks={player_blanks}"
            )


def normalize_venue_name_expr(
    column: str,
) -> pl.Expr:
    return (
        pl.col(column)
        .cast(
            pl.String,
            strict=False,
        )
        .fill_null("")
        .str.strip_chars()
        .str.replace_all(
            r"\s+",
            " ",
        )
    )


def normalize_id_expr(
    column: str,
) -> pl.Expr:
    return (
        pl.col(column)
        .cast(
            pl.String,
            strict=False,
        )
        .fill_null("")
        .str.strip_chars()
        .str.replace(
            r"\.0$",
            "",
        )
    )


def validate_ncaam_venue_context(
    validation: Validation,
    path: Path,
    columns: list[str],
) -> None:
    column_set = set(columns)

    missing = [
        column
        for column
        in NCAAM_GAMES_VENUE_COLUMNS
        if column not in column_set
    ]

    missing_source_ids = [
        column
        for column
        in NCAAM_GAMES_SOURCE_TEAM_ID_COLUMNS
        if column not in column_set
    ]

    if missing:
        validation.fail(
            f"{path} NCAAM venue context "
            f"missing columns {missing}"
        )
        return

    if missing_source_ids:
        validation.fail(
            f"{path} NCAAM source team IDs "
            f"missing columns {missing_source_ids}"
        )
        return

    frame = pl.scan_parquet(path)

    neutral_text = (
        pl.col("neutral_site")
        .cast(
            pl.String,
            strict=False,
        )
        .fill_null("")
        .str.strip_chars()
        .str.to_lowercase()
    )

    stats = (
        frame.select(
            [
                pl.col("neutral_site")
                .null_count()
                .alias(
                    "neutral_nulls"
                ),
                neutral_text
                .eq("")
                .sum()
                .alias(
                    "neutral_blanks"
                ),
                (
                    ~neutral_text.is_in(
                        [
                            "true",
                            "false",
                            "1",
                            "0",
                            "yes",
                            "no",
                        ]
                    )
                    & neutral_text.ne("")
                )
                .sum()
                .alias(
                    "neutral_invalid"
                ),
                (
                    normalize_id_expr(
                        "home_id"
                    )
                    != normalize_id_expr(
                        "home_team_id"
                    )
                )
                .sum()
                .alias(
                    "home_id_mismatch"
                ),
                (
                    normalize_id_expr(
                        "away_id"
                    )
                    != normalize_id_expr(
                        "away_team_id"
                    )
                )
                .sum()
                .alias(
                    "away_id_mismatch"
                ),
                (
                    normalize_venue_name_expr(
                        "venue_full_name"
                    )
                    != normalize_venue_name_expr(
                        "venue_name"
                    )
                )
                .sum()
                .alias(
                    "venue_name_mismatch"
                ),
                pl.col("venue_id")
                .is_not_null()
                .sum()
                .alias(
                    "venue_id_present"
                ),
                normalize_venue_name_expr(
                    "venue_full_name"
                )
                .ne("")
                .sum()
                .alias(
                    "venue_name_present"
                ),
            ]
        )
        .collect()
        .to_dicts()[0]
    )

    if stats["neutral_invalid"]:
        validation.fail(
            f"{path} neutral_site has "
            "unrecognized nonblank values="
            f"{stats['neutral_invalid']}"
        )

    if stats["home_id_mismatch"]:
        validation.fail(
            f"{path} home_team_id does not "
            "preserve source home_id rows="
            f"{stats['home_id_mismatch']}"
        )

    if stats["away_id_mismatch"]:
        validation.fail(
            f"{path} away_team_id does not "
            "preserve source away_id rows="
            f"{stats['away_id_mismatch']}"
        )

    if stats["venue_name_mismatch"]:
        validation.fail(
            f"{path} venue_name is not the "
            "normalized venue_full_name rows="
            f"{stats['venue_name_mismatch']}"
        )

    if (
        not stats["neutral_invalid"]
        and not stats["home_id_mismatch"]
        and not stats["away_id_mismatch"]
        and not stats[
            "venue_name_mismatch"
        ]
    ):
        validation.passed(
            f"{path} NCAAM venue context "
            "preserved "
            f"neutral_nulls="
            f"{stats['neutral_nulls']} "
            f"neutral_blanks="
            f"{stats['neutral_blanks']} "
            f"venue_id_present="
            f"{stats['venue_id_present']} "
            f"venue_name_present="
            f"{stats['venue_name_present']}"
        )


def detect_pro_lineup_identity(
    columns: list[str],
) -> tuple[
    bool,
    list[str],
    str,
]:
    column_set = set(columns)

    missing = [
        column
        for column
        in PRO_LINEUP_IDENTITY
        if column not in column_set
    ]

    if missing:
        return (
            False,
            [],
            (
                "missing expected per-game "
                f"lineup columns {missing}"
            ),
        )

    return (
        True,
        list(PRO_LINEUP_IDENTITY),
        (
            "game/action/period plus "
            "home/away five-player identity"
        ),
    )


def validate_lineups(
    validation: Validation,
    path: Path,
    league: str,
    columns: list[str],
) -> None:
    column_set = set(columns)

    if league == "ncaam":
        missing = [
            column
            for column
            in NCAAM_LINEUP_IDENTITY
            if column not in column_set
        ]

        if missing:
            validation.fail(
                f"{path} NCAAM lineup "
                f"identity missing {missing}; "
                "expected "
                f"{list(NCAAM_LINEUP_IDENTITY)}"
            )

        else:
            validation.passed(
                f"{path} NCAAM lineup "
                "identity="
                f"{list(NCAAM_LINEUP_IDENTITY)}"
            )

        return

    (
        valid,
        identity_columns,
        identity_type,
    ) = detect_pro_lineup_identity(
        columns
    )

    if valid:
        validation.passed(
            f"{path} "
            f"{league.upper()} lineup "
            f"identity type={identity_type} "
            f"columns={identity_columns}"
        )

    else:
        validation.fail(
            f"{path} "
            f"{league.upper()} lineup "
            f"identity invalid: "
            f"{identity_type}"
        )


def collect_configured_limits(
    node: Any,
    path: tuple[str, ...] = (),
) -> list[
    tuple[
        str,
        str,
        int,
    ]
]:
    found: list[
        tuple[
            str,
            str,
            int,
        ]
    ] = []

    if isinstance(node, dict):
        for (
            raw_key,
            value,
        ) in node.items():
            key = clean(
                raw_key
            ).lower()

            child_path = (
                *path,
                key,
            )

            path_text = ".".join(
                child_path
            )

            if (
                not isinstance(
                    value,
                    bool,
                )
                and isinstance(
                    value,
                    (
                        int,
                        float,
                    ),
                )
                and int(value) > 0
            ):
                if key in GAME_LIMIT_KEYS:
                    found.append(
                        (
                            path_text,
                            "games",
                            int(value),
                        )
                    )

                elif key in ROW_LIMIT_KEYS:
                    found.append(
                        (
                            path_text,
                            "rows",
                            int(value),
                        )
                    )

                elif (
                    key == "limit"
                    and any(
                        token
                        in ".".join(
                            path
                        ).lower()
                        for token in (
                            "loader",
                            "truncate",
                            "truncation",
                            "season",
                        )
                    )
                ):
                    found.append(
                        (
                            path_text,
                            "generic",
                            int(value),
                        )
                    )

            found.extend(
                collect_configured_limits(
                    value,
                    child_path,
                )
            )

    elif isinstance(node, list):
        for (
            index,
            value,
        ) in enumerate(node):
            found.extend(
                collect_configured_limits(
                    value,
                    (
                        *path,
                        str(index),
                    ),
                )
            )

    return found


def true_truncation_flags(
    node: Any,
    path: tuple[str, ...] = (),
) -> list[str]:
    found: list[str] = []

    if isinstance(node, dict):
        for (
            raw_key,
            value,
        ) in node.items():
            key = clean(
                raw_key
            ).lower()

            child_path = (
                *path,
                key,
            )

            if (
                key
                in TRUNCATION_BOOLEAN_KEYS
                and value is True
            ):
                found.append(
                    ".".join(
                        child_path
                    )
                )

            found.extend(
                true_truncation_flags(
                    value,
                    child_path,
                )
            )

    elif isinstance(node, list):
        for (
            index,
            value,
        ) in enumerate(node):
            found.extend(
                true_truncation_flags(
                    value,
                    (
                        *path,
                        str(index),
                    ),
                )
            )

    return found


def validate_manifest(
    validation: Validation,
    path: Path,
    league: str,
    internal_season: int,
    sdv_season: int,
    actual_tables: dict[
        str,
        dict[str, Any],
    ],
) -> None:
    try:
        manifest = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:
        validation.fail(
            f"Cannot read manifest "
            f"{path}: {exc}"
        )
        return

    if not isinstance(
        manifest,
        dict,
    ):
        validation.fail(
            f"{path} root is not an object"
        )
        return

    manifest_version = clean(
        manifest.get(
            "sportsdataverse_version"
        )
    )

    if (
        manifest_version
        != EXPECTED_SDV_VERSION
    ):
        validation.fail(
            f"{path} "
            "sportsdataverse_version="
            f"{manifest_version!r}; "
            "expected="
            f"{EXPECTED_SDV_VERSION!r}"
        )

    if (
        clean(
            manifest.get("league")
        ).upper()
        != league.upper()
    ):
        validation.fail(
            f"{path} league="
            f"{manifest.get('league')!r}; "
            f"expected={league.upper()!r}"
        )

    try:
        manifest_internal = int(
            manifest.get(
                "internal_season"
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        manifest_internal = None

    if (
        manifest_internal
        != internal_season
    ):
        validation.fail(
            f"{path} internal_season="
            f"{manifest.get('internal_season')!r}; "
            f"expected={internal_season}"
        )

    try:
        manifest_sdv = int(
            manifest.get("sdv_season")
        )

    except (
        TypeError,
        ValueError,
    ):
        manifest_sdv = None

    if manifest_sdv != sdv_season:
        validation.fail(
            f"{path} sdv_season="
            f"{manifest.get('sdv_season')!r}; "
            f"expected={sdv_season}"
        )

    entries = manifest.get("tables")

    if not isinstance(
        entries,
        dict,
    ):
        validation.fail(
            f"{path} tables is not an object"
        )
        return

    for table in TABLES:
        entry = entries.get(table)

        if not isinstance(
            entry,
            dict,
        ):
            validation.fail(
                f"{path} missing/invalid "
                f"table entry={table}"
            )
            continue

        actual = actual_tables.get(
            table
        )

        if actual is None:
            continue

        try:
            manifest_rows = int(
                entry.get("rows")
            )

        except (
            TypeError,
            ValueError,
        ):
            manifest_rows = -1

        if (
            manifest_rows
            != actual["rows"]
        ):
            validation.fail(
                f"{path} table={table} "
                "rows manifest="
                f"{entry.get('rows')!r} "
                f"actual={actual['rows']}"
            )

        if (
            entry.get("columns")
            != actual["columns"]
        ):
            validation.fail(
                f"{path} table={table} "
                "manifest schema does not "
                "match parquet schema"
            )

        expected_filename = (
            f"{table}.parquet"
        )

        if (
            clean(
                entry.get("filename")
            )
            != expected_filename
        ):
            validation.fail(
                f"{path} table={table} "
                "wrong filename="
                f"{entry.get('filename')!r}"
            )

        status = clean(
            entry.get("status")
        )

        if (
            status
            not in VALID_MANIFEST_STATUSES
        ):
            validation.fail(
                f"{path} table={table} "
                f"invalid status={status!r}"
            )

        if not clean(
            entry.get("source_loader")
        ):
            validation.fail(
                f"{path} table={table} "
                "missing source_loader"
            )

        if not clean(
            entry.get("source_function")
        ):
            validation.fail(
                f"{path} table={table} "
                "missing source_function"
            )

        coverage_status = clean(
            entry.get(
                "coverage_status"
            )
        )

        if (
            coverage_status
            and coverage_status
            != "complete"
        ):
            validation.warn(
                f"{path} table={table} "
                "coverage_status="
                f"{coverage_status}"
            )

    for flag in true_truncation_flags(
        manifest
    ):
        validation.fail(
            f"{path} reports reached "
            "truncation/limit flag at "
            f"{flag}"
        )

    validation.passed(
        f"{league}:{internal_season} "
        "manifest compared to "
        "parquet tables"
    )


def validate_truncation_limits(
    validation: Validation,
    limits: list[
        tuple[
            str,
            str,
            int,
        ]
    ],
    league: str,
    internal_season: int,
    actual_tables: dict[
        str,
        dict[str, Any],
    ],
) -> None:
    if not limits:
        validation.passed(
            f"{league}:{internal_season} "
            "no configured loader "
            "truncation guard"
        )
        return

    for (
        config_path,
        limit_type,
        limit,
    ) in limits:
        if (
            limit_type
            in {
                "games",
                "generic",
            }
            and "games"
            in actual_tables
        ):
            game_count = int(
                actual_tables[
                    "games"
                ].get(
                    "unique_games",
                    actual_tables[
                        "games"
                    ]["rows"],
                )
            )

            if game_count == limit:
                validation.fail(
                    f"{league}:{internal_season} "
                    "games stopped exactly at "
                    f"configured limit={limit} "
                    f"({config_path})"
                )

        if (
            limit_type
            in {
                "rows",
                "generic",
            }
        ):
            for (
                table,
                info,
            ) in actual_tables.items():
                if (
                    int(info["rows"])
                    == limit
                ):
                    validation.fail(
                        f"{league}:{internal_season} "
                        f"table={table} rows "
                        "equal configured "
                        f"limit={limit} "
                        f"({config_path})"
                    )


def validate_season(
    validation: Validation,
    storage_cfg: dict[str, Any],
    season_cfg: dict[str, Any],
    league: str,
    internal_season: int,
    limits: list[
        tuple[
            str,
            str,
            int,
        ]
    ],
) -> None:
    root = storage_root(
        storage_cfg
    )

    season_dir = (
        root
        / league
        / str(internal_season)
    )

    validation.emit(
        "--- "
        f"LEAGUE={league.upper()} "
        "INTERNAL_SEASON="
        f"{internal_season} "
        "---"
    )

    try:
        sdv_season = (
            mapped_sdv_season(
                season_cfg,
                league,
                internal_season,
            )
        )

    except Exception as exc:
        validation.fail(
            f"{league}:{internal_season} "
            "season mapping failed: "
            f"{exc}"
        )
        return

    validation.passed(
        f"{league}:{internal_season} "
        f"maps to SDV {sdv_season}"
    )

    if not season_dir.exists():
        validation.fail(
            "Missing configured "
            f"directory {season_dir}"
        )
        return

    missing_files = [
        str(
            season_dir
            / filename
        )
        for filename in REQUIRED_FILES
        if not (
            season_dir
            / filename
        ).is_file()
    ]

    if missing_files:
        validation.fail(
            f"{league}:{internal_season} "
            "missing required files "
            f"{missing_files}"
        )
        return

    odds_path = (
        season_dir
        / "odds.parquet"
    )

    if odds_path.exists():
        validation.fail(
            f"{league}:{internal_season} "
            "forbidden odds.parquet "
            "exists in SDV history"
        )

    else:
        validation.passed(
            f"{league}:{internal_season} "
            "required file set present"
        )

    actual_tables: dict[
        str,
        dict[str, Any],
    ] = {}

    for table in TABLES:
        path = (
            season_dir
            / f"{table}.parquet"
        )

        try:
            (
                row_count,
                columns,
            ) = parquet_info(path)

        except Exception as exc:
            validation.fail(
                f"Cannot read {path}: "
                f"{exc}"
            )
            continue

        validation.tables_checked += 1

        if row_count <= 0:
            validation.fail(
                f"{path} has zero rows"
            )
            continue

        validation.passed(
            f"{path} "
            f"rows={row_count} "
            f"columns={len(columns)}"
        )

        source_loader = (
            validate_metadata(
                validation,
                path,
                league,
                internal_season,
                sdv_season,
            )
        )

        validate_required_keys(
            validation,
            path,
            table,
            columns,
        )

        if (
            league == "ncaam"
            and table == "games"
        ):
            validate_ncaam_venue_context(
                validation,
                path,
                columns,
            )

        if table == "lineups":
            validate_lineups(
                validation,
                path,
                league,
                columns,
            )

        actual_tables[
            table
        ] = {
            "rows": row_count,
            "columns": columns,
            "source_loader": (
                source_loader
            ),
        }

    games_path = (
        season_dir
        / "games.parquet"
    )

    if (
        "games"
        in actual_tables
        and "game_id"
        in actual_tables[
            "games"
        ]["columns"]
    ):
        try:
            game_ids = (
                pl.scan_parquet(
                    games_path
                )
                .select(
                    pl.col("game_id")
                    .cast(
                        pl.String,
                        strict=False,
                    )
                    .str.strip_chars()
                    .alias("game_id")
                )
                .collect()
                .get_column("game_id")
            )

            null_count = (
                game_ids.null_count()
            )

            non_null_ids = (
                game_ids.drop_nulls()
            )

            blank_count = int(
                (
                    non_null_ids
                    == ""
                ).sum()
            )

            non_blank_ids = (
                non_null_ids.filter(
                    non_null_ids
                    != ""
                )
            )

            unique_count = int(
                non_blank_ids.n_unique()
            )

            if (
                null_count
                or blank_count
            ):
                validation.fail(
                    f"{games_path} "
                    "game_id "
                    f"nulls={null_count} "
                    f"blanks={blank_count}"
                )

            if (
                unique_count
                != len(
                    non_blank_ids
                )
            ):
                duplicate_count = (
                    len(
                        non_blank_ids
                    )
                    - unique_count
                )

                validation.fail(
                    f"{games_path} "
                    "duplicate canonical "
                    "game_id rows="
                    f"{duplicate_count}"
                )

            else:
                validation.passed(
                    f"{games_path} "
                    "canonical game_id "
                    f"unique count={unique_count}"
                )

            canonical_ids = set(
                non_blank_ids.to_list()
            )

            actual_tables[
                "games"
            ][
                "unique_games"
            ] = unique_count

            for table in GAME_KEY_TABLES:
                table_info = (
                    actual_tables.get(
                        table
                    )
                )

                if table_info is None:
                    continue

                child_path = (
                    season_dir
                    / f"{table}.parquet"
                )

                if (
                    "game_id"
                    not in table_info[
                        "columns"
                    ]
                ):
                    validation.fail(
                        f"{child_path} "
                        "missing game_id "
                        "required for integrity"
                    )
                    continue

                child_ids = unique_game_ids(
                    child_path
                )

                orphan_ids = sorted(
                    child_ids
                    - canonical_ids
                )

                if orphan_ids:
                    validation.fail(
                        f"{child_path} has "
                        f"{len(orphan_ids)} "
                        "game_ids absent from "
                        "games.parquet "
                        f"sample={orphan_ids[:20]}"
                    )

                else:
                    validation.passed(
                        f"{child_path} "
                        "game_id integrity "
                        "passed "
                        "unique_games="
                        f"{len(child_ids)}"
                    )

                table_info[
                    "unique_games"
                ] = len(child_ids)

        except Exception as exc:
            validation.fail(
                f"{league}:{internal_season} "
                "game-id integrity failed: "
                f"{exc}"
            )

    manifest_path = (
        season_dir
        / "manifest.json"
    )

    validate_manifest(
        validation,
        manifest_path,
        league,
        internal_season,
        sdv_season,
        actual_tables,
    )

    validate_truncation_limits(
        validation,
        limits,
        league,
        internal_season,
        actual_tables,
    )

    validation.seasons_checked += 1


def main() -> int:
    validation = Validation()

    validation.emit(
        "=== SDV STORAGE VALIDATION "
        f"{datetime.now(timezone.utc).isoformat()} "
        "==="
    )

    validation.emit(
        f"storage_config={STORAGE_CONFIG}"
    )

    validation.emit(
        f"season_config={SEASON_CONFIG}"
    )

    validation.emit(
        f"validation_log={VALIDATION_LOG}"
    )

    try:
        storage_cfg = read_yaml(
            STORAGE_CONFIG
        )

        season_cfg = read_yaml(
            SEASON_CONFIG
        )

    except Exception as exc:
        validation.fail(
            "Configuration load failed: "
            f"{exc}"
        )

        validation.emit(
            "STATUS: FAILED"
        )

        validation.write_log()

        print(
            "SDV storage validation "
            f"FAILED: {exc}"
        )

        return 1

    if (
        storage_cfg.get(
            "schema_version"
        )
        == 1
    ):
        validation.passed(
            "sdv_storage.yaml "
            "schema_version=1"
        )

    else:
        validation.fail(
            "sdv_storage.yaml "
            "schema_version must be 1"
        )

    if (
        season_cfg.get(
            "schema_version"
        )
        == 1
    ):
        validation.passed(
            "sdv_seasons.yaml "
            "schema_version=1"
        )

    else:
        validation.fail(
            "sdv_seasons.yaml "
            "schema_version must be 1"
        )

    sdv_section = (
        storage_cfg.get(
            "sportsdataverse"
        )
        or {}
    )

    if not isinstance(
        sdv_section,
        dict,
    ):
        sdv_section = {}

    configured_version = clean(
        sdv_section.get(
            "expected_version"
        )
    )

    if (
        configured_version
        == EXPECTED_SDV_VERSION
    ):
        validation.passed(
            "Configured "
            "SportsDataVerse version="
            f"{configured_version}"
        )

    else:
        validation.fail(
            "Configured "
            "SportsDataVerse version="
            f"{configured_version!r}; "
            "expected="
            f"{EXPECTED_SDV_VERSION}"
        )

    try:
        installed_version = (
            importlib.metadata.version(
                "sportsdataverse"
            )
        )

    except (
        importlib.metadata.PackageNotFoundError
    ):
        installed_version = (
            "NOT_INSTALLED"
        )

    if (
        installed_version
        == EXPECTED_SDV_VERSION
    ):
        validation.passed(
            "Installed "
            "SportsDataVerse version="
            f"{installed_version}"
        )

    else:
        validation.fail(
            "Installed "
            "SportsDataVerse version="
            f"{installed_version!r}; "
            "expected="
            f"{EXPECTED_SDV_VERSION}"
        )

    configured_tables = (
        storage_cfg.get("tables")
    )

    if (
        configured_tables
        == list(TABLES)
    ):
        validation.passed(
            "Configured required "
            "table list matches "
            "validator"
        )

    else:
        validation.fail(
            "Configured tables="
            f"{configured_tables!r}; "
            "expected="
            f"{list(TABLES)!r}"
        )

    configured_limits = (
        collect_configured_limits(
            storage_cfg
        )
    )

    validation.emit(
        "INFO | "
        "configured_truncation_limits="
        f"{configured_limits}"
    )

    try:
        jobs = configured_jobs(
            storage_cfg
        )

    except Exception as exc:
        jobs = []

        validation.fail(
            "Historical season "
            "configuration invalid: "
            f"{exc}"
        )

    validation.emit(
        "INFO | "
        f"configured_jobs={jobs}"
    )

    for (
        league,
        internal_season,
    ) in jobs:
        try:
            validate_season(
                validation,
                storage_cfg,
                season_cfg,
                league,
                internal_season,
                configured_limits,
            )

        except Exception as exc:
            validation.fail(
                "Unexpected validation "
                "error "
                f"{league}:"
                f"{internal_season}: "
                f"{exc}"
            )

    validation.emit(
        "SUMMARY | "
        "seasons_checked="
        f"{validation.seasons_checked} "
        "tables_checked="
        f"{validation.tables_checked} "
        "warnings="
        f"{len(validation.warnings)} "
        "errors="
        f"{len(validation.errors)}"
    )

    if validation.errors:
        validation.emit(
            "STATUS: FAILED"
        )

        validation.write_log()

        print(
            "SDV storage validation "
            "FAILED: "
            f"{len(validation.errors)} "
            "error(s). See "
            f"{VALIDATION_LOG}"
        )

        return 1

    validation.emit(
        "STATUS: SUCCESS"
    )

    validation.write_log()

    print(
        "SDV storage validation "
        "complete: SUCCESS. See "
        f"{VALIDATION_LOG}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )