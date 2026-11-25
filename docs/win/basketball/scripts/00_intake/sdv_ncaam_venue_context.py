#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/sdv_ncaam_venue_context.py
"""Preserve and validate explicit NCAAM venue context in SDV history."""
from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml

BASE = Path("docs/win/basketball")
STORAGE_CONFIG = BASE / "config/sdv_storage.yaml"
ERROR_DIR = BASE / "errors/00_intake"
LOG_FILE = ERROR_DIR / "sdv_ncaam_venue_context.txt"

REQUIRED_SOURCE_COLUMNS = (
    "game_id",
    "neutral_site",
    "venue_id",
    "venue_full_name",
    "home_id",
    "away_id",
)

REQUIRED_OUTPUT_COLUMNS = (
    "game_id",
    "neutral_site",
    "venue_id",
    "venue_full_name",
    "venue_name",
    "home_team_id",
    "away_team_id",
)


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def log(message: str) -> None:
    ERROR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            f"{datetime.now(timezone.utc).isoformat()} | "
            f"{message}\n"
        )


def read_yaml(path: Path) -> dict[str, Any]:
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
            f"YAML root must be a mapping: {path}"
        )

    return payload


def storage_root(cfg: dict[str, Any]) -> Path:
    storage = cfg.get("storage")

    if not isinstance(storage, dict):
        raise ValueError(
            "storage section missing from sdv_storage.yaml"
        )

    root = clean(
        storage.get("root")
    )

    if not root:
        raise ValueError(
            "storage.root is empty"
        )

    return Path(root)


def configured_ncaam_seasons(
    cfg: dict[str, Any],
) -> list[int]:
    section = cfg.get(
        "historical_internal_seasons"
    )

    if not isinstance(section, dict):
        raise ValueError(
            "historical_internal_seasons must be a mapping"
        )

    seasons = section.get("ncaam")

    if (
        not isinstance(seasons, list)
        or not seasons
    ):
        raise ValueError(
            "historical_internal_seasons.ncaam "
            "must be a non-empty list"
        )

    return sorted(
        {
            int(season)
            for season in seasons
        }
    )


def resolve_seasons(
    cfg: dict[str, Any],
    requested: list[int] | None,
) -> list[int]:
    configured = configured_ncaam_seasons(cfg)

    if not requested:
        return configured

    requested_set = {
        int(season)
        for season in requested
    }

    invalid = sorted(
        requested_set
        - set(configured)
    )

    if invalid:
        raise ValueError(
            "Requested NCAAM internal season "
            "is not configured as historical: "
            f"{invalid}"
        )

    return [
        season
        for season in configured
        if season in requested_set
    ]


def normalize_venue_name_expr() -> pl.Expr:
    source = (
        pl.col("venue_full_name")
        .cast(
            pl.String,
            strict=False,
        )
    )

    return (
        pl.when(
            source.is_null()
        )
        .then(
            pl.lit(
                None,
                dtype=pl.String,
            )
        )
        .otherwise(
            source
            .str.strip_chars()
            .str.replace_all(
                r"\s+",
                " ",
            )
        )
        .alias("venue_name")
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


def require_columns(
    frame: pl.DataFrame,
    required: tuple[str, ...],
    *,
    path: Path,
    label: str,
) -> None:
    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            f"{path} missing {label} columns "
            f"{missing}"
        )


def validate_neutral_site(
    frame: pl.DataFrame,
    path: Path,
) -> dict[str, int]:
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
        frame.lazy()
        .select(
            [
                pl.col("neutral_site")
                .null_count()
                .alias("nulls"),
                neutral_text
                .eq("")
                .sum()
                .alias("blanks"),
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
                .alias("invalid"),
            ]
        )
        .collect()
        .to_dicts()[0]
    )

    if stats["invalid"]:
        raise RuntimeError(
            f"{path} neutral_site contains "
            "unrecognized nonblank values: "
            f"invalid={stats['invalid']}"
        )

    return {
        key: int(value)
        for key, value in stats.items()
    }


def validate_team_ids(
    frame: pl.DataFrame,
    path: Path,
) -> None:
    stats = (
        frame.lazy()
        .select(
            [
                normalize_id_expr("home_id")
                .eq("")
                .sum()
                .alias("home_source_blank"),
                normalize_id_expr("away_id")
                .eq("")
                .sum()
                .alias("away_source_blank"),
                (
                    normalize_id_expr("home_id")
                    != normalize_id_expr(
                        "home_team_id"
                    )
                )
                .sum()
                .alias("home_mismatch"),
                (
                    normalize_id_expr("away_id")
                    != normalize_id_expr(
                        "away_team_id"
                    )
                )
                .sum()
                .alias("away_mismatch"),
            ]
        )
        .collect()
        .to_dicts()[0]
    )

    if (
        stats["home_source_blank"]
        or stats["away_source_blank"]
    ):
        raise RuntimeError(
            f"{path} source team IDs contain blanks: "
            f"home={stats['home_source_blank']} "
            f"away={stats['away_source_blank']}"
        )

    if (
        stats["home_mismatch"]
        or stats["away_mismatch"]
    ):
        raise RuntimeError(
            f"{path} normalized team IDs do not "
            "preserve source IDs: "
            f"home_mismatch={stats['home_mismatch']} "
            f"away_mismatch={stats['away_mismatch']}"
        )


def validate_venue_name(
    frame: pl.DataFrame,
    path: Path,
) -> dict[str, int]:
    expected = (
        pl.col("venue_full_name")
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

    actual = (
        pl.col("venue_name")
        .cast(
            pl.String,
            strict=False,
        )
        .fill_null("")
        .str.strip_chars()
    )

    stats = (
        frame.lazy()
        .select(
            [
                (
                    expected != actual
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
                expected
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

    if stats["venue_name_mismatch"]:
        raise RuntimeError(
            f"{path} venue_name is not "
            "normalized venue_full_name for "
            f"{stats['venue_name_mismatch']} rows"
        )

    return {
        key: int(value)
        for key, value in stats.items()
    }


def validate_frame(
    frame: pl.DataFrame,
    path: Path,
) -> dict[str, int]:
    require_columns(
        frame,
        REQUIRED_SOURCE_COLUMNS,
        path=path,
        label="source venue/context",
    )

    require_columns(
        frame,
        REQUIRED_OUTPUT_COLUMNS,
        path=path,
        label="output venue/context",
    )

    if frame.height <= 0:
        raise RuntimeError(
            f"{path} has zero rows"
        )

    neutral_stats = validate_neutral_site(
        frame,
        path,
    )

    validate_team_ids(
        frame,
        path,
    )

    venue_stats = validate_venue_name(
        frame,
        path,
    )

    blank_game_ids = int(
        frame.select(
            pl.col("game_id")
            .cast(
                pl.String,
                strict=False,
            )
            .fill_null("")
            .str.strip_chars()
            .eq("")
            .sum()
        )
        .item()
    )

    if blank_game_ids:
        raise RuntimeError(
            f"{path} game_id contains null or "
            f"blank values: rows={blank_game_ids}"
        )

    if venue_stats["venue_id_present"] <= 0:
        raise RuntimeError(
            f"{path} has no game with a venue_id: "
            f"rows={frame.height}"
        )

    unique_games = int(
        frame.select(
            pl.col("game_id")
            .cast(
                pl.String,
                strict=False,
            )
            .str.strip_chars()
            .n_unique()
            .alias("unique_games")
        )
        .item()
    )

    if unique_games != frame.height:
        raise RuntimeError(
            f"{path} game_id is not unique: "
            f"rows={frame.height} "
            f"unique_games={unique_games}"
        )

    return {
        "rows": int(frame.height),
        "unique_games": unique_games,
        "neutral_site_nulls": (
            neutral_stats["nulls"]
        ),
        "neutral_site_blanks": (
            neutral_stats["blanks"]
        ),
        **venue_stats,
    }


def enrich_frame(
    frame: pl.DataFrame,
    path: Path,
) -> pl.DataFrame:
    require_columns(
        frame,
        REQUIRED_SOURCE_COLUMNS,
        path=path,
        label="source venue/context",
    )

    enriched = frame.with_columns(
        [
            pl.col("home_id").alias(
                "home_team_id"
            ),
            pl.col("away_id").alias(
                "away_team_id"
            ),
            normalize_venue_name_expr(),
        ]
    )

    validate_frame(
        enriched,
        path,
    )

    return enriched


def write_parquet_atomic(
    path: Path,
    frame: pl.DataFrame,
) -> None:
    tmp = Path(
        f"{path}.tmp"
    )

    if tmp.exists():
        tmp.unlink()

    try:
        frame.write_parquet(
            tmp,
            compression="zstd",
        )
        tmp.replace(path)

    finally:
        if tmp.exists():
            tmp.unlink()


def load_manifest(
    manifest_path: Path,
) -> dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            manifest_path
        )

    payload = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"{manifest_path} root is not an object"
        )

    if (
        clean(payload.get("league")).upper()
        != "NCAAM"
    ):
        raise RuntimeError(
            f"{manifest_path} is not an NCAAM manifest"
        )

    tables = payload.get("tables")

    if not isinstance(tables, dict):
        raise RuntimeError(
            f"{manifest_path} missing tables object"
        )

    games = tables.get("games")

    if not isinstance(games, dict):
        raise RuntimeError(
            f"{manifest_path} missing games table entry"
        )

    return payload


def update_manifest(
    manifest_path: Path,
    *,
    games_path: Path,
    frame: pl.DataFrame,
) -> None:
    payload = load_manifest(
        manifest_path
    )

    games = payload["tables"]["games"]

    games["rows"] = int(frame.height)
    games["columns"] = list(frame.columns)
    games["filename"] = "games.parquet"
    games["file"] = str(games_path)

    payload["generated_at_utc"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    payload["ncaam_venue_context"] = {
        "status": "ready",
        "script": (
            "docs/win/basketball/scripts/"
            "00_intake/sdv_ncaam_venue_context.py"
        ),
        "fields": list(
            REQUIRED_OUTPUT_COLUMNS[1:]
        ),
        "venue_name_definition": (
            "venue_full_name trimmed with "
            "internal whitespace collapsed"
        ),
        "home_team_id_definition": (
            "preserved from source home_id"
        ),
        "away_team_id_definition": (
            "preserved from source away_id"
        ),
        "neutral_site_definition": (
            "preserved from SDV schedule; "
            "not inferred from home_team"
        ),
    }

    tmp = Path(
        f"{manifest_path}.tmp"
    )

    if tmp.exists():
        tmp.unlink()

    try:
        tmp.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        tmp.replace(manifest_path)

    finally:
        if tmp.exists():
            tmp.unlink()


def process_season(
    root: Path,
    internal_season: int,
    *,
    validate_only: bool,
) -> dict[str, int]:
    season_dir = (
        root
        / "ncaam"
        / str(internal_season)
    )

    games_path = (
        season_dir
        / "games.parquet"
    )

    manifest_path = (
        season_dir
        / "manifest.json"
    )

    if not games_path.exists():
        raise FileNotFoundError(
            games_path
        )

    frame = pl.read_parquet(
        games_path
    )

    if validate_only:
        stats = validate_frame(
            frame,
            games_path,
        )

        log(
            "VALID | "
            f"internal_season={internal_season} "
            f"rows={stats['rows']} "
            f"venue_id_present="
            f"{stats['venue_id_present']} "
            f"venue_name_present="
            f"{stats['venue_name_present']}"
        )

        return stats

    enriched = enrich_frame(
        frame,
        games_path,
    )

    load_manifest(
        manifest_path
    )

    original_bytes = games_path.read_bytes()

    write_parquet_atomic(
        games_path,
        enriched,
    )

    try:
        update_manifest(
            manifest_path,
            games_path=games_path,
            frame=enriched,
        )

    except Exception:
        games_path.write_bytes(
            original_bytes
        )

        log(
            "ROLLBACK | games.parquet restored "
            f"internal_season={internal_season}"
        )

        raise

    stored = pl.read_parquet(
        games_path
    )

    stats = validate_frame(
        stored,
        games_path,
    )

    log(
        "UPDATED | "
        f"internal_season={internal_season} "
        f"rows={stats['rows']} "
        f"venue_id_present="
        f"{stats['venue_id_present']} "
        f"venue_name_present="
        f"{stats['venue_name_present']}"
    )

    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preserve explicit NCAAM venue context "
            "in historical SDV games.parquet files."
        )
    )

    parser.add_argument(
        "--internal-season",
        action="append",
        type=int,
        help=(
            "Configured NCAAM internal season. "
            "May be repeated. Default: all configured "
            "historical NCAAM seasons."
        ),
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate stored NCAAM venue context "
            "without rewriting parquet or manifest."
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=STORAGE_CONFIG,
        help="Path to sdv_storage.yaml.",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    ERROR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_FILE.write_text(
        (
            "=== SDV NCAAM VENUE CONTEXT "
            f"{datetime.now(timezone.utc).isoformat()} "
            "===\n"
        ),
        encoding="utf-8",
    )

    try:
        cfg = read_yaml(
            args.config
        )

        root = storage_root(cfg)

        seasons = resolve_seasons(
            cfg,
            args.internal_season,
        )

        log(
            "CONFIG VALID | "
            f"root={root} "
            f"seasons={seasons} "
            f"validate_only={args.validate_only}"
        )

        for season in seasons:
            process_season(
                root,
                season,
                validate_only=args.validate_only,
            )

        log(
            "STATUS: SUCCESS | "
            f"seasons={len(seasons)}"
        )

        print(
            "SDV NCAAM venue context "
            "complete: SUCCESS."
        )

        return 0

    except Exception as exc:
        log(
            f"FATAL: {exc}"
        )
        log(
            traceback.format_exc().rstrip()
        )
        log(
            "STATUS: FAILED"
        )

        print(
            "SDV NCAAM venue context "
            f"FAILED: {exc}"
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )