#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/sdv_season_mapping.py
#
# Central SportsDataVerse season-ID translation.
#
# Reads:
#   docs/win/basketball/config/sdv_seasons.yaml
#
# Purpose:
# - Prevent internal basketball season labels from being passed directly to SDV.
# - Require an explicit league-specific internal-season -> SDV-season mapping.
# - Fail closed when a league or season has not been configured.
#
# Examples:
#   internal NBA season 2025 -> SDV NBA season 2026
#   label 2025_NBA         -> SDV NBA season 2026

from __future__ import annotations

import argparse
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


BASE = Path("docs/win/basketball")

CONFIG_PATH = BASE / "config/sdv_seasons.yaml"

LOG_PATH = (
    BASE
    / "errors/00_intake/sdv_season_mapping.txt"
)


def _utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def log(message: str) -> None:
    try:
        LOG_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with LOG_PATH.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                f"{_utc_now()} | {message}\n"
            )

    except OSError as exc:
        print(
            f"sdv_season_mapping log write failed: {exc}",
            file=sys.stderr,
        )


SUPPORTED_LEAGUES = {
    "nba": "NBA",
    "ncaam": "NCAAM",
    "wnba": "WNBA",
}

INTERNAL_LABEL_RE = re.compile(
    r"^(?P<season>\d{4})_(?P<league>NBA|NCAAM|WNBA)$",
    re.IGNORECASE,
)


def normalize_league(value: Any) -> str:
    league = (
        ""
        if value is None
        else str(value).strip().lower()
    )

    if league not in SUPPORTED_LEAGUES:
        raise ValueError(
            f"Unsupported league {value!r}; "
            f"expected one of: "
            f"{', '.join(sorted(SUPPORTED_LEAGUES))}"
        )

    return league


def normalize_internal_season(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"Invalid internal season {value!r}"
        )

    text = (
        ""
        if value is None
        else str(value).strip()
    )

    if not re.fullmatch(r"\d{4}", text):
        raise ValueError(
            f"Invalid internal season {value!r}; "
            f"expected a four-digit year"
        )

    return int(text)


def load_config(
    path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing SDV season mapping config: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(
            f"{path} must contain a top-level mapping"
        )

    if config.get("schema_version") != 1:
        raise ValueError(
            f"{path} schema_version must be 1"
        )

    if (
        str(
            config.get(
                "mapping_policy",
                "",
            )
        )
        .strip()
        .lower()
        != "explicit_only"
    ):
        raise ValueError(
            f"{path} mapping_policy must be "
            f"'explicit_only'"
        )

    leagues = config.get("leagues")

    if not isinstance(leagues, dict):
        raise ValueError(
            f"{path} must contain a top-level "
            f"'leagues' mapping"
        )

    return config


def validate_league_config(
    config: dict[str, Any],
    league: str,
) -> dict[str, Any]:
    key = normalize_league(league)

    leagues = config.get("leagues") or {}
    league_cfg = leagues.get(key)

    if not isinstance(league_cfg, dict):
        raise ValueError(
            f"Missing SDV season mapping config "
            f"for league={key}"
        )

    status = (
        str(
            league_cfg.get(
                "status",
                "",
            )
        )
        .strip()
        .lower()
    )

    if status != "active":
        raise ValueError(
            f"SDV season mapping for league={key} "
            f"is not active; status={status!r}"
        )

    mappings = league_cfg.get("mappings")

    if not isinstance(mappings, dict):
        raise ValueError(
            f"SDV season mappings for league={key} "
            f"must be a mapping"
        )

    if not mappings:
        raise ValueError(
            f"SDV season mappings for league={key} "
            f"are empty"
        )

    normalized: dict[int, int] = {}

    for raw_internal, raw_sdv in mappings.items():
        internal = normalize_internal_season(
            raw_internal
        )
        sdv = normalize_internal_season(
            raw_sdv
        )

        if internal in normalized:
            raise ValueError(
                f"Duplicate internal season mapping "
                f"for league={key} season={internal}"
            )

        normalized[internal] = sdv

    result = dict(league_cfg)
    result["mappings"] = normalized

    return result


def sdv_season_id(
    league: str,
    internal_season: Any,
    config_path: Path = CONFIG_PATH,
) -> int:
    try:
        key = normalize_league(league)
        internal = normalize_internal_season(
            internal_season
        )

        config = load_config(
            config_path
        )

        league_cfg = validate_league_config(
            config,
            key,
        )

        mappings: dict[int, int] = (
            league_cfg["mappings"]
        )

        if internal not in mappings:
            raise ValueError(
                f"No explicit SDV season mapping for "
                f"league={key} internal_season={internal}. "
                f"Add it to {config_path} before querying SDV."
            )

        return mappings[internal]

    except Exception as exc:
        log(
            "ERROR | "
            f"league={league!r} "
            f"internal_season={internal_season!r} | "
            f"{exc}"
        )
        raise


def parse_internal_season_label(
    label: str,
) -> tuple[str, int]:
    text = str(label).strip()

    match = INTERNAL_LABEL_RE.fullmatch(
        text
    )

    if match is None:
        raise ValueError(
            f"Invalid internal season label {label!r}; "
            f"expected YYYY_NBA, YYYY_NCAAM, or YYYY_WNBA"
        )

    league = normalize_league(
        match.group("league")
    )

    internal = normalize_internal_season(
        match.group("season")
    )

    return league, internal


def sdv_season_from_label(
    label: str,
    config_path: Path = CONFIG_PATH,
) -> int:
    try:
        league, internal = (
            parse_internal_season_label(
                label
            )
        )

    except ValueError as exc:
        log(
            f"ERROR | label={label!r} | {exc}"
        )
        raise

    return sdv_season_id(
        league,
        internal,
        config_path=config_path,
    )


def sdv_season_from_filename(
    path: str | Path,
    config_path: Path = CONFIG_PATH,
) -> int:
    filename = Path(path).name

    stem = filename

    if stem.lower().endswith(".csv"):
        stem = stem[:-4]

    return sdv_season_from_label(
        stem,
        config_path=config_path,
    )


def validate_all_active_mappings(
    config_path: Path = CONFIG_PATH,
) -> None:
    try:
        config = load_config(
            config_path
        )

        leagues = config.get("leagues") or {}

        for league, league_cfg in leagues.items():
            if not isinstance(
                league_cfg,
                dict,
            ):
                raise ValueError(
                    f"Invalid league config for {league!r}"
                )

            status = (
                str(
                    league_cfg.get(
                        "status",
                        "",
                    )
                )
                .strip()
                .lower()
            )

            if status == "active":
                validate_league_config(
                    config,
                    league,
                )

    except Exception as exc:
        log(
            f"ERROR | validate | {exc}"
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Translate internal basketball season IDs "
            "to explicit SportsDataVerse season IDs."
        )
    )

    group = parser.add_mutually_exclusive_group(
        required=False
    )

    group.add_argument(
        "--label",
        help=(
            "Internal season label such as 2025_NBA"
        ),
    )

    group.add_argument(
        "--file",
        help=(
            "Internal season filename such as "
            "2025_NBA.csv"
        ),
    )

    parser.add_argument(
        "--league",
        choices=sorted(
            SUPPORTED_LEAGUES
        ),
        help=(
            "League when using --internal-season"
        ),
    )

    parser.add_argument(
        "--internal-season",
        help=(
            "Internal four-digit season start/calendar year"
        ),
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Validate all active mappings and exit"
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help=(
            f"Mapping config path "
            f"(default: {CONFIG_PATH})"
        ),
    )

    return parser


def main() -> int:
    parser = build_parser()

    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_PATH.write_text(
        "=== SDV SEASON MAPPING "
        f"{_utc_now()} ===\n",
        encoding="utf-8",
    )

    try:
        args = parser.parse_args()

    except SystemExit as exc:
        if exc.code in (0, None):
            log("STATUS: SUCCESS")

        else:
            log(
                "FATAL: invalid command-line arguments"
            )
            log("STATUS: FAILED")

        raise

    try:
        if args.validate:
            validate_all_active_mappings(
                config_path=args.config
            )
            print("SDV season mappings valid.")
            log(
                "VALIDATE | all active mappings valid"
            )

        elif args.label:
            result = sdv_season_from_label(
                args.label,
                config_path=args.config,
            )
            print(result)
            log(
                f"LABEL | {args.label} -> {result}"
            )

        elif args.file:
            result = sdv_season_from_filename(
                args.file,
                config_path=args.config,
            )
            print(result)
            log(
                f"FILE | {args.file} -> {result}"
            )

        elif (
            args.league
            and args.internal_season
        ):
            result = sdv_season_id(
                args.league,
                args.internal_season,
                config_path=args.config,
            )
            print(result)
            log(
                f"SEASON | {args.league} "
                f"{args.internal_season} -> {result}"
            )

        else:
            log(
                "FATAL: Provide --validate, --label, "
                "--file, or both --league and "
                "--internal-season"
            )
            log("STATUS: FAILED")

            parser.error(
                "Provide --validate, --label, --file, "
                "or both --league and --internal-season"
            )

    except Exception as exc:
        log(f"FATAL: {exc}")
        log(traceback.format_exc().rstrip())
        log("STATUS: FAILED")

        print(
            f"SDV season mapping FAILED: {exc}",
            file=sys.stderr,
        )

        return 1

    log("STATUS: SUCCESS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
