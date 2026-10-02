#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/sdv_predict.py
"""Production SDV Model V1 inference.

Authoritative slates:
    docs/win/basketball/daily_games/{league}/{game_date}_{LEAGUE}.csv

Feature implementation reused from:
    docs/win/basketball/scripts/00_intake/sdv_feature_generation.py

Model artifacts:
    docs/win/basketball/models/sdv/{league}/

Outputs:
    docs/win/basketball/00_intake/predictions_sdv/{league}/
        {game_date}_{LEAGUE}_predictions.csv

Default behavior:
- scan every available daily-games file for the selected league;
- create an SDV prediction file for every valid non-empty slate;
- skip empty slates;
- if one date fails, log that date and continue processing the others.

Optional --game-date remains available only as a filter for an explicit
single-date rerun. It is not required.

Fail-closed guarantees for every date that is processed:
- every daily slate row must have canonical game_id;
- every daily slate game must receive one feature row;
- every prediction must retain the daily canonical game_id;
- feature/model versions must match exactly;
- model/schema/coefficient ordering must match;
- no game inside a processed slate is silently dropped.

The daily_games file is authoritative for slate membership and identity.
SportsDataVerse schedule information is used only to enrich the daily
slate with team IDs, venue, neutral-site, and other feature context.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import sdv_canonical_games
import sdv_feature_generation as feature_generation
from sdv_season_mapping import sdv_season_id


BASE = Path("docs/win/basketball")
CONFIG_PATH = BASE / "config/sdv_model.yaml"
DAILY_GAMES_ROOT = BASE / "daily_games"
DEFAULT_MODEL_ROOT = BASE / "models/sdv"
PREDICTION_ROOT = BASE / "00_intake/predictions_sdv"
ERROR_DIR = BASE / "errors/00_intake"
LOG_FILE = ERROR_DIR / "sdv_predict.txt"

LEAGUE_LABELS = {
    "nba": "NBA",
    "ncaam": "NCAAM",
    "wnba": "WNBA",
}

REQUIRED_ARTIFACT_FILES = (
    "margin_model.json",
    "total_model.json",
    "feature_schema.json",
    "metadata.json",
)

PREDICTION_FIELDS = [
    "sport",
    "league",
    "game_id",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "model_source",
    "model_version",
    "feature_version",
    "home_prob",
    "away_prob",
    "raw_home_ml_prob",
    "raw_away_ml_prob",
    "home_projected_points",
    "away_projected_points",
    "total_projected_points",
    "expected_margin",
    "expected_total",
    "margin_residual_mean",
    "margin_residual_std",
    "total_residual_mean",
    "total_residual_std",
    "feature_generated_at_utc",
    "prediction_generated_at_utc",
]

PROBABILITY_EPSILON = 1e-12


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


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


def normalize_game_date(value: Any) -> str:
    text = clean(value)

    if not text:
        return ""

    normalized = text[:10].replace("-", "_").replace("/", "_")

    if not re.fullmatch(r"\d{4}_\d{2}_\d{2}", normalized):
        return ""

    try:
        datetime.strptime(normalized, "%Y_%m_%d")

    except ValueError:
        return ""

    return normalized


def normalize_team_key(value: Any) -> str:
    return " ".join(clean(value).casefold().split())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)

    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()} | {message}\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"JSON root must be object: {path}"
        )

    return payload


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        result = [
            dict(row)
            for row in csv.DictReader(handle)
        ]
    return result


def required_mapping(
    parent: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    value = parent.get(key)

    if not isinstance(value, dict):
        raise ValueError(
            f"Missing mapping: {key}"
        )

    return value


def to_float_required(
    value: Any,
    label: str,
) -> float:
    if isinstance(value, bool):
        result = float(value)

    else:
        text = clean(value)

        if not text:
            raise RuntimeError(
                f"{label} is blank"
            )

        try:
            result = float(text)

        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{label} is not numeric: {value!r}"
            ) from exc

    if not math.isfinite(result):
        raise RuntimeError(
            f"{label} is not finite"
        )

    return result


def configured_model_root(
    cfg: dict[str, Any],
) -> Path:
    artifact_cfg = required_mapping(
        cfg,
        "artifacts",
    )

    root = clean(
        artifact_cfg.get("root")
    )

    if not root:
        raise ValueError(
            "artifacts.root is blank"
        )

    result = Path(root)

    if (
        result.as_posix()
        != DEFAULT_MODEL_ROOT.as_posix()
    ):
        raise ValueError(
            "SDV model root mismatch: "
            f"configured={result} "
            f"expected={DEFAULT_MODEL_ROOT}"
        )

    if not bool(
        artifact_cfg.get(
            "require_feature_version_match",
            False,
        )
    ):
        raise ValueError(
            "artifacts."
            "require_feature_version_match "
            "must be true"
        )

    if not bool(
        artifact_cfg.get(
            "require_model_version_match",
            False,
        )
    ):
        raise ValueError(
            "artifacts."
            "require_model_version_match "
            "must be true"
        )

    return result


def discover_daily_game_dates(
    league: str,
) -> list[str]:
    label = LEAGUE_LABELS[league]

    folder = (
        DAILY_GAMES_ROOT
        / league
    )

    if not folder.exists():
        return []

    dates: list[str] = []

    pattern = re.compile(
        rf"^(\d{{4}}_\d{{2}}_\d{{2}})_"
        rf"{re.escape(label)}\.csv$"
    )

    for path in sorted(
        folder.glob(
            f"*_{label}.csv"
        )
    ):
        match = pattern.fullmatch(
            path.name
        )

        if not match:
            continue

        game_date = normalize_game_date(
            match.group(1)
        )

        if game_date:
            dates.append(game_date)

    return sorted(
        set(dates)
    )


def load_daily_slate(
    league: str,
    game_date: str,
) -> tuple[
    Path,
    list[dict[str, Any]],
]:
    label = LEAGUE_LABELS[league]

    path = (
        DAILY_GAMES_ROOT
        / league
        / f"{game_date}_{label}.csv"
    )

    rows = read_csv_rows(path)

    if not rows:
        return path, []

    seen_ids: set[str] = set()

    for index, row in enumerate(
        rows,
        start=1,
    ):
        row_date = normalize_game_date(
            row.get("game_date")
        )

        if row_date != game_date:
            raise RuntimeError(
                f"{path}: row={index} "
                "game_date mismatch "
                f"expected={game_date} "
                f"actual={row_date}"
            )

        game_id = clean_id(
            row.get("game_id")
        )

        if not game_id:
            raise RuntimeError(
                f"{path}: row={index} "
                "has blank canonical game_id"
            )

        if game_id in seen_ids:
            raise RuntimeError(
                f"{path}: duplicate "
                f"game_id={game_id}"
            )

        seen_ids.add(game_id)

        home_team = clean(
            row.get("home_team")
        )

        away_team = clean(
            row.get("away_team")
        )

        if not home_team or not away_team:
            raise RuntimeError(
                f"{path}: game_id={game_id} "
                "has blank home/away team"
            )

        row["game_id"] = game_id
        row["game_date"] = game_date

    return path, rows


def internal_season_from_game_date(
    league: str,
    game_date: str,
) -> int:
    parsed = datetime.strptime(
        game_date,
        "%Y_%m_%d",
    )

    if league == "wnba":
        return parsed.year

    if league in {
        "nba",
        "ncaam",
    }:
        if parsed.month >= 10:
            return parsed.year

        return parsed.year - 1

    raise ValueError(
        f"Unsupported league={league}"
    )


def canonical_season_path(
    cfg: dict[str, Any],
    league: str,
    internal_season: int,
) -> Path:
    paths = (
        feature_generation
        .configured_paths(cfg)
    )

    label = LEAGUE_LABELS[league]

    return (
        paths["canonical_current_root"]
        / league
        / (
            f"{internal_season}_"
            f"{label}_games.csv"
        )
    )


def load_cached_canonical_rows(
    cfg: dict[str, Any],
    league: str,
    internal_season: int,
) -> list[dict[str, Any]]:
    path = canonical_season_path(
        cfg,
        league,
        internal_season,
    )

    if not path.exists():
        log(
            "CACHED CANONICAL MISSING | "
            f"league={LEAGUE_LABELS[league]} "
            f"path={path}"
        )

        return []

    rows = read_csv_rows(path)

    log(
        "CACHED CANONICAL LOADED | "
        f"league={LEAGUE_LABELS[league]} "
        f"internal_season={internal_season} "
        f"rows={len(rows)} "
        f"path={path}"
    )

    return rows


def fetch_date_specific_sdv_rows(
    league: str,
    game_date: str,
    internal_season: int,
    mapped_sdv_season: int,
) -> list[dict[str, Any]]:
    date_query = int(
        game_date.replace("_", "")
    )

    limit = 2000
    fetched_at = utc_now()

    try:
        frame = (
            sdv_canonical_games
            .fetch_schedule(
                league,
                date_query,
                limit,
            )
        )

        if (
            frame is None
            or frame.empty
        ):
            log(
                "DATE SDV EMPTY | "
                f"league={LEAGUE_LABELS[league]} "
                f"game_date={game_date}"
            )

            return []

        sdv_canonical_games.validate_source_schema(
            frame,
            league,
            limit,
        )

        rows = (
            sdv_canonical_games
            .canonicalize_schedule(
                frame,
                league=league,
                internal_season=internal_season,
                sdv_season=mapped_sdv_season,
                fetched_at_utc=fetched_at,
            )
        )

        rows = [
            row
            for row in rows
            if normalize_game_date(
                row.get("game_date")
            )
            == game_date
        ]

        log(
            "DATE SDV READY | "
            f"league={LEAGUE_LABELS[league]} "
            f"game_date={game_date} "
            f"rows={len(rows)}"
        )

        return rows

    except Exception as exc:
        log(
            "DATE SDV WARNING | "
            f"league={LEAGUE_LABELS[league]} "
            f"game_date={game_date} "
            f"error={exc}"
        )

        return []


def add_team_id_candidate(
    mapping: dict[str, set[str]],
    name: Any,
    team_id: Any,
) -> None:
    key = normalize_team_key(name)
    identifier = clean_id(team_id)

    if key and identifier:
        mapping[key].add(identifier)


def add_canonical_team_ids(
    mapping: dict[str, set[str]],
    rows: list[dict[str, Any]],
) -> None:
    for row in rows:
        add_team_id_candidate(
            mapping,
            row.get("home_team"),
            row.get("home_team_id"),
        )

        add_team_id_candidate(
            mapping,
            row.get("away_team"),
            row.get("away_team_id"),
        )


def add_history_team_ids(
    mapping: dict[str, set[str]],
    cfg: dict[str, Any],
    league: str,
) -> None:
    paths = (
        feature_generation
        .configured_paths(cfg)
    )

    history_root = (
        paths["history_input_root"]
    )

    seasons = (
        feature_generation
        .history_seasons(
            history_root,
            league,
        )
    )

    home_name_columns = (
        "home_display_name",
        "home_short_display_name",
        "home_location",
        "home_name",
    )

    away_name_columns = (
        "away_display_name",
        "away_short_display_name",
        "away_location",
        "away_name",
    )

    for season in seasons:
        path = (
            history_root
            / league
            / str(season)
            / "games.parquet"
        )

        if not path.exists():
            continue

        frame = pl.read_parquet(path)
        columns = set(frame.columns)

        home_id_column = (
            "home_team_id"
            if "home_team_id" in columns
            else "home_id"
        )

        away_id_column = (
            "away_team_id"
            if "away_team_id" in columns
            else "away_id"
        )

        selected_columns = [
            column
            for column in (
                *home_name_columns,
                *away_name_columns,
                home_id_column,
                away_id_column,
            )
            if column in columns
        ]

        if not selected_columns:
            continue

        for row in frame.select(
            selected_columns
        ).to_dicts():
            home_id = row.get(
                home_id_column
            )

            away_id = row.get(
                away_id_column
            )

            for column in home_name_columns:
                if column in row:
                    add_team_id_candidate(
                        mapping,
                        row.get(column),
                        home_id,
                    )

            for column in away_name_columns:
                if column in row:
                    add_team_id_candidate(
                        mapping,
                        row.get(column),
                        away_id,
                    )


def build_team_id_map(
    cfg: dict[str, Any],
    league: str,
    cached_rows: list[dict[str, Any]],
    date_rows: list[dict[str, Any]],
) -> dict[str, str]:
    candidates: dict[
        str,
        set[str],
    ] = defaultdict(set)

    add_canonical_team_ids(
        candidates,
        cached_rows,
    )

    add_canonical_team_ids(
        candidates,
        date_rows,
    )

    add_history_team_ids(
        candidates,
        cfg,
        league,
    )

    ambiguous = {
        name: sorted(values)
        for name, values
        in candidates.items()
        if len(values) > 1
    }

    if ambiguous:
        log(
            "TEAM ID AMBIGUITIES | "
            f"league={LEAGUE_LABELS[league]} "
            f"count={len(ambiguous)}"
        )

    return {
        name: next(iter(values))
        for name, values
        in candidates.items()
        if len(values) == 1
    }


def exact_context_by_game_id(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[
        str,
        dict[str, Any],
    ] = {}

    for row in rows:
        game_id = clean_id(
            row.get("game_id")
            or row.get("id")
        )

        if not game_id:
            continue

        if game_id in result:
            raise RuntimeError(
                "Duplicate context "
                f"game_id={game_id}"
            )

        result[game_id] = row

    return result


def assert_team_identity(
    slate_row: dict[str, Any],
    context_row: dict[str, Any],
) -> None:
    game_id = clean_id(
        slate_row.get("game_id")
    )

    slate_home = normalize_team_key(
        slate_row.get("home_team")
    )

    slate_away = normalize_team_key(
        slate_row.get("away_team")
    )

    context_home = normalize_team_key(
        context_row.get("home_team")
    )

    context_away = normalize_team_key(
        context_row.get("away_team")
    )

    if (
        context_home
        and slate_home
        and context_home != slate_home
    ):
        raise RuntimeError(
            "Canonical home-team identity "
            "mismatch for "
            f"game_id={game_id}: "
            f"daily={slate_row.get('home_team')} "
            f"sdv={context_row.get('home_team')}"
        )

    if (
        context_away
        and slate_away
        and context_away != slate_away
    ):
        raise RuntimeError(
            "Canonical away-team identity "
            "mismatch for "
            f"game_id={game_id}: "
            f"daily={slate_row.get('away_team')} "
            f"sdv={context_row.get('away_team')}"
        )


def build_feature_targets(
    cfg: dict[str, Any],
    league: str,
    game_date: str,
    internal_season: int,
    mapped_sdv_season: int,
    slate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cached_rows = (
        load_cached_canonical_rows(
            cfg,
            league,
            internal_season,
        )
    )

    date_rows = (
        fetch_date_specific_sdv_rows(
            league,
            game_date,
            internal_season,
            mapped_sdv_season,
        )
    )

    exact_rows: dict[
        str,
        dict[str, Any],
    ] = {}

    for source_rows in (
        cached_rows,
        date_rows,
    ):
        for game_id, row in (
            exact_context_by_game_id(
                source_rows
            ).items()
        ):
            if game_id in exact_rows:
                old = exact_rows[game_id]

                old_identity = (
                    normalize_team_key(
                        old.get("home_team")
                    ),
                    normalize_team_key(
                        old.get("away_team")
                    ),
                )

                new_identity = (
                    normalize_team_key(
                        row.get("home_team")
                    ),
                    normalize_team_key(
                        row.get("away_team")
                    ),
                )

                if old_identity != new_identity:
                    raise RuntimeError(
                        "Conflicting SDV contexts "
                        f"for game_id={game_id}"
                    )

            exact_rows[game_id] = row

    team_id_map = build_team_id_map(
        cfg,
        league,
        cached_rows,
        date_rows,
    )

    targets: list[
        dict[str, Any]
    ] = []

    fallback_count = 0

    for slate_row in slate_rows:
        game_id = clean_id(
            slate_row["game_id"]
        )

        exact = exact_rows.get(
            game_id
        )

        if exact is not None:
            assert_team_identity(
                slate_row,
                exact,
            )

            target = dict(exact)

            target["sport"] = (
                clean(
                    slate_row.get("sport")
                )
                or "basketball"
            )

            target["league"] = (
                LEAGUE_LABELS[league]
            )

            target["game_id"] = game_id
            target["game_date"] = game_date

            target["game_time"] = (
                clean(
                    slate_row.get("game_time")
                )
                or clean(
                    exact.get("game_time")
                )
            )

            target["home_team"] = clean(
                slate_row.get("home_team")
            )

            target["away_team"] = clean(
                slate_row.get("away_team")
            )

            target["internal_season"] = str(
                internal_season
            )

            target["sdv_season"] = str(
                mapped_sdv_season
            )

            targets.append(target)
            continue

        home_team = clean(
            slate_row.get("home_team")
        )

        away_team = clean(
            slate_row.get("away_team")
        )

        home_team_id = (
            team_id_map.get(
                normalize_team_key(
                    home_team
                )
            )
        )

        away_team_id = (
            team_id_map.get(
                normalize_team_key(
                    away_team
                )
            )
        )

        if not home_team_id:
            raise RuntimeError(
                "Unable to resolve SDV "
                "home_team_id for daily "
                f"game_id={game_id} "
                f"team={home_team!r}"
            )

        if not away_team_id:
            raise RuntimeError(
                "Unable to resolve SDV "
                "away_team_id for daily "
                f"game_id={game_id} "
                f"team={away_team!r}"
            )

        fallback_count += 1

        targets.append(
            {
                "sport": (
                    clean(
                        slate_row.get("sport")
                    )
                    or "basketball"
                ),
                "league": (
                    LEAGUE_LABELS[league]
                ),
                "internal_season": str(
                    internal_season
                ),
                "sdv_season": str(
                    mapped_sdv_season
                ),
                "game_date": game_date,
                "game_time": clean(
                    slate_row.get(
                        "game_time"
                    )
                ),
                "home_team": home_team,
                "away_team": away_team,
                "game_id": game_id,
                "home_team_id": (
                    home_team_id
                ),
                "away_team_id": (
                    away_team_id
                ),
                "neutral_site": "",
                "venue_id": "",
                "venue_full_name": "",
                "venue_name": "",
                "home_venue_id": "",
                "away_venue_id": "",
                "status": "",
                "source": (
                    "daily_games_sdv_"
                    "team_id_fallback"
                ),
                "fetched_at_utc": (
                    utc_now()
                ),
            }
        )

    if len(targets) != len(
        slate_rows
    ):
        raise RuntimeError(
            "Feature target count does "
            "not match daily slate"
        )

    log(
        "FEATURE TARGETS READY | "
        f"league={LEAGUE_LABELS[league]} "
        f"game_date={game_date} "
        f"targets={len(targets)} "
        "fallback_context_rows="
        f"{fallback_count}"
    )

    return targets


def generate_current_features(
    cfg: dict[str, Any],
    league: str,
    game_date: str,
    internal_season: int,
    mapped_sdv_season: int,
    slate_rows: list[dict[str, Any]],
) -> Path:
    paths = (
        feature_generation
        .configured_paths(cfg)
    )

    history_root = (
        paths["history_input_root"]
    )

    (
        _,
        team_index,
        player_index,
        home_venue_index,
        league_efficiency_index,
    ) = (
        feature_generation
        .build_indexes(
            history_root,
            league,
            cfg,
        )
    )

    targets = build_feature_targets(
        cfg,
        league,
        game_date,
        internal_season,
        mapped_sdv_season,
        slate_rows,
    )

    generated_at = utc_now()

    rows = (
        feature_generation
        .generate_rows(
            targets,
            canonical=True,
            league=league,
            cfg=cfg,
            team_index=team_index,
            player_index=player_index,
            home_venue_index=(
                home_venue_index
            ),
            league_efficiency_index=(
                league_efficiency_index
            ),
            generated_at=generated_at,
        )
    )

    label = LEAGUE_LABELS[league]

    output = (
        paths["current_output_root"]
        / league
        / (
            f"{game_date}_"
            f"{label}_features.parquet"
        )
    )

    feature_generation.write_features(
        output,
        rows,
        cfg,
    )

    log(
        "CURRENT FEATURES READY | "
        f"league={label} "
        f"game_date={game_date} "
        f"rows={len(rows)} "
        f"path={output}"
    )

    return output


def load_feature_rows(
    path: Path,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pl.read_parquet(path)

    if frame.is_empty():
        raise RuntimeError(
            "Generated feature file "
            "contains zero rows: "
            f"{path}"
        )

    by_id: dict[
        str,
        dict[str, Any],
    ] = {}

    for row in frame.to_dicts():
        game_id = clean_id(
            row.get("game_id")
        )

        if not game_id:
            raise RuntimeError(
                "Generated feature row "
                "has blank game_id"
            )

        if game_id in by_id:
            raise RuntimeError(
                "Generated features contain "
                f"duplicate game_id={game_id}"
            )

        row["game_id"] = game_id
        by_id[game_id] = row

    return by_id


def assert_complete_feature_coverage(
    slate_rows: list[dict[str, Any]],
    features_by_id: dict[
        str,
        dict[str, Any],
    ],
) -> None:
    slate_ids = {
        clean_id(row["game_id"])
        for row in slate_rows
    }

    feature_ids = set(
        features_by_id
    )

    missing = sorted(
        slate_ids - feature_ids
    )

    extra = sorted(
        feature_ids - slate_ids
    )

    if missing:
        raise RuntimeError(
            "CURRENT SLATE FEATURE COVERAGE "
            "FAILED. Canonical games without "
            f"feature rows={missing}"
        )

    if extra:
        raise RuntimeError(
            "CURRENT SLATE FEATURE COVERAGE "
            "FAILED. Feature rows not present "
            f"in canonical slate={extra}"
        )


def load_artifacts(
    cfg: dict[str, Any],
    league: str,
) -> dict[str, Any]:
    root = (
        configured_model_root(cfg)
        / league
    )

    missing = [
        str(root / filename)
        for filename
        in REQUIRED_ARTIFACT_FILES
        if not (
            root / filename
        ).exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing SDV model artifacts: "
            f"{missing}"
        )

    return {
        "root": root,
        "margin": read_json(
            root / "margin_model.json"
        ),
        "total": read_json(
            root / "total_model.json"
        ),
        "schema": read_json(
            root / "feature_schema.json"
        ),
        "metadata": read_json(
            root / "metadata.json"
        ),
    }


def exact_single_version(
    values: list[Any],
    label: str,
) -> str:
    normalized = {
        clean(value)
        for value in values
    }

    if (
        "" in normalized
        or len(normalized) != 1
    ):
        raise RuntimeError(
            f"{label} mismatch: "
            f"{sorted(normalized)}"
        )

    return next(iter(normalized))


def validate_artifacts(
    cfg: dict[str, Any],
    league: str,
    artifacts: dict[str, Any],
) -> tuple[str, str]:
    margin = artifacts["margin"]
    total = artifacts["total"]
    schema = artifacts["schema"]
    metadata = artifacts["metadata"]

    label = LEAGUE_LABELS[league]

    for artifact_name, artifact in (
        ("margin_model", margin),
        ("total_model", total),
        ("feature_schema", schema),
        ("metadata", metadata),
    ):
        artifact_league = clean(
            artifact.get("league")
        ).upper()

        if artifact_league != label:
            raise RuntimeError(
                f"{artifact_name} league "
                "mismatch: "
                f"expected={label} "
                f"actual={artifact_league}"
            )

    configured_feature_version = (
        feature_generation
        .production_feature_version(cfg)
    )

    training_cfg = required_mapping(
        cfg,
        "training",
    )

    configured_model_version = clean(
        training_cfg.get("model_version")
    )

    feature_version = (
        exact_single_version(
            [
                configured_feature_version,
                margin.get(
                    "feature_version"
                ),
                total.get(
                    "feature_version"
                ),
                schema.get(
                    "feature_version"
                ),
                metadata.get(
                    "feature_version"
                ),
            ],
            "feature_version",
        )
    )

    model_version = (
        exact_single_version(
            [
                configured_model_version,
                margin.get(
                    "model_version"
                ),
                total.get(
                    "model_version"
                ),
                schema.get(
                    "model_version"
                ),
                metadata.get(
                    "model_version"
                ),
            ],
            "model_version",
        )
    )

    margin_encoder = margin.get(
        "encoder"
    )

    total_encoder = total.get(
        "encoder"
    )

    if not isinstance(
        margin_encoder,
        dict,
    ):
        raise RuntimeError(
            "margin_model encoder missing"
        )

    if not isinstance(
        total_encoder,
        dict,
    ):
        raise RuntimeError(
            "total_model encoder missing"
        )

    if margin_encoder != total_encoder:
        raise RuntimeError(
            "Margin and total encoders "
            "do not match"
        )

    encoded_order = schema.get(
        "encoded_feature_order"
    )

    raw_order = schema.get(
        "raw_feature_order"
    )

    if not isinstance(
        encoded_order,
        list,
    ):
        raise RuntimeError(
            "Invalid encoded_feature_order"
        )

    if not isinstance(
        raw_order,
        list,
    ):
        raise RuntimeError(
            "Invalid raw_feature_order"
        )

    if (
        margin_encoder.get(
            "encoded_feature_order"
        )
        != encoded_order
    ):
        raise RuntimeError(
            "Model encoded feature order "
            "does not match feature schema"
        )

    numeric_order = (
        margin_encoder.get(
            "numeric_feature_order"
        )
    )

    categorical_order = (
        margin_encoder.get(
            "categorical_feature_order"
        )
    )

    if not isinstance(
        numeric_order,
        list,
    ):
        raise RuntimeError(
            "Invalid numeric feature order"
        )

    if not isinstance(
        categorical_order,
        list,
    ):
        raise RuntimeError(
            "Invalid categorical feature order"
        )

    if (
        numeric_order
        + categorical_order
        != raw_order
    ):
        raise RuntimeError(
            "Raw feature order mismatch"
        )

    for model_name, model in (
        ("margin", margin),
        ("total", total),
    ):
        coefficients = model.get(
            "coefficients"
        )

        if not isinstance(
            coefficients,
            list,
        ):
            raise RuntimeError(
                f"{model_name}: "
                "coefficients invalid"
            )

        if len(coefficients) != len(
            encoded_order
        ):
            raise RuntimeError(
                f"{model_name}: "
                "coefficient count does "
                "not match schema"
            )

        names: list[str] = []

        for expected_position, item in enumerate(
            coefficients
        ):
            if not isinstance(
                item,
                dict,
            ):
                raise RuntimeError(
                    f"{model_name}: "
                    "invalid coefficient entry"
                )

            if int(
                item.get(
                    "position",
                    -1,
                )
            ) != expected_position:
                raise RuntimeError(
                    f"{model_name}: "
                    "coefficient position mismatch"
                )

            feature = clean(
                item.get("feature")
            )

            names.append(feature)

            to_float_required(
                item.get("value"),
                (
                    f"{model_name} "
                    f"coefficient {feature}"
                ),
            )

        if names != encoded_order:
            raise RuntimeError(
                f"{model_name}: "
                "coefficient order does "
                "not match schema"
            )

        residual = model.get(
            "residual_distribution"
        )

        if not isinstance(
            residual,
            dict,
        ):
            raise RuntimeError(
                f"{model_name}: "
                "residual distribution missing"
            )

        residual_std = (
            to_float_required(
                residual.get("std"),
                (
                    f"{model_name} "
                    "residual std"
                ),
            )
        )

        if residual_std <= 0:
            raise RuntimeError(
                f"{model_name}: "
                "residual std must be positive"
            )

        to_float_required(
            residual.get("mean"),
            (
                f"{model_name} "
                "residual mean"
            ),
        )

    contract = metadata.get(
        "version_enforcement_contract"
    )

    if not isinstance(
        contract,
        dict,
    ):
        raise RuntimeError(
            "Metadata version contract "
            "missing"
        )

    for key in (
        "predictor_must_require_exact_model_version_match",
        "predictor_must_require_exact_feature_version_match",
        "predictor_must_refuse_on_mismatch",
    ):
        if not bool(
            contract.get(
                key,
                False,
            )
        ):
            raise RuntimeError(
                "Metadata version contract "
                f"failed: {key}"
            )

    return (
        model_version,
        feature_version,
    )


def validate_feature_row_schema(
    row: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    raw_features = schema.get(
        "raw_features"
    )

    if not isinstance(
        raw_features,
        list,
    ):
        raise RuntimeError(
            "feature_schema raw_features "
            "is invalid"
        )

    for expected_position, item in enumerate(
        raw_features
    ):
        if not isinstance(
            item,
            dict,
        ):
            raise RuntimeError(
                "Invalid raw feature schema"
            )

        if int(
            item.get(
                "position",
                -1,
            )
        ) != expected_position:
            raise RuntimeError(
                "Raw feature position mismatch"
            )

        name = clean(
            item.get("name")
        )

        if not name:
            raise RuntimeError(
                "Blank model feature name"
            )

        if name not in row:
            raise RuntimeError(
                "Current feature row missing "
                "required model input: "
                f"game_id={row.get('game_id')} "
                f"feature={name}"
            )


def numeric_feature_value(
    row: dict[str, Any],
    name: str,
    fill_value: float,
) -> float:
    value = row.get(name)

    if (
        value is None
        or clean(value) == ""
    ):
        return fill_value

    try:
        result = float(value)

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeError(
            "Feature is not numeric: "
            f"game_id={row.get('game_id')} "
            f"feature={name} "
            f"value={value!r}"
        ) from exc

    if not math.isfinite(result):
        raise RuntimeError(
            "Feature is non-finite: "
            f"game_id={row.get('game_id')} "
            f"feature={name}"
        )

    return result


def encode_feature_row(
    row: dict[str, Any],
    model: dict[str, Any],
    schema: dict[str, Any],
) -> np.ndarray:
    validate_feature_row_schema(
        row,
        schema,
    )

    encoder = model["encoder"]

    encoded_order = encoder.get(
        "encoded_feature_order"
    )

    if not isinstance(
        encoded_order,
        list,
    ):
        raise RuntimeError(
            "Invalid encoded feature order"
        )

    feature_index = {
        clean(feature): position
        for position, feature
        in enumerate(encoded_order)
    }

    vector = np.zeros(
        len(encoded_order),
        dtype=float,
    )

    intercept = encoder.get(
        "intercept"
    )

    if not isinstance(
        intercept,
        dict,
    ):
        raise RuntimeError(
            "Model intercept metadata missing"
        )

    intercept_index = int(
        intercept.get(
            "encoded_index",
            -1,
        )
    )

    if (
        intercept_index < 0
        or intercept_index >= len(vector)
    ):
        raise RuntimeError(
            "Invalid intercept index"
        )

    vector[intercept_index] = (
        to_float_required(
            intercept.get("value"),
            "intercept value",
        )
    )

    numeric_scaling = encoder.get(
        "numeric_scaling"
    )

    if not isinstance(
        numeric_scaling,
        list,
    ):
        raise RuntimeError(
            "numeric_scaling invalid"
        )

    for item in numeric_scaling:
        name = clean(
            item.get("name")
        )

        if name not in feature_index:
            raise RuntimeError(
                "Numeric feature absent "
                "from encoded order: "
                f"{name}"
            )

        fill_value = to_float_required(
            item.get(
                "missing_fill_value"
            ),
            f"{name} fill value",
        )

        mean_value = to_float_required(
            item.get("mean"),
            f"{name} mean",
        )

        std_value = to_float_required(
            item.get("std"),
            f"{name} std",
        )

        if std_value <= 0:
            raise RuntimeError(
                f"{name}: std must be positive"
            )

        raw_value = numeric_feature_value(
            row,
            name,
            fill_value,
        )

        vector[
            feature_index[name]
        ] = (
            raw_value - mean_value
        ) / std_value

    categorical_encoding = encoder.get(
        "categorical_encoding"
    )

    if not isinstance(
        categorical_encoding,
        list,
    ):
        raise RuntimeError(
            "categorical_encoding invalid"
        )

    for item in categorical_encoding:
        name = clean(
            item.get("name")
        )

        missing_token = clean(
            item.get("missing_token")
        )

        unknown_token = clean(
            item.get("unknown_token")
        )

        mapping = item.get(
            "encoded_index_by_level"
        )

        if not isinstance(
            mapping,
            dict,
        ):
            raise RuntimeError(
                f"{name}: categorical "
                "mapping missing"
            )

        category = (
            clean(row.get(name))
            or missing_token
        )

        if category not in mapping:
            category = unknown_token

        if category not in mapping:
            raise RuntimeError(
                f"{name}: unknown category "
                "token not encoded"
            )

        encoded_index = int(
            mapping[category]
        )

        if (
            encoded_index < 0
            or encoded_index >= len(vector)
        ):
            raise RuntimeError(
                f"{name}: encoded index "
                "out of bounds"
            )

        vector[encoded_index] = 1.0

    if not np.all(
        np.isfinite(vector)
    ):
        raise RuntimeError(
            "Encoded feature vector "
            "contains non-finite values "
            f"for game_id={row.get('game_id')}"
        )

    return vector


def model_coefficients(
    model: dict[str, Any],
) -> np.ndarray:
    rows = model.get(
        "coefficients"
    )

    if not isinstance(
        rows,
        list,
    ):
        raise RuntimeError(
            "Model coefficients invalid"
        )

    return np.asarray(
        [
            to_float_required(
                item.get("value"),
                "model coefficient",
            )
            for item in rows
        ],
        dtype=float,
    )


def predict_model(
    feature_row: dict[str, Any],
    model: dict[str, Any],
    schema: dict[str, Any],
) -> float:
    vector = encode_feature_row(
        feature_row,
        model,
        schema,
    )

    coefficients = (
        model_coefficients(model)
    )

    if (
        vector.shape[0]
        != coefficients.shape[0]
    ):
        raise RuntimeError(
            "Inference vector/coefficient "
            "length mismatch"
        )

    prediction = float(
        np.dot(
            vector,
            coefficients,
        )
    )

    if not math.isfinite(
        prediction
    ):
        raise RuntimeError(
            "Non-finite prediction "
            f"game_id="
            f"{feature_row.get('game_id')}"
        )

    return prediction


def residual_parameters(
    model: dict[str, Any],
) -> tuple[float, float]:
    residual = model.get(
        "residual_distribution"
    )

    if not isinstance(
        residual,
        dict,
    ):
        raise RuntimeError(
            "Residual distribution missing"
        )

    mean_value = to_float_required(
        residual.get("mean"),
        "residual mean",
    )

    std_value = to_float_required(
        residual.get("std"),
        "residual std",
    )

    if std_value <= 0:
        raise RuntimeError(
            "Residual std must be positive"
        )

    return (
        mean_value,
        std_value,
    )


def normal_cdf(
    value: float,
) -> float:
    probability = (
        0.5
        * (
            1.0
            + math.erf(
                value
                / math.sqrt(2.0)
            )
        )
    )

    return min(
        max(
            probability,
            PROBABILITY_EPSILON,
        ),
        1.0 - PROBABILITY_EPSILON,
    )


def moneyline_probabilities(
    expected_margin: float,
    residual_mean: float,
    residual_std: float,
) -> tuple[float, float]:
    adjusted_margin = (
        expected_margin
        + residual_mean
    )

    home_probability = normal_cdf(
        adjusted_margin
        / residual_std
    )

    away_probability = (
        1.0 - home_probability
    )

    if abs(
        (
            home_probability
            + away_probability
        )
        - 1.0
    ) > 1e-12:
        raise RuntimeError(
            "Moneyline complement "
            "validation failed"
        )

    return (
        home_probability,
        away_probability,
    )


def validate_current_feature_version(
    feature_row: dict[str, Any],
    expected_feature_version: str,
) -> None:
    actual = clean(
        feature_row.get(
            "feature_version"
        )
    )

    if (
        actual
        != expected_feature_version
    ):
        raise RuntimeError(
            "FEATURE/MODEL VERSION "
            "MISMATCH | "
            f"game_id="
            f"{feature_row.get('game_id')} "
            f"feature={actual!r} "
            f"model="
            f"{expected_feature_version!r}"
        )


def build_prediction_row(
    *,
    slate_row: dict[str, Any],
    feature_row: dict[str, Any],
    artifacts: dict[str, Any],
    model_version: str,
    feature_version: str,
    prediction_generated_at: str,
) -> dict[str, Any]:
    validate_current_feature_version(
        feature_row,
        feature_version,
    )

    margin_model = artifacts[
        "margin"
    ]

    total_model = artifacts[
        "total"
    ]

    schema = artifacts[
        "schema"
    ]

    expected_margin = predict_model(
        feature_row,
        margin_model,
        schema,
    )

    expected_total = predict_model(
        feature_row,
        total_model,
        schema,
    )

    (
        margin_residual_mean,
        margin_residual_std,
    ) = residual_parameters(
        margin_model
    )

    (
        total_residual_mean,
        total_residual_std,
    ) = residual_parameters(
        total_model
    )

    (
        raw_home_ml_prob,
        raw_away_ml_prob,
    ) = moneyline_probabilities(
        expected_margin,
        margin_residual_mean,
        margin_residual_std,
    )

    home_projected_points = (
        expected_total
        + expected_margin
    ) / 2.0

    away_projected_points = (
        expected_total
        - expected_margin
    ) / 2.0

    return {
        "sport": (
            clean(
                slate_row.get("sport")
            )
            or "Basketball"
        ),
        "league": clean(
            slate_row.get("league")
        ),
        "game_id": clean_id(
            slate_row["game_id"]
        ),
        "game_date": clean(
            slate_row["game_date"]
        ),
        "game_time": clean(
            slate_row.get("game_time")
        ),
        "home_team": clean(
            slate_row.get("home_team")
        ),
        "away_team": clean(
            slate_row.get("away_team")
        ),
        "model_source": "sdv",
        "model_version": (
            model_version
        ),
        "feature_version": (
            feature_version
        ),
        "home_prob": (
            raw_home_ml_prob
        ),
        "away_prob": (
            raw_away_ml_prob
        ),
        "raw_home_ml_prob": (
            raw_home_ml_prob
        ),
        "raw_away_ml_prob": (
            raw_away_ml_prob
        ),
        "home_projected_points": (
            home_projected_points
        ),
        "away_projected_points": (
            away_projected_points
        ),
        "total_projected_points": (
            expected_total
        ),
        "expected_margin": (
            expected_margin
        ),
        "expected_total": (
            expected_total
        ),
        "margin_residual_mean": (
            margin_residual_mean
        ),
        "margin_residual_std": (
            margin_residual_std
        ),
        "total_residual_mean": (
            total_residual_mean
        ),
        "total_residual_std": (
            total_residual_std
        ),
        "feature_generated_at_utc": clean(
            feature_row.get(
                "feature_generated_at_utc"
            )
        ),
        "prediction_generated_at_utc": (
            prediction_generated_at
        ),
    }


def write_predictions(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise RuntimeError(
            "Refusing zero-row "
            f"prediction file: {path}"
        )

    game_ids = [
        clean_id(
            row.get("game_id")
        )
        for row in rows
    ]

    if any(
        not game_id
        for game_id in game_ids
    ):
        raise RuntimeError(
            "Prediction output contains "
            "blank game_id"
        )

    if len(game_ids) != len(
        set(game_ids)
    ):
        raise RuntimeError(
            "Prediction output contains "
            "duplicate game_id"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        f"{path}.tmp"
    )

    if tmp.exists():
        tmp.unlink()

    try:
        with tmp.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=PREDICTION_FIELDS,
                extrasaction="ignore",
            )

            writer.writeheader()
            writer.writerows(rows)

        tmp.replace(path)

    finally:
        if tmp.exists():
            tmp.unlink()


def run_prediction(
    cfg: dict[str, Any],
    league: str,
    game_date: str,
) -> Path | None:
    label = LEAGUE_LABELS[league]

    (
        daily_path,
        slate_rows,
    ) = load_daily_slate(
        league,
        game_date,
    )

    if not slate_rows:
        log(
            "NO GAMES | "
            f"league={label} "
            f"game_date={game_date} "
            f"path={daily_path} | "
            "skipped"
        )

        return None

    log(
        "DAILY SLATE READY | "
        f"league={label} "
        f"game_date={game_date} "
        f"rows={len(slate_rows)} "
        f"path={daily_path}"
    )

    internal_season = (
        internal_season_from_game_date(
            league,
            game_date,
        )
    )

    mapped_sdv_season = (
        sdv_season_id(
            league,
            internal_season,
        )
    )

    log(
        "SEASON READY | "
        f"league={label} "
        f"game_date={game_date} "
        f"internal_season="
        f"{internal_season} "
        f"sdv_season="
        f"{mapped_sdv_season}"
    )

    feature_path = (
        generate_current_features(
            cfg,
            league,
            game_date,
            internal_season,
            mapped_sdv_season,
            slate_rows,
        )
    )

    features_by_id = (
        load_feature_rows(
            feature_path
        )
    )

    assert_complete_feature_coverage(
        slate_rows,
        features_by_id,
    )

    log(
        "FEATURE COVERAGE PASS | "
        f"league={label} "
        f"game_date={game_date} "
        f"rows={len(slate_rows)}"
    )

    artifacts = load_artifacts(
        cfg,
        league,
    )

    (
        model_version,
        feature_version,
    ) = validate_artifacts(
        cfg,
        league,
        artifacts,
    )

    prediction_generated_at = (
        utc_now()
    )

    predictions: list[
        dict[str, Any]
    ] = []

    for slate_row in slate_rows:
        game_id = clean_id(
            slate_row["game_id"]
        )

        feature_row = (
            features_by_id.get(
                game_id
            )
        )

        if feature_row is None:
            raise RuntimeError(
                "Canonical current game "
                "lacks feature row: "
                f"game_id={game_id}"
            )

        predictions.append(
            build_prediction_row(
                slate_row=slate_row,
                feature_row=feature_row,
                artifacts=artifacts,
                model_version=(
                    model_version
                ),
                feature_version=(
                    feature_version
                ),
                prediction_generated_at=(
                    prediction_generated_at
                ),
            )
        )

    slate_ids = {
        clean_id(
            row["game_id"]
        )
        for row in slate_rows
    }

    prediction_ids = {
        clean_id(
            row["game_id"]
        )
        for row in predictions
    }

    if slate_ids != prediction_ids:
        raise RuntimeError(
            "Prediction coverage does not "
            "exactly match canonical slate"
        )

    output_path = (
        PREDICTION_ROOT
        / league
        / (
            f"{game_date}_"
            f"{label}_predictions.csv"
        )
    )

    write_predictions(
        output_path,
        predictions,
    )

    log(
        "PREDICTIONS READY | "
        f"league={label} "
        f"game_date={game_date} "
        f"rows={len(predictions)} "
        f"model_version={model_version} "
        f"feature_version="
        f"{feature_version} "
        f"path={output_path}"
    )

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate production SDV-only "
            "basketball predictions."
        )
    )

    parser.add_argument(
        "--league",
        required=True,
        choices=sorted(
            LEAGUE_LABELS
        ),
    )

    parser.add_argument(
        "--game-date",
        required=False,
        help=(
            "Optional single-date filter "
            "in YYYY_MM_DD or YYYY-MM-DD. "
            "If omitted, every available "
            "daily-games date for the "
            "selected league is processed."
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
    )

    return parser


def main() -> int:
    args = (
        build_parser()
        .parse_args()
    )

    ERROR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_FILE.write_text(
        (
            "=== SDV PRODUCTION "
            "PREDICTION "
            f"{utc_now()} ===\n"
        ),
        encoding="utf-8",
    )

    try:
        cfg = (
            feature_generation
            .read_yaml(
                args.config
            )
        )

        feature_generation.validate_config(
            cfg
        )

        if args.game_date:
            game_date = normalize_game_date(
                args.game_date
            )

            if not game_date:
                raise ValueError(
                    "Invalid --game-date "
                    f"{args.game_date!r}"
                )

            game_dates = [
                game_date
            ]

        else:
            game_dates = (
                discover_daily_game_dates(
                    args.league
                )
            )

        label = LEAGUE_LABELS[
            args.league
        ]

        if not game_dates:
            log(
                "STATUS: SUCCESS | "
                f"league={label} "
                "dates_found=0 "
                "dates_written=0 "
                "rows_written=0 "
                "dates_skipped=0 "
                "date_errors=0"
            )

            print(
                "SDV production prediction "
                "complete: SUCCESS. "
                f"league={label} "
                "dates_found=0 "
                "dates_written=0 "
                "rows_written=0 "
                "dates_skipped=0 "
                "date_errors=0"
            )

            return 0

        dates_written = 0
        rows_written = 0
        dates_skipped = 0
        date_errors = 0

        for game_date in game_dates:
            try:
                output_path = run_prediction(
                    cfg,
                    args.league,
                    game_date,
                )

                if output_path is None:
                    dates_skipped += 1
                    continue

                rows = read_csv_rows(
                    output_path
                )

                dates_written += 1
                rows_written += len(rows)

                log(
                    "DATE SUCCESS | "
                    f"league={label} "
                    f"game_date={game_date} "
                    f"rows={len(rows)} "
                    f"path={output_path}"
                )

            except Exception as exc:
                date_errors += 1

                log(
                    "DATE FAILED | "
                    f"league={label} "
                    f"game_date={game_date} "
                    f"error={exc}"
                )

                log(
                    traceback
                    .format_exc()
                    .rstrip()
                )

        log(
            "STATUS: SUCCESS | "
            f"league={label} "
            f"dates_found={len(game_dates)} "
            f"dates_written={dates_written} "
            f"rows_written={rows_written} "
            f"dates_skipped={dates_skipped} "
            f"date_errors={date_errors}"
        )

        print(
            "SDV production prediction "
            "complete: SUCCESS. "
            f"league={label} "
            f"dates_found={len(game_dates)} "
            f"dates_written={dates_written} "
            f"rows_written={rows_written} "
            f"dates_skipped={dates_skipped} "
            f"date_errors={date_errors}"
        )

        return 0

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
            "SDV production prediction "
            f"FAILED: {exc}"
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )