#!/usr/bin/env python3
"""Shared behavior-preserving utilities for basketball pipeline scripts."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


def resolve_repository_path(
    path: Path,
    *,
    strict: bool = False,
) -> Path:
    candidate = path if path.is_absolute() else Path.cwd() / path
    safe_path = candidate.resolve(strict=strict)

    try:
        safe_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Path escapes repository root: {path}"
        ) from exc

    return safe_path


def identity_key(
    row: dict,
    clean,
    unresolved_team,
) -> tuple[str, str, str, str]:
    league = clean(row.get("league")).upper()
    game_date = clean(row.get("game_date"))
    home_team = clean(row.get("home_team"))
    away_team = clean(row.get("away_team"))

    if unresolved_team(home_team) or unresolved_team(away_team):
        return league, game_date, "", ""

    return (
        league,
        game_date,
        home_team.casefold(),
        away_team.casefold(),
    )
