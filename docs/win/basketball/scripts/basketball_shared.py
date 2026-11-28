#!/usr/bin/env python3
"""Shared behavior-preserving utilities for basketball pipeline scripts.

Error log (appended, one line per error; never written on success):
  docs/win/basketball/errors/99_validation/basketball_shared.txt
"""

from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]

ERROR_LOG = (
    REPO_ROOT
    / "docs"
    / "win"
    / "basketball"
    / "errors"
    / "99_validation"
    / "basketball_shared.txt"
)


def log_error(message: str) -> None:
    """Append one error line to the shared-utilities log.

    The log location is anchored to REPO_ROOT, so it does not depend on the
    current working directory. Logging never raises: a log failure must not
    hide the original error.
    """
    try:
        ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ERROR_LOG.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{datetime.now(timezone.utc).isoformat()} | ERROR | "
                f"{message.rstrip()}\n"
            )
    except OSError:
        pass


def resolve_repository_path(
    path: Path,
    *,
    strict: bool = False,
) -> Path:
    # Relative paths are repository-relative, not cwd-relative, so callers
    # work no matter which folder the script is launched from.
    candidate = path if path.is_absolute() else REPO_ROOT / path

    try:
        safe_path = candidate.resolve(strict=strict)
    except (OSError, RuntimeError) as exc:
        log_error(
            f"resolve_repository_path failed: path={path} strict={strict} "
            f"{type(exc).__name__}: {exc}"
        )
        raise

    try:
        safe_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        log_error(
            f"Path escapes repository root: {path} -> {safe_path}"
        )
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
