#!/usr/bin/env python3
"""Validated launcher for basketball grading.

The locked-picks/replay grading implementation remains in
01_basketball_results_grade_core.py.

This launcher:
- clears stale mismatch diagnostics before every run;
- rejects conflicting duplicate final scores;
- adds a safe grading fallback for selected rows whose game_id does not match
  the final-score game_id but whose game_date + home_team + away_team uniquely
  identify the same completed game.
"""
from __future__ import annotations
from functools import lru_cache

import csv
import importlib.util
import sys
from pathlib import Path

import pandas as pd

BASE = Path("docs/win/basketball")
SELECT_BASE = BASE / "04_select"
RESULTS = BASE / "05_final_scores/results"
ERROR_DIR = BASE / "errors/05_final_scores"
CORE_PATH = Path(__file__).with_name("01_basketball_results_grade_core.py")
LEAGUES = ["nba", "ncaam", "wnba"]


def clean(v) -> str:
    return "" if v is None else str(v).strip()


def clean_date(v) -> str:
    return clean(v).replace("-", "_").replace("/", "_")


@lru_cache(maxsize=1)
def _team_normalizer():
    """Use the same NBA team map as the final-scores fetch script."""
    path = Path(__file__).with_name("00_basketball_final_scores.py")
    spec = importlib.util.spec_from_file_location("basketball_finals_aliases", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import final-score normalizer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize_team


def clean_team(v, league: str = "") -> str:
    return _team_normalizer()(v, league)




def comp(row: dict, league: str | None = None) -> tuple[str, str, str]:
    effective_league = league if league is not None else clean(row.get("league"))
    return (
        clean_date(row.get("game_date")),
        clean_team(row.get("home_team"), effective_league),
        clean_team(row.get("away_team"), effective_league),
    )




def score_sig(row: dict) -> tuple[str, str]:
    return clean(row.get("home_score")), clean(row.get("away_score"))


def clear_stale_diagnostics() -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    for league in LEAGUES:
        for name in [
            f"{league}_game_id_no_match.csv",
            f"{league}_locked_game_id_no_match.csv",
        ]:
            (ERROR_DIR / name).unlink(missing_ok=True)


def validate_final_scores() -> None:
    errors: list[str] = []
    for league in LEAGUES:
        by_id: dict[str, tuple[tuple[str, str, str], tuple[str, str], str]] = {}
        by_comp: dict[tuple[str, str, str], tuple[tuple[str, str], str, str]] = {}
        folder = RESULTS / league
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.csv")):
            with path.open(newline="", encoding="utf-8-sig") as handle:
                for n, row in enumerate(csv.DictReader(handle), start=2):
                    key = comp(row, league)
                    scores = score_sig(row)
                    gid = clean(row.get("game_id"))
                    if not all(key) or not all(scores):
                        continue
                    location = f"{path}:{n}"
                    if gid:
                        prior = by_id.get(gid)
                        if prior and (prior[0] != key or prior[1] != scores):
                            errors.append(
                                f"{league.upper()} game_id {gid} has conflicting finals: "
                                f"{prior[2]} vs {location}"
                            )
                        else:
                            by_id[gid] = (key, scores, location)
                    prior_comp = by_comp.get(key)
                    if prior_comp and prior_comp[0] != scores:
                        errors.append(
                            f"{league.upper()} {key} has conflicting scores: "
                            f"{prior_comp[1]} vs {location}"
                        )
                    elif (prior_comp and prior_comp[2] and gid
                          and prior_comp[2] != gid):
                        errors.append(
                            f"{league.upper()} {key} has conflicting game IDs: "
                            f"{prior_comp[2]} vs {gid} at {prior_comp[1]} vs {location}"
                        )
                    elif prior_comp is None:
                        by_comp[key] = (scores, location, gid)
                    elif not prior_comp[2] and gid:
                        by_comp[key] = (scores, location, gid)
    if errors:
        raise RuntimeError(
            "Final-score integrity validation failed:\n" + "\n".join(errors[:50])
        )




def load_core():
    spec = importlib.util.spec_from_file_location(
        "basketball_results_grade_core",
        CORE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load grading core: {CORE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_final_score_composites(league: str) -> dict[tuple[str, str, str], tuple[str, str]]:
    """Return unique composite -> score pairs after validate_final_scores()."""
    matches: dict[tuple[str, str, str], tuple[str, str]] = {}
    folder = RESULTS / league

    if not folder.exists():
        return matches

    for path in sorted(folder.glob("*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = comp(row, league)
                scores = score_sig(row)

                if not all(key) or not all(scores):
                    continue

                prior = matches.get(key)
                if prior is not None and prior != scores:
                    raise RuntimeError(
                        f"{league.upper()} composite fallback is ambiguous for {key}: "
                        f"{prior} vs {scores}"
                    )

                matches[key] = scores

    return matches


def _iter_pick_rows(league: str):
    for folder_name in ("daily_picks", "locked_picks"):
        folder = SELECT_BASE / league / folder_name
        if not folder.exists():
            continue

        for path in sorted(folder.glob("*.csv")):
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    yield row


def install_composite_score_fallback(core) -> None:
    """Extend the core's score loader with safe alias rows for mismatched IDs.

    The core still grades by game_id. For a selected row whose game_id is absent
    from score data, this wrapper adds an in-memory score alias only when
    game_date + home_team + away_team identify exactly one non-conflicting final.
    No score CSV is rewritten and the selected row's original game_id is preserved.
    """
    original_loader = core.load_scores_for_league

    def load_scores_for_league_with_fallback(league: str) -> pd.DataFrame:
        scores = original_loader(league)
        if scores.empty or "game_id" not in scores.columns:
            return scores

        known_ids = {
            clean(value)
            for value in scores["game_id"].dropna().tolist()
            if clean(value)
        }

        by_comp = _read_final_score_composites(league)
        aliases: list[dict[str, object]] = []

        for pick in _iter_pick_rows(league):
            gid = clean(pick.get("game_id"))
            if not gid or gid in known_ids:
                continue

            key = comp(pick, league)
            if not all(key):
                continue

            score_pair = by_comp.get(key)
            if score_pair is None:
                continue

            aliases.append(
                {
                    "game_id": gid,
                    "home_score": score_pair[0],
                    "away_score": score_pair[1],
                }
            )
            known_ids.add(gid)

        if aliases:
            alias_df = pd.DataFrame(aliases)
            scores = pd.concat([scores, alias_df], ignore_index=True)
            scores = scores.drop_duplicates(subset=["game_id"], keep="last")
            core.log(
                f"[{league}] composite final-score fallback added "
                f"{len(alias_df)} game_id alias(es)"
            )

        return scores

    core.load_scores_for_league = load_scores_for_league_with_fallback


def main() -> None:
    clear_stale_diagnostics()
    validate_final_scores()

    core = load_core()
    install_composite_score_fallback(core)
    core.main()

    text = core.LOG_FILE.read_text(encoding="utf-8", errors="replace")
    if "STATUS: COMPLETED WITH ERRORS" in text or "STATUS: FAILED" in text:
        sys.exit(1)


if __name__ == "__main__":
    main()
