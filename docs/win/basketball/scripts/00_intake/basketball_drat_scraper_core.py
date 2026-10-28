
#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/basketball_drat_scraper_core.py

"""
DRatings basketball scraper.

Preservation rules:
- Upcoming games may be inserted or updated.
- In-progress games never overwrite saved predictions.
- Completed games never overwrite saved predictions.
- Previously saved games missing from a new scrape are retained.
- Live/completed games without saved predictions are not inserted.
- Existing JSON is validated before replacement.
- Output is written atomically.

The output JSON schema and directory structure are unchanged.

This module is loaded by basketball_drat_scraper.py, which controls
which leagues are active using season_dates.yaml.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import traceback

from datetime import datetime
from pathlib import Path

import pytz
from playwright.sync_api import sync_playwright


URLS = {
    "nba": (
        "https://www.dratings.com/predictor/"
        "nba-basketball-predictions/"
    ),
    "ncaa": (
        "https://www.dratings.com/predictor/"
        "ncaa-basketball-predictions/"
    ),
    "wnba": (
        "https://www.dratings.com/predictor/"
        "wnba-basketball-predictions/"
    ),
}

UTC = pytz.utc
ET = pytz.timezone("America/New_York")

BASE = Path("docs/win/basketball")
OUTPUT_ROOT = BASE / "00_intake/drat_raw"
ERROR_DIR = BASE / "errors/00_intake"

ERROR_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "basketball_drat_scraper.txt"

with LOG_FILE.open("w", encoding="utf-8") as handle:
    handle.write(
        "=== basketball_drat_scraper RUN "
        f"{datetime.now(ET).isoformat()} ===\n"
    )


GAME_FIELDS = (
    "sport",
    "date_time",
    "team1",
    "team2",
    "team1_win_pct",
    "team2_win_pct",
    "team1_moneyline",
    "team2_moneyline",
    "team1_spread",
    "team2_spread",
    "proj_score_1",
    "proj_score_2",
    "total",
    "over_line",
    "under_line",
    "score1",
    "score2",
    "game_status",
)

PREDICTION_FIELDS = (
    "team1_win_pct",
    "team2_win_pct",
    "team1_moneyline",
    "team2_moneyline",
    "team1_spread",
    "team2_spread",
    "proj_score_1",
    "proj_score_2",
    "total",
    "over_line",
    "under_line",
)


def log(message: str) -> None:
    line = (
        f"{datetime.now(ET).isoformat()} | "
        f"{message}"
    )

    print(line, flush=True)

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(line + "\n")


def normalize_cell(value) -> str:
    value = "" if value is None else str(value)

    return (
        value.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
        .replace("\xa0", " ")
        .strip()
    )


def convert_utc_to_et(date_time_str: str) -> str:
    try:
        dt = datetime.strptime(
            date_time_str.strip(),
            "%m/%d/%Y %I:%M %p",
        )

        dt_utc = UTC.localize(dt)
        dt_et = dt_utc.astimezone(ET)

        return dt_et.strftime(
            "%m/%d/%Y %I:%M %p"
        )

    except (
        AttributeError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return date_time_str


def split_pair(value):
    value = normalize_cell(value)

    if not value:
        return "", ""

    if "\n" in value:
        parts = [
            p.strip()
            for p in re.split(r"\n+", value)
            if p.strip()
        ]

    elif "|" in value:
        parts = [
            p.strip()
            for p in value.split("|")
            if p.strip()
        ]

    else:
        parts = [value.strip()]

    if len(parts) >= 2:
        return parts[0], parts[1]

    if len(parts) == 1:
        return parts[0], ""

    return "", ""


def split_pair_expected(
    value,
    label: str,
    sport: str,
    row,
) -> tuple:
    a, b = split_pair(value)

    if a and not b:
        log(
            "MALFORMED PAIR | "
            f"sport={sport} | "
            f"field={label} | "
            f"value={json.dumps(normalize_cell(value))} | "
            f"row={json.dumps(row)}"
        )

    return a, b


def strip_record(team: str) -> str:
    return re.sub(
        r"\s*\(\d+[-–]\d+[-–]?\d*\)\s*$",
        "",
        (team or "").strip(),
    ).strip()


def strip_wnba_record(team: str) -> str:
    return strip_record(team)


def is_game_row(row) -> bool:
    if len(row) < 5:
        return False

    teams = normalize_cell(row[1])

    return "\n" in teams


def is_score(value) -> bool:
    try:
        number = float(str(value).strip())

        return (
            0 <= number
            and number == int(number)
            and number < 250
        )

    except (ValueError, TypeError):
        return False


def is_summary_row(row) -> bool:
    if not row:
        return False

    first = normalize_cell(row[0]).lower()

    return first in {
        "sportsbooks",
        "dratings",
    }


def empty_game(
    sport: str,
    date_time: str,
    team1: str,
    team2: str,
) -> dict:
    return {
        "sport": sport,
        "date_time": date_time,
        "team1": team1,
        "team2": team2,
        "team1_win_pct": "",
        "team2_win_pct": "",
        "team1_moneyline": "",
        "team2_moneyline": "",
        "team1_spread": "",
        "team2_spread": "",
        "proj_score_1": "",
        "proj_score_2": "",
        "total": "",
        "over_line": "",
        "under_line": "",
        "score1": "",
        "score2": "",
        "game_status": "",
    }


def parse_nba_ncaa(row, sport):
    """
    Parse an upcoming NBA/NCAAM prediction row.

    Live/completed tables are handled separately and are
    never passed to this function for prediction updates.
    """
    if is_summary_row(row):
        return None

    if not is_game_row(row):
        return None

    try:
        date_time = convert_utc_to_et(
            normalize_cell(row[0]).replace(
                "\n", " "
            )
        )

        team1, team2 = split_pair_expected(
            row[1], "teams", sport, row
        )

        wp1, wp2 = split_pair_expected(
            row[2], "win_pct", sport, row
        )

        ml1, ml2 = split_pair_expected(
            row[3], "moneyline", sport, row
        )

        sp1, sp2 = split_pair_expected(
            row[4], "spread", sport, row
        )

        if not team1 or not team2:
            log(
                f"{sport.upper()} REJECTED EMPTY TEAM | "
                f"row={json.dumps(row)}"
            )
            return None

        game = empty_game(
            sport,
            date_time,
            team1,
            team2,
        )

        game.update({
            "team1_win_pct": wp1,
            "team2_win_pct": wp2,
            "team1_moneyline": ml1,
            "team2_moneyline": ml2,
            "team1_spread": sp1,
            "team2_spread": sp2,
        })

        if len(row) >= 10:
            proj1, proj2 = split_pair_expected(
                row[5],
                "projected_scores",
                sport,
                row,
            )

            over_line, under_line = (
                split_pair_expected(
                    row[7],
                    "over_under",
                    sport,
                    row,
                )
            )

            game.update({
                "proj_score_1": proj1,
                "proj_score_2": proj2,
                "total": normalize_cell(row[6]),
                "over_line": over_line,
                "under_line": under_line,
            })

        elif (
            len(row) >= 9
            and not is_score(
                normalize_cell(row[5])
            )
        ):
            over_line, under_line = (
                split_pair_expected(
                    row[6],
                    "over_under",
                    sport,
                    row,
                )
            )

            game.update({
                "total": normalize_cell(row[5]),
                "over_line": over_line,
                "under_line": under_line,
                "game_status": " ".join(
                    p
                    for p in re.split(
                        r"\n+",
                        normalize_cell(row[7]),
                    )
                    if p.strip()
                ),
            })

            score1, score2 = split_pair_expected(
                row[8],
                "score",
                sport,
                row,
            )

            game["score1"] = score1
            game["score2"] = score2

        elif len(row) >= 7:
            score1, score2 = split_pair_expected(
                row[5],
                "score",
                sport,
                row,
            )

            if (
                not score2
                and len(row) > 6
                and is_score(row[6])
            ):
                score2 = normalize_cell(row[6])

            game["score1"] = score1
            game["score2"] = score2

        return game

    except Exception as exc:
        log(
            "NBA/NCAA PARSE ERROR | "
            f"sport={sport} | "
            f"error={exc} | "
            f"row={json.dumps(row)}\n"
            f"{traceback.format_exc()}"
        )
        return None


def parse_wnba(row):
    if is_summary_row(row):
        return None

    if len(row) not in (8, 10):
        log(
            "WNBA REJECTED ROW LENGTH | "
            f"len={len(row)} | "
            f"row={json.dumps(row)}"
        )
        return None

    try:
        date_time = convert_utc_to_et(
            normalize_cell(row[0]).replace(
                "\n", " "
            )
        )

        team1, team2 = split_pair_expected(
            row[1],
            "teams",
            "wnba",
            row,
        )

        team1 = strip_wnba_record(team1)
        team2 = strip_wnba_record(team2)

        if not team1 or not team2:
            return None

        if team1.lower() in {
            "sportsbooks",
            "dratings",
        }:
            return None

        game = empty_game(
            "wnba",
            date_time,
            team1,
            team2,
        )

        wp1, wp2 = split_pair_expected(
            row[2],
            "win_pct",
            "wnba",
            row,
        )

        ml1, ml2 = split_pair_expected(
            row[3],
            "moneyline",
            "wnba",
            row,
        )

        sp1, sp2 = split_pair_expected(
            row[4],
            "spread",
            "wnba",
            row,
        )

        game.update({
            "team1_win_pct": wp1,
            "team2_win_pct": wp2,
            "team1_moneyline": ml1,
            "team2_moneyline": ml2,
            "team1_spread": sp1,
            "team2_spread": sp2,
        })

        if len(row) == 10:
            proj1, proj2 = split_pair_expected(
                row[5],
                "projected_scores",
                "wnba",
                row,
            )

            over_line, under_line = (
                split_pair_expected(
                    row[7],
                    "over_under",
                    "wnba",
                    row,
                )
            )

            game.update({
                "proj_score_1": proj1,
                "proj_score_2": proj2,
                "total": normalize_cell(row[6]),
                "over_line": over_line,
                "under_line": under_line,
            })

        else:
            score1, score2 = split_pair_expected(
                row[5],
                "score",
                "wnba",
                row,
            )

            game["score1"] = score1
            game["score2"] = score2

        return game

    except Exception as exc:
        log(
            f"WNBA PARSE ERROR | "
            f"error={exc} | "
            f"row={json.dumps(row)}\n"
            f"{traceback.format_exc()}"
        )
        return None


def parse_row(row, sport):
    if sport == "wnba":
        return parse_wnba(row)

    return parse_nba_ncaa(row, sport)


def normalize_section_heading(text: str) -> str:
    text = " ".join(
        normalize_cell(text).lower().split()
    )

    if "games in progress" in text:
        return "in_progress"

    if (
        "completed games" in text
        or "final games" in text
    ):
        return "completed"

    if (
        "upcoming games" in text
        or "upcoming" == text
    ):
        return "upcoming"

    return ""


def scrape_page(page, url):
    """
    Read game rows together with their table section.

    A row is eligible for a prediction update only when
    its table belongs to the Upcoming Games section.

    Unknown table sections are fail-closed.
    """
    page.goto(
        url,
        wait_until="domcontentloaded",
        timeout=60000,
    )

    page.wait_for_selector(
        "table",
        timeout=60000,
    )

    extracted = page.evaluate(
        """() => {
            const nodes = Array.from(
                document.querySelectorAll(
                    'h1,h2,h3,h4,h5,h6,table'
                )
            );

            let section = '';
            const output = [];

            function classify(text) {
                const value = (text || '')
                    .toLowerCase()
                    .replace(/\\s+/g, ' ')
                    .trim();

                if (value.includes('games in progress')) {
                    return 'in_progress';
                }
                if (value.includes('completed games')) {
                    return 'completed';
                }
                if (value.includes('final games')) {
                    return 'completed';
                }
                if (value.includes('upcoming games')) {
                    return 'upcoming';
                }
                return '';
            }

            for (const node of nodes) {
                if (node.tagName !== 'TABLE') {
                    const detected = classify(
                        node.innerText || node.textContent
                    );
                    if (detected) {
                        section = detected;
                    }
                    continue;
                }

                const headers = Array.from(
                    node.querySelectorAll('thead th')
                ).map(cell =>
                    (cell.innerText || '').trim()
                );

                const rows = Array.from(
                    node.querySelectorAll('tbody tr')
                );

                for (const row of rows) {
                    const cells = Array.from(
                        row.querySelectorAll('td')
                    ).map(cell =>
                        (cell.innerText || '').trim()
                    );

                    output.push({
                        section: section,
                        headers: headers,
                        cells: cells
                    });
                }
            }

            return output;
        }"""
    )

    if not isinstance(extracted, list):
        raise RuntimeError(
            "DRatings table extraction returned "
            "invalid data"
        )

    results = []

    for index, item in enumerate(extracted):
        if not isinstance(item, dict):
            raise RuntimeError(
                "DRatings table extraction contained "
                "an invalid row"
            )

        cells = [
            normalize_cell(value)
            for value in item.get("cells", [])
        ]

        section = clean_section(
            item.get("section")
        )

        headers = [
            normalize_cell(value)
            for value in item.get("headers", [])
        ]

        log(
            "RAW TABLE ROW | "
            f"idx={index} | "
            f"section={section or 'unknown'} | "
            f"cell_count={len(cells)} | "
            f"headers={json.dumps(headers)} | "
            f"cells={json.dumps(cells)}"
        )

        results.append({
            "section": section,
            "headers": headers,
            "cells": cells,
        })

    return results


def clean_section(value: str) -> str:
    section = normalize_cell(value).lower()

    if section in {
        "upcoming",
        "in_progress",
        "completed",
    }:
        return section

    return "unknown"


def game_identity(game: dict) -> tuple:
    """
    The existing raw DRatings JSON does not contain
    ESPN game IDs.

    Therefore, identify games using the existing
    league, converted ET date/time, and two team names.
    """
    return (
        normalize_cell(
            game.get("sport")
        ).casefold(),
        normalize_cell(
            game.get("date_time")
        ).casefold(),
        strip_record(
            normalize_cell(game.get("team1"))
        ).casefold(),
        strip_record(
            normalize_cell(game.get("team2"))
        ).casefold(),
    )


def validate_game(game: dict) -> None:
    if not isinstance(game, dict):
        raise ValueError(
            "Saved DRatings game is not an object"
        )

    if any(
        not normalize_cell(game.get(field))
        for field in (
            "sport",
            "date_time",
            "team1",
            "team2",
        )
    ):
        raise ValueError(
            "DRatings game missing required identity"
        )

    # Preserve the existing JSON schema.
    if not set(GAME_FIELDS).issubset(game.keys()):
        missing = sorted(
            set(GAME_FIELDS) - set(game.keys())
        )
        raise ValueError(
            "Saved DRatings game missing fields: "
            f"{missing}"
        )


def validate_game_collection(
    games: list[dict],
) -> None:
    seen = set()

    for game in games:
        validate_game(game)

        key = game_identity(game)

        if key in seen:
            raise ValueError(
                "Duplicate DRatings game identity: "
                f"{key}"
            )

        seen.add(key)


def load_saved_games(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(
            f"Saved DRatings JSON is not a list: {path}"
        )

    validate_game_collection(data)

    log(
        f"EXISTING SNAPSHOT | "
        f"path={path} | games={len(data)}"
    )

    return data


def is_usable_upcoming_prediction(
    game: dict,
    sport: str,
    row: list[str],
) -> bool:
    """
    Ensure a candidate is a pregame prediction.

    The NBA/NCAAM upcoming table has projected scores
    and win probabilities.

    A completed score row must never be mistaken for
    a fresh pregame projection.
    """
    if not game:
        return False

    if game.get("score1") or game.get("score2"):
        return False

    if normalize_cell(game.get("game_status")):
        return False

    if not all(
        normalize_cell(game.get(field))
        for field in (
            "team1_win_pct",
            "team2_win_pct",
        )
    ):
        return False

    if sport in {"nba", "ncaa"}:
        if len(row) < 10:
            return False

        if not all(
            normalize_cell(game.get(field))
            for field in (
                "proj_score_1",
                "proj_score_2",
            )
        ):
            return False

    if sport == "wnba":
        if len(row) != 10:
            return False

        if not all(
            normalize_cell(game.get(field))
            for field in (
                "proj_score_1",
                "proj_score_2",
            )
        ):
            return False

    return True


def merge_upcoming_prediction(
    saved: dict,
    incoming: dict,
) -> dict:
    """
    Update only prediction-related fields.

    Preserve the original game identity and any
    other existing fields.
    """
    result = saved.copy()

    for field in PREDICTION_FIELDS:
        new_value = normalize_cell(
            incoming.get(field)
        )

        if new_value:
            result[field] = new_value

    # Never replace the identity of an existing game.
    for field in (
        "sport",
        "date_time",
        "team1",
        "team2",
    ):
        result[field] = saved[field]

    # A pregame prediction must not be replaced by
    # a live score or final score.
    for field in (
        "score1",
        "score2",
        "game_status",
    ):
        result[field] = saved[field]

    return result


def atomic_write_json(
    path: Path,
    games: list[dict],
) -> None:
    validate_game_collection(games)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:
        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                games,
                handle,
                indent=2,
                ensure_ascii=False,
            )
            handle.write("\n")

        os.replace(temp_path, path)

    finally:
        if temp_path.exists():
            temp_path.unlink()


def merge_scraped_games(
    existing: list[dict],
    scraped: list[dict],
    sport: str,
) -> tuple[list[dict], dict]:
    """
    Apply section-specific preservation rules.

    Only Upcoming Games rows may change predictions.
    In-progress, completed and unknown-section rows
    cannot change any saved record.
    """
    validate_game_collection(existing)

    by_key = {
        game_identity(game): game.copy()
        for game in existing
    }

    counters = {
        "new": 0,
        "updated": 0,
        "unchanged": 0,
        "protected_live": 0,
        "protected_completed": 0,
        "protected_unknown": 0,
        "skipped_invalid": 0,
    }

    seen_current = set()

    for item in scraped:
        section = item["section"]
        row = item["cells"]

        if not is_game_row(row):
            continue

        # The first columns (date and team names) are
        # shared by the different game table sections.
        # Do not parse live scores as predictions.
        if section != "upcoming":
            if section == "in_progress":
                counters["protected_live"] += 1
            elif section == "completed":
                counters["protected_completed"] += 1
            else:
                counters["protected_unknown"] += 1

            log(
                "PREGAME PREDICTION PROTECTED | "
                f"sport={sport} | "
                f"section={section} | "
                f"teams={normalize_cell(row[1])}"
            )
            continue

        parsed = parse_row(row, sport)

        if not is_usable_upcoming_prediction(
            parsed,
            sport,
            row,
        ):
            counters["skipped_invalid"] += 1

            log(
                "INVALID UPCOMING PREDICTION | "
                f"sport={sport} | "
                f"row={json.dumps(row)}"
            )
            continue

        validate_game(parsed)
        key = game_identity(parsed)

        if key in seen_current:
            raise ValueError(
                "Duplicate game on current DRatings "
                f"page: {key}"
            )

        seen_current.add(key)

        if key in by_key:
            merged = merge_upcoming_prediction(
                by_key[key],
                parsed,
            )

            if merged != by_key[key]:
                by_key[key] = merged
                counters["updated"] += 1

                log(
                    "PREGAME PREDICTION UPDATED | "
                    f"sport={sport} | "
                    f"game={key}"
                )

            else:
                counters["unchanged"] += 1

        else:
            by_key[key] = parsed
            counters["new"] += 1

            log(
                "NEW PREGAME PREDICTION | "
                f"sport={sport} | "
                f"game={key}"
            )

    # Existing games not returned on the latest page
    # remain in by_key, unchanged.
    games = list(by_key.values())

    validate_game_collection(games)

    return games, counters


def main():
    files_written = []
    errors = []

    run_date = datetime.now(ET).strftime(
        "%Y_%m_%d"
    )

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True
            )

            page = browser.new_page()

            page.set_extra_http_headers({
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            })

            try:
                for sport, url in URLS.items():
                    log(
                        f"Scraping {sport.upper()}"
                    )

                    try:
                        label = (
                            "ncaam"
                            if sport == "ncaa"
                            else sport
                        )

                        out_dir = (
                            OUTPUT_ROOT / label
                        )

                        path = (
                            out_dir
                            / f"{run_date}_{label}_raw.json"
                        )

                        # Read and validate the prior
                        # snapshot before collecting new data.
                        existing = load_saved_games(
                            path
                        )

                        scraped = scrape_page(
                            page,
                            url,
                        )

                        log(
                            f"RAW ROWS | "
                            f"sport={sport} | "
                            f"count={len(scraped)}"
                        )

                        if not scraped:
                            raise RuntimeError(
                                f"{sport.upper()} returned "
                                "zero table rows"
                            )

                        games, counts = (
                            merge_scraped_games(
                                existing,
                                scraped,
                                sport,
                            )
                        )

                        if not games:
                            raise RuntimeError(
                                f"{sport.upper()} returned "
                                "no usable pregame "
                                "predictions and has "
                                "no saved predictions"
                            )

                        # No write when the current page
                        # does not change any saved record.
                        # This protects the original JSON
                        # bytes and prevents needless churn.
                        if games != existing:
                            atomic_write_json(
                                path,
                                games,
                            )

                            log(
                                f"WROTE {path} "
                                f"({len(games)} games)"
                            )

                        else:
                            log(
                                f"UNCHANGED {path} "
                                f"({len(games)} games)"
                            )

                        files_written.append(
                            (str(path), len(games))
                        )

                        log(
                            "SCRAPE SUMMARY | "
                            f"sport={sport} | "
                            f"existing={len(existing)} | "
                            f"final={len(games)} | "
                            f"new={counts['new']} | "
                            f"updated={counts['updated']} | "
                            f"unchanged={counts['unchanged']} | "
                            f"protected_live="
                            f"{counts['protected_live']} | "
                            f"protected_completed="
                            f"{counts['protected_completed']} | "
                            f"protected_unknown="
                            f"{counts['protected_unknown']} | "
                            f"invalid="
                            f"{counts['skipped_invalid']}"
                        )

                    except Exception as exc:
                        errors.append(
                            f"{sport}: {exc}"
                        )

                        log(
                            f"ERROR scraping {sport}: "
                            f"{exc}\n"
                            f"{traceback.format_exc()}"
                        )

                    time.sleep(
                        random.uniform(2, 4)
                    )

            finally:
                browser.close()

        log("--- SUMMARY ---")

        log(
            f"Files processed: "
            f"{len(files_written)}"
        )

        for path, count in files_written:
            log(
                f"FILE: {path} "
                f"({count} games)"
            )

        if errors:
            log(
                "STATUS: FAILED | "
                + "; ".join(errors)
            )

            raise RuntimeError(
                "DRatings scrape failed for: "
                + "; ".join(errors)
            )

        log("STATUS: SUCCESS")

    except Exception as exc:
        log(
            f"FATAL ERROR: {exc}\n"
            f"{traceback.format_exc()}"
        )

        log("STATUS: FAILED")
        raise

    print(
        "Basketball drat scraper complete."
    )


if __name__ == "__main__":
    main()
