#!/usr/bin/env python3
"""Failure-signaling launcher for basketball_game_id_core.py.

The core scans and repairs full history. After successful completion, this
launcher replaces the historical mismatch listing in the operational log with
current New York date results based on the core's canonical team/date keys and
unambiguous schedule game-ID rules.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

CORE_PATH = Path(__file__).with_name("basketball_game_id_core.py")
BASE = Path("docs/win/basketball")
NY = ZoneInfo("America/New_York")
LABELS = {"nba": "NBA", "ncaam": "NCAAM", "wnba": "WNBA"}


def load_core():
    spec = importlib.util.spec_from_file_location("basketball_game_id_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean(value) -> str:
    return "" if value is None else str(value).strip()


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def key(row: dict, league_label: str, core):
    """Use the same canonical match key as the ID-injection core."""
    return core.make_key(
        row.get("game_date"),
        row.get("home_team"),
        row.get("away_team"),
        league_label,
    )


def _issue(row: dict, league_label: str, source: str, reason: str) -> dict:
    return {
        "league": league_label,
        "source": source,
        "reason": reason,
        **row,
    }


def unique_schedule(rows: list[dict], league_label: str, core):
    """Exclude nonunique game IDs, but permit repeated identical schedule exports."""
    entries = defaultdict(list)
    issues = []
    for row in rows:
        k = key(row, league_label, core)
        gid = core.clean_value(row.get("game_id"))
        if k is None or not gid:
            issues.append(_issue(
                row, league_label, "daily_games", "invalid_identity_or_missing_schedule_id"
            ))
            continue
        entries[k].append(row)

    id_to_keys = defaultdict(set)
    for k, candidates in entries.items():
        for row in candidates:
            id_to_keys[core.clean_value(row.get("game_id"))].add(k)

    unique = {}
    for k, candidates in entries.items():
        ids = {core.clean_value(row.get("game_id")) for row in candidates}
        ambiguous = len(ids) != 1 or any(len(id_to_keys[gid]) != 1 for gid in ids)
        if ambiguous:
            issues.extend(
                _issue(row, league_label, "daily_games", "ambiguous_schedule_ids_or_game_id_reused")
                for row in candidates
            )
        else:
            unique[k] = candidates[0]
    return unique, issues


def unique_predictions(rows: list[dict], league_label: str, core):
    """Duplicate/invalid prediction matchups are not counted as matched."""
    keys = [key(row, league_label, core) for row in rows]
    counts = Counter(k for k in keys if k is not None)
    unique = {}
    unmatched = []
    issues = []
    for row, k in zip(rows, keys):
        if k is None:
            unmatched.append(_issue(row, league_label, "predictions", "invalid_date_or_team"))
        elif counts[k] > 1:
            issues.append(_issue(row, league_label, "predictions", "duplicate_source_matchup"))
            unmatched.append(_issue(row, league_label, "predictions", "ambiguous_duplicate_prediction_matchup"))
        else:
            unique[k] = row
    return unique, unmatched, issues


def current_mismatches(core, game_date: str | None = None):
    game_date = game_date or datetime.now(NY).strftime("%Y_%m_%d")
    daily_missing = []
    prediction_missing = []
    duplicates = []

    for league, label in LABELS.items():
        daily_path = BASE / f"daily_games/{league}/{game_date}_{label}.csv"
        pred_path = BASE / f"00_intake/predictions/{league}/{game_date}_{label}_predictions.csv"
        daily_map, daily_issues = unique_schedule(read_rows(daily_path), label, core)
        pred_map, pred_invalid, pred_issues = unique_predictions(read_rows(pred_path), label, core)
        duplicates.extend(daily_issues)
        duplicates.extend(pred_issues)
        prediction_missing.extend({**row, "source_file": str(pred_path)} for row in pred_invalid)

        for k in sorted(set(daily_map) - set(pred_map)):
            daily_missing.append({"league": label, **daily_map[k], "source_file": str(daily_path)})

        for k in sorted(set(pred_map) - set(daily_map)):
            reason = (
                "reversed_home_away_not_auto_assigned"
                if (k[0], k[2], k[1]) in daily_map
                else "no_validated_matchup_alias_or_schedule_game"
            )
            prediction_missing.append({
                "league": label,
                **pred_map[k],
                "source_file": str(pred_path),
                "reason": reason,
            })

    return game_date, daily_missing, prediction_missing, duplicates


def _rewrite_mismatch_counts(lines, daily_missing_count, prediction_missing_count, duplicate_count):
    replacements = (
        (
            "Sportsbook rows with no prediction match:",
            "Current-date daily games with no prediction match:",
            daily_missing_count,
        ),
        (
            "Prediction rows with no sportsbook match:",
            "Current-date prediction rows with no daily-game match:",
            prediction_missing_count,
        ),
        (
            "Duplicate key rows:",
            "Current-date ambiguous or duplicate key rows:",
            duplicate_count,
        ),
    )
    for idx, line in enumerate(lines):
        if " | " not in line:
            continue
        for old_label, new_label, count in replacements:
            if old_label in line:
                prefix = line.split(old_label, 1)[0]
                lines[idx] = f"{prefix}{new_label} {count}"
                break


def _find_mismatch_status(lines):
    marker_index = next(
        (index for index, line in enumerate(lines)
         if "--- SMALL MISMATCH REPORT ---" in line),
        None,
    )
    if marker_index is None:
        raise RuntimeError("Unable to locate mismatch report marker in basketball_game_id log")
    status_index = next(
        (index for index in range(len(lines) - 1, marker_index, -1)
         if "STATUS:" in lines[index]),
        None,
    )
    if status_index is None:
        raise RuntimeError("Unable to locate final status in basketball_game_id log")
    return marker_index, status_index


def _append_mismatch_section(report, title, rows, formatter):
    report.append(f"{datetime.now().isoformat()} | {title}")
    if not rows:
        report.append(f"{datetime.now().isoformat()} |   none")
        return
    for row in rows:
        report.append(f"{datetime.now().isoformat()} |   {formatter(row)}")


def rewrite_operational_log(log_path: Path, original_text: str, core) -> None:
    game_date, daily_missing, prediction_missing, duplicates = current_mismatches(core)
    lines = original_text.splitlines()
    _rewrite_mismatch_counts(
        lines, len(daily_missing), len(prediction_missing), len(duplicates)
    )
    marker_index, status_index = _find_mismatch_status(lines)
    prefix_lines = lines[:marker_index]
    status_line = lines[status_index]
    report = [
        f"{datetime.now().isoformat()} | --- CURRENT-DATE MISMATCH REPORT ---",
        f"{datetime.now().isoformat()} | Operational mismatch date: {game_date} (America/New_York)",
        f"{datetime.now().isoformat()} | Matching: canonical date/team aliases; unique game ID; home/away orientation retained",
    ]

    _append_mismatch_section(
        report,
        "Current-date daily games with no prediction match:",
        daily_missing,
        lambda row: (
            f"{row['league']} | {clean(row.get('game_date'))} | "
            f"{clean(row.get('home_team'))} | {clean(row.get('away_team'))} | "
            f"game_id={clean(row.get('game_id'))}"
        ),
    )
    _append_mismatch_section(
        report,
        "Current-date prediction rows with no daily-game match:",
        prediction_missing,
        lambda row: (
            f"{row['league']} | {clean(row.get('game_date'))} | "
            f"{clean(row.get('home_team'))} | {clean(row.get('away_team'))} | "
            f"reason={clean(row.get('reason'))} | file={row.get('source_file', '')}"
        ),
    )
    _append_mismatch_section(
        report,
        "Current-date ambiguous or duplicate key rows:",
        duplicates,
        lambda row: (
            f"{row['league']} | {clean(row.get('game_date'))} | "
            f"{clean(row.get('home_team'))} | {clean(row.get('away_team'))} | "
            f"reason={clean(row.get('reason'))} | source={row['source']}"
        ),
    )
    log_path.write_text("\n".join(prefix_lines + report + [status_line]) + "\n", encoding="utf-8")


def main() -> None:
    core = load_core()
    core.main()
    original_text = core.LOG_FILE.read_text(encoding="utf-8", errors="replace")
    fatal_markers = ("ERROR loading", "ERROR updating", "FATAL ERROR", "STATUS: FAILED")
    if any(marker in original_text for marker in fatal_markers):
        sys.exit(1)
    rewrite_operational_log(core.LOG_FILE, original_text, core)


if __name__ == "__main__":
    main()
