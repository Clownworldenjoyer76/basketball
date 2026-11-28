#!/usr/bin/env python3
# docs/win/basketball/scripts/04_select/basketball_select_bets.py
"""Rebuild historical and current basketball selections using today's filters.

Every run reads all stored stage-3 EV/Kelly files, applies the current
markets.yaml and model_config.yaml rules through basketball_select_bets_core.py,
and replaces only the rebuildable 04_select/{league}/daily_picks files.

Immutable 04_select/{league}/locked_picks are never modified. Replacements are
staged and published only after the core reports a successful selection.
BASKETBALL_FULL_REBUILD continues to control other pipeline stages, but the
selection stage always re-filters all available historical dates.

Launcher log: docs/win/basketball/errors/00_intake/basketball_select_bets.txt
Core log:     docs/win/basketball/errors/04_select/select_bets.txt
"""
from __future__ import annotations

import csv
import importlib.util
import re
import shutil
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
ROOT = SCRIPTS_DIR.parents[3]  # repository root (docs/win/basketball/scripts/04_select/ -> repo root)
BASE = ROOT / "docs/win/basketball"
CORE_PATH = Path(__file__).resolve().with_name("basketball_select_bets_core.py")
REAL_INPUT = BASE / "03_edges/ev_kelly"
REAL_SELECT = BASE / "04_select"
ERROR_DIR = BASE / "errors/00_intake"
LOG_FILE = ERROR_DIR / "basketball_select_bets.txt"
LEAGUES = ("nba", "ncaam", "wnba")
MARKETS = ("moneyline", "spread", "total")
DATE_PATTERN = re.compile(r"^\d{4}_\d{2}_\d{2}$")


def log(message: str, level: str = "INFO") -> None:
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now().isoformat()} | {level:<5} | {message.rstrip()}\n")


def start_log() -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(
        f"=== basketball_select_bets launcher RUN {datetime.now().isoformat()} ===\n",
        encoding="utf-8",
    )


def load_core():
    spec = importlib.util.spec_from_file_location(
        "basketball_select_bets_core", CORE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def remove_leftover_work_folders() -> None:
    """Remove temporary folders left behind by a killed earlier run."""
    if not REAL_SELECT.exists():
        return
    for leftover in REAL_SELECT.glob(".selection_rebuild_*"):
        if leftover.is_dir():
            shutil.rmtree(leftover)
            log(f"REMOVED leftover work folder: {leftover}", "WARN")


def verify_historical_sources() -> int:
    """Do not discard an old selection if its original market input vanished.

    Returns the number of stored selected rows that were checked.
    """
    available = set()
    for league in LEAGUES:
        for market in MARKETS:
            folder = REAL_INPUT / league / market
            for source in folder.glob("*.csv"):
                prefix = source.name[:10]
                if DATE_PATTERN.fullmatch(prefix) and source.name == (
                    f"{prefix}_{league.upper()}_{market}.csv"
                ):
                    available.add((league, prefix, market))

    if not available:
        raise RuntimeError(f"No historical or current EV/Kelly files in {REAL_INPUT}")

    checked = 0
    for league in LEAGUES:
        folder = REAL_SELECT / league / "daily_picks"
        for selected in folder.glob("*_selected.csv"):
            with selected.open(newline="", encoding="utf-8-sig") as handle:
                for row_number, row in enumerate(csv.DictReader(handle), start=2):
                    game_date = str(row.get("game_date") or "").strip().replace("-", "_")
                    market = str(row.get("market_type") or "").strip().lower()
                    if (league, game_date, market) not in available:
                        raise RuntimeError(
                            "Cannot re-filter historical selection: missing "
                            f"EV/Kelly input for {league}/{game_date}/{market}; "
                            f"file={selected}:{row_number}"
                        )
                    checked += 1
    log(f"VERIFIED: {len(available)} EV/Kelly input files; {checked} stored selected rows all have their input")
    return checked


def publish_rebuilt_daily_picks(staged: Path, backups: Path) -> None:
    """Swap only daily_picks, restoring old directories if publication fails."""
    swaps = []
    try:
        for league in LEAGUES:
            destination = REAL_SELECT / league / "daily_picks"
            source = staged / league / "daily_picks"
            backup = backups / league / "daily_picks"
            source.mkdir(parents=True, exist_ok=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup.parent.mkdir(parents=True, exist_ok=True)

            had_original = destination.exists()
            if had_original:
                destination.rename(backup)
            try:
                source.rename(destination)
            except Exception:
                if had_original:
                    backup.rename(destination)
                raise
            swaps.append((destination, backup, had_original))
            log(f"PUBLISHED {destination}")
    except Exception:
        for destination, backup, had_original in reversed(swaps):
            if destination.exists():
                shutil.rmtree(destination)
            if had_original:
                backup.rename(destination)
        log("PUBLISH FAILED: previous daily_picks folders restored", "ERROR")
        raise


def main() -> None:
    start_log()
    log("Mode: re-filter all historical and current dates")
    log(f"INPUT : {REAL_INPUT}")
    log(f"OUTPUT: {REAL_SELECT}")

    REAL_SELECT.mkdir(parents=True, exist_ok=True)
    remove_leftover_work_folders()
    verify_historical_sources()
    core = load_core()
    log(f"LOADED core: {CORE_PATH}")

    # Keeping this temporary directory under REAL_SELECT permits directory
    # renames on the same filesystem, including Windows drive-backed repos.
    with tempfile.TemporaryDirectory(prefix=".selection_rebuild_", dir=REAL_SELECT) as work:
        root = Path(work)
        stage = root / "new"
        core.INPUT_DIR = REAL_INPUT
        core.SELECT_DIR = stage
        core.main()

        log(f"Core log: {core.LOG_FILE}")
        if not core.LOG_FILE.is_file():
            raise RuntimeError(f"Selection log missing: {core.LOG_FILE}")
        statuses = [
            line for line in core.LOG_FILE.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if "STATUS:" in line
        ]
        log(f"Core final status line: {statuses[-1].strip() if statuses else '(none)'}")
        if not statuses or "STATUS: SUCCESS" not in statuses[-1]:
            raise RuntimeError(
                f"Selection failed; historical files preserved. Log: {core.LOG_FILE}"
            )

        publish_rebuilt_daily_picks(stage, root / "backup")

    log("Original locked_picks were not changed.")
    log("STATUS: SUCCESS")
    print("Historical and current daily selections rebuilt using the current filters.")
    print("Original locked_picks were not changed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            log(f"FATAL ERROR: {exc}\n{traceback.format_exc()}", "ERROR")
            log("STATUS: FAILED")
        except Exception:  # noqa: BLE001
            pass
        print(f"STATUS: FAILED | {exc}", file=sys.stderr)
        sys.exit(1)
