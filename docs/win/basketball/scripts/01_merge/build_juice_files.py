#!/usr/bin/env python3
# docs/win/basketball/scripts/01_merge/build_juice_files.py
"""Incremental/full-rebuild launcher for build_juice_files_core.py.

Launcher log: docs/win/basketball/errors/00_intake/build_juice_files.txt
Core log:     docs/win/basketball/errors/01_merge/build_juice_files.txt
"""
from __future__ import annotations
import importlib.util, os, shutil, sys, tempfile, traceback
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
ROOT = SCRIPTS_DIR.parents[3]  # repository root (docs/win/basketball/scripts/01_merge/ -> repo root)
BASE = ROOT / "docs/win/basketball"
CORE_PATH = Path(__file__).resolve().with_name("build_juice_files_core.py")
REAL_INPUT = BASE / "01_merge"
REAL_OUTPUT = BASE / "01_merge/01_merguiced"
ERROR_DIR = BASE / "errors/00_intake"
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "build_juice_files.txt"
LEAGUES = ["nba", "ncaam", "wnba"]
MARKETS = ["moneyline", "spread", "total"]


def log(message: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now().isoformat()} | {message}\n")


def start_log() -> None:
    LOG_FILE.write_text(f"=== build_juice_files launcher RUN {datetime.now().isoformat()} ===\n", encoding="utf-8")


try:
    from zoneinfo import ZoneInfo

    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    from basketball_shared import resolve_repository_path

    NY = ZoneInfo("America/New_York")
except Exception as exc:  # noqa: BLE001
    start_log()
    log(f"FATAL ERROR: import/setup failed: {exc}\n{traceback.format_exc()}")
    log("STATUS: FAILED")
    sys.exit(1)


def truthy(v): return str(v or "").strip().lower() in {"1", "true", "yes", "on"}


def load_core():
    spec = importlib.util.spec_from_file_location("build_juice_files_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {CORE_PATH}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read_core_log(core) -> str:
    return core.LOG_FILE.read_text(encoding="utf-8", errors="replace")


def core_failed(text: str) -> bool:
    return "STATUS: FAILED" in text or "STATUS: COMPLETED WITH ERRORS" in text or "ERROR processing" in text


def run_core(core) -> int:
    """Run core.main(); return its exit code (0 if it returned normally)."""
    try:
        core.main()
        return 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        log(f"Core exited with code {code}")
        return code


def run_full(core) -> int:
    code = run_core(core)
    text = read_core_log(core)
    return 1 if code or core_failed(text) else 0


def run_incremental(core) -> int:
    date = datetime.now(NY).strftime("%Y_%m_%d")
    log(f"Incremental mode: {date}")
    with tempfile.TemporaryDirectory(prefix="basketball_juice_") as td:
        root = Path(td)
        inp = root / "input"
        out = root / "output"
        has_input = set()
        for lg in LEAGUES:
            up = lg.upper()
            for market in MARKETS:
                src = REAL_INPUT / lg / market / f"{date}_{up}_{market}.csv"
                if src.exists():
                    dst = inp / lg / market / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    has_input.add((lg, market))
                else:
                    log(f"NO MERGE INPUT | {up} {market} {date}")
        core.INPUT_DIR = inp
        core.OUTPUT_DIR = out
        code = run_core(core)
        text = read_core_log(core)
        with_errors = "STATUS: COMPLETED WITH ERRORS" in text
        if "STATUS: FAILED" in text or (code and not with_errors):
            log("Core failed; existing outputs left unchanged.")
            return 1
        for lg in LEAGUES:
            up = lg.upper()
            for market in MARKETS:
                dest = resolve_repository_path(REAL_OUTPUT / lg / market / f"{date}_{up}_{market}.csv", strict=False)
                src = out / lg / market / dest.name
                if src.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
                    log(f"COPIED {dest.name}")
                elif (lg, market) not in has_input:
                    if dest.exists():
                        dest.unlink()
                        log(f"REMOVED {dest.name} | no merge input exists for today")
                else:
                    log(f"KEPT existing {dest.name} | merge input exists but no output was built")
    return 1 if (code or core_failed(text)) else 0


def main():
    start_log()
    full = truthy(os.getenv("BASKETBALL_FULL_REBUILD"))
    log(f"Mode: {'full_rebuild' if full else 'incremental_current_date'}")
    try:
        core = load_core()
    except Exception as exc:  # noqa: BLE001
        log(f"FATAL ERROR: could not load core (config or import error): {exc}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        sys.exit(1)
    try:
        failed = run_full(core) if full else run_incremental(core)
    except Exception as exc:  # noqa: BLE001
        log(f"FATAL ERROR: {exc}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        sys.exit(1)
    log(f"STATUS: {'FAILED' if failed else 'SUCCESS'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
