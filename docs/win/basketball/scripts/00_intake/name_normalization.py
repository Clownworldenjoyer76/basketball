#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/name_normalization.py

import sys
import csv
import re
import unicodedata
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from basketball_shared import resolve_repository_path


# =========================
# LOGGER UTILITY
# =========================

def _write_signal_summary(summary_dir, ts, df):
    play_cols = [
        column
        for column in ("home_play", "away_play", "over_play", "under_play")
        if column in df.columns
    ]
    if not play_cols:
        return

    signals = df[df[play_cols].any(axis=1)].copy()
    if signals.empty:
        return

    base_cols = ["game_date", "home_team", "away_team"]
    edge_cols = [column for column in df.columns if "edge_pct" in column]
    final_cols = [column for column in base_cols + edge_cols if column in signals.columns]

    summary_path = summary_dir / "condensed_summary.txt"
    with summary_path.open("a", encoding="utf-8") as audit_handle:
        audit_handle.write(f"\n--- BETTING SIGNALS: {ts} ---\n")
        audit_handle.write(signals[final_cols].to_string(index=False))
        audit_handle.write("\n" + "=" * 30 + "\n")


def audit(log_path, stage, status, msg="", df=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = Path(log_path)
    safe_path = resolve_repository_path(log_path, strict=False)
    safe_path.parent.mkdir(parents=True, exist_ok=True)

    with safe_path.open("a", encoding="utf-8") as audit_handle:
        audit_handle.write(f"\n[{ts}] [{stage}] {status}\n")
        if msg:
            audit_handle.write(f"  MSG: {msg}\n")
        if df is not None and isinstance(df, pd.DataFrame):
            audit_handle.write(f"  STATS: {len(df)} rows | {len(df.columns)} cols\n")
            audit_handle.write(f"  NULLS: {df.isnull().sum().sum()} total\n")
            audit_handle.write(f"  SAMPLE:\n{df.head(3).to_string(index=False)}\n")
        audit_handle.write("-" * 40 + "\n")

    if df is not None and isinstance(df, pd.DataFrame):
        _write_signal_summary(safe_path.parent, ts, df)


# =========================
# PATHS
# =========================

SPORTSBOOK_DIR = Path("docs/win/basketball/00_intake/sportsbook")
PREDICTIONS_DIR = Path("docs/win/basketball/00_intake/predictions")
DAILY_GAMES_DIR = Path("docs/win/basketball/daily_games")
NBA_MAP_FILE = Path("docs/win/basketball/maps/team_map_nba.csv")
NCAAM_MAP_FILE = Path("docs/win/basketball/maps/team_map_ncaam.csv")
WNBA_MAP_FILE = Path("docs/win/basketball/maps/team_map_wnba.csv")
NO_MAP_DIR = Path("docs/win/basketball/maps/no_map")
NO_MAP_DIR.mkdir(parents=True, exist_ok=True)
NO_MAP_FILE = NO_MAP_DIR / "no_map_basketball.csv"
ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "name_normalization_log.txt"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("")


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as log_handle:
        log_handle.write(f"{datetime.now(timezone.utc).isoformat()} | {msg}\n")


# =========================
# LOAD TEAM MAPS
# =========================

def _team_key(league: str, value: str) -> str:
    """Use the same NCAAM accent/punctuation folding as game-ID assignment."""
    name = str(value or "").strip()
    if league != "ncaam":
        return name.lower()
    name = unicodedata.normalize("NFKD", name).casefold()
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    return " ".join(re.findall(r"[a-z0-9]+", name))


team_map = {}


def load_map(map_file: Path):
    if not map_file.exists():
        log(f"WARNING: {map_file.name} not found")
        return

    with map_file.open(newline="", encoding="utf-8-sig") as map_handle:
        map_reader = csv.DictReader(map_handle)
        for number, map_row in enumerate(map_reader, start=2):
            map_league = (map_row.get("league") or "").strip().lower()
            alias = (map_row.get("alias") or "").strip()
            canonical_team = (map_row.get("canonical_team") or "").strip()
            if not map_league or not alias or not canonical_team:
                continue
            key = (map_league, _team_key(map_league, alias))
            existing = team_map.get(key)
            if existing is not None and existing != canonical_team:
                raise ValueError(
                    f"Conflicting team alias {alias!r} ({map_league}) in "
                    f"{map_file}:{number}: {existing!r} vs {canonical_team!r}"
                )
            team_map[key] = canonical_team

            # Canonical names must normalize to themselves as well.
            canonical_key = (map_league, _team_key(map_league, canonical_team))
            canonical_existing = team_map.get(canonical_key)
            if canonical_existing is not None and canonical_existing != canonical_team:
                raise ValueError(
                    f"Conflicting canonical team {canonical_team!r} in {map_file}:{number}"
                )
            team_map[canonical_key] = canonical_team


load_map(NBA_MAP_FILE)
load_map(NCAAM_MAP_FILE)
load_map(WNBA_MAP_FILE)


# =========================
# TARGET FILES: SPORTSBOOK, PREDICTIONS, AND DAILY GAMES
# =========================

target_files = {}
for league in ("nba", "ncaam", "wnba"):
    for f in sorted((SPORTSBOOK_DIR / league).glob("*_odds.csv")):
        target_files[f] = league
    for f in sorted((PREDICTIONS_DIR / league).glob("*_predictions.csv")):
        target_files[f] = league
    for f in sorted((DAILY_GAMES_DIR / league).glob("*.csv")):
        target_files[f] = league


# =========================
# PROCESS FILES
# =========================

unmapped = set()
files_processed = 0
rows_processed = 0
rows_updated = 0

for csv_file, folder_league in sorted(target_files.items()):
    files_processed += 1
    updated_rows = []
    modified = False

    with csv_file.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows_processed += 1
            row_league = (row.get("league") or "").strip().lower()
            league = row_league or folder_league
            if row_league and row_league != folder_league:
                log(
                    f"WARNING: league={row_league!r} disagrees with folder "
                    f"league={folder_league!r} in {csv_file}; row unchanged"
                )
                updated_rows.append(row)
                continue

            for side in ("home_team", "away_team"):
                team = (row.get(side) or "").strip()
                if not team:
                    continue
                key = (league, _team_key(league, team))
                canonical = team_map.get(key)
                if canonical is not None:
                    if row.get(side) != canonical:
                        row[side] = canonical
                        modified = True
                        rows_updated += 1
                else:
                    unmapped.add((league, team))
            updated_rows.append(row)

    if modified and fieldnames:
        with csv_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)

    if updated_rows:
        audit(
            LOG_FILE,
            "NORMALIZATION",
            "SUCCESS",
            msg=f"Processed {csv_file} | changed={modified}",
            df=pd.DataFrame(updated_rows),
        )


# =========================
# WRITE UNMAPPED
# =========================

existing = set()
if NO_MAP_FILE.exists():
    with NO_MAP_FILE.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "league" in reader.fieldnames and "team" in reader.fieldnames:
            for row in reader:
                league = (row.get("league") or "").strip().lower()
                team = (row.get("team") or "").strip()
                if league and team:
                    existing.add((league, team))

new_only = unmapped - existing
combined = existing | unmapped
with NO_MAP_FILE.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["league", "team"])
    for league, team in sorted(combined):
        writer.writerow([league, team])


# =========================
# LOG SUMMARY
# =========================

log(
    f"SUMMARY: files_processed={files_processed}, "
    f"rows_processed={rows_processed}, rows_updated={rows_updated}, "
    f"unmapped_found={len(unmapped)}, unmapped_new_added={len(new_only)}"
)
print("Basketball name normalization complete.")
