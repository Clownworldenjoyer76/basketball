#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/name_normalization.py

import sys
import csv
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
        for column in (
            "home_play",
            "away_play",
            "over_play",
            "under_play",
        )
        if column in df.columns
    ]
    if not play_cols:
        return

    signals = df[df[play_cols].any(axis=1)].copy()
    if signals.empty:
        return

    base_cols = ["game_date", "home_team", "away_team"]
    edge_cols = [
        column
        for column in df.columns
        if "edge_pct" in column
    ]
    final_cols = [
        column
        for column in base_cols + edge_cols
        if column in signals.columns
    ]

    summary_path = summary_dir / "condensed_summary.txt"
    with summary_path.open("a", encoding="utf-8") as audit_handle:
        audit_handle.write(f"\n--- BETTING SIGNALS: {ts} ---\n")
        audit_handle.write(signals[final_cols].to_string(index=False))
        audit_handle.write("\n" + "=" * 30 + "\n")


def audit(log_path, stage, status, msg="", df=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = Path(log_path)
    safe_path = resolve_repository_path(
        log_path,
        strict=False,
    )

    safe_path.parent.mkdir(parents=True, exist_ok=True)

    with safe_path.open("a", encoding="utf-8") as audit_handle:
        audit_handle.write(f"\n[{ts}] [{stage}] {status}\n")
        if msg:
            audit_handle.write(f"  MSG: {msg}\n")
        if df is not None and isinstance(df, pd.DataFrame):
            audit_handle.write(
                f"  STATS: {len(df)} rows | {len(df.columns)} cols\n"
            )
            audit_handle.write(f"  NULLS: {df.isnull().sum().sum()} total\n")
            audit_handle.write(
                f"  SAMPLE:\n{df.head(3).to_string(index=False)}\n"
            )
        audit_handle.write("-" * 40 + "\n")

    if df is not None and isinstance(df, pd.DataFrame):
        _write_signal_summary(safe_path.parent, ts, df)


# =========================
# PATHS
# =========================

SPORTSBOOK_DIR = Path(
    "docs/win/basketball/00_intake/sportsbook"
)

PREDICTIONS_DIR = Path(
    "docs/win/basketball/00_intake/predictions"
)

NBA_MAP_FILE = Path(
    "docs/win/basketball/maps/team_map_nba.csv"
)

NCAAM_MAP_FILE = Path(
    "docs/win/basketball/maps/team_map_ncaam.csv"
)

WNBA_MAP_FILE = Path(
    "docs/win/basketball/maps/team_map_wnba.csv"
)

NO_MAP_DIR = Path(
    "docs/win/basketball/maps/no_map"
)

NO_MAP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NO_MAP_FILE = (
    NO_MAP_DIR
    / "no_map_basketball.csv"
)

ERROR_DIR = Path(
    "docs/win/basketball/errors/00_intake"
)

ERROR_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE = (
    ERROR_DIR
    / "name_normalization_log.txt"
)

with open(
    LOG_FILE,
    "w",
    encoding="utf-8",
) as f:
    f.write("")


def log(msg: str) -> None:
    with open(
        LOG_FILE,
        "a",
        encoding="utf-8",
    ) as log_handle:
        log_handle.write(
            f"{datetime.now(timezone.utc).isoformat()} | "
            f"{msg}\n"
        )


# =========================
# LOAD TEAM MAPS
# =========================

team_map = {}


def load_map(map_file: Path):
    if not map_file.exists():
        log(
            f"WARNING: {map_file.name} not found"
        )
        return

    with open(
        map_file,
        newline="",
        encoding="utf-8",
    ) as map_handle:
        map_reader = csv.DictReader(map_handle)

        for map_row in map_reader:
            map_league = (
                map_row.get("league", "")
                .strip()
                .lower()
            )

            alias = (
                map_row.get("alias", "")
                .strip()
                .lower()
            )

            canonical_team = (
                map_row.get("canonical_team", "")
                .strip()
            )

            if map_league and alias and canonical_team:
                team_map[
                    (map_league, alias)
                ] = canonical_team


load_map(NBA_MAP_FILE)
load_map(NCAAM_MAP_FILE)
load_map(WNBA_MAP_FILE)


# =========================
# TARGET FILES ONLY
# =========================

target_files = []

for f in (
    SPORTSBOOK_DIR
    / "nba"
).glob("*_odds.csv"):
    target_files.append(f)

for f in (
    SPORTSBOOK_DIR
    / "ncaam"
).glob("*_odds.csv"):
    target_files.append(f)

for f in (
    SPORTSBOOK_DIR
    / "wnba"
).glob("*_odds.csv"):
    target_files.append(f)

for f in (
    PREDICTIONS_DIR
    / "nba"
).glob("*_predictions.csv"):
    target_files.append(f)

for f in (
    PREDICTIONS_DIR
    / "ncaam"
).glob("*_predictions.csv"):
    target_files.append(f)

for f in (
    PREDICTIONS_DIR
    / "wnba"
).glob("*_predictions.csv"):
    target_files.append(f)


# =========================
# PROCESS FILES
# =========================

unmapped = set()
files_processed = 0
rows_processed = 0
rows_updated = 0

for csv_file in target_files:
    files_processed += 1
    updated_rows = []
    modified = False

    with open(
        csv_file,
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        for row in reader:
            rows_processed += 1

            league = (
                row.get("league", "")
                .strip()
                .lower()
            )

            for side in [
                "home_team",
                "away_team",
            ]:
                team = (
                    row.get(side, "")
                    .strip()
                )

                if not team:
                    continue

                key = (
                    league,
                    team.lower(),
                )

                if key in team_map:
                    canonical = team_map[key]

                    if row.get(side) != canonical:
                        row[side] = canonical
                        modified = True
                        rows_updated += 1

                else:
                    unmapped.add(
                        (league, team)
                    )

            updated_rows.append(row)

    if modified and fieldnames:
        with open(
            csv_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(updated_rows)

    if updated_rows:
        df_audit = pd.DataFrame(
            updated_rows
        )

        audit(
            LOG_FILE,
            "NORMALIZATION",
            "SUCCESS",
            msg=f"Processed {csv_file.name}",
            df=df_audit,
        )


# =========================
# WRITE UNMAPPED
# =========================

existing = set()

if NO_MAP_FILE.exists():
    with open(
        NO_MAP_FILE,
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        if (
            reader.fieldnames
            and "league" in reader.fieldnames
            and "team" in reader.fieldnames
        ):
            for row in reader:
                league = (
                    row.get("league") or ""
                ).strip().lower()

                team = (
                    row.get("team") or ""
                ).strip()

                if league and team:
                    existing.add(
                        (league, team)
                    )

new_only = unmapped - existing
combined = existing | unmapped

with open(
    NO_MAP_FILE,
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)

    writer.writerow([
        "league",
        "team",
    ])

    for league, team in sorted(combined):
        writer.writerow([
            league,
            team,
        ])


# =========================
# LOG SUMMARY
# =========================

log(
    f"SUMMARY: "
    f"files_processed={files_processed}, "
    f"rows_processed={rows_processed}, "
    f"rows_updated={rows_updated}, "
    f"unmapped_found={len(unmapped)}, "
    f"unmapped_new_added={len(new_only)}"
)

print(
    "Basketball name normalization complete."
)
