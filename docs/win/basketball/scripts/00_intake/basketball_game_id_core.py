#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/basketball_game_id.py

import csv
import traceback
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# =========================
# PATHS
# =========================

LEAGUES = {
    "nba": {
        "league_label": "NBA",
        "daily_games_dir": Path("docs/win/basketball/daily_games/nba"),
        "predictions_dir": Path("docs/win/basketball/00_intake/predictions/nba"),
        "final_scores_dir": Path("docs/win/basketball/05_final_scores/results/nba"),
    },
    "ncaam": {
        "league_label": "NCAAM",
        "daily_games_dir": Path("docs/win/basketball/daily_games/ncaam"),
        "predictions_dir": Path("docs/win/basketball/00_intake/predictions/ncaam"),
        "final_scores_dir": Path("docs/win/basketball/05_final_scores/results/ncaam"),
    },
    "wnba": {
        "league_label": "WNBA",
        "daily_games_dir": Path("docs/win/basketball/daily_games/wnba"),
        "predictions_dir": Path("docs/win/basketball/00_intake/predictions/wnba"),
        "final_scores_dir": Path("docs/win/basketball/05_final_scores/results/wnba"),
    },
}

ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "basketball_game_id.txt"

with open(LOG_FILE, "w", encoding="utf-8") as startup_log_handle:
    startup_log_handle.write(f"=== basketball_game_id RUN {datetime.now().isoformat()} ===\n")


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


# =========================
# HELPERS
# =========================

def clean_value(val):
    if val is None:
        return ""
    return str(val).strip()


def make_key(game_date, home_team, away_team):
    return (
        clean_value(game_date),
        clean_value(home_team).lower(),
        clean_value(away_team).lower(),
    )


def ensure_fieldnames(fieldnames, wanted):
    fieldnames = list(fieldnames or [])
    for col in wanted:
        if col not in fieldnames:
            fieldnames.append(col)
    return fieldnames


def read_csv_rows(path: Path):
    repo_root = Path(__file__).resolve().parents[5]
    candidate = path if path.is_absolute() else Path.cwd() / path
    safe_path = candidate.resolve(strict=True)
    try:
        safe_path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {path}") from exc

    with safe_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def write_csv_rows(path: Path, fieldnames, rows):
    repo_root = Path(__file__).resolve().parents[5]
    candidate = path if path.is_absolute() else Path.cwd() / path
    safe_path = candidate.resolve()
    try:
        safe_path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {path}") from exc

    with safe_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_daily_games_for_league(daily_games_dir: Path, league_label: str):
    daily_map = {}
    daily_keys_by_date = defaultdict(set)
    duplicate_rows = []
    files_found = 0
    rows_loaded = 0

    if not daily_games_dir.exists():
        log(f"DAILY GAMES DIR NOT FOUND: {daily_games_dir}")
        return daily_map, daily_keys_by_date, duplicate_rows, files_found, rows_loaded

    for csv_path in sorted(daily_games_dir.glob("*.csv")):
        files_found += 1
        try:
            rows, _ = read_csv_rows(csv_path)
            for row in rows:
                rows_loaded += 1

                game_date = clean_value(row.get("game_date"))
                home_team = clean_value(row.get("home_team"))
                away_team = clean_value(row.get("away_team"))
                game_id = clean_value(row.get("game_id"))
                game_time = clean_value(row.get("game_time"))

                key = make_key(game_date, home_team, away_team)
                daily_keys_by_date[game_date].add(key)

                if key in daily_map:
                    duplicate_rows.append({
                        "league": league_label,
                        "source_file": str(csv_path),
                        "game_date": game_date,
                        "home_team": home_team,
                        "away_team": away_team,
                        "game_id": game_id,
                    })
                else:
                    daily_map[key] = {
                        "game_id": game_id,
                        "game_time": game_time,
                        "source_file": str(csv_path),
                    }

        except Exception as e:
            log(f"ERROR loading daily games file {csv_path}: {e}\n{traceback.format_exc()}")

    return daily_map, daily_keys_by_date, duplicate_rows, files_found, rows_loaded


def track_duplicate_row(
    row: dict,
    seen_keys: set,
    duplicate_rows: list,
    league_label: str,
    csv_path: Path,
):
    key = make_key(
        row.get("game_date"),
        row.get("home_team"),
        row.get("away_team"),
    )

    if key in seen_keys:
        duplicate_rows.append({
            "league": league_label,
            "file": str(csv_path),
            "game_date": clean_value(
                row.get("game_date")
            ),
            "home_team": clean_value(
                row.get("home_team")
            ),
            "away_team": clean_value(
                row.get("away_team")
            ),
        })
    else:
        seen_keys.add(
            key
        )

    return key

def update_predictions(predictions_dir: Path, daily_map: dict, league_label: str):
    files_processed = 0
    rows_processed = 0
    rows_updated = 0
    matched_keys = set()
    prediction_no_match = []
    duplicate_prediction_rows = []

    if not predictions_dir.exists():
        log(f"PREDICTIONS DIR NOT FOUND: {predictions_dir}")
        return (
            files_processed,
            rows_processed,
            rows_updated,
            matched_keys,
            prediction_no_match,
            duplicate_prediction_rows,
        )

    for csv_path in sorted(predictions_dir.glob("*.csv")):
        try:
            files_processed += 1
            rows, fieldnames = read_csv_rows(csv_path)
            fieldnames = ensure_fieldnames(fieldnames, ["game_id", "game_time"])

            modified = False
            seen_keys = set()

            for row in rows:
                rows_processed += 1

                key = track_duplicate_row(
                    row,
                    seen_keys,
                    duplicate_prediction_rows,
                    league_label,
                    csv_path,
                )

                if key in daily_map:
                    matched_keys.add(key)
                    source_game_id = clean_value(daily_map[key].get("game_id"))
                    source_game_time = clean_value(daily_map[key].get("game_time"))

                    if clean_value(row.get("game_id")) != source_game_id:
                        row["game_id"] = source_game_id
                        modified = True
                        rows_updated += 1

                    if clean_value(row.get("game_time")) != source_game_time:
                        row["game_time"] = source_game_time
                        modified = True
                        rows_updated += 1
                else:
                    prediction_no_match.append({
                        "league": league_label,
                        "file": str(csv_path),
                        "game_date": clean_value(row.get("game_date")),
                        "home_team": clean_value(row.get("home_team")),
                        "away_team": clean_value(row.get("away_team")),
                    })

            if modified:
                write_csv_rows(csv_path, fieldnames, rows)
                log(f"UPDATED PREDICTIONS: {csv_path}")
            else:
                log(f"NO CHANGES PREDICTIONS: {csv_path}")

        except Exception as e:
            log(f"ERROR updating predictions file {csv_path}: {e}\n{traceback.format_exc()}")

    return (
        files_processed,
        rows_processed,
        rows_updated,
        matched_keys,
        prediction_no_match,
        duplicate_prediction_rows,
    )


def update_final_scores(final_scores_dir: Path, daily_map: dict, league_label: str):
    files_processed = 0
    rows_processed = 0
    rows_updated = 0
    duplicate_final_score_rows = []

    if not final_scores_dir.exists():
        log(f"FINAL SCORES DIR NOT FOUND: {final_scores_dir}")
        return files_processed, rows_processed, rows_updated, duplicate_final_score_rows

    for csv_path in sorted(final_scores_dir.glob("*.csv")):
        try:
            files_processed += 1
            rows, fieldnames = read_csv_rows(csv_path)
            fieldnames = ensure_fieldnames(fieldnames, ["game_id"])

            modified = False
            seen_keys = set()

            for row in rows:
                rows_processed += 1

                key = track_duplicate_row(
                    row,
                    seen_keys,
                    duplicate_final_score_rows,
                    league_label,
                    csv_path,
                )

                if key in daily_map:
                    source_game_id = clean_value(daily_map[key].get("game_id"))
                    if clean_value(row.get("game_id")) != source_game_id:
                        row["game_id"] = source_game_id
                        modified = True
                        rows_updated += 1

            if modified:
                write_csv_rows(csv_path, fieldnames, rows)
                log(f"UPDATED FINAL SCORES: {csv_path}")
            else:
                log(f"NO CHANGES FINAL SCORES: {csv_path}")

        except Exception as e:
            log(f"ERROR updating final scores file {csv_path}: {e}\n{traceback.format_exc()}")

    return files_processed, rows_processed, rows_updated, duplicate_final_score_rows


# =========================
# MAIN
# =========================

def main():
    total_daily_files_found = 0
    total_daily_rows_loaded = 0
    total_prediction_files_processed = 0
    total_prediction_rows_processed = 0
    total_prediction_updates = 0
    total_final_score_files_processed = 0
    total_final_score_rows_processed = 0
    total_final_score_updates = 0

    sportsbook_no_prediction_match = []
    prediction_no_sportsbook_match = []
    duplicate_key_rows = []

    try:
        for cfg in LEAGUES.values():
            league_label = cfg["league_label"]
            log(f"--- LEAGUE {league_label} ---")

            (
                daily_map,
                _,
                daily_duplicate_rows,
                daily_files_found,
                daily_rows_loaded,
            ) = load_daily_games_for_league(cfg["daily_games_dir"], league_label)

            total_daily_files_found += daily_files_found
            total_daily_rows_loaded += daily_rows_loaded
            duplicate_key_rows.extend(daily_duplicate_rows)

            (
                prediction_files_processed,
                prediction_rows_processed,
                prediction_updates,
                matched_prediction_keys,
                prediction_no_match,
                duplicate_prediction_rows,
            ) = update_predictions(
                cfg["predictions_dir"],
                daily_map,
                league_label,
            )

            total_prediction_files_processed += prediction_files_processed
            total_prediction_rows_processed += prediction_rows_processed
            total_prediction_updates += prediction_updates
            prediction_no_sportsbook_match.extend(prediction_no_match)
            duplicate_key_rows.extend(duplicate_prediction_rows)

            (
                final_score_files_processed,
                final_score_rows_processed,
                final_score_updates,
                duplicate_final_score_rows,
            ) = update_final_scores(
                cfg["final_scores_dir"],
                daily_map,
                league_label,
            )

            total_final_score_files_processed += final_score_files_processed
            total_final_score_rows_processed += final_score_rows_processed
            total_final_score_updates += final_score_updates
            duplicate_key_rows.extend(duplicate_final_score_rows)

            unmatched_daily_keys = set(daily_map.keys()) - matched_prediction_keys
            for key in sorted(unmatched_daily_keys):
                meta = daily_map[key]
                sportsbook_no_prediction_match.append({
                    "league": league_label,
                    "source_file": meta.get("source_file", ""),
                    "game_date": key[0],
                    "home_team": key[1],
                    "away_team": key[2],
                    "game_id": meta.get("game_id", ""),
                })

        # =========================
        # SUMMARY / MISMATCH REPORT
        # =========================

        log("--- SUMMARY ---")
        log(f"Daily games files found: {total_daily_files_found}")
        log(f"Daily games rows loaded: {total_daily_rows_loaded}")
        log(f"Prediction files processed: {total_prediction_files_processed}")
        log(f"Prediction rows processed: {total_prediction_rows_processed}")
        log(f"Prediction field updates made: {total_prediction_updates}")
        log(f"Final score files processed: {total_final_score_files_processed}")
        log(f"Final score rows processed: {total_final_score_rows_processed}")
        log(f"Final score field updates made: {total_final_score_updates}")
        log(f"Sportsbook rows with no prediction match: {len(sportsbook_no_prediction_match)}")
        log(f"Prediction rows with no sportsbook match: {len(prediction_no_sportsbook_match)}")
        log(f"Duplicate key rows: {len(duplicate_key_rows)}")

        log("--- SMALL MISMATCH REPORT ---")

        log("Sportsbook rows with no prediction match:")
        if sportsbook_no_prediction_match:
            for row in sportsbook_no_prediction_match[:25]:
                log(
                    f"  {row['league']} | {row['game_date']} | "
                    f"{row['home_team']} | {row['away_team']} | game_id={row['game_id']} | "
                    f"source={row['source_file']}"
                )
            if len(sportsbook_no_prediction_match) > 25:
                log(f"  ... and {len(sportsbook_no_prediction_match) - 25} more")
        else:
            log("  none")

        log("Prediction rows with no sportsbook match:")
        if prediction_no_sportsbook_match:
            for row in prediction_no_sportsbook_match[:25]:
                log(
                    f"  {row['league']} | {row['game_date']} | "
                    f"{row['home_team']} | {row['away_team']} | file={row['file']}"
                )
            if len(prediction_no_sportsbook_match) > 25:
                log(f"  ... and {len(prediction_no_sportsbook_match) - 25} more")
        else:
            log("  none")

        log("Duplicate key rows:")
        if duplicate_key_rows:
            for row in duplicate_key_rows[:25]:
                if "source_file" in row:
                    log(
                        f"  {row['league']} | {row['game_date']} | "
                        f"{row['home_team']} | {row['away_team']} | source={row['source_file']}"
                    )
                else:
                    log(
                        f"  {row['league']} | {row['game_date']} | "
                        f"{row['home_team']} | {row['away_team']} | file={row['file']}"
                    )
            if len(duplicate_key_rows) > 25:
                log(f"  ... and {len(duplicate_key_rows) - 25} more")
        else:
            log("  none")

        log("STATUS: SUCCESS")
        print("Basketball game_id injection complete.")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise


if __name__ == "__main__":
    main()