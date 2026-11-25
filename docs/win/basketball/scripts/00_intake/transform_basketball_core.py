
# docs/win/basketball/scripts/00_intake/transform_basketball_core.py

import math
import os
import re
import json
import traceback
import pandas as pd
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)

PROB_SUM_TOLERANCE = 0.1

LEAGUE_CONFIG = {
    "nba": {
        "league_label": "NBA",
        "input_dir": Path("docs/win/basketball/00_intake/drat_raw/nba"),
        "predictions_dir": "docs/win/basketball/00_intake/predictions/nba",
        "final_scores_dir": "docs/win/basketball/05_final_scores/results/nba",
        "log_file": ERROR_DIR / "transform_basketball_nba.txt",
    },
    "ncaam": {
        "league_label": "NCAAM",
        "input_dir": Path("docs/win/basketball/00_intake/drat_raw/ncaam"),
        "predictions_dir": "docs/win/basketball/00_intake/predictions/ncaam",
        "final_scores_dir": "docs/win/basketball/05_final_scores/results/ncaam",
        "log_file": ERROR_DIR / "transform_basketball_ncaam.txt",
    },
    "wnba": {
        "league_label": "WNBA",
        "input_dir": Path("docs/win/basketball/00_intake/drat_raw/wnba"),
        "predictions_dir": "docs/win/basketball/00_intake/predictions/wnba",
        "final_scores_dir": "docs/win/basketball/05_final_scores/results/wnba",
        "log_file": ERROR_DIR / "transform_basketball_wnba.txt",
    },
}


def init_log(league_key: str) -> None:
    cfg = LEAGUE_CONFIG[league_key]
    with open(cfg["log_file"], "w", encoding="utf-8") as f:
        f.write(
            f"=== transform_basketball {cfg['league_label']} RUN "
            f"{datetime.now().isoformat()} ===\n"
        )


def log(league_key: str, msg: str) -> None:
    cfg = LEAGUE_CONFIG[league_key]
    with open(cfg["log_file"], "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


def parse_date(date_str: str) -> str:
    try:
        dt = datetime.strptime(date_str.strip(), "%m/%d/%Y %I:%M %p")
        return dt.strftime("%Y_%m_%d")
    except ValueError:
        return date_str.strip().replace("/", "_").replace(" ", "_")


def parse_time(date_str: str) -> str:
    try:
        dt = datetime.strptime(date_str.strip(), "%m/%d/%Y %I:%M %p")
        return dt.strftime("%I:%M %p")
    except ValueError:
        parts = date_str.strip().split(" ")
        if len(parts) >= 2:
            return " ".join(parts[1:])
        return ""


def strip_record(name: str) -> str:
    return re.sub(r"\s*\(\d+[-–]\d+[-–]?\d*\)\s*$", "", str(name)).strip()


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save(df: pd.DataFrame, path: str, files_written: list, league_key: str):
    ensure_dir(path)
    tmp_path = path + ".tmp"
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    files_written.append((path, len(df)))
    log(league_key, f"WROTE {path} ({len(df)} rows)")
    print(f"  Saved {len(df)} rows -> {path}")


def load_json(path: Path) -> list:
    if not path or not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    return result


def games_to_df(games: list) -> pd.DataFrame:
    df = pd.DataFrame(games)
    if df.empty:
        return df
    df["game_date"] = df["date_time"].apply(parse_date)
    df["game_time"] = df["date_time"].apply(parse_time)
    df["team1"] = df["team1"].apply(strip_record)
    df["team2"] = df["team2"].apply(strip_record)
    return df


def require_number(value, label: str, context: str) -> float:
    if value is None:
        raise ValueError(f"{label} is missing: {context}")
    try:
        is_null = bool(pd.isna(value))
    except (TypeError, ValueError):
        is_null = False
    if is_null:
        raise ValueError(f"{label} is null/NaN: {context}")
    text = str(value).replace("%", "").strip()
    if text == "":
        raise ValueError(f"{label} is blank: {context}")
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"{label} is not numeric ({value!r}): {context}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite ({value!r}): {context}")
    return number


def is_valid_game_date(value) -> bool:
    text = str(value)
    if not re.fullmatch(r"\d{4}_\d{2}_\d{2}", text):
        return False
    try:
        datetime.strptime(text, "%Y_%m_%d")
    except ValueError:
        return False
    return True


def drop_bad_dates(
    df: pd.DataFrame,
    league_key: str,
    source_name: str,
    stats: dict,
) -> pd.DataFrame:
    valid = df["game_date"].apply(is_valid_game_date)
    for _, row in df[~valid].iterrows():
        log(
            league_key,
            "ERROR unparseable date, game skipped: "
            f"file={source_name} date_time={row.get('date_time')!r} "
            f"home={row.get('team2')!r} away={row.get('team1')!r}",
        )
    stats["date_rows_skipped"] = stats.get("date_rows_skipped", 0) + int((~valid).sum())
    return df[valid].copy()


def check_no_duplicate_games(df: pd.DataFrame, source_name: str) -> None:
    seen = set()
    duplicates = set()
    for key in zip(
        df["game_date"],
        df["team1"].astype(str).str.strip().str.casefold(),
        df["team2"].astype(str).str.strip().str.casefold(),
    ):
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if duplicates:
        raise ValueError(
            f"Duplicate games inside {source_name}: {sorted(duplicates)[:10]}"
        )


def process_predictions(
    df: pd.DataFrame,
    files_written: list,
    league_key: str,
    stats: dict,
):
    cfg = LEAGUE_CONFIG[league_key]
    league_label = cfg["league_label"]

    mask = df["score1"].isna() | (df["score1"].astype(str).str.strip() == "")
    upcoming = df[mask].copy()
    stats["upcoming_games"] += len(upcoming)

    if upcoming.empty:
        log(league_key, f"No upcoming {league_label} games found in this file.")
        return

    today_ny = datetime.now(ZoneInfo("America/New_York")).strftime("%Y_%m_%d")
    pending = []

    for date_val, group in upcoming.groupby("game_date"):
        path = f"{cfg['predictions_dir']}/{date_val}_{league_label}_predictions.csv"

        # A historical pregame snapshot must never be replaced by a later
        # transformation of raw data that contains fewer/different games.
        if (
            re.fullmatch(r"\d{4}_\d{2}_\d{2}", str(date_val))
            and str(date_val) < today_ny
            and Path(path).is_file()
        ):
            log(league_key, f"PRESERVED HISTORICAL PREDICTIONS: {path}")
            continue

        rows = []
        for _, row in group.iterrows():
            context = (
                f"game_date={date_val} home={row['team2']!r} away={row['team1']!r}"
            )
            home_prob = require_number(
                row["team2_win_pct"], "home win pct", context
            ) / 100
            away_prob = require_number(
                row["team1_win_pct"], "away win pct", context
            ) / 100
            for label, value in (("home_prob", home_prob), ("away_prob", away_prob)):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"{label}={value:.6f} outside 0-1: {context}"
                    )
            prob_sum = home_prob + away_prob
            if abs(prob_sum - 1.0) > PROB_SUM_TOLERANCE:
                raise ValueError(
                    f"home_prob + away_prob = {prob_sum:.6f}, differs from 1 by more "
                    f"than tolerance {PROB_SUM_TOLERANCE}: {context}"
                )

            away_proj = require_number(
                row["proj_score_1"], "away projected score", context
            )
            home_proj = require_number(
                row["proj_score_2"], "home projected score", context
            )
            total_proj = round(away_proj + home_proj, 1)

            rows.append({
                "sport": "Basketball",
                "league": league_label,
                "game_id": "",
                "game_date": date_val,
                "game_time": row["game_time"],
                "home_team": row["team2"],
                "away_team": row["team1"],
                "home_prob": f"{home_prob:.6f}",
                "away_prob": f"{away_prob:.6f}",
                "home_projected_points": home_proj,
                "away_projected_points": away_proj,
                "total_projected_points": total_proj,
            })

        out = pd.DataFrame(rows, columns=[
            "sport", "league", "game_id", "game_date", "game_time",
            "home_team", "away_team", "home_prob", "away_prob",
            "home_projected_points", "away_projected_points", "total_projected_points",
        ])

        if Path(path).is_file():
            existing_rows = len(
                pd.read_csv(path, dtype=str, keep_default_na=False)
            )
            if len(out) < existing_rows:
                log(
                    league_key,
                    f"KEPT EXISTING PREDICTIONS (new rows fewer): {path} "
                    f"existing_rows={existing_rows} new_rows={len(out)}",
                )
                continue

        pending.append((path, out))

    for path, out in pending:
        save(out, path, files_written, league_key)
        stats["prediction_files_written"] += 1
        stats["prediction_rows_written"] += len(out)


def process_final_scores(
    df: pd.DataFrame,
    files_written: list,
    league_key: str,
    stats: dict,
):
    cfg = LEAGUE_CONFIG[league_key]
    league_label = cfg["league_label"]

    mask = df["score1"].notna() & (df["score1"].astype(str).str.strip() != "")
    completed = df[mask].copy()
    stats["completed_games"] += len(completed)

    if completed.empty:
        log(league_key, f"No completed {league_label} games found in this file.")
        return

    pending = []

    for date_val, group in completed.groupby("game_date"):
        path = f"{cfg['final_scores_dir']}/{date_val}_final_scores_{league_label}.csv"

        if Path(path).exists():
            log(league_key, f"SKIPPED existing final scores file: {path}")
            continue

        rows = []
        for _, row in group.iterrows():
            context = (
                f"game_date={date_val} home={row['team2']!r} away={row['team1']!r}"
            )
            away_score = int(require_number(row["score1"], "away score", context))
            home_score = int(require_number(row["score2"], "home score", context))
            total = away_score + home_score
            away_spread = away_score - home_score
            home_spread = home_score - away_score

            rows.append({
                "sport": "Basketball",
                "league": league_label,
                "game_id": "",
                "game_date": date_val,
                "home_team": row["team2"],
                "away_team": row["team1"],
                "home_score": home_score,
                "away_score": away_score,
                "total": total,
                "home_spread": home_spread,
                "away_spread": away_spread,
            })

        out = pd.DataFrame(rows, columns=[
            "sport", "league", "game_id", "game_date",
            "home_team", "away_team", "home_score", "away_score",
            "total", "home_spread", "away_spread",
        ])
        pending.append((path, out))

    for path, out in pending:
        save(out, path, files_written, league_key)
        stats["final_score_files_written"] += 1
        stats["final_score_rows_written"] += len(out)


def write_summary(
    league_key: str,
    files_written: list,
    stats: dict,
    status: str,
):
    cfg = LEAGUE_CONFIG[league_key]
    league_label = cfg["league_label"]

    log(league_key, "--- SUMMARY ---")
    log(league_key, f"League: {league_label}")
    log(league_key, f"Input directory: {cfg['input_dir']}")
    log(league_key, f"Input files found: {stats['input_files_found']}")
    log(league_key, f"Input files processed: {stats['input_files_processed']}")
    log(league_key, f"Raw games loaded: {stats['games_loaded']}")
    log(league_key, f"Rows skipped (unparseable date): {stats.get('date_rows_skipped', 0)}")
    log(league_key, f"Upcoming games found: {stats['upcoming_games']}")
    log(league_key, f"Completed games found: {stats['completed_games']}")
    log(league_key, f"Prediction files written: {stats['prediction_files_written']}")
    log(league_key, f"Prediction rows written: {stats['prediction_rows_written']}")
    log(league_key, f"Final score files written: {stats['final_score_files_written']}")
    log(league_key, f"Final score rows written: {stats['final_score_rows_written']}")
    log(league_key, f"File-level errors: {stats['file_errors']}")
    log(league_key, f"Total files written: {len(files_written)}")
    for path, count in files_written:
        log(league_key, f"  FILE: {path} ({count} rows)")
    log(league_key, f"STATUS: {status}")


def process_league(league_key: str):
    cfg = LEAGUE_CONFIG[league_key]
    league_label = cfg["league_label"]
    input_dir = cfg["input_dir"]

    init_log(league_key)
    log(
        league_key,
        "CHECKS | probability_range=0-1 "
        f"probability_sum_tolerance={PROB_SUM_TOLERANCE}",
    )

    files_written = []
    stats = {
        "input_files_found": 0,
        "input_files_processed": 0,
        "games_loaded": 0,
        "upcoming_games": 0,
        "completed_games": 0,
        "prediction_files_written": 0,
        "prediction_rows_written": 0,
        "final_score_files_written": 0,
        "final_score_rows_written": 0,
        "file_errors": 0,
        "date_rows_skipped": 0,
    }

    try:
        if not input_dir.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: {input_dir}"
            )

        run_date = (
            os.getenv("DATE")
            or datetime.now(ZoneInfo("America/New_York")).strftime("%Y_%m_%d")
        )

        current_raw_path = (
            input_dir
            / f"{run_date}_{league_key}_raw.json"
        )

        if not current_raw_path.exists():
            raise FileNotFoundError(
                f"Missing current raw input for "
                f"{league_label}: {current_raw_path}"
            )

        current_games = load_json(current_raw_path)

        if not current_games:
            raise RuntimeError(
                f"Current raw input contains zero usable "
                f"{league_label} games: {current_raw_path}"
            )

        json_files = sorted(input_dir.glob("*.json"))
        stats["input_files_found"] = len(json_files)

        for json_path in json_files:
            log(league_key, f"Processing {league_label}: {json_path}")
            try:
                games = load_json(json_path)
                stats["input_files_processed"] += 1
                stats["games_loaded"] += len(games)
                log(league_key, f"{league_label} games loaded from {json_path.name}: {len(games)}")

                df = games_to_df(games)
                if not df.empty:
                    df = drop_bad_dates(df, league_key, json_path.name, stats)
                    check_no_duplicate_games(df, json_path.name)
                if not df.empty:
                    process_predictions(df, files_written, league_key, stats)
                    process_final_scores(df, files_written, league_key, stats)
                else:
                    log(league_key, f"{league_label}: no data to process in {json_path.name}")

            except Exception as e:
                stats["file_errors"] += 1
                log(league_key, f"ERROR processing {json_path}: {e}\n{traceback.format_exc()}")

        if stats["file_errors"] > 0 and len(files_written) > 0:
            status = "PARTIAL"
        elif stats["file_errors"] > 0 and len(files_written) == 0:
            status = "FAILED"
        elif len(files_written) == 0:
            status = "SUCCESS (nothing to write)"
        else:
            status = "SUCCESS"

        write_summary(league_key, files_written, stats, status)
        return status

    except Exception as e:
        log(league_key, f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        status = "FAILED"
        if len(files_written) > 0:
            status = "PARTIAL"
        write_summary(league_key, files_written, stats, status)
        raise


def main():
    failed = []
    for league_key in ["nba", "ncaam", "wnba"]:
        try:
            status = process_league(league_key)
        except Exception as exc:
            failed.append(league_key)
            print(f"{league_key} FAILED: {exc}")
            continue
        if str(status).startswith(("FAILED", "PARTIAL")):
            failed.append(league_key)
            print(f"{league_key} finished with status: {status}")

    if failed:
        print(f"\nLeagues with errors: {failed}")
        raise SystemExit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
