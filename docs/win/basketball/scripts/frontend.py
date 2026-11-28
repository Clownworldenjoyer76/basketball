#!/usr/bin/env python3
"""Export the two basketball frontend CSV files (NBA, WNBA, NCAAM).

Run from any working directory:
    python frontend.py

Writes:
    frontend/basketball_slate.csv  odds (every game) + predictions + picks, all dates
    frontend/basketball_final.csv  graded final files of all leagues stacked

Slate rows: one row per game; a game with selected bets gets one row per bet
instead, with the bet columns filled. Games without a bet have blank bet columns.
Source files are never modified.

Log (overwritten each run):
    docs/win/basketball/errors/99_validation/frontend.txt
"""

from __future__ import annotations

import csv
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path

LEAGUES = ("nba", "wnba", "ncaam")
SLATE_FIRST_COLUMNS = (
    "sport", "league", "game_date", "game_id", "game_time", "home_team", "away_team",
)
FINAL_FIRST_COLUMNS = ("sport", "league", "game_id", "game_date")


def find_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "docs/win/basketball").is_dir():
            return parent
    raise FileNotFoundError("Could not find docs/win/basketball above " + str(here))


ROOT = find_root()
BB = ROOT / "docs/win/basketball"
OUT = ROOT / "frontend"
SLATE_DEST = OUT / "basketball_slate.csv"
FINAL_DEST = OUT / "basketball_final.csv"
LOG_FILE = BB / "errors/99_validation/frontend.txt"
DUPLICATE_GAME_IDS = [0]


def init_log() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(
        f"=== frontend RUN {datetime.now(timezone.utc).isoformat()} ===\n",
        encoding="utf-8",
    )


def log(message: str, level: str = "INFO") -> None:
    print(f"WARNING: {message}" if level == "WARN" else message)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{datetime.now(timezone.utc).isoformat()} | {level:<5} | "
            f"{message.rstrip()}\n"
        )


def clean(value: str | None) -> str:
    if value is None:
        return ""
    result = value.strip()
    return "" if result.lower() in {"nan", "none", "null"} else result


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if not headers:
            return [], []
        if any(not h for h in headers) or len(headers) != len(set(headers)):
            raise ValueError(f"Invalid or duplicate CSV headers in {path}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Malformed CSV row in {path} at line {line_number}")
            rows.append({k: clean(v) for k, v in row.items()})
    return headers, rows


def index_by_game_id(rows: list[dict[str, str]], source: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        game_id = row.get("game_id", "")
        if not game_id:
            log(f"Skipping row with blank game_id in {rel(source)}", "WARN")
            continue
        if game_id in result:
            DUPLICATE_GAME_IDS[0] += 1
            log(f"Duplicate game_id={game_id} in {rel(source)}; using the last one", "WARN")
        result[game_id] = row
    return result


def fill_missing(target: dict[str, str], source_row: dict[str, str]) -> None:
    for field, value in source_row.items():
        if value and not target.get(field, ""):
            target[field] = value


def write_csv(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def ordered_headers(first: tuple[str, ...], seen: list[str]) -> list[str]:
    headers = [h for h in first if h in seen]
    headers += [h for h in seen if h not in headers]
    return headers


def odds_dir(league: str) -> Path:
    return BB / f"00_intake/sportsbook/sportsbook_cleaned/{league}"


def odds_dates(league: str) -> list[str]:
    folder = odds_dir(league)
    if not folder.is_dir():
        return []
    pattern = re.compile(rf"(\d{{4}}_\d{{2}}_\d{{2}})_{league}_odds\.csv", re.IGNORECASE)
    return sorted({
        match.group(1)
        for path in folder.iterdir()
        if path.is_file() and (match := pattern.fullmatch(path.name))
    })


def find_file(folder: Path, name: str) -> Path:
    """Return folder/name, matching the file name case-insensitively."""
    exact = folder / name
    if exact.is_file() or not folder.is_dir():
        return exact
    for path in folder.iterdir():
        if path.is_file() and path.name.lower() == name.lower():
            return path
    return exact


def build_slate() -> int:
    upper_cols: list[str] = []
    all_rows: list[dict[str, str]] = []
    total_games = total_picks = 0

    def include(fields: list[str]) -> None:
        for field in fields:
            if field not in upper_cols:
                upper_cols.append(field)

    for league in LEAGUES:
        league_up = league.upper()
        dates = odds_dates(league)
        if not dates:
            log(f"No odds files found for {league_up}; skipping", "WARN")
            continue
        for date in dates:
            odds_path = find_file(odds_dir(league), f"{date}_{league_up}_odds.csv")
            pred_path = find_file(
                BB / f"00_intake/predictions/predictions_cleaned/{league}",
                f"{date}_{league_up}_predictions.csv",
            )
            picks_path = find_file(
                BB / f"04_select/{league}/daily_picks", f"{date}_{league}_selected.csv"
            )

            odds_headers, odds_rows = read_csv(odds_path)
            if not odds_rows:
                log(f"Empty odds file {rel(odds_path)}; skipping", "WARN")
                continue
            pred_headers, pred_rows = read_csv(pred_path)
            pick_headers, pick_rows = read_csv(picks_path)
            include(odds_headers)
            include(pred_headers)
            include(pick_headers)

            games = index_by_game_id(odds_rows, odds_path)
            for game_id, game in games.items():
                game["league"] = league_up
                if not game.get("game_date"):
                    game["game_date"] = date

            preds = index_by_game_id(pred_rows, pred_path)
            for game_id, row in preds.items():
                game = games.get(game_id)
                if game is None:
                    log(f"Skipping unmatched game_id={game_id} in {rel(pred_path)}", "WARN")
                    continue
                fill_missing(game, row)

            picks_by_game: dict[str, list[dict[str, str]]] = {}
            for pick in pick_rows:
                game_id = pick.get("game_id", "")
                if game_id not in games:
                    log(f"Selected bet has unmatched game_id={game_id} in {rel(picks_path)}", "WARN")
                picks_by_game.setdefault(game_id, []).append(pick)

            for game_id, game in games.items():
                bets = picks_by_game.get(game_id)
                if bets:
                    for bet in bets:
                        row = dict(game)
                        fill_missing(row, bet)
                        all_rows.append(row)
                        total_picks += 1
                else:
                    all_rows.append(game)
                total_games += 1
            # Selected bets for games that are not in the odds file are kept as-is.
            for game_id, bets in picks_by_game.items():
                if game_id in games:
                    continue
                for bet in bets:
                    row = dict(bet)
                    row["league"] = league_up
                    if not row.get("game_date"):
                        row["game_date"] = date
                    all_rows.append(row)
                    total_picks += 1
            log(f"{league_up} {date}: {len(games)} games, {len(pick_rows)} selected bets")

    if not all_rows:
        raise FileNotFoundError("No basketball games found; slate not written")

    headers = ordered_headers(SLATE_FIRST_COLUMNS, upper_cols)
    for row in all_rows:
        for field in headers:
            row.setdefault(field, "")
    write_csv(SLATE_DEST, headers, all_rows)
    log(f"WROTE {rel(SLATE_DEST)}: {len(all_rows)} rows ({total_games} games, {total_picks} selected bets)")
    return len(all_rows)


def build_final() -> int:
    columns: list[str] = []
    all_rows: list[dict[str, str]] = []
    for league in LEAGUES:
        league_up = league.upper()
        path = find_file(
            BB / f"05_final_scores/results/{league}/graded", f"{league_up}_final.csv"
        )
        headers, rows = read_csv(path)
        if not rows:
            log(f"No final rows for {league_up} ({rel(path)}); skipping", "WARN")
            continue
        for field in headers:
            if field not in columns:
                columns.append(field)
        if "league" not in columns:
            columns.append("league")
        for row in rows:
            if not row.get("league"):
                row["league"] = league_up
            all_rows.append(row)
        log(f"{league_up} final: {len(rows)} rows")

    if not all_rows:
        raise FileNotFoundError("No basketball final rows found; final file not written")

    headers = ordered_headers(FINAL_FIRST_COLUMNS, columns)
    for row in all_rows:
        for field in headers:
            row.setdefault(field, "")
    write_csv(FINAL_DEST, headers, all_rows)
    log(f"WROTE {rel(FINAL_DEST)}: {len(all_rows)} rows")
    return len(all_rows)


def main() -> None:
    init_log()
    log(f"root={ROOT}")
    try:
        build_slate()
        build_final()
    except Exception as exc:
        log(f"FATAL: {type(exc).__name__}: {exc}", "ERROR")
        log(traceback.format_exc(), "ERROR")
        log("STATUS: FAILED", "ERROR")
        raise
    log(f"SUMMARY | duplicate_game_ids_keep_last={DUPLICATE_GAME_IDS[0]}")
    log("Export complete: 2 CSV files (slate, final)")
    log("STATUS: SUCCESS")


if __name__ == "__main__":
    main()
