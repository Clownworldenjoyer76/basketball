# docs/win/basketball/scripts/00_intake/basketball_drat_scraper.py

import json
import time
import random
import traceback
import re
from pathlib import Path
from datetime import datetime
import pytz
from playwright.sync_api import sync_playwright

URLS = {
    "nba":  "https://www.dratings.com/predictor/nba-basketball-predictions/",
    "ncaa": "https://www.dratings.com/predictor/ncaa-basketball-predictions/",
    "wnba": "https://www.dratings.com/predictor/wnba-basketball-predictions/",
}

UTC = pytz.utc
ET  = pytz.timezone("America/New_York")

ERROR_DIR = Path("docs/win/basketball/errors/00_intake")
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "basketball_drat_scraper.txt"

with open(LOG_FILE, "w", encoding="utf-8") as startup_log_handle:
    startup_log_handle.write(f"=== basketball_drat_scraper RUN {datetime.now(ET).isoformat()} ===\n")


def log(msg: str) -> None:
    line = f"{datetime.now(ET).isoformat()} | {msg}"
    print(line, flush=True)

    with open(LOG_FILE, "a", encoding="utf-8") as log_handle:
        log_handle.write(line + "\n")


def convert_utc_to_et(date_time_str: str) -> str:
    try:
        dt     = datetime.strptime(date_time_str.strip(), "%m/%d/%Y %I:%M %p")
        dt_utc = UTC.localize(dt)
        dt_et  = dt_utc.astimezone(ET)
        return dt_et.strftime("%m/%d/%Y %I:%M %p")
    except (AttributeError, TypeError, ValueError, OverflowError):
        return date_time_str


def normalize_cell(value) -> str:
    value = "" if value is None else str(value)
    value = value.replace("\r\n", "\n")
    value = value.replace("\r", "\n")
    value = value.replace("\u2028", "\n")
    value = value.replace("\u2029", "\n")
    value = value.replace("\xa0", " ")
    return value.strip()


def split_pair(value):
    value = normalize_cell(value)

    if not value:
        return "", ""

    if "\n" in value:
        parts = [p.strip() for p in re.split(r"\n+", value) if p.strip()]
    elif "|" in value:
        parts = [p.strip() for p in value.split("|") if p.strip()]
    else:
        parts = [value.strip()]

    if len(parts) >= 2:
        return parts[0], parts[1]

    if len(parts) == 1:
        return parts[0], ""

    return "", ""


def split_pair_expected(value, label: str, sport: str, row) -> tuple:
    a, b = split_pair(value)

    if a and not b:
        log(
            f"MALFORMED PAIR | sport={sport} | field={label} | "
            f"value={json.dumps(normalize_cell(value))} | "
            f"row={json.dumps(row)}"
        )

    return a, b


def strip_record(team: str) -> str:
    return re.sub(
        r"\s*\(\d+[-–]\d+[-–]?\d*\)\s*$",
        "",
        (team or "").strip()
    ).strip()


def strip_wnba_record(team: str) -> str:
    return strip_record(team)


def is_game_row(row):
    if len(row) < 5:
        return False

    teams = normalize_cell(row[1])
    return "\n" in teams


def is_score(s):
    try:
        v = float(str(s).strip())
        return 0 <= v == int(v) < 250
    except (ValueError, TypeError):
        return False


def is_summary_row(row):
    if not row:
        return False

    first = normalize_cell(row[0]).lower()
    return first in {"sportsbooks", "dratings"}


def parse_nba_ncaa(row, sport):
    if is_summary_row(row):
        return None

    if not is_game_row(row):
        return None

    try:
        date_time = convert_utc_to_et(normalize_cell(row[0]).replace("\n", " "))

        team1, team2 = split_pair_expected(row[1], "teams", sport, row)
        wp1, wp2     = split_pair_expected(row[2], "win_pct", sport, row)
        ml1, ml2     = split_pair_expected(row[3], "moneyline", sport, row)
        sp1, sp2     = split_pair_expected(row[4], "spread", sport, row)

        if not team1 or not team2:
            log(
                f"{sport.upper()} REJECTED EMPTY TEAM | "
                f"row_len={len(row)} | row={json.dumps(row)}"
            )
            return None

        proj1 = proj2 = total = over_line = under_line = ""
        score1 = score2 = game_status = ""

        # IMPORTANT:
        # NBA/NCAAM future rows can have numeric projected scores like:
        #   row[5] = "107.0\n104.5"
        # Those are NOT final scores.
        # So 10-cell rows must be handled as future prediction rows BEFORE any score logic.
        if len(row) >= 10:
            proj1, proj2 = split_pair_expected(row[5], "projected_scores", sport, row)
            total = normalize_cell(row[6]) if len(row) > 6 else ""
            over_line, under_line = split_pair_expected(row[7], "over_under", sport, row)

        # Some older/in-between rows may have total/O-U/status/score.
        elif len(row) >= 9 and not is_score(normalize_cell(row[5])):
            total = normalize_cell(row[5]) if len(row) > 5 else ""
            over_line, under_line = split_pair_expected(row[6], "over_under", sport, row)
            game_status = " ".join(
                [p for p in re.split(r"\n+", normalize_cell(row[7])) if p.strip()]
            ) if len(row) > 7 else ""
            score1, score2 = split_pair_expected(row[8], "score", sport, row)

        # Completed rows:
        #   row[5] = final score away/home
        elif len(row) >= 7:
            score1, score2 = split_pair_expected(row[5], "score", sport, row)

            if not score2 and len(row) > 6 and is_score(row[6]):
                score2 = normalize_cell(row[6])

        return {
            "sport":           sport,
            "date_time":       date_time,
            "team1":           team1,
            "team2":           team2,
            "team1_win_pct":   wp1,
            "team2_win_pct":   wp2,
            "team1_moneyline": ml1,
            "team2_moneyline": ml2,
            "team1_spread":    sp1,
            "team2_spread":    sp2,
            "proj_score_1":    proj1,
            "proj_score_2":    proj2,
            "total":           total,
            "over_line":       over_line,
            "under_line":      under_line,
            "score1":          score1,
            "score2":          score2,
            "game_status":     game_status,
        }

    except Exception as e:
        log(
            f"NBA/NCAA PARSE ERROR | sport={sport} | "
            f"row_len={len(row)} | error={e} | "
            f"row={json.dumps(row)}\n{traceback.format_exc()}"
        )
        return None


def parse_wnba(row):
    if is_summary_row(row):
        return None

    if len(row) not in (8, 10):
        log(
            f"WNBA REJECTED ROW LENGTH | "
            f"len={len(row)} | row={json.dumps(row)}"
        )
        return None

    try:
        date_time = convert_utc_to_et(normalize_cell(row[0]).replace("\n", " "))

        team1, team2 = split_pair_expected(row[1], "teams", "wnba", row)

        team1 = strip_wnba_record(team1)
        team2 = strip_wnba_record(team2)

        if not team1 or not team2:
            log(
                f"WNBA REJECTED EMPTY TEAM | "
                f"row={json.dumps(row)}"
            )
            return None

        if team1.lower() == "sportsbooks" or team1.lower() == "dratings":
            log(
                f"WNBA REJECTED HEADER ROW | "
                f"row={json.dumps(row)}"
            )
            return None

        wp1, wp2 = split_pair_expected(row[2], "win_pct", "wnba", row)
        ml1, ml2 = split_pair_expected(row[3], "moneyline", "wnba", row) if len(row) > 3 else ("", "")
        sp1, sp2 = split_pair_expected(row[4], "spread", "wnba", row) if len(row) > 4 else ("", "")

        proj1 = proj2 = total = over_line = under_line = ""
        score1 = score2 = game_status = ""

        if len(row) == 10:
            proj1, proj2 = split_pair_expected(row[5], "projected_scores", "wnba", row)
            total = normalize_cell(row[6]) if len(row) > 6 else ""
            over_line, under_line = split_pair_expected(row[7], "over_under", "wnba", row) if len(row) > 7 else ("", "")

        elif len(row) == 8:
            score1, score2 = split_pair_expected(row[5], "score", "wnba", row)

        return {
            "sport":           "wnba",
            "date_time":       date_time,
            "team1":           team1,
            "team2":           team2,
            "team1_win_pct":   wp1,
            "team2_win_pct":   wp2,
            "team1_moneyline": ml1,
            "team2_moneyline": ml2,
            "team1_spread":    sp1,
            "team2_spread":    sp2,
            "proj_score_1":    proj1,
            "proj_score_2":    proj2,
            "total":           total,
            "over_line":       over_line,
            "under_line":      under_line,
            "score1":          score1,
            "score2":          score2,
            "game_status":     game_status,
        }

    except Exception as e:
        log(
            f"WNBA PARSE ERROR | "
            f"row_len={len(row)} | error={e} | "
            f"row={json.dumps(row)}\n{traceback.format_exc()}"
        )
        return None


def parse_row(row, sport):
    if sport == "wnba":
        return parse_wnba(row)

    return parse_nba_ncaa(row, sport)


def scrape_page(page, url):
    page.goto(url, wait_until="domcontentloaded", timeout=60000)

    page.wait_for_selector("table", timeout=60000)

    rows = page.query_selector_all("table tbody tr")

    parsed_rows = []

    for idx, r in enumerate(rows):
        cells = [normalize_cell(c.inner_text()) for c in r.query_selector_all("td")]

        log(
            f"RAW TABLE ROW | idx={idx} | "
            f"cell_count={len(cells)} | "
            f"cells={json.dumps(cells)}"
        )

        parsed_rows.append(cells)

    return parsed_rows


def main():
    files_written = []

    try:
        date = datetime.now(ET).strftime("%Y_%m_%d")

        base_out_dir = Path("docs/win/basketball/00_intake/drat_raw")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)

            page = browser.new_page()

            page.set_extra_http_headers({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            })

            for sport, url in URLS.items():
                log(f"Scraping {sport.upper()}")

                try:
                    raw = scrape_page(page, url)

                    log(f"RAW ROWS: {len(raw)}")

                    games = []

                    for idx, row in enumerate(raw):
                        try:
                            parsed = parse_row(row, sport)

                            if parsed:
                                games.append(parsed)

                                log(
                                    f"PARSED GAME | sport={sport} | "
                                    f"idx={idx} | "
                                    f"{parsed.get('team1')} vs {parsed.get('team2')}"
                                )

                            else:
                                log(
                                    f"REJECTED ROW | sport={sport} | "
                                    f"idx={idx} | "
                                    f"row_len={len(row)} | "
                                    f"row={json.dumps(row)}"
                                )

                        except Exception as e:
                            log(
                                f"ROW ERROR | sport={sport} | "
                                f"idx={idx} | "
                                f"row_len={len(row)} | "
                                f"error={e} | "
                                f"row={json.dumps(row)}\n"
                                f"{traceback.format_exc()}"
                            )

                    label = "ncaam" if sport == "ncaa" else sport

                    out_dir = base_out_dir / label
                    out_dir.mkdir(parents=True, exist_ok=True)

                    path = out_dir / f"{date}_{label}_raw.json"

                    with open(path, "w", encoding="utf-8") as json_handle:
                        json.dump(games, json_handle, indent=2)

                    files_written.append((str(path), len(games)))

                    log(f"WROTE {path} ({len(games)} games)")

                except Exception as e:
                    log(f"ERROR scraping {sport}: {e}\n{traceback.format_exc()}")

                time.sleep(random.uniform(2, 4))

            browser.close()

        log("--- SUMMARY ---")
        log(f"Files written: {len(files_written)}")

        for path, count in files_written:
            log(f"FILE: {path} ({count} games)")

        log("STATUS: SUCCESS")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise

    print("Basketball drat scraper complete.")


if __name__ == "__main__":
    main()
