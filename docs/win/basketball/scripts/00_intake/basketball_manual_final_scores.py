#!/usr/bin/env python3
"""DRatings manual basketball final scores -> existing 18-field drat_raw JSON.

Accepts copied DRatings final-score tables with line breaks inside cells,
optional/blank moneyline and spread columns, and repeated table headers.

For games already present in raw JSON, adds only scores/status through the
existing shared merger, retaining original pregame predictions and odds.

Log (overwritten each run):
  docs/win/basketball/errors/00_intake/basketball_manual_final_scores.txt
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from basketball_manual_input import ERROR_DIR, LEAGUE_SPORT, init_log, log, merge_into_files

LOG_FILE = ERROR_DIR / "basketball_manual_final_scores.txt"


DATE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}(?:\s|$)")
TIME = re.compile(r"^\d{1,2}:\d{2}\s*[AP]M$", re.IGNORECASE)
PERCENT = re.compile(r"^\d+(?:\.\d+)?%$")
MONEYLINE = re.compile(r"^[+-]\d{2,5}$")
SPREAD = re.compile(r"^[+-]\d+(?:½|¼|¾|\.\d+)?[+-]\d{2,4}$")
SCORE = re.compile(r"^\d+$")
LOG_LOSS = re.compile(r"^[+-]?\d+\.\d+$")
RECORD = re.compile(r"\s*\(\d+[-–]\d+(?:[-–]\d+)?\)\s*$")

RAW_FIELDS = (
    "sport", "date_time", "team1", "team2", "team1_win_pct", "team2_win_pct",
    "team1_moneyline", "team2_moneyline", "team1_spread", "team2_spread",
    "proj_score_1", "proj_score_2", "total", "over_line", "under_line",
    "score1", "score2", "game_status",
)

# DRatings repeats these headings between portions of long tables. Some
# browsers copy a heading across multiple lines; others join the words.
HEADER_WORDS = {
    "time", "teams", "win", "best", "ml", "spread", "final", "points",
    "sportsbook", "dratings", "log", "loss", "bet", "value", "more",
    "details", "total", "o", "u", "svg",
}


def _header_or_separator(token: str) -> bool:
    if re.fullmatch(r"[|:\-\s]+", token):
        return True
    words = re.findall(r"[a-z]+", token.lower())
    return bool(words) and all(word in HEADER_WORDS for word in words)


def _flatten_clipboard(text: str) -> list[str]:
    """Turn multiline TSV, Markdown, or HTML into ordered nonempty tokens."""
    # Some copied tables include a literal continuation backslash at EOL.
    text = re.sub(r"\\(?=\r?\n)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\n)]*\)", r"\1", text)
    # Split Markdown columns before <br> converts a cell into multiple lines.
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            line = "\t".join(part.strip() for part in stripped[1:-1].split("|"))
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", text)
    text = re.sub(r"(?i)</\s*t[dh]\s*>", "\t", text)
    text = re.sub(r"(?i)</\s*tr\s*>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).replace("\xa0", " ")
    tokens = []
    for token in re.split(r"[\r\n\t]+", text):
        token = token.strip()
        if token and not _header_or_separator(token):
            tokens.append(token)
    return tokens


def _expand_paired_numbers(tokens: list[str]) -> list[str]:
    """Split numeric pairs copied on one line without splitting team names."""
    out = []
    for token in tokens:
        pair = token.split()
        if len(pair) == 2 and any(
            all(pattern.fullmatch(part) for part in pair)
            for pattern in (PERCENT, MONEYLINE, SPREAD, SCORE, LOG_LOSS)
        ):
            out.extend(pair)
        else:
            out.append(token)
    return out


def _game_blocks(tokens: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if DATE.match(token):
            if current:
                blocks.append(current)
            current = [token]
        elif current:
            current.append(token)
    if current:
        blocks.append(current)
    if not blocks:
        raise ValueError("No dated games found in pasted final-score table")
    return blocks


def _parse_date_time(first: str, rest: list[str], basis: str) -> tuple[datetime, list[str]]:
    combined = first.strip()
    if not re.search(r"\b\d{1,2}:\d{2}\s*[AP]M\b", combined, re.IGNORECASE):
        if not rest or not TIME.fullmatch(rest[0]):
            raise ValueError(f"Missing game time after {first!r}")
        combined += " " + rest[0]
        rest = rest[1:]
    try:
        naive = datetime.strptime(combined, "%m/%d/%Y %I:%M %p")
    except ValueError as exc:
        raise ValueError(f"Invalid game date/time {combined!r}") from exc
    if basis == "utc":
        converted = naive.replace(tzinfo=timezone.utc).astimezone(
            ZoneInfo("America/New_York")
        )
    else:
        converted = naive.replace(tzinfo=ZoneInfo("America/New_York"))
    return converted, rest


def _parse_block(block: list[str], league: str, basis: str) -> dict[str, str]:
    when, tokens = _parse_date_time(block[0], block[1:], basis)
    tokens = _expand_paired_numbers(tokens)
    label = when.strftime("%m/%d/%Y %I:%M %p")

    # Some clipboard formats keep two teams with records in one cell.
    if tokens:
        matches = re.findall(r"[^()]+?\(\d+[-–]\d+(?:[-–]\d+)?\)", tokens[0])
        if len(matches) == 2 and "".join(matches).strip() == tokens[0].strip():
            tokens = [matches[0].strip(), matches[1].strip()] + tokens[1:]

    if len(tokens) < 4:
        raise ValueError(f"Missing teams or win percentages for {label}")
    team1, team2 = tokens[0:2]
    if not team1 or not team2 or PERCENT.fullmatch(team1) or PERCENT.fullmatch(team2):
        raise ValueError(f"Invalid teams for {label}: {team1!r}, {team2!r}")

    prob1, prob2 = tokens[2:4]
    for probability in (prob1, prob2):
        if not PERCENT.fullmatch(probability) or not 0 <= float(probability[:-1]) <= 100:
            raise ValueError(f"Invalid win percentage for {label}: {probability!r}")
    if abs(float(prob1[:-1]) + float(prob2[:-1]) - 100.0) > 0.2:
        raise ValueError(f"Win probabilities do not total 100% for {label}")

    i = 4
    moneylines = []
    while i < len(tokens) and MONEYLINE.fullmatch(tokens[i]):
        moneylines.append(tokens[i])
        i += 1
    if len(moneylines) not in (0, 2):
        raise ValueError(f"Expected zero or two moneylines for {label}; found {len(moneylines)}")

    spreads = []
    while i < len(tokens) and SPREAD.fullmatch(tokens[i]):
        spreads.append(tokens[i])
        i += 1
    if len(spreads) not in (0, 2):
        raise ValueError(f"Expected zero or two spreads for {label}; found {len(spreads)}")

    if i + 2 > len(tokens) or not SCORE.fullmatch(tokens[i]) or not SCORE.fullmatch(tokens[i + 1]):
        raise ValueError(f"Missing or invalid final points for {label}")
    score1, score2 = tokens[i:i + 2]
    i += 2

    # The optional sportsbook/DRatings log-loss columns are not raw JSON fields.
    remaining = tokens[i:]
    if len(remaining) > 2 or any(not LOG_LOSS.fullmatch(v) for v in remaining):
        raise ValueError(f"Unexpected text after final points for {label}: {remaining!r}")

    row = dict.fromkeys(RAW_FIELDS, "")
    row.update({
        "sport": LEAGUE_SPORT[league],
        "date_time": label,
        "team1": team1,
        "team2": team2,
        "team1_win_pct": prob1,
        "team2_win_pct": prob2,
        "team1_moneyline": moneylines[0] if moneylines else "",
        "team2_moneyline": moneylines[1] if moneylines else "",
        "team1_spread": spreads[0] if spreads else "",
        "team2_spread": spreads[1] if spreads else "",
        "score1": score1,
        "score2": score2,
        "game_status": "Final",
    })
    return row


def parse_final_scores(text: str, league: str, basis: str) -> list[dict[str, str]]:
    """Parse all dates, requiring every game block to be valid before writing."""
    if league not in LEAGUE_SPORT:
        raise ValueError(f"Unsupported league: {league}")
    if basis not in {"et", "utc"}:
        raise ValueError(f"Unsupported time basis: {basis}")
    if not text.strip():
        raise ValueError("Pasted data is empty")

    rows = []
    seen = set()
    for number, block in enumerate(_game_blocks(_flatten_clipboard(text)), 1):
        try:
            game = _parse_block(block, league, basis)
        except ValueError as exc:
            raise ValueError(f"Game row {number}: {exc}") from exc
        date = datetime.strptime(game["date_time"], "%m/%d/%Y %I:%M %p").date()
        # Match the shared JSON merger's date-and-teams identity rule.
        identity = (
            date,
            RECORD.sub("", game["team1"]).strip().casefold(),
            RECORD.sub("", game["team2"]).strip().casefold(),
        )
        if identity in seen:
            raise ValueError(f"Duplicate pasted matchup: {identity}")
        seen.add(identity)
        rows.append(game)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--league", choices=sorted(LEAGUE_SPORT), required=True)
    ap.add_argument("--raw-file", type=Path, required=True)
    ap.add_argument("--time-basis", choices=["et", "utc"], required=True)
    args = ap.parse_args(argv)

    init_log(LOG_FILE, "basketball_manual_final_scores")
    log(LOG_FILE, f"ARGS | league={args.league} | raw_file={args.raw_file} | time_basis={args.time_basis}")
    try:
        text = args.raw_file.read_text(encoding="utf-8-sig")
        rows = parse_final_scores(text, args.league, args.time_basis)
        paths = merge_into_files(rows, args.league, "final")
    except Exception as exc:
        log(LOG_FILE, f"ERROR: {exc}\n{traceback.format_exc()}")
        log(LOG_FILE, "STATUS: FAILED")
        raise
    for path, added, updated in paths:
        print(f"WROTE/VERIFIED {path}: added={added} updated={updated}")
        log(LOG_FILE, f"WROTE/VERIFIED {path}: added={added} updated={updated}")
    print(f"SUCCESS: parsed={len(rows)} league={args.league} files={len(paths)}")
    log(LOG_FILE, f"SUMMARY | parsed={len(rows)} | league={args.league} | files={len(paths)}")
    log(LOG_FILE, "STATUS: SUCCESS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
