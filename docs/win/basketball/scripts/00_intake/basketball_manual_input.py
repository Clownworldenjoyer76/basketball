#!/usr/bin/env python3
"""DRatings manual basketball predictions -> existing 18-field drat_raw JSON.

Accepts copied DRatings tables in Markdown, HTML, TSV, or line-oriented text.
This module also provides parsing/upsert helpers for basketball_manual_final_scores.py.
Run from the basketball repository root.
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import sys
import tempfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path("docs/win/basketball/00_intake/drat_raw")
LEAGUE_SPORT = {"nba": "nba", "ncaam": "ncaa", "wnba": "wnba"}
FIELDS = (
    "sport", "date_time", "team1", "team2", "team1_win_pct", "team2_win_pct",
    "team1_moneyline", "team2_moneyline", "team1_spread", "team2_spread",
    "proj_score_1", "proj_score_2", "total", "over_line", "under_line",
    "score1", "score2", "game_status",
)
PREDICTION_FIELDS = (
    "team1_win_pct", "team2_win_pct", "team1_moneyline", "team2_moneyline",
    "team1_spread", "team2_spread", "proj_score_1", "proj_score_2",
    "total", "over_line", "under_line",
)
DATE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")
TIME = re.compile(r"\b\d{1,2}:\d{2}\s*[AP]M\b", re.I)
PERCENT = re.compile(r"^\d+(?:\.\d+)?%$")
MONEYLINE = re.compile(r"^[+-]\d{2,5}$")
SPREAD = re.compile(r"^[+-]\d+(?:½|¼|¾|\.\d+)?[+-]\d{2,4}$")
NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")
OU_LINE = re.compile(r"^[ouOU]\d+(?:½|¼|¾|\.\d+)?[+-]\d{2,4}$")
RECORD = re.compile(r"\s*\(\d+[-–]\d+(?:[-–]\d+)?\)\s*$")
IGNORED = {
    "time", "teams", "win", "best", "ml", "best ml", "best spread", "spread",
    "points", "total points", "total", "best o/u", "o/u", "bet value",
    "more details", "svg", "final points", "sportsbook log loss", "dratings log loss",
    "upcoming games", "completed games", "final games", "in progress games",
}


class _Tables(HTMLParser):
    """HTML table cell extractor; preserves <br> as line breaks."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append("\n")

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None and self.row is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row:
                self.rows.append(self.row)
            self.row = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def plain_markup(value: str) -> str:
    """Keep anchor labels, not URLs; keep pairs split by <br>."""
    value = re.sub(r"\[([^\]]+)\]\((?:[^()]|\([^()]*\))*\)", r"\1", value)
    value = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = html.unescape(value)
    return value.replace("\xa0", " ").replace("\r", "").strip()


def _tokens(value: str) -> list[str]:
    return [v.strip() for v in plain_markup(value).split("\n") if v.strip()]


def _paired(value: str, label: str, pattern: re.Pattern | None = None) -> tuple[str, str]:
    parts = _tokens(value)
    if len(parts) == 1 and pattern is not None:
        parts = re.findall(pattern.pattern, parts[0], flags=pattern.flags)
    if len(parts) == 1 and label == "teams":
        # Rendered clipboard sometimes separates the teams only at the two record suffixes.
        matches = re.findall(r".+?\(\d+[-–]\d+(?:[-–]\d+)?\)", parts[0])
        if len(matches) == 2 and not parts[0].replace(matches[0], "", 1).replace(matches[1], "", 1).strip():
            parts = [m.strip() for m in matches]
    if len(parts) != 2:
        raise ValueError(f"Expected two {label}, found {len(parts)} in {value!r}")
    return parts[0], parts[1]


def parse_datetime(text: str, time_basis: str) -> datetime:
    date = DATE.search(plain_markup(text))
    at = TIME.search(plain_markup(text))
    if not date or not at:
        raise ValueError(f"Missing date/time: {text!r}")
    naive = datetime.strptime(f"{date.group()} {at.group().upper().replace(' ', '')}", "%m/%d/%Y %I:%M%p")
    if time_basis == "utc":
        return naive.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("America/New_York"))
    if time_basis == "et":
        return naive.replace(tzinfo=ZoneInfo("America/New_York"))
    raise ValueError("--time-basis must be 'et' or 'utc'")


def empty_game(league: str, when: datetime, team1: str, team2: str) -> dict[str, str]:
    row = {key: "" for key in FIELDS}
    row.update({
        "sport": LEAGUE_SPORT[league],
        "date_time": when.strftime("%m/%d/%Y %I:%M %p"),
        "team1": team1.strip(),
        "team2": team2.strip(),
    })
    return row


def _pct(v: str) -> str:
    v = v.strip()
    if not PERCENT.fullmatch(v) or not (0 <= float(v[:-1]) <= 100):
        raise ValueError(f"Invalid win percentage: {v!r}")
    return v


def _num(v: str, name: str) -> str:
    v = v.strip()
    if not NUMBER.fullmatch(v) or float(v) < 0:
        raise ValueError(f"Invalid {name}: {v!r}")
    return v


def _optional_pair(value: str, name: str, regex: re.Pattern) -> tuple[str, str]:
    if not plain_markup(value).strip():
        return "", ""
    result = _paired(value, name, regex)
    for part in result:
        if not regex.fullmatch(part):
            raise ValueError(f"Invalid {name}: {part!r}")
    return result


def _row_prediction(cells: list[str], league: str, basis: str) -> dict[str, str]:
    if len(cells) < 8:
        raise ValueError(f"Prediction row needs at least 8 columns; found {len(cells)}: {cells!r}")
    when = parse_datetime(cells[0], basis)
    team1, team2 = _paired(cells[1], "teams")
    prob1, prob2 = _paired(cells[2], "win percentages", re.compile(r"\d+(?:\.\d+)?%"))
    prob1, prob2 = _pct(prob1), _pct(prob2)
    if abs(float(prob1[:-1]) + float(prob2[:-1]) - 100) > 0.2:
        raise ValueError(f"Win percentages do not total 100: {prob1}, {prob2}")
    ml1, ml2 = _optional_pair(cells[3], "moneylines", MONEYLINE)
    sp1, sp2 = _optional_pair(cells[4], "spreads", SPREAD)
    point1, point2 = _paired(cells[5], "projected points", re.compile(r"\d+(?:\.\d+)?"))
    point1, point2 = _num(point1, "away projected points"), _num(point2, "home projected points")
    total = _num(plain_markup(cells[6]), "total")
    if abs(float(total) - (float(point1) + float(point2))) > 0.15:
        raise ValueError(f"Projected total does not match team scores: {point1} + {point2} != {total}")
    ou1, ou2 = _optional_pair(cells[7], "over/under lines", OU_LINE)
    if ou1 and (not ou1.lower().startswith("o") or not ou2.lower().startswith("u")):
        raise ValueError("Best O/U must list over first and under second")
    row = empty_game(league, when, team1, team2)
    row.update({
        "team1_win_pct": prob1, "team2_win_pct": prob2,
        "team1_moneyline": ml1, "team2_moneyline": ml2,
        "team1_spread": sp1, "team2_spread": sp2,
        "proj_score_1": point1, "proj_score_2": point2,
        "total": total, "over_line": ou1, "under_line": ou2,
    })
    return row


def _row_final(cells: list[str], league: str, basis: str) -> dict[str, str]:
    if len(cells) < 6:
        raise ValueError(f"Final row needs at least 6 columns; found {len(cells)}: {cells!r}")
    when = parse_datetime(cells[0], basis)
    team1, team2 = _paired(cells[1], "teams")
    prob1, prob2 = _paired(cells[2], "win percentages", re.compile(r"\d+(?:\.\d+)?%"))
    prob1, prob2 = _pct(prob1), _pct(prob2)
    if abs(float(prob1[:-1]) + float(prob2[:-1]) - 100) > 0.2:
        raise ValueError(f"Win percentages do not total 100: {prob1}, {prob2}")
    ml1, ml2 = _optional_pair(cells[3], "moneylines", MONEYLINE)
    sp1, sp2 = _optional_pair(cells[4], "spreads", SPREAD)
    score1, score2 = _paired(cells[5], "final points", re.compile(r"\d+(?:\.\d+)?"))
    if not score1.isdigit() or not score2.isdigit():
        raise ValueError(f"Final scores must be nonnegative integers: {score1}, {score2}")
    row = empty_game(league, when, team1, team2)
    row.update({
        "team1_win_pct": prob1, "team2_win_pct": prob2,
        "team1_moneyline": ml1, "team2_moneyline": ml2,
        "team1_spread": sp1, "team2_spread": sp2,
        "score1": score1, "score2": score2, "game_status": "Final",
    })
    return row


def _markdown_rows(text: str) -> list[list[str]]:
    rows = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [part.strip() for part in stripped.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells):
            continue
        rows.append(cells)
    return rows


def _tsv_rows(text: str) -> list[list[str]]:
    if "\t" not in text:
        return []
    return [row for row in csv.reader(io.StringIO(text), delimiter="\t") if row]


def _flat_rows(text: str, source: str) -> list[list[str]]:
    """Fallback for DRatings copy/paste as one token per line instead of a table."""
    text = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", text)
    text = re.sub(r"\[([^\]]+)\]\((?:[^()]|\([^()]*\))*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = html.unescape(text)
    lines = [v.strip() for v in re.split(r"[\r\n\t]+", text) if v.strip()]
    lines = [v for v in lines if re.sub(r"\s+", " ", v).lower() not in IGNORED]
    groups: list[list[str]] = []
    active = None
    for line in lines:
        if DATE.search(line) and (not active or len(active) > 1):
            if active:
                groups.append(active)
            active = [line]
        elif active is not None:
            active.append(line)
    if active:
        groups.append(active)
    result = []
    for group in groups:
        first = group[0]
        if not TIME.search(first):
            if len(group) < 2 or not TIME.fullmatch(group[1]):
                raise ValueError(f"Missing game time after {first!r}")
            first += " " + group[1]
            tokens = group[2:]
        else:
            tokens = group[1:]
        # Expand paired machine tokens that appear on one line.
        expanded = []
        for token in tokens:
            pieces = token.split()
            if len(pieces) == 2 and (
                all(PERCENT.fullmatch(p) for p in pieces) or
                all(MONEYLINE.fullmatch(p) for p in pieces) or
                all(SPREAD.fullmatch(p) for p in pieces) or
                all(OU_LINE.fullmatch(p) for p in pieces) or
                all(NUMBER.fullmatch(p) for p in pieces)
            ):
                expanded.extend(pieces)
            else:
                expanded.append(token)
        tokens = expanded
        if len(tokens) < 2:
            raise ValueError(f"Missing teams for {first}")
        if RECORD.search(tokens[0]) and len(re.findall(r"\(\d+[-–]\d+(?:[-–]\d+)?\)", tokens[0])) == 2:
            a, b = _paired(tokens[0], "teams")
            tokens = [a, b] + tokens[1:]
        teams = "\n".join(tokens[:2]); ix = 2
        if len(tokens) < ix + 4:
            raise ValueError(f"Not enough percentages/moneylines for {first}")
        pct = "\n".join(tokens[ix:ix+2]); ix += 2
        ml = "\n".join(tokens[ix:ix+2]); ix += 2
        if len(tokens) < ix + 2:
            raise ValueError(f"Missing spread tokens for {first}")
        sp = "\n".join(tokens[ix:ix+2]); ix += 2
        if source == "predictions":
            if len(tokens) < ix + 5:
                raise ValueError(f"Missing projected points/total/O-U for {first}")
            points = "\n".join(tokens[ix:ix+2]); ix += 2
            total = tokens[ix]; ix += 1
            ou = "\n".join(tokens[ix:ix+2]); ix += 2
            result.append([first, teams, pct, ml, sp, points, total, ou])
        else:
            if len(tokens) < ix + 2:
                raise ValueError(f"Missing final points for {first}")
            final_points = "\n".join(tokens[ix:ix+2])
            result.append([first, teams, pct, ml, sp, final_points])
    return result


def parse_text(text: str, league: str, source: str, time_basis: str) -> list[dict[str, str]]:
    if league not in LEAGUE_SPORT:
        raise ValueError(f"Unsupported league {league!r}")
    if source not in {"predictions", "final"}:
        raise ValueError(f"Unsupported source {source!r}")
    if not text.strip():
        raise ValueError("Pasted data is empty")
    minimum = 8 if source == "predictions" else 6
    parser = _row_prediction if source == "predictions" else _row_final
    table_rows = _markdown_rows(text)
    if not table_rows and ("<table" in text.lower() or "<tr" in text.lower()):
        doc = _Tables()
        doc.feed(text)
        table_rows = doc.rows
    if not table_rows and "\t" in text:
        table_rows = _tsv_rows(text)
    if table_rows:
        usable = [row for row in table_rows if row and DATE.search(plain_markup(row[0]))]
        if not usable:
            raise ValueError("No game rows found in pasted table")
    else:
        usable = _flat_rows(text, source)
    if not usable:
        raise ValueError("No valid game blocks found in pasted data")
    parsed: list[dict[str, str]] = []
    for i, row in enumerate(usable, 1):
        if len(row) < minimum:
            raise ValueError(f"Game row {i} has {len(row)} columns; expected at least {minimum}")
        try:
            parsed.append(parser(row, league, time_basis))
        except Exception as exc:
            raise ValueError(f"Game row {i}: {exc}") from exc
    seen = set()
    for row in parsed:
        key = match_key(row)
        if key in seen:
            raise ValueError(f"Duplicate pasted matchup: {key}")
        seen.add(key)
    return parsed


def strip_record(team: str) -> str:
    return RECORD.sub("", team).strip().casefold()


def _date(row: dict[str, str]) -> str:
    return datetime.strptime(row["date_time"], "%m/%d/%Y %I:%M %p").strftime("%Y_%m_%d")


def match_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (_date(row), strip_record(row["team1"]), strip_record(row["team2"]))


def load_existing(path: Path, league: str) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Existing raw JSON is not an array: {path}")
    keys = set()
    for item in rows:
        if not isinstance(item, dict) or not set(FIELDS).issubset(item):
            raise ValueError(f"Existing raw JSON has invalid 18-field game: {path}")
        if item["sport"] != LEAGUE_SPORT[league]:
            raise ValueError(f"Wrong sport in existing raw JSON: {path}")
        # Existing DRatings JSON can contain "Postponed" or "TBD" as date_time.
        # These records must be preserved, not treated as invalid input.
        try:
            key = match_key(item)
        except ValueError:
            key = (str(item["date_time"]), strip_record(item["team1"]), strip_record(item["team2"]))
        if key in keys:
            raise ValueError(f"Duplicate matchup in existing raw JSON: {key} in {path}")
        keys.add(key)
    return rows


def _merge(old: dict[str, str], incoming: dict[str, str], source: str) -> dict[str, str]:
    updated = dict(old)
    if source == "predictions":
        if old.get("score1", "").strip() or old.get("score2", "").strip():
            # Never alter pregame projections on a finalized game.
            if any(old.get(key, "") != incoming[key] for key in PREDICTION_FIELDS):
                raise ValueError(f"Cannot replace predictions for finalized game: {match_key(old)}")
            return updated
        for key in PREDICTION_FIELDS:
            updated[key] = incoming[key]
    else:
        if (old.get("score1", "").strip() or old.get("score2", "").strip()) and (
            old.get("score1") != incoming["score1"] or old.get("score2") != incoming["score2"]
        ):
            raise ValueError(f"Conflicting final scores for {match_key(old)}; refusing overwrite")
        # Preserve all previously captured pregame numbers and game identity.
        updated["score1"] = incoming["score1"]
        updated["score2"] = incoming["score2"]
        updated["game_status"] = "Final"
    return updated


def merge_into_files(rows: list[dict[str, str]], league: str, source: str, root: Path = BASE) -> list[tuple[Path, int, int]]:
    """Validate and stage all affected date files before any writes."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(_date(row), []).append(row)
    staged = []
    for date, new_rows in sorted(grouped.items()):
        path = root / league / f"{date}_{league}_raw.json"
        existing = load_existing(path, league)
        merged = [dict(row) for row in existing]
        by_key = {}
        for index, game in enumerate(merged):
            try:
                by_key[match_key(game)] = index
            except ValueError:
                # Preserve postponed/TBD raw records unchanged.
                pass
        added = updated = 0
        for row in new_rows:
            key = match_key(row)
            if key in by_key:
                idx = by_key[key]
                replacement = _merge(merged[idx], row, source)
                if merged[idx] != replacement:
                    merged[idx] = replacement
                    updated += 1
            else:
                by_key[key] = len(merged)
                merged.append(row)
                added += 1
        # Scraper also rejects duplicates using exact ET date/time/team identity.
        exact = set()
        for game in merged:
            identity = (game["sport"], game["date_time"], strip_record(game["team1"]), strip_record(game["team2"]))
            if identity in exact:
                raise ValueError(f"Duplicate raw identity in {path}: {identity}")
            exact.add(identity)
        staged.append((path, merged, added, updated))
    results = []
    for path, merged, added, updated in staged:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(merged, ensure_ascii=False, indent=2) + "\n"
        if not path.exists() or path.read_text(encoding="utf-8-sig") != serialized:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".manual_", suffix=".tmp", delete=False) as f:
                temp = Path(f.name)
                f.write(serialized)
            temp.replace(path)
        results.append((path, added, updated))
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--league", choices=sorted(LEAGUE_SPORT), required=True)
    ap.add_argument("--raw-file", type=Path, required=True)
    ap.add_argument("--time-basis", choices=["et", "utc"], required=True,
                    help="Whether pasted times are already Eastern or are UTC, like the scraper's webpage input")
    args = ap.parse_args(argv)
    text = args.raw_file.read_text(encoding="utf-8-sig")
    rows = parse_text(text, args.league, "predictions", args.time_basis)
    paths = merge_into_files(rows, args.league, "predictions")
    for path, added, updated in paths:
        print(f"WROTE/VERIFIED {path}: added={added} updated={updated}")
    print(f"SUCCESS: parsed={len(rows)} league={args.league} files={len(paths)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
