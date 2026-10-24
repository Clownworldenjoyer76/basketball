
#!/usr/bin/env python3
"""Merge basketball markets and model probabilities into Games Today CSVs.

Run from the repository root. Requires only the Python standard library.
The existing published filenames and directory structure are unchanged.
"""

import csv
import math
import os
import re
import tempfile
from pathlib import Path

INPUT_ROOT = Path("docs/win/basketball/01_merge")
MODEL_ROOT = Path("docs/win/basketball/03_edges/ev_kelly/wnba")
OUTPUT_ROOT = Path("frontend/data/index_home/basketball")
LEAGUES = ("WNBA", "NBA", "NCAAM")
MARKETS = ("spread", "total", "moneyline")
DATE_PATTERN = re.compile(r"\d{4}_\d{2}_\d{2}\Z")
MODEL_FIELDS = {
    "moneyline": ("home_model_prob", "away_model_prob"),
    "spread": ("home_spread_model_prob", "away_spread_model_prob"),
    "total": ("over_model_prob", "under_model_prob"),
}


def input_path(league: str, market: str, date: str) -> Path:
    return INPUT_ROOT / league.lower() / market / f"{date}_{league}_{market}.csv"


def model_path(league: str, market: str, date: str) -> Path:
    # The EV/Kelly folder is named "wnba" for all three leagues as specified.
    return MODEL_ROOT / market / f"{date}_{league}_{market}.csv"


def available_dates(league: str) -> list[str]:
    dates = set()
    for market in MARKETS:
        suffix = f"_{league}_{market}.csv"
        for path in (INPUT_ROOT / league.lower() / market).glob(f"*{suffix}"):
            date = path.name[: -len(suffix)]
            if path.is_file() and DATE_PATTERN.fullmatch(date):
                dates.add(date)
    return sorted(dates)


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or "game_id" not in fields or len(fields) != len(set(fields)):
            raise ValueError(f"Invalid or duplicate CSV headers in {path}")

        rows = []
        seen_ids = set()
        for line_no, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Unexpected extra CSV columns in {path}, line {line_no}")
            game_id = (row.get("game_id") or "").strip()
            if not game_id:
                raise ValueError(f"Missing game_id in {path}, line {line_no}")
            if game_id in seen_ids:
                raise ValueError(f"Duplicate game_id {game_id} in {path}")
            seen_ids.add(game_id)
            rows.append(row)

    return fields, rows


def validate_match(
    league: str,
    date: str,
    game_id: str,
    merged: dict[str, str],
    model: dict[str, str],
    source: Path,
) -> None:
    expected = {
        "league": league,
        "game_date": date,
        "home_team": merged.get("home_team", ""),
        "away_team": merged.get("away_team", ""),
    }
    for field, value in expected.items():
        actual = (model.get(field) or "").strip()
        if not value or actual.casefold() != value.strip().casefold():
            raise ValueError(
                f"Model match mismatch: {source}, game_id={game_id}, "
                f"{field}={actual!r}, expected={value!r}"
            )


def checked_probability(
    value: str, source: Path, game_id: str, field: str, required: bool
) -> str:
    raw = (value or "").strip()
    if not raw:
        if required:
            raise ValueError(
                f"Missing required model probability: {source}, "
                f"game_id={game_id}, field={field}"
            )
        return ""
    try:
        number = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid model probability: {source}, game_id={game_id}, "
            f"field={field}, value={raw!r}"
        ) from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(
            f"Out-of-range model probability: {source}, game_id={game_id}, "
            f"field={field}, value={raw!r}"
        )
    return raw


def combine(league: str, date: str) -> int:
    fields: list[str] = []
    games: dict[str, dict[str, str]] = {}

    # Retain the existing three-market merge and all original columns.
    for market in MARKETS:
        source = input_path(league, market, date)
        source_fields, source_rows = load_csv(source)
        for field in source_fields:
            if field not in fields:
                fields.append(field)
        for row in source_rows:
            game_id = row["game_id"].strip()
            combined = games.setdefault(game_id, {})
            for field, value in row.items():
                value = value or ""
                previous = combined.get(field, "")
                if previous and value and previous != value:
                    raise ValueError(
                        f"Conflicting value: {league} {date} game_id={game_id} "
                        f"column={field!r}, existing={previous!r}, {market}={value!r}"
                    )
                if value or field not in combined:
                    combined[field] = value

    # Add the model-specific fields without removing or renaming old fields.
    for field in ("home_prob", "away_prob", *sum(MODEL_FIELDS.values(), ())):
        if field not in fields:
            fields.append(field)

    for market in ("moneyline", "spread", "total"):
        source = model_path(league, market, date)
        if not source.is_file():
            if market == "moneyline":
                raise ValueError(f"Required moneyline model CSV missing: {source}")
            print(f"WARN {league} {date}: optional {market} model CSV missing: {source}")
            continue

        model_fields, model_rows = load_csv(source)
        required_headers = (*MODEL_FIELDS[market], "league", "game_date", "home_team", "away_team")
        missing_headers = [field for field in required_headers if field not in model_fields]
        if missing_headers:
            raise ValueError(f"Missing model headers in {source}: {', '.join(missing_headers)}")

        model_by_id = {row["game_id"].strip(): row for row in model_rows}
        unused_ids = sorted(set(model_by_id) - set(games))
        if unused_ids:
            print(f"WARN {league} {date} {market}: {len(unused_ids)} model game_id(s) absent from merge: {', '.join(unused_ids)}")

        for game_id, combined in games.items():
            model = model_by_id.get(game_id)
            if model is None:
                if market == "moneyline":
                    raise ValueError(
                        f"Missing moneyline model match: {source}, game_id={game_id}"
                    )
                print(f"WARN {league} {date} {market}: no model row for game_id={game_id}")
                continue

            validate_match(league, date, game_id, combined, model, source)
            for field in MODEL_FIELDS[market]:
                combined[field] = checked_probability(
                    model.get(field, ""), source, game_id, field,
                    required=(market == "moneyline"),
                )

            if market == "moneyline":
                # Win probabilities must originate in the moneyline model,
                # never sportsbook implied/market probabilities or 01_merge.
                combined["home_prob"] = combined["home_model_prob"]
                combined["away_prob"] = combined["away_model_prob"]

    destination = OUTPUT_ROOT / league / f"{date}_{league}_games_today.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(games.values())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)

    print(f"WROTE {destination} ({len(games)} games)")
    return len(games)


def main() -> None:
    output_files = 0
    for league in LEAGUES:
        for date in available_dates(league):
            missing = [market for market in MARKETS if not input_path(league, market, date).is_file()]
            if missing:
                print(f"SKIP {league} {date}: missing {', '.join(missing)} 01_merge CSV(s)")
                continue
            moneyline_model = model_path(league, "moneyline", date)
            if not moneyline_model.is_file():
                print(f"SKIP {league} {date}: required moneyline model CSV missing: {moneyline_model}")
                continue
            combine(league, date)
            output_files += 1
    print(f"DONE: {output_files} CSV file(s) written")


if __name__ == "__main__":
    main()
