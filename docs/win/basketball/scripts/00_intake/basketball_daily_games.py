#!/usr/bin/env python3
# docs/win/basketball/scripts/00_intake/basketball_daily_games.py
"""
Build canonical per-date basketball game identity files.

Primary identity source:
  docs/win/basketball/00_intake/sdv/canonical_games/{league}/*_games.csv

Those files are refreshed from SportsDataVerse/ESPN by sdv_canonical_games.py,
which uses explicit season translation from sdv_season_mapping.py.

Compatibility/fallback source:
  docs/win/basketball/00_intake/sportsbook/{league}/*.csv

Identity is league + game_date + normalized home + normalized away.
SDV/ESPN numeric game IDs are loaded first and remain canonical. Sportsbook
history may fill missing games/times and may contribute legacy/custom aliases.
Conflicting numeric IDs for one identity are fatal.
"""
from __future__ import annotations

import csv
import sys
import traceback
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

from sdv_canonical_games import build_current_canonical_games

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from basketball_shared import identity_key as shared_identity_key


BASE = Path("docs/win/basketball")

LEAGUES = {
    "nba": {
        "league_label": "NBA",
        "sdv_input_dir": BASE / "00_intake/sdv/canonical_games/nba",
        "sportsbook_input_dir": BASE / "00_intake/sportsbook/nba",
        "output_dir": BASE / "daily_games/nba",
    },
    "ncaam": {
        "league_label": "NCAAM",
        "sdv_input_dir": BASE / "00_intake/sdv/canonical_games/ncaam",
        "sportsbook_input_dir": BASE / "00_intake/sportsbook/ncaam",
        "output_dir": BASE / "daily_games/ncaam",
    },
    "wnba": {
        "league_label": "WNBA",
        "sdv_input_dir": BASE / "00_intake/sdv/canonical_games/wnba",
        "sportsbook_input_dir": BASE / "00_intake/sportsbook/wnba",
        "output_dir": BASE / "daily_games/wnba",
    },
}

for league_cfg in LEAGUES.values():
    league_cfg["output_dir"].mkdir(parents=True, exist_ok=True)

ERROR_DIR = BASE / "errors/00_intake"
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "basketball_daily_games.txt"
ALIAS_FILE = ERROR_DIR / "basketball_game_id_aliases.csv"
NY = ZoneInfo("America/New_York")

# Explicit schedule cancellation signals. Conditional/"if necessary" alone is
# not proof that a game is canceled; avoid guessing from that phrase.
INACTIVE_STATUSES = (
    "canceled", "cancelled", "postponed", "not necessary", "not needed",
    "will not be played", "abandoned",
)

FIELDNAMES = [
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "game_id",
]

ALIAS_FIELDS = [
    "league",
    "game_date",
    "home_team",
    "away_team",
    "alias_game_id",
    "canonical_game_id",
    "source_file",
]

UNRESOLVED_TEAM_NAMES = {
    "tbd",
    "to be determined",
    "unknown",
}


def log(msg: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


def clean(v) -> str:
    return "" if v is None else str(v).strip()


def unresolved_team(value) -> bool:
    team = clean(value).casefold()

    if not team or team in UNRESOLVED_TEAM_NAMES:
        return True

    # ESPN playoff placeholders can expose unresolved matchup labels such as
    # "Dream/Mystics" before the participating team is known.
    return "/" in team


def is_inactive_schedule_status(value: object) -> bool:
    status = clean(value).casefold()
    return any(text in status for text in INACTIVE_STATUSES)


def active_or_future_day(value: str, *, current_day: str) -> bool:
    # Lexical comparison valid exclusively for normalized YYYY_MM_DD dates.
    try:
        return (datetime.strptime(value, "%Y_%m_%d").strftime("%Y_%m_%d")
                >= current_day)
    except ValueError:
        return False


def american_price(value: object) -> bool:
    try:
        number = float(clean(value))
    except (ValueError, TypeError):
        return False
    return number == number and (number >= 100 or number <= -100) and abs(number) < 100000


def has_quoted_market(source: dict) -> bool:
    ml = all(american_price(source.get(k)) for k in (
        "home_dk_moneyline_american", "away_dk_moneyline_american"))
    spread = all(american_price(source.get(k)) for k in (
        "home_dk_spread_american", "away_dk_spread_american")) and all(
            clean(source.get(k)) for k in ("home_spread", "away_spread"))
    total = all(american_price(source.get(k)) for k in (
        "dk_total_over_american", "dk_total_under_american")) and clean(
            source.get("total")) != ""
    return bool(ml or spread or total)


def credible_current_sportsbook_fallback(source: dict, *, observed: datetime) -> bool:
    """Only an actual, timestamped, priced quote may create an active fixture.

    This is NOT independent proof that the game will occur or that its
    scheduled start time is correct; the PIT gate remains responsible for that.
    """
    if not clean(source.get("sportsbook_provider")):
        return False
    stamp = clean(source.get("scraped_at_utc"))
    if not stamp:
        return False
    try:
        issued = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return False
    if issued.tzinfo is None or issued.astimezone(timezone.utc) > observed.astimezone(timezone.utc):
        return False
    if not clean(source.get("game_time")):
        return False
    return has_quoted_market(source)


def build_row(row: dict) -> dict:
    return {k: clean(row.get(k)) for k in FIELDNAMES}


def identity_key(row: dict) -> tuple[str, str, str, str]:
    return shared_identity_key(
        row,
        clean,
        unresolved_team,
    )


def id_rank(game_id: str) -> int:
    gid = clean(game_id)
    if not gid:
        return 0
    return 2 if gid.isdigit() else 1


def sort_key(row: dict):
    return (
        clean(row.get("game_date")),
        clean(row.get("game_time")),
        clean(row.get("home_team")),
        clean(row.get("away_team")),
        clean(row.get("game_id")),
    )


def choose(
    existing: dict,
    candidate: dict,
    source_file: str,
    aliases: list[dict],
) -> dict:
    old_id = clean(existing.get("game_id"))
    new_id = clean(candidate.get("game_id"))

    if old_id.isdigit() and new_id.isdigit() and old_id != new_id:
        raise ValueError(
            "Conflicting numeric game_ids for the same game identity: "
            f"{identity_key(existing)} -> {old_id} vs {new_id}"
        )

    if id_rank(new_id) > id_rank(old_id):
        canonical, alias = new_id, old_id
        chosen = candidate.copy()
        if not clean(chosen.get("game_time")):
            chosen["game_time"] = existing.get("game_time", "")
    else:
        canonical, alias = old_id, new_id
        chosen = existing
        if not clean(chosen.get("game_time")) and clean(candidate.get("game_time")):
            chosen = chosen.copy()
            chosen["game_time"] = candidate.get("game_time", "")

    if alias and alias != canonical:
        aliases.append(
            {
                "league": clean(chosen.get("league")),
                "game_date": clean(chosen.get("game_date")),
                "home_team": clean(chosen.get("home_team")),
                "away_team": clean(chosen.get("away_team")),
                "alias_game_id": alias,
                "canonical_game_id": canonical,
                "source_file": source_file,
            }
        )

    return chosen


def write_aliases(rows: list[dict]) -> None:
    unique = {}
    for row in rows:
        key = tuple(row.get(c, "") for c in ALIAS_FIELDS[:-1])
        unique[key] = row

    with ALIAS_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALIAS_FIELDS)
        writer.writeheader()
        writer.writerows(unique.values())


def remove_unresolved_only_output_files(
    output_dir: Path,
    league_label: str,
    active_dates: set[str],
) -> int:
    removed = 0

    for path in sorted(
        output_dir.glob(f"*_{league_label}.csv")
    ):
        game_date = path.name.removesuffix(
            f"_{league_label}.csv"
        )

        if game_date in active_dates:
            continue

        with path.open(
            newline="",
            encoding="utf-8-sig",
        ) as f:
            rows = list(csv.DictReader(f))

        if not rows:
            continue

        if all(
            unresolved_team(row.get("home_team"))
            or unresolved_team(row.get("away_team"))
            for row in rows
        ):
            path.unlink()
            removed += 1
            log(
                "REMOVED stale unresolved-only daily-games file | "
                f"league={league_label} | "
                f"path={path} | "
                f"rows={len(rows)}"
            )

    return removed


def remove_stale_suppressed_daily_file(
    *, output_dir: Path, league_label: str, active_dates: set[str],
    rejected: dict[str, set[tuple[str, str, str, str]]],
    current_day: str,
) -> int:
    """Remove a future/today ghost slate only when every old row was rejected.

    Other historical/current games are never silently deleted. A mixed file
    raises an error rather than silently certifying an incomplete slate.
    """
    removed = 0
    for game_date, rejected_keys in rejected.items():
        if game_date in active_dates or not active_or_future_day(game_date, current_day=current_day):
            continue
        target = output_dir / f"{game_date}_{league_label}.csv"
        if not target.exists():
            continue
        with target.open(newline="", encoding="utf-8-sig") as file:
            old_rows = list(csv.DictReader(file))
        if not old_rows:
            continue
        all_rejected = all(identity_key(row) in rejected_keys for row in old_rows)
        if not all_rejected:
            raise RuntimeError(
                f"Stale daily slate contains rows not supported by current sources; "
                f"manual review required instead of deleting: {target}"
            )
        target.unlink()
        removed += 1
        log(f"REMOVED obsolete active daily slate: {target} ({len(old_rows)} rows)")
    return removed


def load_source_files(
    *,
    league_label: str,
    source_kind: str,
    folder: Path,
    pattern: str,
    canonical: dict[tuple[str, str, str, str], dict],
    aliases: list[dict],
    inactive_keys: set[tuple[str, str, str, str]] | None = None,
    rejected: dict[str, set[tuple[str, str, str, str]]] | None = None,
    observed: datetime | None = None,
) -> tuple[int, int, int]:
    files = 0
    rows_read = 0
    duplicates = 0
    inactive_keys = inactive_keys if inactive_keys is not None else set()
    rejected = rejected if rejected is not None else {}
    observed = observed if observed is not None else datetime.now(timezone.utc)
    current_day = observed.astimezone(NY).strftime("%Y_%m_%d")

    if not folder.exists():
        log(f"{source_kind.upper()} INPUT DIR NOT FOUND: {folder}")
        return files, rows_read, duplicates

    for csv_path in sorted(folder.glob(pattern)):
        files += 1
        log(f"PROCESSING {source_kind.upper()} {csv_path}")

        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for source in reader:
                rows_read += 1
                row = build_row(source)

                if not row["league"]:
                    row["league"] = league_label
                if not row["sport"]:
                    row["sport"] = "basketball"

                if (
                    not row["game_date"]
                    or not row["home_team"]
                    or not row["away_team"]
                ):
                    log(
                        f"SKIPPED malformed {source_kind} identity row "
                        f"in {csv_path.name}"
                    )
                    continue

                if (
                    unresolved_team(row["home_team"])
                    or unresolved_team(row["away_team"])
                ):
                    log(
                        "SKIPPED unresolved "
                        f"{source_kind} identity row "
                        f"in {csv_path.name} | "
                        f"game_date={row['game_date']} | "
                        f"game_id={row['game_id']} | "
                        f"home_team={row['home_team']!r} | "
                        f"away_team={row['away_team']!r}"
                    )
                    continue

                key = identity_key(row)

                if not all(key):
                    log(
                        f"SKIPPED malformed {source_kind} identity key "
                        f"in {csv_path.name}"
                    )
                    continue

                active_date = active_or_future_day(row["game_date"], current_day=current_day)
                if source_kind == "sdv" and is_inactive_schedule_status(source.get("status")):
                    inactive_keys.add(key)
                    canonical.pop(key, None)
                    if active_date:
                        rejected.setdefault(row["game_date"], set()).add(key)
                    log(f"SKIPPED inactive canonical fixture: {row['game_date']} {key}")
                    continue
                if source_kind == "sportsbook" and key in inactive_keys:
                    if active_date:
                        rejected.setdefault(row["game_date"], set()).add(key)
                    log(f"SKIPPED sportsbook fixture rejected by canonical status: {row['game_date']} {key}")
                    continue
                if (source_kind == "sportsbook" and active_date and key not in canonical
                        and not credible_current_sportsbook_fallback(source, observed=observed)):
                    rejected.setdefault(row["game_date"], set()).add(key)
                    log(f"SKIPPED unpriced/undated sportsbook-only active fixture: {row['game_date']} {key}")
                    continue

                if key in canonical:
                    duplicates += 1
                    canonical[key] = choose(
                        canonical[key],
                        row,
                        str(csv_path),
                        aliases,
                    )
                else:
                    canonical[key] = row

    return files, rows_read, duplicates


def main() -> None:
    LOG_FILE.write_text(
        f"=== basketball_daily_games RUN {datetime.now().isoformat()} ===\n",
        encoding="utf-8",
    )

    files_written = []
    aliases: list[dict] = []
    errors = 0

    total_sdv_files = 0
    total_sdv_rows = 0
    total_sportsbook_files = 0
    total_sportsbook_rows = 0
    total_identity_duplicates = 0
    total_rows_written = 0

    try:
        refreshed = build_current_canonical_games()
        log(f"SDV CANONICAL REFRESHED: {len(refreshed)} season files")

        for cfg in LEAGUES.values():
            league_label = cfg["league_label"]
            canonical: dict[tuple[str, str, str, str], dict] = {}
            inactive_keys: set[tuple[str, str, str, str]] = set()
            rejected: dict[str, set[tuple[str, str, str, str]]] = {}
            observed = datetime.now(timezone.utc)

            (
                sdv_files,
                sdv_rows,
                sdv_duplicates,
            ) = load_source_files(
                league_label=league_label,
                source_kind="sdv",
                folder=cfg["sdv_input_dir"],
                pattern="*_games.csv",
                canonical=canonical,
                aliases=aliases,
                inactive_keys=inactive_keys,
                rejected=rejected,
                observed=observed,
            )

            (
                sportsbook_files,
                sportsbook_rows,
                sportsbook_duplicates,
            ) = load_source_files(
                league_label=league_label,
                source_kind="sportsbook",
                folder=cfg["sportsbook_input_dir"],
                pattern="*.csv",
                canonical=canonical,
                aliases=aliases,
                inactive_keys=inactive_keys,
                rejected=rejected,
                observed=observed,
            )

            total_sdv_files += sdv_files
            total_sdv_rows += sdv_rows
            total_sportsbook_files += sportsbook_files
            total_sportsbook_rows += sportsbook_rows
            total_identity_duplicates += (
                sdv_duplicates + sportsbook_duplicates
            )

            by_date: dict[str, list[dict]] = {}
            for row in canonical.values():
                by_date.setdefault(row["game_date"], []).append(row)

            remove_unresolved_only_output_files(
                cfg["output_dir"],
                league_label,
                set(by_date),
            )
            remove_stale_suppressed_daily_file(
                output_dir=cfg["output_dir"], league_label=league_label,
                active_dates=set(by_date), rejected=rejected,
                current_day=observed.astimezone(NY).strftime("%Y_%m_%d"),
            )

            for game_date, rows in sorted(by_date.items()):
                rows = sorted(rows, key=sort_key)
                out_path = (
                    cfg["output_dir"]
                    / f"{game_date}_{league_label}.csv"
                )

                with out_path.open(
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=FIELDNAMES,
                    )
                    writer.writeheader()
                    writer.writerows(rows)

                files_written.append((str(out_path), len(rows)))
                total_rows_written += len(rows)
                log(f"WROTE {out_path} ({len(rows)} rows)")

    except Exception as exc:
        errors += 1
        log(f"FATAL: {exc}\n{traceback.format_exc()}")

    write_aliases(aliases)

    log("--- SUMMARY ---")
    log(f"SDV season files read: {total_sdv_files}")
    log(f"SDV rows read: {total_sdv_rows}")
    log(f"Sportsbook files read: {total_sportsbook_files}")
    log(f"Sportsbook rows read: {total_sportsbook_rows}")
    log(f"Duplicate identities resolved: {total_identity_duplicates}")
    log(f"ID aliases recorded: {len(aliases)}")
    log(f"Rows written: {total_rows_written}")
    log(f"Files written: {len(files_written)}")
    log(f"Errors: {errors}")
    log(f"STATUS: {'SUCCESS' if errors == 0 else 'FAILED'}")

    if errors:
        sys.exit(1)

    print("Basketball daily games complete.")


if __name__ == "__main__":
    main()
