#!/usr/bin/env python3
"""Isolated 2026 WNBA ESPN reconstruction and point-in-time SDV replay.

No writes under --repo. ESPN scoreboards and summaries are persisted to --output
for reproducibility. Replays the production feature-generation functions in
memory with a read-only monkeypatch for history reads; NEVER invokes their writers.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

try:
    NY = ZoneInfo("America/New_York")
except Exception:
    # Full default 2026 audit window is in US daylight saving time.
    # Exact UTC-04 fallback avoids requiring tzdata just to scan ESPN dates.
    NY = timezone(timedelta(hours=-4))
TEAM_NAMES = {
    "atlanta dream", "chicago sky", "connecticut sun", "dallas wings",
    "golden state valkyries", "indiana fever", "las vegas aces",
    "los angeles sparks", "minnesota lynx", "new york liberty",
    "phoenix mercury", "portland fire", "seattle storm",
    "toronto tempo", "washington mystics",
}
TEAM_ALIASES = {
    "la sparks": "los angeles sparks", "ny liberty": "new york liberty",
    "gs valkyries": "golden state valkyries", "vegas aces": "las vegas aces",
    "conn sun": "connecticut sun", "wash mystics": "washington mystics",
    "minn lynx": "minnesota lynx", "tor tempo": "toronto tempo",
}
REQUIRED_TEAM_STATS = (
    "field_goals_made", "field_goals_attempted",
    "three_point_field_goals_made", "free_throws_attempted",
    "offensive_rebounds", "defensive_rebounds", "total_turnovers",
)


def norm(name):
    x = re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()
    return TEAM_ALIASES.get(x, x)


def num(v):
    try:
        if v is None or str(v).strip() in ("", "--", "-"):
            return None
        n = float(str(v).replace(",", "").replace("%", "").strip())
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def integer(v):
    n = num(v)
    return int(n) if n is not None and n >= 0 and n == int(n) else None


def key(d, h, a, hs, a_s):
    return (str(d), norm(h), norm(a), integer(hs), integer(a_s))


def iso_date(utc):
    return datetime.fromisoformat(str(utc).replace("Z", "+00:00")).astimezone(NY).strftime("%Y_%m_%d")


def get_json(url, retries=3):
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "WNBA-SDV-Isolated-Audit/1.0", "Accept": "application/json"})
            with urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(attempt)
    raise RuntimeError(f"Unable to retrieve {url}: {last}")


def save_csv(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not columns:
        columns = list(dict.fromkeys(k for row in rows for k in row)) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        out = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        out.writeheader()
        out.writerows(rows)


def log_issue(issues, severity, reason, **details):
    issues.append({"severity": severity, "reason": reason, **details})


def read_archive(base, start, end):
    folder = base / "05_final_scores/results/wnba"
    valid = {}; by_match = defaultdict(list); by_fixture = defaultdict(list)
    invalid = []; dups = []
    for path in sorted(folder.glob("2026_*_final_scores_WNBA.csv")):
        with path.open(encoding="utf-8-sig", newline="") as f:
            for idx, row in enumerate(csv.DictReader(f), 2):
                d = (row.get("game_date") or "").strip()
                if not (start <= d <= end):
                    continue
                gid = (row.get("game_id") or "").strip()
                info = {"archive_id": gid, "game_date": d, "home": row.get("home_team", ""),
                        "away": row.get("away_team", ""), "home_score": row.get("home_score", ""),
                        "away_score": row.get("away_score", ""), "source": path.name,
                        "line": idx}
                k = key(d, info["home"], info["away"], info["home_score"], info["away_score"])
                if not gid or not k[1] or not k[2] or k[3] is None or k[4] is None:
                    invalid.append(info)
                    continue
                if gid in valid:
                    dups.append(info)
                    continue
                valid[gid] = info
                by_match[k].append(info)
                by_fixture[k[:3]].append(info)
    return valid, by_match, by_fixture, invalid, dups


def scoreboard_event(event, issues):
    gid = str(event.get("id", "")).strip()
    if not gid or event.get("status", {}).get("type", {}).get("completed") is not True:
        return None
    stype = integer(event.get("season", {}).get("type"))
    if stype not in (2, 3):
        return None
    try:
        d = iso_date(event["date"])
        competitors = event["competitions"][0]["competitors"]
        h = [c for c in competitors if c.get("homeAway") == "home"]
        a = [c for c in competitors if c.get("homeAway") == "away"]
        if len(h) != 1 or len(a) != 1:
            raise ValueError("home/away competitor count")
        h, a = h[0], a[0]
        hn, an = h["team"]["displayName"], a["team"]["displayName"]
        hs, av = integer(h.get("score")), integer(a.get("score"))
        if hs is None or av is None:
            raise ValueError("missing scoreboard scores")
        comp = event["competitions"][0]
        return {"game_id": gid, "game_date": d, "utc_start": event["date"],
                "home": hn, "away": an,
                "home_id": str(h["team"]["id"]), "away_id": str(a["team"]["id"]),
                "home_score": hs, "away_score": av, "season_type": stype,
                "neutral_site": bool(comp.get("neutralSite", False)),
                "venue_id": str((comp.get("venue") or {}).get("id") or ""),
                "venue_name": str((comp.get("venue") or {}).get("fullName") or "")}
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        log_issue(issues, "error", "SCOREBOARD_PARSE", game_id=gid, detail=str(exc))
        return None


def normstat(label):
    return re.sub(r"[^a-z0-9]", "", str(label or "").lower())


def pair(v):
    match = re.match(r"^\s*(\d+)\s*[-/]\s*(\d+)\s*$", str(v or ""))
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)


def team_statistics(entry):
    stats = {}
    for it in entry.get("statistics", []):
        raw_name = str(it.get("name") or it.get("label") or "")
        v = it.get("displayValue", it.get("value"))
        stats[normstat(raw_name)] = v
    def get(*names):
        for name in names:
            value = stats.get(normstat(name))
            if value is not None:
                return num(value)
        return None
    fg = pair(stats.get("fieldgoalsmadefieldgoalsattempted") or stats.get("fieldgoals") or stats.get("fg"))
    three = pair(stats.get("threepointfieldgoalsmadethreepointfieldgoalsattempted") or stats.get("threepointfieldgoals") or stats.get("3pt"))
    ft = pair(stats.get("freethrowsmadefreethrowsattempted") or stats.get("freethrows") or stats.get("ft"))
    return {
        "field_goals_made": get("fieldGoalsMade") if fg[0] is None else fg[0],
        "field_goals_attempted": get("fieldGoalsAttempted") if fg[1] is None else fg[1],
        "three_point_field_goals_made": get("threePointFieldGoalsMade") if three[0] is None else three[0],
        "three_point_field_goals_attempted": get("threePointFieldGoalsAttempted") if three[1] is None else three[1],
        "free_throws_made": get("freeThrowsMade") if ft[0] is None else ft[0],
        "free_throws_attempted": get("freeThrowsAttempted") if ft[1] is None else ft[1],
        "offensive_rebounds": get("offensiveRebounds", "offRebounds", "offensive"),
        "defensive_rebounds": get("defensiveRebounds", "defRebounds", "defensive"),
        "rebounds": get("totalRebounds", "rebounds"),
        "total_turnovers": get("turnovers", "totalTurnovers"),
        "turnovers": get("turnovers", "totalTurnovers"),
        "assists": get("assists"), "steals": get("steals"), "blocks": get("blocks"),
    }


def parse_min(v):
    if v is None or str(v) in ("--", "-"):
        return None
    if ":" in str(v):
        a, b = str(v).split(":", 1)
        if a.strip().isdigit() and b.strip().isdigit():
            return round(int(a) + int(b)/60, 3)
    return num(v)


def player_rows(entry, game, gid, issues):
    result = []
    team = entry.get("team") or {}
    tid = str(team.get("id") or "")
    for group in entry.get("statistics") or []:
        names = [normstat(x) for x in group.get("labels", [])]
        for p in group.get("athletes", []) or []:
            athlete = p.get("athlete") or {}
            pid = str(athlete.get("id") or "")
            if not pid:
                continue
            vals = p.get("stats") or []
            obj = dict(zip(names, vals))
            def val(*keys):
                for k in keys:
                    if normstat(k) in obj:
                        return obj[normstat(k)]
                return None
            fg = pair(val("FG")); three = pair(val("3PT")); ft = pair(val("FT"))
            result.append({
                "game_id": gid, "season": 2026, "game_date": game["game_date"],
                "game_date_time": game["utc_start"], "team_id": tid,
                "athlete_id": pid, "player_id": pid,
                "minutes": parse_min(val("MIN")), "points": num(val("PTS")),
                "rebounds": num(val("REB")), "assists": num(val("AST")),
                "steals": num(val("STL")), "blocks": num(val("BLK")),
                "turnovers": num(val("TO", "TOV")),
                "field_goals_made": fg[0], "field_goals_attempted": fg[1],
                "three_point_field_goals_made": three[0],
                "three_point_field_goals_attempted": three[1],
                "free_throws_made": ft[0], "free_throws_attempted": ft[1],
            })
    return result


def summary_tables(s, game, issues):
    gid = game["game_id"]
    bt = s.get("boxscore") or {}
    ts = bt.get("teams") or []
    ps = bt.get("players") or []
    if len(ts) != 2 or len(ps) != 2:
        log_issue(issues, "error", "SUMMARY_INCOMPLETE_GROUPS", game_id=gid,
                  team_groups=len(ts), player_groups=len(ps))
        return None
    team_ids = {game["home_id"], game["away_id"]}
    try:
        h = (s.get("header") or {}).get("competitions") or []
        if h and h[0].get("status", {}).get("type", {}).get("completed") is False:
            raise ValueError("summary game not marked complete")
        # Prefer summary competition scores if supplied to cross-validate scoreboard.
        if h:
            competitors = h[0].get("competitors") or []
            for c in competitors:
                side = c.get("homeAway")
                if side in ("home", "away"):
                    expected = game[side + "_score"]
                    found = integer(c.get("score"))
                    if found is not None and found != expected:
                        raise ValueError(f"summary/scoreboard {side} score conflict: {found} vs {expected}")
        rows = []
        for entry in ts:
            tid = str((entry.get("team") or {}).get("id") or "")
            if tid not in team_ids:
                raise ValueError(f"summary team id {tid} not in scoreboard IDs {team_ids}")
            op = game["away_id"] if tid == game["home_id"] else game["home_id"]
            sc = game["home_score"] if tid == game["home_id"] else game["away_score"]
            osc = game["away_score"] if tid == game["home_id"] else game["home_score"]
            values = team_statistics(entry)
            missing = [col for col in REQUIRED_TEAM_STATS if values.get(col) is None]
            if missing:
                raise ValueError("missing required ESPN box stats: " + ",".join(missing))
            rows.append({"game_id": gid, "season": 2026,
                         "game_date": game["game_date"], "game_date_time": game["utc_start"],
                         "team_id": tid, "opponent_team_id": op,
                         "team_home_away": "home" if tid == game["home_id"] else "away",
                         "team_score": sc, "opponent_team_score": osc, **values})
        if len({r["team_id"] for r in rows}) != 2:
            raise ValueError("duplicate summary team id")
        players = []
        for entry in ps:
            tid = str((entry.get("team") or {}).get("id") or "")
            if tid not in team_ids:
                raise ValueError("player group team id does not match scoreboard")
            players.extend(player_rows(entry, game, gid, issues))
        counts = Counter(p["team_id"] for p in players)
        if any(counts.get(tid, 0) < 5 for tid in team_ids):
            raise ValueError(f"too few parsed player rows: {dict(counts)}")
        if sum(p.get("points") is not None for p in players) < 10:
            raise ValueError("player points unavailable")
        game_row = {
            "game_id": gid, "id": gid, "game_date": game["game_date"],
            "date": game["utc_start"], "game_date_time": game["utc_start"],
            "internal_season": 2026, "sdv_season": 2026,
            "home_team_id": game["home_id"], "away_team_id": game["away_id"],
            "home_id": game["home_id"], "away_id": game["away_id"],
            "home_score": game["home_score"], "away_score": game["away_score"],
            "home_display_name": game["home"], "away_display_name": game["away"],
            "neutral_site": game["neutral_site"], "venue_id": game["venue_id"],
            "home_venue_id": game["venue_id"] if not game["neutral_site"] else None,
            "away_venue_id": None, "venue_full_name": game["venue_name"],
        }
        return game_row, rows, players
    except (KeyError, ValueError, TypeError) as exc:
        log_issue(issues, "error", "SUMMARY_SCHEMA_OR_SCORE_CONFLICT", game_id=gid, detail=str(exc))
        return None


def build(args, base, output):
    import polars as pl
    start = date.fromisoformat(args.from_date)
    end = date.fromisoformat(args.through_date)
    score_dir = output / "espn_scoreboards"
    summary_dir = output / "espn_summaries"
    score_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    issues = []
    arch, arch_by_key, arch_by_fixture, invalid_archive, duplicates = read_archive(
        base, start.strftime("%Y_%m_%d"), end.strftime("%Y_%m_%d"))
    for r in invalid_archive:
        log_issue(issues, "warning", "INVALID_ARCHIVE_ROW", **r)
    for r in duplicates:
        log_issue(issues, "error", "DUPLICATE_ARCHIVE_ID", **r)
    games = {}
    scanned = 0
    for i in range((end-start).days + 1):
        d = start + timedelta(days=i)
        tag = d.strftime("%Y%m%d")
        path = score_dir / (tag + ".json")
        if path.is_file() and not args.refresh:
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            try:
                data = get_json(f"https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates={tag}&limit=500")
                path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                if args.delay_ms:
                    time.sleep(args.delay_ms/1000)
            except Exception as exc:
                log_issue(issues, "error", "SCOREBOARD_HTTP_FAILURE", game_date=tag, detail=str(exc))
                continue
        scanned += 1
        for event in data.get("events", []):
            obj = scoreboard_event(event, issues)
            if obj is None:
                continue
            if not (args.from_date.replace("-", "_") <= obj["game_date"] <= args.through_date.replace("-", "_")):
                continue
            if norm(obj["home"]) not in TEAM_NAMES or norm(obj["away"]) not in TEAM_NAMES:
                log_issue(issues, "info", "EXCLUDED_NON_WNBA_CLUB_GAME", game_id=obj["game_id"],
                          game_date=obj["game_date"], home=obj["home"], away=obj["away"])
                continue
            if obj["game_id"] in games and obj != games[obj["game_id"]]:
                log_issue(issues, "error", "DUPLICATE_ESPN_DIFFERENT_CONTENT", game_id=obj["game_id"])
                continue
            games[obj["game_id"]] = obj
        if (i+1) % 30 == 0:
            print(f"ESPN dates scanned {i+1}/{(end-start).days+1}; league games {len(games)}", flush=True)
    if scanned != (end - start).days + 1:
        log_issue(issues, "error", "INCOMPLETE_ESPN_DATE_COVERAGE", retrieved=scanned,
                  expected=(end-start).days + 1)
    matches = 0; aliases = 0; missing_archive = 0; archive_disputed = 0
    trusted = []; box_rows = []; players = []; audit_matches = []
    summaries_attempted = 0
    consumed_archive_ids = set()
    for k, game in sorted(games.items(), key=lambda x: (x[1]["game_date"], x[0])):
        key0 = key(game["game_date"], game["home"], game["away"], game["home_score"], game["away_score"])
        matches_for_game = arch_by_key.get(key0, [])
        same_fixture = arch_by_fixture.get(key0[:3], [])
        conflict = bool(same_fixture and not matches_for_game)
        if conflict:
            archive_disputed += 1
            log_issue(issues, "error", "ARCHIVE_ESPN_SCORE_CONFLICT_QUARANTINED", game_id=k,
                      game_date=game["game_date"], home=game["home"], away=game["away"],
                      espn_score=f'{game["home_score"]}-{game["away_score"]}',
                      archive_scores=";".join(f'{r["home_score"]}-{r["away_score"]}' for r in same_fixture))
        if len(matches_for_game) > 1:
            log_issue(issues, "error", "AMBIGUOUS_ARCHIVE_IDENTITY", game_id=k)
            continue
        if matches_for_game:
            a = matches_for_game[0]
            consumed_archive_ids.add(a["archive_id"])
            matches += 1
            aliases += int(k != a["archive_id"])
            audit_matches.append({"game_date": game["game_date"], "espn_id": k,
                                  "archive_id": a["archive_id"], "home": game["home"],
                                  "away": game["away"], "score": f'{game["home_score"]}-{game["away_score"]}'})
        elif not conflict:
            missing_archive += 1
            log_issue(issues, "warning", "ESPN_GAME_MISSING_IN_ARCHIVE", game_id=k,
                      game_date=game["game_date"], home=game["home"], away=game["away"])
        if conflict:
            continue   # Quarantined disputed games are never used for feature history or replay.
        spath = summary_dir / (k + ".json")
        if spath.is_file() and not args.refresh:
            summary = json.loads(spath.read_text(encoding="utf-8"))
        else:
            try:
                summary = get_json(f"https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary?event={k}")
                spath.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
            except Exception as exc:
                log_issue(issues, "error", "SUMMARY_HTTP_FAILURE", game_id=k, detail=str(exc))
                continue
        summaries_attempted += 1
        parsed = summary_tables(summary, game, issues)
        if parsed is None:
            if summaries_attempted == 12 and len(trusted) < 2:
                save_csv(output/"quarantine_and_flags.csv", issues)
                raise RuntimeError("Most initial ESPN summaries failed schema validation; inspect quarantine_and_flags.csv before downloading all games")
            continue
        grow, tb, plrows = parsed
        trusted.append(grow)
        box_rows.extend(tb)
        players.extend(plrows)
        if len(trusted) % 50 == 0:
            print(f"Validated complete ESPN box scores: {len(trusted)}", flush=True)
        if args.delay_ms:
            time.sleep(args.delay_ms/1000)
    for aid, item in arch.items():
        if aid not in consumed_archive_ids:
            log_issue(issues, "warning", "ARCHIVE_GAME_UNMATCHED", **item)
    # An ESPN-only game requires a validated summary boxscore as independent support;
    # all archive conflicts are excluded rather than implicitly resolving disputed scores.
    # Other discrepancies
    # are quarantined, and the original archive is never imported into staged history.
    stage_dir = output / "isolated_history" / "wnba" / "2026"
    stage_dir.mkdir(parents=True, exist_ok=True)
    # Use explicit optional schema columns for clarity; string/date fields stay faithful.
    for rows, filename in ((trusted,"games"),(box_rows,"team_game"),(players,"player_game")):
        if not rows:
            continue
        pl.DataFrame(rows, infer_schema_length=None, strict=False).write_parquet(stage_dir/(filename+".parquet"))
    save_csv(output/"espn_vs_archive_crosswalk.csv", audit_matches)
    save_csv(output/"quarantine_and_flags.csv", issues)
    save_csv(output/"validated_2026_games.csv", [dict(game_id=g["game_id"],game_date=g["game_date"],
             home_team_id=g["home_team_id"],away_team_id=g["away_team_id"],
             home_score=g["home_score"],away_score=g["away_score"]) for g in trusted])
    manifest = {"source": "ESPN daily scoreboard and individual game summary",
                "retrieved_date_window": [args.from_date, args.through_date],
                "loaded_dates": scanned, "expected_dates": (end-start).days+1,
                "completed_wnba_regular_playoff": len(games),
                "matched_to_archive": matches, "alias_ids": aliases,
                "espn_only_archive_missing": missing_archive,
                "score_conflicts_quarantined": archive_disputed,
                "boxscore_validated_games": len(trusted),
                "boxscore_team_rows": len(box_rows), "player_rows": len(players),
                "quarantine_counts": dict(Counter(x["reason"] for x in issues)),
                "allstar_or_non_club_exclusions": sum(x["reason"]=="EXCLUDED_NON_WNBA_CLUB_GAME" for x in issues),
                "source_statistics_method": "ESPN summary team boxscore; estimated possessions when historical possessions unavailable",
                "status": "candidate_research_only_no_production_use"}
    (output/"build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nISOLATED HISTORY RECONSTRUCTION")
    for name, val in manifest.items():
        if name not in ("quarantine_counts",):
            print(f"{name}: {val}")
    print("Quarantine/flags:", manifest["quarantine_counts"])
    if args.stage == "build":
        return manifest
    fatal = sum(x["severity"] == "error" and x["reason"] in (
        "SCOREBOARD_HTTP_FAILURE", "INCOMPLETE_ESPN_DATE_COVERAGE",
        "DUPLICATE_ESPN_DIFFERENT_CONTENT") for x in issues)
    if fatal:
        raise RuntimeError(f"Replay blocked: ESPN coverage or identity errors ({fatal}); inspect quarantine_and_flags.csv")
    if len(trusted) < len(games)*0.90:
        raise RuntimeError(f"Replay blocked: only {len(trusted)}/{len(games)} validated boxscores")
    return manifest


def load_past_archive(base, from_date, through_date):
    preds = base/"00_intake/predictions/predictions_cleaned/wnba"
    scorebase = base/"05_final_scores/results/wnba"
    result = []
    for path in sorted(preds.glob("*_WNBA_predictions.csv")):
        d = path.name[:10]
        if d < from_date.replace("-", "_") or d > through_date.replace("-", "_"):
            continue
        scorefile = scorebase/(d+"_final_scores_WNBA.csv")
        if not scorefile.is_file():
            continue
        with scorefile.open(encoding="utf-8-sig", newline="") as f:
            scores = {str(r.get("game_id") or ""):r for r in csv.DictReader(f)}
        with path.open(encoding="utf-8-sig", newline="") as f:
            for p in csv.DictReader(f):
                gid = str(p.get("game_id") or "")
                s = scores.get(gid)
                if not s or str(p.get("model_source") or "").strip().lower() != "ensemble":
                    continue
                if norm(p.get("home_team")) != norm(s.get("home_team")) or norm(p.get("away_team")) != norm(s.get("away_team")):
                    continue
                if p.get("game_date") != s.get("game_date"):
                    continue
                if any(num(p.get(c)) is None for c in ("home_prob","dratings_home_prob","sdv_home_prob","sdv_expected_margin")):
                    continue
                margin = num(s.get("home_score"))-num(s.get("away_score"))
                if margin == 0:
                    continue
                result.append((p,s))
    return result


def model_replay(feature, model):
    enc = model["encoder"]
    order = enc["encoded_feature_order"]
    beta = [float(c["value"]) for c in model["coefficients"]]
    if len(beta) != len(order):
        raise ValueError("Coefficient mismatch")
    vec = [0.0]*len(order)
    index = {n:i for i,n in enumerate(order)}
    vec[int(enc["intercept"]["encoded_index"])]=float(enc["intercept"]["value"])
    for n in enc["numeric_scaling"]:
        name=n["name"]
        raw=feature.get(name)
        v=num(raw)
        if v is None:
            v=num(n["missing_fill_value"])
        if v is None:
            raise ValueError(f"Invalid numeric feature {name}")
        sd=float(n["std"])
        if sd <= 0:
            raise ValueError("Invalid feature standard deviation: "+name)
        vec[index[name]]=(v-float(n["mean"]))/sd
    for n in enc["categorical_encoding"]:
        token=str(feature.get(n["name"]) or "").strip() or str(n["missing_token"])
        mapping=n["encoded_index_by_level"]
        if token not in mapping:
            token=str(n["unknown_token"])
        vec[int(mapping[token])]=1
    return sum(v*b for v,b in zip(vec,beta))


def n_cdf(v):
    return min(max(0.5*(1+math.erf(v/math.sqrt(2))),1e-12), 1-1e-12)


def proper_scores(rows, col):
    if not rows:
        return {"N":0}
    brier=0; ll=0; picks=0; err_margin=[]
    for x in rows:
        p=min(max(x[col],1e-12),1-1e-12)
        y=x["y"]
        brier+=(p-y)**2
        ll-= y*math.log(p)+(1-y)*math.log(1-p)
        picks+=int((p>=0.5)==bool(y))
    return {"N":len(rows),"Brier":round(brier/len(rows),4),"LogLoss":round(ll/len(rows),4),
            "Picks":f"{picks}/{len(rows)}"}


def replay(args, base, output, manifest):
    import polars as pl
    sys.path.insert(0,str((base/"scripts/00_intake").resolve()))
    try:
        import sdv_feature_generation as fg
    except ImportError as e:
        raise RuntimeError(f"Cannot load production feature generator (polars/PyYAML required): {e}") from e
    import yaml
    cfg=yaml.safe_load((base/"config/sdv_model.yaml").read_text(encoding="utf-8"))
    model=json.loads((base/"models/sdv/wnba/margin_model.json").read_text(encoding="utf-8"))
    oldroot=(base/"00_intake/sdv/history").resolve()
    newdir=(output/"isolated_history/wnba/2026").resolve()
    for name in ("games","team_game","player_game"):
        if not (newdir/(name+".parquet")).is_file():
            raise RuntimeError("Missing staged table "+name)
    # Read-only shim: production builders read 2023-25 from their original locations;
    # add 2026 isolated parquet tables in RAM, do not alter original import or config.
    original_seasons=fg.history_seasons
    original_read=fg.read_history_table
    def extra_seasons(path,league):
        years=original_seasons(path,league)
        return sorted(set(years)|{2026}) if league=="wnba" else years
    def extra_read(path,league,years,table,*,required=True):
        older=[x for x in years if x!=2026]
        entries=original_read(path,league,older,table,required=required)
        if league=="wnba" and 2026 in years and table in ("games","team_game","player_game"):
            entries += pl.read_parquet(newdir/(table+".parquet")).to_dicts()
        return entries
    fg.history_seasons=extra_seasons
    fg.read_history_table=extra_read
    try:
        (_,team_index,player_index,venue_index,league_index)=fg.build_indexes(oldroot,"wnba",cfg)
    finally:
        fg.history_seasons=original_seasons
        fg.read_history_table=original_read
    print("\nREPLAYING PREEXISTING ARCHIVED PREDICTION GAMES AGAINST ISOLATED 2026 HISTORY")
    archived=load_past_archive(base,args.score_from_date,args.through_date)
    hist_games={str(row["game_id"]):row for row in pl.read_parquet(newdir/"games.parquet").to_dicts()}
    recorded=[]; excluded=[]; failures=[]
    residual=model["residual_distribution"]
    sd=float(residual["std"]); mean=float(residual["mean"])
    for p,s in archived:
        gid=str(p["game_id"])
        game=hist_games.get(gid)
        if game is None:
            excluded.append({"game_id":gid,"reason":"NO_VALIDATED_2026_ESPN_GAME","date":p["game_date"]})
            continue
        if (norm(p["home_team"])!=norm(game.get("home_display_name")) or
            norm(p["away_team"])!=norm(game.get("away_display_name")) or
            p["game_date"]!=game["game_date"]):
            excluded.append({"game_id":gid,"reason":"GAME_IDENTITY_MISMATCH","date":p["game_date"]})
            continue
        if integer(s["home_score"])!=integer(game["home_score"]) or integer(s["away_score"])!=integer(game["away_score"]):
            excluded.append({"game_id":gid,"reason":"ARCHIVE_ESPN_SCORE_CONFLICT","date":p["game_date"]})
            continue
        # Omit target time deliberately: strict prior-date only, avoiding same-day
        # scoreboard visibility (conservative PIT gate).
        target={"game_id":gid, "game_date":p["game_date"], "game_time":"",
                "home_team_id":str(game["home_team_id"]),"away_team_id":str(game["away_team_id"]),
                "neutral_site":game.get("neutral_site"),"venue_id":game.get("venue_id"),
                "internal_season":2026,"sdv_season":2026}
        try:
            f=fg.feature_row(target, canonical=True,league="wnba",cfg=cfg,
                feature_version=fg.active_feature_version(cfg),generated_at="isolated_diagnostic",
                team_index=team_index,player_index=player_index,home_venue_index=venue_index,
                league_efficiency_index=league_index,**dict(zip(("team_windows", "opponent_rating_games", "player_game_window", "player_top_n", "venue_history_games", "team_pseudo_games", "opponent_pseudo_games", "player_pseudo_games", "player_weights"),fg.generation_settings(cfg))))
            margin=model_replay(f,model)
            prob=n_cdf((margin+mean)/sd)
            y=int(integer(s["home_score"])>integer(s["away_score"]))
            recorded.append({"date":p["game_date"],"game_id":gid,"home":p["home_team"],"away":p["away_team"],
                             "y":y,"p_dr":float(p["dratings_home_prob"]),"p_sdv_orig":float(p["sdv_home_prob"]),
                             "p_ens_orig":float(p["home_prob"]),"p_sdv_repaired":prob,
                             "p_ens_repaired":0.5*(float(p["dratings_home_prob"])+prob),
                             "margin_sdv_orig":float(p["sdv_expected_margin"]),
                             "margin_sdv_repaired":margin,
                             "home_rest_days":f.get("home_rest_days"),
                             "away_rest_days":f.get("away_rest_days"),
                             "home_games_prior":f.get("home_games_prior"),
                             "away_games_prior":f.get("away_games_prior")})
        except Exception as exc:
            failures.append({"date":p["game_date"],"game_id":gid,"error":str(exc),
                             "trace":traceback.format_exc(limit=3)})
    save_csv(output/"replay_game_by_game.csv",recorded)
    save_csv(output/"replay_excluded.csv",excluded)
    save_csv(output/"replay_errors.csv",failures)
    print(f"Archive predictions eligible: {len(archived)}, evaluated: {len(recorded)}, excluded: {len(excluded)}, errors: {len(failures)}")
    if not recorded:
        raise RuntimeError("No validated archived predictions could be replayed; inspect replay_errors.csv")
    for k,label in (("p_dr","DRatings"),("p_sdv_orig","Original SDV"),
                    ("p_sdv_repaired","SDV with rebuilt 2026 history"),
                    ("p_ens_orig","Original 50/50 ensemble"),
                    ("p_ens_repaired","50/50 ensemble with rebuilt SDV")):
        print(f"{label:36s} {proper_scores(recorded,k)}")
    print("Mean corrected SDV margin:",round(sum(r["margin_sdv_repaired"] for r in recorded)/len(recorded),3))
    print("Mean home/away rest:",round(sum((num(r["home_rest_days"]) or 0) for r in recorded)/len(recorded),3),
          round(sum((num(r["away_rest_days"]) or 0) for r in recorded)/len(recorded),3))
    result={"eligible":len(archived),"evaluated":len(recorded),"excluded":len(excluded),"errors":len(failures),
            "scores":{label:proper_scores(recorded,k) for k,label in (("p_dr","dratings"),
              ("p_sdv_orig","sdv_original"),("p_sdv_repaired","sdv_2026_rebuilt"),
              ("p_ens_orig","ensemble_original"),("p_ens_repaired","ensemble_2026_rebuilt"))},
            "limitations":["Retrospective model comparison, not proven historical PIT provenance",
                           "2026 ESPN boxscore-derived statistics may differ from SDV historical source semantics",
                           "Missing SDV possession events; fallback boxscore possessions used for reconstructed games",
                           "Target-day earlier games excluded conservatively",
                           "If other dates failed ESPN API or ESPN boxscore validity, results are incomplete"]}
    (output/"replay_summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("ISOLATED OUTPUT:",output)
    return result


def self_test():
    assert norm("LA Sparks")=="los angeles sparks"
    assert key("2026_08_04","Golden State Valkyries","Toronto Tempo",92,81)!=key("2026_08_04","Golden State Valkyries","Toronto Tempo",92,82)
    assert pair("33-81")== (33,81)
    assert num("--") is None and integer("92")==92
    mock={"statistics":[{"name":"fieldGoalsMade-fieldGoalsAttempted","displayValue":"32-65"},
                        {"name":"threePointFieldGoalsMade-threePointFieldGoalsAttempted","displayValue":"8-21"},
                        {"name":"freeThrowsMade-freeThrowsAttempted","displayValue":"10-12"},
                        {"name":"offensiveRebounds","displayValue":"10"},
                        {"name":"defensiveRebounds","displayValue":"29"},
                        {"name":"turnovers","displayValue":"11"}]}
    v=team_statistics(mock)
    assert v["field_goals_made"]==32 and v["three_point_field_goals_made"]==8
    assert v["free_throws_attempted"]==12
    assert all(v[k] is not None for k in REQUIRED_TEAM_STATS)
    print("SELF TEST PASSED")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",type=Path,required=False)
    ap.add_argument("--output",type=Path,required=False)
    ap.add_argument("--from-date",default="2026-04-25")
    ap.add_argument("--through-date",default="2026-10-23")
    ap.add_argument("--score-from-date",default="2026-08-28")
    ap.add_argument("--stage",choices=("all","build","replay"),default="all")
    ap.add_argument("--refresh",action="store_true")
    ap.add_argument("--delay-ms",type=int,default=120)
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    if args.self_test:
        self_test(); return
    if args.repo is None or args.output is None:
        ap.error("--repo and --output required")
    repo=args.repo.resolve()
    out=args.output.resolve()
    if out==repo or repo in out.parents:
        raise SystemExit("REFUSING TO WRITE INSIDE REPOSITORY")
    if args.stage=="replay" and not (out/"build_manifest.json").exists():
        raise SystemExit("Replay requires completed isolated build_manifest.json")
    if args.stage=="build" or args.stage=="all":
        if args.stage=="all" and (out/"replay_summary.json").exists():
            raise SystemExit("Refusing to overwrite existing completed replay. Use a new output directory")
        out.mkdir(parents=True,exist_ok=True)
    base=repo/"docs/win/basketball"
    for req in (base/"models/sdv/wnba/margin_model.json",base/"config/sdv_model.yaml",
                base/"05_final_scores/results/wnba"):
        if not req.exists():
            raise SystemExit("Missing required local source: "+str(req))
    if args.stage=="replay":
        manifest=json.loads((out/"build_manifest.json").read_text(encoding="utf-8"))
        if manifest["loaded_dates"]!=manifest["expected_dates"]:
            raise SystemExit("Replay blocked: incomplete date coverage in manifest")
    else:
        manifest=build(args,base,out)
    if args.stage in ("all","replay"):
        if manifest["loaded_dates"]!=manifest["expected_dates"]:
            raise RuntimeError("Replay blocked: incomplete ESPN date coverage")
        if manifest["boxscore_validated_games"] < manifest["completed_wnba_regular_playoff"] * 0.90:
            raise RuntimeError("Replay blocked: fewer than 90% of completed WNBA games have validated box scores")
        replay(args,base,out,manifest)


if __name__=="__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
