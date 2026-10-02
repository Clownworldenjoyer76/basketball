#!/usr/bin/env python3
# docs/win/basketball/scripts/05_final_scores/04_basketball_results_dashboard.py
#
# Rebuilds:
#   frontend/basketball_dashboard.html
#   frontend/nba_dashboard.html
#   frontend/ncaam_dashboard.html
#   frontend/wnba_dashboard.html
#
# Current-season inputs:
#   docs/win/basketball/05_final_scores/{league}_summary_grand_total.csv
#   docs/win/basketball/05_final_scores/{league}_summary_overall.csv
#   docs/win/basketball/05_final_scores/reports/{league}/...
#
# Optional historical-season inputs:
#   docs/win/basketball/05_final_scores/seasons/<season>/
#
# Each historical season directory should mirror the current Stage 05 layout.

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import html
import json
import traceback

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

LEAGUES = ("nba", "ncaam", "wnba")
MARKETS = ("moneyline", "spread", "total")

LEAGUE_DISPLAY = {
    "all": "All",
    "nba": "NBA",
    "ncaam": "NCAAM",
    "wnba": "WNBA",
}

BASE = Path("docs/win/basketball/05_final_scores")
REPORTS = BASE / "reports"
SEASONS_ROOT = BASE / "seasons"

MASTER_OUTPUT = Path("frontend/basketball_dashboard.html")
LEAGUE_OUTPUTS = {
    "nba": Path("frontend/nba_dashboard.html"),
    "ncaam": Path("frontend/ncaam_dashboard.html"),
    "wnba": Path("frontend/wnba_dashboard.html"),
}

ERROR_DIR = Path("docs/win/basketball/errors/05_final_scores")
LOG_FILE = ERROR_DIR / "04_basketball_results_dashboard.txt"

ERROR_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

RUN_STARTED = datetime.now(UTC)
RUN_COUNTS = {
    "warnings": 0,
    "errors": 0,
    "input_files": 0,
    "input_rows": 0,
    "output_files": 0,
    "output_rows": 0,
}
INPUT_FILES_SEEN: set[str] = set()


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def reset_log() -> None:
    LOG_FILE.write_text(
        "=== 04_basketball_results_dashboard ===\n"
        f"START_TIMESTAMP_UTC: {RUN_STARTED.isoformat()}\n",
        encoding="utf-8",
    )


def log(level: str, message: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"{now_utc()} | {level} | {message}\n")


def warn(message: str) -> None:
    RUN_COUNTS["warnings"] += 1
    log("WARNING", message)


def error(message: str) -> None:
    RUN_COUNTS["errors"] += 1
    log("ERROR", message)


def log_input(path: Path, rows: int, exists: bool) -> None:
    key = str(path)
    if key in INPUT_FILES_SEEN:
        return

    INPUT_FILES_SEEN.add(key)
    RUN_COUNTS["input_files"] += 1
    RUN_COUNTS["input_rows"] += rows
    log("INFO", f"INPUT | file={path} | exists={int(exists)} | rows={rows}")


def log_output(path: Path, rows: int, bytes_written: int) -> None:
    RUN_COUNTS["output_files"] += 1
    RUN_COUNTS["output_rows"] += rows
    log("INFO", f"OUTPUT | file={path} | rows={rows} | bytes={bytes_written}")


def finish(status: str) -> None:
    ended = datetime.now(UTC)

    summary = (
        f"INPUT_SUMMARY | files={RUN_COUNTS['input_files']} | rows={RUN_COUNTS['input_rows']}",
        f"OUTPUT_SUMMARY | files={RUN_COUNTS['output_files']} | rows={RUN_COUNTS['output_rows']}",
        f"WARNING_COUNT: {RUN_COUNTS['warnings']}",
        f"ERROR_COUNT: {RUN_COUNTS['errors']}",
        f"END_TIMESTAMP_UTC: {ended.isoformat()}",
        f"STATUS: {status}",
    )

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.writelines(
            f"{line}\n"
            for line in summary
        )


# ============================================================
# DATA HELPERS
# ============================================================


def safe_read(path: Path, *, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        log_input(path, 0, False)
        if required:
            warn(f"Required dashboard input missing: {path}")
        else:
            log("INFO", f"Optional dashboard input missing: {path}")
        return pd.DataFrame()

    try:
        frame = pd.read_csv(path)
        log_input(path, len(frame), True)
        return frame
    except Exception as exc:
        log_input(path, 0, True)
        warn(f"Unable to read {path}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def clean_scalar(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def df_to_records(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []

    return [
        {key: clean_scalar(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def first_row(frame: pd.DataFrame) -> dict:
    rows = df_to_records(frame.head(1))
    return rows[0] if rows else {}


def first_nonblank(row: dict, keys: tuple[str, ...]):
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def normalize_side_rows(frame: pd.DataFrame) -> list[dict]:
    records = df_to_records(frame)

    for row in records:
        if first_nonblank(row, ("bucket",)) is not None:
            continue

        value = first_nonblank(row, ("side_group", "side", "variable"))
        if value is not None:
            row["bucket"] = value

    return records


def add_league(rows: list[dict], league: str) -> list[dict]:
    display = LEAGUE_DISPLAY[league]
    return [{"league": display, **row} for row in rows]


def to_number(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


# ============================================================
# SEASONS
# ============================================================


def season_label(name: str) -> str:
    return name.replace("_", "-").strip()


def discover_seasons() -> list[tuple[str, str, Path]]:
    found: list[tuple[str, str, Path]] = [("current", "Current", BASE)]

    if not SEASONS_ROOT.exists():
        return found

    historical = sorted(
        [path for path in SEASONS_ROOT.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )

    for path in historical:
        found.append((path.name, season_label(path.name), path))

    return found


# ============================================================
# LEAGUE DATA COLLECTION
# ============================================================


def collect_market_data(league: str, root: Path, market: str) -> dict:
    market_dir = root / "reports" / league / market
    market_data = {"by": {}, "by_side": {}}

    dimensions = (
        "ev",
        "kelly",
        "odds",
        "win_prob",
        "edge_vs_market",
        "dow",
        "month",
    )

    for dimension in dimensions:
        overall = market_dir / f"{league}_{market}_by_{dimension}.csv"
        side_suffix = "over_under" if market == "total" else "home_away"
        side = market_dir / f"{league}_{market}_by_{dimension}_{side_suffix}_summary.csv"

        market_data["by"][dimension] = df_to_records(safe_read(overall))
        market_data["by_side"][dimension] = normalize_side_rows(safe_read(side))

    if market in ("spread", "total"):
        side_suffix = "over_under" if market == "total" else "home_away"
        market_data["by"]["side"] = df_to_records(
            safe_read(market_dir / f"{league}_{market}_by_side.csv")
        )
        market_data["by_side"]["side"] = normalize_side_rows(
            safe_read(
                market_dir / f"{league}_{market}_by_side_{side_suffix}_summary.csv"
            )
        )

    if market == "total":
        market_data["by"]["total_range"] = df_to_records(
            safe_read(market_dir / f"{league}_{market}_by_total_range.csv")
        )
        market_data["by_side"]["total_range"] = normalize_side_rows(
            safe_read(
                market_dir / f"{league}_{market}_by_total_range_over_under_summary.csv"
            )
        )

    return market_data


def collect_league_data(league: str, root: Path) -> dict:
    is_current = root == BASE
    report_root = root / "reports" / league

    return {
        "league": league.upper(),
        "display": LEAGUE_DISPLAY[league],
        "grand_total": first_row(
            safe_read(
                root / f"{league}_summary_grand_total.csv",
                required=is_current,
            )
        ),
        "by_market_summary": df_to_records(
            safe_read(
                root / f"{league}_summary_overall.csv",
                required=is_current,
            )
        ),
        "quality": df_to_records(
            safe_read(report_root / "quality" / f"{league}_model_quality_all.csv")
        ),
        "overview": {
            "by_market": df_to_records(
                safe_read(report_root / "overview" / f"{league}_summary_by_market.csv")
            ),
            "by_side_group": normalize_side_rows(
                safe_read(
                    report_root
                    / "overview"
                    / f"{league}_summary_by_side_group.csv"
                )
            ),
            "by_date": df_to_records(
                safe_read(report_root / "overview" / f"{league}_summary_by_date.csv")
            ),
        },
        "markets": {
            market: collect_market_data(league, root, market)
            for market in MARKETS
        },
    }


# ============================================================
# ALL-LEAGUES AGGREGATION
# ============================================================


def collect_all_data(payloads: dict[str, dict]) -> dict:
    grand_rows = [
        payload.get("grand_total") or {}
        for payload in payloads.values()
        if payload.get("grand_total")
    ]

    bets = int(sum(to_number(row.get("bets")) for row in grand_rows))
    wins = int(sum(to_number(row.get("wins")) for row in grand_rows))
    losses = int(sum(to_number(row.get("losses")) for row in grand_rows))
    pushes = int(sum(to_number(row.get("pushes")) for row in grand_rows))
    units_flat = sum(to_number(row.get("units_flat")) for row in grand_rows)
    units_kelly = sum(to_number(row.get("units_kelly")) for row in grand_rows)

    grand_total = {
        "league": "ALL",
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_pct": wins / (wins + losses) if (wins + losses) else None,
        "units_flat": units_flat if grand_rows else None,
        "roi_flat": units_flat / bets if bets else None,
        "units_kelly": units_kelly if grand_rows else None,
        "roi_kelly": None,
    }

    by_market_summary: list[dict] = []
    quality: list[dict] = []
    overview = {
        "by_market": [],
        "by_side_group": [],
        "by_date": [],
    }
    markets = {
        market: {"by": {}, "by_side": {}}
        for market in MARKETS
    }

    for league, payload in payloads.items():
        by_market_summary.extend(
            add_league(payload.get("by_market_summary") or [], league)
        )
        quality.extend(add_league(payload.get("quality") or [], league))

        league_overview = payload.get("overview") or {}
        for key in overview:
            overview[key].extend(
                add_league(league_overview.get(key) or [], league)
            )

        league_markets = payload.get("markets") or {}
        for market in MARKETS:
            market_payload = league_markets.get(market) or {"by": {}, "by_side": {}}

            for view in ("by", "by_side"):
                for dimension, rows in (market_payload.get(view) or {}).items():
                    markets[market][view].setdefault(dimension, [])
                    markets[market][view][dimension].extend(
                        add_league(rows or [], league)
                    )

    return {
        "league": "ALL",
        "display": "All",
        "is_all": True,
        "grand_total": grand_total,
        "by_market_summary": by_market_summary,
        "quality": quality,
        "overview": overview,
        "markets": markets,
    }


# ============================================================
# CSS
# ============================================================

CSS = r"""
:root {
  --bg:#0e1117;
  --panel:#161b22;
  --panel-2:#1c232c;
  --text:#e6edf3;
  --muted:#8b949e;
  --accent:#58a6ff;
  --good:#3fb950;
  --bad:#f85149;
  --border:#30363d;
  --table-border:rgba(139,148,158,.22);
}

* { box-sizing:border-box; }

html, body {
  margin:0;
  padding:0;
  background:var(--bg);
  color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
}

header {
  padding:18px 24px;
  border-bottom:1px solid var(--border);
}

header h1 {
  margin:0;
  font-size:22px;
  font-weight:700;
}

.selector-bar {
  padding:10px 24px;
  border-bottom:1px solid var(--border);
  background:var(--panel);
  display:flex;
  gap:6px;
  flex-wrap:wrap;
  align-items:center;
}

.league-bar {
  position:sticky;
  top:0;
  z-index:10;
}

.selector-bar .lbl {
  color:var(--muted);
  font-size:12px;
  text-transform:uppercase;
  letter-spacing:.06em;
  margin-right:6px;
}

.league-btn,
.season-btn {
  background:var(--panel-2);
  border:1px solid var(--border);
  color:var(--text);
  padding:6px 14px;
  border-radius:4px;
  cursor:pointer;
  font-size:13px;
}

.league-btn.active,
.season-btn.active {
  background:var(--accent);
  color:#0e1117;
  border-color:var(--accent);
  font-weight:600;
}

main {
  padding:18px 24px;
  max-width:1500px;
  margin:0 auto;
}

.league-section { display:none; }
.league-section.active { display:block; }

h2 {
  font-size:16px;
  margin:24px 0 8px;
  color:var(--text);
  border-bottom:1px solid var(--border);
  padding-bottom:4px;
}

.kpis {
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
  gap:10px;
  margin:12px 0 4px;
}

.kpi {
  background:var(--panel);
  border:1px solid var(--border);
  border-radius:6px;
  padding:12px;
}

.kpi .label {
  color:var(--muted);
  font-size:11px;
  text-transform:uppercase;
  letter-spacing:.05em;
}

.kpi .value {
  font-size:22px;
  margin-top:4px;
  font-weight:600;
}

.kpi .value.good { color:var(--good); }
.kpi .value.bad { color:var(--bad); }

.tabs {
  display:flex;
  gap:4px;
  margin:16px 0 0;
  flex-wrap:wrap;
}

.tab {
  background:var(--panel);
  border:1px solid var(--border);
  padding:6px 12px;
  border-radius:6px 6px 0 0;
  cursor:pointer;
  color:var(--muted);
  font-size:13px;
}

.tab.active {
  background:var(--panel-2);
  color:var(--text);
  border-bottom-color:var(--panel-2);
}

.tab-body {
  background:var(--panel-2);
  border:1px solid var(--border);
  border-top:none;
  padding:14px;
  border-radius:0 6px 6px 6px;
}

.controls {
  display:flex;
  gap:8px;
  flex-wrap:wrap;
  align-items:center;
  margin-bottom:10px;
}

.controls label {
  color:var(--muted);
  font-size:12px;
}

.controls select {
  background:var(--panel);
  color:var(--text);
  border:1px solid var(--border);
  padding:4px 8px;
  border-radius:4px;
  font-size:13px;
}

.scroll {
  max-height:60vh;
  overflow:auto;
  border:1px solid var(--table-border);
  border-radius:6px;
}

table {
  width:100%;
  border-collapse:collapse;
  font-size:13px;
}

th,
td {
  padding:7px 9px;
  text-align:center;
  border:1px solid var(--table-border);
  white-space:nowrap;
  vertical-align:middle;
}

th {
  color:var(--muted);
  text-transform:uppercase;
  font-size:11px;
  letter-spacing:.05em;
  cursor:pointer;
  user-select:none;
  position:sticky;
  top:0;
  background:var(--panel-2);
}

th .arrow { opacity:.4; margin-left:4px; }
th.sorted .arrow { opacity:1; color:var(--accent); }
td.num { font-variant-numeric:tabular-nums; }

.pos { color:var(--good); }
.neg { color:var(--bad); }
.muted { color:var(--muted); }

td.win-pct-strong {
  background:rgba(63,185,80,.24);
  color:#7ee787;
  font-weight:700;
}

td.win-pct-green {
  background:rgba(63,185,80,.16);
  color:#56d364;
  font-weight:600;
}

td.win-pct-light {
  background:rgba(63,185,80,.09);
  color:#8ddb8c;
  font-weight:600;
}

td.win-pct-neutral {
  background:rgba(139,148,158,.09);
  color:#c9d1d9;
}

td.win-pct-red {
  background:rgba(248,81,73,.14);
  color:#ff7b72;
  font-weight:600;
}
"""


# ============================================================
# JAVASCRIPT
# ============================================================

JS = r"""
let ACTIVE_SEASON = null;
let ACTIVE_LEAGUE = null;

const LEAGUE_LABELS = {
  all: 'All',
  nba: 'NBA',
  ncaam: 'NCAAM',
  wnba: 'WNBA'
};

function fmtPct(value) {
  if (value == null || isNaN(value)) return '';
  return (Number(value) * 100).toFixed(2) + '%';
}

function fmtNum(value, decimals) {
  if (value == null || isNaN(value)) return '';
  return Number(value).toFixed(decimals);
}

function fmtInt(value) {
  if (value == null || isNaN(value)) return '';
  return Number(value).toLocaleString();
}

function signedClass(value) {
  if (value == null || isNaN(value)) return '';
  return Number(value) > 0 ? 'pos' : (Number(value) < 0 ? 'neg' : '');
}

function winPctClass(value) {
  if (value == null || isNaN(value)) return '';

  const pct = Number(value);
  if (pct >= 0.80) return 'win-pct-strong';
  if (pct >= 0.70) return 'win-pct-green';
  if (pct >= 0.60) return 'win-pct-light';
  if (pct >= 0.50) return 'win-pct-neutral';
  return 'win-pct-red';
}

function isWinPctKey(key) {
  return String(key || '').toLowerCase() === 'win_pct';
}

function showTab(host, key) {
  host.querySelectorAll(':scope > .tabs .tab').forEach(tab => {
    tab.classList.toggle('active', tab.dataset.key === key);
  });

  host.querySelectorAll(':scope > .tab-body > .tab-panel').forEach(panel => {
    panel.style.display = panel.dataset.key === key ? '' : 'none';
  });
}

function renderTable(data, columns, container) {
  if (!data || !data.length) {
    container.innerHTML = '<div class="muted">No rows.</div>';
    return;
  }

  const wrap = document.createElement('div');
  wrap.className = 'scroll';

  const table = document.createElement('table');
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  const tbody = document.createElement('tbody');

  let sortKey = null;
  let sortDirection = 'desc';

  columns.forEach(column => {
    const th = document.createElement('th');
    th.innerHTML = column.label + ' <span class="arrow">&#9662;</span>';

    th.onclick = () => {
      if (sortKey === column.key) {
        sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
      } else {
        sortKey = column.key;
        sortDirection = 'desc';
      }

      drawRows();
      thead.querySelectorAll('th').forEach(item => item.classList.remove('sorted'));
      th.classList.add('sorted');
    };

    headerRow.appendChild(th);
  });

  thead.appendChild(headerRow);

  function drawRows() {
    const rows = data.slice();

    if (sortKey) {
      rows.sort((a, b) => {
        const av = a[sortKey];
        const bv = b[sortKey];

        if (av == null) return 1;
        if (bv == null) return -1;

        const an = Number(av);
        const bn = Number(bv);
        const bothNumeric = !isNaN(an) && !isNaN(bn);

        if (bothNumeric) {
          return sortDirection === 'asc' ? an - bn : bn - an;
        }

        return sortDirection === 'asc'
          ? String(av).localeCompare(String(bv))
          : String(bv).localeCompare(String(av));
      });
    }

    tbody.innerHTML = '';

    rows.forEach(row => {
      const tr = document.createElement('tr');

      columns.forEach(column => {
        const td = document.createElement('td');
        const value = row[column.key];
        let colorClass = '';

        if (column.fmt === 'int') {
          td.classList.add('num');
          td.textContent = fmtInt(value);
        } else if (column.fmt === 'pct') {
          td.classList.add('num');
          td.textContent = fmtPct(value);

          if (isWinPctKey(column.key)) {
            const winClass = winPctClass(value);
            if (winClass) td.classList.add(winClass);
          }

          if (column.color) colorClass = signedClass(value);
        } else if (column.fmt === 'num') {
          td.classList.add('num');
          td.textContent = fmtNum(
            value,
            column.decimals == null ? 2 : column.decimals
          );
          if (column.color) colorClass = signedClass(value);
        } else if (column.fmt === 'roi') {
          td.classList.add('num');
          td.textContent = fmtPct(value);
          colorClass = signedClass(value);
        } else {
          td.textContent = value == null ? '' : String(value);
        }

        if (colorClass) td.classList.add(colorClass);
        tr.appendChild(td);
      });

      tbody.appendChild(tr);
    });
  }

  drawRows();
  table.appendChild(thead);
  table.appendChild(tbody);
  wrap.appendChild(table);
  container.innerHTML = '';
  container.appendChild(wrap);
}

const STANDARD_COLUMNS = [
  { key:'bucket', label:'Bucket' },
  { key:'bets', label:'Bets', fmt:'int' },
  { key:'wins', label:'W', fmt:'int' },
  { key:'losses', label:'L', fmt:'int' },
  { key:'pushes', label:'P', fmt:'int' },
  { key:'win_pct', label:'Win %', fmt:'pct' },
  { key:'units_flat', label:'Units (flat)', fmt:'num', decimals:2, color:true },
  { key:'roi_flat', label:'ROI flat', fmt:'roi' },
  { key:'units_kelly', label:'Units (Kelly)', fmt:'num', decimals:4, color:true },
  { key:'roi_kelly', label:'ROI Kelly', fmt:'roi' },
  { key:'avg_ev', label:'Avg EV', fmt:'pct' },
  { key:'avg_edge_vs_market_pp', label:'Avg edge (pp)', fmt:'num', decimals:2 },
  { key:'avg_kelly_pct', label:'Avg Kelly', fmt:'pct' },
  { key:'avg_model_prob', label:'Avg model p', fmt:'pct' },
  { key:'avg_odds_american', label:'Avg odds', fmt:'num', decimals:0 }
];

const SIDE_COLUMNS = [
  { key:'side_group', label:'Side' },
  { key:'bucket', label:'Bucket' },
  ...STANDARD_COLUMNS.filter(column => column.key !== 'bucket')
];

const QUALITY_COLUMNS = [
  { key:'scope', label:'Scope' },
  { key:'market_type', label:'Market' },
  { key:'model_source', label:'Model source' },
  { key:'model_version', label:'Model version' },
  { key:'rows', label:'Rows', fmt:'int' },
  { key:'probability_n', label:'Prob N', fmt:'int' },
  { key:'brier_score', label:'Brier', fmt:'num', decimals:4 },
  { key:'log_loss', label:'Log loss', fmt:'num', decimals:4 },
  { key:'calibration_error', label:'Calibration err', fmt:'num', decimals:4 },
  { key:'margin_n', label:'Margin N', fmt:'int' },
  { key:'margin_mae', label:'Margin MAE', fmt:'num', decimals:3 },
  { key:'margin_rmse', label:'Margin RMSE', fmt:'num', decimals:3 },
  { key:'total_n', label:'Total N', fmt:'int' },
  { key:'total_mae', label:'Total MAE', fmt:'num', decimals:3 },
  { key:'total_rmse', label:'Total RMSE', fmt:'num', decimals:3 },
  { key:'clv_n', label:'CLV N', fmt:'int' },
  { key:'avg_clv', label:'Avg CLV', fmt:'num', decimals:3, color:true },
  { key:'clv_units', label:'CLV units' },
  { key:'prob_disagreement_n', label:'Prob dis N', fmt:'int' },
  { key:'avg_model_vs_market_prob_pp', label:'Model-market p (pp)', fmt:'num', decimals:3, color:true },
  { key:'mean_abs_model_vs_market_prob_pp', label:'Abs p gap (pp)', fmt:'num', decimals:3 },
  { key:'line_disagreement_n', label:'Line dis N', fmt:'int' },
  { key:'avg_model_vs_market_line', label:'Model-market line', fmt:'num', decimals:3, color:true },
  { key:'mean_abs_model_vs_market_line', label:'Abs line gap', fmt:'num', decimals:3 }
];

function uniqueFilterValues(rows, key) {
  const seen = new Set();
  const values = [];

  rows.forEach(row => {
    const raw = row[key];
    if (raw == null) return;

    const text = String(raw).trim();
    if (!text) return;
    if (text.toUpperCase() === 'ALL') return;

    const normalized = text.toUpperCase();
    if (seen.has(normalized)) return;

    seen.add(normalized);
    values.push(text);
  });

  return values.sort((a, b) => a.localeCompare(b));
}

function optionHtml(values) {
  return '<option value="__ALL__">All</option>' +
    values.map(value =>
      '<option value="' +
      String(value).replace(/&/g, '&amp;').replace(/"/g, '&quot;') +
      '">' + value + '</option>'
    ).join('');
}

function buildQualityArea(section, data) {
  const area = section.querySelector('.quality-area');
  const rows = data.quality || [];

  if (!rows.length) {
    area.innerHTML = '<div class="muted">No model-quality rows available.</div>';
    return;
  }

  const scopes = uniqueFilterValues(rows, 'scope');
  const markets = uniqueFilterValues(rows, 'market_type');
  const sources = uniqueFilterValues(rows, 'model_source');
  const versions = uniqueFilterValues(rows, 'model_version');

  area.innerHTML =
    '<div class="controls">' +
      '<label>Scope: <select class="quality-scope">' + optionHtml(scopes) + '</select></label>' +
      '<label>Market: <select class="quality-market">' + optionHtml(markets) + '</select></label>' +
      '<label>Source: <select class="quality-source">' + optionHtml(sources) + '</select></label>' +
      '<label>Version: <select class="quality-version">' + optionHtml(versions) + '</select></label>' +
    '</div>' +
    '<div class="quality-table"></div>';

  const scopeSelect = area.querySelector('.quality-scope');
  const marketSelect = area.querySelector('.quality-market');
  const sourceSelect = area.querySelector('.quality-source');
  const versionSelect = area.querySelector('.quality-version');
  const table = area.querySelector('.quality-table');

  function refresh() {
    const filtered = rows.filter(row =>
      (scopeSelect.value === '__ALL__' || String(row.scope) === scopeSelect.value) &&
      (marketSelect.value === '__ALL__' || String(row.market_type) === marketSelect.value) &&
      (sourceSelect.value === '__ALL__' || String(row.model_source) === sourceSelect.value) &&
      (versionSelect.value === '__ALL__' || String(row.model_version) === versionSelect.value)
    );

    const columns = data.is_all
      ? [{ key:'league', label:'League' }, ...QUALITY_COLUMNS]
      : QUALITY_COLUMNS;

    renderTable(filtered, columns, table);
  }

  [scopeSelect, marketSelect, sourceSelect, versionSelect].forEach(select => {
    select.onchange = refresh;
  });

  refresh();
}

function selectLeague(league) {
  ACTIVE_LEAGUE = league;

  document.querySelectorAll('.league-btn').forEach(button => {
    button.classList.toggle('active', button.dataset.league === league);
  });

  document.querySelectorAll('.league-section').forEach(section => {
    section.classList.toggle('active', section.dataset.league === league);
  });

  try {
    localStorage.setItem('basketball_dash_league', league);
  } catch (error) {}
}

function selectSeason(season) {
  if (!ALL_DATA[season]) return;

  ACTIVE_SEASON = season;

  document.querySelectorAll('.season-btn').forEach(button => {
    button.classList.toggle('active', button.dataset.season === season);
  });

  const seasonLeagues = ALL_DATA[season].leagues || {};
  Object.keys(seasonLeagues).forEach(league => {
    buildLeagueSection(league, seasonLeagues[league]);
  });

  if (ACTIVE_LEAGUE && seasonLeagues[ACTIVE_LEAGUE]) {
    selectLeague(ACTIVE_LEAGUE);
  }

  try {
    localStorage.setItem('basketball_dash_season', season);
  } catch (error) {}
}

function buildLeagueSection(league, data) {
  const section = document.querySelector(
    '.league-section[data-league="' + league + '"]'
  );

  if (!section) return;

  const grand = data.grand_total || {};

  function kpi(label, value, format) {
    let display = 'N/A';
    let cssClass = '';

    if (value !== null && value !== undefined && value !== '') {
      if (format === 'pct') {
        display = fmtPct(value);
        cssClass = signedClass(value) === 'pos' ? 'good' : (
          signedClass(value) === 'neg' ? 'bad' : ''
        );
      } else if (format === 'int') {
        display = fmtInt(value);
      } else if (format === 'signed') {
        display = (Number(value) >= 0 ? '+' : '') + fmtNum(value, 2);
        cssClass = Number(value) > 0 ? 'good' : (Number(value) < 0 ? 'bad' : '');
      } else {
        display = String(value);
      }
    }

    return '<div class="kpi"><div class="label">' + label +
      '</div><div class="value ' + cssClass + '">' + display + '</div></div>';
  }

  section.querySelector('.kpis').innerHTML = [
    kpi('Bets', grand.bets, 'int'),
    kpi('Wins', grand.wins, 'int'),
    kpi('Losses', grand.losses, 'int'),
    kpi('Pushes', grand.pushes, 'int'),
    kpi('Win %', grand.win_pct, 'pct'),
    kpi('Units (flat)', grand.units_flat, 'signed'),
    kpi('ROI flat', grand.roi_flat, 'pct'),
    kpi('Units (Kelly)', grand.units_kelly, 'signed'),
    kpi('ROI Kelly', grand.roi_kelly, 'pct')
  ].join('');

  let marketSummaryColumns = [
    { key:'market_type', label:'Market' },
    { key:'Win', label:'W', fmt:'int' },
    { key:'Loss', label:'L', fmt:'int' },
    { key:'Push', label:'P', fmt:'int' },
    { key:'Total', label:'Total', fmt:'int' },
    { key:'Win_Pct', label:'Win %', fmt:'pct' }
  ];

  if (data.is_all) {
    marketSummaryColumns = [
      { key:'league', label:'League' },
      ...marketSummaryColumns
    ];
  }

  renderTable(
    data.by_market_summary || [],
    marketSummaryColumns,
    section.querySelector('.by-market-summary')
  );

  buildQualityArea(section, data);

  MARKETS.forEach(market => {
    const panel = section.querySelector('.panel-' + market);
    const marketData = (data.markets || {})[market] || { by:{}, by_side:{} };

    const dimensions = Array.from(new Set([
      ...Object.keys(marketData.by || {}),
      ...Object.keys(marketData.by_side || {})
    ])).filter(dimension =>
      ((marketData.by || {})[dimension] || []).length ||
      ((marketData.by_side || {})[dimension] || []).length
    );

    if (!dimensions.length) {
      panel.innerHTML = '<div class="muted">No report rows available for this market.</div>';
      return;
    }

    const sideViewLabel = market === 'total'
      ? 'Split by side (Over / Under)'
      : 'Split by side (Home / Away)';

    panel.innerHTML =
      '<div class="controls">' +
        '<label>Dimension: <select class="dim-select">' +
          dimensions.map(dimension =>
            '<option value="' + dimension + '">' +
            dimension.replaceAll('_', ' ') + '</option>'
          ).join('') +
        '</select></label>' +
        '<label>View: <select class="view-select">' +
          '<option value="overall">Overall</option>' +
          '<option value="side">' + sideViewLabel + '</option>' +
        '</select></label>' +
      '</div>' +
      '<div class="market-table"></div>';

    const dimensionSelect = panel.querySelector('.dim-select');
    const viewSelect = panel.querySelector('.view-select');
    const target = panel.querySelector('.market-table');

    function refreshMarket() {
      const dimension = dimensionSelect.value;
      const sideView = viewSelect.value === 'side';
      const rows = sideView
        ? ((marketData.by_side || {})[dimension] || [])
        : ((marketData.by || {})[dimension] || []);

      let columns = sideView ? SIDE_COLUMNS : STANDARD_COLUMNS;

      if (data.is_all) {
        columns = [{ key:'league', label:'League' }, ...columns];
      }

      renderTable(rows, columns, target);
    }

    dimensionSelect.onchange = refreshMarket;
    viewSelect.onchange = refreshMarket;
    refreshMarket();
  });

  const marketArea = section.querySelector('.market-area');
  marketArea.querySelectorAll(':scope > .tabs .tab').forEach(tab => {
    tab.onclick = () => showTab(marketArea, tab.dataset.key);
  });

  let overviewMarketColumns = [
    { key:'market_type', label:'Market' },
    ...STANDARD_COLUMNS.filter(column => column.key !== 'bucket')
  ];

  let overviewSideColumns = [
    { key:'bucket', label:'Side' },
    ...STANDARD_COLUMNS.filter(column => column.key !== 'bucket')
  ];

  let overviewDateColumns = [
    { key:'bucket', label:'Date' },
    ...STANDARD_COLUMNS.filter(column => column.key !== 'bucket')
  ];

  if (data.is_all) {
    overviewMarketColumns = [{ key:'league', label:'League' }, ...overviewMarketColumns];
    overviewSideColumns = [{ key:'league', label:'League' }, ...overviewSideColumns];
    overviewDateColumns = [{ key:'league', label:'League' }, ...overviewDateColumns];
  }

  const overview = data.overview || {};

  renderTable(
    overview.by_market || [],
    overviewMarketColumns,
    section.querySelector('.overview-market')
  );

  renderTable(
    overview.by_side_group || [],
    overviewSideColumns,
    section.querySelector('.overview-side')
  );

  renderTable(
    overview.by_date || [],
    overviewDateColumns,
    section.querySelector('.overview-date')
  );

  const overviewArea = section.querySelector('.overview-area');
  overviewArea.querySelectorAll(':scope > .tabs .tab').forEach(tab => {
    tab.onclick = () => showTab(overviewArea, tab.dataset.key);
  });
}
"""


# ============================================================
# HTML BUILDERS
# ============================================================


def league_section_html(league: str, display: str) -> str:
    return f"""
<section class="league-section" data-league="{html.escape(league)}">
  <h2>{html.escape(display)} Analytics</h2>
  <div class="kpis"></div>

  <h2>By Market</h2>
  <div class="by-market-summary"></div>

  <h2>Model Quality</h2>
  <div class="quality-area"></div>

  <h2>Per Market Drilldown</h2>
  <div class="market-area">
    <div class="tabs">
      <div class="tab active" data-key="moneyline">Moneyline</div>
      <div class="tab" data-key="spread">Spread</div>
      <div class="tab" data-key="total">Total</div>
    </div>
    <div class="tab-body">
      <div class="tab-panel panel-moneyline" data-key="moneyline"></div>
      <div class="tab-panel panel-spread" data-key="spread" style="display:none"></div>
      <div class="tab-panel panel-total" data-key="total" style="display:none"></div>
    </div>
  </div>

  <h2>Overview</h2>
  <div class="overview-area">
    <div class="tabs">
      <div class="tab active" data-key="market">By market</div>
      <div class="tab" data-key="side">By side group</div>
      <div class="tab" data-key="date">By date</div>
    </div>
    <div class="tab-body">
      <div class="tab-panel overview-market" data-key="market"></div>
      <div class="tab-panel overview-side" data-key="side" style="display:none"></div>
      <div class="tab-panel overview-date" data-key="date" style="display:none"></div>
    </div>
  </div>
</section>
"""


def build_page(
    leagues: tuple[str, ...],
    *,
    include_all: bool,
    page_title: str,
    hide_league_bar: bool,
) -> str:
    seasons = discover_seasons()
    payload: dict[str, dict] = {}

    for season_key, label, root in seasons:
        league_payloads = {
            league: collect_league_data(league, root)
            for league in leagues
        }

        page_leagues = dict(league_payloads)
        if include_all:
            page_leagues = {
                "all": collect_all_data(league_payloads),
                **league_payloads,
            }

        payload[season_key] = {
            "label": label,
            "leagues": page_leagues,
        }

    nav_leagues: list[tuple[str, str]] = []
    if include_all:
        nav_leagues.append(("all", "All"))
    nav_leagues.extend((league, LEAGUE_DISPLAY[league]) for league in leagues)

    league_buttons = "\n".join(
        f'<button class="league-btn{" active" if index == 0 else ""}" '
        f'data-league="{html.escape(league)}" '
        f'onclick="selectLeague(\'{html.escape(league)}\')">'
        f'{html.escape(display)}</button>'
        for index, (league, display) in enumerate(nav_leagues)
    )

    season_buttons = "\n".join(
        f'<button class="season-btn{" active" if index == 0 else ""}" '
        f'data-season="{html.escape(season_key)}" '
        f'onclick="selectSeason(\'{html.escape(season_key)}\')">'
        f'{html.escape(label)}</button>'
        for index, (season_key, label, _) in enumerate(seasons)
    )

    sections = "\n".join(
        league_section_html(league, display)
        for league, display in nav_leagues
    )

    first_season = seasons[0][0]
    first_league = nav_leagues[0][0]

    season_bar_style = " style=\"display:none\"" if len(seasons) <= 1 else ""
    league_bar_style = " style=\"display:none\"" if hide_league_bar else ""

    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(page_title)}</title>
<link rel="stylesheet" href="assets/css/matstheme.css">
<style>{CSS}</style>
</head>
<body>
<div id="nav-placeholder"></div>

<header>
  <h1 id="page-title">{html.escape(page_title)}</h1>
</header>

<div class="selector-bar season-bar"{season_bar_style}>
  <span class="lbl">Season:</span>
  {season_buttons}
</div>

<div class="selector-bar league-bar"{league_bar_style}>
  <span class="lbl">League:</span>
  {league_buttons}
</div>

<main>
{sections}
</main>

<script src="assets/js/shared/nav.js"></script>
<script>
const MARKETS = ['moneyline','spread','total'];
{JS}
const ALL_DATA = {payload_json};

document.addEventListener('DOMContentLoaded', () => {{
  let initialSeason = '{html.escape(first_season)}';

  try {{
    const storedSeason = localStorage.getItem('basketball_dash_season');
    if (storedSeason && ALL_DATA[storedSeason]) initialSeason = storedSeason;
  }} catch (error) {{}}

  let initialLeague = '{html.escape(first_league)}';

  try {{
    const storedLeague = localStorage.getItem('basketball_dash_league');
    const seasonLeagues = (ALL_DATA[initialSeason] || {{}}).leagues || {{}};
    if (storedLeague && seasonLeagues[storedLeague]) initialLeague = storedLeague;
  }} catch (error) {{}}

  ACTIVE_LEAGUE = initialLeague;
  selectSeason(initialSeason);
  selectLeague(initialLeague);
}});
</script>
</body>
</html>"""


# ============================================================
# OUTPUT
# ============================================================


def write_page(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    log_output(path, content.count("\n") + 1, len(content.encode("utf-8")))
    log("INFO", f"dashboard -> {path}")


def run() -> None:
    master = build_page(
        LEAGUES,
        include_all=True,
        page_title="Basketball Analytics",
        hide_league_bar=False,
    )
    write_page(MASTER_OUTPUT, master)

    for league in LEAGUES:
        display = LEAGUE_DISPLAY[league]
        page = build_page(
            (league,),
            include_all=False,
            page_title=f"{display} Analytics",
            hide_league_bar=True,
        )
        write_page(LEAGUE_OUTPUTS[league], page)


def main() -> None:
    reset_log()
    status = "FAILED"

    try:
        run()
        status = "SUCCESS"
    except Exception as exc:
        error(f"Unhandled exception: {type(exc).__name__}: {exc}")
        trace = traceback.format_exc()
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(trace)
            if not trace.endswith("\n"):
                handle.write("\n")
        raise
    finally:
        finish(status)


if __name__ == "__main__":
    main()
