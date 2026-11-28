#!/usr/bin/env python3
"""Write a dedicated, always-fresh WNBA residual-drift status file.

This reporter is informational only. It never changes the pipeline health decision.

Log (overwritten each run):
  docs/win/basketball/errors/99_validation/write_wnba_bias_drift_warning.txt
"""
from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("docs/win/basketball")
HEALTH_FILE = BASE / "pipeline_health.json"
OUTPUT_FILE = BASE / "errors/99_validation/write_wnba_bias_drift_warning.txt"


def render(payload: dict) -> str:
    drift = payload.get("wnba_bias_drift") or {}
    windows = drift.get("windows") or {}
    warnings = drift.get("warnings") or []
    # No drift section, no windows, or zero matched games means the drift
    # calculation produced nothing: report that instead of a false "OK".
    unavailable = not windows or not drift.get("matched_games")
    status = "WARNING" if warnings else ("UNAVAILABLE" if unavailable else "OK")
    lines = [
        f"=== WNBA bias drift {datetime.now(timezone.utc).isoformat()} ===",
        f"pipeline_health_generated_at_utc: {payload.get('generated_at_utc', '')}",
        f"definition: {drift.get('definition', '')}",
        f"warning_threshold_abs_points: {drift.get('warning_threshold_abs_points', '')}",
        f"matched_games: {drift.get('matched_games', 0)}",
    ]
    for n in ("25", "50", "100"):
        values = windows.get(n) or {}
        lines.append(
            f"window_{n}: games={values.get('games', 0)} "
            f"margin_mean_residual={values.get('margin_mean_residual')} "
            f"total_mean_residual={values.get('total_mean_residual')} "
            f"warning={bool(values.get('warning', False))}"
        )
    if warnings:
        for warning in warnings:
            lines.append(f"WARN: {warning}")
    elif not unavailable:
        lines.append("WARN: none")
    if unavailable and not warnings:
        lines.append("ERROR: WNBA drift data unavailable: pipeline_health.json has no drift windows or zero matched games")
    lines.append(f"STATUS: {status}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = json.loads(HEALTH_FILE.read_text(encoding="utf-8"))
        text = render(payload)
    except Exception as exc:
        text = (
            f"=== WNBA bias drift {datetime.now(timezone.utc).isoformat()} ===\n"
            f"ERROR: unable to read pipeline health: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}"
            "STATUS: UNAVAILABLE\n"
        )
    # Always overwrite: a resolved warning cannot linger from an older run.
    OUTPUT_FILE.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
