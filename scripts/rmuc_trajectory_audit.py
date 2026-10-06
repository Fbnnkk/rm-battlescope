#!/usr/bin/env python3
"""Audit trajectory quality for every game in the RMUC 2026 SQLite dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rmuc_trajectory.audit import audit_database, write_audit_outputs
from rmuc_trajectory.paths import DEFAULT_OUTPUT_ROOT


DEFAULT_DB = ROOT / "dataset" / "rmuc_2026_region_dataset.sqlite"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "trajectory_audit"
    )
    parser.add_argument("--max-gap", type=int, default=3)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--max-speed", type=float, default=8.0)
    args = parser.parse_args()
    if args.max_gap < 0 or args.smooth_window < 1 or args.smooth_window % 2 == 0:
        parser.error("max-gap 必须非负，smooth-window 必须为正奇数")
    try:
        audits = audit_database(
            args.db,
            max_gap=args.max_gap,
            smooth_window=args.smooth_window,
            max_speed_mps=args.max_speed,
        )
        summary = write_audit_outputs(args.output_dir, audits)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
