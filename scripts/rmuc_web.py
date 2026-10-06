#!/usr/bin/env python3
"""Run the local RMUC match browser and replay generator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rmuc_trajectory.webapp import create_server
from rmuc_trajectory.paths import DEFAULT_OUTPUT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=ROOT / "dataset" / "rmuc_2026_region_dataset.sqlite",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "web_replays"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    server, application = create_server(
        ROOT, args.db, args.output_dir, host=args.host, port=args.port
    )
    url = f"http://{args.host}:{server.server_port}/"
    print(f"RMUC 比赛浏览器已启动: {url}")
    print("按 Ctrl+C 停止服务")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止服务…")
    finally:
        server.shutdown()
        server.server_close()
        application.executor.shutdown(wait=False, cancel_futures=True)
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
