#!/usr/bin/env python3
"""Read-only command-line explorer for the RMUC 2026 regional SQLite dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "rmuc_2026_region_dataset" / "rmuc_2026_region_dataset.sqlite"
MAX_ROWS = 100_000


def open_readonly(path: Path) -> sqlite3.Connection:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SQLite 文件不存在: {path}")
    uri = f"{path.as_uri()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA temp_store = MEMORY")
    return connection


def positive_limit(value: str) -> int:
    limit = int(value)
    if not 1 <= limit <= MAX_ROWS:
        raise argparse.ArgumentTypeError(f"limit 必须在 1 到 {MAX_ROWS} 之间")
    return limit


def add_condition(
    clauses: list[str], parameters: list[Any], expression: str, value: Any
) -> None:
    if value is not None:
        clauses.append(expression)
        parameters.append(value)


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def render_table(rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        print("没有匹配记录。")
        return
    columns = list(rows[0])
    text_rows = [["" if row[col] is None else str(row[col]) for col in columns] for row in rows]
    widths = [
        min(40, max(len(str(col)), *(len(values[index]) for values in text_rows)))
        for index, col in enumerate(columns)
    ]

    def format_values(values: Sequence[str]) -> str:
        cells = []
        for value, width in zip(values, widths):
            shown = value if len(value) <= width else value[: max(0, width - 1)] + "…"
            cells.append(shown.ljust(width))
        return " | ".join(cells)

    print(format_values(columns))
    print("-+-".join("-" * width for width in widths))
    for values in text_rows:
        print(format_values(values))


def emit_rows(
    rows: Sequence[dict[str, Any]], output_format: str, output: Path | None
) -> None:
    if output is not None:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "csv":
            with output.open("w", newline="", encoding="utf-8-sig") as handle:
                if rows:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        elif output_format == "json":
            output.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        elif output_format == "jsonl":
            with output.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            raise ValueError("写入文件时 format 必须是 csv、json 或 jsonl")
        print(f"已写出 {len(rows)} 行: {output}")
        return

    if output_format == "table":
        render_table(rows)
    elif output_format == "json":
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    elif output_format == "jsonl":
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
    elif output_format == "csv":
        if rows:
            writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def command_summary(connection: sqlite3.Connection, _args: argparse.Namespace) -> None:
    counts = {
        table: connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
        for table in ("matches", "timeseries", "events")
    }
    match_overview = dict(
        connection.execute(
            """
            SELECT count(DISTINCT 场次号 || ':' || 赛区) AS matches,
                   count(DISTINCT game_id) AS games,
                   min(开始时间) AS first_started_at,
                   max(开始时间) AS last_started_at,
                   min(时长秒) AS min_duration_seconds,
                   max(时长秒) AS max_duration_seconds,
                   round(avg(时长秒), 1) AS avg_duration_seconds
            FROM matches
            """
        ).fetchone()
    )
    school_count = connection.execute(
        """
        SELECT count(*) FROM (
          SELECT 红方学校 AS 学校 FROM matches
          UNION
          SELECT 蓝方学校 AS 学校 FROM matches
        )
        """
    ).fetchone()[0]
    regions = rows_to_dicts(
        connection.execute(
            """
            SELECT 赛区 AS region, count(DISTINCT 场次号) AS matches, count(*) AS games
            FROM matches GROUP BY 赛区 ORDER BY 赛区
            """
        )
    )
    event_types = rows_to_dicts(
        connection.execute(
            """
            SELECT 事件类型 AS event_type, count(*) AS rows
            FROM events GROUP BY 事件类型 ORDER BY rows DESC
            """
        )
    )
    result = {
        "sqlite_version": sqlite3.sqlite_version,
        "quick_check": connection.execute("PRAGMA quick_check").fetchone()[0],
        "row_counts": counts,
        "match_overview": {**match_overview, "schools": school_count},
        "regions": regions,
        "event_types": event_types,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_schema(connection: sqlite3.Connection, _args: argparse.Namespace) -> None:
    result: dict[str, Any] = {"tables": {}, "indexes": []}
    tables = connection.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    for table in tables:
        name = table["name"]
        result["tables"][name] = {
            "sql": table["sql"],
            "columns": rows_to_dicts(connection.execute(f'PRAGMA table_info("{name}")')),
        }
    result["indexes"] = rows_to_dicts(
        connection.execute(
            """
            SELECT name, tbl_name AS table_name, sql
            FROM sqlite_master WHERE type='index' ORDER BY name
            """
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def query_matches(connection: sqlite3.Connection, args: argparse.Namespace) -> list[dict[str, Any]]:
    clauses: list[str] = []
    parameters: list[Any] = []
    add_condition(clauses, parameters, "game_id = ?", args.game_id)
    add_condition(clauses, parameters, "赛区 = ?", args.region)
    if args.school is not None:
        clauses.append("(红方学校 = ? OR 蓝方学校 = ?)")
        parameters.extend([args.school, args.school])
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    sql = f"SELECT * FROM matches{where} ORDER BY 开始时间, 场次号, 局号 LIMIT ?"
    parameters.append(args.limit)
    return rows_to_dicts(connection.execute(sql, parameters))


def query_events(connection: sqlite3.Connection, args: argparse.Namespace) -> list[dict[str, Any]]:
    clauses: list[str] = []
    parameters: list[Any] = []
    add_condition(clauses, parameters, "game_id = ?", args.game_id)
    add_condition(clauses, parameters, "赛区 = ?", args.region)
    add_condition(clauses, parameters, "学校名 = ?", args.school)
    add_condition(clauses, parameters, "事件类型 = ?", args.event_type)
    add_condition(clauses, parameters, "机器人类型 = ?", args.robot_type)
    add_condition(clauses, parameters, "robot_id = ?", args.robot_id)
    if args.start is not None:
        clauses.append("时刻秒 >= ?")
        parameters.append(args.start)
    if args.end is not None:
        clauses.append("时刻秒 <= ?")
        parameters.append(args.end)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    sql = f"SELECT * FROM events{where} ORDER BY game_id, 时刻秒, rowid LIMIT ?"
    parameters.append(args.limit)
    return rows_to_dicts(connection.execute(sql, parameters))


def query_timeseries(
    connection: sqlite3.Connection, args: argparse.Namespace
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    parameters: list[Any] = []
    add_condition(clauses, parameters, "game_id = ?", args.game_id)
    add_condition(clauses, parameters, "赛区 = ?", args.region)
    add_condition(clauses, parameters, "学校名 = ?", args.school)
    add_condition(clauses, parameters, "机器人类型 = ?", args.robot_type)
    add_condition(clauses, parameters, "robot_id = ?", args.robot_id)
    if args.start is not None:
        clauses.append("时刻秒 >= ?")
        parameters.append(args.start)
    if args.end is not None:
        clauses.append("时刻秒 <= ?")
        parameters.append(args.end)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    sql = f"SELECT * FROM timeseries{where} ORDER BY game_id, 时刻秒, robot_id LIMIT ?"
    parameters.append(args.limit)
    return rows_to_dicts(connection.execute(sql, parameters))


def add_common_query_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--game-id", type=int, help="局唯一编号")
    parser.add_argument("--region", help="赛区，如东部赛区")
    parser.add_argument("--school", help="学校全名")
    parser.add_argument("--limit", type=positive_limit, default=20, help="最多返回行数")
    parser.add_argument(
        "--format", choices=("table", "json", "jsonl", "csv"), default="table"
    )
    parser.add_argument("--output", type=Path, help="写出路径；扩展名不会自动推断格式")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite 数据库路径")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary", help="数据库完整性与规模概览")
    summary.set_defaults(handler=command_summary)

    schema = subparsers.add_parser("schema", help="输出表、字段和索引结构")
    schema.set_defaults(handler=command_schema)

    matches = subparsers.add_parser("matches", help="查询比赛/局索引")
    add_common_query_arguments(matches)
    matches.set_defaults(query=query_matches)

    events = subparsers.add_parser("events", help="查询离散事件")
    add_common_query_arguments(events)
    events.add_argument("--event-type", help="如发弹、受击、装配成功")
    events.add_argument("--robot-type", help="如英雄、工程、步兵3")
    events.add_argument("--robot-id", type=int)
    events.add_argument("--start", type=float, help="起始时刻秒（含）")
    events.add_argument("--end", type=float, help="结束时刻秒（含）")
    events.set_defaults(query=query_events)

    timeseries = subparsers.add_parser("timeseries", help="查询逐秒状态")
    add_common_query_arguments(timeseries)
    timeseries.add_argument("--robot-type", help="如英雄、工程、步兵3")
    timeseries.add_argument("--robot-id", type=int)
    timeseries.add_argument("--start", type=float, help="起始时刻秒（含）")
    timeseries.add_argument("--end", type=float, help="结束时刻秒（含）")
    timeseries.set_defaults(query=query_timeseries)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "start", None) is not None and getattr(args, "end", None) is not None:
        if args.start > args.end:
            parser.error("--start 不能大于 --end")
    if getattr(args, "output", None) is not None and args.format == "table":
        parser.error("使用 --output 时请显式指定 --format csv、json 或 jsonl")
    try:
        with open_readonly(args.db) as connection:
            if hasattr(args, "handler"):
                args.handler(connection, args)
            else:
                rows = args.query(connection, args)
                emit_rows(rows, args.format, args.output)
    except (FileNotFoundError, sqlite3.Error, OSError, ValueError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
