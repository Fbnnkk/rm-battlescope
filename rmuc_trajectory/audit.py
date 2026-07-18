"""Batch quality audit for every game in the RMUC trajectory dataset."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from .pipeline import DEFAULT_SPEED_MPS, TrackKey, clean_track, open_readonly


@dataclass
class GameAudit:
    game_id: int
    region: str
    schedule: str
    round_number: int
    red_school: str
    blue_school: str
    duration_seconds: int
    track_count: int
    raw_rows: int
    frame_slots: int
    missing_frames: int
    out_of_bounds: int
    spike_outliers: int
    interpolated_frames: int
    residual_jumps: int
    out_of_bounds_rate: float
    missing_rate: float
    max_clean_speed_mps: float
    review_reasons: list[str]

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["review_reasons"] = "; ".join(self.review_reasons)
        return row


def _match_index(connection: sqlite3.Connection) -> dict[int, dict[str, Any]]:
    return {
        int(row["game_id"]): dict(row)
        for row in connection.execute("SELECT * FROM matches ORDER BY game_id")
    }


def _track_groups(connection: sqlite3.Connection) -> Iterator[tuple[int, TrackKey, list[sqlite3.Row]]]:
    rows = connection.execute(
        """
        SELECT game_id, 时刻秒, robot_id, 机器人类型, 阵营, 学校名, x, y
        FROM timeseries INDEXED BY ix_ts_game
        WHERE 机器人类型 NOT IN ('基地', '前哨站')
        ORDER BY game_id, 阵营, robot_id, 时刻秒, rowid
        """
    )
    current_identity: tuple[int, str, int, str, str] | None = None
    current_rows: list[sqlite3.Row] = []
    for row in rows:
        identity = (
            int(row["game_id"]),
            str(row["阵营"]),
            int(row["robot_id"]),
            str(row["机器人类型"]),
            str(row["学校名"]),
        )
        if current_identity is not None and identity != current_identity:
            game_id, camp, robot_id, robot_type, school = current_identity
            yield game_id, TrackKey(camp, robot_id, robot_type, school), current_rows
            current_rows = []
        current_identity = identity
        current_rows.append(row)
    if current_identity is not None:
        game_id, camp, robot_id, robot_type, school = current_identity
        yield game_id, TrackKey(camp, robot_id, robot_type, school), current_rows


def audit_database(
    database: Path,
    max_gap: int = 3,
    smooth_window: int = 3,
    max_speed_mps: float = 8.0,
    review_out_of_bounds_rate: float = 0.01,
    review_missing_rate: float = 0.05,
) -> list[GameAudit]:
    with open_readonly(database) as connection:
        matches = _match_index(connection)
        diagnostics: dict[int, list[Any]] = defaultdict(list)
        for game_id, key, rows in _track_groups(connection):
            speed_limit = max(DEFAULT_SPEED_MPS.get(key.robot_type, max_speed_mps), max_speed_mps)
            diagnostics[game_id].append(
                clean_track(
                    key,
                    rows,
                    start=None,
                    end=None,
                    max_gap=max_gap,
                    smooth_window=smooth_window,
                    max_speed_mps=speed_limit,
                ).diagnostics
            )

    audits: list[GameAudit] = []
    for game_id, match in matches.items():
        tracks = diagnostics.get(game_id, [])
        raw_rows = sum(track.raw_rows for track in tracks)
        frame_slots = sum(track.frame_count for track in tracks)
        missing_frames = sum(track.missing_frames for track in tracks)
        out_of_bounds = sum(track.out_of_bounds for track in tracks)
        residual_jumps = sum(track.residual_jump_count for track in tracks)
        out_rate = out_of_bounds / raw_rows if raw_rows else 1.0
        missing_rate = missing_frames / frame_slots if frame_slots else 1.0
        reasons: list[str] = []
        if out_rate > review_out_of_bounds_rate:
            reasons.append(f"越界率>{review_out_of_bounds_rate:.1%}")
        if missing_rate > review_missing_rate:
            reasons.append(f"缺帧率>{review_missing_rate:.1%}")
        if residual_jumps:
            reasons.append("存在清洗后超速断点")
        if len(tracks) < 10:
            reasons.append("移动实体轨迹少于10条")
        audits.append(
            GameAudit(
                game_id=game_id,
                region=str(match["赛区"]),
                schedule=str(match["赛程"]),
                round_number=int(match["局号"]),
                red_school=str(match["红方学校"]),
                blue_school=str(match["蓝方学校"]),
                duration_seconds=int(match["时长秒"]),
                track_count=len(tracks),
                raw_rows=raw_rows,
                frame_slots=frame_slots,
                missing_frames=missing_frames,
                out_of_bounds=out_of_bounds,
                spike_outliers=sum(track.spike_outliers for track in tracks),
                interpolated_frames=sum(track.interpolated_frames for track in tracks),
                residual_jumps=residual_jumps,
                out_of_bounds_rate=round(out_rate, 6),
                missing_rate=round(missing_rate, 6),
                max_clean_speed_mps=max(
                    (track.max_clean_speed_mps for track in tracks), default=0.0
                ),
                review_reasons=reasons,
            )
        )
    return sorted(audits, key=lambda audit: audit.game_id)


def write_audit_outputs(output_dir: Path, audits: list[GameAudit]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [audit.to_row() for audit in audits]
    csv_path = output_dir / "trajectory_quality_by_game.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    flagged = [audit for audit in audits if audit.review_reasons]
    worst = sorted(
        audits,
        key=lambda audit: (
            audit.out_of_bounds_rate,
            audit.missing_rate,
            audit.residual_jumps,
        ),
        reverse=True,
    )[:20]
    summary = {
        "game_count": len(audits),
        "flagged_game_count": len(flagged),
        "total_rows": sum(audit.raw_rows for audit in audits),
        "total_out_of_bounds": sum(audit.out_of_bounds for audit in audits),
        "total_missing_frames": sum(audit.missing_frames for audit in audits),
        "total_residual_jumps": sum(audit.residual_jumps for audit in audits),
        "thresholds": {
            "out_of_bounds_rate": 0.01,
            "missing_rate": 0.05,
            "minimum_track_count": 10,
        },
        "worst_games": [audit.to_row() for audit in worst],
    }
    (output_dir / "trajectory_quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# RMUC 2026 全库轨迹质量审计",
        "",
        f"- 总局数：{len(audits)}",
        f"- 需要人工复核：{len(flagged)}",
        f"- 移动机器人原始行：{summary['total_rows']}",
        f"- 场外坐标：{summary['total_out_of_bounds']}",
        f"- 清洗后超速断点：{summary['total_residual_jumps']}",
        "",
        "## 越界率最高的 20 局",
        "",
        "| game_id | 赛程 | 局 | 越界率 | 缺帧率 | 断点 | 复核原因 |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for audit in worst:
        lines.append(
            f"| {audit.game_id} | {audit.schedule} | {audit.round_number} | "
            f"{audit.out_of_bounds_rate:.2%} | {audit.missing_rate:.2%} | "
            f"{audit.residual_jumps} | {'；'.join(audit.review_reasons) or '-'} |"
        )
    (output_dir / "trajectory_quality_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary
