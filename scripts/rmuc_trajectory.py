#!/usr/bin/env python3
"""Reconstruct and visualize continuous RMUC 2026 robot trajectories."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rmuc_trajectory.field import default_canvas
from rmuc_trajectory.combat import infer_attacks
from rmuc_trajectory.buffs import build_buff_intervals
from rmuc_trajectory.dart import infer_dart_impacts
from rmuc_trajectory.revival import infer_paid_revivals, infer_respawn_intervals
from rmuc_trajectory.pipeline import (
    event_alignment_summary,
    load_and_clean_tracks,
    load_match_events,
    quality_report,
)
from rmuc_trajectory.render import render_gif, render_interactive_html, render_static
from rmuc_trajectory.scoring import compute_score_report
from rmuc_trajectory.review import build_review
from rmuc_trajectory.exports import write_scores_csv
from rmuc_trajectory.paths import DEFAULT_OUTPUT_ROOT


DEFAULT_DB = ROOT / "dataset" / "rmuc_2026_region_dataset.sqlite"


def odd_positive(value: str) -> int:
    number = int(value)
    if number < 1 or number % 2 == 0:
        raise argparse.ArgumentTypeError("必须是正奇数")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--game-id", type=int, required=True)
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--camp", action="append", choices=("红", "蓝"))
    parser.add_argument("--robot-type", action="append")
    parser.add_argument("--robot-id", action="append", type=int)
    parser.add_argument("--event-type", action="append", help="仅嵌入指定事件类型，可重复")
    parser.add_argument("--max-gap", type=int, default=3, help="最多插值的连续缺帧数")
    parser.add_argument("--smooth-window", type=odd_positive, default=3)
    parser.add_argument("--max-speed", type=float, default=8.0, help="地面轨迹跳变阈值 m/s")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--return-url", help="相对演示目录链接（静态导出，可选）")
    parser.add_argument("--gif", action="store_true", help="额外输出 GIF 动画")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--trail-seconds", type=int, default=20)
    parser.add_argument("--hide-raw", action="store_true", help="静态图不绘制原始轨迹")
    parser.add_argument("--scores", action="store_true", default=True, help="输出评分明细 JSON（默认启用）")
    parser.add_argument("--no-scores", dest="scores", action="store_false", help="仅生成轨迹，不计算评分")
    parser.add_argument("--min-confidence", choices=("high", "medium", "low"), default="low", help="参与评分的最低攻击可信度")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.start is not None and args.end is not None and args.start > args.end:
        parser.error("--start 不能大于 --end")
    if args.max_gap < 0 or args.max_speed <= 0 or args.fps <= 0 or args.trail_seconds <= 0:
        parser.error("max-gap、max-speed、fps 和 trail-seconds 必须为正值")

    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / "trajectories" / f"{args.game_id}-{uuid.uuid4().hex[:8]}"
    if (output_dir / "trajectory.html").exists():
        parser.error("输出目录已有回放，请选择新目录以避免覆盖")
    parameters = {
        "game_id": args.game_id,
        "start": args.start,
        "end": args.end,
        "camps": args.camp,
        "robot_types": args.robot_type,
        "robot_ids": args.robot_id,
        "event_types": args.event_type,
        "max_gap": args.max_gap,
        "smooth_window": args.smooth_window,
        "max_speed_mps": args.max_speed,
    }
    try:
        match, tracks = load_and_clean_tracks(
            args.db,
            args.game_id,
            start=args.start,
            end=args.end,
            camps=args.camp,
            robot_types=args.robot_type,
            robot_ids=args.robot_id,
            max_gap=args.max_gap,
            smooth_window=args.smooth_window,
            max_speed_mps=args.max_speed,
        )
        _, objectives = load_and_clean_tracks(
            args.db,
            args.game_id,
            start=args.start,
            end=args.end,
            camps=args.camp,
            robot_types=("基地", "前哨站"),
            max_gap=args.max_gap,
            smooth_window=1,
            max_speed_mps=args.max_speed,
            include_static=True,
        )
        canvas = default_canvas(ROOT)
        events = load_match_events(
            args.db,
            args.game_id,
            start=args.start,
            end=args.end,
            event_types=args.event_type,
        )
        match["_is_partial"] = bool((args.start or 0) > 0 or
                                    (args.end is not None and args.end < float(match.get("时长秒") or 420)))
        output_dir.mkdir(parents=True, exist_ok=True)
        report = quality_report(match, tracks, parameters)
        report["event_alignment"] = event_alignment_summary(events, tracks)
        attacks, attack_summary = infer_attacks(events, tracks + objectives)
        match_end = max(float(track.times[-1]) for track in tracks if len(track.times))
        buff_intervals, buff_summary = build_buff_intervals(
            events, tracks + objectives, match_end
        )
        paid_revivals, revival_summary = infer_paid_revivals(tracks)
        respawn_intervals, respawn_summary = infer_respawn_intervals(
            tracks, paid_revivals, match_end
        )
        dart_impacts, dart_summary = infer_dart_impacts(events, objectives)
        if args.scores:
            scores, scoring_summary, timeseries_scores = compute_score_report(
                match, tracks + objectives, events, attacks, buff_intervals, min_confidence=args.min_confidence
            )
            report["scoring"] = scores
            report["scoring_summary"] = scoring_summary
            score_path = output_dir / "scores.json"
            score_path.write_text(
                json.dumps({"scores": scores, "summary": scoring_summary},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            write_scores_csv(output_dir / "scores.csv", scores)
            review = build_review(match, tracks + objectives, events, scores, timeseries_scores, scoring_summary)
            (output_dir / "review.json").write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding="utf-8")
            ts_path = output_dir / "timeseries_scores.json"
            ts_path.write_text(
                json.dumps(timeseries_scores, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        report["attack_inference"] = {
            **attack_summary,
            "heading_convention": "0° 指向 +x，正角度逆时针",
            "method": "同口径发弹窗口 + 敌对阵营 + 枪口偏角 + 距离/时间软约束",
            "warning": "这是遥测启发式推断，不等同于裁判系统确认的命中归因",
        }
        report["objective_tracking"] = {
            "track_count": len(objectives),
            "position_source": "数据库建筑坐标为(0,0)，展示位置由规则手册俯视画布近似标定",
            "ballistic_link": "英雄42mm发弹与基地/前哨站受击窗口绘制为抛射形事件关联",
        }
        report["buff_timeline"] = buff_summary
        report["paid_revival_inference"] = revival_summary
        report["respawn_timeline"] = respawn_summary
        report["dart_inference"] = dart_summary
        report["summary"].update(report["event_alignment"])
        report["summary"].update(attack_summary)
        report["summary"]["dart_hit_count"] = len(dart_impacts)
        report["summary"]["inferred_paid_revival_count"] = revival_summary[
            "inferred_paid_revival_count"
        ]
        report["summary"]["objective_track_count"] = len(objectives)
        report["summary"]["respawn_interval_count"] = len(respawn_intervals)
        report["summary"]["deployed_hero_lob_count"] = sum(
            attack.special_mode == "hero_deployed_lob" for attack in attacks
        )
        report_path = output_dir / "quality_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        render_static(
            output_dir / "trajectory.png",
            canvas,
            match,
            tracks,
            show_raw=not args.hide_raw,
        )
        render_interactive_html(
            output_dir / "trajectory.html",
            canvas,
            match,
            tracks,
            events,
            attacks,
            objectives,
            buff_intervals,
            paid_revivals,
            respawn_intervals,
            dart_impacts,
            scores=scores if args.scores else None,
            review=review if args.scores else None,
            return_url=args.return_url,
            timeseries_scores=timeseries_scores if args.scores else None,
        )
        if args.gif:
            render_gif(
                output_dir / "trajectory.gif",
                canvas,
                match,
                tracks,
                fps=args.fps,
                trail_seconds=args.trail_seconds,
            )
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1

    print(json.dumps({"output_dir": str(output_dir.resolve()), **report["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
