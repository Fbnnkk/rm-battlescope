"""Build rule-driven buff intervals from start-only RMUC event records."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .pipeline import CleanTrack, MatchEvent


BUFF_DURATIONS = {
    "飞坡": 30.0,
    "过中央高地": 30.0,
    # The regional dataset calls this traversal "台阶跨越" while V2.1.0
    # describes the comparable highland traversal as a 30-second buff.
    "台阶跨越": 30.0,
    "小能量机关增益": 45.0,
}
TERRAIN_BUFFS = {"飞坡", "过中央高地", "台阶跨越"}
BIG_RUNE_DURATIONS = {5: 30.0, 6: 35.0, 7: 40.0, 8: 45.0, 9: 50.0, 10: 60.0}


@dataclass(frozen=True)
class BuffInterval:
    camp: str
    robot_id: int
    robot_type: str
    category: str
    start: float
    end: float
    duration: float
    end_reason: str
    rule_basis: str

    def to_jsonable(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("start", "end", "duration"):
            result[key] = round(float(result[key]), 3)
        return result


def _arm_count(events: list[MatchEvent], buff: MatchEvent) -> int | None:
    values: list[int] = []
    for event in events:
        if (
            event.event_type != "能量机关"
            or event.camp != buff.camp
            or event.time > buff.time
            or buff.time - event.time > 2.0
            or not event.note
        ):
            continue
        match = re.search(r"arm_cnt=([0-9.]+)", event.note)
        if match:
            values.append(int(float(match.group(1))))
    return max(values) if values else None


def _death_time(track: CleanTrack | None, start: float, planned_end: float) -> float | None:
    if track is None:
        return None
    selected = (track.times > start) & (track.times <= planned_end)
    indices = np.flatnonzero(selected & np.isfinite(track.health) & (track.health <= 0.0))
    return float(track.times[indices[0]]) if indices.size else None


def build_buff_intervals(
    events: list[MatchEvent], tracks: list[CleanTrack], match_end: float
) -> tuple[list[BuffInterval], dict[str, Any]]:
    """Convert buff start events into intervals using the V2.1.0 timetable.

    The SQLite schema contains no universal buff-end flag. Terrain buffs are
    additionally truncated at the first zero-health frame, as required by the
    rule manual.
    """

    lookup = {(track.key.camp, track.key.robot_id): track for track in tracks}
    unique: dict[tuple[str, int, str, float], MatchEvent] = {}
    for event in events:
        if event.event_type == "增益" and event.camp and event.robot_id is not None and event.category:
            unique.setdefault((event.camp, event.robot_id, event.category, event.time), event)

    intervals: list[BuffInterval] = []
    unknown_categories: set[str] = set()
    for event in unique.values():
        category = str(event.category)
        if category == "大能量机关增益":
            arms = _arm_count(events, event)
            normalized_arms = min(10, max(5, arms or 5))
            duration = BIG_RUNE_DURATIONS[normalized_arms]
            basis = f"大能量机关{arms if arms is not None else '未知'}灯臂，按V2.1.0表5-18"
        elif category in BUFF_DURATIONS:
            duration = BUFF_DURATIONS[category]
            basis = "V2.1.0第102页/第110页；台阶跨越按高地类映射"
        else:
            unknown_categories.add(category)
            duration = max(0.0, match_end - event.time)
            basis = "数据无结束标志且规则映射未知，保留至本局结束"

        planned_end = min(match_end, event.time + duration)
        end_reason = "规则计时结束" if planned_end < match_end else "本局结束"
        if category in TERRAIN_BUFFS:
            death = _death_time(lookup.get((str(event.camp), int(event.robot_id))), event.time, planned_end)
            if death is not None:
                planned_end = death
                end_reason = "机器人战亡清除"
        intervals.append(
            BuffInterval(
                camp=str(event.camp),
                robot_id=int(event.robot_id),
                robot_type=event.robot_type or "未知",
                category=category,
                start=float(event.time),
                end=float(planned_end),
                duration=max(0.0, float(planned_end - event.time)),
                end_reason=end_reason,
                rule_basis=basis,
            )
        )

    intervals.sort(key=lambda item: (item.start, item.camp, item.robot_id, item.category))
    summary = {
        "interval_count": len(intervals),
        "terrain_death_truncations": sum(item.end_reason == "机器人战亡清除" for item in intervals),
        "unknown_categories": sorted(unknown_categories),
        "has_dataset_end_flag": False,
        "rule_version": "RoboMaster 2026 比赛规则手册 V2.1.0",
        "regional_version_warning": "区域赛数据早于V2.1.0，台阶跨越按高地类30秒近似映射",
    }
    return intervals, summary
