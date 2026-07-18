"""Interpret native dart-hit events using the RMUC 2026 V2.1 rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .pipeline import CleanTrack, MatchEvent


BASE_TARGETS = {
    200: ("固定目标", 0.0),
    300: ("随机固定目标", 0.0),
    625: ("随机移动目标", 10.0),
    1000: ("末端移动目标", 10.0),
}
FIXED_BLIND_DURATIONS = (10.0, 5.0, 3.0, 2.0)


@dataclass(frozen=True)
class DartImpact:
    time: float
    attacker_camp: str
    target_camp: str
    target_robot_id: int
    target_type: str
    target_profile: str
    damage: float
    blind_duration: float
    blind_end: float
    buff_suppression_duration: float
    buff_suppression_end: float
    target_point_disabled_until: float
    confidence: str

    def to_jsonable(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "time",
            "damage",
            "blind_duration",
            "blind_end",
            "buff_suppression_duration",
            "buff_suppression_end",
            "target_point_disabled_until",
        ):
            result[key] = round(float(result[key]), 3)
        return result


def infer_dart_impacts(
    events: list[MatchEvent], objectives: list[CleanTrack]
) -> tuple[list[DartImpact], dict[str, Any]]:
    objective_lookup = {track.key.robot_id: track for track in objectives}
    hit_order: dict[str, int] = {}
    stacked_moving_blind_end: dict[str, float] = {}
    fixed_blind_end: dict[str, float] = {}
    impacts: list[DartImpact] = []
    for event in events:
        if event.event_type != "飞镖命中" or event.target_robot_id is None:
            continue
        objective = objective_lookup.get(event.target_robot_id)
        if objective is None:
            continue
        attacker_camp = event.camp or ("蓝" if objective.key.camp == "红" else "红")
        damage = abs(float(event.value or 0.0))
        rounded_damage = int(round(damage))
        confidence = "high"
        if objective.key.robot_type == "前哨站":
            target_profile = "前哨站固定靶"
            buff_suppression = 0.0
            moving = False
        else:
            target_profile, moving_suppression = BASE_TARGETS.get(
                rounded_damage, ("基地飞镖靶（类型未定）", 0.0)
            )
            confidence = "high" if rounded_damage in BASE_TARGETS else "low"
            buff_suppression = (
                0.0
                if target_profile == "固定目标"
                else moving_suppression
            )
            moving = target_profile in {"随机移动目标", "末端移动目标"}

        order = hit_order.get(attacker_camp, 0)
        if moving:
            blind_duration = 10.0
            blind_start = max(event.time, stacked_moving_blind_end.get(objective.key.camp, event.time))
            blind_end = blind_start + blind_duration
            stacked_moving_blind_end[objective.key.camp] = blind_end
        else:
            blind_duration = FIXED_BLIND_DURATIONS[min(order, 3)]
            # Fixed targets refresh the masking window but must not shorten a
            # longer window that is already active. Moving targets are the
            # only profiles for which V2.1 explicitly says durations stack.
            blind_end = max(
                fixed_blind_end.get(objective.key.camp, event.time),
                event.time + blind_duration,
            )
            fixed_blind_end[objective.key.camp] = blind_end
            hit_order[attacker_camp] = order + 1
        if target_profile == "随机固定目标":
            buff_suppression = blind_duration
        impacts.append(
            DartImpact(
                time=float(event.time),
                attacker_camp=attacker_camp,
                target_camp=objective.key.camp,
                target_robot_id=objective.key.robot_id,
                target_type=objective.key.robot_type,
                target_profile=target_profile,
                damage=damage,
                blind_duration=blind_duration,
                blind_end=blind_end,
                buff_suppression_duration=buff_suppression,
                buff_suppression_end=float(event.time + buff_suppression),
                # This 30-second effect is conditional on the guide light. The
                # dataset has no light flag, so expose it as a possible window.
                target_point_disabled_until=float(event.time + 30.0),
                confidence=confidence,
            )
        )
    summary = {
        "dart_hit_count": len(impacts),
        "identified_target_profile_count": sum(item.confidence == "high" for item in impacts),
        "unknown_target_profile_count": sum(item.confidence != "high" for item in impacts),
        "mapping": "基地伤害200/300/625/1000对应固定/随机固定/随机移动/末端移动；前哨站为固定靶",
        "blind_timing": "固定/随机固定靶按10/5/3/2秒刷新且不缩短现有遮挡；随机移动/末端移动靶每次10秒叠加",
        "warning": "增益点失效30秒需飞镖引导灯亮起，数据库无该标志，因此仅展示为条件窗口",
    }
    return impacts, summary
