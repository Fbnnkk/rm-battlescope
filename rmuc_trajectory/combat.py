"""Infer plausible projectile attack relationships from event and pose telemetry."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from .pipeline import CleanTrack, MatchEvent


PROJECTILE_WINDOWS = {"17mm": 2.0, "42mm": 3.0}
PROJECTILE_SPEEDS = {"17mm": 25.0, "42mm": 16.5}


@dataclass(frozen=True)
class AttackInference:
    attacker_robot_id: int
    attacker_type: str
    attacker_camp: str
    victim_robot_id: int
    victim_type: str
    victim_camp: str
    caliber: str
    shot_time: float
    hit_time: float
    attacker_xy: tuple[float, float]
    victim_xy: tuple[float, float]
    heading_deg: float
    angle_error_deg: float
    distance_m: float
    delay_s: float
    burst_count: int
    continuous_count: int
    score: float
    confidence: str
    ambiguous: bool
    damage: float | None = None
    special_mode: str | None = None
    special_confidence: str | None = None
    special_reason: str | None = None

    def to_jsonable(self) -> dict[str, Any]:
        result = asdict(self)
        result["attacker_xy"] = [round(value, 3) for value in self.attacker_xy]
        result["victim_xy"] = [round(value, 3) for value in self.victim_xy]
        for key in ("shot_time", "hit_time", "heading_deg", "angle_error_deg", "distance_m", "delay_s", "score"):
            result[key] = round(float(result[key]), 3)
        if result["damage"] is not None:
            result["damage"] = round(float(result["damage"]), 3)
        return result


def _track_lookup(tracks: list[CleanTrack]) -> dict[tuple[str, int], CleanTrack]:
    return {(track.key.camp, track.key.robot_id): track for track in tracks}


def _state(track: CleanTrack, time: float) -> tuple[np.ndarray, float] | None:
    frame = int(round(time))
    indices = np.flatnonzero(track.times == frame)
    if not indices.size:
        return None
    index = int(indices[0])
    point = track.clean_xy[index]
    heading = track.heading_deg[index]
    if not np.isfinite(point).all() or not np.isfinite(heading):
        return None
    return point, float(heading)


def _angle_error(heading_deg: float, vector: np.ndarray) -> float:
    target = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    return abs((target - heading_deg + 180.0) % 360.0 - 180.0)


def _hero_deployed_evidence(track: CleanTrack, time: float, camp: str) -> tuple[bool, str]:
    recent = (track.times >= time - 2.0) & (track.times <= time)
    points = track.clean_xy[recent]
    powers = track.chassis_power[recent]
    valid_points = points[np.isfinite(points).all(axis=1)]
    valid_powers = powers[np.isfinite(powers)]
    if len(valid_points) < 2 or not len(valid_powers):
        return False, "部署遥测不足"
    displacement = float(np.linalg.norm(valid_points[-1] - valid_points[0]))
    power = float(np.median(valid_powers))
    x, y = valid_points[-1]
    # Approximate the two V2.1 deployment zones from the rule figure. Position
    # is supporting evidence only; no native deployment-mode flag is present.
    inside_zone = (camp == "红" and x <= 6.0 and y >= 10.0) or (
        camp == "蓝" and x >= 22.0 and y <= 5.8
    )
    detected = displacement <= 0.15 and power <= 5.0 and inside_zone
    return detected, f"2秒位移{displacement:.2f}m，底盘功率中位数{power:.1f}W，位于己方部署区近似范围"


def infer_attacks(
    events: list[MatchEvent], tracks: list[CleanTrack]
) -> tuple[list[AttackInference], dict[str, int]]:
    """Match each projectile hit to the most plausible recent opposing shot.

    Heading follows the dataset's empirically verified convention: zero points
    along +x and positive angles rotate counterclockwise.
    """

    lookup = _track_lookup(tracks)
    shots = [
        event
        for event in events
        if event.event_type == "发弹" and event.category in PROJECTILE_WINDOWS
    ]
    hits = [
        event
        for event in events
        if event.event_type == "受击" and event.category in PROJECTILE_WINDOWS
    ]
    hit_windows: dict[tuple[str | None, int | None, str | None, float], MatchEvent] = {}
    for hit in hits:
        hit_windows.setdefault((hit.camp, hit.robot_id, hit.category, hit.time), hit)
    inferences: list[AttackInference] = []
    unresolved = 0

    for hit in hit_windows.values():
        if hit.camp is None or hit.robot_id is None:
            unresolved += 1
            continue
        victim_track = lookup.get((hit.camp, hit.robot_id))
        victim_state = _state(victim_track, hit.time) if victim_track else None
        if victim_state is None:
            unresolved += 1
            continue
        victim_xy, _ = victim_state
        window = PROJECTILE_WINDOWS[hit.category]

        grouped: dict[tuple[str, int, float], list[MatchEvent]] = {}
        for shot in shots:
            if (
                shot.category != hit.category
                or shot.camp is None
                or shot.robot_id is None
                or shot.camp == hit.camp
                or shot.time > hit.time
                or hit.time - shot.time > window
            ):
                continue
            grouped.setdefault((shot.camp, shot.robot_id, shot.time), []).append(shot)

        candidates: list[tuple[float, MatchEvent, np.ndarray, float, float, float, int]] = []
        for burst in grouped.values():
            shot = burst[0]
            attacker_track = lookup.get((shot.camp, shot.robot_id))
            attacker_state = _state(attacker_track, shot.time) if attacker_track else None
            if attacker_state is None:
                continue
            attacker_xy, heading = attacker_state
            vector = victim_xy - attacker_xy
            distance = float(np.linalg.norm(vector))
            if distance < 0.05:
                continue
            delay = max(0.0, float(hit.time - shot.time))
            angle = _angle_error(heading, vector)
            allowed_distance = PROJECTILE_SPEEDS[hit.category] * (delay + 1.0) + 3.0
            angle_score = math.exp(-((angle / 24.0) ** 2))
            objective_ballistic = (
                hit.category == "42mm"
                and shot.robot_type == "英雄"
                and victim_track.key.robot_type in {"基地", "前哨站"}
            )
            if objective_ballistic:
                # Static-object hit and opposing hero shot in the same short
                # window are strong evidence even when planar yaw is noisy.
                angle_score = max(angle_score, 0.65)
            time_score = math.exp(-(delay / max(window, 0.1)))
            travel_score = math.exp(-max(0.0, distance - allowed_distance) / 4.0)
            burst_boost = min(1.15, 1.0 + 0.03 * (len(burst) - 1))
            score = angle_score * (0.55 + 0.45 * time_score) * travel_score * burst_boost
            candidates.append((score, shot, attacker_xy, heading, angle, distance, len(burst)))

        if not candidates:
            unresolved += 1
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        score, shot, attacker_xy, heading, angle, distance, burst_count = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else 0.0
        ambiguous = second_score >= score * 0.85
        margin = score - second_score
        objective_ballistic = (
            hit.category == "42mm"
            and shot.robot_type == "英雄"
            and victim_track.key.robot_type in {"基地", "前哨站"}
        )
        if angle <= 15.0 and score >= 0.45 and margin >= 0.15 and not ambiguous:
            confidence = "high"
        elif objective_ballistic and score >= 0.45 and margin >= 0.10 and not ambiguous:
            confidence = "medium"
        elif angle <= 32.0 and score >= 0.22 and margin >= 0.05 and not ambiguous:
            confidence = "medium"
        else:
            confidence = "low"
        deployed_detected = False
        deployed_reason = None
        if objective_ballistic and victim_track.key.robot_type == "基地":
            deployed_detected, deployed_reason = _hero_deployed_evidence(
                lookup[(str(shot.camp), int(shot.robot_id))], shot.time, str(shot.camp)
            )
        deployed_detected = (
            deployed_detected and abs(float(hit.value or 0.0)) in {225.0, 300.0}
        )
        inferences.append(
            AttackInference(
                attacker_robot_id=int(shot.robot_id),
                attacker_type=shot.robot_type or "未知",
                attacker_camp=str(shot.camp),
                victim_robot_id=int(hit.robot_id),
                victim_type=hit.robot_type or "未知",
                victim_camp=str(hit.camp),
                caliber=str(hit.category),
                shot_time=float(shot.time),
                hit_time=float(hit.time),
                attacker_xy=(float(attacker_xy[0]), float(attacker_xy[1])),
                victim_xy=(float(victim_xy[0]), float(victim_xy[1])),
                heading_deg=heading,
                angle_error_deg=angle,
                distance_m=distance,
                delay_s=max(0.0, float(hit.time - shot.time)),
                burst_count=burst_count,
                continuous_count=1,
                score=score,
                confidence=confidence,
                ambiguous=ambiguous,
                damage=abs(float(hit.value)) if hit.value is not None else None,
                special_mode="hero_deployed_lob" if deployed_detected else None,
                special_confidence="high" if deployed_detected else None,
                special_reason=deployed_reason,
            )
        )

    grouped_sequences: dict[tuple[str, int, str, int, str], list[int]] = {}
    for index, item in enumerate(inferences):
        key = (
            item.attacker_camp,
            item.attacker_robot_id,
            item.victim_camp,
            item.victim_robot_id,
            item.caliber,
        )
        grouped_sequences.setdefault(key, []).append(index)
    continuous_sequences = 0
    max_continuous_count = 1 if inferences else 0
    for indices in grouped_sequences.values():
        indices.sort(key=lambda index: inferences[index].hit_time)
        sequence: list[int] = []
        for index in indices:
            if sequence and inferences[index].hit_time - inferences[sequence[-1]].hit_time > 2.0:
                count = len(sequence)
                if count > 1:
                    continuous_sequences += 1
                max_continuous_count = max(max_continuous_count, count)
                for member in sequence:
                    inferences[member] = replace(inferences[member], continuous_count=count)
                sequence = []
            sequence.append(index)
        if sequence:
            count = len(sequence)
            if count > 1:
                continuous_sequences += 1
            max_continuous_count = max(max_continuous_count, count)
            for member in sequence:
                inferences[member] = replace(inferences[member], continuous_count=count)

    summary = {
        "projectile_hit_count": len(hits),
        "projectile_hit_window_count": len(hit_windows),
        "inferred_attack_count": len(inferences),
        "high_confidence": sum(item.confidence == "high" for item in inferences),
        "medium_confidence": sum(item.confidence == "medium" for item in inferences),
        "low_confidence": sum(item.confidence == "low" for item in inferences),
        "ambiguous": sum(item.ambiguous for item in inferences),
        "unresolved": unresolved,
        "continuous_attack_sequences": continuous_sequences,
        "max_continuous_attack_count": max_continuous_count,
    }
    return inferences, summary
