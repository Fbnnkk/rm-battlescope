"""Infer death/respawn intervals and immediate paid revivals from telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .pipeline import CleanTrack


@dataclass(frozen=True)
class RevivalInference:
    time: float
    camp: str
    robot_id: int
    robot_type: str
    health: float
    max_health: float
    coin_before: float
    coin_after: float
    observed_coin_drop: float
    confidence: str
    reason: str

    def to_jsonable(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("time", "health", "max_health", "coin_before", "coin_after", "observed_coin_drop"):
            result[key] = round(float(result[key]), 3)
        return result


@dataclass(frozen=True)
class RespawnInterval:
    death_time: float
    end_time: float
    revive_time: float | None
    camp: str
    robot_id: int
    robot_type: str
    method: str
    required_progress: int
    paid_revivals_before_death: int
    observed_duration: float | None
    confidence: str

    def to_jsonable(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("death_time", "end_time", "revive_time", "observed_duration"):
            if result[key] is not None:
                result[key] = round(float(result[key]), 3)
        return result


def _largest_coin_drop(track: CleanTrack, revival_index: int) -> tuple[float, float, float]:
    best = (0.0, 0.0, 0.0)
    start = max(1, revival_index - 1)
    end = min(len(track.times) - 1, revival_index + 1)
    for index in range(start, end + 1):
        before = track.remaining_coins[index - 1]
        after = track.remaining_coins[index]
        if not np.isfinite(before) or not np.isfinite(after):
            continue
        drop = float(before - after)
        if drop > best[2]:
            best = (float(before), float(after), drop)
    return best


def infer_paid_revivals(
    tracks: list[CleanTrack], minimum_coin_drop: float = 50.0
) -> tuple[list[RevivalInference], dict[str, Any]]:
    """Detect full-health revivals accompanied by a nearby team-coin drop.

    The database has no explicit buyback event. Under the rule manual, normal
    timer revival restores 10% health while paid immediate revival restores
    100%, so the combination is a strong but still inferred signal.
    """

    revivals: list[RevivalInference] = []
    for track in tracks:
        if track.key.robot_type in {"基地", "前哨站", "空中"}:
            continue
        for index in range(1, len(track.times)):
            previous_health = track.health[index - 1]
            health = track.health[index]
            max_health = track.max_health[index]
            if (
                not np.isfinite(previous_health)
                or not np.isfinite(health)
                or not np.isfinite(max_health)
                or previous_health > 0.0
                or max_health <= 0.0
                or health / max_health < 0.75
                or track.times[index] - track.times[index - 1] > 2.0
            ):
                continue
            coin_before, coin_after, drop = _largest_coin_drop(track, index)
            if drop < minimum_coin_drop:
                continue
            full_ratio = float(health / max_health)
            confidence = "high" if full_ratio >= 0.95 and drop >= 100.0 else "medium"
            revivals.append(
                RevivalInference(
                    time=float(track.times[index]),
                    camp=track.key.camp,
                    robot_id=track.key.robot_id,
                    robot_type=track.key.robot_type,
                    health=float(health),
                    max_health=float(max_health),
                    coin_before=coin_before,
                    coin_after=coin_after,
                    observed_coin_drop=drop,
                    confidence=confidence,
                    reason="战亡后恢复至少75%血量，且同一秒附近队伍剩余金币下降",
                )
            )

    revivals.sort(key=lambda item: (item.time, item.camp, item.robot_id))
    summary = {
        "inferred_paid_revival_count": len(revivals),
        "high_confidence": sum(item.confidence == "high" for item in revivals),
        "medium_confidence": sum(item.confidence == "medium" for item in revivals),
        "has_native_buyback_event": False,
        "method": "0血量→至少75%血量，并在±1秒窗口检测到队伍剩余金币下降至少50",
        "warning": "立即复活由遥测识别，不是数据库原生事件",
    }
    return revivals, summary


def infer_respawn_intervals(
    tracks: list[CleanTrack],
    paid_revivals: list[RevivalInference],
    match_end: float | None = None,
) -> tuple[list[RespawnInterval], dict[str, Any]]:
    """Build observed death intervals and attach the V2.1 respawn-bar length.

    The bar length is rule-derived, while its displayed completion time is read
    from the next frame in which health becomes positive. This keeps accelerated
    supply/base respawns honest even though the dataset has no native progress
    or supply-zone occupancy field.
    """

    paid_lookup = {
        (item.camp, item.robot_id, int(round(item.time))): item for item in paid_revivals
    }
    intervals: list[RespawnInterval] = []
    for track in tracks:
        paid_count = 0
        for index in range(1, len(track.times)):
            previous = track.health[index - 1]
            current = track.health[index]
            if not np.isfinite(previous) or not np.isfinite(current):
                continue
            if previous <= 0.0 or current > 0.0:
                continue
            death_time = float(track.times[index])
            revive_index = next(
                (
                    candidate
                    for candidate in range(index + 1, len(track.times))
                    if np.isfinite(track.health[candidate]) and track.health[candidate] > 0.0
                ),
                None,
            )
            revive_time = (
                float(track.times[revive_index]) if revive_index is not None else None
            )
            paid = (
                paid_lookup.get((track.key.camp, track.key.robot_id, int(round(revive_time))))
                if revive_time is not None
                else None
            )
            method = "paid" if paid else ("timer" if revive_time is not None else "unrevived")
            # Manual V2.1: round 10 + elapsed/10 + 20 * prior paid revivals.
            required = int(np.floor(10.0 + death_time / 10.0 + 20.0 * paid_count + 0.5))
            end_time = revive_time if revive_time is not None else float(
                match_end if match_end is not None else track.times[-1]
            )
            intervals.append(
                RespawnInterval(
                    death_time=death_time,
                    end_time=end_time,
                    revive_time=revive_time,
                    camp=track.key.camp,
                    robot_id=track.key.robot_id,
                    robot_type=track.key.robot_type,
                    method=method,
                    required_progress=required,
                    paid_revivals_before_death=paid_count,
                    observed_duration=(revive_time - death_time) if revive_time is not None else None,
                    confidence="high" if revive_time is not None else "medium",
                )
            )
            if paid:
                paid_count += 1

    intervals.sort(key=lambda item: (item.death_time, item.camp, item.robot_id))
    summary = {
        "death_interval_count": len(intervals),
        "timer_respawn_count": sum(item.method == "timer" for item in intervals),
        "paid_respawn_count": sum(item.method == "paid" for item in intervals),
        "unrevived_count": sum(item.method == "unrevived" for item in intervals),
        "rule": "required=round(10+death_elapsed/10+20*prior_paid_revivals)",
        "warning": "数据库无原生复活进度；已完成区间按观测复活时刻回放，未完成区间按基础1点/秒估算",
    }
    return intervals, summary
