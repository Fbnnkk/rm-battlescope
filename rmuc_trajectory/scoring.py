"""Compute per-robot performance scores (0-10 scale) from match replay data.

Scoring is based on RMUC 2026 rules and uses data already loaded by the
replay pipeline: CleanTrack, MatchEvent, AttackInference, BuffInterval.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .buffs import TERRAIN_BUFFS, BuffInterval
from .combat import AttackInference
from .pipeline import CleanTrack, MatchEvent, TrackKey

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

STARTING_SCORE = 5.0
MIN_SCORE = 0.0
MAX_SCORE = 10.0

# -- universal weights ------------------------------------------------------
SCORE_KILL = 0.5
SCORE_ASSIST = 0.25
SCORE_DEATH = -0.5
SCORE_DAMAGE_PER_100HP = 0.10
SCORE_STRUCTURE_DAMAGE_PER_100HP = 0.20
SCORE_DAMAGE_RECEIVED_PER_100HP = -0.05
SCORE_RAMP = 0.12         # 飞坡
SCORE_STEP = 0.04          # 台阶跨越
SCORE_HIGHLAND_PER_SEC = 0.004  # 过中央高地
SCORE_PRESSURE_PER_SEC = 0.005   # 进入敌方半场（压迫感）
SCORE_WIN = 0.3
SCORE_FIRST_BLOOD = 0.5
SCORE_AFK_PENALTY = -1.5

# -- hero -------------------------------------------------------------------
SCORE_HERO_KILL = 0.55
SCORE_HERO_DEATH = -0.55
SCORE_HERO_42MM_SHOT = 0.03
SCORE_HERO_GOLD_SPENT_PER_COIN = -0.0005

# -- infantry 3/4 -----------------------------------------------------------
SCORE_INFANTRY_KILL_HERO_BONUS = 0.3
SCORE_INFANTRY_KILL_SENTRY_BONUS = 0.3
SCORE_INFANTRY_KILL_STREAK_PER_KILL = 0.1

# -- sentry -----------------------------------------------------------------
SCORE_SENTRY_KILL = 0.7
SCORE_SENTRY_DEATH = -0.7
SCORE_SENTRY_SURVIVAL_PER_SEC = 0.0012

# -- engineer ---------------------------------------------------------------
SCORE_ENGINEER_DEATH = -0.6
SCORE_ENGINEER_SURVIVAL_PER_SEC = 0.003
ENGINEER_TERRAIN_MULTIPLIER = 1.3

# -- aerial -----------------------------------------------------------------
SCORE_AERIAL_KILL = 0.45
SCORE_AERIAL_DEATH = -0.55
SCORE_AERIAL_ENEMY_HALF_PER_SEC = 0.002

# Zero-centred logistic efficiency: dps < midpoint → negative rate,
# dps > midpoint → positive rate.  Inactive aerials lose score over time
# while active ones gain — the median naturally converges to ~5.0.
AERIAL_LOGISTIC_MAX_RATE = 0.004   # max |rate| per second from efficiency
AERIAL_LOGISTIC_MIDPOINT = 2.5     # DPS at which rate crosses zero
AERIAL_LOGISTIC_STEEPNESS = 2.0    # transition sharpness

AERIAL_ROLLING_WINDOW = 10  # seconds for rolling efficiency

# -- energy mechanism (rune) activation ------------------------------------
# One-time score awarded when the energy mechanism is activated.
# Arm-hit events are grouped into activation sessions and scored once per
# session — individual lamp strikes are not scored separately.
SCORE_RUNE_LARGE = 0.3          # large rune (rune_type=1.0)
SCORE_RUNE_SMALL = 0.2          # small rune (rune_type=0.0)
RUNE_ACTIVATION_MAX_GAP = 10.0  # seconds between arm hits before new activation

# Heuristic: activating robot is near centre, heading toward it, on own half.
RUNE_CENTER = (14.0, 7.5)       # field center (28×15 m)
RUNE_DIST_MIN = 3.0
RUNE_DIST_MAX = 9.0
RUNE_OWN_HALF_MARGIN = 1.0   # allow 1 m into 'wrong' half near centre line
RUNE_ACTIVATOR_TYPES = {"步兵3", "步兵4", "空中"}

RADAR_JAM_DURATION_S = 30.0  # per RMUC V2.1 rules

# Per-type scaling factors (set to 1.0 — baseline anchored at 5.0,
# median varies naturally by type reflecting actual game impact).
TYPE_SCALING = {
    "英雄": 1.0,
    "工程": 1.0,
    "步兵3": 1.0,
    "步兵4": 1.0,
    "哨兵": 1.0,
    "空中": 1.0,
}

# ---------------------------------------------------------------------------
# Internal data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _KillEvent:
    time: float
    killer_camp: str
    killer_id: int
    killer_type: str
    victim_camp: str
    victim_id: int
    victim_type: str
    damage: float | None
    caliber: str | None
    assist_attackers: tuple[tuple[str, int, str], ...]  # (camp, id, type)


@dataclass(frozen=True)
class _DeathEvent:
    time: float
    camp: str
    robot_id: int
    robot_type: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_scores(
    match: dict[str, Any],
    tracks: list[CleanTrack],
    events: list[MatchEvent],
    attacks: list[AttackInference],
    buff_intervals: list[BuffInterval],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute per-robot performance scores for a single match.

    Returns ``(score_dicts, summary)`` where each score dict describes one
    robot and the summary aggregates match-level statistics.
    """
    # Separate mobile robots from static objectives
    mobile_tracks = [t for t in tracks if t.key.robot_type not in {"基地", "前哨站", "飞镖"}]
    track_lookup: dict[tuple[str, int], CleanTrack] = {
        (t.key.camp, t.key.robot_id): t for t in tracks
    }

    # ---- detect kills, deaths, assists -----------------------------------
    kill_events, death_events, assist_map = _detect_kills_and_deaths(
        mobile_tracks, attacks
    )

    # ---- aggregate damage stats ------------------------------------------
    robot_dmg, base_dmg, outpost_dmg, dmg_received = _compute_damage_stats(attacks)

    # ---- base HP gap for multiplier -------------------------------------
    base_hp: dict[str, float] = {}
    for t in tracks:
        if t.key.robot_type == "基地":
            hp = t.health
            valid = hp[np.isfinite(hp) & (hp > 0)]
            base_hp[t.key.camp] = float(valid[-1]) if len(valid) > 0 else 0.0

    # ---- aggregate terrain stats -----------------------------------------
    terrain_stats = _compute_terrain_scores(buff_intervals)

    # ---- first blood -----------------------------------------------------
    first_blood = _find_first_blood(kill_events)

    # ---- energy mechanism (rune) activations -----------------------------
    rune_acts = _detect_rune_activations(events, mobile_tracks)
    rune_scores: dict[tuple[str, int], float] = defaultdict(float)
    for ra in rune_acts:
        rune_scores[(ra.camp, ra.robot_id)] += ra.score

    # ---- win bonus -------------------------------------------------------
    winner = match.get("胜方", "")

    # ---- score each robot ------------------------------------------------
    score_list: list[dict[str, Any]] = []
    for track in mobile_tracks:
        key = (track.key.camp, track.key.robot_id)
        my_base_hp = base_hp.get(track.key.camp, 0.0)
        enemy_camp = "红" if track.key.camp == "蓝" else "蓝"
        enemy_base_hp = base_hp.get(enemy_camp, 0.0)
        base_hp_gap = my_base_hp - enemy_base_hp

        stats = _score_one_robot(
            track=track,
            kill_events=kill_events,
            death_events=death_events,
            assist_map=assist_map,
            robot_dmg=robot_dmg,
            base_dmg=base_dmg,
            outpost_dmg=outpost_dmg,
            dmg_received=dmg_received,
            terrain_stats=terrain_stats,
            first_blood=first_blood,
            winner=winner,
            events=events,
            rune_act_count=len([ra for ra in rune_acts
                                if ra.camp == track.key.camp and ra.robot_id == track.key.robot_id]),
            rune_score=rune_scores.get(key, 0.0),
            base_hp_gap=base_hp_gap,
        )
        score_list.append(stats)

    # Sort: winner first, then by type rank, then by robot_id
    type_rank = {"英雄": 0, "工程": 1, "步兵3": 2, "步兵4": 3, "哨兵": 4, "空中": 5}

    def _sort_key(d: dict) -> tuple[int, int, int, int]:
        is_winner = 0 if d["camp"] == winner else 1
        rank = type_rank.get(d["robot_type"], 99)
        return (is_winner, rank, d["robot_id"])

    score_list.sort(key=_sort_key)

    # ---- summary ---------------------------------------------------------
    scores_only = [s["total_score"] for s in score_list]
    afk_count = sum(
        1 for s in score_list if s["components"].get("afk_penalty", {}).get("value")
    )

    summary: dict[str, Any] = {
        "robot_count": len(score_list),
        "min_score": round(min(scores_only) if scores_only else 0.0, 2),
        "max_score": round(max(scores_only) if scores_only else 0.0, 2),
        "mean_score": round(float(np.mean(scores_only)) if scores_only else 0.0, 2),
        "kill_count": len(kill_events),
        "death_count": len(death_events),
        "assist_count": sum(assist_map.values()),
        "first_blood_time": round(first_blood.time, 1) if first_blood else None,
        "first_blood_robot": (
            {
                "camp": first_blood.killer_camp,
                "robot_id": first_blood.killer_id,
                "robot_type": first_blood.killer_type,
            }
            if first_blood
            else None
        ),
        "afk_robot_count": afk_count,
        "score_range": [MIN_SCORE, MAX_SCORE],
        "rule_version": "RMUC 2026 V2.1 scoring",
        "notes": [
            "Gold spending estimated from team-level coin data, attributed to hero.",
            "Damage scoring covers projectile damage (17mm/42mm) from AttackInference only.",
            "Radar jam duration assumed 30s per event per RMUC V2.1 rules.",
            "Deaths without attributed attacks still incur the death penalty.",
        ],
    }
    return score_list, summary


# ---------------------------------------------------------------------------
# Kill / death / assist detection
# ---------------------------------------------------------------------------


def _detect_kills_and_deaths(
    tracks: list[CleanTrack],
    attacks: list[AttackInference],
) -> tuple[list[_KillEvent], list[_DeathEvent], dict[tuple[str, int], int]]:
    kill_events: list[_KillEvent] = []
    death_events: list[_DeathEvent] = []
    assist_count: dict[tuple[str, int], int] = defaultdict(int)

    for track in tracks:
        health = track.health
        times = track.times
        camp = track.key.camp
        rid = track.key.robot_id
        rtype = track.key.robot_type

        for i in range(1, len(health)):
            if health[i - 1] > 0 and health[i] <= 0:
                death_time = times[i]
                death_events.append(_DeathEvent(death_time, camp, rid, rtype))

                # find attacks on this victim near death time
                relevant = [
                    a
                    for a in attacks
                    if a.victim_camp == camp
                    and a.victim_robot_id == rid
                    and (death_time - 5.0) <= a.hit_time <= (death_time + 1.0)
                ]
                if not relevant:
                    continue

                relevant.sort(key=lambda a: a.hit_time, reverse=True)
                killing_blow = relevant[0]

                # assist candidates: hits within 5s before death, excluding the killer
                seen: set[tuple[str, int]] = {
                    (killing_blow.attacker_camp, killing_blow.attacker_robot_id)
                }
                assist_attackers: list[tuple[str, int, str]] = []
                for a in relevant[1:]:
                    if (death_time - a.hit_time) > 5.0:
                        continue
                    akey = (a.attacker_camp, a.attacker_robot_id)
                    if akey not in seen:
                        seen.add(akey)
                        assist_attackers.append(
                            (a.attacker_camp, a.attacker_robot_id, a.attacker_type)
                        )
                        assist_count[akey] += 1

                kill_events.append(
                    _KillEvent(
                        time=death_time,
                        killer_camp=killing_blow.attacker_camp,
                        killer_id=killing_blow.attacker_robot_id,
                        killer_type=killing_blow.attacker_type,
                        victim_camp=camp,
                        victim_id=rid,
                        victim_type=rtype,
                        damage=killing_blow.damage,
                        caliber=killing_blow.caliber,
                        assist_attackers=tuple(assist_attackers),
                    )
                )

    return kill_events, death_events, dict(assist_count)


# ---------------------------------------------------------------------------
# Damage aggregation
# ---------------------------------------------------------------------------

def _compute_damage_stats(
    attacks: list[AttackInference],
) -> tuple[
    dict[tuple[str, int], float],
    dict[tuple[str, int], float],
    dict[tuple[str, int], float],
    dict[tuple[str, int], float],
]:
    robot_dmg: dict[tuple[str, int], float] = defaultdict(float)
    base_dmg: dict[tuple[str, int], float] = defaultdict(float)
    outpost_dmg: dict[tuple[str, int], float] = defaultdict(float)
    received: dict[tuple[str, int], float] = defaultdict(float)

    for a in attacks:
        dmg = a.damage or 0.0
        atk_key = (a.attacker_camp, a.attacker_robot_id)
        vic_key = (a.victim_camp, a.victim_robot_id)

        if a.victim_type == "基地":
            base_dmg[atk_key] += dmg
        elif a.victim_type == "前哨站":
            outpost_dmg[atk_key] += dmg
        else:
            robot_dmg[atk_key] += dmg
        received[vic_key] += dmg

    return dict(robot_dmg), dict(base_dmg), dict(outpost_dmg), dict(received)


# ---------------------------------------------------------------------------
# Terrain scoring
# ---------------------------------------------------------------------------


def _compute_terrain_scores(
    buff_intervals: list[BuffInterval],
) -> dict[tuple[str, int], dict[str, float]]:
    stats: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: {"飞坡": 0, "台阶跨越": 0, "过中央高地": 0.0}
    )
    for bi in buff_intervals:
        if bi.category not in TERRAIN_BUFFS:
            continue
        key = (bi.camp, bi.robot_id)
        if bi.category in {"飞坡", "台阶跨越"}:
            stats[key][bi.category] += 1
        else:
            stats[key][bi.category] += bi.duration
    return dict(stats)


# ---------------------------------------------------------------------------
# First blood
# ---------------------------------------------------------------------------


def _find_first_blood(kill_events: list[_KillEvent]) -> _KillEvent | None:
    if not kill_events:
        return None
    return min(kill_events, key=lambda k: k.time)


# ---------------------------------------------------------------------------
# Energy mechanism (rune) activation detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RuneActivation:
    time: float
    camp: str
    robot_id: int
    robot_type: str
    rune_type: str  # "large" or "small"
    avg_round: float
    arm_count: int
    score: float


def _detect_rune_activations(
    events: list[MatchEvent],
    tracks: list[CleanTrack],
) -> list[_RuneActivation]:
    """Detect energy-mechanism activations from per-arm hit events.

    Arm-hit events are grouped into activation sessions (arm_cnt reset
    to 1 after ≥3 arms, or gap > 10 s).  Each session is scored once.
    The activating robot is identified by distance to centre, heading
    toward centre, and own-half position.  When two robots are equally
    likely the score is split between them.
    """
    import math
    import re

    track_lookup: dict[tuple[str, int], CleanTrack] = {
        (t.key.camp, t.key.robot_id): t for t in tracks
    }

    # ---- collect & parse raw arm-hit events --------------------------------
    _RawHit = tuple[float, str, str, int, float]  # time, camp, rune_type, arm_cnt, avg_round
    raw_hits: list[_RawHit] = []
    for event in events:
        if event.event_type != "能量机关" or not event.camp:
            continue
        if not event.category or not event.note:
            continue
        arm_match = re.search(r"arm_cnt=([0-9.]+)", event.note)
        round_match = re.search(r"avg_round=([0-9.]+)", event.note)
        if not arm_match or not round_match:
            continue
        rune_type = "small" if "rune_type=0.0" in event.category else "large"
        arm_count = int(float(arm_match.group(1)))
        avg_round = float(round_match.group(1))
        raw_hits.append((event.time, event.camp, rune_type, arm_count, avg_round))

    if not raw_hits:
        return []

    raw_hits.sort(key=lambda h: (h[1], h[0]))  # by camp, then time

    # ---- group into activation sessions per camp ---------------------------
    groups: list[list[_RawHit]] = []
    for camp in sorted({h[1] for h in raw_hits}):
        camp_hits = sorted(
            [h for h in raw_hits if h[1] == camp],
            key=lambda h: (h[0], h[3]),
        )
        current: list[_RawHit] = []
        max_arm_seen = 0
        for hit in camp_hits:
            arm = hit[3]
            gap = hit[0] - current[-1][0] if current else 999.0
            start_new = False
            if current and gap > RUNE_ACTIVATION_MAX_GAP:
                start_new = True
            elif current and arm == 1 and max_arm_seen >= 3:
                start_new = True
            if start_new:
                groups.append(current)
                current = []
                max_arm_seen = 0
            current.append(hit)
            if arm > max_arm_seen:
                max_arm_seen = arm
        if current:
            groups.append(current)

    # ---- for each group, find activator(s) & score once --------------------
    cx, cy = RUNE_CENTER
    FIELD_HALF_X = 14.0

    activations: list[_RuneActivation] = []

    for group in groups:
        t0 = group[0][0]          # activation start time
        camp = group[0][1]
        rune_type = group[0][2]
        best_round = max(h[4] for h in group)

        # Score candidates: distance (40%) + heading (60%)
        candidates: list[tuple[float, int, str]] = []  # (score, robot_id, type)
        for key, track in track_lookup.items():
            if key[0] != camp:
                continue
            if track.key.robot_type not in RUNE_ACTIVATOR_TYPES:
                continue
            idx = int(round(t0 - track.times[0]))
            if idx < 0 or idx >= len(track.times):
                continue
            px = track.clean_xy[idx, 0]
            py = track.clean_xy[idx, 1]
            if not np.isfinite(px) or not np.isfinite(py):
                continue

            # Must be near centre
            dist = math.hypot(px - cx, py - cy)
            if dist < RUNE_DIST_MIN or dist > RUNE_DIST_MAX:
                continue

            # Must be on own half (with 1 m margin near centre line)
            half_ok = True
            if camp == "红" and px > FIELD_HALF_X + RUNE_OWN_HALF_MARGIN:
                half_ok = False
            if camp == "蓝" and px < FIELD_HALF_X - RUNE_OWN_HALF_MARGIN:
                half_ok = False
            if not half_ok:
                continue

            # Distance score: prefer ~5 m from centre
            dist_score = max(0.0, 1.0 - abs(dist - 5.0) / 3.0)

            # Heading score: prefer facing centre (never reject, just de-prioritise)
            heading = float(track.heading_deg[idx])
            angle_score = 0.1  # floor — never reject due to angle alone
            if np.isfinite(heading):
                heading_rad = math.radians(heading)
                to_center = math.atan2(cy - py, cx - px)
                angle_diff = abs(math.degrees(
                    math.atan2(math.sin(to_center - heading_rad),
                               math.cos(to_center - heading_rad))
                ))
                angle_score = max(0.1, 1.0 - angle_diff / 90.0)

            total = dist_score * 0.4 + angle_score * 0.6
            candidates.append((total, key[1], track.key.robot_type))

        if not candidates:
            continue

        candidates.sort(key=lambda c: c[0], reverse=True)
        best_score = candidates[0][0]

        # Split score among robots within 15 % of the best candidate
        threshold = best_score * 0.85
        top = [c for c in candidates if c[0] >= threshold]
        share = 1.0 / len(top)

        base = SCORE_RUNE_LARGE if rune_type == "large" else SCORE_RUNE_SMALL

        for _, rid, rtype in top:
            activations.append(_RuneActivation(
                time=t0, camp=camp, robot_id=rid,
                robot_type=rtype,
                rune_type=rune_type, avg_round=best_round,
                arm_count=max(h[3] for h in group),
                score=base * share,
            ))

    return activations


def _compute_kill_streaks(
    kill_events: list[_KillEvent],
    death_events: list[_DeathEvent],
    camp: str,
    robot_id: int,
) -> tuple[int, int]:
    """Return (max_streak, total_bonus_kills) for one robot."""
    my_kills = sorted(
        [k for k in kill_events if k.killer_camp == camp and k.killer_id == robot_id],
        key=lambda k: k.time,
    )
    my_deaths = sorted(
        [d.time for d in death_events if d.camp == camp and d.robot_id == robot_id]
    )

    if not my_kills:
        return 0, 0

    max_streak = 1
    current_streak = 1
    death_idx = 0
    total_bonus = 0

    for i, k in enumerate(my_kills):
        # check if a death occurred between previous kill (if any) and this one
        if i > 0:
            prev_time = my_kills[i - 1].time
            died_between = any(prev_time < dt <= k.time for dt in my_deaths)
            if died_between:
                current_streak = 1
            else:
                current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak

        # bonus kills = streak length - 1 (the first kill in each streak is "free")
        if current_streak > 1:
            total_bonus += 1

    return max_streak, total_bonus


# ---------------------------------------------------------------------------
# Survival time
# ---------------------------------------------------------------------------


def _compute_survival_time(
    track: CleanTrack,
    death_events: list[_DeathEvent],
) -> float:
    """Total seconds the robot was alive (health > 0)."""
    camp = track.key.camp
    rid = track.key.robot_id
    alive = np.isfinite(track.health) & (track.health > 0)
    return float(np.count_nonzero(alive))


# ---------------------------------------------------------------------------
# Aerial helpers
# ---------------------------------------------------------------------------


def _merge_jam_intervals(
    events: list[MatchEvent],
    camp: str,
    robot_id: int,
) -> float:
    """Total non-overlapping radar-jammed duration for an aerial robot."""
    starts = [
        e.time
        for e in events
        if e.event_type == "雷达反制UAV"
        and e.robot_id == robot_id
        and e.camp == camp
    ]
    if not starts:
        return 0.0

    starts.sort()
    total = 0.0
    cur_start = starts[0]
    cur_end = cur_start + RADAR_JAM_DURATION_S

    for t in starts[1:]:
        end = t + RADAR_JAM_DURATION_S
        if t <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start = t
            cur_end = end
    total += cur_end - cur_start
    return total


def _compute_enemy_half_time(track: CleanTrack) -> float:
    """Seconds spent on the enemy half of the field while alive."""
    camp = track.key.camp
    half = 14.0  # FIELD_WIDTH_M / 2
    x = track.clean_xy[:, 0]
    alive = np.isfinite(track.health) & (track.health > 0)

    if camp == "红":
        on_enemy = x > half
    else:
        on_enemy = x < half

    return float(np.count_nonzero(on_enemy & alive))


# ---------------------------------------------------------------------------
# Per-robot scoring
# ---------------------------------------------------------------------------


def _score_one_robot(
    track: CleanTrack,
    kill_events: list[_KillEvent],
    death_events: list[_DeathEvent],
    assist_map: dict[tuple[str, int], int],
    robot_dmg: dict[tuple[str, int], float],
    base_dmg: dict[tuple[str, int], float],
    outpost_dmg: dict[tuple[str, int], float],
    dmg_received: dict[tuple[str, int], float],
    terrain_stats: dict[tuple[str, int], dict[str, float]],
    first_blood: _KillEvent | None,
    winner: str,
    events: list[MatchEvent],
    rune_act_count: int = 0,
    rune_score: float = 0.0,
    base_hp_gap: float = 0.0,
) -> dict[str, Any]:
    camp = track.key.camp
    rid = track.key.robot_id
    rtype = track.key.robot_type
    school = track.key.school
    key = (camp, rid)

    # counts
    my_kills = [k for k in kill_events if k.killer_camp == camp and k.killer_id == rid]
    my_deaths = [
        d for d in death_events if d.camp == camp and d.robot_id == rid
    ]
    kill_count = len(my_kills)
    death_count = len(my_deaths)
    assist_count = assist_map.get(key, 0)

    # damage
    dmg_robot = robot_dmg.get(key, 0.0)
    dmg_base = base_dmg.get(key, 0.0)
    dmg_outpost = outpost_dmg.get(key, 0.0)
    dmg_recv = dmg_received.get(key, 0.0)

    # ---- base damage multiplier based on HP gap -------------------------
    gap = base_hp_gap
    if abs(gap) <= 300:
        base_mult = 2.0
    elif 300 < gap <= 1000:
        base_mult = 1.5
    else:
        base_mult = 1.0

    # terrain
    terrain = terrain_stats.get(key, {"飞坡": 0, "台阶跨越": 0, "过中央高地": 0.0})
    bump_count = int(terrain.get("飞坡", 0))
    step_count = int(terrain.get("台阶跨越", 0))
    highland_sec = float(terrain.get("过中央高地", 0.0))

    # win
    is_winner = camp == winner

    # first blood
    has_first_blood = (
        first_blood is not None
        and first_blood.killer_camp == camp
        and first_blood.killer_id == rid
    )

    # ---- role-specific logic ---------------------------------------------
    hero = None
    infantry = None
    sentry = None
    engineer = None
    aerial = None

    # Kill / death weights (may be overridden per role)
    kill_weight = SCORE_KILL
    death_weight = SCORE_DEATH

    if rtype == "英雄":
        kill_weight = SCORE_HERO_KILL
        death_weight = SCORE_HERO_DEATH
        # 42mm shots
        c42 = track.cumulative_42mm
        valid_42 = c42[np.isfinite(c42)]
        shots_42 = int(valid_42[-1] - valid_42[0]) if len(valid_42) >= 2 else 0
        # gold spent
        spent_arr = track.total_coins - track.remaining_coins
        valid_spent = spent_arr[np.isfinite(spent_arr)]
        gold_spent = float(np.max(valid_spent)) if len(valid_spent) else 0.0
        hero = {
            "42mm_shots": {"value": shots_42, "score": round(shots_42 * SCORE_HERO_42MM_SHOT, 4)},
            "gold_spent": {
                "value": round(gold_spent, 0),
                "score": round(gold_spent * SCORE_HERO_GOLD_SPENT_PER_COIN, 4),
            },
        }

    elif rtype in {"步兵3", "步兵4"}:
        hero_kill_count = sum(
            1 for k in my_kills if k.victim_type == "英雄"
        )
        sentry_kill_count = sum(
            1 for k in my_kills if k.victim_type == "哨兵"
        )
        max_streak, streak_bonus = _compute_kill_streaks(
            kill_events, death_events, camp, rid
        )
        infantry = {
            "hero_kills": {
                "value": hero_kill_count,
                "score": round(hero_kill_count * SCORE_INFANTRY_KILL_HERO_BONUS, 4),
            },
            "sentry_kills": {
                "value": sentry_kill_count,
                "score": round(sentry_kill_count * SCORE_INFANTRY_KILL_SENTRY_BONUS, 4),
            },
            "max_kill_streak": {
                "value": max_streak,
                "bonus_kills": streak_bonus,
                "score": round(streak_bonus * SCORE_INFANTRY_KILL_STREAK_PER_KILL, 4),
            },
        }

    elif rtype == "哨兵":
        kill_weight = SCORE_SENTRY_KILL
        death_weight = SCORE_SENTRY_DEATH
        surv = _compute_survival_time(track, death_events)
        sentry = {
            "survival_duration": {
                "value": surv,
                "score": round(surv * SCORE_SENTRY_SURVIVAL_PER_SEC, 4),
            },
        }

    elif rtype == "工程":
        death_weight = SCORE_ENGINEER_DEATH
        surv = _compute_survival_time(track, death_events)
        engineer = {
            "survival_duration": {
                "value": surv,
                "score": round(surv * SCORE_ENGINEER_SURVIVAL_PER_SEC, 4),
            },
            "terrain_multiplier": ENGINEER_TERRAIN_MULTIPLIER,
        }

    elif rtype == "空中":
        kill_weight = SCORE_AERIAL_KILL
        death_weight = SCORE_AERIAL_DEATH
        # Zero-centred efficiency: dps < midpoint → lose score, dps > midpoint → gain
        dmg = dmg_robot + dmg_base + dmg_outpost
        active_time = float(track.times[-1] - track.times[0]) if len(track.times) > 0 else 0.0
        jammed = _merge_jam_intervals(events, camp, rid)
        effective_time = max(1.0, active_time - jammed)
        dps = dmg / effective_time
        import math
        # rate ∈ [-MAX_RATE, +MAX_RATE], zero at dps=midpoint
        k = AERIAL_LOGISTIC_STEEPNESS
        d0 = AERIAL_LOGISTIC_MIDPOINT
        rate = AERIAL_LOGISTIC_MAX_RATE * (
            2.0 / (1.0 + math.exp(-k * (dps - d0))) - 1.0
        )
        eff_score = active_time * rate
        # enemy half
        enemy_sec = _compute_enemy_half_time(track)
        aerial = {
            "active_time": round(active_time, 1),
            "jammed_time": round(jammed, 1),
            "output_efficiency_dps": round(dps, 3),
            "efficiency_score": round(eff_score, 4),
            "enemy_half_time": {
                "value": enemy_sec,
                "score": round(enemy_sec * SCORE_AERIAL_ENEMY_HALF_PER_SEC, 4),
            },
        }

    # ---- assemble components ---------------------------------------------
    terrain_bump_score = bump_count * SCORE_RAMP
    terrain_step_score = step_count * SCORE_STEP
    terrain_highland_score = highland_sec * SCORE_HIGHLAND_PER_SEC

    if rtype == "工程":
        terrain_bump_score *= ENGINEER_TERRAIN_MULTIPLIER
        terrain_step_score *= ENGINEER_TERRAIN_MULTIPLIER
        terrain_highland_score *= ENGINEER_TERRAIN_MULTIPLIER

    # Universal: enemy-half pressure (all types)
    enemy_half_sec = _compute_enemy_half_time(track)
    pressure_score = enemy_half_sec * SCORE_PRESSURE_PER_SEC

    components: dict[str, dict[str, Any]] = {
        "kills": {"value": kill_count, "score": round(kill_count * kill_weight, 4)},
        "assists": {"value": assist_count, "score": round(assist_count * SCORE_ASSIST, 4)},
        "deaths": {"value": death_count, "score": round(death_count * death_weight, 4)},
        "damage_to_robots": {
            "value": round(dmg_robot, 1),
            "score": round(dmg_robot / 100.0 * SCORE_DAMAGE_PER_100HP, 4),
        },
        "damage_to_base": {
            "value": round(dmg_base, 1),
            "score": round(dmg_base / 100.0 * SCORE_STRUCTURE_DAMAGE_PER_100HP * base_mult, 4),
            "multiplier": base_mult,
            "base_hp_gap": round(gap, 0),
        },
        "damage_to_outpost": {
            "value": round(dmg_outpost, 1),
            "score": round(dmg_outpost / 100.0 * SCORE_STRUCTURE_DAMAGE_PER_100HP, 4),
        },
        "damage_received": {
            "value": round(dmg_recv, 1),
            "score": round(dmg_recv / 100.0 * SCORE_DAMAGE_RECEIVED_PER_100HP, 4),
        },
        "terrain_bumps": {"value": bump_count, "score": round(terrain_bump_score, 4)},
        "terrain_steps": {"value": step_count, "score": round(terrain_step_score, 4)},
        "terrain_highland": {
            "value": round(highland_sec, 1),
            "score": round(terrain_highland_score, 4),
        },
        "pressure": {
            "value": round(enemy_half_sec, 1),
            "score": round(pressure_score, 4),
        },
        "win_bonus": {"value": is_winner, "score": SCORE_WIN if is_winner else 0.0},
    }

    # first blood
    components["first_blood"] = {
        "value": has_first_blood,
        "score": SCORE_FIRST_BLOOD if has_first_blood else 0.0,
    }

    # ---- compute total ---------------------------------------------------
    total = STARTING_SCORE
    for comp in components.values():
        total += comp["score"]

    if hero:
        total += hero["42mm_shots"]["score"]
        total += hero["gold_spent"]["score"]
    if infantry:
        total += infantry["hero_kills"]["score"]
        total += infantry["sentry_kills"]["score"]
        total += infantry["max_kill_streak"]["score"]
    if sentry:
        total += sentry["survival_duration"]["score"]
    if engineer:
        total += engineer["survival_duration"]["score"]
    if aerial:
        total += aerial["efficiency_score"]
        total += aerial["enemy_half_time"]["score"]

    # Energy mechanism (rune) activation
    components["rune_activations"] = {
        "value": rune_act_count,
        "score": round(rune_score, 4),
    }
    total += rune_score

    # AFK check (engineer excluded — their role is survival/defence)
    is_afk = (
        rtype != "工程"
        and kill_count == 0
        and assist_count == 0
        and dmg_robot == 0
        and dmg_base == 0 and dmg_outpost == 0
        and death_count >= 2
    )
    afk_score = SCORE_AFK_PENALTY if is_afk else 0.0
    components["afk_penalty"] = {"value": is_afk, "score": afk_score}
    total += afk_score

    total = max(MIN_SCORE, min(MAX_SCORE, total))

    # Per-type league calibration scaling (delta from baseline only)
    scale = TYPE_SCALING.get(rtype, 1.0)
    total = STARTING_SCORE + (total - STARTING_SCORE) * scale
    total = max(MIN_SCORE, min(MAX_SCORE, total))

    # ---- build result dict -----------------------------------------------
    result: dict[str, Any] = {
        "camp": camp,
        "robot_id": rid,
        "robot_type": rtype,
        "school": school,
        "total_score": round(total, 2),
        "components": components,
    }
    if hero:
        result["hero"] = hero
    if infantry:
        result["infantry"] = infantry
    if sentry:
        result["sentry"] = sentry
    if engineer:
        result["engineer"] = engineer
    if aerial:
        result["aerial"] = aerial

    return result


# ---------------------------------------------------------------------------
# Timeseries scoring (per-second cumulative scores)
# ---------------------------------------------------------------------------


def compute_timeseries_scores(
    match: dict[str, Any],
    tracks: list[CleanTrack],
    events: list[MatchEvent],
    attacks: list[AttackInference],
    buff_intervals: list[BuffInterval],
) -> dict[str, list[dict[str, Any]]]:
    """Compute per-second cumulative scores for every robot.

    Returns a dict keyed by camp (``"红"`` / ``"蓝"``) whose values are
    lists of per-robot timeseries dicts.  Each dict contains::

        {"camp", "robot_id", "robot_type",
         "times": [0, 1, ..., N],   # integer-second time grid
         "score": [5.0, ..., 8.2]}  # cumulative score at each second

    The final value ``score[-1]`` equals the ``total_score`` from
    :func:`compute_scores`.
    """
    mobile_tracks = [t for t in tracks if t.key.robot_type not in {"基地", "前哨站", "飞镖"}]
    camps = sorted({t.key.camp for t in mobile_tracks})

    # ---- reuse shared event detection ------------------------------------
    kill_events, death_events, assist_map = _detect_kills_and_deaths(
        mobile_tracks, attacks
    )
    first_blood = _find_first_blood(kill_events)
    winner = match.get("胜方", "")

    # ---- determine time grid ----------------------------------------------
    all_times = np.concatenate([t.times for t in mobile_tracks])
    t_min = int(np.floor(np.min(all_times)))
    t_max = int(np.ceil(np.max(all_times)))
    time_grid = np.arange(t_min, t_max + 1, dtype=float)
    n_steps = len(time_grid)

    # ---- build base HP arrays for per-second gap multiplier ----------------
    base_hp_arrays: dict[str, np.ndarray] = {}
    for t in tracks:
        if t.key.robot_type == "基地":
            bp = np.full(n_steps, np.nan)
            t_rel = (t.times - t_min).astype(int)
            valid = (t_rel >= 0) & (t_rel < n_steps)
            bp[t_rel[valid]] = t.health[valid]
            base_hp_arrays[t.key.camp] = bp

    # ---- build per-robot arrays -------------------------------------------
    robot_keys: list[tuple[str, int]] = []
    robot_arrays: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    for track in mobile_tracks:
        key = (track.key.camp, track.key.robot_id)
        robot_keys.append(key)
        rtype = track.key.robot_type

        # Map track times to grid indices
        t_rel = (track.times - t_min).astype(int)
        valid = (t_rel >= 0) & (t_rel < n_steps)

        alive = np.zeros(n_steps, dtype=bool)
        alive[t_rel[valid]] = np.isfinite(track.health[valid]) & (track.health[valid] > 0)

        # Enemy half
        half = 14.0
        x = track.clean_xy[:, 0]
        if track.key.camp == "红":
            enemy = x > half
        else:
            enemy = x < half
        enemy_half = np.zeros(n_steps, dtype=bool)
        enemy_half[t_rel[valid]] = enemy[valid] & alive[t_rel[valid]]

        # Cumulative shots / coins from track arrays
        shots_42 = np.zeros(n_steps, dtype=float)
        c42 = track.cumulative_42mm
        valid_42 = c42[np.isfinite(c42)]
        if len(valid_42) > 0:
            shots_42[t_rel[valid]] = valid_42[valid] - valid_42[0]

        spent = np.zeros(n_steps, dtype=float)
        raw_spent = track.total_coins - track.remaining_coins
        valid_raw = raw_spent[np.isfinite(raw_spent)]
        if len(valid_raw) > 0:
            spent[t_rel[valid]] = np.maximum(0, valid_raw[valid])

        robot_arrays[key] = {
            "type": rtype,
            "alive": alive,
            "enemy_half": enemy_half,
            "shots_42": shots_42,
            "spent": spent,
            "kills": np.zeros(n_steps, dtype=float),
            "deaths": np.zeros(n_steps, dtype=float),
            "assists": np.zeros(n_steps, dtype=float),
            "dmg_robot": np.zeros(n_steps, dtype=float),
            "dmg_base": np.zeros(n_steps, dtype=float),
            "dmg_outpost": np.zeros(n_steps, dtype=float),
            "dmg_received": np.zeros(n_steps, dtype=float),
            "highland_sec": np.zeros(n_steps, dtype=float),
            "ramp_count": np.zeros(n_steps, dtype=float),
            "step_count": np.zeros(n_steps, dtype=float),
            "hero_kill": np.zeros(n_steps, dtype=float),
            "sentry_kill": np.zeros(n_steps, dtype=float),
            "streak_bonus": np.zeros(n_steps, dtype=float),
            "rune_score": np.zeros(n_steps, dtype=float),
        }

    # ---- bin events -------------------------------------------------------
    def _sec_idx(t: float) -> int:
        return int(round(t)) - t_min

    # Damage from attacks
    for a in attacks:
        idx = _sec_idx(a.hit_time)
        if 0 <= idx < n_steps:
            dmg = a.damage or 0.0
            atk_key = (a.attacker_camp, a.attacker_robot_id)
            vic_key = (a.victim_camp, a.victim_robot_id)
            if atk_key in robot_arrays:
                if a.victim_type == "基地":
                    robot_arrays[atk_key]["dmg_base"][idx] += dmg
                elif a.victim_type == "前哨站":
                    robot_arrays[atk_key]["dmg_outpost"][idx] += dmg
                else:
                    robot_arrays[atk_key]["dmg_robot"][idx] += dmg
            if vic_key in robot_arrays:
                robot_arrays[vic_key]["dmg_received"][idx] += dmg

    # Kills and assists
    for k in kill_events:
        idx = _sec_idx(k.time)
        if 0 <= idx < n_steps:
            killer_key = (k.killer_camp, k.killer_id)
            if killer_key in robot_arrays:
                robot_arrays[killer_key]["kills"][idx] += 1
                if k.victim_type == "英雄":
                    robot_arrays[killer_key]["hero_kill"][idx] += 1
                if k.victim_type == "哨兵":
                    robot_arrays[killer_key]["sentry_kill"][idx] += 1
            for acamp, aid, _ in k.assist_attackers:
                akey = (acamp, aid)
                if akey in robot_arrays:
                    robot_arrays[akey]["assists"][idx] += 1

    # Deaths
    for d in death_events:
        idx = _sec_idx(d.time)
        if 0 <= idx < n_steps:
            dkey = (d.camp, d.robot_id)
            if dkey in robot_arrays:
                robot_arrays[dkey]["deaths"][idx] += 1

    # Buff / terrain intervals
    for bi in buff_intervals:
        if bi.category not in TERRAIN_BUFFS:
            continue
        key = (bi.camp, bi.robot_id)
        if key not in robot_arrays:
            continue
        start_idx = max(0, _sec_idx(bi.start))
        end_idx = min(n_steps - 1, _sec_idx(bi.end))
        if start_idx <= end_idx:
            if bi.category in {"飞坡", "台阶跨越"}:
                if bi.category == "飞坡":
                    robot_arrays[key]["ramp_count"][start_idx] += 1
                else:
                    robot_arrays[key]["step_count"][start_idx] += 1
            else:
                robot_arrays[key]["highland_sec"][start_idx : end_idx + 1] += 1.0

    # ---- bin rune activations --------------------------------------------
    rune_acts = _detect_rune_activations(events, mobile_tracks)
    for ra in rune_acts:
        idx = _sec_idx(ra.time)
        if 0 <= idx < n_steps:
            key = (ra.camp, ra.robot_id)
            if key in robot_arrays:
                robot_arrays[key]["rune_score"][idx] += ra.score

    # ---- compute infantry kill streaks over time --------------------------
    for key in robot_keys:
        arr = robot_arrays[key]
        if arr["type"] not in {"步兵3", "步兵4"}:
            continue
        camp, rid = key
        my_kills = sorted(
            [k for k in kill_events if k.killer_camp == camp and k.killer_id == rid],
            key=lambda k: k.time,
        )
        my_death_times = sorted(
            [d.time for d in death_events if d.camp == camp and d.robot_id == rid]
        )
        streak = 0
        prev_kill_time = -999.0
        death_ptr = 0
        for k in my_kills:
            # check if death occurred since last kill
            died = False
            while death_ptr < len(my_death_times) and my_death_times[death_ptr] <= k.time:
                if my_death_times[death_ptr] > prev_kill_time:
                    died = True
                death_ptr += 1
            if died:
                streak = 1
            else:
                streak += 1
            prev_kill_time = k.time
            if streak > 1:
                idx = _sec_idx(k.time)
                if 0 <= idx < n_steps:
                    robot_arrays[key]["streak_bonus"][idx] += 1

    # ---- compute cumulative scores ----------------------------------------
    result: dict[str, list[dict[str, Any]]] = {}
    for camp in camps:
        result[camp] = []

    for key in robot_keys:
        camp, rid = key
        arr = robot_arrays[key]
        rtype = arr["type"]

        # Role-specific weights
        if rtype == "英雄":
            kw, dw = SCORE_HERO_KILL, SCORE_HERO_DEATH
        elif rtype == "哨兵":
            kw, dw = SCORE_SENTRY_KILL, SCORE_SENTRY_DEATH
        elif rtype == "工程":
            kw, dw = SCORE_KILL, SCORE_ENGINEER_DEATH
        elif rtype == "空中":
            kw, dw = SCORE_AERIAL_KILL, SCORE_AERIAL_DEATH
        else:
            kw, dw = SCORE_KILL, SCORE_DEATH

        is_winner = camp == winner
        is_fb = first_blood is not None and first_blood.killer_camp == camp and first_blood.killer_id == rid

        # Build cumulative score array
        score = np.full(n_steps, STARTING_SCORE)

        # Step events (cumulative sum of deltas)
        score += np.cumsum(arr["kills"] * kw)
        score += np.cumsum(arr["deaths"] * dw)
        score += np.cumsum(arr["assists"] * SCORE_ASSIST)

        # Damage (cumulative)
        score += np.cumsum(arr["dmg_robot"]) / 100.0 * SCORE_DAMAGE_PER_100HP
        # Base damage: apply multiplier based on per-second HP gap
        my_bhp = base_hp_arrays.get(camp, np.full(n_steps, np.nan))
        enemy_bhp = base_hp_arrays.get("红" if camp == "蓝" else "蓝", np.full(n_steps, np.nan))
        gap_arr = my_bhp - enemy_bhp
        gap_ok = np.isfinite(gap_arr)
        base_m = np.ones(n_steps)
        base_m[gap_ok & (np.abs(gap_arr) <= 300)] = 2.0
        base_m[gap_ok & (gap_arr > 300) & (gap_arr <= 1000)] = 1.5
        score += np.cumsum(arr["dmg_base"] * base_m) / 100.0 * SCORE_STRUCTURE_DAMAGE_PER_100HP
        score += np.cumsum(arr["dmg_outpost"]) / 100.0 * SCORE_STRUCTURE_DAMAGE_PER_100HP
        score += np.cumsum(arr["dmg_received"]) / 100.0 * SCORE_DAMAGE_RECEIVED_PER_100HP

        # Terrain
        score += np.cumsum(arr["ramp_count"] * SCORE_RAMP)
        score += np.cumsum(arr["step_count"] * SCORE_STEP)
        score += np.cumsum(arr["highland_sec"]) * SCORE_HIGHLAND_PER_SEC

        # Pressure: enemy-half time (universal)
        score += np.cumsum(arr["enemy_half"].astype(float)) * SCORE_PRESSURE_PER_SEC

        # Engineer terrain multiplier
        if rtype == "工程":
            # Subtract original terrain contributions and add multiplied ones
            terrain_orig = (
                np.cumsum(arr["ramp_count"] * SCORE_RAMP)
                + np.cumsum(arr["step_count"] * SCORE_STEP)
                + np.cumsum(arr["highland_sec"]) * SCORE_HIGHLAND_PER_SEC
            )
            score += terrain_orig * (ENGINEER_TERRAIN_MULTIPLIER - 1.0)

        # Role-specific
        if rtype == "英雄":
            score += arr["shots_42"] * SCORE_HERO_42MM_SHOT
            score += np.maximum.accumulate(arr["spent"]) * SCORE_HERO_GOLD_SPENT_PER_COIN
        elif rtype in {"步兵3", "步兵4"}:
            score += np.cumsum(arr["hero_kill"] * SCORE_INFANTRY_KILL_HERO_BONUS)
            score += np.cumsum(arr["sentry_kill"] * SCORE_INFANTRY_KILL_SENTRY_BONUS)
            score += np.cumsum(arr["streak_bonus"] * SCORE_INFANTRY_KILL_STREAK_PER_KILL)
        elif rtype == "哨兵":
            score += np.cumsum(arr["alive"].astype(float)) * SCORE_SENTRY_SURVIVAL_PER_SEC
        elif rtype == "工程":
            score += np.cumsum(arr["alive"].astype(float)) * SCORE_ENGINEER_SURVIVAL_PER_SEC
        elif rtype == "空中":
            score += np.cumsum(arr["enemy_half"].astype(float)) * SCORE_AERIAL_ENEMY_HALF_PER_SEC
            # Zero-centred rolling-10s efficiency
            cum_dmg = np.cumsum(arr["dmg_robot"] + arr["dmg_base"] + arr["dmg_outpost"])
            w = AERIAL_ROLLING_WINDOW
            roll_dmg = cum_dmg.copy()
            roll_dmg[w:] = cum_dmg[w:] - cum_dmg[:-w]
            roll_time = np.minimum(float(w), np.arange(1, n_steps + 1, dtype=float))
            roll_dps = roll_dmg / np.maximum(1.0, roll_time)
            import math
            k = AERIAL_LOGISTIC_STEEPNESS
            d0 = AERIAL_LOGISTIC_MIDPOINT
            rate = AERIAL_LOGISTIC_MAX_RATE * (
                2.0 / (1.0 + np.exp(-k * (roll_dps - d0))) - 1.0
            )
            score += np.cumsum(rate)

        # Win bonus: ramped in over the final 20% of match time
        if is_winner:
            ramp_start = max(0, n_steps - max(30, n_steps // 5))
            if ramp_start < n_steps:
                ramp = np.zeros(n_steps)
                ramp[ramp_start:] = SCORE_WIN * np.linspace(0, 1, n_steps - ramp_start)
                score += ramp

        # First blood
        if is_fb:
            fb_idx = _sec_idx(first_blood.time)  # type: ignore[union-attr]
            if 0 <= fb_idx < n_steps:
                score[fb_idx:] += SCORE_FIRST_BLOOD

        # Rune activations (cumulative, discrete events)
        score += np.cumsum(arr["rune_score"])

        # AFK: applied from the second death onward (not retroactive to t=0)
        total_kills = arr["kills"].sum()
        total_assists = arr["assists"].sum()
        total_dmg = arr["dmg_robot"].sum() + arr["dmg_base"].sum() + arr["dmg_outpost"].sum()
        total_deaths = arr["deaths"].sum()
        is_afk = rtype != "工程" and total_kills == 0 and total_assists == 0 and total_dmg == 0 and total_deaths >= 2
        if is_afk:
            my_death_times = sorted(
                d.time for d in death_events if d.camp == camp and d.robot_id == rid
            )
            if len(my_death_times) >= 2:
                afk_idx = _sec_idx(my_death_times[1])  # confirmed AFK at second death
                if 0 <= afk_idx < n_steps:
                    score[afk_idx:] += SCORE_AFK_PENALTY

        score = np.clip(score, MIN_SCORE, MAX_SCORE)
        scale = TYPE_SCALING.get(rtype, 1.0)
        score = STARTING_SCORE + (score - STARTING_SCORE) * scale
        score = np.clip(score, MIN_SCORE, MAX_SCORE)

        result[camp].append({
            "camp": camp,
            "robot_id": rid,
            "robot_type": rtype,
            "times": time_grid.tolist(),
            "score": [round(float(s), 4) for s in score],
            "kills": [int(x) for x in np.cumsum(arr["kills"])],
            "assists": [int(x) for x in np.cumsum(arr["assists"])],
            "deaths": [int(x) for x in np.cumsum(arr["deaths"])],
            "damage": [round(float(x), 1) for x in np.cumsum(arr["dmg_robot"] + arr["dmg_base"] + arr["dmg_outpost"])],
        })

    # Sort within each camp by type rank
    type_rank = {"英雄": 0, "工程": 1, "步兵3": 2, "步兵4": 3, "哨兵": 4, "空中": 5}
    for camp in result:
        result[camp].sort(key=lambda d: type_rank.get(d["robot_type"], 99))

    return result
