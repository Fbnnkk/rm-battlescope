# -*- coding: utf-8 -*-
"""Regression tests for the explicitly retained legacy v2 scoring rules."""

from __future__ import annotations

import unittest

from rmuc_trajectory.buffs import BuffInterval
from rmuc_trajectory.combat import AttackInference
from rmuc_trajectory.pipeline import TrackKey, clean_track
from rmuc_trajectory.legacy_scoring import (
    MIN_SCORE,
    MAX_SCORE,
    SCORE_AFK_PENALTY,
    SCORE_AERIAL_DEATH,
    SCORE_AERIAL_ENEMY_HALF_PER_SEC,
    SCORE_AERIAL_KILL,
    SCORE_ASSIST,
    SCORE_DAMAGE_PER_100HP,
    SCORE_DAMAGE_RECEIVED_PER_100HP,
    SCORE_ENGINEER_DEATH,
    SCORE_ENGINEER_SURVIVAL_PER_SEC,
    SCORE_FIRST_BLOOD,
    SCORE_HERO_42MM_SHOT,
    SCORE_HERO_DEATH,
    SCORE_HERO_GOLD_SPENT_PER_COIN,
    SCORE_HERO_KILL,
    SCORE_HIGHLAND_PER_SEC,
    SCORE_INFANTRY_KILL_HERO_BONUS,
    SCORE_INFANTRY_KILL_SENTRY_BONUS,
    SCORE_INFANTRY_KILL_STREAK_PER_KILL,
    SCORE_RAMP,
    SCORE_SENTRY_DEATH,
    SCORE_SENTRY_KILL,
    SCORE_SENTRY_SURVIVAL_PER_SEC,
    SCORE_STRUCTURE_DAMAGE_PER_100HP,
    SCORE_WIN,
    ENGINEER_TERRAIN_MULTIPLIER,
    RADAR_JAM_DURATION_S,
    compute_scores,
)

# Chinese string constants (avoid encoding issues in source)
RED = "红"          # 红
BLUE = "蓝"         # 蓝
HERO = "英雄"   # 英雄
ENGINEER = "工程"  # 工程
INF3 = "步兵3"     # 步兵3
INF4 = "步兵4"     # 步兵4
SENTRY = "哨兵"    # 哨兵
AERIAL = "空中"    # 空中
OUTPOST = "前哨站"  # 前哨站
BASE = "基地"            # 基地
SCHOOL = "测试大学"  # 测试大学
WINNER_KEY = "胜方"     # 胜方

# timeseries Chinese column names (used as dict keys)
TS_TIME = "时刻秒"       # 时刻秒
TS_HP = "当前血量"   # 当前血量
TS_MAX_HP = "最大血量"  # 最大血量
TS_HEADING = "枪口朝向"  # 枪口朝向
TS_CHASSIS = "底盘功率"  # 底盘功率
TS_SMALL_HEAT = "小热量"     # 小热量
TS_SMALL_HEAT_LIMIT = "小热量上限"  # 小热量上限
TS_LARGE_HEAT = "大热量"     # 大热量
TS_LARGE_HEAT_LIMIT = "大热量上限"  # 大热量上限
TS_C17 = "累计17mm发弹"   # 累计17mm发弹
TS_C42 = "累计42mm发弹"   # 累计42mm发弹
TS_TOTAL_COINS = "队伍总金币"    # 队伍总金币
TS_REMAIN_COINS = "队伍剩余金币"  # 队伍剩余金币
TS_VULNERABLE = "是否易伤"  # 是否易伤


class FakeRow(dict):
    pass


def _row(t, x_val, hp, cum_42mm=0.0, total_coins=500.0, remain_coins=500.0):
    """Build a FakeRow with proper Chinese column keys."""
    return FakeRow({
        TS_TIME: t,
        "x": x_val, "y": 7.5, "z": 0.0,
        TS_HP: hp,
        TS_MAX_HP: 400.0,
        TS_HEADING: 0.0,
        TS_CHASSIS: 0.0,
        TS_SMALL_HEAT: 0.0,
        TS_SMALL_HEAT_LIMIT: 100.0,
        TS_LARGE_HEAT: 0.0,
        TS_LARGE_HEAT_LIMIT: 200.0,
        TS_C17: 0.0,
        TS_C42: cum_42mm,
        TS_TOTAL_COINS: total_coins,
        TS_REMAIN_COINS: remain_coins,
        TS_VULNERABLE: 0,
    })


def _track(
    camp=RED,
    robot_id=1,
    robot_type=HERO,
    school=SCHOOL,
    times=None,
    health=None,
    x_vals=None,
    cum_42mm=None,
    total_coins=None,
    remain_coins=None,
):
    """Build a minimal CleanTrack for scoring tests."""
    n = len(health) if health else (len(times) if times else 3)
    if times is None:
        times = list(range(n))
    if health is None:
        health = [100.0] * n
    if x_vals is None:
        x_vals = [1.0] * n
    if cum_42mm is None:
        cum_42mm = [0.0] * n
    if total_coins is None:
        total_coins = [500.0] * n
    if remain_coins is None:
        remain_coins = [500.0] * n

    rows = [
        _row(t, x_vals[i], health[i], cum_42mm[i], total_coins[i], remain_coins[i])
        for i, t in enumerate(times)
    ]
    return clean_track(
        TrackKey(camp, robot_id, robot_type, school),
        rows,  # type: ignore[arg-type]
        start=times[0],
        end=times[-1],
        max_gap=1,
        smooth_window=1,
        max_speed_mps=8.0,
    )


def _attack(
    attacker_camp=RED,
    attacker_id=1,
    attacker_type=HERO,
    victim_camp=BLUE,
    victim_id=2,
    victim_type=INF3,
    hit_time=1.0,
    damage=50.0,
    caliber="17mm",
    shot_time=None,
):
    """Build an AttackInference."""
    if shot_time is None:
        shot_time = hit_time - 0.2
    return AttackInference(
        attacker_robot_id=attacker_id,
        attacker_type=attacker_type,
        attacker_camp=attacker_camp,
        victim_robot_id=victim_id,
        victim_type=victim_type,
        victim_camp=victim_camp,
        caliber=caliber,
        shot_time=shot_time,
        hit_time=hit_time,
        attacker_xy=(1.0, 7.5),
        victim_xy=(2.0, 7.5),
        heading_deg=0.0,
        angle_error_deg=5.0,
        distance_m=1.0,
        delay_s=0.2,
        burst_count=1,
        continuous_count=1,
        score=0.8,
        confidence="high",
        ambiguous=False,
        damage=damage,
    )


# ---------------------------------------------------------------------------
# Universal tests
# ---------------------------------------------------------------------------


class UniversalScoringTests(unittest.TestCase):
    def test_base_score_is_five(self):
        track = _track()
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], [], [])
        # Delta from 5.0 is 0, so scaling does not affect the baseline
        self.assertAlmostEqual(scores[0]["total_score"], 5.0, places=2)

    def test_win_bonus(self):
        track = _track()
        match = {WINNER_KEY: RED}
        scores, _ = compute_scores(match, [track], [], [], [])
        # Only the +0.3 win bonus is scaled; baseline 5.0 stays intact
        expected = 5.0 + SCORE_WIN * 1.015
        self.assertAlmostEqual(scores[0]["total_score"], expected, places=2)

    def test_kill_detection(self):
        t_victim = _track(camp=BLUE, robot_id=2, robot_type=INF3, health=[100, 50, 0])
        t_killer = _track()
        attacks = [_attack(hit_time=2.1, damage=50.0)]
        match = {WINNER_KEY: BLUE}
        scores, summary = compute_scores(match, [t_killer, t_victim], [], attacks, [])
        killer = [s for s in scores if s["robot_id"] == 1][0]
        self.assertEqual(killer["components"]["kills"]["value"], 1)
        self.assertAlmostEqual(killer["components"]["kills"]["score"], SCORE_HERO_KILL)
        self.assertEqual(summary["kill_count"], 1)

    def test_kill_window_too_far(self):
        t_victim = _track(camp=BLUE, robot_id=2, robot_type=INF3,
                          health=[100] * 20 + [0])
        t_killer = _track()
        attacks = [_attack(hit_time=5.0, damage=50.0)]
        match = {WINNER_KEY: BLUE}
        scores, summary = compute_scores(match, [t_killer, t_victim], [], attacks, [])
        killer = [s for s in scores if s["robot_id"] == 1][0]
        self.assertEqual(killer["components"]["kills"]["value"], 0)
        self.assertEqual(summary["kill_count"], 0)

    def test_assist_attribution(self):
        health_vals = [100] * 10 + [50, 0]
        t_victim = _track(camp=BLUE, robot_id=3, robot_type=INF4, health=health_vals)
        t_hero = _track()
        t_inf = _track(robot_id=2, robot_type=INF3)
        attacks = [
            _attack(attacker_id=2, attacker_type=INF3,
                    victim_id=3, hit_time=9.0, damage=40.0),
            _attack(victim_id=3, hit_time=10.8, damage=50.0),
        ]
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [t_hero, t_inf, t_victim], [], attacks, [])
        hero = [s for s in scores if s["robot_id"] == 1][0]
        inf = [s for s in scores if s["robot_id"] == 2][0]
        self.assertEqual(hero["components"]["kills"]["value"], 1)
        self.assertEqual(inf["components"]["assists"]["value"], 1)

    def test_death_counted(self):
        track = _track(health=[100, 0, 0])
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], [], [])
        self.assertEqual(scores[0]["components"]["deaths"]["value"], 1)
        self.assertAlmostEqual(scores[0]["components"]["deaths"]["score"], SCORE_HERO_DEATH)

    def test_multiple_deaths(self):
        track = _track(health=[100, 0, 50, 0])
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], [], [])
        self.assertEqual(scores[0]["components"]["deaths"]["value"], 2)

    def test_damage_to_robots(self):
        track = _track()
        attacks = [_attack(damage=300.0)]
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], attacks, [])
        expected = 300.0 / 100.0 * SCORE_DAMAGE_PER_100HP
        self.assertAlmostEqual(scores[0]["components"]["damage_to_robots"]["score"], expected)

    def test_damage_to_structures(self):
        track = _track()
        attacks = [_attack(victim_id=111, victim_type=OUTPOST, damage=500.0)]
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], attacks, [])
        expected = 500.0 / 100.0 * SCORE_STRUCTURE_DAMAGE_PER_100HP
        self.assertAlmostEqual(scores[0]["components"]["damage_to_outpost"]["score"], expected)
        self.assertAlmostEqual(scores[0]["components"]["damage_to_base"]["value"], 0)

    def test_damage_received(self):
        track = _track(camp=BLUE, robot_id=2, robot_type=INF3)
        attacks = [_attack(damage=200.0)]
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], attacks, [])
        expected = 200.0 / 100.0 * SCORE_DAMAGE_RECEIVED_PER_100HP
        self.assertAlmostEqual(scores[0]["components"]["damage_received"]["score"], expected)

    def test_terrain_bump(self):
        track = _track(times=list(range(50)))
        buffs = [
            BuffInterval(RED, 1, HERO, "飞坡", 5.0, 35.0, 30.0,
                         "规则计时结束", "V2.1.0"),
        ]
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], [], buffs)
        self.assertAlmostEqual(scores[0]["components"]["terrain_bumps"]["score"], SCORE_RAMP)
        self.assertEqual(scores[0]["components"]["terrain_bumps"]["value"], 1)

    def test_terrain_highland(self):
        track = _track(times=list(range(50)))
        buffs = [
            BuffInterval(RED, 1, HERO,
                         "过中央高地",
                         10.0, 40.0, 30.0,
                         "规则计时结束", "V2.1.0"),
        ]
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], [], buffs)
        expected = 30.0 * SCORE_HIGHLAND_PER_SEC
        self.assertAlmostEqual(scores[0]["components"]["terrain_highland"]["score"], expected)

    def test_first_blood(self):
        t1 = _track(health=[100] * 10)
        t2 = _track(camp=BLUE, robot_id=2, robot_type=INF3, health=[100] * 10)
        t3 = _track(robot_id=3, robot_type=INF3, health=[100] * 10 + [0])
        t_v1 = _track(camp=BLUE, robot_id=4, robot_type=INF4, health=[100] * 5 + [0])
        attacks = [
            _attack(victim_id=4, victim_type=INF4, hit_time=5.1, damage=100),
            _attack(attacker_camp=BLUE, attacker_id=2, attacker_type=INF3,
                    victim_camp=RED, victim_id=3, victim_type=INF3,
                    hit_time=10.1, damage=100),
        ]
        match = {WINNER_KEY: BLUE}
        scores, summary = compute_scores(match, [t1, t2, t3, t_v1], [], attacks, [])
        fb_hero = [s for s in scores if s["robot_id"] == 1][0]
        self.assertTrue(fb_hero["components"]["first_blood"]["value"])
        self.assertAlmostEqual(fb_hero["components"]["first_blood"]["score"], SCORE_FIRST_BLOOD)
        self.assertEqual(summary["first_blood_robot"]["robot_id"], 1)

    def test_afk_penalty(self):
        track = _track(health=[100, 0, 50, 0])
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], [], [])
        self.assertTrue(scores[0]["components"]["afk_penalty"]["value"])
        self.assertAlmostEqual(scores[0]["components"]["afk_penalty"]["score"], SCORE_AFK_PENALTY)

    def test_no_afk_if_has_damage(self):
        track = _track(health=[100, 0, 50, 0])
        attacks = [_attack(damage=10.0)]
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], attacks, [])
        self.assertFalse(scores[0]["components"]["afk_penalty"]["value"])

    def test_score_clamped_to_zero(self):
        track = _track(health=[100, 0, 50, 0, 30, 0, 10, 0])
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], [], [])
        self.assertGreaterEqual(scores[0]["total_score"], MIN_SCORE)

    def test_score_clamped_to_ten(self):
        track = _track()
        attacks = [_attack(damage=100.0, hit_time=t) for t in range(1, 101)]
        match = {WINNER_KEY: RED}
        scores, _ = compute_scores(match, [track], [], attacks, [])
        self.assertLessEqual(scores[0]["total_score"], MAX_SCORE)


# ---------------------------------------------------------------------------
# Hero tests
# ---------------------------------------------------------------------------


class HeroTests(unittest.TestCase):
    def test_hero_kill_override(self):
        t_victim = _track(camp=BLUE, robot_id=2, robot_type=INF3, health=[100, 0])
        t_hero = _track()
        attacks = [_attack(hit_time=1.1)]
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [t_hero, t_victim], [], attacks, [])
        hero = [s for s in scores if s["robot_type"] == HERO][0]
        self.assertAlmostEqual(hero["components"]["kills"]["score"], SCORE_HERO_KILL)

    def test_hero_death_override(self):
        track = _track(health=[100, 0])
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], [], [])
        self.assertAlmostEqual(scores[0]["components"]["deaths"]["score"], SCORE_HERO_DEATH)

    def test_hero_42mm_shots(self):
        track = _track(cum_42mm=[0.0, 5.0, 10.0, 10.0])
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], [], [])
        self.assertEqual(scores[0]["hero"]["42mm_shots"]["value"], 10)
        self.assertAlmostEqual(scores[0]["hero"]["42mm_shots"]["score"],
                               10 * SCORE_HERO_42MM_SHOT)

    def test_hero_gold_spent(self):
        track = _track(total_coins=[500, 600, 700], remain_coins=[500, 400, 300])
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], [], [])
        self.assertEqual(scores[0]["hero"]["gold_spent"]["value"], 400)
        self.assertAlmostEqual(scores[0]["hero"]["gold_spent"]["score"],
                               400 * SCORE_HERO_GOLD_SPENT_PER_COIN)


# ---------------------------------------------------------------------------
# Infantry tests
# ---------------------------------------------------------------------------


class InfantryTests(unittest.TestCase):
    def test_infantry_kill_hero_bonus(self):
        t_hero = _track(camp=BLUE, robot_id=1, health=[100, 0])
        t_inf = _track(robot_id=3, robot_type=INF3)
        attacks = [_attack(attacker_id=3, attacker_type=INF3,
                           victim_id=1, victim_type=HERO, hit_time=1.1)]
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [t_inf, t_hero], [], attacks, [])
        inf = [s for s in scores if s["robot_type"] == INF3][0]
        self.assertEqual(inf["infantry"]["hero_kills"]["value"], 1)
        self.assertAlmostEqual(inf["infantry"]["hero_kills"]["score"],
                               SCORE_INFANTRY_KILL_HERO_BONUS)

    def test_infantry_kill_sentry_bonus(self):
        t_sentry = _track(camp=BLUE, robot_id=7, robot_type=SENTRY, health=[100, 0])
        t_inf = _track(robot_id=4, robot_type=INF4)
        attacks = [_attack(attacker_id=4, attacker_type=INF4,
                           victim_id=7, victim_type=SENTRY, hit_time=1.1)]
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [t_inf, t_sentry], [], attacks, [])
        inf = [s for s in scores if s["robot_type"] == INF4][0]
        self.assertEqual(inf["infantry"]["sentry_kills"]["value"], 1)
        self.assertAlmostEqual(inf["infantry"]["sentry_kills"]["score"],
                               SCORE_INFANTRY_KILL_SENTRY_BONUS)

    def test_infantry_kill_same_type(self):
        """Infantry killing another infantry of the same type."""
        t_victim = _track(camp=BLUE, robot_id=2, robot_type=INF3, health=[100, 0])
        t_inf = _track(robot_type=INF3, health=[100] * 5)
        attacks = [_attack(attacker_type=INF3, victim_id=2, hit_time=1.1)]
        match = {WINNER_KEY: BLUE}
        scores, summary = compute_scores(match, [t_inf, t_victim], [], attacks, [])
        self.assertEqual(summary["kill_count"], 1)
        inf = [s for s in scores if s["camp"] == RED and s["robot_type"] == INF3][0]
        self.assertEqual(inf["components"]["kills"]["value"], 1)

    def test_kill_streak(self):
        t_v1 = _track(camp=BLUE, robot_id=2, robot_type=INF3, health=[100, 0])
        t_v2 = _track(camp=BLUE, robot_id=3, robot_type=INF3, health=[100, 100, 0])
        t_v3 = _track(camp=BLUE, robot_id=4, robot_type=INF3, health=[100]*3+[0])
        t_inf = _track(robot_type=INF3, health=[100] * 5)
        attacks = [
            _attack(attacker_type=INF3, victim_id=2, hit_time=1.1),
            _attack(attacker_type=INF3, victim_id=3, hit_time=2.1),
            _attack(attacker_type=INF3, victim_id=4, hit_time=3.1),
        ]
        match = {WINNER_KEY: BLUE}
        scores, summary = compute_scores(match, [t_inf, t_v1, t_v2, t_v3], [], attacks, [])
        self.assertEqual(summary["kill_count"], 3)
        inf = [s for s in scores if s["camp"] == RED and s["robot_type"] == INF3][0]
        self.assertEqual(inf["components"]["kills"]["value"], 3)
        self.assertEqual(inf["infantry"]["max_kill_streak"]["value"], 3)
        self.assertEqual(inf["infantry"]["max_kill_streak"]["bonus_kills"], 2)
        self.assertAlmostEqual(inf["infantry"]["max_kill_streak"]["score"],
                               2 * SCORE_INFANTRY_KILL_STREAK_PER_KILL)

    def test_streak_resets_on_death(self):
        t_inf = _track(robot_type=INF3, health=[100, 0, 50, 0])
        t_v1 = _track(camp=BLUE, robot_id=2, robot_type=INF3, health=[100, 0])
        t_v2 = _track(camp=BLUE, robot_id=3, robot_type=INF3, health=[100, 100, 0])
        attacks = [
            _attack(attacker_type=INF3, victim_id=2, hit_time=1.1),
            _attack(attacker_type=INF3, victim_id=3, hit_time=3.1),
        ]
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [t_inf, t_v1, t_v2], [], attacks, [])
        inf = [s for s in scores if s["robot_type"] == INF3][0]
        self.assertEqual(inf["infantry"]["max_kill_streak"]["bonus_kills"], 0)


# ---------------------------------------------------------------------------
# Sentry tests
# ---------------------------------------------------------------------------


class SentryTests(unittest.TestCase):
    def test_sentry_kill_override(self):
        t_victim = _track(camp=BLUE, robot_id=2, robot_type=INF3, health=[100, 0])
        t_sentry = _track(robot_id=7, robot_type=SENTRY)
        attacks = [_attack(attacker_id=7, attacker_type=SENTRY, hit_time=1.1)]
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [t_sentry, t_victim], [], attacks, [])
        sentry = [s for s in scores if s["robot_type"] == SENTRY][0]
        self.assertAlmostEqual(sentry["components"]["kills"]["score"], SCORE_SENTRY_KILL)

    def test_sentry_death_override(self):
        track = _track(robot_id=7, robot_type=SENTRY, health=[100, 0])
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], [], [])
        self.assertAlmostEqual(scores[0]["components"]["deaths"]["score"], SCORE_SENTRY_DEATH)

    def test_sentry_survival(self):
        track = _track(robot_id=7, robot_type=SENTRY, health=[100] * 300)
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], [], [])
        expected = 300 * SCORE_SENTRY_SURVIVAL_PER_SEC
        self.assertAlmostEqual(scores[0]["sentry"]["survival_duration"]["score"], expected)


# ---------------------------------------------------------------------------
# Engineer tests
# ---------------------------------------------------------------------------


class EngineerTests(unittest.TestCase):
    def test_engineer_death_override(self):
        track = _track(robot_id=5, robot_type=ENGINEER, health=[100, 0])
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], [], [])
        self.assertAlmostEqual(scores[0]["components"]["deaths"]["score"], SCORE_ENGINEER_DEATH)

    def test_engineer_survival(self):
        track = _track(robot_id=5, robot_type=ENGINEER, health=[100] * 100)
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], [], [])
        expected = 100 * SCORE_ENGINEER_SURVIVAL_PER_SEC
        self.assertAlmostEqual(scores[0]["engineer"]["survival_duration"]["score"], expected)

    def test_engineer_terrain_multiplier(self):
        track = _track(robot_id=5, robot_type=ENGINEER, times=list(range(50)))
        buffs = [
            BuffInterval(RED, 5, ENGINEER,
                         "飞坡", 5.0, 35.0, 30.0,
                         "规则计时结束", "V2.1.0"),
        ]
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], [], buffs)
        expected = SCORE_RAMP * ENGINEER_TERRAIN_MULTIPLIER
        self.assertAlmostEqual(scores[0]["components"]["terrain_bumps"]["score"], expected)
        self.assertEqual(scores[0]["engineer"]["terrain_multiplier"], ENGINEER_TERRAIN_MULTIPLIER)


# ---------------------------------------------------------------------------
# Aerial tests
# ---------------------------------------------------------------------------


class AerialTests(unittest.TestCase):
    def test_aerial_kill_override(self):
        t_victim = _track(camp=BLUE, robot_id=2, robot_type=INF3, health=[100, 0])
        t_aerial = _track(robot_id=8, robot_type=AERIAL)
        attacks = [_attack(attacker_id=8, attacker_type=AERIAL, hit_time=1.1)]
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [t_aerial, t_victim], [], attacks, [])
        aerial = [s for s in scores if s["robot_type"] == AERIAL][0]
        self.assertAlmostEqual(aerial["components"]["kills"]["score"], SCORE_AERIAL_KILL)

    def test_aerial_death_override(self):
        track = _track(robot_id=8, robot_type=AERIAL, health=[100, 0])
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], [], [])
        self.assertAlmostEqual(scores[0]["components"]["deaths"]["score"], SCORE_AERIAL_DEATH)

    def test_aerial_efficiency(self):
        track = _track(robot_id=8, robot_type=AERIAL, times=list(range(100)))
        attacks = [_attack(attacker_id=8, attacker_type=AERIAL,
                           hit_time=t, damage=10.0) for t in range(10, 60)]
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], attacks, [])
        a = scores[0]["aerial"]
        self.assertGreater(a["output_efficiency_dps"], 0)
        self.assertGreater(a["efficiency_score"], 0)

    def test_aerial_jammed_time(self):
        track = _track(robot_id=8, robot_type=AERIAL, times=list(range(100)))
        from rmuc_trajectory.pipeline import MatchEvent
        events = [
            MatchEvent(20, "雷达反制UAV", 8, AERIAL,
                       RED, SCHOOL,
                       None, None,
                       "被反制", None,
                       "反制方=蓝"),
        ]
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], events, [], [])
        a = scores[0]["aerial"]
        self.assertAlmostEqual(a["jammed_time"], RADAR_JAM_DURATION_S)

    def test_aerial_enemy_half(self):
        track = _track(camp=BLUE, robot_id=8, robot_type=AERIAL,
                       x_vals=[10.0] * 50, health=[100] * 50)
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], [], [])
        expected = 50 * SCORE_AERIAL_ENEMY_HALF_PER_SEC
        self.assertAlmostEqual(scores[0]["aerial"]["enemy_half_time"]["score"], expected)

    def test_aerial_not_on_enemy_half(self):
        track = _track(robot_id=8, robot_type=AERIAL,
                       x_vals=[10.0] * 50, health=[100] * 50)
        match = {WINNER_KEY: BLUE}
        scores, _ = compute_scores(match, [track], [], [], [])
        self.assertAlmostEqual(scores[0]["aerial"]["enemy_half_time"]["value"], 0)


if __name__ == "__main__":
    unittest.main()
