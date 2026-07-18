from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path

import numpy as np

from rmuc_trajectory.combat import infer_attacks
from rmuc_trajectory.buffs import build_buff_intervals
from rmuc_trajectory.dart import infer_dart_impacts
from rmuc_trajectory.field import FieldCanvas
from rmuc_trajectory.revival import infer_paid_revivals, infer_respawn_intervals
from rmuc_trajectory.pipeline import (
    MatchEvent,
    TrackKey,
    clean_track,
    event_alignment_summary,
    load_events,
)


class FakeRow(dict):
    pass


class TrajectoryCleaningTests(unittest.TestCase):
    def test_default_canvas_uses_inner_field_not_outer_baffles(self) -> None:
        canvas = FieldCanvas(Path("field.jpeg"))
        self.assertEqual(canvas.crop_pixels(1683, 938), (100, 69, 1576, 856))

    def test_health_and_heading_are_carried_into_clean_frames(self) -> None:
        rows = [
            FakeRow(时刻秒=0, x=1.0, y=1.0, 当前血量=200.0, 最大血量=400.0, 枪口朝向=170.0, 小热量=20, 小热量上限=100, 队伍总金币=10, 队伍剩余金币=10, 是否易伤=0),
            FakeRow(时刻秒=1, x=2.0, y=1.0, 当前血量=100.0, 最大血量=400.0, 枪口朝向=None, 小热量=50, 小热量上限=100, 队伍总金币=20, 队伍剩余金币=5, 是否易伤=1),
            FakeRow(时刻秒=2, x=3.0, y=1.0, 当前血量=0.0, 最大血量=400.0, 枪口朝向=-170.0, 小热量=0, 小热量上限=100, 队伍总金币=30, 队伍剩余金币=15, 是否易伤=0),
        ]
        track = clean_track(
            TrackKey("红", 1, "英雄", "测试大学"), rows,  # type: ignore[arg-type]
            start=0, end=2, max_gap=1, smooth_window=1, max_speed_mps=8.0,
        )
        np.testing.assert_allclose(track.health, [200.0, 100.0, 0.0])
        np.testing.assert_allclose(track.small_heat, [20.0, 50.0, 0.0])
        np.testing.assert_allclose(track.total_coins, [10.0, 20.0, 30.0])
        self.assertEqual(track.to_jsonable()["vulnerable"], [False, True, False])
        self.assertAlmostEqual(abs(track.heading_deg[1]), 180.0, places=5)

    def test_static_objective_uses_canvas_calibrated_position(self) -> None:
        rows = [FakeRow(时刻秒=0, x=0.0, y=0.0), FakeRow(时刻秒=1, x=0.0, y=0.0)]
        track = clean_track(
            TrackKey("蓝", 111, "前哨站", "测试大学"), rows,  # type: ignore[arg-type]
            start=0, end=1, max_gap=1, smooth_window=1, max_speed_mps=8.0,
        )
        np.testing.assert_allclose(track.clean_xy, [[17.12, 11.32], [17.12, 11.32]])

    def test_rule_buff_intervals_use_duration_and_death_clear(self) -> None:
        rows = [
            FakeRow(时刻秒=second, x=1.0, y=1.0, 当前血量=0 if second == 20 else 100, 最大血量=100)
            for second in range(0, 201)
        ]
        track = clean_track(
            TrackKey("红", 3, "步兵3", "测试大学"), rows,  # type: ignore[arg-type]
            start=0, end=200, max_gap=1, smooth_window=1, max_speed_mps=8.0,
        )
        events = [
            MatchEvent(5, "增益", 3, "步兵3", "红", "测试大学", None, None, "飞坡", None, None),
            MatchEvent(50, "增益", 3, "步兵3", "红", "测试大学", None, None, "小能量机关增益", None, None),
            MatchEvent(100, "能量机关", None, None, "红", "测试大学", None, None, "rune_type=0.0", 9, "arm_cnt=8.0,avg_round=8.5"),
            MatchEvent(100, "增益", 3, "步兵3", "红", "测试大学", None, None, "大能量机关增益", None, None),
        ]
        intervals, summary = build_buff_intervals(events, [track], 200)
        by_category = {item.category: item for item in intervals}
        self.assertEqual(by_category["飞坡"].end, 20)
        self.assertEqual(by_category["飞坡"].end_reason, "机器人战亡清除")
        self.assertEqual(by_category["小能量机关增益"].end, 95)
        self.assertEqual(by_category["大能量机关增益"].end, 145)
        self.assertFalse(summary["has_dataset_end_flag"])

    def test_infers_full_health_revival_with_coin_drop(self) -> None:
        rows = [
            FakeRow(时刻秒=0, x=1.0, y=1.0, 当前血量=100, 最大血量=100, 队伍剩余金币=500),
            FakeRow(时刻秒=1, x=1.0, y=1.0, 当前血量=0, 最大血量=100, 队伍剩余金币=500),
            FakeRow(时刻秒=2, x=1.0, y=1.0, 当前血量=100, 最大血量=100, 队伍剩余金币=300),
        ]
        track = clean_track(
            TrackKey("红", 4, "步兵4", "测试大学"), rows,  # type: ignore[arg-type]
            start=0, end=2, max_gap=1, smooth_window=1, max_speed_mps=8.0,
        )
        revivals, summary = infer_paid_revivals([track])
        self.assertEqual(len(revivals), 1)
        self.assertEqual(revivals[0].time, 2)
        self.assertEqual(revivals[0].observed_coin_drop, 200)
        self.assertEqual(revivals[0].confidence, "high")
        self.assertFalse(summary["has_native_buyback_event"])
        intervals, interval_summary = infer_respawn_intervals([track], revivals, 2)
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0].method, "paid")
        self.assertEqual(intervals[0].required_progress, 10)
        self.assertEqual(intervals[0].observed_duration, 1)
        self.assertEqual(interval_summary["paid_respawn_count"], 1)

    def test_maps_dart_damage_to_target_and_blind_duration(self) -> None:
        objective = clean_track(
            TrackKey("蓝", 110, "基地", "乙校"),
            [FakeRow(时刻秒=10, x=0, y=0, 当前血量=5000, 最大血量=5000)],  # type: ignore[arg-type]
            start=10, end=10, max_gap=1, smooth_window=1, max_speed_mps=8,
        )
        events = [
            MatchEvent(10, "飞镖命中", None, None, "红", "甲校", 110, "基地", "飞镖命中", 300, None)
        ]
        impacts, summary = infer_dart_impacts(events, [objective])
        self.assertEqual(impacts[0].target_profile, "随机固定目标")
        self.assertEqual(impacts[0].blind_duration, 10)
        self.assertEqual(impacts[0].buff_suppression_duration, 10)
        self.assertEqual(summary["identified_target_profile_count"], 1)

    def test_fixed_dart_blind_refreshes_without_shortening(self) -> None:
        objective = clean_track(
            TrackKey("蓝", 110, "基地", "乙校"),
            [FakeRow(时刻秒=0, x=0, y=0, 当前血量=5000, 最大血量=5000)],  # type: ignore[arg-type]
            start=0, end=0, max_gap=1, smooth_window=1, max_speed_mps=8,
        )
        events = [
            MatchEvent(0, "飞镖命中", None, None, "红", "甲校", 110, "基地", "飞镖命中", 200, None),
            MatchEvent(2, "飞镖命中", None, None, "红", "甲校", 110, "基地", "飞镖命中", 200, None),
            MatchEvent(8, "飞镖命中", None, None, "红", "甲校", 110, "基地", "飞镖命中", 200, None),
        ]
        impacts, _ = infer_dart_impacts(events, [objective])
        self.assertEqual([item.blind_duration for item in impacts], [10, 5, 3])
        self.assertEqual([item.blind_end for item in impacts], [10, 10, 11])

    def test_rejects_bounds_and_single_frame_spike_then_interpolates(self) -> None:
        rows = [
            FakeRow(时刻秒=0, x=1.0, y=1.0),
            FakeRow(时刻秒=1, x=2.0, y=1.0),
            FakeRow(时刻秒=2, x=20.0, y=14.0),
            FakeRow(时刻秒=3, x=4.0, y=1.0),
            FakeRow(时刻秒=4, x=40.0, y=1.0),
            FakeRow(时刻秒=5, x=6.0, y=1.0),
        ]
        track = clean_track(
            TrackKey("红", 1, "英雄", "测试大学"),
            rows,  # type: ignore[arg-type]
            start=0,
            end=5,
            max_gap=1,
            smooth_window=1,
            max_speed_mps=8.0,
        )
        self.assertEqual(track.diagnostics.out_of_bounds, 1)
        self.assertEqual(track.diagnostics.spike_outliers, 1)
        self.assertEqual(track.diagnostics.interpolated_frames, 2)
        np.testing.assert_allclose(track.clean_xy[2], [3.0, 1.0])
        np.testing.assert_allclose(track.clean_xy[4], [5.0, 1.0])
        self.assertEqual(track.diagnostics.residual_jump_count, 0)

    def test_load_events_preserves_timeline_fields(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.execute(
            """
            CREATE TABLE events(
              game_id INT, 时刻秒 REAL, 事件类型 TEXT, robot_id INT,
              机器人类型 TEXT, 阵营 TEXT, 学校名 TEXT, 目标robot_id INT,
              目标类型 TEXT, 类别 TEXT, 数值 REAL, 备注 TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO events VALUES(1, 12.5, '受击', 3, '步兵3', '红', '测试大学', NULL, NULL, '17mm', -20, NULL)"
        )
        events = load_events(connection, 1)
        connection.close()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].time, 12.5)
        self.assertEqual(events[0].event_type, "受击")
        self.assertEqual(events[0].value, -20.0)

    def test_event_alignment_counts_positionable_events(self) -> None:
        rows = [FakeRow(时刻秒=1, x=1.0, y=1.0), FakeRow(时刻秒=2, x=2.0, y=1.0)]
        track = clean_track(
            TrackKey("红", 3, "步兵3", "测试大学"),
            rows,  # type: ignore[arg-type]
            start=1,
            end=2,
            max_gap=1,
            smooth_window=1,
            max_speed_mps=8.0,
        )
        events = [
            MatchEvent(1.2, "受击", 3, "步兵3", "红", "测试大学", None, None, "17mm", -20, None),
            MatchEvent(1.0, "装配成功", None, None, "红", "测试大学", None, None, "等级1", 10, None),
        ]
        summary = event_alignment_summary(events, [track])
        self.assertEqual(summary["event_count"], 2)
        self.assertEqual(summary["events_with_subject_robot"], 1)
        self.assertEqual(summary["events_positioned_on_track"], 1)

    def test_infers_aligned_recent_enemy_shot_as_attack(self) -> None:
        attacker_rows = [
            FakeRow(时刻秒=10, x=1.0, y=1.0, 当前血量=400, 最大血量=400, 枪口朝向=0.0),
            FakeRow(时刻秒=11, x=1.0, y=1.0, 当前血量=400, 最大血量=400, 枪口朝向=0.0),
        ]
        victim_rows = [
            FakeRow(时刻秒=10, x=5.0, y=1.0, 当前血量=400, 最大血量=400, 枪口朝向=180.0),
            FakeRow(时刻秒=11, x=5.0, y=1.0, 当前血量=380, 最大血量=400, 枪口朝向=180.0),
        ]
        tracks = [
            clean_track(TrackKey("红", 3, "步兵3", "甲校"), attacker_rows, start=10, end=11, max_gap=1, smooth_window=1, max_speed_mps=8),  # type: ignore[arg-type]
            clean_track(TrackKey("蓝", 4, "步兵4", "乙校"), victim_rows, start=10, end=11, max_gap=1, smooth_window=1, max_speed_mps=8),  # type: ignore[arg-type]
        ]
        events = [
            MatchEvent(10.0, "发弹", 3, "步兵3", "红", "甲校", None, None, "17mm", 1, None),
            MatchEvent(11.0, "受击", 4, "步兵4", "蓝", "乙校", None, None, "17mm", -20, None),
        ]
        attacks, summary = infer_attacks(events, tracks)
        self.assertEqual(len(attacks), 1)
        self.assertEqual(attacks[0].attacker_robot_id, 3)
        self.assertEqual(attacks[0].victim_robot_id, 4)
        self.assertEqual(attacks[0].confidence, "high")
        self.assertEqual(summary["unresolved"], 0)

    def test_infers_deployed_hero_base_lob_from_power_position_and_damage(self) -> None:
        hero_rows = [
            FakeRow(时刻秒=t, x=4.7, y=11.5, 当前血量=400, 最大血量=400, 枪口朝向=-11, 底盘功率=0)
            for t in (8, 9, 10, 11)
        ]
        base_rows = [
            FakeRow(时刻秒=t, x=0, y=0, 当前血量=5000 if t < 11 else 4700, 最大血量=5000, 枪口朝向=180)
            for t in (8, 9, 10, 11)
        ]
        tracks = [
            clean_track(TrackKey("红", 1, "英雄", "甲校"), hero_rows, start=8, end=11, max_gap=1, smooth_window=1, max_speed_mps=8),  # type: ignore[arg-type]
            clean_track(TrackKey("蓝", 110, "基地", "乙校"), base_rows, start=8, end=11, max_gap=1, smooth_window=1, max_speed_mps=8),  # type: ignore[arg-type]
        ]
        events = [
            MatchEvent(10, "发弹", 1, "英雄", "红", "甲校", None, None, "42mm", None, None),
            MatchEvent(11, "受击", 110, "基地", "蓝", "乙校", None, None, "42mm", -300, None),
        ]
        attacks, _ = infer_attacks(events, tracks)
        self.assertEqual(attacks[0].special_mode, "hero_deployed_lob")
        self.assertEqual(attacks[0].damage, 300)


if __name__ == "__main__":
    unittest.main()
