from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from rmuc_trajectory.webapp import search_matches, validate_job_payload


class WebApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Path(self.temp_dir.name) / "matches.sqlite"
        connection = sqlite3.connect(self.database)
        connection.execute(
            """
            CREATE TABLE matches(
              赛区 TEXT, 场次号 INT, 赛程 TEXT, 局号 INT, game_id INT,
              web_game_id INT, 红方学校 TEXT, 蓝方学校 TEXT, 胜方 TEXT,
              开始时间 TEXT, 时长秒 INT
            )
            """
        )
        connection.executemany(
            "INSERT INTO matches VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            [
                ("南部赛区", 2, "第2场", 1, 200, 2, "甲大学", "目标学院", "红", "2026-01-02", 420),
                ("南部赛区", 1, "第1场", 1, 100, 1, "目标学院", "乙大学", "蓝", "2026-01-01", 420),
                ("北部赛区", 3, "第3场", 1, 300, 3, "目标学院附中", "丙大学", "红", "2026-01-03", 420),
            ],
        )
        connection.commit()
        connection.close()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_matches_are_chronological_without_search(self) -> None:
        matches = search_matches(self.database)
        self.assertEqual([match["game_id"] for match in matches], [100, 200, 300])

    def test_search_ranks_exact_school_before_prefix_match(self) -> None:
        matches = search_matches(self.database, "目标学院")
        self.assertEqual([match["game_id"] for match in matches], [100, 200, 300])

    def test_job_payload_validation(self) -> None:
        request = validate_job_payload(
            {
                "game_id": "100",
                "start": "20",
                "end": "200",
                "max_gap": "3",
                "smooth_window": "5",
                "max_speed": "8",
            }
        )
        self.assertEqual(request.game_id, 100)
        self.assertEqual(request.smooth_window, 5)
        with self.assertRaisesRegex(ValueError, "奇数"):
            validate_job_payload({"game_id": 100, "smooth_window": 4})


if __name__ == "__main__":
    unittest.main()
