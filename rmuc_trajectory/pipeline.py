"""Read, validate, repair and diagnose continuous RMUC trajectory frames."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .field import (
    FIELD_HEIGHT_M,
    FIELD_WIDTH_M,
    INNER_FIELD_RECT_PX,
    OBJECTIVE_POSITIONS,
    SOURCE_IMAGE_SIZE,
)


GROUND_TYPES = {"英雄", "工程", "步兵3", "步兵4", "哨兵"}
DEFAULT_SPEED_MPS = {"空中": 15.0}
EXTRA_TELEMETRY = (
    ("z", "z"),
    ("chassis_power", "底盘功率"),
    ("small_heat", "小热量"),
    ("small_heat_limit", "小热量上限"),
    ("large_heat", "大热量"),
    ("large_heat_limit", "大热量上限"),
    ("cumulative_17mm", "累计17mm发弹"),
    ("cumulative_42mm", "累计42mm发弹"),
    ("total_coins", "队伍总金币"),
    ("remaining_coins", "队伍剩余金币"),
    ("vulnerable", "是否易伤"),
)


@dataclass(frozen=True, order=True)
class TrackKey:
    camp: str
    robot_id: int
    robot_type: str
    school: str

    @property
    def label(self) -> str:
        return f"{self.camp}方 {self.robot_type} #{self.robot_id}"


@dataclass
class TrackDiagnostics:
    label: str
    raw_rows: int
    frame_count: int
    missing_frames: int
    duplicate_rows: int
    null_coordinates: int
    out_of_bounds: int
    spike_outliers: int
    interpolated_frames: int
    remaining_frames: int
    residual_jump_count: int
    raw_path_m: float
    clean_path_m: float
    max_raw_speed_mps: float
    max_clean_speed_mps: float


@dataclass
class CleanTrack:
    key: TrackKey
    times: np.ndarray
    raw_xy: np.ndarray
    clean_xy: np.ndarray
    health: np.ndarray
    max_health: np.ndarray
    heading_deg: np.ndarray
    z: np.ndarray
    chassis_power: np.ndarray
    small_heat: np.ndarray
    small_heat_limit: np.ndarray
    large_heat: np.ndarray
    large_heat_limit: np.ndarray
    cumulative_17mm: np.ndarray
    cumulative_42mm: np.ndarray
    total_coins: np.ndarray
    remaining_coins: np.ndarray
    vulnerable: np.ndarray
    observed_mask: np.ndarray
    interpolated_mask: np.ndarray
    continuity_mask: np.ndarray
    diagnostics: TrackDiagnostics

    def to_jsonable(self, decimals: int = 3) -> dict[str, Any]:
        def point(value: np.ndarray) -> list[float] | None:
            if not np.isfinite(value).all():
                return None
            return [round(float(value[0]), decimals), round(float(value[1]), decimals)]

        return {
            "key": asdict(self.key),
            "label": self.key.label,
            "times": [float(value) for value in self.times],
            "raw_xy": [point(value) for value in self.raw_xy],
            "clean_xy": [point(value) for value in self.clean_xy],
            "health": [round(float(value), decimals) if np.isfinite(value) else None for value in self.health],
            "max_health": [round(float(value), decimals) if np.isfinite(value) else None for value in self.max_health],
            "heading_deg": [round(float(value), decimals) if np.isfinite(value) else None for value in self.heading_deg],
            "z": [round(float(value), decimals) if np.isfinite(value) else None for value in self.z],
            "chassis_power": [round(float(value), decimals) if np.isfinite(value) else None for value in self.chassis_power],
            "small_heat": [round(float(value), decimals) if np.isfinite(value) else None for value in self.small_heat],
            "small_heat_limit": [round(float(value), decimals) if np.isfinite(value) else None for value in self.small_heat_limit],
            "large_heat": [round(float(value), decimals) if np.isfinite(value) else None for value in self.large_heat],
            "large_heat_limit": [round(float(value), decimals) if np.isfinite(value) else None for value in self.large_heat_limit],
            "cumulative_17mm": [round(float(value), decimals) if np.isfinite(value) else None for value in self.cumulative_17mm],
            "cumulative_42mm": [round(float(value), decimals) if np.isfinite(value) else None for value in self.cumulative_42mm],
            "total_coins": [round(float(value), decimals) if np.isfinite(value) else None for value in self.total_coins],
            "remaining_coins": [round(float(value), decimals) if np.isfinite(value) else None for value in self.remaining_coins],
            "vulnerable": [bool(value >= 0.5) if np.isfinite(value) else False for value in self.vulnerable],
            "interpolated": [bool(value) for value in self.interpolated_mask],
            "continuity": [bool(value) for value in self.continuity_mask],
            "diagnostics": asdict(self.diagnostics),
        }


@dataclass(frozen=True)
class MatchEvent:
    time: float
    event_type: str
    robot_id: int | None
    robot_type: str | None
    camp: str | None
    school: str | None
    target_robot_id: int | None
    target_type: str | None
    category: str | None
    value: float | None
    note: str | None

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def open_readonly(path: Path) -> sqlite3.Connection:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SQLite 文件不存在: {path}")
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA temp_store = MEMORY")
    return connection


def load_match(connection: sqlite3.Connection, game_id: int) -> dict[str, Any]:
    row = connection.execute(
        "SELECT * FROM matches WHERE game_id = ? LIMIT 1", (game_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"matches 中不存在 game_id={game_id}")
    return dict(row)


def load_rows(
    connection: sqlite3.Connection,
    game_id: int,
    start: float | None = None,
    end: float | None = None,
    camps: Iterable[str] | None = None,
    robot_types: Iterable[str] | None = None,
    robot_ids: Iterable[int] | None = None,
    include_static: bool = False,
) -> list[sqlite3.Row]:
    clauses = ["game_id = ?"]
    if not include_static:
        clauses.append("机器人类型 NOT IN ('基地', '前哨站')")
    parameters: list[Any] = [game_id]
    if start is not None:
        clauses.append("时刻秒 >= ?")
        parameters.append(start)
    if end is not None:
        clauses.append("时刻秒 <= ?")
        parameters.append(end)

    for column, values in (
        ("阵营", list(camps or [])),
        ("机器人类型", list(robot_types or [])),
        ("robot_id", list(robot_ids or [])),
    ):
        if values:
            placeholders = ",".join("?" for _ in values)
            clauses.append(f'"{column}" IN ({placeholders})')
            parameters.extend(values)

    sql = f"""
        SELECT 时刻秒, robot_id, 机器人类型, 阵营, 学校名, x, y,
               当前血量, 最大血量, 枪口朝向, z, 底盘功率,
               小热量, 小热量上限, 大热量, 大热量上限,
               累计17mm发弹, 累计42mm发弹, 队伍总金币, 队伍剩余金币,
               是否易伤
        FROM timeseries
        WHERE {' AND '.join(clauses)}
        ORDER BY 阵营, robot_id, 时刻秒, rowid
    """
    return connection.execute(sql, parameters).fetchall()


def load_events(
    connection: sqlite3.Connection,
    game_id: int,
    start: float | None = None,
    end: float | None = None,
    event_types: Iterable[str] | None = None,
) -> list[MatchEvent]:
    clauses = ["game_id = ?"]
    parameters: list[Any] = [game_id]
    if start is not None:
        clauses.append("时刻秒 >= ?")
        parameters.append(start)
    if end is not None:
        clauses.append("时刻秒 <= ?")
        parameters.append(end)
    selected_types = list(event_types or [])
    if selected_types:
        placeholders = ",".join("?" for _ in selected_types)
        clauses.append(f"事件类型 IN ({placeholders})")
        parameters.extend(selected_types)
    rows = connection.execute(
        f"""
        SELECT 时刻秒, 事件类型, robot_id, 机器人类型, 阵营, 学校名,
               目标robot_id, 目标类型, 类别, 数值, 备注
        FROM events
        WHERE {' AND '.join(clauses)}
        ORDER BY 时刻秒, rowid
        """,
        parameters,
    )
    return [
        MatchEvent(
            time=float(row["时刻秒"]),
            event_type=str(row["事件类型"]),
            robot_id=int(row["robot_id"]) if row["robot_id"] is not None else None,
            robot_type=str(row["机器人类型"]) if row["机器人类型"] is not None else None,
            camp=str(row["阵营"]) if row["阵营"] is not None else None,
            school=str(row["学校名"]) if row["学校名"] is not None else None,
            target_robot_id=(
                int(row["目标robot_id"]) if row["目标robot_id"] is not None else None
            ),
            target_type=str(row["目标类型"]) if row["目标类型"] is not None else None,
            category=str(row["类别"]) if row["类别"] is not None else None,
            value=float(row["数值"]) if row["数值"] is not None else None,
            note=str(row["备注"]) if row["备注"] is not None else None,
        )
        for row in rows
    ]


def load_match_events(
    database: Path,
    game_id: int,
    start: float | None = None,
    end: float | None = None,
    event_types: Iterable[str] | None = None,
) -> list[MatchEvent]:
    with open_readonly(database) as connection:
        return load_events(
            connection,
            game_id,
            start=start,
            end=end,
            event_types=event_types,
        )


def event_alignment_summary(
    events: list[MatchEvent], tracks: list[CleanTrack]
) -> dict[str, int]:
    lookup = {(track.key.camp, track.key.robot_id): track for track in tracks}
    with_subject = 0
    positioned = 0
    static_subject = 0
    unmatched_robot = 0
    missing_position = 0
    for event in events:
        if event.robot_id is None or event.camp is None:
            continue
        with_subject += 1
        if event.robot_type in {"基地", "前哨站"}:
            static_subject += 1
            continue
        track = lookup.get((event.camp, event.robot_id))
        if track is None:
            unmatched_robot += 1
            continue
        frame = int(round(event.time))
        indices = np.flatnonzero(track.times == frame)
        if indices.size and np.isfinite(track.clean_xy[indices[0]]).all():
            positioned += 1
        else:
            missing_position += 1
    return {
        "event_count": len(events),
        "events_with_subject_robot": with_subject,
        "events_positioned_on_track": positioned,
        "events_static_subject": static_subject,
        "events_unmatched_robot": unmatched_robot,
        "events_missing_clean_position": missing_position,
    }


def _max_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return round(float(np.max(finite)), 3) if finite.size else 0.0


def _path_and_speeds(times: np.ndarray, xy: np.ndarray) -> tuple[float, np.ndarray]:
    if len(times) < 2:
        return 0.0, np.array([], dtype=float)
    valid = np.isfinite(xy).all(axis=1)
    delta_xy = np.diff(xy, axis=0)
    delta_t = np.diff(times)
    adjacent = valid[:-1] & valid[1:] & (delta_t > 0)
    distances = np.linalg.norm(delta_xy, axis=1)
    speeds = np.full(len(delta_t), np.nan)
    speeds[adjacent] = distances[adjacent] / delta_t[adjacent]
    return float(np.nansum(np.where(adjacent, distances, 0.0))), speeds


def _interpolate_short_gaps(xy: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    result = xy.copy()
    interpolated = np.zeros(len(result), dtype=bool)
    valid = np.isfinite(result).all(axis=1)
    index = 0
    while index < len(result):
        if valid[index]:
            index += 1
            continue
        gap_start = index
        while index < len(result) and not valid[index]:
            index += 1
        gap_end = index
        gap_size = gap_end - gap_start
        if gap_size <= max_gap and gap_start > 0 and gap_end < len(result):
            left = result[gap_start - 1]
            right = result[gap_end]
            for offset, frame in enumerate(range(gap_start, gap_end), start=1):
                weight = offset / (gap_size + 1)
                result[frame] = left * (1.0 - weight) + right * weight
                interpolated[frame] = True
            valid[gap_start:gap_end] = True
    return result, interpolated


def _interpolate_scalar(values: np.ndarray, max_gap: int) -> np.ndarray:
    result = values.copy()
    valid = np.isfinite(result)
    index = 0
    while index < len(result):
        if valid[index]:
            index += 1
            continue
        start = index
        while index < len(result) and not valid[index]:
            index += 1
        end = index
        size = end - start
        if size <= max_gap and start > 0 and end < len(result):
            result[start:end] = np.linspace(result[start - 1], result[end], size + 2)[1:-1]
            valid[start:end] = True
    return result


def _interpolate_heading(values: np.ndarray, max_gap: int) -> np.ndarray:
    radians = np.deg2rad(values)
    sin_values = _interpolate_scalar(np.sin(radians), max_gap)
    cos_values = _interpolate_scalar(np.cos(radians), max_gap)
    result = np.rad2deg(np.arctan2(sin_values, cos_values))
    result[~(np.isfinite(sin_values) & np.isfinite(cos_values))] = np.nan
    return result


def _row_value(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (IndexError, KeyError):
        return default


def _smooth_segments(xy: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return xy.copy()
    if window % 2 == 0:
        raise ValueError("平滑窗口必须是奇数")
    result = xy.copy()
    valid = np.isfinite(xy).all(axis=1)
    radius = window // 2
    for index in range(radius, len(xy) - radius):
        section = xy[index - radius : index + radius + 1]
        if valid[index - radius : index + radius + 1].all():
            weights = np.arange(1, radius + 2, dtype=float)
            weights = np.concatenate([weights, weights[-2::-1]])
            result[index] = np.average(section, axis=0, weights=weights)
    return result


def clean_track(
    key: TrackKey,
    rows: list[sqlite3.Row],
    start: float | None,
    end: float | None,
    max_gap: int,
    smooth_window: int,
    max_speed_mps: float,
) -> CleanTrack:
    by_time: dict[int, list[tuple[Any, ...]]] = {}
    for row in rows:
        frame = int(round(float(row["时刻秒"])))
        by_time.setdefault(frame, []).append(
            (
                row["x"],
                row["y"],
                _row_value(row, "当前血量"),
                _row_value(row, "最大血量"),
                _row_value(row, "枪口朝向"),
                *(_row_value(row, column) for _, column in EXTRA_TELEMETRY),
            )
        )

    first = int(np.ceil(start)) if start is not None else min(by_time)
    last = int(np.floor(end)) if end is not None else max(by_time)
    times = np.arange(first, last + 1, dtype=float)
    raw_xy = np.full((len(times), 2), np.nan, dtype=float)
    health = np.full(len(times), np.nan, dtype=float)
    max_health = np.full(len(times), np.nan, dtype=float)
    heading = np.full(len(times), np.nan, dtype=float)
    telemetry = {
        name: np.full(len(times), np.nan, dtype=float)
        for name, _ in EXTRA_TELEMETRY
    }
    duplicate_rows = 0
    null_coordinates = 0

    for frame, coordinates in by_time.items():
        if frame < first or frame > last:
            continue
        duplicate_rows += max(0, len(coordinates) - 1)
        finite = [
            (float(values[0]), float(values[1]))
            for values in coordinates
            for x, y in [(values[0], values[1])]
            if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)
        ]
        null_coordinates += len(coordinates) - len(finite)
        if finite:
            raw_xy[frame - first] = np.median(np.asarray(finite), axis=0)
        for target, value_index in ((health, 2), (max_health, 3), (heading, 4)):
            values = [
                float(item[value_index])
                for item in coordinates
                if item[value_index] is not None and np.isfinite(item[value_index])
            ]
            if values:
                target[frame - first] = float(np.median(values))
        for value_index, (name, _) in enumerate(EXTRA_TELEMETRY, start=5):
            values = [
                float(item[value_index])
                for item in coordinates
                if item[value_index] is not None and np.isfinite(item[value_index])
            ]
            if values:
                telemetry[name][frame - first] = float(np.median(values))

    objective_position = OBJECTIVE_POSITIONS.get((key.camp, key.robot_type))
    if objective_position is not None:
        raw_xy[:] = np.asarray(objective_position, dtype=float)

    clean_xy = raw_xy.copy()
    finite = np.isfinite(clean_xy).all(axis=1)
    in_bounds = (
        finite
        & (clean_xy[:, 0] >= 0.0)
        & (clean_xy[:, 0] <= FIELD_WIDTH_M)
        & (clean_xy[:, 1] >= 0.0)
        & (clean_xy[:, 1] <= FIELD_HEIGHT_M)
    )
    out_of_bounds = int(np.count_nonzero(finite & ~in_bounds))
    clean_xy[~in_bounds] = np.nan

    spike_mask = np.zeros(len(clean_xy), dtype=bool)
    for index in range(1, len(clean_xy) - 1):
        before, current, after = clean_xy[index - 1 : index + 2]
        if not np.isfinite([before, current, after]).all():
            continue
        speed_before = np.linalg.norm(current - before)
        speed_after = np.linalg.norm(after - current)
        bridge_speed = np.linalg.norm(after - before) / 2.0
        if (
            speed_before > max_speed_mps
            and speed_after > max_speed_mps
            and bridge_speed <= max_speed_mps
        ):
            spike_mask[index] = True
    clean_xy[spike_mask] = np.nan

    observed_mask = np.isfinite(clean_xy).all(axis=1)
    clean_xy, interpolated_mask = _interpolate_short_gaps(clean_xy, max_gap=max_gap)
    clean_xy = _smooth_segments(clean_xy, window=smooth_window)
    health = _interpolate_scalar(health, max_gap=max_gap)
    max_health = _interpolate_scalar(max_health, max_gap=max_gap)
    heading = _interpolate_heading(heading, max_gap=max_gap)
    for name in telemetry:
        telemetry[name] = _interpolate_scalar(telemetry[name], max_gap=max_gap)
    valid_health = np.isfinite(health) & np.isfinite(max_health) & (max_health > 0)
    health[valid_health] = np.clip(health[valid_health], 0.0, max_health[valid_health])

    raw_path, raw_speeds = _path_and_speeds(times, raw_xy)
    clean_path, clean_speeds = _path_and_speeds(times, clean_xy)
    continuity = np.zeros(len(times), dtype=bool)
    if len(times) > 1:
        continuity[1:] = np.isfinite(clean_speeds) & (clean_speeds <= max_speed_mps)
    residual_jumps = int(np.count_nonzero(np.isfinite(clean_speeds) & (clean_speeds > max_speed_mps)))

    diagnostics = TrackDiagnostics(
        label=key.label,
        raw_rows=len(rows),
        frame_count=len(times),
        missing_frames=int(np.count_nonzero(~np.isfinite(raw_xy).all(axis=1))),
        duplicate_rows=duplicate_rows,
        null_coordinates=null_coordinates,
        out_of_bounds=out_of_bounds,
        spike_outliers=int(np.count_nonzero(spike_mask)),
        interpolated_frames=int(np.count_nonzero(interpolated_mask)),
        remaining_frames=int(np.count_nonzero(np.isfinite(clean_xy).all(axis=1))),
        residual_jump_count=residual_jumps,
        raw_path_m=round(raw_path, 3),
        clean_path_m=round(clean_path, 3),
        max_raw_speed_mps=_max_finite(raw_speeds),
        max_clean_speed_mps=_max_finite(clean_speeds),
    )
    return CleanTrack(
        key=key,
        times=times,
        raw_xy=raw_xy,
        clean_xy=clean_xy,
        health=health,
        max_health=max_health,
        heading_deg=heading,
        **telemetry,
        observed_mask=observed_mask,
        interpolated_mask=interpolated_mask,
        continuity_mask=continuity,
        diagnostics=diagnostics,
    )


def load_and_clean_tracks(
    database: Path,
    game_id: int,
    start: float | None = None,
    end: float | None = None,
    camps: Iterable[str] | None = None,
    robot_types: Iterable[str] | None = None,
    robot_ids: Iterable[int] | None = None,
    max_gap: int = 3,
    smooth_window: int = 3,
    max_speed_mps: float = 8.0,
    include_static: bool = False,
) -> tuple[dict[str, Any], list[CleanTrack]]:
    with open_readonly(database) as connection:
        match = load_match(connection, game_id)
        rows = load_rows(
            connection,
            game_id,
            start=start,
            end=end,
            camps=camps,
            robot_types=robot_types,
            robot_ids=robot_ids,
            include_static=include_static,
        )
    if not rows:
        raise ValueError("筛选条件下没有可用的机器人状态")

    grouped: dict[TrackKey, list[sqlite3.Row]] = {}
    for row in rows:
        key = TrackKey(
            camp=str(row["阵营"]),
            robot_id=int(row["robot_id"]),
            robot_type=str(row["机器人类型"]),
            school=str(row["学校名"]),
        )
        grouped.setdefault(key, []).append(row)

    tracks = []
    for key, track_rows in sorted(grouped.items()):
        speed_limit = max(DEFAULT_SPEED_MPS.get(key.robot_type, max_speed_mps), max_speed_mps)
        tracks.append(
            clean_track(
                key,
                track_rows,
                start=start,
                end=end,
                max_gap=max_gap,
                smooth_window=smooth_window,
                max_speed_mps=speed_limit,
            )
        )
    return match, tracks


def quality_report(
    match: dict[str, Any], tracks: list[CleanTrack], parameters: dict[str, Any]
) -> dict[str, Any]:
    return {
        "match": match,
        "field": {
            "width_m": FIELD_WIDTH_M,
            "height_m": FIELD_HEIGHT_M,
            "coordinate_interpretation": "x: 红方(左)到蓝方(右), y: 画布下到上",
            "canvas_source_size_px": list(SOURCE_IMAGE_SIZE),
            "canvas_inner_field_rect_px": list(INNER_FIELD_RECT_PX),
            "canvas_accuracy": "以内侧停机坪/有效场地角点映射28m×15m，仅用于轨迹解释",
        },
        "parameters": parameters,
        "summary": {
            "track_count": len(tracks),
            "raw_rows": sum(track.diagnostics.raw_rows for track in tracks),
            "out_of_bounds": sum(track.diagnostics.out_of_bounds for track in tracks),
            "spike_outliers": sum(track.diagnostics.spike_outliers for track in tracks),
            "interpolated_frames": sum(
                track.diagnostics.interpolated_frames for track in tracks
            ),
            "residual_jump_count": sum(
                track.diagnostics.residual_jump_count for track in tracks
            ),
        },
        "tracks": [asdict(track.diagnostics) for track in tracks],
    }
