"""Local HTTP application for selecting and generating RMUC match replays."""

from __future__ import annotations

import json
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from .buffs import build_buff_intervals
from .combat import infer_attacks
from .dart import infer_dart_impacts
from .field import default_canvas
from .pipeline import (
    event_alignment_summary,
    load_and_clean_tracks,
    load_match_events,
    open_readonly,
    quality_report,
)
from .render import render_interactive_html
from .revival import infer_paid_revivals, infer_respawn_intervals


ProgressCallback = Callable[[int, str], None]


def search_matches(database: Path, keyword: str = "") -> list[dict[str, Any]]:
    """Return matches chronologically, or relevance-ranked for a keyword."""

    query = keyword.strip()
    connection = open_readonly(database)
    try:
        if not query:
            rows = connection.execute(
                """
                SELECT 赛区, 场次号, 赛程, 局号, game_id, web_game_id,
                       红方学校, 蓝方学校, 胜方, 开始时间, 时长秒
                FROM matches
                ORDER BY game_id
                """
            ).fetchall()
        else:
            contains = f"%{query}%"
            prefix = f"{query}%"
            rows = connection.execute(
                """
                SELECT 赛区, 场次号, 赛程, 局号, game_id, web_game_id,
                       红方学校, 蓝方学校, 胜方, 开始时间, 时长秒
                FROM matches
                WHERE 红方学校 LIKE ? OR 蓝方学校 LIKE ?
                   OR 赛区 LIKE ? OR 赛程 LIKE ? OR CAST(game_id AS TEXT) LIKE ?
                ORDER BY CASE
                    WHEN 红方学校 = ? OR 蓝方学校 = ? THEN 0
                    WHEN 红方学校 LIKE ? OR 蓝方学校 LIKE ? THEN 1
                    ELSE 2
                END,
                game_id
                """,
                (
                    contains,
                    contains,
                    contains,
                    contains,
                    contains,
                    query,
                    query,
                    prefix,
                    prefix,
                ),
            ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


@dataclass(frozen=True)
class ReplayRequest:
    game_id: int
    start: float | None = None
    end: float | None = None
    max_gap: int = 3
    smooth_window: int = 3
    max_speed: float = 8.0


def validate_job_payload(payload: dict[str, Any]) -> ReplayRequest:
    try:
        game_id = int(payload["game_id"])
        start = float(payload["start"]) if payload.get("start") not in (None, "") else None
        end = float(payload["end"]) if payload.get("end") not in (None, "") else None
        max_gap = int(payload.get("max_gap", 3))
        smooth_window = int(payload.get("smooth_window", 3))
        max_speed = float(payload.get("max_speed", 8.0))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("参数格式不正确") from error
    if game_id <= 0:
        raise ValueError("game_id 必须为正整数")
    if start is not None and start < 0:
        raise ValueError("开始时间不能小于0")
    if end is not None and end < 0:
        raise ValueError("结束时间不能小于0")
    if start is not None and end is not None and start > end:
        raise ValueError("开始时间不能晚于结束时间")
    if max_gap < 0 or max_gap > 30:
        raise ValueError("最大插值缺帧数必须在0到30之间")
    if smooth_window < 1 or smooth_window > 9 or smooth_window % 2 == 0:
        raise ValueError("平滑窗口必须是1到9之间的奇数")
    if max_speed <= 0 or max_speed > 30:
        raise ValueError("速度阈值必须在0到30m/s之间")
    return ReplayRequest(game_id, start, end, max_gap, smooth_window, max_speed)
def generate_replay(
    root: Path,
    database: Path,
    output_dir: Path,
    request: ReplayRequest,
    progress: ProgressCallback,
) -> dict[str, Any]:
    """Run the existing parser in observable stages and generate a player."""

    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = {
        "game_id": request.game_id,
        "start": request.start,
        "end": request.end,
        "max_gap": request.max_gap,
        "smooth_window": request.smooth_window,
        "max_speed_mps": request.max_speed,
        "source": "local_web_app",
    }
    progress(5, "读取车辆连续帧")
    match, tracks = load_and_clean_tracks(
        database,
        request.game_id,
        start=request.start,
        end=request.end,
        max_gap=request.max_gap,
        smooth_window=request.smooth_window,
        max_speed_mps=request.max_speed,
    )
    progress(18, "读取基地与前哨站")
    _, objectives = load_and_clean_tracks(
        database,
        request.game_id,
        start=request.start,
        end=request.end,
        robot_types=("基地", "前哨站"),
        max_gap=request.max_gap,
        smooth_window=1,
        max_speed_mps=request.max_speed,
        include_static=True,
    )
    progress(30, "读取比赛事件")
    events = load_match_events(
        database, request.game_id, start=request.start, end=request.end
    )
    canvas = default_canvas(root)
    progress(44, "推断攻击关系")
    attacks, attack_summary = infer_attacks(events, tracks + objectives)
    match_end = max(float(track.times[-1]) for track in tracks if len(track.times))
    progress(58, "计算增益、复活与飞镖状态")
    buff_intervals, buff_summary = build_buff_intervals(
        events, tracks + objectives, match_end
    )
    paid_revivals, revival_summary = infer_paid_revivals(tracks)
    respawn_intervals, respawn_summary = infer_respawn_intervals(
        tracks, paid_revivals, match_end
    )
    dart_impacts, dart_summary = infer_dart_impacts(events, objectives)
    progress(72, "整理质量报告")
    report = quality_report(match, tracks, parameters)
    report["event_alignment"] = event_alignment_summary(events, tracks)
    report["attack_inference"] = attack_summary
    report["buff_timeline"] = buff_summary
    report["paid_revival_inference"] = revival_summary
    report["respawn_timeline"] = respawn_summary
    report["dart_inference"] = dart_summary
    report["objective_tracking"] = {
        "track_count": len(objectives),
        "position_source": "规则画布近似标定",
    }
    report["summary"].update(report["event_alignment"])
    report["summary"].update(attack_summary)
    report["summary"].update(
        {
            "dart_hit_count": len(dart_impacts),
            "inferred_paid_revival_count": len(paid_revivals),
            "respawn_interval_count": len(respawn_intervals),
            "objective_track_count": len(objectives),
            "deployed_hero_lob_count": sum(
                attack.special_mode == "hero_deployed_lob" for attack in attacks
            ),
        }
    )
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    progress(84, "生成交互回放")
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
        return_url="/",
    )
    progress(100, "解析完成")
    return report["summary"]


@dataclass
class JobState:
    job_id: str
    request: ReplayRequest
    output_dir: Path
    status: str = "queued"
    progress: int = 0
    stage: str = "等待解析"
    replay_url: str | None = None
    error: str | None = None
    summary: dict[str, Any] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, progress: int, stage: str) -> None:
        with self.lock:
            self.status = "running" if progress < 100 else "done"
            self.progress = progress
            self.stage = stage

    def to_jsonable(self) -> dict[str, Any]:
        with self.lock:
            return {
                "job_id": self.job_id,
                "status": self.status,
                "progress": self.progress,
                "stage": self.stage,
                "replay_url": self.replay_url,
                "error": self.error,
                "summary": self.summary,
            }


class ReplayApplication:
    def __init__(self, root: Path, database: Path, output_root: Path) -> None:
        self.root = root.resolve()
        self.database = database.resolve()
        self.output_root = output_root.resolve()
        self.index_path = self.root / "rmuc_web" / "index.html"
        self.jobs: dict[str, JobState] = {}
        self.jobs_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rmuc-replay")

    def create_job(self, request: ReplayRequest) -> JobState:
        connection = open_readonly(self.database)
        try:
            exists = connection.execute(
                "SELECT 1 FROM matches WHERE game_id = ? LIMIT 1", (request.game_id,)
            ).fetchone()
        finally:
            connection.close()
        if not exists:
            raise ValueError(f"不存在 game_id={request.game_id} 的比赛")
        job_id = uuid.uuid4().hex[:12]
        state = JobState(job_id, request, self.output_root / job_id)
        with self.jobs_lock:
            self.jobs[job_id] = state
        self.executor.submit(self._run_job, state)
        return state

    def _run_job(self, state: JobState) -> None:
        try:
            state.update(1, "任务已启动")
            summary = generate_replay(
                self.root,
                self.database,
                state.output_dir,
                state.request,
                state.update,
            )
            with state.lock:
                state.summary = summary
                state.replay_url = f"/replays/{state.job_id}/trajectory.html"
                state.status = "done"
                state.progress = 100
                state.stage = "解析完成，正在进入回放"
        except Exception as error:  # keep background failures visible in the UI
            traceback.print_exc()
            with state.lock:
                state.status = "error"
                state.stage = "解析失败"
                state.error = str(error)

    def get_job(self, job_id: str) -> JobState | None:
        with self.jobs_lock:
            return self.jobs.get(job_id)


def make_handler(application: ReplayApplication) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "RMUCReplay/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {format % args}")

        def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_file(self, path: Path, content_type: str) -> None:
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self.send_file(application.index_path, "text/html; charset=utf-8")
                return
            if parsed.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            if parsed.path == "/api/matches":
                keyword = parse_qs(parsed.query).get("q", [""])[0]
                self.send_json({"matches": search_matches(application.database, keyword)})
                return
            if parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.rsplit("/", 1)[-1]
                state = application.get_job(job_id)
                if state is None:
                    self.send_json({"error": "任务不存在"}, HTTPStatus.NOT_FOUND)
                else:
                    self.send_json(state.to_jsonable())
                return
            if parsed.path.startswith("/replays/") and parsed.path.endswith("/trajectory.html"):
                parts = parsed.path.strip("/").split("/")
                if len(parts) != 3:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                state = application.get_job(parts[1])
                if state is None or state.status != "done":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self.send_file(state.output_dir / "trajectory.html", "text/html; charset=utf-8")
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/jobs":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 64_000:
                    raise ValueError("请求正文为空或过大")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                request = validate_job_payload(payload)
                state = application.create_job(request)
            except (ValueError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json(state.to_jsonable(), HTTPStatus.ACCEPTED)

    return Handler


def create_server(
    root: Path,
    database: Path,
    output_root: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> tuple[ThreadingHTTPServer, ReplayApplication]:
    application = ReplayApplication(root, database, output_root)
    server = ThreadingHTTPServer((host, port), make_handler(application))
    return server, application
