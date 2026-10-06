"""Static, animated and interactive renderers for cleaned RMUC tracks."""

from __future__ import annotations

import base64
import html as html_lib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .field import FORTRESS_POSITIONS, FORTRESS_RADIUS_M, FieldCanvas
from .pipeline import CleanTrack, MatchEvent
from .combat import AttackInference
from .buffs import BuffInterval
from .dart import DartImpact
from .revival import RespawnInterval, RevivalInference


TYPE_MARKERS = {
    "英雄": "o",
    "工程": "s",
    "步兵3": "^",
    "步兵4": "v",
    "空中": "P",
    "哨兵": "D",
}
TYPE_DASHES = {
    "英雄": "-",
    "工程": "--",
    "步兵3": "-.",
    "步兵4": ":",
    "空中": (0, (5, 2)),
    "哨兵": (0, (2, 1)),
}
CAMP_COLORS = {"红": "#ff3b3b", "蓝": "#3282ff"}


def _configure_matplotlib_fonts() -> None:
    from matplotlib import font_manager, rcParams

    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
            family = font_manager.FontProperties(fname=str(candidate)).get_name()
            rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
            break
    rcParams["axes.unicode_minus"] = False


def _load_cropped_canvas(canvas: FieldCanvas) -> np.ndarray:
    from PIL import Image

    with Image.open(canvas.image_path) as image:
        crop = canvas.crop_pixels(*image.size)
        return np.asarray(image.convert("RGB").crop(crop))


def setup_field_axis(axis: Any, canvas: FieldCanvas) -> None:
    image = _load_cropped_canvas(canvas)
    axis.imshow(
        image,
        extent=(0, canvas.width_m, 0, canvas.height_m),
        origin="upper",
        alpha=0.82,
        zorder=0,
    )
    axis.set_xlim(0, canvas.width_m)
    axis.set_ylim(0, canvas.height_m)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x / m（红方 → 蓝方）")
    axis.set_ylabel("y / m")
    axis.grid(color="white", alpha=0.16, linewidth=0.5)


def _segments(track: CleanTrack) -> list[np.ndarray]:
    segments: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    for index, point in enumerate(track.clean_xy):
        finite = np.isfinite(point).all()
        connected = index == 0 or bool(track.continuity_mask[index])
        if not finite or (current and not connected):
            if current:
                segments.append(current)
            current = []
        if finite:
            current.append(point)
    if current:
        segments.append(current)
    return [np.asarray(segment) for segment in segments]


def render_static(
    output: Path,
    canvas: FieldCanvas,
    match: dict[str, Any],
    tracks: list[CleanTrack],
    show_raw: bool = True,
    dpi: int = 180,
) -> None:
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts()
    figure, axis = plt.subplots(figsize=(14, 8.2), constrained_layout=True)
    setup_field_axis(axis, canvas)
    for track in tracks:
        color = CAMP_COLORS.get(track.key.camp, "#f4c542")
        if show_raw:
            axis.plot(
                track.raw_xy[:, 0],
                track.raw_xy[:, 1],
                color=color,
                alpha=0.18,
                linewidth=0.7,
                zorder=1,
            )
        for segment_index, segment in enumerate(_segments(track)):
            axis.plot(
                segment[:, 0],
                segment[:, 1],
                color=color,
                linestyle=TYPE_DASHES.get(track.key.robot_type, "-"),
                linewidth=1.5,
                alpha=0.9,
                label=track.key.label if segment_index == 0 else None,
                zorder=2,
            )
        finite_indices = np.flatnonzero(np.isfinite(track.clean_xy).all(axis=1))
        if finite_indices.size:
            first, last = finite_indices[0], finite_indices[-1]
            axis.scatter(
                *track.clean_xy[first],
                color=color,
                marker=TYPE_MARKERS.get(track.key.robot_type, "o"),
                s=28,
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )
            axis.scatter(
                *track.clean_xy[last],
                color=color,
                marker="x",
                s=30,
                linewidths=1.1,
                zorder=3,
            )
    title = (
        f"{match['赛程']} 第{match['局号']}局 | "
        f"{match['红方学校']} vs {match['蓝方学校']}"
    )
    axis.set_title(title)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def render_gif(
    output: Path,
    canvas: FieldCanvas,
    match: dict[str, Any],
    tracks: list[CleanTrack],
    fps: int = 12,
    trail_seconds: int = 20,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    _configure_matplotlib_fonts()
    frame_times = np.unique(np.concatenate([track.times for track in tracks]))
    figure, axis = plt.subplots(figsize=(11.2, 6.4), constrained_layout=True)
    setup_field_axis(axis, canvas)
    artists = []
    for track in tracks:
        color = CAMP_COLORS.get(track.key.camp, "#f4c542")
        (trail,) = axis.plot([], [], color=color, linewidth=1.5, alpha=0.88)
        (marker,) = axis.plot(
            [],
            [],
            linestyle="none",
            marker=TYPE_MARKERS.get(track.key.robot_type, "o"),
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=6,
            label=track.key.label,
        )
        artists.append((track, trail, marker))
    time_label = axis.text(
        0.01,
        0.99,
        "",
        transform=axis.transAxes,
        va="top",
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
    )
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4, fontsize=7)

    def update(frame_time: float) -> list[Any]:
        changed: list[Any] = [time_label]
        time_label.set_text(f"t = {frame_time:.0f} s")
        for track, trail, marker in artists:
            within = (track.times <= frame_time) & (track.times >= frame_time - trail_seconds)
            points = track.clean_xy[within].copy()
            continuity = track.continuity_mask[within]
            if len(points) > 1:
                points[1:][~continuity[1:]] = np.nan
            trail.set_data(points[:, 0], points[:, 1])
            current = np.flatnonzero(track.times == frame_time)
            if current.size and np.isfinite(track.clean_xy[current[0]]).all():
                marker.set_data([track.clean_xy[current[0], 0]], [track.clean_xy[current[0], 1]])
            else:
                marker.set_data([], [])
            changed.extend([trail, marker])
        return changed

    animation = FuncAnimation(
        figure, update, frames=frame_times, interval=1000 / fps, blit=False
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(figure)


def render_interactive_html(
    output: Path,
    canvas: FieldCanvas,
    match: dict[str, Any],
    tracks: list[CleanTrack],
    events: list[MatchEvent] | None = None,
    attacks: list[AttackInference] | None = None,
    objectives: list[CleanTrack] | None = None,
    buff_intervals: list[BuffInterval] | None = None,
    paid_revivals: list[RevivalInference] | None = None,
    respawn_intervals: list[RespawnInterval] | None = None,
    dart_impacts: list[DartImpact] | None = None,
    return_url: str | None = None,
    scores: list[dict[str, Any]] | None = None,
    timeseries_scores: dict[str, list[dict[str, Any]]] | None = None,
    review: dict[str, Any] | None = None,
    backend: dict[str, str] | None = None,
) -> None:
    image_b64 = base64.b64encode(canvas.image_path.read_bytes()).decode("ascii")
    payload = {
        "delivery": {
            "mode": "backend" if backend else "static",
            "export_url": (backend or {}).get("export_url"),
            "return_url": return_url,
            "csv_url": "scores.csv" if (output.parent / "scores.csv").is_file() else None,
        },
        "field": {
            "width": canvas.width_m,
            "height": canvas.height_m,
            "crop": [
                canvas.crop_left,
                canvas.crop_top,
                canvas.crop_right,
                canvas.crop_bottom,
            ],
            "fortresses": {
                camp: [position[0], position[1]]
                for camp, position in FORTRESS_POSITIONS.items()
            },
            "fortress_radius": FORTRESS_RADIUS_M,
        },
        "match": match,
        "tracks": [track.to_jsonable() for track in tracks],
        "events": [event.to_jsonable() for event in (events or [])],
        "attacks": [attack.to_jsonable() for attack in (attacks or [])],
        "objectives": [track.to_jsonable() for track in (objectives or [])],
        "buff_intervals": [interval.to_jsonable() for interval in (buff_intervals or [])],
        "paid_revivals": [revival.to_jsonable() for revival in (paid_revivals or [])],
        "respawn_intervals": [interval.to_jsonable() for interval in (respawn_intervals or [])],
        "dart_impacts": [impact.to_jsonable() for impact in (dart_impacts or [])],
        "scores": scores or [],
        "timeseries_scores": timeseries_scores or {},
        "review": review or {},
    }
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return_button = (
        f'<a class="return-button" href="{html_lib.escape(return_url, quote=True)}">← 返回比赛列表</a>'
        if return_url
        else ""
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RM BattleScope · 战绩与复盘</title>
<style>
:root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
body {{ margin: 0; background: #151515; color: #f4f4f4; }}
main {{ max-width: 1600px; margin: auto; padding: 16px; }}
.controls, .filters {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }}
button, input {{ font: inherit; }} button {{ padding: 6px 12px; }}
button[aria-pressed="true"] {{ background: #5b4aa8; color: #fff; border-color: #b8a8ff; }}
.return-button {{ display: inline-flex; align-items: center; padding: 6px 11px; border: 1px solid #666; border-radius: 2px; color: #f4f4f4; background: #292929; text-decoration: none; }}
.return-button:hover {{ background: #3a3a3a; border-color: #999; }}
.playback {{ display: flex; flex: 1 1 900px; gap: 5px; align-items: center; flex-wrap: wrap; }}
input[type=range] {{ flex: 1; min-width: 220px; }}
#customSpeed {{ flex: 0 1 160px; min-width: 120px; }}
.stage {{ position: relative; aspect-ratio: 28 / 15; width: 100%; }}
canvas {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
.battle-grid {{ display: grid; grid-template-columns: minmax(168px, 202px) minmax(0, 1fr) minmax(168px, 202px); gap: 8px; align-items: start; }}
.roster {{ display: grid; max-width: 200px; gap: 5px; }}
.roster-title {{ display: flex; justify-content: space-between; align-items: baseline; padding: 2px 4px; }}
.roster-title strong {{ white-space: nowrap; }} .roster-title .muted {{ min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; text-align: right; }}
.unit-card, .objective-card {{ border: 1px solid #3b3b3b; border-left: 3px solid var(--team); background: rgba(29,29,29,.92); padding: 5px 7px; min-width: 0; }}
.roster.blue .unit-card, .roster.blue .objective-card {{ border-left: 1px solid #3b3b3b; border-right: 3px solid var(--team); }}
.unit-head, .objective-head {{ display: flex; justify-content: space-between; gap: 5px; align-items: center; font-size: 12px; }}
.unit-name {{ display: flex; align-items: center; gap: 5px; min-width: 0; }}
.unit-number {{ width: 19px; height: 19px; display: inline-grid; place-items: center; border: 2px solid var(--team); border-radius: 50%; font-weight: 500; flex: 0 0 auto; }}
.hp-track, .mini-track {{ height: 5px; background: #343434; margin-top: 4px; overflow: hidden; }}
.hp-fill, .mini-fill {{ height: 100%; width: 0; background: var(--team); transition: width .12s linear; }}
.respawn-row {{ display: none; margin-top: 4px; color: #ffd54a; font-size: 11px; }}
.respawn-row.active {{ display: block; }}
.respawn-track {{ height: 6px; margin-top: 2px; background: #343434; overflow: hidden; }}
.respawn-fill {{ height: 100%; width: 0; background: #ffd54a; transition: width .12s linear; }}
.respawn-fill.paid {{ background: #69f0ae; }}
.unit-stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2px 6px; margin-top: 4px; color: #cfcfcf; font-size: 11px; }}
.dmg-text {{ font-size: 11px; color: #ff9800; min-width: 38px; text-align: right; flex-shrink: 0; }}
.kda-text {{ font-size: 11px; color: #aaa; min-width: 42px; text-align: center; flex-shrink: 0; }}
.score-row {{ display: flex; align-items: center; gap: 4px; margin-top: 3px; }}
.score-label {{ font-size: 10px; color: #888; width: 24px; flex-shrink: 0; }}
.score-value {{ font-size: 13px; font-weight: 700; min-width: 30px; text-align: right; flex-shrink: 0; }}
.score-track {{ flex: 1; height: 4px; background: #343434; border-radius: 2px; overflow: hidden; }}
.score-fill {{ height: 100%; border-radius: 2px; transition: width .15s linear; }}
.score-fill.high {{ background: #69f0ae; }}
.score-fill.mid {{ background: #ffd54a; }}
.score-fill.low {{ background: #ff5252; }}
.roster-title {{ display: flex; justify-content: space-between; align-items: baseline; }}
.team-score {{ font-size: 15px; font-weight: 700; white-space: nowrap; }}
.unit-status {{ min-height: 14px; margin-top: 2px; font-size: 11px; color: #ffd54a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.vulnerable-card {{ box-shadow: 0 0 10px rgba(255,51,77,.72), inset 0 0 8px rgba(255,51,77,.18); border-color: #ff334d; }}
.buyback-card {{ box-shadow: 0 0 12px rgba(105,240,174,.9), inset 0 0 9px rgba(105,240,174,.24); border-color: #69f0ae; }}
.dart-hit-card {{ box-shadow: 0 0 13px rgba(255,77,166,.9), inset 0 0 8px rgba(255,77,166,.25); border-color: #ff4da6; }}
.blind-card {{ box-shadow: 0 0 12px rgba(164,92,255,.8), inset 0 0 10px rgba(72,34,112,.32); border-color: #a45cff; }}
.objectives {{ display: grid; gap: 5px; margin-top: 3px; }}
.center-column {{ min-width: 0; }}
.economy {{ margin-top: 7px; display: grid; gap: 4px; }}
.economy-lane {{ position: relative; height: 22px; background: #242424; border: 1px solid #3b3b3b; overflow: hidden; }}
.economy-fill {{ position: absolute; top: 0; bottom: 0; width: 0; transition: width .12s linear; opacity: .68; }}
.economy-fill.red {{ right: 50%; background: #ff3b3b; }} .economy-fill.blue {{ left: 50%; background: #3282ff; }}
.economy-fill.remaining {{ top: 12px; height: 10px; bottom: auto; opacity: 1; }}
.economy-center {{ position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: #ddd; z-index: 3; }}
.economy-label {{ position: absolute; z-index: 4; top: 2px; font-size: 11px; text-shadow: 0 1px 2px #000; }}
.economy-label.red {{ left: 5px; }} .economy-label.blue {{ right: 5px; text-align: right; }}
.assembly-marker {{ position: absolute; top: 0; bottom: 0; width: 2px; background: #ce7dff; z-index: 5; }}
.assembly-marker::after {{ content: attr(data-label); position: absolute; top: 1px; left: 3px; color: #f0d7ff; font-size: 10px; white-space: nowrap; }}
.assembly-marker.red::after {{ left: auto; right: 3px; }}
.assembly-marker.label-hidden::after {{ display: none; }}
.legend {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; font-size: 13px; }}
.item {{ display: inline-flex; align-items: center; gap: 4px; }}
.item::before {{ content: '●'; color: var(--c); }}
.detail {{ min-height: 1.5em; margin-top: 8px; }}
.muted {{ color: #b8b8b8; }}
@media (max-width: 980px) {{ .battle-grid {{ grid-template-columns: 1fr; }} .center-column {{ order: -1; }} .roster {{ grid-template-columns: repeat(2,minmax(0,1fr)); }} .roster-title, .objectives {{ grid-column: 1 / -1; }} }}
</style>
</head>
<body><main>
<div class="bs-replay-toolbar" id="bsReplayToolbar">
  <div class="controls bs-primary-playback">
    <button id="bsPlayToggle" type="button">播放</button>
    <button id="bsPrev" type="button" aria-label="后退5秒">−5秒</button>
    <button id="bsNext" type="button" aria-label="前进5秒">+5秒</button>
    <label>倍速 <select id="bsPlaybackSpeed" aria-label="常用播放倍速"><option>0.5</option><option selected>1</option><option>1.5</option><option>2</option><option>3</option><option>5</option><option>8</option></select>×</label>
    <label>时间 <output id="time"></output></label>
    <label class="muted">倒计时 <output id="countdown">07:00</output></label>
  </div>
  <input id="slider" type="range" step="1" aria-label="比赛时刻">
  <div id="bsSeekContext" class="bs-muted" aria-live="polite"></div>
</div>
<details class="bs-replay-settings" id="bsReplaySettings"><summary>回放设置 · 播放模式 / 轨迹 / 事件 / 置信度 / 时刻链接</summary><div id="bsSettingsBody">
<div class="controls">
  {return_button}
  <span class="playback" aria-label="播放模式">
    <button id="framePlay" type="button" aria-pressed="false">按帧速播</button>
    <button type="button" class="speed-play" data-speed="1" aria-pressed="false">1×按秒</button>
    <button type="button" class="speed-play" data-speed="1.5" aria-pressed="false">1.5×</button>
    <button type="button" class="speed-play" data-speed="2" aria-pressed="false">2×</button>
    <button type="button" class="speed-play" data-speed="3" aria-pressed="false">3×</button>
    <button type="button" class="speed-play" data-speed="5" aria-pressed="false">5×</button>
    <button id="customPlay" type="button" aria-pressed="false">无极 1.00×</button>
    <label>倍速 <output id="customSpeedValue">1.00×</output></label>
    <input id="customSpeed" type="range" min="0.25" max="8" step="0.05" value="1" aria-label="无极播放倍速">
    <button id="pause" type="button" disabled>暂停</button>
  </span>
</div>
<div class="controls">  <label><input id="raw" type="checkbox"> 原始点</label>
  <label><input id="damageToggle" type="checkbox" checked> 显示伤害</label>
  <label>尾迹 <input id="trail" type="number" min="1" max="120" value="20" style="width:4em"> 秒</label>

</div>
<div id="trackFilters" class="filters" aria-label="轨迹筛选"></div>
<div id="eventFilters" class="filters" aria-label="事件筛选"></div>
<div id="attackFilters" class="filters" aria-label="攻击推断筛选">
  <span class="muted">攻击推断</span>
  <label class="item" style="--c:#69f0ae"><input type="checkbox" data-confidence="high" checked> 高可信</label>
  <label class="item" style="--c:#ffd54a"><input type="checkbox" data-confidence="medium" checked> 中可信</label>
  <label class="item" style="--c:#b8b8b8"><input type="checkbox" data-confidence="low"> 低可信</label>
</div>
</div></details>
<div class="battle-grid">
  <aside id="redRoster" class="roster red" style="--team:#ff3b3b" aria-label="红方机器人状态"></aside>
  <section class="center-column">
    <div class="stage"><canvas id="field" aria-label="RMUC 2026 场地连续帧轨迹"></canvas></div>
    <div class="economy" aria-label="双方经济和装配进度">
      <div id="totalEconomyLane" class="economy-lane"><span class="economy-center"></span><span id="redTotalLabel" class="economy-label red"></span><span id="blueTotalLabel" class="economy-label blue"></span><span id="redTotalBar" class="economy-fill red"></span><span id="blueTotalBar" class="economy-fill blue"></span></div>
      <div id="remainingEconomyLane" class="economy-lane"><span class="economy-center"></span><span id="redRemainingLabel" class="economy-label red"></span><span id="blueRemainingLabel" class="economy-label blue"></span><span id="redRemainingBar" class="economy-fill red remaining"></span><span id="blueRemainingBar" class="economy-fill blue remaining"></span></div>
    </div>
  </section>
  <aside id="blueRoster" class="roster blue" style="--team:#3282ff" aria-label="蓝方机器人状态"></aside>
</div>
<div id="legend" class="legend"></div>
<div id="detail" class="detail" aria-live="polite"></div>
<p class="muted">编号圆圈表示当前车辆，圈内填充表示血量，外环逆时针表示枪口热量。阵亡时红叉表示不可用，绿色填充表示直至实际复活时刻的进度；前哨站转动依据规则时间表和首次战亡/基地护甲条件，方向无原生数据；堡垒状态由规则画布位置与易伤遥测联合推定；基地护甲无原生通知，按飞镖或堡垒条件推定。</p>
</main>
<script id="payload" type="application/json">{data}</script>
<script>
const data = JSON.parse(document.getElementById('payload').textContent);
function replayEscape(value) {{ return String(value??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
const canvas = document.getElementById('field'); const ctx = canvas.getContext('2d');
const slider = document.getElementById('slider'); const framePlay = document.getElementById('framePlay'); const pauseButton=document.getElementById('pause');
const speedButtons=[...document.querySelectorAll('.speed-play')]; const customPlay=document.getElementById('customPlay'); const customSpeed=document.getElementById('customSpeed'); const customSpeedValue=document.getElementById('customSpeedValue'); const playbackButtons=[framePlay,...speedButtons,customPlay];
const timeOut = document.getElementById('time'); const countdownOut=document.getElementById('countdown'); const rawToggle = document.getElementById('raw');
const trailInput = document.getElementById('trail'); const damageToggle=document.getElementById('damageToggle'); const image = new Image();
image.src = 'data:image/jpeg;base64,{image_b64}';
const colors = {{'红':'#ff3b3b','蓝':'#3282ff'}};
const eventColors = {{'发弹':'#f5f5f5','受击':'#ff334d','增益':'#62e58c','装配成功':'#ce7dff','能量机关':'#ff8a4c','飞镖闸门开':'#4dd5ff','飞镖命中':'#ff4da6','雷达反制UAV':'#ff6b6b'}};
const attackColors = {{high:'#69f0ae',medium:'#ffd54a',low:'#b8b8b8'}};
const allTimes = data.tracks.flatMap(t => t.times); const minT = Math.min(...allTimes), maxT = Math.max(...allTimes);
const requestedT=Number(new URLSearchParams(location.hash.slice(1)).get('t'));
slider.min = minT; slider.max = maxT; slider.step='0.01'; slider.value = Number.isFinite(requestedT) ? Math.max(minT,Math.min(maxT,requestedT)) : minT; let timer = null, animationFrame=null;
const visibleTracks = new Set(data.tracks.map((_,i)=>i));
const eventTypes = [...new Set(data.events.map(e=>e.event_type))];
const visibleEvents = new Set(eventTypes.filter(type=>type!=='发弹'));
const visibleAttackConfidence = new Set(['high','medium']);
const allEntities=[...data.tracks,...data.objectives];
const unitOrder=['英雄','工程','步兵3','步兵4','空中','哨兵'];
const fortressEligibleTypes=new Set(['英雄','步兵3','步兵4','哨兵']);
function oppositeCamp(camp) {{ return camp==='红'?'蓝':'红'; }}
function firstOutpostDestroyed(camp) {{
  const track=data.objectives.find(item=>item.key.camp===camp&&item.key.robot_type==='前哨站'); if(!track) return null;
  const index=track.health.findIndex(value=>value!==null&&value<=0); return index>=0?track.times[index]:null;
}}
const firstOutpostDestroyedTimes={{'红':firstOutpostDestroyed('红'),'蓝':firstOutpostDestroyed('蓝')}};
function inferDartArmorOpen(camp) {{
  const impacts=data.dart_impacts.filter(item=>item.target_camp===camp&&item.target_type==='基地').sort((a,b)=>a.time-b.time); let fixed=[];
  for(const impact of impacts) {{
    if(impact.target_profile==='随机移动目标'||impact.target_profile==='末端移动目标') return {{time:impact.time,source:`飞镖${{impact.target_profile}}命中`,confidence:'high'}};
    if(impact.target_profile==='固定目标'||impact.target_profile==='随机固定目标') {{ if(fixed.length&&impact.time-fixed[fixed.length-1].time>45) fixed=[]; fixed.push(impact); if(fixed.length>=4) return {{time:impact.time,source:'同轮4发固定靶飞镖命中',confidence:'high'}}; }}
  }} return null;
}}
function insideFortress(point,camp) {{ const center=data.field.fortresses[camp]; return Boolean(point&&center&&Math.hypot(point[0]-center[0],point[1]-center[1])<=data.field.fortress_radius); }}
function geofencePoint(track,index) {{ return track?.raw_xy?.[index]||track?.clean_xy?.[index]||null; }}
function inferFortressArmorOpen(targetCamp) {{
  const activation=firstOutpostDestroyedTimes[targetCamp]; if(activation===null) return null; const enemy=oppositeCamp(targetCamp), candidates=[];
  data.tracks.filter(track=>track.key.camp===enemy&&fortressEligibleTypes.has(track.key.robot_type)).forEach(track=>{{ let progress=0,last=null,confirmed=false;
    for(let i=0;i<track.times.length;i++) {{ const time=track.times[i]; if(time<Math.max(180,activation)) continue; const alive=track.health[i]!==null&&track.health[i]>0;
      if(alive&&insideFortress(geofencePoint(track,i),targetCamp)) {{ if(last===null||time-last>4) {{ progress=0; confirmed=false; }} progress+=1; last=time; confirmed=confirmed||Boolean(track.vulnerable[i]); if(progress>=20&&confirmed) {{ candidates.push({{time,source:`${{enemy}}方${{unitNumber(track.key.robot_id)}}占领${{targetCamp}}方堡垒20秒`,confidence:'medium'}}); break; }} }}
    }}
  }}); return candidates.sort((a,b)=>a.time-b.time)[0]||null;
}}
const baseArmorStates={{}}; for(const camp of ['红','蓝']) {{ const candidates=[inferDartArmorOpen(camp),inferFortressArmorOpen(camp)].filter(Boolean).sort((a,b)=>a.time-b.time); baseArmorStates[camp]=candidates[0]||null; }}
function outpostRotationState(camp,now) {{
  const opponentArmor=baseArmorStates[oppositeCamp(camp)]?.time??Infinity, destroyed=firstOutpostDestroyedTimes[camp]??Infinity, stopTime=Math.min(180,destroyed,opponentArmor), effective=Math.min(now,stopTime), angle=effective<=5?Math.PI*.08*effective*effective:Math.PI*2+Math.PI*.8*(effective-5);
  if(now>=stopTime) {{ const reason=stopTime===destroyed?'首次被击毁':stopTime===opponentArmor?'对方基地护甲展开':'比赛进行3分钟'; return {{mode:'stopped',label:`停转·${{reason}}`,stopTime,angle}}; }}
  if(now<5) return {{mode:'accelerating',label:`加速旋转·${{(Math.max(0,now)/5*.8).toFixed(2)}}π rad/s`,stopTime,angle}};
  return {{mode:'rotating',label:'匀速旋转·0.8π rad/s·方向未知',stopTime,angle}};
}}
function fortressEffect(track,now,index) {{
  if(!fortressEligibleTypes.has(track.key.robot_type)||track.health[index]===null||track.health[index]<=0) return null; const own=track.key.camp, enemy=oppositeCamp(own);
  const point=geofencePoint(track,index);
  if(firstOutpostDestroyedTimes[own]!==null&&now>=firstOutpostDestroyedTimes[own]&&insideFortress(point,own)) {{ const base=data.objectives.find(item=>item.key.camp===own&&item.key.robot_type==='基地'), bi=frameIndex(base,now), max=bi>=0?base.max_health[bi]:null, past=bi>=0?base.health.slice(0,bi+1).filter(value=>value!==null):[]; const delta=max!==null&&past.length?Math.max(0,max-Math.min(...past)):0, cooling=Math.min(75,Math.floor(delta/40)); return {{kind:'amplified',zoneCamp:own,label:`己方堡垒增幅·50%防御·冷却+${{cooling}}`}}; }}
  const armor=baseArmorStates[enemy]; if(now>=180&&firstOutpostDestroyedTimes[enemy]!==null&&now>=firstOutpostDestroyedTimes[enemy]&&(!armor||now<armor.time)&&insideFortress(point,enemy)) return {{kind:'vulnerable',zoneCamp:enemy,label:`敌方堡垒虚弱·100%易伤${{track.vulnerable[index]?'·遥测确认':'·位置推定'}}`}};
  return null;
}}
document.getElementById('trackFilters').innerHTML = data.tracks.map((t,i)=>`<label class="item" style="--c:${{colors[t.key.camp] || '#ffd54a'}}"><input type="checkbox" data-track="${{i}}" checked>${{replayEscape(t.label)}}</label>`).join('');
document.getElementById('eventFilters').innerHTML = eventTypes.length ? '<span class="muted">事件</span>'+eventTypes.map(type=>`<label class="item" style="--c:${{eventColors[type] || '#f5f5f5'}}"><input type="checkbox" data-event="${{replayEscape(type)}}" ${{type==='发弹'?'':'checked'}}>${{replayEscape(type)}}</label>`).join('') : '';
document.getElementById('legend').innerHTML = '<span class="muted">箭头由攻击方指向受击方；青色分段环为前哨站旋转，橙色展开线为基地护甲，绿色/红色虚线区为堡垒增幅/虚弱；红色也表示普通受击和碎盾易伤。实线为高可信，虚线为中/低可信。</span>';
function entityId(camp,robotId) {{ return `${{camp}}-${{robotId}}`; }}
function frameIndex(track,now) {{ return track ? track.times.indexOf(Math.floor(Number(now)+1e-6)) : -1; }}
function poseAt(track,now,useRaw=false) {{
  const index=frameIndex(track,now); if(index<0) return null; const points=useRaw?track.raw_xy:track.clean_xy, point=points[index]; if(!point) return null;
  const next=index+1, fraction=Math.max(0,Math.min(1,Number(now)-track.times[index])); let interpolated=point, heading=track.heading_deg[index];
  const canInterpolate=fraction>0&&next<track.times.length&&track.times[next]===track.times[index]+1&&points[next]&&(useRaw||track.continuity[next]);
  if(canInterpolate) {{ interpolated=[point[0]+(points[next][0]-point[0])*fraction,point[1]+(points[next][1]-point[1])*fraction]; const nextHeading=track.heading_deg[next]; if(heading!==null&&nextHeading!==null) {{ const delta=(nextHeading-heading+540)%360-180; heading=heading+delta*fraction; }} }}
  return {{index,point:interpolated,heading,connected:canInterpolate}};
}}
function rosterMarkup(camp) {{
  const tracks=unitOrder.map(type=>data.tracks.find(track=>track.key.camp===camp&&track.key.robot_type===type)).filter(Boolean);
  const school=tracks[0]?.key.school||'';
  const units=tracks.map(track=>`<article id="unit-${{entityId(camp,track.key.robot_id)}}" class="unit-card"><div class="unit-head"><span class="unit-name"><span class="unit-number">${{unitNumber(track.key.robot_id)}}</span><span>${{replayEscape(track.key.robot_type)}}</span></span><span id="hp-text-${{entityId(camp,track.key.robot_id)}}">—</span></div><div class="hp-track"><div id="hp-fill-${{entityId(camp,track.key.robot_id)}}" class="hp-fill"></div></div><div id="respawn-${{entityId(camp,track.key.robot_id)}}" class="respawn-row"><span id="respawn-text-${{entityId(camp,track.key.robot_id)}}"></span><div class="respawn-track"><div id="respawn-fill-${{entityId(camp,track.key.robot_id)}}" class="respawn-fill"></div></div></div><div class="unit-stats"><span id="heat-${{entityId(camp,track.key.robot_id)}}">热量 —</span><span id="shots-${{entityId(camp,track.key.robot_id)}}">发弹 —</span></div><div class="score-row"><span id="kda-${{entityId(camp,track.key.robot_id)}}" class="kda-text">0/0/0</span><span id="dmg-${{entityId(camp,track.key.robot_id)}}" class="dmg-text">0</span><span class="score-label">评分</span><span id="score-text-${{entityId(camp,track.key.robot_id)}}" class="score-value">5.0</span><div class="score-track"><div id="score-fill-${{entityId(camp,track.key.robot_id)}}" class="score-fill mid" style="width:50%"></div></div></div><div id="status-${{entityId(camp,track.key.robot_id)}}" class="unit-status"></div></article>`).join('');
  const objectives=data.objectives.filter(track=>track.key.camp===camp).map(track=>`<article id="objective-card-${{entityId(camp,track.key.robot_id)}}" class="objective-card"><div class="objective-head"><span>${{replayEscape(track.key.robot_type)}}</span><span id="objective-hp-${{entityId(camp,track.key.robot_id)}}">—</span></div><div class="hp-track"><div id="objective-fill-${{entityId(camp,track.key.robot_id)}}" class="hp-fill"></div></div><div id="objective-status-${{entityId(camp,track.key.robot_id)}}" class="unit-status"></div></article>`).join('');
  return `<div class="roster-title"><div><strong>${{camp}}方</strong><span class="muted">${{replayEscape(school)}}</span></div><span id="team-score-${{camp}}" class="team-score">—</span></div>${{units}}<div class="objectives">${{objectives}}</div>`;
}}
document.getElementById('redRoster').innerHTML=rosterMarkup('红');
document.getElementById('blueRoster').innerHTML=rosterMarkup('蓝');
function heatState(track,index) {{
  const large=track.large_heat[index], largeLimit=track.large_heat_limit[index], small=track.small_heat[index], smallLimit=track.small_heat_limit[index];
  if(track.key.robot_type==='英雄' && large!==null && largeLimit>0) return {{value:large,limit:largeLimit,caliber:'42'}};
  if(small!==null && smallLimit>0) return {{value:small,limit:smallLimit,caliber:'17'}};
  return {{value:null,limit:null,caliber:''}};
}}
function activeBlind(camp,now) {{
  return data.dart_impacts.filter(item=>item.target_camp===camp&&item.time<=now&&now<item.blind_end).sort((a,b)=>b.blind_end-a.blind_end)[0]||null;
}}
function activeUavCountermeasure(camp,now) {{
  return data.events.filter(event=>event.event_type==='雷达反制UAV'&&event.camp===camp&&event.time<=now&&now<event.time+45).sort((a,b)=>b.time-a.time)[0]||null;
}}
function activeRespawn(track,now) {{
  return data.respawn_intervals.find(item=>item.camp===track.key.camp&&item.robot_id===track.key.robot_id&&item.death_time<=now&&now<item.end_time)||null;
}}
function respawnProgress(interval,now) {{
  if(!interval) return 0; const duration=interval.observed_duration||interval.required_progress||1; return Math.max(0,Math.min(1,(now-interval.death_time)/Math.max(1,duration)));
}}
function recentStatuses(track,now,index) {{
  const statuses=[]; if(track.vulnerable[index]) statuses.push('易伤');
  const fortress=fortressEffect(track,now,index); if(fortress) statuses.push(fortress.label);
  const blind=['基地','前哨站'].includes(track.key.robot_type)?null:activeBlind(track.key.camp,now);
  if(blind) statuses.push(`飞镖致盲·${{Math.ceil(blind.blind_end-now)}}s·${{blind.target_profile}}`);
  const suppressed=data.dart_impacts.filter(item=>item.target_camp===track.key.camp&&item.buff_suppression_duration>0&&item.time<=now&&now<item.buff_suppression_end).sort((a,b)=>b.buff_suppression_end-a.buff_suppression_end)[0];
  if(suppressed) statuses.push(`地形/机关增益失效·${{Math.ceil(suppressed.buff_suppression_end-now)}}s`);
  const activeBuffs=data.buff_intervals.filter(buff=>buff.camp===track.key.camp&&buff.robot_id===track.key.robot_id&&buff.start<=now&&now<buff.end);
  const byCategory=new Map(); activeBuffs.forEach(buff=>{{ const existing=byCategory.get(buff.category); if(!existing||buff.end>existing.end) byCategory.set(buff.category,buff); }});
  byCategory.forEach(buff=>{{ const remaining=Math.max(1,Math.ceil(buff.end-now)); statuses.push(`${{buff.category}}·${{remaining}}s`); }});
  const revival=data.paid_revivals.find(item=>item.camp===track.key.camp&&item.robot_id===track.key.robot_id&&item.time<=now&&now<item.time+4);
  if(revival) {{ const elapsed=now-revival.time; statuses.push(elapsed<3?`立即复活·无敌${{Math.ceil(3-elapsed)}}s`:`立即复活·功率强化${{Math.ceil(4-elapsed)}}s`); }}
  const countermeasure=track.key.robot_type==='空中'?activeUavCountermeasure(track.key.camp,now):null;
  if(countermeasure) statuses.push(`雷达反制·发射锁定 ${{Math.ceil(countermeasure.time+45-now)}}s`);
  if(data.events.some(event=>event.camp===track.key.camp&&event.robot_id===track.key.robot_id&&event.event_type==='受击'&&event.category==='判罚'&&event.time<=now&&now-event.time<=4)) statuses.push('判罚');
  return [...new Set(statuses)].join('  ');
}}
function updateRosters(now) {{
  data.tracks.forEach(track=>{{ const index=frameIndex(track,now); if(index<0) return; const id=entityId(track.key.camp,track.key.robot_id); const health=track.health[index], maxHealth=track.max_health[index]; const hpRatio=health!==null&&maxHealth>0?Math.max(0,Math.min(1,health/maxHealth)):0; const heat=heatState(track,index);
    document.getElementById(`hp-text-${{id}}`).textContent=health!==null&&maxHealth!==null?`${{Math.round(health)}} / ${{Math.round(maxHealth)}}`:'—'; document.getElementById(`hp-fill-${{id}}`).style.width=`${{hpRatio*100}}%`;
    document.getElementById(`heat-${{id}}`).textContent=heat.value!==null?`${{heat.caliber}}热 ${{Math.round(heat.value)}}/${{Math.round(heat.limit)}}`:'热量 —';
    const shots=track.key.robot_type==='英雄'?track.cumulative_42mm[index]:track.cumulative_17mm[index]; document.getElementById(`shots-${{id}}`).textContent=shots!==null?`发弹 ${{Math.round(shots)}}`:'发弹 —';
    const status=recentStatuses(track,now,index), card=document.getElementById(`unit-${{id}}`); document.getElementById(`status-${{id}}`).textContent=status; card.classList.toggle('vulnerable-card',Boolean(track.vulnerable[index])); card.classList.toggle('buyback-card',data.paid_revivals.some(item=>item.camp===track.key.camp&&item.robot_id===track.key.robot_id&&item.time<=now&&now<item.time+4)); card.classList.toggle('blind-card',Boolean(activeBlind(track.key.camp,now)));
    const death=activeRespawn(track,now), row=document.getElementById(`respawn-${{id}}`), fill=document.getElementById(`respawn-fill-${{id}}`);
    row.classList.toggle('active',Boolean(death));
    if(death) {{ const ratio=respawnProgress(death,now), remaining=Math.max(0,Math.ceil((death.revive_time||death.death_time+death.required_progress)-now)); fill.style.width=`${{ratio*100}}%`; fill.classList.toggle('paid',death.method==='paid'); document.getElementById(`respawn-text-${{id}}`).textContent=death.method==='paid'?`立即复活前 ${{remaining}}s`:(death.method==='timer'?`复活读条 ${{remaining}}s`:`复活读条 ≥${{remaining}}s`); }}
  }});
  data.objectives.forEach(track=>{{ const index=frameIndex(track,now); if(index<0) return; const id=entityId(track.key.camp,track.key.robot_id), health=track.health[index], maxHealth=track.max_health[index]; const ratio=health!==null&&maxHealth>0?Math.max(0,Math.min(1,health/maxHealth)):0; document.getElementById(`objective-hp-${{id}}`).textContent=health!==null&&maxHealth!==null?`${{Math.round(health)}} / ${{Math.round(maxHealth)}}`:'—'; document.getElementById(`objective-fill-${{id}}`).style.width=`${{ratio*100}}%`;
    const status=document.getElementById(`objective-status-${{id}}`); if(track.key.robot_type==='前哨站') status.textContent=outpostRotationState(track.key.camp,now).label; else {{ const armor=baseArmorStates[track.key.camp]; status.textContent=armor&&now>=armor.time?`护甲展开·${{armor.source}}`:'护甲闭合'; }}
    document.getElementById(`objective-card-${{id}}`).classList.toggle('dart-hit-card',data.dart_impacts.some(item=>item.target_robot_id===track.key.robot_id&&item.time<=now&&now<item.time+3)); }});
  updateScores(now);
}}
const teamEconomy={{}}; ['红','蓝'].forEach(camp=>{{ teamEconomy[camp]=data.tracks.find(track=>track.key.camp===camp)||data.objectives.find(track=>track.key.camp===camp); }});
const maxTotalCoins=Math.max(1,...Object.values(teamEconomy).flatMap(track=>(track?.total_coins||[]).filter(value=>value!==null)));
function addAssemblyMarkers() {{
  const lane=document.getElementById('totalEconomyLane'), markers={{'红':[],'蓝':[]}}; data.events.filter(event=>event.event_type==='装配成功'&&(event.camp==='红'||event.camp==='蓝')).forEach(event=>{{ const track=teamEconomy[event.camp], index=frameIndex(track,Math.round(event.time)); if(!track||index<0) return; const total=track.total_coins[index]; if(total===null) return; const marker=document.createElement('span'), direction=event.camp==='红'?-1:1, endpoint=50+direction*Math.min(50,total/maxTotalCoins*50); marker.className=`assembly-marker ${{event.camp==='红'?'red':'blue'}}`; marker.style.left=`${{endpoint}}%`; marker.style.display='none'; marker.dataset.camp=event.camp; marker.dataset.time=String(event.time); marker.dataset.endpoint=String(endpoint); marker.dataset.baseLabel=`${{Math.round(event.time)}}s ${{event.category||'装配'}}`; marker.dataset.label=marker.dataset.baseLabel; lane.appendChild(marker); markers[event.camp].push(marker); }});
  for(const camp of ['红','蓝']) {{ const ordered=markers[camp].sort((a,b)=>Number(a.dataset.endpoint)-Number(b.dataset.endpoint)); let cluster=[], clusterIndex=0; const flush=()=>{{ if(!cluster.length) return; cluster.forEach(marker=>marker.dataset.cluster=`${{camp}}-${{clusterIndex}}`); clusterIndex+=1; cluster=[]; }}; ordered.forEach(marker=>{{ if(cluster.length&&Math.abs(Number(marker.dataset.endpoint)-Number(cluster[cluster.length-1].dataset.endpoint))>7) flush(); cluster.push(marker); }}); flush(); }}
}}
addAssemblyMarkers();
function updateAssemblyMarkers(now) {{ const visibleByCluster=new Map(); document.querySelectorAll('.assembly-marker').forEach(marker=>{{ const visible=Number(marker.dataset.time)<=now; marker.style.display=visible?'block':'none'; marker.classList.add('label-hidden'); marker.dataset.label=marker.dataset.baseLabel; if(visible) {{ if(!visibleByCluster.has(marker.dataset.cluster)) visibleByCluster.set(marker.dataset.cluster,[]); visibleByCluster.get(marker.dataset.cluster).push(marker); }} }}); visibleByCluster.forEach(markers=>{{ const camp=markers[0].dataset.camp, ordered=markers.sort((a,b)=>Number(a.dataset.endpoint)-Number(b.dataset.endpoint)), anchor=camp==='红'?ordered[ordered.length-1]:ordered[0]; anchor.dataset.label=ordered.sort((a,b)=>Number(a.dataset.time)-Number(b.dataset.time)).map(marker=>marker.dataset.baseLabel).join(' · '); anchor.classList.remove('label-hidden'); }}); }}

function updateScores(now) {{
  if (!data.timeseries_scores) return;
  const camps = ['红','蓝'];
  camps.forEach(camp => {{
    const entries = data.timeseries_scores[camp] || [];
    let teamTotal = 0, ratedCount = 0;
    entries.forEach(entry => {{
      const times = entry.times;
      const scores = entry.score;
      if (!times || !scores || times.length === 0) return;
      // Binary search for current time
      let idx = 0;
      for (let i = times.length - 1; i >= 0; i--) {{
        if (times[i] <= now + 0.5) {{ idx = i; break; }}
      }}
      const score = scores[idx];
      const id = entityId(entry.camp, entry.robot_id);
      const textEl = document.getElementById(`score-text-${{id}}`);
      const kdaEl = document.getElementById(`kda-${{id}}`);
      if (kdaEl && entry.kills !== undefined) {{ kdaEl.textContent = `${{entry.kills[idx]}}/${{entry.assists[idx]}}/${{entry.deaths[idx]}}`;
      const dmgEl = document.getElementById(`dmg-${{id}}`); if (dmgEl && entry.damage !== undefined) {{ dmgEl.textContent = Math.round(entry.damage[idx]); }} }}
      const fillEl = document.getElementById(`score-fill-${{id}}`);
      if (textEl) {{
        textEl.textContent = score===null ? '未评级' : score.toFixed(1);
        if (score >= 7.5) textEl.style.color = '#69f0ae';
        else if (score >= 5.5) textEl.style.color = '#ffd54a';
        else textEl.style.color = '#ff5252';
      }}
      if (fillEl) {{
        fillEl.style.width = `${{score===null ? 0 : score * 10}}%`;
        fillEl.classList.remove('high', 'mid', 'low');
        if (score >= 7.5) fillEl.classList.add('high');
        else if (score >= 5.5) fillEl.classList.add('mid');
        else fillEl.classList.add('low');
      }}
      if(score!==null) {{ teamTotal += score; ratedCount++; }}
    }});
    const teamEl = document.getElementById(`team-score-${{camp}}`);
    if (teamEl) teamEl.textContent = ratedCount ? `均分 ${{(teamTotal/ratedCount).toFixed(1)}} · ${{ratedCount}}个` : '暂无评级';
  }});
}}
function updateEconomy(now) {{
  for(const camp of ['红','蓝']) {{ const track=teamEconomy[camp], index=frameIndex(track,now); if(!track||index<0) continue; const total=track.total_coins[index]??0, remaining=track.remaining_coins[index]??0; const side=camp==='红'?'red':'blue'; document.getElementById(`${{side}}TotalBar`).style.width=`${{Math.min(50,total/maxTotalCoins*50)}}%`; document.getElementById(`${{side}}RemainingBar`).style.width=`${{Math.min(50,remaining/maxTotalCoins*50)}}%`; document.getElementById(`${{side}}TotalLabel`).textContent=`${{camp}} 总经济 ${{Math.round(total)}}`; document.getElementById(`${{side}}RemainingLabel`).textContent=`${{camp}} 现有 ${{Math.round(remaining)}}`; }}
  updateAssemblyMarkers(now);
}}
function resize() {{ const r=canvas.getBoundingClientRect(), dpr=devicePixelRatio||1; canvas.width=Math.round(r.width*dpr); canvas.height=Math.round(r.height*dpr); draw(); }}
function xy(p) {{ return [p[0]/data.field.width*canvas.width, (1-p[1]/data.field.height)*canvas.height]; }}
function unitNumber(robotId) {{ const value=Number(robotId); return value>100 ? value%100 : value; }}
function drawBlindBadge(cx,cy,dpr) {{
  const radius=8*dpr; ctx.save(); ctx.fillStyle='rgba(49,27,70,.96)'; ctx.strokeStyle='#c084ff'; ctx.lineWidth=1.6*dpr; ctx.beginPath(); ctx.arc(cx,cy,radius,0,Math.PI*2); ctx.fill(); ctx.stroke();
  ctx.strokeStyle='#f5eaff'; ctx.lineWidth=1.5*dpr; ctx.lineCap='round'; ctx.beginPath(); ctx.moveTo(cx-5*dpr,cy); ctx.quadraticCurveTo(cx,cy-4*dpr,cx+5*dpr,cy); ctx.quadraticCurveTo(cx,cy+4*dpr,cx-5*dpr,cy); ctx.stroke(); ctx.beginPath(); ctx.moveTo(cx-5*dpr,cy-5*dpr); ctx.lineTo(cx+5*dpr,cy+5*dpr); ctx.stroke(); ctx.restore();
}}
function drawVulnerableBadge(cx,cy,dpr) {{
  ctx.save(); ctx.fillStyle='rgba(80,8,16,.96)'; ctx.strokeStyle='#ff3b4f'; ctx.lineWidth=1.7*dpr; ctx.lineJoin='round'; ctx.beginPath(); ctx.moveTo(cx,cy-8*dpr); ctx.lineTo(cx+7*dpr,cy-5*dpr); ctx.lineTo(cx+6*dpr,cy+1*dpr); ctx.quadraticCurveTo(cx+4*dpr,cy+6*dpr,cx,cy+8*dpr); ctx.quadraticCurveTo(cx-4*dpr,cy+6*dpr,cx-6*dpr,cy+1*dpr); ctx.lineTo(cx-7*dpr,cy-5*dpr); ctx.closePath(); ctx.fill(); ctx.stroke();
  ctx.strokeStyle='#ffd4d9'; ctx.lineWidth=1.8*dpr; ctx.beginPath(); ctx.moveTo(cx+1*dpr,cy-7*dpr); ctx.lineTo(cx-2*dpr,cy-1*dpr); ctx.lineTo(cx+2*dpr,cy+1*dpr); ctx.lineTo(cx-2*dpr,cy+7*dpr); ctx.stroke(); ctx.restore();
}}
function drawRobotStatusBadges(track,index,x,y,radius,dpr,now) {{
  if(activeRespawn(track,now)) return;
  const badges=[]; if(!['基地','前哨站'].includes(track.key.robot_type)&&activeBlind(track.key.camp,now)) badges.push('blind'); if(track.vulnerable[index]) badges.push('vulnerable'); if(!badges.length) return;
  const gap=18*dpr, start=x-(badges.length-1)*gap/2, cy=y-radius-10*dpr; badges.forEach((kind,i)=>{{ const cx=start+i*gap; if(kind==='blind') drawBlindBadge(cx,cy,dpr); else drawVulnerableBadge(cx,cy,dpr); }});
}}
function drawRobotIcon(track,index,point=track.clean_xy[index],headingOverride=track.heading_deg[index],now=track.times[index]) {{
  if(!point) return; const [x,y]=xy(point); const dpr=devicePixelRatio||1;
  const radius=11*dpr; const color=colors[track.key.camp] || '#ffd54a';
  const health=track.health[index], maxHealth=track.max_health[index];
  const death=activeRespawn(track,now), ratio=death?respawnProgress(death,now):((health!==null && maxHealth!==null && maxHealth>0) ? Math.max(0,Math.min(1,health/maxHealth)) : 0);
  ctx.save(); ctx.globalAlpha=1;
  if(death) {{ ctx.shadowColor='#ff334d'; ctx.shadowBlur=(10+4*Math.sin((now-death.death_time)*5))*dpr; }} else if(track.vulnerable[index]) {{ ctx.shadowColor='#ff334d'; ctx.shadowBlur=14*dpr; }}
  ctx.fillStyle='rgba(20,20,20,.88)'; ctx.beginPath(); ctx.arc(x,y,radius,0,Math.PI*2); ctx.fill();
  ctx.beginPath(); ctx.arc(x,y,radius-1*dpr,0,Math.PI*2); ctx.clip();
  ctx.fillStyle=death?'#40e878':color; ctx.globalAlpha=0.92; ctx.fillRect(x-radius,y+radius-2*radius*ratio,2*radius,2*radius*ratio);
  ctx.restore();
  ctx.save(); ctx.globalAlpha=1; ctx.strokeStyle=color; ctx.lineWidth=2*dpr; ctx.beginPath(); ctx.arc(x,y,radius,0,Math.PI*2); ctx.stroke();
  const heat=heatState(track,index), heatRatio=!death&&heat.value!==null&&heat.limit>0?Math.max(0,Math.min(1,heat.value/heat.limit)):0, heatRadius=15.5*dpr;
  ctx.strokeStyle='rgba(255,255,255,.24)'; ctx.lineWidth=2.5*dpr; ctx.beginPath(); ctx.arc(x,y,heatRadius,0,Math.PI*2); ctx.stroke();
  if(heatRatio>0) {{ ctx.strokeStyle=heatRatio>.85?'#ff5252':'#ffb74d'; ctx.lineWidth=3.2*dpr; ctx.lineCap='round'; ctx.beginPath(); ctx.arc(x,y,heatRadius,-Math.PI/2,-Math.PI/2-Math.PI*2*heatRatio,true); ctx.stroke(); }}
  const heading=headingOverride;
  if(!death&&heading!==null) {{ const rad=heading*Math.PI/180; ctx.strokeStyle='#fff'; ctx.lineWidth=2*dpr; ctx.beginPath(); ctx.moveTo(x+Math.cos(rad)*radius*.55,y-Math.sin(rad)*radius*.55); ctx.lineTo(x+Math.cos(rad)*radius*1.65,y-Math.sin(rad)*radius*1.65); ctx.stroke(); }}
  ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.font=`500 ${{13*dpr}}px system-ui,sans-serif`; ctx.lineWidth=3*dpr; ctx.strokeStyle='rgba(0,0,0,.82)';
  const countermeasure=track.key.robot_type==='空中'?activeUavCountermeasure(track.key.camp,now):null;
  if(death) {{ ctx.strokeStyle='rgba(35,0,0,.9)'; ctx.lineWidth=5*dpr; ctx.lineCap='round'; ctx.beginPath(); ctx.moveTo(x-radius*.62,y-radius*.62); ctx.lineTo(x+radius*.62,y+radius*.62); ctx.moveTo(x+radius*.62,y-radius*.62); ctx.lineTo(x-radius*.62,y+radius*.62); ctx.stroke(); ctx.strokeStyle='#ff334d'; ctx.lineWidth=2.5*dpr; ctx.stroke(); }}
  else if(countermeasure) {{ ctx.strokeStyle='rgba(25,0,0,.9)'; ctx.lineWidth=5*dpr; ctx.lineCap='round'; ctx.beginPath(); ctx.moveTo(x-radius*.58,y-radius*.58); ctx.lineTo(x+radius*.58,y+radius*.58); ctx.moveTo(x+radius*.58,y-radius*.58); ctx.lineTo(x-radius*.58,y+radius*.58); ctx.stroke(); ctx.strokeStyle='#ff4055'; ctx.lineWidth=2.4*dpr; ctx.stroke(); }}
  else {{ const label=String(unitNumber(track.key.robot_id)); ctx.strokeText(label,x,y); ctx.fillStyle='#fff'; ctx.fillText(label,x,y); }} ctx.restore(); drawRobotStatusBadges(track,index,x,y,radius,dpr,now);
}}
function drawTrack(track, trackIndex, now, useRaw, showIcon=true) {{
  if(!visibleTracks.has(trackIndex)) return;
  const points = useRaw ? track.raw_xy : track.clean_xy; const color=colors[track.key.camp] || '#ffd54a';
  const trail=Math.max(1,Number(trailInput.value)||20); ctx.strokeStyle=color; ctx.lineWidth=2*(devicePixelRatio||1); ctx.globalAlpha=useRaw?0.28:0.9;
  ctx.beginPath(); let active=false;
  for(let i=0;i<track.times.length;i++) {{
    if(track.times[i] > now) break;
    if(track.times[i] < now-trail || !points[i]) {{ active=false; continue; }}
    const [x,y]=xy(points[i]); const connected=useRaw || i===0 || track.continuity[i];
    if(!active || !connected) ctx.moveTo(x,y); else ctx.lineTo(x,y); active=true;
  }}
  const pose=poseAt(track,now,useRaw); if(pose&&Number(now)%1!==0) {{ const [x,y]=xy(pose.point); if(active&&pose.connected) ctx.lineTo(x,y); else ctx.moveTo(x,y); }} ctx.stroke();
  if(showIcon && !useRaw && pose) drawRobotIcon(track,pose.index,pose.point,pose.heading,now);
}}
function eventPosition(event, now) {{
  if(event.robot_id===null) return null;
  const track=allEntities.find((track)=>track.key.robot_id===event.robot_id && track.key.camp===event.camp);
  if(!track) return null; const mobileIndex=data.tracks.indexOf(track); if(mobileIndex>=0&&!visibleTracks.has(mobileIndex)) return null; return poseAt(track,now,false)?.point||null;
}}
function drawObjectiveHud(now) {{
  const dpr=devicePixelRatio||1;
  data.objectives.forEach(track=>{{ const pose=poseAt(track,now,false); if(!pose) return; const [x,y]=xy(pose.point), index=pose.index, health=track.health[index], maxHealth=track.max_health[index]; if(health===null||maxHealth===null||maxHealth<=0) return;
    const ratio=Math.max(0,Math.min(1,health/maxHealth)), width=(track.key.robot_type==='基地'?72:58)*dpr, height=7*dpr, top=y-(track.key.robot_type==='基地'?32:25)*dpr, left=x-width/2, color=colors[track.key.camp]||'#fff';
    ctx.save(); ctx.globalAlpha=.96; ctx.fillStyle='rgba(10,10,10,.9)'; ctx.fillRect(left-2*dpr,top-2*dpr,width+4*dpr,height+4*dpr); ctx.fillStyle='#383838'; ctx.fillRect(left,top,width,height); ctx.fillStyle=color; ctx.fillRect(left,top,width*ratio,height); ctx.strokeStyle='#f2f2f2'; ctx.lineWidth=1*dpr; ctx.strokeRect(left,top,width,height);
    ctx.textAlign='center'; ctx.textBaseline='bottom'; ctx.font=`600 ${{10*dpr}}px system-ui,sans-serif`; ctx.lineWidth=3*dpr; ctx.strokeStyle='rgba(0,0,0,.92)'; const label=`${{track.key.camp}}方${{replayEscape(track.key.robot_type)}} ${{Math.round(health)}}/${{Math.round(maxHealth)}}`; ctx.strokeText(label,x,top-3*dpr); ctx.fillStyle='#fff'; ctx.fillText(label,x,top-3*dpr); ctx.restore();
  }});
}}
function drawArenaMechanics(now) {{
  const dpr=devicePixelRatio||1, details=[];
  data.objectives.filter(track=>track.key.robot_type==='前哨站').forEach(track=>{{ const pose=poseAt(track,now,false); if(!pose) return; const [x,y]=xy(pose.point), state=outpostRotationState(track.key.camp,now), radius=20*dpr; ctx.save(); ctx.translate(x,y); ctx.rotate(state.angle); ctx.strokeStyle=state.mode==='stopped'?'#ff5268':'#5de7ff'; ctx.lineWidth=2.5*dpr; ctx.lineCap='round'; for(let segment=0;segment<4;segment++) {{ ctx.beginPath(); ctx.arc(0,0,radius,segment*Math.PI/2,segment*Math.PI/2+Math.PI*.28); ctx.stroke(); }} ctx.restore();
    ctx.save(); ctx.font=`600 ${{9*dpr}}px system-ui,sans-serif`; ctx.textAlign='center'; ctx.textBaseline='top'; ctx.lineWidth=3*dpr; ctx.strokeStyle='rgba(0,0,0,.9)'; const short=state.mode==='stopped'?'前哨站·停转':state.mode==='accelerating'?'前哨站·加速':'前哨站·旋转'; ctx.strokeText(short,x,y+22*dpr); ctx.fillStyle=state.mode==='stopped'?'#ff9aa8':'#b8f5ff'; ctx.fillText(short,x,y+22*dpr); ctx.restore();
  }});
  for(const camp of ['红','蓝']) {{ const armor=baseArmorStates[camp]; if(!armor||now<armor.time) continue; const base=data.objectives.find(track=>track.key.camp===camp&&track.key.robot_type==='基地'), pose=poseAt(base,now,false); if(!pose) continue; const [x,y]=xy(pose.point), phase=Math.max(0,now-armor.time), pulse=Math.max(0,1-phase/3); ctx.save(); ctx.strokeStyle='#ffad42'; ctx.lineWidth=(2.2+pulse*1.8)*dpr; ctx.globalAlpha=.8+pulse*.2; for(let panel=0;panel<6;panel++) {{ const a=panel*Math.PI/3, inner=20*dpr, outer=(29+pulse*7)*dpr; ctx.beginPath(); ctx.moveTo(x+Math.cos(a-.13)*inner,y+Math.sin(a-.13)*inner); ctx.lineTo(x+Math.cos(a)*outer,y+Math.sin(a)*outer); ctx.lineTo(x+Math.cos(a+.13)*inner,y+Math.sin(a+.13)*inner); ctx.stroke(); }} ctx.restore(); if(phase<3) details.push(`${{armor.time.toFixed(1)}}s ${{camp}}方基地护甲展开（${{armor.source}}，${{armor.confidence==='high'?'高':'中'}}可信）`); }}
  const occupied=new Map(); data.tracks.forEach((track,i)=>{{ if(!visibleTracks.has(i)) return; const index=frameIndex(track,now); if(index<0) return; const effect=fortressEffect(track,now,index); if(!effect) return; const key=`${{effect.zoneCamp}}-${{effect.kind}}`; if(!occupied.has(key)) occupied.set(key,{{...effect,units:[]}}); occupied.get(key).units.push(unitNumber(track.key.robot_id)); }});
  occupied.forEach(item=>{{ const center=data.field.fortresses[item.zoneCamp], [x,y]=xy(center), radius=data.field.fortress_radius/data.field.width*canvas.width; ctx.save(); ctx.globalAlpha=.92; ctx.strokeStyle=item.kind==='amplified'?'#69f0ae':'#ff4055'; ctx.fillStyle=item.kind==='amplified'?'rgba(105,240,174,.12)':'rgba(255,64,85,.12)'; ctx.lineWidth=2.4*dpr; ctx.setLineDash([6*dpr,4*dpr]); ctx.beginPath(); ctx.arc(x,y,radius,0,Math.PI*2); ctx.fill(); ctx.stroke(); ctx.setLineDash([]); ctx.font=`600 ${{10*dpr}}px system-ui,sans-serif`; ctx.textAlign='center'; ctx.textBaseline='bottom'; ctx.lineWidth=3*dpr; ctx.strokeStyle='rgba(0,0,0,.9)'; const label=`${{item.kind==='amplified'?'堡垒增幅':'堡垒虚弱'}} · ${{item.units.join('/')}}`; ctx.strokeText(label,x,y-radius-3*dpr); ctx.fillStyle='#fff'; ctx.fillText(label,x,y-radius-3*dpr); ctx.restore(); }});
  return details;
}}
function drawEvents(now) {{
  const current=data.events.filter(event=>Math.round(event.time)===Math.floor(now+1e-6) && visibleEvents.has(event.event_type));
  current.forEach((event,index)=>{{ const point=eventPosition(event,now); if(!point) return; const [x,y]=xy(point), dpr=devicePixelRatio||1; ctx.save(); ctx.globalAlpha=1; ctx.strokeStyle=eventColors[event.event_type]||'#f5f5f5'; ctx.lineWidth=(event.event_type==='受击'?3:2)*dpr; const radius=(9+index%3*4)*dpr; ctx.beginPath(); ctx.arc(x,y,radius,0,Math.PI*2); ctx.stroke(); if(event.event_type==='受击') {{ ctx.globalAlpha=.72; for(let ray=0;ray<6;ray++) {{ const a=ray*Math.PI/3+now*.8; ctx.beginPath(); ctx.moveTo(x+Math.cos(a)*radius*.75,y+Math.sin(a)*radius*.75); ctx.lineTo(x+Math.cos(a)*radius*1.45,y+Math.sin(a)*radius*1.45); ctx.stroke(); }} }} ctx.restore(); }});
  return current.filter(event=>event.event_type!=='飞镖命中').slice(0,5).map(event=>`${{event.time.toFixed(1)}}s ${{event.camp||''}}方 ${{event.robot_type||''}} ${{event.event_type}}${{event.category?'·'+event.category:''}}${{event.value!==null?' ('+event.value+')':''}}`);
}}
function drawHeavyObjectiveHits(now) {{
  const dpr=devicePixelRatio||1, active=data.events.filter(event=>event.event_type==='受击'&&event.category==='42mm'&&(event.robot_type==='基地'||event.robot_type==='前哨站')&&event.time<=now&&now-event.time<2), groups=new Map();
  active.forEach(event=>{{ const key=`${{event.camp}}-${{event.robot_id}}`; if(!groups.has(key)) groups.set(key,[]); groups.get(key).push(event); }});
  groups.forEach(events=>{{ const latest=events.reduce((a,b)=>a.time>b.time?a:b), point=eventPosition(latest,now); if(!point) return; const [x,y]=xy(point), phase=now-latest.time, count=events.length; ctx.save(); ctx.globalAlpha=Math.max(.25,1-phase/2); ctx.strokeStyle='#ffd740'; ctx.lineWidth=2.6*dpr; ctx.setLineDash([7*dpr,4*dpr]); for(let ring=0;ring<Math.min(5,count);ring++) {{ ctx.beginPath(); ctx.arc(x,y,(22+ring*7+phase*8)*dpr,0,Math.PI*2); ctx.stroke(); }} ctx.setLineDash([]); ctx.font=`700 ${{11*dpr}}px system-ui,sans-serif`; ctx.textAlign='center'; ctx.textBaseline='top'; ctx.lineWidth=3*dpr; ctx.strokeStyle='rgba(0,0,0,.9)'; const label=`42mm重击 ×${{count}}`; ctx.strokeText(label,x,y+26*dpr); ctx.fillStyle='#fff19a'; ctx.fillText(label,x,y+26*dpr); ctx.restore(); }});
}}
function drawDamageNumbers(now) {{
  if(!damageToggle.checked) return; const dpr=devicePixelRatio||1, recent=data.events.filter(event=>event.event_type==='受击'&&event.value!==null&&event.value<0&&event.time<=now&&now-event.time<1.35), groups=new Map();
  recent.forEach(event=>{{ const key=`${{event.camp}}-${{event.robot_id}}`; if(!groups.has(key)) groups.set(key,[]); groups.get(key).push(event); }});
  groups.forEach(events=>{{ const latest=events.reduce((a,b)=>a.time>b.time?a:b), point=eventPosition(latest,now); if(!point) return; const [x,y]=xy(point), sum=events.reduce((total,event)=>total+Math.abs(Number(event.value)||0),0), count=events.length, phase=now-latest.time, progress=Math.min(1,phase/1.35), bounce=Math.abs(Math.sin(progress*Math.PI*2.4))*(1-progress), roll=count>1?Math.sin(progress*Math.PI*5)*4*dpr:0, drawX=x+roll, drawY=y-(28+progress*22+bounce*12)*dpr, alpha=Math.max(.18,1-progress);
    ctx.save(); ctx.globalAlpha=alpha; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.font=`800 ${{(count>1?18:15)*dpr}}px system-ui,sans-serif`; ctx.lineWidth=4*dpr; ctx.strokeStyle='rgba(20,0,0,.92)'; const label=`-${{Math.round(sum)}}`; ctx.strokeText(label,drawX,drawY); ctx.fillStyle=count>1?'#ffcf5a':'#ff5268'; ctx.fillText(label,drawX,drawY); ctx.restore();
  }});
}}
function drawDartEffects(now) {{
  const current=data.dart_impacts.filter(item=>item.time<=now&&now<item.time+3);
  const details=[]; current.forEach(item=>{{ const target=data.objectives.find(track=>track.key.robot_id===item.target_robot_id); if(!target) return; const index=frameIndex(target,Math.round(item.time)); if(index<0||!target.clean_xy[index]) return; const [x,y]=xy(target.clean_xy[index]), dpr=devicePixelRatio||1, phase=now-item.time, radius=(18+phase*10)*dpr;
    ctx.save(); ctx.globalAlpha=Math.max(.3,1-phase*.25); ctx.strokeStyle='#ff4da6'; ctx.fillStyle='rgba(255,77,166,.16)'; ctx.lineWidth=3*dpr; ctx.beginPath(); ctx.arc(x,y,radius,0,Math.PI*2); ctx.fill(); ctx.stroke();
    ctx.strokeStyle='#ffd54a'; ctx.lineWidth=2*dpr; for(let ray=0;ray<8;ray++) {{ const angle=ray*Math.PI/4; ctx.beginPath(); ctx.moveTo(x+Math.cos(angle)*radius*.45,y+Math.sin(angle)*radius*.45); ctx.lineTo(x+Math.cos(angle)*radius*1.25,y+Math.sin(angle)*radius*1.25); ctx.stroke(); }}
    ctx.textAlign='center'; ctx.textBaseline='bottom'; ctx.font=`500 ${{12*dpr}}px system-ui,sans-serif`; ctx.lineWidth=3*dpr; ctx.strokeStyle='rgba(0,0,0,.85)'; const label=`${{item.target_profile}} · -${{Math.round(item.damage)}} · 致盲${{Math.round(item.blind_duration)}}s`; ctx.strokeText(label,x,y-radius-5*dpr); ctx.fillStyle='#fff'; ctx.fillText(label,x,y-radius-5*dpr); ctx.restore();
    details.push(`${{item.time.toFixed(1)}}s ${{item.attacker_camp}}方飞镖命中${{item.target_camp}}方${{item.target_type}} · ${{item.target_profile}} · ${{Math.round(item.damage)}}伤害 · 致盲${{Math.round(item.blind_duration)}}s${{item.buff_suppression_duration>0?' · 地形/机关增益失效'+Math.round(item.buff_suppression_duration)+'s':''}}`);
  }}); return details;
}}
function drawRevivalEffects(now) {{
  const current=data.paid_revivals.filter(item=>item.time<=now&&now<item.time+4); const details=[];
  current.forEach(item=>{{ const track=data.tracks.find(track=>track.key.camp===item.camp&&track.key.robot_id===item.robot_id); if(!track) return; const index=frameIndex(track,Math.round(item.time)); if(index<0||!track.clean_xy[index]) return; const [x,y]=xy(track.clean_xy[index]), dpr=devicePixelRatio||1, phase=now-item.time, radius=(20+phase*5)*dpr;
    ctx.save(); ctx.globalAlpha=Math.max(.35,1-phase*.16); ctx.strokeStyle='#69f0ae'; ctx.lineWidth=3*dpr; ctx.beginPath(); ctx.arc(x,y,radius,0,Math.PI*2); ctx.stroke(); ctx.strokeStyle='#4dd5ff'; ctx.lineWidth=2*dpr; ctx.beginPath(); ctx.arc(x,y,radius+6*dpr,-Math.PI*.8,Math.PI*.35); ctx.stroke();
    ctx.fillStyle='rgba(105,240,174,.18)'; ctx.beginPath(); ctx.arc(x,y,radius*.72,0,Math.PI*2); ctx.fill(); ctx.textAlign='center'; ctx.textBaseline='bottom'; ctx.font=`500 ${{12*dpr}}px system-ui,sans-serif`; ctx.lineWidth=3*dpr; ctx.strokeStyle='rgba(0,0,0,.85)'; const label=`立即复活 · ${{item.camp}}${{unitNumber(item.robot_id)}}`; ctx.strokeText(label,x,y-radius-5*dpr); ctx.fillStyle='#e8fff4'; ctx.fillText(label,x,y-radius-5*dpr); ctx.restore();
    details.push(`${{item.time.toFixed(1)}}s 立即复活：${{item.camp}}${{unitNumber(item.robot_id)}} · 满血复活 · 金币下降${{Math.round(item.observed_coin_drop)}}`);
  }}); return details;
}}
function drawDeployedHeroEffects(now) {{
  const current=data.attacks.filter(item=>item.special_mode==='hero_deployed_lob'&&item.hit_time<=now&&now<item.hit_time+3), details=[];
  current.forEach(item=>{{ const [x,y]=xy(item.victim_xy), dpr=devicePixelRatio||1, phase=now-item.hit_time, fade=Math.max(.18,1-phase/3), radius=(15+phase*16)*dpr; ctx.save(); ctx.globalAlpha=fade;
    const glow=ctx.createRadialGradient(x,y,0,x,y,radius); glow.addColorStop(0,'rgba(255,255,210,.95)'); glow.addColorStop(.28,'rgba(255,175,35,.72)'); glow.addColorStop(1,'rgba(255,86,16,0)'); ctx.fillStyle=glow; ctx.beginPath(); ctx.arc(x,y,radius,0,Math.PI*2); ctx.fill();
    ctx.strokeStyle='#ffc928'; ctx.lineWidth=3*dpr; ctx.beginPath(); ctx.arc(x,y,radius*.72,0,Math.PI*2); ctx.stroke(); ctx.strokeStyle='#fff2a8'; ctx.lineWidth=2*dpr; for(let ray=0;ray<12;ray++) {{ const angle=ray*Math.PI/6+phase*.18, inner=radius*.28, outer=radius*(.85+(ray%3)*.14); ctx.beginPath(); ctx.moveTo(x+Math.cos(angle)*inner,y+Math.sin(angle)*inner); ctx.lineTo(x+Math.cos(angle)*outer,y+Math.sin(angle)*outer); ctx.stroke(); }}
    ctx.textAlign='center'; ctx.textBaseline='bottom'; ctx.font=`500 ${{12*dpr}}px system-ui,sans-serif`; ctx.lineWidth=3*dpr; ctx.strokeStyle='rgba(0,0,0,.86)'; const label=`部署吊射命中 · -${{Math.round(item.damage||0)}}`; ctx.strokeText(label,x,y-radius-5*dpr); ctx.fillStyle='#fff7c4'; ctx.fillText(label,x,y-radius-5*dpr); ctx.restore();
    if(Math.round(item.hit_time)!==now) details.push(`${{item.hit_time.toFixed(1)}}s 英雄部署吊射命中${{item.victim_camp}}方基地 · ${{Math.round(item.damage||0)}}伤害`);
  }}); return details;
}}
function attackTracksVisible(attack) {{
  const attacker=data.tracks.findIndex((t,i)=>visibleTracks.has(i)&&t.key.camp===attack.attacker_camp&&t.key.robot_id===attack.attacker_robot_id);
  const mobileVictim=data.tracks.findIndex((t,i)=>visibleTracks.has(i)&&t.key.camp===attack.victim_camp&&t.key.robot_id===attack.victim_robot_id);
  const objectiveVictim=data.objectives.some(t=>t.key.camp===attack.victim_camp&&t.key.robot_id===attack.victim_robot_id);
  return attacker>=0 && (mobileVictim>=0 || objectiveVictim);
}}
function drawArrow(attack) {{
  const dpr=devicePixelRatio||1; const raw1=xy(attack.attacker_xy), raw2=xy(attack.victim_xy);
  const dx=raw2[0]-raw1[0], dy=raw2[1]-raw1[1], length=Math.hypot(dx,dy); if(length<36*dpr) return;
  const ux=dx/length, uy=dy/length; const startTrim=15*dpr, endTrim=19*dpr;
  const x1=raw1[0]+ux*startTrim, y1=raw1[1]+uy*startTrim, x2=raw2[0]-ux*endTrim, y2=raw2[1]-uy*endTrim;
  const ballistic=attack.attacker_type==='英雄'&&attack.caliber==='42mm'&&(attack.victim_type==='基地'||attack.victim_type==='前哨站');
  const controlX=(x1+x2)/2, controlY=(y1+y2)/2-Math.min(90*dpr,length*.24);
  const angle=ballistic?Math.atan2(y2-controlY,x2-controlX):Math.atan2(y2-y1,x2-x1); const deployed=attack.special_mode==='hero_deployed_lob', color=deployed?'#ffc928':attackColors[attack.confidence];
  const weight=1.7+Math.min(3.2,Math.log2(1+attack.continuous_count)*.9+Math.log2(1+attack.burst_count)*.35);
  const lineWidth=weight*dpr, head=(7.5+weight*1.7)*dpr; const dashed=attack.confidence!=='high';
  ctx.save(); ctx.globalAlpha=.9; ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=lineWidth; ctx.lineCap='round'; ctx.lineJoin='round'; ctx.setLineDash(dashed?[7*dpr,5*dpr]:[]);
  ctx.beginPath(); ctx.moveTo(x1,y1); if(ballistic) ctx.quadraticCurveTo(controlX,controlY,x2,y2); else ctx.lineTo(x2,y2); ctx.stroke(); ctx.setLineDash([]);
  ctx.beginPath(); ctx.moveTo(x2,y2); ctx.lineTo(x2-head*Math.cos(angle-Math.PI/6),y2-head*Math.sin(angle-Math.PI/6)); ctx.lineTo(x2-head*Math.cos(angle+Math.PI/6),y2-head*Math.sin(angle+Math.PI/6)); ctx.closePath(); ctx.fill();
  if(deployed) {{ ctx.strokeStyle='#fff2a8'; ctx.lineWidth=2*dpr; for(let ring=0;ring<2;ring++) {{ ctx.beginPath(); ctx.arc(raw2[0],raw2[1],(15+ring*7)*dpr,0,Math.PI*2); ctx.stroke(); }} }}
  ctx.fillStyle=colors[attack.attacker_camp]||color; ctx.beginPath(); ctx.arc(x1,y1,(3.2+weight*.35)*dpr,0,Math.PI*2); ctx.fill();
  if(length>100*dpr) {{
    const victimLabel=(attack.victim_type==='基地'||attack.victim_type==='前哨站')?`${{attack.victim_camp}}${{attack.victim_type}}`:`${{attack.victim_camp}}${{unitNumber(attack.victim_robot_id)}}`;
    const label=deployed?`部署吊射推定 · -${{Math.round(attack.damage||0)}}`:`${{attack.attacker_camp}}${{unitNumber(attack.attacker_robot_id)}} → ${{victimLabel}}${{attack.continuous_count>1?' ×'+attack.continuous_count:''}}`;
    const offset=10*dpr, baseX=ballistic?(.25*x1+.5*controlX+.25*x2):(x1+x2)/2, baseY=ballistic?(.25*y1+.5*controlY+.25*y2):(y1+y2)/2, midX=baseX-uy*offset, midY=baseY+ux*offset;
    ctx.font=`500 ${{11*dpr}}px system-ui,sans-serif`; ctx.textAlign='center'; ctx.textBaseline='middle'; const textWidth=ctx.measureText(label).width; const padX=5*dpr, boxH=17*dpr;
    ctx.globalAlpha=.82; ctx.fillStyle='rgba(16,16,16,.88)'; ctx.fillRect(midX-textWidth/2-padX,midY-boxH/2,textWidth+padX*2,boxH);
    ctx.globalAlpha=1; ctx.fillStyle='#fff'; ctx.fillText(label,midX,midY);
  }}
  ctx.restore();
}}
function drawAttacks(now) {{
  const current=data.attacks.filter(a=>Math.round(a.hit_time)===Math.floor(now+1e-6) && visibleAttackConfidence.has(a.confidence) && attackTracksVisible(a));
  current.sort((a,b)=>a.continuous_count-b.continuous_count).forEach(drawArrow);
  return current.slice(0,5).map(a=>{{ const objective=a.victim_type==='基地'||a.victim_type==='前哨站', victim=objective?`${{a.victim_camp}}${{a.victim_type}}`:`${{a.victim_camp}}${{unitNumber(a.victim_robot_id)}}`; return `${{a.hit_time.toFixed(1)}}s 推断：${{a.attacker_camp}}${{unitNumber(a.attacker_robot_id)}} → ${{victim}} · ${{a.caliber}}${{a.damage!==null?' · '+Math.round(a.damage)+'伤害':''}} · ${{a.confidence==='high'?'高':a.confidence==='medium'?'中':'低'}}可信${{a.special_mode==='hero_deployed_lob'?' · 英雄部署吊射推定（150%攻击增益）':a.continuous_count>1?' · 连续×'+a.continuous_count:''}}${{objective&&a.attacker_type==='英雄'&&a.caliber==='42mm'?' · 抛射事件关联':' · 偏角'+a.angle_error_deg.toFixed(1)+'°'}}`; }});
}}
function draw() {{
  const play=document.getElementById('bsPlayToggle'); if(play) play.textContent=timer||animationFrame!==null?'暂停':'播放';
  if(!canvas.width) return; ctx.clearRect(0,0,canvas.width,canvas.height);
  if(image.complete) {{ const [l,t,r,b]=data.field.crop; ctx.globalAlpha=0.82; ctx.drawImage(image,image.width*l,image.height*t,image.width*(r-l),image.height*(b-t),0,0,canvas.width,canvas.height); }}
  const now=Number(slider.value); if(rawToggle.checked) data.tracks.forEach((t,i)=>drawTrack(t,i,now,true,false)); data.tracks.forEach((t,i)=>drawTrack(t,i,now,false,false));
  const attackDetails=drawAttacks(now);
  const dartDetails=drawDartEffects(now);
  const deployedDetails=drawDeployedHeroEffects(now);
  data.tracks.forEach((track,i)=>{{ if(!visibleTracks.has(i)) return; const pose=poseAt(track,now,false); if(pose) drawRobotIcon(track,pose.index,pose.point,pose.heading,now); }});
  const revivalDetails=drawRevivalEffects(now);
  const eventDetails=drawEvents(now); drawHeavyObjectiveHits(now); drawObjectiveHud(now); const arenaDetails=drawArenaMechanics(now); drawDamageNumbers(now);
  const details=[...dartDetails,...deployedDetails,...revivalDetails,...arenaDetails,...eventDetails,...attackDetails]; document.getElementById('detail').textContent=details.join('；') || `${{now.toFixed(Number.isInteger(now)?0:2)}}s 无已选事件或攻击推断`;
  updateRosters(now); updateEconomy(now); ctx.globalAlpha=1; timeOut.value=`${{now.toFixed(Number.isInteger(now)?0:2)}} s`; const remaining=Math.max(0,Math.ceil(420-now)); countdownOut.value=`${{String(Math.floor(remaining/60)).padStart(2,'0')}}:${{String(remaining%60).padStart(2,'0')}}`;
}}
function stop() {{ if(timer) clearInterval(timer); if(animationFrame!==null) cancelAnimationFrame(animationFrame); timer=null; animationFrame=null; playbackButtons.forEach(button=>button.setAttribute('aria-pressed','false')); pauseButton.disabled=true; document.getElementById('bsPlayToggle').textContent='播放'; }}
function startFramePlayback(button) {{ stop(); button.setAttribute('aria-pressed','true'); pauseButton.disabled=false; slider.step='1'; slider.value=Math.floor(Number(slider.value)); timer=setInterval(()=>{{ let value=Math.floor(Number(slider.value))+1; if(value>maxT) value=minT; slider.value=value; draw(); }},100); }}
function startSmoothPlayback(speedProvider,button) {{ stop(); button.setAttribute('aria-pressed','true'); pauseButton.disabled=false; document.getElementById('bsPlayToggle').textContent='暂停'; slider.step='any'; let previous=performance.now(), playhead=Number(slider.value); const tick=timestamp=>{{ const elapsed=Math.max(0,(timestamp-previous)/1000); previous=timestamp; const speed=Math.max(.05,Number(speedProvider())||1); playhead+=elapsed*speed; const duration=maxT-minT; if(duration>0&&playhead>maxT) playhead=minT+((playhead-minT)%duration); slider.value=String(playhead); draw(); animationFrame=requestAnimationFrame(tick); }}; animationFrame=requestAnimationFrame(tick); }}
framePlay.addEventListener('click',()=>startFramePlayback(framePlay));
speedButtons.forEach(button=>button.addEventListener('click',()=>startSmoothPlayback(()=>Number(button.dataset.speed)||1,button)));
customPlay.addEventListener('click',()=>startSmoothPlayback(()=>Number(customSpeed.value)||1,customPlay));
customSpeed.addEventListener('input',()=>{{ const label=`${{Number(customSpeed.value).toFixed(2)}}×`; customSpeedValue.value=label; customPlay.textContent=`无极 ${{label}}`; }});
pauseButton.addEventListener('click',stop);
slider.addEventListener('input',()=>{{ if(timer||animationFrame!==null) stop(); slider.step='0.01'; draw(); }}); [rawToggle,trailInput,damageToggle].forEach(el=>el.addEventListener('input',draw));
document.getElementById('trackFilters').addEventListener('change',event=>{{ const value=Number(event.target.dataset.track); if(event.target.checked) visibleTracks.add(value); else visibleTracks.delete(value); draw(); }});
document.getElementById('eventFilters').addEventListener('change',event=>{{ const value=event.target.dataset.event; if(event.target.checked) visibleEvents.add(value); else visibleEvents.delete(value); draw(); }});
document.getElementById('attackFilters').addEventListener('change',event=>{{ const value=event.target.dataset.confidence; if(event.target.checked) visibleAttackConfidence.add(value); else visibleAttackConfidence.delete(value); draw(); }});
document.getElementById('bsPlayToggle').onclick=()=>{{ if(timer||animationFrame!==null) stop(); else startSmoothPlayback(()=>Number(customSpeed.value)||1,customPlay); }};
document.getElementById('bsPlaybackSpeed').onchange=event=>{{ customSpeed.value=event.target.value; customSpeed.dispatchEvent(new Event('input')); }};
image.addEventListener('load',draw); window.addEventListener('resize',resize); resize();
</script></body></html>"""
    ui_root = Path(__file__).resolve().parents[1] / "rmuc_web"
    html = html.replace("</style>", (ui_root / "replay.css").read_text(encoding="utf-8") + "\n</style>", 1)
    html = html.replace("</script></body></html>", "</script><script>\n" + (ui_root / "replay.js").read_text(encoding="utf-8") + "\n</script></body></html>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
