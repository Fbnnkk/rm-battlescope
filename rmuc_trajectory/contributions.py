"""Custom role contributions from observed regional data, not official points.

One timed evidence ledger drives both scorecards and curves. Tactical rewards
describe observable signs, not proven causality or terrain line of sight.
"""
from collections import defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
import re

import numpy as np

from .buffs import TERRAIN_BUFFS
from .field import OBJECTIVE_POSITIONS

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/scoring_v3.json"
LABELS = {
    "kills": "击杀贡献", "assists": "协同助攻", "deaths": "阵亡代价",
    "damage_to_robots": "对机器人输出", "damage_to_base": "基地攻坚",
    "damage_to_outpost": "前哨站攻坚", "damage_received": "承受伤害（仅观测）",
    "pressure": "前线交战控制（位置推断）", "suppression": "迫退 / 火力受限（推断）",
    "terrain_harassment": "跨地形后有效骚扰", "terrain_logistics": "地形机动后装配",
    "objective_defense": "建筑附近防守交战（推断）", "ally_cover": "掩护工程 / 英雄（推断）",
    "rune_activations": "成功能量机关支援（归因推断）", "assembly": "科技核心装配支援",
    "radar_marking": "雷达标记覆盖", "radar_assists": "标记期间协同输出",
    "radar_counter": "反制无人机", "dart_hits": "飞镖有效命中",
    "win_bonus": "胜负（仅观测）", "first_blood": "首杀（仅观测）",
}
DIMENSIONS = {
    "输出": {"damage_to_robots"}, "攻坚": {"damage_to_base", "damage_to_outpost", "dart_hits"},
    "协同": {"kills", "assists", "ally_cover"},
    "控场": {"pressure", "suppression", "objective_defense"},
    "机动": {"terrain_harassment", "terrain_logistics"},
    "支援": {"assembly", "rune_activations", "radar_marking", "radar_assists", "radar_counter"},
    "代价": {"deaths"},
}
NOTES = [
    "5分为中性起点；各兵种按职责权重评价，跨兵种不应直接当成能力排名。",
    "分项按 C×(1−exp(−贡献量/S)) 递减计分；正向分项之和超过5分时等比折算，限制堆叠刷分。",
    "伤害统计保留原始HP；计分时高/中/低可信分别按1/0.7/0.35权重折算。低可信多候选按匹配权重分摊同一份伤害，仅计输出，不推算击杀或控场。",
    "敌半场停留、空发弹、单纯存活、承伤、团队胜负、开闸次数都不直接加减分。",
    "控场需前线位置、邻近敌人及近期有效命中；压制需后撤或发弹下降。未命中定向开火仅在确认后撤后低权重计分，均属推断。",
    "地形触发后15秒内形成有效敌半场输出才计骚扰；连续触发去重。",
    "工程装配、飞镖命中、雷达标记与反制使用原始事件；不套用全国赛增益时长。",
    "每项贡献按确认时刻入账，曲线与最终战绩使用同一账本；片段不补算窗外贡献。",
]


def load_config():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    specs = list(config["components"].values())
    specs += [r[k] for r in config["roles"].values() for k in ("robot_damage", "structure_damage", "death")]
    if any(not math.isfinite(cap) or not math.isfinite(scale) or cap < 0 or scale <= 0 for cap, scale in specs):
        raise ValueError("评分配置的上限必须非负，尺度必须为正数")
    if not 0 < config["low_confidence_weight"] <= config["medium_confidence_weight"] <= 1 or not 0 < config["positive_budget"] <= 5:
        raise ValueError("证据权重及正向预算超出允许范围")
    for key in ("pressure_radius_m", "pressure_recent_hit_s", "retreat_window_s", "retreat_distance_m",
                "suppression_cooldown_s", "terrain_followup_s", "terrain_min_damage", "terrain_cooldown_s", "assist_window_s",
                "kill_window_s", "fire_min_shots", "fire_heading_deg"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"评分配置 {key} 必须为正数")
    if not 0 < config['fire_suppression_weight'] <= 1:
        raise ValueError('未命中火力压制的证据权重必须在0到1之间')
    return config


def bounded(value, spec):
    cap, scale = spec
    return float(cap * -math.expm1(-max(0.0, value) / scale))


def finite(value):
    return value is not None and math.isfinite(float(value))


def other(camp):
    return "蓝" if camp == "红" else "红"


def index(track, time):
    if track is None or not len(track.times):
        return None
    i = int(np.searchsorted(track.times, time, side="right"))-1
    return i if i >= 0 and time-track.times[i] <= 1.001 else None


def sample(track, time, position=True):
    i = index(track, time)
    if i is None or not track.observed_mask[i] or not finite(track.health[i]) or track.health[i] <= 0:
        return None
    if position and (not np.isfinite(track.clean_xy[i]).all() or not np.isfinite(track.raw_xy[i]).all()
                     or np.linalg.norm(track.raw_xy[i]-track.clean_xy[i]) > 1.5):
        return None
    return i


class Evaluation:
    def __init__(self, match, tracks, events, attacks, buffs, min_confidence="low"):
        if min_confidence not in {"high", "medium", "low"}:
            raise ValueError("评分证据必须为 high、medium 或 low")
        self.config, self.match, self.tracks, self.events = load_config(), match, tracks, events
        self.policy, self.all_attacks = min_confidence, attacks
        self.lookup = {(t.key.camp, t.key.robot_id): t for t in tracks if len(t.times)}
        self.entities = {key: {"camp": t.key.camp, "robot_id": t.key.robot_id,
                              "robot_type": t.key.robot_type, "school": t.key.school}
                         for key, t in self.lookup.items()
                         if t.key.robot_type not in {"基地", "前哨站", "飞镖", "雷达"}}
        allowed = {"high"} if min_confidence == "high" else {"high", "medium"}
        self.attacks = sorted([a for a in attacks if a.confidence in allowed and not a.ambiguous
                              and a.attacker_camp in {"红", "蓝"} and a.victim_camp == other(a.attacker_camp)
                              and finite(a.damage) and a.damage > 0 and finite(a.hit_time)
                              and finite(a.shot_time) and a.shot_time <= a.hit_time], key=lambda a: a.hit_time)
        self.damage_attacks = list(self.attacks)
        if min_confidence == "low":
            for a in attacks:
                if not (a.confidence == "low" or a.ambiguous):
                    continue
                if not (a.attacker_camp in {"红", "蓝"} and a.victim_camp == other(a.attacker_camp)
                        and finite(a.damage) and a.damage > 0 and finite(a.hit_time)
                        and finite(a.shot_time) and a.shot_time <= a.hit_time):
                    continue
                # Only candidates explicitly retained by the inference engine
                # may receive uncertain damage; an unsupported guess stays unknown.
                candidates = [c for c in a.damage_candidates
                              if c.get("attacker_camp") == a.attacker_camp
                              and (c.get("attacker_camp"), c.get("attacker_robot_id")) in self.entities
                              and finite(c.get("share")) and c["share"] > 0
                              and finite(c.get("shot_time")) and c["shot_time"] <= a.hit_time]
                share_sum = sum(c["share"] for c in candidates)
                for c in candidates:
                    # Missing/unsupported candidates retain their unknown share.
                    self.damage_attacks.append(replace(a,
                        attacker_camp=c["attacker_camp"], attacker_robot_id=c["attacker_robot_id"],
                        attacker_type=c["attacker_type"], shot_time=c["shot_time"], confidence="low",
                        damage=a.damage*c["share"]/max(1.0, share_sum)))
        self.items, self.by_actor = [], defaultdict(list)
        self.outgoing, self.incoming, self.shots = defaultdict(list), defaultdict(list), defaultdict(list)
        for a in self.attacks:
            self.outgoing[(a.attacker_camp, a.attacker_robot_id)].append(a)
            self.incoming[(a.victim_camp, a.victim_robot_id)].append(a)
        for e in events:
            if e.event_type == "发弹":
                self.shots[(e.camp, e.robot_id)].append(e.time)
        times = [float(v) for t in tracks if len(t.times) for v in (t.times[0], t.times[-1])]
        self.lo = min(times) if times else min((e.time for e in events), default=0)
        self.hi = max(times) if times else max((e.time for e in events), default=0)
        self.damage_and_support()
        self.deaths_and_kills()
        self.tactical()
        self.special_events(buffs)
        self.items.sort(key=lambda i: (i["time"], i["camp"], i["robot_id"], i["key"]))
        for item in self.items:
            self.by_actor[(item["camp"], item["robot_id"])].append(item)

    def add(self, time, actor, key, value, reason, confidence="recorded", *, units=None, target="", start=None):
        if actor not in self.entities or not finite(time) or time < self.lo or time > self.hi:
            return
        self.items.append({"time": float(time), "camp": actor[0], "robot_id": actor[1], "key": key,
                           "label": LABELS[key], "value": float(value), "units": float(value if units is None else units),
                           "reason": reason, "confidence": confidence, "target": target,
                           "start": float(start) if start is not None else None})

    def support(self, camp, role):
        rid = (9 if role == "雷达" else 8)+(100 if camp == "蓝" else 0)
        actual = next((key for key, t in self.lookup.items() if key[0] == camp and t.key.robot_type == role), None)
        key = actual or (camp, rid)
        self.entities.setdefault(key, {"camp": camp, "robot_id": key[1], "robot_type": role,
                                      "school": self.match.get(camp+"方学校", ""), "event_only": True,
                                      "synthetic_id": actual is None})
        return key

    def weight(self, attack):
        return 1.0 if attack.confidence == "high" else self.config[attack.confidence+"_confidence_weight"]

    def damage_and_support(self):
        for a in self.damage_attacks:
            actor, victim = (a.attacker_camp, a.attacker_robot_id), (a.victim_camp, a.victim_robot_id)
            target = f"{a.victim_camp}方{a.victim_type} #{a.victim_robot_id}"
            key = {"基地": "damage_to_base", "前哨站": "damage_to_outpost"}.get(a.victim_type, "damage_to_robots")
            reason = (f"{a.caliber}低可信伤害分摊；同秒{a.hit_count}次受击，候选按匹配权重分摊，计分权重{self.weight(a):g}，不用于击杀/控场"
                      if a.confidence == "low" else f"{a.caliber}命中归因；同秒{a.hit_count}次受击合计，伤害统计保留原值")
            self.add(a.hit_time, actor, key, a.damage, reason,
                     a.confidence, units=a.damage*self.weight(a), target=target)
            self.add(a.hit_time, victim, "damage_received", a.damage, "受击归因，不把承伤直接当成失误", a.confidence)
            if a.confidence == "low":
                continue
            vt = self.lookup.get(victim)
            vi = index(vt, a.hit_time)
            if vi is not None and vt.observed_mask[vi] and vt.vulnerable[vi] >= .5:
                self.add(a.hit_time, self.support(a.attacker_camp, "雷达"), "radar_assists", a.damage,
                         "敌方被本队雷达标记期间受到友军归因伤害；不等同于增伤因果", a.confidence,
                         units=a.damage*self.weight(a), target=target)
            vi, ai = sample(vt, a.hit_time), sample(self.lookup.get(actor), a.hit_time)
            if a.victim_type in {"基地", "前哨站"} or vi is None or ai is None:
                continue
            own = [t for t in self.tracks if t.key.camp == actor[0] and t.key.robot_type in {"基地", "前哨站"}]
            if any(sample(t, a.hit_time, False) is not None and
                   np.linalg.norm(vt.clean_xy[vi]-OBJECTIVE_POSITIONS[(actor[0], t.key.robot_type)]) <= 3.5 for t in own):
                self.add(a.hit_time, actor, "objective_defense", a.damage,
                         "打击己方存活建筑约3.5米内敌人；建筑位置为地图近似值", "inferred",
                         units=a.damage*self.weight(a), target=target)
            if any(a.hit_time-5 <= b.hit_time <= a.hit_time and b.victim_type in {"工程", "英雄"} for b in self.outgoing[victim]):
                self.add(a.hit_time, actor, "ally_cover", a.damage,
                         "回击过去5秒攻击本队工程或英雄的敌人；无法证明拦截成功", "inferred",
                         units=a.damage*self.weight(a), target=target)
        for t in self.tracks:
            if t.key.robot_type in {"基地", "前哨站", "雷达", "飞镖"}:
                continue
            for i, time in enumerate(t.times):
                if t.observed_mask[i] and finite(t.vulnerable[i]) and t.vulnerable[i] >= .5 and t.health[i] > 0:
                    self.add(time, self.support(other(t.key.camp), "雷达"), "radar_marking", 1,
                             "逐秒原始遥测确认敌方被雷达标记；单位为目标秒", "observed", target=f"{t.key.camp}方{t.key.robot_type}")

    def deaths_and_kills(self):
        for key, ent in list(self.entities.items()):
            t = self.lookup.get(key)
            if t is None or ent.get("event_only"):
                continue
            for i in range(1, len(t.times)):
                if not (t.observed_mask[i-1] and t.observed_mask[i] and t.times[i]-t.times[i-1] <= 1.1
                        and t.health[i-1] > 0 and t.health[i] <= 0):
                    continue
                time = float(t.times[i])
                self.add(time, key, "deaths", 1, "相邻有效血量帧由正值降至0；未知受击不假造击杀", "observed")
                hits = [a for a in self.incoming[key] if time-self.config["assist_window_s"] <= a.hit_time <= time]
                if not hits:
                    continue
                latest = max(a.hit_time for a in hits)
                if latest < time-self.config['kill_window_s']:
                    continue
                # A later unassigned/uncertain hit may be the fatal blow. Do
                # not silently promote the previous reliable shooter to killer.
                if any(e.event_type == '受击' and e.category in {'17mm', '42mm'}
                       and (e.camp, e.robot_id) == key and latest < e.time <= time
                       for e in self.events):
                    continue
                if any((a.victim_camp, a.victim_robot_id) == key and latest <= a.hit_time <= time
                       and (a.ambiguous or a.confidence not in
                            ({'high'} if self.policy == 'high' else {'high', 'medium'}))
                       for a in self.all_attacks):
                    continue
                last = {(a.attacker_camp, a.attacker_robot_id) for a in hits if a.hit_time == latest}
                if len(last) != 1:
                    continue
                killer = next(iter(last))
                damage = defaultdict(float)
                for a in hits:
                    damage[(a.attacker_camp, a.attacker_robot_id)] += a.damage*self.weight(a)
                for actor, dealt in damage.items():
                    share, kill = dealt/sum(damage.values()), actor == killer
                    self.add(time, actor, "kills" if kill else "assists", 1,
                             f"阵亡前8秒伤害占比 {share:.0%}；最后命中归因，不代表官方击杀统计", "inferred",
                             units=.5+.5*share if kill else share, target=f"{key[0]}方{ent['robot_type']} #{key[1]}")

    def tactical(self):
        seconds = np.arange(math.ceil(self.lo), math.floor(self.hi)+1)
        for victim, incoming in self.incoming.items():
            t = self.lookup.get(victim)
            if t is None or t.key.robot_type in {"基地", "前哨站", "空中"}:
                continue
            for time in seconds:
                vi = sample(t, time)
                if vi is None:
                    continue
                candidates = {}
                for a in incoming:
                    if not time-self.config["pressure_recent_hit_s"] <= a.hit_time <= time:
                        continue
                    actor = (a.attacker_camp, a.attacker_robot_id)
                    at = self.lookup.get(actor)
                    ai = sample(at, time)
                    if ai is None:
                        continue
                    pos = at.clean_xy[ai]
                    front = pos[0] >= 13 if actor[0] == "红" else pos[0] <= 15
                    if front and np.linalg.norm(pos-t.clean_xy[vi]) <= self.config["pressure_radius_m"]:
                        candidates[actor] = max(candidates.get(actor, 0), self.weight(a))
                for actor, weight in candidates.items():
                    self.add(time, actor, "pressure", 1/len(candidates),
                             f"前线存活、敌人{self.config['pressure_radius_m']:g}米内且过去{self.config['pressure_recent_hit_s']:g}秒有效命中；多人分摊目标秒", "inferred",
                             units=weight/len(candidates), target=f"{victim[0]}方{t.key.robot_type}")
            self.suppression(victim, incoming)
        self.fire_suppression()
        # Merge all suppression sources chronologically: a later observation
        # never removes an earlier score, and one retreat cannot be farmed.
        groups = defaultdict(list)
        for item in self.items:
            if item['key']=='suppression':groups[(item['target'], item['time'])].append(item)
        last, keep = defaultdict(lambda: -math.inf), set()
        for (target, time), group in sorted(groups.items(), key=lambda g:g[0][1]):
            if time-last[target] < self.config['suppression_cooldown_s']:
                continue
            hits=[item for item in group if not item.get('fire_only')]
            for item in hits or group:keep.add(id(item))
            last[target]=time
        self.items=[i for i in self.items if i['key']!='suppression' or id(i) in keep]

    def suppression(self, victim, incoming):
        t, last = self.lookup[victim], -math.inf
        for a in incoming:
            time = a.hit_time+self.config["retreat_window_s"]
            if time > self.hi or time-last < self.config["suppression_cooldown_s"]:
                continue
            vi0, vi1 = sample(t, a.hit_time), sample(t, time)
            if vi0 is None or vi1 is None:
                continue
            frames = np.flatnonzero((t.times >= a.hit_time) & (t.times <= time))
            if len(frames) < 3 or any(sample(t, t.times[i]) is None for i in frames):
                continue
            contributors = {}
            p0, p1 = t.clean_xy[vi0], t.clean_xy[vi1]
            retreat_dx = (p1[0]-p0[0])*(1 if victim[0] == "蓝" else -1)
            shots = self.shots[victim]
            before = sum(a.hit_time-4 <= s < a.hit_time for s in shots)
            after = sum(a.hit_time <= s <= time for s in shots)
            for b in incoming:
                if not a.hit_time-3 <= b.hit_time <= a.hit_time:
                    continue
                actor = (b.attacker_camp, b.attacker_robot_id)
                at = self.lookup.get(actor)
                ai0, ai1 = sample(at, a.hit_time), sample(at, time)
                if ai0 is None or ai1 is None:
                    continue
                ap0, ap1 = at.clean_xy[ai0], at.clean_xy[ai1]
                front = ap0[0] >= 13 if actor[0] == "红" else ap0[0] <= 15
                if not front or np.linalg.norm(ap0-p0) > self.config["pressure_radius_m"]:
                    continue
                retreat = retreat_dx >= self.config["retreat_distance_m"] and np.linalg.norm(ap0-p1)-np.linalg.norm(ap0-p0) >= self.config["retreat_distance_m"]
                limited = before >= 4 and after <= before*.4 and np.linalg.norm(ap1-p1) <= self.config["pressure_radius_m"]
                if retreat or limited:
                    reason = (f"命中后{self.config['retreat_window_s']:g}秒敌方向己方后撤且远离攻击点≥{self.config['retreat_distance_m']:g}米" if retreat else
                              f"命中后{self.config['retreat_window_s']:g}秒敌方发弹降至此前4秒的40%以下，仍在交战距离内")
                    contributors[actor] = (self.weight(b), reason)
            if contributors:
                for actor, (weight, reason) in contributors.items():
                    self.add(time, actor, "suppression", 1/len(contributors), reason+f"；仅为压制迹象，不能证明因果；{self.config['suppression_cooldown_s']:g}秒去重", "inferred",
                             units=weight/len(contributors), target=f"{victim[0]}方{t.key.robot_type} #{victim[1]}", start=a.hit_time)
                last = time

    def fire_suppression(self):
        """Lower-weight threat proxy: aimed bursts plus measured enemy retreat.

        No hit attribution is invented. Heading and position alone never earn
        points; an observed retreat must complete the evidence window.
        """
        groups=defaultdict(list)
        for actor, shots in self.shots.items():
            t=self.lookup.get(actor)
            if t is None or t.key.robot_type not in {'英雄','步兵3','步兵4','哨兵','空中'}:
                continue
            counts=defaultdict(int)
            for time in shots:counts[float(time)]+=1
            for start, count in sorted(counts.items()):
                time=start+self.config['retreat_window_s']
                if count < self.config['fire_min_shots'] or time > self.hi:
                    continue
                ai=sample(t,start)
                if ai is None or sample(t,time) is None or not finite(t.heading_deg[ai]):
                    continue
                p=t.clean_xy[ai]
                if not (p[0]>=13 if actor[0]=='红' else p[0]<=15):
                    continue
                candidates=[]
                for victim, vt in self.lookup.items():
                    if victim[0]==actor[0] or vt.key.robot_type in {'基地','前哨站','空中','雷达','飞镖'}:
                        continue
                    vi=sample(vt,start)
                    if vi is None:
                        continue
                    q=vt.clean_xy[vi]
                    distance=float(np.linalg.norm(q-p))
                    if not .5 <= distance <= self.config['pressure_radius_m']:
                        continue
                    angle=math.degrees(math.atan2(q[1]-p[1],q[0]-p[0]))
                    error=abs((t.heading_deg[ai]-angle+180)%360-180)
                    if error<=self.config['fire_heading_deg']:
                        candidates.append((error+distance*.1,victim))
                if not candidates:
                    continue
                candidates.sort()
                # Two equally aligned targets cannot identify a threatened
                # robot; do not guess or award both in full.
                if len(candidates)>1 and candidates[1][0]-candidates[0][0]<2:
                    continue
                victim=candidates[0][1];vt=self.lookup[victim]
                vi0,vi1=sample(vt,start),sample(vt,time)
                frames=np.flatnonzero((vt.times>=start)&(vt.times<=time))
                if vi1 is None or len(frames)<3 or any(sample(vt,vt.times[i]) is None for i in frames):
                    continue
                q0,q1=vt.clean_xy[vi0],vt.clean_xy[vi1]
                dx=(q1[0]-q0[0])*(1 if victim[0]=='蓝' else -1)
                if dx<self.config['retreat_distance_m'] or np.linalg.norm(q1-p)-np.linalg.norm(q0-p)<self.config['retreat_distance_m']:
                    continue
                if any(start-3<=a.hit_time<=time for a in self.incoming[victim]):
                    continue  # the more informative hit-based path handles this
                groups[(victim,time,start)].append(actor)
        for (victim,time,start),actors in groups.items():
            for actor in set(actors):
                self.add(time,actor,'suppression',1/len(set(actors)),
                         f"前线定向密集开火（同秒≥{self.config['fire_min_shots']}发、偏角≤{self.config['fire_heading_deg']}°），后{self.config['retreat_window_s']:g}秒敌方向己方后撤且远离攻击点≥{self.config['retreat_distance_m']:g}米；没有确认命中，按{self.config['fire_suppression_weight']:g}权重计压制迹象",
                         'inferred',units=self.config['fire_suppression_weight']/len(set(actors)),
                         target=f"{victim[0]}方{self.lookup[victim].key.robot_type} #{victim[1]}",start=start)
                if self.items and self.items[-1]['key']=='suppression':self.items[-1]['fire_only']=True

    def special_events(self, buffs):
        events, seen, terrain = sorted(self.events, key=lambda e: e.time), set(), defaultdict(list)
        for b in buffs:
            if b.category in TERRAIN_BUFFS:
                terrain[(b.camp, b.robot_id)].append((b.start, b.category))
        for e in events:
            identity = (e.time, e.event_type, e.camp, e.robot_id, e.category, e.value, e.target_robot_id)
            if identity in seen:
                continue
            seen.add(identity)
            if e.event_type == "增益" and e.category in TERRAIN_BUFFS and e.robot_id is not None:
                terrain[(e.camp, e.robot_id)].append((e.time, e.category))
            if e.event_type == "装配成功" and e.camp in {"红", "蓝"}:
                actors = [key for key, ent in self.entities.items() if key[0] == e.camp and ent["robot_type"] == "工程"]
                if e.robot_id is not None:
                    actors = [key for key in actors if key[1] == e.robot_id]
                grade = re.search(r"等级\s*(\d+)", e.category or "")
                if len(actors) == 1 and grade and 1 <= int(grade.group(1)) <= 5:
                    level = int(grade.group(1))
                    duration = f"，耗时{e.value:g}秒" if finite(e.value) else ""
                    penalty = f"，备注：{e.note}" if e.note else ""
                    self.add(e.time, actors[0], "assembly", 1,
                             f"原始装配成功：等级{level}{duration}{penalty}；归到本队唯一工程，等级权重{level*.5+.25:g}，不假算金币",
                             "role_inferred" if e.robot_id is None else "recorded", units=level*.5+.25)
            elif e.event_type == "飞镖闸门开" and e.camp in {"红", "蓝"}:
                self.support(e.camp, "飞镖")
            elif e.event_type == "飞镖命中" and e.camp in {"红", "蓝"} and finite(e.value) and e.value > 0 and e.target_type in {"基地", "前哨站"}:
                actor, target = self.support(e.camp, "飞镖"), f"{other(e.camp)}方{e.target_type}"
                self.add(e.time, actor, "dart_hits", 1, "原始飞镖命中；开闸次数不作为发射次数", target=target)
                self.add(e.time, actor, "damage_to_base" if e.target_type == "基地" else "damage_to_outpost", e.value,
                         "原始飞镖命中伤害，与弹丸推断分开归因", target=target)
            elif e.event_type == "雷达反制UAV" and e.camp in {"红", "蓝"}:
                note = re.search(r"反制方\s*=\s*([红蓝])", e.note or "")
                camp = note.group(1) if note else other(e.camp)
                if camp == other(e.camp):
                    self.add(e.time, self.support(camp, "雷达"), "radar_counter", 1,
                             "原始雷达反制：主体为被反制无人机，计给反制方；不假设持续时间", target=f"{e.camp}方空中")
        self.terrain(terrain)
        self.rune(events)

    def terrain(self, terrain):
        for actor, triggers in list(terrain.items()):
            if actor not in self.entities:
                continue
            last = -math.inf
            for start, category in sorted(set(triggers)):
                if start-last < self.config["terrain_cooldown_s"]:
                    continue
                hits = [a for a in self.outgoing[actor] if start <= a.shot_time <= a.hit_time <= start+self.config["terrain_followup_s"]
                        and (a.victim_type in {"基地", "前哨站"} or
                             (a.victim_xy is not None and np.isfinite(a.victim_xy).all() and
                              (a.victim_xy[0] >= 14 if actor[0] == "红" else a.victim_xy[0] <= 14)))]
                amount = 0.0
                for a in hits:
                    amount += a.damage
                    if amount >= self.config["terrain_min_damage"]:
                        self.add(a.hit_time, actor, "terrain_harassment", 1,
                                 f"{category}后{self.config['terrain_followup_s']:g}秒内对敌半场机器人/建筑造成≥{self.config['terrain_min_damage']:g}HP归因伤害；无有效后续不奖励", "inferred",
                                 units=self.weight(a), start=start, target=f"{a.victim_camp}方{a.victim_type}")
                        last = a.hit_time
                        break
                else:
                    assembly = next((i for i in self.items if (i["camp"], i["robot_id"]) == actor and i["key"] == "assembly"
                                     and start <= i["time"] <= start+20), None)
                    if assembly:
                        self.add(assembly["time"], actor, "terrain_logistics", 1,
                                 f"{category}后20秒出现装配成功，作为任务机动迹象", "inferred", start=start)
                        last = assembly["time"]

    def rune(self, events):
        last = defaultdict(lambda: -math.inf)
        for e in events:
            if e.event_type != "增益" or e.category not in {"小能量机关增益", "大能量机关增益"} or e.camp not in {"红", "蓝"} or e.time-last[e.camp] < 10:
                continue
            arms = [a for a in events if a.event_type == "能量机关" and a.camp == e.camp and e.time-10 <= a.time <= e.time]
            if not arms:
                continue
            arm, candidates = arms[-1], []
            for actor, ent in self.entities.items():
                if actor[0] != e.camp or ent["robot_type"] not in {"步兵3", "步兵4", "空中"}:
                    continue
                t = self.lookup.get(actor)
                i = sample(t, arm.time)
                if i is None or not finite(t.heading_deg[i]):
                    continue
                pos, heading = t.clean_xy[i], t.heading_deg[i]
                dist = np.linalg.norm(pos-np.array([14, 7.5]))
                if not 3 <= dist <= 9 or (pos[0] > 15 if e.camp == "红" else pos[0] < 13):
                    continue
                angle = math.degrees(math.atan2(7.5-pos[1], 14-pos[0]))
                diff = abs((heading-angle+180)%360-180)
                if diff <= 45 and any(arm.time-3 <= s <= arm.time for s in self.shots[actor]):
                    candidates.append((1-diff/90, actor))
            if candidates:
                top = [actor for score, actor in candidates if score >= max(s for s, _ in candidates)*.85]
                for actor in top:
                    self.add(e.time, actor, "rune_activations", 1/len(top),
                             f"实际{e.category}确认成功；中心附近朝向及发弹推断执行者，{len(top)}个候选均分", "inferred")
                last[e.camp] = e.time

    def assessment(self, actor, until=None):
        entity = self.entities[actor]
        items = [i for i in self.by_actor[actor] if until is None or i['time'] <= until]
        contributions = [i for i in items if i['key'] != 'damage_received' and i['units'] != 0]
        if contributions:
            return {'rating_eligible': True, 'assessment_kind': 'observed_contributions',
                    'assessment_status': '按已观测贡献暂评',
                    'assessment_reason': '存在可计分贡献；只评价已覆盖的职责，不代表完整能力。'}
        t = self.lookup.get(actor)
        if t is not None and entity['robot_type'] in {'英雄', '步兵3', '步兵4', '哨兵', '空中'}:
            within = t.times <= (self.hi if until is None else until)
            alive = within & t.observed_mask & np.isfinite(t.health) & (t.health > 0)
            valid = alive & np.isfinite(t.clean_xy).all(axis=1) & np.isfinite(t.heading_deg)
            ammunition = np.isfinite(t.cumulative_17mm) | np.isfinite(t.cumulative_42mm)
            if np.count_nonzero(valid & ammunition) >= 2 and np.count_nonzero(within) > 0 and np.mean((t.observed_mask & np.isfinite(t.health))[within]) >= .5:
                return {'rating_eligible': True, 'assessment_kind': 'observed_neutral',
                        'assessment_status': '已观测活动 · 中性表现',
                        'assessment_reason': '至少两帧有效存活位置、朝向及发弹遥测；未发现可计分贡献，保留中性起点，不能据此断言没有战术贡献。'}
        role_reason = {'飞镖': '开闸不能确认发射、未命中或伤害；缺少可评价的命中记录。',
                       '工程': '移动或存活遥测不能确认装配、运输或兑换表现。',
                       '雷达': '缺少标记覆盖或反制证据；无法确认全程支援表现。'}
        return {'rating_eligible': False, 'assessment_kind': 'insufficient_evidence',
                'assessment_status': '证据不足 · 暂不评级',
                'assessment_reason': role_reason.get(entity['robot_type'], '有效活动遥测或贡献证据不足，未观测不等于零贡献。')}

    def row(self, entity, values, units, explain=True):
        role = self.config["roles"].get(entity["robot_type"], self.config["roles"]["步兵3"])
        components = {k: {"value": round(values.get(k, 0), 3), "score": 0.0} for k in LABELS}
        for key, spec in self.config["components"].items():
            components[key]["score"] = bounded(units.get(key, 0), spec)
        components["damage_to_robots"]["score"] = bounded(units.get("damage_to_robots", 0), role["robot_damage"])
        structure = sum(units.get(k, 0) for k in ("damage_to_base", "damage_to_outpost"))
        for key in ("damage_to_base", "damage_to_outpost"):
            components[key]["score"] = bounded(structure, role["structure_damage"])*units.get(key, 0)/structure if structure else 0.0
        components["deaths"]["score"] = -bounded(units.get("deaths", 0), role["death"])
        positive = sum(max(0, v["score"]) for v in components.values())
        budget = self.config["positive_budget"]
        factor = min(1.0, budget/positive) if positive else 1.0
        for value in components.values():
            if value["score"] > 0:
                value["score"] *= factor
            value["score"] = round(value["score"], 6)
        raw = round(5+sum(v["score"] for v in components.values()), 6)
        total = round(max(0, min(10, raw)), 2)
        row = {**entity, "total_score": total, "raw_score": raw, "starting_score": 5, "scale": 1.0,
               "grade": "S" if total >= 8.5 else "A" if total >= 7 else "B" if total >= 5.5 else "C" if total >= 4 else "D",
               "components": components, "mission": role["mission"], "positive_budget_factor": round(factor, 6)}
        if not explain:
            return row
        actor = (entity["camp"], entity["robot_id"])
        row["explanation"] = sorted([{"key": k, "label": LABELS[k], **v} for k, v in components.items()], key=lambda c: abs(c["score"]), reverse=True)
        row["dimensions"] = [{"label": label, "score": round(sum(components[k]["score"] for k in keys), 6)} for label, keys in DIMENSIONS.items()]
        row["contribution_events"] = [{k: v for k, v in i.items() if k not in {"camp", "robot_id", "units"}} for i in self.by_actor[actor] if i["key"] != "damage_received"]
        row.update(self.assessment(actor))
        row['calculated_score'] = total
        if not row['rating_eligible']:
            row['total_score'], row['grade'] = None, None
        t = self.lookup.get(actor)
        relevant = [a for a in self.all_attacks if (a.attacker_camp, a.attacker_robot_id) == actor]
        accepted = self.outgoing[actor]
        allocated = [a for a in self.damage_attacks if (a.attacker_camp, a.attacker_robot_id) == actor
                     and self.lo <= a.hit_time <= self.hi]
        relevant = [a for a in self.all_attacks if (a.attacker_camp, a.attacker_robot_id) == actor
                    or any((c['attacker_camp'], c['attacker_robot_id']) == actor for c in a.damage_candidates)]
        received_events = [e for e in self.events if e.event_type=='受击' and e.category in {'17mm','42mm'}
                           and (e.camp,e.robot_id)==actor and finite(e.value)]
        received_raw = sum(abs(e.value) for e in received_events)
        received_attributed = sum(a.damage for a in self.damage_attacks if (a.victim_camp, a.victim_robot_id) == actor)
        row["evidence"] = {
            "health_coverage": round(float(np.mean(np.isfinite(t.health))), 3) if t is not None else None,
            "position_coverage": round(float(np.mean(np.isfinite(t.clean_xy).all(axis=1))), 3) if t is not None else None,
            "inferred_attacks": len(relevant), "scored_attacks": len(allocated), "excluded_attacks": max(0, len(relevant)-len(allocated)),
            "high_confidence_attacks": sum(a.confidence == "high" for a in accepted),
            "observed_projectile_damage_received": received_raw,
            "attributed_projectile_damage_received": received_attributed,
            "mode": "event_only" if entity.get("event_only") else "telemetry_and_events",
            "limitations": ["位置控场、压制和掩护是可回看的迹象，无法确认视线遮挡或战术因果。",
                            "未观测贡献不等于没有贡献；数据不足不会自动判定挂机。"],
        }
        damage_items = [i for i in self.by_actor[actor] if i['key'] in
                        {'damage_to_robots', 'damage_to_base', 'damage_to_outpost'}]
        for level, field in [('high', 'high_confidence_damage'), ('medium', 'medium_confidence_damage'), ('recorded', 'recorded_damage')]:
            row['evidence'][field] = round(sum(i['value'] for i in damage_items if i['confidence'] == level), 3)
        row['evidence']['reliable_damage'] = round(sum(i['value'] for i in damage_items if i['confidence'] != 'low'), 3)
        row['evidence']['low_confidence_damage'] = round(sum(i['value'] for i in damage_items if i['confidence'] == 'low'), 3)
        row['evidence']['low_confidence_damage_windows'] = len([i for i in damage_items if i['confidence'] == 'low'])
        if entity["robot_type"] == "工程":
            row["evidence"]["limitations"].append("装配按本队唯一工程归因；没有完整矿石运输、失败尝试、准确兑换金币或救援记录。")
        elif entity.get("event_only"):
            row["evidence"]["limitations"].append("仅对已记录事件评分；没有移动遥测，不能推断未命中次数或全程支援质量。")
        if entity["robot_type"] == "英雄" and t is not None:
            shots = sum(max(0, float(t.cumulative_42mm[i]-t.cumulative_42mm[i-1])) for i in range(1, len(t.times))
                        if finite(t.cumulative_42mm[i]) and finite(t.cumulative_42mm[i-1]) and t.observed_mask[i] and t.observed_mask[i-1])
            spent = t.total_coins-t.remaining_coins
            valid = spent[np.isfinite(spent)]
            row["hero"] = {"42mm_shots": {"value": shots, "score": 0},
                           "gold_spent": {"value": max(0, float(valid[-1]-valid[0])) if len(valid) else 0, "score": 0}}
        if entity["robot_type"] == "空中" and t is not None:
            active = int(np.count_nonzero(t.observed_mask & np.isfinite(t.health) & (t.health > 0)))
            row["aerial"] = {"observed_active_seconds": active, "output_efficiency_dps": round(values.get("damage_to_robots", 0)/max(1, active), 3),
                             "counter_events_received": sum(e.event_type == "雷达反制UAV" and e.camp == entity["camp"] for e in self.events)}
        return row

    def scores(self):
        result = []
        for actor, entity in self.entities.items():
            values, units = defaultdict(float), defaultdict(float)
            for item in self.by_actor[actor]:
                values[item["key"]] += item["value"]
                units[item["key"]] += item["units"]
            result.append(self.row(entity, values, units))
        order = {role: i for i, role in enumerate(self.config["roles"])}
        result.sort(key=lambda r: (r["camp"] != self.match.get("胜方"), order.get(r["robot_type"], 99), r["robot_id"]))
        numbers = [r["total_score"] for r in result if r["rating_eligible"]]
        raw_damage = sum(abs(e.value) for e in self.events if e.event_type=='受击'
                         and e.category in {'17mm','42mm'} and finite(e.value))
        damage_ledger = [i for i in self.items if i['key'] in
                        {'damage_to_robots', 'damage_to_base', 'damage_to_outpost'}
                        and i['reason'].startswith(('17mm', '42mm'))]
        attributed = sum(i['value'] for i in damage_ledger)
        reliable = sum(i['value'] for i in damage_ledger if i['confidence'] != 'low')
        low = attributed - reliable
        targets = []
        for camp, rid in sorted({(e.camp, e.robot_id) for e in self.events if e.event_type == '受击'
                               and e.category in {'17mm', '42mm'} and e.camp in {'红', '蓝'} and e.robot_id is not None}):
            native = [e for e in self.events if e.event_type == '受击' and e.category in {'17mm', '42mm'}
                      and (e.camp, e.robot_id) == (camp, rid) and finite(e.value)]
            target_name = f"{camp}方{native[0].robot_type} #{rid}" if native else ''
            observed = sum(abs(e.value) for e in native)
            reliable_target = sum(i['value'] for i in damage_ledger if i['target'] == target_name and i['confidence'] != 'low')
            low_target = sum(i['value'] for i in damage_ledger if i['target'] == target_name and i['confidence'] == 'low')
            targets.append({'camp': camp, 'robot_id': rid, 'robot_type': native[0].robot_type if native else '未知',
                            'observed_damage': observed, 'reliable_damage': reliable_target,
                            'low_confidence_damage': low_target,
                            'high_confidence_damage': sum(i['value'] for i in damage_ledger if i['target']==target_name and i['confidence']=='high'),
                            'medium_confidence_damage': sum(i['value'] for i in damage_ledger if i['target']==target_name and i['confidence']=='medium'),
                            'unattributed_damage': max(0, observed - reliable_target - low_target)})
        summary = {"robot_count": len(result), "min_score": min(numbers, default=None), "max_score": max(numbers, default=None),
                   "mean_score": round(float(np.mean(numbers)), 2) if numbers else None,
                   "rated_robot_count": len(numbers), "unrated_robot_count": len(result)-len(numbers),
                   "kill_count": sum(r["components"]["kills"]["value"] for r in result),
                   "death_count": sum(r["components"]["deaths"]["value"] for r in result),
                   "assist_count": sum(r["components"]["assists"]["value"] for r in result),
                   "afk_robot_count": 0, "first_blood_time": None, "first_blood_robot": None,
                   "rule_version": self.config["version"], "score_range": [0, 10], "min_confidence": self.policy,
                   "excluded_attack_count": len(self.all_attacks)-len({(a.victim_camp,a.victim_robot_id,a.caliber,a.hit_time) for a in self.damage_attacks}), "is_partial": bool(self.match.get("_is_partial")),
                   "configuration": self.config, "contribution_event_count": len(self.items),
                   "observed_projectile_damage": raw_damage, "attributed_projectile_damage": attributed,
                   "reliable_projectile_damage": reliable,
                   "high_confidence_projectile_damage": sum(i["value"] for i in damage_ledger if i["confidence"]=="high"),
                   "medium_confidence_projectile_damage": sum(i["value"] for i in damage_ledger if i["confidence"]=="medium"), "low_confidence_projectile_damage": low,
                   "unattributed_projectile_damage": max(0, raw_damage-attributed), "damage_targets": targets,
                   "projectile_attribution_coverage": round(attributed/raw_damage,4) if raw_damage else None,
                   "unattributed_death_count": sum(r['components']['deaths']['value']-r['components']['kills']['value'] for r in result),
                   "notes": ["证据不足的兵种暂不评级，不纳入均分或最高表现；均分分母只包含已评级实体。内部5分起点不作为缺失证据的成绩。",
                             "高/中/低可信攻击来源均为遥测启发式归因，非官方确认；原始受击HP与来源推断分开统计。",
                             "等级阈值和职责权重是自定义值；测试检查计算一致性，尚未通过录像或人工标注验证战术合理性。",
                             *NOTES[:2],
                             f"伤害统计保留原始HP；计分权重：高可信1，中可信{self.config['medium_confidence_weight']:g}，低可信{self.config['low_confidence_weight']:g}。低可信多候选分摊同一份伤害，仅计输出，不推算击杀或控场。",
                             *NOTES[3:5],
                             f"地形触发后{self.config['terrain_followup_s']:g}秒内形成≥{self.config['terrain_min_damage']:g}HP敌半场输出才计骚扰；连续触发去重。",
                             *NOTES[6:]]}
        return result, summary

    def series(self):
        if not self.entities:
            return {}
        grid = np.arange(math.floor(self.lo), math.ceil(self.hi)+1, dtype=float)
        result, lookup = {}, {}
        totals = {actor: (defaultdict(float), defaultdict(float)) for actor in self.entities}
        for actor, ent in self.entities.items():
            entry = {k: ent[k] for k in ("camp", "robot_id", "robot_type")}
            entry.update(times=grid.tolist(), score=[], calculated_score=[], rating_eligible=self.assessment(actor)["rating_eligible"], kills=[], assists=[], deaths=[], damage=[])
            result.setdefault(actor[0], []).append(entry)
            lookup[actor] = entry
        cursor = 0
        for now in grid:
            limit = self.hi+.51 if now == grid[-1] else now
            while cursor < len(self.items) and self.items[cursor]["time"] <= limit:
                item = self.items[cursor]
                values, units = totals[(item["camp"], item["robot_id"])]
                values[item["key"]] += item["value"]
                units[item["key"]] += item["units"]
                cursor += 1
            for actor, ent in self.entities.items():
                values, units = totals[actor]
                row, entry = self.row(ent, values, units, explain=False), lookup[actor]
                entry["calculated_score"].append(row["total_score"])
                entry["score"].append(row["total_score"] if self.assessment(actor, limit)["rating_eligible"] else None)
                for key in ("kills", "assists", "deaths"):
                    entry[key].append(row["components"][key]["value"])
                entry["damage"].append(round(sum(values.get(k, 0) for k in ("damage_to_robots", "damage_to_base", "damage_to_outpost")), 1))
        return result
