"""Shared CSV export for standalone and HTTP reports."""

import csv
from pathlib import Path


def write_scores_csv(path: Path, scores: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        contribution_columns = [('pressure','前线控制目标秒'),('suppression','压制迹象次数'),
                                ('terrain_harassment','地形骚扰次数'),('assembly','装配成功次数'),
                                ('rune_activations','成功能量机关支援'),('radar_marking','雷达标记目标秒'),
                                ('radar_counter','雷达反制次数'),('dart_hits','飞镖命中次数')]
        writer.writerow(["阵营", "学校", "兵种", "编号", "表现分", "等级", "击杀", "助攻", "阵亡", "机器人伤害", "建筑伤害", "血量数据覆盖率",
                         '旧规则评分','职责','证据模式','高可信归因伤害HP','中可信归因伤害HP','原始命中事件伤害HP','低可信分摊伤害HP','评级状态','评级依据'] + [label for _,label in contribution_columns]
                        + [label+'得分' for label in ('输出','攻坚','协同','控场','机动','支援','代价')])
        for s in scores:
            c = s["components"]
            writer.writerow([s["camp"], s["school"], s["robot_type"], s["robot_id"], s["total_score"], s["grade"],
                             c["kills"]["value"], c["assists"]["value"], c["deaths"]["value"],
                             c["damage_to_robots"]["value"], c["damage_to_base"]["value"] + c["damage_to_outpost"]["value"],
                             s["evidence"]["health_coverage"],s.get('legacy_score'),s.get('mission'),s['evidence'].get('mode'),s['evidence'].get('high_confidence_damage',0),s['evidence'].get('medium_confidence_damage',0),s['evidence'].get('recorded_damage',0),s['evidence'].get('low_confidence_damage',0),s['assessment_status'],s['assessment_reason']]
                            + [c[key]['value'] for key,_ in contribution_columns]
                            + [d['score'] for d in s.get('dimensions',[])])
