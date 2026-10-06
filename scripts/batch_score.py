#!/usr/bin/env python3
"""Inspect role distributions across regional matches without forcing medians.

Reports are diagnostic samples, not a claim that heuristic scores have been
validated against expert strategy judgments. The SQLite database stays read-only.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rmuc_trajectory.buffs import build_buff_intervals
from rmuc_trajectory.combat import infer_attacks
from rmuc_trajectory.paths import DEFAULT_OUTPUT_ROOT
from rmuc_trajectory.pipeline import load_and_clean_tracks, load_match_events, open_readonly
from rmuc_trajectory.scoring import compute_score_report


def sample_matches(matches, count):
    regions = sorted({m['赛区'] for m in matches})
    chosen = []
    for i, region in enumerate(regions):
        group = sorted([m for m in matches if m['赛区']==region], key=lambda m: (m['时长秒'] or 0, m['game_id']))
        take = min(len(group), count//len(regions)+(i < count%len(regions)))
        if take:
            chosen.extend(group[int(j)] for j in np.linspace(0,len(group)-1,take))
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample',type=int,default=30,help='Across all regions and duration quantiles')
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=None)
    args=parser.parse_args()
    if args.sample <= 0:parser.error('--sample must be positive')
    database=ROOT/'dataset'/'rmuc_2026_region_dataset.sqlite'
    connection=open_readonly(database)
    try:matches=[dict(r) for r in connection.execute('SELECT * FROM matches ORDER BY game_id')]
    finally:connection.close()
    selected=matches if args.all else sample_matches(matches,min(args.sample,len(matches)))
    output=args.output_dir or DEFAULT_OUTPUT_ROOT/('scoring-v3-audit-'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=True)
    report_path=output/'scoring-audit.json'
    if report_path.exists():raise FileExistsError(f'不会覆盖已有审查文件：{report_path}')
    roles,records,errors=defaultdict(list),[],[]
    start=time.perf_counter()
    for index,m in enumerate(selected):
        gid=m['game_id']
        try:
            match,tracks=load_and_clean_tracks(database,gid)
            _,objectives=load_and_clean_tracks(database,gid,robot_types=('基地','前哨站'),include_static=True)
            events=load_match_events(database,gid)
            attacks,_=infer_attacks(events,tracks+objectives)
            buffs,_=build_buff_intervals(events,tracks+objectives,max(t.times[-1] for t in tracks))
            scores,summary,series=compute_score_report(match,tracks+objectives,events,attacks,buffs)
            for s in scores:
                line=next(l for l in series[s['camp']] if l['robot_id']==s['robot_id'])
                assert line['score'][-1]==s['total_score'],'curve endpoint mismatch'
                assert abs(5+sum(c['score'] for c in s['explanation'])-s['raw_score'])<.00002
                assert abs(sum(c['score'] for c in s['dimensions'])+5-s['raw_score'])<.00002
                assert s['total_score'] is None or 0 <= s['total_score'] <= 10
                roles[s['robot_type']].append(s)
            records.append({'game_id':gid,'region':match['赛区'],'duration':match['时长秒'],
                            'entities':len(scores),'rated_entities':summary['rated_robot_count'],'unrated_entities':summary['unrated_robot_count'],
                            'projectile_attribution_coverage':summary['projectile_attribution_coverage'],
                            'low_confidence_damage':summary['low_confidence_projectile_damage'],'unattributed_damage':summary['unattributed_projectile_damage'],'ledger_items':summary['contribution_event_count'],
                            'accepted_attacks':len(attacks)-summary['excluded_attack_count'],
                            'endpoints_consistent':True,'contributions_consistent':True})
        except Exception as exc:errors.append({'game_id':gid,'error':str(exc)})
        print(f'{index+1}/{len(selected)} · {gid} · errors {len(errors)}',flush=True)
    stats={}
    for role,rows in roles.items():
        rated=[s for s in rows if s['rating_eligible']]
        numbers=np.array([s['total_score'] for s in rated], dtype=float)
        raw=np.array([s['components']['damage_to_robots']['value'] for s in rows])
        nonzero=raw[raw>0]
        stats[role]={'count':len(rows),'rated_count':len(rated),'unrated_count':len(rows)-len(rated),'p25':round(float(np.percentile(numbers,25)),3) if len(numbers) else None,
                     'median':round(float(np.median(numbers)),3) if len(numbers) else None,'p75':round(float(np.percentile(numbers,75)),3) if len(numbers) else None,
                     'min':float(min(numbers)) if len(numbers) else None,'max':float(max(numbers)) if len(numbers) else None,
                     'grade_counts':{g:sum(s['grade']==g for s in rated) for g in 'SABCD'},
                     'neutral_count':sum(s['assessment_kind']=='observed_neutral' for s in rows),
                     'max_score_count':int(sum(numbers==10)),
                     'positive_damage_median':float(np.median(nonzero)) if len(nonzero) else None,
                     'contribution_counts':{k:round(sum(s['components'][k]['value'] for s in rows),3)
                                            for k in ('kills','assists','pressure','suppression','terrain_harassment',
                                                      'objective_defense','ally_cover','assembly','rune_activations',
                                                      'radar_counter','radar_marking','dart_hits')}}
        print(role,json.dumps(stats[role],ensure_ascii=False),flush=True)
    report={'selection':'all' if args.all else '各赛区按比赛时长等距取样（含最短与最长）',
            'requested':len(selected),'completed':len(records),'errors':errors,
            'elapsed_seconds':round(time.perf_counter()-start,2),'roles':stats,'matches':records,
            'notes':['仅检查分数分布与计算一致性；未用专家标注校准，不强行把各兵种中位数调成5。',
                     '包含低可信伤害分摊，计分权重0.35；分布只统计已评级实体，证据不足数量另列。',
                     '归因覆盖是已分配HP占原始受击HP的比例，不能代表来源推断准确率。缺少同步录像及人工伤害/职责标注，无法校准归因准确率或等级阈值。']}
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved',report_path,flush=True)
    return 1 if errors else 0


if __name__=='__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
