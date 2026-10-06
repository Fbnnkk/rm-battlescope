"""Evidence-backed scorecards and grouped, navigable match turning points."""
import math
import re


def number(note, key):
    found = re.search(rf"(?:^|[,;\s]){key}=([-+0-9.]+)", note or "")
    try:
        value = float(found.group(1)) if found else None
        return value if value is not None and math.isfinite(value) else None
    except ValueError:
        return None


def event_detail(event):
    if event.event_type == '装配成功':
        grade = re.fullmatch(r'等级\s*([1-5])', event.category or '')
        return ' · '.join([('科技核心等级' + grade.group(1)) if grade else '装配成功记录',
                          f'耗时{event.value:g}秒' if event.value is not None else '耗时未记录'])
    if event.event_type == '飞镖命中':
        return f"命中{event.target_type or '目标'} · 原始伤害{event.value:g} HP" if event.value is not None else '原始命中事件'
    if event.event_type == '飞镖闸门开':
        return '发射闸门开启记录；不能据此确认发射或命中'
    if event.event_type == '雷达反制UAV':
        return f"{event.camp or '未知'}方无人机被反制；主体是被反制方"
    return event.category or '原始事件记录'


def build_moments(tracks, events):
    moments = []
    for track in tracks:
        for i in range(1, len(track.times)):
            if not (track.observed_mask[i-1] and track.observed_mask[i] and track.times[i]-track.times[i-1] <= 1.1):
                continue
            previous, current = track.health[i-1], track.health[i]
            if not (math.isfinite(previous) and math.isfinite(current)):
                continue
            structure = track.key.robot_type in {'基地', '前哨站'}
            if previous > 0 and current == 0:
                kind, ending = ('建筑被摧毁', '被摧毁') if structure else ('阵亡', '阵亡')
            elif previous == 0 and current > 0:
                kind, ending = '复活', '恢复血量'
            else:
                continue
            moments.append({'time': float(track.times[i]), 'camp': track.key.camp,
                            'robot_id': track.key.robot_id, 'kind': kind,
                            'title': f'{track.key.camp}方{track.key.robot_type}{ending}',
                            'detail': f'相邻原始遥测血量 {previous:g} → {current:g} HP',
                            'raw_records': [{'source': '原始血量遥测', 'time': float(track.times[i]),
                                             'previous_health': float(previous), 'health': float(current)}]})
    groups = {}
    rune_sessions = []
    for event in sorted(events, key=lambda e: e.time):
        if event.event_type == '能量机关':
            identity = (event.camp, event.robot_id, event.category)
            session = groups.get(identity)
            arms = number(event.note, 'arm_cnt')
            reset = session is not None and arms is not None and session['last_arms'] is not None and arms < session['last_arms']
            if session is None or event.time-session['end'] > 12 or reset:
                session = {'time': event.time, 'end': event.time, 'camp': event.camp,
                           'robot_id': event.robot_id, 'kind': '能量机关',
                           'title': f"{event.camp or '未知'}方能量机关击打过程", 'raw_records': [], 'last_arms': None}
                rune_sessions.append(session); groups[identity] = session
            session['end'] = event.time
            session['last_arms'] = arms
            session['raw_records'].append(event.to_jsonable())
        elif event.event_type in {'装配成功', '飞镖命中', '飞镖闸门开', '雷达反制UAV'} or (event.event_type == '增益' and event.category in {'小能量机关增益', '大能量机关增益'}):
            moments.append({'time': event.time, 'camp': event.camp, 'robot_id': event.robot_id,
                            'kind': '机关增益' if event.event_type == '增益' else event.event_type,
                            'title': f"{event.camp or '未知'}方{event.robot_type or ''} · {event.category if event.event_type == '增益' else event.event_type}",
                            'detail': event_detail(event), 'raw_records': [event.to_jsonable()]})
    for session in rune_sessions:
        records = session['raw_records']
        arms = [number(r.get('note'), 'arm_cnt') for r in records]
        rounds = [number(r.get('note'), 'avg_round') for r in records]
        arms, rounds = [v for v in arms if v is not None], [v for v in rounds if v is not None]
        parts = [f"{len(records)}条连续击打记录，持续{session['end']-session['time']:g}秒"]
        if arms: parts.append(f'激活臂数最高{max(arms):g}')
        if rounds: parts.append(f'末次平均环数{rounds[-1]:g}')
        buffs = [e for e in events if e.event_type == '增益' and e.camp == session['camp']
                 and e.category in {'小能量机关增益', '大能量机关增益'} and session['time'] <= e.time <= session['end']+3]
        parts.append('期间/随后记录' + buffs[0].category if buffs else '未关联到成功增益记录；击打不等于激活成功')
        session['detail'] = ' · '.join(parts)
        session['record_count'] = len(records)
        session.pop('last_arms')
        moments.append(session)
    return sorted(moments, key=lambda m: m['time'])


def build_review(match, tracks, events, scores, series, scoring_summary):
    teams = []
    for camp in ('红', '蓝'):
        members = [s for s in scores if s['camp'] == camp]
        rated = [s for s in members if s.get('rating_eligible', s['total_score'] is not None)]
        def total(key):
            return sum(s['components'].get(key, {}).get('value', 0) for s in members)
        teams.append({'camp': camp, 'school': match.get(camp+'方学校', ''),
                      'mean_score': round(sum(s['total_score'] for s in rated)/len(rated), 2) if rated else None,
                      'rated_count': len(rated), 'unrated_count': len(members)-len(rated), 'entity_count': len(members),
                      'kills': total('kills'), 'deaths': total('deaths'),
                      'damage': round(total('damage_to_robots')+total('damage_to_base')+total('damage_to_outpost'), 1),
                      'structure_damage': round(total('damage_to_base')+total('damage_to_outpost'), 1),
                      'objective_damage': [t for t in scoring_summary.get('damage_targets', []) if t['camp'] != camp and t['robot_type'] in {'基地','前哨站'}]})
    rated = [s for s in scores if s.get('rating_eligible', s['total_score'] is not None)]
    high = max(rated, key=lambda s: (s['total_score'], s.get('raw_score', 0)), default=None)
    times = [v for t in tracks if len(t.times) for v in (float(t.times[0]), float(t.times[-1]))]
    return {'teams': teams, 'moments': build_moments(tracks, events),
            'best': {k: high[k] for k in ('camp','robot_id','robot_type','school','total_score','grade')} if high else None,
            'scoring': scoring_summary, 'window': [min(times), max(times)] if times else [0,0],
            'notes': scoring_summary.get('notes', [])}
