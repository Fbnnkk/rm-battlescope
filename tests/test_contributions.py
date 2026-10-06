"""Scenario tests for v3: credit useful actions without rewarding idle farming."""
import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from test_scoring import _track, _attack, RED, BLUE, HERO, INF3, INF4, ENGINEER, SENTRY, AERIAL, BASE
from rmuc_trajectory.pipeline import MatchEvent
from rmuc_trajectory.scoring import compute_scores, compute_timeseries_scores, compute_score_report
from rmuc_trajectory.contributions import Evaluation, load_config


def event(time, kind, camp=RED, robot_id=None, category=None, value=None, note=None, target_type=None):
    return MatchEvent(time, kind, robot_id, None, camp, '测试学校', None, target_type, category, value, note)


def row(scores, role=HERO, camp=RED):
    return next(s for s in scores if s['camp']==camp and s['robot_type']==role)


class ContributionTests(unittest.TestCase):
    def score(self, tracks=None, attacks=None, events=None, buffs=None, match=None):
        return compute_scores(match or {}, tracks or [_track()], events or [], attacks or [], buffs or [])[0]

    def test_idle_survival_ammo_and_enemy_half_are_neutral_for_all_roles(self):
        for role in (HERO, INF3, INF4, ENGINEER, SENTRY, AERIAL):
            t=_track(robot_type=role,times=list(range(60)),x_vals=[20]*60,cum_42mm=list(range(60)))
            s=row(self.score([t],match={'胜方':RED}),role)
            self.assertEqual(s['calculated_score'],5)
            self.assertEqual(s['total_score'],None if role==ENGINEER else 5)

    def test_receiving_damage_is_not_a_direct_penalty(self):
        a=_attack(attacker_camp=BLUE,attacker_id=2,victim_camp=RED,victim_id=1,damage=60)
        s=row(self.score(attacks=[a]))
        self.assertEqual(s['components']['damage_received']['value'],60)
        self.assertEqual(s['total_score'],5)

    def test_same_team_hits_invalid_damage_and_unknown_damage_do_not_count(self):
        attacks=[replace(_attack(),victim_camp=RED),replace(_attack(),damage=-1),
                 replace(_attack(),damage=float('nan')),replace(_attack(),damage=None)]
        s,summary=compute_scores({},[_track()],[],attacks,[])
        self.assertEqual(s[0]['total_score'],5)
        self.assertEqual(summary['excluded_attack_count'],4)

    def test_medium_damage_retains_stats_with_discounted_points(self):
        hi=row(self.score(attacks=[_attack(damage=100)]))
        mid=row(self.score(attacks=[replace(_attack(damage=100),confidence='medium')]))
        self.assertEqual(mid['components']['damage_to_robots']['value'],100)
        self.assertLess(mid['total_score'],hi['total_score'])

    def test_diminishing_damage_returns(self):
        def points(d):return row(self.score(attacks=[_attack(damage=d)]))['raw_score']
        self.assertGreater(points(1000),points(500))
        self.assertLess(points(1000)-points(500),points(500)-points(0))
        self.assertLessEqual(points(100000),10)

    def test_hero_has_more_objective_credit_than_infantry(self):
        hero=row(self.score(attacks=[_attack(victim_type=BASE,damage=600)]))
        infantry=row(self.score([_track(robot_type=INF3)],[_attack(attacker_type=INF3,victim_type=BASE,damage=600)]),INF3)
        self.assertGreater(hero['total_score'],infantry['total_score'])

    def test_base_hp_gap_and_final_winner_do_not_rewrite_damage(self):
        t=_track();a=_attack(victim_type=BASE,damage=100)
        s0=row(self.score([t],[a],match={'胜方':BLUE}))
        red=_track(robot_id=10,robot_type=BASE,health=[5000]*3)
        blue=_track(camp=BLUE,robot_id=110,robot_type=BASE,health=[100]*3)
        s1=row(self.score([t,red,blue],[a],match={'胜方':RED}))
        self.assertEqual(s0['total_score'],s1['total_score'])

    def front(self, retreat=False):
        t=_track(robot_type=INF3,times=list(range(11)),x_vals=[15]*11)
        x=[18,18,18,18,18,18.5,19,19.5,20,20,20] if retreat else [18]*11
        v=_track(camp=BLUE,robot_id=2,robot_type=INF3,times=list(range(11)),x_vals=x)
        a=replace(_attack(attacker_type=INF3,hit_time=4,damage=60),attacker_xy=(15,7.5),victim_xy=(18,7.5))
        return t,v,a

    def test_front_control_requires_an_actual_opponent_and_recent_hit(self):
        t,v,a=self.front()
        self.assertEqual(row(self.score([t,v]),INF3)['components']['pressure']['value'],0)
        self.assertEqual(row(self.score([t],[a]),INF3)['components']['pressure']['value'],0)
        s=row(self.score([t,v],[a]),INF3)
        self.assertGreater(s['components']['pressure']['score'],0)
        self.assertEqual(s['components']['suppression']['value'],0)

    def test_front_control_is_shared_per_target_second(self):
        t,v,a=self.front()
        t2=_track(robot_id=3,robot_type=INF4,times=list(range(11)),x_vals=[16]*11)
        a2=replace(a,attacker_robot_id=3,attacker_type=INF4,attacker_xy=(16,7.5))
        solo=row(self.score([t,v],[a]),INF3)['components']['pressure']['value']
        scores=self.score([t,t2,v],[a,a2])
        shared=sum(s['components']['pressure']['value'] for s in scores if s['camp']==RED)
        self.assertEqual(solo,shared)

    def test_retreat_credit_waits_for_confirmation_and_never_leaks_future(self):
        t,v,a=self.front(True)
        args=({},[t,v],[],[a],[])
        scores,_=compute_scores(*args);series=compute_timeseries_scores(*args)
        s=row(scores,INF3)
        proof=next(i for i in s['contribution_events'] if i['key']=='suppression')
        self.assertEqual(proof['time'],8)
        self.assertEqual(proof['start'],4)
        curve=series[RED][0]['score']
        self.assertGreater(curve[8],curve[7])
        prefix=[replace(tr,**{name: getattr(tr,name)[:8] for name in tr.__dataclass_fields__ if isinstance(getattr(tr,name),np.ndarray)}) for tr in (t,v)]
        prefix_score=row(self.score(prefix,[a]),INF3)
        self.assertEqual(prefix_score['components']['suppression']['value'],0)
        self.assertEqual(prefix_score['total_score'],curve[7])

    def test_gaps_cleaned_spikes_and_dead_targets_cannot_establish_retreat(self):
        for field in ('gap','spike','dead'):
            t,v,a=self.front(True)
            if field=='gap':v.observed_mask[6]=False
            elif field=='spike':v.raw_xy[6]=[0,0]
            else:v.health[7:]=0
            self.assertEqual(row(self.score([t,v],[a]),INF3)['components']['suppression']['value'],0)

    def test_fire_reduction_is_only_inferred_when_enemy_previously_fired(self):
        t,v,a=self.front()
        events=[event(i,'发弹',BLUE,2) for i in (0,1,2,3)]
        s=row(self.score([t,v],[a],events),INF3)
        self.assertEqual(s['components']['suppression']['value'],1)
        self.assertIn('40%',next(i['reason'] for i in s['contribution_events'] if i['key']=='suppression'))

    def test_aimed_fire_retreat_can_contribute_without_inventing_damage(self):
        t,v,_=self.front(True)
        events=[event(4,'发弹',robot_id=1)]*3
        s=row(self.score([t,v],events=events),INF3)
        self.assertEqual(s['components']['suppression']['value'],1)
        self.assertEqual(s['components']['damage_to_robots']['value'],0)
        proof=next(i for i in s['contribution_events'] if i['key']=='suppression')
        self.assertEqual(proof['time'],8)
        self.assertIn('没有确认命中',proof['reason'])
        hit_based=row(self.score([t,v],[_attack(attacker_type=INF3,hit_time=4)],events),INF3)
        self.assertEqual(hit_based['components']['suppression']['value'],1)
        self.assertGreater(hit_based['components']['suppression']['score'],s['components']['suppression']['score'])

    def test_empty_fire_wrong_heading_and_stationary_enemies_do_not_reward(self):
        for mode in ('direction','stationary','few'):
            t,v,_=self.front(mode!='stationary')
            if mode=='direction':t.heading_deg[:]=180
            events=[event(4,'发弹',robot_id=1)]*(2 if mode=='few' else 3)
            s=row(self.score([t,v],events=events),INF3)
            self.assertEqual(s['total_score'],5)

    def test_fire_suppression_ambiguous_targets_and_observation_gaps_are_rejected(self):
        t,v,_=self.front(True)
        second=_track(camp=BLUE,robot_id=3,robot_type=SENTRY,times=list(range(11)),x_vals=[18]*11)
        events=[event(4,'发弹',robot_id=1)]*3
        self.assertEqual(row(self.score([t,v,second],events=events),INF3)['components']['suppression']['value'],0)
        v.observed_mask[6]=False
        self.assertEqual(row(self.score([t,v],events=events),INF3)['components']['suppression']['value'],0)

    def test_old_nonfatal_hit_cannot_be_claimed_as_later_kill(self):
        t=_track(times=list(range(11)))
        v=_track(camp=BLUE,robot_id=2,times=list(range(11)),health=[100]*10+[0])
        self.assertEqual(row(self.score([t,v],[_attack(hit_time=4)]))['components']['kills']['value'],0)

    def test_terrain_requires_useful_enemy_half_followup_and_dedupes(self):
        t,v,a=self.front()
        events=[event(1,'增益',robot_id=1,category='飞坡'),event(1,'增益',robot_id=1,category='台阶跨越'),event(2,'增益',robot_id=1,category='飞坡')]
        idle=row(self.score([t,v],events=events),INF3)
        active=row(self.score([t,v],[a],events),INF3)
        self.assertEqual(idle['components']['terrain_harassment']['value'],0)
        self.assertEqual(active['components']['terrain_harassment']['value'],1)
        own_half=replace(a,victim_xy=(8,7.5))
        self.assertEqual(row(self.score([t,v],[own_half],events),INF3)['components']['terrain_harassment']['value'],0)

    def test_terrain_expiry_and_low_damage_do_not_reward(self):
        t,v,a=self.front()
        events=[event(1,'增益',robot_id=1,category='飛坡')]
        self.assertEqual(row(self.score([t,v],[a],events),INF3)['components']['terrain_harassment']['value'],0)
        events=[event(1,'增益',robot_id=1,category='飞坡')]
        self.assertEqual(row(self.score([t,v],[replace(a,damage=20)],events),INF3)['components']['terrain_harassment']['value'],0)

    def test_engineering_assembly_has_grade_weights_and_duplicate_protection(self):
        t=_track(robot_type=ENGINEER,times=list(range(11)))
        low=event(5,'装配成功',category='等级1',value=15)
        high=event(5,'装配成功',category='等级3',value=60)
        slo=row(self.score([t],events=[low]),ENGINEER)
        shi=row(self.score([t],events=[high,high]),ENGINEER)
        self.assertGreater(shi['total_score'],slo['total_score'])
        self.assertEqual(shi['components']['assembly']['value'],1)
        self.assertIn('60秒',shi['contribution_events'][0]['reason'])

    def test_engineering_assembly_is_not_assigned_to_wrong_or_ambiguous_robot(self):
        t=_track(robot_type=ENGINEER,times=list(range(11)))
        t2=_track(robot_id=2,robot_type=ENGINEER,times=list(range(11)))
        event0=event(5,'装配成功',category='等级2',value=15)
        for events,tracks in [([replace(event0,camp=BLUE)],[t]),([event0],[t,t2])]:
            self.assertTrue(all(s['components']['assembly']['value']==0 for s in self.score(tracks,events=events)))

    def test_engineering_terrain_can_support_mission_without_combat(self):
        t=_track(robot_type=ENGINEER,times=list(range(11)))
        events=[event(1,'增益',robot_id=1,category='台阶跨越'),event(7,'装配成功',category='等级2',value=12)]
        s=row(self.score([t],events=events),ENGINEER)
        self.assertEqual(s['components']['terrain_logistics']['value'],1)
        self.assertEqual(s['components']['terrain_harassment']['value'],0)

    def test_radar_mark_and_counter_credit_the_opposite_of_affected_robot(self):
        t=_track(camp=BLUE,robot_id=2,robot_type=INF3,times=list(range(11)))
        t.vulnerable[3:7]=1
        events=[event(5,'雷达反制UAV',camp=BLUE,note='反制方=红')]
        scores=self.score([t],events=events)
        s=row(scores,'雷达')
        self.assertTrue(s['event_only'])
        self.assertEqual(s['components']['radar_marking']['value'],4)
        self.assertEqual(s['components']['radar_counter']['value'],1)
        self.assertIsNone(s['evidence']['position_coverage'])

    def test_radar_counter_contradictory_note_not_scored(self):
        scores=self.score(events=[event(1,'雷达反制UAV',camp=BLUE,note='反制方=蓝')])
        self.assertFalse(any(s['robot_type']=='雷达' for s in scores))

    def test_radar_support_damage_is_not_double_counted_as_robot_output(self):
        t,v,a=self.front();v.vulnerable[:]=1
        s=row(self.score([t,v],[a]),'雷达')
        self.assertEqual(s['components']['radar_assists']['value'],60)
        self.assertEqual(s['components']['damage_to_robots']['value'],0)

    def test_dart_uses_direct_hit_damage_not_gate_accuracy(self):
        events=[event(1,'飞镖闸门开',value=3),event(2,'飞镖命中',value=625,target_type=BASE)]
        scores=self.score(events=events)
        s=row(scores,'飞镖')
        self.assertEqual(s['components']['dart_hits']['value'],1)
        self.assertEqual(s['components']['damage_to_base']['value'],625)
        self.assertEqual(row(scores)['components']['damage_to_base']['value'],0)
        gate=row(self.score(events=events[:1]),'飞镖')
        self.assertIsNone(gate['total_score'])
        self.assertIn('证据不足',gate['assessment_status'])

    def test_rune_needs_confirmed_team_buff_not_just_arm_hits(self):
        t=_track(robot_type=INF3,times=list(range(11)),x_vals=[9]*11)
        events=[event(4,'能量机关',category='rune_type=1.0',note='arm_cnt=1.0,avg_round=0.0'),event(4,'发弹',robot_id=1)]
        self.assertEqual(row(self.score([t],events=events),INF3)['components']['rune_activations']['value'],0)
        events += [event(5,'增益',robot_id=1,category='大能量机关增益')]
        self.assertEqual(row(self.score([t],events=events),INF3)['components']['rune_activations']['value'],1)

    def test_rune_multiple_candidates_share_one_success(self):
        t=_track(robot_type=INF3,times=list(range(11)),x_vals=[9]*11)
        t2=_track(robot_id=3,robot_type=INF4,times=list(range(11)),x_vals=[9]*11)
        events=[event(4,'能量机关'),event(4,'发弹',robot_id=1),event(4,'发弹',robot_id=3),
                event(5,'增益',robot_id=1,category='大能量机关增益'),event(5,'增益',robot_id=3,category='大能量机关增益')]
        scores=self.score([t,t2],events=events)
        self.assertEqual(sum(s['components']['rune_activations']['value'] for s in scores),1)

    def test_kill_never_uses_a_future_hit_and_ties_do_not_fake_last_hit(self):
        t=_track(times=[0,1,2,3,4]);v=_track(camp=BLUE,robot_id=2,times=[0,1,2,3,4],health=[100,100,100,0,0])
        a=_attack(hit_time=4)
        self.assertEqual(row(self.score([t,v],[a]))['components']['kills']['value'],0)
        a=_attack(hit_time=3);a2=replace(a,attacker_robot_id=3)
        self.assertEqual(row(self.score([t,v],[a,a2]))['components']['kills']['value'],0)

    def test_assist_credit_reflects_damage_share_without_inventing_official_kills(self):
        t=_track(times=[0,1,2,3,4]);t2=_track(robot_id=3,robot_type=INF3,times=[0,1,2,3,4])
        v=_track(camp=BLUE,robot_id=2,times=[0,1,2,3,4],health=[100,100,100,0,0])
        attacks=[_attack(attacker_id=3,attacker_type=INF3,hit_time=1,damage=90),_attack(hit_time=3,damage=10)]
        scores=self.score([t,t2,v],attacks)
        self.assertEqual(row(scores)['components']['kills']['value'],1)
        self.assertEqual(row(scores,INF3)['components']['assists']['value'],1)
        self.assertIn('90%',next(i['reason'] for i in row(scores,INF3)['contribution_events'] if i['key']=='assists'))

    def test_cover_requires_enemy_that_really_attacked_protected_role(self):
        t=_track(robot_type=SENTRY,times=list(range(6)));v=_track(camp=BLUE,robot_id=2,times=list(range(6)))
        hero=_track(robot_id=3,times=list(range(6)))
        ret=_attack(attacker_type=SENTRY,hit_time=3)
        attack=_attack(attacker_camp=BLUE,attacker_id=2,victim_camp=RED,victim_id=3,victim_type=HERO,hit_time=2)
        no=row(self.score([t,v,hero],[ret]),SENTRY)
        yes=row(self.score([t,v,hero],[attack,ret]),SENTRY)
        self.assertEqual(no['components']['ally_cover']['value'],0)
        self.assertGreater(yes['components']['ally_cover']['value'],0)

    def test_objective_defense_requires_living_own_building_and_nearby_enemy(self):
        t=_track(robot_type=SENTRY,times=list(range(6)),x_vals=[3]*6)
        v=_track(camp=BLUE,robot_id=2,times=list(range(6)),x_vals=[4]*6)
        base=_track(robot_id=10,robot_type=BASE,times=list(range(6)))
        a=_attack(attacker_type=SENTRY,hit_time=3)
        self.assertGreater(row(self.score([t,v,base],[a]),SENTRY)['components']['objective_defense']['score'],0)
        base.health[:]=0
        self.assertEqual(row(self.score([t,v,base],[a]),SENTRY)['components']['objective_defense']['score'],0)

    def test_direct_damage_coverage_is_distinct_from_telemetry_coverage(self):
        e=event(1,'受击',BLUE,2,category='17mm',value=100)
        s,summary=compute_scores({},[_track()], [e],[_attack(damage=20)],[])
        self.assertEqual(summary['projectile_attribution_coverage'],.2)
        self.assertEqual(s[0]['evidence']['health_coverage'],1)

    def test_dimensions_components_series_and_legacy_comparison_agree(self):
        t,v,a=self.front(True)
        events=[event(2,'飞镖命中',BLUE,value=625,target_type=BASE)]
        scores,summary,series=compute_score_report({},[t,v],events,[a],[])
        for s in scores:
            line=next(l for l in series[s['camp']] if l['robot_id']==s['robot_id'])
            self.assertEqual(line['score'][-1],s['total_score'])
            self.assertAlmostEqual(5+sum(i['score'] for i in s['explanation']),s['raw_score'],places=5)
            self.assertAlmostEqual(sum(i['score'] for i in s['dimensions']),s['raw_score']-5,places=5)
            self.assertTrue(all(math.isfinite(i['score']) for i in s['explanation']))
            self.assertLessEqual(sum(max(0,i['score']) for i in s['explanation']),5.00001)
        self.assertEqual(row(scores,INF3)['legacy_score'] is None,False)
        self.assertEqual(summary['configuration']['version'],summary['rule_version'])

    def test_bad_rule_configuration_is_rejected(self):
        config=load_config();config['roles']['英雄']['robot_damage'][1]=0
        with patch('rmuc_trajectory.contributions.json.loads',return_value=config),self.assertRaises(ValueError):
            load_config()

    def test_same_causal_curve_as_prefix_for_every_second(self):
        t,v,a=self.front(True)
        scores,summary,series=compute_score_report({},[t,v],[],[a],[])
        for time in range(11):
            tracks=[replace(tr,**{name:getattr(tr,name)[:time+1] for name in tr.__dataclass_fields__ if isinstance(getattr(tr,name),np.ndarray)}) for tr in (t,v)]
            prefix=row(self.score(tracks,[a] if a.hit_time<=time else []),INF3)
            self.assertEqual(prefix['total_score'],series[RED][0]['score'][time])


if __name__=='__main__':unittest.main()
