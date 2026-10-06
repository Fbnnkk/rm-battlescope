"""Public ratings, aggregation denominators and review event semantics."""
from dataclasses import replace
import unittest
import numpy as np

from test_scoring import _track, _attack
from rmuc_trajectory.pipeline import MatchEvent
from rmuc_trajectory.scoring import compute_score_report
from rmuc_trajectory.review import build_review, build_moments


def event(time, kind, camp='红', category=None, note=None, value=None):
    return MatchEvent(time, kind, None, None, camp, '测试', None, None, category, value, note)


class ReviewEvidenceTests(unittest.TestCase):
    def report(self, tracks, events=None, attacks=None):
        scores, summary, series=compute_score_report({},tracks,events or [],attacks or [],[])
        return scores, summary, series, build_review({},tracks,events or [],scores,series,summary)

    def test_gate_only_dart_is_null_in_score_grade_curve_and_legacy_comparison(self):
        scores, summary, series, review=self.report([_track()], [event(1,'飞镖闸门开')])
        dart=next(s for s in scores if s['robot_type']=='飞镖')
        self.assertFalse(dart['rating_eligible'])
        self.assertIsNone(dart['total_score'])
        self.assertIsNone(dart['grade'])
        self.assertIsNone(dart['legacy_score'])
        self.assertIsNone(dart['score_change'])
        self.assertEqual(dart['calculated_score'],5)
        self.assertEqual(next(t for t in series['红'] if t['robot_type']=='飞镖')['score'],[None]*3)
        red=review['teams'][0]
        self.assertEqual((red['rated_count'],red['unrated_count'],red['mean_score']),(1,1,5))
        self.assertNotEqual(review['best']['robot_type'],'飞镖')

    def test_valid_activity_with_no_contribution_can_be_neutral(self):
        scores,_,series,_=self.report([_track()])
        self.assertTrue(scores[0]['rating_eligible'])
        self.assertEqual(scores[0]['assessment_kind'],'observed_neutral')
        self.assertEqual(scores[0]['total_score'],5)
        self.assertEqual(series['红'][0]['score'],[None,5,5])

    def test_inactive_aerial_and_missing_operational_evidence_are_not_neutral_grades(self):
        for role in ('空中','工程'):
            with self.subTest(role=role):
                t=_track(robot_type=role, health=[0,0,0] if role=='空中' else [100]*3)
                scores,summary,_,review=self.report([t])
                self.assertIsNone(scores[0]['total_score'])
                self.assertIsNone(review['best'])
                self.assertIsNone(review['teams'][0]['mean_score'])
                self.assertEqual(summary['rated_robot_count'],0)

    def test_contribution_is_rated_without_claiming_full_role_observability(self):
        tracks=[_track(robot_type='工程')]
        scores,_,_,_=self.report(tracks,[event(1,'装配成功',category='等级2',value=7)])
        self.assertTrue(scores[0]['rating_eligible'])
        self.assertIn('已覆盖',scores[0]['assessment_reason'])
        self.assertEqual(scores[0]['contribution_events'][0]['confidence'],'role_inferred')

    def test_damage_event_and_inference_have_distinct_source_fields(self):
        events=[MatchEvent(1,'飞镖命中',None,'飞镖','红','测试',110,'基地',None,625,None)]
        scores,_,_,_=self.report([_track()],events,[_attack(damage=20)])
        hero=next(s for s in scores if s['robot_type']=='英雄')
        dart=next(s for s in scores if s['robot_type']=='飞镖')
        self.assertEqual(hero['evidence']['high_confidence_damage'],20)
        self.assertEqual(hero['evidence']['recorded_damage'],0)
        self.assertEqual(dart['evidence']['recorded_damage'],625)
        self.assertEqual(dart['evidence']['high_confidence_damage'],0)
        self.assertEqual(dart['contribution_events'][0]['confidence'],'recorded')

    def test_rune_process_merges_records_but_does_not_invent_success(self):
        events=[event(t,'能量机关',category='rune_type=7.0',note=f'arm_cnt={t}.0,avg_round=2.5') for t in (1,2,3)]
        moments=build_moments([],events)
        self.assertEqual(len(moments),1)
        self.assertEqual(moments[0]['record_count'],3)
        self.assertEqual(moments[0]['end'],3)
        self.assertIn('激活臂数最高3',moments[0]['detail'])
        self.assertIn('末次平均环数2.5',moments[0]['detail'])
        self.assertIn('未关联',moments[0]['detail'])
        self.assertNotIn('大能量',moments[0]['title'])
        self.assertEqual(moments[0]['raw_records'][0]['category'],'rune_type=7.0')

    def test_rune_arm_reset_gap_camp_and_type_split_sessions(self):
        events=[event(1,'能量机关',category='rune_type=0',note='arm_cnt=3'),
                event(2,'能量机关',category='rune_type=0',note='arm_cnt=1'),
                event(3,'能量机关','蓝',category='rune_type=0',note='arm_cnt=2'),
                event(4,'能量机关',category='rune_type=1',note='arm_cnt=2'),
                event(20,'能量机关',category='rune_type=0',note='arm_cnt=2')]
        self.assertEqual(len(build_moments([],events)),5)

    def test_confirmed_buff_is_distinct_from_arm_hits(self):
        events=[event(1,'能量机关',note='arm_cnt=1'),event(2,'增益',category='小能量机关增益')]
        moments=build_moments([],events)
        self.assertEqual(len(moments),2)
        self.assertEqual(moments[1]['kind'],'机关增益')
        self.assertIn('记录小能量机关增益',moments[0]['detail'])

    def test_building_turning_point_and_missing_frame_do_not_conflate(self):
        t=_track(robot_type='基地',health=[100,0,100])
        moments=build_moments([t],[])
        self.assertEqual(moments[0]['kind'],'建筑被摧毁')
        self.assertEqual(moments[1]['kind'],'复活')
        t.observed_mask[1]=False
        self.assertEqual(build_moments([t],[]),[])
