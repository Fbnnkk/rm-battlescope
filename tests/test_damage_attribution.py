"""Damage conservation and uncertainty regression tests."""
from dataclasses import replace
import unittest

from test_scoring import _track
from rmuc_trajectory.combat import infer_attacks
from rmuc_trajectory.pipeline import MatchEvent
from rmuc_trajectory.scoring import compute_score_report, compute_scores


def event(time, kind, rid, camp, value=None):
    return MatchEvent(time, kind, rid, '步兵3', camp, '测试', None, None, '17mm', value, None)


class DamageAttributionTests(unittest.TestCase):
    def scene(self, heading=0, second=False):
        attacker = _track(robot_type='步兵3', x_vals=[1]*3)
        attacker.heading_deg[:] = heading
        victim = _track(camp='蓝', robot_id=103, robot_type='步兵3', x_vals=[5]*3, health=[100, 0, 0])
        tracks = [attacker, victim]
        events = [event(0, '发弹', 1, '红'), event(1, '发弹', 1, '红')]
        if second:
            tracks.append(_track(robot_id=3, robot_type='步兵3', x_vals=[1]*3))
            events.append(event(1, '发弹', 3, '红'))
        events.extend([event(1, '受击', 103, '蓝', -20)] * 5)
        return tracks, events

    def test_repeated_native_hits_are_bullets_not_duplicates(self):
        tracks, events = self.scene()
        attacks, summary = infer_attacks(events, tracks)
        self.assertEqual(summary['projectile_hit_count'], 5)
        self.assertEqual(summary['projectile_hit_window_count'], 1)
        self.assertEqual(attacks[0].damage, 100)
        self.assertEqual(attacks[0].hit_count, 5)

    def test_adjacent_bursts_from_same_shooter_do_not_cause_ambiguity(self):
        tracks, events = self.scene()
        attacks, _ = infer_attacks(events, tracks)
        self.assertFalse(attacks[0].ambiguous)
        self.assertEqual(attacks[0].confidence, 'high')
        self.assertEqual(len(attacks[0].damage_candidates), 1)

    def test_two_attackers_split_one_hit_total(self):
        tracks, events = self.scene(second=True)
        attacks, _ = infer_attacks(events, tracks)
        self.assertTrue(attacks[0].ambiguous)
        scores, summary, series = compute_score_report({}, tracks, events, attacks, [])
        amounts = [s['components']['damage_to_robots']['value'] for s in scores if s['camp']=='红']
        self.assertEqual(amounts, [50, 50])
        self.assertEqual(summary['low_confidence_projectile_damage'], 100)
        self.assertEqual(summary['excluded_attack_count'], 0)
        for s in scores:
            self.assertEqual(s['components']['kills']['value'], 0)
            self.assertEqual(s['components']['assists']['value'], 0)
            self.assertEqual(s['components']['pressure']['value'], 0)
            t=next(t for t in series[s['camp']] if t['robot_id']==s['robot_id'])
            self.assertEqual(t['score'][-1], s['total_score'])
            self.assertEqual(t['damage'][-1], s['components']['damage_to_robots']['value'])

    def test_medium_policy_keeps_uncertain_damage_unassigned(self):
        tracks, events = self.scene(second=True)
        attacks, _ = infer_attacks(events, tracks)
        scores, summary = compute_scores({}, tracks, events, attacks, [], min_confidence='medium')
        self.assertEqual(summary['attributed_projectile_damage'], 0)
        self.assertEqual(summary['unattributed_projectile_damage'], 100)
        self.assertTrue(all(s['components']['damage_to_robots']['value']==0 for s in scores))

    def test_extremely_bad_heading_is_not_forced_into_personal_damage(self):
        tracks, events = self.scene(heading=180)
        attacks, _ = infer_attacks(events, tracks)
        self.assertEqual(attacks[0].damage_candidates, ())
        _, summary = compute_scores({}, tracks, events, attacks, [])
        self.assertEqual(summary['unattributed_projectile_damage'], 100)

    def test_unresolved_and_missing_candidates_retain_unknown_share(self):
        tracks, events = self.scene(second=True)
        attacks, _ = infer_attacks(events, tracks)
        # Removing the second shooter's track must not transfer its share.
        _, summary = compute_scores({}, tracks[:2], events, attacks, [])
        self.assertEqual(summary['low_confidence_projectile_damage'], 50)
        self.assertEqual(summary['unattributed_projectile_damage'], 50)
        self.assertEqual(sum(t['observed_damage'] for t in summary['damage_targets']), 100)

    def test_no_shot_evidence_preserves_observed_damage(self):
        tracks, events = self.scene()
        events=[e for e in events if e.event_type=='受击']
        attacks, summary = infer_attacks(events, tracks)
        self.assertEqual(attacks, [])
        self.assertEqual(summary['unresolved'], 1)
        _, summary = compute_scores({}, tracks, events, attacks, [])
        self.assertEqual(summary['observed_projectile_damage'], 100)
        self.assertEqual(summary['unattributed_projectile_damage'], 100)

    def test_low_weight_discounts_score_not_damage_hp(self):
        tracks, events = self.scene(heading=40)
        attacks, _ = infer_attacks(events, tracks)
        self.assertEqual(attacks[0].confidence, 'low')
        low, _ = compute_scores({}, tracks, events, attacks, [])
        high, _ = compute_scores({}, tracks, events, [replace(attacks[0], confidence='high')], [])
        low_row=next(s for s in low if s['camp']=='红')
        high_row=next(s for s in high if s['camp']=='红')
        self.assertEqual(low_row['components']['damage_to_robots']['value'], 100)
        self.assertLess(low_row['components']['damage_to_robots']['score'], high_row['components']['damage_to_robots']['score'])

    def test_nonfinite_hit_value_does_not_poison_damage_total(self):
        tracks, events = self.scene()
        events.append(event(1, '受击', 103, '蓝', float('nan')))
        attacks, _ = infer_attacks(events, tracks)
        self.assertEqual(attacks[0].damage, 100)

    def test_uncertain_fatal_hit_does_not_promote_previous_reliable_shooter(self):
        tracks, events = self.scene(second=True)
        attacks, _ = infer_attacks(events, tracks)
        reliable = replace(attacks[0], hit_time=0, shot_time=0, confidence='high', ambiguous=False, damage=20)
        scores, _ = compute_scores({}, tracks, events, [reliable, *attacks], [])
        self.assertTrue(all(s['components']['kills']['value']==0 for s in scores))

    def test_direct_dart_damage_stays_separate_from_projectile_reconciliation(self):
        tracks, events = self.scene()
        events.append(MatchEvent(1, '飞镖命中', None, '飞镖', '红', '测试', 110, '基地', None, 625, None))
        attacks, _ = infer_attacks(events, tracks)
        scores, summary = compute_scores({}, tracks, events, attacks, [])
        self.assertEqual(summary['attributed_projectile_damage'], 100)
        dart=next(s for s in scores if s['robot_type']=='飞镖')
        self.assertEqual(dart['evidence']['reliable_damage'], 625)
