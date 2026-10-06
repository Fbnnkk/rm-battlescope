"""Regression tests for trustworthy scorecards, exports and job reuse."""

import unittest
import json
import sqlite3
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from test_scoring import _attack, _track, AERIAL, BASE, BLUE, ENGINEER, HERO, INF3, RED, SENTRY
import test_webapp
from rmuc_trajectory.buffs import BuffInterval
from rmuc_trajectory.pipeline import MatchEvent
from rmuc_trajectory.review import build_review
from rmuc_trajectory.scoring import compute_scores, compute_timeseries_scores
from rmuc_trajectory.webapp import ReplayApplication, create_server, validate_job_payload


class ScorecardTests(unittest.TestCase):
    def assert_series_matches(self, tracks, attacks=None, buffs=None, match=None, events=None):
        args = (match or {"胜方": RED}, tracks, events or [], attacks or [], buffs or [])
        scores, summary = compute_scores(*args)
        series = compute_timeseries_scores(*args)
        for s in scores:
            line = next(t for t in series[s["camp"]] if t["robot_id"] == s["robot_id"])
            self.assertEqual(line["score"][-1], s["total_score"])
            self.assertEqual(line["kills"][-1], s["components"]["kills"]["value"])
            self.assertAlmostEqual(5 + sum(c["score"] for c in s["explanation"]), s["raw_score"], places=3)
        return scores, series, summary

    def test_endpoint_consistency_for_every_role(self):
        for role in (HERO, INF3, SENTRY, ENGINEER, AERIAL):
            with self.subTest(role=role):
                t = _track(robot_type=role, health=[100, 0, 100, 100], x_vals=[15]*4)
                attack = _attack(attacker_type=role, hit_time=2.5, damage=25)
                buffs = [BuffInterval(RED, 1, role, "过中央高地", 0.2, 2.7, 2.5, "规则计时结束", "test")]
                self.assert_series_matches([t], [attack], buffs)

    def test_sparse_coins_and_shots_do_not_crash(self):
        t = _track(robot_type=HERO, times=list(range(6)))
        t.cumulative_42mm = np.array([0, np.nan, 2, 3, np.nan, 5.])
        t.total_coins[2] = np.nan
        t.remaining_coins[4] = np.nan
        self.assert_series_matches([t])

    def test_victory_bonus_is_not_known_before_end(self):
        t = _track(robot_type=HERO, times=list(range(5)))
        _, series, _ = self.assert_series_matches([t])
        self.assertEqual(series[RED][0]["score"][:-1], [None,5.0,5.0,5.0])
        self.assertEqual(series[RED][0]["score"][-1], 5.0)  # team outcome is observational in v3

    def test_partial_window_has_no_win_or_first_blood_bonus(self):
        t = _track(robot_type=INF3, times=[60,61,62])
        victim = _track(camp=BLUE,robot_id=2,robot_type=INF3,times=[60,61,62],health=[100,0,0])
        s, _, _ = self.assert_series_matches([t,victim], [_attack(attacker_type=INF3,hit_time=61)], match={"胜方":RED,"_is_partial":True})
        self.assertEqual(s[0]["components"]["win_bonus"]["score"],0)
        self.assertEqual(s[0]["components"]["first_blood"]["score"],0)

    def test_low_confidence_and_ambiguous_attacks_are_excluded(self):
        t = _track()
        attacks = [replace(_attack(damage=100),confidence="low"),replace(_attack(damage=100),ambiguous=True)]
        s, summary = compute_scores({"胜方":BLUE},[t],[],attacks,[])
        self.assertEqual(s[0]["total_score"],5)
        self.assertEqual(summary["excluded_attack_count"],2)
        self.assertEqual(s[0]["evidence"]["excluded_attacks"],2)

    def test_strict_confidence_policy(self):
        attack = replace(_attack(damage=100),confidence="medium")
        s, _ = compute_scores({"胜方":BLUE},[_track()],[],[attack],[],min_confidence="high")
        self.assertEqual(s[0]["components"]["damage_to_robots"]["value"],0)

    def test_team_coins_do_not_penalize_hero(self):
        t = _track(total_coins=[500]*3,remain_coins=[500,100,0])
        scores,_ = compute_scores({"胜方":BLUE},[t],[],[],[])
        self.assertEqual(scores[0]["hero"]["gold_spent"]["value"],500)
        self.assertEqual(scores[0]["hero"]["gold_spent"]["score"],0)

    def test_unknown_base_hp_does_not_receive_tie_multiplier(self):
        scores,_=compute_scores({"胜方":BLUE},[_track()],[],[_attack(victim_type=BASE,damage=100)],[])
        changed,_=compute_scores({"胜方":RED},[_track()],[],[_attack(victim_type=BASE,damage=100)],[])
        self.assertEqual(scores[0]["components"]["damage_to_base"]["score"],changed[0]["components"]["damage_to_base"]["score"])

    def test_jam_is_clipped_at_end(self):
        t=_track(robot_id=6,robot_type=AERIAL,times=list(range(11)))
        event=MatchEvent(9,"雷达反制UAV",6,AERIAL,RED,"学校",None,None,None,None,None)
        scores,_,_=self.assert_series_matches([t],events=[event])
        self.assertEqual(scores[0]["aerial"]["counter_events_received"],1)
        radar=next(s for s in scores if s["robot_type"]=="雷达")
        self.assertEqual(radar['camp'],BLUE)

    def test_review_includes_destroyed_building(self):
        base=_track(robot_type=BASE,health=[100,0])
        review=build_review({},[base],[],[],{}, {})
        self.assertIn("被摧毁",review["moments"][0]["title"])

    def test_missing_health_is_not_death_or_alive(self):
        t=_track(robot_type=SENTRY,times=[0,1,2])
        t.health=np.array([100,np.nan,0])
        scores,_,_=self.assert_series_matches([t])
        self.assertEqual(scores[0]["components"]["deaths"]["value"],0)
        self.assertAlmostEqual(scores[0]["evidence"]["health_coverage"],.667)

    def test_empty_series(self):
        self.assertEqual(compute_timeseries_scores({},[],[],[],[]),{})


class InputTests(unittest.TestCase):
    def test_non_object_and_nonfinite_parameters(self):
        for payload in ([],None,{"game_id":True},{"game_id":100,"start":"nan"},
                        {"game_id":100,"end":"inf"},{"game_id":100,"max_speed":"NaN"},
                        {"game_id":100,"min_confidence":"unknown"}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_job_payload(payload)


class JobCacheTests(unittest.TestCase):
    def setUp(self):
        # Reuse the match fixture without inheriting/re-running its test cases.
        self.fixture=test_webapp.WebApplicationTests(methodName='test_job_payload_validation')
        self.fixture.setUp()
        self.root=Path(self.fixture.temp_dir.name)
        self.app=ReplayApplication(self.root,self.fixture.database,self.root/'replays')

    def tearDown(self):
        self.app.executor.shutdown(wait=True,cancel_futures=True)
        self.fixture.tearDown()

    def test_duplicate_jobs_and_restart_reuse_completed_replay(self):
        def generate(root, database, output, request, progress):
            output.mkdir(parents=True)
            (output/'trajectory.html').write_text('<html>test</html>',encoding='utf-8')
            progress(100,'complete')
            return {"track_count":1}
        request=validate_job_payload({"game_id":100})
        with patch('rmuc_trajectory.webapp.generate_replay',side_effect=generate) as generated:
            first=self.app.create_job(request)
            second=self.app.create_job(request)
            self.assertIs(first,second)
            self.app.executor.shutdown(wait=True)
            self.assertEqual(generated.call_count,1)
        self.assertEqual(first.status,'done')
        restored=ReplayApplication(self.root,self.fixture.database,self.root/'replays')
        try:
            self.assertEqual(restored.create_job(request).job_id,first.job_id)
            self.assertEqual(len(restored.recent_replays()),1)
        finally:
            restored.executor.shutdown(wait=True)

    def test_time_window_bounds(self):
        for payload in ({"game_id":100,"start":420},{"game_id":100,"end":421}):
            with self.assertRaises(ValueError):
                self.app.create_job(validate_job_payload(payload))

    def test_progress_100_is_not_published_done_without_url(self):
        from rmuc_trajectory.webapp import JobState
        state=JobState('test',validate_job_payload({"game_id":100}),self.root)
        state.update(100,'rendered')
        self.assertEqual(state.to_jsonable()['status'],'running')
        self.assertIsNone(state.to_jsonable()['replay_url'])

    def test_manifest_uses_key_from_job_creation(self):
        def generate(root, database, output, request, progress):
            output.mkdir()
            (output/'trajectory.html').write_text('test',encoding='utf-8')
            self.app.cache_key=lambda request:'changed-source'
            return {}
        with patch('rmuc_trajectory.webapp.generate_replay',side_effect=generate):
            state=self.app.create_job(validate_job_payload({"game_id":100}))
            self.app.executor.shutdown(wait=True)
        saved=json.loads((state.output_dir/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['cache_key'],state.cache_key)
        self.assertNotEqual(saved['cache_key'],'changed-source')


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_webapp.WebApplicationTests(methodName='test_job_payload_validation')
        self.fixture.setUp()
        self.root=Path(self.fixture.temp_dir.name)
        self.server,self.app=create_server(Path(__file__).resolve().parents[1],self.fixture.database,self.root/'replays',port=0)
        self.app_handler_print=patch('builtins.print')
        self.app_handler_print.start()
        self.worker=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.worker.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join()
        self.app.executor.shutdown(wait=True,cancel_futures=True)
        self.app_handler_print.stop()
        self.fixture.tearDown()

    def test_invalid_json_shape_returns_400_json(self):
        request=Request(self.url+'/api/jobs',data=b'[]',headers={'Content-Type':'application/json'})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request)
        self.assertEqual(caught.exception.code,400)
        self.assertIn('error',json.loads(caught.exception.read()))

    def test_sqlite_failure_returns_503_json(self):
        connection=sqlite3.connect(self.fixture.database)
        try:
            connection.execute('DROP TABLE matches')
            connection.commit()
        finally:
            connection.close()
        with self.assertRaises(HTTPError) as caught:
            urlopen(self.url+'/api/matches')
        self.assertEqual(caught.exception.code,503)
        self.assertIn('error',json.loads(caught.exception.read()))

    def test_replay_files_are_allowlisted(self):
        from rmuc_trajectory.webapp import JobState
        output=self.root/'replays'/'abcd12345678'
        output.mkdir()
        (output/'scores.csv').write_text('score\n5',encoding='utf-8')
        (output/'private.txt').write_text('hidden',encoding='utf-8')
        self.app.jobs['abcd12345678']=JobState('abcd12345678',validate_job_payload({'game_id':100}),output,status='done')
        with urlopen(self.url+'/replays/abcd12345678/scores.csv') as response:
            self.assertEqual(response.status,200)
        with self.assertRaises(HTTPError) as caught:
            urlopen(self.url+'/replays/abcd12345678/private.txt')
        self.assertEqual(caught.exception.code,404)

    def export_fixture(self):
        from rmuc_trajectory.webapp import JobState
        output=self.root/'replays'/'abcd12345678'
        output.mkdir()
        for name,content in {'quality_report.json':{'match':{'时长秒':420},'scoring':[]},
                             'review.json':{},'timeseries_scores.json':{}}.items():
            (output/name).write_text(json.dumps(content),encoding='utf-8')
        self.app.jobs['abcd12345678']=JobState('abcd12345678',validate_job_payload({'game_id':100}),output,status='done')
        return output

    def test_export_is_saved_in_replay_directory_and_can_be_read(self):
        output=self.export_fixture()
        body={'kind':'scorecard','notes':[{'time':33,'text':'复盘备注'}]}
        request=Request(self.url+'/api/replays/abcd12345678/exports',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request) as response:
            self.assertEqual(response.status,201)
            saved=json.loads(response.read())
        self.assertEqual(Path(saved['path']).parent,output/'exports')
        with urlopen(self.url+saved['url']) as response:
            content=json.loads(response.read())
        self.assertEqual(content['notes'],body['notes'])
        self.assertIn('scores',content)

    def test_invalid_export_notes_are_rejected(self):
        self.export_fixture()
        request=Request(self.url+'/api/replays/abcd12345678/exports',data=b'{"kind":"notes","notes":[{"time":999,"text":"test"}]}',headers={'Content-Type':'application/json'})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request)
        self.assertEqual(caught.exception.code,400)
