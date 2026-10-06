"""Portable configuration and static/backend export contracts."""
import csv
import json
import re
import tempfile
import unittest
from pathlib import Path

from test_scoring import _track
from test_review_evidence import event
from rmuc_trajectory.paths import get_output_root
from rmuc_trajectory.exports import write_scores_csv
from rmuc_trajectory.field import default_canvas
from rmuc_trajectory.render import render_interactive_html
from rmuc_trajectory.scoring import compute_score_report


class ReleaseExportTests(unittest.TestCase):
    def test_output_default_is_inside_supplied_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(get_output_root(root, {}), root.resolve()/'outputs')

    def test_local_config_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'.battlescope.local.json').write_text(json.dumps({'output_dir':'local-results'}),encoding='utf-8')
            self.assertEqual(get_output_root(root, {}),root.resolve()/'local-results')
            self.assertEqual(get_output_root(root, {'BATTLESCOPE_OUTPUT_DIR':'override'}),root.resolve()/'override')

    def test_explicit_absolute_output_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root/'external'/'reports'
            self.assertEqual(get_output_root(root, {'BATTLESCOPE_OUTPUT_DIR':str(target)}),target.resolve())

    def test_invalid_local_setting_fails_instead_of_falling_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config=root/'.battlescope.local.json'
            config.write_text('{"output_dir":null}',encoding='utf-8')
            with self.assertRaises(ValueError):
                get_output_root(root, {})

    def test_csv_unrated_values_remain_empty_and_schema_matches(self):
        scores,_,_=compute_score_report({},[_track()],[event(1,'飞镖闸门开')],[],[])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scores.csv'
            write_scores_csv(path,scores)
            with path.open(encoding='utf-8-sig',newline='') as stream:
                rows=list(csv.DictReader(stream))
            self.assertTrue(all(None not in row for row in rows))
            dart=next(row for row in rows if row['兵种']=='飞镖')
            self.assertEqual((dart['表现分'],dart['等级']),('',''))
            self.assertIn('暂不评级',dart['评级状态'])

    def test_static_and_backend_capabilities_are_explicit(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'trajectory.html'
            canvas=default_canvas(root)
            render_interactive_html(output,canvas,{},[_track()],return_url='../../index.html')
            html=output.read_text(encoding='utf-8')
            self.assertIn('"mode": "static"',html)
            self.assertIn('"return_url": "../../index.html"',html)
            self.assertIn('"csv_url": null',html)
            (output.parent/'scores.csv').write_text('header',encoding='utf-8')
            render_interactive_html(output,canvas,{},[_track()],backend={'export_url':'/api/replays/test/exports'})
            html=output.read_text(encoding='utf-8')
            self.assertIn('"mode": "backend"',html)
            self.assertIn('"csv_url": "scores.csv"',html)
            self.assertIn('"export_url": "/api/replays/test/exports"',html)


if __name__=='__main__':
    unittest.main()
