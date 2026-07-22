#!/usr/bin/env python3
"""Batch score all matches for calibration. Output to outputs/scoring_calibration.json."""
import sqlite3, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rmuc_trajectory.scoring import compute_scores
from rmuc_trajectory.pipeline import load_and_clean_tracks, load_match_events, open_readonly
from rmuc_trajectory.combat import infer_attacks
from rmuc_trajectory.buffs import build_buff_intervals

db_path = ROOT / "rmuc_2026_region_dataset" / "rmuc_2026_region_dataset.sqlite"

conn = open_readonly(db_path)
game_ids = [r[0] for r in conn.execute("SELECT DISTINCT game_id FROM matches ORDER BY game_id").fetchall()]
conn.close()
print(f"Total games: {len(game_ids)}", flush=True)

all_scores = defaultdict(list)
errors = []

for i, gid in enumerate(game_ids):
    try:
        match, tracks = load_and_clean_tracks(db_path, gid)
        _, objectives = load_and_clean_tracks(db_path, gid, robot_types=('基地', '前哨站'), include_static=True)
        events = load_match_events(db_path, gid)
        attacks, _ = infer_attacks(events, tracks + objectives)
        match_end = max(float(t.times[-1]) for t in tracks if len(t.times))
        buff_intervals, _ = build_buff_intervals(events, tracks + objectives, match_end)
        scores, _ = compute_scores(match, tracks, events, attacks, buff_intervals)
        for s in scores:
            all_scores[s['robot_type']].append(s['total_score'])
    except Exception as e:
        errors.append(f"game {gid}: {e}")

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(game_ids)} games, {len(errors)} errors", flush=True)

print(f"Done. Errors: {len(errors)}", flush=True)
for e in errors[:5]:
    print(f"  {e}")

# Stats
print("\n=== Per-type score distribution (raw) ===")
stats = {}
for rtype in sorted(all_scores.keys()):
    scores = np.array(all_scores[rtype])
    p25 = float(np.percentile(scores, 25))
    p50 = float(np.percentile(scores, 50))
    p75 = float(np.percentile(scores, 75))
    mean = float(np.mean(scores))
    factor = 5.0 / p50 if p50 > 0 else 1.0
    stats[rtype] = {
        'count': len(scores),
        'p25': round(p25, 3), 'p50': round(p50, 3), 'p75': round(p75, 3),
        'mean': round(mean, 3),
        'factor': round(factor, 4),
        'p25_scaled': round(p25 * factor, 3),
        'p50_scaled': round(p50 * factor, 3),
        'p75_scaled': round(p75 * factor, 3),
        'min': round(float(scores.min()), 3),
        'max': round(float(scores.max()), 3),
    }
    print(f"  {rtype}: n={len(scores)} p25={p25:.2f} p50={p50:.2f} p75={p75:.2f} mean={mean:.2f} factor={factor:.3f}")
    print(f"    scaled: p25={p25*factor:.2f} p50={p50*factor:.2f} p75={p75*factor:.2f}")

out_path = ROOT / "outputs" / "scoring_calibration.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {out_path}")
