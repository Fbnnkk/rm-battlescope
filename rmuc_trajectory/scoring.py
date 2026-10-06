"""Default role contribution scoring v3; legacy rules remain explicitly callable."""
from .contributions import Evaluation, LABELS as COMPONENT_LABELS
from .legacy_scoring import STARTING_SCORE, MIN_SCORE, MAX_SCORE, TYPE_SCALING
from .legacy_scoring import compute_scores as compute_legacy_scores
from .legacy_scoring import compute_timeseries_scores as compute_legacy_timeseries_scores


def compute_scores(match, tracks, events, attacks, buff_intervals, *, min_confidence="low"):
    return Evaluation(match, tracks, events, attacks, buff_intervals, min_confidence).scores()


def compute_timeseries_scores(match, tracks, events, attacks, buff_intervals, *, min_confidence="low"):
    return Evaluation(match, tracks, events, attacks, buff_intervals, min_confidence).series()


def compute_score_report(match, tracks, events, attacks, buff_intervals, *, min_confidence="low"):
    evaluation = Evaluation(match, tracks, events, attacks, buff_intervals, min_confidence)
    scores, summary = evaluation.scores()
    previous, _ = compute_legacy_scores(match, tracks, events, attacks, buff_intervals, min_confidence="medium" if min_confidence == "low" else min_confidence)
    lookup = {(r["camp"], r["robot_id"]): r["total_score"] for r in previous}
    for row in scores:
        old = lookup.get((row["camp"], row["robot_id"]))
        row["legacy_score"] = old if row["rating_eligible"] else None
        row["score_change"] = round(row["total_score"]-old, 2) if old is not None and row["rating_eligible"] else None
    return scores, summary, evaluation.series()


def explain_score(row):
    return sorted([{"key": k, "label": COMPONENT_LABELS.get(k, k), **v}
                   for k, v in row.get("components", {}).items()],
                  key=lambda c: abs(c["score"]), reverse=True)
