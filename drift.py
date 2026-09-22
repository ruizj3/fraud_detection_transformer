"""Lightweight data-drift monitoring.

Compares the distribution of newly loaded data against a reference snapshot
(stats captured the last time the model was trained) so training only
re-splits/retrains once the data has actually moved, instead of on every run.
"""
import json
import logging

logger = logging.getLogger(__name__)


def compute_reference_stats(df, cat_cols, cont_cols):
    stats = {"categorical": {}, "continuous": {}}
    for col in cat_cols:
        stats["categorical"][col] = df[col].astype(str).value_counts(normalize=True).to_dict()
    for col in cont_cols:
        series = df[col].astype(float)
        stats["continuous"][col] = {"mean": float(series.mean()), "std": float(series.std() or 1.0)}
    return stats


def save_reference_stats(stats, path):
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_reference_stats(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _categorical_drift(reference_props, df, col):
    # Total variation distance between the reference and current category proportions.
    current_props = df[col].astype(str).value_counts(normalize=True).to_dict()
    categories = set(reference_props) | set(current_props)
    return 0.5 * sum(abs(reference_props.get(c, 0.0) - current_props.get(c, 0.0)) for c in categories)


def _continuous_drift(reference_stat, df, col):
    # Standardized mean shift: how many reference std-devs the current mean has moved.
    current_mean = df[col].astype(float).mean()
    return abs(current_mean - reference_stat["mean"]) / (reference_stat["std"] or 1.0)


def compute_drift_score(reference_stats, df, cat_cols, cont_cols):
    """Returns (overall_score, per_feature_scores). Overall is the max across features."""
    scores = {}
    for col in cat_cols:
        if col in reference_stats["categorical"]:
            scores[col] = _categorical_drift(reference_stats["categorical"][col], df, col)
    for col in cont_cols:
        if col in reference_stats["continuous"]:
            scores[col] = _continuous_drift(reference_stats["continuous"][col], df, col)
    overall = max(scores.values()) if scores else 0.0
    return overall, scores


def has_drifted(reference_stats, df, cat_cols, cont_cols, threshold):
    """No reference snapshot yet (first run) always counts as drifted, to force an initial train."""
    if reference_stats is None:
        return True, float("inf"), {}
    overall, scores = compute_drift_score(reference_stats, df, cat_cols, cont_cols)
    return overall >= threshold, overall, scores
