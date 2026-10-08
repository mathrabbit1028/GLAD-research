"""Binary graph anomaly metrics. Label 1 and high scores mean anomaly."""
import math
from collections.abc import Sequence


def classification_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict[str, float]:
    """Compute ROC AUC, F1, and accuracy from anomaly-oriented scores."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("Scores and labels must be aligned and nonempty")
    if set(labels) != {0, 1} or not all(math.isfinite(float(score)) for score in scores):
        raise ValueError("Evaluation needs finite scores and both binary classes")
    positives = sum(labels)
    negatives = len(labels) - positives
    # Midranks handle equal scores exactly as ROC AUC's tie convention.
    ranked = sorted(zip(scores, labels), key=lambda pair: pair[0])
    positive_rank_sum = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][0] == ranked[start][0]:
            end += 1
        midrank = (start + 1 + end) / 2
        positive_rank_sum += midrank * sum(label for _, label in ranked[start:end])
        start = end
    auc = (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    predictions = [int(score >= threshold) for score in scores]
    tp = sum(label == prediction == 1 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    correct = sum(label == prediction for label, prediction in zip(labels, predictions))
    return {"auc": float(auc),
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
            "accuracy": float(correct / len(labels))}
