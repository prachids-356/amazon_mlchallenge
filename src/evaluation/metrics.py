import re

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_f05(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return (1 + 0.5**2) * (precision * recall) / ((0.5**2 * precision) + recall)


def calculate_binary_metrics(labels: pd.Series, predictions: pd.Series):
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision,
        'recall': recall,
        'f1': f1_score(labels, predictions, zero_division=0),
        'f05': calculate_f05(precision, recall),
    }


def _parse_match_ids(raw_value):
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return set()
    if isinstance(raw_value, (list, tuple, set)):
        ids = [str(item).strip() for item in raw_value]
    else:
        text = str(raw_value).strip()
        if not text:
            return set()
        ids = re.split(r'[\s,]+', text)
    return {item for item in ids if item}


def evaluate_predictions(predictions: pd.DataFrame, ground_truth: pd.DataFrame):
    """
    Evaluates predictions at the Source 1 entity level.
    predictions: DataFrame with 'source1_entity_id' and 'predicted_match_ids' (list or space-separated string)
    ground_truth: DataFrame with 'source1_entity_id' and 'actual_match_ids' (list or space-separated string)
    """
    if predictions.empty and ground_truth.empty:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'f05': 0.0, 'accuracy': 0.0}

    pred_map = predictions.set_index('source1_entity_id')['predicted_match_ids'].to_dict() if not predictions.empty else {}
    gt_map = ground_truth.set_index('source1_entity_id')['actual_match_ids'].to_dict() if not ground_truth.empty else {}

    all_entity_ids = sorted(set(pred_map) | set(gt_map))
    precisions = []
    recalls = []
    f05_scores = []
    accuracies = []

    for entity_id in all_entity_ids:
        pred_ids = _parse_match_ids(pred_map.get(entity_id, ''))
        true_ids = _parse_match_ids(gt_map.get(entity_id, ''))

        tp = len(pred_ids & true_ids)
        fp = len(pred_ids - true_ids)
        fn = len(true_ids - pred_ids)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f05_scores.append(calculate_f05(precision, recall))

        if pred_ids == true_ids:
            accuracies.append(1.0)
        else:
            accuracies.append(0.0)

    if not all_entity_ids:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'f05': 0.0, 'accuracy': 0.0}

    return {
        'precision': sum(precisions) / len(precisions),
        'recall': sum(recalls) / len(recalls),
        'f1': sum(2 * p * r / (p + r) if (p + r) > 0 else 0.0 for p, r in zip(precisions, recalls)) / len(precisions),
        'f05': sum(f05_scores) / len(f05_scores),
        'accuracy': sum(accuracies) / len(accuracies),
    }
