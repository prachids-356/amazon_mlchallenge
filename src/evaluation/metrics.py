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

def evaluate_predictions(predictions: pd.DataFrame, ground_truth: pd.DataFrame):
    """
    Evaluates predictions at the Source 1 entity level.
    predictions: DataFrame with 'source1_entity_id' and 'predicted_match_ids' (list or space-separated string)
    ground_truth: DataFrame with 'source1_entity_id' and 'actual_match_ids' (list or space-separated string)
    """
    # implementation will calculate macro-average F0.5
    pass
