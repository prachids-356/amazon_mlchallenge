import pandas as pd

def calculate_f05(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return (1 + 0.5**2) * (precision * recall) / ((0.5**2 * precision) + recall)

def evaluate_predictions(predictions: pd.DataFrame, ground_truth: pd.DataFrame):
    """
    Evaluates predictions at the Source 1 entity level.
    predictions: DataFrame with 'source1_entity_id' and 'predicted_match_ids' (list or space-separated string)
    ground_truth: DataFrame with 'source1_entity_id' and 'actual_match_ids' (list or space-separated string)
    """
    # implementation will calculate macro-average F0.5
    pass
