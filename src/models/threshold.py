import pandas as pd
from typing import Tuple
from src.evaluation.metrics import calculate_f05

def tune_threshold(df_val: pd.DataFrame, prob_col: str) -> Tuple[float, float]:
    """
    Finds the threshold that maximizes macro-averaged F0.5 (calculated per S1 entity).
    df_val must contain 'source1_entity_id', 'candidate_entity_id', 'label', and prob_col.
    """
    import numpy as np
    
    if df_val.empty:
        return 0.5, 0.0
        
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.5
    best_f05 = 0.0
    
    s1_groups = df_val.groupby('source1_entity_id')
    
    for th in thresholds:
        f05_scores = []
        for _, group in s1_groups:
            preds = (group[prob_col] >= th).astype(int)
            labels = group['label']
            
            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            
            # Singleton handling: if true matches == 0 and we predicted 0 matches, score is 1.0
            if labels.sum() == 0:
                if preds.sum() == 0:
                    f05_scores.append(1.0)
                else:
                    f05_scores.append(0.0)
                continue
                
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            f05 = calculate_f05(precision, recall)
            f05_scores.append(f05)
            
        macro_f05 = np.mean(f05_scores) if f05_scores else 0.0
        
        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_thresh = th
            
    return best_thresh, best_f05
    
def apply_threshold(df: pd.DataFrame, prob_col: str, threshold: float) -> pd.DataFrame:
    """
    Applies the decision threshold to yield final predictions.
    Handles duplicates by taking the max probability per S1->Cand pair.
    """
    df = df.copy()
    df['is_match'] = (df[prob_col] >= threshold).astype(int)
    
    # Filter to positive matches
    matches = df[df['is_match'] == 1].copy()
    
    # Remove duplicates (e.g., if a candidate was yielded by multiple blocking rules)
    matches = matches.sort_values(prob_col, ascending=False).drop_duplicates(
        subset=['source1_entity_id', 'candidate_entity_id']
    )
    
    return matches
