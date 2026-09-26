import pandas as pd

def create_labels(features_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels candidate pairs as 1 (match) or 0 (hard negative).
    ground_truth_df expects columns: 'source1_entity_id', 'matched_entity_ids' 
    (where matched_entity_ids is a space-separated string of S2/S3 IDs, or empty).
    """
    if features_df.empty or ground_truth_df.empty:
        features_df['label'] = 0
        return features_df

    # Explode ground truth so we have a simple S1 -> S2/S3 mapping
    # Assuming matched_entity_ids is space-separated or comma-separated
    gt = ground_truth_df.copy()
    
    # Handle potentially missing or different separator, defaulting to comma
    gt['matched_list'] = gt['matched_entity_ids'].apply(
        lambda x: str(x).split(',') if pd.notna(x) else []
    )
    
    gt_exploded = gt.explode('matched_list')
    gt_exploded = gt_exploded[gt_exploded['matched_list'] != ""]
    gt_exploded = gt_exploded.rename(columns={
        'source1_entity_id': 's1_id', 
        'matched_list': 'cand_id'
    })
    
    # Create a set of true pairs for O(1) lookup
    true_pairs = set(zip(gt_exploded['s1_id'], gt_exploded['cand_id']))
    
    # Label the features
    features_df['label'] = features_df.apply(
        lambda r: 1 if (r['source1_entity_id'], r['candidate_entity_id']) in true_pairs else 0, 
        axis=1
    )
    
    return features_df
