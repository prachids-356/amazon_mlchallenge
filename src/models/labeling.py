import re

import pandas as pd


def _parse_entity_ids(raw_value):
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return []
    if isinstance(raw_value, (list, tuple, set)):
        values = [str(item).strip() for item in raw_value]
    else:
        text = str(raw_value).strip()
        if not text:
            return []
        values = re.split(r'[\s,]+', text)
    return [value for value in values if value]


def create_labels(features_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels candidate pairs as 1 (match) or 0 (hard negative).
    ground_truth_df expects columns: 'source1_entity_id', 'matched_entity_ids' 
    (where matched_entity_ids is a space-separated string of S2/S3 IDs, or empty).
    """
    if features_df.empty or ground_truth_df.empty:
        features_df['label'] = 0
        return features_df

    # Explode ground truth so we have a simple S1 -> S2/S3 mapping.
    gt = ground_truth_df.copy()
    gt['matched_list'] = gt['matched_entity_ids'].apply(_parse_entity_ids)

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
