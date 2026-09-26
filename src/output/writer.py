import pandas as pd
from typing import List

def format_predictions_for_submission(
    source1_test_ids: List[str], 
    final_matches_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Ensures every Source 1 test entity appears exactly once.
    Empty list if no matches.
    """
    if final_matches_df.empty:
        # All singletons
        return pd.DataFrame({
            'source1_entity_id': source1_test_ids,
            'matched_entity_ids': [""] * len(source1_test_ids)
        })
        
    # Group matches by source1_entity_id
    grouped = final_matches_df.groupby('source1_entity_id')['candidate_entity_id'].apply(
        lambda x: ",".join(list(set(x)))
    ).reset_index()
    grouped.rename(columns={'candidate_entity_id': 'matched_entity_ids'}, inplace=True)
    
    # Merge with ALL source1 test ids to handle singletons
    s1_df = pd.DataFrame({'source1_entity_id': source1_test_ids})
    submission = pd.merge(s1_df, grouped, on='source1_entity_id', how='left')
    
    # Fill missing matches with empty string
    submission['matched_entity_ids'] = submission['matched_entity_ids'].fillna("")
    
    return submission
