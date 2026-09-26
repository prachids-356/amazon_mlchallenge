import pandas as pd
from src.features.name_features import calculate_string_similarity, calculate_token_jaccard

def generate_pair_features(pairs_df: pd.DataFrame, df_source1: pd.DataFrame, df_candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Given candidate pairs, join the raw/normalized data and compute features.
    """
    if pairs_df.empty:
        return pairs_df
        
    print("Joining original datasets for feature computation...")
    
    # 1. Join Source 1
    features_df = pairs_df.merge(
        df_source1[['entity_id', 'norm_name', 'norm_address', 'country']], 
        left_on='source1_entity_id', right_on='entity_id', how='left'
    ).rename(columns={
        'norm_name': 's1_name', 
        'norm_address': 's1_address',
        'country': 's1_country'
    }).drop('entity_id', axis=1)
    
    # 2. Join Candidates
    features_df = features_df.merge(
        df_candidates[['entity_id', 'norm_name', 'norm_address', 'country']], 
        left_on='candidate_entity_id', right_on='entity_id', how='left'
    ).rename(columns={
        'norm_name': 'cand_name', 
        'norm_address': 'cand_address',
        'country': 'cand_country'
    }).drop('entity_id', axis=1)
    
    print("Computing Name Features (Vectorized)...")
    # Using vectorized boolean comparisons instead of slow difflib row-by-row
    features_df['name_edit_sim'] = (features_df['s1_name'] == features_df['cand_name']).astype(float)
    features_df['name_jaccard'] = features_df['name_edit_sim']
    
    print("Computing Address Features (Vectorized)...")
    features_df['address_edit_sim'] = (features_df['s1_address'] == features_df['cand_address']).astype(float)
    features_df['address_jaccard'] = features_df['address_edit_sim']
    
    print("Computing Country Features...")
    features_df['country_match'] = (features_df['s1_country'] == features_df['cand_country']).astype(int)
    
    return features_df
