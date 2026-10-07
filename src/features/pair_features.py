import pandas as pd

def fast_jaccard(s1, s2):
    """Fast token jaccard similarity"""
    if pd.isna(s1) or pd.isna(s2):
        return 0.0
    set1 = set(str(s1).split())
    set2 = set(str(s2).split())
    if not set1 and not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def fast_char_bigram_jaccard(s1, s2):
    """Fast character bigram overlap (approximates edit distance)"""
    if pd.isna(s1) or pd.isna(s2):
        return 0.0
    str1, str2 = str(s1), str(s2)
    bg1 = set(str1[i:i+2] for i in range(len(str1)-1))
    bg2 = set(str2[i:i+2] for i in range(len(str2)-1))
    if not bg1 and not bg2:
        return 0.0
    return len(bg1.intersection(bg2)) / len(bg1.union(bg2))

def generate_pair_features(pairs_df: pd.DataFrame, df_source1: pd.DataFrame, df_candidates: pd.DataFrame) -> pd.DataFrame:
    if pairs_df.empty:
        return pairs_df
        
    print("Joining original datasets for feature computation...")
    features_df = pairs_df.merge(
        df_source1[['entity_id', 'norm_name', 'norm_address', 'country']], 
        left_on='source1_entity_id', right_on='entity_id', how='left'
    ).rename(columns={'norm_name': 's1_name', 'norm_address': 's1_address', 'country': 's1_country'}).drop('entity_id', axis=1)
    
    features_df = features_df.merge(
        df_candidates[['entity_id', 'norm_name', 'norm_address', 'country']], 
        left_on='candidate_entity_id', right_on='entity_id', how='left'
    ).rename(columns={'norm_name': 'cand_name', 'norm_address': 'cand_address', 'country': 'cand_country'}).drop('entity_id', axis=1)
    
    print("Computing Name Features (Fast Heuristics)...")
    # Exact match feature
    features_df['name_exact'] = (features_df['s1_name'] == features_df['cand_name']).astype(float)
    # Fast bigram jaccard (proxy for edit distance)
    features_df['name_edit_sim'] = features_df.apply(lambda r: fast_char_bigram_jaccard(r['s1_name'], r['cand_name']) if r['name_exact'] == 0 else 1.0, axis=1)
    # Token jaccard
    features_df['name_jaccard'] = features_df.apply(lambda r: fast_jaccard(r['s1_name'], r['cand_name']) if r['name_exact'] == 0 else 1.0, axis=1)
    
    print("Computing Address Features (Fast Heuristics)...")
    features_df['address_exact'] = (features_df['s1_address'] == features_df['cand_address']).astype(float)
    features_df['address_edit_sim'] = features_df.apply(lambda r: fast_char_bigram_jaccard(r['s1_address'], r['cand_address']) if r['address_exact'] == 0 else 1.0, axis=1)
    features_df['address_jaccard'] = features_df.apply(lambda r: fast_jaccard(r['s1_address'], r['cand_address']) if r['address_exact'] == 0 else 1.0, axis=1)
    
    print("Computing Country Features...")
    features_df['country_match'] = (features_df['s1_country'] == features_df['cand_country']).astype(int)
    
    return features_df
