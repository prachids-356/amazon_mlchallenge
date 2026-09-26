import pandas as pd

def exact_match_blocking(df_source1: pd.DataFrame, df_candidates: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Blocks on exact normalized name and exact normalized address.
    Returns a dataframe of candidate pairs.
    """
    if df_source1.empty or df_candidates.empty:
        return pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id', 'candidate_source'])
        
    merged = pd.merge(
        df_source1[['entity_id', 'norm_name', 'norm_address']],
        df_candidates[['entity_id', 'norm_name', 'norm_address']],
        on=['norm_name', 'norm_address'],
        suffixes=('_s1', '_cand')
    )
    
    pairs = merged[['entity_id_s1', 'entity_id_cand']].copy()
    pairs.rename(columns={'entity_id_s1': 'source1_entity_id', 'entity_id_cand': 'candidate_entity_id'}, inplace=True)
    pairs['candidate_source'] = source_name
    return pairs
