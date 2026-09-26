import pandas as pd

def name_token_blocking(df_source1: pd.DataFrame, df_candidates: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Blocks on the first token of the normalized business name to increase recall.
    """
    if df_source1.empty or df_candidates.empty:
        return pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id', 'candidate_source'])

    # Create blocking key: first token of the normalized name
    def get_first_token(name):
        tokens = str(name).split()
        return tokens[0] if tokens else ""

    # Work on copies to prevent SettingWithCopyWarning
    s1_df = df_source1[['entity_id', 'norm_name']].copy()
    cand_df = df_candidates[['entity_id', 'norm_name']].copy()
    
    s1_df['block_key'] = s1_df['norm_name'].apply(get_first_token)
    cand_df['block_key'] = cand_df['norm_name'].apply(get_first_token)
    
    # Filter out empty block keys
    s1_df = s1_df[s1_df['block_key'] != ""]
    cand_df = cand_df[cand_df['block_key'] != ""]

    merged = pd.merge(
        s1_df[['entity_id', 'block_key']],
        cand_df[['entity_id', 'block_key']],
        on=['block_key'],
        suffixes=('_s1', '_cand')
    )
    
    pairs = merged[['entity_id_s1', 'entity_id_cand']].copy()
    pairs.rename(columns={'entity_id_s1': 'source1_entity_id', 'entity_id_cand': 'candidate_entity_id'}, inplace=True)
    pairs['candidate_source'] = source_name
    pairs['blocking_rule'] = 'name_first_token'
    
    return pairs
