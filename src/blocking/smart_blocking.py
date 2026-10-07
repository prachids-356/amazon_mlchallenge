import pandas as pd

def smart_multi_key_blocking(df_source1: pd.DataFrame, df_candidates: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Blocks on Country + First 4 letters of Name + First word of Address.
    This safely restricts combinations to avoid memory explosion, while allowing
    typos in the rest of the name or address to be matched.
    """
    if df_source1.empty or df_candidates.empty:
        return pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id', 'candidate_source'])

    print(f"Generating smart block keys for {source_name}...")
    
    def generate_key(name, address, country):
        name_part = str(name)[:4].strip() if pd.notna(name) else ""
        addr_tokens = str(address).split() if pd.notna(address) else []
        addr_part = addr_tokens[0] if addr_tokens else ""
        return f"{country}_{name_part}_{addr_part}"

    s1_df = df_source1[['entity_id', 'norm_name', 'norm_address', 'country']].copy()
    cand_df = df_candidates[['entity_id', 'norm_name', 'norm_address', 'country']].copy()
    
    s1_df['block_key'] = s1_df.apply(lambda r: generate_key(r['norm_name'], r['norm_address'], r['country']), axis=1)
    cand_df['block_key'] = cand_df.apply(lambda r: generate_key(r['norm_name'], r['norm_address'], r['country']), axis=1)
    
    # Filter out empty keys to prevent massive collisions on missing data
    s1_df = s1_df[s1_df['block_key'] != "__"]
    cand_df = cand_df[cand_df['block_key'] != "__"]

    merged = pd.merge(
        s1_df[['entity_id', 'block_key']],
        cand_df[['entity_id', 'block_key']],
        on='block_key',
        suffixes=('_s1', '_cand')
    )
    
    pairs = merged[['entity_id_s1', 'entity_id_cand']].copy()
    pairs.rename(columns={'entity_id_s1': 'source1_entity_id', 'entity_id_cand': 'candidate_entity_id'}, inplace=True)
    pairs['candidate_source'] = source_name
    pairs['blocking_rule'] = 'smart_multi_key'
    
    return pairs
