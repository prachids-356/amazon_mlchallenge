import pandas as pd
from .exact_blocking import exact_match_blocking
from .smart_blocking import smart_multi_key_blocking

def generate_candidates(df_source1: pd.DataFrame, df_candidates: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Union of exact match and smart multi-key blocking to maximize candidate recall safely.
    """
    print(f"Running exact match blocking for {source_name}...")
    exact_pairs = exact_match_blocking(df_source1, df_candidates, source_name)
    exact_pairs['blocking_rule'] = 'exact_match'
    
    print(f"Running smart multi-key blocking for {source_name}...")
    smart_pairs = smart_multi_key_blocking(df_source1, df_candidates, source_name)
    
    # Union all strategies
    all_pairs = pd.concat([exact_pairs, smart_pairs], ignore_index=True)
    
    # Deduplicate candidate pairs
    dedup_pairs = all_pairs.drop_duplicates(subset=['source1_entity_id', 'candidate_entity_id'])
    
    print(f"Yielded {len(dedup_pairs)} unique candidate pairs for {source_name} (down from {len(all_pairs)} total matches).")
    
    return dedup_pairs
