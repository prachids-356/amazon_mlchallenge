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
    """Character bigram Jaccard similarity."""
    if pd.isna(s1) or pd.isna(s2):
        return 0.0

    str1, str2 = str(s1), str(s2)

    bg1 = {str1[i:i+2] for i in range(len(str1) - 1)}
    bg2 = {str2[i:i+2] for i in range(len(str2) - 1)}

    if not bg1 and not bg2:
        return 0.0

    intersection = len(bg1 & bg2)
    union_size = len(bg1) + len(bg2) - intersection

    return intersection / union_size



def _compute_series_similarity(left_values, right_values, metric_fn, exact_flags=None, chunk_size=200_000):
    """Compute similarity values in chunks while keeping exact matches at 1.0."""
    results = [1.0] * len(left_values)

    if exact_flags is not None:
        exact_mask = exact_flags.to_numpy(dtype=bool)
        non_exact_idx = [i for i, is_exact in enumerate(exact_mask) if not is_exact]
    else:
        non_exact_idx = list(range(len(left_values)))

    for start in range(0, len(non_exact_idx), chunk_size):
        batch = non_exact_idx[start:start + chunk_size]
        for pos in batch:
            results[pos] = metric_fn(left_values.iloc[pos], right_values.iloc[pos])

    return pd.Series(results, index=left_values.index)


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
    s1_name = features_df['s1_name'].fillna('').astype(str)
    cand_name = features_df['cand_name'].fillna('').astype(str)
    name_exact = (s1_name == cand_name)
    features_df['name_exact'] = name_exact.astype(float)
    features_df['name_edit_sim'] = _compute_series_similarity(
        s1_name, cand_name, fast_char_bigram_jaccard, exact_flags=name_exact, chunk_size=200_000
    )
    features_df['name_jaccard'] = _compute_series_similarity(
        s1_name, cand_name, fast_jaccard, exact_flags=name_exact, chunk_size=200_000
    )

    print("Computing Address Features (Fast Heuristics)...")
    s1_address = features_df['s1_address'].fillna('').astype(str)
    cand_address = features_df['cand_address'].fillna('').astype(str)
    address_exact = (s1_address == cand_address)
    features_df['address_exact'] = address_exact.astype(float)
    features_df['address_edit_sim'] = _compute_series_similarity(
        s1_address, cand_address, fast_char_bigram_jaccard, exact_flags=address_exact, chunk_size=200_000
    )
    features_df['address_jaccard'] = _compute_series_similarity(
        s1_address, cand_address, fast_jaccard, exact_flags=address_exact, chunk_size=200_000
    )

    print("Computing Country Features...")
    features_df['country_match'] = (features_df['s1_country'] == features_df['cand_country']).astype(int)

    return features_df
