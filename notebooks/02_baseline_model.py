import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.ingestion.loader import load_train_data
from src.preprocessing.name_normalizer import normalize_name
from src.preprocessing.address_normalizer import normalize_address
from src.blocking.exact_blocking import exact_match_blocking

def run_baseline():
    print("Starting Baseline Pipeline...")
    try:
        data = load_train_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Dataset not found. Skipping execution.")
        return
        
    s1 = data["source1"].copy()
    s2 = data["source2"].copy()
    s3 = data["source3"].copy()

    # Apply Normalization
    print("Normalizing names and addresses...")
    for df in [s1, s2, s3]:
        if not df.empty:
            df['norm_name'] = df['business_name'].apply(normalize_name)
            df['norm_address'] = df['business_address'].apply(normalize_address)

    # Candidate Generation (Exact Match Baseline)
    print("Generating candidates using Exact Match Blocking...")
    pairs_s2 = exact_match_blocking(s1, s2, "source2")
    pairs_s3 = exact_match_blocking(s1, s3, "source3")
    
    all_candidates = pd.concat([pairs_s2, pairs_s3], ignore_index=True)
    print(f"Total candidates generated: {len(all_candidates)}")
    
    # Save candidates
    from src import config
    all_candidates.to_csv(config.CANDIDATE_PAIRS, sep='\t', index=False)
    print(f"Saved candidates to {config.CANDIDATE_PAIRS}")
    
if __name__ == "__main__":
    run_baseline()
