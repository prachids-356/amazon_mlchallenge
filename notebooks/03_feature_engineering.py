import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.ingestion.loader import load_train_data
from src.preprocessing.name_normalizer import normalize_name
from src.preprocessing.address_normalizer import normalize_address
from src.blocking.blocker import generate_candidates
from src.features.pair_features import generate_pair_features

def run_feature_engineering():
    print("Starting Advanced Candidate Generation and Feature Engineering...")
    try:
        data = load_train_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Dataset not found. Skipping execution.")
        return
        
    s1 = data["source1"].copy()
    s2 = data["source2"].copy()
    s3 = data["source3"].copy()

    # Preprocessing
    for df in [s1, s2, s3]:
        if not df.empty:
            df['norm_name'] = df['business_name'].apply(normalize_name)
            df['norm_address'] = df['business_address'].apply(normalize_address)

    # Candidate Generation (Phase 2)
    print("--- Phase 2: Candidate Generation ---")
    candidates_s2 = generate_candidates(s1, s2, "source2")
    candidates_s3 = generate_candidates(s1, s3, "source3")
    
    all_candidates = pd.concat([candidates_s2, candidates_s3], ignore_index=True)
    
    # Feature Engineering (Phase 3)
    print("--- Phase 3: Feature Engineering ---")
    combined_candidates_data = pd.concat([s2, s3], ignore_index=True)
    features = generate_pair_features(all_candidates, s1, combined_candidates_data)
    
    # Save the prepared training features
    from src import config
    features_path = config.OUTPUT_DIR / "train_features.tsv"
    features.to_csv(features_path, sep='\t', index=False)
    print(f"Saved feature-engineered dataset to {features_path}")

if __name__ == "__main__":
    run_feature_engineering()
