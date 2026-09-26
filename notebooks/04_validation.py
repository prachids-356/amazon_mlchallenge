import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.ingestion.loader import load_tsv
from src import config
from src.models.labeling import create_labels
from src.models.matcher import EntityMatcher
from src.models.threshold import tune_threshold, apply_threshold

def run_validation():
    print("Starting Phase 4 & 5: Model Training and Decision Layer...")
    
    # 1. Load Features and Ground Truth
    try:
        features_df = pd.read_csv(config.OUTPUT_DIR / "train_features.tsv", sep='\t')
        ground_truth_df = load_tsv(config.TRAIN_GT)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure earlier steps were run and ground truth is in place.")
        return
        
    if features_df.empty:
        print("No features to train on. Exiting.")
        return

    # 2. Labeling
    print("Labeling dataset with Ground Truth...")
    labeled_df = create_labels(features_df, ground_truth_df)
    
    # Simple temporal/random split for validation could be done here.
    # For now, we'll train and tune on the same set (Phase 4).
    # In a true setup, we would split s1_entities into train/val.
    
    # 3. Train Matcher
    matcher = EntityMatcher()
    matcher.fit(labeled_df)
    
    # 4. Predict
    print("Predicting probabilities...")
    labeled_df['match_prob'] = matcher.predict_proba(labeled_df)
    
    # 5. Tune Decision Threshold
    print("Tuning decision threshold...")
    best_thresh, best_f05 = tune_threshold(labeled_df['label'], labeled_df['match_prob'])
    print(f"Best Threshold: {best_thresh:.3f} | Estimated Global F0.5: {best_f05:.3f}")
    
    # 6. Apply Decision Layer
    print("Applying decision layer and handling singletons...")
    final_matches = apply_threshold(labeled_df, 'match_prob', best_thresh)
    
    print(f"Yielded {len(final_matches)} positive matches out of {len(labeled_df)} candidates.")
    print("Validation pipeline complete.")

if __name__ == "__main__":
    run_validation()
