import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split
from src.ingestion.loader import load_train_data, load_test_data
from src.preprocessing.name_normalizer import normalize_name
from src.preprocessing.address_normalizer import normalize_address
from src.blocking.blocker import generate_candidates
from src.features.pair_features import generate_pair_features
from src.models.labeling import create_labels
from src.models.matcher import EntityMatcher
from src.models.threshold import tune_threshold, apply_threshold
from src.output.writer import format_predictions_for_submission
from src import config

def preprocess_datasets(datasets):
    for name, df in datasets.items():
        if not df.empty and name != 'ground_truth':
            df['norm_name'] = df['business_name'].apply(normalize_name)
            df['norm_address'] = df['business_address'].apply(normalize_address)
    return datasets

def run_end_to_end():
    print("--- 1. LOADING DATA ---")
    try:
        train_data = load_train_data()
        test_data = load_test_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("--- 2. PREPROCESSING ---")
    train_data = preprocess_datasets(train_data)
    test_data = preprocess_datasets(test_data)

    print("--- 3. SPLIT FOR VALIDATION ---")
    s1_full = train_data['source1']
    if not s1_full.empty:
        s1_train, s1_val = train_test_split(s1_full, test_size=0.2, random_state=42)
    else:
        s1_train, s1_val = s1_full, s1_full

    s2_tr, s3_tr = train_data['source2'], train_data['source3']
    s23_combined = pd.concat([s2_tr, s3_tr], ignore_index=True)

    print("--- 4. TRAINING PIPELINE ---")
    # Candidates for Train split
    cands_s2_tr = generate_candidates(s1_train, s2_tr, 'source2')
    cands_s3_tr = generate_candidates(s1_train, s3_tr, 'source3')
    train_cands = pd.concat([cands_s2_tr, cands_s3_tr], ignore_index=True)
    
    # Features & Labels
    train_features = generate_pair_features(train_cands, s1_train, s23_combined)
    train_labels = create_labels(train_features, train_data['ground_truth'])
    
    # Train Model
    matcher = EntityMatcher()
    matcher.fit(train_labels)
    
    print("--- 5. VALIDATION PIPELINE (THRESHOLD TUNING) ---")
    # Candidates for Val split
    cands_s2_val = generate_candidates(s1_val, s2_tr, 'source2')
    cands_s3_val = generate_candidates(s1_val, s3_tr, 'source3')
    val_cands = pd.concat([cands_s2_val, cands_s3_val], ignore_index=True)
    
    val_features = generate_pair_features(val_cands, s1_val, s23_combined)
    val_labels = create_labels(val_features, train_data['ground_truth'])
    
    if not val_labels.empty:
        val_labels['match_prob'] = matcher.predict_proba(val_labels)
        best_thresh, best_f05 = tune_threshold(val_labels, 'match_prob')
        print(f"Optimal Decision Threshold learned: {best_thresh:.3f} with Validation F0.5 = {best_f05:.3f}")
    else:
        best_thresh = 0.5
        print("Validation set empty, defaulting threshold to 0.5")

    print("--- 6. TEST INFERENCE PIPELINE ---")
    s1_te, s2_te, s3_te = test_data['source1'], test_data['source2'], test_data['source3']
    
    # Test Candidate Generation
    cands_s2_te = generate_candidates(s1_te, s2_te, 'source2')
    cands_s3_te = generate_candidates(s1_te, s3_te, 'source3')
    all_te_cands = pd.concat([cands_s2_te, cands_s3_te], ignore_index=True)
    
    # Save candidate_pairs.tsv as per requirement
    all_te_cands.to_csv(config.CANDIDATE_PAIRS, sep='\t', index=False)
    
    # Test Features
    test_features = generate_pair_features(all_te_cands, s1_te, pd.concat([s2_te, s3_te], ignore_index=True))
    
    # Test Predictions
    if not test_features.empty:
        test_features['match_prob'] = matcher.predict_proba(test_features)
        final_test_matches = apply_threshold(test_features, 'match_prob', best_thresh)
    else:
        final_test_matches = pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id'])
        
    print("--- 7. OUTPUT GENERATION ---")
    s1_test_ids = s1_te['entity_id'].tolist() if not s1_te.empty else []
    submission_df = format_predictions_for_submission(s1_test_ids, final_test_matches)
    
    submission_df.to_csv(config.MATCHING_RESULTS, sep='\t', index=False)
    print(f"Pipeline complete! Outputs written to:\n - {config.CANDIDATE_PAIRS}\n - {config.MATCHING_RESULTS}")

if __name__ == "__main__":
    run_end_to_end()
