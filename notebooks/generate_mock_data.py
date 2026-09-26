import pandas as pd
import os
from src.config import TRAIN_DIR, TEST_DIR

def create_mock_data():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # --- TRAIN DATA ---
    s1_train = pd.DataFrame([
        {'entity_id': 'S1-001', 'business_name': 'Amazon Inc.', 'business_address': '410 Terry Ave N, Seattle, WA', 'country': 'US'},
        {'entity_id': 'S1-002', 'business_name': 'Flipkart Pvt Ltd', 'business_address': 'Outer Ring Rd, Bengaluru', 'country': 'India'},
        {'entity_id': 'S1-003', 'business_name': 'Singleton Corp', 'business_address': '123 Lonely St', 'country': 'US'},
    ])
    
    s2_train = pd.DataFrame([
        {'entity_id': 'S2-001', 'business_name': 'Amazon Corporation', 'business_address': '410 Terry Avenue North Seattle', 'country': 'US'},
        {'entity_id': 'S2-002', 'business_name': 'Flipkart Private Limited', 'business_address': 'Outer Ring Road, Bangalore', 'country': 'India'},
        {'entity_id': 'S2-003', 'business_name': 'Random Business LLC', 'business_address': '456 Nowhere Ave', 'country': 'US'},
    ])

    s3_train = pd.DataFrame([
        {'entity_id': 'S3-001', 'business_name': 'Amazon', 'business_address': 'Terry Ave N, Seattle WA', 'country': 'US'},
        {'entity_id': 'S3-002', 'business_name': 'Flipkart', 'business_address': 'ORR, Bengaluru, KA', 'country': 'India'},
    ])

    gt_train = pd.DataFrame([
        {'source1_entity_id': 'S1-001', 'matched_entity_ids': 'S2-001,S3-001'},
        {'source1_entity_id': 'S1-002', 'matched_entity_ids': 'S2-002,S3-002'},
        {'source1_entity_id': 'S1-003', 'matched_entity_ids': ''}, # Singleton
    ])

    # Save Train Data
    s1_train.to_csv(TRAIN_DIR / 'train_source1.tsv', sep='\t', index=False)
    s2_train.to_csv(TRAIN_DIR / 'train_source2.tsv', sep='\t', index=False)
    s3_train.to_csv(TRAIN_DIR / 'train_source3.tsv', sep='\t', index=False)
    gt_train.to_csv(TRAIN_DIR / 'train_ground_truth.tsv', sep='\t', index=False)

    # --- TEST DATA ---
    s1_test = pd.DataFrame([
        {'entity_id': 'S1-101', 'business_name': 'Google LLC', 'business_address': '1600 Amphitheatre Pkwy', 'country': 'US'},
        {'entity_id': 'S1-102', 'business_name': 'Loreal Paris', 'business_address': '14 Rue Royale', 'country': 'France'}, # France included per PDF
    ])

    s2_test = pd.DataFrame([
        {'entity_id': 'S2-101', 'business_name': 'Google Limited Liability Corp', 'business_address': '1600 Amphitheatre Parkway, Mountain View', 'country': 'US'},
    ])

    s3_test = pd.DataFrame([
        {'entity_id': 'S3-101', 'business_name': 'Google', 'business_address': 'Amphitheatre Pkwy', 'country': 'US'},
        {'entity_id': 'S3-102', 'business_name': 'Loreal', 'business_address': '14 Rue Royale, Paris', 'country': 'France'},
    ])

    # Save Test Data
    s1_test.to_csv(TEST_DIR / 'test_source1.tsv', sep='\t', index=False)
    s2_test.to_csv(TEST_DIR / 'test_source2.tsv', sep='\t', index=False)
    s3_test.to_csv(TEST_DIR / 'test_source3.tsv', sep='\t', index=False)

    print("Mock dataset generated successfully!")

if __name__ == "__main__":
    create_mock_data()
