import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = BASE_DIR / "output"

# Train files
TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

# Test files
TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

# Output files
MATCHING_RESULTS = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS = OUTPUT_DIR / "candidate_pairs.tsv"
