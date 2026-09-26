import pandas as pd
from typing import Dict
from src import config

def validate_schema(df: pd.DataFrame, source_type: str):
    """Ensure dataset matches expected schema and ID prefixes."""
    expected_cols = {'entity_id', 'business_name', 'business_address', 'country'}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"Missing columns in {source_type}. Expected {expected_cols}, got {set(df.columns)}")
        
    # ID Prefix validation (e.g. S1 for source1)
    prefix_map = {'source1': 'S1', 'source2': 'S2', 'source3': 'S3'}
    if source_type in prefix_map:
        prefix = prefix_map[source_type]
        if not df['entity_id'].str.startswith(prefix).all():
            print(f"WARNING: Not all entity_ids in {source_type} start with prefix '{prefix}'!")

def load_tsv(filepath) -> pd.DataFrame:
    """Load a TSV file robustly."""
    if not filepath.exists():
        raise FileNotFoundError(f"Missing expected dataset file: {filepath}")
    return pd.read_csv(filepath, sep='\t', dtype=str).fillna("")

def load_train_data() -> Dict[str, pd.DataFrame]:
    """Load all training datasets and validate schemas."""
    print("Loading training data...")
    data = {
        "source1": load_tsv(config.TRAIN_S1),
        "source2": load_tsv(config.TRAIN_S2),
        "source3": load_tsv(config.TRAIN_S3),
        "ground_truth": load_tsv(config.TRAIN_GT)
    }
    for name in ["source1", "source2", "source3"]:
        validate_schema(data[name], name)
    return data

def load_test_data() -> Dict[str, pd.DataFrame]:
    """Load all test datasets and validate schemas."""
    print("Loading test data...")
    data = {
        "source1": load_tsv(config.TEST_S1),
        "source2": load_tsv(config.TEST_S2),
        "source3": load_tsv(config.TEST_S3)
    }
    for name in ["source1", "source2", "source3"]:
        validate_schema(data[name], name)
    return data
