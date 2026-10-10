import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.blocking.blocker import generate_candidates
from src.evaluation.metrics import calculate_binary_metrics
from src.features.pair_features import generate_pair_features
from src.models.labeling import create_labels
from src.models.matcher import EntityMatcher
from src.models.threshold import tune_threshold
from src.preprocessing.address_normalizer import normalize_address
from src.preprocessing.name_normalizer import normalize_name

DATA_DIR = ROOT / "data" / "train"
OUTPUT_DIR = ROOT / "demo"
SAMPLE_SIZE = 1000


def load_sample(filename: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / filename, sep="\t", dtype=str, nrows=SAMPLE_SIZE).fillna("")


def load_candidate_sample(filename: str, required_ids: set[str]) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(DATA_DIR / filename, sep="\t", dtype=str, chunksize=100_000):
        selected = chunk[chunk["entity_id"].isin(required_ids)]
        if not selected.empty:
            chunks.append(selected)
        if sum(len(item) for item in chunks) >= SAMPLE_SIZE:
            break
    return pd.concat(chunks, ignore_index=True).fillna("") if chunks else pd.DataFrame()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["norm_name"] = df["business_name"].apply(normalize_name)
    df["norm_address"] = df["business_address"].apply(normalize_address)
    return df


def main() -> None:
    ground_truth = pd.read_csv(DATA_DIR / "train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    source1 = prepare(load_sample("train_source1.tsv"))
    ground_truth = ground_truth[ground_truth["source1_entity_id"].isin(source1["entity_id"])]

    required_ids = set()
    for raw_ids in ground_truth["matched_entity_ids"]:
        required_ids.update(item for item in re.split(r"[\s,]+", raw_ids) if item)
    source2 = prepare(load_candidate_sample("train_source2.tsv", required_ids))
    source3 = prepare(load_candidate_sample("train_source3.tsv", required_ids))
    source2 = pd.concat([source2, prepare(load_sample("train_source2.tsv"))], ignore_index=True).drop_duplicates("entity_id")
    source3 = pd.concat([source3, prepare(load_sample("train_source3.tsv"))], ignore_index=True).drop_duplicates("entity_id")

    train_s1, val_s1 = train_test_split(source1, test_size=0.2, random_state=42)
    candidates_train = pd.concat(
        [
            generate_candidates(train_s1, source2, "source2"),
            generate_candidates(train_s1, source3, "source3"),
        ],
        ignore_index=True,
    )
    candidates_val = pd.concat(
        [
            generate_candidates(val_s1, source2, "source2"),
            generate_candidates(val_s1, source3, "source3"),
        ],
        ignore_index=True,
    )
    combined_candidates = pd.concat([source2, source3], ignore_index=True)

    train_features = create_labels(
        generate_pair_features(candidates_train, train_s1, combined_candidates), ground_truth
    )
    val_features = create_labels(
        generate_pair_features(candidates_val, val_s1, combined_candidates), ground_truth
    )

    if train_features.empty or val_features.empty or train_features["label"].nunique() < 2:
        raise RuntimeError("The demo sample does not contain enough positive and negative pairs.")

    matcher = EntityMatcher()
    matcher.fit(train_features)
    val_features["match_prob"] = matcher.predict_proba(val_features)
    threshold, tuned_f05 = tune_threshold(val_features, "match_prob")
    predictions = (val_features["match_prob"] >= threshold).astype(int)
    metrics = calculate_binary_metrics(val_features["label"], predictions)

    report = {
        "evaluation_type": "sample validation",
        "sample_size_per_source": SAMPLE_SIZE,
        "validation_entities": int(len(val_s1)),
        "training_candidate_pairs": int(len(train_features)),
        "validation_candidate_pairs": int(len(val_features)),
        "threshold": round(float(threshold), 4),
        "tuned_macro_f05": round(float(tuned_f05), 4),
        **{key: round(float(value), 4) for key, value in metrics.items()},
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "demo_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    val_features[["source1_entity_id", "candidate_entity_id", "label", "match_prob"]].to_csv(
        OUTPUT_DIR / "demo_predictions.tsv", sep="\t", index=False
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
