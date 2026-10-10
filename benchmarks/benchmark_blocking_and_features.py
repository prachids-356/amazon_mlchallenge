import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.blocking.blocker import generate_candidates
from src.features.pair_features import generate_pair_features
from src.ingestion.loader import load_train_data
from src.preprocessing.name_normalizer import normalize_name
from src.preprocessing.address_normalizer import normalize_address


def _ensure_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'norm_name' not in df.columns:
        df['norm_name'] = df['business_name'].apply(normalize_name)
    if 'norm_address' not in df.columns:
        df['norm_address'] = df['business_address'].apply(normalize_address)
    return df


def benchmark_candidate_generation(sample_size: int = 1000):
    data = load_train_data()
    s1 = _ensure_normalized_columns(data['source1'].head(sample_size).copy())
    s2 = _ensure_normalized_columns(data['source2'].head(sample_size).copy())
    s3 = _ensure_normalized_columns(data['source3'].head(sample_size).copy())

    start = time.time()
    cands_s2 = generate_candidates(s1, s2, 'source2')
    cands_s3 = generate_candidates(s1, s3, 'source3')
    total = pd.concat([cands_s2, cands_s3], ignore_index=True)
    elapsed = time.time() - start

    summary = {
        'sample_size': sample_size,
        'source2_candidates': int(len(cands_s2)),
        'source3_candidates': int(len(cands_s3)),
        'total_candidates': int(len(total)),
        'candidate_generation_time_seconds': round(elapsed, 4),
    }

    print(f"Sample size: {sample_size}")
    print(f"Candidates source2: {len(cands_s2)}")
    print(f"Candidates source3: {len(cands_s3)}")
    print(f"Total candidates: {len(total)}")
    print(f"Candidate generation time: {elapsed:.2f}s")

    print("Candidate rule counts:")
    if not total.empty:
        counts = total['blocking_rule'].value_counts().to_dict()
        print(counts)
        summary['blocking_rule_counts'] = counts

    return total, summary


def benchmark_feature_generation(sample_pairs: pd.DataFrame, chunk_size: int = 200_000):
    if sample_pairs.empty:
        print("No sample pairs to benchmark.")
        return {'feature_generation_time_seconds': 0.0, 'feature_rows': 0, 'columns': []}

    data = load_train_data()
    s1 = _ensure_normalized_columns(data['source1'].copy())
    s23 = pd.concat([
        _ensure_normalized_columns(data['source2'].copy()),
        _ensure_normalized_columns(data['source3'].copy())
    ], ignore_index=True)

    start = time.time()
    features = generate_pair_features(sample_pairs, s1, s23)
    elapsed = time.time() - start

    print(f"Feature generation time: {elapsed:.2f}s")
    print(f"Feature rows: {len(features)}")
    print(f"Columns: {list(features.columns)}")
    print(features.head(3).to_string(index=False))

    summary = {
        'feature_generation_time_seconds': round(elapsed, 4),
        'feature_rows': int(len(features)),
        'columns': list(features.columns),
        'chunk_size': chunk_size,
    }
    return summary


def write_summary(output_dir: Path, summary: dict, prefix: str = "benchmark"):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{prefix}.csv"
    json_path = output_dir / f"{prefix}.json"

    pd.Series(summary).to_csv(csv_path)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"Saved benchmark summary to: {csv_path}")
    print(f"Saved benchmark summary JSON to: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark blocking and feature generation on a representative sample.")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of Source 1 rows to sample for the benchmark.")
    parser.add_argument("--chunk-size", type=int, default=200_000, help="Chunk size used during non-exact similarity computation.")
    parser.add_argument("--output-dir", type=str, default="benchmarks", help="Directory for summary CSV/JSON outputs.")
    args = parser.parse_args()

    bench_dir = Path(args.output_dir)
    sample_pairs, candidate_summary = benchmark_candidate_generation(sample_size=args.sample_size)
    feature_summary = benchmark_feature_generation(sample_pairs, chunk_size=args.chunk_size)

    combined_summary = {
        **candidate_summary,
        **feature_summary,
    }
    write_summary(bench_dir, combined_summary, prefix="benchmark_summary")
