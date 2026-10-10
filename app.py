from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "benchmarks" / "benchmark_summary.json"
RESULTS_PATH = ROOT / "output" / "matching_results.tsv"
DEMO_METRICS_PATH = ROOT / "demo" / "demo_validation.json"
DEMO_RESULTS_PATH = ROOT / "demo" / "demo_predictions.tsv"
REQUIRED_DATA_FILES = [
    ROOT / "data" / "train" / "train_source1.tsv",
    ROOT / "data" / "train" / "train_source2.tsv",
    ROOT / "data" / "train" / "train_source3.tsv",
    ROOT / "data" / "train" / "train_ground_truth.tsv",
    ROOT / "data" / "test" / "test_source1.tsv",
    ROOT / "data" / "test" / "test_source2.tsv",
    ROOT / "data" / "test" / "test_source3.tsv",
]
DATA_AVAILABLE = all(path.exists() for path in REQUIRED_DATA_FILES)


st.set_page_config(page_title="Entity Resolution Lab", page_icon="ER", layout="wide")
st.title("Business Entity Resolution")
st.caption("Amazon ML Challenge pipeline: blocking, similarity features, and precision-focused matching")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


summary = load_json(SUMMARY_PATH)
demo_metrics = load_json(DEMO_METRICS_PATH)
left, middle, right = st.columns(3)
left.metric("Sample candidates", summary.get("total_candidates", "-"))
middle.metric("Candidate generation", f"{summary['candidate_generation_time_seconds']:.2f}s" if summary else "-")
right.metric("Feature generation", f"{summary['feature_generation_time_seconds']:.2f}s" if summary else "-")

st.subheader("Evaluation")
if demo_metrics:
    st.caption("Reproducible sample validation using the real pipeline components (1,000 sampled records per source).")
    metric_columns = st.columns(5)
    metric_columns[0].metric("Accuracy", f"{demo_metrics['accuracy']:.2%}")
    metric_columns[1].metric("Precision", f"{demo_metrics['precision']:.2%}")
    metric_columns[2].metric("Recall", f"{demo_metrics['recall']:.2%}")
    metric_columns[3].metric("F1", f"{demo_metrics['f1']:.2%}")
    metric_columns[4].metric("F0.5", f"{demo_metrics['f05']:.2%}")
if DATA_AVAILABLE:
    st.info(
        "The challenge metric is validation F0.5. Run the pipeline to calculate validation accuracy, "
        "precision, recall, F1, and F0.5 on the held-out training split."
    )
else:
    st.info(
        "This public demo includes the code and benchmark results, but not the private challenge datasets. "
        "Add the required files under data/ to run matching locally."
    )

if st.button("Run full pipeline", type="primary", disabled=not DATA_AVAILABLE):
    with st.status("Running pipeline...", expanded=True) as status:
        process = subprocess.run(
            [sys.executable, str(ROOT / "src" / "pipeline.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        st.code(process.stdout or process.stderr)
        if process.returncode == 0:
            status.update(label="Pipeline completed", state="complete")
        else:
            status.update(label="Pipeline failed", state="error")

if RESULTS_PATH.exists():
    st.subheader("Latest matching output")
    results = pd.read_csv(RESULTS_PATH, sep="\t")
    st.write(f"{len(results):,} Source 1 entities in the latest output")
    st.dataframe(results.head(100), width="stretch")
elif DEMO_RESULTS_PATH.exists():
    st.subheader("Sample validation output")
    demo_results = pd.read_csv(DEMO_RESULTS_PATH, sep="\t")
    st.caption("These are validation candidate predictions from the reproducible demo sample, not hidden test-set results.")
    st.dataframe(demo_results.head(100), width="stretch")
else:
    st.caption("Matching output is generated locally from the challenge data and is not included in this public demo.")

st.subheader("Blocking rules")
st.write("Exact name/address, first normalized name token, and country + name-prefix + address-token blocking.")