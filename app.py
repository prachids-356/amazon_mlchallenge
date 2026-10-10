from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "benchmarks" / "benchmark_summary.json"
RESULTS_PATH = ROOT / "output" / "matching_results.tsv"


st.set_page_config(page_title="Entity Resolution Lab", page_icon="ER", layout="wide")
st.title("Business Entity Resolution")
st.caption("Amazon ML Challenge pipeline: blocking, similarity features, and precision-focused matching")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


summary = load_json(SUMMARY_PATH)
left, middle, right = st.columns(3)
left.metric("Sample candidates", summary.get("total_candidates", "-"))
middle.metric("Candidate generation", f"{summary['candidate_generation_time_seconds']:.2f}s" if summary else "-")
right.metric("Feature generation", f"{summary['feature_generation_time_seconds']:.2f}s" if summary else "-")

st.subheader("Evaluation")
st.info(
    "The challenge metric is validation F0.5. Run the pipeline to calculate validation accuracy, "
    "precision, recall, F1, and F0.5 on the held-out training split."
)

if st.button("Run full pipeline", type="primary"):
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
else:
    st.warning("No matching output exists yet. Run the pipeline after placing the challenge data under data/.")

st.subheader("Blocking rules")
st.write("Exact name/address, first normalized name token, and country + name-prefix + address-token blocking.")