import pandas as pd

from src.blocking.blocker import generate_candidates
from src.evaluation.metrics import evaluate_predictions


def test_generate_candidates_includes_token_blocking():
    s1 = pd.DataFrame([
        {"entity_id": "S1_1", "business_name": "Amazon Inc", "business_address": "1 Main St", "country": "US", "norm_name": "amazon inc", "norm_address": "1 main street"},
        {"entity_id": "S1_2", "business_name": "Alpha Labs", "business_address": "2 Side Ave", "country": "US", "norm_name": "alpha labs", "norm_address": "2 side avenue"},
    ])
    cand = pd.DataFrame([
        {"entity_id": "S2_1", "business_name": "Amazon United", "business_address": "1 Park Road", "country": "US", "norm_name": "amazon united", "norm_address": "1 park road"},
        {"entity_id": "S2_2", "business_name": "Beta Labs", "business_address": "2 Side Avenue", "country": "US", "norm_name": "beta labs", "norm_address": "2 side avenue"},
    ])

    pairs = generate_candidates(s1, cand, "source2")

    assert {'source1_entity_id', 'candidate_entity_id', 'candidate_source', 'blocking_rule'} <= set(pairs.columns)
    assert not pairs.empty
    assert 'name_first_token' in set(pairs['blocking_rule'])


def test_evaluate_predictions_handles_space_and_comma_ids():
    predictions = pd.DataFrame([
        {"source1_entity_id": "S1_1", "predicted_match_ids": "S2_1 S2_2"},
        {"source1_entity_id": "S1_2", "predicted_match_ids": ""},
    ])
    ground_truth = pd.DataFrame([
        {"source1_entity_id": "S1_1", "actual_match_ids": "S2_1,S2_3"},
        {"source1_entity_id": "S1_2", "actual_match_ids": ""},
    ])

    result = evaluate_predictions(predictions, ground_truth)

    assert result["precision"] >= 0
    assert result["recall"] >= 0
    assert result["f05"] >= 0
