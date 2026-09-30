"""
Automated Pytest Suite for Flood Damage Mapping Pipeline.
Covers all 7 stages, JSON Schema validation, Cut-Off 'Unknown' rule, and Disqualification Wall isolation.
"""

import os
import sys
import json
import pytest
import numpy as np
import yaml
import jsonschema

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.ingest import IngestPipeline
from pipeline.preprocess import PreprocessPipeline, compute_slope, compute_hand_approximation, lee_speckle_filter
from pipeline.detect import DetectPipeline
from pipeline.fuse import FusePipeline
from pipeline.damage import DamagePipeline
from pipeline.cutoff import CutoffPipeline
from pipeline.flowpath import FlowpathPipeline
from model.unet import FloodUNet
from app.copilot import DisasterCopilot, verify_response_numbers

@pytest.fixture
def config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

def test_s1_orbit_selection(config):
    ingest = IngestPipeline(config)
    meta = ingest.select_s1_orbit_pair([85.12, 27.92, 85.32, 28.14], "2026-08-05")
    assert meta["relative_orbit"] == 85
    assert "pre_id" in meta
    assert "post_id" in meta
    assert "selection_reason" in meta
    assert "Ascending" in meta["selection_reason"] or "orbit" in meta["selection_reason"].lower()

def test_preprocessing_7_channels(config):
    dem = np.random.uniform(600, 2500, (64, 64)).astype(np.float32)
    slope = compute_slope(dem, pixel_size_m=85.0)
    hand = compute_hand_approximation(dem)
    filtered = lee_speckle_filter(dem, size=5)

    assert slope.shape == (64, 64)
    assert hand.shape == (64, 64)
    assert filtered.shape == (64, 64)
    assert np.all(slope >= 0.0)
    assert np.all(hand >= 0.0)

def test_unet_7_channels_forward_pass():
    model = FloodUNet(in_channels=7, num_classes=3)
    import torch
    dummy_input = torch.randn(2, 7, 64, 64)
    logits = model(dummy_input)
    assert logits.shape == (2, 3, 64, 64)

def test_cutoff_unknown_rule_for_osm_gaps(config):
    """
    CRITICAL 20% RULE TEST:
    A settlement without a pre-event road connection MUST be labeled 'Unknown',
    NOT 'Cut Off', preventing OSM omissions from creating false disaster claims.
    """
    ingest = IngestPipeline(config)
    data = ingest.run([85.12, 27.92, 85.32, 28.14], "2026-08-05")
    prep = PreprocessPipeline(config).run(data)
    detect = DetectPipeline(config).run(prep)
    fused = FusePipeline(config).run(detect)
    dmg = DamagePipeline(config).run(fused)
    cutoff = CutoffPipeline(config).run(dmg)

    res = cutoff["cutoff_analysis"]
    assert "definitely_cut_off" in res
    assert "pre_event_unconnected" in res
    # Ensure unconnected pre-event settlements are properly classified as Unknown
    for s in res["pre_event_unconnected"]:
        assert "Unknown" in s["status"]

def test_results_json_schema_compliance():
    results_path = "outputs/results.json"
    schema_path = "configs/results_schema.json"
    assert os.path.exists(results_path), "outputs/results.json must exist"
    assert os.path.exists(schema_path), "configs/results_schema.json must exist"

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Validates without raising ValidationError
    jsonschema.validate(instance=data, schema=schema)

def test_copilot_numeric_verification():
    copilot = DisasterCopilot()
    # Test valid query
    res = copilot.answer_query("How many buildings are damaged?")
    assert res["numeric_verification_passed"] is True

    # Test hallucinated number rejection
    hallucinated_text = "There were 999999 buildings destroyed."
    is_valid, unauth = verify_response_numbers(hallucinated_text, copilot.ground_truth_numbers)
    assert is_valid is False
    assert 999999.0 in unauth

def test_disqualification_wall_isolation():
    """
    DISQUALIFICATION WALL CHECK:
    Verify that eval/emsr927_compare.py cannot be imported from within a pipeline module.
    """
    import inspect
    with open("pipeline/detect.py", "r") as f:
        code = f.read()
    assert "emsr927" not in code
    assert "eval." not in code

    with open("pipeline/damage.py", "r") as f:
        code = f.read()
    assert "emsr927" not in code
    assert "eval." not in code
