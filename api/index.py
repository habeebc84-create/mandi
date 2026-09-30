"""
Vercel Serverless Function Entrypoint for Mandi 3D Flood Command Center.
"""

import os
import sys
import json
from typing import Optional, Set, Any, Tuple
import re

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="Mandi 3D Flood Command Center - Vercel")

CURRENT_DIR = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(CURRENT_DIR, "results.json")
if not os.path.exists(RESULTS_PATH):
    RESULTS_PATH = os.path.join(CURRENT_DIR, "..", "outputs", "results.json")

# Import the full 3D dashboard HTML from app.dashboard.app
try:
    from app.dashboard.app import DASHBOARD_HTML
except ImportError:
    DASHBOARD_HTML = "<h1>Mandi 3D Flood Command Center is loading...</h1>"

def extract_all_numbers_from_dict(obj: Any) -> Set[float]:
    numbers = set()
    if isinstance(obj, (int, float)):
        numbers.add(round(float(obj), 2))
        numbers.add(float(int(obj)))
    elif isinstance(obj, dict):
        for val in obj.values():
            numbers.update(extract_all_numbers_from_dict(val))
    elif isinstance(obj, list):
        numbers.add(float(len(obj)))
        for item in obj:
            numbers.update(extract_all_numbers_from_dict(item))
    return numbers

def verify_response_numbers(response_text: str, ground_truth_numbers: Set[float]) -> Tuple[bool, list]:
    allowed_whitelist = {2026.0, 2021.0, 2023.0, 1.0, 2.0, 3.0, 4.0, 7.0, 12.0, 27.0, 30.0, 70.0, 35.0}
    tokens = re.findall(r"\b\d+(?:\.\d+)?\b", response_text)
    unauthorized = []
    for token in tokens:
        val = float(token)
        if val in allowed_whitelist:
            continue
        if round(val, 2) not in ground_truth_numbers and float(int(val)) not in ground_truth_numbers:
            unauthorized.append(val)
    return len(unauthorized) == 0, unauthorized

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return DASHBOARD_HTML

@app.get("/api/status")
@app.get("/health")
def get_status():
    return {
        "status": "online",
        "app": "Mandi 3D Flood Command Center",
        "deployment": "Vercel Serverless",
        "version": "1.0.0",
        "results_ready": os.path.exists(RESULTS_PATH)
    }

@app.get("/api/results")
def get_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return JSONResponse(
        status_code=404,
        content={"error": "results.json not found."}
    )

@app.get("/api/copilot/query")
def copilot_query(q: str, lang: str = "en"):
    if not os.path.exists(RESULTS_PATH):
        return {"error": "results.json not found", "valid": False}
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    gt_numbers = extract_all_numbers_from_dict(results)
    query_lower = q.lower()
    dmg = results.get("damage_summary", {})
    cutoff = results.get("cutoff_analysis", {})
    haz = results.get("hazard_summary", {})
    meta = results.get("metadata", {})

    if any(w in query_lower for w in ["cut off", "cutoff", "isolated", "settlement", "village"]):
        count = len(cutoff.get("definitely_cut_off", []))
        names = [f"{s['name']} (Pre-event: {s['pre_dist_km']} km)" for s in cutoff.get("definitely_cut_off", [])]
        unknowns = len(cutoff.get("pre_event_unconnected", []))
        answer = (
            f"A total of {count} settlements are confirmed definitely cut off from road access to medical and municipal hubs: "
            f"{'; '.join(names)}. An additional {unknowns} settlements lacked pre-event road documentation in OpenStreetMap "
            f"and are conservatively classified as 'Unknown' to prevent false disaster claims."
        )
    elif any(w in query_lower for w in ["damage", "building", "road", "bridge"]):
        bldgs_hit = dmg["buildings"]["likely_hit"]
        bldg_pct = dmg["buildings"]["likely_hit_percent"]
        roads_km = dmg["roads_km"]["likely_hit_km"]
        bridges_hit = dmg["bridges"]["likely_hit"]
        bridges_adj = dmg["bridges"]["adjacent_to_corridor"]
        answer = (
            f"Damage Assessment Summary: {bldgs_hit} buildings likely hit ({bldg_pct}% of assessed structures), "
            f"{roads_km} km of road network severed, and {bridges_hit} bridges directly hit "
            f"({bridges_adj} bridges adjacent to the active debris corridor)."
        )
    else:
        answer = (
            f"Emergency Assessment for {meta.get('event_name', 'Himalayan Flood')}: "
            f"{dmg['buildings']['likely_hit']} buildings hit, {dmg['roads_km']['likely_hit_km']} km roads severed, "
            f"and {len(cutoff.get('definitely_cut_off', []))} settlements cut off."
        )

    is_valid, unauth = verify_response_numbers(answer, gt_numbers)
    return {
        "query": q,
        "response": answer,
        "numeric_verification_passed": is_valid,
        "language": lang
    }
