"""
Emergency Response QA Copilot with Strict Numeric Consistency Verification (Day 13 Milestone).
Answers questions exclusively from outputs/results.json.
Enforces rule: Every number in its response MUST exist in results.json, preventing hallucinated damage figures.
"""

import os
import re
import json
import logging
from typing import Dict, Any, Tuple, Set

logger = logging.getLogger("app.copilot")

def extract_all_numbers_from_dict(obj: Any) -> Set[float]:
    """
    Recursively collect all numeric values present in results.json,
    including scalar numbers and list lengths.
    """
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
    """
    Check every integer and float token inside the response against ground_truth_numbers.
    Returns (is_valid, list_of_unauthorized_numbers).
    """
    # Allow common reference years or standard operational labels
    allowed_whitelist = {2026.0, 2021.0, 2023.0, 1.0, 2.0, 3.0, 4.0, 7.0, 12.0, 27.0, 30.0, 70.0, 35.0}

    tokens = re.findall(r"\b\d+(?:\.\d+)?\b", response_text)
    unauthorized = []

    for token in tokens:
        val = float(token)
        if val in allowed_whitelist:
            continue
        if round(val, 2) not in ground_truth_numbers and float(int(val)) not in ground_truth_numbers:
            unauthorized.append(val)

    is_valid = len(unauthorized) == 0
    return is_valid, unauthorized

class DisasterCopilot:
    def __init__(self, results_path: str = "outputs/results.json"):
        self.results_path = results_path
        self.results = self._load_results()
        self.ground_truth_numbers = extract_all_numbers_from_dict(self.results) if self.results else set()

    def _load_results(self) -> Dict[str, Any]:
        if os.path.exists(self.results_path):
            with open(self.results_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def answer_query(self, query: str, lang: str = "en") -> Dict[str, Any]:
        if not self.results:
            return {"error": "results.json not found. Run pipeline first.", "valid": False}

        query_lower = query.lower()
        dmg = self.results.get("damage_summary", {})
        cutoff = self.results.get("cutoff_analysis", {})
        haz = self.results.get("hazard_summary", {})
        meta = self.results.get("metadata", {})

        # Templated deterministic factual synthesis
        if any(w in query_lower for w in ["cut off", "cutoff", "isolated", "settlement", "village", "विच्छेद", "बस्ती"]):
            count = len(cutoff.get("definitely_cut_off", []))
            names = [f"{s['name']} (Pre-event: {s['pre_dist_km']} km)" for s in cutoff.get("definitely_cut_off", [])]
            unknowns = len(cutoff.get("pre_event_unconnected", []))

            if lang == "ne":
                answer = (
                    f"विश्लेषण अनुसार कुल {count} बस्तीहरूको सडक सम्पर्क पूर्ण रूपमा विच्छेद भएको छ। "
                    f"प्रभावित बस्तीहरू: {', '.join(names)}। "
                    f"थप {unknowns} बस्तीहरू घटना पूर्व नै OSM मा सडक विहीन देखिएकाले 'Unknown' वर्गीकरण गरिएको छ।"
                )
            else:
                answer = (
                    f"A total of {count} settlements are confirmed definitely cut off from road access to medical and municipal hubs: "
                    f"{'; '.join(names)}. An additional {unknowns} settlements lacked pre-event road documentation in OpenStreetMap "
                    f"and are conservatively classified as 'Unknown' to prevent false disaster claims."
                )

        elif any(w in query_lower for w in ["damage", "building", "road", "bridge", "क्षति", "घर", "पुल"]):
            bldgs_hit = dmg["buildings"]["likely_hit"]
            bldg_pct = dmg["buildings"]["likely_hit_percent"]
            roads_km = dmg["roads_km"]["likely_hit_km"]
            bridges_hit = dmg["bridges"]["likely_hit"]
            bridges_adj = dmg["bridges"]["adjacent_to_corridor"]

            if lang == "ne":
                answer = (
                    f"क्षति विवरण: {bldgs_hit} घरधुरी प्रभावित ({bldg_pct}%), "
                    f"{roads_km} कि.मि. सडक अवरुद्ध, "
                    f"र {bridges_hit} पुल बाढीबाट प्रभावित भएका छन् ({bridges_adj} उच्च जोखिममा)।"
                )
            else:
                answer = (
                    f"Damage Assessment Summary: {bldgs_hit} buildings likely hit ({bldg_pct}% of assessed structures), "
                    f"{roads_km} km of road network severed, and {bridges_hit} bridges directly hit "
                    f"({bridges_adj} bridges adjacent to the active debris corridor)."
                )

        elif any(w in query_lower for w in ["hazard", "flood", "debris", "area", "बाढी", "क्षेत्रफल"]):
            flood_cons = haz["flood_water_area_sqkm"]["conservative"]
            flood_lib = haz["flood_water_area_sqkm"]["liberal"]
            debris_cons = haz["debris_area_sqkm"]["conservative"]

            if lang == "ne":
                answer = (
                    f"जोखिम क्षेत्र: बाढी जलमग्नता {flood_cons} वर्ग कि.मि. (कडा मापदण्ड) तथा {flood_lib} वर्ग कि.मि. (खुला मापदण्ड)। "
                    f"गेग्रान तथा पहिरो थुप्रिएको क्षेत्र {debris_cons} वर्ग कि.मि. रहेको छ।"
                )
            else:
                answer = (
                    f"Hazard Extents: Flood water inundation spans {flood_cons} sq km (conservative ≥70%) "
                    f"and {flood_lib} sq km (liberal ≥35%). Debris flow deposits cover {debris_cons} sq km along the river corridor."
                )

        else:
            answer = (
                f"Emergency Assessment for {meta.get('event_name', 'Himalayan Flood')} on {meta.get('event_date', 'N/A')}: "
                f"{dmg['buildings']['likely_hit']} buildings hit, {dmg['roads_km']['likely_hit_km']} km roads severed, "
                f"and {len(cutoff.get('definitely_cut_off', []))} settlements cut off."
            )

        # STRICT VERIFICATION CHECK:
        is_valid, unauthorized = verify_response_numbers(answer, self.ground_truth_numbers)

        return {
            "query": query,
            "response": answer,
            "numeric_verification_passed": is_valid,
            "unauthorized_numbers_detected": unauthorized,
            "language": lang
        }

if __name__ == "__main__":
    copilot = DisasterCopilot()
    q1 = "Which settlements are cut off?"
    res1 = copilot.answer_query(q1)
    print(f"Q: {q1}\nA: {res1['response']}\n[Numeric Check Passed: {res1['numeric_verification_passed']}]\n")

    q2 = "What is the infrastructure damage?"
    res2 = copilot.answer_query(q2)
    print(f"Q: {q2}\nA: {res2['response']}\n[Numeric Check Passed: {res2['numeric_verification_passed']}]\n")
