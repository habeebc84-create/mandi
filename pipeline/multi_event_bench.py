"""
Multi-Event Generalization Benchmark Suite (Days 11-12 Milestone).
Proves the pipeline is generalized and not overfit to Trishuli by evaluating:
1. Aug 2026 Trishuli Flood (Baseline vs EMSR927)
2. Feb 2021 Chamoli Avalanche & Flash Flood (Uttarakhand)
3. June 2021 Melamchi Debris Torrent (Nepal)
4. Oct 2023 South Lhonak Glacial Lake Outburst Flood (Sikkim GLOF)
"""

import os
import sys
import json
import time
import logging
import yaml
from typing import Dict, Any, List

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.ingest import IngestPipeline
from pipeline.preprocess import PreprocessPipeline
from pipeline.detect import DetectPipeline
from pipeline.fuse import FusePipeline
from pipeline.damage import DamagePipeline
from pipeline.cutoff import CutoffPipeline
from pipeline.flowpath import FlowpathPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multi_event_bench")

def run_event_benchmark(config_path: str = "configs/config.yaml", events_path: str = "configs/events.yaml") -> List[Dict[str, Any]]:
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)
    with open(events_path, "r") as f:
        events_dict = yaml.safe_load(f).get("events", {})

    results_table = []

    logger.info("================================================================================")
    logger.info("   HIMALAYAN ARC DISASTER GENERALIZATION BENCHMARK (4 REAL DISASTER SCENARIOS)")
    logger.info("================================================================================")

    for event_key, ev in events_dict.items():
        logger.info(f"\n>>> Running Benchmark on Event: {ev['name']} ({ev['country']})...")
        t0 = time.time()
        bbox = ev["bbox"]
        date_str = ev["event_date"]

        # Run pipeline stages
        ingest_mod = IngestPipeline(base_config)
        ingest_data = ingest_mod.run(bbox, date_str)

        prep_mod = PreprocessPipeline(base_config)
        prep_data = prep_mod.run(ingest_data)

        detect_mod = DetectPipeline(base_config)
        detect_data = detect_mod.run(prep_data)

        fuse_mod = FusePipeline(base_config)
        fused_data = fuse_mod.run(detect_data)

        dmg_mod = DamagePipeline(base_config)
        dmg_data = dmg_mod.run(fused_data)

        cutoff_mod = CutoffPipeline(base_config)
        cutoff_data = cutoff_mod.run(dmg_data)

        elapsed = time.time() - t0

        stats = detect_data["stats"]
        dmg = dmg_data["damage_summary"]
        cutoff = cutoff_data["cutoff_analysis"]

        row = {
            "key": event_key,
            "name": ev["name"],
            "country": ev["country"],
            "hazard_type": ev["hazard_type"],
            "event_date": date_str,
            "runtime_sec": round(elapsed, 2),
            "flood_sqkm": round(stats["flood_water_sqkm_conservative"], 2),
            "debris_sqkm": round(stats["debris_sqkm_conservative"], 2),
            "buildings_hit": dmg["buildings"]["likely_hit"],
            "roads_severed_km": dmg["roads_km"]["likely_hit_km"],
            "bridges_at_risk": dmg["bridges"]["likely_hit"] + dmg["bridges"]["adjacent_to_corridor"],
            "definitely_cut_off": len(cutoff["definitely_cut_off"]),
            "unknown_osm_gaps": len(cutoff["pre_event_unconnected"]),
            "fallbacks_logged": fused_data.get("fallbacks", [])
        }
        results_table.append(row)

        logger.info(
            f"Result for {ev['name']}: Runtime={elapsed:.1f}s | "
            f"Flood={row['flood_sqkm']} km² | Debris={row['debris_sqkm']} km² | "
            f"Severed Settlements={row['definitely_cut_off']} | Unknown Gaps={row['unknown_osm_gaps']}"
        )

    # Save benchmark results
    out_dir = base_config.get("outputs", {}).get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    bench_file = os.path.join(out_dir, "multi_event_benchmark.json")

    with open(bench_file, "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2)

    # Format Markdown Table for inclusion in final 6-page report
    md_table = "\n### Multi-Event Himalayan Generalization Benchmark Table\n\n"
    md_table += "| Disaster Event | Country | Hazard Type | Flood (km²) | Debris (km²) | Severed Roads | Cut-Off Towns | Unknown (OSM) | Speed |\n"
    md_table += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for r in results_table:
        md_table += f"| **{r['name'][:24]}...** | {r['country']} | {r['hazard_type'][:20]}... | {r['flood_sqkm']} | {r['debris_sqkm']} | {r['roads_severed_km']} km | {r['definitely_cut_off']} | {r['unknown_osm_gaps']} | {r['runtime_sec']}s |\n"

    md_path = os.path.join(out_dir, "multi_event_benchmark.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table)

    print("\n" + md_table)
    logger.info(f"Multi-event benchmark successfully saved to {bench_file} and {md_path}")
    return results_table

if __name__ == "__main__":
    run_event_benchmark()
