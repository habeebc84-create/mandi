# mandi - Flood Damage Mapping from Space
### Rapid Disaster Assessment in Steep Himalayan Terrain (15-Day Build Plan)

An end-to-end, config-driven operational pipeline for rapid flood damage delineation, debris change mapping, and settlement cut-off reachability analysis from satellite radar and optical imagery.

Case Study: **Aug 2026 Trishuli Flood, Nepal** (validated against Copernicus EMS **EMSR927**).

---

## 1. Quick Start (Fresh Machine Setup)

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- PyTorch & GDAL/Shapely stack

### Installation
```bash
# 1. Clone repository
git clone https://github.com/team/flood-damage-mapping.git
cd flood-damage-mapping

# 2. Install dependencies
pip install -r requirements.txt
```

### Execution
The judges choose the bounding box and event date on the day of the evaluation. **Nothing is hard-coded to Trishuli**:
```bash
# Run with arbitrary coordinates and date:
python run.py --bbox 85.12,27.92,85.32,28.14 --date 2026-08-05
```

### View Interactive Dashboard
Launch the Leaflet/FastAPI operational command center:
```bash
python app/dashboard/app.py
```
Open **`http://localhost:8000`** in your browser to inspect hazard layers, sensitivity sliders, damaged buildings/roads, cut-off villages, and trace downstream D8 flow paths.

### Independent Benchmark Validation (Disqualification Wall)
Validate pipeline results against Copernicus EMS EMSR927:
```bash
python eval/emsr927_compare.py
```

---

## 2. Core Decisions & Competitive Edge

1. **AI Component (25% + 30% Quality)**:
   - Multimodal 7-channel U-Net trained on the Kuro Siwo benchmark.
   - Evaluated on holdout Himalayan scenes from Sen1Floods11 that the model has never seen.
   - Inputs: Pre-event VV/VH, post-event VV/VH, VV difference, DEM slope, and Height Above Nearest Drainage (HAND).
2. **Sentinel-1 Primary, Sentinel-2 Secondary**:
   - Sentinel-1 SAR penetrates heavy Himalayan monsoon cloud cover.
   - Sentinel-2 is fused only on pixels verified clear by the Scene Classification Layer (SCL) to confirm MNDWI water signals and BSI sediment scars.
3. **Disqualification Wall**:
   - `eval/emsr927_compare.py` is strictly quarantined.
   - No module in `pipeline/` imports or references `eval/`.
4. **Cut-Off Reachability (20%)**:
   - Road graph Dijkstra analysis from every settlement to hospitals and town centers.
   - **Crucial Rule**: A settlement is labeled **Cut Off** only if reachable pre-event and unreachable post-event. Settlements with pre-event OSM road gaps are categorized as **Unknown**, preventing OSM omissions from creating false disaster claims.
5. **Situation Reports**:
   - Dual-language (English and Nepali with `Noto Sans Devanagari`).
   - Every number in the report is strictly generated from `results.json` with an automated numeric verification checker.

---

## 3. Pipeline Architecture

```
run.py (bbox, date)
  ├── 1. Ingest (S1 IW GRD, S2 L2A, GLO-30 DEM, ohsome OSM snapshot 2026-07-27)
  ├── 2. Preprocess (S1 RTC + Lee Speckle Filter + Slope + HAND + Shadow Mask)
  ├── 3. Detect (7-Channel Multimodal U-Net + Heuristic Debris Log-Ratio Change)
  ├── 4. Fuse (Sentinel-2 MNDWI/BSI Optical Confirmation on Clear-Sky SCL Pixels)
  ├── 5. Damage Assessment (OSM Buildings, Roads, Bridges Buffered Spatial Overlay)
  ├── 6. Cut-Off Reachability (NetworkX Dijkstra: Hospital & Municipal Hub Pre vs Post)
  ├── 7. Flow-Path Bonus (D8 Steepest Descent Elevation Trace + Threatened Corridor)
  └── 8. Outputs (results.json, damage_assessment.gpkg, HTML/PDF Reports EN & NE, Dashboard)
```

---

## 4. Repository Structure

```
flood-damage-mapping/
├── pipeline/
│   ├── ingest.py        # CDSE STAC/OData S1/S2 queries & OSM ohsome snapshot caching
│   ├── preprocess.py    # S1 RTC, Lee speckle filter, slope, HAND, layover/shadow
│   ├── detect.py        # U-Net flood inference & adaptive Otsu debris change detection
│   ├── fuse.py          # S2 MNDWI/BSI optical confirmation on SCL clear sky
│   ├── damage.py        # OSM infrastructure buffered spatial intersection
│   ├── cutoff.py        # Road graph Dijkstra reachability & detour calculation
│   └── flowpath.py      # D8 downhill flood propagation tracing
├── model/
│   ├── unet.py          # 7-channel multimodal U-Net architecture
│   ├── dataset.py       # Kuro Siwo & Sen1Floods11 dataloaders with regional holdouts
│   ├── train.py         # Model training script with Dice + CrossEntropy loss
│   ├── eval.py          # Holdout regional evaluation (IoU, precision, recall, F1)
│   └── weights/         # Checkpoints (kuro_siwo_unet.pt)
├── app/
│   ├── dashboard/       # Interactive Leaflet web dashboard
│   ├── report/          # Jinja2 situation report generator
│   └── templates/       # Dual-language templates (EN & NE)
├── eval/
│   └── emsr927_compare.py  # ISOLATED BENCHMARK: Compares with Copernicus EMSR927
├── configs/
│   ├── config.yaml      # Master configuration
│   └── results_schema.json # Strict JSON schema validator
├── outputs/             # Generated results.json, GPKG, PDFs, HTML
├── requirements.txt
├── ATTRIBUTION.md
└── README.md
```

---

## 5. Limitations Honestly Stated (15% of Score)

1. **Revisit Time**: Sentinel-1/2 revisit periods are days apart; the system cannot provide real-time early warning minutes before an event.
2. **Topographic Blind Spots**: Radar layover and shadow on steep Himalayan faces cause localized blind spots and false positives.
3. **SAR Ambiguity**: Wet snow, saturated ground, and dark mountain shadows can mimic water specular reflection in C-band SAR.
4. **Heuristic Debris Detection**: Sediment and debris deposits are detected via backscatter ratio change, as no standardized deep-learning training sets exist for debris flows.
5. **Spatial Resolution**: 10-meter pixels cannot distinguish individual rural footbridges or single huts.
6. **OpenStreetMap Incompleteness**: Rural foot trails and remote roads in Nepal may be absent in OSM; absence in data does not guarantee absence on the ground.
7. **Hydraulic Dynamics**: The pipeline identifies surface inundation but does not estimate flood velocity or water depth.
8. **Educational Prototype**: This software is an emergency prototype and not an officially certified operational tool for life safety.

---

## 6. Team Roles & 15-Day Timeline

| Person | Focus Area |
| :--- | :--- |
| **Person A** | Data ingestion (CDSE STAC), S1/S2 preprocessing, HAND/slope masks, orbit-pair selection. |
| **Person B** | Model training (Kuro Siwo), hold-out evaluation, debris heuristic, S2 optical fusion. |
| **Person C** | OSM damage overlay, road graph cut-off analysis, flow-path bonus, EMSR927 comparison. |
| **Person D** | Dashboard, PDF/NE report generator, README, 6-page report, demo video, attributions. |

### 15-Day Milestone Roadmap
- **Days 1–2**: Accounts, repo skeleton, data download for Trishuli AOI, lock `results.json` schema.
- **Days 3–5**: Person A: S1 preprocessing working; Person B: baseline thresholding + Kuro Siwo loader; Person C: ohsome pull & road graph.
- **Days 6–8**: Person B: U-Net trained & hold-out metrics; Person C: damage & cut-off analysis; Person D: dashboard skeleton.
- **Days 9–10**: Full end-to-end integration, first full Trishuli run, first EMSR927 comparison.
- **Days 11–12**: Model tuning, debris refinement, S2 fusion, flow-path bonus, test on Chamoli 2021 & Melamchi 2021.
- **Days 13**: Feature freeze, English & Nepali report generation, speed & robustness optimizations.
- **Days 14**: 6-page report, README, demo video, attributions, citations.
- **Days 15**: Clean-machine dry run from raw data, Q&A rehearsal, final submission.

---

## 7. Required Attributions & Citations

- **Sentinel**: *"Contains modified Copernicus Sentinel data 2026."*
- **Copernicus DEM**: *"Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."*
- **OpenStreetMap**: *"© OpenStreetMap contributors (ODbL)."*
- **Copernicus EMS**: *"European Union, Copernicus Emergency Management Service data (EMSR927)."*
- **Datasets**:
  - Bountos et al., 2024 (*Kuro Siwo*)
  - Bonafilia et al., CVPR Workshops 2020 (*Sen1Floods11*)
