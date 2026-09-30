# System Architecture & Technical Specifications
## Flood Damage Mapping from Space (15-Day Rapid Disaster Pipeline)

```
run.py (--bbox W,S,E,N --date YYYY-MM-DD)
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

## 1. Disqualification Quarantine Wall
Copernicus EMS (EMSR927), UNOSAT, and post-event OSM edits reside strictly inside `eval/emsr927_compare.py`. 
No module inside `pipeline/` imports or references `eval/`. This preserves competitive integrity and avoids disqualification.

---

## 2. Ingestion & Preprocessing
* **Copernicus Data Space Ecosystem (CDSE)**:
  * STAC/OData API queries automatically select matching relative orbits (same-track) with ~12-day baseline repeat.
  * Same-track geometry guarantees identical radar incidence angles, preventing steep mountain faces from creating artificial backscatter shifts.
* **Himalayan Terrain Corrections**:
  * Slope mask computed from 30m Copernicus GLO-30 DEM.
  * Height Above Nearest Drainage (HAND) separates low-lying floodplain inundation from steep ridge radar shadows.
  * Multi-look Lee speckle filter suppresses radar speckle while preserving sharp valley boundaries.

---

## 3. Multimodal AI Segmentation & Debris Detection
* **7-Channel Input Tensor**:
  1. Sentinel-1 Pre-event VV (dB)
  2. Sentinel-1 Pre-event VH (dB)
  3. Sentinel-1 Post-event VV (dB)
  4. Sentinel-1 Post-event VH (dB)
  5. VV Difference (Post - Pre dB)
  6. Normalized Terrain Slope ($[0, 1]$)
  7. Normalized HAND ($[0, 1]$)
* **Classes**:
  * 0: Background
  * 1: Permanent Water Channel
  * 2: Flood Inundation
* **Debris Scars**:
  * Heuristic change-detection inside active river corridors ($HAND < 25\text{m}$) using adaptive Otsu log-ratio thresholding.

---

## 4. Optical Fusion (Sentinel-2)
* Secondary confirmation applied **only** to cloud-free pixels identified by Scene Classification Layer (SCL classes 4, 5, 6).
* Monsoon cloud cover (>70%) automatically triggers graceful fallback to pure Sentinel-1 SAR.

---

## 5. Cut-Off Settlement Analysis (20% Judging Weight)
* **Road Graph Routing**: Constructed via NetworkX with edge weights based on real road lengths.
* **Destinations**: Pre-event hospitals, clinics, and municipal town centers.
* **False Positive Prevention**:
  * A settlement is classified as **Cut Off** if and only if it was reachable pre-event and is disconnected post-event.
  * Settlements with incomplete OSM connections pre-event are labeled **Unknown**, never falsely claimed as disaster cut-offs.
  * Detour impact is reported as increased travel distance (km and percentage) for settlements retaining connectivity.

---

## 6. Dual-Language Reports & Numeric Verification
* Every statistic rendered in English (`app/templates/en/`) and Nepali (`app/templates/ne/`) is bound strictly to `results.json`.
* An automated regex consistency checker ensures no hallucinated numbers can enter LLM-assisted summaries.
