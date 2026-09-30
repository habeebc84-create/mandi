"""
Detection module for flood and debris hazards.
Runs multimodal U-Net inference for water/flood segmentation,
and applies an adaptive change-detection heuristic for mountain debris flows.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import torch
from model.unet import load_flood_model

logger = logging.getLogger("pipeline.detect")

def otsu_threshold(image: np.ndarray) -> float:
    """
    Otsu thresholding for separating debris change signals in SAR backscatter.
    """
    vals = image[~np.isnan(image)]
    if len(vals) == 0:
        return 0.0
    hist, bin_edges = np.histogram(vals, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = len(vals)
    current_max, threshold = 0.0, bin_centers[25]
    weight_bg, sum_bg = 0.0, 0.0
    sum_total = np.dot(bin_centers, hist)

    for i in range(len(hist)):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += bin_centers[i] * hist[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between_class_var = weight_bg * weight_fg * ((mean_bg - mean_fg) ** 2)
        if between_class_var > current_max:
            current_max = between_class_var
            threshold = bin_centers[i]
    return threshold

class DetectPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detect_cfg = config.get("detection", {})
        self.weights_path = self.detect_cfg.get("model_weights", "model/weights/kuro_siwo_unet.pt")
        self.thresholds = self.detect_cfg.get("thresholds", {"conservative": 0.70, "liberal": 0.35})
        self.device = "cuda" if torch.cuda.is_available() and self.detect_cfg.get("device") != "cpu" else "cpu"
        self.model = load_flood_model(self.weights_path, device=self.device)

    def run(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute detection pipeline:
        1. Deep Learning U-Net for permanent water and flood water
        2. Log-ratio backscatter change detection for debris flow deposits
        3. Multi-threshold sensitivity output
        """
        logger.info("Executing AI detection and debris change analysis...")
        features = preprocessed_data["features"] # Shape (7, H, W)
        dem = preprocessed_data["dem"]
        hand_m = preprocessed_data["hand_m"]
        layover_mask = preprocessed_data["layover_shadow_mask"]
        diff_vv = preprocessed_data["diff_vv"]
        H, W = dem.shape

        # 1. U-Net Inference
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(feat_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        prob_background = probs[0]
        prob_perm_water = probs[1]
        prob_flood_water = probs[2]

        # Physics-grounded SAR backscatter & HAND baseline integration
        flood_signal = (features[2] < -19.5) & (hand_m < 20.0) & (~layover_mask)
        perm_signal = (features[0] < -21.0) & (features[2] < -21.0) & (hand_m < 8.0)

        # Blend model logits with SAR physical evidence
        prob_flood_water = np.maximum(prob_flood_water, np.where(flood_signal, 0.88, 0.05).astype(np.float32))
        prob_perm_water = np.maximum(prob_perm_water, np.where(perm_signal, 0.95, 0.02).astype(np.float32))

        # Suppress radar layover/shadow false positives in steep slopes
        prob_flood_water[layover_mask] = 0.01

        # 2. Debris Flow Detection (Heuristic change detection in river corridor)
        # Log ratio of backscatter change inside HAND < 25m corridor
        corridor_mask = (hand_m < 25.0) & (prob_flood_water < 0.4) & (~layover_mask)
        debris_change = diff_vv[corridor_mask]
        
        adaptive_thresh = otsu_threshold(np.abs(debris_change)) if len(debris_change) > 0 else 3.5
        logger.info(f"Debris heuristic Otsu threshold: {adaptive_thresh:.2f} dB")

        # Debris shows up as significant structural change (roughening or scouring)
        debris_prob = np.zeros((H, W), dtype=np.float32)
        debris_raw = (np.abs(diff_vv) > adaptive_thresh) & corridor_mask & (hand_m > 4.0)
        debris_prob[debris_raw] = 0.78

        # 3. Create Multi-Threshold Hazard Masks
        cons_thresh = self.thresholds.get("conservative", 0.70)
        lib_thresh = self.thresholds.get("liberal", 0.35)

        flood_mask_conservative = prob_flood_water >= cons_thresh
        flood_mask_liberal = prob_flood_water >= lib_thresh

        debris_mask_conservative = debris_prob >= 0.70
        debris_mask_liberal = debris_prob >= 0.40

        combined_hazard_conservative = flood_mask_conservative | debris_mask_conservative
        combined_hazard_liberal = flood_mask_liberal | debris_mask_liberal

        # Estimate surface area in sq km (assuming pixel size)
        pixel_size = self.config.get("preprocessing", {}).get("pixel_size_m", 10.0)
        pixel_area_sqkm = (pixel_size * pixel_size) / 1e6

        stats = {
            "permanent_water_sqkm": float(np.sum(prob_perm_water >= 0.5) * pixel_area_sqkm),
            "flood_water_sqkm_conservative": float(np.sum(flood_mask_conservative) * pixel_area_sqkm),
            "flood_water_sqkm_liberal": float(np.sum(flood_mask_liberal) * pixel_area_sqkm),
            "debris_sqkm_conservative": float(np.sum(debris_mask_conservative) * pixel_area_sqkm),
            "debris_sqkm_liberal": float(np.sum(debris_mask_liberal) * pixel_area_sqkm),
            "total_hazard_sqkm_conservative": float(np.sum(combined_hazard_conservative) * pixel_area_sqkm),
            "total_hazard_sqkm_liberal": float(np.sum(combined_hazard_liberal) * pixel_area_sqkm),
        }

        logger.info(f"Hazard Areas: Flood={stats['flood_water_sqkm_conservative']:.2f} sq km (cons) / {stats['flood_water_sqkm_liberal']:.2f} sq km (lib); Debris={stats['debris_sqkm_conservative']:.2f} sq km")

        return {
            "prob_perm_water": prob_perm_water,
            "prob_flood_water": prob_flood_water,
            "prob_debris": debris_prob,
            "flood_mask_conservative": flood_mask_conservative,
            "flood_mask_liberal": flood_mask_liberal,
            "debris_mask_conservative": debris_mask_conservative,
            "debris_mask_liberal": debris_mask_liberal,
            "combined_hazard_conservative": combined_hazard_conservative,
            "combined_hazard_liberal": combined_hazard_liberal,
            "stats": stats,
            "preprocessed": preprocessed_data
        }
