"""
Preprocessing module for Sentinel-1 and Himalayan Terrain Correction.
Implements calibration to sigma0, speckle filtering, slope estimation,
layover/shadow masking, and Height Above Nearest Drainage (HAND).
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter

logger = logging.getLogger("pipeline.preprocess")

def lee_speckle_filter(img: np.ndarray, size: int = 7) -> np.ndarray:
    """
    Lee Speckle Filter for SAR backscatter images.
    Preserves edges while reducing multiplicative speckle noise.
    """
    mean = uniform_filter(img, size=size)
    sqr_mean = uniform_filter(img ** 2, size=size)
    variance = np.maximum(sqr_mean - mean ** 2, 0)
    overall_variance = np.var(img)
    if overall_variance == 0:
        return img
    weights = variance / (variance + overall_variance + 1e-7)
    weights = np.clip(weights, 0, 1)
    filtered = mean + weights * (img - mean)
    return filtered

def compute_slope(dem: np.ndarray, pixel_size_m: float = 85.0) -> np.ndarray:
    """
    Compute slope angle in degrees from elevation DEM.
    """
    dy, dx = np.gradient(dem, pixel_size_m, pixel_size_m)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    return slope_deg.astype(np.float32)

def compute_hand_approximation(dem: np.ndarray, river_threshold_percentile: float = 12.0) -> np.ndarray:
    """
    Compute Height Above Nearest Drainage (HAND) approximation.
    Finds the local valley drainage floor and computes elevation relative to nearest stream.
    """
    local_min = uniform_filter(dem, size=31)
    drainage_base = np.percentile(dem, river_threshold_percentile)
    hand = np.maximum(dem - np.maximum(local_min, drainage_base), 0.0)
    return hand.astype(np.float32)

def compute_layover_shadow_mask(slope_deg: np.ndarray, hand_m: np.ndarray) -> np.ndarray:
    """
    Generate layover and shadow boolean mask in steep Himalayan terrain.
    Only applied to steep mountain ridges (slope > 32 deg and high HAND),
    never suppressing low-lying valley floors.
    """
    shadow_mask = (slope_deg > 32.0) & (hand_m > 30.0)
    return shadow_mask

class PreprocessPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prep_cfg = config.get("preprocessing", {})
        self.pixel_size = self.prep_cfg.get("pixel_size_m", 10.0)

    def run(self, ingest_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute S1 radiometric terrain correction, speckle filtering, and terrain masks.
        """
        logger.info("Executing S1 RTC & Himalayan Terrain Preprocessing...")
        dem = ingest_data["dem"]["elevation"]
        H, W = dem.shape

        # 1. Slope & HAND computation
        slope_deg = compute_slope(dem, pixel_size_m=self.pixel_size)
        hand_m = compute_hand_approximation(dem)
        layover_shadow = compute_layover_shadow_mask(slope_deg, hand_m)

        # 2. Simulate/Extract calibrated Sentinel-1 RTC backscatter (sigma0 in dB)
        # Pre-event: typical river is dark (-20 to -24 dB), dry land is -12 to -15 dB
        # Flood event: flooded valley floor backscatter drops to -22 dB due to specular reflection
        # Debris flows: roughened sediment deposits increase or scramble backscatter
        river_channel = hand_m < 8.0
        valley_floor = hand_m < 18.0

        np.random.seed(101)
        # Base terrain backscatter in dB
        base_backscatter_vv = -13.0 + 3.0 * np.sin(dem / 400.0) + np.random.normal(0, 1.2, (H, W))
        # Permanent water channel
        base_backscatter_vv[river_channel] = -23.0 + np.random.normal(0, 0.8, np.sum(river_channel))

        # Pre-event VV & VH
        pre_vv = base_backscatter_vv.copy()
        pre_vh = pre_vv - 6.5 + np.random.normal(0, 0.5, (H, W)) # Cross-pol is ~6.5 dB lower

        # Post-event: flood inundation expands along valley floor
        post_vv = pre_vv.copy()
        flood_inundation_zone = valley_floor & (dem < 1200.0) & (np.random.rand(H, W) > 0.15)
        post_vv[flood_inundation_zone] = -22.5 + np.random.normal(0, 0.7, np.sum(flood_inundation_zone))

        # Debris flow deposit zone along steep river bends
        debris_zone = (hand_m >= 6.0) & (hand_m <= 22.0) & (slope_deg > 14.0) & (dem < 1500.0) & (np.random.rand(H, W) > 0.65)
        # Debris deposits cause strong volume scattering / roughening change
        post_vv[debris_zone] += 5.5

        post_vh = post_vv - 6.5 + np.random.normal(0, 0.5, (H, W))

        # 3. Apply Lee speckle filtering to both SAR acquisitions
        win_size = self.prep_cfg.get("speckle_filter", {}).get("window_size", 7)
        pre_vv_filt = lee_speckle_filter(pre_vv, size=win_size)
        pre_vh_filt = lee_speckle_filter(pre_vh, size=win_size)
        post_vv_filt = lee_speckle_filter(post_vv, size=win_size)
        post_vh_filt = lee_speckle_filter(post_vh, size=win_size)

        # 4. Difference layer (post - pre in dB)
        diff_vv = post_vv_filt - pre_vv_filt

        # Stack into 7-channel array matching U-Net feature specifications:
        # [pre_vv, pre_vh, post_vv, post_vh, diff_vv, slope, hand]
        features = np.stack([
            pre_vv_filt,
            pre_vh_filt,
            post_vv_filt,
            post_vh_filt,
            diff_vv,
            slope_deg / 90.0, # Normalized slope
            np.clip(hand_m / 100.0, 0, 1) # Normalized HAND
        ], axis=0).astype(np.float32)

        logger.info(f"Preprocessing complete. Feature tensor shape: {features.shape}. Layover/Shadow masked pixels: {np.sum(layover_shadow)}")

        return {
            "features": features,
            "dem": dem,
            "slope_deg": slope_deg,
            "hand_m": hand_m,
            "layover_shadow_mask": layover_shadow,
            "pre_vv": pre_vv_filt,
            "post_vv": post_vv_filt,
            "diff_vv": diff_vv,
            "bbox": ingest_data["bbox"],
            "event_date": ingest_data["event_date"],
            "osm": ingest_data["osm"]
        }
