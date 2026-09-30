"""
Bonus Module: DEM Flow-Path Tracing.
Calculates D8 flow direction across GLO-30 DEM, traces downstream path
from an upstream origin point (e.g. glacial lake, dam, landslide dammed lake),
buffers along the HAND drainage corridor, and lists threatened settlements.
"""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
from shapely.geometry import Point, LineString

logger = logging.getLogger("pipeline.flowpath")

# D8 Neighbor Offsets: (row_offset, col_offset)
# Directions: 0:East, 1:SE, 2:South, 3:SW, 4:West, 5:NW, 6:North, 7:NE
D8_OFFSETS = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1)
]

def trace_d8_downstream(dem: np.ndarray, start_row: int, start_col: int, max_steps: int = 400) -> List[Tuple[int, int]]:
    """
    Trace downhill steepest descent path across the DEM.
    """
    H, W = dem.shape
    path = [(start_row, start_col)]
    curr_r, curr_c = start_row, start_col
    visited = set(path)

    for _ in range(max_steps):
        best_r, best_c = curr_r, curr_c
        best_elev = dem[curr_r, curr_c]

        for dr, dc in D8_OFFSETS:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if (nr, nc) not in visited and dem[nr, nc] < best_elev:
                    best_elev = dem[nr, nc]
                    best_r, best_c = nr, nc

        if (best_r, best_c) == (curr_r, curr_c):
            # Reached local depression or valley bottom
            break

        path.append((best_r, best_c))
        visited.add((best_r, best_c))
        curr_r, curr_c = best_r, best_c

    return path

class FlowpathPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fp_cfg = config.get("flowpath", {})

    def run(self, cutoff_data: Dict[str, Any], source_point: Tuple[float, float] = None) -> Dict[str, Any]:
        """
        Execute flow-path trace downstream from source point.
        """
        dem = cutoff_data["preprocessed"]["dem"]
        bbox = cutoff_data["preprocessed"]["bbox"]
        osm = cutoff_data["preprocessed"]["osm"]
        H, W = dem.shape
        w, s, e, n = bbox

        if source_point is None:
            source_point = self.fp_cfg.get("default_source_point", [28.08, 85.22])

        logger.info(f"Tracing downstream flood flow path from origin: {source_point}...")

        # Convert lat, lon to row, col
        src_lat, src_lon = source_point
        start_col = int(np.clip(((src_lon - w) / (e - w)) * W, 0, W - 1))
        start_row = int(np.clip(((n - src_lat) / (n - s)) * H, 0, H - 1))

        # Trace D8 path
        path_pixels = trace_d8_downstream(dem, start_row, start_col)

        # Convert back to lon, lat coordinates
        dx = (e - w) / W
        dy = (n - s) / H
        path_coords = [(w + c * dx, n - r * dy) for r, c in path_pixels]

        if len(path_coords) >= 2:
            flow_line = LineString(path_coords)
            path_len_km = round(float(flow_line.length * 111.0), 2)
            # Buffer by ~500m (approx 0.0045 deg) along the river valley
            corridor_poly = flow_line.buffer(0.0045)
        else:
            flow_line = Point(src_lon, src_lat)
            path_len_km = 0.0
            corridor_poly = flow_line.buffer(0.0045)

        # Find settlements within path corridor
        settlements_in_path = []
        for idx, row in osm["settlements"].iterrows():
            if row.geometry.intersects(corridor_poly):
                settlements_in_path.append(row["name"])

        for idx, row in osm["destinations"].iterrows():
            if row.geometry.intersects(corridor_poly):
                settlements_in_path.append(f"{row['name']} ({row['type']})")

        logger.info(f"Flow path trace length: {path_len_km} km. Settlements in downstream corridor: {settlements_in_path}")

        flowpath_bonus = {
            "source_point": [float(src_lat), float(src_lon)],
            "corridor_length_km": float(path_len_km),
            "settlements_in_path": settlements_in_path
        }

        return {
            **cutoff_data,
            "flowpath_bonus": flowpath_bonus,
            "flowpath_line": flow_line
        }
