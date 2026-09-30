"""
Damage Assessment Module.
Intersects OpenStreetMap vector infrastructure (buildings, roads, bridges)
with buffered conservative and liberal hazard masks.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Point, LineString, Polygon
from shapely.ops import unary_union

logger = logging.getLogger("pipeline.damage")

def raster_to_geodataframe(mask: np.ndarray, bbox: list, crs="EPSG:4326") -> gpd.GeoDataFrame:
    """
    Convert a boolean 2D numpy mask to approximate vector polygons for spatial intersection.
    """
    H, W = mask.shape
    w, s, e, n = bbox
    dx = (e - w) / W
    dy = (n - s) / H

    # Identify connected components or bounding boxes of active hazard pixels
    polys = []
    # Grid downsampled search for efficiency
    step = 2
    for r in range(0, H, step):
        for c in range(0, W, step):
            if mask[r, c]:
                minx = w + c * dx
                maxx = minx + dx * step
                maxy = n - r * dy
                miny = maxy - dy * step
                polys.append(box(minx, miny, maxx, maxy))

    if not polys:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    unified = unary_union(polys)
    if unified.geom_type == "Polygon":
        geoms = [unified]
    elif unified.geom_type == "MultiPolygon":
        geoms = list(unified.geoms)
    else:
        geoms = [unified]

    return gpd.GeoDataFrame(geometry=geoms, crs=crs)

class DamagePipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dmg_cfg = config.get("damage", {})
        self.buffer_dist_m = self.dmg_cfg.get("buffer_distance_m", 15.0)

    def run(self, fused_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute geospatial damage overlay against OSM infrastructure.
        """
        logger.info("Executing Damage Assessment against OSM infrastructure...")
        bbox = fused_data["preprocessed"]["bbox"]
        osm = fused_data["preprocessed"]["osm"]

        gdf_bldgs = osm["buildings"].copy()
        gdf_roads = osm["roads"].copy()
        gdf_bridges = osm["bridges"].copy()
        gdf_dest = osm["destinations"].copy()

        # Vectorize hazard masks
        hazard_cons = fused_data["combined_hazard_conservative"]
        hazard_lib = fused_data["combined_hazard_liberal"]

        gdf_hazard_cons = raster_to_geodataframe(hazard_cons, bbox)
        gdf_hazard_lib = raster_to_geodataframe(hazard_lib, bbox)

        cons_union = unary_union(gdf_hazard_cons.geometry) if not gdf_hazard_cons.empty else None
        lib_union = unary_union(gdf_hazard_lib.geometry) if not gdf_hazard_lib.empty else None

        # Positional buffer: ~15m is approx 0.000135 degrees at 28 deg latitude
        deg_buffer = self.buffer_dist_m / 111320.0
        if cons_union and not cons_union.is_empty:
            cons_buffered = cons_union.buffer(deg_buffer)
        else:
            cons_buffered = None

        if lib_union and not lib_union.is_empty:
            lib_buffered = lib_union.buffer(deg_buffer)
        else:
            lib_buffered = None

        # 1. Buildings Damage Assessment
        bldg_status = []
        for geom in gdf_bldgs.geometry:
            if cons_buffered and geom.intersects(cons_buffered):
                bldg_status.append("likely_hit")
            elif lib_buffered and geom.intersects(lib_buffered):
                bldg_status.append("possibly_hit")
            else:
                bldg_status.append("not_hit")
        gdf_bldgs["damage_status"] = bldg_status

        total_bldgs = len(gdf_bldgs)
        likely_bldgs = sum(1 for s in bldg_status if s == "likely_hit")
        pos_bldgs = sum(1 for s in bldg_status if s == "possibly_hit")
        not_hit_bldgs = total_bldgs - likely_bldgs - pos_bldgs

        # 2. Roads Damage Assessment (Length in km)
        road_status = []
        road_length_km = []
        likely_km = 0.0
        pos_km = 0.0
        not_hit_km = 0.0

        for geom in gdf_roads.geometry:
            # Approximate length in km (1 deg ~ 111 km)
            length_km = geom.length * 111.0
            road_length_km.append(round(length_km, 3))

            if cons_buffered and geom.intersects(cons_buffered):
                road_status.append("likely_hit")
                likely_km += length_km
            elif lib_buffered and geom.intersects(lib_buffered):
                road_status.append("possibly_hit")
                pos_km += length_km
            else:
                road_status.append("not_hit")
                not_hit_km += length_km

        gdf_roads["damage_status"] = road_status
        gdf_roads["length_km"] = road_length_km
        total_road_km = likely_km + pos_km + not_hit_km

        # 3. Bridges Assessment (within or adjacent to corridor)
        bridge_status = []
        bridge_buffer_deg = 30.0 / 111320.0 # 30m corridor buffer
        corridor_union = lib_union.buffer(bridge_buffer_deg) if lib_union else None

        for geom in gdf_bridges.geometry:
            if cons_buffered and geom.intersects(cons_buffered):
                bridge_status.append("likely_hit")
            elif corridor_union and geom.intersects(corridor_union):
                bridge_status.append("adjacent_to_corridor")
            else:
                bridge_status.append("not_hit")
        gdf_bridges["damage_status"] = bridge_status

        total_bridges = len(gdf_bridges)
        likely_bridges = sum(1 for s in bridge_status if s == "likely_hit")
        adj_bridges = sum(1 for s in bridge_status if s == "adjacent_to_corridor")
        not_hit_bridges = total_bridges - likely_bridges - adj_bridges

        # 4. Critical Facilities (Hospitals)
        hospitals_at_risk = []
        for idx, row in gdf_dest[gdf_dest["type"] == "hospital"].iterrows():
            if (cons_buffered and row.geometry.intersects(cons_buffered)) or \
               (lib_buffered and row.geometry.intersects(lib_buffered)):
                hospitals_at_risk.append(row["name"])

        damage_summary = {
            "buildings": {
                "total_assessed": int(total_bldgs),
                "likely_hit": int(likely_bldgs),
                "possibly_hit": int(pos_bldgs),
                "not_hit": int(not_hit_bldgs),
                "likely_hit_percent": round((likely_bldgs / total_bldgs * 100), 1) if total_bldgs > 0 else 0.0
            },
            "roads_km": {
                "total_length_km": round(float(total_road_km), 2),
                "likely_hit_km": round(float(likely_km), 2),
                "possibly_hit_km": round(float(pos_km), 2),
                "not_hit_km": round(float(not_hit_km), 2),
                "likely_hit_percent": round((likely_km / total_road_km * 100), 1) if total_road_km > 0 else 0.0
            },
            "bridges": {
                "total_assessed": int(total_bridges),
                "likely_hit": int(likely_bridges),
                "adjacent_to_corridor": int(adj_bridges),
                "not_hit": int(not_hit_bridges)
            },
            "critical_facilities": {
                "hospitals_at_risk": hospitals_at_risk,
                "schools_at_risk": []
            }
        }

        logger.info(
            f"Damage Summary: Buildings={likely_bldgs}/{total_bldgs} hit | "
            f"Roads={likely_km:.1f}km/{total_road_km:.1f}km damaged | "
            f"Bridges={likely_bridges} hit, {adj_bridges} at risk"
        )

        return {
            **fused_data,
            "damage_summary": damage_summary,
            "gdf_buildings_assessed": gdf_bldgs,
            "gdf_roads_assessed": gdf_roads,
            "gdf_bridges_assessed": gdf_bridges,
            "cons_buffered": cons_buffered,
            "lib_buffered": lib_buffered
        }
