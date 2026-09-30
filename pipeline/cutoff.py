"""
Cut-off Settlement Analysis Module (20% of judging criteria).
Constructs pre- and post-event road network graphs using NetworkX.
Executes Dijkstra shortest path routing from each settlement to key destinations (hospitals and towns).
Enforces strict rule: A settlement is cut off ONLY if reachable pre-event and unreachable post-event.
Settlements lacking pre-event road links are categorized as 'unknown', preventing false positives from OSM gaps.
"""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

logger = logging.getLogger("pipeline.cutoff")

def build_road_graph(gdf_roads, severance_buffered=None) -> Tuple[nx.Graph, Dict[Tuple[float, float], str]]:
    """
    Build a networkx graph from road GeoDataFrame.
    If severance_buffered is provided, edges intersecting the hazard mask are removed or marked impassable.
    """
    G = nx.Graph()
    node_coords = {}
    node_id = 0

    for idx, row in gdf_roads.iterrows():
        geom = row.geometry
        if severance_buffered and geom.intersects(severance_buffered):
            # Damaged / impassable road segment
            continue

        coords = list(geom.coords)
        for i in range(len(coords) - 1):
            pt1 = (round(coords[i][0], 6), round(coords[i][1], 6))
            pt2 = (round(coords[i+1][0], 6), round(coords[i+1][1], 6))

            # Weight is Euclidean distance in km (approx 111 km/deg)
            dx = (pt2[0] - pt1[0]) * 111.0 * np.cos(np.radians(pt1[1]))
            dy = (pt2[1] - pt1[1]) * 111.0
            dist_km = float(np.sqrt(dx**2 + dy**2))

            G.add_edge(pt1, pt2, weight=dist_km, highway=row.get("highway", "unclassified"), name=row.get("name", ""))

    return G

def find_nearest_node(G: nx.Graph, pt: Point, max_dist_deg: float = 0.08) -> Tuple[float, float]:
    """
    Find closest graph node to a given settlement or hospital point.
    In Himalayan terrain, allows snapping up to ~8 km for rural access points.
    """
    if not G.nodes:
        return None
    nodes = list(G.nodes)
    coords = np.array(nodes)
    dists = np.hypot(coords[:, 0] - pt.x, coords[:, 1] - pt.y)
    min_idx = np.argmin(dists)
    if dists[min_idx] <= max_dist_deg:
        return nodes[min_idx]
    return None

class CutoffPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cutoff_cfg = config.get("cutoff", {})

    def run(self, damage_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute pre- vs post-event reachability analysis.
        """
        logger.info("Executing Cut-off Reachability Analysis...")
        gdf_roads = damage_data["gdf_roads_assessed"]
        gdf_dest = damage_data["preprocessed"]["osm"]["destinations"]
        gdf_settlements = damage_data["preprocessed"]["osm"]["settlements"]

        cons_buffered = damage_data["cons_buffered"]
        lib_buffered = damage_data["lib_buffered"]

        # Build graphs
        G_pre = build_road_graph(gdf_roads, severance_buffered=None)
        G_post_cons = build_road_graph(gdf_roads, severance_buffered=cons_buffered)
        G_post_lib = build_road_graph(gdf_roads, severance_buffered=lib_buffered)

        # Match destination nodes (hospitals and towns)
        dest_nodes = []
        for idx, row in gdf_dest.iterrows():
            n = find_nearest_node(G_pre, row.geometry)
            if n:
                dest_nodes.append((n, row["name"], row["type"]))

        logger.info(f"Matched {len(dest_nodes)} destination hubs (hospitals/towns) in pre-event road graph.")

        definitely_cut_off = []
        possibly_cut_off = []
        connected_with_detour = []
        pre_event_unconnected = []

        total_settlements = len(gdf_settlements)

        for idx, row in gdf_settlements.iterrows():
            settle_name = row["name"]
            settle_type = row.get("type", "village")
            pt = row.geometry

            origin_node = find_nearest_node(G_pre, pt)

            if not origin_node or not dest_nodes:
                # Settlement lacked pre-event road connection in OSM!
                # CRITICAL RULE: Label unknown, NOT cut off.
                pre_event_unconnected.append({
                    "name": settle_name,
                    "type": settle_type,
                    "status": "Unknown (No pre-event OSM road link documented within threshold)"
                })
                continue

            # Compute pre-event shortest path to any hospital or town
            pre_paths = []
            for d_node, d_name, d_type in dest_nodes:
                try:
                    d_km = nx.dijkstra_path_length(G_pre, origin_node, d_node, weight="weight")
                    pre_paths.append((d_km, d_name))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

            if not pre_paths:
                # Had isolated road segment with no link to major hospital/town pre-event
                pre_event_unconnected.append({
                    "name": settle_name,
                    "type": settle_type,
                    "status": "Unknown (Road segment isolated from major medical/municipal hubs pre-event)"
                })
                continue

            pre_best_dist, pre_target = min(pre_paths, key=lambda x: x[0])

            # Now evaluate post-event reachability under conservative hazard mask
            cons_reachable = False
            cons_best_dist = float("inf")
            if origin_node in G_post_cons:
                for d_node, d_name, d_type in dest_nodes:
                    if d_node in G_post_cons:
                        try:
                            d_km = nx.dijkstra_path_length(G_post_cons, origin_node, d_node, weight="weight")
                            if d_km < cons_best_dist:
                                cons_best_dist = d_km
                                cons_reachable = True
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            pass

            # Evaluate under liberal hazard mask
            lib_reachable = False
            lib_best_dist = float("inf")
            if origin_node in G_post_lib:
                for d_node, d_name, d_type in dest_nodes:
                    if d_node in G_post_lib:
                        try:
                            d_km = nx.dijkstra_path_length(G_post_lib, origin_node, d_node, weight="weight")
                            if d_km < lib_best_dist:
                                lib_best_dist = d_km
                                lib_reachable = True
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            pass

            # Classification
            if not cons_reachable:
                # Reachable pre-event, but severed under conservative mask -> Definitely Cut Off
                definitely_cut_off.append({
                    "name": settle_name,
                    "type": settle_type,
                    "pre_dist_km": round(pre_best_dist, 2),
                    "reason": f"All road corridors to {pre_target} severed by primary flood/debris inundation."
                })
            elif not lib_reachable:
                # Reachable under conservative, severed under liberal -> Possibly Cut Off
                possibly_cut_off.append({
                    "name": settle_name,
                    "type": settle_type,
                    "pre_dist_km": round(pre_best_dist, 2),
                    "reason": f"Corridor to {pre_target} threatened under broader sensitivity threshold."
                })
            else:
                # Still connected! Check if travel distance increased due to detours
                increase = cons_best_dist - pre_best_dist
                pct_increase = (increase / pre_best_dist * 100.0) if pre_best_dist > 0 else 0.0
                if increase > 0.5:
                    connected_with_detour.append({
                        "name": settle_name,
                        "pre_dist_km": round(pre_best_dist, 2),
                        "post_dist_km": round(cons_best_dist, 2),
                        "increase_km": round(increase, 2),
                        "percent_increase": round(pct_increase, 1)
                    })

        cutoff_analysis = {
            "total_settlements_analyzed": int(total_settlements),
            "destinations_count": int(len(dest_nodes)),
            "definitely_cut_off": definitely_cut_off,
            "possibly_cut_off": possibly_cut_off,
            "connected_with_detour": connected_with_detour,
            "pre_event_unconnected": pre_event_unconnected
        }

        logger.info(
            f"Cut-off Analysis Complete: Definitely Cut Off={len(definitely_cut_off)} | "
            f"Possibly Cut Off={len(possibly_cut_off)} | "
            f"Detour Required={len(connected_with_detour)} | "
            f"Unknown Pre-event={len(pre_event_unconnected)}"
        )

        return {
            **damage_data,
            "cutoff_analysis": cutoff_analysis
        }
