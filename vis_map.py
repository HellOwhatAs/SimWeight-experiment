import folium
import pickle
import pandas as pd
from typing import List, Tuple, Dict, Iterable
from tqdm import tqdm

def base_edge_map(
        nodes: pd.DataFrame, edges: pd.DataFrame,
        color: str = "#BEBEBE", weight: int = 1, zoom_start: int = 10, tiles: str = '', attr: str = '') -> folium.Map:
    edge_locations: List[List[Tuple[float, float]]] = []
    nodes_loc = nodes[["y", "x"]].to_numpy().tolist()
    for _, edge in tqdm(edges.iterrows(), dynamic_ncols=True):
        u, v = edge['u'], edge['v']
        edge_locations.append([
            (nodes_loc[u][0], nodes_loc[u][1]),
            (nodes_loc[v][0], nodes_loc[v][1])
        ])
    center = list(sum(sum(j[k] for j in i) / len(i) for i in edge_locations) / len(edge_locations) for k in range(2))
    m = folium.Map(
        center,
        tiles=tiles,
        attr=attr,
        zoom_start=zoom_start,
        prefer_canvas=True
    )
    folium.PolyLine(
        locations=edge_locations,
        color=color,
        weight=weight
    ).add_to(m)
    return m

def add_trips(m: folium.Map, nodes: pd.DataFrame, edges: pd.DataFrame, trips: List[List[int]], color: str = "#FF0000", weight: int = 3):
    trip_locations: List[List[Tuple[float, float]]] = []
    for trip in tqdm(trips, dynamic_ncols=True):
        trip_loc = []
        for idx, edge_id in enumerate(trip):
            edge = edges.loc[edge_id]
            u, v = edge['u'], edge['v']
            if idx == 0: trip_loc.append((nodes.loc[u]['y'], nodes.loc[u]['x']))
            trip_loc.append((nodes.loc[v]['y'], nodes.loc[v]['x']))
        trip_locations.append(trip_loc)
    if not trip_locations: return
    folium.PolyLine(
        locations=trip_locations,
        color=color,
        weight=weight
    ).add_to(m)

def add_nodes(m: folium.Map, nodes: pd.DataFrame, node_ids: Dict[int, str]):
    for node_id in node_ids:
        folium.Circle(
            (nodes.loc[node_id]['y'], nodes.loc[node_id]['x']),
            popup = node_ids[node_id],

        ).add_to(m)

def add_edges(m: folium.Map, nodes: pd.DataFrame, edges: pd.DataFrame, edge_ids: Iterable[int], color: str = "#FF0000", weight: int = 3):
    edge_locations: List[List[Tuple[float, float]]] = []
    for edge_id in edge_ids:
        edge = edges.loc[edge_id]
        u, v = edge['u'], edge['v']
        edge_locations.append([(nodes.loc[u]['y'], nodes.loc[u]['x']), (nodes.loc[v]['y'], nodes.loc[v]['x'])])
    if not edge_locations: return
    
    folium.PolyLine(
        locations=edge_locations,
        color=color,
        weight=weight
    ).add_to(m)