import os
from typing import List, Tuple, Dict, Literal
import geopandas as gpd
import pandas as pd
import pickle

Trips = Dict[Literal["train", "test", "valid"], Dict[Tuple[int, int], List[List[int]]]]
Result = Tuple[pd.DataFrame, pd.DataFrame, Trips]
"""
Return type of `extract_data.extract`
`Result = (nodes, edges, {litstr: {(u, v): trips}})`
"""

def remove_loops(path):
    reduced = []
    last_occ = {p:-1 for p in path}
    for i in range(len(path)-1,-1,-1):
        if last_occ[path[i]] == -1:
            last_occ[path[i]] = i
    current = 0
    while(current < len(path)):
        reduced.append(path[current])
        current = last_occ[path[current]] + 1
    return reduced

def groupy_uv(trips: List[List[int]], edges: List[Tuple[int, int]]) -> Dict[Tuple[int, int], List[List[int]]]:
    uv2trips: Dict[Tuple[int, int], List[List[int]]] = {}
    for trip in trips:
        key = (edges[trip[0]][0], edges[trip[-1]][1])
        if key[0] == key[1]: continue
        if key in uv2trips: uv2trips[key].append(trip)
        else: uv2trips[key] = [trip]
    return uv2trips

def extract(path: str = "./preprocessed_data/beijing_data", removeloops: bool = True) -> Result:
    edge_df: pd.DataFrame = gpd.read_file(os.path.join(path, "map/edges.shp"), ignore_geometry=True)
    node_df: pd.DataFrame = gpd.read_file(os.path.join(path, "map/nodes.shp"), ignore_geometry=True)

    map_edge_id_to_u_v = edge_df[['u', 'v']].to_numpy()
    map_u_v_to_edge_id = {(u, v): i for i, (u, v) in enumerate(map_edge_id_to_u_v)}

    map_node_osmid_to_id = {j: i for i, j in enumerate(node_df[["osmid"]].to_numpy().flatten())}
    edge_df['u'] = edge_df['u'].map(map_node_osmid_to_id)
    edge_df['v'] = edge_df['v'].map(map_node_osmid_to_id)

    with open(os.path.join(path, "preprocessed_test_trips_all.pkl"), "rb") as f:
        data_test = pickle.load(f)
    with open(os.path.join(path, "preprocessed_train_trips_all.pkl"), "rb") as f:
        data_train = pickle.load(f)
    with open(os.path.join(path, "preprocessed_validation_trips_all.pkl"), "rb") as f:
        data_valid = pickle.load(f)
    data_test = [(idx, [map_u_v_to_edge_id[tuple(map_edge_id_to_u_v[e])] for e in t]) for (idx, t, _) in data_test]
    data_train = [(idx, [map_u_v_to_edge_id[tuple(map_edge_id_to_u_v[e])] for e in t]) for (idx, t, _) in data_train]
    data_valid = [(idx, [map_u_v_to_edge_id[tuple(map_edge_id_to_u_v[e])] for e in t]) for (idx, t, _) in data_valid]
    
    if removeloops:
        data_test = [(idx, remove_loops(t)) for (idx,t) in data_test]
        data_train = [(idx, remove_loops(t)) for (idx,t) in data_train]
        data_valid = [(idx, remove_loops(t)) for (idx,t) in data_valid]
        
    data_test = [t for (_, t) in data_test if len(t) >= 5]
    data_train = [t for (_, t) in data_train if len(t) >= 5]
    data_valid = [t for (_, t) in data_valid if len(t) >= 5]

    edges_list = edge_df[["u", "v"]].to_numpy().tolist()
    data_test = groupy_uv(data_test, edges_list)
    data_train = groupy_uv(data_train, edges_list)
    data_valid = groupy_uv(data_valid, edges_list)

    return node_df, edge_df, {"train": data_train, "test": data_test, "valid": data_valid}


if __name__ == "__main__":
    # output files avaliable at https://www.kaggle.com/code/xjq701229/simweight-data-source
    with open("beijing.pkl", "wb") as f:
        pickle.dump(extract("./preprocessed_data/beijing_data/"), f)
    with open("chengdu.pkl", "wb") as f:
        pickle.dump(extract("./preprocessed_data/chengdu_data/"), f)
    with open("cityindia.pkl", "wb") as f:
        pickle.dump(extract("./preprocessed_data/cityindia_data/"), f)
    with open("harbin.pkl", "wb") as f:
        pickle.dump(extract("./preprocessed_data/harbin_data/"), f)
    with open("porto.pkl", "wb") as f:
        pickle.dump(extract("./preprocessed_data/porto_data/"), f)