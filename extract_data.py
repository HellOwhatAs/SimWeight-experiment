import os
from typing import List, Tuple, Dict
from typing_extensions import Literal
import geopandas as gpd
import pandas as pd
import pickle
from multiprocessing import Pool

Trips = Dict[Literal["train", "test", "valid"], Dict[Tuple[int, int], List[Tuple[List[int], Tuple[int, int]]]]]
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

def groupby_uv(trips: List[Tuple[List[int], Tuple[int, int]]], edges: List[Tuple[int, int]]) -> Dict[Tuple[int, int], List[Tuple[List[int], Tuple[int, int]]]]:
    uv2trips: Dict[Tuple[int, int], List[Tuple[List[int], Tuple[int, int]]]] = {}
    for trip, time in trips:
        key = (edges[trip[0]][0], edges[trip[-1]][1])
        if key[0] == key[1]: continue
        if key in uv2trips: uv2trips[key].append((trip, time))
        else: uv2trips[key] = [(trip, time)]
    return uv2trips

def groupby_io(trips: List[List[int]], edges: List[Tuple[int, int]]) -> Dict[Tuple[int, int], List[List[int]]]:
    io2trips: Dict[Tuple[int, int], List[List[int]]] = {}
    for trip in trips:
        key = (edges[trip[0]][0], edges[trip[-1]][1])
        if key[0] == key[1]: continue
        if key in io2trips: io2trips[key].append(trip)
        else: io2trips[key] = [trip]
    return io2trips

def extract(path: str = "./preprocessed_data/beijing_data", removeloops: bool = True) -> Result:
    edge_df: pd.DataFrame = gpd.read_file(os.path.join(path, "map/edges.shp"), ignore_geometry=True)
    node_df: pd.DataFrame = gpd.read_file(os.path.join(path, "map/nodes.shp"), ignore_geometry=True)

    map_node_osmid_to_id = {j: i for i, j in enumerate(node_df[["osmid"]].to_numpy().flatten())}
    edge_df['u'] = edge_df['u'].map(map_node_osmid_to_id)
    edge_df['v'] = edge_df['v'].map(map_node_osmid_to_id)

    with open(os.path.join(path, "preprocessed_test_trips_all.pkl"), "rb") as f:
        data_test = pickle.load(f)
    with open(os.path.join(path, "preprocessed_train_trips_all.pkl"), "rb") as f:
        data_train = pickle.load(f)
    with open(os.path.join(path, "preprocessed_validation_trips_all.pkl"), "rb") as f:
        data_valid = pickle.load(f)
    
    if removeloops:
        data_test = [(idx, remove_loops(path), time) for (idx, path, time) in data_test]
        data_train = [(idx, remove_loops(path), time) for (idx, path, time) in data_train]
        data_valid = [(idx, remove_loops(path), time) for (idx, path, time) in data_valid]
        
    data_test = [(path, time) for (_, path, time) in data_test if len(path) >= 5]
    data_train = [(path, time) for (_, path, time) in data_train if len(path) >= 5]
    data_valid = [(path, time) for (_, path, time) in data_valid if len(path) >= 5]

    edges_list = edge_df[["u", "v"]].to_numpy().tolist()
    data_test = groupby_uv(data_test, edges_list)
    data_train = groupby_uv(data_train, edges_list)
    data_valid = groupby_uv(data_valid, edges_list)

    return node_df, edge_df, {"train": data_train, "test": data_test, "valid": data_valid}

def worker(name: str):
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(extract(f"./preprocessed_data/{name}_data/"), f)

if __name__ == "__main__":
    # output files avaliable at https://www.kaggle.com/code/xjq701229/simweight-data-source
    with Pool() as p:
        p.map(worker, ["beijing", "chengdu", "cityindia", "harbin", "porto"])