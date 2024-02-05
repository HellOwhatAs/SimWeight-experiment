from extract_data import Result
from typing import Dict, Set, Tuple
import pickle
import utils_rs
import time
from tqdm import tqdm
import vis_map
import more_itertools

with open("./beijing.pkl", "rb") as f:
    tmp: Result = pickle.load(f)
    (nodes, edges, trips) = tmp
    g = utils_rs.DiGraph(nodes.shape[0], [(i['u'], i['v']) for _, i in edges.iterrows()], edges["length"])

def test_length_acc():
    trips_test = list(more_itertools.flatten(trips["test"].values()))
    t = time.time()
    res = g.experiment(trips_test)
    print(time.time() - t)
    print(res / len(trips_test))

def test_yen():
    map = vis_map.base_edge_map(nodes, edges,
        tiles= 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',
        attr='高德-常规图')
    yen_trips = [path for path, _ in g.yen(10893, 7595, 200)]
    vis_map.add_edges(map, nodes, edges, set(more_itertools.flatten(yen_trips)), color="#00FF00")
    vis_map.add_nodes(map, nodes, {
        10893: "start",
        7595: "target"
    })
    map.save("tmp0.html")

def test_map_case():
    c: Dict[Tuple[int, int], Set[Tuple[int]]] = {}
    tmp = edges[["u", "v"]].to_numpy().tolist()
    for trip in tqdm(more_itertools.flatten(trips["train"].values())):
        key = (tmp[trip[0]][0], tmp[trip[-1]][1])
        if key in c: c[key].add(tuple(trip))
        else: c[key] = {tuple(trip)}
    max_val = max(c.items(), key=lambda x: len(x[1]))
        
    map = vis_map.base_edge_map(nodes, edges,
        tiles= 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',
        attr='高德-常规图')
    vis_map.add_trips(map, nodes, edges, max_val[1])
    yen_trips = [path for path, _ in g.yen(max_val[0][0], max_val[0][1], len(max_val[1]))]
    vis_map.add_trips(map, nodes, edges, yen_trips, color="#00FF00")
    vis_map.add_nodes(map, nodes, {
        max_val[0][0]: f"start: {max_val[0][0]}",
        max_val[0][1]: f"target: {max_val[0][1]}"
    })
    map.save("tmp.html")

def test_neg_sample():
    map = vis_map.base_edge_map(nodes, edges,
        tiles= 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',
        attr='高德-常规图')
    tmp = g.bidirectional_dijkstra([], 10893, 7595, 200)
    vis_map.add_edges(map, nodes, edges, more_itertools.flatten(tmp))
    vis_map.add_nodes(map, nodes, {
        10893: "start",
        7595: "target"
    })
    map.save("tmp1.html")

def test_db():
    db = utils_rs.Sqlite("tmp.db")
    db.insert("train", 114514, 1919810, [[1, 2, 3], [111111111, 456]])
    del db

    db = utils_rs.Sqlite("tmp.db", delete=False)
    print(db.get("train", 114514, 1919810))
    del db

    import os
    os.remove("tmp.db")

def vis_neg_samples():
    from neg_sample import SampleLoader
    sql = SampleLoader("./beijing.db", "train")
    u, v = 5625, 9249
    map = vis_map.base_edge_map(nodes, edges,
        tiles= 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',
        attr='高德-常规图')
    
    pos_edges = set(more_itertools.flatten(trips["train"][(u, v)]))
    neg_edges = set(more_itertools.flatten(sql.get(u, v)))

    vis_map.add_edges(map, nodes, edges, pos_edges)
    vis_map.add_edges(map, nodes, edges, neg_edges, color="#00FF00")
    vis_map.add_edges(map, nodes, edges, pos_edges & neg_edges, color="#FF00FF")

    vis_map.add_nodes(map, nodes, {
        u: f"start: {u}",
        v: f"target: {v}"
    })
    map.save("tmp1.html")

def test_neg_samples_valid():
    from neg_sample import SampleLoader
    sql = SampleLoader("./beijing.db", "test")
    edges_loc = edges[["u", "v"]].to_numpy().tolist()
    for u, v in tqdm(sql.keys()):
        samples = sql.get(u, v)
        for trip in samples:
            assert all(edges_loc[a][1] == edges_loc[b][0] for a, b in more_itertools.pairwise(trip)), (u, v)
            assert edges_loc[trip[0]][0] == u and edges_loc[trip[-1]][1] == v, (u, v)

test_neg_samples_valid()