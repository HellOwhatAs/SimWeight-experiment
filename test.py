from extract_data import Result
from typing import Dict, Set, Tuple
import pickle
import utils_rs
import time
from tqdm import tqdm
import vis_map

with open("./beijing.pkl", "rb") as f:
    tmp: Result = pickle.load(f)
    (nodes, edges, trips) = tmp
    g = utils_rs.DiGraph(nodes.shape[0], [(i['u'], i['v']) for _, i in edges.iterrows()], edges["length"])

def test_length_acc():
    t = time.time()
    res = g.experiment(trips["test"])
    print(time.time() - t)
    print(res / len(trips["test"]))

def test_yen():
    map = vis_map.base_edge_map(nodes, edges,
        tiles= 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',
        attr='高德-常规图')
    yen_trips = [path for path, _ in g.yen(10893, 7595, 3000)]
    vis_map.add_trips(map, nodes, edges, yen_trips, color="#00FF00")
    vis_map.add_nodes(map, nodes, {
        10893: "start",
        7595: "target"
    })
    map.save("tmp0.html")

def test_map_case():
    c: Dict[Tuple[int, int], Set[Tuple[int]]] = {}
    tmp = edges[["u", "v"]].to_numpy().tolist()
    for trip in tqdm(trips["train"]):
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

test_yen()