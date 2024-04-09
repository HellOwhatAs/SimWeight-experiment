from extract_data import Result
from typing import Dict, List, Set, Tuple, Optional
from contextlib import contextmanager
import pickle
import utils_rs
from tqdm import tqdm
import vis_map
import more_itertools
from rower_model import Rower
import torch
import cmap
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

class Test:
    def __init__(self, city: str) -> None:
        self.city = city

        with open(f"./{city}.pkl", "rb") as f:
            tmp: Result = pickle.load(f)
        (self.nodes, self.edges, self.trips) = tmp
        self.g = utils_rs.DiGraph(self.nodes.shape[0], [(i['u'], i['v']) for _, i in self.edges.iterrows()], self.edges["length"])

        self.model = Rower(self.edges)
        self.model.load_state_dict(torch.load(f'{city}_model_weights.pth', map_location='cpu'))

    @contextmanager
    def with_weight(self, new_weight: List[float]):
        old_weight = self.g.weight
        self.g.weight = new_weight
        try: yield
        finally: self.g.weight = old_weight

    def acc_old(self):
        trips_test = list(more_itertools.flatten(self.trips["test"].values()))
        baseline = self.g.experiment_old(trips_test)
        with self.with_weight(self.model.get_weight().flatten().cpu().numpy()):
            acc = self.g.experiment_old(trips_test)
        return baseline / len(trips_test), acc / len(trips_test)

    def acc(self):
        baseline = self.g.experiment(self.trips["test"])
        with self.with_weight(self.model.get_weight().flatten().cpu().numpy()):
            acc = self.g.experiment(self.trips["test"])
        return baseline / len(self.trips["test"]), acc / len(self.trips["test"])

    def vis_yen(self, u: int = 10893, v: int = 7595, k: int = 200):
        nodes, edges, g = self.nodes, self.edges, self.g
        map = vis_map.base_edge_map(nodes, edges,
            tiles= 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',
            attr='高德-常规图', zoom_start=12)
        yen_trips = [path for path, _ in g.yen(u, v, k)]
        vis_map.add_edges(map, nodes, edges, set(more_itertools.flatten(yen_trips)), color="#CE4257")
        vis_map.add_nodes(map, nodes, {
            u: {
                'popup': "start",
                'color': '#0000FF',
            },
            v: {
                'popup': "target",
                'color': '#A020F0',
            }
        })
        map.save("yen.html")

    def vis_bidijkstra(self, u: int = 10893, v: int = 7595, k: int = 200):
        nodes, edges, g = self.nodes, self.edges, self.g
        map = vis_map.base_edge_map(nodes, edges,
            tiles= 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',
            attr='高德-常规图', zoom_start=12)
        tmp = g.bidirectional_dijkstra(u, v, k)
        vis_map.add_edges(map, nodes, edges, more_itertools.flatten(tmp), color="#CE4257")
        vis_map.add_nodes(map, nodes, {
            u: {
                'popup': "start",
                'color': '#0000FF',
            },
            v: {
                'popup': "target",
                'color': '#A020F0',
            }
        })
        map.save("bidijkstra.html")

    def vis_why_dynamic(self):
        nodes, edges, trips, g = self.nodes, self.edges, self.trips, self.g
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
        vis_map.add_trips(map, nodes, edges, max_val[1], color='#4AAD52')
        # tmp_trips = [path for path, _ in g.yen(max_val[0][0], max_val[0][1], 2000)]
        tmp_trips = g.bidirectional_dijkstra(max_val[0][0], max_val[0][1], len(max_val[1]))
        vis_map.add_trips(map, nodes, edges, tmp_trips, color="#CE4257")
        vis_map.add_nodes(map, nodes, {
            max_val[0][0]: {
                'popup': f"start: {max_val[0][0]}",
                'color': '#0000FF',
            },
            max_val[0][1]: {
                'popup': f"target: {max_val[0][1]}",
                'color': '#A020F0',
            }
        })
        map.save("why_dynamic.html")

    def vis_improve(self):
        nodes, edges, trips, g = self.nodes, self.edges, self.trips, self.g
        c: Dict[Tuple[int, int], Set[Tuple[int]]] = {}
        tmp = edges[["u", "v"]].to_numpy().tolist()
        for trip in tqdm(more_itertools.flatten(trips["train"].values())):
            key = (tmp[trip[0]][0], tmp[trip[-1]][1])
            if key in c: c[key].add(tuple(trip))
            else: c[key] = {tuple(trip)}
        max_val = max(c.items(), key=lambda x: len(x[1]))
        
        yen_trips = [path for path, _ in g.yen(max_val[0][0], max_val[0][1], 1)]
        map = vis_map.base_edge_map(nodes, edges,
            tiles= 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',
            attr='高德-常规图')
        vis_map.add_trips(map, nodes, edges, max_val[1], color='#4AAD52')
        vis_map.add_trips(map, nodes, edges, yen_trips, color="#CE4257")
        vis_map.add_nodes(map, nodes, {
            max_val[0][0]: {
                'popup': f"start: {max_val[0][0]}",
                'color': '#0000FF',
            },
            max_val[0][1]: {
                'popup': f"target: {max_val[0][1]}",
                'color': '#A020F0',
            }
        })
        map.save("before.html")

        with self.with_weight(self.model.get_weight().flatten().cpu().numpy()):
            yen_trips = [path for path, _ in g.yen(max_val[0][0], max_val[0][1], 1)]
        map = vis_map.base_edge_map(nodes, edges,
            tiles= 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',
            attr='高德-常规图')
        vis_map.add_trips(map, nodes, edges, max_val[1], color='#4AAD52')
        vis_map.add_trips(map, nodes, edges, yen_trips, color="#CE4257")
        vis_map.add_nodes(map, nodes, {
            max_val[0][0]: {
                'popup': f"start: {max_val[0][0]}",
                'color': '#0000FF',
            },
            max_val[0][1]: {
                'popup': f"target: {max_val[0][1]}",
                'color': '#A020F0',
            }
        })
        map.save("after.html")

    @staticmethod
    def histogram_equalization(data: np.ndarray, bins: int = 1024):
        hist, bins = np.histogram(data, bins=bins, range=(data.min(), data.max()))
        cdf = hist.cumsum()
        cdf_normalized = cdf / float(cdf.max())
        equalized_data = np.interp(data, bins[:-1], cdf_normalized)
        return equalized_data

    def vis_delta_weight(self):
        old_weight = torch.tensor(self.g.weight)
        new_weight = torch.tensor(self.model.get_weight().flatten().cpu().numpy())
        c_weight = np.array(sorted((new_weight / old_weight / 2)))
        cweight: List[float] = self.histogram_equalization(c_weight, 100000).tolist()

        cm = cmap.Colormap('viridis_r')
        color = [cm(i).hex for i in cweight]
        m = vis_map.colored_edge_map(self.nodes, self.edges, color, zoom_start=12)
        m.save("delta_weight.html")

    def plot_weight_distribute(self, axes: Optional[Axes] = None):
        """
        ```
        test = Test('beijing')
        test.plot_weight_distribute()
        plt.show()
        
        # or

        test = Test('beijing')
        ax = plt.subplot()
        test.plot_weight_distribute(ax)
        plt.show()
        ```
        """
        if axes is None: axes = plt.subplot()
        
        old_weight = torch.tensor(self.g.weight)
        new_weight = torch.tensor(self.model.get_weight().flatten().cpu().numpy())
        c_weight = np.array(sorted((new_weight / old_weight / 2)))
        cweight: List[float] = self.histogram_equalization(c_weight, 100000).tolist()

        cm = cmap.Colormap('viridis_r')
        color = [cm(i).hex for i in cweight]
        axes.set_yscale('log')
        axes.scatter(range(len(c_weight)), c_weight, linewidths=0, edgecolors=None, alpha=0.1, c=color)

    def vis_unlearned_edges(self):
        new_weight = torch.tensor(self.model.get_weight().flatten().cpu().numpy())
        tmp_edges = self.edges.copy(deep=True)
        tmp_edges['weight'] = new_weight

        vis_map.base_edge_map(
            self.nodes,
            tmp_edges.loc[(tmp_edges['length'] == tmp_edges['weight']).abs() < 1e-5],
            zoom_start=12,
            color='black'
        ).save("unlearned_edges.html")

if __name__ == '__main__':
    test = Test('cityindia')
    test.vis_unlearned_edges()