from extract_data import Result, groupby_io
from typing import Dict, List, Set, Tuple, Optional
from contextlib import contextmanager
import pickle
import utils_rs
from math import inf
from tqdm import tqdm
import vis_map, folium
import more_itertools, itertools
import pandas as pd, random
from rower_model import Rower
import torch
import cmap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
plt.rc('font', family='Times New Roman')

class Test:
    def __init__(self, city: str, data_fp: Optional[str] = None, model_fp: Optional[str] = None) -> None:
        self.city = city

        with open(data_fp if data_fp is not None else f"./{city}.pkl", "rb") as f:
            tmp: Result = pickle.load(f)
        (nodes, edges, trips) = tmp
        trips = {
            k: {k1: [i for i, _ in v1] for k1, v1 in v.items()}
            for k, v in trips.items()
        }

        nodes_cross: List[Tuple[List[int], List[int]]] = [([], []) for _ in range(nodes.shape[0])]
        for edge_idx, edge in edges.iterrows():
            u, v = int(edge['u']), int(edge['v'])
            nodes_cross[u][1].append(edge_idx)
            nodes_cross[v][0].append(edge_idx)
        transformed_edges: List[Tuple[int, int]] = []
        transformed_length: List[float] = []
        for in_edges, out_edges in nodes_cross:
            for in_edge, out_edge in itertools.product(in_edges, out_edges):
                transformed_edges.append((in_edge, out_edge))
                transformed_length.append((edges["length"][in_edge] + edges["length"][out_edge]) / 2)
        self.edges = pd.DataFrame(transformed_edges, columns=["u", "v"], dtype=np.int64)
        self.edges["length"] = transformed_length
        self.original_edges_length = edges["length"].to_list()

        ee2te: Dict[Tuple[int, int], int] = {key: idx for idx, key in enumerate(transformed_edges)}
        
        self.trips = {
            k: groupby_io(
                (
                    [ee2te[ee] for ee in more_itertools.pairwise(trip)]
                    for trip in more_itertools.flatten(trips for _, trips in v.items())
                ), transformed_edges
            )
            for k, v in trips.items()
        }
        self.trips["valid"] = dict(random.sample(sorted(self.trips["valid"].items()), 10000))

        self.g = utils_rs.DiGraph(edges.shape[0], [(getattr(i, 'u'), getattr(i, 'v')) for i in self.edges.itertuples()], self.edges["length"])

        self.model = Rower(self.edges)
        self.model.load_state_dict(torch.load(model_fp if model_fp is not None else f'{city}_model_weights.pth', map_location='cpu'))

    @contextmanager
    def weight_being(self, new_weight: List[float]):
        old_weight = self.g.weight
        self.g.weight = new_weight
        try: yield
        finally: self.g.weight = old_weight

    def acc_old(self):
        trips_test = list(more_itertools.flatten(self.trips["valid"].values()))
        baseline = self.g.experiment_old(trips_test)
        with self.weight_being(self.model.get_weight().flatten().cpu().numpy()):
            acc = self.g.experiment_old(trips_test)
        return baseline / len(trips_test), acc / len(trips_test)

    def acc(self):
        baseline = self.g.experiment(self.trips["valid"])
        with self.weight_being(self.model.get_weight().flatten().cpu().numpy()):
            acc = self.g.experiment(self.trips["valid"])
        return baseline / len(self.trips["valid"]), acc / len(self.trips["valid"])
    
    def acc_jaccard(self):
        baseline = self.g.experiment_path_jaccard(self.trips["valid"])
        with self.weight_being(self.model.get_weight().flatten().cpu().numpy()):
            acc = self.g.experiment_path_jaccard(self.trips["valid"])
        return baseline / len(self.trips["valid"]), acc / len(self.trips["valid"])
    
    def acc_lengths_jaccard(self):
        baseline = self.g.experiment_path_lengths_jaccard(self.trips["valid"], self.edges["length"])
        with self.weight_being(self.model.get_weight().flatten().cpu().numpy()):
            acc = self.g.experiment_path_lengths_jaccard(self.trips["valid"], self.edges["length"])
        return baseline / len(self.trips["valid"]), acc / len(self.trips["valid"])
    
    def acc_neuromlr(self):
        baseline_precision, baseline_recall = self.g.experiment_neuromlr(self.trips["valid"], self.original_edges_length)
        with self.weight_being(self.model.get_weight().flatten().cpu().numpy()):
            precision, recall = self.g.experiment_neuromlr(self.trips["valid"], self.original_edges_length)
        return (
            (baseline_precision / len(self.trips["valid"]), precision / len(self.trips["valid"])),
            (baseline_recall / len(self.trips["valid"]), recall / len(self.trips["valid"]))
        )

    def acc_lev_distance(self):
        baseline = self.g.experiment_path_lev_distance(self.trips["valid"])
        with self.weight_being(self.model.get_weight().flatten().cpu().numpy()):
            acc = self.g.experiment_path_lev_distance(self.trips["valid"])
        return baseline / len(self.trips["valid"]), acc / len(self.trips["valid"])
    
    def acc_top(self, k: int):
        baseline = self.g.experiment_top(self.trips["valid"], k)
        with self.weight_being(self.model.get_weight().flatten().cpu().numpy()):
            acc = self.g.experiment_top(self.trips["valid"], k)
        return baseline / len(self.trips["valid"]), acc / len(self.trips["valid"])

    @staticmethod
    def histogram_equalization(data: np.ndarray, bins: int = 1024):
        hist, bins = np.histogram(data, bins=bins, range=(data.min(), data.max()))
        cdf = hist.cumsum()
        cdf_normalized = cdf / float(cdf.max())
        equalized_data = np.interp(data, bins[:-1], cdf_normalized)
        return equalized_data

    def plot_weight_distribute(self, axes: Optional[Axes] = None, stroke_color: str = '#737373'):
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

        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.grid()

        axes.spines['bottom'].set_color(stroke_color) 
        axes.spines['left'].set_color(stroke_color)
        axes.xaxis.label.set_color(stroke_color)
        axes.yaxis.label.set_color(stroke_color)
        axes.tick_params(axis='x', colors=stroke_color)
        axes.tick_params(axis='y', colors=stroke_color)
        
        old_weight = torch.tensor(self.g.weight)
        new_weight = torch.tensor(self.model.get_weight().flatten().cpu().numpy())
        c_weight = np.array(sorted((new_weight / old_weight / 2)))
        cweight: List[float] = self.histogram_equalization(c_weight, 100000).tolist()

        cm = cmap.Colormap('viridis_r')
        color = [cm(i).hex for i in cweight]
        axes.set_yscale('log')
        axes.scatter(range(len(c_weight)), c_weight, linewidths=0, edgecolors=None, alpha=0.1, c=color)
        axes.set_xlabel('sorted edges')
        axes.set_ylabel(r'$\frac{w(e)}{\mathrm{length}(e)}$')

if __name__ == '__main__':
    for city in ('beijing', ):
        test = Test(city, data_fp=f"./{city}.pkl", model_fp=f"./{city}_model_weights.pth")
        fmt = city + ' {\n' + (
            f'    Sim = {test.acc()},\n' +
            # f'    SimTop3 = {test.acc_top(3)},\n' + # too slow
            # f'    Jaccard = {test.acc_jaccard()},\n' +
            # f'    LengthsJaccard = {test.acc_lengths_jaccard()},\n' +
            # f'    LevDistance = {test.acc_lev_distance()}\n' +
            f'    Precision = {(prec_recall := test.acc_neuromlr())[0]},\n' +
            f'    Recall = {prec_recall[1]},\n' +
            '}'
        )
        print(fmt)