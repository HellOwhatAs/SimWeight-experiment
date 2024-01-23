from typing import List, Tuple

class DiGraph:
    def __init__(self, n: int, edges: List[Tuple[int, int]]) -> None:
        self.n: int
        "num of vertexs"
        self.edges: List[Tuple[int, int]]
        "`edges[edge_idx] = (u, v)`"
        self.adjlist: List[List[Tuple[int, int]]]
        "`adjlist[u] = [(v1, edge_idx), ..., (vn, edge_idx)]`"

    def experiment(self, weight: List[int], trips: List[List[int]]) -> int:
        """
        Args:
            weight: `weight[edge_idx]` is the weight of the `edge_idx` edge
            trips: `[trip1, ..., tripn]` where `trip = [edge_idx, ...]`

        Returns:
            int: num of trips that is shortest path under weight
        """