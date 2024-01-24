from typing import List, Tuple, Optional

class DiGraph:
    def __init__(self, n: int, edges: List[Tuple[int, int]], weight: Optional[List[float]] = None) -> None:
        self.n: int
        "num of vertexs"
        self.edges: List[Tuple[int, int]]
        "`edges[edge_idx] = (u, v)`"
        self.adjlist: List[List[Tuple[int, int]]]
        "`adjlist[u] = [(v1, edge_idx), ..., (vn, edge_idx)]`"
        self.weight: Optional[List[float]]
        "`weight[edge_idx] = weight`"

    def dijkstra(self, u: int, v: int, weight: Optional[List[float]] = None) -> Optional[Tuple[List[int], float]]:
        """
        Compute a shortest path using the Dijkstra search algorithm.
        Args:
            u: start vertex
            v: target vertex
            weight: `weight[edge_idx] = weight`
        Returns:
            (path: List[edge_idx], cost)
            None if target unreachable
        """

    def yen(self, u: int, v: int, k: int, weight: Optional[List[float]] = None) -> List[Tuple[List[int], float]]:
        """
        Compute the k-shortest paths using the Yen’s search algorithm.
        Args:
            u: start vertex
            v: target vertex
            k: the amount of paths requests, including the shortest one
            weight: `weight[edge_idx] = weight`
        Returns:
            List[(path: List[edge_idx], cost)]
        """

    def experiment(self, trips: List[List[int]], weight: Optional[List[int]] = None) -> int:
        """
        Args:
            trips: `[trip1, ..., tripn]` where `trip = [edge_idx, ...]`
            weight: `weight[edge_idx]` is the weight of the `edge_idx` edge

        Returns:
            int: num of trips that is shortest path under weight
        """