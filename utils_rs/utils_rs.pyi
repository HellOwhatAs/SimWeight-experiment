from typing import List, Tuple, Optional, Callable, Dict

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

    def bidirectional_dijkstra(self, u: int, v: int, k: int = 1, weight: Optional[List[float]] = None) -> List[List[int]]:
        """
        Compute at most k paths using the Bidirectional Dijkstra search algorithm.
        Args:
            u: start vertex
            v: target vertex
            k: paths num
            weight: `weight[edge_idx] = weight`
        Returns:
            List[path]
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

    def par_yen(self, chunk: List[Tuple[Tuple[int, int], List[List[int]]]]) -> List[List[List[int]]]:
        """
        parallel version of yen  
        with k = len(List[pos_sample])
        Args:
            chunk: List[(u, v, List[pos_sample])]
        """

    def par_bidirectional_dijkstra(self, chunk: List[Tuple[int, int, int]]) -> List[List[List[int]]]:
        """
        parallel version of bidirectional_dijkstra  
        Args:
            chunk: List[(u, v, k)]
        """
    
    def experiment(self, trips: Dict[Tuple[int, int], List[List[int]]], weight: Optional[List[int]] = None) -> int:
        """
        Args:
            trips: `Dict[(u, v), List[trip]]` where `trip = [edge_idx, ...]`
            weight: `weight[edge_idx]` is the weight of the `edge_idx` edge

        Returns:
            int: num of (u, v) that match shortest path under weight
        """

    def experiment_num(self, trips: Dict[Tuple[int, int], List[List[int]]], weight: Optional[List[int]] = None) -> int:
        ...

    def experiment_old(self, trips: List[List[int]], weight: Optional[List[int]] = None) -> int:
        """
        Args:
            trips: `[trip1, ..., tripn]` where `trip = [edge_idx, ...]`
            weight: `weight[edge_idx]` is the weight of the `edge_idx` edge

        Returns:
            int: num of trips that is shortest path under weight
        """

    def experiment_cme(self, trips: List[List[int]], weight: Optional[List[int]] = None) -> int:
        ...

    def experiment_old_topk(self, trips: List[List[int]], k: int, weight: Optional[List[int]] = None) -> int:
        """
        a trip will acc if the top-k shortest path under given weight contains the trip
        """

    def experiment_top(self, trips: Dict[Tuple[int, int], List[List[int]]], k: int) -> float:
        """
        top k sim
        """

    def experiment_path_jaccard(self, trips: Dict[Tuple[int, int], List[List[int]]], weight: Optional[List[int]] = None) -> float:
        """
        Return sum(max(J(p, dijkstra(u, v)) for p in R) for (u, v), R in trips.values())
        """

    def experiment_path_lengths_jaccard(self, trips: Dict[Tuple[int, int], List[List[int]]], lengths: List[float], weight: Optional[List[int]] = None) -> float:
        """
        `experiment_path_jaccard` but edges are weighted by `lengths`
        """

    def experiment_neuromlr(self, trips: Dict[Tuple[int, int], List[List[int]]], lengths: List[float], weight: Optional[List[int]] = None) -> Tuple[float, float]:
        """
        (Precision, Recall), metrics of NeuroMLR
        """

    def experiment_path_lev_distance(self, trips: Dict[Tuple[int, int], List[List[int]]], weight: Optional[List[int]] = None) -> float:
        """
        Return sum(max((len(p) - lev_distance(p, dijkstra(u, v))) / len(p) for p in R) for (u, v), R in trips.values())
        """