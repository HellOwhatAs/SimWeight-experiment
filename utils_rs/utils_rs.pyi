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

    def yen_drop(self, positive_samples: List[List[int]], u: int, v: int, k: int, weight: Optional[List[float]] = None) -> List[Tuple[List[int], float]]:
        """
        filter yen's result that paths not in positive_samples
        Args:
            positive_samples: `List[List[edge_idx]]`
            u: start vertex
            v: target vertex
            k: the amount of paths requests, including the shortest one
            weight: `weight[edge_idx] = weight`
        Returns:
            List[(path: List[edge_idx], cost)]
        """

    def path_sampling(self, u: int, v: int, pos_samples: List[List[int]], k: int) -> List[List[int]]:
        """
        sample at most k paths from u to v that not in pos_samples
        """

    def par_path_sampling_tosqlite(self, uvs: List[Tuple[int, int]], pos_samples: List[List[List[int]]], k: int, chunk_size: int, path: str, table: str, delete: bool) -> None:
        """
        parallel version of path_sampling that outputs to a sqlite db file
        """

    def experiment(self, trips: List[List[int]], weight: Optional[List[int]] = None) -> int:
        """
        Args:
            trips: `[trip1, ..., tripn]` where `trip = [edge_idx, ...]`
            weight: `weight[edge_idx]` is the weight of the `edge_idx` edge

        Returns:
            int: num of trips that is shortest path under weight
        """


class Sqlite:
    def __init__(self, db_path: str, delete = True) -> None:
        """
        Connect to database
        """
    
    def insert_btyes(self, table: str, data: Tuple[int, int, bytes]) -> None:
        ...

    def insert(self, table: str, u: int, v: int, samples: List[List[int]]) -> None:
        ...

    def get_bytes(self, table: str, u: int, v: int) -> Optional[bytes]:
        ...

    def get(self, table: str, u: int, v: int) -> Optional[List[List[int]]]:
        ...