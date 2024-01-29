from typing import Literal, Optional
from extract_data import Result, Trips
import pickle, sys
import utils_rs
from tqdm import tqdm

Table = Literal["train", "test", "valid"]

def sampling_tosqlite(path: str, g: utils_rs.DiGraph, trips: Trips, table: Table, k: int = 32, chunk_size: int = 4096) -> None:
    v_trips = trips[table]
    uvs = list(v_trips.keys())
    pos_samples = list(v_trips.values())
    pbar = tqdm(total=len(uvs), desc=table, file=sys.stdout)
    g.par_bidirectional_dijkstra_tosqlite(uvs, pos_samples, k, chunk_size, path, table, False, pbar.update)

class SampleLoader:
    def __init__(self, db_path: str, table: str) -> None:
        self.sqlite = utils_rs.Sqlite(db_path, False)
        self.default_table = table
    
    def get(self, u: int, v: int, table: Optional[Table] = None):
        selected_table = self.default_table if table is None else table
        return self.sqlite.get(selected_table, u, v)


if __name__ == "__main__":
    # output file avaliable at https://www.kaggle.com/code/xjq701229/simweight-neg-sample/
    with open("./beijing.pkl", "rb") as f:
        tmp: Result = pickle.load(f)
        (nodes, edges, trips) = tmp
    g = utils_rs.DiGraph(nodes.shape[0], [(i['u'], i['v']) for _, i in edges.iterrows()], edges["length"])
    sampling_tosqlite("beijing.db", g, trips, "train")
    sampling_tosqlite("beijing.db", g, trips, "test")
    sampling_tosqlite("beijing.db", g, trips, "valid")