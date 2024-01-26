from extract_data import Result, Trips
import pickle
import utils_rs

def sampling_tosqlite(path: str, g: utils_rs.DiGraph, trips: Trips, k: int = 32, chunk_size: int = 4096) -> None:
    for table, v_trips in trips.items():
        uvs = list(v_trips.keys())
        pos_samples = list(v_trips.values())
        g.par_path_sampling_tosqlite(uvs, pos_samples, k, chunk_size, path, table, False)

if __name__ == "__main__":
    # output file should be avaliable at https://www.kaggle.com/code/xjq701229/simweight-neg-sample
    with open("./beijing.pkl", "rb") as f:
        tmp: Result = pickle.load(f)
        (nodes, edges, trips) = tmp
    g = utils_rs.DiGraph(nodes.shape[0], [(i['u'], i['v']) for _, i in edges.iterrows()], edges["length"])
    sampling_tosqlite("beijing.db", g, trips)