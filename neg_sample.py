from extract_data import Result, Trips
import pickle
import utils_rs

def trips2trips(g: utils_rs.DiGraph, trips: Trips, k: int = 32) -> Trips:
    res: Trips = {}
    for k_trips, v_trips in trips.items():
        uvs = list(v_trips.keys())
        pos_samples = list(v_trips.values())
        samples = g.par_path_sampling(uvs, pos_samples, k)
        v_samples = {k: v for k, v in zip(uvs, samples)}
        res[k_trips] = v_samples
    return res

if __name__ == "__main__":
    # output file should be avaliable at https://www.kaggle.com/code/xjq701229/simweight-neg-sample
    with open("./beijing.pkl", "rb") as f:
        tmp: Result = pickle.load(f)
        (nodes, edges, trips) = tmp
    g = utils_rs.DiGraph(nodes.shape[0], [(i['u'], i['v']) for _, i in edges.iterrows()], edges["length"])
    with open("./beijing_negsamples.pkl", "wb") as f:
        pickle.dump(trips2trips(g, trips), f)