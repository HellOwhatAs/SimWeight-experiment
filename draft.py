from extract_data import Result
import pickle
import weight_experiment
import time

with open("./beijing.pkl", "rb") as f:
    tmp: Result = pickle.load(f)
    (nodes, edges, trips) = tmp
    
g = weight_experiment.DiGraph(nodes.shape[0], [(i['u'], i['v']) for _, i in edges.iterrows()])
t = time.time()
res = g.experiment(edges["length"], trips["test"])
print(time.time() - t)
print(res / len(trips["test"]))