from extract_data import Result
import pickle
import weight_experiment

with open("./beijing.pkl", "rb") as f:
    tmp: Result = pickle.load(f)
    (nodes, edges, trips) = tmp
    
g = weight_experiment.DiGraph(nodes.shape[0], [(i['u'], i['v']) for _, i in edges.iterrows()])

res = g.experiment(edges["length"].to_list(), trips["test"])

print(res / len(trips))