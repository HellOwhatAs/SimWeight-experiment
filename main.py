from extract_data import Result
from typing import List
import pickle
import os, time
from tqdm import tqdm
import warnings
from more_itertools import chunked, flatten
import random
import torch, utils_rs
from rower_model import Rower, batch_trips, pack_sequence, bpr_loss_reverse

device = torch.device("cuda")
if not torch.cuda.is_available():
    device = torch.device("cpu")
    warnings.warn("Using CPU")

with open("./beijing.pkl", "rb") as f:
    tmp: Result = pickle.load(f)
    (nodes, edges, trips) = tmp

random.seed(42)
trips_train = {k: v for k, v in trips["train"].items()}
trips_test = {k: v for k, v in random.sample(list(trips["test"].items()), 5000)}
total_test = sum(len(i) for i in trips_test.values())

g = utils_rs.DiGraph(nodes.shape[0], [(i['u'], i['v']) for _, i in edges.iterrows()], edges["length"])
model = Rower(edges).to(device)
if os.path.isfile('model_weights.pth'):
    model.load_state_dict(torch.load('model_weights.pth'))
optimizer = torch.optim.Adam(model.parameters())
accs: List[int] = []
losses: List[float] = []
start_time = time.time()

model.train()
for epoch in (pbar := tqdm(range(4000))):
    loss_value = 0
    for chunk in chunked(trips_train.items(), 2 ** 16):
        seq, sep = batch_trips(chunk, g)
        trips_input = pack_sequence(seq, enforce_sorted=False).to(device)
        lengths = model(trips_input)
        loss = bpr_loss_reverse(lengths, sep)
        loss.backward()
        loss_value += loss.item()
        optimizer.step()
        optimizer.zero_grad()

    g.weight = model.get_weight().flatten().cpu().numpy()
    acc = g.experiment_cme(list(flatten(trips_test.values())))
    pbar.set_postfix(acc=f"{acc} / {total_test}", loss=f"{loss_value:.4f}", refresh=False)
    accs.append(acc)
    losses.append(loss_value)

torch.save(model.state_dict(), 'model_weights.pth')

with open("accs.txt", "a") as f:
    f.write("; ".join(map(str, zip(accs, losses))))
    f.write(f'\n#{time.time() - start_time}\n')