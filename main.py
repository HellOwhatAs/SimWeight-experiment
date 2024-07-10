import argparse
from typing import Optional

parser = argparse.ArgumentParser(description='training of Rower model')
parser.add_argument('--log', type=str, default="beijing_accs.txt", help='log file name')
parser.add_argument('--city', type=str, default="beijing.pkl", help='city file name')
parser.add_argument('--model', type=str, default="beijing_model_weights.pth", help='model weights file name')
parser.add_argument('--device', type=str, default="cuda", help='device where model run')
parser.add_argument('--epoch', type=int, default=4000, help='number training runs')
parser.add_argument('--chunk', type=int, default=2 ** 16, help='chunk size')
parser.add_argument('--k', type=int, default=None, help='k sampling paths')
parser.add_argument('--train_num', type=int, default=None, help='num of paths used in training')
parser.add_argument('--test_num', type=int, default=5000, help='num of paths used in testing')
args = parser.parse_args()
log_fname: str = getattr(args, 'log')
city_fname: str = getattr(args, 'city')
model_fname: str = getattr(args, 'model')
device_name: str = getattr(args, 'device')
epoch: int = getattr(args, 'epoch')
chunk_size: int = getattr(args, 'chunk')
k: Optional[int] = getattr(args, 'k')
train_num: Optional[int] = getattr(args, 'train_num')
test_num: int = getattr(args, 'test_num')

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

device = torch.device(device_name)
if not torch.cuda.is_available():
    device = torch.device("cpu")
    warnings.warn("Using CPU")

with open(city_fname, "rb") as f:
    tmp: Result = pickle.load(f)
    (nodes, edges, trips) = tmp

random.seed(42)
trips_train = [(k, [i for i, _ in v]) for k, v in (
    trips["test"].items()
    if train_num is None else 
    random.sample(trips["test"].items(), train_num)
)]
trips_test = {k: v for k, v in [(k, [i for i, _ in v]) for k, v in (
    trips["valid"].items()
    if test_num is None else
    random.sample(trips["valid"].items(), test_num)
)]}
total_test = sum(len(i) for i in trips_test.values())

g = utils_rs.DiGraph(nodes.shape[0], [(i['u'], i['v']) for _, i in edges.iterrows()], edges["length"])
model = Rower(edges).to(device)
if os.path.isfile(model_fname):
    model.load_state_dict(torch.load(model_fname))
optimizer = torch.optim.Adam(model.parameters())
accs: List[int] = []
best_acc: int = 0
losses: List[float] = []
start_time = time.time()

def save_checkpoint():
    torch.save(model.state_dict(), model_fname)

    with open(log_fname, "a") as f:
        f.write("; ".join(map(str, zip(accs, losses))))
        f.write(f'\n#{time.time() - start_time}\n')

model.train()
pbar = tqdm(range(epoch))
for epoch in pbar:
    loss_value = 0
    random.shuffle(trips_train)
    for chunk in chunked(trips_train, chunk_size):
        seq, sep = batch_trips(chunk, g, k)
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

    if acc > best_acc:
        best_acc = acc
        save_checkpoint()