from typing import Tuple, Union, List
from more_itertools import pairwise, flatten
import pandas as pd
import torch
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_packed_sequence
import numpy as np
import utils_rs

class WeightEmbedding(torch.nn.Module):
    
    def __init__(self, num_edges: int, num_fields: Union[Tuple[int], None] = None):
        super().__init__()
        self.embedding = torch.nn.Sequential(
            *flatten([(torch.nn.Linear(i, o), torch.nn.ReLU()) for i, o in pairwise(num_fields)]),
            torch.nn.Linear(num_fields[-1], 1)
        ) if num_fields is not None else None
        self.weight = torch.nn.Parameter(torch.zeros((num_edges, )))

    def forward(self, idx: torch.Tensor, field: Union[torch.Tensor, None] = None) -> torch.Tensor:
        """
        :param idx: tensor of size ``(edge_idxs, 1)``
        :param filed: tensor of size ``(edge_idxs, num_fields)``
        :return: tensor of size ``(edge_idxs,)``
        """
        res = self.weight[idx]
        assert (field is None) == (self.embedding is None), "`field` must consistent with `num_fields` in constructor"
        if self.embedding is not None and field is not None:
            res += self.embedding(field)
        return res.exp()
    
class Rower(torch.nn.Module):

    def __init__(self, edges: pd.DataFrame, attrs: List[str] = ["length"]):
        super().__init__()
        self.edge_base = torch.nn.Parameter(torch.from_numpy(edges[attrs].to_numpy(dtype=np.float32)).view(-1, len(attrs)), requires_grad=False)
        self.edge_weight = WeightEmbedding(edges.shape[0])

    def forward(self, trips: PackedSequence) -> torch.Tensor:
        res: torch.Tensor = self.edge_weight(trips.data.view(-1, 1)) * self.edge_base[trips.data]
        tmp = pad_packed_sequence(PackedSequence(res.squeeze(), trips.batch_sizes, trips.sorted_indices, trips.unsorted_indices), batch_first=True)[0]
        return tmp.sum(dim=1)
    
    def get_weight(self) -> torch.Tensor:
        with torch.no_grad():
            return self.edge_weight(torch.arange(0, self.edge_weight.weight.shape[0]).view(-1, 1)) * self.edge_base

def batch_trips(chunk: List[Tuple[Tuple[int, int], List[List[int]]]], g: utils_rs.DiGraph) -> Tuple[List[torch.LongTensor], List[Tuple[int, int]]]:
    seq: List[torch.LongTensor] = []
    sep: List[Tuple[int, int]] = []
    negative_samples_chunk = g.par_bidirectional_dijkstra([(u, v, len(paths)) for (u, v), paths in chunk])
    for (_, positive_samples), negative_samples in zip(chunk, negative_samples_chunk):
        seq.extend(torch.LongTensor(trip) for trip in positive_samples)
        seq.extend(torch.LongTensor(trip) for trip in negative_samples)
        sep.append((len(positive_samples), len(negative_samples)))
    return seq, sep

def bpr_loss_reverse(lengths: torch.Tensor, sep: List[Tuple[int, int]]) -> torch.Tensor:
    idx = 0
    return sum(
        - torch.nn.functional.logsigmoid(
            - lengths[idx: (idx := idx + pos)] + lengths[idx: (idx := idx + neg)].unsqueeze(1)
        ).sum()
        for (pos, neg) in sep)

if __name__ == "__main__":
    from extract_data import Result
    import pickle
    import os, time
    from tqdm import tqdm
    import warnings
    from more_itertools import chunked
    import random

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