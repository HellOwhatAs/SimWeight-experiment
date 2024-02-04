from typing import Tuple, Union, List
from more_itertools import pairwise, flatten
import pandas as pd
import torch
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_packed_sequence
import numpy as np
from neg_sample import SampleLoader

class WeightEmbedding(torch.nn.Module):
    
    def __init__(self, num_edges: int, num_fields: Union[Tuple[int], None] = None):
        super().__init__()
        self.embedding = torch.nn.Sequential(
            *flatten([(torch.nn.Linear(i, o), torch.nn.ReLU()) for i, o in pairwise(num_fields)]),
            torch.nn.Linear(num_fields[-1], 1)
        ) if num_fields is not None else None
        self.weight = torch.nn.Parameter(torch.rand((num_edges, )))

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
        return res.sigmoid()
    
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

def batch_trips(chunk: List[Tuple[Tuple[int, int], List[List[int]]]], neg: SampleLoader) -> Tuple[List[torch.LongTensor], List[Tuple[int, int]]]:
    seq: List[torch.LongTensor] = []
    sep: List[Tuple[int, int]] = []
    for (u, v), positive_samples in chunk:
        negative_samples = neg.get(u, v)
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
    import utils_rs, more_itertools

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        warnings.warn("Using CPU")

    with open("./beijing.pkl", "rb") as f:
        tmp: Result = pickle.load(f)
        (nodes, edges, trips) = tmp

    trips_test = {k: v for k, v in more_itertools.take(100000, trips["test"].items())}

    g = utils_rs.DiGraph(nodes.shape[0], [(i['u'], i['v']) for _, i in edges.iterrows()], edges["length"])
    model = Rower(edges).to(device)
    if os.path.isfile('model_weights.pth'):
        model.load_state_dict(torch.load('model_weights.pth'))
    optimizer = torch.optim.Adam(model.parameters())
    neg = SampleLoader("./beijing.db", "test")
    acc_samples = list(more_itertools.flatten(trips_test.values()))
    accs: List[int] = []
    start_time = time.time()

    model.train()
    for epoch in range(250):
        for chunk in more_itertools.chunked(pbar := tqdm(trips_test.items(), desc=f"epoch: {epoch}({time.time() - start_time}s)"), 16276):
            seq, sep = batch_trips(chunk, neg)
            trips_input = pack_sequence(seq, enforce_sorted=False).to(device)
            lengths = model(trips_input)
            loss = bpr_loss_reverse(lengths, sep)
            pbar.set_postfix_str(f"loss={loss.item():.4f}", False)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        weight = model.get_weight().flatten().tolist()
        acc = g.experiment(acc_samples, weight)
        print(acc)
        accs.append(acc)

    torch.save(model.state_dict(), 'model_weights.pth')
    
    with open("accs.txt", "a") as f:
        f.write(" ".join(map(str, accs)))
        f.write(f'\n#{time.time() - start_time}\n')