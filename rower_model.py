from typing import Tuple, Union, List, Optional
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
    
    def weight_factor(self) -> torch.Tensor:
        return self.edge_weight(torch.arange(0, self.edge_weight.weight.shape[0]).view(-1, 1))
    
    def grad_weight(self) -> torch.Tensor:
        return self.weight_factor() * self.edge_base
    
    def get_weight(self):
        with torch.no_grad(): return self.grad_weight()

def batch_trips(chunk: List[Tuple[Tuple[int, int], List[List[int]]]], g: utils_rs.DiGraph, k: Optional[int] = None) -> Tuple[List[torch.LongTensor], List[Tuple[int, int]]]:
    seq: List[torch.LongTensor] = []
    sep: List[Tuple[int, int]] = []
    negative_samples_chunk = g.par_bidirectional_dijkstra([(u, v, len(paths) if k is None else k) for (u, v), paths in chunk])
    for (_, positive_samples), negative_samples in zip(chunk, negative_samples_chunk):
        seq.extend(torch.LongTensor(trip) for trip in positive_samples)
        seq.extend(torch.LongTensor(trip) for trip in negative_samples)
        sep.append((len(positive_samples), len(negative_samples)))
    return seq, sep

def bpr_loss_reverse(lengths: torch.Tensor, sep: List[Tuple[int, int]]) -> torch.Tensor:
    idx = 0
    tmp = []
    for (pos, neg) in sep:
        tmp1 = lengths[idx: (idx + pos)]
        idx += pos
        tmp2 = lengths[idx: (idx + neg)]
        idx += neg
        
        tmp.append(- torch.nn.functional.logsigmoid(
            - tmp1 + tmp2.unsqueeze(1)
        ).sum())
    return sum(tmp)