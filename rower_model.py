from typing import Tuple, Union, List
from more_itertools import pairwise, flatten
import pandas as pd
import torch
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_packed_sequence
import numpy as np

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
        return res
    
class Rower(torch.nn.Module):

    def __init__(self, edges: pd.DataFrame, attrs: List[str] = ["length"]):
        super().__init__()
        self.edge_attrs = torch.nn.Parameter(torch.from_numpy(edges[attrs].to_numpy(dtype=np.float32)).view(-1, len(attrs)), requires_grad=False)
        self.edge_weight = WeightEmbedding(edges.shape[0], num_fields=(len(attrs), 16))

    def forward(self, trips: PackedSequence) -> torch.Tensor:
        res: torch.Tensor = self.edge_weight(trips.data.view(-1, 1), self.edge_attrs[trips.data])
        tmp = pad_packed_sequence(PackedSequence(res.squeeze(), trips.batch_sizes, trips.sorted_indices, trips.unsorted_indices), batch_first=True)[0]
        return tmp.sum(dim=1)
    
def bpr_loss_reverse(lengths: torch.Tensor, k: int) -> torch.Tensor:
    return -torch.nn.functional.logsigmoid(lengths[k:].unsqueeze(1) - lengths[:k]).sum()

if __name__ == "__main__":
    from extract_data import Result
    import pickle
    from tqdm import tqdm
    from neg_sample import SampleLoader
    import warnings

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        warnings.warn("Using CPU")

    with open("./beijing.pkl", "rb") as f:
        tmp: Result = pickle.load(f)
        (nodes, edges, trips) = tmp

    model = Rower(edges).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    neg = SampleLoader("./beijing.db", "test")
    
    model.train()
    for (u, v), positive_samples in (pbar := tqdm(trips["test"].items())):
        trips_input = pack_sequence([*(torch.LongTensor(trip) for trip in positive_samples), *(torch.LongTensor(trip) for trip in (neg.get(u, v)) if trip)], enforce_sorted=False).to(device)
        lengths = model(trips_input)
        loss = bpr_loss_reverse(lengths, len(positive_samples))
        pbar.set_postfix_str(f"loss={loss.item():.4f}", False)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
