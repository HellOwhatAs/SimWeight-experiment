from typing import Tuple, Union, List
from more_itertools import pairwise, flatten
import utils_rs
import pandas as pd
import torch
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

    def __init__(self, edges: pd.DataFrame):
        super().__init__()
        self.edge_attrs = torch.nn.Parameter(torch.from_numpy(edges["length"].to_numpy(dtype=np.float32)).view(-1, 1), requires_grad=False)
        self.edge_weight = WeightEmbedding(edges.shape[0], num_fields=(1, 16))

    def forward(self, trips: List[torch.LongTensor]) -> torch.Tensor:
        return torch.stack([self.edge_weight(trip.view(-1, 1), self.edge_attrs[trip]).sum() for trip in trips])
    
def bpr_loss_reverse(positive_lengths: torch.Tensor, negative_lengths: torch.Tensor) -> torch.Tensor:
    return -torch.nn.functional.logsigmoid(negative_lengths.unsqueeze(1) - positive_lengths).sum()

if __name__ == "__main__":
    from extract_data import Result
    import pickle
    from tqdm import tqdm
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
    
    model.train()
    count = 0
    for (u, v), positive_samples in (pbar := tqdm(trips["test"].items())):
        trips_input = [torch.LongTensor(trip).to(device) for trip in positive_samples]
        positive_lengths = model(trips_input)
        # negative_samples = 
        raise NotImplementedError
        trips_input = [torch.LongTensor(trip).to(device) for trip in negative_samples]
        if not trips_input: continue # not exist negative samples
        negative_lengths = model(trips_input)
        loss = bpr_loss_reverse(positive_lengths, negative_lengths)
        pbar.set_postfix_str(f"loss={loss.item():.4f}", False)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
