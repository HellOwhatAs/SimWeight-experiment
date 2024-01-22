from typing import Tuple, Union
from more_itertools import pairwise, flatten
import torch

class WeightEmbedding(torch.nn.Module):
    
    def __init__(self, num_edges: int, num_fields: Union[Tuple[int], None] = None):
        super().__init__()
        if num_fields is not None:
            self.embedding = torch.nn.Sequential(
                *flatten([(torch.nn.Linear(i, o), torch.nn.ReLU()) for i, o in pairwise(num_fields)]),
                torch.nn.Linear(num_fields[-1], 1)
            )
        else:
            self.embedding = None
        self.weight = torch.nn.Parameter(torch.rand((num_edges, )))

    def forward(self, idx: torch.Tensor, filed: Union[torch.Tensor, None] = None) -> torch.Tensor:
        """
        :param idx: tensor of size ``(edge_idxs, 1)``
        :param filed: tensor of size ``(edge_idxs, num_fields)``
        :return: tensor of size ``(edge_idxs,)``
        """
        res = self.weight[idx]
        assert (filed is None) == (self.embedding is None), "`filed` must consistent with `num_fields` in constructor"
        if self.embedding is not None and filed is not None:
            res += self.embedding(filed)
        return res

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

    lossfunc = torch.nn.MSELoss()
    m = WeightEmbedding(edges.shape[0], (1, 16)).to(device)
    optimizer = torch.optim.Adam(m.parameters(), lr=0.01)

    x_idx = torch.arange(0, edges.shape[0]).to(device).view(-1, 1)
    x_train = torch.Tensor(edges["length"].to_numpy()).to(device).view(-1, 1)
    y_train = torch.Tensor(edges["length"].to_numpy()).to(device).view(-1, 1)

    m.train()
    for _ in (pbar := tqdm(range(10000))):
        result = m(x_idx[:1000], x_train[:1000])
        loss = lossfunc(result, y_train[:1000])
        pbar.set_postfix_str(f"loss={loss.item():.5f}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    m.eval()
    with torch.no_grad():
        result = m(x_idx, x_train)
        print(lossfunc(result, y_train))