from typing import List
from rower_model import Rower, pack_sequence, batch_trips, bpr_loss_reverse
import pandas as pd
import torch
import more_itertools
from tqdm import tqdm
import utils_rs
import matplotlib.pyplot as plt
from pprint import pprint


edges = pd.DataFrame(data = {
    "length": [1.0, 4, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1],
    "u": [0, 0, 1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 6, 7, 7, 8],
    "v": [1, 3, 2, 4, 5, 4, 6, 3, 5, 7, 4, 8, 7, 6, 8, 7]
}, dtype=int)

trips = {
    (0, 6): [
        [0, 3, 7, 6],
        [0, 3, 9, 13],
    ],
    (0, 8): [
        [0, 3, 8, 11],
        # [0, 2, 4, 10, 9, 14]
    ]
}
g = utils_rs.DiGraph(9, [(i['u'], i['v']) for _, i in edges.iterrows()], edges["length"])

def rower():
    model = Rower(edges)
    optimizer = torch.optim.Adam(model.parameters())
    accs: List[int] = []
    acc2s = []
    losss: List[float] = []

    pprint(g.yen(0, 6, 32))
    pprint(g.yen(0, 8, 32))

    model.train()
    for epoch in (pbar := tqdm(range(40000))):

        chunk = list(trips.items())
        seq, sep = batch_trips(chunk, g)
        trips_input = pack_sequence(seq, enforce_sorted=False)
        lengths = model(trips_input)
        loss = bpr_loss_reverse(lengths, sep)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        weight = model.get_weight().flatten().tolist()
        acc = g.experiment_num(trips, weight)
        acc2 = g.experiment_cme(list(more_itertools.flatten(trips.values())), weight)
        accs.append(acc)
        acc2s.append(acc2)
        losss.append(loss.item())
        pbar.set_postfix(acc = acc, acc2 = acc2, loss = loss.item())

    plt.plot(accs)
    plt.plot(acc2s)
    plt.plot(losss)
    plt.show()

    pprint(g.yen(0, 6, 32, weight))
    pprint(g.yen(0, 8, 32, weight))
    print(weight)

def sa():
    import numpy as np
    import time

    cnt = 0
    def cost_func(p: np.ndarray):
        nonlocal cnt
        cnt += 1
        return - g.experiment_cme(list(more_itertools.flatten(trips.values())), p)

    from sko.GA import GA

    ga = GA(func=cost_func, n_dim=edges.shape[0], lb=[1] * edges.shape[0], ub=[10] * edges.shape[0], precision=1)
    start_time = time.time()
    best_x, best_y = ga.run()
    print(time.time() - start_time, cnt)
    print('best_x:', best_x, '\n', 'best_y:', best_y)

    pprint(g.yen(0, 6, 32, best_x))
    pprint(g.yen(0, 8, 32, best_x))

rower()