# SimWeight

## Setup

```
python -m venv .venv
.\.venv\Scripts\activate
(.venv) pip install -r requirements.txt
(.venv) cd .\utils_rs\
(.venv) maturin.exe develop -r
```

## Roadmap

### Discussion 2024.1.23

- use float weight for edges
- sample k-shortset paths and drop positive samples by comparing list of edge-id
- have to write shortest path algorithm and k-shortest path algorithm that outputs edge-id
- Eq(7) in paper is wrong.  
  bpr loss is
  $$-\sum_{(i,j)\in G}{\ln  \sigma(\mathrm{length}(j) - \mathrm{length}(i))}$$
  where  
  $$i \text{ is positive path sample}$$  
  $$j \text{ is negitive path sample}$$  
  $$\mathrm{length}(i) = \sum_{e_k \in i} \mathrm{w}(e_k)$$

## TODO
1. negative sampling
   - k shortest path
     - no circle: Yen's Algorithm
     - allow circle: meaningless
   - otherwise?
     > 负样本：通过双向dijkstra算法选出200条路线，代表距离或时间权路网中的最优、次优路线（这个有论文证明合理性）。
