# SimWeight

## Setup
[Python3](www.python.org) and [Rust](https://www.rust-lang.org/)
```
python -m venv .venv
.venv/Scripts/activate
(.venv) pip install -r requirements.txt
(.venv) cd utils_rs
(.venv) maturin develop -r
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

### Discussion 2024.1.25

- give up Yen's Algorithm due to it's time complexity

### Discussion 2024.1.27
- the time complexity of bidirection dijkstra is equal to dijkstra 😨

### Discussion 2024.2.21
- use test dataset to evaluate the weight

### Discussion 2024.3.10
- a explaination of low accuracy: didi online taxi is different from taxi. Online taxi follows navigation most of the time (thanks to Yongbao Song)
- involve other evaluation standard, eg. top3
- start working on the paper

## TODO
- exist multiple shortest paths: does it match the requirement?
- involve other evaluation standard, eg. top3
  |top(k)|length|ROWER|
  |:---: |:----:|:---:|
  |top3  |0.31  |0.43 |
  |top5  |0.34  |0.46 |
  |top10 |0.37  |0.51 |
  |top15 |0.40  |0.53 |
- as a python package that solves "Sim-Weight Estimation Problem"
  with exampe data generator based on hidden weight
- better dataset?
  using osmnx: https://zhuanlan.zhihu.com/p/613801546
- need baselines (not restricted on edge weight)