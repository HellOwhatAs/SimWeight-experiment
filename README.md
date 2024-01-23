# SimWeight

## Setup

```
python -m venv .venv
.\.venv\Scripts\activate
(.venv) pip install -r requirements.txt
(.venv) cd .\weight_experiment\
(.venv) maturin.exe develop -r
```

## Roadmap

> [!NOTE] Discussion 2024.1.23
> - use float weight for edges
> - sample k-shortset paths and drop positive samples by comparing list of edge-id
> - have to write shortest path algorithm and k-shortest path algorithm that outputs edge-id
> - Eq(7) in paper is wrong.  
>   bpr loss is
>   $$-\sum_{(i,j)\in G}{\ln  \sigma(\mathrm{length}(j) - \mathrm{length}(i))}$$
>   where  
>   $$i \text{ is positive path sample} \\
>     j \text{ is negitive path sample} \\
>     \mathrm{length}(i) = \sum_{e_k \in i} \mathrm{w}(e_k)$$