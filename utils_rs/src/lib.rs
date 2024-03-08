use std::{cmp::Reverse, collections::{BinaryHeap, HashMap, HashSet}};
use pyo3::prelude::*;
use rayon::prelude::*;
use pathfinding::prelude::{dijkstra_eid, yen_eid};
use ordered_float::OrderedFloat;
use geo::{HaversineDistance, Point};

#[pyclass]
pub struct DiGraph {
    #[pyo3(get)]
    n: usize,
    edges: Vec<(usize, usize)>,
    #[pyo3(get)]
    adjlist: Vec<Vec<(usize, usize)>>,
    radjlist: Vec<Vec<(usize, usize)>>,
    #[pyo3(get, set)]
    weight: Option<Vec<f64>>,
    pos: Option<Vec<Point<f64>>>
}

impl DiGraph {
    fn determin_weight<'a>(
        self_weight: &'a Option<Vec<f64>>, weight: &'a Option<Vec<f64>>) -> Option<&'a Vec<f64>> {
        match (self_weight, weight) {
            (_, Some(w)) => Some(w),
            (Some(w), None) => Some(w),
            (None, None) => None,
        }
    }
    fn extract_path(&self, start: usize, prev: &Vec<Option<usize>>) -> Vec<usize> {
        let eid = match prev[start] { Some(eid) => eid, _ => return vec![] };
        let next = self.edges[eid].0;
        let mut path = self.extract_path(next, prev);
        path.push(eid);
        path
    }
    fn rextract_path(&self, start: usize, prev: &Vec<Option<usize>>) -> Vec<usize> {
        let eid = match prev[start] { Some(eid) => eid, _ => return vec![] };
        let next = self.edges[eid].1;
        let mut path = self.rextract_path(next, prev);
        path.push(eid);
        path
    }
    fn remove_loop(&self, path: Vec<usize>) -> Vec<usize> {
        assert!(!path.is_empty(), "not empty path");
        let mut vis = HashMap::new();
        vis.insert(self.edges[path[0]].0, 0);
        let mut res = vec![];
        for eid in path.into_iter() {
            let (_, v) = self.edges[eid];
            vis.entry(v)
                .and_modify(|e| {
                    res.truncate(*e);
                })
                .or_insert_with(|| {
                    res.push(eid);
                    res.len()
                });
        }
        res
    }
    fn doit(&self, idx: usize, res: &mut HashSet<Vec<usize>>, prev_f: &Vec<Option<usize>>, prev_b: &Vec<Option<usize>> ) {
        let mut path = self.extract_path(idx, prev_f);
        let mut path2 = self.rextract_path(idx, prev_b);
        path2.reverse();
        path.append(&mut path2);
        let path = self.remove_loop(path);
        if !path.is_empty() {
            res.insert(path);
        }
    }
}


#[pymethods]
impl DiGraph {
    #[new]
    pub fn new(n: usize, edges: Vec<(usize, usize)>, weight: Option<Vec<f64>>, pos: Option<Vec<(f64, f64)>>) -> Self {
        let mut adjlist = vec![vec![]; n];
        let mut radjlist = vec![vec![]; n];
        for (idx, (s, t)) in edges.iter().enumerate() {
            adjlist[*s].push((*t, idx));
            radjlist[*t].push((*s, idx));
        }
        let pos = match pos {
            Some(pos) => Some(pos.into_iter().map(|(x, y)| Point::new(x, y)).collect()),
            _ => None
        };
        Self { n, edges, adjlist, radjlist, weight, pos }
    }

    pub fn dijkstra(&self, u: usize, v: usize, weight: Option<Vec<f64>>) -> Option<(Vec<usize>, f64)> {
        let weight = Self::determin_weight(&self.weight, &weight).expect("must specify weight");
        let successors = |n: &usize| {
            self.adjlist[*n].iter().map(|(t, edge_idx)| (*t, OrderedFloat(weight[*edge_idx]), *edge_idx))
        };
        let (path, cost) = dijkstra_eid(&u, successors, |p| *p == v).unwrap();
        Some((path, cost.into()))
    }

    pub fn bidirectional_dijkstra(&self, u: usize, v: usize, k: usize, weight: Option<Vec<f64>>) -> Vec<Vec<usize>> {
        let weight = Self::determin_weight(&self.weight, &weight).expect("must specify weight");
        let (mut dis_f, mut dis_b) = (vec![OrderedFloat(f64::INFINITY); self.n], vec![OrderedFloat(f64::INFINITY); self.n]);
        dis_f[u] = OrderedFloat(0.0); dis_b[v] = OrderedFloat(0.0);
        let (mut vis_f, mut vis_b) = (vec![false; self.n], vec![false; self.n]);
        let (mut prev_f, mut prev_b) = (vec![None; self.n], vec![None; self.n]);
        let (mut pq_f, mut pq_b) = (BinaryHeap::from([Reverse((OrderedFloat(0.0), u))]), BinaryHeap::from([Reverse((OrderedFloat(0.0), v))]));
        
        let mut res = HashSet::new();
        while let (Some(Reverse((_, f))), Some(Reverse((_, b)))) = (pq_f.pop(), pq_b.pop()) {
            if !vis_b[b] {
                vis_b[b] = true;
                if vis_f[b] {
                    self.doit(b, &mut res, &prev_f, &prev_b);
                }
                for &(s, eid) in &self.radjlist[b] {
                    if vis_b[s] { continue; }
                    let new_dist = OrderedFloat(dis_b[b].0 + weight[eid]);
                    if new_dist < dis_b[s] {
                        dis_b[s] = new_dist;
                        prev_b[s] = Some(eid);
                        pq_b.push(Reverse((new_dist, s)));
                    }
                }    
            }
            
            if !vis_f[f] {
                vis_f[f] = true;
                if vis_b[f] { 
                    self.doit(f, &mut res, &prev_f, &prev_b);
                }
                for &(t, eid) in &self.adjlist[f] {
                    if vis_f[t] { continue; }
                    let new_dist = OrderedFloat(dis_f[f].0 + weight[eid]);
                    if new_dist < dis_f[t] {
                        dis_f[t] = new_dist;
                        prev_f[t] = Some(eid);
                        pq_f.push(Reverse((new_dist, t)));
                    }
                }
            }

            if res.len() >= k { break; }
        }

        res.into_iter().collect()
    }

    pub fn bidirectional_astar(&self, u: usize, v: usize, k: usize, weight: Option<Vec<f64>>) -> Vec<Vec<usize>> {
        assert!(self.pos.is_some(), "A-star requires pos");

        let pos = self.pos.as_ref().unwrap();
        let mut cache = vec![vec![None; self.n]; self.n];
        let mut h = |u: usize, v: usize| -> f64 {
            *cache[u][v].get_or_insert_with(|| pos[u].haversine_distance(&pos[v]))
        };

        let weight = Self::determin_weight(&self.weight, &weight).expect("must specify weight");
        let (mut dis_f, mut dis_b) = (vec![OrderedFloat(f64::INFINITY); self.n], vec![OrderedFloat(f64::INFINITY); self.n]);
        {
            let huv = h(u, v);
            dis_f[u] = OrderedFloat(0.0 + huv); dis_b[v] = OrderedFloat(0.0 + huv);
        }
        let (mut vis_f, mut vis_b) = (vec![false; self.n], vec![false; self.n]);
        let (mut prev_f, mut prev_b) = (vec![None; self.n], vec![None; self.n]);
        let (mut pq_f, mut pq_b) = (BinaryHeap::from([Reverse((OrderedFloat(0.0), u))]), BinaryHeap::from([Reverse((OrderedFloat(0.0), v))]));
        
        let mut res = HashSet::new();
        while let (Some(Reverse((_, f))), Some(Reverse((_, b)))) = (pq_f.pop(), pq_b.pop()) {
            if !vis_b[b] {
                vis_b[b] = true;
                if vis_f[b] {
                    self.doit(b, &mut res, &prev_f, &prev_b);
                }
                for &(s, eid) in &self.radjlist[b] {
                    if vis_b[s] { continue; }
                    let new_dist = OrderedFloat(dis_b[b].0 + weight[eid]);
                    if new_dist < dis_b[s] {
                        dis_b[s] = new_dist;
                        prev_b[s] = Some(eid);
                        pq_b.push(Reverse((new_dist + h(u, s), s)));
                    }
                }    
            }
            
            if !vis_f[f] {
                vis_f[f] = true;
                if vis_b[f] { 
                    self.doit(f, &mut res, &prev_f, &prev_b);
                }
                for &(t, eid) in &self.adjlist[f] {
                    if vis_f[t] { continue; }
                    let new_dist = OrderedFloat(dis_f[f].0 + weight[eid]);
                    if new_dist < dis_f[t] {
                        dis_f[t] = new_dist;
                        prev_f[t] = Some(eid);
                        pq_f.push(Reverse((new_dist + h(t, v), t)));
                    }
                }
            }

            if res.len() >= k { break; }
        }

        res.into_iter().collect()
    }

    pub fn yen(&self, u: usize, v: usize, k: usize, weight: Option<Vec<f64>>) -> Vec<(Vec<usize>, f64)> {
        let weight = Self::determin_weight(&self.weight, &weight).expect("must specify weight");
        let successors = |n: &usize| {
            self.adjlist[*n].iter().map(|(t, edge_idx)| (*t, OrderedFloat(weight[*edge_idx]), *edge_idx))
        };
        let eid2uvc = |eid: usize| (
            self.edges[eid].0,
            self.edges[eid].1,
            OrderedFloat(weight[eid])
        );
        yen_eid( &u, successors, eid2uvc, |p| *p == v, k)
            .into_iter()
            .map(|(path, cost)| (path, cost.into()))
            .collect()
    }

    pub fn par_yen(&self, chunk: Vec<((usize, usize), Vec<Vec<usize>>)>) -> Vec<Vec<Vec<usize>>> {
        chunk.into_par_iter().map(|((u, v), samples)| {
            let k = samples.len();
            self.yen(u, v, k, None).into_iter().map(|(p, _)| p).collect()
        }).collect()
    }
    
    pub fn par_bidirectional_dijkstra(&self, chunk: Vec<(usize, usize, usize)>) -> Vec<Vec<Vec<usize>>> {
        chunk.into_par_iter().map(|(u, v, k)| {
            self.bidirectional_dijkstra(u, v, k, None)
        }).collect()
    }

    pub fn par_bidirectional_astar(&self, chunk: Vec<(usize, usize, usize)>) -> Vec<Vec<Vec<usize>>> {
        chunk.into_par_iter().map(|(u, v, k)| {
            self.bidirectional_astar(u, v, k, None)
        }).collect()
    }

    pub fn experiment(&self, trips: HashMap<(usize, usize), Vec<Vec<usize>>>, weight: Option<Vec<f64>>) -> usize {
        let weight = Self::determin_weight(&self.weight, &weight).expect("must specify weight");
        let successors = |n: &usize| {
            self.adjlist[*n].iter().map(|(t, edge_idx)| (*t, OrderedFloat(weight[*edge_idx]), *edge_idx))
        };
        trips.into_par_iter().map(|((u, v), trips)| {
            let trips: HashSet<&Vec<usize>> = HashSet::from_iter(trips.iter());
            let (pred, _) = dijkstra_eid(&u, successors, |p| *p == v).unwrap();
            trips.contains(&pred) as usize
        }).sum()
    }

    pub fn experiment_num(&self, trips: HashMap<(usize, usize), Vec<Vec<usize>>>, weight: Option<Vec<f64>>) -> usize {
        let weight = Self::determin_weight(&self.weight, &weight).expect("must specify weight");
        let successors = |n: &usize| {
            self.adjlist[*n].iter().map(|(t, edge_idx)| (*t, OrderedFloat(weight[*edge_idx]), *edge_idx))
        };
        trips.into_par_iter().map(|((u, v), trips)| {
            let trips: HashSet<OrderedFloat<f64>> = HashSet::from_iter(trips.iter().map(|path| path.iter().map(|eid| OrderedFloat(weight[*eid])).sum()));
            let (_, f) = dijkstra_eid(&u, successors, |p| *p == v).unwrap();
            trips.contains(&f) as usize
        }).sum()
    }
    
    pub fn experiment_old(&self, trips: Vec<Vec<usize>>, weight: Option<Vec<f64>>) -> usize {
        let weight = Self::determin_weight(&self.weight, &weight).expect("must specify weight");
        let successors = |n: &usize| {
            self.adjlist[*n].iter().map(|(t, edge_idx)| (*t, OrderedFloat(weight[*edge_idx]), *edge_idx))
        };
        trips.into_par_iter().map(|trip| {
            let (pred, _) = dijkstra_eid(&self.edges[*trip.first().unwrap()].0, successors, |p| *p == self.edges[*trip.last().unwrap()].1).unwrap();
            (pred == trip) as usize
        }).sum()
    }

    pub fn experiment_cme(&self, trips: Vec<Vec<usize>>, weight: Option<Vec<f64>>) -> usize {
        let weight = Self::determin_weight(&self.weight, &weight).expect("must specify weight");
        let successors = |n: &usize| {
            self.adjlist[*n].iter().map(|(t, edge_idx)| (*t, OrderedFloat(weight[*edge_idx]), *edge_idx))
        };
        trips.into_par_iter().map(|trip| {
            let d: OrderedFloat<f64> = trip.iter().map(|eid| OrderedFloat(weight[*eid])).sum();
            let (_, f) = dijkstra_eid(&self.edges[*trip.first().unwrap()].0, successors, |p| *p == self.edges[*trip.last().unwrap()].1).unwrap();
            (d == f) as usize
        }).sum()
    }
}

#[pymodule]
fn utils_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DiGraph>()?;
    Ok(())
}