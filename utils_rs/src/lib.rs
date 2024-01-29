use std::{cmp::Reverse, collections::{BinaryHeap, HashSet}};
use pyo3::prelude::*;
use rayon::prelude::*;
use pathfinding::prelude::{dijkstra_eid, yen_eid};
use ordered_float::OrderedFloat;

#[pyclass]
struct DiGraph {
    #[pyo3(get)]
    n: usize,
    #[pyo3(get)]
    edges: Vec<(usize, usize)>,
    #[pyo3(get)]
    adjlist: Vec<Vec<(usize, usize)>>,
    radjlist: Vec<Vec<(usize, usize)>>,
    #[pyo3(get)]
    weight: Option<Vec<f64>>
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
    fn remove_loops(&self, path: Vec<usize>) -> Vec<usize> {
        let mut vis = HashSet::new();
        let mut res = vec![];
        if let Some(eid) = path.first() {
            vis.insert(self.edges[*eid].0);
        } else { return res; }
        for eid in path {
            let t = self.edges[eid].1;
            if !vis.contains(&t) {
                res.push(eid);
                vis.insert(t);
            }
            else {
                while let Some(eid) = res.last() {
                    if self.edges[*eid].1 != t { res.pop().unwrap(); }
                    else { break; }
                }
            }
        }
        res
    }
}


#[pymethods]
impl DiGraph {
    #[new]
    fn new(n: usize, edges: Vec<(usize, usize)>, weight: Option<Vec<f64>>) -> Self {
        let mut adjlist = vec![vec![]; n];
        let mut radjlist = vec![vec![]; n];
        for (idx, (s, t)) in edges.iter().enumerate() {
            adjlist[*s].push((*t, idx));
            radjlist[*t].push((*s, idx));
        }
        Self { n, edges, adjlist, radjlist, weight }
    }

    pub fn dijkstra(&self, u: usize, v: usize, weight: Option<Vec<f64>>) -> Option<(Vec<usize>, f64)> {
        let weight = Self::determin_weight(&self.weight, &weight).expect("must specify weight");
        let successors = |n: &usize| {
            self.adjlist[*n].iter().map(|(t, edge_idx)| (*t, OrderedFloat(weight[*edge_idx]), *edge_idx))
        };
        let (path, cost) = dijkstra_eid(&u, successors, |p| *p == v).unwrap();
        Some((path, cost.into()))
    }

    pub fn bidirectional_dijkstra(&self, positive_samples: Vec<Vec<usize>>, u: usize, v: usize, k: usize, weight: Option<Vec<f64>>) -> Vec<Vec<usize>> {
        let weight = Self::determin_weight(&self.weight, &weight).expect("must specify weight");
        let (mut dis_f, mut dis_b) = (vec![OrderedFloat(f64::INFINITY); self.n], vec![OrderedFloat(f64::INFINITY); self.n]);
        dis_f[u] = OrderedFloat(0.0); dis_b[v] = OrderedFloat(0.0);
        let (mut prev_f, mut prev_b) = (vec![None; self.n], vec![None; self.n]);
        let (mut pq_f, mut pq_b) = (BinaryHeap::from([Reverse((OrderedFloat(0.0), u))]), BinaryHeap::from([Reverse((OrderedFloat(0.0), v))]));
        let mut mu = OrderedFloat(f64::INFINITY);
        while let (Some(Reverse((OrderedFloat(df), f))), Some(Reverse((OrderedFloat(db), b)))) = (pq_f.pop(), pq_b.pop()) {
            for &(s, eid) in &self.radjlist[b] {
                let new_dist = OrderedFloat(db + weight[eid]);
                if new_dist < dis_b[s] {
                    dis_b[s] = new_dist;
                    prev_b[s] = Some(eid);
                    pq_b.push(Reverse((new_dist, s)));
                }
                mu = std::cmp::min(mu, new_dist + dis_f[s]);
            }
            for &(t, eid) in &self.adjlist[f] {
                let new_dist = OrderedFloat(df + weight[eid]);
                if new_dist < dis_f[t] {
                    dis_f[t] = new_dist;
                    prev_f[t] = Some(eid);
                    pq_f.push(Reverse((new_dist, t)));
                }
                mu = std::cmp::min(mu, new_dist + dis_b[t]);
            }
        }
        let mut kmin = BinaryHeap::from_iter((0..self.n).map(|idx| Reverse((dis_f[idx] + dis_b[idx], idx))));
        let mut res = HashSet::new();
        let filter: HashSet<&Vec<usize>> = HashSet::from_iter(positive_samples.iter());
        while let Some(Reverse((_, idx))) = kmin.pop() {
            if res.len() >= k { break; }
            let mut path = self.extract_path(idx, &prev_f);
            let mut path2 = self.rextract_path(idx, &prev_b);
            path2.reverse();
            path.append(&mut path2);
            path = self.remove_loops(path);
            if !filter.contains(&path) {
                res.insert(path);
            }
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

    pub fn yen_drop(&self, positive_samples: Vec<Vec<usize>>, u: usize, v: usize, k: usize, weight: Option<Vec<f64>>) -> Vec<(Vec<usize>, f64)> {
        let positive_samples_set: HashSet<&Vec<usize>> = HashSet::from_iter(positive_samples.iter());
        self.yen(u, v, k, weight).into_iter().filter(|(positive_path, _)| !positive_samples_set.contains(positive_path)).collect()
    }

    pub fn experiment(&self, trips: Vec<Vec<usize>>, weight: Option<Vec<f64>>) -> usize {
        let weight = Self::determin_weight(&self.weight, &weight).expect("must specify weight");
        let successors = |n: &usize| {
            self.adjlist[*n].iter().map(|(t, edge_idx)| (*t, OrderedFloat(weight[*edge_idx]), *edge_idx))
        };
        trips.into_par_iter().map(|trip| {
            let (pred, _) = dijkstra_eid(&self.edges[*trip.first().unwrap()].0, successors, |p| *p == self.edges[*trip.last().unwrap()].1).unwrap();
            (pred == trip) as usize
        }).sum()
    }
}

#[pymodule]
fn utils_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DiGraph>()?;
    Ok(())
}