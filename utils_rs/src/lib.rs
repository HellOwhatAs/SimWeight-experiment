use std::collections::HashSet;
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
}


#[pymethods]
impl DiGraph {
    #[new]
    fn new(n: usize, edges: Vec<(usize, usize)>, weight: Option<Vec<f64>>) -> Self {
        let mut adjlist = vec![vec![]; n];
        for (idx, (s, t)) in edges.iter().enumerate() {
            adjlist[*s].push((*t, idx));
        }
        Self { n, edges, adjlist, weight }
    }

    pub fn dijkstra(&self, u: usize, v: usize, weight: Option<Vec<f64>>) -> Option<(Vec<usize>, f64)> {
        let weight = Self::determin_weight(&self.weight, &weight).expect("must specify weight");
        let successors = |n: &usize| {
            self.adjlist[*n].iter().map(|(t, edge_idx)| (*t, OrderedFloat(weight[*edge_idx]), *edge_idx))
        };
        let (path, cost) = dijkstra_eid(&u, successors, |p| *p == v).unwrap();
        Some((path, cost.into()))
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