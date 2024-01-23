use pyo3::prelude::*;
use rayon::prelude::*;
use pathfinding::directed::dijkstra::dijkstra_eid;
use ordered_float::OrderedFloat;

#[pyclass]
struct DiGraph {
    n: usize,
    edges: Vec<(usize, usize)>,
    adjlist: Vec<Vec<(usize, usize)>>
}

#[pymethods]
impl DiGraph {
    #[new]
    fn new(n: usize, edges: Vec<(usize, usize)>) -> Self {
        let mut adjlist = vec![vec![]; n];
        for (idx, (s, t)) in edges.iter().enumerate() {
            adjlist[*s].push((*t, idx));
        }
        Self { n, edges, adjlist }
    }
    #[getter]
    fn n(&self) -> PyResult<usize> {
        Ok(self.n)
    }
    #[getter]
    fn edges(&self) -> PyResult<Vec<(usize, usize)>> {
        Ok(self.edges.clone())
    }
    #[getter]
    fn adjlist(&self) -> PyResult<Vec<Vec<(usize, usize)>>> {
        Ok(self.adjlist.clone())
    }
    pub fn experiment(&self, weight: Vec<f64>, trips: Vec<Vec<usize>>) -> usize {
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