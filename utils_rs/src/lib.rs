use rusqlite::Connection;
use bincode::{serialize, deserialize};
use pyo3::prelude::*;
use rayon::prelude::*;
use pathfinding::prelude::{dijkstra_eid, yen_eid};
use ordered_float::OrderedFloat;
use std::collections::HashSet;


#[pyclass]
pub struct Sqlite {
    conn: Connection
}

impl Sqlite {
    fn init(db_path: &str) -> Connection {
        let conn = Connection::open(db_path).expect("Open-db failed");
        conn.execute_batch(
            "CREATE TABLE train (
                u       INTEGER,
                v       INTEGER,
                data    BLOB,
                PRIMARY KEY (u, v)
            );
            CREATE TABLE test (
                u       INTEGER,
                v       INTEGER,
                data    BLOB,
                PRIMARY KEY (u, v)
            );
            CREATE TABLE valid (
                u       INTEGER,
                v       INTEGER,
                data    BLOB,
                PRIMARY KEY (u, v)
            );"
        ).expect("Init-db failed");
        conn
    }

    fn serialize(samples: &Vec<Vec<usize>>) -> Vec<u8> {
        serialize(samples).expect("Serialization failed")
    }

    fn deserialize(blob: &Vec<u8>) -> Vec<Vec<usize>> {
        deserialize(blob).expect("Deserialize failed")
    }
}

#[pymethods]
impl Sqlite {
    #[new]
    fn new(db_path: &str, delete: Option<bool>) -> Self {
        let delete = match delete { Some(false) => false, _ => true };
        let conn = match (std::path::Path::new(db_path).exists(), delete) {
            (true, true) => {
                std::fs::remove_file(db_path).expect("Delete Failed");
                Self::init(db_path)
            },
            (true, false) => Connection::open(db_path).expect("Open-db failed"),
            (false, _) => Self::init(db_path)
        };
        Sqlite { conn }
    }

    pub fn insert_btyes(&mut self, table: &str, data: Vec<(usize, usize, Vec<u8>)>) {
        assert!(["train", "test", "valid"].contains(&table));
        let transaction = self.conn.transaction().expect("Initialize transaction failed");
        for data in data {
            transaction.execute(
                &format!("INSERT INTO {table} VALUES (?1, ?2, ?3)"),
                data,
            ).expect("Insert failed");
        }
        transaction.commit().expect("Transaction commit failed");
    }

    pub fn insert(&mut self, table: &str,  u: usize, v: usize, samples: Vec<Vec<usize>>) {
        assert!(["train", "test", "valid"].contains(&table));
        let blob = Self::serialize(&samples);
        self.insert_btyes(table, vec![(u, v, blob)])
    }

    pub fn get_bytes(&self, table: &str, u: usize, v: usize) -> Option<Vec<u8>> {
        assert!(["train", "test", "valid"].contains(&table));
        let mut stmt = self.conn.prepare(&format!("SELECT data FROM {table} WHERE u = ?1 AND v = ?2")).expect("Sql failed");
        let mut binding = stmt.query([u, v]).expect("Binding parameters failed");
        let rows = binding.next().expect(&format!("({u}, {v}) not found"))?;
        let blob: Vec<u8> = rows.get(0).unwrap();
        Some(blob)
    }

    pub fn get(&self, table: &str, u: usize, v: usize) -> Option<Vec<Vec<usize>>> {
        assert!(["train", "test", "valid"].contains(&table));
        let blob = self.get_bytes(table, u, v)?;
        let samples: Vec<Vec<usize>> = Self::deserialize(&blob);
        Some(samples)
    }
}

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

    pub fn path_sampling(&self, u: usize, v: usize, pos_samples: Vec<Vec<usize>>, k: usize, weight: Option<Vec<f64>>) -> Vec<Vec<usize>> {
        let weight = Self::determin_weight(&self.weight, &weight).expect("must specify weight");
        let mut filtered_edges: HashSet<usize> = HashSet::new();
        let pos_samples: HashSet<&Vec<usize>> = HashSet::from_iter(pos_samples.iter());
        let mut result = vec![];
        let successors = |n: &usize| {
            self.adjlist[*n].iter().map(|(t, edge_idx)| (*t, OrderedFloat(weight[*edge_idx]), *edge_idx))
        };
        let Some((path, _)) = dijkstra_eid(&u, successors, |p| *p == v) else { return vec![]; };
        filtered_edges.insert(path[path.len() / 2]);
        if !pos_samples.contains(&path) {
            result.push(path);
        }
        loop {
            let filtered_successor = |n: &usize| {
                successors(n)
                    .into_iter()
                    .filter(|(_, _, eid)| !filtered_edges.contains(eid))
                    .collect::<Vec<_>>()
            };
            if let Some((path, _)) = dijkstra_eid(&u, filtered_successor, |p| *p == v) {
                filtered_edges.insert(path[path.len() / 2]);
                if !pos_samples.contains(&path) {
                    result.push(path);
                    if result.len() >= k { break; }
                }
            } else { break; }
        }
        result
    }

    pub fn par_path_sampling_tosqlite(&self, uvs: Vec<(usize, usize)>, pos_samples: Vec<Vec<Vec<usize>>>, k: usize, chunk_size: usize, path: &str, table: &str, delete: bool) {
        let mut db = Sqlite::new(path, Some(delete));
        std::iter::zip(uvs.chunks(chunk_size), pos_samples.chunks(chunk_size)).for_each(|(uvs, pos_samples)| {
            let batch: Vec<Vec<u8>> = uvs.par_iter().zip(pos_samples).map(|(&(u, v), samples)| {
                Sqlite::serialize(&self.path_sampling(u, v, samples.clone(), k, None))
            }).collect();
            let data = uvs.iter().zip(batch).map(|(&(u, v), sample)| (u, v, sample)).collect();
            db.insert_btyes(table, data);
        });
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
    m.add_class::<Sqlite>()?;
    Ok(())
}