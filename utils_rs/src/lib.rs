use std::{cmp::Reverse, collections::{BinaryHeap, HashMap, HashSet}};
use rusqlite::Connection;
use bincode::{serialize, deserialize};
use pyo3::prelude::*;
use rayon::prelude::*;
use pathfinding::prelude::{dijkstra_eid, yen_eid};
use ordered_float::OrderedFloat;
use std::io::{Read, Write};


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
                length  INTEGER,
                data    BLOB,
                PRIMARY KEY (u, v)
            );
            CREATE TABLE test (
                u       INTEGER,
                v       INTEGER,
                length  INTEGER,
                data    BLOB,
                PRIMARY KEY (u, v)
            );
            CREATE TABLE valid (
                u       INTEGER,
                v       INTEGER,
                length  INTEGER,
                data    BLOB,
                PRIMARY KEY (u, v)
            );"
        ).expect("Init-db failed");
        conn
    }

    fn serialize(samples: &Vec<Vec<usize>>) -> (usize, Vec<u8>) {
        use flate2::Compression;
        use flate2::write::ZlibEncoder;
        let blob = serialize(samples).expect("Serialization failed");
        let length = blob.len();
        let mut e = ZlibEncoder::new(Vec::new(), Compression::default());
        e.write_all(&blob).expect("Compress failed");
        (length, e.finish().expect("Finish ZlibEncoder failed"))
    }

    fn deserialize(length: usize, c: &Vec<u8>) -> Vec<Vec<usize>> {
        use flate2::read::ZlibDecoder;
        let mut d = ZlibDecoder::new(&**c);
        let mut blob = vec![0; length];
        d.read(&mut blob).expect("Decompress failed");
        deserialize(&blob).expect("Deserialize failed")
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

    pub fn insert_btyes(&mut self, table: &str, data: Vec<(usize, usize, usize, Vec<u8>)>) {
        assert!(["train", "test", "valid"].contains(&table));
        let transaction = self.conn.transaction().expect("Initialize transaction failed");
        for data in data {
            transaction.execute(
                &format!("INSERT INTO {table} VALUES (?1, ?2, ?3, ?4)"),
                data,
            ).expect("Insert failed");
        }
        transaction.commit().expect("Transaction commit failed");
    }

    pub fn insert(&mut self, table: &str,  u: usize, v: usize, samples: Vec<Vec<usize>>) {
        assert!(["train", "test", "valid"].contains(&table));
        let (length, blob) = Self::serialize(&samples);
        self.insert_btyes(table, vec![(u, v, length, blob)])
    }

    pub fn get_bytes(&self, table: &str, u: usize, v: usize) -> Option<(usize, Vec<u8>)> {
        assert!(["train", "test", "valid"].contains(&table));
        let mut stmt = self.conn.prepare(&format!("SELECT length, data FROM {table} WHERE u = ?1 AND v = ?2")).expect("Sql failed");
        let mut binding = stmt.query([u, v]).expect("Binding parameters failed");
        let rows = binding.next().expect(&format!("({u}, {v}) not found"))?;
        let length: usize = rows.get(0).unwrap();
        let blob: Vec<u8> = rows.get(1).unwrap();
        Some((length, blob))
    }

    pub fn get(&self, table: &str, u: usize, v: usize) -> Option<Vec<Vec<usize>>> {
        assert!(["train", "test", "valid"].contains(&table));
        let (length, blob) = self.get_bytes(table, u, v)?;
        let samples: Vec<Vec<usize>> = Self::deserialize(length, &blob);
        Some(samples)
    }

    pub fn keys(&self, table: &str) -> Vec<(usize, usize)> {
        let mut stmt = self.conn.prepare(&format!("SELECT u, v FROM {table}")).expect("Sql failed");
        let person_iter = stmt.query_map([], |row| {
            let (u, v): (usize, usize) = (row.get(0).unwrap(), row.get(1).unwrap());
            Ok((u, v))
        }).expect("Binding parameters fails");
        person_iter.map(Result::unwrap).collect()
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
        while let (Some(Reverse((OrderedFloat(df), f))), Some(Reverse((OrderedFloat(db), b)))) = (pq_f.pop(), pq_b.pop()) {
            for &(s, eid) in &self.radjlist[b] {
                let new_dist = OrderedFloat(db + weight[eid]);
                if new_dist < dis_b[s] {
                    dis_b[s] = new_dist;
                    prev_b[s] = Some(eid);
                    pq_b.push(Reverse((new_dist, s)));
                }
            }
            for &(t, eid) in &self.adjlist[f] {
                let new_dist = OrderedFloat(df + weight[eid]);
                if new_dist < dis_f[t] {
                    dis_f[t] = new_dist;
                    prev_f[t] = Some(eid);
                    pq_f.push(Reverse((new_dist, t)));
                }
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

    pub fn par_bidirectional_dijkstra_tosqlite(&self, uvs: Vec<(usize, usize)>, pos_samples: Vec<Vec<Vec<usize>>>, k: usize, chunk_size: usize, path: &str, table: &str, delete: bool, callback: Option<Py<PyAny>>) {
        let mut db = Sqlite::new(path, Some(delete));
        std::iter::zip(uvs.chunks(chunk_size), pos_samples.chunks(chunk_size)).for_each(|(uvs, pos_samples)| {
            let batch: Vec<(usize, Vec<u8>)> = uvs.par_iter().zip(pos_samples).map(|(&(u, v), samples)| {
                Sqlite::serialize(&self.bidirectional_dijkstra(samples.clone(), u, v, k, None))
            }).collect();
            let data = uvs.iter().zip(batch).map(|(&(u, v), (length, sample))| (u, v, length, sample)).collect();
            db.insert_btyes(table, data);
            if let Some(f) = &callback {
                Python::with_gil(|py| {
                    f.call1(py, (uvs.len(),)).unwrap();
                });
            }
        });
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
}

#[pymodule]
fn utils_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DiGraph>()?;
    m.add_class::<Sqlite>()?;
    Ok(())
}