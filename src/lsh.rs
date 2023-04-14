use std::cmp::Reverse;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};

use crate::{Team, TeamId};
use rayon::prelude::*;

const HASH_COUNT: usize = 100;
const BAND_SIZE: usize = 2;
const SHINGLE_SIZE: usize = 6;

fn chunked_min_hash(document: &str) -> Vec<(usize, u64)> {
    // single hash function. for justification, see https://robertheaton.com/2014/05/02/jaccard-similarity-and-minhash-for-winners/
    // and http://web.eecs.utk.edu/~jplank/plank/classes/cs494/494/notes/Min-Hash/index.html
    let shingle_count = document.len() - SHINGLE_SIZE + 1;

    let mut heap = BinaryHeap::with_capacity(shingle_count);

    let mut hashes = vec![];
    for idx in 0..shingle_count {
        let shingle = &document[idx..idx + SHINGLE_SIZE];
        let mut hasher = DefaultHasher::new();
        shingle.hash(&mut hasher);
        let shingle_hash = hasher.finish();
        heap.push(Reverse(shingle_hash));
    }

    for _ in 0..HASH_COUNT {
        // try to gracefully handle shingle_count < HASH_COUNT situation. it should still work,
        // at least under certain conditions
        if heap.is_empty() {
            break;
        }
        hashes.push(heap.pop().unwrap().0);
    }

    hashes
        .chunks(BAND_SIZE)
        .map(|chunk| {
            let mut hasher = DefaultHasher::new();
            chunk.hash(&mut hasher);
            hasher.finish()
        })
        .enumerate()
        .collect()
}

pub fn string_shingles(document: &str) -> HashSet<u64> {
    let shingle_count = document.len() - SHINGLE_SIZE;
    let mut shingles = HashSet::new();
    for idx in 0..shingle_count {
        let shingle = &document[idx..idx + SHINGLE_SIZE];
        let mut hasher = DefaultHasher::new();
        shingle.hash(&mut hasher);
        let shingle_hash = hasher.finish();
        shingles.insert(shingle_hash);
    }
    shingles
}

pub fn jaccard_similarity(a: &HashSet<u64>, b: &HashSet<u64>) -> f32 {
    let intersection_cardinality = a.intersection(b).count();
    (intersection_cardinality as f32) / ((a.len() + b.len() - intersection_cardinality) as f32)
}

pub fn neighbor_similarities<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(
    query: &str,
    n: usize,
    matches: &HashSet<TeamId>,
    archive: &Archive<A, Fitness>,
) -> Vec<f32> {
    let query_shingles = string_shingles(query);
    let mut similar_matches: Vec<f32> = matches
        .par_iter()
        .map(|m| {
            let entry_index = archive.id_to_entries_index.get(m).unwrap();
            let entry = &archive.entries[*entry_index];
            let match_shingles = string_shingles(&entry.team.behavior_descriptor);
            jaccard_similarity(&query_shingles, &match_shingles)
        })
        .collect();
    similar_matches.sort_by(|a, b| b.partial_cmp(a).unwrap());
    if similar_matches.len() > n {
        similar_matches.resize(n, 0.0);
    }
    similar_matches
}

pub type Buckets = Vec<HashMap<u64, Vec<usize>>>;

pub fn initialize_buckets() -> Buckets {
    let mut buckets: Vec<HashMap<u64, Vec<usize>>> = vec![];

    let bucket_count = HASH_COUNT / BAND_SIZE;
    for _ in 0..bucket_count {
        buckets.push(HashMap::new());
    }
    buckets
}

pub fn index_teams<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(
    teams: &Vec<Team<A, Fitness>>,
    buckets: &mut Buckets,
) {
    let chunked_min_hashes: Vec<Vec<(usize, u64)>> = teams
        .par_iter()
        .map(|team| chunked_min_hash(&team.behavior_descriptor))
        .collect();

    let team_ids: Vec<usize> = teams.iter().map(|team| team.id).collect();

    for (chunked_min_hash, team_id) in chunked_min_hashes.iter().zip(team_ids.iter()) {
        for (bucket_index, min_hash) in chunked_min_hash.iter() {
            let bucket = &mut buckets[*bucket_index];
            bucket.entry(*min_hash).or_insert(vec![]).push(*team_id);
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct ArchiveEntry<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
> {
    pub team: Team<A, Fitness>,
    pub generation_added: usize,
}

#[derive(Debug)]
pub struct Archive<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
> {
    pub entries: Vec<ArchiveEntry<A, Fitness>>,
    pub id_to_entries_index: HashMap<TeamId, usize>,
    pub distance_cache: HashMap<(TeamId, TeamId), f32>,
}

pub fn search_index<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(
    archive: &Archive<A, Fitness>,
    buckets: &mut Buckets,
    query: &str,
    n: usize,
) -> Vec<f32> {
    let mut matches: HashSet<usize> = HashSet::new();
    let query_signature = chunked_min_hash(query);
    for (bucket_index, min_hash) in query_signature.iter() {
        let bucket = &mut buckets[*bucket_index];
        if let Some(b) = bucket.get(min_hash) {
            matches.extend(b);
        }
    }

    neighbor_similarities(query, n, &matches, archive)
}
