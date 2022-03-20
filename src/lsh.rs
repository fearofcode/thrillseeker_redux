use std::cmp::Reverse;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::{Hash, Hasher};

use rayon::prelude::*;

const HASH_COUNT: usize = 50;
const BAND_SIZE: usize = 2;
const SHINGLE_SIZE: usize = 4;

fn chunked_min_hash(document: &str) -> Vec<(usize, u64)> {
    // single hash function. for justification, see https://robertheaton.com/2014/05/02/jaccard-similarity-and-minhash-for-winners/
    // and http://web.eecs.utk.edu/~jplank/plank/classes/cs494/494/notes/Min-Hash/index.html
    let shingle_count = document.len() - SHINGLE_SIZE;

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

fn string_shingles(document: &str) -> HashSet<u64> {
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

fn jaccard_similarity(a: &HashSet<u64>, b: &HashSet<u64>) -> f32 {
    let intersection_cardinality = a.intersection(b).count();
    (intersection_cardinality as f32) / ((a.len() + b.len() - intersection_cardinality) as f32)
}

fn nearest_neighbors(
    query: &str,
    n: usize,
    matches: &HashSet<usize>,
    documents: &[String],
) -> Vec<(usize, f32)> {
    let query_shingles = string_shingles(query);
    let mut similar_matches: Vec<(usize, f32)> = matches
        .par_iter()
        .map(|m| {
            let document = &documents[*m];
            let match_shingles = string_shingles(document);
            let similarity = jaccard_similarity(&query_shingles, &match_shingles);
            (*m, similarity)
        })
        .collect();
    similar_matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    if similar_matches.len() > n {
        similar_matches.resize(n, (0, 0.0));
    }
    similar_matches
}

fn index_documents(documents: &mut Vec<String>) -> Vec<HashMap<u64, Vec<usize>>> {
    let mut buckets: Vec<HashMap<u64, Vec<usize>>> = vec![];

    let bucket_count = HASH_COUNT / BAND_SIZE;
    for _ in 0..bucket_count {
        buckets.push(HashMap::new());
    }

    let chunked_min_hashes: Vec<Vec<(usize, u64)>> = documents
        .par_iter()
        .map(|document| chunked_min_hash(document))
        .collect();

    for (document_index, chunked_min_hash) in chunked_min_hashes.iter().enumerate() {
        for (bucket_index, min_hash) in chunked_min_hash.iter() {
            let bucket = &mut buckets[*bucket_index];
            bucket
                .entry(*min_hash)
                .or_insert(vec![])
                .push(document_index);
        }
    }
    buckets
}

fn search_index(
    documents: &[String],
    buckets: &mut Vec<HashMap<u64, Vec<usize>>>,
    query: &str,
    n: usize,
) -> (HashSet<usize>, Vec<(usize, f32)>) {
    let mut matches: HashSet<usize> = HashSet::new();
    let query_signature = chunked_min_hash(query);
    for (bucket_index, min_hash) in query_signature.iter() {
        let bucket = &mut buckets[*bucket_index];
        if bucket.contains_key(min_hash) {
            matches.extend(&bucket[min_hash]);
        }
    }

    let top_neighbors = nearest_neighbors(query, n, &matches, documents);
    (matches, top_neighbors)
}