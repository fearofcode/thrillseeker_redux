use std::cmp::{max, min, Reverse};
use std::collections::hash_map::DefaultHasher;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::iter::StepBy;
use std::ops::Range;

use rayon::prelude::*;

const HASH_COUNT: usize = 50;
const BAND_SIZE: usize = 2;
// enough for three positions in ant trail
const SHINGLE_SIZE: usize = 4 * 3 + 2;
// one position in ant trail
const SHINGLE_STRIDE: usize = 4 + 1;

// without the following types, the types become confusing.
#[derive(Hash, Eq, PartialEq, Ord, PartialOrd, Copy, Clone, Debug)]
struct HashCode(u64);

#[derive(Debug, Hash, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct TeamRecord {
    pub(crate) team_id: u64,
    pub(crate) behavior_descriptor: String
}

#[derive(Debug, Hash, Clone)]
pub struct Bucket(pub HashMap<HashCode, Vec<usize>>);

#[derive(Debug, Hash, Clone)]
pub struct BandedBucket(pub Vec<Bucket>);

#[derive(Debug, Hash, Clone)]
pub struct Archive(pub HashMap<usize, (usize, String)>);

fn chunked_min_hash(document: &str) -> Vec<(usize, HashCode)> {
    // single hash function. for justification, see https://robertheaton.com/2014/05/02/jaccard-similarity-and-minhash-for-winners/
    // and http://web.eecs.utk.edu/~jplank/plank/classes/cs494/494/notes/Min-Hash/index.html
    let shingle_count = document_shingle_count(document);

    let mut heap = BinaryHeap::with_capacity(shingle_count);

    let mut hashes = vec![];
    for idx in shingle_iterator(document) {
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
            HashCode(hasher.finish())
        })
        .enumerate()
        .collect()
}

fn shingle_iterator(document: &str) -> StepBy<Range<usize>> {
    (0..(document.len() - SHINGLE_SIZE)).step_by(SHINGLE_STRIDE)
}

fn document_shingle_count(document: &str) -> usize {
    if (document.len() - SHINGLE_SIZE) % SHINGLE_STRIDE == 0 {
        (document.len() - SHINGLE_SIZE) / SHINGLE_STRIDE
    } else {
        // document len 99, shingle size 5, shingle count = 99 // 5 + 1 == 19 + 1 == 20
        (document.len() - SHINGLE_SIZE) / SHINGLE_STRIDE + 1
    }
}

fn string_shingles(document: &str) -> HashSet<HashCode> {
    let mut shingles = HashSet::new();
    for idx in shingle_iterator(document) {
        let shingle = &document[idx..idx + SHINGLE_SIZE];
        let mut hasher = DefaultHasher::new();
        shingle.hash(&mut hasher);
        let shingle_hash = HashCode(hasher.finish());
        shingles.insert(shingle_hash);
    }
    shingles
}

fn jaccard_similarity(a: &HashSet<HashCode>, b: &HashSet<HashCode>) -> f32 {
    let intersection_cardinality = a.intersection(b).count();
    (intersection_cardinality as f32) / ((a.len() + b.len() - intersection_cardinality) as f32)
}

pub fn index_documents(archive: &Archive) -> BandedBucket {
    let mut buckets = empty_buckets();

    let chunked_min_hashes: Vec<Vec<(usize, HashCode)>> = archive.0
        .par_iter()
        .map(|(_team_id, (_team_index, descriptor))| chunked_min_hash(&descriptor))
        .collect();

    for (document_index, chunked_min_hash) in chunked_min_hashes.iter().enumerate() {
        let team_record = archive.0[&document_index].clone();
        for (bucket_index, min_hash) in chunked_min_hash.iter() {
            let bucket = &mut buckets.0[*bucket_index].0;
            bucket
                .entry(*min_hash)
                .or_insert(vec![])
                .push(team_record.clone());
        }
    }
    buckets
}

pub fn empty_buckets() -> BandedBucket {
    let mut buckets: BandedBucket = BandedBucket(vec![]);

    let bucket_count = HASH_COUNT / BAND_SIZE;
    for _ in 0..bucket_count {
        buckets.0.push(Bucket(HashMap::new()));
    }
    buckets
}

pub fn search_index(
    archive: &Archive,
    buckets: &mut BandedBucket,
    query: &TeamRecord,
    similarity_cache: &mut HashMap<(u64, u64), f32>,
    n: usize,
    max_similarity: f32
) -> Vec<(usize, f32)> {
    let mut matches: HashSet<TeamRecord> = HashSet::new();
    let query_signature = chunked_min_hash(&query.behavior_descriptor);
    for (bucket_index, min_hash) in query_signature.iter() {
        let bucket = &mut buckets.0[*bucket_index].0;
        if bucket.contains_key(min_hash) {
            let bucket_matches = &bucket[min_hash];
            matches.extend(bucket_matches);
        }
    }

    nearest_neighbors(query, n, &matches, archive, similarity_cache, max_similarity)
}


fn nearest_neighbors(
    query: &TeamRecord,
    n: usize,
    matches: &HashSet<TeamRecord>,
    archive: &Archive,
    similarity_cache: &mut HashMap<(u64, u64), f32>,
    max_similarity: f32
) -> Vec<(usize, f32)> {
    if matches.is_empty() {
        return vec![];
    }
    let team_id = query.team_id;
    let query_shingles = string_shingles(query);
    let mut similar_matches: Vec<(usize, f32)> = matches
        // .par_iter() TODO does this help? it would make caching with similarity_cache harder :(
        // @Performance determine if par_iter() would be faster
        .iter()
        // filter out ourselves
        .filter(|match_id| match_id.clone() != team_id)
        .map(|match_index| {
            let key: (u64, u64) = (
                min(team_id, *match_index.team_id),
                max(team_id, *match_index.team_id),
            );
            if similarity_cache.contains_key(&key) {
                return (*match_index, *similarity_cache.get(&key).unwrap());
            }
            let team_record = archive[*match_index];
            let match_shingles = string_shingles(team_record.behavior_descriptor);
            let similarity = jaccard_similarity(&query_shingles, &match_shingles);
            similarity_cache.insert(key, similarity);
            (*match_index, similarity)
        })
        .filter(|(_m, similarity)| *similarity <= max_similarity)
        .collect();
    similar_matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    if similar_matches.len() > n {
        similar_matches.resize(n, (0, 0.0));
    }
    similar_matches
}

pub fn merge_into_archives(
    generation_archive: &Archive,
    archive: &mut Archive,
) {
    for (generation_team_id, generation_value) in generation_archive.iter() {
        archive.0.insert(*generation_team_id, generation_value.clone());
    }
}

pub fn merge_into_run_index(
    generation_index: &BandedBucket,
    run_index: &mut BandedBucket,
) {
    for (generation_bucket, archive_bucket) in generation_index.0.iter().zip(run_index.0.iter_mut()) {
        for (key, value) in generation_bucket.0.iter() {
            archive_bucket.0
                .entry(*key)
                .or_insert_with(|| vec![])
                .extend(value.clone());
        }
    }
}
