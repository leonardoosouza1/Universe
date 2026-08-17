//! iara-hdc — Hyperdimensional Computing (HDC) encoding and retrieval.
//!
//! Implements the triple-signature HDC pipeline extracted from byte-language-lab
//! where the same logic was duplicated across 13+ binary files.
//!
//! # Representations
//!
//! Each text string is encoded into a [`Sig`] containing three complementary
//! 1024-bit hypervectors:
//!
//! | field    | encoding                                      | sensitivity         |
//! |----------|-----------------------------------------------|---------------------|
//! | `bag`    | bundle (majority vote) of unigram HVs         | vocabulary, order-free |
//! | `bigram` | bundle of `HV[i] XOR rot1(HV[i+1])`           | local adjacency     |
//! | `seq`    | XOR of `HV[byte] XOR pos_HV[min(pos, 255)]`   | strict byte order   |
//!
//! Combined similarity: `0.5 * cos(bag) + 0.35 * cos(bigram) + 0.15 * cos(seq)`
//!
//! # Retrieval
//!
//! [`HdcIndex::search`] uses two-phase retrieval:
//! 1. HDC similarity against all entries (fast bitwise ops)
//! 2. Levenshtein re-rank of the top-K candidates (`0.4 * hdc + 0.6 * lev`)
//!
//! # Quick example
//! ```rust
//! use iara_hdc::{build_pos_hvs, encode, similarity, HV_U64};
//!
//! let byte_hvs = [[0u64; HV_U64]; 256];
//! let pos_hvs  = build_pos_hvs();
//! let a = encode("hello world", &byte_hvs, &pos_hvs);
//! let b = encode("hello world", &byte_hvs, &pos_hvs);
//! assert!((similarity(&a, &b) - 1.0).abs() < 1e-6);
//! ```

use std::{fs, io};
use std::io::{Read, Write};

use iara_byte::normalize_hdc;

// ── Constants ──────────────────────────────────────────────────────────────

/// Hypervector dimensionality in bits.
pub const HV_BITS: usize = 1024;

/// Number of `u64` words per hypervector (`HV_BITS / 64 = 16`).
pub const HV_U64: usize = HV_BITS / 64;

/// A 1024-bit hypervector: 16 × u64 in little-endian word order.
///
/// 128 bytes per vector — two cache lines on x86-64.
pub type Hv = [u64; HV_U64];

// ── Core Hv operations ────────────────────────────────────────────────────

/// Hamming distance between two hypervectors.
///
/// Counts differing bits via XOR + popcount. Result is in `[0, HV_BITS]`.
#[inline]
pub fn hamming(a: &Hv, b: &Hv) -> u32 {
    let mut h = 0u32;
    for i in 0..HV_U64 {
        h += (a[i] ^ b[i]).count_ones();
    }
    h
}

/// Elementwise XOR of two hypervectors (HDC binding operation).
///
/// Returns a new `Hv`; neither input is mutated.
#[inline]
pub fn hv_xor(a: &Hv, b: &Hv) -> Hv {
    let mut out = [0u64; HV_U64];
    for i in 0..HV_U64 {
        out[i] = a[i] ^ b[i];
    }
    out
}

/// Rotate a hypervector left by 1 word (64-bit) position (HDC permutation).
///
/// Word at index `i` moves to `(i + 1) % HV_U64`. Cheap and sufficient to
/// encode bigram order: `HV[i] XOR rot1(HV[i+1])` is unique to the pair and
/// changes when either element or their relative order changes.
#[inline]
pub fn hv_rot1(hv: &Hv) -> Hv {
    let mut out = [0u64; HV_U64];
    for i in 0..HV_U64 {
        out[(i + 1) % HV_U64] = hv[i];
    }
    out
}

/// Bundle (majority vote) of a slice of hypervectors.
///
/// Each output bit is `1` iff strictly more than half the inputs have that bit
/// set. Ties (even count) yield `0`. Returns the zero vector for an empty slice.
///
/// Bundling is the HDC superposition operation: it approximates the "mean" of
/// the input set without the XOR cancellation that plagues pure accumulation.
pub fn bundle(hvs: &[Hv]) -> Hv {
    if hvs.is_empty() {
        return [0u64; HV_U64];
    }
    let n = hvs.len();
    let half = n / 2;
    // u16 per bit — sufficient for up to 65 535 inputs.
    let mut counts = vec![0u16; HV_BITS];
    for hv in hvs {
        for w in 0..HV_U64 {
            for b in 0..64usize {
                if (hv[w] >> b) & 1 == 1 {
                    counts[w * 64 + b] += 1;
                }
            }
        }
    }
    let mut out = [0u64; HV_U64];
    for p in 0..HV_BITS {
        if counts[p] as usize > half {
            out[p / 64] |= 1u64 << (p % 64);
        }
    }
    out
}

/// Cosine similarity approximated via Hamming distance.
///
/// `cosine_hv(a, b) = 1.0 - hamming(a, b) / HV_BITS`
///
/// Range: `[0.0, 1.0]`. Identical vectors → 1.0; maximally different → 0.0.
#[inline]
pub fn cosine_hv(a: &Hv, b: &Hv) -> f32 {
    1.0 - hamming(a, b) as f32 / HV_BITS as f32
}

// ── Position vectors ──────────────────────────────────────────────────────

/// Build 256 deterministic pseudo-random position hypervectors via xorshift64.
///
/// Seed is fixed (`0xDEADBEEFCAFEBABE`) so vectors are identical across all
/// runs and machines — no file needed. Position `p` represents "slot `p` in
/// the byte sequence".
pub fn build_pos_hvs() -> [Hv; 256] {
    let mut out = [[0u64; HV_U64]; 256];
    let mut s: u64 = 0xDEADBEEF_CAFEBABE;
    for p in 0..256 {
        for w in 0..HV_U64 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            out[p][w] = s;
        }
    }
    out
}

// ── Triple signature ──────────────────────────────────────────────────────

/// Triple-component HDC signature for a text string.
///
/// Combining three views makes retrieval robust to different kinds of variation:
/// - `bag` captures vocabulary overlap regardless of order
/// - `bigram` is sensitive to local structure while tolerating typos that swap
///   non-adjacent characters
/// - `seq` anchors each byte to an absolute position, penalising reorderings
#[derive(Clone, Debug)]
pub struct Sig {
    /// Majority-vote bundle of unigram hypervectors (bag-of-bytes).
    pub bag: Hv,
    /// Bundle of `HV[i] XOR rot1(HV[i+1])` for all adjacent byte pairs.
    pub bigram: Hv,
    /// XOR accumulation of `HV[byte] XOR pos_HV[min(pos, 255)]`.
    pub seq: Hv,
}

/// Encode a text string into a triple [`Sig`].
///
/// The text is normalized via [`iara_byte::normalize_hdc`] (diacritics stripped,
/// lowercased, non-alphanumeric → space, runs collapsed). Then:
///
/// - `bag`    = `bundle([byte_hvs[b] for b in norm])`
/// - `bigram` = `bundle([hv_xor(byte_hvs[b_i], hv_rot1(byte_hvs[b_{i+1}])) ...])`
/// - `seq`    = `XOR(hv_xor(byte_hvs[b], pos_hvs[min(pos,255)]) for each byte)`
///
/// For single-byte inputs the bigram falls back to the unigram list.
pub fn encode(text: &str, byte_hvs: &[Hv; 256], pos_hvs: &[Hv; 256]) -> Sig {
    let norm = normalize_hdc(text);

    // bag: bundle of unigram HVs
    let unigrams: Vec<Hv> = norm.iter().map(|&b| byte_hvs[b as usize]).collect();
    let bag = bundle(&unigrams);

    // bigram: bundle of position-encoded adjacent pairs
    let bigrams: Vec<Hv> = if norm.len() >= 2 {
        norm.windows(2)
            .map(|w| hv_xor(&byte_hvs[w[0] as usize], &hv_rot1(&byte_hvs[w[1] as usize])))
            .collect()
    } else {
        // Single byte or empty: reuse unigrams so bigram ≈ bag
        unigrams.clone()
    };
    let bigram = bundle(&bigrams);

    // seq: XOR accumulation with position binding
    let mut seq = [0u64; HV_U64];
    for (pos, &b) in norm.iter().enumerate() {
        let p = pos.min(255);
        let bound = hv_xor(&byte_hvs[b as usize], &pos_hvs[p]);
        for k in 0..HV_U64 {
            seq[k] ^= bound[k];
        }
    }

    Sig { bag, bigram, seq }
}

/// Weighted combined similarity between two [`Sig`]s.
///
/// `0.5 * cos(bag) + 0.35 * cos(bigram) + 0.15 * cos(seq)` — range `[0.0, 1.0]`.
#[inline]
pub fn similarity(a: &Sig, b: &Sig) -> f32 {
    0.50 * cosine_hv(&a.bag,    &b.bag)
        + 0.35 * cosine_hv(&a.bigram, &b.bigram)
        + 0.15 * cosine_hv(&a.seq,    &b.seq)
}

// ── Byte HV file I/O ──────────────────────────────────────────────────────

/// Load 256 byte hypervectors from a binary cache file.
///
/// Format: 256 × `HV_U64` × 8 bytes (u64 little-endian) = 32 768 bytes total.
/// The cache is produced by the Qwen centroid extraction binary in
/// `byte-language-lab` and stored at `data/byte_hvs_qwen.bin` by convention.
pub fn load_byte_hvs(path: &str) -> io::Result<[Hv; 256]> {
    let mut f = fs::File::open(path)?;
    let mut out = [[0u64; HV_U64]; 256];
    for hv in &mut out {
        for w in hv.iter_mut() {
            let mut buf = [0u8; 8];
            f.read_exact(&mut buf)?;
            *w = u64::from_le_bytes(buf);
        }
    }
    Ok(out)
}

// ── Edit-distance helpers ─────────────────────────────────────────────────

/// Classic Levenshtein distance between two strings (character-level DP).
///
/// Allocates an `(m+1) × (n+1)` matrix. Fast for the short strings typical in
/// HDC re-ranking (usually under 100 characters).
pub fn levenshtein(a: &str, b: &str) -> usize {
    let av: Vec<char> = a.chars().collect();
    let bv: Vec<char> = b.chars().collect();
    let m = av.len();
    let n = bv.len();
    if m == 0 { return n; }
    if n == 0 { return m; }
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m { dp[i][0] = i; }
    for j in 0..=n { dp[0][j] = j; }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if av[i - 1] == bv[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[m][n]
}

/// Levenshtein similarity normalized to `[0.0, 1.0]`.
///
/// `1.0 - levenshtein(a, b) / max(chars(a), chars(b))`
///
/// Returns `1.0` for two empty strings.
#[inline]
pub fn lev_sim(a: &str, b: &str) -> f32 {
    let d = levenshtein(a, b);
    let max_len = a.chars().count().max(b.chars().count()).max(1);
    1.0 - d as f32 / max_len as f32
}

// ── Index types ───────────────────────────────────────────────────────────

/// A single indexed entry: a user utterance paired with a model response.
pub struct Entry {
    /// Original user text (unnormalized).
    pub key: String,
    /// `normalize_hdc(key)` as a `String` — used for Levenshtein re-ranking.
    pub key_norm: String,
    /// Precomputed triple signature.
    pub sig: Sig,
    /// Associated model response.
    pub value: String,
}

/// In-memory HDC retrieval index.
///
/// Build it by encoding your corpus entries and constructing this struct directly:
///
/// ```rust,no_run
/// use iara_hdc::{HdcIndex, Entry, encode, build_pos_hvs, load_byte_hvs};
/// use iara_byte::normalize_hdc;
///
/// let byte_hvs = load_byte_hvs("data/byte_hvs_qwen.bin").unwrap();
/// let pos_hvs  = build_pos_hvs();
/// let entries: Vec<Entry> = vec![];  // fill from corpus
/// let index = HdcIndex { entries, byte_hvs, pos_hvs };
/// ```
pub struct HdcIndex {
    /// All indexed key-value entries.
    pub entries: Vec<Entry>,
    /// 256 byte hypervectors (Qwen-derived centroids or random fallback).
    pub byte_hvs: [Hv; 256],
    /// 256 deterministic position hypervectors (from [`build_pos_hvs`]).
    pub pos_hvs: [Hv; 256],
}

impl HdcIndex {
    /// Search the index for the `top_k` nearest entries to `query`.
    ///
    /// # Algorithm
    /// 1. Encode `query` into a [`Sig`].
    /// 2. Score all entries with [`similarity`]; keep top-`top_k`.
    /// 3. Re-rank: `0.4 * hdc_score + 0.6 * lev_sim(query_norm, entry.key_norm)`.
    ///
    /// Returns up to `top_k` `(score, &Entry)` pairs sorted descending.
    pub fn search<'a>(&'a self, query: &str, top_k: usize) -> Vec<(f32, &'a Entry)> {
        let q_sig  = encode(query, &self.byte_hvs, &self.pos_hvs);
        let q_norm = String::from_utf8_lossy(&normalize_hdc(query)).into_owned();

        // Phase 1: HDC scoring over the full index
        let mut hdc_scores: Vec<(f32, usize)> = self.entries.iter().enumerate()
            .map(|(i, e)| (similarity(&q_sig, &e.sig), i))
            .collect();
        hdc_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        hdc_scores.truncate(top_k);

        // Phase 2: Levenshtein re-rank
        let mut reranked: Vec<(f32, usize)> = hdc_scores.into_iter()
            .map(|(hdc_s, idx)| {
                let lev = lev_sim(&q_norm, &self.entries[idx].key_norm);
                (0.4 * hdc_s + 0.6 * lev, idx)
            })
            .collect();
        reranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        reranked.into_iter()
            .map(|(score, idx)| (score, &self.entries[idx]))
            .collect()
    }
}

// ── Binary index I/O ──────────────────────────────────────────────────────

/// Magic bytes identifying an IARA HDC index file: ASCII `HDCX` = `0x48444358`.
pub const INDEX_MAGIC: u32 = 0x48444358;
/// Current binary index format version.
pub const INDEX_VERSION: u32 = 1;

fn write_hv(f: &mut fs::File, hv: &Hv) -> io::Result<()> {
    for &w in hv.iter() {
        f.write_all(&w.to_le_bytes())?;
    }
    Ok(())
}

fn read_hv(f: &mut fs::File) -> io::Result<Hv> {
    let mut out = [0u64; HV_U64];
    for w in out.iter_mut() {
        let mut buf = [0u8; 8];
        f.read_exact(&mut buf)?;
        *w = u64::from_le_bytes(buf);
    }
    Ok(out)
}

/// Serialize a corpus index to a binary file.
///
/// # Format
/// ```text
/// [magic     u32 LE = 0x48444358 "HDCX"]
/// [version   u32 LE = 1]
/// [n_entries u32 LE]
/// for each entry:
///   [key_len   u16 LE] [key UTF-8 bytes]
///   [value_len u16 LE] [value UTF-8 bytes]
///   [bag    HV  128 bytes — 16 × u64 LE]
///   [bigram HV  128 bytes]
///   [seq    HV  128 bytes]
/// ```
///
/// `byte_hvs` and `pos_hvs` are **not** stored in the index: `byte_hvs` live in
/// their own cache file; `pos_hvs` are deterministically regenerated via
/// [`build_pos_hvs`].
pub fn save_index(
    path: &str,
    entries: &[Entry],
    _byte_hvs: &[Hv; 256],
    _pos_hvs: &[Hv; 256],
) -> io::Result<()> {
    let mut f = fs::File::create(path)?;
    f.write_all(&INDEX_MAGIC.to_le_bytes())?;
    f.write_all(&INDEX_VERSION.to_le_bytes())?;
    f.write_all(&(entries.len() as u32).to_le_bytes())?;
    for entry in entries {
        let kb = entry.key.as_bytes();
        let vb = entry.value.as_bytes();
        f.write_all(&(kb.len() as u16).to_le_bytes())?;
        f.write_all(kb)?;
        f.write_all(&(vb.len() as u16).to_le_bytes())?;
        f.write_all(vb)?;
        write_hv(&mut f, &entry.sig.bag)?;
        write_hv(&mut f, &entry.sig.bigram)?;
        write_hv(&mut f, &entry.sig.seq)?;
    }
    Ok(())
}

/// Load a corpus index from a binary file produced by [`save_index`].
///
/// Returns `(entries, zero_byte_hvs)`. The second value is a zeroed
/// `[[0u64; HV_U64]; 256]` — callers must load the real byte HVs via
/// [`load_byte_hvs`] and construct an [`HdcIndex`] with them.
pub fn load_index(path: &str) -> io::Result<(Vec<Entry>, [Hv; 256])> {
    let mut f = fs::File::open(path)?;
    let mut buf4 = [0u8; 4];

    f.read_exact(&mut buf4)?;
    let magic = u32::from_le_bytes(buf4);
    if magic != INDEX_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("wrong magic: expected 0x{:08X}, got 0x{:08X}", INDEX_MAGIC, magic),
        ));
    }

    f.read_exact(&mut buf4)?;
    let _version = u32::from_le_bytes(buf4);

    f.read_exact(&mut buf4)?;
    let n = u32::from_le_bytes(buf4) as usize;

    let mut entries = Vec::with_capacity(n);
    for _ in 0..n {
        let mut buf2 = [0u8; 2];

        f.read_exact(&mut buf2)?;
        let key_len = u16::from_le_bytes(buf2) as usize;
        let mut key_bytes = vec![0u8; key_len];
        f.read_exact(&mut key_bytes)?;
        let key = String::from_utf8_lossy(&key_bytes).into_owned();

        f.read_exact(&mut buf2)?;
        let val_len = u16::from_le_bytes(buf2) as usize;
        let mut val_bytes = vec![0u8; val_len];
        f.read_exact(&mut val_bytes)?;
        let value = String::from_utf8_lossy(&val_bytes).into_owned();

        let bag    = read_hv(&mut f)?;
        let bigram = read_hv(&mut f)?;
        let seq    = read_hv(&mut f)?;

        let key_norm = String::from_utf8_lossy(&normalize_hdc(&key)).into_owned();
        entries.push(Entry {
            key,
            key_norm,
            sig: Sig { bag, bigram, seq },
            value,
        });
    }

    Ok((entries, [[0u64; HV_U64]; 256]))
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic byte HVs for testing (xorshift64, seed 0xABCDEF0123456789).
    fn test_byte_hvs() -> [Hv; 256] {
        let mut out = [[0u64; HV_U64]; 256];
        let mut s: u64 = 0xABCD_EF01_2345_6789;
        for hv in &mut out {
            for w in hv.iter_mut() {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                *w = s;
            }
        }
        out
    }

    // ── hamming ──────────────────────────────────────────────────────────

    #[test]
    fn hamming_identical_is_zero() {
        let v: Hv = [0xDEAD_BEEF_CAFE_BABE; HV_U64];
        assert_eq!(hamming(&v, &v), 0);
    }

    #[test]
    fn hamming_xor_equals_popcount() {
        let a: Hv = [u64::MAX; HV_U64];
        let b: Hv = [0u64; HV_U64];
        // XOR → all ones → popcount = HV_BITS
        assert_eq!(hamming(&a, &b), HV_BITS as u32);
        // Cross-check: manual XOR then popcount
        let xored = hv_xor(&a, &b);
        let manual: u32 = xored.iter().map(|w| w.count_ones()).sum();
        assert_eq!(hamming(&a, &b), manual);
    }

    #[test]
    fn hamming_single_bit_difference() {
        let mut a = [0u64; HV_U64];
        let mut b = [0u64; HV_U64];
        a[0] = 1;
        b[0] = 0;
        assert_eq!(hamming(&a, &b), 1);
    }

    // ── bundle ────────────────────────────────────────────────────────────

    #[test]
    fn bundle_of_identical_returns_same() {
        let v: Hv = [0xAAAA_AAAA_AAAA_AAAA; HV_U64];
        assert_eq!(bundle(&[v, v, v]), v);
    }

    #[test]
    fn bundle_empty_is_zero() {
        assert_eq!(bundle(&[]), [0u64; HV_U64]);
    }

    #[test]
    fn bundle_majority_two_ones_one_zero() {
        let ones:  Hv = [u64::MAX; HV_U64];
        let zeros: Hv = [0u64; HV_U64];
        assert_eq!(bundle(&[ones, ones, zeros]), ones);
    }

    #[test]
    fn bundle_majority_one_one_two_zeros() {
        let ones:  Hv = [u64::MAX; HV_U64];
        let zeros: Hv = [0u64; HV_U64];
        assert_eq!(bundle(&[ones, zeros, zeros]), zeros);
    }

    #[test]
    fn bundle_tie_gives_zero_bits() {
        let ones:  Hv = [u64::MAX; HV_U64];
        let zeros: Hv = [0u64; HV_U64];
        // Exactly half (2 of 2) → tie → 0
        assert_eq!(bundle(&[ones, zeros]), zeros);
    }

    // ── hv_rot1 ───────────────────────────────────────────────────────────

    #[test]
    fn rot1_shifts_word_forward() {
        let mut v = [0u64; HV_U64];
        v[0] = 42;
        let r = hv_rot1(&v);
        assert_eq!(r[1], 42);
        assert_eq!(r[0], 0);
    }

    #[test]
    fn rot1_last_word_wraps_to_first() {
        let mut v = [0u64; HV_U64];
        v[HV_U64 - 1] = 7;
        let r = hv_rot1(&v);
        assert_eq!(r[0], 7);
        assert_eq!(r[HV_U64 - 1], 0);
    }

    // ── encode + similarity ───────────────────────────────────────────────

    #[test]
    fn encode_same_text_similarity_is_one() {
        let byte_hvs = test_byte_hvs();
        let pos_hvs  = build_pos_hvs();
        let a = encode("hello world", &byte_hvs, &pos_hvs);
        let b = encode("hello world", &byte_hvs, &pos_hvs);
        let sim = similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6, "expected 1.0, got {sim}");
    }

    #[test]
    fn encode_different_texts_lower_similarity() {
        let byte_hvs = test_byte_hvs();
        let pos_hvs  = build_pos_hvs();
        let a = encode("hello world", &byte_hvs, &pos_hvs);
        let b = encode("xyzzy frobnicator", &byte_hvs, &pos_hvs);
        let sim = similarity(&a, &b);
        assert!(sim < 0.99, "expected sim < 0.99 for unrelated texts, got {sim}");
    }

    #[test]
    fn encode_similar_texts_score_higher_than_unrelated() {
        let byte_hvs = test_byte_hvs();
        let pos_hvs  = build_pos_hvs();
        let base    = encode("hello world", &byte_hvs, &pos_hvs);
        let close   = encode("hello world again", &byte_hvs, &pos_hvs);
        let distant = encode("zzz zzz zzz zzz zzz", &byte_hvs, &pos_hvs);
        let sim_close   = similarity(&base, &close);
        let sim_distant = similarity(&base, &distant);
        assert!(sim_close > sim_distant,
            "close sim={sim_close} should be > distant sim={sim_distant}");
    }

    #[test]
    fn encode_empty_string_self_similarity_is_one() {
        let byte_hvs = test_byte_hvs();
        let pos_hvs  = build_pos_hvs();
        let a = encode("", &byte_hvs, &pos_hvs);
        let b = encode("", &byte_hvs, &pos_hvs);
        assert!((similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    // ── levenshtein ───────────────────────────────────────────────────────

    #[test]
    fn levenshtein_kitten_sitting_is_3() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn levenshtein_identical_is_zero() {
        assert_eq!(levenshtein("abc", "abc"), 0);
    }

    #[test]
    fn levenshtein_empty_a_is_len_b() {
        assert_eq!(levenshtein("", "abc"), 3);
    }

    #[test]
    fn levenshtein_empty_b_is_len_a() {
        assert_eq!(levenshtein("abc", ""), 3);
    }

    #[test]
    fn levenshtein_both_empty_is_zero() {
        assert_eq!(levenshtein("", ""), 0);
    }

    #[test]
    fn levenshtein_single_substitution() {
        assert_eq!(levenshtein("cat", "bat"), 1);
    }

    // ── lev_sim ───────────────────────────────────────────────────────────

    #[test]
    fn lev_sim_identical_is_one() {
        assert!((lev_sim("hello", "hello") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn lev_sim_both_empty_is_one() {
        assert!((lev_sim("", "") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn lev_sim_in_valid_range() {
        let s = lev_sim("kitten", "sitting");
        assert!(s >= 0.0 && s <= 1.0, "lev_sim out of range: {s}");
    }

    // ── cosine_hv ─────────────────────────────────────────────────────────

    #[test]
    fn cosine_identical_is_one() {
        let v: Hv = [0xC0FFEE; HV_U64];
        assert!((cosine_hv(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_all_ones_vs_all_zeros_is_zero() {
        let a: Hv = [u64::MAX; HV_U64];
        let b: Hv = [0u64; HV_U64];
        assert!((cosine_hv(&a, &b) - 0.0).abs() < 1e-6);
    }

    // ── index round-trip ─────────────────────────────────────────────────

    #[test]
    fn index_save_load_round_trip() {
        let byte_hvs = test_byte_hvs();
        let pos_hvs  = build_pos_hvs();

        let entries = vec![
            Entry {
                key:      "hello".to_string(),
                key_norm: "hello".to_string(),
                sig:      encode("hello", &byte_hvs, &pos_hvs),
                value:    "world".to_string(),
            },
            Entry {
                key:      "foo bar".to_string(),
                key_norm: "foo bar".to_string(),
                sig:      encode("foo bar", &byte_hvs, &pos_hvs),
                value:    "baz".to_string(),
            },
        ];

        let path = "/tmp/iara_hdc_test_round_trip.bin";
        save_index(path, &entries, &byte_hvs, &pos_hvs).unwrap();
        let (loaded, _) = load_index(path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].key,   "hello");
        assert_eq!(loaded[0].value, "world");
        assert_eq!(loaded[1].key,   "foo bar");
        assert_eq!(loaded[1].value, "baz");

        // Signatures survive the round-trip exactly
        assert_eq!(loaded[0].sig.bag,    entries[0].sig.bag);
        assert_eq!(loaded[0].sig.bigram, entries[0].sig.bigram);
        assert_eq!(loaded[0].sig.seq,    entries[0].sig.seq);
        assert_eq!(loaded[1].sig.bag,    entries[1].sig.bag);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn index_wrong_magic_returns_error() {
        let path = "/tmp/iara_hdc_test_bad_magic.bin";
        std::fs::write(path, b"BAAD\x01\x00\x00\x00\x00\x00\x00\x00").unwrap();
        let result = load_index(path);
        assert!(result.is_err(), "expected error for bad magic");
        let _ = std::fs::remove_file(path);
    }

    // ── HdcIndex::search ─────────────────────────────────────────────────

    #[test]
    fn search_exact_match_is_top_result() {
        let byte_hvs = test_byte_hvs();
        let pos_hvs  = build_pos_hvs();

        let texts = ["oi tudo bem", "como vai voce", "bom dia", "boa tarde"];
        let entries: Vec<Entry> = texts.iter().map(|&t| Entry {
            key:      t.to_string(),
            key_norm: String::from_utf8_lossy(&normalize_hdc(t)).into_owned(),
            sig:      encode(t, &byte_hvs, &pos_hvs),
            value:    format!("resposta: {t}"),
        }).collect();

        let index = HdcIndex { entries, byte_hvs, pos_hvs };
        let results = index.search("oi tudo bem", 3);

        assert!(!results.is_empty(), "search returned no results");
        assert_eq!(results[0].1.key, "oi tudo bem",
            "top result should be the exact match");
        assert!(results[0].0 > 0.9,
            "exact match should score > 0.9, got {}", results[0].0);
    }

    #[test]
    fn search_returns_at_most_top_k() {
        let byte_hvs = test_byte_hvs();
        let pos_hvs  = build_pos_hvs();

        let texts = ["alpha", "beta", "gamma"];
        let entries: Vec<Entry> = texts.iter().map(|&t| Entry {
            key:      t.to_string(),
            key_norm: t.to_string(),
            sig:      encode(t, &byte_hvs, &pos_hvs),
            value:    t.to_string(),
        }).collect();

        let index = HdcIndex { entries, byte_hvs, pos_hvs };
        // Request more than available — should return all (3), not panic
        assert!(index.search("alpha", 10).len() <= 3);
        // Request fewer — should return exactly top_k
        assert_eq!(index.search("alpha", 2).len(), 2);
    }

    #[test]
    fn search_scores_are_descending() {
        let byte_hvs = test_byte_hvs();
        let pos_hvs  = build_pos_hvs();

        let texts = ["hello world", "helo world", "hi world", "goodbye world"];
        let entries: Vec<Entry> = texts.iter().map(|&t| Entry {
            key:      t.to_string(),
            key_norm: String::from_utf8_lossy(&normalize_hdc(t)).into_owned(),
            sig:      encode(t, &byte_hvs, &pos_hvs),
            value:    t.to_string(),
        }).collect();

        let index = HdcIndex { entries, byte_hvs, pos_hvs };
        let results = index.search("hello world", 4);

        // Scores must be monotonically non-increasing
        for w in results.windows(2) {
            assert!(w[0].0 >= w[1].0,
                "scores not descending: {} < {}", w[0].0, w[1].0);
        }
    }
}
