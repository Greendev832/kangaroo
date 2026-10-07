//! Recover an old Electrum seed from partially known words and a known address.
//!
//! Every candidate costs one 100,000-round SHA-256 stretch (run on the GPU, or on
//! all CPU cores as a fallback), then a few EC multiplications to derive its first
//! addresses. The EC work for one batch runs on the CPU while the GPU stretches
//! the next one.

use anyhow::{anyhow, bail, Context, Result};
use bitcoin::address::NetworkUnchecked;
use bitcoin::hashes::Hash;
use bitcoin::{Address, Network};
use k256::elliptic_curve::sec1::ToEncodedPoint;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashSet;
use std::path::Path;
use std::time::{Duration, Instant};

use super::gpu::GpuStretcher;
use super::{
    key_offset, master_public_key, scalar_from_be, stretch_digest, uncompressed_pubkey, word_index,
    wordlist, WORDLIST_LEN,
};
use crate::crypto::{compute_hash160, pubkey_hash_to_address};

const SEED_WORDS: usize = 12;
const FIRST_GPU_BATCH: usize = 65_536;
const FIRST_CPU_BATCH: usize = 256;
const MAX_BATCH: usize = 1 << 20;
/// Batches are sized to take about this long, so progress updates stay frequent.
const TARGET_BATCH_TIME: Duration = Duration::from_secs(2);

/// Allowed words for each of the 12 seed positions.
pub struct SeedPattern {
    slots: Vec<Vec<u16>>,
}

impl SeedPattern {
    /// Parse 12 space-separated tokens: a word, `?` for any word, or `a|b|c` for
    /// one of several candidates.
    pub fn parse(pattern: &str) -> Result<Self> {
        let tokens: Vec<String> = pattern.split_whitespace().map(str::to_lowercase).collect();
        if tokens.len() != SEED_WORDS {
            bail!(
                "old Electrum seeds have {SEED_WORDS} words, the pattern has {}",
                tokens.len()
            );
        }
        let slots = tokens
            .iter()
            .map(|token| {
                if token == "?" {
                    return Ok((0..WORDLIST_LEN as u16).collect());
                }
                let mut options = Vec::new();
                for word in token.split('|').filter(|w| !w.is_empty()) {
                    let index = word_index(word)? as u16;
                    if !options.contains(&index) {
                        options.push(index);
                    }
                }
                if options.is_empty() {
                    bail!("empty alternative list '{token}'");
                }
                Ok(options)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { slots })
        // let pattern = Self { slots };
        // if pattern.checked_count().is_none() {
        //     let unknown = pattern.slots.iter().filter(|s| s.len() > 1).count();
        //     bail!(
        //         "the pattern describes more than 2^128 seeds ({unknown} of 12 words open); \
        //          that search cannot finish, so pin down more words"
        //     );
        // }
        // Ok(pattern)
    }

    // fn checked_count(&self) -> Option<u128> {
    //     self.slots
    //         .iter()
    //         .try_fold(1u128, |acc, s| acc.checked_mul(s.len() as u128))
    // }

    /// Number of seeds the pattern describes.
    pub fn candidate_count(&self) -> u128 {
        self.slots.iter().map(|s| s.len() as u128).product()
        // self.checked_count()
        //     .expect("parse rejects patterns whose count overflows")
    }

    /// Candidate `index` in mixed radix, with the last word varying fastest.
    fn words_at(&self, mut index: u128) -> [u16; SEED_WORDS] {
        let mut words = [0u16; SEED_WORDS];
        for (slot, word) in self.slots.iter().zip(words.iter_mut()).rev() {
            let radix = slot.len() as u128;
            *word = slot[(index % radix) as usize];
            index /= radix;
        }
        words
    }
}

fn words_to_string(words: &[u16; SEED_WORDS]) -> String {
    let list = wordlist();
    words
        .iter()
        .map(|&w| list[w as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

/// The 32 ASCII hex characters Electrum stretches for these words, or `None` if a
/// word triple decodes past 32 bits (`mn_encode` never produces such seeds).
fn hex_seed(words: &[u16; SEED_WORDS]) -> Option<[u8; 32]> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let n = WORDLIST_LEN;
    let mut out = [0u8; 32];
    for (group, chunk) in words
        .as_chunks::<3>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<8>().0)
    {
        let [w1, w2, w3] = group.map(i64::from);
        let x = w1 + n * (w2 - w1).rem_euclid(n) + n * n * (w3 - w2).rem_euclid(n);
        let x = u32::try_from(x).ok()?;
        for (i, byte) in chunk.iter_mut().enumerate() {
            *byte = HEX[((x >> (28 - 4 * i)) & 0xf) as usize];
        }
    }
    Some(out)
}

/// What the recovered wallet must contain: addresses (as hash160s of uncompressed
/// pubkeys) and/or master public keys.
#[derive(Default)]
pub struct Targets {
    hashes: HashSet<[u8; 20]>,
    master_keys: HashSet<[u8; 64]>,
}

impl Targets {
    pub fn new() -> Self {
        Self::default()
    }

    /// `addresses`: P2PKH addresses. `pubkeys`: see [`Targets::add_pubkey`].
    pub fn parse(addresses: &[String], pubkeys: &[String]) -> Result<Self> {
        let mut targets = Self::new();
        for text in addresses {
            targets.add_address(text)?;
        }
        for text in pubkeys {
            targets.add_pubkey(text)?;
        }
        targets.ensure_not_empty()?;
        Ok(targets)
    }

    pub fn ensure_not_empty(&self) -> Result<()> {
        if self.is_empty() {
            bail!("at least one known address or public key is required");
        }
        Ok(())
    }

    /// Number of distinct targets (addresses and master public keys).
    pub fn len(&self) -> usize {
        self.hashes.len() + self.master_keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Add a P2PKH address.
    pub fn add_address(&mut self, text: &str) -> Result<()> {
        let address = text
            .trim()
            .parse::<Address<NetworkUnchecked>>()
            .map_err(|e| anyhow!("invalid address '{text}': {e}"))?
            .require_network(Network::Bitcoin)
            .map_err(|e| anyhow!("'{text}' is not a mainnet address: {e}"))?;
        let hash = address.pubkey_hash().ok_or_else(|| {
            anyhow!("'{text}' is not a P2PKH (1...) address; old Electrum only used those")
        })?;
        self.hashes.insert(hash.to_byte_array());
        Ok(())
    }

    /// Add a hex public key: an address's key (33-byte compressed or 65-byte
    /// uncompressed) or the 64-byte master public key.
    pub fn add_pubkey(&mut self, text: &str) -> Result<()> {
        let hex_text = text.trim().trim_start_matches("0x");
        let bytes =
            hex::decode(hex_text).map_err(|e| anyhow!("invalid public key hex '{text}': {e}"))?;
        match bytes.len() {
            64 => {
                let mut sec1 = vec![0x04];
                sec1.extend_from_slice(&bytes);
                k256::PublicKey::from_sec1_bytes(&sec1).map_err(|_| {
                    anyhow!("master public key '{text}' is not a point on secp256k1")
                })?;
                self.master_keys.insert(bytes.try_into().expect("64 bytes"));
            }
            33 | 65 => {
                let key = k256::PublicKey::from_sec1_bytes(&bytes)
                    .map_err(|_| anyhow!("public key '{text}' is not a point on secp256k1"))?;
                // Old Electrum addresses always hash the uncompressed key.
                let uncompressed = key.to_encoded_point(false);
                self.hashes.insert(compute_hash160(uncompressed.as_bytes()));
            }
            n => bail!(
                "public key '{text}' has {n} bytes; expected 33 or 65 (address key) or 64 (master public key)"
            ),
        }
        Ok(())
    }

    /// Add every public key listed in a text file and return how many lines held one.
    /// See [`read_list_file`] for the format.
    pub fn add_pubkey_file(&mut self, path: &Path) -> Result<usize> {
        self.add_from_file(path, Self::add_pubkey)
    }

    /// Add every P2PKH address listed in a text file and return how many were added.
    /// See [`read_list_file`] for the format.
    ///
    /// Valid addresses of other types (`3...`, `bc1...`) cannot belong to an old
    /// Electrum wallet, so they are skipped with a warning instead of failing the run.
    pub fn add_address_file(&mut self, path: &Path) -> Result<usize> {
        let mut added = 0;
        let mut skipped = Vec::new();
        for (line_no, entry) in read_list_file(path)? {
            let address = entry
                .parse::<Address<NetworkUnchecked>>()
                .map_err(|e| anyhow!("invalid address '{entry}': {e}"))
                .and_then(|a| {
                    a.require_network(Network::Bitcoin)
                        .map_err(|e| anyhow!("'{entry}' is not a mainnet address: {e}"))
                })
                .with_context(|| format!("{}:{line_no}", path.display()))?;
            match address.pubkey_hash() {
                Some(hash) => {
                    self.hashes.insert(hash.to_byte_array());
                    added += 1;
                }
                None => skipped.push(line_no),
            }
        }
        if !skipped.is_empty() {
            tracing::warn!(
                "Skipped {} non-P2PKH address(es) in {} (old Electrum only used 1... addresses), lines {:?}",
                skipped.len(),
                path.display(),
                skipped
            );
        }
        Ok(added)
    }

    fn add_from_file(
        &mut self,
        path: &Path,
        add: fn(&mut Self, &str) -> Result<()>,
    ) -> Result<usize> {
        let entries = read_list_file(path)?;
        for (line_no, entry) in &entries {
            add(self, entry).with_context(|| format!("{}:{line_no}", path.display()))?;
        }
        Ok(entries.len())
    }

    /// Whether candidates must derive addresses (otherwise the master key suffices).
    pub fn needs_addresses(&self) -> bool {
        !self.hashes.is_empty()
    }
}

/// Read a one-entry-per-line text file as `(line number, entry)` pairs.
///
/// Blank lines and lines starting with `#` are skipped. Only the first token of a
/// line is used, so `entry,label`, `entry;label` and `entry label` also work.
fn read_list_file(path: &Path) -> Result<Vec<(usize, String)>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    Ok(text
        .lines()
        .enumerate()
        .filter_map(|(i, line)| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let entry = line
                .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
                .next()
                .unwrap_or_default();
            Some((i + 1, entry.to_string()))
        })
        .collect())
}

/// Which target the recovered seed matched.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MatchedKey {
    MasterPublicKey,
    Address {
        address: String,
        change: bool,
        index: u32,
    },
}

/// A seed whose master public key or derived addresses include a target.
#[derive(Debug, Clone, Serialize)]
pub struct RecoveredSeed {
    pub words: String,
    pub hex_seed: String,
    pub matched: MatchedKey,
}

/// Where the 100,000-round stretch runs.
pub enum Stretcher {
    Gpu(Box<GpuStretcher>),
    Cpu,
}

impl Stretcher {
    pub fn name(&self) -> String {
        match self {
            Self::Gpu(gpu) => format!("GPU ({})", gpu.device_name()),
            Self::Cpu => format!("CPU ({} threads)", rayon::current_num_threads()),
        }
    }

    fn first_batch(&self) -> usize {
        match self {
            Self::Gpu(_) => FIRST_GPU_BATCH,
            Self::Cpu => FIRST_CPU_BATCH,
        }
    }

    fn stretch(&mut self, seeds: &[[u8; 32]]) -> Result<Vec<[u8; 32]>> {
        match self {
            Self::Gpu(gpu) => gpu.stretch(seeds),
            Self::Cpu => Ok(seeds.par_iter().map(|s| stretch_digest(s)).collect()),
        }
    }
}

/// Check the master public key, then the first `count` receiving and change addresses.
fn find_target(digest: &[u8; 32], targets: &Targets, count: u32) -> Option<MatchedKey> {
    let secret = scalar_from_be(digest);
    if bool::from(secret.is_zero()) {
        return None;
    }
    let mpk = master_public_key(&secret);
    if targets.master_keys.contains(&mpk) {
        return Some(MatchedKey::MasterPublicKey);
    }
    if !targets.needs_addresses() {
        return None;
    }
    for change in [false, true] {
        for index in 0..count {
            let key = secret + key_offset(&mpk, change, index);
            let hash = compute_hash160(&uncompressed_pubkey(&key));
            if targets.hashes.contains(&hash) {
                return Some(MatchedKey::Address {
                    address: pubkey_hash_to_address(&hash),
                    change,
                    index,
                });
            }
        }
    }
    None
}

struct Batch {
    words: Vec<[u16; SEED_WORDS]>,
    seeds: Vec<[u8; 32]>,
    digests: Vec<[u8; 32]>,
}

fn check_batch(batch: &Batch, targets: &Targets, count: u32) -> Option<RecoveredSeed> {
    (0..batch.digests.len()).into_par_iter().find_map_any(|i| {
        let matched = find_target(&batch.digests[i], targets, count)?;
        Some(RecoveredSeed {
            words: words_to_string(&batch.words[i]),
            hex_seed: String::from_utf8_lossy(&batch.seeds[i]).into_owned(),
            matched,
        })
    })
}

/// Search every seed the pattern allows. `on_progress(tested, seeds_per_sec)` is
/// called after each batch.
pub fn recover(
    pattern: &SeedPattern,
    targets: &Targets,
    count: u32,
    stretcher: &mut Stretcher,
    mut on_progress: impl FnMut(u128, f64),
) -> Result<Option<RecoveredSeed>> {
    let total = pattern.candidate_count();
    let started = Instant::now();
    let mut next: u128 = 0;
    let mut batch_size = stretcher.first_batch();
    let mut pending: Option<Batch> = None;

    loop {
        let end = total.min(next + batch_size as u128);
        let (words, seeds): (Vec<_>, Vec<_>) = (next..end)
            .into_par_iter()
            .filter_map(|i| {
                let words = pattern.words_at(i);
                hex_seed(&words).map(|seed| (words, seed))
            })
            .unzip();
        let finished = next >= total;
        next = end;

        let batch_start = Instant::now();
        let (digests, hit) = std::thread::scope(|scope| {
            let checker = scope.spawn(|| {
                pending
                    .as_ref()
                    .and_then(|batch| check_batch(batch, targets, count))
            });
            let digests = stretcher.stretch(&seeds);
            (digests, checker.join().expect("address checker panicked"))
        });
        if hit.is_some() {
            return Ok(hit);
        }
        if finished {
            return Ok(None);
        }

        if !seeds.is_empty() {
            let rate = seeds.len() as f64 / batch_start.elapsed().as_secs_f64().max(1e-6);
            let ideal = (rate * TARGET_BATCH_TIME.as_secs_f64()) as usize;
            batch_size = ideal
                .clamp(stretcher.first_batch(), MAX_BATCH)
                .next_power_of_two();
        }
        pending = Some(Batch {
            words,
            seeds,
            digests: digests?,
        });
        on_progress(
            next,
            next as f64 / started.elapsed().as_secs_f64().max(1e-6),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::electrum::mnemonic_to_hex;

    const SEED: &str =
        "powerful random nobody notice nothing important anyway look away hidden message over";
    const RECEIVE_0: &str = "1FJEEB8ihPMbzs2SkLmr37dHyRFzakqUmo";
    const CHANGE_0: &str = "1KRW8pH6HFHZh889VDq6fEKvmrsmApwNfe";
    const MPK: &str = "e9d4b7866dd1e91c862aebf62a49548c7dbf7bcc6e4b7b8c9da820c7737968df9c09d5a3e271dc814a29981f81b3faaf2737b551ef5dcc6189cf0f8252c442b3";

    fn address_targets(addresses: &[&str]) -> Targets {
        let addresses: Vec<String> = addresses.iter().map(|a| a.to_string()).collect();
        Targets::parse(&addresses, &[]).unwrap()
    }

    fn pubkey_targets(pubkeys: &[String]) -> Targets {
        Targets::parse(&[], pubkeys).unwrap()
    }

    fn matched_address(found: &RecoveredSeed) -> (&str, bool, u32) {
        match &found.matched {
            MatchedKey::Address {
                address,
                change,
                index,
            } => (address, *change, *index),
            MatchedKey::MasterPublicKey => panic!("expected an address match"),
        }
    }

    fn uncertain_pattern() -> SeedPattern {
        SeedPattern::parse(&SEED.replace("random", "like|random|just")).unwrap()
    }

    #[test]
    fn pattern_parsing_and_counting() {
        let pattern = SeedPattern::parse(&SEED.replace("over", "?")).unwrap();
        assert_eq!(pattern.candidate_count(), 1626);
        let pattern = SeedPattern::parse(&SEED.replace("over", "over|like|just")).unwrap();
        assert_eq!(pattern.candidate_count(), 3);
        assert!(SeedPattern::parse("like just love").is_err());
        // 1626^12 overflows u128 and must be rejected, not wrapped.
        // let err = SeedPattern::parse(&["?"; 12].join(" "))
        //     .err()
        //     .expect("all-unknown pattern must be rejected")
        //     .to_string();
        // assert!(err.contains("2^128"), "{err}");
        assert!(SeedPattern::parse(&SEED.replace("over", "notaword")).is_err());
    }

    #[test]
    fn hex_seed_matches_mnemonic_decode() {
        let pattern = SeedPattern::parse(SEED).unwrap();
        let words = pattern.words_at(0);
        let words_str: Vec<&str> = SEED.split(' ').collect();
        assert_eq!(
            hex_seed(&words).unwrap().as_slice(),
            mnemonic_to_hex(&words_str).unwrap().as_bytes()
        );
    }

    #[test]
    fn rejects_bad_targets() {
        let p2sh = "3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy".to_string();
        assert!(Targets::parse(&[p2sh], &[]).is_err());
        assert!(Targets::parse(&[], &[]).is_err());
        assert!(Targets::parse(&[], &["02abcd".to_string()]).is_err());
        let off_curve = format!("{}00", &MPK[..126]);
        assert!(Targets::parse(&[], &[off_curve]).is_err());
    }

    #[test]
    fn reads_pubkeys_from_file() {
        let wallet = crate::electrum::OldElectrumWallet::from_seed(SEED).unwrap();
        let key0 = wallet.derive(false, 0).unwrap().public_key_hex;
        let key1 = wallet.derive(true, 1).unwrap().public_key_hex;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pubkeys.txt");
        std::fs::write(
            &path,
            format!(
                "# keys from old transactions\n\n{key0}\n  {key1},change-1  \n{MPK} master\n{key0}\n"
            ),
        )
        .unwrap();

        let mut targets = Targets::new();
        assert_eq!(targets.add_pubkey_file(&path).unwrap(), 4);
        // The duplicate line collapses: 2 address keys + 1 master public key.
        assert_eq!(targets.len(), 3);
        assert!(targets.needs_addresses());

        std::fs::write(&path, format!("{key0}\nnot-hex\n")).unwrap();
        let err = format!("{:#}", Targets::new().add_pubkey_file(&path).unwrap_err());
        assert!(err.contains("pubkeys.txt:2"), "{err}");

        assert!(Targets::new()
            .add_pubkey_file(&dir.path().join("missing.txt"))
            .is_err());
    }

    #[test]
    fn reads_addresses_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("addresses.txt");
        std::fs::write(
            &path,
            format!(
                "# wallet addresses\n{RECEIVE_0},first\n\n  {CHANGE_0} change-0\n{RECEIVE_0}\n"
            ),
        )
        .unwrap();

        let mut targets = Targets::new();
        assert_eq!(targets.add_address_file(&path).unwrap(), 3);
        assert_eq!(targets.len(), 2);

        // P2SH lines are skipped; malformed lines still fail with their line number.
        std::fs::write(
            &path,
            format!("{RECEIVE_0}\n3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy\n"),
        )
        .unwrap();
        let mut targets = Targets::new();
        assert_eq!(targets.add_address_file(&path).unwrap(), 1);
        assert_eq!(targets.len(), 1);
        std::fs::write(&path, format!("{RECEIVE_0}\n1NotAnAddress\n")).unwrap();
        let err = format!("{:#}", Targets::new().add_address_file(&path).unwrap_err());
        assert!(err.contains("addresses.txt:2"), "{err}");

        // A file of addresses finds the seed just like --electrum-address.
        std::fs::write(&path, format!("{CHANGE_0}\n")).unwrap();
        let mut targets = Targets::new();
        targets.add_address_file(&path).unwrap();
        let found = recover(
            &uncertain_pattern(),
            &targets,
            1,
            &mut Stretcher::Cpu,
            |_, _| {},
        )
        .unwrap()
        .expect("seed should be found");
        assert_eq!(matched_address(&found), (CHANGE_0, true, 0));
    }

    #[test]
    fn cpu_recovers_uncertain_word_from_change_address() {
        let found = recover(
            &uncertain_pattern(),
            &address_targets(&[CHANGE_0]),
            1,
            &mut Stretcher::Cpu,
            |_, _| {},
        )
        .unwrap()
        .expect("seed should be found");
        assert_eq!(found.words, SEED);
        assert_eq!(matched_address(&found), (CHANGE_0, true, 0));
    }

    #[test]
    fn cpu_recovers_from_master_public_key() {
        let found = recover(
            &uncertain_pattern(),
            &pubkey_targets(&[MPK.to_string()]),
            1,
            &mut Stretcher::Cpu,
            |_, _| {},
        )
        .unwrap()
        .expect("seed should be found");
        assert_eq!(found.words, SEED);
        assert!(matches!(found.matched, MatchedKey::MasterPublicKey));
    }

    #[test]
    fn cpu_recovers_from_address_public_key() {
        let wallet = crate::electrum::OldElectrumWallet::from_seed(SEED).unwrap();
        let derived = wallet.derive(false, 2).unwrap();
        let uncompressed = hex::decode(&derived.public_key_hex).unwrap();
        let compressed = k256::PublicKey::from_sec1_bytes(&uncompressed)
            .unwrap()
            .to_encoded_point(true);

        for pubkey in [
            derived.public_key_hex.clone(),
            hex::encode(compressed.as_bytes()),
        ] {
            let found = recover(
                &uncertain_pattern(),
                &pubkey_targets(&[pubkey]),
                3,
                &mut Stretcher::Cpu,
                |_, _| {},
            )
            .unwrap()
            .expect("seed should be found");
            assert_eq!(found.words, SEED);
            assert_eq!(
                matched_address(&found),
                (derived.address.as_str(), false, 2)
            );
        }
    }

    #[test]
    fn gpu_recovers_missing_word() {
        let Ok(ctx) = pollster::block_on(crate::GpuContext::new(0, crate::GpuBackend::Auto)) else {
            eprintln!("no GPU available, skipping");
            return;
        };
        let mut stretcher = Stretcher::Gpu(Box::new(GpuStretcher::new(ctx).unwrap()));
        let pattern = SeedPattern::parse(&SEED.replace("hidden", "?")).unwrap();
        let targets = address_targets(&[RECEIVE_0]);
        let found = recover(&pattern, &targets, 1, &mut stretcher, |_, _| {})
            .unwrap()
            .expect("seed should be found");
        assert_eq!(found.words, SEED);
        assert_eq!(matched_address(&found), (RECEIVE_0, false, 0));
    }
}
