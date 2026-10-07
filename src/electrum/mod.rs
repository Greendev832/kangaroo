//! Old-style (pre-2.0, 2011-2014) Electrum wallets.
//!
//! These wallets derive everything from a 128-bit seed, shown to the user as 12
//! words from a 1626-word list or as 32 hex characters:
//!
//! - master secret = sha256 stretched 100,000 times over the hex seed's ASCII
//! - offset(n, change) = sha256d("n:change:" || master public key x||y)
//! - private key = master secret + offset, address = P2PKH of the uncompressed pubkey

use anyhow::{anyhow, bail, Result};
use bitcoin::hashes::{sha256, sha256d, Hash};
use bitcoin::{Network, PrivateKey};
use k256::elliptic_curve::ops::{MulByGenerator, Reduce};
use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::{ProjectivePoint, Scalar, U256 as K256U256};
use serde::Serialize;
use std::sync::OnceLock;

use crate::crypto::{compute_hash160, pubkey_hash_to_address};

pub mod gpu;
pub mod recover;

const WORDLIST_TEXT: &str = include_str!("old_wordlist.txt");
const WORDLIST_LEN: i64 = 1626;
pub(crate) const STRETCH_ROUNDS: u32 = 100_000;

pub(crate) fn wordlist() -> &'static [&'static str] {
    static WORDS: OnceLock<Vec<&'static str>> = OnceLock::new();
    WORDS.get_or_init(|| WORDLIST_TEXT.lines().collect())
}

pub(crate) fn word_index(word: &str) -> Result<i64> {
    wordlist()
        .iter()
        .position(|w| *w == word)
        .map(|i| i as i64)
        .ok_or_else(|| anyhow!("'{word}' is not in the old Electrum wordlist"))
}

/// Decode old Electrum mnemonic words to the hex seed (Electrum's `mn_decode`).
pub fn mnemonic_to_hex(words: &[&str]) -> Result<String> {
    if words.is_empty() || !words.len().is_multiple_of(3) {
        bail!(
            "old Electrum mnemonics have a multiple of 3 words, got {}",
            words.len()
        );
    }
    let n = WORDLIST_LEN;
    let mut out = String::with_capacity(words.len() / 3 * 8);
    for [word1, word2, word3] in words.as_chunks::<3>().0 {
        let w1 = word_index(word1)?;
        let w2 = word_index(word2)?;
        let w3 = word_index(word3)?;
        let x = w1 + n * (w2 - w1).rem_euclid(n) + n * n * (w3 - w2).rem_euclid(n);
        out.push_str(&format!("{x:08x}"));
    }
    Ok(out)
}

/// Normalize user input to the hex seed Electrum stretches.
///
/// Accepts 12 or 24 mnemonic words, or a 32/64-character hex seed, matching
/// Electrum's `is_old_seed`.
pub fn seed_to_hex(seed: &str) -> Result<String> {
    let normalized = seed
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    if normalized.is_empty() {
        bail!("empty Electrum seed");
    }

    let is_hex = normalized.bytes().all(|b| b.is_ascii_hexdigit());
    if is_hex && (normalized.len() == 32 || normalized.len() == 64) {
        return Ok(normalized);
    }

    let words: Vec<&str> = normalized.split(' ').collect();
    if words.len() != 12 && words.len() != 24 {
        bail!(
            "old Electrum seeds are 12 or 24 words or 32/64 hex characters, got {} words",
            words.len()
        );
    }
    mnemonic_to_hex(&words)
}

pub(crate) fn scalar_from_be(bytes: &[u8; 32]) -> Scalar {
    Scalar::reduce(K256U256::from_be_slice(bytes))
}

/// Stretched seed digest: x = sha256(x || seed) 100,000 times, starting from x = seed.
pub fn stretch_digest(hex_seed: &[u8]) -> [u8; 32] {
    let mut buf = Vec::with_capacity(32 + hex_seed.len());
    buf.extend_from_slice(hex_seed);
    buf.extend_from_slice(hex_seed);
    let mut x = sha256::Hash::hash(&buf).to_byte_array();
    for _ in 1..STRETCH_ROUNDS {
        buf.clear();
        buf.extend_from_slice(&x);
        buf.extend_from_slice(hex_seed);
        x = sha256::Hash::hash(&buf).to_byte_array();
    }
    x
}

pub(crate) fn uncompressed_pubkey(secret: &Scalar) -> [u8; 65] {
    let encoded = ProjectivePoint::mul_by_generator(secret)
        .to_affine()
        .to_encoded_point(false);
    let mut out = [0u8; 65];
    out.copy_from_slice(encoded.as_bytes());
    out
}

/// Master public key (x || y, no 0x04 prefix) for a master secret.
pub(crate) fn master_public_key(master_secret: &Scalar) -> [u8; 64] {
    let mut mpk = [0u8; 64];
    mpk.copy_from_slice(&uncompressed_pubkey(master_secret)[1..]);
    mpk
}

/// Per-address offset added to the master secret: sha256d("index:change:" || mpk).
pub(crate) fn key_offset(mpk: &[u8; 64], change: bool, index: u32) -> Scalar {
    let mut preimage = format!("{index}:{}:", u8::from(change)).into_bytes();
    preimage.extend_from_slice(mpk);
    scalar_from_be(&sha256d::Hash::hash(&preimage).to_byte_array())
}

/// One derived key of an old Electrum wallet.
#[derive(Debug, Clone, Serialize)]
pub struct DerivedAddress {
    pub change: bool,
    pub index: u32,
    pub address: String,
    /// Uncompressed SEC1 public key (65 bytes hex)
    pub public_key_hex: String,
    /// Raw private key (32 bytes hex, big-endian)
    pub private_key_hex: String,
    /// Uncompressed-key WIF, as exported by Electrum
    pub wif: String,
}

/// An old (2012-2013) Electrum wallet restored from its seed.
pub struct OldElectrumWallet {
    hex_seed: String,
    master_secret: Scalar,
    /// Master public key: uncompressed point without the 0x04 prefix (x || y)
    mpk: [u8; 64],
}

impl OldElectrumWallet {
    /// Restore from 12/24 mnemonic words or a 32/64-character hex seed.
    pub fn from_seed(seed: &str) -> Result<Self> {
        let hex_seed = seed_to_hex(seed)?;
        let master_secret = scalar_from_be(&stretch_digest(hex_seed.as_bytes()));
        if bool::from(master_secret.is_zero()) {
            bail!("seed stretches to an invalid (zero) key");
        }
        Ok(Self {
            hex_seed,
            master_secret,
            mpk: master_public_key(&master_secret),
        })
    }

    pub fn hex_seed(&self) -> &str {
        &self.hex_seed
    }

    /// Master public key in Electrum's format (128 hex chars, x || y).
    pub fn master_public_key_hex(&self) -> String {
        hex::encode(self.mpk)
    }

    /// Derive receiving (`change = false`) or change (`change = true`) key `index`.
    pub fn derive(&self, change: bool, index: u32) -> Result<DerivedAddress> {
        let secret = self.master_secret + key_offset(&self.mpk, change, index);
        if bool::from(secret.is_zero()) {
            bail!("derived key {index} is invalid (zero)");
        }
        let private_key: [u8; 32] = secret.to_bytes().into();
        let public_key = uncompressed_pubkey(&secret);

        let mut wif_key = PrivateKey::from_slice(&private_key, Network::Bitcoin)
            .map_err(|e| anyhow!("invalid derived private key: {e}"))?;
        wif_key.compressed = false;

        Ok(DerivedAddress {
            change,
            index,
            address: pubkey_hash_to_address(&compute_hash160(&public_key)),
            public_key_hex: hex::encode(public_key),
            private_key_hex: hex::encode(private_key),
            wif: wif_key.to_wif(),
        })
    }

    /// The first `count` receiving or change addresses.
    pub fn addresses(&self, change: bool, count: u32) -> Result<Vec<DerivedAddress>> {
        (0..count).map(|i| self.derive(change, i)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Vectors from Electrum's tests (test_mnemonic.py, test_wallet_vertical.py).
    const SEED_WORDS: &str =
        "powerful random nobody notice nothing important anyway look away hidden message over";
    const SEED_HEX: &str = "acb740e454c3134901d7c8f16497cc1c";
    const MPK: &str = "e9d4b7866dd1e91c862aebf62a49548c7dbf7bcc6e4b7b8c9da820c7737968df9c09d5a3e271dc814a29981f81b3faaf2737b551ef5dcc6189cf0f8252c442b3";

    #[test]
    fn wordlist_has_1626_unique_words() {
        let words = wordlist();
        assert_eq!(words.len(), 1626);
        let unique: std::collections::HashSet<_> = words.iter().collect();
        assert_eq!(unique.len(), 1626);
    }

    #[test]
    fn decodes_mnemonic_like_electrum() {
        let words =
            "hardly point goal hallway patience key stone difference ready caught listen fact";
        assert_eq!(
            seed_to_hex(words).unwrap(),
            "8edad31a95e7d59f8837667510d75a4d"
        );
        assert_eq!(seed_to_hex(SEED_WORDS).unwrap(), SEED_HEX);
    }

    #[test]
    fn restores_electrum_test_wallet_from_words_and_hex() {
        for seed in [SEED_WORDS, SEED_HEX, "  Powerful RANDOM nobody notice nothing important anyway look away hidden message over\n"] {
            let wallet = OldElectrumWallet::from_seed(seed).unwrap();
            assert_eq!(wallet.hex_seed(), SEED_HEX);
            assert_eq!(wallet.master_public_key_hex(), MPK);
            assert_eq!(
                wallet.derive(false, 0).unwrap().address,
                "1FJEEB8ihPMbzs2SkLmr37dHyRFzakqUmo"
            );
            assert_eq!(
                wallet.derive(true, 0).unwrap().address,
                "1KRW8pH6HFHZh889VDq6fEKvmrsmApwNfe"
            );
        }
    }

    #[test]
    fn derived_private_key_matches_address() {
        let wallet = OldElectrumWallet::from_seed(SEED_HEX).unwrap();
        let derived = wallet.derive(false, 3).unwrap();
        let wif = PrivateKey::from_wif(&derived.wif).unwrap();
        assert!(!wif.compressed);
        assert_eq!(
            hex::encode(wif.inner.secret_bytes()),
            derived.private_key_hex
        );
        let pubkey = hex::decode(&derived.public_key_hex).unwrap();
        assert_eq!(pubkey.len(), 65);
        let hash = compute_hash160(&pubkey);
        assert_eq!(pubkey_hash_to_address(&hash), derived.address);
    }

    #[test]
    fn rejects_bad_seeds() {
        assert!(seed_to_hex("").is_err());
        assert!(seed_to_hex("like just love").is_err());
        let bad_word = SEED_WORDS.replace("powerful", "notaword");
        let err = seed_to_hex(&bad_word).unwrap_err().to_string();
        assert!(err.contains("notaword"), "{err}");
    }
}
