//! Kangaroo initialization and jump table generation.

use crate::convert::{affine_to_gpu, scalar_be_to_limbs};
use crate::crypto::{Point, U256};
use crate::gpu::{GpuAffinePoint, GpuKangaroo};
use crate::math::negate_256_be;
use anyhow::Result;
use k256::elliptic_curve::ops::{MulByGenerator, Reduce};
use k256::U256 as K256U256;
use k256::{ProjectivePoint, Scalar};
use rayon::prelude::*;
use std::ops::Neg;

pub type JumpPointTable = Vec<GpuAffinePoint>;
pub type JumpDistanceTable = Vec<[u32; 8]>;
pub type JumpTables = (JumpPointTable, JumpDistanceTable);

pub fn generate_jump_tables(range_bits: u32, base_point: &ProjectivePoint) -> JumpTables {
    const TABLE_SIZE: usize = 256;

    let base_is_generator = *base_point == ProjectivePoint::GENERATOR;
    let jump_exp = range_bits / 2;

    let entries: Vec<(GpuAffinePoint, [u32; 8])> = (0..TABLE_SIZE)
        .into_par_iter()
        .map(|i| {
            let jump_scalar_bytes = make_step_scalar_bytes(i as u32, jump_exp, 0x811c9dc5u32);
            let jump_scalar = Scalar::reduce(K256U256::from_be_slice(&jump_scalar_bytes));
            let jump_point = mul_base(base_point, &jump_scalar, base_is_generator).to_affine();
            (
                affine_to_gpu(&jump_point),
                scalar_be_to_limbs(&jump_scalar_bytes),
            )
        })
        .collect();

    let (jump_points, jump_distances): (Vec<_>, Vec<_>) = entries.into_iter().unzip();

    tracing::debug!("Jump table generated: {} entries", TABLE_SIZE);

    (jump_points, jump_distances)
}

fn make_step_scalar_bytes(index: u32, exp_bits: u32, salt: u32) -> [u8; 32] {
    let mut h = salt;
    h = (h ^ index).wrapping_mul(0x01000193);

    let num_bytes = exp_bits.div_ceil(8);
    let limit_byte = (num_bytes as usize).min(32);
    let mut scalar_bytes = [0u8; 32];

    #[allow(clippy::needless_range_loop)]
    for b in (32 - limit_byte)..32 {
        h = (h ^ (b as u32)).wrapping_mul(0x01000193);
        scalar_bytes[b] = (h & 0xFF) as u8;
    }

    let rem = exp_bits % 8;
    if rem != 0 {
        let mask = (1u8 << rem) - 1;
        if 32 - limit_byte < 32 {
            scalar_bytes[32 - limit_byte] &= mask;
        }
    }

    if scalar_bytes.iter().all(|&x| x == 0) {
        scalar_bytes[31] = 1;
    }

    scalar_bytes
}

/// Places kangaroos at random positions inside their set.
///
/// With the negation map, walks diffuse around their start instead of travelling,
/// so coverage comes from where kangaroos are (re)spawned:
/// - tame (ktype=0): uniform over the whole search interval `[start, start + 2^range_bits)`
/// - wild_1 (ktype=1): uniform over `P + [-2^(range_bits-1), 2^(range_bits-1))`
/// - wild_2 (ktype=2): same offsets around `-P`, for cross-wild collisions
///
/// Tame and wild sets therefore overlap by at least half the interval for any key.
pub struct KangarooSpawner {
    pubkey: Point,
    neg_pubkey: Point,
    start: U256,
    range_mask: K256U256,
    range_middle: K256U256,
    base_point: ProjectivePoint,
    base_is_generator: bool,
}

impl KangarooSpawner {
    pub fn new(
        pubkey: &Point,
        start: &U256,
        range_bits: u32,
        base_point: &ProjectivePoint,
    ) -> Result<Self> {
        anyhow::ensure!(
            range_bits > 0 && range_bits <= 255,
            "range_bits must be 1..=255"
        );
        let range_size = K256U256::ONE.shl_vartime(range_bits as usize);
        Ok(Self {
            pubkey: *pubkey,
            neg_pubkey: pubkey.neg(),
            start: *start,
            range_mask: range_size.wrapping_sub(&K256U256::ONE),
            range_middle: K256U256::ONE.shl_vartime((range_bits - 1) as usize),
            base_point: *base_point,
            base_is_generator: *base_point == ProjectivePoint::GENERATOR,
        })
    }

    /// Spawn a kangaroo of `ktype` at a position derived from `(global_id, epoch)`.
    ///
    /// The same inputs always give the same kangaroo; bump `epoch` for a fresh position.
    pub fn spawn(&self, ktype: u32, global_id: u32, epoch: u64) -> GpuKangaroo {
        let offset = random_u256(global_id, epoch) & self.range_mask;

        let (point, dist) = match ktype {
            0 => init_tame_kangaroo_at_offset(
                &self.start,
                &offset,
                &self.base_point,
                self.base_is_generator,
            ),
            1 => init_wild_kangaroo_at_offset(
                &self.pubkey,
                &offset,
                &self.range_middle,
                &self.base_point,
                self.base_is_generator,
            ),
            _ => init_wild_kangaroo_at_offset(
                &self.neg_pubkey,
                &offset,
                &self.range_middle,
                &self.base_point,
                self.base_is_generator,
            ),
        };

        let gpu_point = affine_to_gpu(&point);

        GpuKangaroo {
            x: gpu_point.x,
            y: gpu_point.y,
            dist,
            ktype,
            is_active: 1,
            cycle_counter: 0,
            checkpoint_x: gpu_point.x[0],
            last_jump: 0xFFFFFFFF,
            _padding: [0; 3],
        }
    }
}

/// Kangaroo type for local index `i`: first third tame, second wild_1, rest wild_2.
pub fn ktype_for_index(i: u32, num_kangaroos: u32) -> u32 {
    let one_third = num_kangaroos / 3;
    if i < one_third {
        0
    } else if i < 2 * one_third {
        1
    } else {
        2
    }
}

/// Initialize kangaroo positions.
///
/// Split into three sets: tame (ktype=0), wild_1 (ktype=1), wild_2 (ktype=2).
/// Wild_2 uses the negated public key for cross-wild collision detection.
///
/// `kangaroo_offset` shifts global indices so multiple GPU workers get unique positions.
/// For single-GPU, pass 0. For multi-GPU, GPU N gets offset = N * num_kangaroos.
///
/// All range/offset math uses full 256-bit arithmetic to correctly handle
/// range_bits >= 128 (fixes silent degradation from u128 clamping).
pub fn initialize_kangaroos(
    pubkey: &Point,
    start: &U256,
    range_bits: u32,
    num_kangaroos: u32,
    base_point: &ProjectivePoint,
    kangaroo_offset: u32,
    _global_kangaroo_count: u32,
) -> Result<Vec<GpuKangaroo>> {
    anyhow::ensure!(
        num_kangaroos >= 3,
        "Multi-set requires at least 3 kangaroos"
    );
    let spawner = KangarooSpawner::new(pubkey, start, range_bits, base_point)?;

    tracing::debug!(
        "Kangaroo init: range_bits={}, num_kangaroos={}",
        range_bits,
        num_kangaroos
    );

    let kangaroos: Vec<GpuKangaroo> = (0..num_kangaroos)
        .into_par_iter()
        .map(|i| {
            spawner.spawn(
                ktype_for_index(i, num_kangaroos),
                kangaroo_offset + i,
                0,
            )
        })
        .collect();

    Ok(kangaroos)
}

/// Deterministic 256 pseudo-random bits from `(global_id, epoch)` (splitmix64 stream).
fn random_u256(global_id: u32, epoch: u64) -> K256U256 {
    let mut state = (u64::from(global_id) << 32)
        ^ epoch.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ 0xCAFE_BABE_D15E_A5E5;
    let mut bytes = [0u8; 32];
    for chunk in bytes.as_chunks_mut::<8>().0 {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        *chunk = z.to_le_bytes();
    }
    K256U256::from_le_slice(&bytes)
}

/// Extract big-endian [u8; 32] from K256U256 via limb decomposition.
fn u256_to_be_bytes(val: &K256U256) -> [u8; 32] {
    let limbs = val.as_limbs();
    let n = limbs.len();
    let mut bytes = [0u8; 32];
    for i in 0..n {
        let be = limbs[n - 1 - i].0.to_be_bytes();
        let sz = be.len();
        bytes[i * sz..(i + 1) * sz].copy_from_slice(&be);
    }
    bytes
}

fn mul_base(
    base_point: &ProjectivePoint,
    scalar: &Scalar,
    base_is_generator: bool,
) -> ProjectivePoint {
    if base_is_generator {
        ProjectivePoint::mul_by_generator(scalar)
    } else {
        *base_point * *scalar
    }
}

/// Initialize a tame kangaroo at a specific offset from start.
fn init_tame_kangaroo_at_offset(
    start: &U256,
    offset: &K256U256,
    base_point: &ProjectivePoint,
    base_is_generator: bool,
) -> (k256::AffinePoint, [u32; 8]) {
    let start_uint = K256U256::from_le_slice(start);
    let sum = start_uint.wrapping_add(offset);
    let scalar = Scalar::reduce(sum);
    let point = mul_base(base_point, &scalar, base_is_generator);

    // Distance = offset relative to start (full 256-bit)
    let offset_be = u256_to_be_bytes(offset);
    (point.to_affine(), scalar_be_to_limbs(&offset_be))
}

/// Initialize a wild kangaroo at a specific offset, centered around the range midpoint.
///
/// Maps raw offset in `[0, range)` to centered offset in `[-range/2, range/2)`.
/// Uses `sbb` (subtract-with-borrow) for sign detection in full 256-bit space.
fn init_wild_kangaroo_at_offset(
    pubkey: &Point,
    raw_offset: &K256U256,
    range_middle: &K256U256,
    base_point: &ProjectivePoint,
    base_is_generator: bool,
) -> (k256::AffinePoint, [u32; 8]) {
    // Detect sign: sbb returns borrow != 0 when raw_offset < range_middle
    let (diff, borrow) = raw_offset.sbb(range_middle, k256::elliptic_curve::bigint::Limb::ZERO);
    let is_negative = borrow != k256::elliptic_curve::bigint::Limb::ZERO;

    if !is_negative {
        // raw_offset >= range_middle: positive direction
        // diff = raw_offset - range_middle (exact, no wrap)
        let scalar = Scalar::reduce(diff);
        let offset_point = mul_base(base_point, &scalar, base_is_generator);
        let wild_point = *pubkey + offset_point;

        let delta_be = u256_to_be_bytes(&diff);
        (wild_point.to_affine(), scalar_be_to_limbs(&delta_be))
    } else {
        // raw_offset < range_middle: negative direction
        let delta = range_middle.wrapping_sub(raw_offset);
        let scalar = Scalar::reduce(delta);
        let offset_point = mul_base(base_point, &scalar, base_is_generator);
        let wild_point = *pubkey - offset_point;

        // Store negative offset as two's complement
        let delta_be = u256_to_be_bytes(&delta);
        let neg_bytes = negate_256_be(&delta_be);
        (wild_point.to_affine(), scalar_be_to_limbs(&neg_bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wild2_init_with_negated_pubkey() {
        let pubkey_hex = "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c";
        let pubkey = crate::crypto::parse_pubkey(pubkey_hex).expect("Failed to parse pubkey");
        let neg_pubkey = pubkey.neg();

        let range_middle = K256U256::ONE.shl_vartime(19);
        let offset = K256U256::from(1000u64);

        let (point, dist) = init_wild_kangaroo_at_offset(
            &neg_pubkey,
            &offset,
            &range_middle,
            &ProjectivePoint::GENERATOR,
            true,
        );

        let gpu_point = affine_to_gpu(&point);

        assert!(
            gpu_point.x.iter().any(|&x| x != 0),
            "GPU point x should have at least one non-zero element"
        );
        assert!(
            dist.iter().any(|&d| d != 0),
            "Distance should have at least one non-zero element"
        );
    }

    #[test]
    fn test_three_set_distribution() {
        let pubkey_hex = "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c";
        let pubkey = crate::crypto::parse_pubkey(pubkey_hex).expect("Failed to parse pubkey");
        let start = [0u8; 32];
        let range_bits = 20u32;
        let num_kangaroos = 4096u32;

        let kangaroos = initialize_kangaroos(
            &pubkey,
            &start,
            range_bits,
            num_kangaroos,
            &ProjectivePoint::GENERATOR,
            0,
            num_kangaroos,
        )
        .unwrap();
        assert_eq!(kangaroos.len(), num_kangaroos as usize);

        let tame = kangaroos.iter().filter(|k| k.ktype == 0).count();
        let wild1 = kangaroos.iter().filter(|k| k.ktype == 1).count();
        let wild2 = kangaroos.iter().filter(|k| k.ktype == 2).count();

        assert_eq!(tame, 1365);
        assert_eq!(wild1, 1365);
        assert_eq!(wild2, 1366);
        assert_eq!(tame + wild1 + wild2, num_kangaroos as usize);
    }

    /// Tames must reach the top of the interval: keys there were unsolvable when tames
    /// only covered the first third.
    #[test]
    fn test_tames_cover_whole_interval() {
        let pubkey_hex = "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c";
        let pubkey = crate::crypto::parse_pubkey(pubkey_hex).expect("Failed to parse pubkey");
        let range_bits = 20u32;
        let kangaroos = initialize_kangaroos(
            &pubkey,
            &[0u8; 32],
            range_bits,
            3000,
            &ProjectivePoint::GENERATOR,
            0,
            3000,
        )
        .unwrap();

        let tame_offsets: Vec<u32> = kangaroos
            .iter()
            .filter(|k| k.ktype == 0)
            .map(|k| k.dist[0])
            .collect();
        let range = 1u32 << range_bits;
        assert!(tame_offsets.iter().all(|&o| o < range));
        assert!(tame_offsets.iter().any(|&o| o > range / 10 * 9));
        assert!(tame_offsets.iter().any(|&o| o < range / 10));
    }

    #[test]
    fn test_spawn_epochs_give_fresh_positions() {
        let pubkey_hex = "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c";
        let pubkey = crate::crypto::parse_pubkey(pubkey_hex).expect("Failed to parse pubkey");
        let spawner =
            KangarooSpawner::new(&pubkey, &[0u8; 32], 40, &ProjectivePoint::GENERATOR).unwrap();

        let a = spawner.spawn(1, 7, 1);
        assert_eq!(a.x, spawner.spawn(1, 7, 1).x, "spawn must be deterministic");
        assert_ne!(a.x, spawner.spawn(1, 7, 2).x, "new epoch must move the kangaroo");
        assert_eq!(a.checkpoint_x, a.x[0]);
        assert_eq!(a.is_active, 1);
    }

    #[test]
    fn test_minimum_kangaroo_count() {
        let pubkey_hex = "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c";
        let pubkey = crate::crypto::parse_pubkey(pubkey_hex).expect("Failed to parse pubkey");
        let start = [0u8; 32];
        let result =
            initialize_kangaroos(&pubkey, &start, 20, 2, &ProjectivePoint::GENERATOR, 0, 2);
        assert!(result.is_err(), "Should fail for num_kangaroos < 3");
        assert!(result.unwrap_err().to_string().contains("at least 3"));
    }

    #[test]
    fn test_multi_gpu_workers_have_disjoint_initial_states() {
        use std::collections::HashSet;

        let pubkey_hex = "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c";
        let pubkey = crate::crypto::parse_pubkey(pubkey_hex).expect("Failed to parse pubkey");
        let start = [0u8; 32];
        let range_bits = 24u32;
        let per_gpu_k = 512u32;
        let total_k = per_gpu_k * 2;

        let gpu0 = initialize_kangaroos(
            &pubkey,
            &start,
            range_bits,
            per_gpu_k,
            &ProjectivePoint::GENERATOR,
            0,
            total_k,
        )
        .unwrap();

        let gpu1 = initialize_kangaroos(
            &pubkey,
            &start,
            range_bits,
            per_gpu_k,
            &ProjectivePoint::GENERATOR,
            per_gpu_k,
            total_k,
        )
        .unwrap();

        type KangarooState = ([u32; 8], [u32; 8], [u32; 8], u32);

        let gpu0_states: HashSet<KangarooState> = gpu0
            .iter()
            .map(|k| (k.x, k.y, k.dist, k.ktype))
            .collect();

        for k in &gpu1 {
            let state = (k.x, k.y, k.dist, k.ktype);
            assert!(
                !gpu0_states.contains(&state),
                "GPU workers must not share initial kangaroo states"
            );
        }
    }

    /// Verify initialization works correctly for range_bits >= 128.
    /// This is the core regression test for issue #70.
    #[test]
    fn test_initialize_kangaroos_large_range() {
        let pubkey_hex = "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c";
        let pubkey = crate::crypto::parse_pubkey(pubkey_hex).expect("Failed to parse pubkey");
        let start = [0u8; 32];
        let range_bits = 135u32;
        let num_kangaroos = 6u32;

        let kangaroos = initialize_kangaroos(
            &pubkey,
            &start,
            range_bits,
            num_kangaroos,
            &ProjectivePoint::GENERATOR,
            0,
            num_kangaroos,
        )
        .unwrap();

        assert_eq!(kangaroos.len(), num_kangaroos as usize);

        for k in &kangaroos {
            assert!(
                k.x.iter().any(|&v| v != 0),
                "kangaroo should have non-zero position"
            );
        }

        assert!(kangaroos.iter().any(|k| k.ktype == 0));
        assert!(kangaroos.iter().any(|k| k.ktype == 1));
        assert!(kangaroos.iter().any(|k| k.ktype == 2));
    }

    /// Verify initialization at range_bits = 200 (well above u128 limit).
    #[test]
    fn test_initialize_kangaroos_200bit_range() {
        let pubkey_hex = "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c";
        let pubkey = crate::crypto::parse_pubkey(pubkey_hex).expect("Failed to parse pubkey");
        let start = [0u8; 32];
        let range_bits = 200u32;
        let num_kangaroos = 9u32;

        let kangaroos = initialize_kangaroos(
            &pubkey,
            &start,
            range_bits,
            num_kangaroos,
            &ProjectivePoint::GENERATOR,
            0,
            num_kangaroos,
        )
        .unwrap();

        assert_eq!(kangaroos.len(), num_kangaroos as usize);

        for k in &kangaroos {
            assert!(
                k.x.iter().any(|&v| v != 0),
                "kangaroo should have non-zero position"
            );
            assert_eq!(k.is_active, 1);
        }
    }

    /// range_bits > 255 must fail (2^256 overflows U256).
    #[test]
    fn test_range_bits_overflow_rejected() {
        let pubkey_hex = "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c";
        let pubkey = crate::crypto::parse_pubkey(pubkey_hex).expect("Failed to parse pubkey");
        let start = [0u8; 32];
        let result =
            initialize_kangaroos(&pubkey, &start, 256, 6, &ProjectivePoint::GENERATOR, 0, 6);
        assert!(result.is_err());
    }

    /// range_bits = 0 must fail.
    #[test]
    fn test_range_bits_zero_rejected() {
        let pubkey_hex = "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c";
        let pubkey = crate::crypto::parse_pubkey(pubkey_hex).expect("Failed to parse pubkey");
        let start = [0u8; 32];
        let result = initialize_kangaroos(&pubkey, &start, 0, 6, &ProjectivePoint::GENERATOR, 0, 6);
        assert!(result.is_err());
    }

    #[test]
    fn test_u256_be_bytes_roundtrip() {
        let val = K256U256::from(0xDEAD_BEEF_u64);
        let be = u256_to_be_bytes(&val);
        let recovered = K256U256::from_be_slice(&be);
        assert_eq!(val, recovered);

        let large = K256U256::ONE.shl_vartime(200);
        let be2 = u256_to_be_bytes(&large);
        let recovered2 = K256U256::from_be_slice(&be2);
        assert_eq!(large, recovered2);
    }

    #[test]
    fn test_generate_jump_tables_lengths() {
        let (jump_points, jump_distances) = generate_jump_tables(40, &ProjectivePoint::GENERATOR);

        assert_eq!(jump_points.len(), 256);
        assert_eq!(jump_distances.len(), 256);
    }
}
