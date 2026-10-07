# Kangaroo

[![Crates.io](https://img.shields.io/crates/v/kangaroo?style=flat&colorA=130f40&colorB=474787)](https://crates.io/crates/kangaroo)
[![Downloads](https://img.shields.io/crates/d/kangaroo?style=flat&colorA=130f40&colorB=474787)](https://crates.io/crates/kangaroo)
[![License](https://img.shields.io/crates/l/kangaroo?style=flat&colorA=130f40&colorB=474787)](LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/oritwoen/kangaroo)

GPU-accelerated Pollard's Kangaroo algorithm for solving the Elliptic Curve Discrete Logarithm Problem (ECDLP) on secp256k1.

## Features

- 🖥️ **Cross-platform GPU** — Vulkan (AMD, NVIDIA, Intel), Metal (Apple Silicon), DX12 (Windows) via wgpu
- 🦀 **Pure Rust + WGSL** — no CUDA dependency, compute shaders compiled at runtime
- ✍️ **Solve from a signature `r`** — find the nonce k from only the x-coordinate of R = k×G (`--r`)
- ⚡ **Distinguished Points** — efficient collision detection with auto-tuned DP bits
- 🔄 **Negation map** — Y-parity directed walks with checkpoint-based cycle detection
- 🦘 **Multi-set kangaroos** — tame, wild1, wild2 herds; each kangaroo respawns at a fresh random position after every DP
- 🎯 **Modular constraints** — if k ≡ R (mod M), reduce search space by factor M
- ⚙️ **Auto-tuning** — herd size, DP bits and steps per dispatch chosen per range and GPU at startup
- 🖧 **Multi-GPU** — `--gpu 0,1` or `--gpu all`
- 📊 **Built-in benchmarks** — `--benchmark` to test hardware, `--save-benchmarks` to record results
- 📦 **Data providers** — pluggable puzzle sources (boha integration for Bitcoin puzzles)
- 💻 **CPU fallback** — pure CPU solver for testing and comparison

## Why This Project?

Most existing Kangaroo implementations (JeanLucPons/Kangaroo, RCKangaroo, etc.) only support NVIDIA GPUs via CUDA. This implementation uses WebGPU/wgpu which provides cross-platform GPU compute through Vulkan, Metal, and DX12.

## Installation

### Arch Linux (AUR)

```bash
paru -S kangaroo
```

### Cargo

```bash
cargo install kangaroo
```

### From source

```bash
git clone https://github.com/oritwoen/kangaroo
cd kangaroo
cargo build --release
```

### With boha provider

```bash
cargo build --release --features boha
```

## Usage

```bash
kangaroo --pubkey <PUBKEY> --start <START> --range <BITS>
kangaroo --r <R_X> --start <START> --range <BITS>
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-t, --target` | - | Data provider target (e.g., `boha:b1000/135`) |
| `-p, --pubkey` | - | Target public key: compressed (33 bytes) or x-only (32 bytes) hex |
| `--r` | - | Signature `r` value, the x-coordinate of R = k×G (32 bytes hex). Alternative to `--pubkey` |
| `-s, --start` | 0 | Start of search range (hex, without 0x prefix) |
| `-r, --range` | 32 | Search range in bits (key is in [start, start + 2^range - 1]) |
| `-d, --dp-bits` | auto | Distinguished point bits |
| `-k, --kangaroos` | auto | Number of parallel kangaroos |
| `--gpu` | 0 | GPU index, comma-separated indices, or `all` |
| `--include-integrated` | false | Include integrated GPUs in `--gpu all` |
| `--list-gpus` | false | List available GPU devices |
| `--backend` | auto | GPU backend: `auto`, `vulkan`, `dx12`, `metal`, `gl` |
| `-o, --output` | - | Write the found private key (hex) to this file. With `--electrum-recover`: append the seed and its keys (default `electrum_recovered.txt`) |
| `-q, --quiet` | false | Minimal output, just print found key |
| `--max-ops` | 0 | Give up after this many operations (0 = unlimited) |
| `--cpu` | false | Use CPU solver instead of GPU |
| `--json` | false | Print the result (or benchmark results) as JSON on stdout |
| `--benchmark` | false | Run benchmark suite |
| `--save-benchmarks` | false | Save benchmark results to `BENCHMARKS.md` when `--benchmark` is used |
| `--mod-step` | 1 | Modular step M (hex): search only k ≡ R (mod M) |
| `--mod-start` | 0 | Modular residue R (hex): 0 ≤ R < M |
| `--list-providers` | false | List available puzzles from providers |
| `--electrum-seed` | - | Old (2012-2013) Electrum seed (12/24 words or 32/64 hex chars, `-` for stdin): print its addresses and keys |
| `--electrum-count` | 5 | Receiving and change addresses to derive with `--electrum-seed`, or to check per candidate with `--electrum-recover` |
| `--electrum-recover` | - | Recover an old Electrum seed from a 12-word pattern (`?` = unknown word, `a\|b` = uncertain word). GPU by default, `--cpu` to use CPU cores |
| `--electrum-address` | - | Address known to belong to the wallet being recovered (repeatable) |
| `--electrum-address-file` | - | Text file of such addresses, one per line (repeatable) |
| `--electrum-pubkey` | - | Public key known to belong to the wallet (repeatable): an address's key (33/65 bytes hex) or the master public key (64 bytes / 128 hex chars) |
| `--electrum-pubkey-file` | - | Text file of such public keys, one per line (repeatable) |

One of `--target`, `--pubkey` or `--r` is required. Note that `-r` (short) is `--range`, while `--r` (long) is the signature value.

### Examples

**Using data provider (boha):**

```bash
# Solve puzzle using boha data (auto: pubkey, start, range)
kangaroo --target boha:b1000/66

# Override range (search smaller subset)
kangaroo --target boha:b1000/66 --range 60

# List available puzzles
kangaroo --list-providers
```

**Manual parameters:**

```bash
kangaroo \
    --pubkey 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4 \
    --start 8000000000 \
    --range 40
```

**From a signature `r` value (find the nonce k of R = k×G):**

```bash
kangaroo \
    --r 2014e54b6e3b53807bea0d00b532a829eba4b9b093dc2c787acd77dcf0489586 \
    --start 800000000000000 \
    --range 60 \
    -o found_key.txt
```

`r` only fixes R up to sign, so both k and n − k match it. The solver searches for either and reports the one inside `[start, start + 2^range)`. If you don't know where k starts, use `--start 0` with `--range` set to an upper bound on k's bit length. The cost depends only on the range width, not on the start.

**Giving up after a bounded amount of work** (useful when the key may not be in the range, as the search otherwise never ends):

```bash
kangaroo --r <R_X> --start 0 --range 56 --max-ops 5000000000 --json >> results.jsonl
```

**Addresses from an old (2012-2013, pre-2.0) Electrum seed:**

```bash
# Seed on the command line (12 words or 32 hex chars)
kangaroo --electrum-seed "powerful random nobody notice nothing important anyway look away hidden message over" --electrum-count 10

# Read the seed from stdin so it stays out of shell history; --json adds pubkeys and raw private keys
kangaroo --electrum-seed - --json < seed.txt
```

This prints the receiving and change addresses with uncompressed WIF private keys, as old Electrum derived them: the hex seed is stretched with 100,000 rounds of SHA-256 into a master key, and key `n` of a chain is `master + sha256d("n:change:" || master_pubkey)`. Addresses are P2PKH of uncompressed public keys. Newer Electrum seeds (2.0+, BIP32-based) are not supported.

**Recovering an old Electrum seed with missing or uncertain words (GPU):**

```bash
# One word unknown ("?"), one word uncertain ("a|b|c"), and an address the wallet used
kangaroo --electrum-recover "powerful random nobody notice ? important anyway look away hidden message over|other|order" \
    --electrum-address 1FJEEB8ihPMbzs2SkLmr37dHyRFzakqUmo \
    -o recovered_seed.txt
```

When a seed is found it is saved, before anything is printed, to `electrum_recovered.txt` in the current directory (or the file given with `-o`). The file gets the seed words, hex seed, master public key, the matched address, and the first `--electrum-count` receiving and change addresses (at least up to the matched one) with their private keys in hex and WIF. Results are appended, never overwritten, and the file is created readable by your user only (mode 600) because it holds private keys:

```text
=== Recovered old Electrum seed (unix time 1791393576) ===
Seed words:        powerful random nobody notice nothing important anyway look away hidden message over
Seed (hex):        acb740e454c3134901d7c8f16497cc1c
Master public key: e9d4b7866dd1e91c862aebf62a49548c7dbf7bcc6e4b7b8c9da820c7737968df9c09d5a3e271dc814a29981f81b3faaf2737b551ef5dcc6189cf0f8252c442b3
Matched:           19dmGS6TfnJuhQYCMYMMn9YxzuF8LVVgzg (receiving #2)

chain      index  address                             private key (hex)                                                 WIF (uncompressed)
receiving      0  1FJEEB8ihPMbzs2SkLmr37dHyRFzakqUmo  8fdf5bc0fdd0bcfb03dd2d050d903a783e5b36de98f3963a025d2e8f0629faa5  5JuecQZ1nH4VCQRQJTQjB4yu93BU6NmnAkDoGRdHX2PyH2E8QVX
...
```

Instead of an address you can give a public key with `--electrum-pubkey`:

```bash
# The master public key (MPK) old Electrum showed and stored in the wallet file: fastest check
kangaroo --electrum-recover "powerful random nobody notice ? important anyway look away hidden message over" \
    --electrum-pubkey e9d4b7866dd1e91c862aebf62a49548c7dbf7bcc6e4b7b8c9da820c7737968df9c09d5a3e271dc814a29981f81b3faaf2737b551ef5dcc6189cf0f8252c442b3

# The public key of any wallet address (e.g. from a spending transaction), compressed or uncompressed
kangaroo --electrum-recover "..." --electrum-pubkey 045f7ba332df2a7b4f5d13f246e307c9174cfa9b8b05f3b83410a3c23ef8958d610be285963d67c7bc1feb082f168fa9877c25999963ff8b56b242a852b23e25ed
```

Many addresses can be loaded from a text file with `--electrum-address-file`, one per line, in the same format as the public key file below:

```text
# addresses from my old wallet
1FJEEB8ihPMbzs2SkLmr37dHyRFzakqUmo,receiving-0
19dmGS6TfnJuhQYCMYMMn9YxzuF8LVVgzg
```

```bash
kangaroo --electrum-recover "..." --electrum-address-file addresses.txt
```

Many keys can be loaded from a text file with `--electrum-pubkey-file`. Each line holds one key (compressed, uncompressed or master public key); blank lines and `#` comments are skipped, and anything after the key (`,label` or ` label`) is ignored:

```text
# public keys collected from old transactions
045f7ba332df2a7b4f5d13f246e307c9174cfa9b8b05f3b83410a3c23ef8958d610be285963d67c7bc1feb082f168fa9877c25999963ff8b56b242a852b23e25ed,receiving-0
02aecb9d427e10f0c370c32210fe75b6e72ccc4f415076cf1a6318fbed55373888
```

```bash
kangaroo --electrum-recover "..." --electrum-pubkey-file pubkeys.txt
```

All keys go into one lookup set, so checking a candidate against thousands of keys costs the same as checking one. An invalid line stops the run with its file name and line number.

Every candidate seed is stretched with 100,000 SHA-256 rounds on the GPU, then checked on the CPU, overlapped with the next GPU batch: its master public key is compared first, then (if an address or address public key was given) its first `--electrum-count` receiving and change addresses. A master public key needs one EC multiplication per candidate instead of eleven. Old seeds have no checksum, so a known address or key is the only way to tell the right seed apart.

The GPU path is tuned for Apple Silicon (Metal) but runs on any wgpu backend:

- Each GPU dispatch is calibrated to ~40 ms, so the desktop stays responsive while the GPU (which also drives the display) is busy
- Batches start at 65,536 seeds (enough to saturate an M4 Pro) and grow to ~2 s of work each
- The SHA-256 rounds are fully unrolled, and the constant padding block's message schedule is precomputed

Measured on an Apple M4 Pro: about 8,000 seeds/s on the GPU versus about 350 seeds/s on all CPU cores (`--cpu`).

| Unknown words | Candidates | Time on M4 Pro GPU |
|---------------|------------|--------------------|
| 1 | 1,626 | ~1 s |
| 2 | 2.6 million | ~6 minutes |
| 3 | 4.3 billion | ~6 days |

Uncertain words multiply the count by their number of alternatives, so narrowing an unknown word to a few guesses helps a lot.

**With modular constraint (k ≡ 37 mod 60):**

```bash
kangaroo \
    --pubkey 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4 \
    --start 8000000000 \
    --range 40 \
    --mod-step 3c \
    --mod-start 25
```

This reduces the search space by ~60×. Useful when partial key structure is known (e.g., key generated with a predictable step pattern).

## How It Works

The Pollard's Kangaroo algorithm solves the discrete logarithm problem in O(√n) time where n is the search range. It works by:

1. **Tame kangaroos** start at random known keys spread across the whole range
2. **Wild kangaroos** start near the target point P (wild1) and its negation −P (wild2) at random known offsets
3. Every jump depends only on the current point, so once two kangaroos meet they follow the same path
4. When a tame and a wild kangaroo (or a wild1 and a wild2) reach the same point, the private key follows from their distances

**Distinguished Points (DP)**: instead of storing every visited point, the GPU only reports points whose x-coordinate ends in `dp_bits` zero bits (about 1 in 2^dp_bits points). Two kangaroos that meet reach the same next DP, so the CPU detects the collision when the same DP arrives twice. After reporting a DP, a kangaroo is respawned at a fresh random position.

Parameters are chosen automatically:

- **Kangaroos**: the GPU-optimal herd for large ranges, never below 1/8 of it (each step is latency-bound, so smaller herds don't run faster), capped for small ranges so unfinished walks stay a small part of the work
- **DP bits**: low enough that leftover walk work stays small, high enough that the DP table fits (4 bits minimum below 44-bit ranges, 8 above)
- **Steps per dispatch**: calibrated for the best useful throughput within a ~120 ms dispatch budget

The progress bar reports `Ops` (total jumps) and `DPs` (stored distinguished points, split by tame/wild1/wild2).

## Performance

Expected operations: ~2.6 × 2^(range_bits/2) on average (the K-factor). Individual solves vary a lot, typically 0.3× to 2× of that, because the method is a random birthday-style search.

Measured on an Apple M4 Pro (Metal), about 11.5M ops/s on large ranges:

| Range | Typical time |
|-------|--------------|
| 32-bit | ~0.15 s |
| 48-bit | ~5 s |
| 56-bit | ~15 s |
| 60-bit | ~4 min (one measured solve: 6.3 min, K = 4.0) |
| 64-bit | ~1 hour |
| 70-bit | ~7 hours |
| 80-bit | ~1 week |

Each extra 2 bits of range doubles the time.

Run `kangaroo --benchmark` to test your hardware without touching files. Use `kangaroo --benchmark --save-benchmarks` to update [BENCHMARKS.md](BENCHMARKS.md).

## Use Cases

| Use Case | Example |
|----------|---------|
| Partial key decoded | Puzzle gives ~240 bits, need to find remaining ~16 |
| Key in known range | Know key is between X and Y |
| Verify near-solution | Have candidate, search ±N bits around it |
| Weak ECDSA nonce | Signature `r` from a nonce generator that produced a small k |

**NOT useful for:**
- Full 256-bit key search (mathematically impossible)
- Properly generated (random 256-bit) ECDSA nonces
- BIP39 passphrase brute-force (use dictionary attack instead)
- Puzzles without partial key information

## Library Usage

```rust
use kangaroo::{KangarooSolver, GpuContext, GpuBackend, parse_pubkey, parse_hex_u256, verify_key};

fn main() -> anyhow::Result<()> {
    // Compressed (33-byte) or x-only (32-byte, e.g. a signature r) hex
    let pubkey = parse_pubkey("03...")?;
    let start = parse_hex_u256("8000000000")?;

    let ctx = pollster::block_on(GpuContext::new(0, GpuBackend::Auto))?;
    let mut solver = KangarooSolver::new(
        ctx,
        pubkey,
        start,
        40,   // range_bits
        8,    // dp_bits
        8192, // num_kangaroos
    )?;

    loop {
        if let Some(key) = solver.step()? {
            if verify_key(&key, &pubkey) {
                println!("Found: {}", hex::encode(&key));
                break;
            }
        }
    }

    Ok(())
}
```

With an x-only point the solver may return n − k instead of k, as both match the x-coordinate. The CLI normalizes the result into the search range for you.

## Data Providers

Kangaroo supports external data providers for puzzle sources. Providers supply pubkey, key range, and other puzzle metadata.

### boha (optional feature)

[boha](https://github.com/oritwoen/boha) provides crypto puzzle data including Bitcoin Puzzle Transaction (b1000).

Build with boha support:
```bash
cargo build --release --features boha
```

Usage:
```bash
# Solve specific puzzle
kangaroo --target boha:b1000/66

# List solvable puzzles (unsolved with known pubkey)
kangaroo --list-providers
```

Provider validates range overrides - you cannot search outside the puzzle's key range.

## Architecture

```
src/
├── main.rs              # CLI entry point
├── lib.rs               # Library entry + Args + run()
├── solver.rs            # GPU solver coordination
├── cli.rs               # CLI utilities (tracing, progress bar)
├── benchmark.rs         # Built-in benchmark suite
├── modular.rs           # Modular constraint transformation
├── math.rs              # 256-bit arithmetic, DP mask generation
├── convert.rs           # Limb/byte conversions for GPU↔CPU
├── provider/
│   ├── mod.rs           # Provider system interface
│   └── boha.rs          # boha provider (feature-gated)
├── cpu/
│   ├── cpu_solver.rs    # Pure CPU solver (testing/comparison)
│   ├── dp_table.rs      # Distinguished Points collision detection
│   └── init.rs          # Kangaroo initialization + jump tables
├── crypto/
│   └── mod.rs           # k256/secp256k1 wrappers
├── gpu/
│   ├── pipeline.rs      # Compute pipeline setup
│   └── buffers.rs       # GPU buffer management
├── gpu_crypto/
│   ├── context.rs       # GPU context + backend selection
│   └── shaders/         # WGSL shader library
│       ├── field.wgsl   # secp256k1 field arithmetic
│       └── curve.wgsl   # Jacobian point operations
└── shaders/
    └── kangaroo_affine.wgsl  # Main Kangaroo compute shader
```

## Requirements

- Rust 1.88+
- Vulkan-capable GPU (AMD, NVIDIA, Intel) or Metal (macOS)
- On Linux with AMD RADV, Mesa 25.x or newer is required (older Mesa versions may crash on WGSL dynamic indexing in shader loops)
- GPU drivers installed

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

- [JeanLucPons/Kangaroo](https://github.com/JeanLucPons/Kangaroo) - CUDA implementation (NVIDIA only)
- [RCKangaroo](https://github.com/RetiredC/RCKangaroo) - CUDA implementation (NVIDIA only)
- [boha](https://github.com/oritwoen/boha) - Crypto puzzles and bounties data library
