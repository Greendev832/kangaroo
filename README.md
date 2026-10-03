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
| `-o, --output` | - | Write the found private key (hex) to this file |
| `-q, --quiet` | false | Minimal output, just print found key |
| `--max-ops` | 0 | Give up after this many operations (0 = unlimited) |
| `--cpu` | false | Use CPU solver instead of GPU |
| `--json` | false | Print the result (or benchmark results) as JSON on stdout |
| `--benchmark` | false | Run benchmark suite |
| `--save-benchmarks` | false | Save benchmark results to `BENCHMARKS.md` when `--benchmark` is used |
| `--mod-step` | 1 | Modular step M (hex): search only k ≡ R (mod M) |
| `--mod-start` | 0 | Modular residue R (hex): 0 ≤ R < M |
| `--list-providers` | false | List available puzzles from providers |

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
