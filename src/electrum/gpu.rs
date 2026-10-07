//! GPU (wgpu: Metal / Vulkan / DX12) key stretching for old Electrum seeds.
//!
//! Each GPU thread runs the 100,000-round `x = sha256(x || seed)` stretch for one
//! 32-character hex seed. Every round hashes exactly 64 bytes, so it is one data
//! block plus a constant padding block whose message schedule is precomputed.

use anyhow::{anyhow, Result};
use std::fmt::Write as _;
use std::time::{Duration, Instant};

use super::STRETCH_ROUNDS;
use crate::gpu_crypto::GpuContext;

const WORKGROUP_SIZE: u32 = 64;
const POLL_TIMEOUT: Duration = Duration::from_secs(30);

/// Per-dispatch GPU time to aim for. On Apple Silicon the same GPU drives the
/// display, and short dispatches keep the desktop responsive during long runs.
const TARGET_DISPATCH: Duration = Duration::from_millis(40);
const CALIBRATION_ROUNDS: u32 = 100;

const SHA256_IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// K[i] + W[i] for the padding block of a 64-byte message (0x80, zeros, length 512).
fn padding_block_kw() -> [u32; 64] {
    let mut w = [0u32; 64];
    w[0] = 0x8000_0000;
    w[15] = 512;
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    std::array::from_fn(|i| SHA256_K[i].wrapping_add(w[i]))
}

/// Emit 64 fully unrolled compression rounds. With `schedule`, the message words
/// live in `w0..w15` and are expanded in place; otherwise `kw` holds K+W constants.
fn emit_rounds(out: &mut String, schedule: bool, kw: &[u32; 64]) {
    for i in 0..64 {
        let constant = if schedule {
            if i >= 16 {
                let _ = writeln!(
                    out,
                    "        w{0} = ssig1(w{1}) + w{2} + ssig0(w{3}) + w{0};",
                    i % 16,
                    (i - 2) % 16,
                    (i - 7) % 16,
                    (i - 15) % 16
                );
            }
            format!("0x{:08x}u + w{}", SHA256_K[i], i % 16)
        } else {
            format!("0x{:08x}u", kw[i])
        };
        let _ = writeln!(
            out,
            "        t1 = h + bsig1(e) + ch(e, f, g) + {constant};\n        \
             t2 = bsig0(a) + maj(a, b, c);\n        \
             h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;"
        );
    }
}

fn shader_source() -> String {
    let iv = SHA256_IV;
    let kw = padding_block_kw();
    let mut s = String::new();

    s.push_str(
        "override WORKGROUP_SIZE: u32 = 64u;

struct Params {
    count: u32,
    rounds: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// 8 big-endian words per seed: the 32 ASCII hex characters.
@group(0) @binding(1) var<storage, read> seeds: array<u32>;
// 8 big-endian words per seed: the current x, updated in place.
@group(0) @binding(2) var<storage, read_write> states: array<u32>;

fn rotr(x: u32, n: u32) -> u32 { return (x >> n) | (x << (32u - n)); }
fn bsig0(x: u32) -> u32 { return rotr(x, 2u) ^ rotr(x, 13u) ^ rotr(x, 22u); }
fn bsig1(x: u32) -> u32 { return rotr(x, 6u) ^ rotr(x, 11u) ^ rotr(x, 25u); }
fn ssig0(x: u32) -> u32 { return rotr(x, 7u) ^ rotr(x, 18u) ^ (x >> 3u); }
fn ssig1(x: u32) -> u32 { return rotr(x, 17u) ^ rotr(x, 19u) ^ (x >> 10u); }
fn ch(x: u32, y: u32, z: u32) -> u32 { return (x & y) ^ (~x & z); }
fn maj(x: u32, y: u32, z: u32) -> u32 { return (x & y) ^ (x & z) ^ (y & z); }

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= params.count) {
        return;
    }
    let base = id * 8u;
",
    );
    for i in 0..8 {
        let _ = writeln!(s, "    let s{i} = seeds[base + {i}u];");
    }
    for i in 0..8 {
        let _ = writeln!(s, "    var x{i} = states[base + {i}u];");
    }
    s.push_str(
        "    var t1: u32;
    var t2: u32;
    for (var r = 0u; r < params.rounds; r = r + 1u) {
",
    );

    // Block 1: x || seed.
    for i in 0..8 {
        let _ = writeln!(s, "        var w{i} = x{i};");
    }
    for i in 8..16 {
        let _ = writeln!(s, "        var w{i} = s{};", i - 8);
    }
    for (name, value) in ["a", "b", "c", "d", "e", "f", "g", "h"].iter().zip(iv) {
        let _ = writeln!(s, "        var {name} = 0x{value:08x}u;");
    }
    emit_rounds(&mut s, true, &kw);
    for (i, (name, value)) in ["a", "b", "c", "d", "e", "f", "g", "h"]
        .iter()
        .zip(iv)
        .enumerate()
    {
        let _ = writeln!(s, "        let m{i} = 0x{value:08x}u + {name};");
    }

    // Block 2: constant padding.
    for (i, name) in ["a", "b", "c", "d", "e", "f", "g", "h"].iter().enumerate() {
        let _ = writeln!(s, "        {name} = m{i};");
    }
    emit_rounds(&mut s, false, &kw);
    for (i, name) in ["a", "b", "c", "d", "e", "f", "g", "h"].iter().enumerate() {
        let _ = writeln!(s, "        x{i} = m{i} + {name};");
    }

    s.push_str("    }\n");
    for i in 0..8 {
        let _ = writeln!(s, "    states[base + {i}u] = x{i};");
    }
    s.push_str("}\n");
    s
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    count: u32,
    rounds: u32,
    _pad: [u32; 2],
}

struct Buffers {
    capacity: usize,
    seeds: wgpu::Buffer,
    states: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// Runs the old-Electrum seed stretch for batches of seeds on one GPU.
pub struct GpuStretcher {
    ctx: GpuContext,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    buffers: Option<Buffers>,
    /// Stretch rounds per dispatch, calibrated per batch size.
    rounds_per_dispatch: u32,
    calibrated_for: usize,
}

impl GpuStretcher {
    pub fn new(ctx: GpuContext) -> Result<Self> {
        let shader = ctx.create_shader_module("Electrum Stretch Shader", &[&shader_source()]);

        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Electrum Stretch Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<Params>() as u64
                            ),
                        },
                        count: None,
                    },
                    storage(1, true),
                    storage(2, false),
                ],
            });
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Electrum Stretch Pipeline Layout"),
                bind_group_layouts: &[&layout],
                immediate_size: 0,
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Electrum Stretch Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("WORKGROUP_SIZE", f64::from(WORKGROUP_SIZE))],
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            });
        let params = ctx.create_buffer::<Params>(
            "Electrum Stretch Params",
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            1,
        )?;

        Ok(Self {
            ctx,
            pipeline,
            layout,
            params,
            buffers: None,
            rounds_per_dispatch: CALIBRATION_ROUNDS,
            calibrated_for: 0,
        })
    }

    pub fn device_name(&self) -> &str {
        self.ctx.device_name()
    }

    fn ensure_capacity(&mut self, count: usize) -> Result<()> {
        if self.buffers.as_ref().is_some_and(|b| b.capacity >= count) {
            return Ok(());
        }
        let words = (count * 8) as u64;
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let seeds = self
            .ctx
            .create_buffer::<u32>("Electrum Seeds", storage, words)?;
        let states = self.ctx.create_buffer::<u32>(
            "Electrum States",
            storage | wgpu::BufferUsages::COPY_SRC,
            words,
        )?;
        let staging = self.ctx.create_buffer::<u32>(
            "Electrum Staging",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            words,
        )?;
        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Electrum Stretch Bind Group"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: seeds.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: states.as_entire_binding(),
                    },
                ],
            });
        self.buffers = Some(Buffers {
            capacity: count,
            seeds,
            states,
            staging,
            bind_group,
        });
        Ok(())
    }

    fn wait(&self, submission: wgpu::SubmissionIndex) -> Result<()> {
        self.ctx
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(POLL_TIMEOUT),
            })
            .map_err(|e| anyhow!("GPU poll failed: {e:?}"))?;
        Ok(())
    }

    fn dispatch(&self, count: u32, rounds: u32) -> wgpu::SubmissionIndex {
        let buffers = self.buffers.as_ref().expect("buffers allocated");
        self.ctx.queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&Params {
                count,
                rounds,
                _pad: [0; 2],
            }),
        );
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Electrum Stretch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Electrum Stretch Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &buffers.bind_group, &[]);
            pass.dispatch_workgroups(count.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()))
    }

    /// Stretch a batch of 32-character hex seeds (ASCII bytes) into their digests.
    pub fn stretch(&mut self, seeds: &[[u8; 32]]) -> Result<Vec<[u8; 32]>> {
        if seeds.is_empty() {
            return Ok(Vec::new());
        }
        let count = u32::try_from(seeds.len()).map_err(|_| anyhow!("batch too large"))?;
        self.ensure_capacity(seeds.len())?;

        let words: Vec<u32> = seeds
            .iter()
            .flat_map(|seed| {
                seed.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| u32::from_be_bytes(*chunk))
            })
            .collect();
        {
            let buffers = self.buffers.as_ref().expect("buffers allocated");
            let bytes = bytemuck::cast_slice(&words);
            self.ctx.queue.write_buffer(&buffers.seeds, 0, bytes);
            self.ctx.queue.write_buffer(&buffers.states, 0, bytes);
        }

        let mut done = 0u32;
        if self.calibrated_for != seeds.len() {
            let start = Instant::now();
            let submission = self.dispatch(count, CALIBRATION_ROUNDS);
            self.wait(submission)?;
            let per_round = start.elapsed().as_secs_f64() / f64::from(CALIBRATION_ROUNDS);
            let rounds = TARGET_DISPATCH.as_secs_f64() / per_round.max(1e-9);
            self.rounds_per_dispatch = (rounds as u32).clamp(10, STRETCH_ROUNDS);
            self.calibrated_for = seeds.len();
            done = CALIBRATION_ROUNDS;
        }

        while done < STRETCH_ROUNDS {
            let rounds = self.rounds_per_dispatch.min(STRETCH_ROUNDS - done);
            self.dispatch(count, rounds);
            done += rounds;
        }

        let buffers = self.buffers.as_ref().expect("buffers allocated");
        let size = u64::from(count) * 32;
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Electrum Readback"),
            });
        encoder.copy_buffer_to_buffer(&buffers.states, 0, &buffers.staging, 0, size);
        let submission = self.ctx.queue.submit(Some(encoder.finish()));

        let slice = buffers.staging.slice(0..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let result = (|| -> Result<Vec<[u8; 32]>> {
            // The whole batch is queued at this point, so allow for all of it.
            let budget = POLL_TIMEOUT
                + TARGET_DISPATCH * (STRETCH_ROUNDS / self.rounds_per_dispatch.max(1)) * 4;
            self.ctx
                .device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: Some(budget),
                })
                .map_err(|e| anyhow!("GPU poll failed: {e:?}"))?;
            rx.recv_timeout(POLL_TIMEOUT)
                .map_err(|e| anyhow!("GPU readback callback not received: {e}"))?
                .map_err(|e| anyhow!("failed to map GPU readback buffer: {e:?}"))?;

            let data = slice.get_mapped_range();
            let digests = data
                .as_chunks::<32>()
                .0
                .iter()
                .map(|state| {
                    let mut digest = [0u8; 32];
                    for (out, word) in digest
                        .as_chunks_mut::<4>()
                        .0
                        .iter_mut()
                        .zip(state.as_chunks::<4>().0)
                    {
                        *out = u32::from_le_bytes(*word).to_be_bytes();
                    }
                    digest
                })
                .collect();
            Ok(digests)
        })();
        buffers.staging.unmap();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_schedule_matches_reference_sha256() {
        use bitcoin::hashes::{sha256, Hash};
        // A 64-byte message hashed with the precomputed padding block must match.
        let kw = padding_block_kw();
        let msg = [0x5au8; 64];
        let mut state = SHA256_IV;
        let mut w: [u32; 64] = [0; 64];
        for (i, chunk) in msg.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*chunk);
        }
        let compress = |state: &mut [u32; 8], kw_at: &dyn Fn(usize) -> u32| {
            let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
            for i in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ (!e & g);
                let t1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(kw_at(i));
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let t2 = s0.wrapping_add(maj);
                h = g;
                g = f;
                f = e;
                e = d.wrapping_add(t1);
                d = c;
                c = b;
                b = a;
                a = t1.wrapping_add(t2);
            }
            for (s, v) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
                *s = s.wrapping_add(v);
            }
        };
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        compress(&mut state, &|i| SHA256_K[i].wrapping_add(w[i]));
        compress(&mut state, &|i| kw[i]);

        let expected = sha256::Hash::hash(&msg).to_byte_array();
        let got: Vec<u8> = state.iter().flat_map(|v| v.to_be_bytes()).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn gpu_stretch_matches_cpu() {
        let Ok(ctx) = pollster::block_on(GpuContext::new(0, crate::GpuBackend::Auto)) else {
            eprintln!("no GPU available, skipping");
            return;
        };
        let mut stretcher = GpuStretcher::new(ctx).unwrap();
        let seeds: Vec<[u8; 32]> = [
            "acb740e454c3134901d7c8f16497cc1c",
            "8edad31a95e7d59f8837667510d75a4d",
            "00000000000000000000000000000000",
        ]
        .iter()
        .map(|s| s.as_bytes().try_into().unwrap())
        .collect();
        let digests = stretcher.stretch(&seeds).unwrap();
        for (seed, digest) in seeds.iter().zip(&digests) {
            assert_eq!(*digest, super::super::stretch_digest(seed));
        }
    }
}
