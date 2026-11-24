---
title: How to Profile TPU Code
description: Practical Tutorials
published: true
---
## 9. Profiling TPU Code

Everything so far has been napkin math: count the FLOPs, count the bytes, divide by a hardware number. Profiling is basically that run in reverse: capture a trace, find the fat op, read its shapes, and compare its time against the roofline. The gap tells us where to dig.

## A Thousand-Foot View of the TPU Software Stack

Between our Python and the TPU there are several layers:

- **JAX:** What we actually write: NumPy-style linear algebra, no mention of hardware.
- **StableHLO:** `jax.jit` traces the function into this portable IR (intermediate representation).
- **HLO:** XLA lowers StableHLO to HLO, decides which ops get fused and how arrays sit in memory, splits the program across devices and adds the collectives (Chapter 10). So it's already the per-device program, and it's what a profile shows.
- **LLO:** A TPU-specific low-level IR where the hardware gets scheduled (copies between memories, buffers pushed through the systolic array), then machine code.
- **Pallas:** The escape hatch. When we can't steer the compiler from JAX, we write the kernel by hand.

So we fix performance in JAX but diagnose it in HLO, which `jax.jit(f).lower(*args).compile().as_text()` prints without a profiler.

## The JAX Profiler: A Multi-Purpose TPU Profiler

The profiler ships with JAX. We wrap our code in `jax.profiler.trace`, and it records every op's duration along with the HLO and memory usage:

```python
import jax
import jax.numpy as jnp

@jax.jit
def up_proj(x, w):
    with jax.named_scope("up_proj"):          # label that shows up in the Trace Viewer
        return jnp.einsum("bd,df->bf", x, w)

x, w = jnp.ones((256, 512)), jnp.ones((512, 2048))
up_proj(x, w).block_until_ready()             # compile first, outside the trace

with jax.profiler.trace("/tmp/jax-trace"):
    up_proj(x, w).block_until_ready()         # wait for the device before the trace closes
```

The trace opens in TensorBoard with the XProf plugin (`pip install xprof`, then `tensorboard --logdir /tmp/jax-trace`). Annotated screenshots of each view are at [jax-ml.github.io/scaling-book/profiling/](https://jax-ml.github.io/scaling-book/profiling/).

### Trace Viewer

The most useful view: a timeline of everything that ran on each TPU core.

- We usually look at one core (TPU:0), since every core typically runs the same program on its own shard.
- The top row, XLA Ops, is the real thing: actual HLO ops with actual durations. The rows under it are approximate, rebuilt from `jax.named_scope`, `jax.named_call` and the stack trace.
- A Transformer shows up as the same block repeated once per layer; zoom into one and we can pick out attention and the MLP.
- Clicking an op gives its HLO line, duration, source line, and a link into the Graph Viewer.
- Navigation is WASD, like a game: A/D to pan, W/S to zoom.

### How to read an XLA op

One HLO line is just a name, an output, some inputs and a few attributes. From the example profile:

```text
%fusion.1 = bf16[8,1024,4096]{2,1,0:T(8,128)(2,1)} fusion(bf16[8,1024,8192]{2,1,0:T(8,128)(2,1)} %fusion.31), kind=kCustom, calls=%all-reduce-scatter.1
```

- **Op name:** `%fusion.1`. A fusion is a bundle of ops the compiler merged into one: at most one matmul plus whatever pointwise VPU work hangs off it. A bare matmul shows up as `dot`.
- **Output dtype and shape:** `bf16[8,1024,4096]`, so 67,108,864 bytes at 2 bytes each, around 67 MB.
- **Layout:** `{2,1,0:T(8,128)(2,1)}`. Before the colon is the physical order of the dimensions, most minor first; read right to left it's 0, 1, 2, so memory matches the logical shape. `T(8,128)` means 8 by 128 tiles, and `(2,1)` is an inner tiling bf16 gets so every load is a multiple of 4 bytes.
- **Memory space:** a trailing `S(1)` inside the braces would mean VMEM. `S(0)`, or nothing like here, means HBM.
- **Inputs:** everything inside `fusion(...)`. Just `%fusion.31` here, a `bf16[8,1024,8192]` array (134 MB). Input names are how we walk the graph backwards.
- **Attributes:** `calls=%all-reduce-scatter.1` names the sub-computation the fusion wraps. 8192 wide goes in, 4096 comes out: a small ReduceScatter, summing partial results across devices and keeping its own half.

Tiling can mostly wait until it costs something: padding inflates odd shapes (an `f32[3,5]` with `T(2,2)` tiles becomes `[4,6]`, 24 elements instead of 15), and retiling copies aren't free.

The thing to burn in: **every shape in an op line is the shape on one device**, not the global array. Here's a toy feed-forward block on 8 fake CPU devices (`jax.config.update('jax_num_cpu_devices', 8)`) with a 4x2 mesh: batch sharded over `X`, the hidden dimension $F$ of both weights over `Y`, and global shapes $x[32, 4, 64]$, $W_{\text{in}}[64, 512]$ and $W_{\text{out}}[512, 64]$ (CPU HLO from JAX 0.10, trimmed, so no tiling):

```text
ENTRY %main.0_spmd (param: f32[8,4,64], param.1: f32[64,256], param.2: f32[256,64]) -> f32[8,4,64] {
  %ynn_fusion.1 = f32[32,256]{1,0} fusion(%bitcast, %param.1), kind=kCustom, calls=%fused_computation.1
  %ynn_fusion = f32[32,64]{1,0} fusion(%ynn_fusion.1, %param.2), kind=kCustom, calls=%fused_computation
  ROOT %all-reduce = f32[8,4,64]{2,1,0} all-reduce(%bitcast.1), channel_id=1, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, to_apply=%add.clone
}
```

The batch of 32 shows up as 8 and $F = 512$ as 256. The fusion lines never say "sharded"; they just have smaller shapes. The mesh only shows up in the collective's `replica_groups`: four groups of two, so the sum runs along the 2-wide model axis. To get global shapes back, we multiply each sharded dimension by its number of shards.

### Graph Viewer

The same HLO drawn as boxes and arrows. It's the easy way to open up a fusion and see what's inside, and hovering over a box usually shows the Python line behind it. The Trace Viewer tells us which op is slow; the Graph Viewer tells us what it is.

### Looking at a real(ish) example profile

The example is a mock Transformer on a TPU v2-8 (8 cores, 4x2), with $2.3 \times 10^{13}$ FLOPs/s per core and $1.2 \times 10^{11}$ bytes/s of bidirectional ICI.

**A matmul (FLOPs).** The MLP up-projection has local inputs `bf16[8,1024,8192]` and `bf16[8192,16384]`. With 4-way data parallelism on the batch and 2-way model parallelism on $F$, globally it's $X[32, 1024, 8192] \cdot W_{\text{in}}[8192, 32768]$. Each shard sees 8,192 tokens, way past compute-bound, so it's just FLOPs over FLOPs/s:

$$
T_{\text{math}} = \frac{2 \times 32 \times 1024 \times 8192 \times 32768 \text{ FLOPs}}{8 \times 2.3 \times 10^{13} \text{ FLOPs/s}} = \frac{1.759 \times 10^{13}}{1.84 \times 10^{14}} = \mathbf{95.6 \text{ ms}}
$$

The trace shows 96.99 ms. Right on the roofline.

**A collective (bytes).** The ReduceScatter we decoded above moves a `bf16[8,1024,8192]` array of partial sums over a model axis only 2 wide, so it's a single hop:

$$
T_{\text{comms}} = \frac{2 \times 8 \times 1024 \times 8192 \text{ bytes}}{1.2 \times 10^{11} \text{ bytes/s}} = \mathbf{1.12 \text{ ms}}
$$

The trace shows 1.13 ms.

**Attention.** The Q projection's op line is `bf16[8,1024,8192]` times `bf16[8192,8,256,1]` giving `bf16[8,8,1024,256]`, so 8 heads on each device. $2 \times 8 \times 1024 \times 8192 \times 8 \times 256 \approx 2.75 \times 10^{11}$ FLOPs over $2.3 \times 10^{13}$ FLOPs/s is around **11.95 ms**, an eighth of the up-projection. The screenshot shows 13.88 ms, about 86% of the roofline. The text says 32 heads, but the op shape implies 16, so we go with 16.

So for every wide bar we decide whether it's doing FLOPs or moving bytes, do one division, and compare. An op that's 2x or 10x over its number is where the time went.

### Memory Profile

HBM usage over the life of the program and where the peak is. It's the tool for debugging OOMs (out of memory errors), and it shows headroom too: in the example about 7.5 GB is parameters and about 8.5 GB is free.

## Worked Problems

**Question 1 [reading an unfamiliar profile]:** We're given a [Colab and its profile](https://colab.research.google.com/drive/1LfLO3OTr-_MWFPxUN36KJ3cqH0BcAoli?usp=sharing). From the profile alone, before reading the code, work out what looks suspicious, exactly what computation is happening and what each op is doing, and what the true shapes of the matrices are and how they're sharded.

- We start on the XLA Ops row for TPU:0 and list one step: a `reduce`, two big fusions and an `all-reduce`. Then we click the widest ops and read their lines. The solution quotes the first big fusion:

  ```text
  %fusion.1 = bf16[4096]{0:T(1024)(128)(2,1)} fusion(bf16[4096,8192]{1,0:T(8,128)(2,1)} %param.1, bf16[8192]{0:T(1024)(128)(2,1)} %reduce.6), kind=kLoop, calls=%fused_computation.1
  ```

- The output is `bf16[4096]`. The inputs are `%param.1`, a `bf16[4096,8192]` matrix straight from the arguments (so a weight), and a `bf16[8192]` vector from `%reduce.6`. 8192 goes in and 4096 comes out, so 8192 is contracted. Per shard it's a matrix-vector product:

  $$
  \text{bf16}[8192] \cdot \text{bf16}[4096, 8192] \to \text{bf16}[4096]
  $$

- The final `all-reduce` has `replica_groups={{0,16,32,48,64,80,96,112}, ...}`. Each group has 8 devices, so partial results are summed 8 ways: 8-way model parallelism. Meaning, the 4096 we see is one eighth of the real dimension.
- **What computation is happening?** Two matmuls back to back with $D = 8192$ and $F = 32768$: $x[B, D] \cdot W_1[F, D] \to h[B, F]$, then $h[B, F] \cdot W_2[D, F] \to y[B, D]$. Up to the wide dimension and back down, like an MLP block with the nonlinearity taken out.
- **What are the true shapes and how are they sharded?**
    - $4096 \times 8 = 32{,}768$, so the first matmul is really $\text{bf16}[8, 8192] \cdot \text{bf16}[32768, 8192] \to \text{bf16}[8, 32768]$.
    - $W_1$ is sharded 8 ways along $F$, so $W_1[F_X, D]$. Each device holds $4096 \times 8192 \times 2 = 67{,}108{,}864$ bytes (64 MiB) of it.
    - The first matmul contracts over $D$, which is whole on every device, so no communication and $h[B, F_X]$ comes out.
    - The second contracts over the sharded $F$. Each device ends up with a full-shaped partial result, $y[B, D]\{U_X\}$, and the `all-reduce` adds them up: Chapter 3's "both multiplicands have sharded contracting dims" case.
- **What is each op doing?** The `reduce` makes the `bf16[8192]` vector, the first fusion is each device's slice of matmul 1, the second should be its partial sum for matmul 2 (`bf16[4096]` in, `bf16[8192]` out), and the `all-reduce` sums those across each group of 8.
- **What looks suspicious?** Per shard, one fusion does $2 \times 4096 \times 8192 = 67{,}108{,}864$ FLOPs and has to pull 67,108,864 bytes of weights out of HBM (plus 24,576 bytes of vectors):

  $$
  \text{Intensity} = \frac{67{,}108{,}864 \text{ FLOPs}}{67{,}133{,}440 \text{ bytes}} \approx \mathbf{1.0 \text{ FLOPs/byte}}
  $$

- A v5e needs around 240 FLOPs/byte to be compute-bound, so these "big" matmuls are memory-bound by more than two orders of magnitude. On a v5e the weight load alone would take 82 μs, and the screenshot shows 209 μs for `fusion.1`, slower than even that.
- So I think the suspicious part is that each device's activation is a single vector. The op line has no batch dimension, and the profiler's FLOP count is for exactly one row. Matrix-vector products are the worst case for a TPU: nearly all the time goes to streaming 64 MiB of weights at about one FLOP per byte. The op is also `kind=kLoop`, where the Transformer matmuls in the notes are `kind=kOutput`, so the compiler isn't treating these as normal matmuls.
- The device ids in `replica_groups` also run up to 112. If those are plain device indices, this profile came from at least 113 devices, not an 8-core slice.
- So: two chained matmuls, $x[8, 8192]$ up to 32,768 and back down to 8,192, with the weights split 8 ways along the 32,768 dimension and one AllReduce to add up the partial outputs.

This question needs a TPU runtime for the Colab, so I'll come back to it.

**Question 2 [mock Transformer benchmark]:** The [Transformer Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) behind the annotated profile in the notes implements a simple mock Transformer. Run it on an 8-core TPU slice (Colab no longer offers the v2-8, so Kaggle or an 8-core GCP slice) and benchmark the naive version with GSPMD partitioning. How long does each part take, and how long should it take? Which sharding is it using? Then fix the sharding (hint: `jax.lax.with_sharding_constraint`); what's the best MFU (model FLOPs utilization) we can get? For reference, the initial version ran at roughly 184 ms/layer and the optimized one at 67 ms/layer, and with a newer XLA compiler it's now roughly 90 ms/layer and 80 ms/layer. Finally, from the profile alone: what sharding strategy is this, what are the batch size, $d_{\text{model}}$ and $d_{\text{ff}}$, how does the time split between attention and the MLP block, and what share of the time should each op take at the roofline?

- **Which config?** The benchmark's config isn't given, so we use the annotated profile's: $B = 32$, $T = 1024$, $D = 8192$, $F = 32768$, $H = 256$ and $N = 16$.
- **How long should each part take?** Per layer, forward only, with a plain two-matmul MLP and K, V and O the same size as $W_Q$. Each op's FLOPs over the slice's $8 \times 2.3 \times 10^{13} = 1.84 \times 10^{14}$ FLOPs/s:
    - MLP up- and down-projections: $2BTDF$, 1.759e13 FLOPs each, so 95.6 ms (39.5%) each.
    - Q, K, V and O projections: $2BTDNH$, 2.199e12 each, so 11.95 ms (4.94%) each.
    - Attention scores ($QK^T$) and the weighted sum over $V$: $2BT^2NH$, 2.75e11 each, so 1.49 ms (0.62%) each.
    - Whole layer: 4.453e13 FLOPs, 242.0 ms.
- So the MLP should take around 191 ms (79.0%) and attention around 51 ms (21.0%).
- Communication is small next to that. Each block ends with one reduction of a 134 MB array over the 2-wide model axis: 1.12 ms as a ReduceScatter, 2.24 ms as a full AllReduce. At most about 4.5 ms a layer, or 1.8%.
- **Which sharding is it using?** We read it off the local shapes and collectives, same as Question 1. A local batch of $B/4$ and a local hidden dim of $F/2$ means 4-way data parallelism with 2-way model parallelism. Signs the partitioner picked something worse: a matmul that sees the full $F$ while the batch is only split 4 ways (duplicated FLOPs), a collective on the fat `[B/4, T, F]` intermediate (4x the bytes, 4.47 ms per hop instead of 1.12 ms), or `copy` ops from retiling.
- **Fixing it.** `jax.lax.with_sharding_constraint` pins the sharding of an intermediate so the compiler doesn't have to guess. We keep the fat intermediate sharded on batch and $F$, and bring the block output back to batch only. This is a sketch: `mesh`, `data` and `model` stand in for whatever the Colab calls them.

  ```python
  import jax
  import jax.numpy as jnp
  from jax.sharding import NamedSharding, PartitionSpec as P

  def ffw(x, w_in, w_out):
      h = jnp.einsum('btd,df->btf', x, w_in)
      h = jax.lax.with_sharding_constraint(h, NamedSharding(mesh, P('data', None, 'model')))
      y = jnp.einsum('btf,fd->btd', h, w_out)
      return jax.lax.with_sharding_constraint(y, NamedSharding(mesh, P('data', None, None)))
  ```

- On JAX 0.10 the mesh axes have to be `AxisType.Auto` for this, since `with_sharding_constraint` raises on the `Explicit` axes `jax.make_mesh` gives by default.
- **What's the best MFU?** MFU is just roofline time over measured time. The annotated run takes about 275 ms a layer against the 242.0 ms roofline, so about 88%. The reference timings must be for a smaller run, since 184 ms is already less than the 191 ms the MLP alone needs. The first Trace Viewer screenshot fits: a 79.057 ms layer with one sequence per device and a quarter of the FLOPs, so a roofline of about 60.5 ms. Against that, 184 ms is about 33% MFU and 67 ms is **around 90% MFU**. The fix was a $184 / 67 = 2.75\times$ speedup when the problem was written and only $90 / 80 = 1.125\times$ now, so these days the compiler finds most of it by itself.
- **From the profile alone:**
    - **Sharding strategy:** 4-way data parallelism on the batch, 2-way Megatron-style model parallelism on $F$ and the heads, plus FSDP on the weights going by the AllGather labels.
    - **Batch size, $d_{\text{model}}$ and $d_{\text{ff}}$:** multiply the up-projection's local `bf16[8,1024,8192]` and `bf16[8192,16384]` back up, giving $B = 32$, $T = 1024$, $d_{\text{model}} = 8192$, $d_{\text{ff}} = 32768$.
    - **Attention vs. the MLP:** measure the two named scopes inside one layer. The roofline says 21.0% and 79.0%.
    - **Each op's share at the roofline:** the percentages above, and they're the same for the smaller run.

This question needs an 8-core TPU slice, so I'll come back to it.
