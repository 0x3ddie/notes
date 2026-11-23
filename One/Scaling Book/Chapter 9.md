---
title: How to Profile TPU Code
description: Practical Tutorials
published: true
---
## 9. Profiling TPU Code

Everything so far has been napkin math: count the FLOPs, count the bytes, divide by a hardware number. This chapter is about checking that math against what the chip actually does. The loop is basically: capture a trace, find the fat op, read its shapes, and compare how long it took against the roofline number we'd predict for it. If the two don't match, the gap tells us where to dig.

### The Software Stack

There are several layers between the Python we write and the TPU:

- **JAX:** What we actually write: NumPy-style linear algebra with no mention of hardware.
- **StableHLO:** `jax.jit` traces the function and emits this portable, platform-agnostic IR (intermediate representation).
- **HLO:** The XLA compiler lowers StableHLO into HLO and runs its passes here: which ops get fused together and how arrays are laid out in memory. XLA is also what splits the program across devices and adds the collectives (Chapter 10), so the HLO we see is already the per-device program. This is what a profile shows us, so it's the layer to learn to read.
- **LLO:** A TPU-specific low-level IR where the hardware gets scheduled: copies between the TPU's memories, buffers pushed into the systolic array and pulled back off. From here it compiles to machine code.
- **Pallas:** The escape hatch. When the compiler does something bad down low and we can't steer it from JAX, we write the kernel by hand.

So we fix performance at the JAX level, but we diagnose it at the HLO level. We don't even need a profiler to see HLO; `jax.jit(f).lower(*args).compile().as_text()` prints it.

### Capturing a Trace

The profiler ships with JAX. We wrap the code we care about in `jax.profiler.trace`, and it records how long every op took along with the HLO and the memory usage:

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

The warm-up call keeps compilation out of the trace, and `block_until_ready()` is there because JAX hands back control before the device has finished.

Per the book, the trace opens in TensorBoard with the XProf plugin (`pip install xprof`, then `tensorboard --logdir /tmp/jax-trace`). The views we care about are the Trace Viewer, the Graph Viewer and the Memory Profile; the book has annotated screenshots of each at [jax-ml.github.io/scaling-book/profiling/](https://jax-ml.github.io/scaling-book/profiling/).

### Trace Viewer

The most useful view. It's a timeline of everything that happened on each TPU core.

- We usually look at one core (TPU:0). Typically every core runs the same program on its own shard, so one is enough.
- The top row, XLA Ops, is the real thing: actual HLO ops with actual durations. The rows underneath are approximate, rebuilt from `jax.named_scope`, `jax.named_call` and the Python stack trace, which is why the label in the snippet is worth adding.
- A Transformer shows up as the same block repeated once per layer. Zoom into one repeat and we can pick out attention and the MLP.
- Clicking an op gives its HLO line, its duration, the line of code it came from, and a link into the Graph Viewer.

Navigation is WASD, like a game: A/D to pan, W/S to zoom.

### Reading an XLA Op

HLO looks scary, but one line is just a name, an output, some inputs and a few attributes. Here's one from the book's example profile:

```text
%fusion.1 = bf16[8,1024,4096]{2,1,0:T(8,128)(2,1)} fusion(bf16[8,1024,8192]{2,1,0:T(8,128)(2,1)} %fusion.31), kind=kCustom, calls=%all-reduce-scatter.1
```

Field by field:

- **Op name:** `%fusion.1`. A fusion is a bundle of ops the compiler merged into one: at most one matmul plus whatever pointwise VPU work hangs off it. A bare matmul shows up as `dot`.
- **Output dtype and shape:** `bf16[8,1024,4096]`. bf16 is 2 bytes per element, so this output is 67,108,864 bytes, around 67 MB.
- **Layout:** `{2,1,0:T(8,128)(2,1)}`. The numbers before the colon say how the dimensions are ordered in physical memory, starting with the most minor one. Read right to left it's 0, 1, 2, so the array is stored in the same order as its logical shape. `T(8,128)` says it's stored in tiles of 8 by 128 and padded up to a whole number of tiles. The extra `(2,1)` is an inner tiling that bf16 arrays get so every load is a multiple of 4 bytes.
- **Memory space:** a trailing `S(1)` inside the braces would mean the array lives in VMEM. `S(0)`, or nothing at all like here, means HBM.
- **Inputs:** everything inside `fusion(...)`. Just one here: `%fusion.31`, a `bf16[8,1024,8192]` array (134 MB) with the same layout. The name tells us which op feeds this one, which is how we walk the graph backwards.
- **Attributes:** `kind=kCustom, calls=%all-reduce-scatter.1`. The book doesn't go through these two. In the compiled HLO text, `calls` names the sub-computation the fusion wraps (the CPU listing below has the same field). So this op takes an 8192-wide array and hands back a 4096-wide one: the book calls it a little ReduceScatter, summing partial results across devices and keeping its own half.

The tiling can mostly be ignored until it costs something. Padding inflates small or odd shapes: an `f32[3,5]` with `T(2,2)` tiles gets padded to `[4,6]`, so 24 elements instead of 15, or 1.6x the memory. XLA will also sometimes insert a copy that retiles an array in the middle of a program, and the book says the overhead of those can be non-trivial.

### Shapes Are Local

The thing to burn in: **every shape in an op line is the shape on one device**, not the global array. Below is a toy feed-forward block compiled on 8 fake CPU devices (`jax.config.update('jax_num_cpu_devices', 8)`) with a 4x2 mesh: batch sharded over `X`, the hidden dimension $F$ of both weights sharded over `Y`, and global shapes $x[32, 4, 64]$, $W_{\text{in}}[64, 512]$ and $W_{\text{out}}[512, 64]$. This is CPU-backend HLO (JAX 0.10), trimmed to the entry line, the two matmuls and the collective, with the `metadata=` and `backend_config=` attributes cut. Being CPU, it has no tiling or `S(1)` annotations and the fusions are named differently than on a TPU:

```text
ENTRY %main.0_spmd (param: f32[8,4,64], param.1: f32[64,256], param.2: f32[256,64]) -> f32[8,4,64] {
  %ynn_fusion.1 = f32[32,256]{1,0} fusion(%bitcast, %param.1), kind=kCustom, calls=%fused_computation.1
  %ynn_fusion = f32[32,64]{1,0} fusion(%ynn_fusion.1, %param.2), kind=kCustom, calls=%fused_computation
  ROOT %all-reduce = f32[8,4,64]{2,1,0} all-reduce(%bitcast.1), channel_id=1, replica_groups={{0,1},{2,3},{4,5},{6,7}}, use_global_device_ids=true, to_apply=%add.clone
}
```

The batch of 32 shows up as 8 (split 4 ways over `X`) and $F = 512$ as 256 (split 2 ways over `Y`). The up-projection outputs `f32[32,256]`: $8 \times 4 = 32$ flattened rows by $F/2$ columns. The fusion lines never say "sharded"; they just have smaller shapes. Where the mesh does show is the collective at the end: `replica_groups` lists which devices talk to each other, here four groups of two. So the sum runs along the 2-wide axis (our model parallelism), and the group size is the size of that mesh axis. To get global shapes back, we multiply each sharded dimension by its number of shards.

### Graph Viewer

The Graph Viewer draws the same HLO as boxes and arrows. It's the easy way to open up a fusion and see what got merged into it, and hovering over a box usually shows the line of Python that produced it. The Trace Viewer tells us which op is slow; the Graph Viewer tells us what that op is.

### Checking an Op Against the Roofline

The book walks through a profile of a mock Transformer on a TPU v2-8 (8 cores, 4x2). The v2 isn't in our reference tables; the book uses $2.3 \times 10^{13}$ FLOPs/s per core and $1.2 \times 10^{11}$ bytes/s of bidirectional ICI bandwidth.

**A matmul (FLOPs).** The MLP up-projection is a fusion with local inputs `bf16[8,1024,8192]` and `bf16[8192,16384]`. Per the book, the sharding is 4-way data parallelism on the batch and 2-way model parallelism on $F$, so globally it's $X[32, 1024, 8192] \cdot W_{\text{in}}[8192, 32768]$. Each shard sees $8 \times 1024 = 8{,}192$ tokens, way past the couple hundred we need to be compute-bound (240 on a v5e), so the estimate is just FLOPs over FLOPs/s:

$$
T_{\text{math}} = \frac{2 \times 32 \times 1024 \times 8192 \times 32768 \text{ FLOPs}}{8 \times 2.3 \times 10^{13} \text{ FLOPs/s}} = \frac{1.759 \times 10^{13}}{1.84 \times 10^{14}} = \mathbf{95.6 \text{ ms}}
$$

The book reports 96 ms for this op in the trace (96.99 ms in its screenshot). Right on the roofline.

**A collective (bytes).** The ReduceScatter we decoded above moves bytes instead of doing math. Each device holds a `bf16[8,1024,8192]` array of partial sums and the model axis is only 2 wide, so it's a single hop:

$$
T_{\text{comms}} = \frac{2 \times 8 \times 1024 \times 8192 \text{ bytes}}{1.2 \times 10^{11} \text{ bytes/s}} = \frac{134{,}217{,}728}{1.2 \times 10^{11}} = \mathbf{1.12 \text{ ms}}
$$

The book reports 1.13 ms.

**Attention.** The book leaves the Q projection as an exercise. Its text gives $W_Q$ the shape $[8192, 32, 256]$ (model dim, heads, head dim), sharded over heads. The op line in its attention screenshot is the per-device view: `bf16[8,1024,8192]` times `bf16[8192,8,256,1]` gives `bf16[8,8,1024,256]`, so 8 heads on each device. Same recipe, straight from the local shapes on one core: $2 \times 8 \times 1024 \times 8192 \times 8 \times 256 \approx 2.75 \times 10^{11}$ FLOPs over $2.3 \times 10^{13}$ FLOPs/s comes to around **11.95 ms**. Exactly an eighth of the up-projection, since $8 \times 256 = 2048$ is an eighth of the local $F$. The screenshot shows 13.88 ms for this op, so it's at about 86% of the roofline.

The text and the screenshot don't agree on the head count. 8 heads per device with a 2-way split is 16 heads, not 32. With 32 heads split 2 ways (16 per device) the estimate would be 23.9 ms, which is more than the op actually took, so the op line is the one to go with.

**TODO(eddie):** check `n_heads` in the Colab. The book's text says 32; its attention screenshot shows 8 per device, which is 16 under a 2-way split.

So for every wide bar we decide whether it's doing FLOPs or moving bytes, do one division, and compare. An op that's 2x or 10x over its number is where the time went.

### Memory Profile

The Memory Profile plots HBM usage over the life of the program, so we can see what's allocated and where the peak is. This is the tool for debugging OOMs (out of memory errors). It also shows headroom: in the book's example about 7.5 GB is model parameters and about 8.5 GB is free, so that program could fit a lot more.

### Takeaway

Profiling is roofline analysis run in reverse. The trace gives us a measured time and the local shapes of every op. The shapes give us FLOPs and bytes, and the hardware numbers turn those into the time it should have taken. Most of the skill is reading one HLO line properly. The shapes are per-device and the collectives (with their `replica_groups`) give away the sharding; the layout braces can wait until a copy shows up.

## Worked Problems

Both problems are built on Colab notebooks and real TPU profiles. What's below is the on-paper half: what to look for, what the roofline says the times should be, and what the book's answer and screenshots show. None of the numbers below were measured for these notes; the measured half is marked TODO.

**Question 1 [reading an unfamiliar profile]:** The book links a [Colab and its profile](https://colab.research.google.com/drive/1LfLO3OTr-_MWFPxUN36KJ3cqH0BcAoli?usp=sharing) and asks us to work out, from the profile alone and before reading the code: what looks suspicious, exactly what computation is happening and what each op is doing, and what the true shapes of the matrices are and how they're sharded.

- This question is really a test of reading HLO lines: local shapes in, global shapes and sharding out. The op names, shapes and timings below come from the book's answer and from the screenshot printed with the question, not from a live trace.
- **How do we go about it?** We start on the XLA Ops row for TPU:0 and list what's in one step. According to the book there are four things: a `reduce`, two big fusions, and an `all-reduce`. Then we click the widest ops and read their lines. The book quotes the first big fusion:

  ```text
  %fusion.1 = bf16[4096]{0:T(1024)(128)(2,1)} fusion(bf16[4096,8192]{1,0:T(8,128)(2,1)} %param.1, bf16[8192]{0:T(1024)(128)(2,1)} %reduce.6), kind=kLoop, calls=%fused_computation.1
  ```

- We decode it the same way as in the notes. The output is `bf16[4096]`. The inputs are `%param.1`, a `bf16[4096,8192]` matrix that comes straight from the function's arguments (so a weight), and a `bf16[8192]` vector produced by `%reduce.6`. A $[4096, 8192]$ matrix and an 8192-vector go in and a 4096-vector comes out, so the 8192 dimension is being contracted. Per shard, this is a matrix-vector product:

  $$
  \text{bf16}[8192] \cdot \text{bf16}[4096, 8192] \to \text{bf16}[4096]
  $$

- Then we look for collectives, since those give away the sharding. The book's answer gives the final `all-reduce` as `replica_groups={{0,16,32,48,64,80,96,112}, ...}`. Each group has 8 devices, so partial results are being summed 8 ways: 8-way model parallelism. Meaning, the 4096 we see is only one eighth of the real dimension.
- **What computation is happening?** Two matmuls back to back. Writing $D = 8192$ and $F = 32768$ (the book's code calls them `w` and `f`), it's $x[B, D] \cdot W_1[F, D] \to h[B, F]$ and then $h[B, F] \cdot W_2[D, F] \to y[B, D]$. Up to the wide dimension and back down, like an MLP block with the nonlinearity taken out.
- **What are the true shapes and how are they sharded?**
    - $4096 \times 8 = 32{,}768$, so the first matmul is really $\text{bf16}[8, 8192] \cdot \text{bf16}[32768, 8192] \to \text{bf16}[8, 32768]$. This is what the solution says.
    - $W_1$ is sharded 8 ways along $F$; in Chapter 3 notation that's $W_1[F_X, D]$. Each device holds `bf16[4096, 8192]`, which is $4096 \times 8192 \times 2 = 67{,}108{,}864$ bytes (64 MiB), out of 536,870,912 bytes (537 MB) for the whole matrix.
    - The first matmul contracts over $D$, which is whole on every device. No communication needed, and the output comes out as $h[B, F_X]$.
    - The second matmul contracts over $F$, which is sharded on $h$ (and presumably on $W_2$ too). Each device ends up with a full-shaped but partial result, $y[B, D]\{U_X\}$, and the `all-reduce` adds them up. That's the "both multiplicands have sharded contracting dims" case from Chapter 3.
- **What is each op doing?** The `reduce` produces the `bf16[8192]` vector that feeds the first fusion. The first fusion is each device's slice of matmul 1. The book doesn't quote the second fusion, but by the same logic it should be each device's partial sum for matmul 2, `bf16[4096]` in and `bf16[8192]` out. The `all-reduce` sums those partial results across the 8 devices in each group.
- **What looks suspicious?** The book's answer doesn't say, so from here down it's a roofline read of the op line and the screenshot, not the book's answer. Per shard, one of these fusions does $2 \times 4096 \times 8192 = 67{,}108{,}864$ FLOPs, and to do it has to pull 67,108,864 bytes of weights out of HBM (the two vectors add another 24,576 bytes):

  $$
  \text{Intensity} = \frac{67{,}108{,}864 \text{ FLOPs}}{67{,}133{,}440 \text{ bytes}} \approx \mathbf{1.0 \text{ FLOPs/byte}}
  $$

- The profiler's own counters agree. The book's screenshot has `fusion.1` selected, and its details list `model_flops` as 67,104,768 (our count minus 4,096, since summing 8,192 products only takes 8,191 additions) and a bytes-accessed line that's cut off at the bottom edge of the image but looks like 67,133,440.
- A v5e needs around 240 FLOPs/byte to be compute-bound, and across our reference table the number runs from 156 to 575. So these "big" matmuls are memory-bound by more than two orders of magnitude, and at best their time is the weight bytes over the HBM bandwidth:

  $$
  T_{\text{fusion}} \approx \frac{67{,}133{,}440 \text{ bytes}}{W_{\text{HBM}}}
  $$

- The book doesn't say which chip produced this profile, so for scale: that's between 9 and 82 μs across the chips in our table (82 μs on a v5e; it's the same 67,108,864 bytes as the weight matrix in Chapter 2's Question 4). The math alone would take under half a microsecond on any of them.
- The screenshot gives `fusion.1` a wall duration of 209,227 ns, so about 209 μs. That's 2.6x slower than even the v5e's memory roofline, and it works out to $3.2 \times 10^{11}$ FLOPs/s: about 1.4% of a single TPU v2 core ($2.3 \times 10^{13}$), and a smaller share of any chip in our table. So whichever chip it was, around 99% of each fusion isn't math.
- The `all-reduce` is tiny too. If the second fusion outputs a `bf16[8192]` per device, the payload is 16,384 bytes, which is under half a microsecond of bandwidth on any ICI link in our table. So it should be latency-bound, the same situation as the $E = F = 256$ case in Chapter 3's pop quiz. (In the screenshot it's the short bar at the end of the step, a small fraction of either fusion.)
- So I think the suspicious part is that the activation on each device is a single vector. The op line has no batch dimension at all, and the profiler's FLOP count is for exactly one row. Two big-looking matmuls that are really matrix-vector products are the worst case for a TPU: nearly all of the time goes to streaming 64 MiB of weights for about one FLOP per byte. The op line also says `kind=kLoop` (the screenshot's category is `loop fusion`), where the matmuls in the book's Transformer screenshots are `kind=kOutput` with category `convolution fusion`. So the compiler isn't handling these the way it handles the Transformer's matmuls. Even with all 8 rows of the batch on every device the intensity would only be around 8. We'd need a couple hundred tokens per device before these chips are doing real work.
- One more thing in the `replica_groups` line: the device ids run up to 112 in steps of 16. If those ids are plain device indices (they are in the CPU listing in the notes, where 8 devices show up as ids 0 to 7), then this profile came from at least 113 devices, and 128 if the groups tile the id range (16 groups of 8). An 8-core slice would have to show different groups. That's inferred from one line; the book doesn't say what the profile was taken on.
- The book's answer stops at the shapes. It doesn't say which part it considers suspicious, what `%reduce.6` is reducing, or where the batch of 8 shows up (the quoted op has no batch dimension). So everything from "What looks suspicious?" down is inference, not something confirmed in a trace.

**TODO(eddie):** open the Colab on a TPU runtime and confirm the op names and timings: which chip it is, the two fusion durations against 67 MB over that chip's HBM bandwidth (the book's screenshot shows 209 μs for the first one), whether a `kLoop` fusion uses the MXU at all, the `all-reduce` duration, what `%reduce.6` reduces over, whether the batch of 8 is sharded or replicated, and how many devices the `replica_groups` cover.

So: two chained matmuls, $x[8, 8192]$ up to 32,768 and back down to 8,192, with the weights split 8 ways along the 32,768 dimension and one AllReduce to add up the partial outputs.

> A matmul with one row per device does about 1 FLOP per byte loaded, so nearly all of the op goes to moving weights instead of doing math.

**Question 2 [mock Transformer benchmark]:** The book's [Transformer Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) (the one behind the annotated profile in the notes) implements a simple mock Transformer. Run it on an 8-core TPU slice (Colab has stopped offering the v2-8, so Kaggle or an 8-core GCP slice) and benchmark the naive version with GSPMD partitioning. How long does each part take, and how long should it take? Which sharding is it using? Then fix the sharding (hint: `jax.lax.with_sharding_constraint`); what's the best MFU (model FLOPs utilization) we can get? For reference, the book says the initial version ran at roughly 184 ms/layer and the optimized one at 67 ms/layer, and that with a newer XLA compiler it's now roughly 90 ms/layer and 80 ms/layer. Finally, answer from the profile alone: what sharding strategy is this, what are the batch size, $d_{\text{model}}$ and $d_{\text{ff}}$, how does the time split between attention and the MLP block, and what share of the time should each op take at the roofline?

- The book gives no answer for this one, and the measured half needs a real TPU. What we can do on paper is work out what the trace should look like, so that when we open it we know which bars are wrong, and check that against the few numbers visible in the book's screenshots.
- **How long should each part take?** Per layer, for a forward pass, with a plain two-matmul MLP (the book's walkthrough shows an up-projection and a second matmul, no gating matrix) and regular multi-head attention:

  | **Op** | **FLOPs** |
  |---|---|
  | MLP up-projection | $2BTDF$ |
  | MLP down-projection | $2BTDF$ |
  | Q, K, V, O projections | $2BTDNH$ each |
  | Attention scores ($QK^T$) | $2BT^2NH$ |
  | Weighted sum over $V$ | $2BT^2NH$ |

- Each one divided by the slice's $8 \times 2.3 \times 10^{13} = 1.84 \times 10^{14}$ FLOPs/s is its roofline time. The whole layer is $4BTDF + 8BTDNH + 4BT^2NH$.
- **Which config?** The book's text doesn't give the config of the benchmark cell. The concrete numbers come from the book's annotated profile of the same Colab, the one walked through in the notes: $B = 32$, $T = 1024$, $D = 8192$, $F = 32768$, $H = 256$, and 8 heads per device in the Q projection's op line, so $N = 16$ under the 2-way model split. (The book's text says 32 heads. That would make each projection 23.9 ms and the layer 292.8 ms, which is longer than the layer in the book's own screenshot, so the table uses the op line.) Plugging those in, and assuming K, V and O are the same size as the $W_Q$ the book shows:

  | **Op** | **FLOPs** | **Roofline time** | **Share of layer** |
  |---|---|---|---|
  | MLP up-projection | 1.759e13 | 95.6 ms | 39.5% |
  | MLP down-projection | 1.759e13 | 95.6 ms | 39.5% |
  | Q, K, V, O projections (each) | 2.199e12 | 11.95 ms | 4.94% |
  | Attention scores | 2.75e11 | 1.49 ms | 0.62% |
  | Weighted sum over V | 2.75e11 | 1.49 ms | 0.62% |
  | **Whole layer** | 4.453e13 | 242.0 ms | 100% |

- So for that config the MLP should take around 191 ms (79.0%) and attention around 51 ms (21.0%). With $F = 4D$ and $NH = D/2$ the MLP share simplifies to $\frac{8D}{10D + T}$, so it sits near four fifths until $T$ gets comparable to $D$.
- Communication is small next to that. With 4-way data parallelism and 2-way model parallelism, each block ends with one reduction of a `bf16[8,1024,8192]` array (134 MB) over the 2-wide model axis: 1.12 ms as a ReduceScatter, 2.24 ms as a full AllReduce. Two blocks per layer is at most about 4.5 ms, or 1.8% of the layer. The book's screenshots also label FSDP AllGathers in front of the matmuls (both of the up-projection's inputs come out of `all-gather` ops), and this count leaves those out.
- **How long does each part actually take?** That's the measured half (see the TODO). The only measured numbers available are the book's own for the annotated run: the up-projection at 96.99 ms (roofline 95.6 ms), the ReduceScatter at 1.13 ms (roofline 1.12 ms), the Q projection at 13.88 ms (roofline 11.95 ms), and a ruler across one whole layer in the screenshot that reads about 275 ms (its label is partly covered by an annotation). Against the 242.0 ms roofline, that run is at an MFU of about 88%.
- **Does this match the reference timings?** No, and that's the interesting part. 184 ms is already less than the 191 ms the MLP alone needs at the roofline, and 67 ms is less than the 96.99 ms the up-projection alone takes in the book's own trace. At 100% MFU, 67 ms on this slice buys at most $1.23 \times 10^{13}$ FLOPs, which is 3.6x fewer than the annotated config needs. So the reference timings have to be for a smaller config than the annotated profile (or for faster chips).
- The book's first Trace Viewer screenshot (the one that introduces the tool) looks like that smaller config. Its ruler shows a 79.057 ms layer, close to the 80 ms/layer the book quotes for the optimized version today. The up-projection selected there has local inputs `f32[1,1024,8192]` and `bf16[8192,32768]`: one sequence per device and the whole $F$ on every device, so a quarter of the annotated run's FLOPs per device. It took 24.5 ms against a one-core roofline of 23.9 ms. If the rest of the model is the same (16 heads of 256), that layer's roofline is 47.8 ms of MLP plus 12.7 ms of attention, so about 60.5 ms. Against that, 184 ms is an MFU of about 33%, 90 ms about 67%, 80 ms about 76%, and 67 ms about 90%.
- So on paper, the book's best reference timing (67 ms/layer for the fixed version) works out to an MFU of around 90%. That's an inference from two of the book's screenshots and not a measurement: the head count in the smaller run isn't visible, and the book never says which config the reference timings belong to. The table above is a template either way; the recipe is the same once the real shapes are read off the trace.
- What we can say from the reference timings alone: fixing the sharding was a $184 / 67 = 2.75\times$ speedup when the problem was written, and only $90 / 80 = 1.125\times$ now. So the compiler finds most of the fix by itself these days. And once we know the layer's FLOPs, the MFU is just roofline time over measured time:

  $$
  \text{MFU} = \frac{T_{\text{roofline}}}{T_{\text{measured}}} = \frac{\text{layer FLOPs}}{1.84 \times 10^{14} \text{ FLOPs/s} \times T_{\text{measured}}}
  $$

- **Capturing and reading the profile:**
    1. Get an 8-core runtime and check that `len(jax.devices())` is 8.
    2. Run the benchmark once so it compiles, then wrap one more call in `jax.profiler.trace(...)` with `block_until_ready()` inside, like the snippet in the notes.
    3. In the Trace Viewer on TPU:0, measure one repeat of the layer block (that's the ms/layer), then click every wide XLA op and write down its name, its local shapes and its duration.
    4. For each one, compute the roofline from the local shapes: FLOPs over $2.3 \times 10^{13}$ for a matmul on one core, bytes over $1.2 \times 10^{11}$ for a collective.
    5. Apply the fix below, re-run, and compare.
- **What sharding is being used, and how do we tell?** From the local shapes and the collectives, same as Question 1. A local batch of $B/4$ and a local hidden dim of $F/2$ means 4-way data parallelism with 2-way model parallelism, which is what the annotated profile shows. Things that would mean the partitioner picked something worse:
    - A matmul fusion that sees the full $F$ while the batch is only split 4 ways, or the full batch with $F$ split. Then devices are duplicating each other's FLOPs, 2x or 4x over. (The full $F$ with the batch split 8 ways is fine. That's pure data parallelism, which is what the smaller run in the book's first screenshot looks like.)
    - A collective on the fat `[B/4, T, F]` intermediate. It's 4x the bytes of `[B/4, T, D]`: 537 MB and 4.47 ms per hop in the annotated config, against 134 MB and 1.12 ms.
    - An `all-gather` that brings back more than the matmul needs. Weight gathers by themselves are expected here (the book's screenshots label them as FSDP AllGathers), so we budget them like any other collective: the `bf16[8192,16384]` that the annotated up-projection takes is 268 MB, 2.24 ms per hop, and the whole $W_{\text{in}}$ would be 537 MB and 4.47 ms.
    - `copy` ops from retiling between layouts.
- **Fixing it.** `jax.lax.with_sharding_constraint` pins the sharding of an intermediate so the compiler doesn't have to guess. We keep the fat intermediate sharded on both batch and $F$, and bring the block output back to being sharded on batch only. This is a sketch of the idea, not the Colab's code: `mesh` is the Colab's mesh and isn't defined in the snippet, and the axis names `data` and `model` are placeholders for whatever that mesh calls them.

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

- On JAX 0.10 the mesh axes have to be `AxisType.Auto` for this: `with_sharding_constraint` raises on `Explicit` axes, which is what `jax.make_mesh` gives by default.
- Constraining the output to `P('data', None, 'model')` instead leaves $D$ split over the model axis, which is the `[8,1024,4096]` output of the ReduceScatter in the annotated profile. On the CPU backend that version compiles to an `all-reduce` followed by a `dynamic-slice`; the book's Graph Viewer screenshot shows the same pair and says the TPU compiles it to a reduce-scatter.
- **Answering from the profile alone:**
    - **What sharding strategy is this?** Read the local shapes and the `replica_groups`. In the annotated profile it's data parallelism on the batch (4-way) with Megatron-style model parallelism on $F$ and on the heads (2-way), plus FSDP on the weights going by the AllGather labels in the book's screenshots.
    - **What are the batch size, $d_{\text{model}}$ and $d_{\text{ff}}$?** Click the up-projection fusion. Its inputs are the local `[B/4, T, D]` and `[D, F/2]`, so we multiply back. In the annotated profile that's `bf16[8,1024,8192]` and `bf16[8192,16384]`, giving $B = 32$, $T = 1024$, $d_{\text{model}} = 8192$, $d_{\text{ff}} = 32768$.
    - **What fraction of time is attention vs. the MLP?** Measure the widths of the two named scopes inside one layer. The roofline says 21.0% and 79.0% for the annotated config.
    - **What fraction should each op take at the roofline?** The share column in the table above. It doesn't change for the smaller run, since a smaller batch scales every op by the same factor.

**TODO(eddie):** run the Colab on a real 8-core TPU slice (Kaggle or GCP) and paste the observed numbers: which TPU generation the runtime is (the $2.3 \times 10^{13}$ FLOPs/s per core used here is the v2 figure), the benchmark's actual config, ms/layer before and after the fix, the duration of each matmul and collective next to its roofline estimate, the attention vs. MLP split, and the resulting MFU. Also check: whether K, V and O really have the same shape as $W_Q$ (and the head count, see the TODO in the notes); whether $B$ in the annotated run is 32 (the book's text says 4-way data parallelism) or 16 (a label on its FFW screenshot says `(B // 2, T, F // 2)`); the exact layer time in the annotated run (about 275 ms, the ruler label is partly hidden); and whether the 79 ms run in the first screenshot really is the benchmark config.

So on paper: for the config in the book's annotated profile, one layer should take about 242 ms on a v2-8, roughly four fifths MLP and one fifth attention, with under 2% for the reductions, and the book's screenshot shows about 275 ms, so around 88% MFU. The question's own reference timings can't be that config. They fit a run a quarter of the size, where 67 ms/layer would be around 90% MFU, but the measured answer has to wait for the real shapes.
