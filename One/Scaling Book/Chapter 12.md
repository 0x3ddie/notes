---
title: How to Think About GPUs
published: true
---
## 12. Thinking About GPUs

This chapter redoes Chapters 2, 3 and 5 for NVIDIA GPUs (mostly the H100 and B200): the chip, the network, the collectives, and the training rooflines.

At the chip level a GPU looks a lot like a TPU: matmul hardware next to a stick of HBM. The difference is how the compute is cut up. Instead of 1 or 2 big TensorCores, a GPU has over a hundred small, mostly independent **Streaming Multiprocessors (SMs)**: 132 on an H100, 148 on a B200. The SM is the GPU's version of a TensorCore, and it has 4 quarters (subpartitions), each with:

- **Tensor Core:** The matmul unit, so the GPU's MXU (confusingly NOT the equivalent of a TPU TensorCore). Almost all the FLOPs/s live here: 990 TFLOPs/s in bf16 on an H100, against about 67 TFLOPs/s from the vector units.

- **CUDA Cores:** 32 fp32 ALUs for vector math (ReLUs, pointwise ops, sums), like the lanes of the TPU's VPU. All 32 run the same instruction each cycle, fed by the Warp Scheduler.

- **Register file:** 16,384 32-bit registers for the CUDA cores (256kB per SM across the 4 quarters).

The twist is SIMT (Single Instruction Multiple Threads): threads in a warp (a group of 32) can branch differently, but divergent branches still run one after the other. An SM also keeps up to 64 warps loaded and switches while one waits on memory; a TPU is basically single-threaded.

### GPU Memory

- **SMEM:** 256kB of L1 cache per SM (the GPU's VMEM), holding the Tensor Core inputs. The B200 adds another 256kB of TMEM per SM to feed its bigger Tensor Cores.

- **L2 cache:** About 50MB on an H100, shared by every SM. Half the size of a TPU's VMEM, much slower (measured around 5.5 TB/s against roughly 40 TB/s for VMEM), and not under our control.

- **HBM:** 80GB at 3.35 TB/s on an H100, 192GB at 8 TB/s on a B200.

For the Chapter 1 roofline, the H100's bf16 critical intensity is around 295 FLOPs/byte and the B200's around 281, next to 240 for a TPU v5e. Meaning, a matmul still needs a batch of a few hundred tokens to be compute-bound.

### GPUs vs. TPUs at the Chip Level

An H100 has $132 \times 4 = 528$ independent vector units of 32 lanes, against 8 units of 1,024 on a TPU v5p. So a GPU is more modular (unrelated kernels can share it with no planning) but less predictable, since a lot gets decided at runtime (what's in L2, which warp goes next). A single GPU is also a bigger chip: an H200 has around $2.2\times$ the FLOPs/s of a v5p and about $1.5\times$ the HBM. TPUs answer with more chips, a stronger network, and far more fast cache (128MB of VMEM on a v5p against 33MB of SMEM on an H100).

## On GPU Networking

This is where the designs really split. GPUs sit at the bottom of a tree of switches: any GPU can reach any other, but the bandwidth depends on how far up the tree a message climbs.

> **Note:** Two unit traps in every problem below. Links are sold in bits (a 400 Gbps cable is 50 GB/s), and they're full-duplex, so the quoted number applies each way at once. Bisection bandwidth is the one figure that usually adds both directions.

### Inside a Node

A **node** is 8 GPUs (H100 or B200) wired all-to-all through NVSwitches over NVLink, the telephone switchboard from Chapter 2. On an H100 each GPU has 18 NVLink 4.0 links at 25 GB/s each way, spread over the node's 4 NVSwitches (64 ports each), so $18 \times 25 = 450\text{ GB/s}$ in and out of every GPU. The B200 doubles the link speed to 50 GB/s, so 900 GB/s per GPU.

### Beyond the Node

Past the node, every GPU has its own 400 Gbps InfiniBand NIC, so a node of 8 has $8 \times 50 = 400\text{ GB/s}$ of egress. NVIDIA's reference H100 SuperPod adds two levels of 64-port InfiniBand switches at 50 GB/s per port:

- **Scalable Unit (SU):** 32 nodes (256 GPUs) under 8 leaf switches.

- **SuperPod:** 4 SUs (1,024 GPUs) under 16 spine switches. Every leaf connects to every spine.

It's a **fat tree**: the cabling gets thicker towards the top (12.8 TB/s out of each SU, 51.2 TB/s across the spine), so any node can send its full 400 GB/s to any other node. The diagrams on [jax-ml.github.io/scaling-book/gpus/](https://jax-ml.github.io/scaling-book/gpus/) help a lot here.

So two numbers to remember for an H100 cluster: **450 GB/s per GPU inside a node, 400 GB/s per node outside it.** That second one is only 50 GB/s per GPU, a $9\times$ drop the moment a message leaves the node. A TPU v5p has 540 GB/s per chip (6 links at 90 GB/s) at any pod size, but only to its neighbors.

The GB200 NVL72 puts 72 GPUs in one NVLink domain at 900 GB/s each, with 3.6 TB/s of node egress. The chips are only about twice as fast, so the cross-node rooflines below get roughly $4\times$ easier.

## Collectives on a Tree

Same four primitives as Chapter 3, but now the cost depends on whether we stay inside a node.

### Intra-node Collectives

AllGather and ReduceScatter run around a ring like on a TPU. Each of the $N - 1$ hops moves $\frac{1}{N}$ of the array at full GPU egress:

$$
T_{\text{AG or RS}} = \frac{\text{bytes} \cdot (N - 1)}{N \cdot W_{\text{GPU}}} \approx \frac{\text{bytes}}{W_{\text{GPU}}}
$$

So about bytes/450e9 on an H100, and an AllReduce is still $2\times$ that. AllToAll is where the switch pays off. No ring to crawl around; every GPU sends each chunk straight to its destination:

$$
T_{\text{AllToAll}} = \frac{\text{bytes} \cdot (N - 1)}{N^2 \cdot W_{\text{GPU}}} \approx \frac{\text{bytes}}{N \cdot W_{\text{GPU}}}
$$

On a torus this was $\text{bytes} / 4W$, so with $N = 8$ it's half the cost. For MoE routing, where only $k$ of the $N$ destinations get anything, it shrinks by another $k/N$.

With **SHARP** (in-network reductions) the switch itself sums what the GPUs send, so an AllReduce costs $\text{bytes} / W_{\text{GPU}}$ instead of twice that. In theory. Measured, it's more like a 30% gain. Spec numbers are optimistic in general: NVLink AllReduce tops out near 370 GB/s for huge messages (around 10GB per device), and a realistic 58MB shard sees closer to 150 GB/s, so treat everything here as an upper bound.

### Cross-node Collectives

Above the node, the fat tree gives every node 400 GB/s to anywhere, so we can picture one big ring through all the nodes with the node's egress as the bandwidth:

$$
T_{\text{AG or RS}} = \max\left[\frac{\text{bytes}}{W_{\text{GPU}}}, \frac{\text{bytes}}{W_{\text{node}}}\right] \approx \frac{\text{bytes}}{400 \times 10^9}
$$

AllToAll does NOT survive the trip. Every GPU really has to reach every other GPU, so all 8 GPUs of a node squeeze through the same 400 GB/s. Over $M$ nodes:

$$
T_{\text{AllToAll}} \approx \frac{\text{bytes}}{M \cdot W_{\text{node}}}
$$

Going from one node to two takes us from $\text{bytes} / (8 \times 450\text{e}9)$ to $\text{bytes} / (2 \times 400\text{e}9)$, which is $4.5\times$ slower.

### A Second Sharded Axis

On a TPU, if the array we're reducing over $X$ is also sharded over $Y$, the collective gets $Y$ times cheaper. On GPUs that's only half true. Say $Y$ is the inner axis (it fills a node first):

$$
T_{\text{node}} = \frac{\text{bytes}}{W_{\text{GPU}} \cdot \min(Y, 8)} \qquad T_{\text{scale-out}} = \frac{\text{bytes}}{W_{\text{node}}} \cdot \frac{8}{\max(Y, 8)}
$$

If $Y \le 8$, the node-level part gets cheaper, but the node still pushes the whole array through its one 400 GB/s uplink, so the total doesn't move. Only when $Y$ spans several nodes does the cross-node cost drop, in proportion to the nodes spanned. I think this is the one fact worth memorizing from the section, since it drives most of the sharding recipe below.

## LLM Rooflines on GPUs

Same setup as Chapter 5: one MLP block ($W_{\text{in}}[D, F]$ and $W_{\text{out}}[F, D]$), $B$ the batch in tokens, $C$ the FLOPs/s per GPU. The only new thing is that $W$ is 450e9 inside a node and 400e9 outside it, so for an H100 $C/W$ is either 2,200 or 2,475.

### 1. Data Parallelism

Pure DP and FSDP (ZeRO) cost the same: a weight-sized AllReduce (or ReduceScatter plus AllGather) in the backward pass. With $X$ replicas:

$$
T_{\text{math}} = \frac{8BDF}{X \cdot C} \qquad T_{\text{comms}} = \frac{8DF}{W} \implies \frac{B}{X} > \frac{C}{W}
$$

So we need around **2,500 tokens per GPU** (2,475 across nodes), against 850 on a TPU v5p using all three torus axes. Three adjustments:

- **MoE:** $E$ experts with $k$ active means $E$ times the weights for only $k$ times the FLOPs, so multiply the threshold by $E/k$. For $E = 128$ and $k = 4$ that's 79,200 tokens per GPU.

- **Only 2-way DP:** The ring factor $(X - 1)/X$ is $\frac{1}{2}$, so the threshold halves to about 1,238.

- **Model parallelism across nodes:** Each node only moves its own slice of the weights, so a replica spanning 8 nodes cuts the threshold by 8.

### 2. Tensor Parallelism

TP moves activations instead of weights: an AllGather and a ReduceScatter of $[B, D]$ around each block.

$$
T_{\text{math}} = \frac{4BDF}{Y \cdot C} \qquad T_{\text{comms}} = \frac{4BD}{W} \implies Y < \frac{F \cdot W}{C}
$$

That's $F/2200$ inside a node and $F/2475$ beyond it. For $F = 28{,}672$ it comes to 13 and 11.6, so TP fits comfortably in one 8-GPU node. Over exactly 2 nodes the discount gives $F > 2475 \cdot (Y - 8)$, which allows 16-way on paper (up to 19 for this $F$). Quiz 5 Question 2 checks how tight that is.

### 3. Expert Parallelism

For MoEs we shard the weights by expert, $W_{\text{in}}[E_Z, D, F]$, and AllToAll each token to wherever its experts live. Across nodes:

$$
T_{\text{math}} = \frac{4BkDF}{Z \cdot C} \qquad T_{\text{comms}} = \frac{4BD(Z - 8)}{W \cdot Z} \cdot \min\left(\frac{8k}{Z}, 1\right)
$$

When $Z$ is small (under $8k$, so a node or two), we need $F > \frac{C}{W} \cdot \frac{Z - 8}{k}$, which is easy. When $Z$ is large, we need $F > 8 \cdot \frac{C}{W}$, about 19,800 on H100s, and past that $Z$ can go all the way up to $E$.

### 4. Pipeline Parallelism

Splitting the layers across nodes only sends a small microbatch of activations between stages:

$$
T_{\text{per-layer comms}} \approx 1.5 \cdot \frac{2BD}{W \cdot N_{\text{layers}}}
$$

Basically free. The price is awkward code, pipeline bubbles, and a bad fit with FSDP (ZeRO-3 would AllGather the weights for every microbatch, so pipelined setups use ZeRO-1).

### Takeaway

Inside a node GPUs are great: 450 GB/s to any neighbor in one hop, and cheap AllToAlls. Leaving the node is a $9\times$ drop per GPU, and that shapes everything else. Anything that spreads a replica over several nodes makes the data-parallel reduction cheaper, which is why the usual recipe is TP or EP inside a node or two, pipelining across many nodes, and DP or ZeRO-1 on the outside. Faster chips on the same network only make this harder, since $C/W$ goes up.

## Worked Problems

Hardware numbers are under Reference Numbers at the end. One heads-up: the solutions mix chip numbers. For the H100 they mostly use 990e12 FLOPs/s and 3.35e12 bytes/s (the table rounds to 9.9e14 and 3.4e12), and for the B200 either 2250e12 or the table's 2.3e15 FLOPs/s. Each problem below follows its solution.

### Quiz 1: GPU hardware

**Question 1 [CUDA cores]:** How many fp32 CUDA cores (ALUs) does an H100 have? A B200? How does that compare with the number of independent ALUs in a TPU v5p?

- Just counting down the chip: SMs, times 4 subpartitions, times 32 fp32 cores. The H100 has 132 SMs and the B200 has 148:

  $$
  \text{H100: } 132 \times 4 \times 32 = \mathbf{16{,}896 \text{ CUDA cores}} \qquad \text{B200: } 148 \times 4 \times 32 = \mathbf{18{,}944 \text{ CUDA cores}}
  $$

- A TPU v5p has 2 TensorCores, each with a VPU of 4 independently programmable units of $8 \times 128$ lanes:

  $$
  2 \times 4 \times 8 \times 128 = \mathbf{8{,}192 \text{ ALUs}}
  $$

- So an H100 has about twice the vector lanes of a v5p ($2.06\times$, and $2.31\times$ for the B200) at roughly the same clock, but split into 528 independent units of 32 instead of 8 big units of 1,024.

**Question 2 [vector FLOPs calculation]:** An H100 has 132 SMs and a 1.59GHz clock (1.98GHz with boost). If every ALU does one vector op per cycle, how many vector fp32 FLOPs/s does the chip get, with and without boost? How does that compare with its matmul FLOPs/s?

- One op per core per cycle, so we just multiply the cores from Question 1 by the clock:

  $$
  16{,}896 \times 1.59 \times 10^9 = 2.69 \times 10^{13} = \mathbf{26.9 \text{ TFLOPs/s}}
  $$

  $$
  16{,}896 \times 1.98 \times 10^9 = 3.35 \times 10^{13} = \mathbf{33.5 \text{ TFLOPs/s (boost)}}
  $$

- The spec sheet says about double ($2 \times 33.5 = 67 \text{ TFLOPs/s}$) because it counts a fused multiply-add as 2 FLOPs per cycle. Fair for a dot product, not for most vector work.
- The Tensor Cores do 990 TFLOPs/s in bf16, around $30\times$ the boosted number ($37\times$ at base clock). Meaning, anything that isn't a matmul runs on a roughly $30\times$ slower machine.

**Question 3 [GPU matmul intensity]:** What is the peak fp16 matmul intensity on an H100? On a B200? What about fp8? Intensity here means the ratio of matmul FLOPs/s to memory bandwidth.

- The hardware critical intensity from Chapter 1, just plugging in the solution's numbers:

  $$
  \text{H100: } \frac{990 \times 10^{12} \text{ FLOPs/s}}{3.35 \times 10^{12} \text{ bytes/s}} = \mathbf{295 \text{ FLOPs/byte}} \qquad \text{B200: } \frac{2250 \times 10^{12}}{8 \times 10^{12}} = \mathbf{281 \text{ FLOPs/byte}}
  $$

- Both chips do exactly $2\times$ the FLOPs/s in fp8, so the fp8 intensities are **591** and **562**. (The solution doubles the rounded 295 and says 590.)
- Right next to the TPU v5e's 240, so a bf16 matmul needs a batch of around 280 to 300 tokens to be compute-bound here too.
- fp8 looks twice as demanding, but fp8 weights are 1 byte each instead of 2, which doubles the matmul's own intensity too. The batch we need stays about the same.

**Question 4 [matmul runtime]:** Using the answer to Question 3, how long should an `fp16[64, 4096] * fp16[4096, 8192]` matmul take on a single B200? How about `fp16[512, 4096] * fp16[4096, 8192]`?

- From Question 3, $B = 64$ should be HBM-bound and $B = 512$ FLOPs-bound (the crossover is around 281). We'll check both, with 8e12 bytes/s and the table's 2.3e15 FLOPs/s as in the solution.
- **$B = 64$:** read $x$ and $W$, write the output, 2 bytes per element:

  $$
  2BD + 2DF + 2BF = 524{,}288 + 67{,}108{,}864 + 1{,}048{,}576 = 68{,}681{,}728 \text{ bytes}
  $$

  $$
  T_{\text{comms}} = \frac{68{,}681{,}728 \text{ bytes}}{8 \times 10^{12} \text{ bytes/s}} = \mathbf{8.6 \ \mu\text{s}} \qquad T_{\text{math}} = \frac{4{,}294{,}967{,}296 \text{ FLOPs}}{2.3 \times 10^{15} \text{ FLOPs/s}} = 1.9 \ \mu\text{s}
  $$

  Memory wins, so about **8.6 μs**.

- **$B = 512$:** same steps.

  $$
  T_{\text{comms}} = \frac{79{,}691{,}776 \text{ bytes}}{8 \times 10^{12} \text{ bytes/s}} = 10.0 \ \mu\text{s} \qquad T_{\text{math}} = \frac{34{,}359{,}738{,}368 \text{ FLOPs}}{2.3 \times 10^{15} \text{ FLOPs/s}} = \mathbf{14.9 \ \mu\text{s}}
  $$

  Now the math is the slow part, so about **15 μs**.

- Notice the weights are 67MB of the roughly 69MB moved at $B = 64$. At small batch we're basically just streaming $W$ out of HBM.
- In practice the solution expects more like 10 to 12 μs and 20 μs, since we never get all the bandwidth or all the FLOPs.

**Question 5 [L1 cache capacity]:** What is the total L1/SMEM capacity of an H100? What about register memory? How does this compare with TPU VMEM capacity?

- Both are per-SM numbers, so we multiply by 132 SMs. SMEM is 256kB per SM:

  $$
  132 \times 256 \text{ kB} = 33{,}792 \text{ kB} = \mathbf{33 \text{ MB}}
  $$

- Registers: $4 \times 16{,}384 \times 4 = 262{,}144 \text{ bytes}$ (4 subpartitions of 16,384 four-byte registers), also 256kB per SM, so another **33 MB** per chip.
- Together that's about **66 MB** of fast memory, roughly half of a TPU v5p's 128MB of VMEM.
- The split is very different though. A TPU has only 256kB of register memory in total, and the H100 has 33MB. The solution's explanation is that VMEM is low-latency enough that spilling registers into it is cheap.

**Question 6 [calculating B200 clock frequency]:** NVIDIA reports that a B200 can do 80 TFLOPs/s of vector fp32 compute. Given that each CUDA core can do 2 FLOPs per cycle with an FMA (fused multiply-add), estimate the peak clock frequency.

- Question 2 run backwards. We're told the FLOPs/s and we know the core count, so we just solve for cycles per second.
- 18,944 CUDA cores from Question 1 at 2 FLOPs each is $18{,}944 \times 2 = 37{,}888$ FLOPs per cycle:

  $$
  \frac{80 \times 10^{12} \text{ FLOPs/s}}{37{,}888 \text{ FLOPs/cycle}} = \mathbf{2.11 \text{ GHz}}
  $$

- A bit above the H100's 1.98GHz boost. The solution calls it high but reasonable, since B200s are generally liquid cooled.

**Question 7 [estimating H100 add runtime]:** Using the figures above, how long should it take to add two `fp32[N]` vectors on a single H100? Work out both $T_{\text{math}}$ and $T_{\text{comms}}$. What is the arithmetic intensity of the operation? If you can get access, run it in PyTorch or JAX for $N = 1024$ and $N = 1024 \times 1024 \times 1024$ and compare.

- An add is a vector op, so it runs on the CUDA cores, not the Tensor Cores. FLOPs: one per element, so $N$. Bytes: two loads and one store at 4 bytes each, so $3 \times 4N = 12N$.

  $$
  \text{Intensity} = \frac{N \text{ FLOPs}}{12N \text{ bytes}} = \mathbf{\frac{1}{12} \text{ FLOPs/byte}}
  $$

- The hardware side is 33.5 TFLOPs/s (Question 2, boost, no FMA) over 3.35e12 bytes/s, about 10 FLOPs/byte. We're at $\frac{1}{12}$, memory-bound by a factor of around 120, and no batch size fixes that.

  $$
  T_{\text{math}} = \frac{N}{3.35 \times 10^{13}} \text{ s} \qquad T_{\text{comms}} = \frac{12N}{3.35 \times 10^{12}} = \frac{N}{2.79 \times 10^{11}} \text{ s}
  $$

  $$
  T = \max(T_{\text{math}}, T_{\text{comms}}) = \mathbf{\frac{N}{2.79 \times 10^{11}} \text{ s}}
  $$

- **$N = 1024$:** $T_{\text{comms}}$ is 3.7 nanoseconds. Nothing on a GPU really finishes that fast, so this one is latency-bound: the time is whatever the kernel launch costs.
- **$N = 1024^3$:** we move $12 \times 2^{30} = 12{,}884{,}901{,}888$ bytes (12 GiB, fits in 80GB):

  $$
  T = \frac{12{,}884{,}901{,}888 \text{ bytes}}{3.35 \times 10^{12} \text{ bytes/s}} = \mathbf{3.85 \text{ ms}}
  $$

- The solution quotes its own JAX timings: about 1.5 μs for $N = 65{,}536$ (roofline 0.23 μs, so latency-bound as expected) and 4.1 ms for $N = 1024^3$, close to the 3.85 ms roofline.
- To check it ourselves:

  ```python
  import time
  import jax
  import jax.numpy as jnp

  add = jax.jit(lambda a, b: a + b)

  for n in (1024, 1024**3):
      a = jnp.ones((n,), dtype=jnp.float32)
      b = jnp.ones((n,), dtype=jnp.float32)
      add(a, b).block_until_ready()  # compile once before timing
      t0 = time.perf_counter()
      for _ in range(100):
          add(a, b).block_until_ready()
      print(n, (time.perf_counter() - t0) / 100, "seconds per add")
  ```

- **TODO(eddie):** run this on a real H100 and paste the observed numbers. Expect microseconds (pure overhead) for $N = 1024$ and roughly 4 ms for $N = 1024^3$.

### Quiz 2: GPU nodes

**Question 1 [total bandwidth for H100 node]:** An 8xH100 node has 4 NVSwitches. What is the total bandwidth of the node? Hint: check the NVLink links and the NVSwitch ports separately.

- Bandwidth can run out at the links leaving the GPUs or at the switch ports, so we work out both and take the smaller.
- **Switches:** $4 \text{ switches} \times 64 \text{ ports} \times 25 \text{ GB/s} = 6.4 \text{ TB/s}$.
- **Links:** each GPU has 18 NVLinks ($5 + 4 + 4 + 5$ to the four switches):

  $$
  8 \text{ GPUs} \times 18 \text{ links} \times 25 \text{ GB/s} = \mathbf{3.6 \text{ TB/s}}
  $$

- The links are the limit, so the node has **3.6 TB/s** in total. The switches have room to spare (only 144 of their 256 ports are used), and since it's full-duplex, 3.6 TB/s can come back at the same time.

**Question 2 [bisection bandwidth]:** Cut a network into two equal halves and add up the bandwidth that crosses the cut. The bisection bandwidth is the smallest value we can get over every way of making that cut. What is it for an 8xH100 node? Hint: the figure usually counts flow in both directions.

- Any even split puts 4 GPUs on each side, and since every GPU reaches every other one through the switches, all splits look the same.
- Each GPU can push its full 450 GB/s across, so one way it's $4 \times 450 \text{ GB/s} = 1.8 \text{ TB/s}$. Adding the other direction:

  $$
  2 \times 4 \times 450 \text{ GB/s} = \mathbf{3.6 \text{ TB/s}}
  $$

- Same number as Question 1, and not by coincidence: whatever the 4 GPUs on one side can send, the switches carry across.

**Question 3 [AllGather cost]:** How long does an AllGather of an array of $B$ bytes take on an 8xH100 node, assuming it's throughput-bound? Then put numbers in for $\text{bf16}[D_X, F]$ with $D = 4096$ and $F = 65{,}536$.

- The ring AllGather from Chapter 3, with NVLink egress standing in for ICI. Each of the $N = 8$ GPUs starts with $\frac{B}{N}$, passes one shard per turn at 450 GB/s, and after $N - 1$ turns everyone has everything:

  $$
  T_{\text{comms}} = (N - 1) \cdot \frac{B}{N \cdot W_{\text{GPU}}} = \frac{7B}{8 \times 450 \times 10^9} \approx \frac{B}{450 \times 10^9}
  $$

- For our array, $B = 4096 \times 65{,}536 \times 2 = 536{,}870{,}912$ bytes:

  $$
  T_{\text{comms}} = \frac{536{,}870{,}912 \times 7}{8 \times 450 \times 10^9} = \mathbf{1.04 \text{ ms}}
  $$

- The lazy $B/W$ version gives 1.19 ms; the $\frac{7}{8}$ stops mattering in bigger rings. The solution says about 1.5 ms in practice, which fits NVLink not delivering its full 450 GB/s.

### Quiz 3: Beyond the node level

**Question 1 [fat tree topology]:** Take the reference 1,024-GPU H100 SuperPod: 128 nodes of 8 GPUs, one 400 Gbps NIC per GPU, 4 SUs of 32 nodes with 8 leaf switches each, 16 spine switches, 64-port switches everywhere, and 2 links from every leaf to every spine. Calculate the bisection bandwidth of the whole pod at the node level, and show that every link is sized so the pod keeps full bisection bandwidth. Hint: work out the link bandwidth and the switch bandwidth at each level.

- 400 Gbps is 50 GB/s per cable and per port. Full bisection bandwidth means every node can use its whole 400 GB/s however we cut the pod, so we check each level carries 400 GB/s per node.
- **Node to leaf:** 8 NICs per node, one cable to each of the SU's 8 leaves: $8 \times 50 = 400$ GB/s per node, 12.8 TB/s per SU. That's 32 of each leaf's 64 ports ($32 \times 50 = 1.6 \text{ TB/s}$).
- **Leaf to spine:** the other 32 ports go up, 2 cables to each of 16 spines: $8 \times 16 \times 2 \times 50 \text{ GB/s} = 12.8 \text{ TB/s}$ per SU.
- **Spine:** 2 cables from each of 32 leaves fills all 64 ports, so 3.2 TB/s per spine, and $16 \times 3.2 = 51.2$ TB/s is exactly $128 \times 400$ GB/s. Every level matches.
- For the bisection, split the nodes into two halves of 64, each sending its full 400 GB/s across:

  $$
  64 \times 400 \text{ GB/s} = 25.6 \text{ TB/s one way} \implies \mathbf{51.2 \text{ TB/s}} \text{ counting both directions}
  $$

- With only 1 cable per leaf-spine pair, a cut along the SUs would get just 12,800 GB/s one way, half of that, so the doubled uplinks are what make the tree "fat". Per GPU it's 50 GB/s each way, the $9\times$ drop from NVLink.

**Question 2 [scaling to a larger DGX pod]:** Say we want to train on 2,048 GPUs instead of 1,024. What's the simplest way to modify the DGX topology above? What about 4,096? Hint: there's no single right answer, but try to keep costs down, and keep link capacity in mind.

- The real constraint is ports: a leaf has 32 facing down and 32 up, and a spine has all 64 facing down.
- **2,048 GPUs:** keep the SU and double everything above it. 8 SUs is 64 leaves with $64 \times 32 = 2{,}048$ uplinks, so $2{,}048 / 64 = 32$ spines, each leaf running 1 cable to each (the same 1.6 TB/s as 2 cables to each of 16). That's $32 \times 3.2 = 102.4 \text{ TB/s}$, or 256 nodes at 400 GB/s. Still a full fat tree, with 96 switches instead of 48.
- **4,096 GPUs:** the ports run out. 16 SUs is 128 leaves, but a spine can only reach 64 leaves and a leaf only 32 spines, so two levels top out at exactly 2,048 GPUs. We need a third level, the core: per the solution, 128 spine and 64 core switches. Checking ports: 128 leaves have 4,096 uplinks, 128 spines (32 down, 32 up) pass 4,096 links on, and $64 \times 64 = 4{,}096$ core ports receive them. 204.8 TB/s at every level, 512 nodes at 400 GB/s.
- The cost: 320 switches for 4,096 GPUs, one per 12.8 GPUs against one per 21.3 in the smaller builds, plus an extra hop of latency. The cheaper option is to oversubscribe the top level and give up full bisection bandwidth.

### Pop Quizzes: collectives

**Pop Quiz 1 [AllGather time]:** On an 8xH100 node, with 450 GB/s of full-duplex bandwidth, how long does $\text{AllGather}(\text{bf16}[B_X, F])$ take for $B = 1024$ and $F = 16{,}384$?

- Same ring as Quiz 2 Question 3, smaller array, $2 \times 1024 \times 16{,}384 = 33{,}554{,}432$ bytes:

  $$
  T_{\text{comms}} \approx \frac{33{,}554{,}432 \text{ bytes}}{450 \times 10^9 \text{ bytes/s}} = 75 \ \mu\text{s} \qquad \text{or exactly} \qquad \frac{33{,}554{,}432 \times 7}{8 \times 450 \times 10^9} = \mathbf{65 \ \mu\text{s}}
  $$

- So around 65 to 75 μs, depending on whether we keep the $\frac{7}{8}$.

**Pop Quiz 2 [AllToAll time]:** Same 8xH100 node, 450 GB/s of unidirectional bandwidth. How long does $\text{AllToAll}_{X \to N}(\text{bf16}[B_X, N])$ take? And if only 4 of the 8 entries are known to be non-zero?

- Careful with the letters: $B$ is the batch dimension here, and $N = 8$ is both the second dimension and the number of GPUs. So the array is $V = 2BN = 16B$ bytes.
- Each GPU starts with $\frac{V}{N}$ bytes (its rows), keeps $\frac{1}{N}$ of them, and sends $\frac{V}{N^2}$ straight to each of the other $N - 1$ GPUs:

  $$
  T_{\text{comms}} = \frac{V \cdot (N - 1)}{N^2 \cdot W_{\text{GPU}}} = \frac{7V}{64 \times 450 \times 10^9} \approx \frac{V}{8 \times 450 \times 10^9} = \mathbf{\frac{V}{3.6 \times 10^{12}} \text{ s}}
  $$

- In terms of the batch, about $4.4 \times 10^{-12} \cdot B$ seconds ($3.9 \times 10^{-12} \cdot B$ with the exact $\frac{7}{64}$).
- **If only 4 of the 8 entries are non-zero:** each GPU only ships the non-zero half of every chunk, so multiply by $k/N = \frac{1}{2}$:

  $$
  T_{\text{comms}} \approx \frac{V}{2 \cdot N \cdot W_{\text{GPU}}} = \mathbf{\frac{V}{7.2 \times 10^{12}} \text{ s}}
  $$

- Meaning, inside a node an AllToAll acts like it has 3.6 TB/s, $8\times$ the AllGather bandwidth, because nobody forwards anyone else's data. That's why MoE expert routing is cheap as long as it stays in one node.

**Pop Quiz 3 [sharding along 2 axes]:** Say we want to do $\text{AllGather}_X(\text{bf16}[D_X, F_Y])$ over a single SU (256 chips), where $Y$ is the inner axis. How long will this take as a function of $D$, $F$ and $Y$?

- The array is $2DF$ bytes. With $Y$ inner, the $Y$ GPUs sharing a shard of $D$ fill a node first, and there are $X = 256/Y$ groups. Each GPU needs the other groups' $D$ shards for its own slice of $F$; the question is which link they come through.
- **Case 1, $Y \le 8$:** every node holds all $Y$ slices of $F$, so it needs every byte it doesn't already have, $\frac{31}{32}$ of the array, all through one 400 GB/s uplink:

  $$
  T_{\text{comms}} = \frac{2DF \cdot 31}{32 \times 400 \times 10^9} \approx \mathbf{\frac{2DF}{400 \times 10^9} \text{ s}}
  $$

  No $Y$ in it at all. Sharding $F$ inside the node shrinks each GPU's share, but the node downloads the same bytes.

- **Case 2, $Y > 8$:** each group spans $Y/8$ nodes, and a node holds 8 different slices of $F$, one per GPU. Each GPU pulls its own bytes through its own 50 GB/s share of the uplink:

  $$
  T_{\text{comms}} = \frac{2DF}{Y \times 50 \times 10^9} \cdot \frac{X - 1}{X} \approx \mathbf{\frac{2DF}{Y \times 50 \times 10^9} \text{ s}}
  $$

- At $Y = 8$ the cases agree ($8 \times 50 = 400$), and the $\frac{X - 1}{X}$ only matters once $X$ is small.
- Plugging in $D = 8192$, $F = 32{,}768$ (so $2DF = 536{,}870{,}912$ bytes):

|**$Y$**|**$X$**|**Nodes per group**|**$T$ with the ring factor**|**$T$ without it**|
|---|---|---|---|---|
|1 to 8|256 to 32|1|1.30 ms|1.34 ms|
|16|16|2|0.63 ms|0.67 ms|
|32|8|4|0.29 ms|0.34 ms|
|64|4|8|0.13 ms|0.17 ms|
|128|2|16|0.04 ms|0.08 ms|

- Meaning, 8-way model parallelism inside a node does nothing for the outer AllGather, and past that every doubling of $Y$ at least halves the cost.

### Quiz 4: Collectives

**Question 1 [SU AllGather]:** Take one SU with $M$ nodes of $N$ GPUs each and run an AllGather across all of it. Exactly how many bytes go into and out of a node-level switch? And the top-level switch?

- Call the array $B$ bytes, so each of the $MN$ GPUs starts with $\frac{B}{MN}$. Treat each node as one switch with $N$ GPUs below and one uplink above, and follow the bytes:
  - In from its GPUs, $N \cdot \frac{B}{MN} = \frac{B}{M}$, and the same $\frac{B}{M}$ out to the top.
  - In from the top, everything the node is missing: $\frac{B(M - 1)}{M}$.
  - Out to its GPUs, all but each one's own shard: $N \cdot \left(B - \frac{B}{MN}\right) = NB - \frac{B}{M}$.

  $$
  \text{Ingress} = \frac{B}{M} + \frac{B(M - 1)}{M} = \mathbf{B} \qquad \text{Egress} = \frac{B}{M} + NB - \frac{B}{M} = \mathbf{NB}
  $$

- **Top-level switch:** $\frac{B}{M}$ arrives from each of the $M$ nodes, so $\mathbf{B}$ in. Each node gets back the $\frac{B(M - 1)}{M}$ it's missing, so $\mathbf{B(M - 1)}$ out.
- Egress sets the time at both levels. At the node each GPU receives about $B$ at 450 GB/s; at the top each node receives $\frac{B(M - 1)}{M}$ at 400 GB/s:

  $$
  T_{\text{node}} \approx \frac{B}{450 \times 10^9} \qquad T_{\text{top}} = \frac{B \cdot (M - 1)}{M \times 400 \times 10^9} = \frac{B}{413 \times 10^9} \text{ for } M = 32
  $$

- The top level is slower, so an AllGather over a whole SU runs at about 413 GB/s, not far off a single node.

**Question 2 [single-node SHARP AR]:** Now a single node of $N$ GPUs doing an AllReduce with SHARP (in-network reductions). Exactly how many bytes does the switch take in and send out?

- Every GPU starts with its own unreduced $B$-byte copy. Split the array into $N$ chunks of $\frac{B}{N}$, with GPU $j$ in charge of finishing chunk $j$.
- **Step 1:** each GPU sends the switch every chunk but its own. Ingress $N \cdot \frac{B(N - 1)}{N} = B(N - 1)$.
- **Step 2:** the switch sums each chunk and sends the partial sum of chunk $j$ to GPU $j$. Egress $N \cdot \frac{B}{N} = B$.
- **Step 3:** GPU $j$ adds its own copy, which finishes the chunk, and sends it back. Ingress $B$.
- **Step 4:** the switch multicasts each finished chunk to the other $N - 1$ GPUs. Egress $B(N - 1)$.

  $$
  \text{Ingress} = B(N - 1) + B = \mathbf{NB} \qquad \text{Egress} = B + B(N - 1) = \mathbf{NB}
  $$

- Per GPU that's exactly $B$ out and $B$ in, so the time is exactly $B / W_{\text{GPU}}$, no $\frac{N - 1}{N}$ this time.
- Without SHARP each GPU egresses $\frac{2B(N - 1)}{N}$ (ReduceScatter plus AllGather), $1.75B$ for $N = 8$. So the theoretical speedup is $1.75\times$, the "75%" that turns into roughly 30% in practice.

**Question 3 [cross-node SHARP AR]:** An array $\text{bf16}[D_X, F_Y]$ is sharded over the $N$ GPUs of one node. With in-network reductions, how long does $\text{AllReduce}(\text{bf16}[D, F_Y] \{U_X\})$ take? What changes once there's more than one node?

- Decoding the notation: the node's $N = XY$ GPUs each hold an unreduced $\text{bf16}[D, F/Y]$, which is $\frac{2DF}{Y}$ bytes, and the $X$ GPUs sharing a slice of $F$ sum theirs.
- That's just Question 2 running $Y$ times side by side on $\frac{2DF}{Y}$ bytes each, so $\frac{2DF}{Y}$ each way per GPU:

  $$
  T_{\text{comms}} = \mathbf{\frac{2DF}{Y \cdot W_{\text{GPU}}}} = \frac{2DF}{Y \times 450 \times 10^9} \text{ s}
  $$

- Inside a node, sharding over $Y$ helps in full: $(X, Y) = (4, 2)$ sees an effective 900 GB/s, and $(2, 4)$ sees 1.8 TB/s.
- **More than one node:** now the reduction over $X$ crosses nodes. A node still holds $Y$ different slices of $F$, which can't be summed with each other, so it sends all of them up: $2DF$ bytes through its 400 GB/s uplink.

  $$
  T_{\text{comms}} = \frac{2DF}{W_{\text{node}}} = \frac{2DF}{400 \times 10^9} \text{ s}
  $$

- The $Y$ is gone. Going from $(2, 4)$ in one node to the same thing across nodes drops us from 1.8 TB/s to 400 GB/s, $4.5\times$ slower. Same lesson as Pop Quiz 3.

**Question 4 [spine level AR cost]:** Same setup as Question 3, except $Y = 256$, which pushes the AllReduce up to the spine level. How long does it take, again with in-network reductions?

- With $Y = 256$ each slice of $F$ lives on exactly one GPU per SU, and we reduce over the $X = 4$ SUs. Basically Pop Quiz 3's second case taken all the way.
- Each GPU holds $\frac{2DF}{256}$ bytes and, with SHARP, sends and receives exactly that through its own 50 GB/s NIC. All 256 can do it at once, since the fat tree gives each SU 12.8 TB/s up to the spine:

  $$
  T_{\text{comms}} = \frac{2DF}{256 \times 50 \times 10^9} = \mathbf{\frac{2DF}{12.8 \times 10^{12}} \text{ s}}
  $$

- That's $32\times$ the bandwidth of the $Y \le 8$ case. For the Pop Quiz 3 array it comes to around 42 μs instead of 1.34 ms. Without SHARP it would be $2 \cdot \frac{X - 1}{X} = 1.5\times$ longer.

**Question 5 [2-way AllGather cost]:** What does an AllGather of $B$ bytes cost over exactly 2 nodes? Keep the exact ring factors instead of the usual approximation, and work out the intra-node cost as well as the cross-node one.

- 2 nodes is 16 GPUs with $\frac{B}{16}$ each. Two links to check: the InfiniBand hop between nodes and the NVLink ring inside each.
- **Cross-node:** each node is missing the other half, $\frac{B}{2}$, and pulls it in at 400 GB/s. The usual ring formula with $M = 2$:

  $$
  T_{\text{cross}} = \frac{B \cdot (M - 1)}{M \cdot W_{\text{node}}} = \frac{B}{2 \times 400 \times 10^9} = \frac{B}{800 \times 10^9}
  $$

- **Intra-node:** each GPU's NIC brings in $\frac{1}{8}$ of the remote half, so every GPU now owns $\frac{B}{8}$ (its own $\frac{B}{16}$ plus $\frac{B}{16}$ remote). The other $\frac{7B}{8}$ reaches it over NVLink:

  $$
  T_{\text{node}} = \frac{B \cdot (N - 1)}{N \cdot W_{\text{GPU}}} = \frac{7B}{8 \times 450 \times 10^9} = \mathbf{\frac{B}{514 \times 10^9} \text{ s}}
  $$

- The node level is slower by about $1.56\times$. So a 2-node AllGather is bound by NVLink, not InfiniBand, and runs at 514 GB/s, same as one node.
- Compare a full SU, where the cross-node term is $B / 413\text{e}9$ and takes over. With 2 nodes, the $\frac{M - 1}{M}$ factor doubles the cross-node bandwidth. That's the "2-node discount" from the notes, and why 2-way data parallelism shows up so often.

### Quiz 5: LLM rooflines

**Question 1 [B200 rooflines]:** A B200 DGX SuperPod (NOT the GB200 NVL72) has $2\times$ the bandwidth within a node (900 GB/s of GPU egress) but the same bandwidth in the scale-out network (400 GB/s per node). The FLOPs/s are in the reference table. How does this change the model-parallel and data-parallel rooflines?

- Every roofline in the notes sees the hardware through $C/W$, so we just recompute it. $C$ goes from 990e12 to 2250e12 FLOPs/s, about $2.27\times$; $W$ doubles inside a node and stays put across nodes.

  $$
  \text{In a node: } \frac{2250 \times 10^{12}}{900 \times 10^9} = \mathbf{2{,}500} \qquad \text{Across nodes: } \frac{2250 \times 10^{12}}{400 \times 10^9} = \mathbf{5{,}625}
  $$

- **Model parallelism (TP):** inside a node the limit goes from $F/2200$ to $F/2500$. For $F = 28{,}672$ that's 13.0 down to 11.5, both still above a node's 8 GPUs, so nothing really changes. Across nodes it goes from $F/2475$ to $F/5625$, only about 5, so cross-node TP gets worse.
- **Data parallelism:** the reduction crosses nodes, so the per-GPU batch we need jumps from 2,475 to **5,625 tokens**.
- A faster chip on the same network is harder to keep compute-bound, not easier. That's what the GB200 NVL72 goes after: 3.6 TB/s of node egress brings the cross-node ratio down to 625.

**Question 2 [how to shard LLaMA-3 70B]:** We're training LLaMA-3 70B with bfloat16 weights and Adam, with the optimizer state kept in fp32.

**1. What's the smallest number of H100s that can hold the weights and the optimizer state?**

- 2 bytes per parameter for the bf16 weight, plus Adam's two moments at 4 bytes each:

  $$
  70 \times 10^9 \times (2 + 8) \text{ bytes} = 140 \text{ GB} + 560 \text{ GB} = 700 \text{ GB} \qquad \frac{700 \text{ GB}}{80 \text{ GB per H100}} = 8.75 \implies \mathbf{9 \text{ GPUs}}
  $$

- GPUs come in nodes of 8, so in practice that's **2 nodes**. And it's only a floor, with no room for gradients or activations.

**2. How long does training take on 4,096 H100s for 15T tokens at 45% MFU (Model FLOPs Utilization)?**

- Napkin math with the $6 \times \text{parameters} \times \text{tokens}$ rule from Chapter 4, over a cluster running at 45% of peak:

  $$
  6 \times 70 \times 10^9 \times 15 \times 10^{12} = 6.3 \times 10^{24} \text{ FLOPs} \qquad 4{,}096 \times 990 \times 10^{12} \times 0.45 = 1.82 \times 10^{18} \text{ FLOPs/s}
  $$

  $$
  \frac{6.3 \times 10^{24}}{1.82 \times 10^{18}} = 3.45 \times 10^6 \text{ s} \approx \mathbf{40 \text{ days}}
  $$

**3. The model has $F = 28{,}672$ and a batch of about 4M tokens. How much model parallelism can we use before it's comms-bound? On 4k chips, does that plus pure DP stay compute-bound? With ZeRO-3 instead? With 8-way pipelining added? Check GPU memory as well as communication.**

- **Model parallelism:** inside a node we can go up to $F / 2200 = 28{,}672 / 2200 \approx 13$-way, more than one node's 8 GPUs, so we stop at **8-way** TP, comfortably compute-bound. The solution lands on 8-way too.
- 16-way over 2 nodes passes the 2-node rule ($28{,}672 > 19{,}800$), but that rule only looks at the InfiniBand hop. Quiz 4 Question 5 showed a 2-node AllGather is really NVLink-bound at 514 GB/s, which makes the limit $F/Y > 2200 \times \frac{7}{8} = 1{,}925$, so $Y < 14.9$. Borderline at best.
- **TODO(eddie):** the solution divides by 1995 here ($28{,}672 / 1995 = 14.4$) where the notes give $990\text{e}12 / 450\text{e}9 = 2200$ (so 13.0). It isn't clear where 1995 comes from in the section's numbers; the node size caps it at 8 either way. Also decide whether to keep the 16-way aside: the section's text allows 16-way, its solution doesn't, and the NVLink argument is our own.
- **8-way TP plus pure DP:** that's $4{,}096 / 8 = 512$ replicas, each keeping its own 700 GB on 8 GPUs, so 87.5 GB per GPU against the H100's 80. Doesn't fit, so **no**.
- It fails on comms too: the per-GPU batch is $4 \times 10^6 / 4{,}096 \approx 977$ tokens against the 2,475 that cross-node DP needs, about $2.5\times$ short (we'd need a 10M token batch). 16-way TP wouldn't save it either. Memory would fit ($700 / 16 = 43.75 \text{ GB}$ per GPU), but even the 2-node threshold of about 1,238 is above 977.
- **ZeRO-3:** sharding everything over all 4,096 GPUs fixes the memory (around 0.17 GB each) but not the comms: same 2,475 tokens per GPU, and we still have 977. So **no**.
- The solution puts ZeRO-3's limit at twice pure DP (4,950) because the weights move too, while the notes and Chapter 5 count the two as the same cost. We're under either way. **TODO(eddie):** pick one to state; the solution and the section's text disagree here.
- **8-way pipelining:** each replica is now 8-way TP times 8 stages, 64 GPUs over 8 nodes, with $4{,}096 / 64 = 64$-way DP on top. Each node only holds $\frac{1}{8}$ of the layers, so it only moves $\frac{1}{8}$ of the weights in the DP reduction, and the effective bandwidth goes from 400 GB/s to $8 \times 400 = 3{,}200 \text{ GB/s}$:

  $$
  \frac{C}{W} = \frac{990 \times 10^{12}}{3{,}200 \times 10^9} \approx \mathbf{309 \text{ tokens per GPU}}
  $$

- We have 977, about $3.2\times$ more than we need, so we're compute-bound. Memory is fine too: $700 / 64 \approx 10.9 \text{ GB}$ per GPU.
- So **yes**: 8-way TP, 8-way PP and 64-way DP works on paper, with ZeRO-1 rather than ZeRO-3 and a decent pipeline implementation.

**Question 3 [Megatron-LM hyperparams]:** A figure from the Megatron-LM repository lists the hyperparameters behind their high MFU numbers. Every config uses a sequence length of 4096. What per-GPU batch size, in tokens, do the 16B, 70B and 314B models run at? Take data parallelism as the outermost axis and bfloat16 reductions: on paper, is each config compute-bound or communication-bound, and could any of them be configured better?

- We use the configs the solution reads off the figure: 16B has 192 sequences on 192 GPUs, 70B has 384 on 768 GPUs with 16-way model parallelism (TP times PP), and 314B has 1,536 on 3,072 GPUs with 64-way.
- **Per-GPU batch:** sequences times 4096 tokens, divided by GPUs.

  $$
  \text{16B: } \frac{192 \times 4096}{192} = \mathbf{4{,}096} \qquad \text{70B: } \frac{384 \times 4096}{768} = \mathbf{2{,}048} \qquad \text{314B: } \frac{1{,}536 \times 4096}{3{,}072} = \mathbf{2{,}048}
  $$

- **Thresholds:** with DP outermost the gradient reduction crosses nodes, so the base requirement is $C/W = 2{,}475$ tokens per GPU, divided by the number of nodes a replica spans. (The solution writes 2,472 here, which looks like a typo.)
- **16B:** 4,096 against 2,475, so compute-bound with about $1.65\times$ to spare, even inside one node.
- **70B:** 16-way is 2 nodes, so $2{,}475 / 2 \approx 1{,}238$ against 2,048. About $1.65\times$ again.
- **314B:** 64-way is 8 nodes, so $2{,}475 / 8 \approx 309$ against 2,048. Compute-bound by a factor of about 6.6.
- All three are compute-bound in theory, the two bigger ones only because their model parallelism spans several nodes.
- **Is there something better?** $1.65\times$ of headroom isn't much for the 16B and 70B, given that measured collective bandwidth falls well short of spec. The 314B has the most slack: the same batch could spread over roughly $6.6\times$ more GPUs, or the per-GPU batch could drop towards 300 tokens.
- **TODO(eddie):** check the TP and PP columns of the Megatron-LM figure. The 16-way and 64-way numbers come from the solution, and the 16B model's split isn't stated there at all.

## Reference Numbers

GPU specs from our references. The V100 has no bf16 Tensor Core figure listed.

|**Model**|**SMs/chip**|**SMEM/SM**|**L2/chip**|**HBM capacity/chip**|**HBM BW/chip (bytes/s)**|**FLOPs/s (bf16/fp16)**|**FLOPs/s (fp8/int8)**|**FLOPs/s (fp4)**|
|---|---|---|---|---|---|---|---|---|
|**V100**|80|96kB|6MB|32GB|9.0e11|n/a|n/a|n/a|
|**A100**|108|192kB|40MB|80GB|2.0e12|3.1e14|6.2e14|n/a|
|**H100**|132|256kB|50MB|80GB|3.4e12|9.9e14|2.0e15|n/a|
|**H200**|132|256kB|50MB|141GB|4.8e12|9.9e14|2.0e15|n/a|
|**B200**|148|256kB|126MB|192GB|8.0e12|2.3e15|4.5e15|9.0e15|

> Every generation has 256kB of register memory per SM, and the B200 adds 256kB of TMEM per SM. Clocks are 1.59GHz (1.98GHz boost) for the H100 and H200; the B200's isn't published.

### Interconnect Figures

NVLink inside a node, by generation:

|**NVLink**|**GPU generation**|**BW/link (GB/s, each way)**|**Links/GPU**|**GPU egress (GB/s)**|**Node size**|**NVSwitches/node**|
|---|---|---|---|---|---|---|
|**3.0**|Ampere (A100)|25|12|300|8|6|
|**4.0**|Hopper (H100, H200)|25|18|450|8|4|
|**5.0**|Blackwell (B200)|50|18|900|8 or 72|2 or 18|

The reference 1,024-GPU H100 SuperPod, level by level:

|**Level**|**GPUs**|**Children**|**Switches per unit**|**Switch BW (TB/s)**|**Cable BW (TB/s)**|**Collective BW (GB/s)**|
|---|---|---|---|---|---|---|
|**Node**|8|8 GPUs|4 NVSwitch|6.4|3.6|450 per GPU|
|**Leaf (SU)**|256|32 nodes|8 InfiniBand|25.6|12.8|400 per node|
|**Spine**|1,024|4 SUs|16 InfiniBand|51.2|51.2|400 per node|

Egress by node type, which is what the rooflines use:

|**Node type**|**GPUs/node**|**GPU egress (bytes/s)**|**Node egress (bytes/s)**|
|---|---|---|---|
|**H100**|8|450e9|400e9|
|**B200**|8|900e9|400e9|
|**GB200 NVL72**|72|900e9|3600e9|

All of these are full-duplex. An NVSwitch has 64 ports at NVLink speed; InfiniBand ports are 400 Gbps (50 GB/s), 64 per switch (3.2 TB/s), with one NIC per GPU. Measured NVLink collectives top out around 370 GB/s instead of 450.
