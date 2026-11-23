---
title: How to Think About GPUs
published: true
---
## 12. Thinking About GPUs

Chapter 2 ended with a short "TPUs vs. GPUs" section: GPUs talk through switches, TPUs through a torus. This chapter is the long version of that for NVIDIA GPUs (mostly the H100 and B200): the chip, then the network, then the collectives from Chapter 3 and the training rooflines from Chapter 5 redone on a GPU cluster.

At the chip level a GPU and a TPU look surprisingly similar: a lot of matmul hardware sitting next to a stick of HBM. The difference is how the compute is cut up. A TPU has 1 or 2 big TensorCores. A GPU has over a hundred small, mostly independent cores called **Streaming Multiprocessors (SMs)**: 132 on an H100 and 148 on a B200.

Every SM is cut into 4 equal quarters (subpartitions, in NVIDIA's terms), and each one has:

- **Tensor Core:** The matmul unit, so the GPU's version of the MXU. Confusingly it is NOT the equivalent of a TPU TensorCore (that would be the whole SM). Nearly all of the chip's FLOPs/s come from here: 990 TFLOPs/s in bf16 on an H100, against about 67 TFLOPs/s from the vector units.

- **CUDA Cores:** 32 fp32 ALUs that do the vector math (ReLUs, pointwise ops, sums), like the lanes of the TPU's VPU. All 32 run the same instruction every cycle. The unit that feeds them work is the Warp Scheduler.

- **Register file:** 16,384 32-bit registers for the CUDA cores.

The names map onto Chapter 2 like this:

|**GPU**|**TPU**|**What it is**|
|---|---|---|
|SM|TensorCore|The core "cell" that holds everything else|
|Tensor Core|MXU|Matmul unit|
|Warp Scheduler|VPU|SIMD vector unit|
|CUDA Core|VPU ALU|One lane of the vector unit|
|SMEM (L1 cache)|VMEM|Fast on-chip cache|
|HBM|HBM|Big, slower main memory|

The CUDA cores have more freedom than the lanes of a VPU. Each one carries its own instruction pointer (SIMT, Single Instruction Multiple Threads), so threads in the same group of 32 (a warp) can branch differently. The catch: the 32 cores still share one instruction per cycle, so the hardware runs the two branches one after the other while the cores on the other branch sit idle. Code that diverges a lot gets slower without any warning. An SM can also keep up to 64 warps loaded and run one while another waits on memory, while a TPU is basically single-threaded.

### GPU Memory

There are more levels of memory than on a TPU. From closest to the math to furthest:

- **Registers:** 256kB per SM.

- **SMEM:** 256kB of L1 cache per SM, holding the inputs for Tensor Core matmuls. The B200's Tensor Cores got big enough that it adds another 256kB of TMEM per SM just to feed them.

- **L2 cache:** About 50MB on an H100, shared by every SM. It's about half the size of a TPU's VMEM and far slower (the book cites a measurement of around 5.5 TB/s, against roughly 40 TB/s for VMEM), and we don't get to control what's in it.

- **HBM:** 80GB at 3.35 TB/s on an H100, 192GB at 8 TB/s on a B200 (that's the book's spec table; its text says 9 TB/s).

So what does the roofline from Chapter 1 look like? The H100's critical intensity for a bf16 matmul is 990e12/3.35e12, around 295 FLOPs/byte, and the B200's is around 281. A TPU v5e is 240. Meaning, the same rule of thumb carries over: a matmul needs a batch of a few hundred tokens before it's compute-bound. Lower precision buys throughput here too (fp8 is $2\times$ the fp16 FLOPs/s, and the B200 adds fp4 at roughly $4\times$).

### GPUs vs. TPUs at the Chip Level

- **Modularity:** An H100 has $132 \times 4 = 528$ independent vector units, each 32 lanes wide. A TPU v5p has 8 units that are each 1,024 wide. Unrelated kernels can run side by side on different SMs with no planning. A TPU has one thread of control, so its compiler has to schedule the memory loads and the MXU and VPU work around each other by hand.

- **Predictability:** The flip side. On a GPU a lot is settled while the program runs (what sits in the L2 cache, which warp goes next), so a kernel can be slow for reasons that never show up in the code. A TPU program is easier to push up to its roofline.

- **Size:** A single GPU is a bigger chip. An H200 has around $2.2\times$ the FLOPs/s of a TPU v5p (9.9e14 against 4.59e14) and about $1.5\times$ the HBM. The TPU answer to that is more chips and a stronger network between them.

- **Fast cache:** TPUs have far more of it: 128MB of VMEM on a v5p against 33MB of SMEM on an H100.

## On GPU Networking

This is where the two designs really split. GPUs sit at the bottom of a tree of switches. Any GPU can reach any other, but how much bandwidth it gets depends on how far up the tree the message has to climb.

> **Note:** Two unit traps show up in every problem below. Network links are sold in bits (a 400 Gbps cable is 50 GB/s), and they're full-duplex, meaning the quoted number is available in each direction at the same time. Bisection bandwidth is the one figure that usually has both directions added together.

### Inside a Node

A **node** is a set of GPUs (8 for the H100 and B200) wired all-to-all through NVSwitches over NVLink. This is the telephone switchboard from Chapter 2. On an H100 node:

- Each NVLink 4.0 link carries 25 GB/s each way.

- Each GPU has 18 of them, spread over the node's 4 NVSwitches, so $18 \times 25 = 450\text{ GB/s}$ in and out of every GPU.

- Each NVSwitch has 64 ports.

The B200 doubles the link speed to 50 GB/s, so 900 GB/s per GPU.

### Beyond the Node

NVLink stops at the edge of the node. Past that, every GPU has its own 400 Gbps InfiniBand NIC, so a node of 8 has $8 \times 50 = 400\text{ GB/s}$ of egress in total. NVIDIA's reference H100 SuperPod stacks nodes up in two more levels:

- **Scalable Unit (SU):** 32 nodes (256 GPUs) under 8 leaf switches.

- **SuperPod:** 4 SUs (1,024 GPUs) under 16 spine switches. Every leaf connects to every spine.

All of these are 64-port InfiniBand switches at 50 GB/s per port. The layout is a **fat tree**: the cabling gets thicker towards the top (12.8 TB/s leaves each SU, 51.2 TB/s crosses the spine), so no level is a choke point and any node can send its full 400 GB/s to any other node. The diagrams on [jax-ml.github.io/scaling-book/gpus/](https://jax-ml.github.io/scaling-book/gpus/) are worth a look for this part.

So there are two numbers to remember for an H100 cluster: **450 GB/s per GPU inside a node, 400 GB/s per node outside it.** That second one is only 50 GB/s per GPU, a $9\times$ drop the moment a message leaves the node. A TPU v5p, for comparison, has 6 links at 90 GB/s each (540 GB/s per chip) at any pod size, but only towards its neighbors.

The GB200 NVL72 changes this by putting 72 GPUs in one NVLink domain at 900 GB/s each, with 3.6 TB/s of node egress. Since the chips are only about twice as fast, the cross-node rooflines below get roughly $4\times$ easier.

## Collectives on a Tree

Same four primitives as Chapter 3, different cost model. On a torus every collective was "bytes over link bandwidth" wherever the chips sat; here the answer depends on whether we stay inside a node.

### Intra-node Collectives

AllGather and ReduceScatter run around a ring exactly like on a TPU. Each of the $N - 1$ hops moves $\frac{1}{N}$ of the array at the full GPU egress:

$$
T_{\text{AG or RS}} = \frac{\text{bytes} \cdot (N - 1)}{N \cdot W_{\text{GPU}}} \approx \frac{\text{bytes}}{W_{\text{GPU}}}
$$

So about bytes/450e9 on an H100, and an AllReduce is still $2\times$ that.

AllToAll is where the switch pays off. There's no ring to crawl around; every GPU sends each chunk straight to where it's going:

$$
T_{\text{AllToAll}} = \frac{\text{bytes} \cdot (N - 1)}{N^2 \cdot W_{\text{GPU}}} \approx \frac{\text{bytes}}{N \cdot W_{\text{GPU}}}
$$

On a torus this was $\text{bytes} / 4W$, so with $N = 8$ it's half the cost. For MoE routing, where only $k$ of the $N$ destinations receive anything, it shrinks by another $k/N$.

The switches can also do math. With **SHARP** (in-network reductions), the switch itself sums what the GPUs send and broadcasts the result, so an AllReduce costs $\text{bytes} / W_{\text{GPU}}$ instead of twice that. In theory. The book's own measurements show a gain of around 30%.

Which brings up the catch: the spec numbers are optimistic. In the book's measurements, NVLink AllReduce bandwidth tops out near 370 GB/s, and only for huge messages (around 10GB per device). A realistic 58MB shard sees closer to 150 GB/s. So everything here is an upper bound.

### Cross-node Collectives

Above the node a reduction climbs the tree one level at a time: the GPUs of a node first, then the nodes under a leaf, then the SUs under the spine. Since the fat tree gives every node 400 GB/s to anywhere, we can picture one big ring through all the nodes, and the cost is bytes over bandwidth again with the node's egress as the bandwidth:

$$
T_{\text{AG or RS}} = \max\left[\frac{\text{bytes}}{W_{\text{GPU}}}, \frac{\text{bytes}}{W_{\text{node}}}\right] \approx \frac{\text{bytes}}{400 \times 10^9}
$$

AllToAll does NOT survive the trip. It isn't hierarchical; every GPU really does have to reach every other GPU, so all 8 GPUs of a node squeeze through the same 400 GB/s. Over $M$ nodes:

$$
T_{\text{AllToAll}} \approx \frac{\text{bytes}}{M \cdot W_{\text{node}}}
$$

Going from one node to two takes us from $\text{bytes} / (8 \times 450\text{e}9)$ to $\text{bytes} / (2 \times 400\text{e}9)$, which is $4.5\times$ slower.

### A Second Sharded Axis

On a TPU, if the array we're reducing over $X$ is also sharded over another axis $Y$, every chip moves $\frac{1}{Y}$ as much data and the collective gets $Y$ times cheaper. On GPUs that's only half true. Say $Y$ is the inner axis (it fills up a node first):

$$
T_{\text{node}} = \frac{\text{bytes}}{W_{\text{GPU}} \cdot \min(Y, 8)} \qquad T_{\text{scale-out}} = \frac{\text{bytes}}{W_{\text{node}}} \cdot \frac{8}{\max(Y, 8)}
$$

If $Y \le 8$, the node-level part gets cheaper, but the node still has to push the whole array through its one 400 GB/s uplink, so the total doesn't move. Only when $Y$ spans several nodes does the cross-node cost drop, in proportion to the number of nodes spanned. I think this is the one fact worth memorizing from the whole section, since it drives most of the sharding recipe below.

## LLM Rooflines on GPUs

Same setup as Chapter 5: one MLP block ($W_{\text{in}}[D, F]$ and $W_{\text{out}}[F, D]$), $B$ the batch in tokens, $C$ the FLOPs/s per GPU, and we compare $T_{\text{math}}$ with $T_{\text{comms}}$. The only new thing is that $W$ is 450e9 when the collective stays inside a node and 400e9 when it doesn't. For an H100 that makes $C/W$ either 2,200 or 2,475.

### 1. Data Parallelism

Pure DP and FSDP (ZeRO) cost the same in the backward pass: a weight-sized AllReduce, or a ReduceScatter plus an AllGather. Per layer, with $X$ replicas:

$$
T_{\text{math}} = \frac{8BDF}{X \cdot C} \qquad T_{\text{comms}} = \frac{8DF}{W} \implies \frac{B}{X} > \frac{C}{W}
$$

So we need around **2,500 tokens per GPU** (2,475 across nodes), against 850 on a TPU v5p using all three torus axes. Three adjustments:

- **MoE:** With $E$ experts and $k$ active per token, there are $E$ times the weights but only $k$ times the FLOPs, so the threshold is multiplied by $E/k$. For $E = 128$ and $k = 4$ that's 79,200 tokens per GPU.

- **Only 2-way DP:** The ring factor $(X - 1)/X$ is $\frac{1}{2}$, so the threshold halves to about 1,238.

- **Model parallelism across nodes:** From the last section, if each replica is spread over several nodes, each node only moves its own slice of the weights. Spanning 8 nodes cuts the threshold by 8.

### 2. Tensor Parallelism

TP moves activations instead of weights: an AllGather and a ReduceScatter of $[B, D]$ around each block.

$$
T_{\text{math}} = \frac{4BDF}{Y \cdot C} \qquad T_{\text{comms}} = \frac{4BD}{W} \implies Y < \frac{F \cdot W}{C}
$$

That's $F/2200$ inside a node and $F/2475$ beyond it. For $F = 28{,}672$ it comes to 13 and 11.6, so TP fits comfortably in one 8-GPU node. With the 2-node discount again, the book's rule for exactly 2 nodes is $F > 2475 \cdot (Y - 8)$, which allows 16-way on paper (up to 19 for this $F$), but not much further. Quiz 5 Question 2 looks at how tight that really is.

### 3. Expert Parallelism

For MoEs we can shard the weights by expert, $W_{\text{in}}[E_Z, D, F]$, and AllToAll each token's activations to wherever its experts live. Across nodes:

$$
T_{\text{math}} = \frac{4BkDF}{Z \cdot C} \qquad T_{\text{comms}} = \frac{4BD(Z - 8)}{W \cdot Z} \cdot \min\left(\frac{8k}{Z}, 1\right)
$$

Two regimes fall out. When $Z$ is small (under $8k$, so think a node or two), we need $F > \frac{C}{W} \cdot \frac{Z - 8}{k}$, which is easy to meet. When $Z$ is large, we need $F > 8 \cdot \frac{C}{W}$, about 19,800 on H100s; if the model clears that bar, $Z$ can go all the way up to $E$.

### 4. Pipeline Parallelism

Splitting the layers across nodes only sends a small microbatch of activations from one stage to the next:

$$
T_{\text{per-layer comms}} \approx 1.5 \cdot \frac{2BD}{W \cdot N_{\text{layers}}}
$$

Basically free. The price is paid elsewhere: awkward code, pipeline bubbles where a stage sits idle, and a bad fit with FSDP (ZeRO-3 would have to AllGather the weights for every microbatch, so pipelined setups use ZeRO-1).

### Takeaway

Inside a node, GPUs are great: 450 GB/s to any neighbor in one hop, and cheap AllToAlls. Leaving the node costs a $9\times$ drop in per-GPU bandwidth, and that shapes everything else.

DP or FSDP needs about 2,500 tokens per GPU (times $E/k$ for an MoE). TP is good for about 8-way, so one node, maybe two. Anything that spreads a model replica over several nodes (pipelining, expert parallelism, 2-node TP) makes the data-parallel reduction cheaper. So the usual recipe is a mix: TP or EP inside a node or two, pipelining across many nodes, and plain DP or ZeRO-1 on the outside.

And faster chips on the same network make all of it harder, since $C/W$ only goes up.

## Worked Problems

All of the hardware numbers used here are collected under Reference Numbers at the end. One heads-up: the book doesn't stick to one set of chip numbers. Its text and solutions mostly use 990e12 FLOPs/s and 3.35e12 bytes/s for the H100, where its spec table rounds to 9.9e14 and 3.4e12. For the B200, some solutions use 2250e12 FLOPs/s and others the table's 2.3e15, and the HBM bandwidth is 8.0e12 in the table and the solutions but 9 TB/s in the text. Each problem below follows the inputs of the book's own solution and says which ones they are.

### Quiz 1: GPU hardware

**Question 1 [CUDA cores]:** How many fp32 CUDA cores (ALUs) does an H100 have? A B200? How does that compare with the number of independent ALUs in a TPU v5p?

- This is just counting down the chip layout: SMs, then subpartitions per SM, then fp32 cores per subpartition.
- An H100 has 132 SMs, 4 subpartitions each, and 32 fp32 cores per subpartition:

  $$
  132 \times 4 \times 32 = \mathbf{16{,}896 \text{ CUDA cores}}
  $$

- A B200 has 148 SMs with the same layout:

  $$
  148 \times 4 \times 32 = \mathbf{18{,}944 \text{ CUDA cores}}
  $$

- A TPU v5p has 2 TensorCores. Each one has a VPU made of 4 independently programmable units, and each unit is $8 \times 128$ lanes:

  $$
  2 \times 4 \times 8 \times 128 = \mathbf{8{,}192 \text{ ALUs}}
  $$

- So an H100 has about twice the vector lanes of a v5p ($2.06\times$, and $2.31\times$ for the B200), running at roughly the same clock speed. The bigger difference is how they're grouped: the H100's are 528 independent little units of 32, and the v5p's are 8 big units of 1,024.

**Question 2 [vector FLOPs calculation]:** An H100 has 132 SMs and a 1.59GHz clock (1.98GHz with boost). If every ALU does one vector op per cycle, how many vector fp32 FLOPs/s does the chip get, with and without boost? How does that compare with its matmul FLOPs/s?

- One op per core per cycle, so we just multiply the core count from Question 1 by the clock:

  $$
  16{,}896 \times 1.59 \times 10^9 = 2.69 \times 10^{13} = \mathbf{26.9 \text{ TFLOPs/s}}
  $$

  $$
  16{,}896 \times 1.98 \times 10^9 = 3.35 \times 10^{13} = \mathbf{33.5 \text{ TFLOPs/s (boost)}}
  $$

- The spec sheet says about double this ($2 \times 33.5 = 67 \text{ TFLOPs/s}$; the book's text has 66), because a fused multiply-add counts as 2 FLOPs in one cycle. That's fair for a dot product but not for most vector work.
- The Tensor Cores do 990 TFLOPs/s in bf16. That's around $30\times$ the boosted vector number ($37\times$ at the base clock).
- Meaning, anything that isn't a matmul runs on a roughly $30\times$ slower machine.

**Question 3 [GPU matmul intensity]:** What is the peak fp16 matmul intensity on an H100? On a B200? What about fp8? Intensity here means the ratio of matmul FLOPs/s to memory bandwidth.

- This is the hardware critical intensity from Chapter 1, with the GPU's numbers plugged in. The solution uses 990e12 FLOPs/s over 3.35e12 bytes/s for the H100 and 2250e12 over 8e12 for the B200, so we do too.

  $$
  \text{H100: } \frac{990 \times 10^{12} \text{ FLOPs/s}}{3.35 \times 10^{12} \text{ bytes/s}} = \mathbf{295 \text{ FLOPs/byte}} \qquad \text{B200: } \frac{2250 \times 10^{12}}{8 \times 10^{12}} = \mathbf{281 \text{ FLOPs/byte}}
  $$

- Both chips do exactly $2\times$ the FLOPs/s in fp8, so the fp8 intensities are **591** and **562** (the solution doubles the rounded 295 and says 590).
- With the rounded table values (9.9e14/3.4e12 and 2.3e15/8.0e12) we'd get 291 and 288. Same answer for napkin math. The one input that would really move it is the 9 TB/s the book's text quotes for the B200's HBM, which gives 250 instead of 281.
- These sit right next to the 240 of a TPU v5e. So the rule from Chapter 1 holds on GPUs too: a bf16 matmul needs a batch of around 280 to 300 tokens to be compute-bound.
- The fp8 number looks twice as demanding, but if the weights are also stored in fp8 then every parameter is 1 byte instead of 2, and the matmul's own intensity doubles as well. The batch size we need stays about the same.

**Question 4 [matmul runtime]:** Using the answer to Question 3, how long should an `fp16[64, 4096] * fp16[4096, 8192]` matmul take on a single B200? How about `fp16[512, 4096] * fp16[4096, 8192]`?

- From Question 3 the B200 flips from memory-bound to compute-bound at a batch of around 281. So $B = 64$ should be limited by HBM bandwidth and $B = 512$ by FLOPs. We'll work out both times for each anyway, with $D = 4096$ and $F = 8192$.
- Inputs, as in the solution: 8e12 bytes/s of HBM bandwidth, and the table's 2.3e15 FLOPs/s for $C$. (The 281 came from 2250e12; the solution switches between the two.)
- **$B = 64$:** we read $x$ and $W$ and write the output, at 2 bytes per element:

  $$
  2BD + 2DF + 2BF = 524{,}288 + 67{,}108{,}864 + 1{,}048{,}576 = 68{,}681{,}728 \text{ bytes}
  $$

  $$
  T_{\text{comms}} = \frac{68{,}681{,}728 \text{ bytes}}{8 \times 10^{12} \text{ bytes/s}} = \mathbf{8.6 \ \mu\text{s}}
  $$

  $$
  T_{\text{math}} = \frac{2BDF}{C} = \frac{4{,}294{,}967{,}296 \text{ FLOPs}}{2.3 \times 10^{15} \text{ FLOPs/s}} = 1.9 \ \mu\text{s}
  $$

  Memory wins by a wide margin, so about **8.6 μs**.

- **$B = 512$:** same steps.

  $$
  T_{\text{comms}} = \frac{79{,}691{,}776 \text{ bytes}}{8 \times 10^{12} \text{ bytes/s}} = 10.0 \ \mu\text{s}
  $$

  $$
  T_{\text{math}} = \frac{34{,}359{,}738{,}368 \text{ FLOPs}}{2.3 \times 10^{15} \text{ FLOPs/s}} = \mathbf{14.9 \ \mu\text{s}}
  $$

  Now the math is the slow part, so about **15 μs**. (With the 2250e12 from Question 3 it would be 15.3 μs. Same answer.)

- Notice the weights are 67MB of the roughly 69MB moved in the first case. At small batch we're basically paying to stream $W$ out of HBM and nothing else.
- Solving for the exact crossover with these shapes gives $B \approx 321$ (314 with 2250e12), a bit above the 281 to 288 that the plain intensity ratio gives, because with $D = 4096$ the activation and output bytes aren't quite negligible.
- These are rooflines. The solution expects something more like 10 to 12 μs and 20 μs in practice, since we never get all of the bandwidth or all of the FLOPs.

**Question 5 [L1 cache capacity]:** What is the total L1/SMEM capacity of an H100? What about register memory? How does this compare with TPU VMEM capacity?

- Both are per-SM numbers, so we multiply by 132 SMs.
- SMEM is 256kB per SM:

  $$
  132 \times 256 \text{ kB} = 33{,}792 \text{ kB} = \mathbf{33 \text{ MB}}
  $$

- Registers: each of the 4 subpartitions has 16,384 registers of 4 bytes, so $4 \times 16{,}384 \times 4 = 262{,}144 \text{ bytes}$, which is also 256kB per SM and another **33 MB** per chip.
- These are power-of-two units: 256kB is 262,144 bytes, and we divide by 1,024 to get MB. So strictly it's 33 MiB of each, or 34.6 million bytes. The solution also says 33MB; the book's GPU vs. TPU table rounds both down to 32MB.
- Together that's about **66 MB** of fast memory, roughly half of a TPU v5p's 128MB of VMEM (52% of it; the solution quotes 120MB for VMEM, which makes it 55%).
- The split is very different though. A TPU has only 256kB of register memory in total, and the H100 has 33MB of it. The solution's explanation is that VMEM is so low-latency that spilling registers into it is cheap, so the TPU doesn't need many.

**Question 6 [calculating B200 clock frequency]:** NVIDIA reports that a B200 can do 80 TFLOPs/s of vector fp32 compute. Given that each CUDA core can do 2 FLOPs per cycle with an FMA (fused multiply-add), estimate the peak clock frequency.

- This is Question 2 run backwards: we know the FLOPs/s and the core count, and want the cycles per second.
- From Question 1 a B200 has 18,944 CUDA cores, so with FMAs it does $18{,}944 \times 2 = 37{,}888$ FLOPs per cycle.

  $$
  \frac{80 \times 10^{12} \text{ FLOPs/s}}{37{,}888 \text{ FLOPs/cycle}} = \mathbf{2.11 \text{ GHz}}
  $$

- That's a bit above the H100's 1.98GHz boost clock. The solution calls it high but reasonable, since B200s are generally liquid cooled.

**Question 7 [estimating H100 add runtime]:** Using the figures above, how long should it take to add two `fp32[N]` vectors on a single H100? Work out both $T_{\text{math}}$ and $T_{\text{comms}}$. What is the arithmetic intensity of the operation? If you can get access, run it in PyTorch or JAX for $N = 1024$ and $N = 1024 \times 1024 \times 1024$ and compare.

- An add is a vector op, so it runs on the CUDA cores, not the Tensor Cores. That changes which FLOPs/s number we use.
- FLOPs: one add per element, so $N$.
- Bytes: we load two vectors and store one, at 4 bytes per element, so $3 \times 4N = 12N$.

  $$
  \text{Intensity} = \frac{N \text{ FLOPs}}{12N \text{ bytes}} = \mathbf{\frac{1}{12} \text{ FLOPs/byte}}
  $$

- The hardware side: 33.5 TFLOPs/s of vector compute (Question 2, with boost and no FMA) over 3.35e12 bytes/s of HBM bandwidth is about 10 FLOPs/byte. We're at $\frac{1}{12}$. So the add is memory-bound by a factor of around 120, and there's nothing a batch size can do about it.

  $$
  T_{\text{math}} = \frac{N}{3.35 \times 10^{13}} \text{ s} \qquad T_{\text{comms}} = \frac{12N}{3.35 \times 10^{12}} = \frac{N}{2.79 \times 10^{11}} \text{ s}
  $$

  $$
  T = \max(T_{\text{math}}, T_{\text{comms}}) = \mathbf{\frac{N}{2.79 \times 10^{11}} \text{ s}}
  $$

- **$N = 1024$:** $T_{\text{comms}}$ is 3.7 nanoseconds. (The math is even less: 1,024 elements only keep 1,024 cores busy, which is 8 of the 132 SMs, for a single cycle.) Nothing on a GPU actually finishes in a few nanoseconds, so here we'd be latency-bound: the time is whatever it costs to launch the kernel.
- **$N = 1024^3$:** we move $12 \times 2^{30} = 12{,}884{,}901{,}888$ bytes (12 GiB in total, which fits in 80GB), so

  $$
  T = \frac{12{,}884{,}901{,}888 \text{ bytes}}{3.35 \times 10^{12} \text{ bytes/s}} = \mathbf{3.85 \text{ ms}}
  $$

- The solution reports its own JAX runs: about 1.5 μs for $N = 65{,}536$ (where this roofline says 0.23 μs, so latency-bound as expected) and 4.1 ms for $N = 1024^3$, close to the 3.85 ms roofline (which it rounds to 3.84). Those timings are the book's, not ours.
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

- There are two places the bandwidth could run out: the links coming out of the GPUs, and the ports on the switches. We work out both and take the smaller one.
- **Switches:** each NVSwitch has 64 ports at 25 GB/s.

  $$
  4 \text{ switches} \times 64 \text{ ports} \times 25 \text{ GB/s} = 6.4 \text{ TB/s}
  $$

- **Links:** each GPU has 18 NVLinks ($5 + 4 + 4 + 5$ to the four switches), so 450 GB/s per GPU.

  $$
  8 \text{ GPUs} \times 18 \text{ links} \times 25 \text{ GB/s} = \mathbf{3.6 \text{ TB/s}}
  $$

- The links are the limit, so the node has **3.6 TB/s** in total. The switches have plenty of room: only 144 of their 256 ports are plugged in.
- This is a full-duplex number. The GPUs can send 3.6 TB/s into the switches and receive 3.6 TB/s back at the same time.

**Question 2 [bisection bandwidth]:** Cut a network into two equal halves and add up the bandwidth that crosses the cut. The bisection bandwidth is the smallest value we can get over every way of making that cut. What is it for an 8xH100 node? Hint: the figure usually counts flow in both directions.

- Any even split puts 4 GPUs on each side. Since every GPU reaches every other one through the switches, all splits look the same.
- Each GPU can push its full 450 GB/s at the other half, so one way we have $4 \times 450 \text{ GB/s} = 1.8 \text{ TB/s}$. Adding the other direction:

  $$
  2 \times 4 \times 450 \text{ GB/s} = \mathbf{3.6 \text{ TB/s}}
  $$

- The both-directions 3.6 TB/s is the figure the solution gives.
- As a check, computing the max-flow over the actual $5 + 4 + 4 + 5$ wiring (a graph calculation, not a measurement) gives 1,800 GB/s one way, whichever 4 GPUs go on each side.
- Same number as Question 1, and that's not a coincidence. It means the switches never get in the way: whatever the 4 GPUs on one side can send, the fabric can carry across. Dividing by the 8 GPUs gives back the 450 GB/s per GPU that we use as the collective bandwidth.

**Question 3 [AllGather cost]:** How long does an AllGather of an array of $B$ bytes take on an 8xH100 node, assuming it's throughput-bound? Then put numbers in for $\text{bf16}[D_X, F]$ with $D = 4096$ and $F = 65{,}536$.

- This is the ring AllGather from Chapter 3, with the NVLink egress standing in for the ICI link.
- With $N = 8$ GPUs, each one starts with $\frac{B}{N}$ bytes. On every turn each GPU passes one shard along at 450 GB/s, and after $N - 1$ turns everybody has everything:

  $$
  T_{\text{comms}} = (N - 1) \cdot \frac{B}{N \cdot W_{\text{GPU}}} = \frac{7B}{8 \times 450 \times 10^9} \approx \frac{B}{450 \times 10^9}
  $$

- For our array:

  $$
  B = 4096 \times 65{,}536 \times 2 \text{ bytes} = 536{,}870{,}912 \text{ bytes}
  $$

  $$
  T_{\text{comms}} = \frac{536{,}870{,}912 \times 7}{8 \times 450 \times 10^9} = \mathbf{1.04 \text{ ms}}
  $$

- With the lazy $B/W$ version it's 1.19 ms. The $\frac{7}{8}$ matters a little at $N = 8$ and stops mattering in bigger rings.
- The solution adds that this takes about 1.5 ms in practice, which fits the note above about NVLink not delivering its full 450 GB/s.

### Quiz 3: Beyond the node level

**Question 1 [fat tree topology]:** Take the reference 1,024-GPU H100 SuperPod: 128 nodes of 8 GPUs, one 400 Gbps NIC per GPU, 4 SUs of 32 nodes with 8 leaf switches each, 16 spine switches, 64-port switches everywhere, and 2 links from every leaf to every spine. Calculate the bisection bandwidth of the whole pod at the node level, and show that every link is sized so the pod keeps full bisection bandwidth. Hint: work out the link bandwidth and the switch bandwidth at each level.

- First the units: 400 Gbps is $400 / 8 = 50 \text{ GB/s}$ per cable and per switch port.
- "Full bisection bandwidth" means every node can use its whole 400 GB/s no matter how we cut the pod in half. So we go up the tree level by level and check that the cables and the switches can both carry 400 GB/s per node.
- **Node to leaf (cables):** each node has 8 NICs, one cable to each of the 8 leaf switches in its SU.

  $$
  8 \times 50 \text{ GB/s} = 400 \text{ GB/s per node} \qquad 32 \text{ nodes} \times 400 \text{ GB/s} = 12.8 \text{ TB/s per SU}
  $$

- **Leaf switches:** a leaf gets one cable from each of the 32 nodes in its SU, so 32 of its 64 ports face down. That's $32 \times 50 = 1.6 \text{ TB/s}$ per leaf, and with 8 leaves, 12.8 TB/s per SU. Exactly 400 GB/s per node.
- **Leaf to spine (cables):** the other 32 ports face up, 2 cables to each of the 16 spines.

  $$
  8 \text{ leaves} \times 16 \text{ spines} \times 2 \times 50 \text{ GB/s} = 12.8 \text{ TB/s per SU}
  $$

  Still 400 GB/s per node.

- **Spine switches:** each spine has 2 cables from each of the 32 leaves, so all 64 ports are used and all face down: 3.2 TB/s per spine.

  $$
  16 \times 3.2 \text{ TB/s} = 51.2 \text{ TB/s} = 128 \text{ nodes} \times 400 \text{ GB/s}
  $$

- So every level matches. There are 1,024 cables from nodes to leaves and 1,024 from leaves to spines, 51.2 TB/s at each level, and no switch has a spare port or an oversubscribed one.
- Now the bisection. Split the 128 nodes into two halves of 64. Each node can send its full 400 GB/s across the cut:

  $$
  64 \times 400 \text{ GB/s} = 25.6 \text{ TB/s one way} \implies \mathbf{51.2 \text{ TB/s}} \text{ counting both directions}
  $$

- The solution stops at 400 GB/s per node across any cut and doesn't quote a pod total. 51.2 TB/s is the spine-level figure in the book's tables, which matches the both-directions number.
- Computing the max-flow over the whole topology (again a graph calculation, not a measurement) gives the same 25,600 GB/s one way, whether the cut follows the SUs or splits them. With only 1 cable per leaf-spine pair, an SU-aligned cut would drop to 12,800 GB/s. So the doubled uplinks are exactly what makes the tree "fat".
- Per GPU that's 50 GB/s in each direction, which is where the $9\times$ drop from NVLink in the notes comes from.

**Question 2 [scaling to a larger DGX pod]:** Say we want to train on 2,048 GPUs instead of 1,024. What's the simplest way to modify the DGX topology above? What about 4,096? Hint: there's no single right answer, but try to keep costs down, and keep link capacity in mind.

- The real constraint here is ports. Every InfiniBand switch has 64 of them. A leaf uses 32 for its nodes and has 32 left to go up. A spine has all 64 facing down.
- **2,048 GPUs:** keep the SU as it is (32 nodes under 8 leaves) and double everything above it. 8 SUs means 64 leaves and $64 \times 32 = 2{,}048$ uplinks, which needs $2{,}048 / 64 = 32$ spines.
- Each leaf only has 32 uplinks, so it can't run 2 cables to each of 32 spines. But it doesn't need to: 1 cable to each of 32 spines is the same 1.6 TB/s as 2 cables to each of 16. Total spine bandwidth is $32 \times 3.2 = 102.4 \text{ TB/s}$, which is 256 nodes at 400 GB/s. Still a full fat tree, with 96 InfiniBand switches instead of 48.
- **4,096 GPUs:** now the ports run out. 16 SUs is 128 leaves, but a spine can only reach 64 leaves (one port each) and a leaf can only reach 32 spines. Two levels of switches top out at exactly 2,048 GPUs.
- So we need a third level. NVIDIA calls it the core, and per the solution the reference design uses 128 spine and 64 core switches. Checking the ports: 128 leaves have 4,096 uplinks, 128 spines split 32 down and 32 up take those and pass 4,096 links on, and $64 \times 64 = 4{,}096$ core ports receive them. 204.8 TB/s at every level, which is 512 nodes at 400 GB/s.
- The cost shows up in the switch count: 320 InfiniBand switches for 4,096 GPUs, or one per 12.8 GPUs, against one per 21.3 GPUs in the two smaller builds. Plus an extra hop of latency. The cheaper alternative would be to oversubscribe the top level and give up full bisection bandwidth.

### Pop Quizzes: collectives

**Pop Quiz 1 [AllGather time]:** On an 8xH100 node, with 450 GB/s of full-duplex bandwidth, how long does $\text{AllGather}(\text{bf16}[B_X, F])$ take for $B = 1024$ and $F = 16{,}384$?

- Same ring as Quiz 2 Question 3, just a smaller array:

  $$
  2 \times 1024 \times 16{,}384 = 33{,}554{,}432 \text{ bytes}
  $$

  $$
  T_{\text{comms}} \approx \frac{33{,}554{,}432 \text{ bytes}}{450 \times 10^9 \text{ bytes/s}} = 75 \ \mu\text{s} \qquad \text{or exactly} \qquad \frac{33{,}554{,}432 \times 7}{8 \times 450 \times 10^9} = \mathbf{65 \ \mu\text{s}}
  $$

- So around 65 to 75 μs, depending on whether we keep the $\frac{7}{8}$.

**Pop Quiz 2 [AllToAll time]:** Same 8xH100 node, 450 GB/s of unidirectional bandwidth. How long does $\text{AllToAll}_{X \to N}(\text{bf16}[B_X, N])$ take? And if only 4 of the 8 entries are known to be non-zero?

- Careful with the letters. Here $B$ is the batch dimension, not a byte count, and $N = 8$ is both the second dimension of the array and the number of GPUs. So the array is $V = 2BN = 16B$ bytes.
- Each GPU starts with $\frac{V}{N}$ bytes (its rows). After the AllToAll it should hold one column instead, so it keeps $\frac{1}{N}$ of what it has and sends $\frac{V}{N^2}$ straight to each of the other $N - 1$ GPUs. Its total egress sets the time:

  $$
  T_{\text{comms}} = \frac{V \cdot (N - 1)}{N^2 \cdot W_{\text{GPU}}} = \frac{7V}{64 \times 450 \times 10^9} \approx \frac{V}{8 \times 450 \times 10^9} = \mathbf{\frac{V}{3.6 \times 10^{12}} \text{ s}}
  $$

- In terms of the batch that's about $4.4 \times 10^{-12} \cdot B$ seconds ($3.9 \times 10^{-12} \cdot B$ with the exact $\frac{7}{64}$).
- **If only 4 of the 8 entries are non-zero:** each GPU only has to ship the non-zero half of every chunk, so the cost is multiplied by $k/N = \frac{1}{2}$:

  $$
  T_{\text{comms}} \approx \frac{V}{2 \cdot N \cdot W_{\text{GPU}}} = \mathbf{\frac{V}{7.2 \times 10^{12}} \text{ s}}
  $$

- What this means: inside a node an AllToAll behaves like it has 3.6 TB/s to work with, $8\times$ the AllGather bandwidth, because nobody has to forward anyone else's data. This is why expert routing in an MoE is cheap as long as it stays in one node.

**Pop Quiz 3 [sharding along 2 axes]:** Say we want to do $\text{AllGather}_X(\text{bf16}[D_X, F_Y])$ over a single SU (256 chips), where $Y$ is the inner axis. How long will this take as a function of $D$, $F$ and $Y$?

- The array is $2DF$ bytes. $Y$ being the inner axis means the $Y$ GPUs that share one shard of $D$ sit next to each other: they fill up a node first and then spill into the next nodes. There are $X = 256/Y$ of these groups.
- Every GPU needs the $D$ shards of the other $X - 1$ groups, but only for its own slice of $F$. That's $\frac{2DF}{Y} \cdot \frac{X - 1}{X}$ bytes per GPU. The question is what link those bytes arrive through.
- **Case 1, $Y \le 8$:** every node holds all $Y$ slices of $F$. Between its 8 GPUs, a node therefore needs every byte of the array it doesn't already have, which is $\frac{31}{32}$ of it, and all of that comes down the node's one 400 GB/s uplink:

  $$
  T_{\text{comms}} = \frac{2DF \cdot 31}{32 \times 400 \times 10^9} \approx \mathbf{\frac{2DF}{400 \times 10^9} \text{ s}}
  $$

  No $Y$ in it at all. Sharding $F$ inside the node shrinks each GPU's share, but the node as a whole downloads the same bytes. (The NVLink part of the gather is always faster than this, so it never sets the time.)

- **Case 2, $Y > 8$:** now each group spans $Y/8$ nodes, and a node holds 8 different slices of $F$, one per GPU. Each GPU pulls its own bytes through its own 50 GB/s share of the uplink:

  $$
  T_{\text{comms}} = \frac{2DF}{Y \times 50 \times 10^9} \cdot \frac{X - 1}{X} \approx \mathbf{\frac{2DF}{Y \times 50 \times 10^9} \text{ s}}
  $$

- At $Y = 8$ the two cases agree, since $8 \times 50 = 400$.
- The solution has the same two cases. For $Y \le 8$ it keeps the $\frac{31}{32}$, and for $Y > 8$ it gives only the approximate form, written as $2DF \cdot 256 / (Y \times 12.8\text{e}12)$. The $\frac{X - 1}{X}$ in the second case is our own addition: it's the usual ring factor with $X$ groups in the ring, and it only matters once $X$ gets small.
- The book plots this for $D = 8192$, $F = 32{,}768$ (so $2DF = 536{,}870{,}912$ bytes). Putting those values into the two formulas above:

|**$Y$**|**$X$**|**Nodes per group**|**$T$ with the ring factor**|**$T$ without it**|
|---|---|---|---|---|
|1 to 8|256 to 32|1|1.30 ms|1.34 ms|
|16|16|2|0.63 ms|0.67 ms|
|32|8|4|0.29 ms|0.34 ms|
|64|4|8|0.13 ms|0.17 ms|
|128|2|16|0.04 ms|0.08 ms|

- So 8-way model parallelism inside a node is free but does nothing for the outer AllGather. Past that, every doubling of $Y$ at least halves the cost. (At $Y = 256$ there's only one group and nothing left to gather.)

### Quiz 4: Collectives

**Question 1 [SU AllGather]:** Take one SU with $M$ nodes of $N$ GPUs each and run an AllGather across all of it. Exactly how many bytes go into and out of a node-level switch? And the top-level switch?

- Call the array $B$ bytes. Each of the $MN$ GPUs starts with $\frac{B}{MN}$ and has to end up with all of it. We treat each node as one switch with $N$ GPUs below it and one uplink above it, and follow the bytes through.
- **Node switch, in from its GPUs:** $N \cdot \frac{B}{MN} = \frac{B}{M}$.
- **Node switch, out to the top:** the same $\frac{B}{M}$, since the rest of the SU needs this node's data.
- **Node switch, in from the top:** everything the node is missing, $\frac{B(M - 1)}{M}$.
- **Node switch, out to its GPUs:** every GPU needs all but its own shard, so $N \cdot \left(B - \frac{B}{MN}\right) = NB - \frac{B}{M}$.

  $$
  \text{Ingress} = \frac{B}{M} + \frac{B(M - 1)}{M} = \mathbf{B} \qquad \text{Egress} = \frac{B}{M} + NB - \frac{B}{M} = \mathbf{NB}
  $$

- **Top-level switch:** $\frac{B}{M}$ arrives from each of the $M$ nodes, so $\mathbf{B}$ in. Each node gets back the $\frac{B(M - 1)}{M}$ it's missing, so $\mathbf{B(M - 1)}$ out.
- Counting shard by shard for $M = 32$ and $N = 8$ gives the same thing: $1B$ in and $8B$ out at a node switch, $1B$ in and $31B$ out at the top.
- Egress is the bigger number at both levels, so that's what sets the time. At the node, each GPU receives $B \left(1 - \frac{1}{MN}\right)$ at 450 GB/s, so about $B / 450\text{e}9$. At the top, each node receives $\frac{B(M - 1)}{M}$ at 400 GB/s:

  $$
  T_{\text{node}} \approx \frac{B}{450 \times 10^9} \qquad T_{\text{top}} = \frac{B \cdot (M - 1)}{M \times 400 \times 10^9} = \frac{B}{413 \times 10^9} \text{ for } M = 32
  $$

- The top level is the slower of the two. So an AllGather over a whole SU runs at about 413 GB/s, not far off what we get inside a single node.

**Question 2 [single-node SHARP AR]:** Now a single node of $N$ GPUs doing an AllReduce with SHARP (in-network reductions). Exactly how many bytes does the switch take in and send out?

- Every GPU starts with its own unreduced copy of the array, $B$ bytes each, and all $N$ copies need summing. Think of the array as $N$ chunks of $\frac{B}{N}$, with GPU $j$ in charge of finishing chunk $j$.
- **Step 1:** each GPU sends the switch every chunk except its own, $\frac{B(N - 1)}{N}$ bytes. Switch ingress: $N \cdot \frac{B(N - 1)}{N} = B(N - 1)$.
- **Step 2:** the switch adds up what it received for each chunk and sends that partial sum of chunk $j$ to GPU $j$. Switch egress: $N \cdot \frac{B}{N} = B$.
- **Step 3:** GPU $j$ adds its own copy of chunk $j$, which finishes that chunk, and sends it back. Switch ingress: $B$.
- **Step 4:** the switch multicasts each finished chunk to the other $N - 1$ GPUs, so every GPU receives $\frac{B(N - 1)}{N}$. Switch egress: $B(N - 1)$.

  $$
  \text{Ingress} = B(N - 1) + B = \mathbf{NB} \qquad \text{Egress} = B + B(N - 1) = \mathbf{NB}
  $$

- Per GPU that's exactly $B$ bytes out and $B$ bytes in. So the time is exactly $B / W_{\text{GPU}}$, with no $\frac{N - 1}{N}$ factor this time.
- Without SHARP each GPU would egress $\frac{2B(N - 1)}{N}$ (a ReduceScatter plus an AllGather), which is $1.75B$ for $N = 8$. So the theoretical speedup is $1.75\times$, and that's the "75%" that turns into roughly 30% in practice.

**Question 3 [cross-node SHARP AR]:** An array $\text{bf16}[D_X, F_Y]$ is sharded over the $N$ GPUs of one node. With in-network reductions, how long does $\text{AllReduce}(\text{bf16}[D, F_Y] \{U_X\})$ take? What changes once there's more than one node?

- Decoding the notation: the node's $N = XY$ GPUs each hold an unreduced $\text{bf16}[D, F/Y]$, which is $\frac{2DF}{Y}$ bytes, and the $X$ GPUs that share a slice of $F$ need to sum theirs.
- That's just Question 2 happening $Y$ times side by side, each on an array of $\frac{2DF}{Y}$ bytes. Each GPU egresses $\frac{2DF}{Y} \cdot \frac{X - 1}{X}$, gets $\frac{2DF}{XY}$ back, returns it, and finally receives $\frac{2DF}{Y} \cdot \frac{X - 1}{X}$. So $\frac{2DF}{Y}$ in each direction per GPU:

  $$
  T_{\text{comms}} = \mathbf{\frac{2DF}{Y \cdot W_{\text{GPU}}}} = \frac{2DF}{Y \times 450 \times 10^9} \text{ s}
  $$

- Inside a node, then, sharding over $Y$ helps in full. With $(X, Y) = (4, 2)$ the reduction sees an effective 900 GB/s, and with $(2, 4)$ it sees 1.8 TB/s.
- **More than one node:** now the reduction over $X$ has to cross nodes. A node still holds $Y$ different slices of $F$, and those can't be summed with each other (they're different parts of the array). So the node has to send all of them up: $2DF$ bytes through its 400 GB/s uplink, not $\frac{2DF}{Y}$.

  $$
  T_{\text{comms}} = \frac{2DF}{W_{\text{node}}} = \frac{2DF}{400 \times 10^9} \text{ s}
  $$

- The $Y$ has disappeared. Going from $(2, 4)$ inside a node to the same thing across nodes takes us from 1.8 TB/s to 400 GB/s, which is $4.5\times$ slower. Same lesson as Pop Quiz 3.

**Question 4 [spine level AR cost]:** Same setup as Question 3, except $Y = 256$, which pushes the AllReduce up to the spine level. How long does it take, again with in-network reductions?

- With $Y = 256$ each slice of $F$ lives on exactly one GPU in an SU, and the $X = 4$ SUs of the SuperPod are what we reduce over. So this is Pop Quiz 3's second case taken as far as it goes.
- Each GPU holds $\frac{2DF}{256}$ bytes and, with SHARP, sends and receives exactly that much through its own 50 GB/s NIC. All 256 GPUs in the SU can do that at once, because the fat tree gives each SU 12.8 TB/s up to the spine ($8 \times 16 \times 2$ cables at 50 GB/s, or 51.2 TB/s over 4 SUs):

  $$
  T_{\text{comms}} = \frac{2DF}{256 \times 50 \times 10^9} = \mathbf{\frac{2DF}{12.8 \times 10^{12}} \text{ s}}
  $$

- That's $32\times$ the bandwidth of the $Y \le 8$ case. For the $D = 8192$, $F = 32{,}768$ array from Pop Quiz 3 it comes to around 42 μs instead of 1.34 ms.
- Without SHARP it would be $2 \cdot \frac{X - 1}{X} = 1.5\times$ longer.

**Question 5 [2-way AllGather cost]:** What does an AllGather of $B$ bytes cost over exactly 2 nodes? Keep the exact ring factors instead of the usual approximation, and work out the intra-node cost as well as the cross-node one.

- 2 nodes is 16 GPUs with $\frac{B}{16}$ each. There are two links to check: the InfiniBand hop between the nodes and the NVLink ring inside each one.
- **Cross-node:** each node is missing the other node's half, $\frac{B}{2}$, and pulls it in at 400 GB/s. This is the usual ring formula with $M = 2$:

  $$
  T_{\text{cross}} = \frac{B \cdot (M - 1)}{M \cdot W_{\text{node}}} = \frac{B}{2 \times 400 \times 10^9} = \frac{B}{800 \times 10^9}
  $$

- **Intra-node:** each GPU's NIC brings in $\frac{1}{8}$ of that remote half, so every GPU now "owns" $\frac{B}{8}$ (its own $\frac{B}{16}$ plus $\frac{B}{16}$ of remote data). The other $\frac{7B}{8}$ has to reach it over NVLink:

  $$
  T_{\text{node}} = \frac{B \cdot (N - 1)}{N \cdot W_{\text{GPU}}} = \frac{7B}{8 \times 450 \times 10^9} = \mathbf{\frac{B}{514 \times 10^9} \text{ s}}
  $$

- The $\frac{7}{8}$ is the solution's factor, and it's what the section's per-level ring formula gives. It depends on each GPU pulling its share of the remote half through its own NIC. If we treated the node as a strict tree instead, with the whole remote half fanned out over NVLink as well, each GPU would receive $\frac{15B}{16}$ that way and the node term would be $B / 480\text{e}9$. NVLink is the limit under both readings.
- The node level is the slower one, by about $1.56\times$. So a 2-node AllGather is bound by NVLink and not by InfiniBand, and it runs at 514 GB/s, the same as a single node.
- Compare that with a full SU, where the cross-node term is $B / 413\text{e}9$ and takes over. With only 2 nodes, the $\frac{M - 1}{M}$ factor hands us double the cross-node bandwidth. This is the "2-node discount" from the notes, and it's the reason 2-way data parallelism shows up so often.

### Quiz 5: LLM rooflines

**Question 1 [B200 rooflines]:** A B200 DGX SuperPod (NOT the GB200 NVL72) has $2\times$ the bandwidth within a node (900 GB/s of GPU egress) but the same bandwidth in the scale-out network (400 GB/s per node). The FLOPs/s are in the reference table. How does this change the model-parallel and data-parallel rooflines?

- Every roofline in the notes depends on the hardware through one ratio, $C/W$. So we just recompute it for both levels and see what moves.
- $C$ goes from 990e12 to 2250e12 FLOPs/s (the solution's inputs), about $2.27\times$ (the solution says $2.25\times$). Inside a node $W$ doubles; across nodes it doesn't change.

  $$
  \text{In a node: } \frac{2250 \times 10^{12}}{900 \times 10^9} = \mathbf{2{,}500} \qquad \text{Across nodes: } \frac{2250 \times 10^{12}}{400 \times 10^9} = \mathbf{5{,}625}
  $$

- For the H100 these were 2,200 and 2,475.
- **Model parallelism (TP):** the limit inside a node goes from $Y < F/2200$ to $Y < F/2500$. For $F = 28{,}672$ that's 13.0 down to 11.5, both still above the 8 GPUs a node actually has. Nothing really changes. Across nodes it goes from $F/2475$ to $F/5625$, which is only about 5 for the same $F$, so cross-node TP gets worse.
- **Data parallelism:** the reduction crosses nodes, so the per-GPU batch we need jumps from 2,475 to **5,625 tokens**.
- (With the table's rounded 2.3e15 the two ratios are 2,556 and 5,750.)
- So a faster chip on the same network is harder to keep compute-bound, not easier. That's the problem the GB200 NVL72 goes after with its 3.6 TB/s of node egress, which brings the cross-node ratio down to 625.

**Question 2 [how to shard LLaMA-3 70B]:** We're training LLaMA-3 70B with bfloat16 weights and Adam, with the optimizer state kept in fp32.

**1. What's the smallest number of H100s that can hold the weights and the optimizer state?**

- Bytes per parameter: 2 for the bf16 weight, plus Adam's two moments at 4 bytes each.

  $$
  70 \times 10^9 \times (2 + 8) \text{ bytes} = 140 \text{ GB} + 560 \text{ GB} = 700 \text{ GB}
  $$

  $$
  \frac{700 \text{ GB}}{80 \text{ GB per H100}} = 8.75 \implies \mathbf{9 \text{ GPUs}}
  $$

- GPUs come in nodes of 8, so in practice that's **2 nodes**. This is only a floor: it leaves no room for gradients or activations.

**2. How long does training take on 4,096 H100s for 15T tokens at 45% MFU (Model FLOPs Utilization)?**

- The $6 \times \text{parameters} \times \text{tokens}$ rule from Chapter 4 gives the total work:

  $$
  6 \times 70 \times 10^9 \times 15 \times 10^{12} = 6.3 \times 10^{24} \text{ FLOPs}
  $$

- The cluster delivers 45% of its peak:

  $$
  4{,}096 \times 990 \times 10^{12} \times 0.45 = 1.82 \times 10^{18} \text{ FLOPs/s}
  $$

  $$
  \frac{6.3 \times 10^{24}}{1.82 \times 10^{18}} = 3.45 \times 10^6 \text{ s} \approx \mathbf{40 \text{ days}}
  $$

**3. The model has $F = 28{,}672$ and its batch was about 4M tokens. How much model parallelism can we use before it's comms-bound? On 4k chips, does that much model parallelism plus pure DP stay compute-bound? Does it with ZeRO-3 instead? With 8-way pipelining added? Check the GPU memory as well as the communication cost.**

- **Model parallelism:** from the TP rule, inside a node we can go up to $F / 2200 = 28{,}672 / 2200 \approx 13$-way. That's more than the 8 GPUs of one node and fewer than the 16 of two, so we stop at **8-way** TP, which is comfortably compute-bound. The solution reasons the same way and also lands on 8-way (its limit is 14.4, see the TODO below).
- What about 16-way over 2 nodes? The book isn't fully consistent here. Its 2-node rule from the notes, $F > 2475 \cdot (Y - 8)$, passes easily at $Y = 16$ since $28{,}672 > 19{,}800$, and the section's text says 16-way TP is generally possible. Its solution to this problem doesn't use that rule and stays at 8.
- Going beyond the book: the 2-node rule only looks at the InfiniBand hop. From Quiz 4 Question 5, a 2-node AllGather is actually limited by the NVLink ring at 514 GB/s. Putting that into the TP roofline gives $F/Y > 2200 \times \frac{7}{8} = 1{,}925$, so $Y < 14.9$, and $F/16 = 1{,}792$ falls about 7% short. This check is ours and not the book's. It points the same way as the solution though: 16-way is borderline at best, and 8-way is the safe answer.
- **TODO(eddie):** the solution divides by 1995 here ($28{,}672 / 1995 = 14.4$) where the notes give $990\text{e}12 / 450\text{e}9 = 2200$ (so 13.0). It isn't clear where 1995 comes from; we couldn't rebuild it from the section's FLOPs/s, bandwidths and ring factors. The conclusion is the same either way, since the node size caps it at 8. Also decide whether to keep the 16-way aside: the section's text allows 16-way, its solution doesn't, and the NVLink argument is our own.
- **8-way TP plus pure DP:** that's $4{,}096 / 8 = 512$ replicas. Memory first: every replica keeps its own 700 GB of weights and optimizer state on 8 GPUs, which is 87.5 GB per GPU. An H100 has 80. It doesn't fit, so **no**.
- It fails on communication too. The per-GPU batch is $4 \times 10^6 / 4{,}096 \approx 977$ tokens, and data parallelism across nodes needs 2,475. We're about $2.5\times$ short (we'd need a 10M token batch).
- The 8 versus 16 question doesn't change this verdict. With 16-way TP the memory would fit ($700 / 16 = 43.75 \text{ GB}$ per GPU), and spanning 2 nodes would halve the threshold to about 1,238, but 977 is still under it.
- **ZeRO-3:** this fixes the memory, since everything is now sharded over all 4,096 GPUs (around 0.17 GB each). It doesn't fix the communication: the roofline is the same 2,475 tokens per GPU and we still have 977. So **no**.
- The solution says ZeRO-3's limit is twice the pure DP one (so 4,950) because the weights have to move too. The data parallelism rule in the notes above and Chapter 5 both treat the two as the same cost, since a ReduceScatter plus an AllGather moves the same bytes as an AllReduce. Either way we're under it. **TODO(eddie):** pick one to state; the book's solution and its own text disagree here.
- **8-way pipelining:** now each replica is 8-way TP times 8 pipeline stages, so 64 GPUs spread over 8 nodes, with $4{,}096 / 64 = 64$-way DP on top. Each node holds only $\frac{1}{8}$ of the layers, so it only has to move $\frac{1}{8}$ of the weights in the data-parallel reduction. Effectively the bandwidth goes from 400 GB/s to $8 \times 400 = 3{,}200 \text{ GB/s}$:

  $$
  \frac{C}{W} = \frac{990 \times 10^{12}}{3{,}200 \times 10^9} \approx \mathbf{309 \text{ tokens per GPU}}
  $$

- We have 977, about $3.2\times$ more than we need, so we're compute-bound. Memory is fine too: $700 / 64 \approx 10.9 \text{ GB}$ per GPU even with every replica holding a full copy.
- So **yes**: 8-way TP, 8-way PP and 64-way DP works on paper. What's left is implementing the pipelining well, and using ZeRO-1 rather than ZeRO-3 with it.

**Question 3 [Megatron-LM hyperparams]:** The book shows a figure from the Megatron-LM repository with the hyperparameters behind their high MFU numbers. Every config uses a sequence length of 4096. What per-GPU batch size, in tokens, do the 16B, 70B and 314B models run at? Take data parallelism as the outermost axis and bfloat16 reductions: on paper, is each config compute-bound or communication-bound, and could any of them be configured better?

- The figure isn't reproduced here, so we use the configs the book's solution reads off it: the 16B model has a batch of 192 sequences on 192 GPUs, the 70B has 384 on 768 GPUs with 16-way model parallelism (TP times PP), and the 314B has 1,536 on 3,072 GPUs with 64-way model parallelism.
- **Per-GPU batch:** sequences times 4096 tokens, divided by GPUs.

  $$
  \text{16B: } \frac{192 \times 4096}{192} = \mathbf{4{,}096} \qquad \text{70B: } \frac{384 \times 4096}{768} = \mathbf{2{,}048} \qquad \text{314B: } \frac{1{,}536 \times 4096}{3{,}072} = \mathbf{2{,}048}
  $$

- **Thresholds:** with DP outermost, the gradient reduction crosses nodes, so the base requirement is $C/W = 2{,}475$ tokens per GPU. Then we divide by the number of nodes each model replica spans. (The solution writes 2,472 at this point, which looks like a slip for the 2,475 in the section's text.)
- **16B:** 4,096 against 2,475. Compute-bound, with about $1.65\times$ to spare, and that holds even if its model parallelism stays inside one node.
- **70B:** 16-way model parallelism is 2 nodes, so the threshold is $2{,}475 / 2 \approx 1{,}238$. We have 2,048. Compute-bound, again by about $1.65\times$.
- **314B:** 64-way is 8 nodes, so the threshold is $2{,}475 / 8 \approx 309$. We have 2,048. Compute-bound by a factor of about 6.6.
- So all three are compute-bound in theory (the solution rounds the last two thresholds to roughly 1k and 300), and the two bigger ones only because their model parallelism spans several nodes. With everything inside one node, 2,048 tokens per GPU would be under the 2,475 line.
- **Is there something better?** The 16B and 70B configs have about $1.65\times$ of headroom, which isn't much once we remember that the collective bandwidth the book measures falls well short of the spec. The 314B has the most slack: on paper the same batch could be spread over roughly $6.6\times$ more GPUs, or the per-GPU batch could drop towards 300 tokens, before communication takes over.
- **TODO(eddie):** check the TP and PP columns of the Megatron-LM figure. The 16-way and 64-way numbers are taken from the book's solution, and the 16B model's split isn't stated there at all.

## Reference Numbers

GPU specs for the models the book tabulates. The V100 has no bf16 Tensor Core figure listed.

|**Model**|**SMs/chip**|**SMEM/SM**|**L2/chip**|**HBM capacity/chip**|**HBM BW/chip (bytes/s)**|**FLOPs/s (bf16/fp16)**|**FLOPs/s (fp8/int8)**|**FLOPs/s (fp4)**|
|---|---|---|---|---|---|---|---|---|
|**V100**|80|96kB|6MB|32GB|9.0e11|n/a|n/a|n/a|
|**A100**|108|192kB|40MB|80GB|2.0e12|3.1e14|6.2e14|n/a|
|**H100**|132|256kB|50MB|80GB|3.4e12|9.9e14|2.0e15|n/a|
|**H200**|132|256kB|50MB|141GB|4.8e12|9.9e14|2.0e15|n/a|
|**B200**|148|256kB|126MB|192GB|8.0e12|2.3e15|4.5e15|9.0e15|

> Every generation has 256kB of register memory per SM, and the B200 adds 256kB of TMEM per SM. Clock speeds are 1.59GHz (1.98GHz boost) for the H100 and H200; the B200's isn't published. The book gives two values for a few of these. For the H100 its text and solutions use 990e12 FLOPs/s and 3.35e12 bytes/s, which this table rounds to 9.9e14 and 3.4e12. For the B200 some solutions use 2250e12 FLOPs/s (Quiz 1 Question 3, Quiz 5 Question 1) and others the table's 2.3e15 (Quiz 1 Question 4), and the text quotes the HBM bandwidth as 9 TB/s against the table's 8.0e12, which is what the solutions use.

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

All of these are full-duplex, so the number applies in each direction at once. An NVSwitch has 64 ports at the NVLink speed. The InfiniBand links and switch ports are 400 Gbps, which is 50 GB/s, with 64 ports per switch (3.2 TB/s). Each GPU has one InfiniBand NIC, so a node's egress is 8 times 50 GB/s. In the book's measurements NVLink collectives top out around 370 GB/s instead of 450.
