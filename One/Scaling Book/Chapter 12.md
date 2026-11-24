---
title: How to Think About GPUs
published: true
---
## 12. Thinking About GPUs

Chapters 2, 3 and 5 again, but for NVIDIA GPUs (mostly the H100 and B200).

## What Is a GPU?

A GPU looks a lot like a TPU, but instead of 1 or 2 big TensorCores it has over a hundred small, mostly independent **Streaming Multiprocessors (SMs)**: 132 on an H100, 148 on a B200. Each SM has 4 subpartitions, each with:

- **Tensor Core:** The matmul unit, so the GPU's MXU (confusingly, not a TPU TensorCore). Almost all the FLOPs/s live here, 990 TFLOPs/s in bf16 on an H100.

- **CUDA Cores:** 32 fp32 ALUs for vector math, like the lanes of the TPU's VPU.

- **Register file:** 16,384 32-bit registers.

An SM also switches between up to 64 warps (groups of 32 threads) to hide memory waits, where a TPU is basically single-threaded.

### Memory

- **SMEM:** 256kB of L1 cache per SM, the GPU's VMEM. The B200 adds 256kB of TMEM per SM.

- **L2 cache:** About 50MB on an H100, shared and not under our control.

- **HBM:** 80GB at 3.35 TB/s on an H100, 192GB at 8 TB/s on a B200.

### Summary of GPU specs

The full table is under Reference Numbers. The bf16 critical intensity is around 295 FLOPs/byte on an H100 and 281 on a B200 (240 on a TPU v5e), so matmuls still need a few hundred tokens of batch to be compute-bound.

### GPUs vs. TPUs at the chip level

An H100 has $132 \times 4 = 528$ independent vector units of 32 lanes, against 8 units of 1,024 on a TPU v5p, so it's more flexible but less predictable. TPUs make up for smaller chips with more of them, a stronger network, and far more fast cache (128MB of VMEM on a v5p against 33MB of SMEM on an H100).

## Networking

GPUs sit at the bottom of a tree of switches: any GPU can reach any other, but the bandwidth depends on how far up the tree a message climbs. Links are quoted in bits (400 Gbps is 50 GB/s) and per direction.

### At the node level

A **node** is 8 GPUs wired all-to-all through NVSwitches, the telephone switchboard from Chapter 2. On an H100 each GPU has 18 NVLink links at 25 GB/s each way, so $18 \times 25 = 450\text{ GB/s}$ in and out of every GPU. The B200 doubles that to 900 GB/s.

## Beyond the node level

Past the node, every GPU has its own 400 Gbps InfiniBand NIC, so a node has $8 \times 50 = 400\text{ GB/s}$ of egress. NVIDIA's reference H100 SuperPod puts 32 nodes under 8 leaf switches (a **Scalable Unit**, or SU) and 4 SUs under 16 spine switches. It's a **fat tree**: the cabling thickens towards the top, so any node can send its full 400 GB/s to any other node. The diagrams on [jax-ml.github.io/scaling-book/gpus/](https://jax-ml.github.io/scaling-book/gpus/) help a lot here.

So two numbers to remember: **450 GB/s per GPU inside a node, 400 GB/s per node outside it.** That second one is only 50 GB/s per GPU, a $9\times$ drop. The GB200 NVL72 eases this with 72 GPUs per NVLink domain and 3.6 TB/s of node egress.

## How Do Collectives Work on GPUs?

### Intra-node collectives

AllGather and ReduceScatter run around a ring like on a TPU, at full GPU egress:

$$
T_{\text{AG or RS}} = \frac{\text{bytes} \cdot (N - 1)}{N \cdot W_{\text{GPU}}} \approx \frac{\text{bytes}}{W_{\text{GPU}}}
$$

An AllReduce is still twice that. AllToAll is where the switch pays off: every GPU sends each chunk straight to its destination for about $\text{bytes} / (N \cdot W_{\text{GPU}})$, half the torus cost at $N = 8$.

With **SHARP** (in-network reductions) the switch does the summing, so an AllReduce costs $\text{bytes} / W_{\text{GPU}}$ instead of twice that. In practice it's more like a 30% gain, and NVLink AllReduce tops out near 370 GB/s.

### Cross-node collectives

Above the node, an AllGather or ReduceScatter is one big ring through the nodes at about $\text{bytes} / 400\text{e}9$. AllToAll does NOT survive the trip: all 8 GPUs of a node squeeze through the same 400 GB/s, so two nodes are already $4.5\times$ slower than one.

On a TPU, sharding the array over a second axis $Y$ makes the collective $Y$ times cheaper. On GPUs that's only half true. With $Y$ as the inner axis:

$$
T_{\text{node}} = \frac{\text{bytes}}{W_{\text{GPU}} \cdot \min(Y, 8)} \qquad T_{\text{scale-out}} = \frac{\text{bytes}}{W_{\text{node}}} \cdot \frac{8}{\max(Y, 8)}
$$

If $Y \le 8$, the node still pushes the whole array through its one uplink, so the total doesn't move. Only when $Y$ spans several nodes does the cross-node cost drop. I think this is the one fact worth memorizing here.

## Rooflines for LLM Scaling on GPUs

Same setup as Chapter 5: one MLP block ($W_{\text{in}}[D, F]$, $W_{\text{out}}[F, D]$), batch $B$ in tokens, $C$ FLOPs/s per GPU. $W$ is 450e9 inside a node and 400e9 outside, so for an H100 $C/W$ is 2,200 or 2,475.

### Data Parallelism

Pure DP and FSDP (ZeRO) both do a weight-sized AllReduce (or ReduceScatter plus AllGather) in the backward pass. With $X$ replicas:

$$
T_{\text{math}} = \frac{8BDF}{X \cdot C} \qquad T_{\text{comms}} = \frac{8DF}{W} \implies \frac{B}{X} > \frac{C}{W}
$$

So we need around **2,500 tokens per GPU** (2,475 across nodes), against 850 on a TPU v5p. Three adjustments:

- **MoE:** multiply by $E/k$ ($E$ experts, $k$ active). For $E = 128$ and $k = 4$ that's 79,200 tokens per GPU.

- **Only 2-way DP:** the ring factor $(X - 1)/X$ halves the threshold to about 1,238.

- **Model parallelism across nodes:** a replica spanning 8 nodes cuts the threshold by 8, since each node only moves its own slice of the weights.

### Tensor Parallelism

TP moves activations instead of weights, which works out to $Y < F \cdot W / C$: $F/2200$ inside a node and $F/2475$ beyond it. For $F = 28{,}672$ that's 13 and 11.6, so TP fits comfortably in one 8-GPU node.

### Expert Parallelism

For MoEs we shard the experts and AllToAll each token to wherever its experts live. Over a node or two that's easy; wider than that we need $F > 8 \cdot C/W$, about 19,800 on H100s.

### Pipeline Parallelism

Splitting the layers across nodes only sends small microbatches of activations between stages, so the comms are basically free. The price is awkward code, pipeline bubbles, and using ZeRO-1 instead of ZeRO-3, which would AllGather the weights every microbatch.

### TLDR of LLM scaling on GPUs

Inside a node GPUs are great; leaving it costs $9\times$ per GPU. Spreading a replica over several nodes makes the DP reduction cheaper, so the usual recipe is TP or EP inside a node or two, pipelining across nodes, and DP or ZeRO-1 outside that. Faster chips on the same network make this harder, since $C/W$ goes up.

## Worked Problems

### Quiz 1: GPU hardware

**Question 1 [CUDA cores]:** How many fp32 CUDA cores (ALUs) does an H100 have? A B200? How does that compare with the number of independent ALUs in a TPU v5p?

- SMs, times 4 subpartitions, times 32 fp32 cores:

  $$
  \text{H100: } 132 \times 4 \times 32 = \mathbf{16{,}896 \text{ CUDA cores}} \qquad \text{B200: } 148 \times 4 \times 32 = \mathbf{18{,}944 \text{ CUDA cores}}
  $$

- A TPU v5p has 2 TensorCores, each with a VPU of 4 independent units of $8 \times 128$ lanes:

  $$
  2 \times 4 \times 8 \times 128 = \mathbf{8{,}192 \text{ ALUs}}
  $$

- So an H100 has about twice the vector lanes, but in 528 small units instead of 8 big ones.

**Question 2 [vector FLOPs calculation]:** An H100 has 132 SMs and a 1.59GHz clock (1.98GHz with boost). If every ALU does one vector op per cycle, how many vector fp32 FLOPs/s does the chip get, with and without boost? How does that compare with its matmul FLOPs/s?

- The cores from Question 1 times the clock:

  $$
  16{,}896 \times 1.59 \times 10^9 = \mathbf{26.9 \text{ TFLOPs/s}} \qquad 16{,}896 \times 1.98 \times 10^9 = \mathbf{33.5 \text{ TFLOPs/s (boost)}}
  $$

- The spec sheet's 67 TFLOPs/s counts a fused multiply-add as 2 FLOPs. The Tensor Cores' 990 TFLOPs/s is around $30\times$ the boosted number, so anything that isn't a matmul runs on a much slower machine.

**Question 3 [GPU matmul intensity]:** What is the peak fp16 matmul intensity on an H100? On a B200? What about fp8? Intensity here means the ratio of matmul FLOPs/s to memory bandwidth.

- FLOPs/s over bytes/s, like Chapter 1:

  $$
  \text{H100: } \frac{990 \times 10^{12} \text{ FLOPs/s}}{3.35 \times 10^{12} \text{ bytes/s}} = \mathbf{295 \text{ FLOPs/byte}} \qquad \text{B200: } \frac{2250 \times 10^{12}}{8 \times 10^{12}} = \mathbf{281 \text{ FLOPs/byte}}
  $$

- Both chips do exactly $2\times$ the FLOPs/s in fp8, so the fp8 intensities are **591** and **562**.
- So a bf16 matmul needs a batch of around 280 to 300 tokens, close to the TPU v5e's 240. fp8 also halves the weight bytes, so that batch stays about the same.

**Question 4 [matmul runtime]:** Using the answer to Question 3, how long should an `fp16[64, 4096] * fp16[4096, 8192]` matmul take on a single B200? How about `fp16[512, 4096] * fp16[4096, 8192]`?

- We read $x$ and $W$ and write the output at 2 bytes per element, with 8e12 bytes/s and the table's 2.3e15 FLOPs/s.
- **$B = 64$:**

  $$
  2BD + 2DF + 2BF = 524{,}288 + 67{,}108{,}864 + 1{,}048{,}576 = 68{,}681{,}728 \text{ bytes}
  $$

  $$
  T_{\text{comms}} = \frac{68{,}681{,}728 \text{ bytes}}{8 \times 10^{12} \text{ bytes/s}} = \mathbf{8.6 \ \mu\text{s}} \qquad T_{\text{math}} = \frac{4{,}294{,}967{,}296 \text{ FLOPs}}{2.3 \times 10^{15} \text{ FLOPs/s}} = 1.9 \ \mu\text{s}
  $$

  Memory wins, so about **8.6 μs**.

- **$B = 512$:**

  $$
  T_{\text{comms}} = \frac{79{,}691{,}776 \text{ bytes}}{8 \times 10^{12} \text{ bytes/s}} = 10.0 \ \mu\text{s} \qquad T_{\text{math}} = \frac{34{,}359{,}738{,}368 \text{ FLOPs}}{2.3 \times 10^{15} \text{ FLOPs/s}} = \mathbf{14.9 \ \mu\text{s}}
  $$

  Now the math is the slow part, so about **15 μs**.

**Question 5 [L1 cache capacity]:** What is the total L1/SMEM capacity of an H100? What about register memory? How does this compare with TPU VMEM capacity?

- SMEM is 256kB per SM, times 132 SMs:

  $$
  132 \times 256 \text{ kB} = 33{,}792 \text{ kB} = \mathbf{33 \text{ MB}}
  $$

- Registers are $4 \times 16{,}384 \times 4 = 262{,}144$ bytes, also 256kB per SM, so another **33 MB**.
- Together that's about **66 MB**, roughly half of a TPU v5p's 128MB of VMEM, but split very differently: a TPU has only 256kB of registers, since spilling into its low-latency VMEM is cheap.

**Question 6 [calculating B200 clock frequency]:** NVIDIA reports that a B200 can do 80 TFLOPs/s of vector fp32 compute. Given that each CUDA core can do 2 FLOPs per cycle with an FMA (fused multiply-add), estimate the peak clock frequency.

- Question 2 run backwards, with $18{,}944 \times 2 = 37{,}888$ FLOPs per cycle:

  $$
  \frac{80 \times 10^{12} \text{ FLOPs/s}}{37{,}888 \text{ FLOPs/cycle}} = \mathbf{2.11 \text{ GHz}}
  $$

- A bit above the H100's 1.98GHz boost, high but reasonable since B200s are generally liquid cooled.

**Question 7 [estimating H100 add runtime]:** Using the figures above, how long should it take to add two `fp32[N]` vectors on a single H100? Work out both $T_{\text{math}}$ and $T_{\text{comms}}$. What is the arithmetic intensity of the operation? If you can get access, run it in PyTorch or JAX for $N = 1024$ and $N = 1024 \times 1024 \times 1024$ and compare.

- An add runs on the CUDA cores. It does $N$ FLOPs and moves $12N$ bytes (two loads and one store at 4 bytes each):

  $$
  \text{Intensity} = \frac{N \text{ FLOPs}}{12N \text{ bytes}} = \mathbf{\frac{1}{12} \text{ FLOPs/byte}}
  $$

- The hardware gives about 10 FLOPs/byte (33.5 TFLOPs/s over 3.35e12 bytes/s), so we're memory-bound whatever $N$ is:

  $$
  T = \max\left(\frac{N}{3.35 \times 10^{13}}, \frac{12N}{3.35 \times 10^{12}}\right) = \mathbf{\frac{N}{2.79 \times 10^{11}} \text{ s}}
  $$

- **$N = 1024$:** $T_{\text{comms}}$ is 3.7 nanoseconds, far less than a kernel launch, so this one is latency-bound.
- **$N = 1024^3$:** we move $12 \times 2^{30} = 12{,}884{,}901{,}888$ bytes:

  $$
  T = \frac{12{,}884{,}901{,}888 \text{ bytes}}{3.35 \times 10^{12} \text{ bytes/s}} = \mathbf{3.85 \text{ ms}}
  $$

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

- We'd expect microseconds of pure overhead for $N = 1024$ and roughly 4 ms for $N = 1024^3$. This question needs an H100, so I'll come back to it.

### Quiz 2: GPU nodes

**Question 1 [total bandwidth for H100 node]:** An 8xH100 node has 4 NVSwitches. What is the total bandwidth of the node? Hint: check the NVLink links and the NVSwitch ports separately.

- Bandwidth can run out at the GPU links or the switch ports, so we take the smaller. The switches have $4 \times 64 \text{ ports} \times 25 \text{ GB/s} = 6.4 \text{ TB/s}$, and the links:

  $$
  8 \text{ GPUs} \times 18 \text{ links} \times 25 \text{ GB/s} = \mathbf{3.6 \text{ TB/s}}
  $$

- The links are the limit, so the node has **3.6 TB/s** in total.

**Question 2 [bisection bandwidth]:** Cut a network into two equal halves and add up the bandwidth that crosses the cut. The bisection bandwidth is the smallest value we can get over every way of making that cut. What is it for an 8xH100 node? Hint: the figure usually counts flow in both directions.

- Through the switches every even split looks the same: 4 GPUs on each side, each pushing 450 GB/s across. Counting both directions:

  $$
  2 \times 4 \times 450 \text{ GB/s} = \mathbf{3.6 \text{ TB/s}}
  $$

**Question 3 [AllGather cost]:** How long does an AllGather of an array of $B$ bytes take on an 8xH100 node, assuming it's throughput-bound? Then put numbers in for $\text{bf16}[D_X, F]$ with $D = 4096$ and $F = 65{,}536$.

- The ring AllGather from Chapter 3 with NVLink in place of ICI: each of the $N = 8$ GPUs passes a $\frac{B}{N}$ shard per turn for $N - 1$ turns.

  $$
  T_{\text{comms}} = (N - 1) \cdot \frac{B}{N \cdot W_{\text{GPU}}} = \frac{7B}{8 \times 450 \times 10^9} \approx \frac{B}{450 \times 10^9}
  $$

- For our array, $B = 4096 \times 65{,}536 \times 2 = 536{,}870{,}912$ bytes:

  $$
  T_{\text{comms}} = \frac{536{,}870{,}912 \times 7}{8 \times 450 \times 10^9} = \mathbf{1.04 \text{ ms}}
  $$

### Quiz 3: Beyond the node level

**Question 1 [fat tree topology]:** Take the reference SuperPod: 1,024 H100s in 128 nodes of 8, one 400 Gbps NIC per GPU, 4 SUs of 32 nodes with 8 leaf switches each, 16 spine switches, 64-port switches everywhere, and 2 links from every leaf to every spine. What's the pod's node-level bisection bandwidth, and is every link sized for full bisection bandwidth? Hint: check the link and switch bandwidth at each level.

- 400 Gbps is 50 GB/s per cable, and full bisection bandwidth means every level carries 400 GB/s per node.
- **Node to leaf:** one cable from each node to each of the SU's 8 leaves is 400 GB/s per node, 12.8 TB/s per SU, using 32 of each leaf's 64 ports.
- **Leaf to spine:** the other 32 go up, 2 cables to each of 16 spines: $8 \times 16 \times 2 \times 50 \text{ GB/s} = 12.8 \text{ TB/s}$ per SU.
- **Spine:** 32 leaves times 2 cables fills all 64 ports, so 3.2 TB/s per spine, and $16 \times 3.2 = 51.2$ TB/s is exactly $128 \times 400$ GB/s. Every level matches.
- For the bisection, two halves of 64 nodes each send 400 GB/s across:

  $$
  64 \times 400 \text{ GB/s} = 25.6 \text{ TB/s one way} \implies \mathbf{51.2 \text{ TB/s}} \text{ counting both directions}
  $$

**Question 2 [scaling to a larger DGX pod]:** Say we want to train on 2,048 GPUs instead of 1,024. What's the simplest way to modify the DGX topology above? What about 4,096? Hint: there's no single right answer, but try to keep costs down, and keep link capacity in mind.

- The constraint is ports: a leaf has 32 facing down and 32 up, and a spine has all 64 facing down.
- **2,048 GPUs:** double everything above the SU. 8 SUs is 64 leaves with $64 \times 32 = 2{,}048$ uplinks, so 32 spines with 1 cable from each leaf. Still a full fat tree at $32 \times 3.2 = 102.4 \text{ TB/s}$, with 96 switches instead of 48.
- **4,096 GPUs:** 16 SUs is 128 leaves, but a spine reaches at most 64 leaves and a leaf 32 spines, so two levels top out at 2,048 GPUs. Per the solution, we add a core level: 128 spine and 64 core switches, with $64 \times 64 = 4{,}096$ core ports to match the 4,096 leaf uplinks.
- That's 320 switches plus an extra hop, so the cheaper option is to oversubscribe the top level and give up full bisection bandwidth.

### Quiz 4: Collectives

**Pop Quiz 1 [AllGather time]:** On an 8xH100 node, with 450 GB/s of full-duplex bandwidth, how long does $\text{AllGather}(\text{bf16}[B_X, F])$ take for $B = 1024$ and $F = 16{,}384$?

- Same ring as Quiz 2 Question 3 with a smaller array, $2 \times 1024 \times 16{,}384 = 33{,}554{,}432$ bytes:

  $$
  T_{\text{comms}} = \frac{33{,}554{,}432 \times 7}{8 \times 450 \times 10^9} = \mathbf{65 \ \mu\text{s}}
  $$

**Pop Quiz 2 [AllToAll time]:** Same 8xH100 node, 450 GB/s of unidirectional bandwidth. How long does $\text{AllToAll}_{X \to N}(\text{bf16}[B_X, N])$ take? And if only 4 of the 8 entries are known to be non-zero?

- Careful with the letters: $B$ is the batch here, and $N = 8$ is both the second dimension and the GPU count, so the array is $V = 2BN = 16B$ bytes.
- Each GPU keeps $\frac{1}{N}$ of its $\frac{V}{N}$ bytes and sends $\frac{V}{N^2}$ straight to each of the other $N - 1$:

  $$
  T_{\text{comms}} = \frac{V \cdot (N - 1)}{N^2 \cdot W_{\text{GPU}}} = \frac{7V}{64 \times 450 \times 10^9} \approx \frac{V}{8 \times 450 \times 10^9} = \mathbf{\frac{V}{3.6 \times 10^{12}} \text{ s}}
  $$

- **If only 4 of the 8 entries are non-zero:** each GPU only ships half of every chunk:

  $$
  T_{\text{comms}} \approx \frac{V}{2 \cdot N \cdot W_{\text{GPU}}} = \mathbf{\frac{V}{7.2 \times 10^{12}} \text{ s}}
  $$

- Meaning, an AllToAll inside a node acts like it has 3.6 TB/s, $8\times$ the AllGather bandwidth, which is why MoE routing is cheap as long as it stays in one node.

**Pop Quiz 3 [sharding along 2 axes]:** Say we want to do $\text{AllGather}_X(\text{bf16}[D_X, F_Y])$ over a single SU (256 chips), where $Y$ is the inner axis. How long will this take as a function of $D$, $F$ and $Y$?

- The array is $2DF$ bytes, and there are $X = 256/Y$ groups of $Y$ GPUs, each group filling a node first.
- **Case 1, $Y \le 8$:** every node holds all $Y$ slices of $F$, so it needs $\frac{31}{32}$ of the array through one 400 GB/s uplink:

  $$
  T_{\text{comms}} = \frac{2DF \cdot 31}{32 \times 400 \times 10^9} \approx \mathbf{\frac{2DF}{400 \times 10^9} \text{ s}}
  $$

- **Case 2, $Y > 8$:** a group spans $Y/8$ nodes, and each GPU pulls its own slice of $F$ through its 50 GB/s share of the uplink:

  $$
  T_{\text{comms}} = \frac{2DF}{Y \times 50 \times 10^9} \cdot \frac{X - 1}{X} \approx \mathbf{\frac{2DF}{Y \times 50 \times 10^9} \text{ s}}
  $$

- For $D = 8192$ and $F = 32{,}768$ ($2DF = 536{,}870{,}912$ bytes), that's about 1.34 ms for any $Y \le 8$, then 0.67, 0.34, 0.17 and 0.08 ms as $Y$ doubles from 16 to 128. Meaning, model parallelism inside a node does nothing for the outer AllGather, and past that every doubling of $Y$ halves the cost.

**Question 1 [SU AllGather]:** Take one SU with $M$ nodes of $N$ GPUs each and run an AllGather across all of it. Exactly how many bytes go into and out of a node-level switch? And the top-level switch?

- For a $B$-byte array, each of the $MN$ GPUs starts with $\frac{B}{MN}$. A node's switch takes in $\frac{B}{M}$ from its GPUs and the missing $\frac{B(M - 1)}{M}$ from the top, sends $\frac{B}{M}$ up, and sends each GPU all but its own shard, $N \cdot \left(B - \frac{B}{MN}\right) = NB - \frac{B}{M}$:

  $$
  \text{Ingress} = \frac{B}{M} + \frac{B(M - 1)}{M} = \mathbf{B} \qquad \text{Egress} = \frac{B}{M} + NB - \frac{B}{M} = \mathbf{NB}
  $$

- **Top-level switch:** $\frac{B}{M}$ from each of the $M$ nodes is $\mathbf{B}$ in, and each node gets back its missing share, so $\mathbf{B(M - 1)}$ out.
- Egress sets the time, at 450 GB/s per GPU and 400 GB/s per node:

  $$
  T_{\text{node}} \approx \frac{B}{450 \times 10^9} \qquad T_{\text{top}} = \frac{B \cdot (M - 1)}{M \times 400 \times 10^9} = \frac{B}{413 \times 10^9} \text{ for } M = 32
  $$

- So an AllGather over a whole SU runs at about 413 GB/s, not far off a single node.

**Question 2 [single-node SHARP AR]:** Now a single node of $N$ GPUs doing an AllReduce with SHARP (in-network reductions). Exactly how many bytes does the switch take in and send out?

- Each GPU starts with its own unreduced $B$-byte copy, split into $N$ chunks, and GPU $j$ finishes chunk $j$.
- **Step 1:** each GPU sends the switch every chunk but its own. Ingress $B(N - 1)$.
- **Step 2:** the switch sums each chunk and sends chunk $j$'s partial sum to GPU $j$. Egress $B$.
- **Step 3:** GPU $j$ adds its own copy and sends the finished chunk back. Ingress $B$.
- **Step 4:** the switch multicasts each finished chunk to the other $N - 1$ GPUs. Egress $B(N - 1)$.

  $$
  \text{Ingress} = B(N - 1) + B = \mathbf{NB} \qquad \text{Egress} = B + B(N - 1) = \mathbf{NB}
  $$

- Per GPU that's $B$ each way, so $B / W_{\text{GPU}}$. Without SHARP each GPU egresses $\frac{2B(N - 1)}{N}$, or $1.75B$ for $N = 8$, so SHARP is $1.75\times$ faster in theory.

**Question 3 [cross-node SHARP AR]:** An array $\text{bf16}[D_X, F_Y]$ is sharded over the $N$ GPUs of one node. With in-network reductions, how long does $\text{AllReduce}(\text{bf16}[D, F_Y] \{U_X\})$ take? What changes once there's more than one node?

- Each of the node's $N = XY$ GPUs holds an unreduced $\frac{2DF}{Y}$ bytes, and the $X$ GPUs sharing a slice of $F$ sum theirs. That's Question 2 running $Y$ times side by side:

  $$
  T_{\text{comms}} = \mathbf{\frac{2DF}{Y \cdot W_{\text{GPU}}}} = \frac{2DF}{Y \times 450 \times 10^9} \text{ s}
  $$

- **More than one node:** the reduction over $X$ now crosses nodes. A node's $Y$ slices of $F$ can't be summed with each other, so all $2DF$ bytes go through its 400 GB/s uplink:

  $$
  T_{\text{comms}} = \frac{2DF}{W_{\text{node}}} = \frac{2DF}{400 \times 10^9} \text{ s}
  $$

- The $Y$ is gone, same lesson as Pop Quiz 3.

**Question 4 [spine level AR cost]:** Same setup as Question 3, except $Y = 256$, which pushes the AllReduce up to the spine level. How long does it take, again with in-network reductions?

- With $Y = 256$ each slice of $F$ lives on one GPU per SU, and we reduce over the $X = 4$ SUs. Each GPU sends and receives its $\frac{2DF}{256}$ bytes through its own 50 GB/s NIC, all 256 at once, since the fat tree gives each SU 12.8 TB/s to the spine:

  $$
  T_{\text{comms}} = \frac{2DF}{256 \times 50 \times 10^9} = \mathbf{\frac{2DF}{12.8 \times 10^{12}} \text{ s}}
  $$

- That's $32\times$ the bandwidth of the $Y \le 8$ case. Without SHARP it would take $2 \cdot \frac{X - 1}{X} = 1.5\times$ longer.

**Question 5 [2-way AllGather cost]:** What does an AllGather of $B$ bytes cost over exactly 2 nodes? Keep the exact ring factors instead of the usual approximation, and work out the intra-node cost as well as the cross-node one.

- 2 nodes is 16 GPUs with $\frac{B}{16}$ each.
- **Cross-node:** each node pulls in the other half at 400 GB/s:

  $$
  T_{\text{cross}} = \frac{B \cdot (M - 1)}{M \cdot W_{\text{node}}} = \frac{B}{2 \times 400 \times 10^9} = \frac{B}{800 \times 10^9}
  $$

- **Intra-node:** each GPU's NIC brings in $\frac{1}{8}$ of the remote half, so every GPU owns $\frac{B}{8}$, and the other $\frac{7B}{8}$ comes over NVLink:

  $$
  T_{\text{node}} = \frac{B \cdot (N - 1)}{N \cdot W_{\text{GPU}}} = \frac{7B}{8 \times 450 \times 10^9} = \mathbf{\frac{B}{514 \times 10^9} \text{ s}}
  $$

- The node level is slower, so a 2-node AllGather is NVLink-bound at 514 GB/s. The $\frac{M - 1}{M}$ factor doubles the cross-node bandwidth here, which is where the 2-way DP discount in the notes comes from.

### Quiz 5: LLM rooflines

**Question 1 [B200 rooflines]:** A B200 DGX SuperPod (NOT the GB200 NVL72) has $2\times$ the bandwidth within a node (900 GB/s of GPU egress) but the same bandwidth in the scale-out network (400 GB/s per node). The FLOPs/s are in the reference table. How does this change the model-parallel and data-parallel rooflines?

- Every roofline sees the hardware through $C/W$, so we recompute it with $C = 2250\text{e}12$:

  $$
  \text{In a node: } \frac{2250 \times 10^{12}}{900 \times 10^9} = \mathbf{2{,}500} \qquad \text{Across nodes: } \frac{2250 \times 10^{12}}{400 \times 10^9} = \mathbf{5{,}625}
  $$

- **Model parallelism (TP):** inside a node the limit goes from $F/2200$ to $F/2500$, 13.0 down to 11.5 for $F = 28{,}672$, still above 8. Across nodes it drops from $F/2475$ to $F/5625$, only about 5.
- **Data parallelism:** the per-GPU batch we need jumps from 2,475 to **5,625 tokens**.
- So a faster chip on the same network is harder to keep busy. The GB200 NVL72's 3.6 TB/s of node egress brings the cross-node ratio down to 625.

**Question 2 [how to shard LLaMA-3 70B]:** We're training LLaMA-3 70B with bfloat16 weights and Adam, with the optimizer state kept in fp32.

**1. What's the smallest number of H100s that can hold the weights and the optimizer state?**

- 2 bytes per parameter for the bf16 weight, plus Adam's two moments at 4 bytes each:

  $$
  70 \times 10^9 \times (2 + 8) \text{ bytes} = 700 \text{ GB} \qquad \frac{700 \text{ GB}}{80 \text{ GB per H100}} = 8.75 \implies \mathbf{9 \text{ GPUs}}
  $$

- GPUs come in nodes of 8, so in practice **2 nodes**, before gradients or activations.

**2. How long does training take on 4,096 H100s for 15T tokens at 45% MFU (Model FLOPs Utilization)?**

- The $6 \times \text{parameters} \times \text{tokens}$ rule from Chapter 4, at 45% of peak:

  $$
  6 \times 70 \times 10^9 \times 15 \times 10^{12} = 6.3 \times 10^{24} \text{ FLOPs} \qquad 4{,}096 \times 990 \times 10^{12} \times 0.45 = 1.82 \times 10^{18} \text{ FLOPs/s}
  $$

  $$
  \frac{6.3 \times 10^{24}}{1.82 \times 10^{18}} = 3.45 \times 10^6 \text{ s} \approx \mathbf{40 \text{ days}}
  $$

**3. The model has $F = 28{,}672$ and a batch of about 4M tokens. How much model parallelism can we use before it's comms-bound? On 4k chips, does that plus pure DP stay compute-bound? With ZeRO-3 instead? With 8-way pipelining added? Check GPU memory as well as communication.**

- **Model parallelism:** $F / 2200 \approx 13$, more than a node's 8 GPUs, so we stop at **8-way** TP.
- **8-way TP plus pure DP:** each replica holds 700 GB on 8 GPUs, 87.5 GB per GPU against the H100's 80, so **no**. The comms fail too: $4 \times 10^6 / 4{,}096 \approx 977$ tokens per GPU against the 2,475 we need.
- **ZeRO-3:** sharding everything over 4,096 GPUs fixes the memory (around 0.17 GB each) but not the comms. On top of the backward-pass reduction, ZeRO-3 also gathers the weights for the forward pass, so the solution doubles DP's bound to about 4,950 tokens per GPU. We have 977, so **no**.
- **8-way pipelining:** a replica is now 8-way TP times 8 stages, 64 GPUs over 8 nodes, with 64-way DP on top. Each node only moves $\frac{1}{8}$ of the weights, so the effective bandwidth is $8 \times 400 = 3{,}200 \text{ GB/s}$:

  $$
  \frac{C}{W} = \frac{990 \times 10^{12}}{3{,}200 \times 10^9} \approx \mathbf{309 \text{ tokens per GPU}}
  $$

- We have 977, about $3.2\times$ more, and memory is fine at $700 / 64 \approx 10.9 \text{ GB}$ per GPU. So **yes**: 8-way TP, 8-way PP and 64-way DP works on paper, with ZeRO-1.

**Question 3 [Megatron-LM hyperparams]:** A figure in the Megatron-LM repository lists the hyperparameters behind their high MFU numbers, all at a sequence length of 4096. What per-GPU batch size, in tokens, do the 16B, 70B and 314B models use? With data parallelism outermost and bfloat16 reductions, is each config compute-bound or communication-bound on paper, and could any be configured better?

- The configs, as the solution reads them off the figure: 16B has 192 sequences on 192 GPUs, 70B has 384 on 768 GPUs with 16-way model parallelism (TP times PP), and 314B has 1,536 on 3,072 GPUs with 64-way.
- **Per-GPU batch:** sequences times 4096 tokens, over GPUs:

  $$
  \text{16B: } \frac{192 \times 4096}{192} = \mathbf{4{,}096} \qquad \text{70B: } \frac{384 \times 4096}{768} = \mathbf{2{,}048} \qquad \text{314B: } \frac{1{,}536 \times 4096}{3{,}072} = \mathbf{2{,}048}
  $$

- With DP outermost we need 2,475 tokens per GPU, divided by the nodes a replica spans:
  - **16B:** 4,096 against 2,475, about $1.65\times$ to spare.
  - **70B:** 16-way is 2 nodes, so 1,238 against 2,048, about $1.65\times$ again.
  - **314B:** 64-way is 8 nodes, so 309 against 2,048, about 6.6 times over.
- All three are compute-bound on paper, but $1.65\times$ isn't much headroom when real collectives fall short of spec. The 314B could spread the same batch over roughly $6.6\times$ more GPUs.

## Reference Numbers

The V100 has no bf16 Tensor Core figure.

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

All of these are full-duplex, with one 400 Gbps (50 GB/s) InfiniBand NIC per GPU.
