---
title: All About Transformer Inference
published: true
---
## 7. Transformer Inference

## The Basics of Transformer Inference

Sampling is simple: feed in a sequence, sample the next token, append it, repeat. Done naively, we re-run the whole sequence every time, so $n$ tokens cost $O(n^2)$ in the MLPs and $O(n^3)$ in attention.

The fix is the **KV cache**. A token only attends to earlier tokens, so we just save each token's keys and values once and reuse them. That gets us to $O(n)$ and $O(n^2)$, and splits inference into two jobs:

- **Prefill:** Run the whole prompt through in one pass, saving its KV cache.
- **Generation:** Feed one sampled token at a time against the cache, until the end token or a length limit.

Same weights, completely different bottlenecks.

> **Note:** the figures are on [jax-ml.github.io/scaling-book/inference/](https://jax-ml.github.io/scaling-book/inference/).

### What do we actually want to optimize?

Training only cares about throughput, but inference also cares about latency: time to first token (TTFT) and time per token after that. Offline evals only care about cost per token, chat needs a low TTFT and tokens faster than people read, and one user on a laptop just wants the lowest latency. Batch size is the knob between them.

### Linear operations: what bottlenecks us?

Every linear layer is the Chapter 1 roofline: $\text{bf16}[B, D]$ times $\text{bf16}[D, F]$. With $B \ll D, F$ the weights dominate the bytes, so the intensity is roughly $\frac{2BDF}{2DF} = B$, and we're compute-bound once:

$$
B_{\text{crit}} = \frac{1.97 \times 10^{14} \text{ FLOPs/s}}{8.2 \times 10^{11} \text{ bytes/s}} \approx \mathbf{240 \text{ tokens}}
$$

That's TPU v5e in bf16, counting tokens, not sequences. int8 weights with bf16 math halve the bytes, so $B_{\text{crit}}$ drops to 120; int8 math as well brings it back to 240.

Prefill gets there for free, since a prompt of a few hundred tokens is already a big batch. Generation makes one token per sequence per step, so it needs 240 users decoding at once. We rarely get there, so every step hauls all the weights out of HBM for a tiny bit of math.

### What about attention?

Here the bytes are the KV cache. With $T$ query tokens attending to $S$ cached ones, we do $4BSTD$ FLOPs and move about $4BSD + 4BTD$ bytes, so the intensity is $\frac{ST}{S + T}$. Prefill has $S = T$, so it's $T/2$ and compute-bound past roughly 480 tokens. Generation has $T = 1$, so it's about 1: permastuck memory-bound, and batching can't help because every sequence brings its own cache.

### Theoretical estimates for LLM latency and throughput

At small batches, a step can't beat pulling everything out of HBM once:

$$
T_{\text{step}} \ge \frac{B \times \text{KV cache bytes} + \text{parameter bytes}}{\text{total HBM bandwidth}}
$$

Throughput is just $B / T_{\text{step}}$. Past $B_{\text{crit}}$ the linear layers go compute-bound, so the parameter term becomes $2 \times B \times \text{param count}$ over FLOPs/s, while the KV cache term stays. Small batch: quick steps, idle FLOPs. Big batch: busy chips, slower tokens, and past $B_{\text{crit}}$ per replica throughput barely improves.

### What about memory?

Memory is just the weights plus the KV cache:

$$
\text{KV cache bytes} = 2 \times \text{bytes per float} \times H \times K \times L \times T
$$

For LLaMA 2-13B ($L = 40$, $K = 40$, $H = 128$) in bf16, one 8,192-token sequence is about 6.7 GB of cache against 26 GB of weights. So just 4 sequences of cache outweigh the whole model.

### Modeling throughput and latency for LLaMA 2-13B

On 8 TPU v5e chips (6.56e12 bytes/s of HBM bandwidth, 128 GB of HBM), batch 1 takes around 5 ms a step, about 200 tokens/s. At most 15 sequences fit, at roughly 19 ms a step and 780 tokens/s. With 8 KV heads instead of 40, 75 sequences fit in the same 19 ms, roughly 3,900 tokens/s. So KV cache size matters about as much as parameter count.

## Tricks for Improving Generation Throughput and Latency

Most of them just shrink the KV cache:

- **GQA (grouped multi-query attention):** Several query heads share each KV head, so the cache shrinks by that ratio and quality barely moves.
- **Local attention:** Some layers only look back a fixed window, so their cache stops growing.
- **Sharing KVs across layers:** Less memory, though steps aren't automatically faster.
- **Quantization:** int8 or int4 weights and KVs mean fewer bytes and bigger batches.
- **Paged attention:** Store the cache in blocks, like an OS page table, so nothing gets padded.
- **Speculative sampling:** A small draft model guesses a few tokens and the big model checks them all in one pass.

Stacked, these can shrink the cache more than $10\times$.

## Distributing Inference Over Multiple Accelerators

### Prefill

Prefill is basically training without the backward pass: model parallelism along $F$ until we'd go ICI-bound, around $F / 2200$ ways per axis on v5e, then sequence parallelism for the leftover chips.

### Generation

Generation is pickier. FSDP is out, since moving weights over ICI is way slower than reading them from HBM, and with one token there's no sequence to shard. That leaves model parallelism, with a twist: we're bottlenecked on weight loading, so more chips means more HBM bandwidth on the same weights, and we keep sharding until ICI time passes HBM time:

$$
\frac{2BD}{W_{\text{ici}}} > \frac{2DF}{Y \cdot W_{\text{hbm}}} \implies Y > \frac{F}{B \cdot \beta}, \qquad \beta = \frac{W_{\text{hbm}}}{W_{\text{ici}}}
$$

On v5e $\beta$ is 8.2e11 / 9.0e10, about 9.1, so with $F = 16{,}384$ and $B = 32$ we can go to about 56-way before latency stops improving.

### Sharding the KV cache

The KV cache gets sharded too, never replicated: KV heads first ($K$ ways at most), then the batch, as $\text{KV}[2, B_Z, S, K_Y, H]$. That costs two AllToAlls per attention layer. For tiny batches or very long contexts we shard the sequence instead.

## Designing an Effective Inference Engine

Scheduling the two jobs:

- **Batched:** Prefill a batch, then generate until everything finishes. Easy, but TTFT is bad and finished sequences leave empty slots.
- **Interleaved:** Generate in a batch and slot in a prefill whenever a slot opens. Good throughput, but every generation stalls while a prefill runs.
- **Disaggregated:** Separate prefill and generation servers, shipping the KV cache between them. The usual answer for latency-sensitive serving.

### Continuous batching

Prefill writes into a KV buffer, generate steps every active slot, and a queue refills a slot as soon as its sequence finishes.

### Prefix caching

When prompts share a prefix (chat histories, system prompts), that part of the KV cache is identical, so we skip prefilling it and route follow-ups back to the replica holding that cache.

### Let's look at an implementation: JetStream

Google's [JetStream](https://github.com/google/JetStream) is an open-source example.

## Worked Problems

Questions 1 to 7 use a made-up model based on LLaMA 2-13B: $L = 64$, $D = 4{,}096$, $F = 16{,}384$, $N = 32$ query heads, $K = 8$ KV heads, $H = 256$ and $V = 32{,}128$. TPU v5e numbers are from Chapter 2: 16 GB of HBM per chip, 8.2e11 bytes/s of HBM bandwidth, 1.97e14 bf16 FLOPs/s, 3.94e14 int8 OPs/s and 9.0e10 bytes/s per ICI link.

**Pop Quiz [bounding step time]:** We take one generate step with a batch of 4 tokens from a 30B parameter dense model on a TPU v5e 4x4 slice. Weights are int8, FLOPs are bf16, the context is 8192 tokens and the KV cache is 100 kB per token. What's a reasonable lower bound on the latency of this step? What if we sample a batch of 256 tokens instead?

- The int8 weights are 30e9 bytes, and one sequence's KV cache is $100{,}000 \times 8192 = 819{,}200{,}000$ bytes (819 MB). 16 chips give us 1.312e13 bytes/s of HBM bandwidth and 3.152e15 FLOPs/s.
- int8 weights with bf16 math put $B_{\text{crit}}$ at 120, so a batch of 4 is all memory-bound:

  $$
  T_{\text{step}} \ge \frac{4 \times 8.192 \times 10^{8} + 30 \times 10^{9} \text{ bytes}}{1.312 \times 10^{13} \text{ bytes/s}} = \mathbf{2.54 \text{ ms}}
  $$

- A batch of 256 is past 120, so the weight term becomes a FLOPs term, but attention still loads every cache:

  $$
  T_{\text{step}} \approx \frac{256 \times 8.192 \times 10^{8}}{1.312 \times 10^{13}} + \frac{2 \times 256 \times 30 \times 10^{9}}{3.152 \times 10^{15}} = 15.98 \text{ ms} + 4.87 \text{ ms} = \mathbf{20.9 \text{ ms}}
  $$

- So a batch of 4 takes at least **2.5 ms** a step and a batch of 256 around **21 ms**, and about 77% of the bigger step is just loading KV caches.

**Question 1 [parameter count]:** How many parameters does the model above have? How large is its KV cache per token in int8? Assume the input and output embedding matrices are shared.

- **MLP:** 3 matrices of $D \times F$ per layer (gate, up and down):

  $$
  3DFL = 3 \times 4096 \times 16384 \times 64 = 12{,}884{,}901{,}888
  $$

- **Attention:** $W_Q$ and $W_O$ are each $D \times N \times H$, and $W_K$ and $W_V$ are each $D \times K \times H$:

  $$
  2DH(N + K)L = 2 \times 4096 \times 256 \times (32 + 8) \times 64 = 5{,}368{,}709{,}120
  $$

- **Vocab:** one shared $D \times V$ matrix, $4096 \times 32128 = 131{,}596{,}288$. Adding them up:

  $$
  \text{Total} = 12{,}884{,}901{,}888 + 5{,}368{,}709{,}120 + 131{,}596{,}288 = 18{,}385{,}207{,}296 \approx \mathbf{18.4 \text{ billion}}
  $$

- The KV cache is a key and a value per KV head per layer, $H$ numbers each, at 1 byte in int8:

  $$
  2LKH = 2 \times 64 \times 8 \times 256 = 262{,}144 \text{ bytes} \approx \mathbf{262 \text{ kB per token}}
  $$

- Meaning, every GB of HBM only buys about 3,800 tokens of context.

**Question 2 [largest batch]:** We serve this model on a TPU v5e 4x4 slice and can fully shard the KV cache over the topology. What's the largest batch size that fits if everything is int8 and we want to support 128k-token sequences? What if we drop the number of KV heads to 1?

- The cache shards fully, so it's just capacity. The slice has $16 \times 16 \text{ GB} = 256 \text{ GB}$, and the int8 weights take 18.4 GB, leaving about 237.6 GB.
- One 128,000-token sequence needs $262{,}144 \times 128{,}000 = 33{,}554{,}432{,}000$ bytes, about 33.55 GB:

  $$
  B = \frac{256 \times 10^9 - 18.39 \times 10^9}{33.55 \times 10^9} = 7.08 \implies \mathbf{B = 7 \text{ sequences}}
  $$

- With $K = 1$ the cache is $8\times$ smaller, 4.194 GB per sequence:

  $$
  B = \frac{237.6 \times 10^9}{4.194 \times 10^9} = 56.65 \implies \mathbf{B = 56 \text{ sequences}}
  $$

- So 16 chips serve just 7 users at once, when the compute roofline wants 240.

**Question 3 [parameter load time]:** How long does it take to load all the parameters from HBM into the MXU, assuming they're int8 and fully sharded across a TPU v5e 4x4 slice? (This is a good lower bound on the per-step latency.)

- All 16 chips load their share in parallel, so it's just total bytes over total bandwidth:

  $$
  T_{\text{HBM}} = \frac{18.39 \times 10^9 \text{ bytes}}{16 \times 8.2 \times 10^{11} \text{ bytes/s}} = \mathbf{1.40 \text{ ms}}
  $$

- No generation step on this slice can beat about **1.4 ms**, so a single sequence tops out around 710 tokens/s.

**Question 4 [sharding a 4x4]:** We want to serve this model on a TPU v5e 4x4 slice with int8 FLOPs and int8 parameters and activations. How would we shard it for both prefill and decode? Hint: first ask what ICI looks like on a 4x4, what the roofline bound on tensor parallelism is, and how we can shard the KV caches. For this sharding, what's the rough per-step latency for generation?

There's no published answer for this one, so we just follow the hint.

- **What does ICI look like on a 4x4?**
  - A v5e only gets wraparound links on an axis of size 16, so a 4x4 (`Mesh({'X': 4, 'Y': 4})`) is a flat grid. Chapter 3's ring formulas assume a wraparound, so comms take roughly twice as long here and every bound below is optimistic.
- **What's the roofline bound on tensor parallelism?**
  - For prefill it's the training bound from Chapter 5. In int8, 1-byte activations and 3.94e14 OPs/s cancel the two factors of 2, so the constant matches bf16:

    $$
    \frac{4BDF}{Y \cdot C_{\text{int8}}} > \frac{2BD}{M_Y \cdot W_{\text{ici}}} \implies Y < M_Y \cdot \frac{F}{2189}
    $$

  - With $F = 16{,}384$ that's about 7.5-way per axis, 15-way over both.
  - Generation is memory-bound, and the looser $Y < F / (B \cdot \beta)$ allows 16-way up to $B \approx 112$.
- **How can we shard the KV caches?**
  - 8 KV heads can't cover 16 chips, so we shard the heads 4 ways along $Y$ and the batch 4 ways along $X$: $\text{KV}[2, B_X, S, K_Y, H]$.
  - A handful of 128k-token sequences won't split evenly, so there we'd shard the sequence instead: $\text{KV}[2, B, S_X, K_Y, H]$.
- **So how do we shard prefill and decode?**
  - **Prefill:** 4-way model parallelism along $Y$ and 4-way sequence parallelism along $X$, with $W_{\text{in}}[D, F_Y]$ and activations $[S_X, D_Y]$. That stays inside the 7.5-way bound, at the cost of replicating the weights across $X$ (4.6 GB per chip).
  - **Decode:** full 16-way model parallelism, $W_{\text{in}}[D, F_{XY}]$ and $W_{\text{out}}[F_{XY}, D]$, with $W_Q$ and $W_O$ split over the 32 query heads. That's about 1.15 GB of weights per chip, leaving 14.85 GB for cache.
- **What's the rough per-step latency for generation?**
  - We're way below 240, so it's all memory bandwidth: the 1.40 ms of weights from Question 3, plus about 20 ns per cached token ($262{,}144 / 1.312 \times 10^{13}$ seconds):

    $$
    T_{\text{step}} \approx 1.40 \text{ ms} + B \times S \times 20 \text{ ns}
    $$

  - A batch of 4 at 128k context is $1.40 + 4 \times 2.56 = \mathbf{11.6 \text{ ms}}$, and a completely full slice is 16 GB / 8.2e11 = 19.5 ms.
  - So a step takes somewhere between **1.4 ms** (tiny batch, short context) and **19.5 ms** (HBM full), and with long contexts it's almost all KV cache loading.

**Question 5 [MoE]:** Pretend the model above is actually a mixture of experts (MoE): every layer has $E$ copies of the FFW block, each token gets routed through $k$ of them, and the $k$ outputs are averaged. Use $E = 16$ and $k = 2$, with everything else as above.

- **How many total and activated parameters does it have? (Activated means used by any given token.)**
  - Only the MLP term changes, from $3DF$ to $3EDF$ per layer:

    $$
    \text{Total} = L \cdot D \cdot (3EF + 2H(N + K)) + DV = 211{,}658{,}735{,}616 \approx \mathbf{212 \text{ billion}}
    $$

  - A token only touches $k$ experts, so we swap $E$ for $k$:

    $$
    \text{Activated} = L \cdot D \cdot (3kF + 2H(N + K)) + DV = 31{,}270{,}109{,}184 \approx \mathbf{31.3 \text{ billion}}
    $$

- **What batch size is needed to become FLOPs bound on TPU v5e?**
  - The MLP loads $E$ experts' worth of bytes, but each token only does $k$ experts' worth of FLOPs. So the intensity drops by $E/k = 8$ and the threshold goes up by the same factor:

    $$
    B_{\text{crit}} = 240 \times \frac{16}{2} = \mathbf{1{,}920 \text{ tokens}}
    $$

- **How large are its KV caches per token?**
  - Unchanged, since the experts only replace the MLP: still 262,144 bytes per token in int8.
- **How many FLOPs are involved in a forward pass with $T$ tokens?**
  - Still 2 FLOPs per parameter per token, just for the activated ones:

    $$
    \text{FLOPs} = 2 \times 31.27 \times 10^{9} \times T \approx \mathbf{6.25 \times 10^{10} \times T}
    $$

  - So an MoE has the FLOPs of a 31B model and the memory footprint of a 212B one.

**Question 6 [expert sharding]:** MoEs allow "expert sharding", where the experts get split along one axis of the mesh. The first FFW weight has shape $[E, D, F]$ and is sharded as $[E_Z, D_X, F_Y]$, with $X$ only used in training as the FSDP axis. Now we want to run inference on TPU v5e.

No published answer here either, so we assume int8 weights like the earlier questions.

- **What's the HBM weight loading time for the MoE on a TPU v5e 8x16 slice with $Y = 8$, $Z = 16$? How much free HBM is available per TPU?**
  - That's 128 chips, and $Z = 16 = E$, so each row of 8 chips along $Y$ owns one expert, cut 8 ways along $F$. There's no $X$ axis at inference, so nothing is replicated. Per chip, the MLP shard is:

    $$
    \frac{3DFL}{Y} = \frac{3 \times 4096 \times 16384 \times 64}{8} = 1{,}610{,}612{,}736 \text{ bytes}
    $$

  - The attention and embedding weights (about 5.5 GB) spread over all 128 chips add 0.04 GB each, so every chip holds about 1.654 GB:

    $$
    T_{\text{HBM}} = \frac{1.654 \times 10^9 \text{ bytes}}{8.2 \times 10^{11} \text{ bytes/s}} = \mathbf{2.02 \text{ ms}}
    $$

  - That floor holds at any batch size: only 2 of the 16 experts are active per token, but the idle experts' chips can't help the active ones load.
  - Free HBM is $16 - 1.65 = \mathbf{14.35 \text{ GB per chip}}$.
- **What is the smallest slice we could fit our model on?**
  - The weights are 211.66 GB and each chip has 16 GB:

    $$
    \frac{211.66 \times 10^9}{16 \times 10^9} = 13.2 \implies \text{at least } 14 \text{ chips}
    $$

  - Slices come in sizes like 2x4, 4x4 and 4x8, so the smallest that fits is a **4x4**, with one whole expert per chip.
  - But each chip then loads 13.23 GB, which takes 16.1 ms a step, $8\times$ slower than the 8x16. So the model fits on a **4x4**, but we wouldn't want to serve on it.

**Question 7 [2D model sharding]:** This one is about what the [ESTI paper](https://arxiv.org/pdf/2211.05102) calls 2D weight-stationary sharding (sketched in Appendix B). The trick is to cut the weights along $D$ as well as $F$, so every chip holds a roughly square block. The algorithm:

1. $\text{In}[B, D_X] = \text{AllGather}_{YZ}(\text{In}[B, D_{XYZ}])$
2. $\text{Tmp}[B, F_{YZ}]\{U_X\} = \text{In}[B, D_X] \cdot_D W_{\text{in}}[D_X, F_{YZ}]$
3. $\text{Tmp}[B, F_{YZ}] = \text{AllReduce}_X(\text{Tmp}[B, F_{YZ}]\{U_X\})$
4. $\text{Out}[B, D_X]\{U_{YZ}\} = \text{Tmp}[B, F_{YZ}] \cdot_F W_{\text{out}}[F_{YZ}, D_X]$
5. $\text{Out}[B, D_{XYZ}] = \text{ReduceScatter}_{YZ}(\text{Out}[B, D_X]\{U_{YZ}\})$

Work out $T_{\text{math}}$ and $T_{\text{comms}}$ for this algorithm. When does it outperform traditional 3D model sharding?

- **Math:** every FLOP is still spread over all $N = XYZ$ chips, so $T_{\text{math}} = \frac{4BDF}{N \cdot C}$ as usual.
- **Comms:** from Chapter 3, an AllGather or ReduceScatter costs the full per-chip array's bytes over $W_{\text{ici}}$ times the number of axes, and an AllReduce costs twice that. In bf16:
  - Steps 1 and 5 run over $YZ$ (2 axes) on $2BD/X$ bytes, so $\frac{2BD}{2X \cdot W_{\text{ici}}}$ each.
  - Step 3 runs over $X$ (1 axis) on $2BF/YZ$ bytes, so $\frac{4BF}{YZ \cdot W_{\text{ici}}}$.

  $$
  T_{\text{2D comms}} = \frac{2BD}{X \cdot W_{\text{ici}}} + \frac{4BF}{YZ \cdot W_{\text{ici}}}
  $$

- With $N = X \cdot YZ$ fixed, a bigger $X$ shrinks the first term and grows the second, so the best split is where they're equal: $YZ = \sqrt{2NF/D}$. Plugging that in:

  $$
  T_{\text{2D comms}} = \frac{\sqrt{32 \cdot F/D} \cdot BD}{\sqrt{N} \cdot W_{\text{ici}}} \approx \frac{11.3 \cdot BD}{\sqrt{N} \cdot W_{\text{ici}}} \text{ for our } F = 4D
  $$

- Traditional model sharding over all 3 axes gathers and then scatters a full $[B, D]$, whatever $N$ is:

  $$
  T_{\text{1D comms}} = \frac{2BD}{3 \cdot W_{\text{ici}}} + \frac{2BD}{3 \cdot W_{\text{ici}}} = \frac{4BD}{3 \cdot W_{\text{ici}}}
  $$

- So 2D has less comms when:

  $$
  \frac{4}{3} > \frac{\sqrt{32 \cdot F/D}}{\sqrt{N}} \implies N > 32 \cdot \frac{F}{D} \cdot \left(\frac{3}{4}\right)^2 = 18 \cdot \frac{F}{D} \implies \mathbf{N > 72 \text{ chips}}
  $$

- 2D weight-stationary sharding wins once we have **more than 72 chips** (for $F = 4D$), since its comms shrink as $1/\sqrt{N}$ instead of staying flat.
