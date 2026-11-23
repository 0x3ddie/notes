---
title: All About Transformer Inference
published: true
---
## 7. Transformer Inference

Sampling from a transformer is simple on paper: feed in a sequence, get a distribution over the next token, sample one, append it, repeat. Done naively, we re-run the whole sequence for every token, so $n$ tokens cost $O(n^2)$ in the MLPs and $O(n^3)$ in attention.

The fix is the **KV cache**. A token only attends to the tokens before it, so we just save every token's keys and values once and reuse them. That drops us to $O(n)$ and $O(n^2)$, and splits inference into two jobs:

- **Prefill:** Run the whole prompt through in one pass, saving the KV cache and the last token's logits.

- **Generation:** Run one sampled token at a time against the cache, appending its keys and values, until the end token or a length limit.

Same weights, same forward pass, completely different bottlenecks.

### What We Optimize

Training only cares about throughput. Inference adds latency: time to first token (TTFT) and time per token after that. Offline batch work (evals, data generation) only cares about cost per token, chat needs a low TTFT and tokens faster than people read, and edge inference (one user on a laptop) wants the lowest latency. Batch size is the knob that trades cost against latency.

### Linear Layers and the 240 Rule

Every linear layer (the MLP matrices and the Q, K, V, O projections) is the same roofline problem from Chapter 1: a $\text{bf16}[B, D]$ batch times a $\text{bf16}[D, F]$ matrix.

$$
T_{\text{math}} = \frac{2BDF}{\text{FLOPs/s}} \qquad T_{\text{comms}} = \frac{2BD + 2DF + 2BF}{\text{HBM bytes/s}}
$$

With $B \ll D, F$ the weights dominate the bytes, so the intensity is roughly $\frac{2BDF}{2DF} = B$. We're compute-bound when $B$ beats the hardware's FLOPs per byte:

$$
B_{\text{crit}} = \frac{1.97 \times 10^{14} \text{ FLOPs/s}}{8.2 \times 10^{11} \text{ bytes/s}} \approx \mathbf{240 \text{ tokens}}
$$

That's TPU v5e in bf16, counting tokens in the matmul, not sequences. int8 weights with bf16 math halve the bytes, so $B_{\text{crit}}$ drops to 120. int8 weights AND int8 math double the FLOPs/s too, so we're back at 240.

This is where the two jobs split:

- **Prefill:** A prompt of a few hundred tokens is already a batch of a few hundred tokens, so we're compute-bound basically for free.

- **Generation:** One token per sequence per step, so hitting 240 means 240 users decoding at once, each with a KV cache in memory. We rarely get there, so every step hauls all the weights out of HBM for a tiny bit of math.

### What About Attention?

For attention the bytes are the KV cache instead of the weights. With $T$ query tokens attending to $S$ cached tokens, we do $4BSTD$ FLOPs (the $QK$ and $AV$ products) and move about $4BSD + 4BTD$ bytes:

$$
\text{Attention intensity} = \frac{4BSTD}{4BSD + 4BTD} = \frac{ST}{S + T}
$$

Prefill has $S = T$, so the intensity is $T/2$ and we're compute-bound past roughly 480 tokens. Generation has $T = 1$, so the intensity is about 1, and batching can't help because every sequence brings its own cache.

> During generation, attention is permastuck memory-bound, and the linear layers are memory-bound until around 240 tokens are batched together. Prefill is compute-bound in both.

### Latency and Throughput

The one formula to remember: at small batches, a generation step can't beat pulling everything out of HBM once.

$$
T_{\text{step}} \ge \frac{B \times \text{KV cache bytes} + \text{parameter bytes}}{\text{total HBM bandwidth}}
$$

Throughput is just $B / T_{\text{step}}$ tokens per second. Once $B$ is big enough for the linear layers to go compute-bound, a FLOPs term replaces the weight loading:

$$
T_{\text{step}} = \frac{B \times \text{KV bytes}}{\text{HBM bandwidth}} + \max\left(\frac{2 \times B \times \text{param count}}{\text{FLOPs/s}}, \frac{\text{parameter bytes}}{\text{HBM bandwidth}}\right)
$$

Small batch: quick steps, idle FLOPs. Big batch: good utilization, but every user waits longer per token. Past $B_{\text{crit}}$ per replica, throughput barely improves.

### On Memory and the KV Cache

Inference memory is just one copy of the weights plus the KV cache, which holds keys and values for every past token in every layer:

$$
\text{KV cache bytes} = 2 \times \text{bytes per float} \times H \times K \times L \times T
$$

Napkin math for LLaMA 2-13B ($L = 40$, $K = 40$, $H = 128$) in bf16: one 8,192-token sequence is about 6.7 GB of cache, and the weights are 26 GB. So just 4 sequences of cache outweigh the whole model.

Put it on 8 TPU v5e chips ($6.56 \times 10^{12}$ bytes/s of HBM bandwidth, 128 GB of HBM). Batch 1 takes around 5 ms a step, so about 200 tokens/s. The most that fits is 15 sequences, at roughly 19 ms a step and 780 tokens/s, nowhere near 240. Make the cache $5\times$ smaller (8 KV heads instead of 40) and 75 sequences fit with the same 19 ms step, so roughly 3,900 tokens/s. Same weights, $5\times$ the throughput. (The solution gets batch 16 by counting 16 GiB per chip, and uses batch 64 for the smaller cache.)

So KV cache size matters about as much as parameter count.

### Tricks for Generation

Most just shrink the KV cache:

- **GQA (grouped multi-query attention):** Fewer KV heads, each shared by several query heads. The cache shrinks by the Q to KV ratio and quality barely moves.

- **Local attention:** Some layers only look back a fixed window, so their cache stops growing.

- **Sharing KVs across layers:** Less memory, but the shared cache may still be read once per layer, so steps aren't automatically faster.

- **Quantization:** int8 or int4 weights and KVs mean fewer bytes, a lower $B_{\text{crit}}$ and bigger batches. Works after training too.

- **Ragged reads and paged attention:** Only read the cache a request actually uses. Paging stores it in blocks, like an OS page table, so nothing is padded.

- **Speculative sampling (Appendix D):** A small draft model guesses a few tokens and the big model checks them all in one batched pass, using FLOPs that were idle anyway.

Stacked, these can shrink the cache more than $10\times$ versus plain multi-head attention.

### Sharding Inference

**Prefill** is basically training without the backward pass: Megatron-style model parallelism (weights sharded along $F$) until we'd go ICI-bound, around $F / 2200$ ways per axis on v5e, then sequence parallelism for the leftover chips.

**Generation** is pickier, since we're memory-bound and latency matters:

- **No FSDP.** Moving weights over ICI is way slower than reading them from HBM.

- **No data parallelism.** That's just separate copies of the model.

- **No sequence sharding.** There's one token.

That leaves model parallelism, with a twist. Training stops adding chips once comms time passes FLOPs time, but generation is bottlenecked on weight loading, and more chips means more HBM bandwidth on the same weights. So we keep sharding until ICI time passes HBM time:

$$
\frac{2BD}{W_{\text{ici}}} > \frac{2DF}{Y \cdot W_{\text{hbm}}} \implies Y > \frac{F}{B \cdot \beta}, \qquad \beta = \frac{W_{\text{hbm}}}{W_{\text{ici}}}
$$

Our references give $\beta = 8.2 \times 10^{11} / 9.0 \times 10^{10} \approx 9.1$, so with $F = 16{,}384$ and $B = 32$ we can go to about 56-way before latency stops improving.

**The KV cache** gets sharded too, never replicated. Shard the KV heads first ($K$ ways at most), then the batch along another axis: $\text{KV}[2, B_Z, S, K_Y, H]$. That costs two AllToAlls per attention layer to reshuffle the queries. For tiny batches or very long contexts we shard the sequence instead.

### Inference Engines

Scheduling the two jobs:

- **Batched:** Prefill a batch, then generate until everything finishes. Easy, but TTFT is bad, finished sequences leave empty slots, and prompts get padded to the longest one.

- **Interleaved:** Prefill at batch 1, generate in a batch, and slot in a new prefill whenever a slot opens. Good bulk throughput, but every generation stalls while a prefill runs.

- **Disaggregated:** Separate prefill and generation servers, shipping the KV cache between them, each sharded and scaled on its own. The usual answer for latency-sensitive serving.

**Continuous batching** is the orchestration underneath: prefill writes into a KV buffer, generate steps every active slot, and a queue refills a slot as soon as its sequence finishes.

**Prefix caching** skips prefill entirely when prompts share a prefix (chat histories, system prompts), since that part of the KV cache is identical. The caches live in spare HBM or host DRAM in a trie with LRU eviction, and follow-ups get routed back to the replica holding them. Google's [JetStream](https://github.com/google/JetStream) is an open-source implementation.

> **Note:** the figures are on [jax-ml.github.io/scaling-book/inference/](https://jax-ml.github.io/scaling-book/inference/). Appendices B and C come up in Questions 7 and 4.

### Takeaway

Prefill behaves like training: lots of tokens, compute-bound, sharded the same way. Generation is a race to drag the weights and KV caches out of HBM every step, and throughput only saturates around 240 tokens per replica. The KV cache decides how big a batch we can afford, so nearly every trick here is about moving fewer bytes per token.

## Worked Problems

Questions 1 to 7 all use a made-up model based on LLaMA 2-13B:

| $L$ (layers) | $D$ (model dim) | $F$ (FFW dim) | $N$ (query heads) | $K$ (KV heads) | $H$ (head dim) | $V$ (vocab) |
|---|---|---|---|---|---|---|
| 64 | 4,096 | 16,384 | 32 | 8 | 256 | 32,128 |

TPU v5e numbers are from our Chapter 2 references: 16 GB of HBM per chip, $8.2 \times 10^{11}$ bytes/s of HBM bandwidth, $1.97 \times 10^{14}$ bf16 FLOPs/s, $3.94 \times 10^{14}$ int8 OPs/s, and $4.5 \times 10^{10}$ bytes/s per ICI link one-way ($9.0 \times 10^{10}$ bidirectional).

**Pop Quiz [bounding step time]:** We take one generate step with a batch of 4 tokens from a 30B parameter dense model on a TPU v5e 4x4 slice. Weights are int8, FLOPs are bf16, the context is 8192 tokens and the KV cache is 100 kB per token. What's a reasonable lower bound on the latency of this step? What if we sample a batch of 256 tokens instead?

- Just the step time formula from the notes, once on each side of $B_{\text{crit}}$.
- The int8 weights are $30 \times 10^9$ bytes, and one sequence's KV cache is $100{,}000 \times 8192 = 819{,}200{,}000$ bytes (819 MB).
- 16 chips give us $16 \times 8.2 \times 10^{11} = 1.312 \times 10^{13}$ bytes/s of HBM bandwidth and $16 \times 1.97 \times 10^{14} = 3.152 \times 10^{15}$ FLOPs/s.
- int8 weights with bf16 math puts $B_{\text{crit}}$ at 120. A batch of 4 is nowhere near it, so it's all memory-bound:

  $$
  T_{\text{step}} \ge \frac{4 \times 8.192 \times 10^{8} + 30 \times 10^{9} \text{ bytes}}{1.312 \times 10^{13} \text{ bytes/s}} = \mathbf{2.54 \text{ ms}}
  $$

- A batch of 256 is past 120, so the linear layers swap weight loading for FLOPs. Attention still pays to load the caches:

  $$
  T_{\text{step}} \approx \frac{256 \times 8.192 \times 10^{8}}{1.312 \times 10^{13}} + \frac{2 \times 256 \times 30 \times 10^{9}}{3.152 \times 10^{15}} = 15.98 \text{ ms} + 4.87 \text{ ms} = \mathbf{20.9 \text{ ms}}
  $$

- A batch of 4 takes at least **2.5 ms** per step, and a batch of 256 takes around **21 ms**.
- What this means: $64\times$ the batch costs about $8.2\times$ the latency, so throughput goes from roughly 1,580 to 12,270 tokens/s. Notice that about 77% of the batch-256 step is just loading KV caches.

**Question 1 [parameter count]:** How many parameters does the model above have? How large is its KV cache per token in int8? Assume the input and output embedding matrices are shared.

- This is just counting matrices, like in Chapter 4.
- **MLP:** 3 matrices of $D \times F$ per layer (gate, up and down projections):

  $$
  3DFL = 3 \times 4096 \times 16384 \times 64 = 12{,}884{,}901{,}888
  $$

- **Attention:** $W_Q$ and $W_O$ are each $D \times N \times H$, and $W_K$ and $W_V$ are each $D \times K \times H$:

  $$
  2DH(N + K)L = 2 \times 4096 \times 256 \times (32 + 8) \times 64 = 5{,}368{,}709{,}120
  $$

- **Vocab:** one shared $D \times V$ matrix, so $4096 \times 32128 = 131{,}596{,}288$.
- Adding them up:

  $$
  \text{Total} = 12{,}884{,}901{,}888 + 5{,}368{,}709{,}120 + 131{,}596{,}288 = 18{,}385{,}207{,}296 \approx \mathbf{18.4 \text{ billion}}
  $$

- The KV cache is a key and a value per KV head per layer, $H$ numbers each, at 1 byte in int8:

  $$
  2LKH = 2 \times 64 \times 8 \times 256 = 262{,}144 \text{ bytes} \approx \mathbf{262 \text{ kB per token}}
  $$

- The model has about **18.4 billion parameters** and its KV cache costs **262 kB per token** in int8. Meaning, every GB of HBM only buys about 3,800 tokens of context.

**Question 2 [largest batch]:** We serve this model on a TPU v5e 4x4 slice and can fully shard the KV cache over the topology. What's the largest batch size that fits if everything is int8 and we want to support 128k-token sequences? What if we drop the number of KV heads to 1?

- We're told the cache shards fully, so this is purely capacity: total HBM minus the weights, over one sequence's cache.
- The slice has $16 \times 16 \text{ GB} = 256 \text{ GB}$ of HBM. The int8 weights take 18.4 GB, leaving about 237.6 GB.
- One 128,000-token sequence needs $262{,}144 \times 128{,}000 = 33{,}554{,}432{,}000$ bytes, about 33.55 GB.

  $$
  B = \frac{256 \times 10^9 - 18.39 \times 10^9}{33.55 \times 10^9} = 7.08 \implies \mathbf{B = 7 \text{ sequences}}
  $$

- With $K = 1$ the cache is $8\times$ smaller: 32,768 bytes per token, or 4.194 GB per sequence.

  $$
  B = \frac{237.6 \times 10^9}{4.194 \times 10^9} = 56.65 \implies \mathbf{B = 56 \text{ sequences}}
  $$

- We can fit **7 sequences** with 8 KV heads and **56** with a single KV head. It's a tight 7, with under 3 GB left across the slice for activations, and reading 128k as 131,072 tokens would only fit 6.
- What this means: 16 chips serving 7 users at once. The compute roofline wants 240 tokens per step, and the memory wall stops us at 7.

**Question 3 [parameter load time]:** How long does it take to load all the parameters from HBM into the MXU, assuming they're int8 and fully sharded across a TPU v5e 4x4 slice? (This is a good lower bound on the per-step latency.)

- Each chip holds 1/16 of the weights (about 1.15 GB) and all 16 load in parallel, so it's just total bytes over total bandwidth:

  $$
  T_{\text{HBM}} = \frac{18.39 \times 10^9 \text{ bytes}}{16 \times 8.2 \times 10^{11} \text{ bytes/s}} = \mathbf{1.40 \text{ ms}}
  $$

- It takes about **1.4 ms** to pull every weight through the MXU once.
- No generation step on this slice can beat that, so a single sequence tops out around 710 tokens/s however small the batch.

**Question 4 [sharding a 4x4]:** We want to serve this model on a TPU v5e 4x4 slice with int8 FLOPs and int8 parameters and activations. How would we shard it for both prefill and decode? Hint: first ask what ICI looks like on a 4x4, what the roofline bound on tensor parallelism is, and how we can shard the KV caches. For this sharding, what's the rough per-step latency for generation?

No published solution for this one, so we're piecing it together from the notes.

- **What does ICI look like on a 4x4?**
  - A v5e only gets wraparound links along an axis of size 16, so a 4x4 slice, `Mesh({'X': 4, 'Y': 4})`, is a flat grid: 3 hops end to end on each axis, 6 hops corner to corner.
  - Chapter 3's ring formulas use the $9.0 \times 10^{10}$ bidirectional number, which assumes a wraparound. Without one, comms take roughly twice as long, so every bound below is optimistic.
- **What's the roofline bound on tensor parallelism?**
  - Compute-bound (prefill) is the training bound from Chapter 5: gathering activations before the MLP and scattering them after has to be faster than the math. In int8, 1-byte activations and $3.94 \times 10^{14}$ OPs/s cancel the two factors of 2, so it's the same constant as bf16:

    $$
    \frac{4BDF}{Y \cdot C_{\text{int8}}} > \frac{2BD}{M_Y \cdot W_{\text{ici}}} \implies Y < M_Y \cdot \frac{F}{2189}
    $$

  - With $F = 16{,}384$ that's about 7.5-way per axis, 15-way on both. Full 16-way is slightly past it (comms at $1.07\times$ the math, $2.1\times$ without wraparounds).
  - Memory-bound (generation) gets the looser $Y < F / (B \cdot \beta)$ from the notes, so 16-way is fine up to $B \approx 112$, and Question 2 caps us at 7 sequences anyway.
- **How can we shard the KV caches?**
  - 8 KV heads and 16 chips, so heads alone can't cover the slice. We shard the heads 4 ways along $Y$ (2 KV heads per chip) and the batch 4 ways along $X$, one split per physical axis: $\text{KV}[2, B_X, S, K_Y, H]$. That's two AllToAlls per attention layer.
  - Heads-first all the way (8 over heads, 2 over batch, on a 4x2x2 logical mesh) works too and loads the cache just as fast, since neither replicates anything.
  - For a handful of 128k-token sequences the batch won't split evenly, so we'd shard the sequence instead: $\text{KV}[2, B, S_X, K_Y, H]$.
- **So how do we shard prefill and decode?**
  - **Prefill** is compute-bound (past 240 tokens per copy of the weights, about 960 tokens with this split), so we stay inside the tensor parallelism bound: 4-way model parallelism along $Y$ and 4-way sequence parallelism along $X$, with weights $W_{\text{in}}[D, F_Y]$ and activations $[S_X, D_Y]$. That's inside 7.5, and right around the 3.7 we get by halving it for the missing wraparound. The cost is replicating the weights across $X$, 4.6 GB per chip.
  - The alternative, 8-way model and 2-way sequence parallelism, has the same comms-to-math ratio ($0.53\times$, or $1.07\times$ without wraparounds) and keeps only 2 copies of the weights (2.3 GB per chip).
  - **Decode** is memory-bound, so we go to full 16-way model parallelism: $W_{\text{in}}[D, F_{XY}]$ and $W_{\text{out}}[F_{XY}, D]$, with $W_Q$ and $W_O$ split over the 32 query heads (2 per chip), activations $[B, D]$ replicated, and the KV cache as above. That's about 1.15 GB of weights per chip, leaving about 14.85 GB each for cache.
  - (We'll ignore $W_K$ and $W_V$, which follow the cache layout and add about 0.2 GB per chip and 0.25 ms.)
  - The two want different layouts, which is the case for disaggregated serving. One slice doing both has to keep both weight copies (about 5.75 GB per chip) or run prefill comms-bound.
- **What's the rough per-step latency for generation?**
  - We're way below $B_{\text{crit}} = 240$, so it's all memory bandwidth. The weights cost the 1.40 ms from Question 3, and every cached token adds $262{,}144 / 1.312 \times 10^{13}$ seconds, about 20 ns:

    $$
    T_{\text{step}} \approx 1.40 \text{ ms} + B \times S \times 20 \text{ ns}
    $$

  - A batch of 4 at 128k context is $1.40 + 4 \times 2.56 = \mathbf{11.6 \text{ ms}}$. The full batch of 7 is 19.3 ms, and a completely full slice is $16 \text{ GB} / 8.2 \times 10^{11} = 19.5 \text{ ms}$. FLOPs don't matter: at batch 7 the math takes about 0.04 ms.
  - So the per-step latency is somewhere between **1.4 ms** (tiny batch, short context) and **19.5 ms** (HBM full), and with long contexts it's almost all KV cache loading.
  - One caveat from Appendix C: at batch 4 the activations are only about 16 kB, under the 45 kB a hop needs to be bandwidth-bound, so collectives cost about 1 μs per hop. Around 4 collectives of up to 6 hops is roughly 24 μs per layer, as long as the 22 μs to load that layer's weights, so at small batches the 1.4 ms floor could be off by around $2\times$.

**TODO(eddie):** no published solution for Question 4. Double-check the prefill split (4-way model and 4-way sequence, versus 8 and 2), the KV cache split (4 over heads and 4 over batch, versus heads-first 8 and 2), and the hop-latency estimate, which assumes 4 collectives per layer at 6 hops each.

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

  - That's about $11.5\times$ the dense model in total but only $1.7\times$ per token. (The solution writes 31.2e9, the same number cut off instead of rounded.)
- **What batch size is needed to become FLOPs bound on TPU v5e?**
  - The MLP now loads $E$ experts' worth of bytes, but each token only does $k$ experts' worth of FLOPs. So the intensity drops by $E/k = 8$ and the threshold goes up by the same factor:

    $$
    B_{\text{crit}} = 240 \times \frac{16}{2} = \mathbf{1{,}920 \text{ tokens}}
    $$

  - That's for bf16 weights and math, or int8 weights and math. int8 weights with bf16 math would be 960.
  - What this means: MoE generation is even more stuck being memory-bound, since it needs $8\times$ the dense model's 240 concurrent tokens.
- **How large are its KV caches per token?**
  - Unchanged, since the experts only replace the MLP: still $2LKH = 262{,}144$ bytes per token in int8.
- **How many FLOPs are involved in a forward pass with $T$ tokens?**
  - Still 2 FLOPs per parameter per token, just for the activated ones:

    $$
    \text{FLOPs} = 2 \times 31.27 \times 10^{9} \times T \approx \mathbf{6.25 \times 10^{10} \times T}
    $$

  - (This leaves out the attention dot products, $4T^2NH$ per layer, which catch up with the linear term around 30k tokens.)
  - So an MoE has the FLOPs of a 31B model and the memory footprint of a 212B one.

**Question 6 [expert sharding]:** MoEs allow "expert sharding", where the experts get split along one axis of the mesh. The first FFW weight has shape $[E, D, F]$ and is sharded as $[E_Z, D_X, F_Y]$, with $X$ only used in training as the FSDP axis. Now we want to run inference on TPU v5e.

No published solution here either. We'll assume int8 weights like the earlier questions; bf16 doubles every byte count.

- **What's the HBM weight loading time for the MoE on a TPU v5e 8x16 slice with $Y = 8$, $Z = 16$? How much free HBM is available per TPU?**
  - 128 chips. Notice $Z = 16 = E$, so each row of 8 chips along $Y$ owns exactly one expert, cut 8 ways along $F$. There's no $X$ axis at inference, so nothing is replicated.
  - Per chip, the MLP shard is one expert over 8:

    $$
    \frac{3DFL}{Y} = \frac{3 \times 4096 \times 16384 \times 64}{8} = 1{,}610{,}612{,}736 \text{ bytes}
    $$

  - The attention and embedding weights (about 5.5 GB), if sharded over all 128 chips, add 0.04 GB each. So every chip holds about 1.654 GB, which is just $211.66 \text{ GB} / 128$.

    $$
    T_{\text{HBM}} = \frac{1.654 \times 10^9 \text{ bytes}}{8.2 \times 10^{11} \text{ bytes/s}} = \mathbf{2.02 \text{ ms}}
    $$

  - Free HBM is $16 - 1.65 = \mathbf{14.35 \text{ GB per chip}}$, about 1,836 GB across the slice. At 262 kB per token that's roughly 7.0 million cached tokens, e.g. a batch of 54 at 128k context.
  - One trap: a token only uses 2 of the 16 experts, so it's tempting to divide the activated 31.3 GB by all 128 chips' bandwidth and get 0.30 ms. But idle experts' chips can't help: an active expert's 8 chips each pull their own 1.61 GB, which takes 1.96 ms, so the floor is about 2 ms at any batch size.
  - Sharding the attention weights 128 ways is the best case; cut 8 ways and replicated across experts, it'd be 2.30 GB per chip, 2.80 ms and 13.7 GB free.
- **What is the smallest slice we could fit our model on?**
  - The weights are 211.66 GB and each chip has 16 GB:

    $$
    \frac{211.66 \times 10^9}{16 \times 10^9} = 13.2 \implies \text{at least } 14 \text{ chips}
    $$

  - Slices come in sizes like 2x4, 4x4 and 4x8, so the smallest with 14 or more chips is a **4x4** (16 chips, 256 GB). It maps nicely too: 16 experts on 16 chips, each holding one whole expert (12.9 GB) plus its share of the attention weights, for 13.2 GB.
  - That leaves 2.77 GB per chip, 44.3 GB in total, about 169k cached tokens: a batch of 20 at 8,192 context, or one 128k sequence.
  - Weight loading goes up to $13.23 \text{ GB} / 8.2 \times 10^{11} = 16.1 \text{ ms}$ per step, $8\times$ slower than on the 8x16.
  - So the model fits on a **4x4**, but the smallest slice that fits isn't the one we'd want to serve on.

**Question 7 [2D model sharding]:** This one is about what the [ESTI paper](https://arxiv.org/pdf/2211.05102) calls 2D weight-stationary sharding (sketched in Appendix B). The trick is to cut the weights along $D$ as well as $F$, so every chip holds a roughly square block. The algorithm:

1. $\text{In}[B, D_X] = \text{AllGather}_{YZ}(\text{In}[B, D_{XYZ}])$
2. $\text{Tmp}[B, F_{YZ}]\{U_X\} = \text{In}[B, D_X] \cdot_D W_{\text{in}}[D_X, F_{YZ}]$
3. $\text{Tmp}[B, F_{YZ}] = \text{AllReduce}_X(\text{Tmp}[B, F_{YZ}]\{U_X\})$
4. $\text{Out}[B, D_X]\{U_{YZ}\} = \text{Tmp}[B, F_{YZ}] \cdot_F W_{\text{out}}[F_{YZ}, D_X]$
5. $\text{Out}[B, D_{XYZ}] = \text{ReduceScatter}_{YZ}(\text{Out}[B, D_X]\{U_{YZ}\})$

Work out $T_{\text{math}}$ and $T_{\text{comms}}$ for this algorithm. When does it outperform traditional 3D model sharding?

- Quick background: Megatron sharding ("1D weight-stationary") only cuts weights along $F$ and gathers activations to a full $[B, D]$, which gets wasteful once each chip's slice of $F$ is much smaller than $D$. Cutting along $D$ too (on axis $X$) means activations only get gathered to $[B, D_X]$, at the cost of an AllReduce in the middle.
- **Math.** Every FLOP is still spread over all $N = XYZ$ chips, so nothing changes:

  $$
  T_{\text{math}} = \frac{4BDF}{N \cdot C}
  $$

- **Comms.** Three collectives, all in bf16 (2 bytes). From Chapter 3, an AllGather or ReduceScatter costs the bytes of the full array per chip (after the gather, or before the scatter) over $W_{\text{ici}}$ times the number of axes. An AllReduce costs twice that.
  - Step 1, AllGather over $YZ$ (2 axes): each chip ends up with $[B, D/X]$, which is $2BD/X$ bytes. Cost: $\frac{2BD}{2X \cdot W_{\text{ici}}}$.
  - Step 3, AllReduce over $X$ (1 axis): each chip holds $[B, F/YZ]$, which is $2BF/YZ$ bytes. Cost: $\frac{4BF}{YZ \cdot W_{\text{ici}}}$.
  - Step 5, ReduceScatter over $YZ$: same as step 1.

  $$
  T_{\text{2D comms}} = \frac{2BD}{X \cdot W_{\text{ici}}} + \frac{4BF}{YZ \cdot W_{\text{ici}}}
  $$

- **Choosing the split.** With $N = X \cdot YZ$ fixed, a bigger $X$ shrinks the first term and grows the second, so the minimum is where they're equal: $YZ = \sqrt{2NF/D}$ and $X = N / YZ$. For $F = 4D$ (our model, $16384 / 4096 = 4$) that's $YZ = \sqrt{8N}$ and $X = \sqrt{N/8}$, and:

  $$
  T_{\text{2D comms}} = \frac{\sqrt{32 \cdot F/D} \cdot BD}{\sqrt{N} \cdot W_{\text{ici}}} = \frac{\sqrt{128} \cdot BD}{\sqrt{N} \cdot W_{\text{ici}}} \approx \frac{11.3 \cdot BD}{\sqrt{N} \cdot W_{\text{ici}}}
  $$

- **The comparison.** Traditional model sharding over all 3 axes gathers and then scatters a full $[B, D]$, which doesn't depend on $N$ at all:

  $$
  T_{\text{1D comms}} = \frac{2BD}{3 \cdot W_{\text{ici}}} + \frac{2BD}{3 \cdot W_{\text{ici}}} = \frac{4BD}{3 \cdot W_{\text{ici}}}
  $$

- The 2D scheme has less comms when:

  $$
  \frac{4}{3} > \frac{\sqrt{32 \cdot F/D}}{\sqrt{N}} \implies N > 32 \cdot \frac{F}{D} \cdot \left(\frac{3}{4}\right)^2 = 18 \cdot \frac{F}{D} \implies \mathbf{N > 72 \text{ chips}}
  $$

- 2D weight-stationary sharding beats plain model sharding once we have **more than 72 chips** (for $F = 4D$).
- What this means: with 1D sharding the comms stay flat while the math shrinks as $1/N$, so we go comms-bound around 20-way ($3F / 2189 \approx 22$ for our $F$). With 2D the comms shrink as $1/\sqrt{N}$, so even comms-bound, adding chips keeps cutting generation latency.
