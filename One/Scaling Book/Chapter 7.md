---
title: All About Transformer Inference
published: true
---
## 7. Transformer Inference

Sampling from a transformer is simple on paper: feed in a sequence, get a distribution over the next token, sample one, append it, repeat. The naive version re-runs the whole sequence for every new token, so generating $n$ tokens costs $O(n^2)$ in the MLPs and $O(n^3)$ in attention. Nobody actually does this.

The fix is the **KV cache**. A token only ever attends to the tokens before it, so we can just save every token's key and value projections the first time we compute them. Later tokens reuse them, and we never touch an old token again. That drops us to $O(n)$ in the MLPs and $O(n^2)$ in attention, and it splits inference into two jobs:

- **Prefill:** Run the whole prompt through the model in one pass. Save the KV cache and the logits of the last token.

- **Generation:** Sample a token, run just that one token through the model against the cache, append its keys and values, and repeat until we hit the end token or a length limit.

Same weights, same forward pass, but the two have completely different bottlenecks. Most of this chapter is about why.

### What We Optimize

Training only cares about throughput. Inference adds latency: the time to first token (TTFT) and the time per token after that. Which one matters depends on the job. Offline batch work (evals, data generation) only cares about cost per token. Chat needs a low TTFT and tokens that arrive faster than people read. Edge inference (one user on a laptop) just wants the lowest latency the hardware allows.

High utilization still lowers cost, but it doesn't automatically make any single user's experience better. Batch size is the knob that trades one for the other.

### Linear Layers and the 240 Rule

Every linear layer (the MLP matrices and the Q, K, V, O projections) is the same roofline problem from Chapter 1: a $\text{bf16}[B, D]$ batch times a $\text{bf16}[D, F]$ matrix.

$$
T_{\text{math}} = \frac{2BDF}{\text{FLOPs/s}} \qquad T_{\text{comms}} = \frac{2BD + 2DF + 2BF}{\text{HBM bytes/s}}
$$

With $B \ll D, F$ the weights dominate the bytes, so the intensity is roughly $\frac{2BDF}{2DF} = B$. We're compute-bound when $B$ beats the hardware's FLOPs per byte:

$$
B_{\text{crit}} = \frac{1.97 \times 10^{14} \text{ FLOPs/s}}{8.2 \times 10^{11} \text{ bytes/s}} \approx \mathbf{240 \text{ tokens}}
$$

That's TPU v5e in bf16. $B$ here counts tokens going through the matmul at once, not sequences. Quantization moves the threshold: int8 weights with bf16 math halve the bytes, so $B_{\text{crit}}$ drops to 120. int8 weights AND int8 math double the FLOPs/s as well, so we're back at 240.

This is where the two jobs split:

- **Prefill:** A prompt of a few hundred tokens already is a batch of a few hundred tokens. We're compute-bound basically for free, so we just maximize FLOPs utilization.

- **Generation:** One token per sequence per step. To reach 240 we need 240 users decoding at the same moment, and 240 KV caches in memory. That's hard, so generation usually sits memory-bound: every step hauls the full set of weights out of HBM to do a tiny amount of math.

### What About Attention?

For attention, the bytes are the KV cache instead of the weights. With $T$ query tokens attending to $S$ cached tokens, we do $4BSTD$ FLOPs (the $QK$ product and the $AV$ product) and move about $4BSD + 4BTD$ bytes:

$$
\text{Attention intensity} = \frac{4BSTD}{4BSD + 4BTD} = \frac{ST}{S + T}
$$

Prefill has $S = T$, so the intensity is $T/2$ and we're compute-bound once the prompt is longer than roughly 480 tokens. Generation has $T = 1$, so the intensity is about 1. And batching can't rescue it, because every sequence brings its own KV cache; the bytes grow exactly as fast as the FLOPs.

> During generation, attention is permastuck memory-bound, and the linear layers are memory-bound until around 240 tokens are batched together. Prefill is compute-bound in both.

### Latency and Throughput

The book says that if we remember one thing from the chapter, it should be this. At small batch sizes a generation step can't be faster than pulling everything out of HBM once:

$$
T_{\text{step}} \ge \frac{B \times \text{KV cache bytes} + \text{parameter bytes}}{\text{total HBM bandwidth}}
$$

Throughput is then just $B / T_{\text{step}}$ tokens per second. Once $B$ is large enough for the linear layers to turn compute-bound, a FLOPs term replaces the weight-loading term:

$$
T_{\text{step}} = \frac{B \times \text{KV bytes}}{\text{HBM bandwidth}} + \max\left(\frac{2 \times B \times \text{param count}}{\text{FLOPs/s}}, \frac{\text{parameter bytes}}{\text{HBM bandwidth}}\right)
$$

At a low batch size each step is quick, but most of the chip's FLOPs sit idle. At a high one the hardware is well used, but every user waits longer per token. Throughput stops improving much once the batch on a replica passes $B_{\text{crit}}$.

### On Memory and the KV Cache

Inference memory is simple: one copy of the weights, no optimizer state, no saved activations. The new thing is the KV cache, which holds keys and values for every past token in every layer:

$$
\text{KV cache bytes} = 2 \times \text{bytes per float} \times H \times K \times L \times T
$$

For LLaMA 2-13B ($L = 40$, $K = 40$, $H = 128$) in bf16, one 8,192-token sequence is about 6.7 GB. The weights are 26 GB. So just 4 sequences of cache outweigh the whole model.

The book models generation for this model on 8 TPU v5e chips ($6.56 \times 10^{12}$ bytes/s of HBM bandwidth). At batch 1 a step takes around 5 ms, so about 200 tokens/s. It says we run out of memory past batch 16 (133 GB, about 20 ms a step), which counts 16 GiB per chip, about 137 GB in total. With the 16 GB ($16 \times 10^9$ bytes) from our reference table we get 128 GB, and the most that fits is 15 sequences, at roughly 19 ms a step and 780 tokens/s. Either way we hit a memory wall long before 240. Now make the cache $5\times$ smaller (8 KV heads instead of 40): 75 sequences fit in the same memory with the same 19 ms step, so roughly 3,900 tokens/s (the book quotes batch 64, at about 3,760 tokens/s). Same weights, $5\times$ the throughput.

So the size of the KV cache matters about as much as the parameter count.

### Tricks for Generation

Most of them shrink the KV cache:

- **Grouped multi-query attention (GQA / GMQA):** Fewer KV heads, shared across many query heads. The cache shrinks by the Q to KV ratio, and quality barely moves.

- **Local attention layers:** Some layers only look back a fixed window, so their cache stops growing past it.

- **Sharing KVs across layers:** Less memory, but the shared cache may still get read once per layer, so the step isn't automatically any faster.

- **Quantization:** int8 or int4 weights and KVs mean fewer bytes to move, a lower $B_{\text{crit}}$ and room for bigger batches. It can be applied after training.

- **Ragged reads and paged attention:** Only read the part of the cache a request actually uses instead of the padded maximum. Paging stores the cache in blocks, like an OS page table, so nothing is padded at all.

- **Speculative sampling (Appendix D):** A small draft model guesses a few tokens, and the big model checks them all in one batched pass, using FLOPs that were sitting idle anyway.

Stacked together, the book says these can shrink the cache by more than $10\times$ against plain multi-head attention.

### Sharding Inference

**Prefill** is basically training without the backward pass, so the same schemes work. The rule: model parallelism (Megatron style, shard the weights along $F$) until we'd be ICI-bound, which is around $F / 2200$ ways per mesh axis on v5e, then sequence parallelism for whatever chips are left.

**Generation** is pickier, since we're memory-bound and latency matters:

- **No FSDP.** FSDP moves weights between chips, and ICI is far slower than HBM. The only thing cheap enough to send around is activations (tiny); weights and caches stay where they are.

- **No data parallelism.** It just replicates the weights. Running separate copies of the model does the same job.

- **No sequence sharding.** There's one token.

That leaves model parallelism, and here's the difference from training. In training we stop adding chips when comms time passes FLOPs time. In generation the FLOPs aren't the bottleneck; weight loading is, and more chips means more HBM bandwidth pulling on the same weights. So we can keep sharding until ICI time passes HBM time:

$$
\frac{2BD}{W_{\text{ici}}} > \frac{2DF}{Y \cdot W_{\text{hbm}}} \implies Y > \frac{F}{B \cdot \beta}, \qquad \beta = \frac{W_{\text{hbm}}}{W_{\text{ici}}}
$$

The book calls $\beta$ "around 8" on v5e; our reference numbers give $8.2 \times 10^{11} / 9.0 \times 10^{10} \approx 9.1$. With $F = 16{,}384$ and $B = 32$ that still allows somewhere around 56 to 64-way sharding before latency stops improving.

**The KV cache** has to be sharded too, and we never want to replicate it. First shard along the KV heads ($K$ ways at most). If there are more chips than heads, shard the batch along another axis: $\text{KV}[2, B_Z, S, K_Y, H]$. The price is two AllToAlls per attention layer, to move the query activations into the batch-sharded layout and back. For tiny batches or very long contexts we can shard the sequence dimension instead.

### Inference Engines

Then there's how the two jobs get scheduled:

- **Batched:** Run a batch of prefills, then generate until everything finishes. Easy to build, but TTFT is bad, finished sequences leave empty slots, and prompts get padded to the longest one.

- **Interleaved:** Prefill at batch 1, generate in a batch, and slot in a new prefill whenever a generation slot opens. Good for bulk throughput, but everyone's generation stalls while a prefill runs.

- **Disaggregated:** Separate prefill servers and generation servers, with the KV cache shipped between them. Each side gets its own sharding and scales on its own. This is the usual answer for latency-sensitive serving.

**Continuous batching** is the orchestration underneath: a prefill function that writes into a KV buffer, a generate function that steps every active slot, and a queue that refills a slot the moment its sequence finishes.

**Prefix caching** skips prefill work entirely. Two prompts with the same prefix have identical KV caches for that prefix, so chat histories and shared system prompts get reused instead of recomputed. The caches live in spare HBM or host DRAM, in a trie with LRU eviction, and follow-up requests have to be routed back to the replica that holds them. Google's [JetStream](https://github.com/google/JetStream) is an open-source implementation.

On the other appendices: A checks the 240 rule against a real profile, B covers 2D weight-stationary sharding (worked in Question 7), and C points out that collectives on small activations are bound by the roughly 1 μs per-hop latency instead of bandwidth (anything under 45 kB per hop on v5e). The figures are on [jax-ml.github.io/scaling-book/inference/](https://jax-ml.github.io/scaling-book/inference/).

### Takeaway

Prefill behaves like training: lots of tokens at once, compute-bound, sharded the same way. Generation is the opposite. With one token per sequence, each step is basically a race to drag the weights and the KV caches out of HBM, so the step time is (KV caches + weights) divided by HBM bandwidth, and throughput only saturates at around 240 tokens per replica. The KV cache decides how big a batch we can afford. Thus nearly every trick here (fewer KV heads, quantization, paging, sharding past the usual limit, separate prefill and generation servers) is about moving fewer bytes per token.

## Worked Problems

The inline quiz comes first, then the book's problem set. Questions 1 to 7 all use a made-up model based on LLaMA 2-13B:

| $L$ (layers) | $D$ (model dim) | $F$ (FFW dim) | $N$ (query heads) | $K$ (KV heads) | $H$ (head dim) | $V$ (vocab) |
|---|---|---|---|---|---|---|
| 64 | 4,096 | 16,384 | 32 | 8 | 256 | 32,128 |

The TPU v5e numbers come from the Chapter 2 references: 16 GB of HBM per chip, $8.2 \times 10^{11}$ bytes/s of HBM bandwidth, $1.97 \times 10^{14}$ bf16 FLOPs/s, $3.94 \times 10^{14}$ int8 OPs/s, and $4.5 \times 10^{10}$ bytes/s per ICI link one-way ($9.0 \times 10^{10}$ bidirectional).

**Pop Quiz [bounding step time]:** We take one generate step with a batch of 4 tokens from a 30B parameter dense model on a TPU v5e 4x4 slice. Weights are int8, FLOPs are bf16, the context is 8192 tokens and the KV cache is 100 kB per token. What's a reasonable lower bound on the latency of this step? What if we sample a batch of 256 tokens instead?

- This is the step time formula from the notes, used once on each side of $B_{\text{crit}}$.
- Bytes first. The int8 weights are $30 \times 10^9$ bytes. One sequence's KV cache is $100{,}000 \times 8192 = 819{,}200{,}000$ bytes (819 MB).
- Then the hardware. 16 chips give us $16 \times 8.2 \times 10^{11} = 1.312 \times 10^{13}$ bytes/s of HBM bandwidth and $16 \times 1.97 \times 10^{14} = 3.152 \times 10^{15}$ FLOPs/s.
- With int8 weights and bf16 math, $B_{\text{crit}}$ is 120. A batch of 4 is nowhere near it, so everything is memory-bound:

  $$
  T_{\text{step}} \ge \frac{4 \times 8.192 \times 10^{8} + 30 \times 10^{9} \text{ bytes}}{1.312 \times 10^{13} \text{ bytes/s}} = \mathbf{2.54 \text{ ms}}
  $$

- A batch of 256 is past 120, so the linear layers are compute-bound and the weight-loading term gets swapped for FLOPs. Attention still pays for loading the caches:

  $$
  T_{\text{step}} \approx \frac{256 \times 8.192 \times 10^{8}}{1.312 \times 10^{13}} + \frac{2 \times 256 \times 30 \times 10^{9}}{3.152 \times 10^{15}} = 15.98 \text{ ms} + 4.87 \text{ ms} = \mathbf{20.9 \text{ ms}}
  $$

- A batch of 4 takes at least **2.5 ms** per step, and a batch of 256 takes around **21 ms**. Same as the book's answer.
- What this means: $64\times$ the batch costs about $8.2\times$ the latency, so throughput goes from roughly 1,580 to 12,270 tokens/s. Also, at batch 256 about 77% of the step is spent loading KV caches, not doing math.

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

- The KV cache stores a key and a value for each of $K$ heads of size $H$ in each of $L$ layers, at 1 byte per number in int8:

  $$
  2LKH = 2 \times 64 \times 8 \times 256 = 262{,}144 \text{ bytes} \approx \mathbf{262 \text{ kB per token}}
  $$

- The model has about **18.4 billion parameters** and its KV cache costs **262 kB per token** in int8, which matches the book. Meaning, every GB of HBM only buys about 3,800 tokens of context.

**Question 2 [largest batch]:** We serve this model on a TPU v5e 4x4 slice and can fully shard the KV cache over the topology. What's the largest batch size that fits if everything is int8 and we want to support 128k-token sequences? What if we drop the number of KV heads to 1?

- This is purely a capacity question: total HBM, minus the weights, divided by one sequence's KV cache.
- The slice has $16 \times 16 \text{ GB} = 256 \text{ GB}$ of HBM. The int8 weights take 18.4 GB, leaving about 237.6 GB.
- One 128,000-token sequence needs $262{,}144 \times 128{,}000 = 33{,}554{,}432{,}000$ bytes, so about 33.55 GB of cache.

  $$
  B = \frac{256 \times 10^9 - 18.39 \times 10^9}{33.55 \times 10^9} = 7.08 \implies \mathbf{B = 7 \text{ sequences}}
  $$

- With $K = 1$ the cache is $8\times$ smaller: 32,768 bytes per token, or 4.194 GB per sequence.

  $$
  B = \frac{237.6 \times 10^9}{4.194 \times 10^9} = 56.65 \implies \mathbf{B = 56 \text{ sequences}}
  $$

- ($W_K$ and $W_V$ shrink too, which takes the weights from 18.4 GB to 17.4 GB, but the answer is still 56.)
- We can fit **7 sequences** with 8 KV heads and **56** with a single KV head. The solution gets the same 7 and "about 56".
- It's a tight 7, though, and it depends on the reading. Like the solution, we took 128k as 128,000 tokens and 16 GB as $16 \times 10^9$ bytes. With 131,072 tokens the division gives 6.9 and we only fit 6 (55 with one KV head), and with 16 GiB chips it's 7 on either token count (59 to 61 with one KV head). On our numbers there's under 3 GB left across the whole slice for activations.
- What this means: 16 chips serving 7 users at once. The compute roofline wants 240 tokens per step, and the memory wall stops us at 7.

**Question 3 [parameter load time]:** How long does it take to load all the parameters from HBM into the MXU, assuming they're int8 and fully sharded across a TPU v5e 4x4 slice? (This is a good lower bound on the per-step latency.)

- Each chip holds 1/16 of the weights (about 1.15 GB) and all 16 load their piece in parallel. So it's just total bytes over total bandwidth:

  $$
  T_{\text{HBM}} = \frac{18.39 \times 10^9 \text{ bytes}}{16 \times 8.2 \times 10^{11} \text{ bytes/s}} = \mathbf{1.40 \text{ ms}}
  $$

- It takes about **1.4 ms** to pull every weight through the MXU once, which matches the book.
- No generation step on this slice can beat that, so a single sequence tops out at around 710 tokens/s no matter how small the batch is.

**Question 4 [sharding a 4x4]:** We want to serve this model on a TPU v5e 4x4 slice with int8 FLOPs and int8 parameters and activations. How would we shard it for both prefill and decode? Hint: first ask what ICI looks like on a 4x4, what the roofline bound on tensor parallelism is, and how we can shard the KV caches. For this sharding, what's the rough per-step latency for generation?

The book doesn't publish a solution for this one, so here's how I'd put the pieces from the notes together.

- **What does ICI look like on a 4x4?**
  - A v5e only has wraparound links along an axis of size 16. So a 4x4 slice, `Mesh({'X': 4, 'Y': 4})`, is a flat grid: two mesh axes, 3 hops end to end on each, 6 hops corner to corner.
  - Each link carries $4.5 \times 10^{10}$ bytes/s one way. The ring formulas from Chapter 3 use the $9.0 \times 10^{10}$ bidirectional number, which assumes a wraparound. Without one, comms take roughly twice as long, so every bound below is on the optimistic side.
- **What's the roofline bound on tensor parallelism?**
  - There are two, depending on whether we're compute-bound.
  - When we're compute-bound (prefill), it's the training bound from Chapter 5: gathering the activations before the MLP and scattering them after has to take less time than the math. In int8 the activations are 1 byte each and the chip does $3.94 \times 10^{14}$ OPs/s, so the two factors of 2 cancel and we get the same constant as bf16:

    $$
    \frac{4BDF}{Y \cdot C_{\text{int8}}} > \frac{2BD}{M_Y \cdot W_{\text{ici}}} \implies Y < M_Y \cdot \frac{F}{2189}
    $$

  - With $F = 16{,}384$ that's about 7.5-way per axis (the book rounds the constant to 2200), so 15-way using both axes. Full 16-way tensor parallelism is already slightly past the limit (comms at $1.07\times$ the math), and without wraparounds it's more like $2.1\times$.
  - When we're memory-bound (generation), it's the looser bound from the notes, $Y < F / (B \cdot \beta)$. With $\beta \approx 9.1$, 16-way sharding is fine up to $B \approx 112$ (or 56 if we only count one-way links). Question 2 caps us at 7 sequences for 128k contexts anyway.
- **How can we shard the KV caches?**
  - We have 8 KV heads and 16 chips, so heads alone can't cover the slice. Shard the heads 4 ways along one axis (2 KV heads per chip) and the batch along the other: $\text{KV}[2, B_X, S, K_Y, H]$. This costs two AllToAlls per attention layer.
  - That split is a judgment call. The book's rule is heads first, as far as they go, then batch. Taken literally that's 8 ways over heads (1 KV head per chip) and 2 over batch, treating the slice as a 4x2x2 logical mesh. I went with 4 and 4 so each split follows one physical axis. Either way nothing is replicated, so the cache loading time below is the same.
  - With only a handful of 128k-token sequences the batch doesn't split evenly, and the sequence is the long dimension anyway, so we'd shard that instead: $\text{KV}[2, B, S_X, K_Y, H]$.
- **So how do we shard prefill and decode?**
  - **Prefill** is compute-bound (past 240 tokens per copy of the weights, so about 960 tokens with the split below), so we shouldn't push tensor parallelism past its bound. I'd do 4-way model parallelism along $Y$ and 4-way sequence parallelism along $X$: weights $W_{\text{in}}[D, F_Y]$, activations $[S_X, D_Y]$. 4-way on a single axis sits inside the 7.5 bound, and right around the 3.7 we get if we halve it for the missing wraparound. The weights get replicated across $X$, which costs 4.6 GB per chip.
  - Again a judgment call. 8-way model parallelism over both axes plus 2-way sequence parallelism (the 4x2x2 view again) has the same ratio of comms to math ($0.53\times$ with ring bandwidth, $1.07\times$ without wraparounds) and only keeps 2 copies of the weights (2.3 GB per chip). I kept 4 and 4 for the same reason as the cache.
  - **Decode** is memory-bound, so we go to full 16-way model parallelism: $W_{\text{in}}[D, F_{XY}]$ and $W_{\text{out}}[F_{XY}, D]$, with $W_Q$ and $W_O$ split over the 32 query heads (2 per chip), activations $[B, D]$ replicated, and the KV cache as above. That's about 1.15 GB of weights per chip, leaving about 14.85 GB each for cache.
  - (The small $W_K$ and $W_V$ matrices follow the cache layout, so they're really sharded 4 ways over the KV heads and replicated along $X$. That adds about 0.2 GB per chip and roughly 0.25 ms to the floor below, which I'm ignoring. With the 8 and 2 split it would be 0.07 GB and 0.08 ms.)
  - The two want different layouts, which is exactly the argument for disaggregated serving. If one slice has to do both, we either keep both copies of the weights (about 5.75 GB per chip) or reuse the decode layout for prefill and accept that prefill is comms-bound.
- **What's the rough per-step latency for generation?**
  - We're far below $B_{\text{crit}} = 240$, so it's all memory bandwidth. The weights cost the 1.40 ms from Question 3, and every cached token adds $262{,}144 / 1.312 \times 10^{13}$ seconds, which is about 20 ns:

    $$
    T_{\text{step}} \approx 1.40 \text{ ms} + B \times S \times 20 \text{ ns}
    $$

  - A batch of 4 at 128k context is $1.40 + 4 \times 2.56 = \mathbf{11.6 \text{ ms}}$. The full batch of 7 is 19.3 ms, and a completely full slice is $16 \text{ GB} / 8.2 \times 10^{11} = 19.5 \text{ ms}$. A batch of 32 at 8,192 context would be about 6.6 ms.
  - FLOPs don't matter here: at batch 7 the math takes about 0.04 ms.
  - So the per-step latency is somewhere between **1.4 ms** (tiny batch, short context) and **19.5 ms** (HBM full), and almost all of it is KV cache loading once the contexts get long.
  - One caveat from Appendix C. At batch 4 the activations are only about 16 kB, far below the 45 kB it takes for a hop to be bandwidth-bound, so every collective costs about 1 μs per hop instead. If a layer needs around 4 collectives of up to 6 hops each, that's roughly 24 μs per layer, about the same as the 22 μs it takes to load that layer's weights. So the 1.4 ms floor could be off by something like $2\times$ at small batch sizes.

**TODO(eddie):** the book has no published solution for Question 4. Check the prefill split (4-way model and 4-way sequence, versus 8 and 2), the KV cache split (4 over heads and 4 over batch, versus the book's heads-first 8 and 2), and the hop-latency estimate, which assumes 4 collectives per layer at 6 hops each.

**Question 5 [MoE]:** Pretend the model above is actually a mixture of experts (MoE): every layer has $E$ copies of the FFW block, each token gets routed through $k$ of them, and the $k$ outputs are averaged. Use $E = 16$ and $k = 2$, with everything else as above.

- **How many total and activated parameters does it have? (Activated means used by any given token.)**
  - Only the MLP term changes. Each layer now has $3EDF$ MLP parameters instead of $3DF$.

    $$
    \text{Total} = L \cdot D \cdot (3EF + 2H(N + K)) + DV = 211{,}658{,}735{,}616 \approx \mathbf{212 \text{ billion}}
    $$

  - A single token only touches $k$ of the experts, so we swap $E$ for $k$:

    $$
    \text{Activated} = L \cdot D \cdot (3kF + 2H(N + K)) + DV = 31{,}270{,}109{,}184 \approx \mathbf{31.3 \text{ billion}}
    $$

  - That's about $11.5\times$ the dense model in total but only $1.7\times$ per token. The solution writes 31.2e9 for the activated count, which is the same number cut off instead of rounded.
- **What batch size is needed to become FLOPs bound on TPU v5e?**
  - For the MLP block we now have to load $E$ experts' worth of bytes, but each token only does $k$ experts' worth of FLOPs. So the intensity per token drops by $E/k = 8$, and the threshold goes up by the same factor:

    $$
    B_{\text{crit}} = 240 \times \frac{16}{2} = \mathbf{1{,}920 \text{ tokens}}
    $$

  - That's with bf16 weights and bf16 math, or int8 weights and int8 math. With int8 weights and bf16 math it would be 960.
  - This matches the book. If we count the attention and embedding weights too (which don't get multiplied by $E$), the whole model's ratio of total to activated parameters is 6.77, which gives around 1,620. Same ballpark.
  - What this means: the dense model needed 240 concurrent tokens to use the chip properly, and the MoE needs $8\times$ that. MoE generation is even more stuck in the memory-bound regime.
- **How large are its KV caches per token?**
  - Unchanged. The experts only replace the MLP, so attention still stores $2LKH = 262{,}144$ bytes per token in int8.
- **How many FLOPs are involved in a forward pass with $T$ tokens?**
  - Still 2 FLOPs per parameter per token, but only for the activated parameters:

    $$
    \text{FLOPs} = 2 \times 31.27 \times 10^{9} \times T \approx \mathbf{6.25 \times 10^{10} \times T}
    $$

  - That's the book's answer too (it writes $2 \times 31.2 \times 10^{9} \times T$). It leaves out the attention dot products, which aren't in the parameter count: $4T^2NH$ per layer in a forward pass (a third of the $12BT^2NH$ training figure from Chapter 4), so about $2.1 \times 10^{6} \times T^2$ here, or half that with a causal mask.
  - That term catches up with the linear one at around 30k tokens. It adds about 27% for an 8,192-token prompt, and at the 128k contexts from Question 2 it's roughly $4\times$ the linear term.
  - So an MoE has the FLOPs of a 31B model and the memory footprint of a 212B one.

**Question 6 [expert sharding]:** MoEs allow "expert sharding", where the experts get split along one axis of the mesh. The first FFW weight has shape $[E, D, F]$ and is sharded as $[E_Z, D_X, F_Y]$, with $X$ only used in training as the FSDP axis. Now we want to run inference on TPU v5e.

The book has no published solution for this one either. I'm assuming int8 weights like the earlier questions; bf16 doubles every byte count.

- **What's the HBM weight loading time for the MoE on a TPU v5e 8x16 slice with $Y = 8$, $Z = 16$? How much free HBM is available per TPU?**
  - We have 128 chips. With $Z = 16 = E$, each row of 8 chips along $Y$ owns exactly one expert, cut 8 ways along $F$. There's no $X$ axis at inference, so nothing is replicated.
  - Per chip, the MLP shard is one expert's weights over 8:

    $$
    \frac{3DFL}{Y} = \frac{3 \times 4096 \times 16384 \times 64}{8} = 1{,}610{,}612{,}736 \text{ bytes}
    $$

  - The attention and embedding weights (about 5.5 GB), if we assume they're fully sharded over all 128 chips, add another 0.04 GB each. So every chip holds about 1.654 GB, which is just $211.66 \text{ GB} / 128$.

    $$
    T_{\text{HBM}} = \frac{1.654 \times 10^9 \text{ bytes}}{8.2 \times 10^{11} \text{ bytes/s}} = \mathbf{2.02 \text{ ms}}
    $$

  - Free HBM is $16 - 1.65 = \mathbf{14.35 \text{ GB per chip}}$, or about 1,836 GB across the slice. At 262 kB per token that's room for roughly 7.0 million cached tokens, e.g. a batch of 54 at 128k context.
  - One trap here. A token only uses 2 of the 16 experts, so it's tempting to divide the activated parameters (31.3 GB) by the bandwidth of all 128 chips and get 0.30 ms. But the chips holding idle experts can't help the busy ones load. An active expert's 8 chips each have to pull their own 1.61 GB, which takes 1.96 ms regardless. So the step floor is about 2 ms at any batch size.
  - The 128-way split of the attention weights is my assumption, and it's the best case. If they were only cut 8 ways along $Y$ and replicated across the expert axis, each chip would hold 2.30 GB: 2.80 ms and 13.7 GB free.
  - (In bf16: 3.31 GB per chip, 4.03 ms, and 12.7 GB free.)
- **What is the smallest slice we could fit our model on?**
  - The weights are 211.66 GB and each chip has 16 GB:

    $$
    \frac{211.66 \times 10^9}{16 \times 10^9} = 13.2 \implies \text{at least } 14 \text{ chips}
    $$

  - Slices come in sizes like 2x4, 4x4 and 4x8, so the smallest one with 14 or more chips is a **4x4** (16 chips, 256 GB). It even maps nicely: 16 experts on 16 chips, with each chip holding one whole expert (12.9 GB) plus its share of the attention weights, for 13.2 GB.
  - That leaves 2.77 GB per chip, 44.3 GB in total. That's about 169k cached tokens: a batch of 20 at 8,192 context, or a single 128k sequence.
  - The weight loading time also goes up to $13.23 \text{ GB} / 8.2 \times 10^{11} = 16.1 \text{ ms}$ per step, $8\times$ slower than on the 8x16.
  - So the model fits on a **4x4**, but the smallest slice that fits is not the one we'd want to serve on. (In bf16 the weights are 423 GB, which needs 27 chips, so a 4x8.)

The book has no published solution for Question 6 either, so it's worth being clear about the two assumptions: int8 weights (bf16 doubles the load time to 4.03 ms) and the attention weights sharded over all 128 chips (2.02 ms, versus 2.80 ms if they're replicated across $Z$).

**Question 7 [2D model sharding]:** This one is about what the [ESTI paper](https://arxiv.org/pdf/2211.05102) calls 2D weight-stationary sharding (the book sketches it in Appendix B). The trick is to cut the weights along $D$ as well as $F$, so every chip holds a roughly square block. That lowers the comms load and lets us scale a little further. The algorithm:

1. $\text{In}[B, D_X] = \text{AllGather}_{YZ}(\text{In}[B, D_{XYZ}])$
2. $\text{Tmp}[B, F_{YZ}]\{U_X\} = \text{In}[B, D_X] \cdot_D W_{\text{in}}[D_X, F_{YZ}]$
3. $\text{Tmp}[B, F_{YZ}] = \text{AllReduce}_X(\text{Tmp}[B, F_{YZ}]\{U_X\})$
4. $\text{Out}[B, D_X]\{U_{YZ}\} = \text{Tmp}[B, F_{YZ}] \cdot_F W_{\text{out}}[F_{YZ}, D_X]$
5. $\text{Out}[B, D_{XYZ}] = \text{ReduceScatter}_{YZ}(\text{Out}[B, D_X]\{U_{YZ}\})$

Work out $T_{\text{math}}$ and $T_{\text{comms}}$ for this algorithm. When does it outperform traditional 3D model sharding?

- Some background from Appendix B first. Normal Megatron sharding is "1D weight-stationary": the weights are only cut along $F$, and the activations get gathered to a full $[B, D]$ before the MLP. With lots of chips, each chip's slice of $F$ gets much smaller than $D$, so we end up shipping a big $[B, D]$ activation around just to feed a skinny slice of weights. The 2D version also cuts the weights along $D$ (on mesh axis $X$). The activations are then only ever gathered to $[B, D_X]$, and in exchange we pay for an AllReduce on the hidden activations in the middle.
- **Math.** Every FLOP is still spread over all $N = XYZ$ chips, so nothing changes:

  $$
  T_{\text{math}} = \frac{4BDF}{N \cdot C}
  $$

- **Comms.** There are three collectives, all in bf16 (2 bytes). From Chapter 3: an AllGather or ReduceScatter costs the bytes of the full array each chip holds (after the gather, or before the scatter), divided by $W_{\text{ici}}$ and by the number of axes it runs over. An AllReduce costs twice that.
  - Step 1, AllGather over $YZ$ (2 axes): each chip ends up with $[B, D/X]$, which is $2BD/X$ bytes. Cost: $\frac{2BD}{2X \cdot W_{\text{ici}}}$.
  - Step 3, AllReduce over $X$ (1 axis): each chip holds $[B, F/YZ]$, which is $2BF/YZ$ bytes. Cost: $\frac{4BF}{YZ \cdot W_{\text{ici}}}$.
  - Step 5, ReduceScatter over $YZ$: same as step 1.

  $$
  T_{\text{2D comms}} = \frac{2BD}{X \cdot W_{\text{ici}}} + \frac{4BF}{YZ \cdot W_{\text{ici}}}
  $$

- **Choosing the split.** With $N = X \cdot YZ$ fixed, a bigger $X$ shrinks the first term and grows the second, so the minimum is where the two terms are equal. That gives $YZ = \sqrt{2NF/D}$ and $X = N / YZ$. For $F = 4D$ (true for our model, $16384 / 4096 = 4$) this is $YZ = \sqrt{8N}$ and $X = \sqrt{N/8}$, and:

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

- A quick check with whole-number splits, in units of $BD / W_{\text{ici}}$: at 64 chips ($X = 4$, $YZ = 16$) the 2D scheme costs 1.5 against Megatron's 1.33, so Megatron still wins. At 128 chips ($X = 4$, $YZ = 32$) it's 1.0 against 1.33, and at 512 chips ($X = 8$, $YZ = 64$) it's 0.5 against 1.33.
- 2D weight-stationary sharding beats plain model sharding once we have **more than 72 chips** (for $F = 4D$), which is the book's answer.
- What this means: with 1D sharding the comms stay flat while the math shrinks as $1/N$, which is why we go comms-bound at around 20-way ($3F / 2189 \approx 22$ for our $F$). With 2D sharding the comms shrink as $1/\sqrt{N}$. We're still comms-bound, but total comms time keeps falling as we add chips, and for generation latency that's exactly what we want.
