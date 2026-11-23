---
title: Serving LLaMA 3 on TPUs
published: true
---
## 8. Serving LLaMA 3-70B

Chapter 7 gave us the inference formulas. This chapter points them at one real model (LLaMA 3-70B) on one real chip (TPU v5e) and asks what anyone serving a model actually wants to know: how cheap can we make a token, and how much latency does that cost us?

So why the v5e? For serving we basically want the most FLOPs per dollar. Using Google Cloud pricing as of February 2025:

| Chip | bf16 FLOPs/s | Price per chip-hour | bf16 FLOPs per dollar |
|---|---|---|---|
| H100 | 9.9e14 | \$10.80 | 3.3e17 |
| TPU v5p | 4.59e14 | \$4.20 | 3.9e17 |
| TPU v5e | 1.97e14 | \$1.20 | 5.9e17 |

FLOPs per dollar is just FLOPs/s times 3,600 seconds over the hourly price. The v5e is the slowest of the three and still buys about 1.5x as many FLOPs per dollar as a v5p and 1.8x as many as an H100. Cost per token is the hourly price divided by the tokens a chip makes in that hour, which is why the problems keep asking for throughput per chip. The catch is that each v5e only has 16GB of HBM, so a 70B model has to be spread over a lot of chips before it even fits.

### The Model and the Chip

| | LLaMA 3-8B | LLaMA 3-70B | LLaMA 3-405B |
|---|---|---|---|
| Layers ($L$) | 32 | 80 | 126 |
| Model dim ($D$) | 4,096 | 8,192 | 16,384 |
| Feed-forward dim ($F$) | 14,336 | 28,672 | 53,248 |
| Query heads ($N$) | 32 | 64 | 128 |
| KV heads ($K$) | 8 | 8 | 8 |
| Head dim ($H$) | 128 | 128 | 128 |
| Vocab ($V$) | 128,256 | 128,256 | 128,256 |
| Parameters | 8.03e9 | 70.55e9 | 405.8e9 |

The 70B column is the one we work through, and we round it to 70e9 parameters. The 8B and 405B columns are for the last few problems and come from each model's `config.json` on HuggingFace (405B is the same config as in Chapter 6). Parameters are counted the same way as in Chapter 6: $3DFL$ for the MLPs, $2DH(N + K)L$ for attention and $2VD$ for the embeddings.

For the TPU v5e, our references from Chapter 2 say:

| | TPU v5e |
|---|---|
| HBM per chip | 16GB |
| HBM bandwidth per chip | 8.2e11 bytes/s |
| bf16 FLOPs/s per chip | 1.97e14 |
| int8 OPs/s per chip | 3.94e14 |
| ICI bandwidth per link (bidirectional) | 9.0e10 bytes/s |
| Slices used here | 2x2, 4x2, 4x4, 4x8, 8x8, up to the 16x16 pod |

### The Method

Every problem below runs the same four steps.

1. **KV cache per token:** $2 \times \text{bytes per number} \times K \times H \times L$. For 70B in int8 that's 163,840 bytes, about 164kB for every token of context in every sequence.
2. **Total memory:** the weights plus $B \times S$ tokens of KV cache, which has to fit in 16GB times the chip count.
3. **Decode step time:** attention reads the whole KV cache on every step and is always memory-bound. The MLP side costs whichever is longer, doing the FLOPs or loading the weights:

   $$
   T_{\text{step}} = \frac{B \times \text{KV bytes per sequence}}{\text{chips} \times W_{\text{hbm}}} + \max\left(\frac{2 \times B \times \text{params}}{\text{chips} \times C}, \frac{\text{param bytes}}{\text{chips} \times W_{\text{hbm}}}\right)
   $$

4. **Throughput:** $B / T_{\text{step}}$ tokens/s, divided by the chip count. Tokens/s/chip is what sets the cost per token.

The $\max$ flips from weight loading to FLOPs at the critical batch size from Chapter 7: 240 with bf16 weights and bf16 FLOPs, 120 with int8 weights and bf16 FLOPs, 240 again with int8 everything. Below that, a decode step is just "read everything in HBM once".

Prefill is the easy half. A prompt of thousands of tokens is way past 240, so it's compute-bound and we only count FLOPs: $2 \times \text{params} \times \text{tokens}$, over chips, FLOPs/s and utilization. 8192 tokens on 16 chips at 40% takes 0.91 seconds, around 47 of the 19.5 ms decode steps we'll see below.

### On Sharding

During generation the only sharding that makes sense is model parallelism: we move small activations over ICI, never weights or KV caches. When we're compute-bound, the Chapter 5 rule caps it at $Y \lesssim M_Y \cdot F / 2200$ (2200 is the v5e's bf16 FLOPs/s over its ICI bandwidth, and $M_Y$ is how many mesh axes we use). For 70B on both axes that's about 26-way, so a 4x4 is fine and a 4x8 isn't.

At small batch sizes we're waiting on weight loading anyway, and ICI only has to beat HBM. So we can shard further to cut latency, up to roughly $Y = F / (\beta B)$ with $\beta = W_{\text{hbm}} / W_{\text{ici}} \approx 9$ from our reference numbers.

### The Latency vs Throughput Tradeoff

Batch size is the knob. Take int8 weights with bf16 FLOPs on a 4x4 and run the step formula:

- **Batch 1:** we pay the full 5.3 ms of weight loading for a single token. Around 5.4 ms per step and 12 tokens/s/chip. Fast, and insanely wasteful.
- **Batch 120 at 2k context:** 8.4 ms per step and about 890 tokens/s/chip. Roughly 1.6x the latency for 77x the throughput.
- **Past 120:** FLOPs and KV loading both grow with the batch, so latency keeps climbing while tokens/s/chip sits flat. Nothing left to gain.
- **Longer context:** at 8k only 138 sequences fit in HBM, a step takes 20 ms (14.1 ms of KV loading plus 6.1 ms of FLOPs), and throughput tops out at 426 tokens/s/chip. At 32k only 34 fit, so we never even reach batch 120.

At 8k context, KV loading overtakes weight loading once the batch passes about 52, and it's slower than the FLOPs for any context past roughly 3,600 tokens. So at long context the KV cache basically IS the step time. You can find interactive plots of the Pareto curve and the time breakdown at [jax-ml.github.io/scaling-book/applied-inference/](https://jax-ml.github.io/scaling-book/applied-inference/).

### Takeaway

For LLaMA 3-70B on v5e, generation is memory-bound almost everywhere we can afford to run it. The numbers worth remembering:

- The KV cache is 164kB per token in int8. One 8k sequence is 1.34GB, and a batch of 32 is 43GB next to 70GB of weights.
- While we're memory-bound, a decode step is the bytes in HBM over the HBM bandwidth: 17.2 ms on a 4x2 at batch 32 (232 tokens/s/chip), and about 19.5 ms on any slice whose HBM is full. Past the critical batch size, FLOPs replace weight loading and the step gets a bit longer.
- The smallest slices that hold the weights (4x4 in bf16, 4x2 in int8, 2x2 in int4) only leave room for 43 sequences. Doubling the slice fits 138 and gets up to 1.6x more queries out of every chip (1.5x for int8 and 1.2x for int4 if the FLOPs stay in bf16).
- An 8k prefill takes 0.91 s on 16 chips, and with 512-token responses it takes about four prefill servers to keep one generate server busy.

The floor for any decode step is loading the weights out of HBM once. At long context, nearly everything above that floor is KV cache.

## Worked Problems

The first twelve are the questions sprinkled through the section, numbered here in order. Questions 13 to 16 are the end-of-section problems 1 to 4, which don't come with published answers.

**Question 1 [KV cache per token]:** How big is LLaMA 3-70B's KV cache per token if we store it in int8? (This is the number that decides how big a batch fits on a given topology.)

- Just the KV cache formula from Chapter 7: every token keeps a key and a value (the 2) for each of the $K$ KV heads, $H$ numbers wide, in all $L$ layers, at 1 byte per number in int8:

  $$
  2 \times K \times H \times L = 2 \times 8 \times 128 \times 80 = \mathbf{163{,}840 \text{ bytes}}
  $$

- So every token of context costs us about **164kB** per sequence. That adds up fast: a batch of 240 at a 32,768 token context is about 1.29TB, roughly 18 times the weights.
- The solution rounds this to 160kB, so its memory totals come out slightly under ours.

**Question 2 [memory to serve]:** Say we serve LLaMA 3-70B at batch size 32 and sequence length 8192, with everything (weights and KV caches) in int8. How much memory is that in total, and what's the smallest slice we could serve it on?

- Counting bytes, KV caches first, with the 163,840 bytes per token from Question 1:

  $$
  163{,}840 \times 8192 \times 32 = 42{,}949{,}672{,}960 \text{ bytes} \approx 42.95 \text{ GB}
  $$

- The weights are 1 byte per parameter, so 70GB. Together:

  $$
  \text{Total} = 42.95 \text{ GB} + 70 \text{ GB} = \mathbf{112.95 \text{ GB}}
  $$

- Each v5e holds 16GB, so $112.95 / 16 = 7.06$ chips, which rounds up to a **4x2** slice (8 chips, 128GB).
- So about 113GB on a 4x2, which is 88% full. Tight enough that real overhead could push us to a 4x4 or a smaller batch.

**Question 3 [decode latency and throughput]:** At that batch size and quantization on a TPU v5e 4x2, roughly what latency should we expect per decode step, and what throughput in tokens/s/chip? What about on a 4x4? Assume the FLOPs are done in bfloat16 and everything is fully sharded.

- First, which regime are we in? With int8 weights and bf16 FLOPs the critical batch size is 120 (Question 4) and we're at 32, so we're memory-bound: on 8 chips the FLOPs take 2.84 ms but loading the weights takes 10.67 ms.
- So the step is just every byte in HBM over the total HBM bandwidth:

  $$
  T_{\text{step}} = \frac{112.95 \times 10^9 \text{ bytes}}{8 \times 8.2 \times 10^{11} \text{ bytes/s}} = \mathbf{17.2 \text{ ms}}
  $$

- Each step makes one token for each of the 32 sequences:

  $$
  \frac{32 \text{ tokens}}{17.22 \text{ ms}} \approx 1{,}859 \text{ tokens/s} \implies \frac{1{,}859}{8 \text{ chips}} = \mathbf{232 \text{ tokens/s/chip}}
  $$

- On a 4x4 the bandwidth doubles, so the step drops to **8.6 ms**. Throughput is 3,717 tokens/s over 16 chips, the same 232 tokens/s/chip.
- Meaning, moving the same batch to a 4x4 halves the latency but doesn't make a token any cheaper.

**Question 4 [compute-bound batch size]:** On TPU v5e, how large does the batch need to be for our matmuls to be compute-bound if we use bfloat16 weights and activations? What about int8 weights with the FLOPs done in bfloat16? And int8 weights with int8 FLOPs?

- Same inequality as Chapters 1 and 7. With $B \ll D, F$ the weight matrix is basically all the bytes, so we compare $2BDF$ FLOPs against one $[D, F]$ matrix.
- **bf16 weights, bf16 FLOPs:** The weights are $2DF$ bytes.

  $$
  \frac{2BDF}{2DF} = B > \frac{1.97 \times 10^{14} \text{ FLOPs/s}}{8.2 \times 10^{11} \text{ bytes/s}} \implies \mathbf{B > 240}
  $$

- **int8 weights, bf16 FLOPs:** The weights shrink to $DF$ bytes and the FLOPs don't change.

  $$
  \frac{2BDF}{DF} = 2B > 240 \implies \mathbf{B > 120}
  $$

- **int8 weights, int8 FLOPs:** Now the chip does $3.94 \times 10^{14}$ OPs/s, so the right side doubles to 480.

  $$
  2B > \frac{3.94 \times 10^{14}}{8.2 \times 10^{11}} = 480 \implies \mathbf{B > 240}
  $$

- So 240, 120 and 240 concurrent sequences. The middle one is worth remembering: shrinking only what sits in HBM halves the batch we need.

**Question 5 [smallest topology]:** With an 8k context, what's the smallest TPU v5e topology that can serve LLaMA 3-70B in bfloat16, in int8 and in int4 (KVs and parameters both)? Treat the KV caches as negligibly small for this one.

- With the KV caches ignored, only the weights have to fit: 70e9 parameters times the bytes per parameter, over 16GB, rounded up to a slice shape that exists.

| dtype | Weights | Chips needed | Smallest slice | HBM left over | KV cache per 8k sequence | Sequences that fit |
|---|---|---|---|---|---|---|
| bf16 | 140GB | 8.75 | 4x4 (16 chips) | 116GB | 2.68GB | 43 |
| int8 | 70GB | 4.38 | 4x2 (8 chips) | 58GB | 1.34GB | 43 |
| int4 | 35GB | 2.19 | 2x2 (4 chips) | 29GB | 0.67GB | 43 |

- So **4x4 for bf16, 4x2 for int8 and 2x2 for int4**. We really can put LLaMA 3-70B on four v5e chips.
- The last three columns are why we wouldn't want to: the leftover HBM fits the same 43 sequences every time, since weights, caches and slice all scale together (barely: 116GB holds 43.2 of the 2.68GB caches). 43 is our max batch size, way under the 120 or 240 we'd want.

**Question 6 [latency at max batch]:** Say we fill each of these topologies with the largest batch that fits. What latency should we expect per generate step?

- This is really just HBM capacity over HBM bandwidth. Batch 43 leaves HBM 99.8% full and is under the critical batch size in all three cases, so each step reads all of HBM once. Every chip does that in parallel, so the slice size cancels out:

  $$
  T_{\text{step}} \approx \frac{16 \times 10^9 \text{ bytes}}{8.2 \times 10^{11} \text{ bytes/s}} = \mathbf{19.5 \text{ ms}}
  $$

- The full step formula at batch 43 gives 19.47 ms for all three (8.80 ms of KV cache plus 10.67 ms of weights), so around **19.5 ms per step** whichever we pick. A median 512 token response takes $512 \times 19.47 \text{ ms} \approx 10$ seconds.
- The solution rounds the step to 19 ms, which is why its throughput numbers in Questions 7 and 8 come out a little higher than ours.

**Question 7 [throughput per chip]:** What does that work out to in throughput per chip for each of the three, counted in queries per second per chip? Assume a median decode length of 512 tokens.

- This is the cost question. Each slice runs 43 queries at once, and each query needs 512 steps of 19.47 ms:

  $$
  \text{QPS per chip} = \frac{B}{T_{\text{step}} \times 512 \times \text{chips}} = \frac{43}{0.01947 \text{ s} \times 512 \times \text{chips}} \approx \frac{4.31}{\text{chips}}
  $$

| dtype | Slice | QPS per chip |
|---|---|---|
| bf16 | 4x4 (16 chips) | **0.27** |
| int8 | 4x2 (8 chips) | **0.54** |
| int4 | 2x2 (4 chips) | **1.08** |

- Every slice finishes the same 4.31 queries per second, so halving the bytes per number doubles the queries per chip.
- These are optimistic, since we've left the forward pass zero working memory. The solution guesses the real numbers are around half.

**Question 8 [doubling the topology]:** What happens to the peak throughput if we double the topology in each of the three cases?

- The weights don't grow, so every byte of new HBM goes to KV caches. For bf16 on a 4x8 that's $512 - 140 = 372\text{ GB}$, and at 2.68GB per sequence:

  $$
  \frac{372 \text{ GB}}{2.684 \text{ GB}} = 138.6 \implies \mathbf{B = 138}
  $$

- Same for int8 on a 4x4 (186GB left) and int4 on a 2x4 (8 chips, 93GB left). So twice the chips holds 3.2x the batch.
- HBM is full again, so if a step is still one full read of HBM it's 19.45 ms, and the slice finishes $138 / (0.01945 \times 512) = 13.86$ queries per second.
- That only holds while 138 is under the critical batch size. With bf16 FLOPs (like Question 3), 138 is under the 240 for bf16 weights but past the 120 for int8 and the 60 for int4, so for those two the $\max$ picks FLOPs. On top of 14.12 ms of KV loading, the MLP goes from 5.34 ms of weight loading to 6.13 ms of FLOPs for int8 (20.2 ms per step) and 12.26 ms for int4 (26.4 ms).
- QPS per chip both ways:

| dtype | Slice | Batch | Step = one HBM read | Full formula, bf16 FLOPs |
|---|---|---|---|---|
| bf16 | 4x8 (32 chips) | 138 | **0.43** | 0.43 |
| int8 | 4x4 (16 chips) | 138 | **0.87** | 0.83 |
| int4 | 2x4 (8 chips) | 138 | **1.73** | 1.28 |

- Against Question 7, doubling the slice gets about 1.6x more queries out of every chip while we stay memory-bound. With bf16 arithmetic, int8 and int4 are compute-bound at this batch, so their gains shrink to 1.5x and 1.2x. The int8 row is the same ceiling as in the notes: $0.832 \times 512 = 426$ tokens/s/chip.

> Fitting the model on the fewest chips doesn't give us the cheapest tokens. While KV cache memory is what caps the batch, adding chips makes every chip more productive.

**Question 9 [sharding on a 4x8]:** Now for the sharding. We're serving in bfloat16 on a TPU v5e 4x8. How should the model be sharded during generation, and can we keep from being communication bound?

- Generation leaves us one choice, model parallelism. FSDP would drag weights over ICI every step, data parallelism just replicates them, and with one new token per step there's no sequence to shard.
- How far can it go? If we're compute-bound, the rule from the notes with both mesh axes ($M_Y = 2$) gives:

  $$
  Y \lesssim \frac{M_Y \cdot F}{2200} = \frac{2 \times 28{,}672}{2200} \approx 26
  $$

- So 16-way on a 4x4 works and 32-way on a 4x8 doesn't, if we're compute-bound. But at a small batch we're waiting on weights, and ICI only has to beat HBM. For one MLP matmul sharded $Y$ ways:

  $$
  T_{\text{ICI}} = \frac{2BD}{W_{\text{ici}}} \qquad T_{\text{HBM}} = \frac{2DF}{Y \cdot W_{\text{hbm}}} \qquad T_{\text{math}} = \frac{2BDF}{Y \cdot C}
  $$

- On the 4x8 ($Y = 32$) at $B = 64$ these come to $11.7 \ \mu\text{s}$, $17.9 \ \mu\text{s}$ and $4.8 \ \mu\text{s}$. HBM is still the slowest, so the ICI traffic hides under the weight loading.
- Break-even is at $B = F / (Y \beta) = 28{,}672 / (32 \times 9.11) \approx 98$. At the full batch of 138 from Question 8, $T_{\text{ICI}}$ is $25.1 \ \mu\text{s}$ against the same $17.9 \ \mu\text{s}$ of HBM, so we're communication bound.
- So: pure model parallelism, and on a 4x8 it only stays clear of the comms bound **below a batch of about 98**. That's a latency play, and it means the 4x8 can't deliver Question 8's peak throughput in bf16. The int8 (4x4) and int4 (2x4) configs are under the 26-way limit, so they're fine.
- The 98 assumes the activations cross one mesh axis. If they get both axes like in the compute-bound rule, $T_{\text{ICI}}$ halves and the break-even doubles to about 197.

**Question 10 [prefill time]:** How long does it take to prefill an 8192 token prompt on 16 TPU v5e chips, if prefill reaches 40% FLOPs utilization?

- 8192 tokens is way past the critical batch size of 240, so prefill is compute-bound and we only count FLOPs. A forward pass is 2 FLOPs per parameter per token:

  $$
  2 \times 70 \times 10^9 \times 8192 \approx 1.147 \times 10^{15} \text{ FLOPs}
  $$

- 16 chips at 40% utilization give us $16 \times 1.97 \times 10^{14} \times 0.4 \approx 1.26 \times 10^{15}$ FLOPs/s, so:

  $$
  T_{\text{prefill}} = \frac{1.147 \times 10^{15} \text{ FLOPs}}{1.26 \times 10^{15} \text{ FLOPs/s}} = \mathbf{0.91 \text{ s}}
  $$

- It takes about **0.91 seconds** to prefill 8192 tokens. That's roughly 47 decode steps' worth of time, and most of what the user waits for the first token.

**Question 11 [KV cache evictions]:** Our median prefill is 8192 tokens, our median decode is 4096 tokens, and the generate batch size is 32. On an average step, how many sequences finish decoding, and how many tokens get evicted from the KV cache?

- Each of the 32 batch slots frees up once every 4096 steps, so in steady state:

  $$
  \frac{32}{4096} \approx \mathbf{0.0078 \text{ sequences per step}}
  $$

- That's one sequence finishing every 128 steps.
- When a sequence leaves, its whole cache goes with it, prefill and decode: $8192 + 4096 = 12{,}288$ tokens.

  $$
  \frac{32}{4096} \times 12{,}288 = \mathbf{96 \text{ tokens per step}}
  $$

- In general it's $B(P + G)/G$ for prefill length $P$ and decode length $G$. Sanity check: each step adds 32 decode tokens plus $32 / 4096 \times 8192 = 64$ prefill tokens on average. 96 in, 96 out.

**Question 12 [disaggregated serving]:** Now we split prefill and generation onto separate servers (disaggregated serving). The median prefill is 8192 tokens, the median decode is 512 tokens, and we use the bfloat16 prefill and generate latencies from the questions above. How many prefill servers do we need per generate server so that both kinds stay fully saturated?

- We just need two rates to match: sequences out of prefill and sequences finished by generate.
- A prefill server (16 chips, Question 10) handles one prompt every 0.91 s, so 1.10 sequences per second.
- A generate server (the bf16 4x4 from Question 6) holds 43 sequences, and each one needs $512 \times 19.47 \text{ ms} = 9.97\text{ s}$ of steps, so it finishes $43 / 9.97 = 4.31$ sequences per second.
- Matching $P$ prefill servers against $G$ generate servers:

  $$
  P \times 1.10 = G \times 4.31 \implies \frac{P}{G} \approx \mathbf{3.9}
  $$

- So we need about **4 prefill servers for every generate server**. Both kinds are 16 chips here, meaning roughly 80% of the chips are doing prefill.
- The solution gets 3 to 1 because it uses a batch of 32 and 19 ms steps.

**Question 13 [LLaMA 3-405B forward pass]:** How many FLOPs per token does one forward pass of LLaMA 3-405B take? If we're FLOPs bound, what's a lower bound on the time for a single forward pass on $N$ TPU v5e chips? And if we're comms bound instead? (We're told to ignore that the model won't fit on one chip.)

- Parameter count first, from the 405B column: MLPs $3DFL \approx 329.8 \times 10^9$, attention $2DH(N_q + K)L \approx 71.9 \times 10^9$ (with $N_q = 128$ query heads, since $N$ means chips here), and embeddings $2VD \approx 4.2 \times 10^9$. Total: about $405.8 \times 10^9$.
- **How many FLOPs per token?** A forward pass is 2 FLOPs per parameter per token:

  $$
  2 \times 405.8 \times 10^9 = \mathbf{8.12 \times 10^{11} \text{ FLOPs per token}}
  $$

  This skips the attention dot products, about 8% extra at an 8192 context.
- **FLOPs bound on $N$ chips:** best case, every chip is busy and nothing waits on communication:

  $$
  T \ge \frac{2 \times \text{params} \times B}{N \times C} = \frac{8.12 \times 10^{11} \times B}{N \times 1.97 \times 10^{14}} = \mathbf{4.12 \text{ ms} \times \frac{B}{N}}
  $$

  So about 4.1 ms per token on one chip.
- **Comms bound:** We'll take "comms" to mean HBM traffic here. Then the pass can't finish faster than every weight can be streamed out of HBM once:

  $$
  T \ge \frac{\text{param bytes}}{N \times W_{\text{hbm}}} = \frac{811.7 \times 10^9 \text{ bytes}}{N \times 8.2 \times 10^{11} \text{ bytes/s}} = \mathbf{\frac{0.99 \text{ s}}{N}} \text{ in bf16}
  $$

  With int8 weights it's half that, $0.495 \text{ s} / N$, and it doesn't depend on the batch at all.
- The real lower bound is whichever is larger, and they cross at $B = 240$ in bf16 (or 120 with int8 weights), the same critical batch size as always.
- No published answer here. "Comms bound" could also mean ICI, but that needs a sharding (like Question 9) before it gives a number.

**Question 14 [serving LLaMA 3-8B]:** We serve LLaMA 3-8B at batch size 240, with the weights and the KV caches both in int8. How many bytes go to (a) the model parameters, (b) the KV caches and (c) the peak working activations, roughly? And what's the smallest topology this runs on? No context length is given, so we assume the 8192 used in the rest of the chapter.

- **(a) Model parameters:** From the 8B column, the MLPs are $3DFL \approx 5.64 \times 10^9$, attention is $2DH(N + K)L \approx 1.34 \times 10^9$ and the embeddings are $2VD \approx 1.05 \times 10^9$. That's $8.03 \times 10^9$ parameters, so **8.03GB** in int8.
- **(b) KV caches:** Per token it's $2 \times 8 \times 128 \times 32 = 65{,}536$ bytes. One 8192 token sequence is $65{,}536 \times 8192 = 536{,}870{,}912$ bytes, and 240 of them:

  $$
  240 \times 536{,}870{,}912 \text{ bytes} \approx \mathbf{128.8 \text{ GB}}
  $$

  That's 16 times the weights.
- **(c) Peak working activations:** A decode step only pushes 240 tokens through, one per sequence. In bf16 the residual stream `bf16[240, 4096]` is 2MB and an MLP hidden activation `bf16[240, 14336]` is 6.9MB. The biggest are the logits `bf16[240, 128256]` at 61.6MB and, if we materialize them (Flash Attention doesn't), the attention scores `bf16[240, 32, 8192]` at 125.8MB. So **roughly 0.2GB** at worst, a rounding error next to (b).
- **Smallest topology:** The total is $8.03 + 128.8 + 0.2 \approx 137.1\text{ GB}$, or 8.57 chips' worth. A 4x2 only has 128GB, less than the KV caches alone, so we need a **4x4** (16 chips, 256GB), which ends up 54% full.
- It's a near miss, though: at 4096 tokens of context the total is about 72.5GB and a 4x2 is plenty, so the topology really hangs on the context length.
- What this means: on the 4x4 a step is 9.82 ms of KV loading plus 1.22 ms of FLOPs, about 11.0 ms, so 89% of the step is reading KV caches. A small model at a big batch is basically a KV cache server.

**Question 15 [serving LLaMA 3-405B]:** How should we serve LLaMA 3-405B on TPU v5e, with int8 weights and bfloat16 FLOPs? Given a hard limit of 15 ms per token, which configuration gets the highest throughput, and what's the theoretical minimum step time? Again there's no context length given, so we keep 8192.

- Bytes first. The weights are 405.8GB in int8, 25.4 chips' worth, so a 4x8 (32 chips) is the smallest slice that holds them. The KV cache is $2 \times 8 \times 128 \times 126 = 258{,}048$ bytes per token, or 2.11GB per 8192 token sequence.
- **Theoretical minimum step time:** the floor is reading the weights out of HBM once, at batch 1 with basically no KV cache:

  $$
  T_{\text{min}} = \frac{405.8 \times 10^9 \text{ bytes}}{\text{chips} \times 8.2 \times 10^{11} \text{ bytes/s}} \approx \frac{495 \text{ ms}}{\text{chips}}
  $$

| Slice | Chips | Weight loading time | HBM left for KV | 8k sequences that fit |
|---|---|---|---|---|
| 4x8 | 32 | 15.5 ms | 106GB | 50 |
| 8x8 | 64 | 7.7 ms | 618GB | 292 |
| 8x16 | 128 | 3.9 ms | 1,642GB | 776 |
| 16x16 | 256 | 1.9 ms | 3,690GB | 1,745 |

- The 4x8 that fits the model takes 15.5 ms just to read its weights, so it misses the 15 ms limit even at batch 1. The next slice up is an **8x8, with a minimum step time of 7.7 ms**. The absolute floor is 1.9 ms on a full pod.
- **Highest throughput under 15 ms:** Easiest to think per chip. For every sequence per chip ($B / \text{chips} = 1$), KV loading costs $2.114 \times 10^9 / (8.2 \times 10^{11}) = 2.58\text{ ms}$ and the FLOPs cost $2 \times 405.8 \times 10^9 / (1.97 \times 10^{14}) = 4.12\text{ ms}$.
- Once the batch is past the critical 120, the step formula is just those two added up:

  $$
  T_{\text{step}} = 6.70 \text{ ms} \times \frac{B}{\text{chips}} \implies \frac{B}{T_{\text{step}} \times \text{chips}} = \frac{1}{6.70 \text{ ms}} \approx \mathbf{149 \text{ tokens/s/chip}}
  $$

- That's the ceiling, and it's flat: past batch 120, more batch only adds latency. The 15 ms limit allows $B / \text{chips} \le 15 / 6.70 = 2.24$, and we need $B \ge 120$ to reach the ceiling, so at least 54 chips. The 8x8 again.
- On the 8x8 at $B = 120$, a step is 4.83 ms of KV loading plus 7.73 ms for the MLP (120 is right at the crossover, so weight loading and FLOPs take the same time), **12.6 ms** in total, for about 9,550 tokens/s in 660GB of the 1,024GB of HBM.
- The largest batch that still meets the limit is $B = 143$: 5.76 ms of KV loading plus 9.21 ms of FLOPs is **14.97 ms**, for 9,555 tokens/s in 708GB. At 144 it's 15.07 ms. So everything from 120 to 143 sits on the same plateau, and the extra 23 sequences cost 2.4 ms per token for nothing.
- Sharding: pure model parallelism again, weights split 64 ways over both axes, KV cache split over its 8 KV heads on one axis and the batch on the other (15 sequences per shard at batch 120).
- So the roofline answer: **an 8x8 slice at 149 tokens/s/chip, about 9,550 tokens/s in total, at any batch from about 120 (12.6 ms per token) up to 143 (15.0 ms)**. Bigger slices don't beat that per chip (an 8x16 at batch 286 gets the same 149), they just serve more users.
- The catch is ICI. 64-way is past this model's compute-bound limit of $2F / 2200 \approx 48$. For one MLP matmul at $B = 120$ on the 8x8, $T_{\text{ICI}}$ is $43.7 \ \mu\text{s}$ on one axis ($21.8 \ \mu\text{s}$ with both) against $16.6 \ \mu\text{s}$ for both the HBM read and the math. ICI only hides under weight loading up to a batch of about 91 (two axes), where the roofline gives 11.4 ms and 125 tokens/s/chip.
- Meaning, the realistic answer is an 8x8 at a batch between 91 and 120, somewhere between 125 and 149 tokens/s/chip depending on how well the ICI traffic overlaps.

**TODO(eddie):** no published answer for this one. Check the 8192 context assumption and how much ICI overhead to charge (149 tokens/s/chip is the roofline with ICI ignored). Also decide whether to headline batch 120 (lowest latency on the plateau) or 143 (largest batch under 15 ms); the roofline throughput is the same.

**Question 16 [implement it yourself]:** The last problem is a build exercise: turn the LLaMA 3-8B weights from [HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/) into a working inference implementation. (1) Load the weights in Colab, inspect every tensor ([TreeScope](https://github.com/google-deepmind/treescope) is suggested), count the parameters and check the MLP vs attention split. (2) Write the full forward pass and sample greedily one token at a time, ideally from the paper alone, falling back on [jax-llm-examples](https://github.com/jax-ml/jax-llm-examples/tree/main/llama3) for details. This fits on a single TPU with more than 16GiB of HBM. (3) Add KV caching. (4) Split it into a prefill server that takes one prompt and a batched decode server. (5) Write Flash Attention in Pallas, preferably without a reference.

No napkin math for this one, so here's what we should expect to see, in build order:

- **Loading and counting:** From the config we expect $3DFL = 5{,}637{,}144{,}576$ parameters in the MLPs (70%), $2DH(N + K)L = 1{,}342{,}177{,}280$ in attention (17%) and $2VD = 1{,}050{,}673{,}152$ in the two embedding tables (13%), plus 266,240 RMSNorm scales for 8,030,261,248 in total. Per layer: query and output projections `[4096, 4096]`, key and value `[1024, 4096]` (8 KV heads of 128), and MLP matrices `[14336, 4096]`, `[14336, 4096]` and `[4096, 14336]`. In bf16 that's about 16.1GB (15.0GiB) of weights, hence the 16GiB hint.
- **Forward pass:** token embedding, then 32 blocks of RMSNorm, attention, residual add, RMSNorm, SwiGLU MLP, residual add, then a final RMSNorm and the projection to 128,256 logits. The easy places to slip are the rotary position embeddings (applied to queries and keys, not added to the residual) and the causal mask. Each KV head is shared by 4 query heads.
- **KV caching:** keep each layer's keys and values as `bf16[B, S, 8, 128]`, run only the newest token through the model, append its keys and values, and attend over the cache. In bf16 that's 131,072 bytes per token, about 1.07GB for a full 8192 token sequence. Without the cache, step $t$ re-runs all $t$ tokens through every MLP; with it, each step runs one.
- **Prefill and decode servers:** prefill runs one prompt at batch 1 and returns its KV cache. The decode side holds a fixed number of batch slots, drops each finished prefill into a free slot, and steps all the active slots together. This is the continuous batching setup from Chapter 7 (JetStream's prefill, insert and generate).
- **Flash Attention in Pallas:** walk the keys and values in blocks, keeping a running max and a running normalizer for the softmax, so the full `[T, S]` score matrix never gets written to HBM.

**TODO(eddie):** actually do this in Colab and paste in the measured parameter count, a sample of generated text, and the tokens/s before and after KV caching.
