---
title: Serving LLaMA 3 on TPUs
published: true
---
## 8. Serving LLaMA 3-70B

Chapter 7 gave us the inference formulas. Now we point them at one real model (LLaMA 3-70B) on one real chip (TPU v5e) and ask what serving actually cares about: how cheap can we make a token, and how much latency does that cost?

## What's the LLaMA Serving Story?

So why the v5e? For serving we basically want the most FLOPs per dollar. On Google Cloud pricing as of February 2025:

- **H100:** 9.9e14 bf16 FLOPs/s at \$10.80 per chip-hour, so 3.3e17 FLOPs per dollar.
- **TPU v5p:** 4.59e14 at \$4.20, so 3.9e17.
- **TPU v5e:** 1.97e14 at \$1.20, so 5.9e17.

That's just FLOPs/s times 3,600 seconds over the hourly price. The v5e is the slowest chip and still buys about 1.5x the FLOPs per dollar of a v5p and 1.8x of an H100. Cost per token is the hourly price over the tokens a chip makes in that hour, so tokens/s/chip is the number we chase. The catch is that a v5e only has 16GB of HBM, so a 70B model has to spread over a lot of chips before it even fits.

LLaMA 3-70B has $L = 80$ layers, $D = 8{,}192$, $F = 28{,}672$, $N = 64$ query heads and $K = 8$ KV heads with $H = 128$, and a vocab of $V = 128{,}256$. Counting like Chapter 6 ($3DFL$ for the MLPs, $2DH(N + K)L$ for attention, $2VD$ for the embeddings) gives 70.55e9 parameters, which we round to 70e9. Our v5e numbers from Chapter 2: 16GB of HBM at 8.2e11 bytes/s, 1.97e14 bf16 FLOPs/s, 3.94e14 int8 OPs/s and 9.0e10 bytes/s of bidirectional ICI per link, on slices from 2x2 up to the 16x16 pod.

Every problem below runs the same four steps:

1. **KV cache per token:** $2 \times \text{bytes per number} \times K \times H \times L$. For 70B in int8 that's 163,840 bytes, about 164kB for every token of context in every sequence.
2. **Total memory:** the weights plus $B \times S$ tokens of KV cache, which has to fit in 16GB times the chip count.
3. **Decode step time:** attention reads the whole KV cache every step and is always memory-bound. The MLP costs whichever is longer, doing the FLOPs or loading the weights:

   $$
   T_{\text{step}} = \frac{B \times \text{KV bytes per sequence}}{\text{chips} \times W_{\text{hbm}}} + \max\left(\frac{2 \times B \times \text{params}}{\text{chips} \times C}, \frac{\text{param bytes}}{\text{chips} \times W_{\text{hbm}}}\right)
   $$

4. **Throughput:** $B / T_{\text{step}}$ tokens/s, divided by the chip count.

### Thinking about throughput

The $\max$ flips from weight loading to FLOPs at the critical batch size from Chapter 7: 240 with bf16 weights and FLOPs, 120 with int8 weights and bf16 FLOPs, 240 again with int8 everything. Below that, a decode step is just "read everything in HBM once", about 19.5 ms on any slice whose HBM is full.

The smallest slices that hold the weights (4x4 in bf16, 4x2 in int8, 2x2 in int4) only leave room for 43 sequences at 8k context, way under 120 or 240. Doubling the slice fits 138 and gets up to 1.6x more queries out of every chip. So fitting the model on the fewest chips doesn't give us the cheapest tokens; while KV cache memory caps the batch, adding chips makes every chip more productive.

For sharding, generation leaves us model parallelism: we move small activations over ICI, never weights or KV caches. When we're compute-bound, the Chapter 5 rule caps it at $Y \lesssim M_Y \cdot F / 2200$ (2200 is the v5e's FLOPs/s over its ICI bandwidth, and $M_Y$ is how many mesh axes we use). For 70B on both axes that's about 26-way, so a 4x4 is fine and a 4x8 isn't. At a small batch we're waiting on weights anyway, so ICI only has to beat HBM and we can go up to roughly $Y = F / (\beta B)$, with $\beta = W_{\text{hbm}} / W_{\text{ici}} \approx 9$.

### What about prefill?

Prefill is the easy half. A prompt of thousands of tokens is way past 240, so it's compute-bound and we only count FLOPs: $2 \times \text{params} \times \text{tokens}$, over chips, FLOPs/s and utilization. 8192 tokens on 16 chips at 40% takes 0.91 seconds, around 47 of the 19.5 ms decode steps. With 512-token responses it takes about four prefill servers to keep one generate server busy.

## Visualizing the Latency Throughput Tradeoff

Batch size is the knob. Take int8 weights with bf16 FLOPs on a 4x4:

- **Batch 1:** we pay the full 5.3 ms of weight loading for a single token. Around 5.4 ms per step and 12 tokens/s/chip. Fast, and insanely wasteful.
- **Batch 120 at 2k context:** 8.4 ms per step and about 890 tokens/s/chip. Roughly 1.6x the latency for 77x the throughput.
- **Past 120:** FLOPs and KV loading both grow with the batch, so latency keeps climbing while tokens/s/chip sits flat.

Longer context makes it worse. At 8k only 138 sequences fit, a step takes 20 ms (14.1 ms of KV loading plus 6.1 ms of FLOPs), and throughput tops out at 426 tokens/s/chip. At 32k only 34 fit, so we never even reach batch 120. KV loading overtakes weight loading once the 8k batch passes about 52, and it's slower than the FLOPs for any context past roughly 3,600 tokens. The interactive Pareto plots are at [jax-ml.github.io/scaling-book/applied-inference/](https://jax-ml.github.io/scaling-book/applied-inference/).

So the floor for any decode step is loading the weights out of HBM once, and at long context nearly everything above that floor is KV cache.

## Worked Problems

Questions 13 to 16 are the end-of-section problems, which don't come with published answers.

**Question 1 [KV cache per token]:** How big is LLaMA 3-70B's KV cache per token if we store it in int8?

- Every token keeps a key and a value (the 2) for each of the $K$ KV heads, $H$ numbers wide, in all $L$ layers, at 1 byte per number:

  $$
  2 \times K \times H \times L = 2 \times 8 \times 128 \times 80 = \mathbf{163{,}840 \text{ bytes}}
  $$

- So every token of context costs about **164kB** per sequence.

**Question 2 [memory to serve]:** Say we serve LLaMA 3-70B at batch size 32 and sequence length 8192, with everything (weights and KV caches) in int8. How much memory is that in total, and what's the smallest slice we could serve it on?

- KV caches first, with the 163,840 bytes per token from Question 1:

  $$
  163{,}840 \times 8192 \times 32 = 42{,}949{,}672{,}960 \text{ bytes} \approx 42.95 \text{ GB}
  $$

- The weights are 70GB in int8, so the total is $42.95 + 70 = \mathbf{112.95 \text{ GB}}$.
- $112.95 / 16 = 7.06$ chips, which rounds up to a **4x2** slice (8 chips, 128GB), 88% full.

**Question 3 [decode latency and throughput]:** At that batch size and quantization on a TPU v5e 4x2, roughly what latency should we expect per decode step, and what throughput in tokens/s/chip? What about on a 4x4? Assume the FLOPs are done in bfloat16 and everything is fully sharded.

- Batch 32 is under the critical 120 for int8 weights with bf16 FLOPs (Question 4), so we're memory-bound: on 8 chips the FLOPs take 2.84 ms but loading the weights takes 10.67 ms. The step is every byte in HBM over the total HBM bandwidth:

  $$
  T_{\text{step}} = \frac{112.95 \times 10^9 \text{ bytes}}{8 \times 8.2 \times 10^{11} \text{ bytes/s}} = \mathbf{17.2 \text{ ms}}
  $$

  $$
  \frac{32 \text{ tokens}}{17.22 \text{ ms}} \approx 1{,}859 \text{ tokens/s} \implies \frac{1{,}859}{8 \text{ chips}} = \mathbf{232 \text{ tokens/s/chip}}
  $$

- On a 4x4 the bandwidth doubles, so the step drops to **8.6 ms** at the same 232 tokens/s/chip. Half the latency, but no cheaper per token.

**Question 4 [compute-bound batch size]:** On TPU v5e, how large does the batch need to be for our matmuls to be compute-bound if we use bfloat16 weights and activations? What about int8 weights with the FLOPs done in bfloat16? And int8 weights with int8 FLOPs?

- With $B \ll D, F$ the weights are basically all the bytes, so we compare $2BDF$ FLOPs against one $[D, F]$ matrix.
- **bf16 weights, bf16 FLOPs:** the weights are $2DF$ bytes.

  $$
  \frac{2BDF}{2DF} = B > \frac{1.97 \times 10^{14} \text{ FLOPs/s}}{8.2 \times 10^{11} \text{ bytes/s}} \implies \mathbf{B > 240}
  $$

- **int8 weights, bf16 FLOPs:** the weights shrink to $DF$ bytes, so $2B > 240 \implies \mathbf{B > 120}$.
- **int8 weights, int8 FLOPs:** the chip now does $3.94 \times 10^{14}$ OPs/s, so $2B > 480 \implies \mathbf{B > 240}$.
- Shrinking only what sits in HBM halves the batch we need.

**Question 5 [smallest topology]:** With an 8k context, what's the smallest TPU v5e topology that can serve LLaMA 3-70B in bfloat16, in int8 and in int4 (KVs and parameters both)? Treat the KV caches as negligibly small for this one.

- Only the weights have to fit: 70e9 parameters times the bytes per parameter, over 16GB, rounded up to a real slice.
- bf16 is 140GB, or 8.75 chips. int8 is 70GB, 4.38 chips. int4 is 35GB, 2.19 chips.
- So **4x4 for bf16, 4x2 for int8 and 2x2 for int4**. We really can put LLaMA 3-70B on four v5e chips.
- The catch: the leftover HBM (116GB, 58GB, 29GB) over 8k caches of 2.68GB, 1.34GB and 0.67GB fits 43 sequences every time. That's our max batch, way under the 120 or 240 we'd want.

**Question 6 [latency at max batch]:** Say we fill each of these topologies with the largest batch that fits. What latency should we expect per generate step?

- Batch 43 leaves HBM 99.8% full and is under the critical batch size in all three cases, so each step reads all of HBM once. Every chip does that in parallel, so the slice size cancels out:

  $$
  T_{\text{step}} \approx \frac{16 \times 10^9 \text{ bytes}}{8.2 \times 10^{11} \text{ bytes/s}} = \mathbf{19.5 \text{ ms}}
  $$

- The full step formula gives 19.47 ms for all three (8.80 ms of KV cache plus 10.67 ms of weights).

**Question 7 [throughput per chip]:** What does that work out to in throughput per chip for each of the three, counted in queries per second per chip? Assume a median decode length of 512 tokens.

- Each slice runs 43 queries at once, and each query needs 512 steps of 19.47 ms:

  $$
  \text{QPS per chip} = \frac{43}{0.01947 \text{ s} \times 512 \times \text{chips}} \approx \frac{4.31}{\text{chips}}
  $$

- So **0.27** for bf16 on 16 chips, **0.54** for int8 on 8 and **1.08** for int4 on 4. Halving the bytes per number doubles the queries per chip.
- These are optimistic, since we've left the forward pass zero working memory. The solution guesses the real numbers are around half.

**Question 8 [doubling the topology]:** What happens to the peak throughput if we double the topology in each of the three cases?

- The weights don't grow, so all the new HBM goes to KV caches. For bf16 on a 4x8 that's $512 - 140 = 372\text{ GB}$:

  $$
  \frac{372 \text{ GB}}{2.684 \text{ GB}} = 138.6 \implies \mathbf{B = 138}
  $$

- Same for int8 on a 4x4 (186GB left) and int4 on a 2x4 (93GB left). Twice the chips holds 3.2x the batch.
- If a step is still one full read of HBM, it's 19.45 ms and each slice finishes $138 / (0.01945 \times 512) = 13.86$ queries per second: **0.43** per chip for bf16 (32 chips), **0.87** for int8 (16) and **1.73** for int4 (8). About 1.6x Question 7.
- With bf16 FLOPs, though, 138 is past the critical 120 for int8 and 60 for int4, so their steps grow to 20.2 ms and 26.4 ms and they only get 0.83 and 1.28 QPS per chip (1.5x and 1.2x).

**Question 9 [sharding on a 4x8]:** Now for the sharding. We're serving in bfloat16 on a TPU v5e 4x8. How should the model be sharded during generation, and can we keep from being communication bound?

- Only model parallelism makes sense. FSDP would drag weights over ICI every step, data parallelism just replicates them, and one token per step leaves no sequence to shard.
- If we're compute-bound, the rule from the notes with both mesh axes ($M_Y = 2$) gives:

  $$
  Y \lesssim \frac{M_Y \cdot F}{2200} = \frac{2 \times 28{,}672}{2200} \approx 26
  $$

- So 32-way on a 4x8 is too far if we're compute-bound. But at a small batch we're waiting on weights, and ICI only has to beat HBM. For one MLP matmul sharded $Y$ ways:

  $$
  T_{\text{ICI}} = \frac{2BD}{W_{\text{ici}}} \qquad T_{\text{HBM}} = \frac{2DF}{Y \cdot W_{\text{hbm}}} \qquad T_{\text{math}} = \frac{2BDF}{Y \cdot C}
  $$

- On the 4x8 ($Y = 32$) at $B = 64$ these are $11.7 \ \mu\text{s}$, $17.9 \ \mu\text{s}$ and $4.8 \ \mu\text{s}$. HBM is the slowest, so the ICI traffic hides under the weight loading.
- Break-even is at $B = F / (Y \beta) = 28{,}672 / (32 \times 9.11) \approx 98$. At Question 8's batch of 138, $T_{\text{ICI}}$ is $25.1 \ \mu\text{s}$ against the same $17.9 \ \mu\text{s}$, so we're communication bound.
- So pure model parallelism, and the 4x8 only stays clear of the comms bound **below a batch of about 98**. It can't deliver Question 8's bf16 peak. The int8 (4x4) and int4 (2x4) setups are under the 26-way limit, so they're fine.

**Question 10 [prefill time]:** How long does it take to prefill an 8192 token prompt on 16 TPU v5e chips, if prefill reaches 40% FLOPs utilization?

- 8192 tokens is way past the critical batch size of 240, so we only count FLOPs, 2 per parameter per token:

  $$
  2 \times 70 \times 10^9 \times 8192 \approx 1.147 \times 10^{15} \text{ FLOPs}
  $$

- 16 chips at 40% give $16 \times 1.97 \times 10^{14} \times 0.4 \approx 1.26 \times 10^{15}$ FLOPs/s, so:

  $$
  T_{\text{prefill}} = \frac{1.147 \times 10^{15} \text{ FLOPs}}{1.26 \times 10^{15} \text{ FLOPs/s}} = \mathbf{0.91 \text{ s}}
  $$

**Question 11 [KV cache evictions]:** Our median prefill is 8192 tokens, our median decode is 4096 tokens, and the generate batch size is 32. On an average step, how many sequences finish decoding, and how many tokens get evicted from the KV cache?

- Each of the 32 batch slots frees up once every 4096 steps:

  $$
  \frac{32}{4096} \approx \mathbf{0.0078 \text{ sequences per step}}
  $$

  So one sequence finishes every 128 steps.
- A finished sequence takes its whole cache with it, $8192 + 4096 = 12{,}288$ tokens:

  $$
  \frac{32}{4096} \times 12{,}288 = \mathbf{96 \text{ tokens per step}}
  $$

**Question 12 [disaggregated serving]:** Now we split prefill and generation onto separate servers (disaggregated serving). The median prefill is 8192 tokens, the median decode is 512 tokens, and we use the bfloat16 prefill and generate latencies from the questions above. How many prefill servers do we need per generate server so that both kinds stay fully saturated?

- A prefill server (16 chips, Question 10) does one prompt every 0.91 s, so 1.10 sequences per second.
- A generate server (the bf16 4x4 from Question 6) holds 43 sequences, each needing $512 \times 19.47 \text{ ms} = 9.97\text{ s}$, so it finishes $43 / 9.97 = 4.31$ per second:

  $$
  P \times 1.10 = G \times 4.31 \implies \frac{P}{G} \approx \mathbf{3.9}
  $$

- So about **4 prefill servers for every generate server**, roughly 80% of the chips doing prefill. The solution gets 3 to 1 because it uses a batch of 32 and 19 ms steps.

**Question 13 [LLaMA 3-405B forward pass]:** How many FLOPs per token does one forward pass of LLaMA 3-405B take? If we're FLOPs bound, what's a lower bound on the time for a single forward pass on $N$ TPU v5e chips? And if we're comms bound instead? (We're told to ignore that the model won't fit on one chip.)

- 405B has $L = 126$, $D = 16{,}384$, $F = 53{,}248$, 128 query heads ($N_q$, since $N$ means chips here) and the same 8 KV heads of 128. MLPs $3DFL \approx 329.8 \times 10^9$, attention $2DH(N_q + K)L \approx 71.9 \times 10^9$, embeddings $2VD \approx 4.2 \times 10^9$. About $405.8 \times 10^9$ in total.
- **How many FLOPs per token?** 2 per parameter:

  $$
  2 \times 405.8 \times 10^9 = \mathbf{8.12 \times 10^{11} \text{ FLOPs per token}}
  $$

- **FLOPs bound on $N$ chips:** every chip busy, nothing waiting on communication:

  $$
  T \ge \frac{8.12 \times 10^{11} \times B}{N \times 1.97 \times 10^{14}} = \mathbf{4.12 \text{ ms} \times \frac{B}{N}}
  $$

- **Comms bound:** taking comms as HBM traffic, the pass can't beat streaming every weight once:

  $$
  T \ge \frac{811.7 \times 10^9 \text{ bytes}}{N \times 8.2 \times 10^{11} \text{ bytes/s}} = \mathbf{\frac{0.99 \text{ s}}{N}} \text{ in bf16}
  $$

  Half that with int8 weights, and it doesn't depend on the batch.
- The real bound is whichever is larger, and they cross at $B = 240$ in bf16 (120 with int8 weights).

**Question 14 [serving LLaMA 3-8B]:** We serve LLaMA 3-8B at batch size 240, with the weights and the KV caches both in int8. How many bytes go to (a) the model parameters, (b) the KV caches and (c) the peak working activations, roughly? And what's the smallest topology this runs on? No context length is given, so we assume the 8192 used in the rest of the chapter.

- 8B has $L = 32$, $D = 4{,}096$, $F = 14{,}336$, 32 query heads and 8 KV heads of 128.
- **(a) Model parameters:** MLPs $3DFL \approx 5.64 \times 10^9$, attention $\approx 1.34 \times 10^9$, embeddings $\approx 1.05 \times 10^9$. That's $8.03 \times 10^9$ parameters, so **8.03GB** in int8.
- **(b) KV caches:** $2 \times 8 \times 128 \times 32 = 65{,}536$ bytes per token, so $65{,}536 \times 8192 = 536{,}870{,}912$ bytes per sequence, and for 240 of them:

  $$
  240 \times 536{,}870{,}912 \text{ bytes} \approx \mathbf{128.8 \text{ GB}}
  $$

  That's 16 times the weights.
- **(c) Peak working activations:** a decode step only pushes 240 tokens through. The biggest are the logits `bf16[240, 128256]` at 61.6MB and, if we materialize them, the attention scores `bf16[240, 32, 8192]` at 125.8MB. So **roughly 0.2GB** at worst.
- **Smallest topology:** $8.03 + 128.8 + 0.2 \approx 137.1\text{ GB}$, or 8.57 chips' worth. A 4x2's 128GB doesn't even hold the KV caches, so we need a **4x4** (16 chips, 256GB), 54% full. At 4096 tokens of context it'd be about 72.5GB and a 4x2 would do.

**Question 15 [serving LLaMA 3-405B]:** How should we serve LLaMA 3-405B on TPU v5e, with int8 weights and bfloat16 FLOPs? Given a hard limit of 15 ms per token, which configuration gets the highest throughput, and what's the theoretical minimum step time? Again there's no context length given, so we keep 8192.

- The weights are 405.8GB in int8, 25.4 chips' worth, so a 4x8 (32 chips) is the smallest slice that holds them. The KV cache is $2 \times 8 \times 128 \times 126 = 258{,}048$ bytes per token, or 2.11GB per 8192 token sequence.
- **Theoretical minimum step time:** reading the weights out of HBM once, at batch 1:

  $$
  T_{\text{min}} = \frac{405.8 \times 10^9 \text{ bytes}}{\text{chips} \times 8.2 \times 10^{11} \text{ bytes/s}} \approx \frac{495 \text{ ms}}{\text{chips}}
  $$

- That's 15.5 ms on the 4x8, which misses the 15 ms limit even at batch 1. The next slice up is an **8x8, with a minimum step time of 7.7 ms**, and 618GB left over for 292 sequences. A full 16x16 pod gets the floor down to 1.9 ms.
- **Highest throughput under 15 ms:** per sequence per chip, KV loading costs $2.114 \times 10^9 / (8.2 \times 10^{11}) = 2.58\text{ ms}$ and the FLOPs cost $2 \times 405.8 \times 10^9 / (1.97 \times 10^{14}) = 4.12\text{ ms}$. Past the critical batch of 120 the step is just those two added up:

  $$
  T_{\text{step}} = 6.70 \text{ ms} \times \frac{B}{\text{chips}} \implies \frac{B}{T_{\text{step}} \times \text{chips}} = \frac{1}{6.70 \text{ ms}} \approx \mathbf{149 \text{ tokens/s/chip}}
  $$

- That ceiling is flat, so more batch only adds latency. The 15 ms limit allows $B / \text{chips} \le 15 / 6.70 = 2.24$, and reaching the ceiling needs $B \ge 120$, so at least 54 chips. The 8x8 again.
- On the 8x8 at $B = 120$, a step is 4.83 ms of KV loading plus 7.73 ms for the MLP (right at the crossover, weights and FLOPs take the same time), so **12.6 ms**.
- Sharding is pure model parallelism: weights split 64 ways over both axes, and the KV cache split over its 8 KV heads on one axis and the batch on the other.
- So: **an 8x8 slice at batch 120, giving 149 tokens/s/chip and about 9,550 tokens/s in total at 12.6 ms per token**. Bigger slices serve more users but don't beat 149 per chip.
- The catch is ICI: 64-way is past this model's compute-bound limit of about 48, so the traffic only hides under weight loading up to a batch of about 91 (125 tokens/s/chip), and the real number lands between 125 and 149 depending on overlap.

**Question 16 [implement it yourself]:** The last problem is a build exercise: turn the LLaMA 3-8B weights from [HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/) into a working inference implementation. (1) Load the weights in Colab, inspect every tensor ([TreeScope](https://github.com/google-deepmind/treescope) is suggested), count the parameters and check the MLP vs attention split. (2) Write the full forward pass and sample greedily one token at a time, ideally from the paper alone, falling back on [jax-llm-examples](https://github.com/jax-ml/jax-llm-examples/tree/main/llama3) for details. This fits on a single TPU with more than 16GiB of HBM. (3) Add KV caching. (4) Split it into a prefill server that takes one prompt and a batched decode server. (5) Write Flash Attention in Pallas, preferably without a reference.

No napkin math here, so this is what we should expect at each step:

- **Loading and counting:** $3DFL = 5{,}637{,}144{,}576$ parameters in the MLPs (70%), $2DH(N + K)L = 1{,}342{,}177{,}280$ in attention (17%) and $2VD = 1{,}050{,}673{,}152$ in the two embedding tables (13%), plus 266,240 RMSNorm scales, for 8,030,261,248 in total. Per layer: query and output projections `[4096, 4096]`, key and value `[1024, 4096]`, and MLP matrices `[14336, 4096]`, `[14336, 4096]` and `[4096, 14336]`. In bf16 that's about 16.1GB (15.0GiB), hence the 16GiB hint.
- **Forward pass:** token embedding, then 32 blocks of RMSNorm, attention, residual add, RMSNorm, SwiGLU MLP, residual add, then a final RMSNorm and the projection to 128,256 logits. The easy slips are the rotary position embeddings (applied to queries and keys, not added to the residual) and the causal mask. Each KV head is shared by 4 query heads.
- **KV caching:** keep each layer's keys and values as `bf16[B, S, 8, 128]`, run only the newest token through the model, append its keys and values, and attend over the cache. In bf16 that's 131,072 bytes per token, about 1.07GB for a full 8192 token sequence.
- **Prefill and decode servers:** prefill runs one prompt at batch 1 and returns its KV cache. The decode side drops each finished prefill into a free batch slot and steps all the active slots together, which is the continuous batching setup from Chapter 7.
- **Flash Attention in Pallas:** walk the keys and values in blocks, keeping a running max and normalizer for the softmax, so the full `[T, S]` score matrix never gets written to HBM.

This one needs a TPU runtime in Colab, so I'll come back to it.
