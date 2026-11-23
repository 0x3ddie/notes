---
title: Training LLaMA 3 on TPUs
published: true
---
## 6. Training LLaMA 3-70B

This chapter is basically one long problem set. There's no new theory; we take the FLOPs counting from Chapter 4 and the sharding bounds from Chapter 5 and point them at a real model, LLaMA 3-70B on TPU v5p. We want four things out of it: the parameter count, the FLOPs bill for a full training run, how long that takes on a pod, and a sharding that doesn't leave the chips waiting on the network. The notes are short on purpose, since the questions do most of the work. The original is at [jax-ml.github.io/scaling-book/applied-training/](https://jax-ml.github.io/scaling-book/applied-training/).

### What LLaMA 3 Looks Like

Nearly everything here is about the 70B model. Its siblings are 8B and 405B, and 405B gets the last question. Everything we need sits in the model's `config.json` on HuggingFace, mapped onto our usual letters:

| Hyperparameter | Value |
|---|---|
| Layers ($L$) | 80 |
| Model dim ($D$) | 8,192 |
| Feed-forward dim ($F$) | 28,672 |
| Query heads ($N$) | 64 |
| KV heads ($K$) | 8 |
| Head dim ($H$) | 128 |
| Vocab size ($V$) | 128,256 |

A few things to notice. $F$ is $3.5D$ rather than the usual $4D$, and the MLP is the gated kind, so it has 3 of those $D \times F$ matrices instead of 2. Attention is GQA: 64 query heads share 8 KV heads, so the K and V projections are 8 times smaller than Q and O. And $N \times H = D$, as usual.

### The Plan of Attack

Every question in this chapter is one of four steps, always in this order:

1. **Count parameters.** Per layer, the MLP has $3DF$ and attention has $2DNH + 2DKH$. Add $2VD$ once for the input and output embeddings.
2. **Count FLOPs.** Training costs $6 \times \text{params}$ FLOPs per token (2 forward, 4 backward), so a whole run is $6 \times \text{params} \times \text{tokens}$.
3. **Turn FLOPs into time and memory.** Time is just the total FLOPs over what the cluster really delivers:

   $$
   T = \frac{\text{total FLOPs}}{\text{chips} \times C \times \text{MFU}}
   $$

   MFU (model FLOPs utilization) is the fraction of peak FLOPs/s we actually get; the book assumes 40%. Memory is 2 bytes per parameter for bf16 weights, 8 bytes per parameter for Adam's two fp32 moments, and $2BD$ bytes for every saved copy of the activations.
4. **Pick a sharding.** Work out the batch per chip and compare it against the bounds from Chapter 5.

### The Bounds We Keep Reusing

Napkin math for TPU v5p: $C = 4.59 \times 10^{14}$ bf16 FLOPs/s, 96GB of HBM, 1.8e11 bytes/s of bidirectional ICI per link, 6.25e9 bytes/s of DCN per chip, and a full pod is 16x20x28 = 8,960 chips. So the ICI intensity is $\alpha = C / W_{\text{ici}} = 2550$.

Take $B$ as the total batch in tokens and $N$ as the number of chips, with $X$-way FSDP spread over $M_X$ mesh axes and $Y$-way tensor parallelism (TP) over $M_Y$ axes:

- **FSDP alone:** compute-bound when $B/N > \alpha / M_X$. With all 3 axes that's 850 tokens per chip.
- **TP alone:** compute-bound when $Y < M_Y F / \alpha$. For $F = 28{,}672$ that's about 11-way per axis, whatever the batch size.
- **FSDP + TP:** compute-bound when $B/N > \alpha^2 / (M_X M_Y F)$, with the best split at $X_{\text{opt}} = \sqrt{(B/F)(M_X/M_Y)N}$. A 3D torus gives $M_X M_Y = 2$, so the bound is 113 tokens per chip for 70B.
- **Across pods:** ICI stops at the edge of a pod, so pods talk over DCN with plain data parallelism. That's fine as long as each pod's batch is above $C / W_{\text{dcn}} = 73{,}440$ tokens.

Why does the mix help? FSDP ships weights around, and that cost doesn't shrink when the batch does. TP ships activations, which do shrink with the batch. So as the batch per chip gets smaller, we trade some FSDP for TP.

### Takeaway

The headline numbers for LLaMA 3-70B on TPU v5p:

- 70.55B parameters, 80% of them in the MLP matrices.
- 4.2e11 FLOPs per token and 6.3e24 FLOPs for 15T tokens. That's 435 years on one chip, or 44 days on a full pod at 40% MFU.
- About 11.2TB of memory at a 4M-token batch (taken as 4e6, like the book's memory table), 94% of it activation checkpoints. That fits on 117 chips, and on a full pod it's 1.25GB per chip. Meaning, we fill a pod because we need the FLOPs. Memory was never the problem.
- For sharding, the 4M batch is 1024 sequences of 4096 tokens, so 4,194,304. Over 8,960 chips that's 468 tokens per chip: under the FSDP bound (850) but over the mixed bound (113). So we mix: roughly 2048-way FSDP with 4-way TP in the book's rounding, or 2240-way FSDP to keep all 8,960 chips busy.
- Spread the same batch over 4 pods and we're at 117 tokens per chip, sitting right on the mixed bound. About 11 days if everything overlaps perfectly.

## Worked Problems

The book leaves its inline questions unnumbered, so I've numbered them 1 to 9 in the order they appear. Questions 10 and 11 are the two problems from the book's own list at the end. Everything assumes TPU v5p: $4.59 \times 10^{14}$ bf16 FLOPs/s and 96GB of HBM per chip, $1.8 \times 10^{11}$ bytes/s of bidirectional ICI per link, and $6.25 \times 10^{9}$ bytes/s of DCN per chip.

> **Note:** The book is a bit loose about what "4M tokens" means. The memory questions (5 and 6) plug in 4e6, and the sharding questions (7 to 9) use $1024 \times 4096 = 4{,}194{,}304$. I use whichever one the book uses in each question, and stay with 4,194,304 for Questions 10 and 11, which have no published answer. The choice moves a few numbers (117 chips becomes 122 in Question 5), but it only comes close to changing a verdict once (Question 10).

**Question 1 [70B parameter count]:** From the config table ($L = 80$, $D = 8192$, $F = 28672$, $N = 64$, $K = 8$, $H = 128$, $V = 128256$), can we work out the LLaMA 3-70B parameter count and actually land on 70B?

- This is just the Chapter 4 accounting applied to real numbers. We have three buckets: the MLP matrices, the attention projections, and the embeddings.
- **MLP:** 3 matrices of size $D \times F$ per layer (the gate, up and down projections):

  $$
  3DFL = 3 \times 8192 \times 28672 \times 80 = 56{,}371{,}445{,}760
  $$

- **Attention:** Q and O are $D \times N \times H$ each. K and V are only $D \times K \times H$ each, since the 64 query heads share 8 KV heads:

  $$
  L(2DNH + 2DKH) = 80 \times (134{,}217{,}728 + 16{,}777{,}216) = 12{,}079{,}595{,}520
  $$

- **Vocab:** separate input and output embeddings, so $2VD = 2 \times 128256 \times 8192 = 2{,}101{,}346{,}304$.
- Adding them up:

  $$
  56.37 \times 10^{9} + 12.08 \times 10^{9} + 2.10 \times 10^{9} = \mathbf{70.55 \times 10^{9} \text{ parameters}}
  $$

- So yes, we get our 70B. The solution says 70.4e9, but only because it rounds each bucket (56.3, 12 and 2.1) before adding. The norm scales add about 1.3M more, which is a rounding error.
- The MLP is 80% of the model, attention is 17%, and the embeddings are 3%. This is why Chapter 5 could work out its sharding bounds from the MLP matrices alone.

**Question 2 [FLOPs per token]:** How many FLOPs does LLaMA 3-70B perform per token in one training step? (This is what sets the price of the whole run.)

- Chapter 4's golden rule: 6 FLOPs per parameter, per token.

  $$
  6 \times 70 \times 10^{9} = \mathbf{4.2 \times 10^{11} \text{ FLOPs per token}}
  $$

- I'm rounding the count to 70B like the book does; with the exact 70.55B it's 4.23e11. Either way, roughly 0.42 trillion FLOPs for every single token.
- On a single TPU v5p at peak, that's $4.2 \times 10^{11} / 4.59 \times 10^{14} \approx 0.92\text{ ms}$ per token. The solution rounds this to 1ms.

**Question 3 [total training FLOPs]:** LLaMA 3 was trained on about 15 trillion tokens. How many FLOPs is that in total?

- Per-token cost times tokens:

  $$
  4.2 \times 10^{11} \times 15 \times 10^{12} = \mathbf{6.3 \times 10^{24} \text{ FLOPs}}
  $$

- For scale: one TPU v5p running at peak would need $6.3 \times 10^{24} / 4.59 \times 10^{14} = 1.37 \times 10^{10}$ seconds, which is about **435 years**. So, more chips.

**Question 4 [one pod at 40% MFU]:** Say we train on a full TPU v5p pod, 16x20x28 = 8960 chips. How long does the run take at 40% MFU in bfloat16, assuming we're compute-bound?

- The pod's peak is $8960 \times 4.59 \times 10^{14} = 4.11 \times 10^{18}$ FLOPs/s. At 40% MFU we really get $1.65 \times 10^{18}$.
- Then we just divide:

  $$
  T = \frac{6.3 \times 10^{24} \text{ FLOPs}}{8960 \times 4.59 \times 10^{14} \times 0.4 \text{ FLOPs/s}} = 3.83 \times 10^{6} \text{ s} = \mathbf{44.3 \text{ days}}
  $$

- About a month and a half on one pod, IF we can actually hold 40% MFU and stay compute-bound. Questions 7 to 9 are about that second "if". (The solution says about 44 days. Carrying the exact 70.55B parameters through instead of 70B would give 44.7.)

**Question 5 [minimum chips for a 4M batch]:** LLaMA 3-70B was pretrained with a batch of about 4M tokens. What's the minimum number of TPU v5p chips we need to train at that batch size? Assume bfloat16 parameters, float32 optimizer state, and activations checkpointed 2 times per layer (at the start of the layer and after attention).

- This is a memory question. Too few FLOPs just makes the run slow; too little HBM means it doesn't run at all. So HBM is the only hard limit on chip count.
- Three things live in HBM during training:
  - **Parameters:** bf16 is 2 bytes each, so $2 \times 70 \times 10^{9} = 140\text{ GB}$.
  - **Optimizer state:** Adam keeps 2 fp32 moments per parameter, so 8 bytes each: $8 \times 70 \times 10^{9} = 560\text{ GB}$.
  - **Activation checkpoints:** each one is a `bf16[B, D]` copy of the residual stream, $2 \times 4 \times 10^{6} \times 8192 = 65.5\text{ GB}$. We keep 2 per layer for 80 layers, so 160 of them: $10.49\text{ TB}$.
- Total:

  $$
  0.14 + 0.56 + 10.49 = 11.19 \text{ TB}
  $$

- Each v5p has 96GB of HBM:

  $$
  \frac{11.19 \times 10^{12} \text{ bytes}}{96 \times 10^{9} \text{ bytes/chip}} = 116.5 \implies \mathbf{117 \text{ chips}}
  $$

- We need at least **117 TPUs**. The activation checkpoints are 94% of that memory, even at only 2 saves per layer. (With the exact 4,194,304 tokens it's 11.7TB and 122 chips. Same story.)
- So why not train on 117 chips? Because of Question 4: $6.3 \times 10^{24} / (117 \times 4.59 \times 10^{14} \times 0.4)$ comes to around 3,394 days, or 9.3 years. The solution gets 3,369 days by scaling its rounded 44 days.
- Also, saving this little means the backward pass has to redo most of the forward pass, so the real cost drifts from $6 \times$ toward $8 \times$ params per token.

**Question 6 [memory per chip on a full pod]:** Same assumptions as Question 5, but now on all 8960 TPU v5p chips. How much memory do we use per chip?

- Nothing about the model changed, so it's the same 11.19TB, just spread thinner (FSDP shards the weights and optimizer state instead of copying them):

  $$
  \frac{11.19 \times 10^{12} \text{ bytes}}{8960 \text{ chips}} = \mathbf{1.25 \text{ GB per chip}}
  $$

- That's 1.3% of the 96GB on each chip. Tiny. (The solution says about 1.3GB; with the exact 4,194,304 tokens it's 1.31GB.)
- Even if we got greedy and saved 12 checkpoints per layer, the activations grow to 62.9TB, the total to 63.6TB, and we'd sit at 7.1GB per chip (the solution rounds this up to 8GB).

> Memory never forces our hand here. We fill a pod because we need the FLOPs.

**Question 7 [FSDP alone, no sequence sharding]:** Keep the same setup: LLaMA 3-70B, a 4M-token batch made of 1024 sequences of length 4096, on a TPU v5p pod of 8960 chips. Can we train with FSDP alone if we aren't allowed any sequence (context) parallelism? Hint: this should be the first thing we try, since it's the simplest scheme and adds no extra communication if it works.

- FSDP shards along the batch. With no sequence sharding, the smallest piece we can hand to a chip is one whole sequence.
- We only have $4{,}194{,}304 / 4096 = 1024$ sequences, and 8960 chips. So we can go 1024-way at most, and the other 7,936 chips get nothing to do.
- On those 1024 chips it would run fine: 4096 tokens per chip clears even the single-axis bound of 2550. But 1024 chips at 40% MFU take 388 days.
- So the answer is **no**, not if we want to use the whole pod. The book admits this is the pedantic version of the question; the next one is the real one.

**Question 8 [FSDP over batch and sequence]:** Now let FSDP shard over both the batch and the sequence axes. Is FSDP by itself enough to train LLaMA 3-70B on all 8960 chips?

- The MLP doesn't care which sequence a token came from, so we're free to split sequences across chips too. Now all 8960 chips get work:

  $$
  \frac{B}{N} = \frac{4{,}194{,}304 \text{ tokens}}{8960 \text{ chips}} = 468 \text{ tokens per chip}
  $$

- FSDP is compute-bound when $B/N > \alpha / M_X$. Giving it all 3 axes of the torus, that's $2550 / 3 = 850$ tokens per chip.
- $468 < 850$. In ratio terms, $T_{\text{math}} / T_{\text{comms}} = 468 \times 3 / 2550 = 0.55$, meaning the weight AllGathers take about 1.8 times as long as the math they're supposed to hide behind.
- So **no**. Pure FSDP leaves us communication bound on a full pod, and that's with the whole 3D torus working for it. It would want a batch of at least $850 \times 8960 \approx 7.6\text{M}$ tokens.

**Question 9 [mixed FSDP + tensor parallelism]:** Now mix tensor parallelism with FSDP. Is there a combination that keeps us compute-bound? If so, how much FSDP and how much TP should we use?

- First we check that a mix can work at all. The bound for FSDP + TP on a 3D torus ($M_X M_Y = 2$) is:

  $$
  \frac{B}{N} > \frac{\alpha^2}{M_X M_Y F} = \frac{2550^2}{2 \times 28672} = 113 \text{ tokens per chip}
  $$

- We have 468, about 4 times the bound. So yes, a compute-bound mix exists.
- Now the split. With 2 axes for FSDP and 1 for TP:

  $$
  X_{\text{opt}} = \sqrt{\frac{B}{F} \cdot \frac{M_X}{M_Y} \cdot N} = \sqrt{\frac{4{,}194{,}304}{28672} \times 2 \times 8960} \approx 1619
  $$

- That leaves $Y = 8960 / 1619 \approx 5.5$ for TP, which isn't a whole number of chips. The solution gets $X_{\text{opt}} = 1618$ (it plugs in $B = 4.19 \times 10^{6}$) and rounds both to powers of two: 2048-way FSDP (1024 over the batch, 2 over the sequence) and 4-way TP.
- Checking both sides of that split. For FSDP, $T_{\text{math}} / T_{\text{comms}} = (B/X) \cdot M_X / \alpha = (4{,}194{,}304 / 2048) \times 2 / 2550 = 1.61$. For TP, it's $F M_Y / (Y \alpha) = 28672 / (4 \times 2550) = 2.81$. Both are above 1, so both kinds of comms hide behind the math.
- So we use roughly **2048-way FSDP and 4-way tensor parallelism**.
- One remark on the solution's split: $2048 \times 4 = 8192$, so it leaves 768 of the pod's 8960 chips idle. Keeping $Y = 4$ and spreading FSDP over the whole pod gives $X = 2240$, which is also compute-bound (1.47 and 2.81). So is $Y = 8$ with $X = 1120$ (2.94 and 1.41), so there's some slack in the choice.

**Question 10 [scaling LLaMA 70B to more chips]:** Say we train LLaMA 3-70B on 4 TPU v5p pods with the same 4M-token batch. What parallelism scheme do we use? Are we compute or communication bound? Roughly how long does training take? Hint: make sure to use the correct roofline bound.

- The book doesn't publish an answer for this one, so this is worked from the Chapter 5 formulas. What changed is that we have 4 times the chips with the same batch, so every chip gets a quarter of the tokens. And ICI doesn't reach across pods, so the pods have to talk over DCN.
- The numbers: $4 \times 8960 = 35{,}840$ chips. Each pod gets $4{,}194{,}304 / 4 = 1{,}048{,}576$ tokens (256 sequences), and each chip gets 117 tokens.
- **Across pods:** plain data parallelism over DCN, with each pod holding a full copy of the model. That's compute-bound when the batch per pod is above $C / W_{\text{dcn}} = 4.59 \times 10^{14} / 6.25 \times 10^{9} = 73{,}440$ tokens. We have about 1.05M, 14 times that. No problem here.
- **Inside a pod:** this is where the hint matters. Against the pure FSDP bound (850), our 117 tokens per chip looks hopeless. But the right bound is the mixed one from Question 9, 113 tokens per chip. $117 > 113$, so on paper we're compute-bound, by about 3%.
- The best split leans harder on TP. A quarter of the batch per pod halves $X_{\text{opt}}$ and doubles $Y$:

  $$
  X_{\text{opt}} = \sqrt{\frac{1{,}048{,}576}{28672} \times 2 \times 8960} \approx 810, \qquad Y = \frac{8960}{810} \approx 11
  $$

- At that exact split, $T_{\text{math}} / T_{\text{comms}} = 1.016$. But we can't build an 11.07-way split, and once $Y$ is rounded to a power of two (like the book does), each choice loses on one side:
  - $1120 \times 8$: FSDP ratio 0.73, TP ratio 1.41.
  - $560 \times 16$: FSDP ratio 1.47, TP ratio 0.70.
- So we're **compute-bound on paper, but sitting right on the roofline**. Each power-of-two split is a little communication bound on one side or the other, and so is the closest one the pod's axes allow ($896 \times 10$: 0.92 and 1.12).
- I'd take the 8-way TP side. The FSDP weight gathers can be started early, while the previous layer is still running. The TP gathers can't, since the activations they move are the layer's own input, so those are the ones we don't want to be slow.
- The book lands in the same place. The cross-pod example at the end of its Section 5 also has 1M tokens per pod (about 112 tokens per chip against a bar of 108, with $F$ rounded to 30,000). It treats that as compute-bound, if a close call, and uses roughly $X = 1024$, $Y = 8$, which is this split rounded to powers of two.
- The scheme: **4-way data parallelism across the pods over DCN, and inside each pod about 1120-way FSDP with 8-way TP**. With only 256 sequences per pod, the FSDP axis has to cut across the sequence too.
- For the time, if we're compute-bound at 40% MFU:

  $$
  T = \frac{6.3 \times 10^{24} \text{ FLOPs}}{35{,}840 \times 4.59 \times 10^{14} \times 0.4 \text{ FLOPs/s}} = 9.57 \times 10^{5} \text{ s} = \mathbf{11.1 \text{ days}}
  $$

- That's a quarter of the single-pod 44 days, as it should be. With the $1120 \times 8$ split, the FSDP gathers take 1.36 times as long as the math does at peak speed, which caps MFU at about 73%. That's still above the 40% we assumed, so as a roofline estimate roughly **11 days** holds. This leans on the comms overlapping fully with the math. If that slowdown stacked on top of the 40% instead, it'd be closer to 15 days.
- Memory is a non-issue: about 0.38GB per chip (78MB of weights and optimizer state, 307MB of activations).
- **TODO(eddie):** the book gives no answer here, and 117 against 113 is too close to call cleanly (with $B = 4 \times 10^{6}$ instead of 4,194,304 it's 112 tokens per chip and the ratio dips to 0.99). Decide whether to call this compute-bound outright, and whether to quote 11 days or the 11 to 15 day range.

**Question 11 [LLaMA 405B]:**

**(a)** Using the LLaMA 3-405B [config](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-405B/blob/main/config.json), write out the same hyperparameter table as we did for 70B. What's the total parameter count? How many FLOPs is one training step? And how many FLOPs is a 15T-token training run?

The config gives:

| Hyperparameter | Value |
|---|---|
| Layers ($L$) | 126 |
| Model dim ($D$) | 16,384 |
| Feed-forward dim ($F$) | 53,248 |
| Query heads ($N$) | 128 |
| KV heads ($K$) | 8 |
| Head dim ($H$) | 128 |
| Vocab size ($V$) | 128,256 |

Compared to 70B: $D$ doubles, $F$ is now $3.25D$, there are 126 layers instead of 80, and the KV heads and vocab stay the same.

* **How many total parameters?**
  - Same three buckets as Question 1.
  - MLP: $3DFL = 3 \times 16384 \times 53248 \times 126 = 329{,}772{,}957{,}696$.
  - Attention: $L(2DNH + 2DKH) = 126 \times (536{,}870{,}912 + 33{,}554{,}432) = 71{,}873{,}593{,}344$.
  - Vocab: $2VD = 2 \times 128256 \times 16384 = 4{,}202{,}692{,}608$.
  - Adding them up:

    $$
    329.77 \times 10^{9} + 71.87 \times 10^{9} + 4.20 \times 10^{9} = \mathbf{405.8 \times 10^{9} \text{ parameters}}
    $$

  - So the name checks out again. The MLP is 81% of it.
* **How many FLOPs per training step?**
  - Per token it's the same $6 \times$ rule, rounding to 405B:

    $$
    6 \times 405 \times 10^{9} = \mathbf{2.43 \times 10^{12} \text{ FLOPs per token}}
    $$

  - That's about 5.8 times the 70B. The book doesn't give a batch size for 405B; at the same 4,194,304-token batch, one step is $2.43 \times 10^{12} \times 4{,}194{,}304 \approx 1.02 \times 10^{19}$ FLOPs.
* **How many FLOPs for 15T tokens?**
  - Per-token cost times tokens again:

    $$
    2.43 \times 10^{12} \times 15 \times 10^{12} = 3.645 \times 10^{25} \approx \mathbf{3.6 \times 10^{25} \text{ FLOPs}}
    $$

  - Around 2,500 years on a single v5p at peak.

**(b)** Now assume we train on 8 TPU v5p pods. What parallelism scheme do we use? How long does training take? Are we compute or comms bound?

The book gives no batch size, MFU or checkpointing scheme for 405B, so this part assumes the 70B setup throughout: a 4,194,304-token batch, 40% MFU and 2 checkpoints per layer. The batch size is the assumption that matters, so the 16M case is checked too.

* **How long would training take?**
  - 8 pods is $8 \times 8960 = 71{,}680$ chips, with a peak of $3.29 \times 10^{19}$ FLOPs/s. Assuming the same 40% MFU as before:

    $$
    T = \frac{3.645 \times 10^{25} \text{ FLOPs}}{71{,}680 \times 4.59 \times 10^{14} \times 0.4 \text{ FLOPs/s}} = 2.77 \times 10^{6} \text{ s} = \mathbf{32 \text{ days}}
    $$

  - About a month, provided we're compute-bound. That depends on the batch size, which the question doesn't give us.
* **What parallelism scheme, and are we compute or comms bound?**
  - The outer layer is the same as Question 10: data parallelism across the 8 pods over DCN, FSDP + TP inside each pod.
  - The bigger $F$ helps us here, since there's more math per token to hide the comms behind. The mixed bound drops to $2550^2 / (2 \times 53248) = 61$ tokens per chip. So 8 pods want a total batch above $61 \times 71{,}680 \approx 4.4\text{M}$ tokens. (Pure FSDP would want about 61M.)
  - With the 70B's batch of 4,194,304 tokens, each pod gets 524,288 tokens (7 times the DCN bound, fine) and each chip gets 58.5. That's just under 61, so we're a hair communication bound: $T_{\text{math}} / T_{\text{comms}} = 0.98$ at the best split of $X_{\text{opt}} \approx 420$, $Y \approx 21$.
  - Rounding $Y$ to a power of two again, the choices are $560 \times 16$ (FSDP ratio 0.73, TP ratio 1.31) or $280 \times 32$ (1.47 and 0.65). Same picture as Question 10 and the same pick: $560 \times 16$, which caps MFU at about 73%. (Giving TP the pod's whole 20 axis, $448 \times 20$, would get closer: 0.92 and 1.04.)
  - The [LLaMA 3 paper](https://arxiv.org/abs/2407.21783) says the 405B run started at a 4M-token batch and later grew it to 8M and then 16M. At 16M (16,777,216 tokens) each chip gets 234 tokens, almost 4 times the bound, and a $1120 \times 8$ split gives ratios of 1.47 and 2.61. Comfortably compute-bound.
  - So: **data parallelism over DCN across the 8 pods, and FSDP + TP inside each pod** ($560 \times 16$ at a 4M batch, $1120 \times 8$ at 16M). At 4M we're sitting on the roofline again, slightly on the comms side; at 16M we're compute-bound. Either way it's roughly **32 days** at 40% MFU, by the same roofline estimate as Question 10 (if the 4M slowdown stacked on top of the 40% instead, closer to 44 days).
  - Memory is still tiny: under 1GB per chip at a 4M batch (452MB of weights and optimizer state, 483MB of activations), even with every pod holding its own copy of the model.
  - The book gives no batch size, no MFU and no answer for 405B, so all of this rests on the 70B numbers (4M tokens, 40% MFU, 2 checkpoints per layer), with the 16M batch as the second case. The paper puts that second doubling, to 16M, at 2.87T tokens into pre-training.
