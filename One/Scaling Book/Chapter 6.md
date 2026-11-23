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

   MFU (model FLOPs utilization) is the fraction of peak FLOPs/s we actually get; we'll assume 40% throughout. Memory is 2 bytes per parameter for bf16 weights, 8 bytes per parameter for Adam's two fp32 moments, and $2BD$ bytes for every saved copy of the activations.
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
- About 11.2TB of memory at a 4M-token batch (taken as 4e6 here), 94% of it activation checkpoints. That fits on 117 chips, and on a full pod it's 1.25GB per chip. Meaning, we fill a pod because we need the FLOPs. Memory was never the problem.
- For sharding, the 4M batch is 1024 sequences of 4096 tokens, so 4,194,304. Over 8,960 chips that's 468 tokens per chip: under the FSDP bound (850) but over the mixed bound (113). So we mix: roughly 2048-way FSDP with 4-way TP, or 2240-way FSDP to keep all 8,960 chips busy.
- Spread the same batch over 4 pods and we're at 117 tokens per chip, sitting right on the mixed bound. About 11 days if everything overlaps perfectly.

## Worked Problems

Questions 1 to 9 are the chapter's inline questions in order, and 10 and 11 are the two problems at the end. Everything is on TPU v5p: $4.59 \times 10^{14}$ bf16 FLOPs/s and 96GB of HBM per chip, $1.8 \times 10^{11}$ bytes/s of bidirectional ICI per link, and $6.25 \times 10^{9}$ bytes/s of DCN per chip.

> **Note:** "4M tokens" gets used two ways. The memory questions (5 and 6) plug in 4e6, and the sharding questions use $1024 \times 4096 = 4{,}194{,}304$. It moves a few numbers, but only comes close to flipping a verdict once, in Question 10.

**Question 1 [70B parameter count]:** From the config table ($L = 80$, $D = 8192$, $F = 28672$, $N = 64$, $K = 8$, $H = 128$, $V = 128256$), can we work out the LLaMA 3-70B parameter count and actually land on 70B?

- Just the Chapter 4 accounting with real numbers. Three buckets: MLP, attention and embeddings.
- **MLP:** 3 matrices of $D \times F$ per layer (gate, up and down):

  $$
  3DFL = 3 \times 8192 \times 28672 \times 80 = 56{,}371{,}445{,}760
  $$

- **Attention:** Q and O are $D \times N \times H$ each. K and V are only $D \times K \times H$, since the 64 query heads share 8 KV heads:

  $$
  L(2DNH + 2DKH) = 80 \times (134{,}217{,}728 + 16{,}777{,}216) = 12{,}079{,}595{,}520
  $$

- **Vocab:** separate input and output embeddings, so $2VD = 2 \times 128256 \times 8192 = 2{,}101{,}346{,}304$.
- Adding them up:

  $$
  56.37 \times 10^{9} + 12.08 \times 10^{9} + 2.10 \times 10^{9} = \mathbf{70.55 \times 10^{9} \text{ parameters}}
  $$

- So yes, we land on 70B. The solution says 70.4e9 because it rounds each bucket (56.3, 12 and 2.1) before adding. The norm scales add about 1.3M more, a rounding error.
- The MLP is 80% of the model, attention 17%, the embeddings 3%, which is why Chapter 5 could get away with only looking at the MLP.

**Question 2 [FLOPs per token]:** How many FLOPs does LLaMA 3-70B perform per token in one training step? (This is what sets the price of the whole run.)

- Chapter 4's golden rule: 6 FLOPs per parameter, per token.

  $$
  6 \times 70 \times 10^{9} = \mathbf{4.2 \times 10^{11} \text{ FLOPs per token}}
  $$

- Rounded to 70B; the exact 70.55B gives 4.23e11. On one TPU v5p at peak that's $4.2 \times 10^{11} / 4.59 \times 10^{14} \approx 0.92\text{ ms}$ per token.

**Question 3 [total training FLOPs]:** LLaMA 3 was trained on about 15 trillion tokens. How many FLOPs is that in total?

- Per-token cost times tokens:

  $$
  4.2 \times 10^{11} \times 15 \times 10^{12} = \mathbf{6.3 \times 10^{24} \text{ FLOPs}}
  $$

- For scale: one v5p at peak would need $6.3 \times 10^{24} / 4.59 \times 10^{14} = 1.37 \times 10^{10}$ seconds, about **435 years**. So, more chips.

**Question 4 [one pod at 40% MFU]:** Say we train on a full TPU v5p pod, 16x20x28 = 8960 chips. How long does the run take at 40% MFU in bfloat16, assuming we're compute-bound?

- The pod's peak is $8960 \times 4.59 \times 10^{14} = 4.11 \times 10^{18}$ FLOPs/s, so $1.65 \times 10^{18}$ at 40% MFU. Then we just divide:

  $$
  T = \frac{6.3 \times 10^{24} \text{ FLOPs}}{8960 \times 4.59 \times 10^{14} \times 0.4 \text{ FLOPs/s}} = 3.83 \times 10^{6} \text{ s} = \mathbf{44.3 \text{ days}}
  $$

- About a month and a half on one pod, IF we can hold 40% MFU and stay compute-bound. Questions 7 to 9 are about that second "if".

**Question 5 [minimum chips for a 4M batch]:** LLaMA 3-70B was pretrained with a batch of about 4M tokens. What's the minimum number of TPU v5p chips we need to train at that batch size? Assume bfloat16 parameters, float32 optimizer state, and activations checkpointed 2 times per layer (at the start of the layer and after attention).

- This is a memory question. Too few FLOPs just makes the run slow; too little HBM means it doesn't run at all.
- Three things live in HBM during training:
  - **Parameters:** 2 bytes each in bf16, so $2 \times 70 \times 10^{9} = 140\text{ GB}$.
  - **Optimizer state:** Adam's 2 fp32 moments, 8 bytes per parameter: $8 \times 70 \times 10^{9} = 560\text{ GB}$.
  - **Activation checkpoints:** each one is a `bf16[B, D]` copy of the residual stream, $2 \times 4 \times 10^{6} \times 8192 = 65.5\text{ GB}$, and 2 per layer over 80 layers is 160 of them: $10.49\text{ TB}$.
- Total:

  $$
  0.14 + 0.56 + 10.49 = 11.19 \text{ TB}
  $$

- Each v5p has 96GB of HBM:

  $$
  \frac{11.19 \times 10^{12} \text{ bytes}}{96 \times 10^{9} \text{ bytes/chip}} = 116.5 \implies \mathbf{117 \text{ chips}}
  $$

- We need at least **117 TPUs**, and 94% of that memory is activation checkpoints.
- But on 117 chips the run would take $6.3 \times 10^{24} / (117 \times 4.59 \times 10^{14} \times 0.4)$, around 3,394 days or 9.3 years.

**Question 6 [memory per chip on a full pod]:** Same assumptions as Question 5, but now on all 8960 TPU v5p chips. How much memory do we use per chip?

- Same 11.19TB, just spread thinner, since FSDP shards the weights and optimizer state instead of copying them:

  $$
  \frac{11.19 \times 10^{12} \text{ bytes}}{8960 \text{ chips}} = \mathbf{1.25 \text{ GB per chip}}
  $$

- That's 1.3% of the 96GB on each chip. Tiny.
- Even with 12 checkpoints per layer it's 63.6TB in total, or 7.1GB per chip. The solution rounds this up to 8GB.

> Memory never forces our hand here. We fill a pod because we need the FLOPs.

**Question 7 [FSDP alone, no sequence sharding]:** Keep the same setup: LLaMA 3-70B, a 4M-token batch made of 1024 sequences of length 4096, on a TPU v5p pod of 8960 chips. Can we train with FSDP alone if we aren't allowed any sequence (context) parallelism? Hint: this should be the first thing we try, since it's the simplest scheme and adds no extra communication if it works.

- FSDP shards along the batch, and with no sequence sharding the smallest piece we can hand a chip is one whole sequence.
- We only have $4{,}194{,}304 / 4096 = 1024$ sequences for 8960 chips. So 1024-way at most, and the other 7,936 chips sit idle.
- Those 1024 chips would run fine (4096 tokens each clears even the single-axis bound of 2550), but at 40% MFU they'd take 388 days.
- So **no**, not if we want the whole pod. This is the pedantic version; Question 8 is the real one.

**Question 8 [FSDP over batch and sequence]:** Now let FSDP shard over both the batch and the sequence axes. Is FSDP by itself enough to train LLaMA 3-70B on all 8960 chips?

- The MLP doesn't care which sequence a token came from, so we can split sequences across chips too. Now every chip gets work:

  $$
  \frac{B}{N} = \frac{4{,}194{,}304 \text{ tokens}}{8960 \text{ chips}} = 468 \text{ tokens per chip}
  $$

- With all 3 torus axes on FSDP the bound is $2550 / 3 = 850$, and $468 < 850$. As a ratio, $T_{\text{math}} / T_{\text{comms}} = 468 \times 3 / 2550 = 0.55$. Meaning, the weight AllGathers take about 1.8 times as long as the math they're supposed to hide behind.
- So **no**, pure FSDP is communication bound on a full pod. It would want a batch of at least $850 \times 8960 \approx 7.6\text{M}$ tokens.

**Question 9 [mixed FSDP + tensor parallelism]:** Now mix tensor parallelism with FSDP. Is there a combination that keeps us compute-bound? If so, how much FSDP and how much TP should we use?

- First, can a mix work at all? The FSDP + TP bound on a 3D torus ($M_X M_Y = 2$):

  $$
  \frac{B}{N} > \frac{\alpha^2}{M_X M_Y F} = \frac{2550^2}{2 \times 28672} = 113 \text{ tokens per chip}
  $$

- We have 468, about 4 times that, so yes.
- Now the split, with 2 axes for FSDP and 1 for TP:

  $$
  X_{\text{opt}} = \sqrt{\frac{B}{F} \cdot \frac{M_X}{M_Y} \cdot N} = \sqrt{\frac{4{,}194{,}304}{28672} \times 2 \times 8960} \approx 1619
  $$

- That leaves $Y = 8960 / 1619 \approx 5.5$. Rounding both to powers of two gives 2048-way FSDP (1024 over the batch, 2 over the sequence) and 4-way TP.
- Checking both sides. FSDP: $(B/X) \cdot M_X / \alpha = (4{,}194{,}304 / 2048) \times 2 / 2550 = 1.61$. TP: $F M_Y / (Y \alpha) = 28672 / (4 \times 2550) = 2.81$. Both above 1, so both kinds of comms hide behind the math.
- So we use roughly **2048-way FSDP and 4-way tensor parallelism**.
- Notice $2048 \times 4 = 8192$ leaves 768 chips idle. $X = 2240$ with the same $Y = 4$ fills the pod and is also compute-bound (1.47 and 2.81).

**Question 10 [scaling LLaMA 70B to more chips]:** Say we train LLaMA 3-70B on 4 TPU v5p pods with the same 4M-token batch. What parallelism scheme do we use? Are we compute or communication bound? Roughly how long does training take? Hint: make sure to use the correct roofline bound.

- Same batch on 4 times the chips, and ICI stops at the pod edge, so the pods talk over DCN.
- $4 \times 8960 = 35{,}840$ chips. Each pod gets $4{,}194{,}304 / 4 = 1{,}048{,}576$ tokens (256 sequences), and each chip gets 117.
- **Across pods:** data parallelism over DCN, each pod holding a full copy of the model. That's compute-bound above $C / W_{\text{dcn}} = 4.59 \times 10^{14} / 6.25 \times 10^{9} = 73{,}440$ tokens per pod, and we have about 1.05M, 14 times that.
- **Inside a pod:** this is where the hint matters. Against the FSDP bound (850), 117 tokens per chip looks hopeless, but the right bound is the mixed one from Question 9, 113. $117 > 113$, so on paper we're compute-bound, by about 3%.
- A quarter of the batch per pod halves $X_{\text{opt}}$ and doubles $Y$:

  $$
  X_{\text{opt}} = \sqrt{\frac{1{,}048{,}576}{28672} \times 2 \times 8960} \approx 810, \qquad Y = \frac{8960}{810} \approx 11
  $$

- At that exact split $T_{\text{math}} / T_{\text{comms}} = 1.016$, but we can't build an 11.07-way split. Rounded to powers of two, each choice loses on one side:
  - $1120 \times 8$: FSDP ratio 0.73, TP ratio 1.41.
  - $560 \times 16$: FSDP ratio 1.47, TP ratio 0.70.
- So we're **compute-bound on paper, but sitting right on the roofline**. Even $896 \times 10$, the closest split the pod's axes allow, comes up a bit short (0.92 and 1.12).
- We'll take the 8-way TP side. FSDP weight gathers can start early while the previous layer runs, but TP gathers move the layer's own input, so those are the ones we don't want slow.
- The scheme: **4-way data parallelism across the pods over DCN, and inside each pod about 1120-way FSDP with 8-way TP**. With only 256 sequences per pod, FSDP has to cut across the sequence too.
- For the time, if we're compute-bound at 40% MFU:

  $$
  T = \frac{6.3 \times 10^{24} \text{ FLOPs}}{35{,}840 \times 4.59 \times 10^{14} \times 0.4 \text{ FLOPs/s}} = 9.57 \times 10^{5} \text{ s} = \mathbf{11.1 \text{ days}}
  $$

- A quarter of the single-pod 44 days. With $1120 \times 8$ the FSDP gathers take 1.36 times as long as the math at peak, capping MFU at about 73%, still above our 40%. So roughly **11 days** if the comms fully overlap with the math, closer to 15 days if that slowdown stacks on top of the 40%.
- Memory is a non-issue: about 0.38GB per chip (78MB of weights and optimizer state, 307MB of activations).
- **TODO(eddie):** no published answer here, and 117 against 113 is too close to call cleanly (with $B = 4 \times 10^{6}$ instead of 4,194,304 it's 112 tokens per chip and the ratio dips to 0.99). Decide whether to call this compute-bound outright, and whether to quote 11 days or the 11 to 15 day range.

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

  - The name checks out again, and the MLP is 81% of it.
* **How many FLOPs per training step?**
  - Same $6 \times$ rule per token, rounding to 405B:

    $$
    6 \times 405 \times 10^{9} = \mathbf{2.43 \times 10^{12} \text{ FLOPs per token}}
    $$

  - About 5.8 times the 70B. At the same 4,194,304-token batch, one step is $2.43 \times 10^{12} \times 4{,}194{,}304 \approx 1.02 \times 10^{19}$ FLOPs.
* **How many FLOPs for 15T tokens?**
  - Per-token cost times tokens again:

    $$
    2.43 \times 10^{12} \times 15 \times 10^{12} = 3.645 \times 10^{25} \approx \mathbf{3.6 \times 10^{25} \text{ FLOPs}}
    $$

  - Around 2,500 years on a single v5p at peak.

**(b)** Now assume we train on 8 TPU v5p pods. What parallelism scheme do we use? How long does training take? Are we compute or comms bound?

No batch size, MFU or checkpointing scheme is given for 405B, so we take the 70B setup (4,194,304-token batch, 40% MFU, 2 checkpoints per layer) and also check a 16M batch.

* **How long would training take?**
  - 8 pods is $8 \times 8960 = 71{,}680$ chips, with a peak of $3.29 \times 10^{19}$ FLOPs/s. At 40% MFU:

    $$
    T = \frac{3.645 \times 10^{25} \text{ FLOPs}}{71{,}680 \times 4.59 \times 10^{14} \times 0.4 \text{ FLOPs/s}} = 2.77 \times 10^{6} \text{ s} = \mathbf{32 \text{ days}}
    $$

  - About a month, provided we're compute-bound.
* **What parallelism scheme, and are we compute or comms bound?**
  - Same outer layer as Question 10: data parallelism across the 8 pods over DCN, FSDP + TP inside each pod.
  - The bigger $F$ helps: more math per token to hide the comms behind. The mixed bound drops to $2550^2 / (2 \times 53248) = 61$ tokens per chip, so 8 pods want a total batch above $61 \times 71{,}680 \approx 4.4\text{M}$ tokens.
  - At 4,194,304 tokens, each pod gets 524,288 (7 times the DCN bound) and each chip gets 58.5. Just under 61, so we're a hair communication bound: $T_{\text{math}} / T_{\text{comms}} = 0.98$ at the best split of $X_{\text{opt}} \approx 420$, $Y \approx 21$.
  - As powers of two it's $560 \times 16$ (FSDP ratio 0.73, TP ratio 1.31) or $280 \times 32$ (1.47 and 0.65). Same pick as Question 10, keeping the TP side above 1: $560 \times 16$, which caps MFU at about 73%.
  - The [LLaMA 3 paper](https://arxiv.org/abs/2407.21783) says the 405B run started at a 4M-token batch, doubled it to 8M, and doubled again to 16M at 2.87T tokens into pre-training. At 16M (16,777,216 tokens) each chip gets 234 tokens, almost 4 times the bound, and $1120 \times 8$ gives ratios of 1.47 and 2.61. Comfortably compute-bound.
  - So: **data parallelism over DCN across the 8 pods, and FSDP + TP inside each pod** ($560 \times 16$ at a 4M batch, $1120 \times 8$ at 16M). At 4M we're on the roofline again, slightly on the comms side; at 16M we're compute-bound. Either way it's roughly **32 days** at 40% MFU, same estimate as Question 10.
  - Memory is still tiny: under 1GB per chip at a 4M batch (452MB of weights and optimizer state, 483MB of activations).
