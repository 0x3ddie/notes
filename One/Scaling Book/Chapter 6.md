---
title: Training LLaMA 3 on TPUs
published: true
---
## 6. Training LLaMA 3-70B

This chapter is basically one long problem set. There's no new theory; we take the FLOPs counting from Chapter 4 and the sharding bounds from Chapter 5 and point them at LLaMA 3-70B on TPU v5p. The original is at [jax-ml.github.io/scaling-book/applied-training/](https://jax-ml.github.io/scaling-book/applied-training/).

### What does LLaMA 3 look like?

LLaMA 3 comes in 8B, 70B and 405B. Nearly everything here is about 70B, and 405B gets the last question. From the model's `config.json` on HuggingFace, in our usual letters: $L = 80$ layers, $D = 8192$, $F = 28672$, $N = 64$ query heads, $K = 8$ KV heads, $H = 128$ and $V = 128256$.

$F$ is $3.5D$ rather than the usual $4D$, and the MLP is gated, so it has 3 of those $D \times F$ matrices instead of 2. Attention is GQA: 64 query heads share 8 KV heads, so the K and V projections are 8 times smaller than Q and O. And $N \times H = D$, as usual.

### Counting parameters and FLOPs

- **Parameters:** per layer the MLP has $3DF$ and attention has $2DNH + 2DKH$, plus $2VD$ once for the input and output embeddings. For 70B that's 70.55B, 80% of it in the MLP.
- **FLOPs:** training costs $6 \times \text{params}$ FLOPs per token (2 forward, 4 backward). That's 4.2e11 per token and 6.3e24 for a 15T-token run.
- **Time:** total FLOPs over what the cluster actually delivers:

  $$
  T = \frac{\text{total FLOPs}}{\text{chips} \times C \times \text{MFU}}
  $$

  MFU (model FLOPs utilization) is the fraction of peak we really get, and we assume 40% throughout. The 70B run is 435 years on one chip at peak, or 44 days on a full pod.
- **Memory:** 2 bytes per parameter for bf16 weights, 8 for Adam's two fp32 moments, and $2BD$ bytes for every saved copy of the activations. At a 4M-token batch that's about 11.2TB, 94% of it activation checkpoints, which fits on 117 chips. Meaning, we fill a pod because we need the FLOPs. Memory was never the problem.

### How to shard LLaMA 3-70B for training

Napkin math for TPU v5p: $C = 4.59 \times 10^{14}$ bf16 FLOPs/s, 96GB of HBM, 1.8e11 bytes/s of bidirectional ICI per link, 6.25e9 bytes/s of DCN per chip, and a full pod is 16x20x28 = 8,960 chips. So $\alpha = C / W_{\text{ici}} = 2550$.

With $B$ the total batch in tokens, $N$ chips, $X$-way FSDP over $M_X$ mesh axes and $Y$-way tensor parallelism (TP) over $M_Y$ axes, the Chapter 5 bounds are:

- **FSDP alone:** compute-bound when $B/N > \alpha / M_X$. With all 3 axes that's 850 tokens per chip.
- **TP alone:** compute-bound when $Y < M_Y F / \alpha$, about 11-way per axis for 70B, whatever the batch.
- **FSDP + TP:** compute-bound when $B/N > \alpha^2 / (M_X M_Y F)$, with the best split at $X_{\text{opt}} = \sqrt{(B/F)(M_X/M_Y)N}$. A 3D torus gives $M_X M_Y = 2$, so that's 113 tokens per chip for 70B.
- **Across pods:** pods talk over DCN with plain data parallelism, which is fine while each pod's batch is above $C / W_{\text{dcn}} = 73{,}440$ tokens.

FSDP ships weights, which don't shrink with the batch; TP ships activations, which do. So as the batch per chip gets smaller, we trade some FSDP for TP.

For 70B, the 4M batch is 1024 sequences of 4096 tokens, or 468 tokens per chip on a full pod. That's under the FSDP bound but over the mixed one, so we mix: roughly 2048-way FSDP with 4-way TP. Over 4 pods it's 117 tokens per chip, right on the mixed bound.

## Worked Problems

Questions 1 to 9 are the chapter's inline questions in order, and 10 and 11 are the two problems at the end, all on the TPU v5p numbers above. The memory questions (5 and 6) take the 4M batch as 4e6 tokens, and the sharding questions as $1024 \times 4096 = 4{,}194{,}304$.

**Question 1 [70B parameter count]:** From the config ($L = 80$, $D = 8192$, $F = 28672$, $N = 64$, $K = 8$, $H = 128$, $V = 128256$), can we work out the LLaMA 3-70B parameter count and actually land on 70B?

- **MLP:** 3 matrices of $D \times F$ per layer (gate, up and down):

  $$
  3DFL = 3 \times 8192 \times 28672 \times 80 = 56{,}371{,}445{,}760
  $$

- **Attention:** Q and O are $D \times N \times H$ each, and K and V are only $D \times K \times H$:

  $$
  L(2DNH + 2DKH) = 80 \times (134{,}217{,}728 + 16{,}777{,}216) = 12{,}079{,}595{,}520
  $$

- **Vocab:** separate input and output embeddings, so $2VD = 2 \times 128256 \times 8192 = 2{,}101{,}346{,}304$.
- Adding them up:

  $$
  56.37 \times 10^{9} + 12.08 \times 10^{9} + 2.10 \times 10^{9} = \mathbf{70.55 \times 10^{9} \text{ parameters}}
  $$

- So yes, we land on 70B, with the MLP at 80%. The solution says 70.4e9 because it rounds each bucket before adding.

**Question 2 [FLOPs per token]:** How many FLOPs does LLaMA 3-70B perform per token in one training step?

- 6 FLOPs per parameter per token (Chapter 4), rounding to 70B:

  $$
  6 \times 70 \times 10^{9} = \mathbf{4.2 \times 10^{11} \text{ FLOPs per token}}
  $$

**Question 3 [total training FLOPs]:** LLaMA 3 was trained on about 15 trillion tokens. How many FLOPs is that in total?

- Per-token cost times tokens:

  $$
  4.2 \times 10^{11} \times 15 \times 10^{12} = \mathbf{6.3 \times 10^{24} \text{ FLOPs}}
  $$

- One v5p at peak would need $6.3 \times 10^{24} / 4.59 \times 10^{14} = 1.37 \times 10^{10}$ seconds, about **435 years**. So, more chips.

**Question 4 [one pod at 40% MFU]:** Say we train on a full TPU v5p pod, 16x20x28 = 8960 chips. How long does the run take at 40% MFU in bfloat16, assuming we're compute-bound?

- We just divide:

  $$
  T = \frac{6.3 \times 10^{24} \text{ FLOPs}}{8960 \times 4.59 \times 10^{14} \times 0.4 \text{ FLOPs/s}} = 3.83 \times 10^{6} \text{ s} = \mathbf{44.3 \text{ days}}
  $$

- About a month and a half, IF we hold 40% MFU and stay compute-bound. Questions 7 to 9 are about that second "if".

**Question 5 [minimum chips for a 4M batch]:** LLaMA 3-70B was pretrained with a batch of about 4M tokens. What's the minimum number of TPU v5p chips we need to train at that batch size? Assume bfloat16 parameters, float32 optimizer state, and activations checkpointed 2 times per layer (at the start of the layer and after attention).

- This is a memory question. Too few FLOPs just makes the run slow; too little HBM means it doesn't run at all. Three things live in HBM:
  - **Parameters:** 2 bytes each in bf16, so $2 \times 70 \times 10^{9} = 140\text{ GB}$.
  - **Optimizer state:** Adam's 2 fp32 moments, $8 \times 70 \times 10^{9} = 560\text{ GB}$.
  - **Activation checkpoints:** each is a `bf16[B, D]` copy of the residual stream, $2 \times 4 \times 10^{6} \times 8192 = 65.5\text{ GB}$, and 2 per layer over 80 layers is 160 of them, $10.49\text{ TB}$.
- That's $0.14 + 0.56 + 10.49 = 11.19$ TB, against 96GB per chip:

  $$
  \frac{11.19 \times 10^{12} \text{ bytes}}{96 \times 10^{9} \text{ bytes/chip}} = 116.5 \implies \mathbf{117 \text{ chips}}
  $$

- We need at least **117 TPUs**, though the run would take about 9.3 years on them.

**Question 6 [memory per chip on a full pod]:** Same assumptions as Question 5, but now on all 8960 TPU v5p chips. How much memory do we use per chip?

- Same 11.19TB, just spread thinner, since FSDP shards the weights and optimizer state instead of copying them:

  $$
  \frac{11.19 \times 10^{12} \text{ bytes}}{8960 \text{ chips}} = \mathbf{1.25 \text{ GB per chip}}
  $$

- That's 1.3% of each chip's HBM. Tiny.

**Question 7 [FSDP alone, no sequence sharding]:** Keep the same setup: LLaMA 3-70B, a 4M-token batch made of 1024 sequences of length 4096, on a TPU v5p pod of 8960 chips. Can we train with FSDP alone if we aren't allowed any sequence (context) parallelism? Hint: this should be the first thing we try, since it's the simplest scheme and adds no extra communication if it works.

- With no sequence sharding, the smallest piece we can hand a chip is one whole sequence, and we only have 1024 of them for 8960 chips.
- Those 1024 chips would run fine (4096 tokens each clears even the single-axis bound of 2550), but the other 7,936 sit idle and the run takes 388 days at 40% MFU.
- So **no**, not if we want the whole pod.

**Question 8 [FSDP over batch and sequence]:** Now let FSDP shard over both the batch and the sequence axes. Is FSDP by itself enough to train LLaMA 3-70B on all 8960 chips?

- Splitting sequences across chips, every chip gets work:

  $$
  \frac{B}{N} = \frac{4{,}194{,}304 \text{ tokens}}{8960 \text{ chips}} = 468 \text{ tokens per chip}
  $$

- With all 3 axes on FSDP the bound is $2550 / 3 = 850$, so $T_{\text{math}} / T_{\text{comms}} = 468 \times 3 / 2550 = 0.55$. Meaning, the weight AllGathers take about 1.8 times as long as the math.
- So **no**, pure FSDP is communication bound on a full pod. It would want a batch of at least $850 \times 8960 \approx 7.6\text{M}$ tokens.

**Question 9 [mixed FSDP + tensor parallelism]:** Now mix tensor parallelism with FSDP. Is there a combination that keeps us compute-bound? If so, how much FSDP and how much TP should we use?

- The mixed bound is $2550^2 / (2 \times 28672) = 113$ tokens per chip, and we have 468, about 4 times that. So yes.
- For the split, with 2 axes for FSDP and 1 for TP:

  $$
  X_{\text{opt}} = \sqrt{\frac{4{,}194{,}304}{28672} \times 2 \times 8960} \approx 1619
  $$

- That leaves $Y = 8960 / 1619 \approx 5.5$. Rounding both to powers of two gives 2048-way FSDP (1024 over the batch, 2 over the sequence) and 4-way TP.
- Checking both sides: FSDP gives $(4{,}194{,}304 / 2048) \times 2 / 2550 = 1.61$ and TP gives $28672 / (4 \times 2550) = 2.81$. Both are above 1, so both comms hide behind the math.
- So we use roughly **2048-way FSDP and 4-way tensor parallelism**.

**Question 10 [scaling LLaMA 70B to more chips]:** Say we train LLaMA 3-70B on 4 TPU v5p pods with the same 4M-token batch. What parallelism scheme do we use? Are we compute or communication bound? Roughly how long does training take? Hint: make sure to use the correct roofline bound.

- $4 \times 8960 = 35{,}840$ chips. Each pod gets $4{,}194{,}304 / 4 = 1{,}048{,}576$ tokens (256 sequences), and each chip gets 117.
- **Across pods:** ICI stops at the pod edge, so the pods do data parallelism over DCN. That needs 73,440 tokens per pod and we have about 1.05M.
- **Inside a pod:** this is where the hint matters. Against the FSDP bound (850), 117 looks hopeless, but the right bound is the mixed one, 113, and $117 > 113$.
- A quarter of the batch per pod halves $X_{\text{opt}}$ and doubles $Y$:

  $$
  X_{\text{opt}} = \sqrt{\frac{1{,}048{,}576}{28672} \times 2 \times 8960} \approx 810, \qquad Y = \frac{8960}{810} \approx 11
  $$

- We round $Y$ down to 8, so $X = 1120$. That keeps the TP side safe (ratio 1.41), which matters because TP gathers move the layer's own input and can't start early. The FSDP gathers then take about 1.36 times as long as the math at peak, which caps MFU at about 73%, still above our 40%.
- So the scheme is **4-way data parallelism across the pods over DCN, and inside each pod about 1120-way FSDP with 8-way TP**, with FSDP cutting across the sequence too.
- At 40% MFU:

  $$
  T = \frac{6.3 \times 10^{24} \text{ FLOPs}}{35{,}840 \times 4.59 \times 10^{14} \times 0.4 \text{ FLOPs/s}} = 9.57 \times 10^{5} \text{ s} = \mathbf{11.1 \text{ days}}
  $$

- So about **11 days**, compute-bound, a quarter of the single-pod 44 days.

**Question 11 [LLaMA 405B]:**

**(a)** Using the LLaMA 3-405B [config](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-405B/blob/main/config.json), list the same hyperparameters as we did for 70B. What's the total parameter count? How many FLOPs is one training step? And how many FLOPs is a 15T-token training run?

The config gives $L = 126$, $D = 16384$, $F = 53248$, $N = 128$, $K = 8$, $H = 128$ and $V = 128256$. Compared to 70B, $D$ doubles, $F$ is now $3.25D$, and the KV heads and vocab stay the same.

* **How many total parameters?**
  - Same three buckets as Question 1. MLP: $3DFL = 3 \times 16384 \times 53248 \times 126 = 329{,}772{,}957{,}696$.
  - Attention: $L(2DNH + 2DKH) = 126 \times (536{,}870{,}912 + 33{,}554{,}432) = 71{,}873{,}593{,}344$.
  - Vocab: $2VD = 2 \times 128256 \times 16384 = 4{,}202{,}692{,}608$.
  - Adding them up:

    $$
    329.77 \times 10^{9} + 71.87 \times 10^{9} + 4.20 \times 10^{9} = \mathbf{405.8 \times 10^{9} \text{ parameters}}
    $$

* **How many FLOPs per training step?**
  - Same rule, rounding to 405B:

    $$
    6 \times 405 \times 10^{9} = \mathbf{2.43 \times 10^{12} \text{ FLOPs per token}}
    $$

  - At a 4,194,304-token batch, one step is about $1.02 \times 10^{19}$ FLOPs.
* **How many FLOPs for 15T tokens?**

    $$
    2.43 \times 10^{12} \times 15 \times 10^{12} = 3.645 \times 10^{25} \approx \mathbf{3.6 \times 10^{25} \text{ FLOPs}}
    $$

**(b)** Now assume we train on 8 TPU v5p pods. What parallelism scheme do we use? How long does training take? Are we compute or comms bound?

No batch size, MFU or checkpointing is given for 405B, so we keep the 70B setup: a 4,194,304-token batch, 40% MFU and 2 checkpoints per layer.

* **How long would training take?**
  - 8 pods is $8 \times 8960 = 71{,}680$ chips. At 40% MFU:

    $$
    T = \frac{3.645 \times 10^{25} \text{ FLOPs}}{71{,}680 \times 4.59 \times 10^{14} \times 0.4 \text{ FLOPs/s}} = 2.77 \times 10^{6} \text{ s} = \mathbf{32 \text{ days}}
    $$

* **What parallelism scheme, and are we compute or comms bound?**
  - Same outer layer as Question 10: data parallelism across the pods over DCN, FSDP + TP inside each pod. Each pod's 524,288 tokens clear the DCN bound easily.
  - The bigger $F$ lowers the mixed bound to $2550^2 / (2 \times 53248) = 61$ tokens per chip. At 4,194,304 tokens each chip gets 58.5, just under, so we're a hair communication bound ($T_{\text{math}} / T_{\text{comms}} = 0.98$ at the best split). Keeping the TP side above 1 like in Question 10, we'd use $560 \times 16$ per pod.
  - The [LLaMA 3 paper](https://arxiv.org/abs/2407.21783) grew the 405B batch from 4M to 16M during pre-training. At 16,777,216 tokens each chip gets 234, almost 4 times the bound, so $1120 \times 8$ is comfortably compute-bound.
  - So: **data parallelism over DCN across the 8 pods, and FSDP + TP inside each pod**. Slightly comms bound at a 4M batch, compute-bound at 16M, and roughly **32 days** at 40% MFU either way.
