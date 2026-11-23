---
title: How to Parallelize a Transformer for Training
description: Overview and Worked Problems
published: true
---
## 5. Parallelizing Training

Chapter 3 was about one sharded matmul and Chapter 4 about the FLOPs of one transformer. Now we put a whole training step on thousands of chips and ask when that stops being efficient.

"Scaling" has a specific meaning here: if we double the chips, we want double the throughput, which is called strong scaling. On a single chip the fight was FLOPs against HBM bandwidth; across chips, it's FLOPs against the network. The trick is the same overlapping from Chapter 2, where the communication hides behind useful math. But adding chips hurts twice: there's more to communicate, and each chip holds a smaller slice of the batch, so there's less math to hide it behind. So for every scheme below we ask the same thing: when does $T_{\text{comms}}$ overtake $T_{\text{math}}$?

### The Setup

To keep the algebra small, we pretend the transformer is just a stack of MLP blocks. The MLP dominates the FLOPs as long as $T < 8D$ (Chapter 4), so we ignore attention, and we drop the gating matrix too. Each layer is then two matmuls:

$$
\text{In}[B, D] \cdot_D W_{\text{in}}[D, F] \cdot_F W_{\text{out}}[F, D] \to \text{Out}[B, D]
$$

A few conventions for the chapter:

- $B$ is the TOTAL batch size in tokens. Not sequences, and not per device.
- $C$ is FLOPs/s per chip and $W$ is the bidirectional network bandwidth ($W_{\text{ici}}$ or $W_{\text{dcn}}$).
- $X$ and $Y$ are the number of chips along the mesh axes we shard over, and $M_X$, $M_Y$ count how many physical mesh axes each one spans. More axes means more links, so comms time drops by roughly $1/M$.

For a TPU v5p, $C = 4.59 \times 10^{14}$ FLOPs/s and $W_{\text{ici}} = 1.8 \times 10^{11}$ bytes/s, so $C / W_{\text{ici}} = 2550$. Basically every rule below is built out of that one number.

Each scheme is then just a choice of how to shard In, $W_{\text{in}}$, $W_{\text{out}}$ and Out.

> **Note:** The MLP doesn't care which sequence a token belongs to, so "batch" sharding can also cut along the sequence (context parallelism). Attention needs extra care when we do that, which we ignore here.

### 1. Data Parallelism

$$
\text{In}[B_X, D] \cdot_D W_{\text{in}}[D, F] \cdot_F W_{\text{out}}[F, D] \to \text{Out}[B_X, D]
$$

We split the batch across chips and give every chip a full copy of the weights. No contracting dimension is sharded, so the forward pass needs zero communication (Case 1 from Chapter 3). The bill comes in the backward pass: each chip only has the gradient from its own slice of the batch, so we AllReduce the gradient of each weight matrix before the update. Nicely, nothing later in the backward pass needs the summed gradients; only the weight update at the end of the step does. So these AllReduces sit off the critical path and can be fitted in around the matmuls.

Per layer in the backward pass, that's two AllReduces, each over $2DF$ bytes (bf16) and each costing twice an AllGather, against four matmuls of $2(B/X)DF$ FLOPs:

$$
T_{\text{comms}} = \frac{2 \cdot 2 \cdot 2DF}{W_{\text{ici}}} = \frac{8DF}{W_{\text{ici}}} \qquad T_{\text{math}} = \frac{4 \cdot 2(B/X)DF}{C} = \frac{8BDF}{X \cdot C}
$$

We stay compute-bound while $T_{\text{math}} > T_{\text{comms}}$, and nearly everything cancels:

$$
\frac{B}{X} > \frac{C}{W_{\text{ici}}}
$$

So the per-device batch has to beat 2550 tokens on a v5p. It's the same shape as the single-chip $B > 240$ rule from Chapter 1, and for the same reason: the math grows with the batch, while the bytes we move (weights) don't. Data parallelism can also run over all three mesh axes at once, which triples the bandwidth and drops the bar to $2550 / 3 = 850$ tokens per chip. A full v5p pod has 8960 chips, so the whole pod needs a batch of about 7.6M tokens. That's not a hard bar to clear.

The real problem is memory. bf16 parameters (2 bytes) plus Adam's two fp32 accumulators (8 bytes) come to 10 bytes per parameter, and that's replicated on EVERY chip. With 96GB of HBM we cap out at $96 \times 10^9 / 10 = 9.6\text{B}$ parameters (the book rounds this down to about 9B), and that's before storing a single activation. So pure data parallelism only works for small models.

### 2. FSDP

FSDP (fully-sharded data parallelism, also called ZeRO-3) keeps the batch sharding and shards the weights and optimizer state along the same axis:

$$
\text{In}[B_X, D] \cdot_D W_{\text{in}}[D_X, F] \cdot_F W_{\text{out}}[F, D_X] \to \text{Out}[B_X, D]
$$

Now the weights have a sharded contracting dimension, so each layer's weights get AllGathered right before they're used (we can prefetch them while the previous layer runs) and dropped right after. On the way back we ReduceScatter the gradients, and each chip updates only its own shard of the optimizer state. Remember $\text{AllReduce} = \text{ReduceScatter} + \text{AllGather}$ from Chapter 3: FSDP takes the AllReduce that data parallelism was already paying for and uses its two halves separately. The backward pass still moves the same bytes per FLOP, and the forward pass now moves some too (in the same proportion), but there's no duplicated memory and no duplicated optimizer work.

Modeling the forward pass this time (two AllGathers of $2DF$ bytes, two matmuls):

$$
T_{\text{comms}} = \frac{4DF}{W_{\text{ici}}} \qquad T_{\text{math}} = \frac{4BDF}{X \cdot C}
$$

Same ratio, same rule: $B/X > 2550 / M_X$. Meaning, any setup that's compute-bound with data parallelism is still compute-bound with FSDP, and we get all that replicated memory back for free.

One thing that feels backwards: with a fixed number of chips, a SMALLER batch makes us more communication bound. And the batch isn't a free knob, since past some size more tokens per step stop helping training. So in practice we usually start with the batch size already decided and a big pile of chips, and the job is to find a sharding that fits one onto the other.

### 3. Tensor Parallelism

$$
\text{In}[B, D_Y] \cdot_D W_{\text{in}}[D, F_Y] \cdot_F W_{\text{out}}[F_Y, D] \to \text{Out}[B, D_Y]
$$

Instead of moving weights, we move activations. The weights are sharded along $F$ and never travel. The activations are sharded along $D$, so we AllGather In before the first matmul, run both matmuls locally (which leaves Out unreduced along $Y$), and ReduceScatter Out at the end. That's one gather and one scatter per layer, not an AllReduce per matrix. The catch is that these ARE on the critical path: the layer can't start until the gather lands.

$$
T_{\text{comms}} = \frac{2 \cdot 2BD}{W_{\text{ici}}} \qquad T_{\text{math}} = \frac{4BDF}{Y \cdot C}
$$

This time $B$ cancels and we get a limit on $Y$ itself:

$$
Y < M_Y \cdot \frac{F}{C / W_{\text{ici}}} = M_Y \cdot \frac{F}{2550}
$$

So tensor parallelism doesn't care about batch size at all. It has a hard ceiling set by how wide the MLP is. With $F \approx 30{,}000$ the ceiling is about 11.8 per axis, so 8-way works and 16-way doesn't (8-way needs $F > 20{,}400$). Typical FFW widths put the ceiling somewhere between 8-way and 16-way. Precision doesn't move it either: going to int8 doubles $C$, but each activation is also half as many bytes.

### 4. Mixing FSDP and Tensor Parallelism

$$
\text{In}[B_X, D_Y] \cdot_D W_{\text{in}}[D_X, F_Y] \cdot_F W_{\text{out}}[F_Y, D_X] \to \text{Out}[B_X, D_Y]
$$

> FSDP pays its comms in weights. Tensor parallelism pays in activations.

The two help each other. Each chip now sees only $B/X$ tokens, so tensor parallelism's activation gathers get cheaper as $X$ grows. And each chip's slice of a weight matrix is only $F/Y$ wide, so FSDP's weight gathers get cheaper as $Y$ grows. With $N = XY$ chips in total:

$$
T_{\text{FSDP}} = \frac{4DF}{Y \cdot W_{\text{ici}} \cdot M_X} \qquad T_{\text{TP}} = \frac{4BD}{X \cdot W_{\text{ici}} \cdot M_Y} \qquad T_{\text{math}} = \frac{4BDF}{N \cdot C}
$$

The math time doesn't depend on how we split $N$; only the comms do. If the two kinds of comms overlap with each other, the total is the max of the two. One grows with $X$ and the other shrinks, so the best split is where they're equal:

$$
X_{\text{opt}} = \sqrt{\frac{B}{F} \cdot \frac{M_X}{M_Y} \cdot N}
$$

Plugging $X_{\text{opt}}$ back in, we're compute-bound when:

$$
\frac{B}{N} > \frac{(C / W_{\text{ici}})^2}{M_X M_Y F}
$$

For $F = 32{,}768$ and $M_X M_Y = 2$ that's about 99 tokens per chip, against 850 for FSDP alone: roughly 8.6 times smaller. It also helps to remember how each ratio $T_{\text{math}} / T_{\text{comms}}$ scales: with $B$ for FSDP, with $\sqrt{B}$ for the mix, and not at all for tensor parallelism. So at huge batches plain FSDP has the most headroom. Under 850 tokens per chip it goes comms-bound, and from there down to about 100 the mix is the only scheme still compute-bound. The plots on [jax-ml.github.io/scaling-book/training/](https://jax-ml.github.io/scaling-book/training/) show this well.

### Pipelining

The last scheme shards along the layer dimension: chip 0 holds the first few layers, runs them, hands the activations to chip 1, and so on, then the gradients come back the other way. Very little has to move, just the activations at each stage boundary going one hop to the next chip, which is why it's popular on GPU clusters where the links between nodes are weaker. The price is the pipeline bubble: the first stage sits idle while everyone downstream works. The usual fixes are microbatching (feed several small batches so every stage stays busy) and interleaving the forward and backward matmuls on each stage. TPU pods are big enough that the book only sketches it, and so will I.

### On Crossing Pods

The biggest v5p slice is a full pod, 8960 chips. Past that we have to go over DCN, which gives each TPU about $6.25 \times 10^9$ bytes/s (25 GB/s per host, shared by 4 chips). The usual recipe is FSDP and tensor parallelism inside each pod, and pure data parallelism across pods. The gradient AllReduce over DCN gets faster as the slice grows, since every chip brings its own share of NIC bandwidth, so the rule ends up being per slice:

$$
\frac{B}{\text{slice}} > \frac{C}{W_{\text{dcn}}} = \frac{4.59 \times 10^{14}}{6.25 \times 10^{9}} = 73{,}440 \text{ tokens}
$$

For a full pod that's about 8 tokens per chip. Tiny. So DCN is rarely the thing that stops us; the ICI rules inside the pod bite first.

As a note, the book's closing takeaway list gives this bar as "about 71,000", but its own division in the cross-pod section (and the 73k in that section's takeaway) gives 73,440, so that's the number used here.

An example with LLaMA-3 70B ($F \approx 30{,}000$) and a 2M token batch: FSDP alone tops out around 2,350 chips ($2 \times 10^6 / 850$), about a quarter of a pod. The mix needs $B/N > 2550^2 / (2 \times 30{,}000) \approx 108$, which allows roughly 18,450 chips, more than one pod holds. So we'd take two pods with 1M tokens each: about 112 tokens per chip (just over the 108 bar) and 1M per pod, way above 73,440.

### Takeaway

| Scheme | What travels | Compute-bound when (v5p) |
|---|---|---|
| Data parallelism | weight gradients (AllReduce, backward only) | $B/X > 2550 / M_X$ |
| FSDP | weights (AllGather) and gradients (ReduceScatter) | $B/X > 2550 / M_X$ |
| Tensor parallelism | activations (AllGather, then ReduceScatter) | $Y < M_Y \cdot F / 2550$ |
| FSDP + tensor parallelism | both | $B/N > 2550^2 / (M_X M_Y F)$ |
| Data parallelism across pods | weight gradients over DCN | $B / \text{slice} > 73{,}440$ |

The common thread: comms hide behind per-chip compute, and both adding chips and shrinking the batch take that compute away. With a big batch or a small model there isn't much to decide; we use data parallelism if the model fits on one chip at 10 bytes per parameter, and FSDP if it doesn't. Where it gets hard is a small batch on a lot of chips. There we add tensor parallelism up to its 8 to 16-way ceiling and let FSDP take the rest. I skipped the backward-pass derivations; they follow from the same sharded matmul rules as Chapter 3, and the book walks through them in its Appendix A.

## Worked Problems

All three problems use LLaMA-2 13B:

| $L$ | $D$ | $F$ | $N$ | $K$ | $H$ | $V$ |
|---|---|---|---|---|---|---|
| 40 | 5,120 | 13,824 | 40 | 40 | 128 | 32,000 |

It has separate embedding and output matrices and a gated MLP block. For hardware we use the TPU v5p numbers from the Chapter 2 references: 96GB of HBM per chip, 4.59e14 bf16 FLOPs/s, and 1.8e11 bytes/s of bidirectional ICI per axis.

**Question 1 [counting parameters]:** How many parameters does LLaMA-2 13B have? (Yes, the answer is in the name, but do the math.) Hint: with a gated MLP there are 3 big FFW matrices per layer, two up-projections and one down-projection. The notes above ignored the gating matrix, but it behaves just like $W_{\text{in}}$.

- This is just the Chapter 4 accounting applied to a real config. Three buckets: MLP, attention and vocab.
- **MLP:** three matrices of $D \times F$ in each of the $L$ layers:

  $$
  3LDF = 3 \times 40 \times 5120 \times 13824 = 8{,}493{,}465{,}600
  $$

- **Attention:** the Q, K, V and output projections. Since $N = K$ and $N \times H = 5120 = D$, each one is $D \times NH$:

  $$
  4DNHL = 4 \times 5120 \times 40 \times 128 \times 40 = 4{,}194{,}304{,}000
  $$

- **Vocab:** the embedding and the output matrix are separate, so we count $V \times D$ twice:

  $$
  2VD = 2 \times 32000 \times 5120 = 327{,}680{,}000
  $$

- Add them up:

  $$
  8{,}493{,}465{,}600 + 4{,}194{,}304{,}000 + 327{,}680{,}000 = \mathbf{13{,}015{,}449{,}600 \text{ parameters}}
  $$

- So **13.0B**, as the name says. The MLP is about 65% of it, attention 32%, and the vocab matrices under 3%. We left out the norm scales, but with two vectors of size $D$ per layer plus a final one that's only $(2L + 1) \times D = 414{,}720$ more, a rounding error.

**Question 2 [training memory]:** Say we train with a batch of 16M tokens using Adam. Ignoring parallelism for now, how much total memory do the model's parameters, optimizer state and activations take? Assume bf16 parameters, fp32 optimizer state, and activations checkpointed three times per layer (after each of the three big FFW matmuls).

- This question is about which of the three actually fills up memory during training. We count bytes per parameter first, then bytes per token.
- Parameters are bf16 (2 bytes each). Adam keeps a first and a second moment for every parameter, both fp32 (4 bytes each). So 10 bytes per parameter:

  $$
  (2 + 4 + 4) \times 13{,}015{,}449{,}600 = 130{,}154{,}496{,}000 \text{ bytes} \approx \mathbf{130 \text{ GB}}
  $$

- For activations: the two up-projections each leave a $[B, F]$ tensor and the down-projection leaves a $[B, D]$ tensor, all in bf16. Per layer that's $2B(D + 2F)$ bytes. Conveniently $D + 2F = 5120 + 27648 = 32768$, so each token costs 65,536 bytes per layer.

  $$
  2 \cdot L \cdot B \cdot (D + 2F) = 2 \times 40 \times 16 \times 10^6 \times 32768 \approx 4.19 \times 10^{13} \text{ bytes} = \mathbf{41.9 \text{ TB}}
  $$

- The total is around **42.1 TB**, and basically all of it is activations: they're about 320 times bigger than the parameters and optimizer state combined.
- What this means: at training batch sizes the model itself is a rounding error in memory. It would take about 438 v5p chips' worth of HBM just to hold everything, which is why sharding the batch is the first thing we reach for.

**Question 3 [choosing a sharding]:** We want to train with a 32k sequence length and a total batch size of 3M tokens on a TPU v5p `16x16x16` slice, with bf16 weights and an fp32 optimizer as above. Assume gradient checkpoints only after the 3 big FFW matrices.

- Some numbers first. The slice has $16 \times 16 \times 16 = 4096$ chips and $4096 \times 96\text{ GB} \approx 393\text{ TB}$ of HBM in total.
- Tokens per chip: $3 \times 10^6 / 4096 \approx 732$. Everything below comes down to comparing this number against a threshold.
- Also, 3M tokens at 32k each is only about 92 to 94 sequences, depending on whether 32k means 32,768 or 32,000 (the solution says 96, taking both numbers as powers of two). Either way we have far fewer sequences than chips. Each sequence ends up spread over around 44 chips, which is the context parallelism from the note above.

**Does pure data parallelism work here? Why or why not?**

- No. Data parallelism puts a full copy of the parameters and optimizer state on every chip, and from Question 2 that's 130 GB against 96 GB of HBM. It doesn't fit, even before a single activation.
- It would also be comms-bound, by the same check as the next part.

**Can pure FSDP do it without being comms-bound? Why or why not? How much memory does each device use with pure FSDP?**

- Memory first. Swapping $B$ for $3 \times 10^6$ in Question 2:

  $$
  2 \times 40 \times 3 \times 10^6 \times 32768 \approx 7.86 \times 10^{12} \text{ bytes}
  $$

- Adding the 130 GB of parameters and optimizer state gives $7.99 \times 10^{12}$ bytes, so basically 8 TB, around 2% of the slice's 393 TB. FSDP shards all of it with no duplication, so each chip holds 1/4096 of the total:

  $$
  \frac{7.99 \times 10^{12} \text{ bytes}}{4096 \text{ chips}} \approx \mathbf{1.95 \text{ GB per chip}}
  $$

- Each chip also briefly holds the gathered weights of the layer it's working on ($3 \times 2DF \approx 0.42\text{ GB}$ for the three FFW matrices, a bit more with the attention projections). Either way, memory is not the problem.
- Now comms. With all three axes on FSDP we need $B/N > 2550 / 3 = 850$ tokens per chip, and we only have 732:

  $$
  \frac{T_{\text{math}}}{T_{\text{comms}}} = \frac{B/N}{850} = \frac{732}{850} \approx 0.86
  $$

- So **no**, pure FSDP is comms-bound here. Plugging into the two-matrix forward-pass formulas from the notes, each layer's matmuls come to about 452 $\mu\text{s}$ against about 524 $\mu\text{s}$ for the weight AllGathers. To be compute-bound on 4096 chips we'd need a batch of at least $850 \times 4096 \approx 3.48\text{M}$ tokens, and we're at 3M. Close, but on the wrong side.

**Does mixed FSDP + tensor parallelism work? Why or why not? If so, what should $X$ and $Y$ be, and how much memory sits on each device? Using only roofline FLOPs estimates and ignoring attention, how long is each training step at 40% MFU?**

- The bar for the mix is $B/N > (C / W_{\text{ici}})^2 / (M_X M_Y F)$. With two axes on FSDP and one on tensor parallelism, $M_X M_Y = 2$:

  $$
  \frac{2550^2}{2 \times 13824} \approx 235 \text{ tokens per chip}
  $$

- We have 732, so **yes**, we can be compute-bound. (The bar is higher than the 99 from the notes because this model's $F$ is smaller.)
- For the split, we use the $X_{\text{opt}}$ rule:

  $$
  X_{\text{opt}} = \sqrt{\frac{B}{F} \cdot \frac{M_X}{M_Y} \cdot N} = \sqrt{\frac{3 \times 10^6}{13824} \times 2 \times 4096} \approx 1333
  $$

- That leaves $Y_{\text{opt}} = 4096 / 1333 \approx 3.1$. Rounding to powers of two, we get $\mathbf{X = 1024}$ for FSDP and $\mathbf{Y = 4}$ for tensor parallelism. $Y = 4$ also sits under the tensor parallelism ceiling of $F / 2550 \approx 5.4$, which $Y = 8$ wouldn't.
- Checking the per-layer times for that split (forward pass, same two-matrix picture):

  $$
  T_{\text{math}} = \frac{4BDF}{N \cdot C} \approx 452 \ \mu\text{s} \qquad T_{\text{FSDP}} = \frac{4DF}{Y \cdot W_{\text{ici}} \cdot M_X} \approx 197 \ \mu\text{s} \qquad T_{\text{TP}} = \frac{4BD}{X \cdot W_{\text{ici}} \cdot M_Y} \approx 333 \ \mu\text{s}
  $$

- Both comms times sit under the math time, so we're compute-bound, with a margin of about 1.36 on the tighter (tensor parallelism) side. The neighbors are worse: $2048 \times 2$ only clears by 1.15 on the FSDP side, and $512 \times 8$ is comms-bound on tensor parallelism (0.68).
- One caveat: the formula treats $X$ and $Y$ as free numbers. Fitting $1024 \times 4$ onto a real `16x16x16` torus means cutting one axis of 16 into groups of 4, so counting $M_X = 2$ and $M_Y = 1$ is only approximate. (Splitting along whole axes would mean $256 \times 16$, which is badly comms-bound on the tensor parallelism side, at about 0.34.) The check also assumes the two kinds of comms overlap with each other, like the notes do; back to back they'd add up to about 530 $\mu\text{s}$, more than the math.
- Memory per device is the same as with pure FSDP, about **1.95 GB per chip**. Nothing is replicated in either scheme, so it's still the 8 TB total divided by 4096.
- For the step time, roofline FLOPs for training are $6 \times \text{tokens} \times \text{parameters}$ (Chapter 4), and the slice delivers 40% of its peak:

  $$
  T_{\text{step}} = \frac{6 \times 3 \times 10^6 \times 13.015 \times 10^9}{4096 \times 4.59 \times 10^{14} \times 0.4} = \frac{2.34 \times 10^{17} \text{ FLOPs}}{7.52 \times 10^{17} \text{ FLOPs/s}} \approx \mathbf{0.31 \text{ s}}
  $$

- Each step takes around **312 ms**, which is about 9.6M tokens per second. The solution says 300 ms. It plugs in 13e9 parameters and 4.6e14 FLOPs/s, which comes to about 310 ms, and then rounds that down.
- As a note, "ignoring attention" hides a lot at this context length. The Chapter 4 ratio of dot-product attention FLOPs to matmul FLOPs is $12T / (18F + 24D)$ when $NH = D$, which comes to about 1.06 at $T = 32{,}768$ (half of that with causal masking). The $T / 8D$ shortcut would say 0.8, but it assumes $F = 4D$ and this model has $F = 2.7D$. So counting those FLOPs too, the roofline estimate grows to about 1.5 to 2 times as long: roughly 0.47 s with causal masking and 0.63 s without.
