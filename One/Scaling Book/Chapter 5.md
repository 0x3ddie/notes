---
title: How to Parallelize a Transformer for Training
description: Overview and Worked Problems
published: true
---
## 5. Parallelizing Training

Chapter 3 was one sharded matmul and Chapter 4 was the FLOPs of one transformer. Now we spread a whole training step over thousands of chips.

We want strong scaling: double the chips, double the throughput. On one chip the fight was FLOPs against HBM bandwidth; across chips it's FLOPs against the network. Adding chips hurts twice: more to communicate, and a smaller slice of the batch per chip, so less math to hide the comms behind. For every scheme below we ask one question: when does $T_{\text{comms}}$ overtake $T_{\text{math}}$?

### The Setup

To keep the algebra small, we pretend the transformer is just a stack of MLP blocks. The MLP dominates the FLOPs as long as $T < 8D$ (Chapter 4), so we ignore attention and drop the gating matrix. Each layer is two matmuls:

$$
\text{In}[B, D] \cdot_D W_{\text{in}}[D, F] \cdot_F W_{\text{out}}[F, D] \to \text{Out}[B, D]
$$

- $B$ is the TOTAL batch size in tokens. Not sequences, and not per device.
- $C$ is FLOPs/s per chip and $W$ is the bidirectional network bandwidth ($W_{\text{ici}}$ or $W_{\text{dcn}}$).
- $X$ and $Y$ are how many chips we shard over, and $M_X$, $M_Y$ count the physical mesh axes each one spans. More axes means more links, so comms time drops by roughly $1/M$.

For a TPU v5p, $C = 4.59 \times 10^{14}$ FLOPs/s and $W_{\text{ici}} = 1.8 \times 10^{11}$ bytes/s, so $C / W_{\text{ici}} = 2550$. Basically every rule below is built out of that one number.

> **Note:** The MLP doesn't care which sequence a token belongs to, so "batch" sharding can also cut along the sequence (context parallelism). Attention needs extra care there, which we ignore.

### 1. Data Parallelism

$$
\text{In}[B_X, D] \cdot_D W_{\text{in}}[D, F] \cdot_F W_{\text{out}}[F, D] \to \text{Out}[B_X, D]
$$

We split the batch and give every chip a full copy of the weights. No contracting dimension is sharded, so the forward pass needs zero communication (Case 1 from Chapter 3). The bill comes in the backward pass: each chip only has the gradient from its own slice of the batch, so we AllReduce each weight gradient before the update. Only the update at the very end needs them, so these AllReduces sit off the critical path.

Per layer, that's two AllReduces of $2DF$ bytes (each costing two AllGathers) against four matmuls of $2(B/X)DF$ FLOPs:

$$
T_{\text{comms}} = \frac{2 \cdot 2 \cdot 2DF}{W_{\text{ici}}} = \frac{8DF}{W_{\text{ici}}} \qquad T_{\text{math}} = \frac{4 \cdot 2(B/X)DF}{C} = \frac{8BDF}{X \cdot C}
$$

We're compute-bound while $T_{\text{math}} > T_{\text{comms}}$, and nearly everything cancels:

$$
\frac{B}{X} > \frac{C}{W_{\text{ici}}}
$$

Meaning, the per-device batch has to beat 2550 tokens on a v5p. Same shape as the $B > 240$ rule from Chapter 1: the math grows with the batch, the weight bytes don't. Over all three mesh axes the bar drops to $2550 / 3 = 850$ tokens per chip, so a full 8960-chip pod needs a batch of about 7.6M tokens. Not hard.

The real problem is memory. bf16 parameters (2 bytes) plus Adam's two fp32 accumulators (8 bytes) is 10 bytes per parameter, on EVERY chip. With 96GB of HBM that caps us at $96 \times 10^9 / 10 = 9.6\text{B}$ parameters before a single activation. Small models only.

### 2. FSDP

FSDP (fully-sharded data parallelism, or ZeRO-3) keeps the batch sharding and shards the weights and optimizer state along the same axis:

$$
\text{In}[B_X, D] \cdot_D W_{\text{in}}[D_X, F] \cdot_F W_{\text{out}}[F, D_X] \to \text{Out}[B_X, D]
$$

Now each layer's weights get AllGathered right before they're used (prefetched while the previous layer runs) and dropped right after. On the way back we ReduceScatter the gradients, and each chip updates only its own shard of the optimizer state. Remember $\text{AllReduce} = \text{ReduceScatter} + \text{AllGather}$ from Chapter 3: FSDP just splits the AllReduce that data parallelism was already paying for. Same bytes per FLOP, no duplicated memory.

Forward pass this time (two AllGathers of $2DF$ bytes, two matmuls):

$$
T_{\text{comms}} = \frac{4DF}{W_{\text{ici}}} \qquad T_{\text{math}} = \frac{4BDF}{X \cdot C}
$$

Same ratio, same rule: $B/X > 2550 / M_X$. So anything compute-bound with data parallelism is still compute-bound with FSDP, and we get the replicated memory back for free.

One thing that feels backwards: on a fixed number of chips, a SMALLER batch makes us more communication bound. And we can't just grow the batch, since past some size more tokens per step stop helping training.

### 3. Tensor Parallelism

$$
\text{In}[B, D_Y] \cdot_D W_{\text{in}}[D, F_Y] \cdot_F W_{\text{out}}[F_Y, D] \to \text{Out}[B, D_Y]
$$

Instead of moving weights, we move activations. The weights are sharded along $F$ and never travel. We AllGather In (sharded along $D$) before the first matmul, run both matmuls locally (leaving Out unreduced along $Y$), and ReduceScatter Out at the end. The catch: these ARE on the critical path, since the layer can't start until the gather lands.

$$
T_{\text{comms}} = \frac{2 \cdot 2BD}{W_{\text{ici}}} \qquad T_{\text{math}} = \frac{4BDF}{Y \cdot C}
$$

This time $B$ cancels and we get a limit on $Y$ itself:

$$
Y < M_Y \cdot \frac{F}{C / W_{\text{ici}}} = M_Y \cdot \frac{F}{2550}
$$

So tensor parallelism doesn't care about batch size; its ceiling is set by how wide the MLP is. With $F \approx 30{,}000$ that's about 11.8 per axis, so 8-way works and 16-way doesn't. Precision doesn't move it either: int8 doubles $C$ but halves the activation bytes.

### 4. Mixing FSDP and Tensor Parallelism

$$
\text{In}[B_X, D_Y] \cdot_D W_{\text{in}}[D_X, F_Y] \cdot_F W_{\text{out}}[F_Y, D_X] \to \text{Out}[B_X, D_Y]
$$

> FSDP pays its comms in weights. Tensor parallelism pays in activations.

The two help each other. Each chip sees only $B/X$ tokens, so the activation gathers shrink as $X$ grows, and each weight slice is only $F/Y$ wide, so the weight gathers shrink as $Y$ grows. With $N = XY$ chips:

$$
T_{\text{FSDP}} = \frac{4DF}{Y \cdot W_{\text{ici}} \cdot M_X} \qquad T_{\text{TP}} = \frac{4BD}{X \cdot W_{\text{ici}} \cdot M_Y} \qquad T_{\text{math}} = \frac{4BDF}{N \cdot C}
$$

The math doesn't depend on the split; only the comms do. If they overlap, the total is the max of the two, so the best split is where they're equal:

$$
X_{\text{opt}} = \sqrt{\frac{B}{F} \cdot \frac{M_X}{M_Y} \cdot N}
$$

Plugging that back in, we're compute-bound when:

$$
\frac{B}{N} > \frac{(C / W_{\text{ici}})^2}{M_X M_Y F}
$$

For $F = 32{,}768$ and $M_X M_Y = 2$ that's about 99 tokens per chip, roughly 8.6 times smaller than FSDP's 850. So plain FSDP is fine down to 850 tokens per chip, and from there down to about 100 only the mix stays compute-bound. The plots on [jax-ml.github.io/scaling-book/training/](https://jax-ml.github.io/scaling-book/training/) show this well.

### Pipelining

The last option shards the layers: chip 0 runs the first few, hands the activations to chip 1, and so on, with gradients coming back the other way. Very little moves, which is why it's popular on GPU clusters with weaker links between nodes. The price is the pipeline bubble, where stages sit idle waiting on each other, usually fixed with microbatching. TPU pods are big enough that it rarely comes up.

### On Crossing Pods

The biggest v5p slice is a full pod, 8960 chips. Past that we go over DCN, about $6.25 \times 10^9$ bytes/s per TPU (25 GB/s per host, shared by 4 chips). The usual recipe: FSDP and tensor parallelism inside each pod, pure data parallelism across pods. Every chip brings its own share of NIC bandwidth, so the rule ends up per slice:

$$
\frac{B}{\text{slice}} > \frac{C}{W_{\text{dcn}}} = \frac{4.59 \times 10^{14}}{6.25 \times 10^{9}} = 73{,}440 \text{ tokens}
$$

For a full pod that's about 8 tokens per chip. Tiny. The ICI rules inside the pod bite first.

Say we train LLaMA-3 70B ($F \approx 30{,}000$) with a 2M token batch. FSDP alone tops out around 2,350 chips ($2 \times 10^6 / 850$), a quarter of a pod. The mix needs $B/N > 2550^2 / (2 \times 30{,}000) \approx 108$, which allows roughly 18,450 chips, more than a pod. So we take two pods at 1M tokens each: about 112 tokens per chip (just over 108) and 1M per pod, way above 73,440.

### Takeaway

| Scheme | What travels | Compute-bound when (v5p) |
|---|---|---|
| Data parallelism | weight gradients (AllReduce, backward only) | $B/X > 2550 / M_X$ |
| FSDP | weights (AllGather) and gradients (ReduceScatter) | $B/X > 2550 / M_X$ |
| Tensor parallelism | activations (AllGather, then ReduceScatter) | $Y < M_Y \cdot F / 2550$ |
| FSDP + tensor parallelism | both | $B/N > 2550^2 / (M_X M_Y F)$ |
| Data parallelism across pods | weight gradients over DCN | $B / \text{slice} > 73{,}440$ |

With a big batch or a small model there isn't much to decide: data parallelism if the model fits on one chip at 10 bytes per parameter, FSDP if it doesn't. The hard case is a small batch on lots of chips, where we add tensor parallelism up to its 8 to 16-way ceiling and let FSDP take the rest.

## Worked Problems

All three problems use LLaMA-2 13B:

| $L$ | $D$ | $F$ | $N$ | $K$ | $H$ | $V$ |
|---|---|---|---|---|---|---|
| 40 | 5,120 | 13,824 | 40 | 40 | 128 | 32,000 |

It has separate embedding and output matrices and a gated MLP block. Hardware is the TPU v5p from our Chapter 2 references: 96GB of HBM per chip, 4.59e14 bf16 FLOPs/s, and 1.8e11 bytes/s of bidirectional ICI per axis.

**Question 1 [counting parameters]:** How many parameters does LLaMA-2 13B have? (Yes, the answer is in the name, but do the math.) Hint: a gated MLP has 3 big FFW matrices per layer, two up-projections and one down-projection. The notes above ignored the gating matrix, but it behaves just like $W_{\text{in}}$.

- Just the Chapter 4 accounting on a real config. Three buckets: MLP, attention and vocab.
- **MLP:** three $D \times F$ matrices in each of the $L$ layers:

  $$
  3LDF = 3 \times 40 \times 5120 \times 13824 = 8{,}493{,}465{,}600
  $$

- **Attention:** the Q, K, V and output projections. Since $N = K$ and $N \times H = 5120 = D$, each one is $D \times NH$:

  $$
  4DNHL = 4 \times 5120 \times 40 \times 128 \times 40 = 4{,}194{,}304{,}000
  $$

- **Vocab:** separate embedding and output matrices, so $V \times D$ twice:

  $$
  2VD = 2 \times 32000 \times 5120 = 327{,}680{,}000
  $$

- Add them up:

  $$
  8{,}493{,}465{,}600 + 4{,}194{,}304{,}000 + 327{,}680{,}000 = \mathbf{13{,}015{,}449{,}600 \text{ parameters}}
  $$

- So **13.0B**, as the name says. The MLP is about 65% of it, attention 32%, and the vocab under 3%. The norm scales only add $(2L + 1) \times D = 414{,}720$ more, a rounding error.

**Question 2 [training memory]:** Say we train with a batch of 16M tokens using Adam. Ignoring parallelism for now, how much total memory do the model's parameters, optimizer state and activations take? Assume bf16 parameters, fp32 optimizer state, and activations checkpointed three times per layer (after each of the three big FFW matmuls).

- Basically asking which of the three fills up memory.
- bf16 parameters are 2 bytes, and Adam's first and second moments are fp32, 4 bytes each. So 10 bytes per parameter:

  $$
  (2 + 4 + 4) \times 13{,}015{,}449{,}600 = 130{,}154{,}496{,}000 \text{ bytes} \approx \mathbf{130 \text{ GB}}
  $$

- Activations: the two up-projections each leave a $[B, F]$ tensor and the down-projection leaves a $[B, D]$, all bf16. Per layer that's $2B(D + 2F)$ bytes, and conveniently $D + 2F = 5120 + 27648 = 32768$:

  $$
  2 \cdot L \cdot B \cdot (D + 2F) = 2 \times 40 \times 16 \times 10^6 \times 32768 \approx 4.19 \times 10^{13} \text{ bytes} = \mathbf{41.9 \text{ TB}}
  $$

- The total is around **42.1 TB**, basically all activations: about 320 times the parameters and optimizer state combined.
- Meaning, at training batch sizes the model itself is a rounding error in memory. Holding it all takes about 438 v5p chips' worth of HBM, which is why we shard the batch first.

**Question 3 [choosing a sharding]:** We want to train with a 32k sequence length and a total batch size of 3M tokens on a TPU v5p `16x16x16` slice, with bf16 weights and an fp32 optimizer as above. Assume gradient checkpoints only after the 3 big FFW matrices.

- Napkin math first. The slice has $16 \times 16 \times 16 = 4096$ chips and $4096 \times 96\text{ GB} \approx 393\text{ TB}$ of HBM.
- Tokens per chip: $3 \times 10^6 / 4096 \approx 732$. Everything below is this number against a threshold.
- 3M tokens at 32k each is only about 92 to 94 sequences, way fewer than chips, so each sequence gets spread over around 44 chips (the context parallelism from the note above).

**Does pure data parallelism work here? Why or why not?**

- No. Every chip holds a full copy of the parameters and optimizer state, which from Question 2 is 130 GB against 96 GB of HBM. Doesn't fit, even before a single activation.

**Can pure FSDP do it without being comms-bound? Why or why not? How much memory does each device use with pure FSDP?**

- Memory first. Swapping $B$ for $3 \times 10^6$ in Question 2:

  $$
  2 \times 40 \times 3 \times 10^6 \times 32768 \approx 7.86 \times 10^{12} \text{ bytes}
  $$

- Adding the 130 GB of parameters and optimizer state gives $7.99 \times 10^{12}$ bytes, basically 8 TB, around 2% of the slice. FSDP shards all of it with no duplication:

  $$
  \frac{7.99 \times 10^{12} \text{ bytes}}{4096 \text{ chips}} \approx \mathbf{1.95 \text{ GB per chip}}
  $$

- Each chip also briefly holds the gathered weights of its current layer ($3 \times 2DF \approx 0.42\text{ GB}$). Memory is not the problem.
- Now comms. With all three axes on FSDP we need $B/N > 2550 / 3 = 850$ tokens per chip, and we only have 732:

  $$
  \frac{T_{\text{math}}}{T_{\text{comms}}} = \frac{B/N}{850} = \frac{732}{850} \approx 0.86
  $$

- So **no**, pure FSDP is comms-bound. Per layer, the matmuls take about 452 $\mu\text{s}$ against about 524 $\mu\text{s}$ for the weight AllGathers. We'd need a batch of at least $850 \times 4096 \approx 3.48\text{M}$ tokens, and we're at 3M. Close, but on the wrong side.

**Does mixed FSDP + tensor parallelism work? Why or why not? If so, what should $X$ and $Y$ be, and how much memory sits on each device? Using only roofline FLOPs estimates and ignoring attention, how long is each training step at 40% MFU?**

- Two axes on FSDP and one on tensor parallelism gives $M_X M_Y = 2$, so the bar for the mix is:

  $$
  \frac{2550^2}{2 \times 13824} \approx 235 \text{ tokens per chip}
  $$

- We have 732, so **yes**. (Higher than the 99 from the notes because this model's $F$ is smaller.)
- For the split:

  $$
  X_{\text{opt}} = \sqrt{\frac{B}{F} \cdot \frac{M_X}{M_Y} \cdot N} = \sqrt{\frac{3 \times 10^6}{13824} \times 2 \times 4096} \approx 1333
  $$

- That leaves $Y_{\text{opt}} = 4096 / 1333 \approx 3.1$. Rounding to powers of two, we get $\mathbf{X = 1024}$ for FSDP and $\mathbf{Y = 4}$ for tensor parallelism. $Y = 4$ also sits under the tensor parallelism ceiling of $F / 2550 \approx 5.4$, which $Y = 8$ wouldn't.
- Checking the per-layer times for that split:

  $$
  T_{\text{math}} = \frac{4BDF}{N \cdot C} \approx 452 \ \mu\text{s} \qquad T_{\text{FSDP}} = \frac{4DF}{Y \cdot W_{\text{ici}} \cdot M_X} \approx 197 \ \mu\text{s} \qquad T_{\text{TP}} = \frac{4BD}{X \cdot W_{\text{ici}} \cdot M_Y} \approx 333 \ \mu\text{s}
  $$

- Both comms times are under the math, so we're compute-bound, with a margin of about 1.36 on the tighter (tensor parallelism) side. The neighbors are worse: $2048 \times 2$ only clears by 1.15, and $512 \times 8$ is comms-bound (0.68). This assumes the two comms overlap; back to back they'd add up to about 530 $\mu\text{s}$, more than the math.
- Memory per device is the same as pure FSDP, about **1.95 GB per chip**, since neither scheme replicates anything.
- Step time: roofline training FLOPs are $6 \times \text{tokens} \times \text{parameters}$ (Chapter 4), at 40% of the slice's peak:

  $$
  T_{\text{step}} = \frac{6 \times 3 \times 10^6 \times 13.015 \times 10^9}{4096 \times 4.59 \times 10^{14} \times 0.4} = \frac{2.34 \times 10^{17} \text{ FLOPs}}{7.52 \times 10^{17} \text{ FLOPs/s}} \approx \mathbf{0.31 \text{ s}}
  $$

- Each step takes around **312 ms**, about 9.6M tokens per second. The solution says 300 ms because it plugs in 13e9 parameters and 4.6e14 FLOPs/s and rounds down.
- As a note, ignoring attention hides a lot at 32k context. From Chapter 4, attention FLOPs over matmul FLOPs is $12T / (18F + 24D)$ when $NH = D$, about 1.06 at $T = 32{,}768$ (half with causal masking). Counting them, a step is more like 0.47 s with causal masking and 0.63 s without.
