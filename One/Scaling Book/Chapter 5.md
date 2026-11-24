---
title: How to Parallelize a Transformer for Training
description: Overview and Worked Problems
published: true
---
## 5. Parallelizing Training

Chapter 3 was one sharded matmul and Chapter 4 was the FLOPs of one transformer. Now we spread a whole training step over thousands of chips.

## What Do We Mean By Scaling?

We want strong scaling: double the chips, double the throughput. On one chip the fight was FLOPs against HBM bandwidth; across chips it's FLOPs against the network. More chips means more to communicate and fewer tokens per chip to hide it behind, so for every scheme below we ask when $T_{\text{comms}}$ overtakes $T_{\text{math}}$.

To keep the algebra small, we pretend the transformer is just a stack of MLP blocks with no gating matrix, which is fair while $T < 8D$ (Chapter 4). Each layer is two matmuls:

$$
\text{In}[B, D] \cdot_D W_{\text{in}}[D, F] \cdot_F W_{\text{out}}[F, D] \to \text{Out}[B, D]
$$

$B$ is the TOTAL batch in tokens (not sequences, not per device), $C$ is FLOPs/s per chip and $W$ is bidirectional bandwidth ($W_{\text{ici}}$ or $W_{\text{dcn}}$). $X$ and $Y$ are how many chips we shard over, spanning $M_X$ and $M_Y$ physical mesh axes; more axes means more links, so comms time drops by roughly $1/M$.

For a TPU v5p, $C = 4.59 \times 10^{14}$ FLOPs/s and $W_{\text{ici}} = 1.8 \times 10^{11}$ bytes/s, so $C / W_{\text{ici}} = 2550$. Basically every rule below is built out of that one number. Also, the MLP doesn't care which sequence a token belongs to, so "batch" sharding can cut along the sequence too (context parallelism).

### Data Parallelism

$$
\text{In}[B_X, D] \cdot_D W_{\text{in}}[D, F] \cdot_F W_{\text{out}}[F, D] \to \text{Out}[B_X, D]
$$

We split the batch and give every chip a full copy of the weights. No contracting dimension is sharded, so the forward pass is free (Case 1 from Chapter 3). The bill comes in the backward pass: each chip only has the gradient from its own slice of the batch, so we AllReduce the weight gradients, off the critical path since only the update needs them.

Per layer, that's two AllReduces of $2DF$ bytes (each costing two AllGathers) against four matmuls of $2(B/X)DF$ FLOPs:

$$
T_{\text{comms}} = \frac{8DF}{W_{\text{ici}}} \qquad T_{\text{math}} = \frac{8BDF}{X \cdot C} \implies \text{compute-bound when } \frac{B}{X} > \frac{C}{W_{\text{ici}}}
$$

Meaning, the per-device batch has to beat 2550 tokens on a v5p. Over all three mesh axes the bar drops to $2550 / 3 = 850$, so a full 8960-chip pod needs a batch of about 7.6M tokens. Not hard.

The real problem is memory. bf16 parameters (2 bytes) plus Adam's two fp32 moments (8 bytes) is 10 bytes per parameter, on EVERY chip. With 96GB of HBM that caps us at $96 \times 10^9 / 10 = 9.6\text{B}$ parameters before a single activation.

### Fully-Sharded Data Parallelism (FSDP)

$$
\text{In}[B_X, D] \cdot_D W_{\text{in}}[D_X, F] \cdot_F W_{\text{out}}[F, D_X] \to \text{Out}[B_X, D]
$$

FSDP (also called ZeRO-3) keeps the batch sharding and shards the weights and optimizer state too. Each layer's weights get AllGathered just before they're used (prefetched while the previous layer runs), and on the way back we ReduceScatter the gradients. Since $\text{AllReduce} = \text{ReduceScatter} + \text{AllGather}$, FSDP just splits the AllReduce that data parallelism was already paying for. In the forward pass that's two AllGathers of $2DF$ bytes against two matmuls:

$$
T_{\text{comms}} = \frac{4DF}{W_{\text{ici}}} \qquad T_{\text{math}} = \frac{4BDF}{X \cdot C}
$$

Same ratio, same rule ($B/X > 2550 / M_X$), and we get the replicated memory back for free.

One thing that feels backwards: on a fixed number of chips, a SMALLER batch makes us more communication bound. And we can't just grow the batch, since past some size more tokens per step stop helping training.

### Tensor Parallelism

$$
\text{In}[B, D_Y] \cdot_D W_{\text{in}}[D, F_Y] \cdot_F W_{\text{out}}[F_Y, D] \to \text{Out}[B, D_Y]
$$

Instead of moving weights, we move activations; the weights are sharded along $F$ and never travel. We AllGather In, run both matmuls locally (leaving Out unreduced along $Y$), and ReduceScatter Out. The catch: these ARE on the critical path, since the layer can't start until the gather lands.

$$
T_{\text{comms}} = \frac{4BD}{W_{\text{ici}}} \qquad T_{\text{math}} = \frac{4BDF}{Y \cdot C} \implies \text{compute-bound when } Y < M_Y \cdot \frac{F}{2550}
$$

This time $B$ cancels, so tensor parallelism doesn't care about batch size; its ceiling is set by how wide the MLP is. With $F \approx 30{,}000$ that's about 11.8 per axis, so 8-way works and 16-way doesn't.

### Combining FSDP and Tensor Parallelism

$$
\text{In}[B_X, D_Y] \cdot_D W_{\text{in}}[D_X, F_Y] \cdot_F W_{\text{out}}[F_Y, D_X] \to \text{Out}[B_X, D_Y]
$$

FSDP pays its comms in weights and tensor parallelism pays in activations, and the two help each other: the activation gathers shrink as $X$ grows and the weight gathers shrink as $Y$ grows. With $N = XY$ chips:

$$
T_{\text{FSDP}} = \frac{4DF}{Y \cdot W_{\text{ici}} \cdot M_X} \qquad T_{\text{TP}} = \frac{4BD}{X \cdot W_{\text{ici}} \cdot M_Y} \qquad T_{\text{math}} = \frac{4BDF}{N \cdot C}
$$

The math doesn't depend on the split. If the two comms overlap, the best split is where they're equal, and plugging it back in gives the bound:

$$
X_{\text{opt}} = \sqrt{\frac{B}{F} \cdot \frac{M_X}{M_Y} \cdot N} \qquad \frac{B}{N} > \frac{(C / W_{\text{ici}})^2}{M_X M_Y F}
$$

For $F = 32{,}768$ and $M_X M_Y = 2$ that's about 99 tokens per chip. So plain FSDP is fine down to 850 tokens per chip, and from there down to about 100 only the mix stays compute-bound. The plots on [jax-ml.github.io/scaling-book/training/](https://jax-ml.github.io/scaling-book/training/) show this well.

### Pipelining

Pipelining shards the layers: chip 0 runs the first few and hands its activations to chip 1. Very little moves, which is why GPU clusters with weak links between nodes like it, but stages sit idle waiting on each other (the pipeline bubble) unless we microbatch. TPU pods are big enough that it rarely comes up.

### Scaling Across Pods

Past a full pod (8960 chips) we go over DCN, about $6.25 \times 10^9$ bytes/s per TPU. The usual recipe is FSDP and tensor parallelism inside each pod and pure data parallelism across pods. Every chip brings its own share of NIC bandwidth, so the rule ends up per slice:

$$
\frac{B}{\text{slice}} > \frac{C}{W_{\text{dcn}}} = \frac{4.59 \times 10^{14}}{6.25 \times 10^{9}} = 73{,}440 \text{ tokens}
$$

For a full pod that's about 8 tokens per chip, so the ICI rules inside the pod bite first.

## Takeaways from LLM Training on TPUs

On a v5p, data parallelism and FSDP need $B/X > 2550 / M_X$, tensor parallelism needs $Y < M_Y \cdot F / 2550$, the mix needs $B/N > 2550^2 / (M_X M_Y F)$, and data parallelism across pods needs 73,440 tokens per slice.

With a big batch or a small model there isn't much to decide: data parallelism if the model fits on one chip at 10 bytes per parameter, FSDP if it doesn't. The hard case is a small batch on lots of chips, where we add tensor parallelism up to its 8 to 16-way ceiling and let FSDP take the rest.

## Some Problems to Work

All three problems use LLaMA-2 13B: $L = 40$, $D = 5120$, $F = 13824$, $N = K = 40$, $H = 128$ and $V = 32000$, with separate embedding and output matrices and a gated MLP. Hardware is the TPU v5p: 96GB of HBM per chip, 4.59e14 bf16 FLOPs/s, and 1.8e11 bytes/s of bidirectional ICI per axis.

**Question 1 [counting parameters]:** How many parameters does LLaMA-2 13B have? Hint: a gated MLP has 3 big FFW matrices per layer, two up-projections and one down-projection, and the gating one behaves just like $W_{\text{in}}$.

- **MLP:** three $D \times F$ matrices in each of the $L$ layers:

  $$
  3LDF = 3 \times 40 \times 5120 \times 13824 = 8{,}493{,}465{,}600
  $$

- **Attention:** Q, K, V and O. Since $N = K$ and $NH = D$, each one is $D \times NH$:

  $$
  4DNHL = 4 \times 5120 \times 40 \times 128 \times 40 = 4{,}194{,}304{,}000
  $$

- **Vocab:** separate embedding and output matrices, so $2VD = 2 \times 32000 \times 5120 = 327{,}680{,}000$.
- Add them up:

  $$
  8{,}493{,}465{,}600 + 4{,}194{,}304{,}000 + 327{,}680{,}000 = \mathbf{13{,}015{,}449{,}600 \text{ parameters}}
  $$

- So **13.0B**, as the name says, and about 65% of it is the MLP.

**Question 2 [training memory]:** Say we train with a batch of 16M tokens using Adam. Ignoring parallelism for now, how much total memory do the parameters, optimizer state and activations take? Assume bf16 parameters, fp32 optimizer state, and activations checkpointed three times per layer (after each of the three big FFW matmuls).

- bf16 parameters are 2 bytes and Adam's two fp32 moments are 4 bytes each, so 10 bytes per parameter:

  $$
  (2 + 4 + 4) \times 13{,}015{,}449{,}600 = 130{,}154{,}496{,}000 \text{ bytes} \approx \mathbf{130 \text{ GB}}
  $$

- Activations: the two up-projections each leave a bf16 $[B, F]$ tensor and the down-projection leaves a $[B, D]$. Per layer that's $2B(D + 2F)$ bytes, and $D + 2F = 32768$:

  $$
  2 \cdot L \cdot B \cdot (D + 2F) = 2 \times 40 \times 16 \times 10^6 \times 32768 \approx 4.19 \times 10^{13} \text{ bytes} = \mathbf{41.9 \text{ TB}}
  $$

- The total is around **42.1 TB**, basically all activations. Meaning, at training batch sizes the model itself is a rounding error in memory.

**Question 3 [choosing a sharding]:** We want to train with a 32k sequence length and a total batch size of 3M tokens on a TPU v5p `16x16x16` slice, with bf16 weights and an fp32 optimizer as above. Assume gradient checkpoints only after the 3 big FFW matrices.

- The slice has $16 \times 16 \times 16 = 4096$ chips, so we get $3 \times 10^6 / 4096 \approx 732$ tokens per chip.
- 3M tokens at 32k each is only about 92 sequences, far fewer than chips, so the batch sharding has to cut across sequences too.

**Does pure data parallelism work here? Why or why not?**

- No. Every chip would hold all 130 GB of parameters and optimizer state from Question 2, and it only has 96 GB of HBM.

**Can pure FSDP do it without being comms-bound? Why or why not? How much memory does each device use with pure FSDP?**

- Swapping $B$ for $3 \times 10^6$ in Question 2 gives $2 \times 40 \times 3 \times 10^6 \times 32768 \approx 7.86 \times 10^{12}$ bytes of activations, or $7.99 \times 10^{12}$ with the 130 GB added. FSDP shards all of it with no duplication:

  $$
  \frac{7.99 \times 10^{12} \text{ bytes}}{4096 \text{ chips}} \approx \mathbf{1.95 \text{ GB per chip}}
  $$

- For comms, with all three axes on FSDP we need $B/N > 2550 / 3 = 850$ tokens per chip, and we only have 732, so $T_{\text{math}} / T_{\text{comms}} = 732 / 850 \approx 0.86$.
- So **no**, pure FSDP is comms-bound. We'd need a batch of at least $850 \times 4096 \approx 3.48\text{M}$ tokens.

**Does mixed FSDP + tensor parallelism work? Why or why not? If so, what should $X$ and $Y$ be, and how much memory sits on each device? Using only roofline FLOPs estimates and ignoring attention, how long is each training step at 40% MFU?**

- Two axes on FSDP and one on tensor parallelism gives $M_X M_Y = 2$, so the bar is:

  $$
  \frac{2550^2}{2 \times 13824} \approx 235 \text{ tokens per chip}
  $$

- We have 732, so **yes**. For the split:

  $$
  X_{\text{opt}} = \sqrt{\frac{3 \times 10^6}{13824} \times 2 \times 4096} \approx 1333
  $$

- That leaves $Y = 4096 / 1333 \approx 3.1$. Rounding to powers of two, we get $\mathbf{X = 1024}$ for FSDP and $\mathbf{Y = 4}$ for tensor parallelism.
- Per layer at that split, $T_{\text{math}} \approx 452 \ \mu\text{s}$ against $T_{\text{FSDP}} \approx 197 \ \mu\text{s}$ and $T_{\text{TP}} \approx 333 \ \mu\text{s}$. Both comms fit under the math (as long as they overlap), so we're compute-bound with a margin of about 1.36.
- Memory per device is the same as pure FSDP, about **1.95 GB per chip**, since neither scheme copies anything.
- Step time is $6 \times \text{tokens} \times \text{parameters}$ FLOPs (Chapter 4) over 40% of the slice's peak:

  $$
  T_{\text{step}} = \frac{6 \times 3 \times 10^6 \times 13.015 \times 10^9}{4096 \times 4.59 \times 10^{14} \times 0.4} = \frac{2.34 \times 10^{17} \text{ FLOPs}}{7.52 \times 10^{17} \text{ FLOPs/s}} \approx \mathbf{0.31 \text{ s}}
  $$

- Each step takes around **312 ms**. The solution rounds this to 300 ms.
