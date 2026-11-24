---
title: All the Transformer Math
published: true
---
## 4. Transformers Overview

We’ll dive deeper into the transformer architecture, calculating FLOPs, bytes, and math in general.

At the lowest level, everything boils down to simple dot products: multiplying 2 lists of numbers. So, the computer needs to take 2 numbers, multiply them together, and add them to the running sum. This is called a MAC operation: Multiply-Accumulate.

- 1 Multiply = 1 FLOP
    
- 1 Add = 1 FLOP
    

So basically every equation we talk about here starts with a 2. For Matrices $A[N \times P]$ and $B[P \times M]$, we get $N \times P \times M$ steps, so we get $2NPM$ FLOPs. Basic rule of thumb.

In higher dimensions, we can look at an example like:

$$
C[G, H, I, J, K, L] \times D[G, H, M, N, K, L]
$$

which just describes a massive, multi-dimensional tensor. To calculate the FLOPs required for any matmul, just multiply all the unique letters together and then multiply by 2. Those brackets include all the contracting dimensions, batching dimensions, and free dimensions.

### Memory vs Math Requirements

Assuming perfectly square matrix multiplications where $N, P, M$ are all size $N$ (same size), compute and memory scale differently:

- **Compute FLOPs** take $2 \cdot N^3$ operations, meaning compute grows cubically: $\mathcal{O}(N^3)$.
    
- **Data (memory)** concerns loading Matrix $A$, Matrix $B$, and writing Matrix $C$, so the memory footprint grows quadratically: $\mathcal{O}(N^2)$.
    

So our math requirements outscale our memory requirements. Which makes TPUs uniquely suited (super fast math, but bad at loading memory (HBM)). Just by scaling up the size of our matrices, we move from a memory-bound regime to compute-bound.

So we’ve established that during simple forward passes, like inference, standard matrix multiplications take $2NPM$ FLOPs. During inference, all we pay is 2 FLOPs per parameter, per token. Training, however, involves calculating 2 separate derivatives that each have its own big matmuls:

1. **The Weight Gradient** tells you how to adjust the weights by multiplying the input activations ($A^T$) by the incoming error; full matmul = 2 FLOPs.
    
2. **The Activation Gradient** passes the error backwards to the previous layer so it can also learn, by multiplying the incoming error by the transposed weights ($B^T$). Also 2 FLOPs.
    

So, during training, we have:

$$
\text{Forward Passes} + \text{Backward Passes (Weights)} + \text{Backward Passes (Activations)} = 6 \text{ FLOPs per parameter, per token}
$$

While training, the golden rule is just $6 \times \text{tokens} \times \text{parameters}$ to figure out how much hardware we need to buy or rent out.

## Transformer Accounting

The following section covers transformer accounting, which can be found in more detail in a previous breakdown of transformers, but for reference, memorizing these variables makes reading research infinitely easier:

- **$B$:** Batch size (number of sequences processed at once; for local models, 1 = just you)
    
- **$T$ (or $S$):** Sequence Length (4096 or 8192 tokens, etc)
    
- **$D$:** Model dimension (core width of the network)
    
- **$F$:** Feed forward dimension (the expanded width inside the MLP, usually $4 \times D$)
    
- **$N$:** Number of query heads
    
- **$K$:** Number of K/V (Key value heads)
    
- **$H$:** Head dimension (usually $D/N$)
    

But you can skip this if you understand the transformer decoder architecture flows. Modern transformers today use pre-norm architectures that are worth touching up on, where the norm occurs before the residual connection, used by models like Llama 3. I’ll go over these briefly:

- **Gating Einsums (SwiGLU):** Originally, for the MLP (multi layer perceptron), we used 2 matrices: 1 to expand the dimension $D \to F$, and one to shrink it back $F \to D$. This approach uses 3 matrices instead, adding a ‘Gating’ matrix to act as a filter, element-wise multiplying against the first matrix. This causes MLP params to increase from $2DF$ to $3DF$, but greatly enhances learning.
    
- **Pre-norm vs Post-norm:** Models used to add residual connections and then normalize the data. This proved to be unstable at training larger scale models, so now we normalize data before going into the MLP/attention blocks.
    
- **Attention Variants (MHA vs GQA vs MQA):** This is about saving memory. In standard MHA (Multi head attention), every query head gets its own key and value head ($N = K$). This takes up a ton of memory for the KV cache during inference. MQA (Multi query attention) is when all query heads share 1 KV head ($K = 1$). GQA is another compromise where query heads are split into groups, and each group takes a KV head.
    

## FLOPs and Param Calculations

This is basically the ‘accounting’ of the model. We know the abstract $6NPM$ rule, but we’ll now apply it precisely to the matrices in our model to see where exactly the compute budget is going.

A rule we want to remember is $T > 8D$, which explains why long-context models are an engineering challenge, and why that challenge gets even worse for smaller models.

Starting with:

### 1. The MLP Budget

The MLP (feed forward) block contains the vast majority of parameters in a transformer. Assuming the modern ‘Gating’ architecture, using 3 matrices, the math is simple:

- **Parameters:** 3 Matrices of size $D \times F$, so $3DF$ total.
    
- **Train FLOPs:** Applying the $6\times$ rule to a batch of $B$ sequences of length $T$, we get $6BT \times 3DF = 18BTDF$ FLOPs.
    

### 2. Attention Budget (Projections vs Dot Products)

Attention has 2 separate costs: projection (setup) and the actual matching (dot products).

- **Projections:** Linear cost in $T$. $W_Q$, $W_K$ and $W_V$ make the queries, keys and values, and $W_O$ maps the result back to width $D$. Q and O are $D \times NH$, K and V are $D \times KH$, so $2D(N+K)H$ parameters and $12BTD(N+K)H$ train FLOPs. For plain MHA ($N = K$ and $NH = D$) that’s just $4D^2$ parameters and $24BTD^2$ FLOPs.

- **Dot Products:** Quadratic cost in $T$. Every query gets scored against every key, then we take a softmax-weighted sum of the values: 2 matmuls of $6BTSNH$ each in training, so $12BT^2NH$ with self-attention ($S = T$). No parameters, pure compute.

As a note, the [transformer diagram](https://jax-ml.github.io/scaling-book/transformers/) shows the shape of each einsum.

### 3. Other Operations

- **Layernorms:** 2 per layer, $O(BTD)$ FLOPs and $2D$ parameters. A rounding error, so we ignore them.

- **Unembedding:** A $[D,V]$ matrix at the end scores every token in the vocab: $DV$ parameters and $6BTDV$ train FLOPs, once, not per layer. The embedding table is another $DV$ parameters, but it’s a lookup, not a matmul.

### 4. General rule of thumb for Transformer FLOPs

Ignoring the dot products (fine for short contexts), the MLP and projections across $L$ layers add up to:

$$
(18BTDF + 12BTD(N+K)H) \cdot L = 6 \cdot BT \cdot (3DF + 2D(N+K)H) \cdot L
$$

The bracket is one layer’s parameter count and $BT$ is the number of tokens, so we land right back on $6 \times \text{tokens} \times \text{parameters}$. Divide by 3 for inference.

### 5. Fractional cost of attention with context length

With the typical $F = 4D$, $NH = D$ and $N = K$, the MLP is $72BTD^2$, the projections are $24BTD^2$, and the dot products are $12BT^2D$:

$$
\frac{\text{dot product FLOPs}}{\text{all other matmul FLOPs}} = \frac{12BT^2D}{72BTD^2 + 24BTD^2} = \frac{T}{8D}
$$

This is where $T > 8D$ comes from: the dot products only take over once the context passes 8 times the model width, 65,536 tokens for $D = 8192$ but only 36,864 for Gemma 27B ($D = 4608$). The MLP and projections scale with $D^2$ and the dot products only with $D$, so smaller models hit the wall sooner.

## Miscellaneous Math

### Sparsity and Mixture-of-Experts

An MoE (Mixture of Experts) model swaps each layer’s MLP for $E$ experts, and a router picks $k$ of them per token. Parameters grow with $E$ but FLOPs only with $k$, so the $6\times$ rule uses the activated parameters. $E/k$ is the sparsity, typically 8 to 64 (DeepSeek v3 has $E = 256$ and $k = 8$, so 32). The price is 2 AllToAlls around every MoE block to route tokens to their experts and back, each about $\frac{1}{4}$ of an AllGather.

### Gradient checkpointing

The backward pass needs the forward pass’s activations. Saving all of them for a 4M token batch ($BT = 4 \times 10^6$), $L = 64$, $D = 8192$ and around 20 activations per layer in bf16:

$$
2 \times 20 \times (4 \times 10^6) \times 8192 \times 64 \approx 8.39 \times 10^{13} \text{ bytes} \approx 84 \text{ TB}
$$

Insanely expensive, so we save some and recompute (rematerialize) the rest:

- **Block remat:** 1 checkpoint per layer instead of 20, so 4.2 TB. But we basically redo the forward pass, so training goes from $6$ to roughly $8 \times \text{tokens} \times \text{parameters}$.
- **Big matmuls only:** Keep the large matmul outputs and recompute the cheap stuff. Around 7 per layer, about 29 TB.

In JAX, this is `jax.checkpoint` (also called `jax.remat`).

### Key-Value (KV) caching

Prefill runs the prompt once and saves every layer’s keys and values, then generation samples 1 token at a time and reuses them. That’s the KV cache, $[2, S, L, K, H]$ per sequence, so in int8:

$$
\text{KV cache} = 2 \cdot S \cdot L \cdot K \cdot H \text{ bytes}
$$

With $S = 8192$, $L = 64$ and $KH = D = 8192$, that’s 8 GiB per sequence (1 MiB per token). GQA ($K \ll N$) shrinks it by a factor of $N/K$.

## What Should You Take Away from this Section?

- Per layer in training (MHA), the MLP is $3DF$ parameters and $18BTDF$ FLOPs, and attention is $4DNH$ and $24BTDNH + 12BT^2NH$. The MLP dominates, which is why $6 \times \text{tokens} \times \text{parameters}$ works so well.
- The dot products only take over once $T > 8D$.
- The KV cache is $2SLKH$ bytes per sequence in int8, and unlike the weights, it isn’t shared across the batch.

## A Few Problems to Work

**Question 1 [counting parameters]:** A model has $D = 4096$, $F = 4D$, $V = 32{,}000$ and $L = 64$. How many parameters is that? What fraction of them belong to attention? And how large is the KV cache per token? Assume $NH = D$, multi-head attention ($N = K$), and int8 KVs.

* **How many parameters?**
    - With $F = 16384$, each layer has $3DF = 201{,}326{,}592$ in the MLP, $4DNH = 67{,}108{,}864$ in attention and $2D = 8{,}192$ in layernorms. That’s 268,443,648 per layer. Plus $2DV = 262{,}144{,}000$ for the embedding and unembedding:

      $$
      \text{Params} = L(3DF + 4DNH + 2D) + 2DV = 17{,}180{,}393{,}472 + 262{,}144{,}000 = \mathbf{17{,}442{,}537{,}472}
      $$

    - So around **17.4B parameters**. The solution says 16B because it rounds the dims first.

* **What fraction are attention parameters?**
    - Ignoring the tiny terms:

      $$
      \frac{4DNH}{4DNH + 3DF} = \frac{4D^2}{4D^2 + 12D^2} = \mathbf{\frac{1}{4}}
      $$

* **How large are the KV caches per token?**
    - 1 key and 1 value vector per layer, each $D$ wide, at 1 byte per element:

      $$
      2 \cdot L \cdot N \cdot H = 2 \times 64 \times 4096 = 524{,}288 \text{ bytes} = \mathbf{512 \text{ KiB per token}}
      $$

    - Meaning, an 8192 token context costs 4 GiB of HBM per sequence.

**Question 2 [sharded matmul FLOPs]:** We run $A[B_X, D_Y] \cdot_D W[D_Y, F]$ on the mesh `{'X': 4, 'Y': 8, 'Z': 4}`. How many FLOPs does that take in total, and how many does each TPU perform?

* **How many FLOPs does each TPU perform?**
    - Each chip just multiplies $A[B/4, D/8]$ by $W[D/8, F]$:

      $$
      \text{FLOPs per TPU} = 2 \cdot \frac{B}{4} \cdot \frac{D}{8} \cdot F = \frac{2BDF}{X \cdot Y} = \mathbf{\frac{BDF}{16}}
      $$

* **How many total FLOPs?**
    - Nothing is sharded over `Z`, so like Question 1 in Chapter 3, each of the 4 `Z` layers does the full matmul by itself. With $4 \times 8 \times 4 = 128$ chips:

      $$
      \text{Total FLOPs} = 128 \times \frac{BDF}{16} = 2BDF \times Z = \mathbf{8BDF}
      $$

    - So the unused axis burns $4\times$ the FLOPs we need, on top of the HBM.

**Question 3 [einsum FLOPs]:** How many FLOPs does the contraction $A[I,J,K,L] * B[I,J,M,N,O] \to C[K,L,M,N,O]$ take?

- The ‘unique letters’ rule. $I$ and $J$ are in both inputs but not the output, so they’re contracting. Nothing is in all three, so no batching dimensions. Every letter counts once, times 2 since we contract:

  $$
  \text{FLOPs} = \mathbf{2 \cdot I \cdot J \cdot K \cdot L \cdot M \cdot N \cdot O}
  $$

**Question 4 [attention intensity]:** Ignoring the Q/K/V/O projections, what is the arithmetic intensity of grouped multi-query attention as a function of query length $T$, KV length $S$ and $G$ query heads per KV head ($N = KG$)? At what context length is attention FLOPs-bound? Then plot the effective cost of attention relative to the FFW block as the context grows. Hint: assume no unnecessary reads or writes, and look at both $T = S$ and $T \ll S$. Assume TPU v5e numbers: $1.97 \times 10^{14}$ bf16 FLOPs/s and $8.2 \times 10^{11}\text{ bytes/s}$ of HBM bandwidth.

- The Chapter 1 intensity question again, for the dot product half of attention in bf16. ‘Efficient’ means the $T \times S$ score matrix never touches HBM.

* **What is the arithmetic intensity?**
    - FLOPs: $QK^T$ and then the multiply by $V$ are $2BTSKGH$ each, so $4BTSKGH$. Bytes: read $Q$ ($2BTKGH$), $K$ and $V$ ($2BSKH$ each), and write an output shaped like $Q$, so $4BKH(TG + S)$:

      $$
      \text{Intensity} = \frac{4BTSKGH}{4BKH(TG + S)} = \mathbf{\frac{TSG}{TG + S}}
      $$

    - $B$ cancels, so a bigger batch doesn’t help. Every sequence brings its own keys and values.

* **At what context length is attention FLOPs-bound?**
    - For prefill or training, $S = T$, so the intensity is $\frac{G}{G+1}T$. The v5e needs $\frac{1.97 \times 10^{14}}{8.2 \times 10^{11}} \approx 240\text{ FLOPs/byte}$:

      $$
      \frac{G}{G+1}T > 240 \implies T > 240 \cdot \frac{G+1}{G}
      $$

    - For plain MHA ($G = 1$) that’s $\mathbf{T > 480}$ **tokens**. The solution says 240 because it drops the $\frac{G}{G+1}$ factor.
    - For generation, $T = 1$ and the intensity tends to $G$, so with MHA we’re permastuck at $1\text{ FLOP/byte}$. Never FLOPs-bound.

* **How does the cost of attention compare to the FFW block as context grows?**
    - No published answer for this part. Effective cost is $\max(T_{\text{math}}, T_{\text{HBM}})$, and with $F = 4D$ and $NH = D$, the FFW forward pass is $24BTD^2$ FLOPs against $4BTSD$ for the dot products.
    - **Prefill ($S = T$):** The ratio is $\frac{T}{6D}$, with a floor below the critical context, where attention waits on HBM:

      $$
      \frac{\text{attention}}{\text{FFW}} = \frac{\max\left(T, \ 240 \cdot \frac{G+1}{G}\right)}{6D}
      $$

    - Attention passes the FFW block at $T = 6D = 24{,}576$ tokens for $D = 4096$.
    - **Generation ($T = 1$):** Attention reads about $\frac{4BSD}{G}$ bytes of KV cache per layer, while the FFW block reads $24D^2$ bytes of weights shared by the batch, taking $\frac{24D^2 \cdot \max(B, 240)}{1.97 \times 10^{14}}$:

      $$
      \frac{\text{attention}}{\text{FFW}} \approx \frac{BS}{6GD} \ \text{ for } B \le 240, \qquad \frac{40S}{GD} \ \text{ for } B \ge 240
      $$

    - Past $B = 240$, attention passes the FFW block at only around 100 tokens of context, which is why a larger $G$ matters so much for serving.

**Question 5 [dot products vs projections]:** For self-attention, at what sequence length do the dot product FLOPs equal the QKVO projection FLOPs?

- Just set the 2 attention costs from the notes equal. With MHA:

  $$
  24BTDNH = 12BT^2NH \implies \mathbf{T = 2D}
  $$

- For $D = 4096$, that’s **8192 tokens**.

**Question 6 [remat FLOPs]:** If the forward pass only saves the outputs of the 7 main matmuls in each transformer layer (Q, K, V, O, plus the 3 FFW matrices), how many extra FLOPs does the backward pass need to ‘rematerialize’ everything else?

- Each weight gradient needs that matmul’s input, so which inputs can’t we rebuild cheaply?
- **Q, K, V and the 2 FFW input matrices:** The normalized residual stream, just a layernorm, so $O(BTD)$.
- **FFW output matrix:** The gated product of the 2 saved FFW activations, elementwise, so $O(BTF)$.
- **O:** $\text{softmax}(QK^T)V$, which we didn’t save. So we redo both attention matmuls, forward only, $2BT^2NH$ each:

  $$
  \text{Extra FLOPs} = 2BT^2NH + 2BT^2NH = \mathbf{4BT^2NH}
  $$

- Ignoring the small leftovers, that’s about 6.7% extra on the whole layer for $D = 4096$ and $T = 8192$, against about 33% for block remat.

**Question 7 [DeepSeek utilization]:** DeepSeek v3 reports 2.79M H800 hours of training on 14.8T tokens ([source](https://arxiv.org/pdf/2412.19437v1)), with 37B activated parameters. Roughly what hardware utilization is that? Hint: they trained with FP8 FLOPs and no structured sparsity. Assume the H800 rate from [this spec sheet](https://lenovopress.lenovo.com/lp1814.pdf): 3,026 TFLOPs/s of FP8 with sparsity, so half of that, $1.513 \times 10^{15}\text{ FLOPs/s}$, without.

- Utilization is the FLOPs the model needed over the FLOPs the hardware could produce. Needed, using the activated parameters since it’s an MoE:

  $$
  6 \times (37 \times 10^9) \times (14.8 \times 10^{12}) \approx 3.29 \times 10^{24} \text{ FLOPs}
  $$

- Available: $2.79 \times 10^6 \times 3600 \approx 1.004 \times 10^{10}$ GPU-seconds at the non-sparse rate:

  $$
  1.004 \times 10^{10} \text{ s} \times 1.513 \times 10^{15} \text{ FLOPs/s} \approx 1.52 \times 10^{25} \text{ FLOPs}
  $$

- Then we just divide:

  $$
  \text{Utilization} = \frac{3.29 \times 10^{24}}{1.52 \times 10^{25}} \approx \mathbf{21.6\%}
  $$

- So roughly **21.6%** of peak FP8, about 1/5. The rule ignores the attention dot products, so the real number is a bit higher.

**Question 8 [MoE batch size]:** An MoE layer holds $E$ copies of the dense MLP block, and every token gets routed through $k$ of them. With int8 weights on a TPU v5e, how many tokens do we need in a batch to be compute-bound? What about DeepSeek, with 256 (routed) experts and $k = 8$? Assume $8.2 \times 10^{11}\text{ bytes/s}$ of HBM bandwidth and bf16 math at $1.97 \times 10^{14}\text{ FLOPs/s}$.

- This is Question 2 from Chapter 1 again (int8 weights, bf16 compute, $B > 120$), except we now load all $E$ experts ($EDF$ bytes) and each token only uses $k$ of them ($2kBDF$ FLOPs). Ignoring the activations like before:

  $$
  \text{Intensity} = \frac{2kBDF}{EDF} = \frac{2kB}{E}
  $$

- The v5e needs around $240\text{ FLOPs/byte}$, so:

  $$
  \frac{2kB}{E} > 240 \implies \mathbf{B > 120 \cdot \frac{E}{k}}
  $$

- For DeepSeek:

  $$
  B > 120 \times \frac{256}{8} = \mathbf{3{,}840 \text{ tokens}}
  $$

- Meaning, the dense threshold of 120 just gets multiplied by the sparsity (32 here), assuming the router spreads tokens evenly. Fine for training, but a massive batch for generation.
