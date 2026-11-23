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
    
- **$T$ (or $S$):** Sequence Length (4096 or 8192 words, etc)
    
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
    
- **Train FLOPs:** Applying the $6\times$ rule to a batch of $B$ sequences of length $T$, we get $6BT \times 3DF$ to $18BTDF$ FLOPs.
    

### 2. Attention Budget (Projections vs Dot Products)

Attention has 2 separate costs: projection (setup) and the actual matching (dot products).

- **Projections:** Linear cost in $T$. There are 4 weight matrices. $W_Q$, $W_K$ and $W_V$ make the queries, keys and values, and $W_O$ maps the result back to width $D$. Q and O are $D \times NH$, K and V are $D \times KH$, so $2D(N+K)H$ parameters and, by the $6\times$ rule, $12BTD(N+K)H$ train FLOPs. For plain MHA ($N = K$ and $NH = D$) that’s just $4D^2$ parameters and $24BTD^2$ FLOPs.

- **Dot Products:** Quadratic cost in $T$. Every query gets scored against every key, we softmax, then take a weighted sum of the values. That’s 2 matmuls ($[T,H] \cdot [H,S]$, then $[T,S] \cdot [S,H]$), each $6BTSNH$ in training across $B$ sequences and $N$ heads. With self-attention ($S = T$) the total is $12BT^2NH$. No parameters, pure compute.

As a note, the [transformer diagram](https://jax-ml.github.io/scaling-book/transformers/) shows the shape of each einsum.

### 3. Everything Else

- **Layernorms:** 2 per layer, $O(BTD)$ FLOPs and $2D$ parameters. A rounding error next to the matmuls, so we ignore them.

- **Unembedding:** A $[D,V]$ matrix at the very end turns the activations into a score for every token in the vocab. That’s $DV$ parameters and $6BTDV$ train FLOPs, paid once, not per layer. The embedding table on the way in is another $DV$ parameters, but it’s a lookup, not a matmul.

### 4. The Rule of Thumb

Ignore the dot products for now (fine for short contexts) and add up the MLP and projections across $L$ layers:

$$
(18BTDF + 12BTD(N+K)H) \cdot L = 6 \cdot BT \cdot (3DF + 2D(N+K)H) \cdot L
$$

The bracket is just one layer’s parameter count and $BT$ is the number of tokens, so the precise accounting lands right back on the golden rule, $6 \times \text{tokens} \times \text{parameters}$. Divide by 3 for inference (forward pass only).

### 5. When Attention Takes Over

Now put the dot products back, with the typical $F = 4D$, $NH = D$ and $N = K$. The MLP is $18BTDF = 72BTD^2$, the projections are $24BTD^2$, and the dot products are $12BT^2D$:

$$
\frac{\text{dot product FLOPs}}{\text{all other matmul FLOPs}} = \frac{12BT^2D}{72BTD^2 + 24BTD^2} = \frac{12BT^2D}{96BTD^2} = \frac{T}{8D}
$$

This is where $T > 8D$ comes from: attention’s quadratic part only becomes the biggest line in the budget once the context passes 8 times the model width. For $D = 8192$ that’s 65,536 tokens (around 64k). For a smaller model like Gemma 27B with $D = 4608$, it’s already 36,864 (around 37k). So why is long context worse for smaller models? The MLP and the projections scale with $D^2$, but the dot products only with $D$, so a wide model’s massive MLP bill hides the attention cost for much longer.

## Miscellaneous Math

### Sparsity and Mixture-of-Experts

An MoE (Mixture of Experts) model swaps the single MLP in each layer for $E$ independent MLPs, called experts, and a router picks only $k$ of them for each token. Parameters grow with $E$, since we store every expert, but FLOPs per token only grow with $k$. So for an MoE the $6\times$ rule uses the activated parameters, not the total.

$E/k$ is the sparsity, typically 8 to 64; DeepSeek v3 has $E = 256$ and $k = 8$, so 32. The price is networking: tokens travel to the chip holding their expert and back, which adds 2 AllToAlls around every MoE block. From Chapter 3, an AllToAll is about $\frac{1}{4}$ the cost of an AllGather, so it’s cheap, but not free.

### Gradient Checkpointing

The backward pass needs the intermediate activations from the forward pass, so by default we save all of them. Napkin math with a batch of 4M tokens ($BT = 4 \times 10^6$), $L = 64$, $D = 8192$, and around 20 saved activations per layer in bf16:

$$
2 \times 20 \times (4 \times 10^6) \times 8192 \times 64 \approx 8.39 \times 10^{13} \text{ bytes} \approx 84 \text{ TB}
$$

Insanely expensive, so we save only some activations and recompute (rematerialize) the rest on the way back:

- **Block remat:** Keep just 1 checkpoint per layer (its input) instead of 20, so 84 TB drops to 4.2 TB. The catch is we basically redo the whole forward pass, so training goes from $6$ to roughly $8 \times \text{tokens} \times \text{parameters}$.

- **Big matmuls only:** Keep every large matmul’s output and only recompute the cheap stuff (activation functions, parts of attention). Around 7 saved activations per layer, so about 29 TB.

In JAX, this is `jax.checkpoint` (also called `jax.remat`).

### KV Caching

Inference has 2 phases. Prefill runs the whole prompt once and saves every attention layer’s keys and values, then generation samples 1 token at a time and reuses them instead of re-processing old tokens. That saved array is the KV cache, shaped $[2, S, L, K, H]$ per sequence (keys plus values), so in int8:

$$
\text{KV cache} = 2 \cdot S \cdot L \cdot K \cdot H \text{ bytes}
$$

With an 8k context ($S = 8192$), $L = 64$ and $KH = NH = D = 8192$, that’s 8 GiB for one sequence (1 MiB per token), and every sequence in the batch needs its own. This is why we want GQA: with $K \ll N$, the cache shrinks by a factor of $N/K$.

### Takeaway

Most of this chapter fits in one table (per layer, training, MHA):

| **Component** | **Params** | **Train FLOPs** |
|---|---|---|
| **MLP** | $3DF$ | $18BTDF$ |
| **Attention** | $4DNH$ | $24BTDNH + 12BT^2NH$ |
| **Layernorms** | $2D$ | $O(BTD)$ |
| **Unembedding** | $DV$ (once, not per layer) | $6BTDV$ |

The MLP holds most of the parameters and FLOPs, which is why $6 \times \text{tokens} \times \text{parameters}$ works so well, and the dot products only take over once $T > 8D$. For inference, remember the KV cache: $2SLKH$ bytes per sequence in int8, and unlike the weights, it isn’t shared across the batch.

## Worked Problems

**Question 1 [counting parameters]:** A model has $D = 4096$, $F = 4D$, $V = 32{,}000$ and $L = 64$. How many parameters is that? What fraction of them belong to attention? And how large is the KV cache per token? Assume $NH = D$, multi-head attention ($N = K$), and int8 KVs.

- Just the takeaway table with numbers plugged in, with $F = 4 \times 4096 = 16384$.

* **How many parameters?**
    - Per layer, the MLP is $3DF = 3 \times 4096 \times 16384 = 201{,}326{,}592$, attention is $4DNH = 4 \times 4096 \times 4096 = 67{,}108{,}864$, and the 2 layernorms are $2D = 8{,}192$. That’s 268,443,648 per layer, or 17,180,393,472 across 64 layers. Plus the embedding and unembedding, $2DV = 2 \times 4096 \times 32{,}000 = 262{,}144{,}000$:

      $$
      \text{Params} = L(3DF + 4DNH + 2D) + 2DV = 17{,}180{,}393{,}472 + 262{,}144{,}000 = \mathbf{17{,}442{,}537{,}472}
      $$

    - So around **17.4B parameters**. The solution says 16B because it rounds the dims before multiplying.

* **What fraction are attention parameters?**
    - Ignoring the tiny terms:

      $$
      \frac{4DNH}{4DNH + 3DF} = \frac{4D^2}{4D^2 + 12D^2} = \mathbf{\frac{1}{4}}
      $$

    - With the exact count it’s $\frac{64 \times 67{,}108{,}864}{17{,}442{,}537{,}472} \approx 24.6\%$, since the vocab matrices take about 1.5%.

* **How large are the KV caches per token?**
    - Each token stores 1 key and 1 value vector per layer, each $KH = NH = D$ wide, at 1 byte per element in int8:

      $$
      2 \cdot L \cdot N \cdot H = 2 \times 64 \times 4096 = 524{,}288 \text{ bytes} = \mathbf{512 \text{ KiB per token}}
      $$

    - Meaning, an 8192 token context costs $524{,}288 \times 8192 = 4{,}294{,}967{,}296$ bytes, or 4 GiB of HBM for every sequence we serve.

**Question 2 [sharded matmul FLOPs]:** We run $A[B_X, D_Y] \cdot_D W[D_Y, F]$ on the mesh `{'X': 4, 'Y': 8, 'Z': 4}`. How many FLOPs does that take in total, and how many does each TPU perform?

- On paper this is $2BDF$ FLOPs. The catch is the mesh axis nobody is using.

* **How many FLOPs does each TPU perform?**
    - $B$ is split 4 ways over `X` and $D$ 8 ways over `Y`, so each chip just multiplies $A[B/4, D/8]$ by $W[D/8, F]$:

      $$
      \text{FLOPs per TPU} = 2 \cdot \frac{B}{4} \cdot \frac{D}{8} \cdot F = \frac{2BDF}{X \cdot Y} = \mathbf{\frac{BDF}{16}}
      $$

* **How many total FLOPs?**
    - Nothing is sharded over `Z`, so just like Question 1 in Chapter 3, the arrays are replicated along it and each of the 4 `Z` layers does the full matmul by itself. With $4 \times 8 \times 4 = 128$ chips:

      $$
      \text{Total FLOPs} = 128 \times \frac{BDF}{16} = 2BDF \times Z = \mathbf{8BDF}
      $$

    - So we burn $4\times$ the FLOPs the math actually needs. The unused axis costs us compute as well as HBM.

**Question 3 [einsum FLOPs]:** How many FLOPs does the contraction $A[I,J,K,L] * B[I,J,M,N,O] \to C[K,L,M,N,O]$ take?

- The ‘unique letters’ rule from the top of the chapter, we just sort the letters first.
- $I$ and $J$ are in both inputs but not the output, so they’re contracting dimensions. $K$, $L$ (only in $A$) and $M$, $N$, $O$ (only in $B$) all survive into the output. Nothing is in all three, so no batching dimensions.
- Every letter gets counted once, and since we do have contracting dimensions, we keep the 2:

  $$
  \text{FLOPs} = \mathbf{2 \cdot I \cdot J \cdot K \cdot L \cdot M \cdot N \cdot O}
  $$

**Question 4 [attention intensity]:** Ignoring the Q/K/V/O projections, what is the arithmetic intensity of grouped multi-query attention as a function of query length $T$, KV length $S$ and multi-query factor $G$ (query heads per KV head, so $N = KG$)? At what context length is attention FLOPs-bound? Then, using our TPUs’ HBM bandwidth, plot the effective cost of attention relative to the FFW block as the context grows. Hint: assume an efficient implementation with no unnecessary reads or writes, and look at both limits, $T = S$ and $T \ll S$. Assume TPU v5e numbers: $1.97 \times 10^{14}$ bf16 FLOPs/s and $8.2 \times 10^{11}\text{ bytes/s}$ of HBM bandwidth.

- The Chapter 1 intensity question, but for the dot product half of attention. Forward pass in bf16 (2 bytes per element), and ‘efficient’ means the $T \times S$ score matrix never gets written to HBM.

* **What is the arithmetic intensity?**
    - FLOPs: scoring $Q[B,T,K,G,H]$ against $K[B,S,K,H]$ and then multiplying by $V[B,S,K,H]$ is $2BTSKGH$ each, so $4BTSKGH$.
    - Bytes: read $Q$ ($2BTKGH$), $K$ and $V$ ($2BSKH$ each), and write an output shaped like $Q$ ($2BTKGH$). Total $4BKH(TG + S)$.

      $$
      \text{Intensity} = \frac{4BTSKGH}{4BKH(TG + S)} = \mathbf{\frac{TSG}{TG + S}}
      $$

    - Notice that $B$ cancels. A bigger batch doesn’t help here, since every sequence brings its own keys and values.

* **At what context length is attention FLOPs-bound?**
    - For prefill or training, $S = T$, so the intensity is $\frac{T^2G}{T(G+1)} = \frac{G}{G+1}T$, linear in context. The v5e needs $\frac{1.97 \times 10^{14}}{8.2 \times 10^{11}} \approx 240\text{ FLOPs/byte}$:

      $$
      \frac{G}{G+1}T > 240 \implies T > 240 \cdot \frac{G+1}{G}
      $$

    - For plain MHA ($G = 1$) that’s $\mathbf{T > 480}$ **tokens**. For $G = 8$ it’s 270, and large $G$ approaches 240. The solution says 240 because it drops the $\frac{G}{G+1}$ factor.
    - For generation, $T = 1$, so the intensity is $\frac{SG}{G + S} \to G$. With MHA we’re permastuck at $1\text{ FLOP/byte}$, and $G = 8$ only gets us to 8. Attention is never FLOPs-bound during generation, no matter the context or batch.

* **How does the cost of attention compare to the FFW block as context grows?**
    - There’s no published answer for this part, so this is our own derivation. ‘Effective’ cost means the time the chip actually spends, $\max(T_{\text{math}}, T_{\text{HBM}})$. With $F = 4D$ and $NH = D$, the FFW forward pass is $24BTD^2$ FLOPs and the attention dot products are $4BTSD$.
    - **Prefill ($S = T$):** The FFW block is compute-bound past 240 tokens (Chapter 1), so if attention is too, the ratio is $\frac{4BT^2D}{24BTD^2} = \frac{T}{6D}$. Below the critical context, attention waits on HBM and costs as if $T$ were sitting at that length:

      $$
      \frac{\text{attention}}{\text{FFW}} = \frac{\max\left(T, \ 240 \cdot \frac{G+1}{G}\right)}{6D}
      $$

    - A flat floor, then a straight line that crosses 1 at $T = 6D = 24{,}576$ tokens for $D = 4096$.
    - **Generation ($T = 1$):** Attention is memory-bound, reading about $\frac{4BSD}{G}$ bytes per layer each step, while the FFW block reads $24D^2$ bytes of weights shared by the whole batch, so its time is $\frac{24D^2 \cdot \max(B, 240)}{1.97 \times 10^{14}}$. Dividing:

      $$
      \frac{\text{attention}}{\text{FFW}} \approx \frac{BS}{6GD} \ \text{ for } B \le 240, \qquad \frac{40S}{GD} \ \text{ for } B \ge 240
      $$

    - Still linear in $S$, but the slope grows with the batch. Up to $B = 240$, attention passes the FFW block once $B \times S > 6GD$ cached tokens (24,576 for $D = 4096$ and $G = 1$), and past that at only around 100 tokens of context.

Instead of a plot, here’s the curve at a few points for $D = 4096$ and $G = 1$ (attention time over FFW time):

| **Context** | **Prefill ($T = S$)** | **Generate, $B = 1$** | **Generate, $B = 32$** | **Generate, $B \ge 240$** |
|---|---|---|---|---|
| 1,024 | 0.042 | 0.042 | 1.33 | 10 |
| 8,192 | 0.33 | 0.33 | 10.7 | 80 |
| 32,768 | 1.33 | 1.33 | 42.7 | 320 |

> In prefill and training, attention stays cheap until the context is several times $D$. In generation it flips: the weights load once for everyone, but the KV caches scale with batch times context, which is why a larger $G$ matters so much for serving.

**Question 5 [dot products vs projections]:** For self-attention, at what sequence length do the dot product FLOPs equal the QKVO projection FLOPs?

- Just set the 2 attention costs from the notes equal. With MHA, the projections are $24BTDNH$ and the dot products are $12BT^2NH$:

  $$
  24BTDNH = 12BT^2NH \implies \mathbf{T = 2D}
  $$

- For $D = 4096$, that’s **8192 tokens**. Below that, attention spends more FLOPs on setup than on the actual matching.

**Question 6 [remat FLOPs]:** If the forward pass only saves the outputs of the 7 main matmuls in each transformer layer (Q, K, V, O, plus the 3 FFW matrices), how many extra FLOPs does the backward pass need to ‘rematerialize’ everything else?

- Each weight gradient needs the input that was fed into that matmul, so which of the 7 inputs can’t we rebuild cheaply?
- **Q, K, V and the 2 FFW input matrices:** Their input is the normalized residual stream, just adds and a layernorm, so $O(BTD)$.
- **FFW output matrix:** Its input is the gated product of the 2 saved FFW activations, an elementwise op, so $O(BTF)$.
- **O:** Its input is $\text{softmax}(QK^T)V$, which we didn’t save. We have $Q$, $K$ and $V$, so we redo both attention matmuls, forward only, $2BT^2NH$ each:

  $$
  \text{Extra FLOPs} = 2BT^2NH + 2BT^2NH = \mathbf{4BT^2NH}
  $$

- Plus the $O(BTD)$ and $O(BTF)$ leftovers, which we ignore as usual.
- What this means: the dot product bill goes from $12BT^2NH$ to $16BT^2NH$, a third more, but against the whole layer ($F = 4D$, MHA) the extra is only $\frac{T}{24D + 3T}$, about 6.7% for $D = 4096$ and $T = 8192$. Block remat adds about 33% ($6\times$ to $8\times$).

**Question 7 [DeepSeek utilization]:** DeepSeek v3 reports 2.79M H800 hours of training on 14.8T tokens ([source](https://arxiv.org/pdf/2412.19437v1)), with 37B activated parameters. Roughly what hardware utilization is that? Hint: they trained with FP8 FLOPs and no structured sparsity. Assume the H800 rate from [this spec sheet](https://lenovopress.lenovo.com/lp1814.pdf): 3,026 TFLOPs/s of FP8 with sparsity, so half of that, $1.513 \times 10^{15}\text{ FLOPs/s}$, without.

- Utilization is the FLOPs the model needed over the FLOPs the hardware could have produced.
- Needed, from the golden rule with the activated parameters (it’s an MoE):

  $$
  6 \times (37 \times 10^9) \times (14.8 \times 10^{12}) \approx 3.29 \times 10^{24} \text{ FLOPs}
  $$

- Available: 2.79M hours is $2.79 \times 10^6 \times 3600 \approx 1.004 \times 10^{10}$ GPU-seconds:

  $$
  1.004 \times 10^{10} \text{ s} \times 1.513 \times 10^{15} \text{ FLOPs/s} \approx 1.52 \times 10^{25} \text{ FLOPs}
  $$

- Then we just divide:

  $$
  \text{Utilization} = \frac{3.29 \times 10^{24}}{1.52 \times 10^{25}} \approx \mathbf{21.6\%}
  $$

- So they got roughly **21.6%** of peak FP8, about 1/5 of what the chips could theoretically do. The rule ignores the attention dot products, so the real number is a bit higher.
- The H800 is basically an H100 with less network bandwidth, so at the H100’s $2.0 \times 10^{15}$ fp8 FLOPs/s instead we’d get about 16.4%.

**TODO(eddie):** 21.6% uses the spec sheet's 3,026 TFLOPs/s and 16.4% the H100's $2.0 \times 10^{15}$. Check which FP8 figure fits the H800s in the paper before publishing, and keep both visible if it stays unclear.

**Question 8 [MoE batch size]:** An MoE layer holds $E$ copies of the dense MLP block, and every token gets routed through $k$ of them. With int8 weights on a TPU v5e, how many tokens do we need in a batch to be compute-bound? What does that come to for DeepSeek, with 256 (routed) experts and $k = 8$? Assume our HBM bandwidth is $8.2 \times 10^{11}\text{ bytes/s}$ and we do the math in bf16 at $1.97 \times 10^{14}\text{ FLOPs/s}$.

- This is Question 2 from Chapter 1 again (int8 weights, bf16 compute), where the dense answer was $2B > 240$, so $B > 120$. The only change is that we now load $E$ weight matrices and each token only uses $k$ of them.
- Bytes: all $E$ experts at 1 byte per parameter, so $E \cdot D \cdot F$. FLOPs: each of the $B$ tokens goes through $k$ experts, so $2 \cdot k \cdot B \cdot D \cdot F$. We ignore the activations like before:

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

- Meaning, the dense threshold of 120 just gets multiplied by the sparsity (32 here): each expert still needs its own 120 tokens on average, assuming the router spreads tokens evenly. Fine for training or prefill, but a massive batch for generation, where each sequence only adds 1 token per step.
