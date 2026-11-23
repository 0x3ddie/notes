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

$$C[G, H, I, J, K, L] \times D[G, H, M, N, K, L]$$

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

$$\text{Forward Passes} + \text{Backward Passes (Weights)} + \text{Backward Passes (Activations)} = 6 \text{ FLOPs per parameter, per token}$$

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

- **Projections:** Linear cost in $T$. Attention carries 4 weight matrices: $W_Q$, $W_K$ and $W_V$ turn the activations into queries, keys and values, and $W_O$ maps the result back to width $D$. The Q and O matrices are each $D \times NH$, the K and V ones $D \times KH$. So we get $2D(N+K)H$ parameters, and the $6\times$ rule gives $12BTD(N+K)H$ train FLOPs. For plain MHA ($N = K$ and $NH = D$) that’s just $4D^2$ parameters and $24BTD^2$ FLOPs.

- **Dot Products:** Quadratic cost in $T$. Every query gets scored against every key, we softmax, then take a weighted sum of the values. That’s 2 matmuls ($[T,H] \cdot [H,S]$, then $[T,S] \cdot [S,H]$), each costing $6BTSNH$ in training across $B$ sequences and $N$ heads. With self-attention ($S = T$) the total is $12BT^2NH$. No parameters here at all, it’s pure compute.

As a note, with a causal mask half of those scores are never needed, but only a dedicated attention kernel (not a plain einsum) actually skips them. The book’s [transformer diagram](https://jax-ml.github.io/scaling-book/transformers/) shows the shapes of each einsum.

### 3. Everything Else

- **Layernorms:** 2 per layer, costing $O(BTD)$ FLOPs and $2D$ parameters. A rounding error next to the matmuls, so we just ignore them.

- **Unembedding:** At the very end, a $[D,V]$ matrix turns the activations into a score for every token in the vocab. That’s $DV$ parameters and $6BTDV$ train FLOPs, paid once and not per layer. The embedding table on the way in is another $DV$ parameters, but it’s a lookup, not a matmul.

### 4. The Rule of Thumb

Say we ignore the dot products for now (fine for shorter contexts) and add up the MLP and the projections across $L$ layers:

$$
(18BTDF + 12BTD(N+K)H) \cdot L = 6 \cdot BT \cdot (3DF + 2D(N+K)H) \cdot L
$$

The bracket on the right is just the parameter count of one layer, and $BT$ is the number of tokens. So the precise accounting lands right back on the golden rule, $6 \times \text{tokens} \times \text{parameters}$. Divide by 3 for inference (forward pass only): $2 \times \text{tokens} \times \text{parameters}$.

### 5. When Attention Takes Over

Now we put the dot products back in. Assume the typical setup of $F = 4D$, $NH = D$ and $N = K$. The MLP is $18BTDF = 72BTD^2$, the projections are $24BTD^2$, and the dot products are $12BT^2D$:

$$
\frac{\text{dot product FLOPs}}{\text{all other matmul FLOPs}} = \frac{12BT^2D}{72BTD^2 + 24BTD^2} = \frac{12BT^2D}{96BTD^2} = \frac{T}{8D}
$$

This is where $T > 8D$ comes from. The quadratic part of attention only becomes the biggest line in the budget once the context is more than 8 times the model width.

For a big model with $D = 8192$, the crossover is at 65,536 tokens (around 64k). For a smaller one like Gemma 27B with $D = 4608$, it’s already at 36,864 (around 37k). So why is long context worse for smaller models? The MLP and the projections scale with $D^2$, but the dot products only scale with $D$. A wide model has a massive MLP bill that hides the attention cost for much longer.

> **Note:** The book’s appendix on Flash Attention covers the memory side of long context (the FLOPs stay quadratic). I’m skipping it here.

## Miscellaneous Math

### Sparsity and Mixture-of-Experts

An MoE (Mixture of Experts) model swaps the single MLP in each layer for $E$ independent MLPs, called experts, and a router picks only $k$ of them for each token. Parameters grow with $E$, because we have to store every expert. FLOPs per token only grow with $k$, so for an MoE the $6\times$ rule uses the activated parameters, not the total.

We call $E/k$ the sparsity. Typical values run from 8 to 64; DeepSeek v3 has $E = 256$ and $k = 8$, so 32.

What we pay for this is networking: tokens have to travel to the chip holding their expert and then come back, which adds 2 AllToAlls around every MoE block. From Chapter 3, an AllToAll is about $\frac{1}{4}$ the cost of an AllGather, so it’s cheap, but not free.

### Gradient Checkpointing

Backprop spends memory to save compute: the backward pass needs the intermediate activations from the forward pass, so by default we save all of them. How much is that? Take a batch of 4M tokens ($BT = 4 \times 10^6$), $L = 64$, $D = 8192$, and around 20 saved activations per layer, all in bf16:

$$
2 \times 20 \times (4 \times 10^6) \times 8192 \times 64 \approx 8.39 \times 10^{13} \text{ bytes} \approx 84 \text{ TB}
$$

That’s insanely expensive, so instead we save only some activations and recompute (rematerialize) the rest on the way back:

- **Block remat:** Keep just 1 checkpoint per layer (its input) instead of 20, so 84 TB drops to 4.2 TB. The catch is that we basically redo the whole forward pass, so training goes from $6$ to roughly $8 \times \text{tokens} \times \text{parameters}$.

- **Big matmuls only:** Keep the output of every large matmul, so only the cheap stuff (activation functions, parts of attention) gets recomputed. That’s around 7 saved activations per layer, which by the same formula is about 29 TB.

In JAX, this is `jax.checkpoint` (also called `jax.remat`).

### KV Caching

Inference has 2 phases. Prefill runs the whole prompt through the model once and saves the key and value projections from every attention layer. Generation then samples 1 token at a time and reuses them, so old tokens never get re-processed. That saved array is the KV cache. For 1 sequence it’s shaped $[2, S, L, K, H]$ (the 2 is keys plus values), so in int8:

$$
\text{KV cache} = 2 \cdot S \cdot L \cdot K \cdot H \text{ bytes}
$$

Plug in an 8k context ($S = 8192$), $L = 64$ and $KH = NH = D = 8192$, and that’s $2 \times 8192 \times 64 \times 8192 = 8{,}589{,}934{,}592$ bytes, or 8 GiB for a single sequence (1 MiB per token). And every sequence in the batch needs its own. This is why we want GQA: with $K \ll N$, the cache shrinks by a factor of $N/K$.

### Takeaway

Most of this chapter fits in one table (per layer, training, assuming MHA):

| **Component** | **Params** | **Train FLOPs** |
|---|---|---|
| **MLP** | $3DF$ | $18BTDF$ |
| **Attention** | $4DNH$ | $24BTDNH + 12BT^2NH$ |
| **Layernorms** | $2D$ | $O(BTD)$ |
| **Unembedding** | $DV$ (once, not per layer) | $6BTDV$ |

As a note, the book’s own summary table lists $12BTDV$ for the vocab row, but its per-operation table and the text next to the rule of thumb both say $6BTDV$. That’s also what the $6\times$ rule gives for a single $[D,V]$ matmul ($2BTDV$ forward, $4BTDV$ backward), so we use $6BTDV$ here.

The MLP holds most of the parameters and most of the FLOPs, which is why $6 \times \text{tokens} \times \text{parameters}$ works so well for training (and $2\times$ for inference). The quadratic dot products only take over once $T > 8D$. For inference, the number to remember is the KV cache: $2SLKH$ bytes per sequence in int8, and unlike the weights, it isn’t shared across the batch.

## Worked Problems

**Question 1 [counting parameters]:** A model has $D = 4096$, $F = 4D$, $V = 32{,}000$ and $L = 64$. How many parameters is that? What fraction of them belong to attention? And how large is the KV cache per token? Assume $NH = D$, multi-head attention ($N = K$), and int8 KVs.

- This question is just the takeaway table with numbers plugged in. We have $F = 4 \times 4096 = 16384$.

* **How many parameters?**
    - Per layer, the MLP is $3DF = 3 \times 4096 \times 16384 = 201{,}326{,}592$, attention is $4DNH = 4 \times 4096 \times 4096 = 67{,}108{,}864$, and the 2 layernorms are $2D = 8{,}192$. That’s 268,443,648 parameters per layer, or 17,180,393,472 across 64 layers.
    - Outside the layers we have the embedding and unembedding matrices, $2DV = 2 \times 4096 \times 32{,}000 = 262{,}144{,}000$.

      $$
      \text{Params} = L(3DF + 4DNH + 2D) + 2DV = 17{,}180{,}393{,}472 + 262{,}144{,}000 = \mathbf{17{,}442{,}537{,}472}
      $$

    - So around **17.4B parameters**. The solution says 16B: it’s the same formula, but with $D$ rounded to 4e3 and $F$ to 16e3 before multiplying (which gives 1.66e10), and the result then rounded down. If the embedding and unembedding shared 1 matrix, we’d count $DV$ once and get 17,311,465,472, so 17.3B.

* **What fraction are attention parameters?**
    - Ignoring the tiny terms, we can do this without any numbers:

      $$
      \frac{4DNH}{4DNH + 3DF} = \frac{4D^2}{4D^2 + 12D^2} = \mathbf{\frac{1}{4}}
      $$

    - With the exact count it’s $\frac{64 \times 67{,}108{,}864}{17{,}442{,}537{,}472} \approx 24.6\%$, since the vocab matrices take about 1.5% for themselves. The MLPs hold basically everything else.

* **How large are the KV caches per token?**
    - Each token stores 1 key and 1 value vector per layer, and with MHA they’re each $KH = NH = D$ wide. In int8 that’s 1 byte per element:

      $$
      2 \cdot L \cdot N \cdot H = 2 \times 64 \times 4096 = 524{,}288 \text{ bytes} = \mathbf{512 \text{ KiB per token}}
      $$

    - Meaning, an 8192 token context costs $524{,}288 \times 8192 = 4{,}294{,}967{,}296$ bytes, or 4 GiB of HBM for every sequence we’re serving.

**Question 2 [sharded matmul FLOPs]:** We run $A[B_X, D_Y] \cdot_D W[D_Y, F]$ on the mesh `{'X': 4, 'Y': 8, 'Z': 4}`. How many FLOPs does that take in total, and how many does each TPU perform?

- On paper this matmul is $2BDF$ FLOPs. The catch is the mesh axis that nobody is using.

* **How many FLOPs does each TPU perform?**
    - $B$ is split 4 ways over `X`, and $D$ is split 8 ways over `Y` on both arrays. So each chip holds $A[B/4, D/8]$ and $W[D/8, F]$, and multiplies just those 2 local pieces:

      $$
      \text{FLOPs per TPU} = 2 \cdot \frac{B}{4} \cdot \frac{D}{8} \cdot F = \frac{2BDF}{X \cdot Y} = \mathbf{\frac{BDF}{16}}
      $$

* **How many total FLOPs?**
    - Nothing is sharded over `Z`. Just like Question 1 in Chapter 3, that means the arrays are replicated along it, so each of the 4 `Z` layers holds a full copy and does the full matmul by itself. We have $4 \times 8 \times 4 = 128$ chips:

      $$
      \text{Total FLOPs} = 128 \times \frac{BDF}{16} = 2BDF \times Z = \mathbf{8BDF}
      $$

    - So we burn $4\times$ the FLOPs the math actually needs. The unused axis costs us compute as well as HBM.
    - One more thing: because $D$ (the contracting dimension) is sharded on both sides, each chip only has a partial sum, so we’d still need an AllReduce or ReduceScatter over `Y` to finish. That’s communication though, it doesn’t change the FLOP count.

**Question 3 [einsum FLOPs]:** How many FLOPs does the contraction $A[I,J,K,L] * B[I,J,M,N,O] \to C[K,L,M,N,O]$ take?

- This is the ‘unique letters’ rule from the top of the chapter, we just need to sort the letters first.
- $I$ and $J$ show up in both inputs but not in the output, so they’re the contracting dimensions (the ones we sum over).
- $K$ and $L$ only live in $A$, and $M$, $N$, $O$ only live in $B$. They all survive into the output.
- A batching dimension would have to be in both inputs and the output. There are none here.
- So every letter gets counted once, and since we do have contracting dimensions, we keep the 2:

  $$
  \text{FLOPs} = \mathbf{2 \cdot I \cdot J \cdot K \cdot L \cdot M \cdot N \cdot O}
  $$

- Another way to see it: the output has $KLMNO$ elements, and each one is a dot product of length $IJ$. If the 2 inputs shared a batch axis, we’d still only count it once.

**Question 4 [attention intensity]:** Ignoring the Q/K/V/O projections, what is the arithmetic intensity of grouped multi-query attention, as a function of the query length $T$, the KV length $S$ and the multi-query factor $G$ (query heads per KV head, so $N = KG$)? At what context length does attention become FLOPs-bound? Then, using the HBM bandwidth of our TPUs, plot how the effective cost of attention relative to the FFW block changes as the context grows. Hint: assume an efficient attention implementation with no unnecessary reads or writes, and look at both limits, $T = S$ and $T \ll S$. Assume TPU v5e numbers: $1.97 \times 10^{14}$ bf16 FLOPs/s and $8.2 \times 10^{11}\text{ bytes/s}$ of HBM bandwidth.

- This is the Chapter 1 intensity question (FLOPs over bytes moved), but for the dot product half of attention. We’re looking at a forward pass in bf16 (2 bytes per element), and ‘efficient’ means the $T \times S$ score matrix never gets written to HBM.

* **What is the arithmetic intensity?**
    - FLOPs: scoring $Q[B,T,K,G,H]$ against $K[B,S,K,H]$ is $2BTSKGH$, and multiplying the softmaxed scores by $V[B,S,K,H]$ is another $2BTSKGH$. Total is $4BTSKGH$.
    - Bytes: we read $Q$ ($2BTKGH$), read $K$ and $V$ ($2BSKH$ each), and write an output with the same shape as $Q$ ($2BTKGH$). Total is $4BTKGH + 4BSKH = 4BKH(TG + S)$.

      $$
      \text{Intensity} = \frac{4BTSKGH}{4BKH(TG + S)} = \mathbf{\frac{TSG}{TG + S}}
      $$

    - Notice that $B$ cancels. For a normal weight matmul the intensity is basically the batch size, but here a bigger batch doesn’t help, because every sequence brings its own keys and values.

* **At what context length is attention FLOPs-bound?**
    - For prefill or training, $S = T$, so the intensity is $\frac{T^2G}{T(G+1)} = \frac{G}{G+1}T$. It grows linearly with context. The v5e needs $\frac{1.97 \times 10^{14}}{8.2 \times 10^{11}} \approx 240\text{ FLOPs/byte}$ to be compute-bound:

      $$
      \frac{G}{G+1}T > 240 \implies T > 240 \cdot \frac{G+1}{G}
      $$

    - For plain MHA ($G = 1$) that’s $\mathbf{T > 480}$ **tokens**. For $G = 8$ it’s 270, and as $G$ gets large it approaches 240.
    - The solution says $S = 240$. It writes the prefill intensity as $\frac{TG}{G+1} = O(T)$ and then treats it as just $T$, dropping the $\frac{G}{G+1}$ factor, so its number is our large $G$ limit. The book’s inference section keeps the factor for MHA ($\frac{ST}{S+T} = \frac{T}{2}$) and quotes roughly 480 tokens, which matches our $G = 1$ case.
    - For generation, $T = 1$ and $S$ is large, so the intensity is $\frac{SG}{G + S} \to G$. With MHA we’re permastuck at $1\text{ FLOP/byte}$, and even $G = 8$ only gets us to 8. That’s nowhere near 240, so attention is never FLOPs-bound during generation, no matter how long the context or how big the batch.

* **How does the cost of attention compare to the FFW block as context grows?**
    - The solution stops at the intensity and never gets to this part, so what follows is our own derivation, with no published answer to check it against.
    - We read ‘effective’ cost as the time the chip actually spends, $\max(T_{\text{math}}, T_{\text{HBM}})$, instead of a raw FLOP count. Take $F = 4D$ and $NH = D$. A forward pass through the FFW block is 3 matmuls, so $6BTDF = 24BTD^2$ FLOPs, and the attention dot products are $4BTSD$.
    - **Prefill ($S = T$):** The FFW block is compute-bound as long as we have more than 240 tokens (Chapter 1). If attention is also compute-bound, the ratio is just FLOPs over FLOPs, $\frac{4BT^2D}{24BTD^2} = \frac{T}{6D}$. Below the critical context from the last part, attention is waiting on HBM, and it costs as much as if $T$ were sitting at that critical length:

      $$
      \frac{\text{attention}}{\text{FFW}} = \frac{\max\left(T, \ 240 \cdot \frac{G+1}{G}\right)}{6D}
      $$

    - So on a plot it’s a flat floor followed by a straight line. For $D = 4096$ and $G = 1$, the floor is $\frac{480}{24{,}576} \approx 0.02$, and the line crosses 1 at $T = 6D = 24{,}576$ tokens.
    - **Generation ($T = 1$):** Attention is memory-bound, and reads about $\frac{4BSD}{G}$ bytes per layer on every step. The FFW block does $24BD^2$ FLOPs but only reads $24D^2$ bytes of weights, which the whole batch shares, so its time is $\frac{24D^2 \cdot \max(B, 240)}{1.97 \times 10^{14}}$. Dividing the 2 times:

      $$
      \frac{\text{attention}}{\text{FFW}} \approx \frac{BS}{6GD} \ \text{ for } B \le 240, \qquad \frac{40S}{GD} \ \text{ for } B \ge 240
      $$

    - Still a straight line in $S$, but now the slope grows with the batch size. For batches up to 240, attention passes the FFW block once the batch is holding more than $6GD$ tokens of cached context in total ($B \times S$), which is only 24,576 tokens for $D = 4096$ and $G = 1$. Past 240, it only takes $S > \frac{GD}{40}$, around 100 tokens.

Here’s the curve at a few points, for $D = 4096$ and $G = 1$ (attention time divided by FFW time):

| **Context** | **Prefill ($T = S$)** | **Generate, $B = 1$** | **Generate, $B = 32$** | **Generate, $B \ge 240$** |
|---|---|---|---|---|
| 1,024 | 0.042 | 0.042 | 1.33 | 10 |
| 8,192 | 0.33 | 0.33 | 10.7 | 80 |
| 32,768 | 1.33 | 1.33 | 42.7 | 320 |

For a feel of the last column: at $S = 8192$, one sequence’s keys and values are about 134 MB per layer, which takes around 164 μs to read from HBM, while its share of the FFW math is around 2 μs.

> During prefill and training, attention stays cheap next to the FFW block until the context is several times $D$. During generation it flips: the weights are loaded once for everyone, but the KV caches scale with batch size times context. That’s why a larger $G$ matters so much for serving.

The question asks for a plot; the table above is the same curve sampled at a few points (on log-log axes it’s a flat floor and then a straight line).

**Question 5 [dot products vs projections]:** For self-attention, at what sequence length do the dot product FLOPs equal the QKVO projection FLOPs?

- We just set the 2 attention costs from the notes equal to each other. With MHA, the projections are $24BTDNH$ and the dot products are $12BT^2NH$:

  $$
  24BTDNH = 12BT^2NH \implies \mathbf{T = 2D}
  $$

- For $D = 4096$, that’s **8192 tokens**. Below that, the attention block spends more FLOPs on its setup than on the actual matching.
- With GQA, the K and V projections get smaller, so the crossover comes earlier: $12BTD(N+K)H = 12BT^2NH$ gives $T = D\left(1 + \frac{1}{G}\right)$, which for $G = 8$ and $D = 4096$ is 4608 tokens.

**Question 6 [remat FLOPs]:** Say that during the forward pass we only save the outputs of the 7 main matmuls in each transformer layer (Q, K, V, O, plus the 3 FFW matrices). How many extra FLOPs does the backward pass need to ‘rematerialize’ everything else?

- To get the gradient of any weight matrix, we need the input that was fed into that matmul. So the real question is: which of the 7 inputs can’t we rebuild cheaply from what we saved?
- **Q, K, V and the 2 FFW input matrices:** Their input is the normalized residual stream. Rebuilding it is just adds and a layernorm, so $O(BTD)$.
- **FFW output matrix:** Its input is the gated product of the 2 saved FFW input activations. That’s an elementwise op, so $O(BTF)$.
- **O:** Its input is the output of the attention dot products, $\text{softmax}(QK^T)V$, and we didn’t save that. We have $Q$, $K$ and $V$, so we have to redo both attention matmuls. Each is a forward pass only, $2BT^2NH$:

  $$
  \text{Extra FLOPs} = 2BT^2NH + 2BT^2NH = \mathbf{4BT^2NH}
  $$

- Plus the $O(BTD)$ and $O(BTF)$ leftovers, which we ignore as usual. Redoing the dot products also gives us back the attention scores, which we need to backprop through the softmax anyway.
- What this means: the dot product bill goes from $12BT^2NH$ to $16BT^2NH$, so a third more. But against the whole layer it’s small. With $F = 4D$ and MHA, the extra is $\frac{T}{24D + 3T}$ of the total train FLOPs, which for $D = 4096$ and $T = 8192$ is about 6.7%. Compare that with block remat, which adds about 33% (from $6\times$ to $8\times$).

**Question 7 [DeepSeek utilization]:** DeepSeek v3 reports 2.79M H800 hours of training on 14.8T tokens ([source](https://arxiv.org/pdf/2412.19437v1)), and the model has 37B activated parameters. Roughly what hardware utilization does that work out to? Hint: they trained with FP8 FLOPs and no structured sparsity. Assume the H800 figure the book’s solution takes from [this spec sheet](https://lenovopress.lenovo.com/lp1814.pdf): 3,026 TFLOPs/s of FP8 with sparsity, so half of that, $1.513 \times 10^{15}\text{ FLOPs/s}$, without.

- Utilization is just the FLOPs the model needed divided by the FLOPs the hardware could have produced in that time.
- For the FLOPs needed, we use the golden rule. It’s an MoE, so we plug in the activated parameters:

  $$
  6 \times (37 \times 10^9) \times (14.8 \times 10^{12}) \approx 3.29 \times 10^{24} \text{ FLOPs}
  $$

- For the FLOPs available, 2.79M hours is $2.79 \times 10^6 \times 3600 \approx 1.004 \times 10^{10}$ GPU-seconds:

  $$
  1.004 \times 10^{10} \text{ s} \times 1.513 \times 10^{15} \text{ FLOPs/s} \approx 1.52 \times 10^{25} \text{ FLOPs}
  $$

- Then we just divide:

  $$
  \text{Utilization} = \frac{3.29 \times 10^{24}}{1.52 \times 10^{25}} \approx \mathbf{21.6\%}
  $$

- So they got roughly **21.6%** of the peak FP8 FLOPs (the solution rounds its way to 21.7%). Meaning, about 1/5 of what the chips could theoretically do went into the matmuls our rule counts. The rule ignores the attention dot products, so the real number is a bit higher.
- The answer moves with the rate we assume. The book’s GPU section gives the H800 the same FLOPs/s as an H100 (just less network bandwidth), and its spec table lists $2.0 \times 10^{15}$ fp8 FLOPs/s for the H100. With that rate we’d get about 16.4% instead.

**TODO(eddie):** 21.6% follows the spec sheet rate in the book’s solution, and 16.4% follows the book’s own GPU table. Check which FP8 figure fits the H800s in the paper before publishing, and keep both visible if it stays unclear.

**Question 8 [MoE batch size]:** An MoE layer holds $E$ copies of the dense MLP block, and every token gets routed through $k$ of them. With int8 weights on a TPU v5e, how many tokens do we need in a batch to be compute-bound? What does that come to for DeepSeek, with 256 (routed) experts and $k = 8$? Assume our HBM bandwidth is $8.2 \times 10^{11}\text{ bytes/s}$ and we do the math in bf16 at $1.97 \times 10^{14}\text{ FLOPs/s}$.

- This is Question 2 from Chapter 1 again (int8 weights, bf16 compute), where the dense answer was $2B > 240$, so $B > 120$. The only thing that changes is that we now have $E$ weight matrices to load, and each token only uses $k$ of them.
- Bytes: for each weight matrix we load all $E$ experts at 1 byte per parameter, so $E \cdot D \cdot F$ bytes. We ignore the activations like before.
- FLOPs: each of the $B$ tokens goes through $k$ experts, so $2 \cdot k \cdot B \cdot D \cdot F$.

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

- Meaning, the dense threshold of 120 just gets multiplied by the sparsity (32 here). Each expert still needs its own 120 tokens on average before its weights are worth the load, and that assumes the router spreads tokens evenly.
- That’s fine for training or prefill, but it’s a massive batch for generation, where each sequence only adds 1 token per step. And if the math also ran in int8 ($3.94 \times 10^{14}\text{ OPs/s}$), the bar doubles to 7,680 tokens.
