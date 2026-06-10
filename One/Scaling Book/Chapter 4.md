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

- **Projections:** Linear cost