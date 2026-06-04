---
title: A Brief Intro to Roofline Analysis
description: Chapter 1 worked problems
published: true
---
## Introduction

### Why do TPUs matter?

Tensor Processing Units (TPUs) are specialized hardware (AI-specific ASICs, or application-specific integrated circuits) used for solving matmul (matrix multiplication) problems at higher speed and less power than any other hardware, including GPUs.

At a broad glance:

- **TPUs** dominate in pre-training LLMs.
    
- **GPUs** dominate in high throughput, video/general ML, and general-purpose use cases (OS, word processing, web servers, etc).
    

Two of the three SOTA models have been trained on TPUs over GPUs (Claude Opus, Google Gemini), so the timing of this writeup could not be more perfect.

> **Critique:** It should be known that while TPUs often offer better tokens/dollar compared to even SOTA Nvidia chips, they generally struggle with non-specialized algorithms or dynamic computation graphs. GPUs historically have been used to innovate because developers can worry about seamless customization and playing around instead of hyper-optimizing for the TPU systolic array, which certainly provides better situational computation. However, it’s likely this changes with how well-maintained and increasingly popular JAX is becoming; you may have seen recently that Meta has begun utilizing Google’s Trillium chips for some of their AI workloads.

### Why is this called the ‘Scaling Book’? What is scaling and why is it important?

Hopefully not to anyone's surprise, the AI game is not purely about tweaking and improving AI algorithms; rather we need to constantly think about how we use our software with hardware constraints. Cutting edge software is unavoidably tied to how efficiently we can use or scale on hardware.

In fact, the perceptron and nascent concept of a ‘neural net’ have been around for a long time, but only became feasibly useful after Alex Krizhevsky wrote incredible CUDA code to be able to use GPUs and materialize AlexNet, one of the first proofs of life of neural nets. Neural nets were essentially useless until then, unable to scale and produce meaningful work on sluggish CPU chips.

The goal of scaling is to be able to use more chips for training or inference while having a proportional increase in throughput. Ideally, we can always just add more chips (parallelism) and get more compute for the model to work with. Realistically, though, this often comes at the cost of added communication overhead between chips, and we become communication bound, unable to scale strongly.

- **Example:** Let's say we have a small model running on a GPU, but we want to scale up since we have more users. But, when we add more GPUs, we find that with each additional unit, we suffer more communication lag; say each additional GPU is added only at 90% efficiency. So, no matter how much money we spend on hardware, we’ll inevitably hit a wall, and likely very early. We want to avoid this.
    

## Rooflines (Constraints)

Algorithms have 3 hard constraints:

1. How fast the computer performs math (Ops/s)
    
2. The bandwidth available to move data around (bytes/s)
    
3. How much storage we have (memory in bytes)
    

Knowing our constraints allows us to find the upper and lower bounds of time of any given computation. So why does an algorithm take a certain amount of time, like 50ms, instead of 50 seconds? This is due to computation, communication between chips, and communication within a chip.

### A Quick Note Before Math

I think its worth clearing up what FLOPs, OPs, TOPs, and TFLOPs are in the beginning:

- **OPs** are just operations. A single operation is like a single $+$, a single $*$, or a single division.
    
- **FLOP** is just an operation for floating point numbers (basically any decimals). Because we use fp numbers so frequently, we generally refer to operations as FLOPs instead of OPs, which are more generic. So a single $+$ operation between 2 decimal numbers is 1 FLOP; it’s an OP for fp numbers.
    
- **TOPs** (Trillion operations per second): A spec sheet that says 100 TOPs usually implies int4/int8 math, which would typically be seen in highly quantized/optimized applications.
    
- **TFLOPs**: 100 Trillion floating point operations are just written as 100 TFLOPs ($10^{12}$).
    
- We have more weird naming like **GOPs/GFLOPs** which are billions of (fl)ops per second ($10^9$), and **POPs/PFLOPs** which are in the quadrillions of (fl)ops ($10^{15}$).
    

### On Computation

At its core, a deep learning model is just a bunch of matmuls, with their operations of floating-point multiplications being known as ‘FLOPs’. Compute time can thus be calculated as:

$$\text{Compute Time} = \frac{\text{Computation FLOPs}}{\text{Accelerator FLOPs}}$$

We take a target # of FLOPs, like $1 \times 10^{12}$, and divide it by how quickly a piece of hardware can perform FLOPs; a Trillium (6th Gen TPU) chip can perform roughly $9.1 \times 10^{14}$ FLOPs.

$$\frac{1 \times 10^{12} \text{ FLOPs}}{9.1 \times 10^{14} \text{ FLOPs/s}} = 1.1\text{ms to perform all those operations on the v6e.}$$

### Communication Within and Between Chips

We measure this with:

$$\text{Communication Time} = \frac{\text{Communication Bytes}}{\text{Network or Bandwidth Bytes}}$$

The bandwidth within and between chips is the bottleneck on how quickly bytes can be transferred. We can find the lower bound of training by using the higher number between compute and comm time, and the upper bound by adding them together.

There are situations where we need to figure out if we are:

- **Compute bound:** Where our hardware is FULLY utilized.
    
- **Communication bound:** Where some parts of the chip may be idle and waiting for bytes to transfer.
    

We can use **operational intensity**, where we divide $\text{Computation FLOPs} / \text{Communication Bytes}$, and effectively find the FLOPs per byte of a given operation. When the intensity is high, we use most of the available FLOPs, suggesting we’re more compute bound, and vice versa. If we know that a chip is performing at a lower operational intensity than its supposed FLOPs/byte, we’re bound by byte loading and can’t fully utilize the hardware, and should probably spend more time working on networking and bandwidth.

- **Example (Dot Product $x \cdot y$):** We need to take $x$ and $y$ from memory, each of which are $2 \cdot N$ ($2N$ bytes), perform $N$ multiplications and $N - 1$ additions, and write 2 bytes back into HBM:
    
    $$\frac{N + N - 1}{2N + 2N + 2} \approx \frac{1}{2}$$
    
    This means 0.5 floating point operations per byte loaded; we’re communication bound.
    

Think of it as a chef and a food prepper. The prepper needs to constantly prepare ingredients for the chef to make; if he can’t prep fast enough, the chef (our hardware) sits idle. If he preps too quickly (bandwidth), our chef can’t cook fast enough.

The reason operational intensity is important: when we diagnose how to optimize our models further, we absolutely need to figure out if we’re hardware-bound, so we can just slap on more compute (more complicated than that but its fine for now), or communication-bound so we can focus on improving bandwidth, maybe invest in some memory hardware. Hence why GPUs use things like NV Link, or TPUs use ICI (inter-core connects).

## Expanding on Matmuls

We can think of matrix multiplications as the core of ML, and I think it’s good to know these problems by heart. Where $X \times Y \to Z$, $X$ has the shape bf16 $[B,D]$, $Y$ has shape bf16 $[D,F]$, and $Z$ has shape bf16 $[B,F]$, the intensity can be found with:

$$\text{Intensity} = \frac{2 \cdot B \cdot D \cdot F}{2 \cdot B \cdot D + 2 \cdot D \cdot F + 2 \cdot B \cdot F}$$

The numerator is roughly our work done, where we multiply the matrices, and the denominator signifies the total size of the matrices we need to move from memory to processor and then back. Again:

$$\text{Intensity} = \frac{\text{Work}}{\text{Weight}}$$

There’s a cool trick: in larger transformer models, the hidden dims ($D,F$) are huge, but batch size $B$ is relatively small. The math can simplify down to:

$$\text{Intensity} \approx B$$

We can assume that our Intensity is just our batch size.

Most of the constraint work we’ve done so far has been within a chip; internal bandwidth and compute time. However, in production, we’ll find that rarely is a single chip used. Instead, most of the important rooflines to think about are communications between chips; thus, we need to think of matrix operations being split, or **sharded**, across multiple TPUs.

We can start with $X$ and $Y$ matrices, both of which are too big to compute efficiently on 1 single TPU. So we have TPU 0 and 1. We must split the inner dimensions ($D$) in half, so TPU 0 performs matmul on the left side of $X$ with the upper half of $Y$, and TPU 1 performs matmul on the right side of $X$ with the lower half of $Y$.

This step gives us 2 partial sums of what we want; how do we combine them? To get final matrix $Z$, we need to send sums to each other over the network cable and combine them. This is where we figure out Network Traffic, which is strictly determined by the Batch Size $B$ and Output Features $F$, resulting in the shape $BF$. Assuming bfloat16 (2 bytes per number), the total data being sent is $2BF$. So we end up with this equation:

$$\text{Network Intensity} = \frac{B \cdot D \cdot F}{2 \cdot B \cdot F}$$

Or, Math work (only $B \cdot D \cdot F$ because each chip does half the work) divided by network traffic (bytes sent, $2BF$). This cancels out to:

$$\text{Network Intensity} = \frac{D}{2}$$

This should be interesting because the bottleneck depends on $D$ and not $B$. Increasing batch size does cause more math to be done, but also proportionally increases how much data is sent over the network; larger inner dimensions $D$ causes the amount of math being done to increase significantly, but the resulting size of $BF$ does not change.

To summarize, network cables between chips are much, much slower than compute time and chip memory itself.


## Question 1 [int8 matmul]
Say we want to do the matmul $X[B,D] \cdot Y[D,F] \to Z[B,F]$ in int8 precision (1 byte per parameter) instead of bfloat16 (2 bytes per parameter) since TPUs/GPUs can do matmuls faster in lower precision.

* **How many bytes need to be loaded from memory? How many need to be written back to memory?**
    This question is about the memory traffic part of the arithmetic intensity equation. We need to take the numbers from the HBM, do the operation (not relevant here) and load the result back into HBM. Meaning, because its int8 (1 byte per element) we load in $[B,D]$ and $[D,F]$ and load back $[B,F]$ bytes without multiplying by 2 (bf16).
    $$\text{Total Bytes} = BD + DF + BF$$
* **How many total OPs are performed?**
    We can just remember this as $2BDF$. As a note, $BF$ comes from how many dot products we have to do, and we get $2D-1$ from how much work is inside a single dot product, $D$ multiplications and $D-1$ for additions of inner elements. Combined into $2BDF - BF$, but we usually just ignore $BF$ as a rounding error.
* **What is the arithmetic intensity?**
    Arithmetic intensity is just Operational Intensity over Network Traffic. Here, it would be:
    $$\text{Intensity} = \frac{2BDF}{BD + DF + BF}$$
    The solution says we can assume batch size $B$ to be much smaller than the hidden dims ($B \ll D$ and $B \ll F$), it simplifies into $\frac{2BDF}{DF}$, resulting in intensity of $2B$.
* **What is a roofline estimate for $T_{\text{math}}$ and $T_{\text{comms}}$? What are reasonable upper and lower bounds for the runtime of the whole operation?**
    Reasonable lower bound is $\max(\text{Math time}, \text{Communication time})$. Upper bound would be $T_{\text{math}}$ + $T_{\text{comms}}$. Upper bound can never be more than twice the lower bound, so optimizing the Arithmetic Intensity is critical. 
    * Math time is OPs/compute speed (given as $3.94 \times 10^{14}$), so $\frac{2BDF}{3.94 \times 10^{14}}$.
    * Communication time is total bytes to be transferred / network comm speed (given). So this is $\frac{BD + DF + BF}{8.2 \times 10^{11}}$.

Assume our HBM bandwidth is $8.2 \times 10^{11}\text{ bytes/s}$ and our int8 peak OPs/s is $3.94 \times 10^{14}$ (about $2\times$ bf16).

---

## Question 1a [fp4 weight-only quantization]
To fit a massive model onto fewer chips, you compress the weights down to FP4 (0.5 bytes per element). However, to keep accuracy high, your input activations, outputs, and the actual compute cores run in bfloat16 (2 bytes per element).

* **How many total FLOPs are performed?** We still perform $2BDF$ OPs/FLOPs. The actual math units are still multiplying the dequantized FP4 numbers (into BF16), quantization only changes the denominator of our intensity equation but not our numerator. 
* **How many bytes need to be loaded from memory? How many need to be written back to memory? What is the formula for total bytes transferred across the memory bus?**
    Question 1 was $BD + DF + BF$ (output). Note that it was a multiple of 1 because of int8 precision. With bf16 and fp4, we change the multiples. This now becomes:
    $$\text{Total Bytes} = 2BD + 0.5DF + 2BF$$
* **What is the arithmetic intensity, and what does it simplify to when applying the assumption $B \ll D, F$?**
    Most often the case, we just take out B from the equation. We know arithmetic intensity is (from part 1 and 2):
    $$\text{Intensity} = \frac{2BDF}{2BD + 0.5DF + 2BF} \approx \frac{2BDF}{0.5DF} = \frac{2B}{0.5} = 4B$$
* **At what exact batch size ($B$) do you cross the hardware's critical intensity and become compute-bound?**
    Hardware critical intensity is $\frac{\text{Peak Compute}}{\text{Peak Memory}}$. Boils down to (how many operations performed / s) / (how many bytes loaded in from HBM / s). Using the provided numbers 1.97 and 8.2, it comes out to $240\text{ FLOPs/byte}$. To become compute bound, arithmetic intensity needs to be greater than hardware critical intensity.
    $$4B > 240 \implies B > 60$$
    This just basically means how many operations we can do per bytes that are loaded in. If we can instantly finish the math and have to wait for HBM to load in bytes, we're underperforming and memory-bound. However, if our batch size or algo density increases enough, and give the hardware so much math to perform that the memory bus can fetch the next chunk of data, our math units are working nonstop. This becomes compute-bound (and is generally the goal).

Assume our HBM bandwidth is $8.2 \times 10^{11}\text{ bytes/s}$ and our peak bfloat16 compute speed is $1.97 \times 10^{14}\text{ FLOPs/s}$. The operation is $X[B, D] \times Y[D, F] \to Z[B, F]$.

---
## Question 1b [B200 int8 matmul]
Let's see what happens when you shift the exact same int8 matmul from Question 1 over to next-generation hardware. You are running a uniform int8 matmul (1 byte per element for activations, weights, and outputs) on an NVIDIA B200 accelerator.

* **What is the hardware's intrinsic critical arithmetic intensity ($\frac{\text{Peak OPs}}{\text{Bandwidth}}$)?**
    $$\text{HCI} = \frac{4.5 \times 10^{15}}{8.0 \times 10^{12}} = 562.5\text{ FLOPs/byte}$$
* **Using the simplified intensity formula for a pure int8 matmul ($I \approx 2B$), what is the new critical batch size threshold ($B_{\text{crit}}$) for this chip?**
    Using the same int8 numbers from Question 1, we have $\frac{2BDF}{BD + DF + BF}$, or $2B$.
    $$2B > 562.5 \implies B > 281.25$$
* **Compare this $B_{\text{crit}}$ to the TPU v5e's threshold ($B > 240$). What does this tell you about how software optimization pressure changes as hardware compute scaling outpaces memory bandwidth scaling?**
    Because compute scaling is faster than memory bandwidth scaling, hardware critical intensity also increases. Algorithmically, we'd be forced to look at increasing batch sizes, heavier quantization or other strategies to make sure we don't become memory-starved. 

Assume our HBM3e memory bandwidth is $8.0 \times 10^{12}\text{ bytes/s}$ ($8.0\text{ TB/s}$) and our peak dense, non-sparse int8 compute speed is $4.5 \times 10^{15}\text{ OPs/s}$. The operation is $X[B, D] \times Y[D, F] \to Z[B, F]$, assuming $B \ll D, F$.

---

## Question 2 [int8 + bf16 matmul]
In practice we often do different weight vs. activation quantization, so we might store our weights in very low precision but keep activations (and compute) in a higher precision. Say we want to quantize our weights in int8 but keep activations (and compute) in bfloat16. At what batch size do we become compute bound? Assume $1.97 \times 10^{14}\text{ bfloat16 FLOPs/s}$.

*Hint: this means specifically `_bf16[B, D] * int8[D, F] -> bf16[B, F]` where B is the “batch size”.*

1. int8 is a 1 byte number, bfloat16 is 2 bytes. We take the same operation as before: $\frac{2BDF}{BD + DF + BF}$. Remember again that the denominator is $X + Y = Z$, or Activations + Weights = Output. Our Activations are 2 bytes, so $2BD$, and weights are 1 byte, so $DF$, so our output is $2BF$. 
2. Using our boundary assumption where B is negligible (compared to DF, loading the weights) we simplify into $\frac{2BDF}{1DF}$, so arithmetic intensity is $2B$.
3. We solve for the hardware intensity, or compute/memory. $\frac{1.97 \times 10^{14}}{8.2 \times 10^{11}} = 240\text{ FLOPs/byte}$.
   $$2B > 240 \implies B > 120$$

---

## Question 2a [int4 weight-only quantization]
In a push to optimize consumer-grade deployment, we decide to store our model weights in an even lower precision: int4 (0.5 bytes per element). However, to prevent catastrophic degradation of the model's performance, the input activations, output tensors, and the internal accumulation math are all maintained in full fp16 precision (2 bytes per element). At what batch size ($B$) do we cross the hardware threshold and become compute-bound?

Assume our HBM memory bandwidth is $8.2 \times 10^{11}\text{ bytes/s}$ and our peak execution speed for fp16 compute is $1.97 \times 10^{14}\text{ FLOPs/s}$. The operation is $X[B, D] \times Y[D, F] \to Z[B, F]$, assuming $B \ll D, F$.

1. Retain $2BDF$ operations but change our denomination to $0.5DF$, intensity is $4B$.
2. Napkin math: FLOPs/HBM is around $240\text{ FLOPs/byte}$.
   $$4B > 240 \implies B > 60$$
   Meaning, we become compute bound at a batch size of 61 tokens. 

---

## Question 2b [fp8 FP-heavy mixed-precision]
Next-generation training and serving architectures often utilize FP8 for both weights and activations to save memory bandwidth, but accumulate the final results in bfloat16. Say we use a mixed setup: the weight matrix Y is stored in FP8 (1 byte per element) and the input activations X are also stored in FP8 (1 byte per element). However, the output matrix Z is written back to HBM in standard bfloat16 (2 bytes per element) to feed safely into the next layer. At what batch size ($B$) do we become compute-bound?

Assume our memory bandwidth is $8.0 \times 10^{12}\text{ bytes/s}$ and our peak FP8 execution speed is $4.5 \times 10^{15}\text{ OPs/s}$. The operation is $X[B, D] \times Y[D, F] \to Z[B, F], assuming $B \ll D, F$.

1. Interesting case where we have 1 byte activation and weights but a 2 byte result. So, $1BD$, $1DF$, and $2BF$. $\frac{2BDF}{1DF}$ still $= 2B$. 
2. Compute/Memory $= \frac{4.5 \times 10^{15}}{8.0 \times 10^{12}} = 562.5\text{ FLOPs/byte}$, so:
   $$2B > 562.5 \implies B > 281.25\text{ tokens per batch}$$

---

## Question 3
Taking the setup from Question 2, make a roofline plot of peak FLOPs/s vs. $B$ for $F=D=4096$ and $F=D=1024$. Use the exact number of bytes loaded, not an approximation.

Walking through this manually instead of dumping a script. Basically want to know the exact numbers, previously we just approximated $B \ll D,F$ and dropped $BD, BF$ from the denominator. Using our Question 2 setup, we ended with ($\text{FLOPs Time}$) / ($\text{Comms Time}$), or $\frac{2BDF}{2BD + DF + 2BF}$. Just plugging in 4096 for roofline 1 and 1024 for line 2. Having the bytes for Comm/FLOPs time, we divide by respective Network Speed/Compute speed (given).

* **For $D=F=4096$:**
    $$T_{\text{comms}} = \frac{2B(4096)+1(4096)(4096)+2(B)(4096)}{8.2 \times 10^{11}} = \frac{16384B + 16777216}{8.2 \times 10^{11}}\text{ seconds}$$
    $$T_{\text{math}} = \frac{2B(4096)(4096)}{1.97 \times 10^{14}}\text{ seconds}$$
    Solving for $B = 241.4$

* **For $D=F=1024$:**
    Repeating for $D=F=1024$ gives us a higher batch size $B = 483$. Meaning, cutting our model dimensions by $4\times$ lets us **double** the batch size needed to hit the compute ceiling on our chip.

---

## Question 3a [Exact Roofline under Heavy Asymmetry]
In many modern transformer architectures (like LLaMA's SwiGLU MLP layers), the projection matrix is highly asymmetric, where the intermediate dimension $F$ is significantly larger than the hidden dimension $D$ (typically $F = \frac{8}{3}D$). Let's evaluate the exact roofline performance for an asymmetric layer where $D = 4096$ and $F = 11008$ using bfloat16 precision (2 bytes per element).

Using the exact, non-approximated byte traffic equation ($\text{Bytes} = 2BD + 2DF + 2BF$), calculate the exact theoretical batch size ($B$) where $\text{flops\_time}$ perfectly equals $\text{comms\_time}$ on a TPU v5e ($1.97 \times 10^{14}\text{ FLOPs/s}$, $8.2 \times 10^{11}\text{ bytes/s}$). Round your final answer to the nearest whole token.

1. $T_{\text{math}} = \frac{2B(4096)(11008)}{1.97 \times 10^{14}} = \frac{90177536 \times B}{1.97 \times 10^{14}}$
2. $T_{\text{comms}} = \frac{2B(4096) + 2(4096)(11008) + 2B(11008)}{8.2 \times 10^{11}} = \frac{30208B + 90177536}{8.2 \times 10^{11}}$
3. Solve for $B = 130.63 = \mathbf{131}$. Because the middle feed forward dimension $F$ was so big, it obviously inflates the total weight matrix size relative to activations, leading to full chip saturation at a lower batch size.

***As a footer***: obviously this is relevant to the quality/cost tradeoffs inference providers need to work with. In a perfect scenario, we could just serve LLMs with full quality, fp32 precision with massive weight matrices ($DF$). In our world, this leads to a hardware limit where our weight matrix can't even fit on a single chip, so we have to split the model leading to new cross-chip network latency. Serving AI to customers starts incurring greater losses.

If we can only serve 1 customer ($B=1$), chip runs thousands of times slower as we're memory-bound waiting for weights to stream out of HBM. Paradox where: increasing batch size increases OPs/byte, but we also can't increase batch size.

This is a regular problem with Anthropic where their models are the most commercially-used and strongest for coding, but due to their lack of infra, they continuously make the decision to quantize their models (lobotomized) or shorten rate limits for users (to help payoff schedules). We've recently seen drops in performance like model intelligence, rate-limiting, TTFT or queueing delays. Interestingly, this causes a cyclical domino effect where oAI/Gemini experience a sudden influx of mad Anthropic customers and reap the benefits, but not for long before their own infra reaches capacity, and the cycle continues. 

---

## Question 4
What if we wanted to perform $\text{int8}[B,D] \cdot_D \text{int8}[B,D,F] \to \text{int8}[B,F]$ where we imagine having a different matrix for each batch element. What is the arithmetic intensity of this operation?

* **What is the arithmetic intensity of this operation?**
    Int8 = 1 byte, so we just alter the denominator of our operation: $\frac{2BDF}{BD + BDF + BF}$. Assuming B is negligible again, we end with $\frac{2BDF}{BDF}$. This results in just a $2$. 
* **What does this 2 mean?**
    Instead of $B=2$, 2 just means we're locked into $2\text{ OPs/byte}$. It doesn't matter if we increase batch size to 64, 512, which would have normally worked, our intensity is permastuck at $2\text{ OPs/byte}$. For reference, a TPU v5e requires $240\text{ OPs/byte}$ to run at max speed/efficiency, so 99% of our chip capacity is stalled while we wait for unique weights to come from memory. 
* **In what scenario would we expand our weight matrix to include the batch dimension?**
    This is a MoE layout that we see with models like Deepseek, Mistral, Llama3 MoE, where the feed forward network is broken up into several independent 'expert' matrices. Every batch of input tokens then goes through a routing algorithm that determines which 'expert' it goes to based on the token content. When a router decides every single token in B goes to a completely new expert, we're forced to load a unique weight matrix for every single token in the batch. Token 1 loads Matrix D,F, Token 2 loads Matrix D,F, etc.
* **But wait, this happens in MoE models, right? And if the model just randomly decides, every token goes to a new expert, this phenomenon would just naturally occur?**
    Yes. So in a nightmare scenario where Tokens 1,2,3,4 go through a router and are assigned Expert 1,2,3,4, we'd need to load that matrix 4 separate times (nonreusable because each expert has unique matrices). We fix this with Top-K routing tricks and Grouped-Expert Attention (Question 4b).

---

## Question 4a [Batch-Dependent Mixed Precision]
To combat the memory bottleneck of the unique batched weights from Question 4, an architecture team proposes keeping the batch-dependent weight tensor in highly compressed FP4 (0.5 bytes per parameter), while keeping the shared token activations and final outputs in standard bfloat16 (2 bytes per parameter) to preserve accuracy.

The operation is: $\text{bf16}[B,D] \cdot_D \text{FP4}[B,D,F] \to \text{bf16}[B,F]$.

* **The Task:** Write out the exact non-approximated intensity equation, simplify it assuming hidden dimensions are massive ($D, F \gg 1$), and find the flat, constant Arithmetic Intensity ($I$). Does this quantization successfully raise the intensity above the TPU v5e's 240 OPs/byte requirement?
* What I'm getting from this is essentially the same approach that doesn't really fix the problem. We have bf16, so (2)BD, (0.5) BDF, and (2)BF, and this just simplifies to 2BDF/0.5BDF, which still results in 4 OPs/byte, so we're still perma-bottlenecked. Clearly, to solve this issue, simple quantization approaches don't work: $$I = \frac{2BDF}{2BD + 0.5BDF + 2BF} \approx \frac{2BDF}{0.5BDF} = \frac{2}{0.5} = 4 \text{ OPs/byte}$$
---

## Question 4b [Grouped-Expert Attention / Matmuls]
Instead of giving every single token its own unique weight matrix, a team designs a "Grouped" setup. The batch $B$ is divided into a fixed number of distinct parallel groups $G$. All tokens assigned to the same group share the exact same weight matrix.

The operation is: $\text{int8}[B,D] \cdot_D \text{int8}[G,D,F] \to \text{int8}[B,F]$, where $G$ is a fixed constant, and $B$ can scale up arbitrarily ($B \gg G$).

* **The Task:** Write out the total memory traffic formula (all operands are 1-byte int8). Simplify the arithmetic intensity under the assumption that the batch size can grow much larger than the fixed group count ($B \gg G$). How does this change the relationship between the batch size $B$ and the overall performance efficiency?
* Equation now becomes:  $$\text{Arithmetic Intensity} = \frac{2BDF}{BD + GDF + BF}$$
* $G$ is essentially a constant number of **active experts**. We no longer scale the expert pool with batch size $B$, and instead becomes a fixed constant where $G$ is just groups of experts that batches can share. Our next step is dividing by B:$$I = \frac{2DF}{D + \frac{GDF}{B} + F}$$
* If we assume that $D = F$, we end with $2D$, but we still have to remove $GDF$. If we unpack a MASSIVE number of input tokens into the system, $B \to \infty$, we end up with:$$\lim_{B \to \infty} \left(\frac{GDF}{B}\right) = 0$$
* Where the infinitely growing batch size drives the entire term down to 0. We end up with:$$I \approx \frac{2DF}{D + F}$$
* And furthermore: $$I \approx \frac{2D^2}{2D} = D$$
* If our arithmetic intensity is dictated by our model's hidden dimensions, and $D=4096$, our operational intensity approaches 4096OPs/byte, which is way higher than our hardware limit of 240OPs/byte. Our MoE models can now run at/beyond the peak compute ceiling of the chip.

## Question 5 [Memory Rooflines for GPUs] 
Using the spec sheet provided by NVIDIA for the H100 SXM, calculate the batch size at which a bfloat16 matrix multiplication will become compute-bound. Note that the Tensor Core FLOPs numbers are twice the true value since they’re only achievable with structured sparsity.

Assume our HBM3 memory bandwidth is $3.35 \times 10^{12}$ bytes/s and our peak dense, non-sparse bfloat16 compute speed is $9.89 \times 10^{14}$ FLOPs/s. The operation is a uniform bfloat16 matmul ($2$ bytes per element) $X[B, D] \times Y[D, F] \to Z[B, F]$, assuming $B \ll D, F$.

The exact performance ratio for uniform 2-byte precision is configured as:

$$I = \frac{2BDF}{2BD + 2DF + 2BF}$$

Applying the small local token assumption ($B \ll D,F$) drops the activation and output data traffic blocks, simplifying the algorithmic density directly to the batch scale:

$$I \approx \frac{2BDF}{2DF} = B \text{ OPs/byte}$$

Evaluating our Hardware Intensity by dividing the peak compute by the peak memory bandwidth yields:

$$\text{Hardware Intensity} = \frac{9.89 \times 10^{14} \text{ FLOPs/s}}{3.35 \times 10^{12} \text{ bytes/s}} \approx \mathbf{295.2 \text{ FLOPs/byte}}$$

To run at maximum efficiency, the NVIDIA H100 requires our software layer to execute at least 295.2 operations for every single byte streamed across the bus. Setting our algorithmic intensity ($B$) higher than this hardware limit shows we cross the inflection point and become compute-bound when the batch size satisfies **$B \geq 296$ tokens**.