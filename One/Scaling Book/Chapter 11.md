---
title: Conclusions and Further Reading
published: true
---
## 11. Conclusions

The book's last section is thanks and a reading list, so here's the short version of everything before it instead.

### Rooflines

An operation takes at least as long as the slower of its math and its data movement, and at most their sum (so never more than twice the lower bound):

$$
T_{\text{lower}} = \max(T_{\text{math}}, T_{\text{comms}})
$$

Which side wins depends on the algorithm's arithmetic intensity (FLOPs per byte moved) versus the hardware's. A TPU v5e does 1.97e14 FLOPs/s and loads 8.2e11 bytes/s from HBM, so roughly 240 FLOPs/byte. A bf16 matmul with $B \ll D, F$ has an intensity of about $B$. Meaning, we need more than 240 tokens per batch to be compute-bound; below that the MXU just waits on memory.

### The Speed Hierarchy

Fastest to slowest: VMEM, HBM, ICI, PCIe, DCN. VMEM is tiny but much faster than HBM: the v5e problem in Chapter 2 takes it as $22\times$, and the book quotes about $7\times$ for TPU7x. On a TPU v5e, HBM is 8.2e11 bytes/s, an ICI link is 9.0e10 (bidirectional), PCIe is around 1.6e10 and DCN is 3.125e9 per chip. So the traffic on each link has to be proportional to its speed, with DCN kept for infrequent transfers.

### Sharding

Whether a sharded matmul needs communication comes down to four cases:

- **Neither input is sharded on the contracting dimension:** multiply locally, no communication.
- **One input is:** AllGather it first.
- **Both are, on the same axis:** multiply locally, then AllReduce (or ReduceScatter) the partial sums.
- **Both have a non-contracting dimension sharded on the same mesh axis:** invalid, so AllGather one of them first.

When throughput-bound, an AllGather or ReduceScatter takes $V / W_{\text{ici}}$ (bytes over bidirectional link bandwidth), however many chips are in the ring. AllReduce is both back to back, so $2\times$. AllToAll is $\frac{1}{4}$ of an AllGather.

### Transformer Math

Every multiply-accumulate is 2 FLOPs, so a forward pass costs about 2 FLOPs per parameter, per token. Training adds two backward matmuls per forward one, which makes it 6. The MLP holds most of the parameters ($3DF$ per layer) and most of the FLOPs; attention's quadratic term only takes over once $T > 8D$.

### Training

The goal is to split those matmuls across chips without comms outgrowing the math.

- **Data parallelism:** shard the batch, replicate the weights. Memory is the limit: bf16 weights plus fp32 Adam state is 10 bytes per parameter.
- **FSDP:** shard the weights and optimizer too, AllGather the weights right before use. Comms-bound once the per-chip batch drops below $C / W_{\text{ici}}$ (FLOPs/s over ICI bandwidth), about 2,550 tokens on a TPU v5p with one mesh axis (plain data parallelism has the same threshold).
- **Tensor parallelism:** shard the weights along $F$ and move activations instead. Comms-bound past $F / 2550$ shards, around 8 to 16 way for most models, whatever the batch size.
- **Pipeline parallelism:** shard along layers. Very little communication, but chips sit idle in the pipeline bubble.

Mixing FSDP with tensor parallelism gets the per-chip batch down to $2550^2 / 2F$, roughly 100 tokens. Past one pod, it's pure data parallelism across pods over DCN.

### Inference

Prefill pushes the whole prompt through at once, so it looks like training and is basically always compute-bound. Generation makes one token per request per step, so reaching the compute-bound batch of 240 takes 240 requests in flight at once, each with its own KV cache. At small batch we're memory-bandwidth-bound, and the step time is just bytes over bandwidth:

$$
T_{\text{step}} \ge \frac{B \times \text{KV cache bytes} + \text{parameter bytes}}{\text{total memory bandwidth}}
$$

The KV cache is the expensive part: $2 \cdot \text{bytes per float} \cdot H \cdot K \cdot L \cdot T$ per sequence. For LLaMA 2-13B at 8,192 tokens in bf16 that's 6.7GB, and four of them outweigh the 26GB of weights. Attention during generation has an intensity of about 1, so batching doesn't rescue it; a smaller cache does (fewer KV heads, quantization). For sharding, only model parallelism is left: the activations move, and the weights and KV caches stay put.

### Takeaway

Basically every chapter is the same move: count the FLOPs, count the bytes, divide each by the right hardware number, take the max. The full book is at [jax-ml.github.io/scaling-book](https://jax-ml.github.io/scaling-book/).

## Further Reading

The book's reading list, with what each one covers:

- [**TPU Deep Dive**](https://henryhmko.github.io/posts/tpu/tpu.html): covers the TPU architecture in depth.
- [**Domain specific architectures for AI inference**](https://fleetwood.dev/posts/domain-specific-architectures): covers AI inference from both the hardware side and the model side.
- [**A Domain-Specific Supercomputer for Training Deep Neural Networks**](https://dl.acm.org/doi/pdf/10.1145/3360307): one of the original TPU papers; covers details of Google's TPU program the book skips.
- [**Making Deep Learning Go Brrrr From First Principles**](https://horace.io/brrr_intro.html): covers rooflines and performance engineering from the GPU and PyTorch side.
- [**Writing TPU Kernels with Pallas**](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html): covers writing custom TPU kernels in Pallas, plus lower-level TPU details.
- [**How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog**](https://siboehm.com/articles/22/CUDA-MMM): covers optimizing a matmul kernel in CUDA; GPU-specific.
- [**Distributed arrays and automatic parallelization**](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html): covers JAX's parallelism APIs; the book points to it for implementing some of its ideas.
- [**Rafi Witten's High Performance LLMs 2024 Class**](https://github.com/rwitten/HighPerfLLMs2024): a course on TPU performance engineering; slides on GitHub.
- [**Efficiently Scaling Transformer Inference**](https://arxiv.org/abs/2211.05102): a paper on the math of Transformer inference; the book credits it as the inspiration for much of its material.
- [**Huggingface Ultra-Scale Playbook**](https://huggingface.co/spaces/nanotron/ultrascale-playbook): roughly a GPU counterpart to the book; covers PyTorch's parallelism and memory-saving techniques for training.
- [**Transformer Inference Arithmetic**](https://kipp.ly/transformer-inference-arithmetic/): a blog on inference arithmetic that shares many of the book's ideas; illustrated.
- [**Stanford CS336 Slides and Videos**](https://stanford-cs336.github.io/spring2025/index.html#coursework): a course covering LLM training and serving; the book points to Assignments 1 and 2.
- [**Stas Bekman's ML Engineering Handbook**](https://github.com/stas00/ml-engineering): covers practical ML infrastructure: negotiating with cloud providers, cluster management, measured GPU throughput.
- [**ezyang's blog**](https://blog.ezyang.com/2026/01/computing-sharding-with-einsum/): written by a PyTorch lead; covers sharding and PyTorch. The book links its [PyTorch internals guide](https://blog.ezyang.com/2019/05/pytorch-internals/) and its [post on sharded matmuls](https://blog.ezyang.com/2026/01/computing-sharding-with-einsum/).
- [**The Anatomy of Collective Communication**](https://www.aleksagordic.com/blog/collective-operations): covers GPU and TPU collectives; the book rates its N-D and GPU coverage above its own.

**TODO(eddie):** add a closing thought in your own words if you want one
