---
title: Conclusions and Further Reading
published: true
---
## 11. Conclusions

This last section is just thanks and a reading list, so here's a short recap instead. Basically every chapter is the same move: count the FLOPs, count the bytes, divide each by the right hardware number, and take the max.

- **Rooflines:** A TPU v5e does 1.97e14 FLOPs/s against 8.2e11 bytes/s of HBM, roughly 240 FLOPs/byte, and a bf16 matmul's intensity is about $B$. So we need more than 240 tokens per batch to be compute-bound.
- **The speed hierarchy:** VMEM, HBM, ICI, PCIe, DCN, fastest to slowest. The traffic on each link has to scale with its speed, so DCN only gets the infrequent transfers.
- **Sharding:** A matmul only needs communication when a contracting dimension is sharded: an AllGather first if one input is, an AllReduce or ReduceScatter after if both are. Throughput-bound, an AllGather takes $V / W_{\text{ici}}$ whatever the ring size.
- **Transformer math:** About 2 FLOPs per parameter per token for a forward pass, 6 for training, and the MLP holds most of the parameters and FLOPs.
- **Training:** Data parallelism and FSDP go comms-bound below about 2,550 tokens per chip on a TPU v5p, tensor parallelism past about $F / 2550$ shards, and mixing the two gets us down to roughly 100 tokens.
- **Inference:** Prefill looks like training and is compute-bound. Generation needs 240 requests in flight to get there, so at small batch the step time is just bytes over bandwidth, and the KV cache decides how big a batch we can afford.

The full series is at [jax-ml.github.io/scaling-book](https://jax-ml.github.io/scaling-book/).

## Further Reading

The reading list from the end of How to Scale Your Model:

- [**TPU Deep Dive**](https://henryhmko.github.io/posts/tpu/tpu.html): the TPU architecture in depth.
- [**Domain specific architectures for AI inference**](https://fleetwood.dev/posts/domain-specific-architectures): AI inference from both the hardware and the model side.
- [**A Domain-Specific Supercomputer for Training Deep Neural Networks**](https://dl.acm.org/doi/pdf/10.1145/3360307): one of the original TPU papers.
- [**Making Deep Learning Go Brrrr From First Principles**](https://horace.io/brrr_intro.html): rooflines and performance engineering from the GPU and PyTorch side.
- [**Writing TPU Kernels with Pallas**](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html): custom TPU kernels in Pallas, plus lower-level TPU details.
- [**How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog**](https://siboehm.com/articles/22/CUDA-MMM): optimizing a matmul kernel in CUDA.
- [**Distributed arrays and automatic parallelization**](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html): JAX's parallelism APIs, for actually implementing some of these ideas.
- [**Rafi Witten's High Performance LLMs 2024 Class**](https://github.com/rwitten/HighPerfLLMs2024): a course on TPU performance engineering, with slides on GitHub.
- [**Efficiently Scaling Transformer Inference**](https://arxiv.org/abs/2211.05102): the math of Transformer inference, credited as the inspiration for a lot of the series.
- [**Huggingface Ultra-Scale Playbook**](https://huggingface.co/spaces/nanotron/ultrascale-playbook): roughly the GPU counterpart to the series, on PyTorch's parallelism and memory-saving tricks for training.
- [**Transformer Inference Arithmetic**](https://kipp.ly/transformer-inference-arithmetic/): an illustrated blog on inference arithmetic.
- [**Stanford CS336 Slides and Videos**](https://stanford-cs336.github.io/spring2025/index.html#coursework): a course on LLM training and serving. The authors point to Assignments 1 and 2.
- [**Stas Bekman's ML Engineering Handbook**](https://github.com/stas00/ml-engineering): practical ML infrastructure, like negotiating with cloud providers, cluster management and measured GPU throughput.
- [**ezyang's blog**](https://blog.ezyang.com/2026/01/computing-sharding-with-einsum/): sharding and PyTorch, from a PyTorch lead. The two posts linked are the [PyTorch internals guide](https://blog.ezyang.com/2019/05/pytorch-internals/) and the [post on sharded matmuls](https://blog.ezyang.com/2026/01/computing-sharding-with-einsum/).
- [**The Anatomy of Collective Communication**](https://www.aleksagordic.com/blog/collective-operations): GPU and TPU collectives. The authors rate its N-D and GPU coverage above their own.
