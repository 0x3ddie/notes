---
title: All about TPUs
description: Overview and Worked Problems
published: true
---
## How TPUs Work

A TPU specializes in matmuls and contains a TensorCore attached to some fast memory (HBM, high bandwidth memory). A TensorCore has several components within that make it so good at math:

- **Matrix Multiply Unit (MXU):** The core of the TensorCore. Performs matmuls using a systolic array.
    
- **Vector Processing Unit (VPU):** Does general math operations like ReLU, pointwise addition or multiplication between vectors, and reductions. Think of it as assisting the MXU.
    
- **Vector Memory (VMEM):** The in-between for the HBM and the MXU. Data loads into the VMEM in order for the MXU to actually do anything; it’s like an L1/L2 cache but larger and programmer-controlled.
    

### Other Components

- **Scalar Unit:** Acts as the CPU by giving instructions to the VPU and MXU.
    
- **HBM (High Bandwidth Memory):** Stores the weights, activations, new batch data, etc. It usually has a capacity in the tens of GB. When operational, tensors come out of HBM into the MXU through the VMEM. MXU results are written back into HBM through VMEM. The speed of this depends on **HBM Bandwidth** (usually 1-2 TB/s), which determines how fast computations can be done.
    

### Pipelining and Overlapping Operations

TPU operations are pipelined and overlapped. When we perform a matmul $X \times Y = Z$, we need to load chunks of $X$ and $Y$ into the MXU from the HBM, going through the VMEM first.

While we’re copying chunks of the matrices from the HBM to VMEM, we’re performing MXU work in parallel and sending results $Z$ from the MXU back to the VMEM, which goes back into HBM. The work, being overlapped, essentially lets us hide the latency.

Our goal is to shoot towards being compute-bound instead of memory-bound; this is fundamentally because we’re loading matrices into a systolic array, specialized for matmuls, and performing around 200 trillion multiply-adds per second. Because the compute is so fast, logically, limits are set on how quickly we can transport data back and forth. Compute-bound, again, just means we need to brute force more chips on-stack.

## On VMEM

VMEM is typically the solution to being memory-bound. It’s lightning fast compared to HBM (around $22\times$ faster, although with a capacity in the MB), while HBM has massive capacity. So we have an imbalance here:

- The MXU basically instantly finishes our math; to keep our MXU running at 100% efficiency, we need a high arithmetic intensity of around **240** when pulling data from HBM.
    
- If we pull data from VMEM, we only need an intensity of **10-20**.
    

TPUs default to being memory bound here. If we’re running small batch operations, and because we can’t fit weights in the tiny VMEM, the MXU has to constantly fetch and wait for data from the HBM while instantly finishing operations. This is severely memory bound.

To summarize the system:

- Reading from the **HBM** starves the MXU, which needs an intensity of 240.
    
- Reading from the **VMEM** provides a perfect, constant feed of data, requiring an intensity of only 10-20, causing our system to be compute-bound instead.
    

If we can engineer our algorithm so that our working data fits perfectly within the VMEM, it is almost a given that we’ll avoid traditional memory-bound issues and default to compute-bound. However, because the cache is so small (few MBs), this is often very challenging.

## On Chip Layouts

Depending on how old the TPU is, we either have separate memory and accelerators (TPU v3 and older), while newer inference chips like v5e only have 1 TPU core per chip. Typically, though, a TPU chip can be arranged as a **‘megacore’** by having 2 TPU cores that share memory and act as 1 large accelerator.

Chips are typically arranged in trays of 4 (so 8 cores, but 4 megacores, meaning 4 chips), connected to a CPU host via a PCIe network. Inference trays with the v5e have 2 trays per host instead of 1, but also 1 core per chip, so 8 chips == 8 cores. The host CPU loads data, executes programs, etc.

As with the $\text{HBM} \leftrightarrow \text{VMEM}$ link, the $\text{CPU} \leftrightarrow \text{HBM}$ PCIe connection also has a specific bandwidth that constrains how quickly we can load from host memory to HBM or vice versa.

## TPU Networks

For GPUs, you might be familiar with GPU networks and Nvidia’s NV Link, which allows GPUs to act as a single compute stack. Google uses the **ICI network**, a direct comparable, to connect TPUs to each other in a Pod.

There are 2 main configurations:

- **2D Torus:** Older gen chips (v2 and v3), inference chips (v5e), and the Trillium generation (TPU v6) connect 4 nearest neighbors with edge links to form a 2D torus.
    
- **3D Torus:** V4 and v5p are connected to the nearest 6 neighbors, making a 3D torus.
    

The toroidal shapes reduce the maximum distance between any 2 nodes from $N$ to $N / 2$, which makes communication much faster.

TPU Pods can get huge with ICI. **Superpods** are maximum pod sizes for specific chips:

- **v4:** $16 \times 16 \times 16$
    
- **v5p:** $16 \times 20 \times 28$
    

These pods are made up of $4 \times 4 \times 4$ cubes (in racks) that are connected to each other via optical wraparound links, from which we can make very large topologies. Smaller topologies like $2 \times 2 \times 1$ or $2 \times 2 \times 2$ can be requested but without wraparounds, which doubles the time of most communications. Any multiple of a full cube $4 \times 4 \times 4$ will have wraparounds.

### Key Difference: TPUs vs. GPUs

- **GPUs** are connected via a hierarchy of switches that allow any GPU to communicate with any other GPU. Nvidia uses dedicated hardware chips called **NVSwitches** for this purpose. For instance, imagine an old-school telephone operator sitting at a switchboard. In an H100 node (8 GPUs) or B200 node (72 GPUs), every single GPU runs an NVLink cable into the NVSwitch. This central connection means that every GPU is only 1 switch/hop away from the other. GPU #1 can talk to GPU #72 with the same speed it can talk to GPU #2. The downside is this is extremely expensive and not proportionally scalable; they also consume power and do no actual math.
    
- **TPUs** are much cheaper since we don’t use switches, and chips connect to each other at the end of their grids. This forms either a 2D or 3D torus shape where each TPU is interconnected with their nearest 4 neighbors. So TPU #1 would plug into the TPU to the East, North, South, etc. This topology means that nodes are dramatically cheaper and simpler to build; to scale, we literally just connect more TPUs and cables at the end of the grid, and bandwidth per chip remains the same throughout. The disadvantage here is that if we want TPU #1 to talk to TPU #72, it has to traverse the physical barrier through all the intermediate TPUs (multi-hop). Thus, we need our software/compiler to be very smart to place all operations next to their immediate neighbor so chips don’t have to multi-hop.
    
### The Speed Hierarchy

$$\begin{array}{rll} \textbf{VMEM:} & \text{On silicon itself, right next to MXU} & (22\times \text{ HBM}) \\ \textbf{HBM:} & \text{High Bandwidth Memory: Same package, different silicon} & (2.8 \text{ TB/s}) \\ \textbf{ICI:} & \text{Inter-Core Connect, cross-chip networking in grid topology} & (90 \text{ GB/s per axis}) \\ \textbf{PCIe:} & \text{Motherboard connecting GPU/TPU nodes to the CPU} & \\ \textbf{DCN:} & \text{Data Center Network, global network} & (6.25 \text{ GB/s - Ultimate bottleneck}) \end{array}$$

For massive scale AI, DCN is a huge bottleneck we face. If our workload is so heavy that we exhaust a single slice (a single continuous ICI grid), we have to connect multiple slices together.

Getting a matrix from a TPU in Slice A to a TPU in Slice B is lengthy:

$$\text{Read from HBM} \to \text{PCIe} \to \text{CPU A} \to \text{Slice A NIC} \to \text{DCN} \to \text{Slice B NIC} \to \text{PCIe} \to \text{TPU B HBM}$$

This throttles throughput from $2.8 \text{ TB/s}$ to $6.25 \text{ GB/s}$, a drop of several orders of magnitude.

### Takeaway

We need to be aware of the advantages/disadvantages of each component and each specific speed. We need to keep our compute cores operating at max efficiency, meaning communication must be proportional to networking speeds.

Ideally, we execute compute locally at MXU/VMEM/HBM, shard model layers locally so we only talk to neighbors via ICI (no hops), and **ONLY** use DCN for infrequent operations like final weight optimizations at the end of training.

> **Note:** If you want to see how systolic arrays work, I made a little interactive tutorial at [systolic.vercel.app](https://systolic.vercel.app/) that you can play around with.

## Worked Problems

### Question 1 [bounding LLM latency]

Say you want to sample from a 200B parameter model in bf16 that’s split across 32 TPU v4p. How long would it take to load all the parameters from HBM into the systolic array? Hint: use the numbers above.

With a 200B parameter model, and each parameter/element is 2 bytes, then our matrix weights are 400 billion bytes total.

Splitting across 32 TPUs = performing 32 loading/math operations in parallel. Using the v4p HBM Bandwidth per chip, which is $1.2 \times 10^{12}$, we can simply take 400gb/32TPUs to get the GB processed per TPU, and then simply divide those GB by the bandwidth of the chip to find how long it would take for each TPU (or all 32 TPUs) to process the entire weight matrix.

$$\text{Workload per chip} = \frac{400 \times 10^9 \text{ bytes}}{32 \text{ chips}} = \mathbf{12.5 \times 10^9 \text{ bytes per chip}}$$

$$\text{Time to load} = \frac{12.5 \times 10^9 \text{ bytes}}{1.2 \times 10^{12} \text{ bytes/s}} = \mathbf{0.01042 \text{ seconds}}$$

It takes us **10.42 milliseconds** to load all parameters from HBM into our systolic array.

### Question 1a [Next-Gen Cluster Bounds]

A 1.2 Trillion ($1.2 \times 10^{12}$) parameter model is deployed in full fp32 precision (4 bytes per parameter) across a cluster of 256 specialized hardware chips. Each individual chip features 192 GB of local HBM storage capacity and an HBM memory bandwidth of $4.8 \times 10^{12} \text{ bytes/s}$. Determine if the parameter array fits within the pooled memory capacity limits. Calculate the exact parallel duration (in milliseconds) required to stream the parameters from HBM into the execution pipelines.

With a 1.2T parameter model, and each parameter/element is 4 bytes, then our matrix weights are 4.8 trillion bytes total.

Splitting across 256 chips = performing 256 loading/math operations in parallel. Using the next-gen chip HBM Bandwidth per chip, which is $4.8 \times 10^{12}$, we can simply take 4.8T bytes/256 chips to get the GB processed per chip, and then simply divide those GB by the bandwidth of the chip to find how long it would take for each chip (or all 256 chips) to process the entire weight matrix.

$$\text{Workload per chip} = \frac{4.8 \times 10^{12} \text{ bytes}}{256 \text{ chips}} = \mathbf{1.875 \times 10^{10} \text{ bytes per chip}}$$

$$\text{Time to load} = \frac{1.875 \times 10^{10} \text{ bytes}}{4.8 \times 10^{12} \text{ bytes/s}} = \mathbf{0.00390625 \text{ seconds}}$$

It takes us **3.91 milliseconds** to load all parameters from HBM into our pipelines.

### Question 1b [Ultra-Low-Bit Edge Quantization]

A 70 Billion ($70 \times 10^9$) parameter model is quantized down to INT4 precision (0.5 bytes per parameter) and sharded across an 8-core edge accelerator system. Each core features 16 GB of local memory capacity and an HBM memory bandwidth of $4.5 \times 10^{11} \text{ bytes/s}$.

1. Determine if the quantized parameter array fits within the edge memory capacity limits.
    
2. Calculate the exact parallel duration (in milliseconds) required to stream the parameters from HBM into the execution pipelines.
    

With a 70B parameter model, and each parameter/element is 0.5 bytes, then our matrix weights are 35 billion bytes total.

Splitting across 8 cores = performing 8 loading/math operations in parallel. Using the edge HBM Bandwidth per core, which is $4.5 \times 10^{11}$, we can simply take 35B bytes/8 cores to get the GB processed per core, and then simply divide those GB by the bandwidth of the core to find how long it would take for each core (or all 8 cores) to process the entire weight matrix.

$$\text{Workload per chip} = \frac{35 \times 10^9 \text{ bytes}}{8 \text{ cores}} = \mathbf{4.375 \times 10^9 \text{ bytes per core}}$$

$$\text{Time to load} = \frac{4.375 \times 10^9 \text{ bytes}}{4.5 \times 10^{11} \text{ bytes/s}} = \mathbf{0.009722 \text{ seconds}}$$

It takes us **9.72 milliseconds** to load all parameters from HBM into our pipelines.

### Question 2 [TPU details]

Consider a full TPU v5e pod. How many total CPU hosts are there? How many TPU TensorCores? What is the total FLOPs/s for the whole pod? What is the total HBM? Do the same exercise for TPU v5p pod.

- **v5e:** Our references say a full v5e pod is a 16x16 shape (256 chips) with a host size of 4x2 (8 chips). Having 1 CPU host per 8 chips, we have 32 CPUs in a full 256 pod. Each TPU has 1 TensorCore, so 256 Cores total. Assuming bf16 parameters, each chip outputs $1.97 \times 10^{14}$ FLOPs, $\times 256$ is 50.4 PFLOPs. Each chip has 16GB HBM, so the pod total as 16x256, 4096 GB HBM.
    
    - $$\text{Total CPU Hosts} = \frac{256 \text{ chips}}{8 \text{ chips/host}} = \mathbf{32 \text{ hosts}}$$
        
    - $$\text{Total TPU TensorCores} = 256 \text{ chips} \times 1 \text{ core/chip} = \mathbf{256 \text{ TensorCores}}$$
        
    - $$\text{Total FLOPs/s (BF16)} = 256 \text{ chips} \times 1.97 \times 10^{14} \text{ FLOPs/s} = \mathbf{5.043 \times 10^{16} \text{ FLOPs/s (50.4 PFLOPs)}}$$
        
    - $$\text{Total HBM} = 256 \text{ chips} \times 16 \text{ GB/chip} = \mathbf{4,096 \text{ GB}}$$
        
- **v5p:** A full v5p pod is 16x20x28, or 8,960 chips. Each CPU host contains 4 chips, so our full pod contains 2,240 CPUs. Because each v5p chip contains 2 TensorCores, we have 17,920 Cores. Assuming bf16 parameters, each chip outputs $4.59 \times 10^{14}$ FLOPs/s, $\times 8,960$ comes out to $4.11 \times 10^{18}$, or 4.11 ExaFLOPs. With 96GB HBM/chip, our full pod has 840TB of HBM.
    
    - $$\text{Total CPU Hosts} = \frac{8,960 \text{ chips}}{4 \text{ chips/host}} = \mathbf{2,240 \text{ hosts}}$$
        
    - $$\text{Total TPU TensorCores} = 8,960 \text{ chips} \times 2 \text{ cores/chip} = \mathbf{17,920 \text{ TensorCores}}$$
        
    - $$\text{Total FLOPs/s (BF16)} = 8,960 \text{ chips} \times 4.59 \times 10^{14} \text{ FLOPs/s} = \mathbf{4.1126 \times 10^{18} \text{ FLOPs/s (4.11 ExaFLOPs)}}$$
        
    - $$\text{Total HBM} = 8,960 \text{ chips} \times 96 \text{ GB/chip} = \mathbf{860,160 \text{ GB (840 TB)}}$$
        

### Question 2a [Partial Slice Provisioning]

An engineering team doesn't buy a full TPU v5p pod; instead, they provision a smaller custom cluster slice with an ICI network shape of $8 \times 8 \times 8$ chips. The host configuration remains standard at 4 chips per CPU host.

Using the same TPU v5p hardware metrics from your notes (2 TensorCores/chip, $4.59 \times 10^{14}$ BF16 FLOPs/s/chip, and 96 GB HBM/chip): Calculate the total number of CPU hosts, total TPU TensorCores, total BF16 FLOPs/s, and total HBM available in this specific $8 \times 8 \times 8$ slice configuration.

- $$\text{Total CPU Hosts} = \frac{512 \text{ chips}}{4 \text{ chips/host}} = \mathbf{128 \text{ hosts}}$$
    
- $$\text{Total TPU TensorCores} = 512 \text{ chips} \times 2 \text{ cores/chip} = \mathbf{1,024 \text{ TensorCores}}$$
    
- $$\text{Total FLOPs/s (BF16)} = 512 \text{ chips} \times 4.59 \times 10^{14} \text{ FLOPs/s} = \mathbf{2.35 \times 10^{17} \text{ FLOPs/s (235 PFLOPs)}}$$
    
- $$\text{Total HBM} = 512 \text{ chips} \times 96 \text{ GB/chip} = \mathbf{49,152 \text{ GB (49.15 TB)}}$$
    

### Question 2b [Int8 Quantized Inference Pod]

A team wants to run a massive quantized inference workload across a full TPU v5e pod ($16 \times 16$ shape, 256 chips, 8 chips per CPU host). Because they are serving the model in int8 precision, the compute performance per chip scales up to $3.94 \times 10^{14}$ OPs/s.

The Task: Calculate the total CPU hosts, total TPU TensorCores, total int8 performance (in PFLOPs or POPs/s), and total HBM for this inference setup. How does switching to int8 alter the compute capacity versus your original bf16 v5e pod notes?

- $$\text{Total CPU Hosts} = \frac{256 \text{ chips}}{8 \text{ chips/host}} = \mathbf{32 \text{ hosts}}$$
    
- $$\text{Total TPU TensorCores} = 256 \text{ chips} \times 1 \text{ core/chip} = \mathbf{256 \text{ TensorCores}}$$
    
- $$\text{Total Peak Compute (INT8)} = 256 \text{ chips} \times 3.94 \times 10^{14} \text{ OPs/s} = \mathbf{1.0086 \times 10^{17} \text{ OPs/s (100.9 POPs/s)}}$$
    
- $$\text{Total HBM} = 256 \text{ chips} \times 16 \text{ GB/chip} = \mathbf{4,096 \text{ GB (4.096 TB)}}$$
    

> Quantizing from bf16 to int8 doubles our compute throughput since we require less hardware area per lane, so our systolic array can pack and process twice as many numbers per cycle without changing any physical footprint.

### Question 3 [PCIe operational intensity]

Imagine we’re forced to store a big weight matrix $A$ of type $\text{bf16}[D,F]$, and a batch of activations $x$ of type $\text{bf16}[B,D]$ in host DRAM and want to do a matrix multiplication on them. This is running on a single host, and we’re using a single TPU v6e chip attached to it. You can assume $B \ll D$, and $F=4D$. What is the smallest batch size $B$ we need to remain FLOPs bound over PCIe? Assume PCIe bandwidth of $1.6 \times 10^{10}$ bytes / second.

We take our Time to compute and Time spent transferring over PCIe:

$$T_{\text{PCIe}} = \frac{2BD + 2DF + 2BF}{1.6 \times 10^{10}} = \frac{2(BD + DF + BF)}{1.6 \times 10^{10}}$$

$$T_{\text{compute}} = \frac{\text{Total FLOPs}}{\text{TPU Compute Speed}} = \frac{2BDF}{9.2 \times 10^{14}}$$

_(Note: Because $F=4D$, the denominators are $8BD^2$ and $8D^2$ for compute and PCIe respectively, but this doesn't change the final calculation as it cancels out)._

Just isolate for batch size:

$$\frac{B}{9.2 \times 10^{14}} > \frac{1}{1.6 \times 10^{10}}$$

$$B > \frac{9.2 \times 10^{14} \text{ FLOPs/s}}{1.6 \times 10^{10} \text{ bytes/s}} \implies B > 57,500$$

The processing batch needs to be at **57,500 tokens minimum** to remain FLOPs bound over PCIe.

> **PCIe info dump:** In typical use cases like gaming, you honestly won't notice the difference between a PCIe Gen 3 and a Gen 5. However, for AI serving and high throughput inference processes, PCIe transfer speeds directly affect how quickly you can serve customers, as it impacts the amount of data you can transfer between your singular TPU and CPU host. For multiple TPUs, we bypass the CPU entirely during calculation and use the ICI to for All-Reduce/All-Gather operations. All TPUs in their nodes output their respective finished portion of the matrix directly to their host CPU.
> 
> PCIe generations and quality differ in Speed per Lane (PCIe runs at 16Ghz FYI) and Lane Width. For AI processes, we would want the highest speeds per lane (+4GB/s per lane) and lane width (x16) for maximum bandwidth. After we hit 16Ghz in PCIe lane speeds, electrical signals travel too fast through the copper and quickly degrade into noise.
> 
> SOTA PCIe generations use a technique called **PAM4 (Pulse Amplitude Modulation)** which uses 4 distinct voltage levels (0v, 1v, 2v, 3v) that allow us to send 2 bits per cycle (00, 01, 10, 11), doubling the throughput of the same copper lane without increasing frequency. Old PCIe uses NRZ, or 2 voltage levels for 1 bit per cycle (0, 1). Still, PCIe bandwidth becomes a looming bottleneck for large scale TPU use cases. Also interestingly, there is an emerging use case for PCIe with photonics, where we use light optics instead of copper to utilize the speed of light.

### Question 3a [Next-Gen PCIe Gen 6 Server Node]

An enterprise team is testing a next-generation AI chip featuring massive matrix processing lanes. They are serving an LLM layer on a single chip, and the weights are being streamed dynamically over a high-end PCIe Gen 6 x16 motherboard bus link.

Hardware Specs:

- **Peak Compute Speed:** $2.4 \times 10^{15} \text{ FLOPs/s}$ (2.4 PFLOPs)
    
- **Motherboard PCIe Bandwidth:** $1.28 \times 10^{11} \text{ bytes/s}$ (128 GB/s)
    

The Task: Using the balanced time inequality framework under the assumption that batch size is negligible relative to the internal model dimensions ($B \ll D$), determine the exact minimum batch size ($B$) required to keep this high-speed processor compute-bound.

Same process where we take the total operations required / peak compute speed $>$ total memory ops / PCIe bandwidth:

$$\frac{2BDF}{2.4 \times 10^{15}} > \frac{2(BD + DF + BF)}{1.28 \times 10^{11}} \implies B > \frac{2.4 \times 10^{15} \text{ FLOPs/s}}{1.28 \times 10^{11} \text{ bytes/s}}$$

Which comes out to: **$B > 18,750$**. Because of the asymptotic reduction, we can jump straight to the HCI calculation and find we need at least 18,750 tokens within each batch to remain compute-bound.

### Question 3b [Edge Mobile Accelerator via PCIe Gen 4]

A robotics lab is building a vision-language system that streams model layers on-demand across a low-power PCIe Gen 4 x4 mobile bus interface to a compact embedded accelerator core.

Hardware Specs:

- **Peak Compute Speed:** $8.0 \times 10^{13} \text{ FLOPs/s}$ (80 TFLOPs)
    
- **Motherboard PCIe Bandwidth:** $8.0 \times 10^9 \text{ bytes/s}$ (8 GB/s)
    

The Task: Using the balanced time inequality framework under the assumption that $B \ll D$, calculate the exact minimum batch size ($B$) needed to ensure the low-power processor pipelines don't stall waiting for the mobile motherboard bus.

Same asymptotic reduction:

$$\frac{2BDF}{8.0 \times 10^{13}} > \frac{2DF}{8.0 \times 10^9} \implies B > \mathbf{10,000 \text{ tokens}}$$

### Question 4 [general matmul latency]

Let’s say we want to multiply a weight matrix $\text{int8}[16384, 4096]$ by an activation matrix of size $\text{int8}[B, 4096]$ where $B$ is some unknown batch size. Let’s say we’re on 1 TPU v5e to start.

How long will this multiplication take as a function of $B$? Hint: it may help to calculate how long it will take to load the arrays from HBM and how long the multiplication will actually take. Which is bottlenecking you?

This question is literally just asking us to algebraically isolate the variable $B$ after accounting for all the bytes. We know our weight matrix bytes ($16384 \times 4096 \times 1\text{ byte} = 67,108,864\text{ bytes}$), activation matrix bytes ($B \times 4096 \times 1\text{ byte} = 4096B$), and output bytes ($B \times 16384 \times 1 = 16384B\text{ bytes}$). Then we also find our OP bytes, which is $2BDF$, so $(2 \times B \times 4096 \times 16384) = 134,217,728B\text{ Operations}$.

$$T_{\text{compute}}(B) = \frac{134,217,728B}{3.94 \times 10^{14}} \approx \mathbf{3.407 \times 10^{-7}B \text{ seconds}}$$

$$T_{\text{HBM}}(B) = \frac{67,108,864 + 20,480B}{8.2 \times 10^{11}} \approx \mathbf{8.184 \times 10^{-5} + 2.498 \times 10^{-8}B \text{ seconds}}$$

We can't execute the math faster than we can load memory, so looking back in Chapter 1, our execution duration is the maximum of these 2 independent times. Using the numbers we got, for small batch sizes like $B = 1$, $T_{\text{HBM}} \approx 81.84 \ \mu\text{s}$ while $T_{\text{compute}} \approx 0.34 \ \mu\text{s}$, meaning a dominant memory bottleneck.

To find the batch size needed to become compute bound, we set the two systems equal to each other:

$$3.407 \times 10^{-7}B = 8.184 \times 10^{-5} + 2.498 \times 10^{-8}B$$

$$3.157 \times 10^{-7}B = 8.184 \times 10^{-5} \implies B \approx \mathbf{259.2}$$

**What if we wanted to run this operation out of VMEM? How long would it take as a function of B?**

This gives us a scenario where we directly load our matrices from VMEM to our MXU. As a refresher, VMEM acts as a high-speed inbetween for the HBM and the MXU. While our MXU is completing math, our VMEM prefetches numbers from the HBM, which allows the MXU to pull in new numbers after it spits the output back into the VMEM. VMEM is extremely tiny, and we'll assume for this question the VMEM to MXU bandwidth is around 20-22x the speed of HBM to VMEM bandwidth.

## Reference Numbers

Here are some specific numbers for our chips:

|**Model**|**Pod size**|**Host size**|**HBM capacity/chip**|**HBM BW/chip (bytes/s)**|**FLOPs/s/chip (bf16)**|**FLOPs/s/chip (int8)**|
|---|---|---|---|---|---|---|
|**TPU v3**|32x32|4x2|32GB|9.0e11|1.4e14|1.4e14|
|**TPU v4p**|16x16x16|2x2x1|32GB|1.2e12|2.75e14|2.75e14|
|**TPU v5p**|16x20x28|2x2x1|96GB|2.8e12|4.59e14|9.18e14|
|**TPU v5e**|16x16|4x2|16GB|8.2e11|1.97e14|3.94e14|
|**TPU v6e**|16x16|4x2|32GB|1.6e12|9.20e14|1.84e15|
|**TPU7x**|7x4x4x576|2x2x1|192GB|7.4e12|2.30e15|4.61e15|

> **Host size** refers to the topology of TPUs connected to a single host (e.g. TPU v5e has a single CPU host connected to 8 TPUs in a 4x2 topology). See the TPU7x documentation for more details on the latest generation.

### Interconnect Figures

|**Model**|**ICI BW/link (one-way, bytes/s)**|**ICI BW/link (bidi, bytes/s)**|
|---|---|---|
|**TPU v3**|1.0e11|2.0e11|
|**TPU v4p**|4.5e10|9.0e10|
|**TPU v5p**|9.0e10|1.8e11|
|**TPU v5e**|4.5e10|9.0e10|
|**TPU v6e**|9.0e10|1.8e11|
|**TPU7x**|9.0e10|1.8e11|

We include both one-way (unidirectional) bandwidth and bidi (bidirectional) bandwidth since unidirectional bandwidth is more true to the hardware but bidirectional bandwidth occurs more often in equations involving a full ring.

PCIe bandwidth is typically around $1.6 \times 10^{10} \text{ bytes/second}$ per TPU ($3.2 \times 10^{10}$ for TPU v6e), while DCN bandwidth is typically around $6.25 \times 10^9 \text{ bytes/second}$ per TPU ($1.25 \times 10^{10}$ for TPU v6e and TPU7x, and $3.125 \times 10^9$ for TPU v5e).