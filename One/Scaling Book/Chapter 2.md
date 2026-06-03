**Question 1 [bounding LLM latency]:** Say you want to sample from a 200B parameter model in bf16 that’s split across 32 TPU v4p. How long would it take to load all the parameters from HBM into the systolic array? _Hint: use the numbers above._
1. With a 200B parameter model, and each parameter/element is 2 bytes, then our matrix weights are 400 billion bytes total. 
2. Splitting across 32 TPUs = performing 32 loading/math operations in parallel. Using the v4p HBM Bandwidth per chip, which is 1.2e12, we can simply take 400gb/32TPUs to get the GB processed per TPU, and then simply divide those GB by the bandwidth of the chip to find how long it would take for each TPU (or all 32 TPUs) to process the entire weight matrix.
3. Workload per chip: $$\frac{400 \times 10^9 \text{ bytes}}{32 \text{ chips}} = \mathbf{12.5 \times 10^9 \text{ bytes per chip}}$$
4. Time to load: $$\frac{12.5 \times 10^9 \text{ bytes}}{1.2 \times 10^{12} \text{ bytes/s}} = \mathbf{0.01042 \text{ seconds}}$$
5. It takes us 10.42 milliseconds to load all parameters from HBM into our systolic array.

**Question 1a [Next-Gen Cluster Bounds]:** A 1.2 Trillion ($1.2 \times 10^{12}$) parameter model is deployed in full `fp32` precision (4 bytes per parameter) across a cluster of 256 specialized hardware chips. Each individual chip features 192 GB of local HBM storage capacity and an HBM memory bandwidth of $4.8 \times 10^{12} \text{ bytes/s}$. Determine if the parameter array fits within the pooled memory capacity limits. Calculate the exact parallel duration (in milliseconds) required to stream the parameters from HBM into the execution pipelines.

1. With a 1.2T parameter model, and each parameter/element is 4 bytes, then our matrix weights are 4.8 trillion bytes total. 

2. Splitting across 256 chips = performing 256 loading/math operations in parallel. Using the next-gen chip HBM Bandwidth per chip, which is 4.8e12, we can simply take 4.8T bytes/256 chips to get the GB processed per chip, and then simply divide those GB by the bandwidth of the chip to find how long it would take for each chip (or all 256 chips) to process the entire weight matrix.

* **Workload per chip:** $\frac{4.8 \times 10^{12} \text{ bytes}}{256 \text{ chips}} = \mathbf{1.875 \times 10^{10} \text{ bytes per chip}}$
* **Time to load:** $\frac{1.875 \times 10^{10} \text{ bytes}}{4.8 \times 10^{12} \text{ bytes/s}} = \mathbf{0.00390625 \text{ seconds}}$

It takes us 3.91 milliseconds to load all parameters from HBM into our pipelines.

---

**Question 1b [Ultra-Low-Bit Edge Quantization]:** A 70 Billion ($70 \times 10^9$) parameter model is quantized down to `INT4` precision (0.5 bytes per parameter) and sharded across an 8-core edge accelerator system. Each core features 16 GB of local memory capacity and an HBM memory bandwidth of $4.5 \times 10^{11} \text{ bytes/s}$. 

Determine if the quantized parameter array fits within the edge memory capacity limits. 

Calculate the exact parallel duration (in milliseconds) required to stream the parameters from HBM into the execution pipelines.

1. With a 70B parameter model, and each parameter/element is 0.5 bytes, then our matrix weights are 35 billion bytes total. 

2. Splitting across 8 cores = performing 8 loading/math operations in parallel. Using the edge HBM Bandwidth per core, which is 4.5e11, we can simply take 35B bytes/8 cores to get the GB processed per core, and then simply divide those GB by the bandwidth of the core to find how long it would take for each core (or all 8 cores) to process the entire weight matrix.

* **Workload per chip:** $\frac{35 \times 10^9 \text{ bytes}}{8 \text{ cores}} = \mathbf{4.375 \times 10^9 \text{ bytes per core}}$
* **Time to load:** $\frac{4.375 \times 10^9 \text{ bytes}}{4.5 \times 10^{11} \text{ bytes/s}} = \mathbf{0.009722 \text{ seconds}}$

It takes us 9.72 milliseconds to load all parameters from HBM into our pipelines.

**Question 2 [TPU details]:** Consider a full TPU v5e pod. How many total CPU hosts are there? How many TPU TensorCores? What is the total FLOPs/s for the whole pod? What is the total HBM? Do the same exercise for TPU v5p pod.

- **v5e:** Our references say a full v5e pod is a 16x16 shape (256 chips) with a host size of 4x2 (8 chips). Having 1 CPU host per 8 chips, we have 32 CPUs in a full 256 pod. Each TPU has 1 TensorCore, so 256 Cores total. Assuming bf16 parameters, each chip outputs 1.97e14 FLOPs, x 256 is 50.4 PFLOPs. Each chip has 16GB HBM, so the pod total as 16x256, 4096 GB HBM.
    
    - **Total CPU Hosts:** $\frac{256 \text{ chips}}{8 \text{ chips/host}} = \mathbf{32 \text{ hosts}}$
        
    - **Total TPU TensorCores:** $256 \text{ chips} \times 1 \text{ core/chip} = \mathbf{256 \text{ TensorCores}}$
        
    - **Total FLOPs/s (BF16):** $256 \text{ chips} \times 1.97 \times 10^{14} \text{ FLOPs/s} = \mathbf{5.043 \times 10^{16} \text{ FLOPs/s (50.4 PFLOPs)}}$
        
    - **Total HBM:** $256 \text{ chips} \times 16 \text{ GB/chip} = \mathbf{4,096 \text{ GB}}$
        
- **v5p:** A full v5p pod is 16x20x21, or 8960 chips. Each CPU host contains 4 chips, so our full pod contains 2240 CPUs. Because each v5p chip contains 2 TensorCores, we have 17920 Cores. Assuming bf16 parameters, each chip outputs 4.59e14 FLOPs/s, x8960 comes out to 4.11e18, or 4.11 ExaFLOPs. With 96GB HBM/chip, our full pod has 840TB of HBM.
    
    - **Total CPU Hosts:** $\frac{8,960 \text{ chips}}{4 \text{ chips/host}} = \mathbf{2,240 \text{ hosts}}$
        
    - **Total TPU TensorCores:** $8,960 \text{ chips} \times 2 \text{ cores/chip} = \mathbf{17,920 \text{ TensorCores}}$
        
    - **Total FLOPs/s (BF16):** $8,960 \text{ chips} \times 4.59 \times 10^{14} \text{ FLOPs/s} = \mathbf{4.1126 \times 10^{18} \text{ FLOPs/s (4.11 ExaFLOPs)}}$
        
    - **Total HBM:** $8,960 \text{ chips} \times 96 \text{ GB/chip} = \mathbf{860,160 \text{ GB (840 TB)}}$

**Question 2a [Partial Slice Provisioning]**
An engineering team doesn't buy a full TPU v5p pod; instead, they provision a smaller custom cluster slice with an ICI network shape of **$8 \times 8 \times 8$ chips**. The host configuration remains standard at 4 chips per CPU host.

Using the same TPU v5p hardware metrics from your notes ($2 \text{ TensorCores/chip}$, $4.59 \times 10^{14} \text{ BF16 FLOPs/s/chip}$, and $96 \text{ GB HBM/chip}$):

- **The Task:** Calculate the total number of CPU hosts, total TPU TensorCores, total BF16 FLOPs/s, and total HBM available in this specific $8 \times 8 \times 8$ slice configuration.
- - **Total CPU Hosts:** $\frac{512 \text{ chips}}{4 \text{ chips/host}} = \mathbf{128 \text{ hosts}}$
    
- **Total TPU TensorCores:** $512 \text{ chips} \times 2 \text{ cores/chip} = \mathbf{1,024 \text{ TensorCores}}$
    
- **Total FLOPs/s (BF16):** $512 \text{ chips} \times 4.59 \times 10^{14} \text{ FLOPs/s} = \mathbf{2.35 \times 10^{17} \text{ FLOPs/s (235 PFLOPs)}}$
    
- **Total HBM:** $512 \text{ chips} \times 96 \text{ GB/chip} = \mathbf{49,152 \text{ GB (49.15 TB)}}$
    
**Question 2b [Int8 Quantized Inference Pod]**
A team wants to run a massive quantized inference workload across a full TPU v5e pod ($16 \times 16$ shape, 256 chips, 8 chips per CPU host). Because they are serving the model in **`int8` precision**, the compute performance per chip scales up to **$3.94 \times 10^{14} \text{ OPs/s}$**.

- **The Task:** Calculate the total CPU hosts, total TPU TensorCores, total `int8` performance (in PFLOPs or POPs/s), and total HBM for this inference setup. How does switching to `int8` alter the compute capacity versus your original `bf16` v5e pod notes?
- **Total CPU Hosts:** $\frac{256 \text{ chips}}{8 \text{ chips/host}} = \mathbf{32 \text{ hosts}}$
    
- **Total TPU TensorCores:** $256 \text{ chips} \times 1 \text{ core/chip} = \mathbf{256 \text{ TensorCores}}$
    
- **Total Peak Compute (INT8):** $256 \text{ chips} \times 3.94 \times 10^{14} \text{ OPs/s} = \mathbf{1.0086 \times 10^{17} \text{ OPs/s (100.9 POPs/s)}}$
    
- **Total HBM:** $256 \text{ chips} \times 16 \text{ GB/chip} = \mathbf{4,096 \text{ GB (4.096 TB)}}$
- Quantizing from bf16 to int8 doubles our compute throughput since we require less hardware area per lane, so our systolic array can pack and process twice as many numbers per cycle without changing any physical footprint.

**Question 3 [PCIe operational intensity]:** Imagine we’re forced to store a big weight matrix A of type bf16[D,F]bf16[D,F], and a batch of activations x of type bf16[B,D]bf16[B,D] in host DRAM and want to do a matrix multiplication on them. This is running on a single host, and we’re using a single TPU v6e chip attached to it. You can assume B≪D, and F=4D (we’ll see in future chapters why these are reasonable assumptions). What is the smallest batch size B we need to remain FLOPs bound over PCIe? Assume PCIe bandwidth of 1.6e10 bytes / second.
- We take our Time to compute and Time spent transferring over PCIe:$$T_{\text{PCIe}} = \frac{2BD + 2DF + 2BF}{1.6 \times 10^{10}} = \frac{2(BD + DF + BF)}{1.6 \times 10^{10}}$$$$T_{\text{compute}} = \frac{\text{Total FLOPs}}{\text{TPU Compute Speed}} = \frac{2BDF}{9.2 \times 10^{14}}$$
- FYI, because F=4D, the denominators are 8BD^2 and 8D^2 for compute and PCIe respectively, but this doesn't change the final calculation (cancels out) so I kept it simple.
    
- Just isolate for batch size:$$\frac{B}{9.2 \times 10^{14}} > \frac{1}{1.6 \times 10^{10}}$$
-  The simple answer is: $$B>\frac{9.2 \times 10^{14} \text{ FLOPs/s}}{1.6 \times 10^{10} \text{ bytes/s}} \implies B > 57,500$$
- The processing batch needs to be at 57,500 tokens minimum to remain FLOPs bound over PCIe.
- PCIe info dump: in typical use cases like gaming, you honestly won't notice the difference between a PCIe Gen 3 and a Gen 5. However, for AI serving and high throughput inference processes, PCIe transfer speeds directly affect how how quickly you can serve customers, as it impacts the amount of data you can transfer between your singular TPU and CPU host (for multiple TPUs, we bypass the CPU entirely during calculation and use the ICI to for All-Reduce/All-Gather operations. All TPUs in their nodes output their respective finished portion of the matrix directly to their host CPU). PCIe generations and quality differ in Speed per Lane (PCIe runs at 16Ghz FYI) and Lane Width. For AI processes, we would want the highest speeds per lane (+4GB/s per lane) and lane width (x16) for maximum bandwidth. After we hit 16Ghz in PCIe lane speeds, electrical signals travel too fast through the copper and quickly degrades into noise. SOTA PCIe generations use a technique: PAM4(Pulse Amplitude Modulation) which uses 4 distinct voltage levels(0v,1v,2v,3v) that allow us to send 2 bits per cycle (00,01,10,11), doubling the throughput of the same copper lane without increasing frequency. Old PCIe uses NRZ, or 2 voltage levels for 1 bit per cycle (0,1). Still, PCIe bandwidth becomes a looming bottleneck for large scale TPU use cases. Also interestingly, there is an emerging use case for PCIe with photonics, where we use light optics instead of copper to utilize the speed of light.
### Question 3a [Next-Gen PCIe Gen 6 Server Node]

An enterprise team is testing a next-generation AI chip featuring massive matrix processing lanes. They are serving an LLM layer on a single chip, and the weights are being streamed dynamically over a high-end **PCIe Gen 6 x16 motherboard bus link**.

- **Hardware Specs:**
    - Peak Compute Speed: $2.4 \times 10^{15} \text{ FLOPs/s}$ (2.4 PFLOPs)
    - Motherboard PCIe Bandwidth: $1.28 \times 10^{11} \text{ bytes/s}$ (128 GB/s)
- **The Task:** Using the balanced time inequality framework under the assumption that batch size is negligible relative to the internal model dimensions ($B \ll D$), determine the exact minimum batch size ($B$) required to keep this high-speed processor compute-bound.
- Same process where we take the total operations required/peak compute speed > total memory ops/PCIe bandwidth:$$\frac{2BDF}{2.4 \times 10^{15}} > \frac{2(BD + DF + BF)}{1.28 \times 10^{11}}$$$$B > \frac{2.4 \times 10^{15} \text{ FLOPs/s}}{1.28 \times 10^{11} \text{ bytes/s}}$$
- Which comes out to: $$B > 18,750$$ where because of the asymptotic reduction, we can jump straight to the HCI calculation and find we need at least 18,750 tokens within each batch to remain compute-bound.

### Question 3b [Edge Mobile Accelerator via PCIe Gen 4]

A robotics lab is building a vision-language system that streams model layers on-demand across a low-power **PCIe Gen 4 x4 mobile bus interface** to a compact embedded accelerator core.

- **Hardware Specs:**
    - Peak Compute Speed: $8.0 \times 10^{13} \text{ FLOPs/s}$ (80 TFLOPs)
    - Motherboard PCIe Bandwidth: $8.0 \times 10^9 \text{ bytes/s}$ (8 GB/s)
- **The Task:** Using the balanced time inequality framework under the assumption that $B \ll D$, calculate the exact minimum batch size ($B$) needed to ensure the low-power processor pipelines don't stall waiting for the mobile motherboard bus.
- Same asymptotic reduction: $$\frac{2BDF}{8.0 \times 10^{13}} > \frac{2DF}{8.0 \times 10^9}$$$$B > \mathbf{10,000 \text{ tokens}}$$
**Question 4 [general matmul latency]:** Let’s say we want to multiply a weight matrix int8[16384, 4096] by an activation matrix of size int8[B, 4096] where B is some unknown batch size. Let’s say we’re on 1 TPU v5e to start.

1. How long will this multiplication take as a function of B? _Hint: it may help to calculate how long it will take to load the arrays from HBM and how long the multiplication will actually take. Which is bottlenecking you?_
	1. This question is literally just asking us to algebraically isolate the variable B after accounting for all the bytes. We know our weight matrix bytes (16384 x 4096 x 1byte = 67,108,864 bytes), activation matrix bytes
2. What if we wanted to run this operation out of VMEM? How long would it take as a function of B?
	1. 
### Reference Numbers

Here are some specific numbers for our chips:

| Model   | Pod size | Host size | HBM capacity/chip | HBM BW/chip (bytes/s) | FLOPs/s/chip (bf16) | FLOPs/s/chip (int8) |
| ------- | -------- | --------- | ----------------- | --------------------- | ------------------- | ------------------- |
| TPU v3  | 32x32    | 4x2       | 32GB              | 9.0e11                | 1.4e14              | 1.4e14              |
| TPU v4p | 16x16x16 | 2x2x1     | 32GB              | 1.2e12                | 2.75e14             | 2.75e14             |
| TPU v5p | 16x20x28 | 2x2x1     | 96GB              | 2.8e12                | 4.59e14             | 9.18e14             |
| TPU v5e | 16x16    | 4x2       | 16GB              | 8.2e11                | 1.97e14             | 3.94e14             |
| TPU v6e | 16x16    | 4x2       | 32GB              | 1.6e12                | 9.20e14             | 1.84e15             |
| TPU7x   | 4x4x576  | 2x2x1     | 192GB             | 7.4e12                | 2.30e15             | 4.61e15             |

Host size refers to the topology of TPUs connected to a single host (e.g. TPU v5e has a single CPU host connected to 8 TPUs in a 4x2 topology). See the [TPU7x documentation](https://docs.cloud.google.com/tpu/docs/tpu7x) for more details on the latest generation. Here are interconnect figures:

|Model|ICI BW/link (one-way, bytes/s)|ICI BW/link (bidi, bytes/s)|
|---|---|---|
|**TPU v3**|1.0e11|2.0e11|
|**TPU v4p**|4.5e10|9.0e10|
|**TPU v5p**|9.0e10|1.8e11|
|**TPU v5e**|4.5e10|9.0e10|
|**TPU v6e**|9.0e10|1.8e11|
|**TPU7x**|9.0e10|1.8e11|

We include both one-way (unidirectional) bandwidth and bidi (bidirectional) bandwidth since unidirectional bandwidth is more true to the hardware but bidirectional bandwidth occurs more often in equations involving a full ring.9

PCIe bandwidth is typically around `1.6e10` bytes / second per TPU (`3.2e10` for TPU v6e), while DCN bandwidth is typically around `6.25e9` bytes / second per TPU (`12.5e9` for TPU v6e and TPU7x, and `3.125e9` for TPU v5e).