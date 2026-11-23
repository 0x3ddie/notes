---
title: Sharded Matrices and How to Multiply Them
published: true
---
## 3. Sharding Matrices

This is where we can highlight the intensity of AI inference (and pre-training). For a single TPU, we must load the entirety of a model's weights into the massive HBM. When an input vector (representing a token) arrives, the TPU begins a relentless pipeline: streaming weights and the input vector from the HBM, through the lightning-fast VMEM, and into the MXU.

Inside the MXU, systolic arrays operate in a continuous wave, locking weights into place just long enough to multiply them against the flowing input vector, before immediately flushing those weights out to make room for the next layer. This massive data-shuttling operation happens for every single token generated. Sharding is born out of strict physical necessity. Modern model weights simply do not fit in the HBM of a single chip, so we must 'shard' the weight matrices across an entire pod of TPUs.

If Google could magically manufacture a giant chip with 100TB of HBM and a supermassive MXU, it would be significantly faster than 8 separate TPUs working in tandem. This is due entirely to the networking bottleneck. In a sharded setup, the 8 chips must constantly pause to communicate their partial sums across the ICI network cables after intermediate layers. Ultimately, they arrive at a final output vector of ~128k numbers (logits). These logits are pushed through a Softmax function to calculate probabilities, allowing the system to pick the final English word from the tokenizer dictionary.

So now we can talk about HOW to shard so that our compute still stays efficient. There exists a notation for sharding, using a named-axis notation variant to describe how the tensor is sharded in blocks across devices. We can assume a 2D or 3D grid of devices exists, called a device mesh, where each axis is given mesh axis names like `X`, `Y`, `Z`. We then specify how that matrix data is spread across the device by describing how each named dimension of that array is separated across the physical mesh axes.

We define our device mesh like:

$$
\text{Mesh}(\text{devices}=((0, 1), (2, 3)), \text{axis\_names}=(\text{'X'}, \text{'Y'}))
$$

This tells us we have 4 TPUs in a $2\times2$ grid, with axis names `X` and `Y`.

For sharding, if we have $A[I_x,J_y]$, this tells us we shard the first axis $I$, along the mesh axis `X`, and the second axis $J$ along the mesh axis `Y`. So, each shard holds $\frac{1}{|X| \cdot |Y|}$ of the array.

It should be noted that when a mesh dimension (`X`, `Y`) has been used to shard a dimension, it can’t be used again to shard another dimension. It’s ‘spent’. So, something like $I_x,J_x$ is forbidden. But, we can have $I_{xy},J$, or $I_y,J_x$, etc. It should also be noted that the order of multiple-axis subscripts matter, like $I_{xy}$ or $I_{yx}$, because it tells us the traversal order of partitioning.

Something I also noticed: There are tradeoffs to sharding in 1D or 2D. When you shard in 2D, like $I_x,J_y$, we split the matrix 4 times across 4 TPUs, like shard 1, shard 2, etc. This typically has the highest memory efficiency, but also uses the most network bandwidth; Requires 2D Reduce-Scatter followed by an All-Gather across both mesh axes. You can say we use 100% HBM and 100% Network overhead. Slower but more memory efficient, 0 duplication.

A 1D shard, like $I_x,Y$, duplicates data since each device holds full columns. Only requires a 1D all-reduce on the `X` axis. You can say we use 200% HBM and 50% network overhead. Much faster but we use massive amounts of HBM copying data.

If we write a fully-replicated form of the matrix as $A[I,J]$, we don’t assign shards and each device has a full copy of the entire matrix.


This is where using JAX comes into play; the named sharding system is very similar to the abstract syntax we’ve been using above. You can use a Colab notebook to try TPUs for free:

Python

```
import jax
import jax.numpy as jnp

# Create our mesh! We're running on a TPU v2-8 4x2 slice with names 'X' and 'Y'.
assert len(jax.devices()) == 8
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'))

# A little utility function to help define our sharding. A PartitionSpec is our
# sharding (a mapping from axes to names).
def P(*args):
  return jax.NamedSharding(mesh, jax.sharding.PartitionSpec(*args))

# We shard both A and B over the non-contracting dimension and A over the contracting dim.
A = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=P('X', 'Y'))
B = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=P(None, 'Y'))

# We can perform a matmul on these sharded arrays! out_shardings tells us how we want
# the output to be sharded. JAX/XLA handles the rest of the sharding for us.
y = jax.jit(lambda A, B: jnp.einsum('BD,DF->BF', A, B), out_shardings=P('X', 'Y'))(A, B)
```

Now that we’ve set up partitioning, how to compute sharded arrays themselves?

I briefly mentioned this earlier, but our overheads depend on the computations involved. For elementwise ops, there is no overhead for operating on a distributed array. But, when we want to perform ops across elements resident on many devices, it gets more complex. If we wanted to simply shard $A[I_x,J] * B[J,K_y] \to C[I_x,K_y]$, we don't need any communication because $J$ is unsharded. But when we want the output to be unsharded, like $C[I,K]$, we either need to copy $A$ and $B$ or $C$ to every device with an AllGather. Our choices have different tradeoffs. We can boil down all sharding possibilities to 4 cases:

### Neither multiplicand has a sharded contracting dimension

This is when the inner dimensions of both $X$ and $Y$ matrices are intact. For example, $[I_x,J]$ and $[J,K_y]$. If the shared dimension $J$ is intact on each chip, we don’t need to do any communication and do the matmul fully locally. This is high speed math with no network delays.

### One multiplicand has a sharded contracting dimension

$$
A[I,J_x] * B[J,K] \to C[I,K]
$$

We can’t simply multiply $A$ and $B$ locally like before; to multiply a row of $A$ by a column of $B$, we need the sum of the entire row, but that row is chopped up between devices. If we do math with a partial row, we get a partial sum.

AllGather is a primitive network operation that removes sharding. By taking data that’s chopped up across TPUs, we force them to share until every device has 100% of the data.

**How:** AllGather works by sharing data in a ring. Imagine our TPUs arranged in a ring formation; like 4 TPUs each with 1 piece of the row. First, we pass over that piece to the right. Now everyone has 2 pieces. We pass over the new piece to the right again. 3 pieces. Now we do a final pass, and come around in a full circle, where everyone has 4 pieces.

We can AllGather in 1 direction or 2 directions.

**Scaling law:** it takes the same amount of time to AllGather across 100 TPUs and 4 TPUs, ONLY if we’re through-put bound already. Meaning, spreading 100 tiny shards across 50 hops mathematically takes the same time as 4 large shards across 4 TPUs.

$$
T_{\text{total}} = \frac{V}{W_{\text{ici}}}
$$

we’re bottlenecked by the speed of each link.

### Both multiplicands have sharded contracting dims

When the inner dims $J$ of both matrices are sharded.

Previously with 1 and 2, either one or both matrices had their inner dimension untouched, meaning a TPU could calculate a finished number for their assignment.

Now, because we’re multiplying the column vector by the row vector, we no longer get a tiny shard, but a full-sized matrix. But, because we only had partial inputs, its a matrix of partial sums. Now, every TPU in the pod has its own partial matrix; we need to stack all these matrices on top to get the full answer.

$\{U_x\}$ notation stands for ‘Unreduced’. Essentially means, this matrix is locally complete, but we haven’t summed the $x$-axes yet, so its incomplete.

**Primitive: AllReduce:** Every TPU takes its incomplete matrix and sends it to the next TPU, and every TPU adds each matrix together. Insanely expensive, around $2\times$ AllGather, basically brute force method, since we’re passing full-sized matrices around. Also, each TPU at the end holds a complete matrix… wasteful.

**Primitive: ReduceScatter:** In a ring, we pass around matrices, adding them up as we go. As we add, we drop pieces we don’t need, resulting in a complete matrix that’s sharded across the TPUs. This is equally intensive to AllGather.

A cool equation:

$$
\text{AllReduce} = \text{ReduceScatter} + \text{AllGather}
$$

We just went over 3 communication primitives: AllGather, ReduceScatter, All Reduce. There is a 4th: AllToAll.

All to All appears in MoE models, and concerns the sharded transposition, or resharding operations.

AlltoAll is purely a rearrangement primitive, and is $\frac{1}{4}$ the cost of an AllGather. This takes place when our current sharding structure, such as an $A[I_x,J]$ is incompatible with the math we need to do next, which requires an $A[I,J_x]$. Thus, we need to transpose those rows into columns, or vice versa. The way this happens is by every TPU chopping up it’s data into tiny blocks, and shipping them along a ring. Each TPU then has 1 block of each row, ending up in a column.

### Both multiplicands have a non-contracting dimension sharded on the same axis

The 4th case, like $A[I_x,J] * B[J,K_x]$. The output would have to be $C[I_x,K_x]$, but a mesh axis can only shard one dimension of an array, so this isn't allowed. Chip $i$ only holds the $i$th block of rows of $A$ and the $i$th block of columns of $B$, so it could only ever compute the diagonal block of $C$. The fix is to AllGather one of the inputs first, like $A[I_x,J] \to A[I,J]$, which puts us back in case 1 with $C[I,K_x]$.

> **Note:** you can find useful GIFs on [jax-ml.github.io/scaling-book/sharding/](https://jax-ml.github.io/scaling-book/sharding/) for references on each primitive, or ask Gemini to visualize the methods/math (like asking TPUs to explain TPUs fr)

To conclude on sharding: There are 4 main communication primitives that define how shards move in pods. In some cases, we might not need to shard at all if our matrices are complete (Case 1), but most cases, and all cases in production environments, will require sharding due to massive sizes and arrays. Many of the cases we described are very rudimentary use cases involving 4, 8 TPUs and some tiny matrices. What about 4000 TPUs? The scale is honestly staggering, so understanding simple concepts like how shards move along ICI to result in final, complete matrices, and which primitive to use or discard to optimize can drastically reduce network and memory latency.

### Problem Set

**Pop Quiz [2D sharding across 1 axis]:** Consider an array `fp32[1024, 4096]` with sharding A[I XY,J] and mesh `{'X': 8, 'Y': 2}`. How much data is held by each device? How much time would it take to load this array from HBM on H100s (assuming `3.4e12` memory bandwidth per chip)?
- Our Mesh means we have an 8x2 grid setup of H100s, or 16 GPUs total. 
- Our array is fp32, so the total bytesize is 4x1024x4096 = 16,777,216 bytes (16 MiB).
- We're told that our array A [I,J] has [I] sharded across X and Y, and J is untouched. This tells us that each individual GPU gets 100% of the columns J and 1/16 of the rows I. 1024/16 = 64, so each GPU gets [64,4096]. Total elements comes to 262,144, x4 = 1048576 bytes.
- $$
  \text{HBM Load Time} = \frac{1{,}048{,}576 \text{ bytes}}{3.4 \times 10^{12} \text{ bytes/s}}
  $$
- .308 microseconds, or 308 nanoseconds.

**Pop Quiz:** Let **A** be an array with shape `int8[128, 2048]`, sharding A[IXY​,J], and mesh `Mesh({'X': 2, 'Y': 8, 'Z': 2})` (so 32 devices total). How much memory does **A** use per device? How much total memory does **A** use across all devices?
- Notice that we now have a Z axis for our TPU cluster, yet our rows are only sharded across X and Y. Meaning, I is split along the X and Y axis, but replicated along the Z axis, so it's only split 16 ways. 
- 128/16 = 8 rows, 2048 columns for each TPU. Memory used per device will be 1 x 8 x 2048 bytes, 16384 bytes. Total memory across devices is 16384 x 32, or 524288 bytes.

**Pop Quiz 2 [AllGather time]:** Using the numbers from Part 2, how long does it take to perform the $\text{AllGather}_Y([E_Y, F]) \rightarrow [E, F]$ on a TPU v5e with a 2D mesh `{'X': 8, 'Y': 4}`, where $E = 2048, F = 8192$ in bfloat16? What about with $E = 256, F = 256$?
* **Part 1 Matrix Configuration ($E = 2048, F = 8192$):**
	* We have 32 TPUs in a grid that is 8 chips wide and 4 chips tall, so 8 columns. Our AllGatherY operation means that all our chips only pass data to their vertical neighbors. This might be a bit confusing, but we have our normal TPU configurations but the operation itself concerns each chip and their vertical neighbors. So our AllGatherY operation means that our chips just move along the vertical line instead of towards everybody else. And as we gather all our shards 'vertically' it should come together as a complete E.
	* This is where we need to look at the mesh configurations too. A v5e only has wraparound cables on an axis with the full 16 chips, and our Y axis only has 4, so there's no wraparound cable that connects the 'bottom' chip directly to the top chip. Meaning, we're throttled by the bottom chip that needs to travel through the other chips vertically. So we're stuck with unidirectional ICI bandwidth.
	-  $$
	   \text{Local Shard Shape} = \text{bf16}[512, 8192] \rightarrow 512 \times 8192 \times 2 \text{ bytes} \approx 8.4\text{ MB}
	   $$
	- Each local shard is 8.4MB, and since we have 4 chips per Y, we split the E row by 4, so 512 elements. That means that in all 8 of our columns, every chip eventually ends up with a finished A[E,F], so 32 identical copies. We can imagine the shard at the bottom of each column travelling through 3 more chips to get to the top for the AllGather, since we're forced to be unidirectional. Hopping 3 times means:
	- $$
	  \text{T}_{\text{comms}} = \frac{3 \times 8.4 \times 10^6 \text{ bytes}}{4.5 \times 10^{10} \text{ bytes/s}} = \mathbf{560 \ \mu\text{s}}
	  $$
	- Plus 3 microseconds for our per-hop processing latency.
	- $$
	  \text{Unified Array Size} = \text{bf16}[2048, 8192] \rightarrow 2048 \times 8192 \times 2 \text{ bytes} \approx 33.55\text{ MB}
	  $$
	- If we change our E = 256 and F = 256,

	  $$
	  \text{Streaming term} = \frac{32 \times 10^3 \text{ bytes}}{4.5 \times 10^{10} \text{ bytes/s}} = 0.7 \ \mu\text{s} < 1 \ \mu\text{s per hop}
	  $$
	- Meaning we're latency-bound with our 3 microsecond hop latency as a minimum.

**Question 1 [replicated sharding]**: An array is sharded A[IX​,J,K,…] (i.e., only sharded across X), with a mesh `Mesh({'X': 4, 'Y': 8, 'Z': 2})`. What is the ratio of the total number of bytes taken up by A across all chips to the size of one copy of the array?
- We see that our mesh describes a 3D configuration of 64 TPUs.
- Our array's I dimension is sharded across X, while J and K are unsharded. So, each chip gets A[I/4, J, K]. 
- In order to get the full array, we need 4 chips, or 1 horizontal line of chips. 64/4=16, meaning we have 16 copies of our completed A.
- What this means: our total memory consumed across 64 TPUs is 16 times larger than the raw unsharded tensor because we're storing 16 completed replicas.

**1a: 2D Sharding on a 3D Mesh**: Consider an weight matrix $W$ with dimensions `fp32[4096, 4096]`. It is sharded as $W[I_X, J_Z]$ (meaning the rows are sharded across $X$, the columns are sharded across $Z$, and $Y$ is left out).

The hardware configuration is `Mesh({'X': 4, 'Y': 4, 'Z': 4})`.
**How much memory does a single shard take up on an individual device?**
- Total elements is 4096 x 4096, bytes would be x4 so 64MB.
- We shard our rows and columns along the X and Z axis respectively, so each device gets 1024x1024 elements, or 4MB.
**What is the ratio of the total memory used across the whole cluster to the size of one un-sharded copy of the array?**
- We're sharding across X and Z but replicating along Y. So, for every 4 X and Z chips, we get 1 complete shard (but Y is replicated 4 times).
- $$
  \text{Ratio} = \frac{64 \text{ devices} \times 4 \text{ MiB per device}}{64 \text{ MiB for 1 copy}} = \frac{256 \text{ MiB}}{64 \text{ MiB}} = \mathbf{4}
  $$

**1b: The Multi-Axis Shard Trap:** Consider an activation tensor $X$ with dimensions `bf16[8, 2048, 4096]`. It is sharded as $X[I_{XY}, J, K]$ (meaning only the first dimension $I$ is sharded, but it is split across _both_ the $X$ and $Y$ hardware axes simultaneously).

The hardware configuration is `Mesh({'X': 2, 'Y': 8, 'Z': 4})`.
**How much memory does a single shard take up on an individual device?**
- Because we shard I along X and Y, we end up with I/16. Meaning, each shard gets 0.5 of an I row. One row is 2048 x 4096 bf16 values, so half a row is 8,388,608 bytes. 
	- FYI: This goes into fractional sharding. Physically we cannot hold half of a row, so we 'pad' the dimension I out to the nearest multiple of the mesh size. JAX won't do this for us: `jax.device_put` raises an error when 8 rows are sharded 16 ways, so we'd pad I ourselves. So 8 would become 16. 8 of our chips hold 1 real row element, while the other 8 hold a dummy padded 0 row. The effective dimension I per device becomes 1 element.
- $$
  \text{Elements per Device} = 1 \times 2048 \times 4096 = 8{,}388{,}608 \text{ elements}
  $$

- $$
  \text{Memory per Device} = 8{,}388{,}608 \times 2 \text{ bytes} = \mathbf{16{,}777{,}216 \text{ bytes}} \ (16 \text{ MiB})
  $$
**What is the ratio of the total memory used across the whole cluster to the size of one un-sharded copy of the array?**
- $$
  \text{Total Cluster Memory} = 64 \text{ devices} \times 16 \text{ MiB/device} = \mathbf{1{,}024 \text{ MiB}} \ (1 \text{ GiB})
  $$

- $$
  \text{Ratio} = \frac{1024 \text{ MiB}}{128 \text{ MiB}} = \mathbf{8}
  $$
**Question 2 [AllGather / AllReduce latency]:** How long should $\text{AllGather}_X([B_X, D_Y])$ take on a TPU v4p $4 \times 4 \times 4$ slice with mesh `Mesh({'X': 4, 'Y': 4, 'Z': 4})` if $B = 1024$ and $D = 4096$ in bfloat16? How about $\text{AllGather}_{XY}([B_X, D_Y])$? How about $\text{AllReduce}_Z([B_X, D_Y]\{U_Z\})$?
- From now on I think we can speed through some calculations since we have all our bandwidth references included:
- Global matrix size would be around 8,388,608 bytes (bfloat16). Since we shard 4 times across X and Y, local shard shapes would be [256, 1024]. Local shard size would therefore be 512kB.
- Because our AllGather only moves across X, D remains sharded by 4, a single X ring of 4 chips would 2BD/Y, or 2.1MB.
- Meaning, the AllGather over X never touches the $D_Y$ sharding. Each X ring of 4 chips just rebuilds its own Y slice, a `bf16[1024, 1024]` block of $2BD/Y = 2{,}097{,}152\text{ bytes}$, from 4 shards of 524,288 bytes. The 4 Y slices do this in parallel on their own rings.
- A v4p $4 \times 4 \times 4$ is a full cube, so every axis has a wraparound and we get the bidirectional 9.0e10 bytes/s.
- **AllGather over X:**

  $$
  T_{\text{comms}} = \frac{2BD/Y}{W_{\text{ici}}} = \frac{2{,}097{,}152 \text{ bytes}}{9.0 \times 10^{10} \text{ bytes/s}} = \mathbf{23.3 \ \mu\text{s}}
  $$

- Latency is only about 2μs (2 hops each way on a ring of 4), so we're bandwidth bound at around **23 microseconds**.
- **AllGather over XY:** both subscripts come off, so every chip ends up with the full 8,388,608 bytes. 4 times the data, but gathering over two axes at once doubles our bandwidth.

  $$
  T_{\text{comms}} = \frac{2BD}{2 \cdot W_{\text{ici}}} = \frac{8{,}388{,}608 \text{ bytes}}{2 \times 9.0 \times 10^{10} \text{ bytes/s}} = \mathbf{46.6 \ \mu\text{s}}
  $$

- The longest path is $(4 + 4)/2 = 4$ hops, about 4μs, so still bandwidth bound. Exactly double part 1.
- **AllReduce over Z:** the 4 chips along each Z ring hold different partial sums of the same local block. The sharding stays $[B_X, D_Y]$, so only the 524,288 byte `bf16[256, 1024]` shard goes around, and we pay for it twice (ReduceScatter plus AllGather).

  $$
  T_{\text{comms}} = 2 \times \frac{2BD/(X \cdot Y)}{W_{\text{ici}}} = \frac{2 \times 524{,}288 \text{ bytes}}{9.0 \times 10^{10} \text{ bytes/s}} = \mathbf{11.65 \ \mu\text{s}}
  $$

- So roughly 23μs, 47μs and 12μs. The AllReduce is supposed to be the expensive primitive, but it's the cheapest here because it only moves a 512kB shard. We pay for the bytes going around the ring, not the name of the primitive.
- The solution adds a trick: all 4 Z planes hold the same copy, so if each plane just keeps its own quarter of the rows, we're at $[B_{XZ}, D_Y]$ for free and can gather over X and Z together (X, Y and Z for part 2). Part 1 drops to $2{,}097{,}152 / (2 \times 9.0 \times 10^{10}) = 11.65$μs and part 2 to $8{,}388{,}608 / (3 \times 9.0 \times 10^{10}) = 31$μs. The solution prints 11.5μs for part 1, which is just half of a rounded 23.

**Question 3 [latency-bound AllGather]:** Same TPU v4p $4 \times 4 \times 4$ slice and mesh `Mesh({'X': 4, 'Y': 4, 'Z': 4})` as Question 2, but now the array is tiny: $\text{AllGather}_X([B_X])$ with $B = 128$ in bfloat16. How long should it take? Hint: we're probably latency bound.
- The whole array is $128 \times 2 = 256\text{ bytes}$, so each chip along X holds 64 bytes. Pushing that over a link at 4.5e10 bytes/s is about 1.4 nanoseconds, but every hop costs around 1μs no matter how little we send. Anything under $4.5 \times 10^{10} \times 10^{-6} = 45\text{ kB}$ per hop is latency bound, and we're about 700 times below that.
- So we just count hops. X has 4 chips and a wraparound, so the farthest chip is 2 hops away.

  $$
  T_{\text{total}} = \max\left[\frac{T_{\text{min}} \cdot \lvert X \rvert}{2}, \frac{V}{W_{\text{ici}}}\right] = \max\left[\frac{1 \ \mu\text{s} \times 4}{2}, \frac{256 \text{ bytes}}{9.0 \times 10^{10} \text{ bytes/s}}\right] = \mathbf{2 \ \mu\text{s}}
  $$

- Roughly **2 microseconds**, and the bandwidth term (about 3 nanoseconds) doesn't even register. Meaning, here the cost scales with ring size, not bytes: the same 256 bytes on a 16 chip ring would take 8μs.

**Question 4 [matmul strategies]:** For $X[B, D] \cdot_D Y[D_X, F] \to Z[B, F]$, Strategy 1 is the usual rule: $\text{AllGather}_X(Y[D_X, F])$, then one local matmul on fully replicated matrices (Case 2). Strategy 2 multiplies the shards where they sit, $X[B, D_X] \cdot_D Y[D_X, F] \to Z[B, F]\{U_X\}$ (Case 3), then runs $\text{AllReduce}_X(Z[B, F]\{U_X\})$. What does each cost in FLOPs and comms? Which is better and why?
- Basically: send the weights around the ring once, or the output twice? Everything is bf16 (2 bytes per element), $C$ is one chip's FLOPs/s and $W_{\text{ici}}$ is the bidirectional ICI bandwidth.
- **Strategy 1:** the AllGather moves all of $Y$, $2DF$ bytes, and then every chip does the whole $2BDF$ matmul. All the chips along X compute the same answer.

  $$
  T_1 = \max\left(\frac{2BDF}{C}, \frac{2DF}{W_{\text{ici}}}\right)
  $$

- **Strategy 2:** $X$ is replicated, so each chip just keeps the slice of $D$ that matches its shard of $Y$, for free. That's $2BDF/X$ FLOPs per chip, so the compute is actually sharded. But every chip ends up with a full sized $Z[B, F]$ of partial sums ($2BF$ bytes), and an AllReduce is twice an AllGather, so $4BF$ bytes worth.

  $$
  T_2 = \max\left(\frac{2BDF}{X \cdot C}, \frac{4BF}{W_{\text{ici}}}\right)
  $$

- **Which is better?** Strategy 2's math term is always the smaller one, so it wins when its comms term beats whichever term limits Strategy 1:

  $$
  \frac{4BF}{W_{\text{ici}}} < \max\left(\frac{2BDF}{C}, \frac{2DF}{W_{\text{ici}}}\right) \implies D > \frac{2C}{W_{\text{ici}}} \ \text{ or } \ D > 2B
  $$

- For a TPU v5p, $C / W_{\text{ici}} = 4.59 \times 10^{14} / 1.8 \times 10^{11} = 2550$, so Strategy 2 wins when $D > 5100$ or $D > 2B$.
- Napkin math with the Question 7 block ($D = 8192$, $F = 32768$, $B = 128$) on a v5p with $X = 4$: Strategy 1 gathers 537MB of weights, around 2.98 ms, for 150μs of math. Strategy 2 reduces an 8.4MB output (16.8MB worth of traffic), around 93μs, for 37μs of math. 32 times faster, which is just $D / (2B)$.
- So for a big model and a small batch, Strategy 2 is better: the output $[B, F]$ is tiny next to the weights $[D, F]$, and we skip the duplicated FLOPs. We don't always do it because the setup is rare: usually the activations are sharded on the same mesh axis as the weights (more in Chapter 5), so $X$ isn't replicated for us to slice.

**Question 5 [minimum latency]:** We want the fastest possible matmul $A[I, J] \cdot_J B[J, K] \to C[I, K]$ on a TPU v4p $4 \times 4 \times 4$. We can shard the inputs any way we like, but $C$ has to end up fully replicated. Which input sharding gives the lowest latency, and what are the total FLOPs and the comms time?
- We have 64 chips at $2.75 \times 10^{14}$ bf16 FLOPs/s each (call it $C_{\text{chip}}$, since $C$ is the output here), and three ICI axes at 9.0e10 bytes/s each. The output is $2IK$ bytes and every chip needs all of it, so we want to split the FLOPs 64 ways while paying the least to put $C$ back together.
- The solution lists four options and leaves the math to us:
    1. **Shard the rows:** $A[I_{XYZ}, J] \cdot B[J, K] \to C[I_{XYZ}, K]$. Case 1, no comms for the matmul, $2IJK/64$ FLOPs per chip. Then an $\text{AllGather}_{XYZ}$ of $C$: $2IK$ bytes over three axes, so $T_{\text{comms}} = 2IK / (3W_{\text{ici}})$.
    2. **Shard the columns:** $A[I, J] \cdot B[J, K_{XYZ}] \to C[I, K_{XYZ}]$. Mirror image of option 1, same cost.
    3. **Shard the contracting dimension:** $A[I, J_{XYZ}] \cdot B[J_{XYZ}, K] \to C[I, K]\{U_{XYZ}\}$. Same FLOPs, but now every chip holds a full sized partial sum and we need an AllReduce: $T_{\text{comms}} = 4IK / (3W_{\text{ici}})$, twice options 1 and 2.
    4. **Don't shard at all:** zero comms, but each chip does all $2IJK$ FLOPs.
- Option 3 is never better than 1 or 2. Sharding vs not sharding comes down to the AllGather against the full matmul:

  $$
  \frac{2IK}{3 \cdot W_{\text{ici}}} < \frac{2IJK}{C_{\text{chip}}} \implies J > \frac{C_{\text{chip}}}{3 \cdot W_{\text{ici}}} = \frac{2.75 \times 10^{14}}{2.7 \times 10^{11}} \approx \mathbf{1019}
  $$

- That assumes the matmul overlaps with the gather (the lower bound from Chapter 1). Back to back, it's $J > 1035$.
- So for any normal sized matrix: shard one of the non-contracting dimensions over all three axes, $A[I_{XYZ}, J]$ with $B$ replicated or $A$ replicated with $B[J, K_{XYZ}]$, and leave $J$ alone. Both are equally good. Total FLOPs are $2IJK$ ($2IJK/64$ per chip), and the comms time is

  $$
  T_{\text{comms}} = \frac{2IK \text{ bytes}}{3 \times 9.0 \times 10^{10} \text{ bytes/s}} = \frac{2IK}{2.7 \times 10^{11}} \text{ seconds}
  $$

- There's a latency floor of about 6μs (the longest path through the cube is $(4 + 4 + 4)/2 = 6$ hops), which only matters when the output is under about 1.6MB. And below a $J$ of about a thousand, option 4 wins: it's faster to never talk, even though that burns 64 times the FLOPs.

**Question 6 [comms vs compute]:** We run $A[I_X, J_Y] \cdot_J B[J_Y, K] \to C[I_X, K]$ on a TPU v5e $4 \times 4$. What communication does it need, and how does the comms time compare to the compute time? Same two questions for $A[I_X, J] \cdot_J B[J_X, K_Y] \to C[I_X, K_Y]$ (the standard training setting: data, tensor and ZeRO sharding combined) and for $A[I_X, J] \cdot_J B[J, K_Y] \to C[I_X, K_Y]$ (the standard inference setting: pure tensor parallelism plus data).
- There's no published solution or sizes for this one, so we keep $I$, $J$, $K$ as symbols and assume bf16. Basically "which of the 4 cases is it?" three times.
- A v5e chip does $1.97 \times 10^{14}$ FLOPs/s and each ICI link carries 4.5e10 bytes/s one way. Same catch as Pop Quiz 2: a $4 \times 4$ v5e slice has no wraparounds, so data at one edge needs 3 hops to reach the other. Gathering $V$ bytes over an axis of 4 takes $\frac{3}{4} V / (4.5 \times 10^{10})$, same as $V / (6.0 \times 10^{10})$.
- In all three, each chip multiplies 1/16 of the problem, $IJK/8$ FLOPs. So $T_{\text{math}} = IJK / (8 \times 1.97 \times 10^{14})$ every time, and only the comms change.
- **First:** $A[I_X, J_Y] \cdot_J B[J_Y, K] \to C[I_X, K]$
    - $J$ is sharded on Y in both inputs, so this is Case 3: multiply locally to get $C[I_X, K]\{U_Y\}$, then $\text{AllReduce}_Y$. Each line of 4 chips along Y reduces an $[I/4, K]$ block, $IK/2$ bytes, in two passes:

      $$
      T_{\text{comms}} = \frac{2 \times IK/2}{6.0 \times 10^{10}} = \frac{IK}{6.0 \times 10^{10}} \text{ s} \qquad \frac{T_{\text{comms}}}{T_{\text{math}}} = \frac{8 \times 1.97 \times 10^{14}}{6.0 \times 10^{10} \cdot J} \approx \frac{26{,}267}{J}
      $$

    - So we're comms bound unless $J > 26{,}267$. With $J = 8192$ the AllReduce takes about 3.2 times as long as the math.
- **Second (training):** $A[I_X, J] \cdot_J B[J_X, K_Y] \to C[I_X, K_Y]$
    - Only $B$ has a sharded contracting dimension (Case 2), on X. So we run $\text{AllGather}_X(B[J_X, K_Y]) \to B[J, K_Y]$, where each line along X gathers a $[J, K/4]$ block of weights ($JK/2$ bytes), and then multiply locally:

      $$
      T_{\text{comms}} = \frac{JK/2}{6.0 \times 10^{10}} = \frac{JK}{1.2 \times 10^{11}} \text{ s} \qquad \frac{T_{\text{comms}}}{T_{\text{math}}} = \frac{8 \times 1.97 \times 10^{14}}{1.2 \times 10^{11} \cdot I} \approx \frac{13{,}133}{I}
      $$

    - The comms don't depend on the batch but the math does, so here the batch has to be big. We're compute bound once $I > 13{,}133$. With $I = 1024$ we'd spend about 12.8 times longer gathering weights than multiplying.
- **Third (inference):** $A[I_X, J] \cdot_J B[J, K_Y] \to C[I_X, K_Y]$
    - Nobody has $J$ sharded, and the two sharded dimensions sit on different mesh axes, so this is Case 1: $T_{\text{comms}} = 0$, all math.
- Meaning, same FLOPs per chip in all three. The inference layout is free, the training layout pays to move weights (so it wants a big batch), and the first one pays to move outputs (so it wants a big $J$).

**Question 7 [sharding a transformer block]:** A Transformer MLP block is $In[B, D] \cdot W_{\text{in}}[D, F] \cdot W_{\text{out}}[F, D]$ with $F \gg D$. With $D = 8192$, $F = 32768$, $B = 128$, all bfloat16, on a TPU v5e $2 \times 2$ slice where we pretend each TPU only has 300MB of memory free, how should we shard $In$, $W_{\text{in}}$, $W_{\text{out}}$ and $Out$ so everything fits and runs as fast as possible? How does the time split between comms and FLOPs? Hint: $Out$ doesn't have to be fully replicated, but it needs the same sharding as $In$ so the layer can be stacked.
- **Memory first.** Each weight matrix is $2 \times 8192 \times 32768 = 536{,}870{,}912\text{ bytes}$, about 537MB. The activations are $2 \times 128 \times 8192 = 2{,}097{,}152\text{ bytes}$, about 2.1MB, which is nothing.
- Together the weights are 1,074MB, and the 4 chips have 1,200MB. No room to replicate anything, so both matrices get split 4 ways over X and Y: $2 \times 134\text{ MB} = 268\text{ MB}$ per chip, with about 32MB left over. The only real choice is which dimension we cut:
    1. $In[B_X, D] \cdot W_{\text{in}}[D_{XY}, F] \cdot W_{\text{out}}[F, D_{XY}] \to Out[B_X, D]$, cutting $D$. This is FSDP (fully sharded data parallelism): the batch and the weights are both sharded, and the weights get AllGathered right before each matmul.
    2. $In[B, D_{XY}] \cdot W_{\text{in}}[D, F_{XY}] \cdot W_{\text{out}}[F_{XY}, D] \to Out[B, D_{XY}]$, cutting $F$. This is tensor parallelism. Chapter 5 covers both properly.
- **FSDP loses here.** Gathering $W_{\text{in}}$ means 537MB landing on a chip with 32MB to spare, so it doesn't even fit. And if it did, each gather takes around 4.5 ms, so about 8.9 ms of comms for the two matrices.
- **Tensor parallelism.** The big matrices never move, only the 2.1MB activations do:
    1. $\text{AllGather}_{XY}(In[B, D_{XY}]) \to In[B, D]$
    2. $In[B, D] \cdot W_{\text{in}}[D, F_{XY}] \to H[B, F_{XY}]$. Case 1, no comms.
    3. $H[B, F_{XY}] \cdot W_{\text{out}}[F_{XY}, D] \to Out[B, D]\{U_{XY}\}$. Case 3, multiplied locally.
    4. $\text{ReduceScatter}_{XY}(Out[B, D]\{U_{XY}\}) \to Out[B, D_{XY}]$. This lands us back on the input's sharding, and a ReduceScatter is half the price of an AllReduce. That's what the hint is for.
- Peak memory per chip is the 268MB of weights plus three small activation arrays, about 275MB. Under the limit.
- **FLOPs.** Each chip does two matmuls of $2 \cdot B \cdot D \cdot F/4$:

  $$
  \text{FLOPs per chip} = 2 \times (2 \times 128 \times 8192 \times 8192) = 34{,}359{,}738{,}368 \approx 3.44 \times 10^{10}
  $$

  $$
  T_{\text{math}} = \frac{3.44 \times 10^{10} \text{ FLOPs}}{1.97 \times 10^{14} \text{ FLOPs/s}} = \mathbf{174 \ \mu\text{s}}
  $$

- No duplicated work: $4BDF \approx 1.37 \times 10^{11}$ FLOPs in total, same as unsharded.
- **Comms.** A v5e $2 \times 2$ has no wraparounds, so each chip has just 2 ICI links (one per axis) at 4.5e10 bytes/s each way. In the AllGather every chip pulls in the other three shards, 3/4 of the 2.1MB:

  $$
  T_{\text{AllGather}} = \frac{\frac{3}{4} \times 2{,}097{,}152 \text{ bytes}}{2 \times 4.5 \times 10^{10} \text{ bytes/s}} = \mathbf{17.5 \ \mu\text{s}}
  $$

- The ReduceScatter moves the same volume back:

  $$
  T_{\text{comms}} = 2 \times 17.5 \ \mu\text{s} \approx \mathbf{35 \ \mu\text{s}}
  $$

- (The multi-axis formula from Question 2 would say 23μs, but it assumes two links into every chip per axis, and a $2 \times 2$ only has one.)
- So the block spends roughly **174μs on FLOPs and 35μs on comms**. The math dominates: around 174μs if the comms overlap with the matmuls, 209μs if they run back to back. Compared to FSDP we move $F / B = 256$ times fewer bytes.
- The idea to keep: put the communication on the $[B, D]$ arrays and never on anything with an $F$ in it.
- FYI, at $B = 128$ the real bottleneck is HBM: 128 FLOPs/byte is under the v5e's 240 from Chapter 1, and streaming each chip's 268MB of weights takes around 327μs at 8.2e11 bytes/s.

**Question 8 [challenge]:** Using the JAX snippet from the notes, put a sharded array on the mesh and time each of the 4 communication primitives (AllGather, AllReduce, ReduceScatter, AllToAll) with pmap or shard_map, using `jax.lax.all_gather`, `jax.lax.psum`, `jax.lax.psum_scatter` and `jax.lax.all_to_all`. What exactly does each one do, and how long does it take?
- **Semantics.** Inside `shard_map` we only see one device's local block, and each function is one of our primitives acting on that block along a named mesh axis. With 4 chips on X and a global $[E, F]$ array:

| Primitive | Call inside `shard_map` | Local block in | Local block out |
|---|---|---|---|
| AllGather | `all_gather(x, 'X', axis=0, tiled=True)` | $[E/4, F]$ | $[E, F]$ |
| AllReduce | `psum(x, 'X')` | $[E, F]$ | $[E, F]$ |
| ReduceScatter | `psum_scatter(x, 'X', scatter_dimension=0, tiled=True)` | $[E, F]$ | $[E/4, F]$ |
| AllToAll | `all_to_all(x, 'X', split_axis=1, concat_axis=0, tiled=True)` | $[E/4, F]$ | $[E, F/4]$ |

- `tiled=True` makes these match our notation by concatenating (or splitting) along an existing axis. Without it, `all_gather` stacks the shards on a new leading axis. For `psum` and `psum_scatter`, every device starts with its own full sized block of partial sums (our $\{U_X\}$).
- **Code.** Same array as Pop Quiz 2, `bf16[2048, 8192]`, so we already have a number for the AllGather:

```python
import time
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

# 8 TPUs as a 4x2 mesh; every collective below runs over the 4 chips on 'X'.
assert len(jax.devices()) == 8
mesh = jax.make_mesh((4, 2), ('X', 'Y'))
N = mesh.shape['X']
E, F = 2048, 8192  # bf16[2048, 8192] is 33.5MB

def shmap(f, in_spec, out_spec):
  # check_vma=False: we promise JAX the replicated outputs really are replicated.
  return jax.jit(jax.shard_map(
      f, mesh=mesh, in_specs=in_spec, out_specs=out_spec, check_vma=False))

# AllGather_X: [E_X, F] -> [E, F]
all_gather = shmap(lambda x: jax.lax.all_gather(x, 'X', axis=0, tiled=True),
                   P('X', None), P(None, None))
# AllReduce_X: [E, F]{U_X} -> [E, F]
all_reduce = shmap(lambda x: jax.lax.psum(x, 'X'), P('X', None), P(None, None))
# ReduceScatter_X: [E, F]{U_X} -> [E_X, F]
reduce_scatter = shmap(
    lambda x: jax.lax.psum_scatter(x, 'X', scatter_dimension=0, tiled=True),
    P('X', None), P('X', None))
# AllToAll_X: [E_X, F] -> [E, F_X]
all_to_all = shmap(
    lambda x: jax.lax.all_to_all(x, 'X', split_axis=1, concat_axis=0, tiled=True),
    P('X', None), P(None, 'X'))

def put(shape):
  x = jax.random.normal(jax.random.key(0), shape, jnp.bfloat16)
  return jax.device_put(x, jax.NamedSharding(mesh, P('X', None)))

sharded = put((E, F))        # local block [E/4, F]
unreduced = put((N * E, F))  # local block [E, F]: one full partial sum per chip

def bench(name, fn, x, iters=100):
  fn(x).block_until_ready()  # compile and warm up
  start = time.perf_counter()
  for _ in range(iters):
    out = fn(x).block_until_ready()
  us = (time.perf_counter() - start) / iters * 1e6
  print(name, x.addressable_shards[0].data.shape, '->',
        out.addressable_shards[0].data.shape, f'{us:.0f} us per call')

bench('AllGather', all_gather, sharded)
bench('AllReduce', all_reduce, unreduced)
bench('ReduceScatter', reduce_scatter, unreduced)
bench('AllToAll', all_to_all, sharded)
```

- **What theory predicts.** $V = 2 \times 2048 \times 8192 = 33{,}554{,}432\text{ bytes}$ in every case: the whole global array for the AllGather and the AllToAll, and the full sized block of partial sums for the two reductions. On a TPU v5e $4 \times 2$ (8 chips, 4.5e10 bytes/s per link one way), the X axis has 4 chips and no wraparound:

| Primitive | Cost | Full ring ($W_{\text{ici}}$ = 9.0e10) | Axis of 4, no wraparound |
|---|---|---|---|
| AllGather | $V / W_{\text{ici}}$ | 373μs | 559μs |
| ReduceScatter | same as AllGather | 373μs | 559μs |
| AllReduce | 2 AllGathers | 746μs | 1,118μs |
| AllToAll | AllGather / 4 on a ring | 93μs | 186μs |

- The last column is what we should actually see. Pop Quiz 2 counting again: the edge link carries 3 of the 4 shards, so the gather takes $\frac{3}{4} V / (4.5 \times 10^{10})$. For the AllToAll, the middle link carries 4 of the 16 blocks each way, $\frac{1}{4} V / (4.5 \times 10^{10})$, so a third of the AllGather instead of a quarter. The latency floor is about 3μs, so all four are bandwidth bound.
- For reference, the published profile of the Pop Quiz 2 AllGather shows 680μs against our 560μs, roughly 20% over (and wall clock per call adds dispatch overhead on top).
- **TODO(eddie):** run this on a real TPU runtime (for example a Colab TPU runtime; the code has only been checked for shapes on CPU with 8 fake devices) and paste the observed time for each primitive next to the predictions. If the runtime isn't a v5e $4 \times 2$, redo the table with that chip's ICI bandwidth first.

**Question 9 [another strategy for sharded matmuls?]:** When only one input has its contracting dimension sharded, the rule so far was to AllGather that input and contract locally. The other idea is to treat it like Case 3: for $A[I, J_X] \cdot_J B[J, K] \to C[I, K]$, (1) $C[I, K]\{U_X\} = A[I, J_X] \cdot B[J_X, K]$, then (2) $C[I, K] = \text{AllReduce}(C[I, K]\{U_X\})$.
- This is Strategy 2 from Question 4 again, just with indices.
- **1. Spell out what each device computes for $A[N, M]$ and $B[M, K]$, with $A$ sharded as $A[I, J_X]$ over $P$ devices and the output replicated on all of them.**
    - Device $p$ (for $p = 0, \dots, P - 1$) holds the columns of $A$ in its range $J_p = \{pM/P, \dots, (p+1)M/P - 1\}$, an $[N, M/P]$ block. $B$ is replicated, so it just keeps the matching rows $J_p$, an $[M/P, K]$ block.
    - Each device multiplies its two blocks. For every $n$ and $k$:

      $$
      C^{(p)}_{nk} = \sum_{m \in J_p} A_{nm} B_{mk}
      $$

    - That's a full $[N, K]$ matrix on every device, but each entry only has the $M/P$ terms this device owns, for $2NMK/P$ FLOPs per device.
    - Then the AllReduce adds the $P$ partial matrices and leaves the total on every device:

      $$
      C_{nk} = \sum_{p=0}^{P-1} C^{(p)}_{nk}
      $$

- **2. What changes if the result can stay sharded (over $N$ or $K$) instead of replicated?**
    - Step 1 is the same. Step 2 becomes a ReduceScatter, $C[N, K]\{U_X\} \to C[N_X, K]$ or $C[N, K_X]$, so device $p$ only keeps the finished sum for its own $N/P$ rows (or $K/P$ columns). Half the comms, since we skip the AllGather at the end.
- **3. On comms alone, how does the part 2 version compare with gathering $A$ first and then multiplying?**
    - Both are one pass around the ring, and when we're throughput bound a pass costs the size of the full array. The AllGather moves $A$ ($NM$ elements), the ReduceScatter moves $C$ ($NK$ elements):

      $$
      \frac{T_{\text{AllGather } A}}{T_{\text{ReduceScatter } C}} = \frac{2NM / W_{\text{ici}}}{2NK / W_{\text{ici}}} = \mathbf{\frac{M}{K}}
      $$

    - So multiplying first wins whenever the output dimension $K$ is smaller than the contracting dimension $M$. If we need the replicated output (part 1), the AllReduce doubles the cost and the break-even moves to $K < M/2$, the same $D > 2B$ from Question 4.

**Question 10 [fun with AllToAll]:** From the notes, an AllToAll takes a quarter of the time of an AllGather or ReduceScatter (when we're throughput bound). Where does that factor of 4 come from, and how would it change if ICI links only went in one direction?
- **1. One direction first. $D$ devices in a ring do an AllGather or a ReduceScatter on an $N \times N$ matrix $A[I_X, J]$ (say $D$ divides $N$). Describe the comms and count the scalars that cross a single ICI link.**
    - Each device holds a strip of $N/D$ rows, $N^2/D$ scalars. In the AllGather every device passes its strip to its neighbor, then keeps forwarding whatever it just received, for $D - 1$ steps.
    - The ReduceScatter is the same loop with sums: at every step a device adds its contribution to the strip passing through and sends it on, and after $D - 1$ steps each strip lands, fully summed, on the device that keeps it.
    - Either way, one strip crosses each link at each of the $D - 1$ steps:

      $$
      (D - 1) \times \frac{N^2}{D} = \mathbf{N^2 \left(1 - \frac{1}{D}\right)} \approx N^2 \text{ scalars per link}
      $$

- **2. Now an AllToAll, still one direction. How is the algorithm different, and how many scalars cross a single link?**
    - An AllToAll $A[I_X, J] \to A[I, J_X]$ doesn't replicate anything, it just moves data to a new home. Each device cuts its strip into $D$ blocks of $N/D \times N/D$, one per destination. One stays put, and the block for the device $d$ hops away rides $d$ links and stops.
    - So the load shrinks as it travels: $D - 1$ blocks on the first link, then $D - 2$, down to 1. With 4 devices that's $3 + 2 + 1 = 6$ blocks over each link, where the AllGather sent 3 full strips, or 12 blocks.

      $$
      \frac{N^2}{D^2} \times \frac{D(D - 1)}{2} = \mathbf{\frac{N^2}{2} \left(1 - \frac{1}{D}\right)} \approx \frac{N^2}{2} \text{ scalars per link}
      $$

- **3. Why is the ratio between parts 1 and 2 such a nice number?**
    - It's exactly 2. An AllGather strip stays full size ($D$ blocks) for all $D - 1$ hops, so $D(D - 1)$ block-hops. An AllToAll strip drops a block at every stop, so $(D - 1) + (D - 2) + \dots + 1 = D(D - 1)/2$. A triangle instead of a rectangle: the average block only rides halfway around the ring.
- **4. With bidirectional links, how does the AllGather time change?**
    - Every strip can go both ways, so each direction of a link carries $(D - 1)/2$ strips instead of $D - 1$. Half the load per link, so half the time. A 2x win.
- **5. And the AllToAll time?**
    - A 4x win, because two things get halved at once. Only about half the blocks go in each direction, and the farthest any of them travels is $D/2$ hops instead of $D - 1$. One direction of a link now carries $1 + 2 + \dots + D/2$, roughly $D^2/8$ blocks, against $D(D - 1)/2 \approx D^2/2$ before. That's $N^2/8$ scalars per link direction.
- **6. So what explains the ratio between AllGather and AllToAll time on a bidirectional ring?**
    - Two separate factors of 2. Even with one way links, an AllGather copies each strip to every device, while an AllToAll delivers each block to exactly one device, on average half a ring away. Then the return direction gives the AllGather twice the links (2x), but gives the AllToAll twice the links and half the distance (4x). So $2 \times \frac{4}{2} = 4$.
    - Counted exactly it's $4(1 - 1/D)$: 3 for a ring of 4, 3.75 for a ring of 16, and 4 in the limit. So "a quarter" is the big ring answer, and small axes get a bit less.
