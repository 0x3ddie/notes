---
title: Sharded Matrices and How to Multiply Them
description: Overview and Worked Problems
published: true
---
## 3. Sharding Matrices

This is where we can highlight the intensity of AI inference (and pre-training). For a single TPU, we must load the entirety of a model's weights into the massive HBM. When an input vector (representing a token) arrives, the TPU begins a relentless pipeline: streaming weights and the input vector from the HBM, through the lightning-fast VMEM, and into the MXU.

Inside the MXU, systolic arrays operate in a continuous wave, locking weights into place just long enough to multiply them against the flowing input vector, before immediately flushing those weights out to make room for the next layer. This massive data-shuttling operation happens for every single token generated. Sharding is born out of strict physical necessity. Modern model weights simply do not fit in the HBM of a single chip, so we must 'shard' the weight matrices across an entire pod of TPUs.

If Google could magically manufacture a giant chip with 100TB of HBM and a supermassive MXU, it would be significantly faster than 8 separate TPUs working in tandem. This is due entirely to the networking bottleneck. In a sharded setup, the 8 chips must constantly pause to communicate their partial sums across the ICI network cables after intermediate layers. Ultimately, they arrive at a final output vector of ~128k numbers (logits). These logits are pushed through a Softmax function to calculate probabilities, allowing the system to pick the final English word from the tokenizer dictionary.

So now we can talk about HOW to shard so that our compute still stays efficient. There exists a notation for sharding, using a named-axis notation variant to describe how the tensor is sharded in blocks across devices. We can assume a 2D or 3D grid of devices exists, called a device mesh, where each axis is given mesh axis names like `X`, `Y`, `Z`. We then specify how that matrix data is spread across the device by describing how each named dimension of that array is separated across the physical mesh axes.

We define our device mesh like:

$$\text{Mesh}(\text{devices}=((0, 1), (2, 3)), \text{axis\_names}=(\text{'X'}, \text{'Y'}))$$

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

$$A[I,J_x] * B[J,K] \to C[I,K]$$

We can’t simply multiply $A$ and $B$ locally like before; to multiply a row of $A$ by a column of $B$, we need the sum of the entire row, but that row is chopped up between devices. If we do math with a partial row, we get a partial sum.

AllGather is a primitive network operation that removes sharding. By taking data that’s chopped up across TPUs, we force them to share until every device has 100% of the data.

**How:** AllGather works by sharing data in a ring. Imagine our TPUs arranged in a ring formation; like 4 TPUs each with 1 piece of the row. First, we pass over that piece to the right. Now everyone has 2 pieces. We pass over the new piece to the right again. 3 pieces. Now we do a final pass, and come around in a full circle, where everyone has 4 pieces.

We can AllGather in 1 direction or 2 directions.

**Scaling law:** it takes the same amount of time to AllGather across 100 TPUs and 4 TPUs, ONLY if we’re through-put bound already. Meaning, spreading 100 tiny shards across 50 hops mathematically takes the same time as 4 large shards across 4 TPUs.

$$T_{\text{total}} = \frac{V}{W_{\text{ici}}}$$

we’re bottlenecked by the speed of each link.

### Both multiplicands have sharded contracting dims

When the inner dims $J$ of both matrices are sharded.

Previously with 1 and 2, either one or both matrices had their inner dimension untouched, meaning a TPU could calculate a finished number for their assignment.

Now, because we’re multiplying the column vector by the row vector, we no longer get a tiny shard, but a full-sized matrix. But, because we only had partial inputs, its a matrix of partial sums. Now, every TPU in the pod has its own partial matrix; we need to stack all these matrices on top to get the full answer.

$\{U_x\}$ notation stands for ‘Unreduced’. Essentially means, this matrix is locally complete, but we haven’t summed the $x$-axes yet, so its incomplete.

**Primitive: AllReduce:** Every TPU takes its incomplete matrix and sends it to the next TPU, and every TPU adds each matrix together. Insanely expensive, around $2\times$ AllGather, basically brute force method, since we’re passing full-sized matrices around. Also, each TPU at the end holds a complete matrix… wasteful.

**Primitive: ReduceScatter:** In a ring, we pass around matrices, adding them up as we go. As we add, we drop pieces we don’t need, resulting in a complete matrix that’s sharded across the TPUs. This is equally intensive to AllGather.

A cool equation:

$$\text{AllReduce} = \text{ReduceScatter} + \text{AllGather}$$

We just went over 3 communication primitives: AllGather, ScatterReduce, All Reduce. There is a 4th: AllToAll.

All to All appears in MoE models, and concerns the sharded transposition, or resharding operations.

AlltoAll is purely a rearrangement primitive, and is $\frac{1}{4}$ the cost of an AllGather. This takes place when our current sharding structure, such as an $A[I_x,J]$ is incompatible with the math we need to do next, which requires an $A[I,J_x]$. Thus, we need to transpose those rows into columns, or vice versa. The way this happens is by every TPU chopping up it’s data into tiny blocks, and shipping them along a ring. Each TPU then has 1 block of each row, ending up in a column.

> **Note:** you can find useful GIFs on [jax-ml.github.io/scaling-book/sharding/](https://jax-ml.github.io/scaling-book/sharding/) for references on each primitive, or ask Gemini to visualize the methods/math (like asking TPUs to explain TPUs fr)

To conclude on sharding: There are 4 main communication primitives that define how shards move in pods. In some cases, we might not need to shard at all if our matrices are complete (Case 1), but most cases, and all cases in production environments, will require sharding due to massive sizes and arrays. Many of the cases we described are very rudimentary use cases involving 4, 8 TPUs and some tiny matrices. What about 4000 TPUs? The scale is honestly staggering, so understanding simple concepts like how shards move along ICI to result in final, complete matrices, and which primitive to use or discard to optimize can drastically reduce network and memory latency.

### Problem Set
