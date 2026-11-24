---
title: Programming TPUs in JAX
description: JAX Intro & Solutions
published: true
---
## 10. Programming TPUs in JAX

Up to now, sharding has been something we did on paper: write $A[B_X, D_Y]$, count the AllGathers, divide bytes by ICI bandwidth. This chapter is where that notation turns into code.

Everything below is written for JAX 0.10 and checked on 8 fake CPU devices (`jax.config.update('jax_num_cpu_devices', 8)`), which is fine for correctness and says nothing about speed. The three snippets in the notes are one session, so the later ones reuse the imports and arrays from the first.

## How Does Parallelism Work in JAX?

JAX has three ways to run one program across many chips, and they basically differ in who decides where the communication goes: the compiler, JAX, or us.

- **Auto:** We write normal single-device code with a global view of every array. If the inputs are sharded, XLA's partitioner (Shardy) picks a sharding for every intermediate and inserts whatever collectives it thinks it needs.
- **Explicit:** Still a global view and no hand-written communication, but the sharding is part of each array's type, and JAX propagates it op by op while tracing. When an op could go two ways, we get an error and have to pick.
- **Manual (`jax.shard_map`):** A per-device view. Nothing moves between devices unless we write the collective ourselves.

I think of auto as the default, since most of a model is elementwise ops and matmuls the compiler handles fine. Explicit is for when we want to know every value's sharding without digging through a profile. Manual is for the few hot spots where we know a better schedule than the compiler, like the collective matmul below. They mix, too: every `shard_map` function here is wrapped in `jax.jit`, and `jax.sharding.auto_axes` and `jax.sharding.explicit_axes` switch between the first two inside one function. Whichever mode we're in, the mental model is still Chapter 3: look at which dimensions are sharded, and we know which collective has to show up.

### Auto sharding mode

The Chapter 3 notation maps straight onto two objects. A mesh is a named grid of devices: `jax.make_mesh((4, 2), ('X', 'Y'))` is our `Mesh({'X': 4, 'Y': 2})`. A `PartitionSpec` (`jax.P`) has one entry per array dimension, naming the mesh axis it's split along:

- $A[B_X, D_Y]$ is `P('X', 'Y')`.
- $W[D_Y, F]$ is `P('Y', None)`. `None` means unsplit, so $W$ is replicated along `X`.
- $A[I_{XY}, J]$ is `P(('X', 'Y'), None)`, and a fully replicated $A[I, J]$ is `P()`.

We get a sharded array with `jax.device_put`, by passing a sharding to a creation function, or from a jitted function's `out_shardings`. One catch: on JAX 0.10, `make_mesh` gives `Explicit` axes unless told otherwise, so auto mode has to be asked for.

```python
import jax
jax.config.update('jax_num_cpu_devices', 8)   # 8 fake CPU devices stand in for a TPU slice
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType

mesh = jax.make_mesh((4, 2), ('X', 'Y'), (AxisType.Auto, AxisType.Auto))
jax.set_mesh(mesh)

rng = np.random.default_rng(0)
A_np = rng.standard_normal((8, 2048), dtype=np.float32)
W_np = rng.standard_normal((2048, 8192), dtype=np.float32)
A = jax.device_put(A_np, jax.P('X', 'Y'))    # A[B_X, D_Y]
W = jax.device_put(W_np, jax.P('Y', None))   # W[D_Y, F]
print(A.addressable_shards[0].data.shape)    # (2, 1024): each device holds B/4 by D/2
print(W.addressable_shards[0].data.shape)    # (1024, 8192)

matmul = jax.jit(lambda A, W: A @ W, out_shardings=jax.P('X', None))   # Out[B_X, F]
out = matmul(A, W)
print(out.sharding.spec, out.addressable_shards[0].data.shape)        # P('X', None) (2, 8192)
```

$D$ is sharded along `Y` on both sides, which is the Chapter 3 case where a device can only compute a partial sum:

$$
\text{Out}[B_X, F]\{U_Y\} = A[B_X, D_Y] \cdot_D W[D_Y, F]
$$

$$
\text{Out}[B_X, F] = \text{AllReduce}_Y(\text{Out}[B_X, F]\{U_Y\})
$$

We only wrote `A @ W`, but the compiled program (`matmul.lower(A, W).compile().as_text()`) has a local `dot` producing `f32[2,8192]` and then exactly one `all-reduce` over the device pairs along `Y`. The compiler added the second step for us.

The catch is that it can also add things we don't want, like a giant AllGather in the middle of a profile. The main lever is `jax.lax.with_sharding_constraint(x, jax.P(...))`, which pins one intermediate's sharding and hopes the rest follows. We're nudging a compiler, not telling it what to do.

### Explicit sharding mode

```python
mesh = jax.make_mesh((4, 2), ('X', 'Y'), (AxisType.Explicit, AxisType.Explicit))
jax.set_mesh(mesh)

A = jax.device_put(A_np, jax.P('X', 'Y'))
W = jax.device_put(W_np, jax.P('Y', None))
print(jax.typeof(A), jax.typeof(W))   # float32[8@X,2048@Y] float32[2048@Y,8192]
print(jax.typeof(jnp.square(A)))      # float32[8@X,2048@Y], elementwise keeps the sharding

try:
    A @ W                             # D is sharded on both sides: AllReduce or ReduceScatter?
except Exception as err:
    print(type(err).__name__)         # ShardingTypeError

out = jnp.einsum('bd,df->bf', A, W, out_sharding=jax.P('X', 'Y'))
print(jax.typeof(out))                # float32[8@X,8192@Y]
```

`jax.typeof` now prints the sharding as part of the type, and elementwise ops carry it along. The matmul gets rejected with a `ShardingTypeError`: with $D$ sharded on both sides, JAX can't know if we want $\text{Out}[B_X, F_Y]$ (a ReduceScatter) or $\text{Out}[B_X, F]$ (an AllReduce), so we pick with `out_sharding`. Meaning, explicit mode still writes the collectives for us. It just refuses to guess the layout.

### Manual sharding mode via shard_map

```python
@jax.jit
@jax.shard_map(in_specs=(jax.P('X', 'Y'), jax.P('Y', None)), out_specs=jax.P('X', None))
def matmul_by_hand(a, w):                  # per device: a is [B/4, D/2], w is [D/2, F]
    partial_sum = a @ w                    # Out[B_X, F] {U_Y}, a sum over half of D
    return jax.lax.psum(partial_sum, 'Y')  # AllReduce over Y finishes the sum

out = matmul_by_hand(A, W)
print(jax.typeof(out))                     # float32[8@X,8192]
```

Inside `shard_map`, `a` is `[2, 1024]` and `w` is `[1024, 8192]`: the body is what ONE device runs. `in_specs` says how the global inputs get cut up, and `out_specs` how the per-device outputs get stitched back together. `a @ w` is the unreduced partial sum and `psum` is the AllReduce, the same two steps as above except now we wrote both. JAX tracks this in the types too: before the `psum` a value is tagged `{V:(X,Y)}` (it varies across both mesh axes), after it `{V:X}`, which is what makes `out_specs=P('X', None)` legal.

Every Chapter 3 primitive has a call. For a local `[2, 1024]` shard on this mesh:

- **AllReduce:** `jax.lax.psum(x, 'Y')` (also `pmean`, `pmax`) stays `[2, 1024]`, identical along `Y`.
- **AllGather:** `jax.lax.all_gather(x, 'Y', axis=1, tiled=True)` gives `[2, 2048]`.
- **ReduceScatter:** `jax.lax.psum_scatter(x, 'Y', scatter_dimension=1, tiled=True)` gives `[2, 512]`.
- **AllToAll:** `jax.lax.all_to_all(x, 'X', 1, 1, tiled=True)` stays `[2, 1024]`, with pieces swapped between devices.
- **One hop:** `jax.lax.ppermute(x, 'Y', perm)` gives a neighbour's shard.

`jax.lax.axis_index('Y')` and `jax.lax.axis_size('Y')` tell a device where it sits on an axis.

Some things are just easier per device. The first 4 elements of every shard, averaged across the mesh, is one `jax.lax.pmean(x[:4], ...)` locally, versus describing a strided slice of the global array and trusting XLA to see that almost nothing needs to move.

The realistic use case is model parallelism: $A[B_X, D_Y] \cdot_D W[D, F_Y] \to \text{Out}[B_X, F_Y]$. The Chapter 3 recipe is to AllGather $A$ along `Y` and then multiply locally. Problem is, that's sequential. While the AllGather is in flight, the MXU does nothing.

There's a cool trick, the collective matmul (Wang et al. 2023): a device doesn't need all of $A$ to start. Its local chunk of $A$ times the matching $D/Y$ rows of $W$ is already one term of the final sum. So:

1. Multiply the chunk we hold by its rows of $W$ and add the result to an accumulator.
2. At the same time, pass that chunk to the next device on the `Y` ring and receive the previous device's chunk.
3. Repeat. After $\lvert Y \rvert - 1$ hops every device has seen every chunk.

Same bytes as the AllGather, they just move while the matmul runs instead of before it, and no device ever holds the gathered $A[B_X, D]$. In code it's a loop of `ppermute` plus a small matmul (Question 3 has one). The profiles on [jax-ml.github.io/scaling-book/jax-stuff/](https://jax-ml.github.io/scaling-book/jax-stuff/) put the blocking version at 311 μs and the collective one at 244 μs, close to the 224 μs of a matmul with no sharded contracting dimension.

> **Note:** On JAX 0.10, the listings on that page that call `jax.make_mesh(axis_sizes=...)` stop with a `TypeError`. The keyword is `axis_shapes` now.

## Worked Problems

All of these assume $N$ devices, here $N = 8$ fake CPU devices on JAX 0.10. The code has only been run on CPU so far: every block matches a NumPy reference, and the collective counts come from the CPU compile (a TPU compile may differ).

**Question 1 [per-shard ops]:** $A$ is an activation array $\text{float32}[S_X, D_Y]$ with $X \cdot Y = N$. We test with `Mesh({'X': 4, 'Y': 2})`, $S = 512$ and $D = 64$, so each shard is `float32[128, 32]`.

**Part 1: Return the mean of each $(X, Y)$ shard as an $[X, Y]$ array, once with `jax.jit` and once with `shard_map`. Profile both: did any communication get added? (None should, but XLA sometimes adds some anyway.)**

- A shard's mean only needs that shard, the cleanest case for the per-device view.

```python
import functools
import jax
jax.config.update('jax_num_cpu_devices', 8)
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType

mesh = jax.make_mesh((4, 2), ('X', 'Y'), (AxisType.Auto, AxisType.Auto))   # N = 8
jax.set_mesh(mesh)
SPEC = jax.P('X', 'Y')

@jax.jit
@jax.shard_map(in_specs=SPEC, out_specs=SPEC)
def shard_mean(a):                    # a is one shard, [S/X, D/Y]
    return a.mean(keepdims=True)      # [1, 1] per device, stitched into [X, Y]

@functools.partial(jax.jit, out_shardings=SPEC)
def shard_mean_jit(a):                # a is the global [S, D] array
    (X, Y), (S, D) = mesh.axis_sizes, a.shape
    blocks = a.reshape(X, S // X, Y, D // Y)   # split each dim into (shard, within-shard)
    return blocks.mean(axis=(1, 3))
```

- The `jax.jit` version has the global view, so it rebuilds the shards with a 4D reshape and hopes XLA notices every block is already local.
- Check: both match the NumPy block means. Communication added? **No.** Zero collectives in either version.
- Profiling them needs a TPU, so I'll come back to it.

**Part 2: Using only `shard_map`, return `roll(x, shift, axis=0) - x`, rolling inside each shard along `X`.**

- Rows wrap within their own block of $S/X$ rows, not the whole array, so per device it's just `jnp.roll` on the local shard.

```python
def shard_roll_diff(a, shift):
    local = lambda s: jnp.roll(s, shift, axis=0) - s    # s is one shard, rows wrap inside it
    return jax.jit(jax.shard_map(local, in_specs=SPEC, out_specs=SPEC))(a)
```

- Check: matches a NumPy roll inside each block and NOT the global roll, which is the point. Zero collectives again.

**Question 2 [mixture of experts]:** $W$: $\text{float32}[E_X, D, F]$ holds $E$ expert matrices, $A$: $\text{float32}[S_X, D]$ the activations, and $B$: $\text{int32}[S_X]$ the routing, with `B[i]` in $[0, E)$ picking token $i$'s expert. We want `Out[i] = A[i] @ W[B[i]]`. We test with `Mesh({'X': 8})`, $E = 16$, $S = 64$, $D = 8$, $F = 12$, so each device holds 2 experts and 8 tokens.

**Part 1: On one device, write a version that never builds an $[S, D, F]$ array. Hint: loop over experts and mask, or sort tokens into a padded $[E, S, D]$ buffer for one batched matmul (why must the second dimension be $S$?). `jax.lax.ragged_dot` is the sorted version without padding.**

- The obvious `einsum('sd,sdf->sf', A, W[B])` builds exactly that $[S, D, F]$. Instead we sort tokens by expert id, `ragged_dot` each group against its own expert, and undo the sort.

```python
import functools
import jax
jax.config.update('jax_num_cpu_devices', 8)
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType

mesh = jax.make_mesh((8,), ('X',), (AxisType.Auto,))    # N = 8 devices on one axis
jax.set_mesh(mesh)
P = jax.P

def moe_local(W, A, B):
    """Out[i] = A[i] @ W[B[i]] for W[E, D, F], A[S, D], B[S]. Ids outside [0, E) give zero rows."""
    E = W.shape[0]
    key = jnp.where((B >= 0) & (B < E), B, E)         # ids that aren't ours sort to the very end
    order = jnp.argsort(key)                          # token indices, grouped by expert
    group_sizes = jnp.bincount(key, length=E)         # how many tokens each expert gets
    out_sorted = jax.lax.ragged_dot(A[order], W, group_sizes)     # [S, F], never [S, D, F]
    return jnp.zeros_like(out_sorted).at[order].set(out_sorted)   # undo the sort
```

- Ids outside $[0, E)$ sort to the end and come back as zero rows. The sharded versions below lean on this.
- The mesh is `Auto` on purpose, since explicit mode refuses Part 2's sort along a sharded dimension.
- Why $S$: shapes are static under jit, and all $S$ tokens could choose the same expert.
- Check: exact match with the per-token einsum for four routings, uniform through all tokens on one expert.

> **Note:** On CPU, `ragged_dot` lowers to the padded $[E, S, D]$ version anyway. The native ragged instruction is TPU only, so the zero rows above have only been checked on CPU.

**Part 2: Now just `jax.jit` the local function on sharded inputs. What communication did the compiler pick, and how long does it take?**

```python
W = jax.device_put(W_np, P('X'))     # W[E_X, D, F]
A = jax.device_put(A_np, P('X'))     # A[S_X, D]
B = jax.device_put(B_np, P('X'))     # B[S_X]
moe_jit = jax.jit(moe_local, out_shardings=P('X'))
```

- What XLA picked: **2 all-gathers and 3 all-reduces**. One all-gather carries all of $B$ (`s32[64]`), and two all-reduces carry the ENTIRE activations and output (`f32[64,8]` and `f32[64,12]`) to every device. Meaning, a global sort turns a problem that's local to each expert into "ship everything everywhere".
- Timing it needs a TPU, so I'll come back to it.

**Part 3: The jit version probably puts all the activations on every device, an $\text{AllGather}_X([S_X, D])$, which costs communication and maybe memory. Redo it with `shard_map` and explicit communication.**

**3.1: First pass: a `jax.lax.all_gather` and the same reordering as Part 1.**

- Each device owns $E/X$ experts. It gathers all of $A$ and $B$, then subtracts its first expert's id, so every token it doesn't own falls outside $[0, E/X)$ and comes back from `moe_local` as a zero row.
- Every token has one owner, so the per-device results sum to the answer. `psum_scatter` does the sum and hands each device its $S/X$ rows.

```python
IN_SPECS = (P('X'), P('X'), P('X'))                     # W[E_X, D, F], A[S_X, D], B[S_X]

def moe_allgather_shard(W, A, B):                       # one device: W[E/X, D, F], A[S/X, D], B[S/X]
    first = jax.lax.axis_index('X') * W.shape[0]        # global id of my first expert
    A_all = jax.lax.all_gather(A, 'X', tiled=True)      # [S, D] lands on every device
    B_all = jax.lax.all_gather(B, 'X', tiled=True)      # [S]
    out = moe_local(W, A_all, B_all - first)            # [S, F], zero rows for tokens I don't own
    return jax.lax.psum_scatter(out, 'X', tiled=True)   # ReduceScatter back to Out[S_X, F]

moe_allgather = jax.jit(jax.shard_map(moe_allgather_shard, in_specs=IN_SPECS, out_specs=P('X')))
```

- Check: exact on the four routings, and the HLO has just the two all-gathers and one reduce-scatter we wrote.
- Still wasteful though. Every device receives all $S$ tokens and mostly runs its experts on somebody else's.

**3.2: Second pass: work raggedly with a `jax.lax.all_to_all` inside a `jax.lax.while_loop`, so nothing of size $[E, S, D]$ exists, the full activations never land on one device, and no compute goes to padding. How much faster is it?**

- Each token should go ONLY to the device that owns its expert, which is an AllToAll. Shapes are static and a device's tokens could all go to one place, so we send rounds of at most `chunk` tokens per (source, destination) pair until nobody has anything left.

```python
def moe_a2a_shard(W, A, B, chunk):
    """Per-device body. Each round ships at most `chunk` tokens to every other device."""
    n = jax.lax.axis_size('X')
    (e_loc, D, F), s_loc = W.shape, A.shape[0]
    first = jax.lax.axis_index('X') * e_loc
    dest = B // e_loc                                  # device that owns each token's expert
    order = jnp.argsort(dest)                          # my tokens, grouped by destination
    A_s, B_s = A[order], B[order]
    counts = jnp.bincount(dest, length=n)              # tokens bound for each device
    starts = jnp.cumsum(counts) - counts
    slot = jnp.arange(chunk)

    def tokens_left(state):
        sent, _, _ = state
        return jax.lax.psum(jnp.sum(counts - sent), 'X') > 0     # same answer on every device

    def one_round(state):
        sent, out, rounds = state
        valid = slot[None, :] < (counts - sent)[:, None]               # [n, chunk]
        pos = jnp.where(valid, (starts + sent)[:, None] + slot[None, :], s_loc)
        send_x = A_s.at[pos].get(mode='fill', fill_value=0)            # [n, chunk, D]
        send_e = B_s.at[pos].get(mode='fill', fill_value=-1)           # [n, chunk], -1 is padding
        recv_x = jax.lax.all_to_all(send_x, 'X', 0, 0)                 # row j: what device j sent me
        recv_e = jax.lax.all_to_all(send_e, 'X', 0, 0)
        y = moe_local(W, recv_x.reshape(n * chunk, D), recv_e.reshape(n * chunk) - first)
        back = jax.lax.all_to_all(y.reshape(n, chunk, F), 'X', 0, 0)   # results go home
        out = out.at[pos].set(back, mode='drop')
        return sent + valid.sum(axis=1), out, rounds + 1

    out0, rounds0 = jax.lax.pcast((jnp.zeros((s_loc, F), A.dtype), 0), 'X', to='varying')
    state = jax.lax.while_loop(tokens_left, one_round, (jnp.zeros_like(counts), out0, rounds0))
    _, out_sorted, rounds = state
    return jnp.zeros_like(out_sorted).at[order].set(out_sorted), jax.lax.pmax(rounds, 'X')

def make_moe_a2a(chunk):
    body = functools.partial(moe_a2a_shard, chunk=chunk)
    return jax.jit(jax.shard_map(body, in_specs=IN_SPECS, out_specs=(P('X'), P())))
```

- Each round, row $j$ of a device's `[n, chunk, D]` buffer holds its next tokens for device $j$, padded with expert id -1. Two `all_to_all`s deliver tokens and ids, the receiver runs `moe_local`, and a third sends the results home. The loop stops on a `psum` of unsent tokens, so every device stops together.
- Check: exact on the four routings for chunk sizes 1, 2, 4 and 8. The HLO has three all-to-alls, two scalar all-reduces and no all-gather, and each token buffer is `f32[1,2,8]` at a chunk of 2, never $[S, D]$.
- How much faster? With balanced routing each device receives about $S/X$ tokens instead of all $S$, roughly $X$ times less data. With skewed routing a hot device still gets everything, just over more rounds.
- The real speedup needs a TPU, so I'll come back to it.

**Part 4: Real MoEs send each token to $k$ experts and average the results. Refactor for $B$: $\text{int32}[S_X, k]$.**

- A token routed to $k$ experts is just $k$ (token, expert) pairs. So we repeat each activation $k$ times, flatten $B$, run any one-expert routine unchanged, and average each token's $k$ rows.

```python
def topk_routing(moe_fn):
    """Lift a one-expert routine to B[S, k]: each (token, expert) pair becomes its own row."""
    def wrapped(W, A, B):
        S, k = B.shape
        out = moe_fn(W, jnp.repeat(A, k, axis=0), B.reshape(S * k))   # [S * k, F]
        return out.reshape(S, k, -1).mean(axis=1)                     # average over the k experts
    return wrapped

a2a_one_expert = lambda W, A, B: moe_a2a_shard(W, A, B, chunk=2)[0]    # drop the round counter

moe_local_topk = jax.jit(topk_routing(moe_local))
moe_allgather_topk = jax.jit(jax.shard_map(
    topk_routing(moe_allgather_shard), in_specs=IN_SPECS, out_specs=P('X')))
moe_a2a_topk = jax.jit(jax.shard_map(
    topk_routing(a2a_one_expert), in_specs=IN_SPECS, out_specs=P('X')))
```

- The wrapper only touches local rows, so it drops into a `shard_map` body without changing the communication.
- Check: for $k = 1, 2, 3$, all three versions match the averaged per-token einsums.

**Question 3 [collective matmuls]:** The notes covered the AllGather collective matmul. Build the other two flavours and chain them into a Transformer MLP. We test with `Mesh({'X': 2, 'Y': 4})`, $B = 8$, $D = 32$, $F = 64$.

**Part 1: Write an AllReduce collective matmul, $A[B_X, D_Y] \cdot_D W[D_Y, F] \to \text{Out}[B_X, F]$ (output replicated across `Y`), that overlaps communication with compute. Hint: tile over the output dimension; `jax.lax.psum` is allowed. Given how XLA treats this, it may not beat the baseline.**

- The naive version, `matmul_by_hand` from the notes, has the whole $[B/X, F]$ partial sum before any of it is sent. Tiling the output lets each tile go out while the next is multiplied.

```python
import jax
jax.config.update('jax_num_cpu_devices', 8)
import jax.numpy as jnp
import numpy as np

mesh = jax.make_mesh((2, 4), ('X', 'Y'))     # axis types default to Explicit on JAX 0.10
jax.set_mesh(mesh)
P = jax.P

def allreduce_matmul_tiled(lhs, rhs, tiles=4):
    """A[B_X, D_Y] @ W[D_Y, F] -> Out[B_X, F]. Per device: lhs [B/X, D/Y], rhs [D/Y, F]."""
    done = [jax.lax.psum(lhs @ cols, 'Y') for cols in jnp.split(rhs, tiles, axis=1)]
    return jnp.concatenate(done, axis=1)     # one small matmul and one small AllReduce per tile
```

- It's a per-device body, run under `jax.shard_map` with `in_specs=(P('X', 'Y'), P('Y', None))` and `out_specs=P('X', None)`.
- Check: exactly equal to `A @ W`. But the HLO has **ONE all-reduce**: XLA merged our four small `psum`s back together. A version XLA can't undo needs the ring, which is what Part 2 builds.

**Part 2: Write an overlapped ReduceScatter collective matmul, $\text{Tmp}[B_X, F_Y] \cdot_F W_2[F_Y, D] \to \text{Out}[B_X, D_Y]$ (a Transformer's down-projection), that passes only as much data as needed. Hint: permute the result while accumulating it.**

- The naive version builds the full $[B/X, D]$ partial product before a `psum_scatter`, though each device only keeps $D/Y$ of the columns. Instead, an accumulator for ONE device's columns, $[B/X, D/Y]$, travels the ring picking up each device's contribution, with $\lvert Y \rvert$ of them in flight at once.

```python
def shift(x, by):
    """Send x to the device `by` steps along the Y ring (by = +1 or -1 is a single hop)."""
    n = jax.lax.axis_size('Y')
    return jax.lax.ppermute(x, 'Y', [(j, (j + by) % n) for j in range(n)])

def reducescatter_matmul(lhs, rhs):
    """L[B_X, K_Y] @ R[K_Y, M] -> Out[B_X, M_Y]. Per device: lhs [B/X, K/Y], rhs [K/Y, M]."""
    n, me = jax.lax.axis_size('Y'), jax.lax.axis_index('Y')
    chunk = rhs.shape[1] // n                    # each device ends up owning M/Y output columns
    acc = 0
    for i in range(n):
        # The accumulator I hold at step i finishes its trip on device me - (i + 1).
        cols = jax.lax.dynamic_slice_in_dim(rhs, (me - i - 1) % n * chunk, chunk, axis=1)
        acc = acc + lhs @ cols                   # add my contribution to that device's columns
        if i < n - 1:
            acc = shift(acc, +1)                 # pass only [B/X, M/Y] along the ring
    return acc
```

- The only real thinking is the index. The accumulator held at step $i$ has $\lvert Y \rvert - 1 - i$ hops left, so it ends on device $\text{me} - (i + 1)$, and that device's columns of $W_2$ are what we slice.
- Every hop carries only $[B/X, D/Y]$, and step $i + 1$'s matmul doesn't wait on step $i$'s permute, so the two can overlap. The HLO has 3 collective-permutes for $\lvert Y \rvert = 4$ and no reduce-scatter.
- Check: exactly equal to `Tmp @ W2`. Same pattern as `matmul_psumscatter_overlapped` in the [JAX `shard_map` notebook](https://docs.jax.dev/en/latest/notebooks/shard_map.html).

Back to Part 1. From Chapter 3, $\text{AllReduce} = \text{ReduceScatter} + \text{AllGather}$, and we just built an overlapped ReduceScatter:

```python
def allreduce_matmul(lhs, rhs):
    """A[B_X, D_Y] @ W[D_Y, F] -> Out[B_X, F], with the reduce half riding the ring."""
    tile = reducescatter_matmul(lhs, rhs)                                      # Out[B_X, F_Y]
    return jax.lax.all_gather(tile, 'Y', axis=1, tiled=True, to='invarying')   # Out[B_X, F]
```

- Check: exactly equal to `A @ W`, with 3 collective-permutes, 1 all-gather and no all-reduce for XLA to merge into.

**Part 3: Chain the notes' AllGather collective matmul and Part 2's ReduceScatter one into a Transformer block, $\text{In}[B_X, D_Y] \cdot_D W_{in}[D, F_Y] \cdot_F W_{out}[F_Y, D] \to \text{Out}[B_X, D_Y]$. (No multiplying $W_{in} \cdot W_{out}$ first, since a real block has a nonlinearity in between.) How much faster is it than a `jax.jit` version?**

```python
def allgather_matmul(lhs, rhs):
    """A[B_X, D_Y] @ W[D, F_Y] -> Out[B_X, F_Y]. Per device: lhs [B/X, D/Y], rhs [D, F/Y]."""
    n, me = jax.lax.axis_size('Y'), jax.lax.axis_index('Y')
    chunk = lhs.shape[1]
    out = 0
    for i in range(n):
        # After i hops I hold the chunk of A that started on device me + i.
        rows = jax.lax.dynamic_slice_in_dim(rhs, (me + i) % n * chunk, chunk, axis=0)
        out = out + lhs @ rows
        if i < n - 1:
            lhs = shift(lhs, -1)                 # hand my chunk of A on, receive the next one
    return out

def mlp_block_shard(x, w_in, w_out, act=lambda h: h):
    hidden = allgather_matmul(x, w_in)                # In[B_X, D_Y] @ W_in[D, F_Y]   -> [B_X, F_Y]
    return reducescatter_matmul(act(hidden), w_out)   # [B_X, F_Y] @ W_out[F_Y, D] -> Out[B_X, D_Y]

BLOCK_SPECS = (P('X', 'Y'), P(None, 'Y'), P('Y', None))
mlp_block = jax.jit(jax.shard_map(mlp_block_shard, in_specs=BLOCK_SPECS, out_specs=P('X', 'Y')))

@jax.jit
def mlp_block_jit(x, w_in, w_out):                    # same math, comms left to the compiler
    hidden = jnp.einsum('bd,df->bf', x, w_in, out_sharding=P('X', 'Y'))
    return jnp.einsum('bf,fd->bd', hidden, w_out, out_sharding=P('X', 'Y'))
```

- `allgather_matmul` is the notes' algorithm as a loop: after $i$ hops a device holds the chunk of $A$ from device $\text{me} + i$, so it multiplies by that device's $D/Y$ rows of $W_{in}$.
- Its $[B_X, F_Y]$ output is exactly what `reducescatter_matmul` wants, so the block is two ring loops back to back and the hidden activation is never gathered.
- Check: exactly equal to `(In @ W_in) @ W_out`, also with a ReLU in between. The HLO has **6 collective-permutes** and nothing else, versus 1 all-gather and 1 all-reduce (the blocking schedule) for the `jax.jit` version.
- How much faster? By whatever share of the jit block's time goes to waiting on those two collectives.
- Timing the two blocks needs a TPU, so I'll come back to it.

**Question 4 [bidirectional comms]:** Every collective matmul so far sends data one way around the ring. Redo the AllReduce and ReduceScatter ones to communicate in both directions. How much faster are they?

- Every hop in Question 3 uses each ICI link in one direction, but the links are bidirectional, so half the wire sits idle.
- The fix is to split each device's output tile in half and send the two halves' accumulators around the ring in opposite directions: same number of hops, half-size messages. Same setup and `shift` helper as Question 3; each tile needs an even number of columns.

```python
def reducescatter_matmul_bidi(lhs, rhs):
    """L[B_X, K_Y] @ R[K_Y, M] -> Out[B_X, M_Y], with each output tile travelling as two halves."""
    n, me = jax.lax.axis_size('Y'), jax.lax.axis_index('Y')
    half = rhs.shape[1] // n // 2

    def cols(dev, top):                            # one half of the columns owned by device `dev`
        start = (2 * (dev % n) + top) * half
        return jax.lax.dynamic_slice_in_dim(rhs, start, half, axis=1)

    lo = hi = 0
    for i in range(n):
        lo = lo + lhs @ cols(me - i - 1, 0)        # low half, headed for device me - (i + 1)
        hi = hi + lhs @ cols(me + i + 1, 1)        # high half, headed for device me + (i + 1)
        if i < n - 1:
            lo, hi = shift(lo, +1), shift(hi, -1)  # two half-size messages, one per direction
    return jnp.concatenate([lo, hi], axis=1)

def allreduce_matmul_bidi(lhs, rhs):
    """A[B_X, D_Y] @ W[D_Y, F] -> Out[B_X, F]: bidirectional ring, then the same AllGather."""
    tile = reducescatter_matmul_bidi(lhs, rhs)
    return jax.lax.all_gather(tile, 'Y', axis=1, tiled=True, to='invarying')
```

- Travelling toward lower device ids, the accumulator held at step $i$ ends on device $\text{me} + (i + 1)$, mirroring the old index. For the AllReduce only the ring half changes; the trailing AllGather is XLA's own.
- Check: both exactly equal to NumPy on rings of 4 and 8. On the ring of 4 the ReduceScatter version goes from 3 collective-permutes of `f32[4,8]` to 6 of `f32[4,4]`.
- Napkin math: each hop costs the tile's bytes over the one-way link bandwidth, and we just halved the bytes per direction.

$$
T_{\text{comms}} = (\lvert Y \rvert - 1) \cdot \frac{\text{tile bytes}}{W_{\text{ici}}} \quad \to \quad (\lvert Y \rvert - 1) \cdot \frac{\text{tile bytes} / 2}{W_{\text{ici}}}
$$

- So **up to $2\times$** on the communication, ONLY if the hops are what each step waits on. If the matmul already hides them, nothing changes.
- Measuring it needs a TPU, so I'll come back to it.
