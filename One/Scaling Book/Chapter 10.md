---
title: Programming TPUs in JAX
description: JAX Intro & Solutions
published: true
---
## 10. Programming TPUs in JAX

Up to now, sharding has been something we did on paper: write $A[B_X, D_Y]$, count the AllGathers, divide bytes by ICI bandwidth. This chapter is where that notation turns into code. JAX has three ways to run one program across many chips, and they basically differ in who decides where the communication goes: the compiler, JAX, or us.

Everything below is written for JAX 0.10 and checked on 8 fake CPU devices (`jax.config.update('jax_num_cpu_devices', 8)`), so correctness can be tested without a TPU. It says nothing about speed. The three snippets in these notes are one session run top to bottom, so the later ones reuse the imports and arrays from the first.

### The Three Modes

- **Auto (`jax.jit` on a mesh with `Auto` axes):** We write normal single-device code with a global view of every array. If the inputs are sharded, XLA's partitioner (Shardy) picks a sharding for every intermediate and inserts whatever AllGathers, AllReduces and ReduceScatters it thinks it needs. The same function runs on 1 chip or on thousands.
- **Explicit (`jax.jit` on a mesh with `Explicit` axes):** Still a global view and still no hand-written communication, but the sharding is now part of each array's type, and JAX (not XLA) propagates it op by op while tracing. When an op could reasonably go two ways, we get an error and have to say which one we mean. Also called sharding-in-types.
- **Manual (`jax.shard_map`):** A per-device view. The function sees one shard, and nothing moves between devices unless we write the collective ourselves.

| Mode | View | We choose the shardings | We write the collectives |
|---|---|---|---|
| Auto | Global | No | No |
| Explicit | Global | Yes | No |
| Manual | Per-device | Yes | Yes |

So when do we reach for which? I think of auto as the default, since most of a model is elementwise ops and matmuls the compiler handles fine. Explicit is for when we want to know the sharding of every value without digging through a profile. Manual is for the few hot spots where we know a better communication schedule than the compiler (the collective matmul below), or for ops that are just awkward to describe globally. They mix, too: every `shard_map` function here is wrapped in `jax.jit`, and `jax.sharding.auto_axes` and `jax.sharding.explicit_axes` switch between the first two inside one function.

### Mesh and PartitionSpec

The Chapter 3 notation maps straight onto two objects. A mesh is a named grid of devices: `jax.make_mesh((4, 2), ('X', 'Y'))` is our `Mesh({'X': 4, 'Y': 2})`. A `PartitionSpec` (`jax.P`) has one entry per array dimension, naming the mesh axis that dimension is split along:

- $A[B_X, D_Y]$ is `P('X', 'Y')`.
- $W[D_Y, F]$ is `P('Y', None)`. `None` means that dimension isn't split, so $W$ is replicated along `X`.
- $A[I_{XY}, J]$ is `P(('X', 'Y'), None)`.
- A fully replicated $A[I, J]$ is `P()`.

Same rule as before: a mesh axis can only be spent once per array. To get a sharded array we can `jax.device_put` an existing one with a spec, pass a sharding to a creation function, or ask a jitted function for `out_shardings`. One version detail: on JAX 0.10, `make_mesh` gives `Explicit` axes unless told otherwise, so auto mode has to be asked for.

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

Each device holds a `[2, 1024]` piece of $A$ and a `[1024, 8192]` piece of $W$. Notice that $D$ is sharded along `Y` on both sides, which is the Chapter 3 case where a device can only compute a partial sum:

$$
\text{Out}[B_X, F]\{U_Y\} = A[B_X, D_Y] \cdot_D W[D_Y, F]
$$

$$
\text{Out}[B_X, F] = \text{AllReduce}_Y(\text{Out}[B_X, F]\{U_Y\})
$$

We only wrote `A @ W`. If we dump the program compiled for our CPU mesh with `matmul.lower(A, W).compile().as_text()`, there's a local `dot` producing `f32[2,8192]` and then exactly one `all-reduce` whose groups are the device pairs along `Y`. The compiler added the second step for us.

The catch is that the compiler can also add things we don't want, like a giant AllGather sitting in the middle of a profile. In auto mode the main lever is `jax.lax.with_sharding_constraint(x, jax.P(...))`, which pins the sharding of one intermediate and hopes the rest follows. It works, but we're nudging a compiler, not telling it what to do.

### Explicit Mode

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

`jax.typeof` now prints the sharding as part of the type, like `float32[8@X,2048@Y]`, and elementwise ops carry it along. The matmul gets rejected with a `ShardingTypeError`: with $D$ sharded on both sides, JAX can't know if we want $\text{Out}[B_X, F_Y]$ (a ReduceScatter) or $\text{Out}[B_X, F]$ (an AllReduce), so we pick with `out_sharding`. Meaning, explicit mode still writes the collectives for us. It just refuses to guess the layout.

### Manual Mode

```python
@jax.jit
@jax.shard_map(in_specs=(jax.P('X', 'Y'), jax.P('Y', None)), out_specs=jax.P('X', None))
def matmul_by_hand(a, w):                  # per device: a is [B/4, D/2], w is [D/2, F]
    partial_sum = a @ w                    # Out[B_X, F] {U_Y}, a sum over half of D
    return jax.lax.psum(partial_sum, 'Y')  # AllReduce over Y finishes the sum

out = matmul_by_hand(A, W)
print(jax.typeof(out))                     # float32[8@X,8192]
```

Inside `shard_map`, `a` is `[2, 1024]` and `w` is `[1024, 8192]`: the body is what ONE device runs. `in_specs` says how the global inputs get cut up, and `out_specs` says how the per-device outputs get stitched back together. `a @ w` is the unreduced partial sum and `psum` is the AllReduce, so it's the same two steps as above, except now we wrote both. JAX tracks this in the types as well: before the `psum` a value is tagged `{V:(X,Y)}` (it varies across both mesh axes), after it `{V:X}`, and that's what makes `out_specs=P('X', None)` legal.

Every Chapter 3 primitive has a call. For a local `[2, 1024]` shard on this mesh:

| Primitive | Inside `shard_map` | Result on each device |
|---|---|---|
| AllReduce | `jax.lax.psum(x, 'Y')`, also `pmean`, `pmax` | `[2, 1024]`, identical along `Y` |
| AllGather | `jax.lax.all_gather(x, 'Y', axis=1, tiled=True)` | `[2, 2048]` |
| ReduceScatter | `jax.lax.psum_scatter(x, 'Y', scatter_dimension=1, tiled=True)` | `[2, 512]` |
| AllToAll | `jax.lax.all_to_all(x, 'X', 1, 1, tiled=True)` | `[2, 1024]`, pieces swapped between devices |
| One hop | `jax.lax.ppermute(x, 'Y', perm)` | `[2, 1024]`, a neighbour's shard |

`jax.lax.axis_index('Y')` and `jax.lax.axis_size('Y')` tell a device where it sits on an axis.

Why go manual at all? Some things are one line per device and a headache globally. Say we want the first 4 elements of every shard, averaged across the mesh. That's `jax.lax.pmean(x[:4], ...)` in the local view, versus describing a strided slice of the global array and trusting XLA to work out that almost nothing needs to move.

### Collective Matmul

The realistic use case is model parallelism: $A[B_X, D_Y] \cdot_D W[D, F_Y] \to \text{Out}[B_X, F_Y]$. The Chapter 3 recipe is to AllGather $A$ along `Y` and then multiply locally. Problem is, that's sequential. While the AllGather is in flight, the MXU does nothing.

There's a cool trick (Wang et al. 2023): a device doesn't need all of $A$ to start. Its local chunk of $A$ times the matching $D/Y$ rows of $W$ is already one valid term of the final sum. So:

1. Multiply the chunk we hold by its rows of $W$ and add the result to an accumulator.
2. At the same time, pass that chunk to the next device on the `Y` ring and receive the previous device's chunk.
3. Repeat. After $\lvert Y \rvert - 1$ hops every device has seen every chunk.

Same bytes as the AllGather, they just move while the matmul runs instead of before it. No device ever holds the gathered $A[B_X, D]$ either, only one chunk at a time. In code it's a loop of `ppermute` plus a small matmul, and there's a version of it in Question 3.

The reference profiles on [jax-ml.github.io/scaling-book/jax-stuff/](https://jax-ml.github.io/scaling-book/jax-stuff/) put the plain jit matmul with the blocking AllGather at 311 μs, the collective version at 244 μs, and the same matmul with no sharded contracting dimension at 224 μs. So the overlapped version lands close to the no-communication baseline.

> **Note:** On JAX 0.10, the listings on that page that call `jax.make_mesh(axis_sizes=...)` stop with a `TypeError`. The keyword is `axis_shapes` now (or just pass the sizes positionally), and that one rename is all they need.

### Takeaway

Auto mode is where we start: write the model once, shard the inputs, and let XLA place the collectives. Explicit mode moves the sharding into the types so the layout is never a surprise. `shard_map` is the escape hatch for the few places where we know the schedule better than the compiler, and it's the only mode where an overlap trick like the collective matmul can be written down at all. Whichever mode we're in, the mental model is still Chapter 3: look at which dimensions are sharded, and we know which collective has to show up.

## Worked Problems

All of these assume $N$ devices, here $N = 8$ fake CPU devices on JAX 0.10. Every code block is checked against a plain NumPy reference, collective counts come from the HLO compiled for this CPU mesh (a TPU compile may differ), and timings are left as TODOs.

**Question 1 [per-shard ops]:** $A$ is an array of activations of shape $\text{float32}[S_X, D_Y]$ with $X \cdot Y = N$. The test uses `Mesh({'X': 4, 'Y': 2})`, $S = 512$ and $D = 64$, so each shard is `float32[128, 32]`.

**Part 1: Compute the mean of each $(X, Y)$ shard as an $[X, Y]$ array, where `arr[i, j]` is the mean of shard $(i, j)$. Do it with `jax.jit` and with `shard_map`, profile both, and check whether any communication got added. (Hint: none should be, but XLA sometimes adds some anyway.)**

- Cleanest case for the per-device view: a shard's mean only needs that shard.

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

- With `shard_map` the body is one line. Each device returns a `[1, 1]` array, and `out_specs=P('X', 'Y')` stitches those into the $[X, Y]$ grid.
- The `jax.jit` version has the global view, so we rebuild the shards with a 4D reshape, average the two within-shard axes, and hope XLA notices every block is already local.
- Check: both return a `[4, 2]` array equal to the NumPy block means.
- Was communication added? **No.** Zero collectives in either version's HLO on the 8-device CPU mesh.
- **TODO(eddie):** profile both versions on a real TPU slice for timings, and confirm the TPU compile of the `jax.jit` version adds no communication either.

**Part 2: Using only `shard_map`, return `roll(x, shift, axis=0) - x` for some shift, where the roll happens inside each shard along `X`.**

- "Inside each shard" means rows wrap within their own block of $S/X$ rows, not the whole array. Per device that's just `jnp.roll` on the local shard.

```python
def shard_roll_diff(a, shift):
    local = lambda s: jnp.roll(s, shift, axis=0) - s    # s is one shard, rows wrap inside it
    return jax.jit(jax.shard_map(local, in_specs=SPEC, out_specs=SPEC))(a)
```

- Check: equals `np.roll` along axis 1 of `x.reshape(X, S // X, D)`, minus the array, for shifts of 1, 5, -3 and 200, and does NOT equal the global roll, which is the point. Zero collectives again.

**Question 2 [mixture of experts]:** $W$: $\text{float32}[E_X, D, F]$ holds $E$ expert matrices, $A$: $\text{float32}[S_X, D]$ the activations, and $B$: $\text{int32}[S_X]$ the routing, where `B[i]` is an integer in $[0, E)$ picking the expert for token $i$. We want `Out[i] = A[i] @ W[B[i]]`. The test uses `Mesh({'X': 8})`, $E = 16$, $S = 64$, $D = 8$, $F = 12$, so each device holds 2 experts and 8 tokens.

**Part 1: Ignoring sharding (small tensors, one device), write a local version that never builds an array of shape $[S, D, F]$. Hints: either loop over experts, multiplying ALL of $A$ by each `W[e]` and masking out tokens that didn't choose $e$, or sort tokens by expert into a padded $[E, S, D]$ buffer, do one batched matmul and mask the padding (why does the second dimension have to be $S$?). `jax.lax.ragged_dot` is the sorted version with no padding.**

- We're told not to build $[S, D, F]$, which is exactly what the obvious `einsum('sd,sdf->sf', A, W[B])` does. So we sort tokens by expert id, `ragged_dot` each contiguous group against its own expert, and undo the sort.

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

- Ids outside $[0, E)$ sort to the end, belong to no group and come back as zero rows. The sharded versions below lean on this.
- The mesh is `Auto` on purpose: on JAX 0.10, `ragged_dot` under an `Explicit` mesh raises a `NotImplementedError` without an `out_sharding`, and Part 2 sorts along a sharded dimension, which explicit mode refuses.
- The hint's question: shapes are static under jit and all $S$ tokens could choose the same expert, so each expert's row of the padded buffer needs room for all of them.
- Check: exact match with the per-token einsum (which does build $[S, D, F]$, fine at test size) for four routings: uniform, all to expert 0, all to the last expert, and two hot experts. Exact because the inputs are small integers in float32.
- The solution takes the easy route instead, a `scan` over experts with a mask, at $E$ times the FLOPs.

> **Note:** On CPU, `ragged_dot` lowers to the masked version anyway: the StableHLO has a `[16, 64, 8]` buffer, which is $[E, S, D]$. The native ragged instruction is TPU only.

**Part 2: Now just `jax.jit` the local function on sharded inputs. Profile it: what communication did the compiler pick, and how long does it take?**

```python
W = jax.device_put(W_np, P('X'))     # W[E_X, D, F]
A = jax.device_put(A_np, P('X'))     # A[S_X, D]
B = jax.device_put(B_np, P('X'))     # B[S_X]
moe_jit = jax.jit(moe_local, out_shardings=P('X'))
```

- Same function, inputs sharded along `X` (`W_np`, `A_np`, `B_np` are the NumPy test arrays), output sharded the same way. Still correct.
- What XLA did to get there: **2 all-gathers and 3 all-reduces**. One all-gather carries `s32[64]`, all of $B$. Two all-reduces carry `f32[64,8]` and `f32[64,12]`, the ENTIRE activations $[S, D]$ and output $[S, F]$, on every device.
- Meaning, a global sort turns a problem that's local to each expert into "ship everything everywhere". That's the blow-up Part 3 is about.
- **TODO(eddie):** benchmark on a real TPU slice: time this and check whether the TPU compile picks the same collectives. Also confirm `moe_local` still returns zero rows for out-of-range ids with the native `ragged_dot` instruction (only checked on the CPU lowering, and 3.1 depends on it).

**Part 3: The jit version probably ends up with all the activations on every device, an $\text{AllGather}_X([S_X, D])$, which costs communication, and memory too if they don't fit on one chip. Redo it with `shard_map` and explicit communication.**

**3.1: First pass, the easy way: a `jax.lax.all_gather` and the same reordering as Part 1.**

- Each device owns $E/X$ experts. It gathers all of $A$ and $B$ and runs `moe_local` on just its own experts: subtracting its first expert's id pushes every token it doesn't own outside $[0, E/X)$, so those become zero rows.
- Every token has exactly one owner, so the per-device results sum to the answer. `psum_scatter` does that sum and hands each device its own $S/X$ rows.

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

- Check: same four routings, exact. The HLO has an all-gather of `f32[64,8]`, one of `s32[64]` and a reduce-scatter of `f32[8,12]`, exactly what we wrote.
- Still wasteful though. Every device receives all $S$ tokens and mostly runs its experts on somebody else's.

**3.2: Second pass: don't materialize anything of size $[E, S, D]$. Work raggedly with a `jax.lax.all_to_all` inside a `jax.lax.while_loop`, so the full activations never land on one device and no compute is wasted on padding. How much faster is it than the original?**

- Each token should go ONLY to the device that owns its expert. That's an AllToAll, the MoE primitive from Chapter 3.
- Static shapes again: in the worst case all of a device's tokens go to one place. So we send in rounds of at most `chunk` tokens per (source, destination) pair until nobody has anything left.

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

- Each device sorts its tokens by destination. A round builds a `[n, chunk, D]` send buffer where row $j$ is the next `chunk` tokens bound for device $j$, padded with expert id -1.
- Two `all_to_all`s deliver the tokens and their ids, the receiver runs `moe_local` on its `n * chunk` rows (padding becomes zero rows), and a third sends the results home. The loop condition is a `psum` of the unsent tokens, so every device stops together.
- Check: exact on the four routings for chunk sizes 1, 2, 4 and 8. The round count is the busiest (source, destination) pair divided by `chunk`, rounded up: 8 rounds when all 8 of a device's tokens go to expert 0 with a chunk of 1, and 1 round with a chunk of 8.
- The HLO has three all-to-alls in the loop, two scalar all-reduces (the stop condition and the round counter's `pmax`) and no all-gather. At a chunk of 2 each token buffer is `f32[1,2,8]` per destination, never $[S, D]$.
- How much faster? From the shapes: with balanced routing each device receives about $S/X$ tokens instead of all $S$, roughly $X$ times less data, and the only waste is padding inside a round. With skewed routing a hot device still gets everything, just over more rounds.
- **TODO(eddie):** benchmark on a real TPU slice against Part 2 and 3.1, with `chunk` near the balanced load of $S/X^2$ tokens per pair.

**Part 4: Real MoEs usually send each token to $k$ experts and average the results. Refactor for that, with $B$: $\text{int32}[S_X, k]$ listing each token's $k$ experts.**

- A token routed to $k$ experts is just $k$ (token, expert) pairs. So we repeat every activation $k$ times, flatten $B$, run any one-expert routine unchanged, and average each token's $k$ rows.

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

- The wrapper only touches local rows, so it drops straight into a `shard_map` body and the communication doesn't change.
- Check: for $k = 1, 2, 3$ with distinct experts per token, the local, `all_gather` and `all_to_all` versions all match the mean of the per-token einsums over experts `B[i, j]`, to float32 tolerance.
- The dispatch now moves $k$ copies of each activation. That's unavoidable: every token really goes to $k$ experts, possibly on $k$ different devices.

**Question 3 [collective matmuls]:** The notes covered the AllGather collective matmul. Build the other two flavours and chain them into the MLP of a Transformer. The test uses `Mesh({'X': 2, 'Y': 4})` with $B = 8$, $D = 32$, $F = 64$.

**Part 1: Start with an AllReduce collective matmul, $A[B_X, D_Y] \cdot_D W[D_Y, F] \to \text{Out}[B_X, F]$, with the output replicated across `Y`. The naive version is a local matmul then an AllReduce. Write one that overlaps the communication with the compute. Hint: tile over the output dimension; `jax.lax.psum` is allowed. Note: with the way XLA treats this, it may not beat the baseline.**

- The naive version is `matmul_by_hand` from the notes: the whole $[B/X, F]$ partial sum has to exist before any of it is sent. Tiling the output lets each tile go out while the next is multiplied.

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

- This is a per-device body. It runs under `jax.shard_map` with `in_specs=(P('X', 'Y'), P('Y', None))` and `out_specs=P('X', None)`, like the other bodies in this question (with `out_specs=P('X', 'Y')` when the output stays sharded).
- Check: exactly equal to NumPy's `A @ W`.
- But the HLO has **ONE all-reduce**, same as the naive version. XLA merged our four small `psum`s into a single all-reduce carrying all four `f32[4,16]` tiles.
- That's the note in the question playing out. A version XLA can't undo needs the ring from Part 2, so we'll come back to it.

**Part 2: The other half is a ReduceScatter collective matmul, $\text{Tmp}[B_X, F_Y] \cdot_F W_2[F_Y, D] \to \text{Out}[B_X, D_Y]$, the down-projection in a Transformer. Write an overlapped version that passes only as much data as is needed. Hint: permute the result while accumulating it.**

- The naive version builds the full $[B/X, D]$ partial product and then calls `psum_scatter`, even though each device only keeps $D/Y$ of those columns.
- Following the hint, an accumulator for ONE device's columns, shape $[B/X, D/Y]$, travels around the ring instead, picking up each device's contribution. There are $\lvert Y \rvert$ of these in flight at once, one per destination.

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

- The only real thinking is the index. The accumulator held at step $i$ has $\lvert Y \rvert - 1 - i$ hops left, so it ends on device $\text{me} - (i + 1)$, and that device's block of $W_2$'s columns is what we slice. On the last step that's our own block, and the accumulator is home.
- Minimal data: every hop carries $[B/X, D/Y]$ (`f32[4,8]` in the test), never the full $[B/X, D]$. The HLO has 3 collective-permutes for $\lvert Y \rvert = 4$ and no reduce-scatter or all-reduce.
- The overlap: step $i + 1$'s matmul doesn't depend on step $i$'s permute, so the hardware can run them at the same time.
- Check: exactly equal to NumPy's `Tmp @ W2`.
- This follows the pattern in the [JAX `shard_map` notebook](https://docs.jax.dev/en/latest/notebooks/shard_map.html). Its `matmul_psumscatter_overlapped` passes the accumulator around the ring the same way, just tiled over the rows of the left operand instead of the columns of $W_2$.

Back to Part 1. From Chapter 3, $\text{AllReduce} = \text{ReduceScatter} + \text{AllGather}$, and we just built an overlapped ReduceScatter:

```python
def allreduce_matmul(lhs, rhs):
    """A[B_X, D_Y] @ W[D_Y, F] -> Out[B_X, F], with the reduce half riding the ring."""
    tile = reducescatter_matmul(lhs, rhs)                                      # Out[B_X, F_Y]
    return jax.lax.all_gather(tile, 'Y', axis=1, tiled=True, to='invarying')   # Out[B_X, F]
```

- The reduce half rides the ring under the matmul. The gather half is a plain AllGather of the finished tiles.
- Check: exactly equal to `A @ W`. The HLO has 3 collective-permutes and 1 all-gather, and this time no all-reduce for XLA to merge things back into.

**Part 3: Combine the AllGather collective matmul from the notes with Part 2's ReduceScatter one into an end-to-end Transformer block, $\text{In}[B_X, D_Y] \cdot_D W_{in}[D, F_Y] \cdot_F W_{out}[F_Y, D] \to \text{Out}[B_X, D_Y]$, with overlapped communication. ($W_{in} \cdot W_{out}$ can't be multiplied first, since a real block has a nonlinearity in between.) How much faster is it than a `jax.jit` implementation?**

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

- `allgather_matmul` is the algorithm from the notes as a plain Python loop. After $i$ hops a device holds the chunk of $A$ that started on device $\text{me} + i$, so it multiplies by that device's $D/Y$ rows of $W_{in}$.
- Its $[B_X, F_Y]$ output is exactly the input sharding `reducescatter_matmul` wants. So the block is just the two ring loops back to back, and the hidden activation is never gathered.
- Check: exactly equal to `(In @ W_in) @ W_out`, also with a ReLU between the two matmuls. The HLO has **6 collective-permutes** and nothing else.
- The explicit-mode `jax.jit` version gives the same numbers with 1 all-gather, 1 all-reduce and no permutes, which is the blocking schedule.
- How much faster? In theory, by whatever share of the jit block's time goes to waiting on those two blocking collectives. The only measured numbers are still the reference profiles' 311 μs versus 244 μs for the single matmul, from the notes.
- **TODO(eddie):** benchmark on a real TPU slice: the collective block versus the jit block, and the naive, tiled and ring AllReduce matmuls from Part 1.

**Question 4 [bidirectional comms]:** Every collective matmul so far sends data around the ring one way only. Redo the AllReduce and ReduceScatter collective matmuls so they communicate in both directions. How much faster are they?

- Every hop in Question 3 uses each ICI link in one direction. The links are bidirectional (the one-way and bidi columns in the Chapter 2 tables), so half the wire sits idle.
- The fix is to split each device's output tile in half and send the two halves' accumulators around the ring in opposite directions. Same number of hops, half-size messages, and both directions of every link are busy.
- Same imports, mesh and `shift` helper as Question 3. Each device's tile needs an even number of columns so it can be cut in two.

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

- The new direction's index mirrors the old one. Travelling toward lower device ids, the accumulator we hold at step $i$ ends on device $\text{me} + (i + 1)$.
- For the AllReduce version only the ring half changes. The trailing AllGather is XLA's own collective, so its routing isn't ours to write.
- Check: both exactly equal to NumPy on a ring of 4 (`Mesh({'X': 2, 'Y': 4})`) and a ring of 8 (`Mesh({'X': 1, 'Y': 8})`). On the ring of 4 the ReduceScatter version goes from 3 collective-permutes of `f32[4,8]` to 6 of `f32[4,4]`: half the bytes per message, one message per direction.
- Napkin math for the speedup: each hop costs the tile's bytes over the one-way link bandwidth, and we just halved the bytes per direction.

$$
T_{\text{comms}} = (\lvert Y \rvert - 1) \cdot \frac{\text{tile bytes}}{W_{\text{ici}}} \quad \to \quad (\lvert Y \rvert - 1) \cdot \frac{\text{tile bytes} / 2}{W_{\text{ici}}}
$$

- So **up to $2\times$** on the communication, and ONLY if the hops are what each step is waiting on. If the matmul already hides the hop, nothing changes.
- The JAX notebook's bidirectional ring matmuls use the same cut-the-block-in-half idea (`matmul_psumscatter_overlapped_bidi` is the closest to ours).
- **TODO(eddie):** benchmark on a real TPU slice, one-way versus bidirectional for both matmuls.
