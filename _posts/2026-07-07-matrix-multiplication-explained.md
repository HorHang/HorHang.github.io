---
layout: post
title: "Triton Matrix Multiplication, Explained: A Tutorial Companion"
description: "A plain-English walkthrough of Triton's matmul kernel: grouped tile ordering lifts an A100 from 220 to 245 TFLOPS, plus pointer arithmetic you can draw by hand."
coverImage: "03-matrix-multiplication-cover.svg"
coverImageAlt: "A 9x9 grid of output tiles for a blocked matrix multiplication, with one tile highlighted at the crossing of a row-strip of A and a column-strip of B."
ogImage: "03-matrix-multiplication-cover.svg"
ogTitle: "Triton Matrix Multiplication, Explained: A Tutorial Companion"
ogDescription: "Grouped tile ordering lifts an A100 from 220 to 245 TFLOPS with no math changed. A plain-English walkthrough of Triton's matmul kernel."
twitterCard: "summary_large_image"
date: "2026-07-07"
lastUpdated: "2026-07-07"
author: "Hang Hor"
authorBio: "Hang Hor is a senior data scientist who writes GPU-kernel and deep-learning explainers while working through the llm.c course in Triton and CUDA."
tags: ["triton", "matrix-multiplication", "gpu-kernels", "cuda", "deep-learning"]
---

The official [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) fits a cuBLAS-competitive kernel into about forty lines. That density is the problem: every line is doing something, and the tutorial trusts you to unpack it. This companion piece slows down the two hardest parts, the program-ID remapping and the pointer arithmetic, using a sketch you can draw on paper before you write a single line.

Why bother, when `torch.matmul` already calls cuBLAS? Because the moment you want a *fused* matmul (add a bias, a GELU, a mask), the vendor library stops helping and you have to write the kernel yourself. Triton is how you do that without dropping to CUDA C++.

> **Key Takeaways**
> - Grouped tile ordering raises A100 throughput from ~220 to ~245 TFLOPS, a >10% win from *scheduling alone* with no math changed ([Triton docs](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)).
> - One Triton program computes a whole `BLOCK_M x BLOCK_N` **tile** of C, not one element. It runs a block-by-block matmul internally.
> - The entire pointer-arithmetic section becomes obvious once you draw the "matmul corner": C's tile sits where a row-strip of A crosses a column-strip of B.
> - Triton was designed to hit vendor-library performance from portable Python-like code ([Tillet et al., MAPL 2019](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)).

## What does one Triton program actually compute?

One Triton program instance computes exactly one `BLOCK_SIZE_M x BLOCK_SIZE_N` **tile** of the output C, many elements at once, not a single scalar. To produce that tile it multiplies a horizontal strip of A (`BLOCK_M` rows by the full `K`) against a vertical strip of B (full `K` by `BLOCK_N` columns), accumulating in the loop.

This is the first place beginners stumble. The scalar rule you learned in school, `C[i,j] = sum over k of A[i,k]*B[k,j]`, still runs, but it runs *hundreds of times inside a single program*, buried in one `tl.dot` call. There are three nested units, and collapsing them is what causes confusion:

| Unit | What it is | Who owns it |
|---|---|---|
| element | one number `C[i,j]` | hardware lanes inside a tile |
| tile / block | a `BLOCK_M x BLOCK_N` chunk of C | **one program (`pid`)** |
| grid | all the tiles together | all programs |

<!-- [UNIQUE INSIGHT] -->
The tutorial's famous figure draws C as a **9x9 grid of squares**, and every square is a *block*, never an element. That distinction trips people up because the runnable demo cells conflate `M` (matrix height in elements) with `num_pid_m` (height in blocks). Set `M=9` with `BLOCK_SIZE_M=3` and you get a 3x3 grid, not the 9x9 one in the picture. You actually need `M=27`. Keeping "elements vs blocks" straight is half the battle.

According to the tutorial's own blocked pseudocode, "each iteration of the doubly-nested for-loop is performed by a dedicated Triton program instance" ([Triton docs](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html), 2026). That single sentence is the whole execution model: you write the body for one tile, and Triton launches a grid of them.

## Why does the launch order change performance?

Because the order in which tiles run decides what stays in L2 cache. Reordering tiles into groups lifts an A100 from roughly 220 to 245 TFLOPS on the tutorial's benchmark, over 10% faster with identical arithmetic ([Triton docs](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html), 2026). The math is untouched; only the *schedule* changed.

Here is the intuition, counted in block-loads. Suppose you want to finish the first 9 output tiles of a 9x9 grid.

- **Row-major order:** those 9 tiles form one full *row* of C. Computing them touches 9 row-blocks of A but **all 81** column-blocks of B, for 90 block-loads.
- **Grouped order:** pack the same 9 tiles into a 3x3 super-tile. Now you need only 3 rows of A (27) plus 3 columns of B (27), for **54 block-loads**.

<figure>
<svg viewBox="0 0 560 220" role="img" aria-label="Bar chart: row-major ordering needs 90 block loads, grouped ordering needs 54, for the same 9 output tiles." xmlns="http://www.w3.org/2000/svg">
  <text x="10" y="24" font-family="sans-serif" font-size="15" font-weight="700" fill="#e07b39">Block-loads to produce the first 9 output tiles</text>
  <g font-family="sans-serif" font-size="13" fill="#8a94a6">
    <text x="10" y="80">Row-major</text>
    <rect x="120" y="66" width="400" height="24" rx="4" fill="#c44e4e"/>
    <text x="528" y="83" text-anchor="end" fill="#fff" font-weight="700">90</text>
    <text x="10" y="140">Grouped</text>
    <rect x="120" y="126" width="240" height="24" rx="4" fill="#3f9d6b"/>
    <text x="368" y="143" font-weight="700">54</text>
  </g>
  <text x="10" y="200" font-family="sans-serif" font-size="12" fill="#8a94a6" opacity="0.7">Source: Triton official tutorial (03-matrix-multiplication), 2026. 40% fewer loads, same 9 outputs.</text>
</svg>
<figcaption>Grouping trades no arithmetic for 40% less memory traffic on the first super-tile.</figcaption>
</figure>

Same nine results, 54 versus 90 loads, because the three loaded rows of A and three columns of B get *reused* across every tile in the super-tile while they are still hot in L2. This is called program-ID swizzling: a column-major remapping so that simultaneously active SMs work on spatially adjacent tiles ([GPU MODE Lecture 14 notes](https://christianjmills.com/posts/cuda-mode-notes/lecture-014/), 2024).

## How do you map a flat program ID to a tile?

Triton launches a 1-D grid, so each program gets a flat integer `pid`, and the kernel must convert it into a 2-D tile coordinate `(pid_m, pid_n)` using the grouped order. The whole eight-line block answers one question, *given a flat `pid`, which tile is mine?*, and you can reconstruct it from a single idea instead of memorizing it.

**The one idea:** row-major is `pid_m = pid // num_pid_n; pid_n = pid % num_pid_n`. Grouping inserts a "band of rows" so consecutive `pid`s march *down a column* before stepping right. That one flip, where the row becomes the fast-varying axis, is the entire optimization.

```python
pid = tl.program_id(axis=0)
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)             # blocks tall
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)             # blocks wide
num_pid_in_group = GROUP_SIZE_M * num_pid_n      # one band = GROUP_SIZE_M full rows
group_id = pid // num_pid_in_group               # which band am I in?
first_pid_m = group_id * GROUP_SIZE_M            # band's starting row
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)   # last band may be short
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)  # row = fast axis (%)
pid_n = (pid % num_pid_in_group) // group_size_m                 # col = slow axis (//)
```

<!-- [PERSONAL EXPERIENCE] -->
The mnemonic that made this stick for me: **in any flattening, `%` gives the fast-varying axis and `//` gives the slow one, always.** Grouped ordering is "down then right," so the row uses `%` and the column uses `//`. If your sanity check prints the same row three times across columns instead of walking down, you flipped the two operators. Trace `pid = 30` on a 9x9 grid with `GROUP_SIZE_M = 3` and you should land on `C[row 3, col 1]`.

The `min(...)` line is the only edge case: when `num_pid_m` is not a multiple of `GROUP_SIZE_M`, the last band is shorter, and clamping the height keeps the inner fold correct. Everything else is forced by the geometry, so there is nothing to guess.

## How does the pointer arithmetic work? Draw the corner

Draw three rectangles: A on the left, B on top, C at the bottom-right corner. The tile of C you own sits exactly at the **crossing** of A's horizontal row-strip and B's vertical column-strip. Once that picture is on paper, every variable in the kernel is just a label on it.

<figure>
<svg viewBox="0 0 620 610" role="img" aria-label="Blocked matrix multiplication layout: A matrix bottom-left (M by K), B matrix top-right (K by N), C matrix bottom-right (M by N). A highlighted Mtile-by-Ktile block of A and a Ktile-by-Ntile block of B combine to produce the Mtile-by-Ntile output block of C at their crossing." xmlns="http://www.w3.org/2000/svg">
  <line x1="290" y1="520" x2="430" y2="520" stroke="#8a94a6" stroke-width="1" stroke-dasharray="4 4" opacity="0.5"/>
  <line x1="452" y1="260" x2="452" y2="498" stroke="#8a94a6" stroke-width="1" stroke-dasharray="4 4" opacity="0.5"/>
  <rect x="340" y="30" width="230" height="230" fill="none" stroke="#8a94a6" stroke-width="1.5"/>
  <rect x="430" y="30" width="44" height="230" fill="#e9c94a" fill-opacity="0.30"/>
  <rect x="430" y="150" width="44" height="44" fill="#d4a017" fill-opacity="0.85"/>
  <rect x="60" y="340" width="230" height="230" fill="none" stroke="#8a94a6" stroke-width="1.5"/>
  <rect x="60" y="498" width="230" height="44" fill="#6aa9e9" fill-opacity="0.30"/>
  <rect x="176" y="498" width="44" height="44" fill="#2f6fc0" fill-opacity="0.85"/>
  <rect x="340" y="340" width="230" height="230" fill="none" stroke="#8a94a6" stroke-width="1.5"/>
  <rect x="430" y="498" width="44" height="44" fill="#3f9d6b" fill-opacity="0.75"/>
  <path d="M46 340 h-6 v230 h6" fill="none" stroke="#8a94a6" stroke-width="1" opacity="0.6"/>
  <path d="M60 324 v-6 h230 v6" fill="none" stroke="#8a94a6" stroke-width="1" opacity="0.6"/>
  <path d="M330 30 h-6 v230 h6" fill="none" stroke="#8a94a6" stroke-width="1" opacity="0.6"/>
  <path d="M340 22 v-6 h230 v6" fill="none" stroke="#8a94a6" stroke-width="1" opacity="0.6"/>
  <path d="M340 584 v6 h230 v-6" fill="none" stroke="#8a94a6" stroke-width="1" opacity="0.6"/>
  <g font-family="sans-serif" fill="#8a94a6">
    <text x="562" y="50" text-anchor="end" font-size="15" font-style="italic">B matrix</text>
    <text x="282" y="360" text-anchor="end" font-size="15" font-style="italic">A matrix</text>
    <text x="562" y="360" text-anchor="end" font-size="15" font-style="italic">C matrix</text>
    <text x="352" y="492" font-size="12">Block(m,n)</text>
    <text x="34" y="460" text-anchor="middle" font-size="15">M</text>
    <text x="175" y="316" text-anchor="middle" font-size="15">K</text>
    <text x="318" y="150" text-anchor="middle" font-size="15">K</text>
    <text x="455" y="18" text-anchor="middle" font-size="15">N</text>
    <text x="455" y="600" text-anchor="middle" font-size="15">N</text>
    <text x="236" y="524" font-size="12">Mtile</text>
    <text x="176" y="562" font-size="12">Ktile</text>
    <text x="484" y="176" font-size="12">Ktile</text>
    <text x="430" y="214" font-size="12">Ntile</text>
    <text x="484" y="524" font-size="12">Mtile</text>
  </g>
</svg>
<figcaption>One program computes the green <em>Block(m,n)</em> of C: it lives exactly where A's Mtile-tall row-strip crosses B's Ntile-wide column-strip. The kernel slides the dark Ktile sub-blocks along K, accumulating their products in fp32.</figcaption>
</figure>

That drawing hands you the three index vectors directly: `offs_am` (my rows of A), `offs_bn` (my cols of B), and `offs_k` (the shared depth). The flat address of any element is `base + row_index * row_stride + col_index * col_stride`, so a whole 2-D block of pointers is one broadcast:

```python
offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
offs_k  = tl.arange(0, BLOCK_SIZE_K)
a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k [None, :] * stride_ak)  # a is [M, K]
b_ptrs = b_ptr + (offs_k [:, None] * stride_bk + offs_bn[None, :] * stride_bn)  # b is [K, N]
```

The rule that removes all guesswork: `[:, None]` indexes the **rows** of the block, `[None, :]` indexes the **cols**, and you pair each stride to the block's logical shape. Because A is `[M, K]` and B is `[K, N]`, the shared K axis is A's *columns* but B's *rows*, which is exactly how `A[i,k]·B[k,j]` lines up. As one community walkthrough puts it, block pointers let you "describe the shape and strides of a tile once and let the compiler handle the addressing" ([Triton Exercises: Block Pointers](https://lweitkamp.github.io/triton_exercises/introduction/block_pointers.html), 2024).

## Why accumulate in a loop over K?

Because K is usually far too wide to load at once, so the kernel walks it in `BLOCK_SIZE_K` chunks and sums the partial products into an fp32 accumulator. The strips slide; the tile stays put.

```python
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
    accumulator = tl.dot(a, b, accumulator)      # acc += a @ b
    a_ptrs += BLOCK_SIZE_K * stride_ak           # A slides RIGHT along K
    b_ptrs += BLOCK_SIZE_K * stride_bk           # B slides DOWN along K
```

Look back at the corner sketch: A's block walks right and B's block walks down, both stepping one `BLOCK_SIZE_K` along the shared axis. Two accuracy details matter here and both are easy to get wrong. First, accumulate in **fp32** even with fp16 inputs, because summing many half-precision products loses bits fast. Second, the `mask=... other=0.0` handles a ragged K tail: out-of-bounds elements load as zero and contribute nothing. When you finally cast down and store, you mask again against the real matrix bounds, because writing garbage past the edge would corrupt neighboring output.

According to Triton's design paper, this tile-level abstraction is what lets a short kernel reach hand-tuned library performance: the compiler owns the intra-tile scheduling, memory coalescing, and register allocation that you would otherwise write by hand in CUDA ([Tillet, Kung & Cox, MAPL 2019](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)). That is the trade: you describe tiles, the compiler sweats the details.

## How does autotuning close the gap to cuBLAS?

Triton benchmarks a list of block-size and scheduling configs on the first call and caches the winner for each input shape. The `@triton.autotune` decorator sweeps combinations of `BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`, `num_warps`, and `num_stages`, then reuses the fastest one keyed on `M, N, K`.

Two knobs deserve a plain-English gloss:

- **`num_warps`**: how many warps cooperate on one tile. `num_warps=8` means 8 × 32 = 256 threads per program ([triton.Config docs](https://triton-lang.org/main/python-api/generated/triton.Config.html), 2026).
- **`num_stages`**: software-pipelining depth, which overlaps the next chunk's memory loads with the current chunk's `tl.dot`. It matters most for matmul on SM80+ GPUs like the A100.

The reason this whole exercise pays off is that matmul is **compute-bound**: it does `2*M*N*K` floating-point operations but touches far fewer bytes, so once tiling keeps the compute units fed, you approach the hardware's peak. Simon Boehm's celebrated CUDA worklog makes the same journey by hand and lands within roughly 95% of cuBLAS after about ten iterative kernels ([siboehm.com](https://siboehm.com/articles/22/CUDA-MMM), 2022). Triton compresses those ten kernels into autotuning plus a tile description.

<figure>
<svg viewBox="0 0 560 220" role="img" aria-label="Bar chart: A100 throughput rises from 220 TFLOPS with naive ordering to 245 TFLOPS with grouped ordering." xmlns="http://www.w3.org/2000/svg">
  <text x="10" y="24" font-family="sans-serif" font-size="15" font-weight="700" fill="#e07b39">A100 throughput: naive vs grouped tile order</text>
  <g font-family="sans-serif" font-size="13" fill="#8a94a6">
    <text x="10" y="80">Naive</text>
    <rect x="120" y="66" width="352" height="24" rx="4" fill="#8892b0"/>
    <text x="480" y="83" font-weight="700">220 TFLOPS</text>
    <text x="10" y="140">Grouped</text>
    <rect x="120" y="126" width="392" height="24" rx="4" fill="#3f9d6b"/>
    <text x="520" y="143" text-anchor="end" fill="#fff" font-weight="700">245</text>
  </g>
  <text x="10" y="200" font-family="sans-serif" font-size="12" fill="#8a94a6" opacity="0.7">Source: Triton official tutorial, 2026. Same kernel math; only the tile schedule differs.</text>
</svg>
<figcaption>A >10% throughput gain from scheduling alone, before any change to the arithmetic.</figcaption>
</figure>

## Frequently Asked Questions

### Does one Triton program compute a single element of C?

No. Each program computes a full `BLOCK_SIZE_M x BLOCK_SIZE_N` tile of C. The scalar multiply-adds still happen, but many of them run inside one `tl.dot`, executed by the warps assigned to that program. On the tutorial's A100 benchmark this tiling is what enables ~245 TFLOPS ([Triton docs](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html), 2026).

### Why is `num_pid_m` computed as `cdiv(M, BLOCK_SIZE_M)` and not just `M`?

Because `num_pid_m` counts *tiles*, not elements. `M` is the matrix height in elements and `BLOCK_SIZE_M` is a tile's height, so their ceiling division gives the number of block-rows. For a 9-block-tall grid with `BLOCK_SIZE_M=3`, you need `M=27`, not `M=9`.

### What does grouped ordering actually change?

Only the sequence in which tiles run; the arithmetic is identical. By running tiles in column-major bands, adjacent programs reuse the same strips of A and B while they are hot in L2. The tutorial reports this lifts an A100 from ~220 to ~245 TFLOPS ([Triton docs](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html), 2026).

### Why accumulate in fp32 when the inputs are fp16?

To preserve precision. A dot product over a long K dimension sums many products, and doing that in fp16 accumulates rounding error quickly. Triton accumulates in fp32 and casts to fp16 only at the end, right before the masked store. That is a standard mixed-precision pattern.

### Can Triton really match cuBLAS?

On common shapes, yes. Triton was explicitly designed to reach vendor-library performance from portable code ([Tillet et al., MAPL 2019](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)). Hand-written CUDA can also get there. Simon Boehm reaches ~95% of cuBLAS after roughly ten kernels ([siboehm.com](https://siboehm.com/articles/22/CUDA-MMM), 2022), but Triton gets you most of the way with far less code.

## Conclusion

The Triton matmul kernel only looks intimidating because two ideas are compressed into a handful of lines. Unpack them and it is almost mechanical:

- **Grouped ordering** is a pure scheduling trick: same math, ~10% faster, because reordered tiles reuse hot L2 data.
- **Pointer arithmetic** falls out of one drawing: C's tile is where A's row-strip crosses B's column-strip, and `[:, None]` / `[None, :]` broadcasts turn that picture into a block of addresses.
- **The K loop** slides both strips while accumulating in fp32, and **autotuning** searches block sizes and pipeline depth to close the last gap to cuBLAS.

Draw the corner before you type. Once the sketch is on paper, you can write the kernel by reading it off the page instead of memorizing forty lines.

**Next:** matmuls rarely live alone. The real payoff is fusing them with the ops around them. See Fused Attention (tutorial 06), where matmuls are chained with softmax inside one Triton kernel.

---

### Sources

- Triton official tutorial, *Matrix Multiplication (03-matrix-multiplication)*, retrieved 2026-07-07, https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- Philippe Tillet, H.T. Kung, David Cox, *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*, MAPL/PLDI 2019, retrieved 2026-07-07, https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf
- Simon Boehm, *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog*, 2022, retrieved 2026-07-07, https://siboehm.com/articles/22/CUDA-MMM
- Triton API docs, *triton.Config* (num_warps, num_stages), retrieved 2026-07-07, https://triton-lang.org/main/python-api/generated/triton.Config.html
- Christian Mills, *GPU MODE Lecture 14: A Practitioner's Guide to Triton*, 2024, retrieved 2026-07-07, https://christianjmills.com/posts/cuda-mode-notes/lecture-014/
- L. Weitkamp, *Triton Exercises: Block Pointers*, retrieved 2026-07-07, https://lweitkamp.github.io/triton_exercises/introduction/block_pointers.html

### About the author

**Hang Hor** is a senior data scientist who writes GPU-kernel and deep-learning explainers while working through the llm.c course, implementing kernels in both Triton and CUDA. This piece grew out of hands-on notes taken while reproducing the official Triton matmul tutorial line by line.

{% raw %}
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "BlogPosting",
      "headline": "Triton Matrix Multiplication, Explained: A Tutorial Companion",
      "description": "A plain-English walkthrough of Triton's matmul kernel: grouped tile ordering lifts an A100 from 220 to 245 TFLOPS, plus pointer arithmetic you can draw by hand.",
      "image": "03-matrix-multiplication-cover.svg",
      "datePublished": "2026-07-07",
      "dateModified": "2026-07-07",
      "keywords": "triton, matrix multiplication, gpu kernels, cuda, deep learning",
      "author": {
        "@type": "Person",
        "name": "Hang Hor",
        "description": "Senior data scientist writing GPU-kernel and deep-learning explainers in Triton and CUDA."
      }
    },
    {
      "@type": "FAQPage",
      "mainEntity": [
        {
          "@type": "Question",
          "name": "Does one Triton program compute a single element of C?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "No. Each program computes a full BLOCK_SIZE_M x BLOCK_SIZE_N tile of C. The scalar multiply-adds still happen, but many of them run inside one tl.dot, executed by the warps assigned to that program. On the tutorial's A100 benchmark this tiling is what enables about 245 TFLOPS."
          }
        },
        {
          "@type": "Question",
          "name": "Why is num_pid_m computed as cdiv(M, BLOCK_SIZE_M) and not just M?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "Because num_pid_m counts tiles, not elements. M is the matrix height in elements and BLOCK_SIZE_M is a tile's height, so their ceiling division gives the number of block-rows. For a 9-block-tall grid with BLOCK_SIZE_M=3, you need M=27, not M=9."
          }
        },
        {
          "@type": "Question",
          "name": "What does grouped ordering actually change?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "Only the sequence in which tiles run; the arithmetic is identical. By running tiles in column-major bands, adjacent programs reuse the same strips of A and B while they are hot in L2. The tutorial reports this lifts an A100 from about 220 to 245 TFLOPS."
          }
        },
        {
          "@type": "Question",
          "name": "Why accumulate in fp32 when the inputs are fp16?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "To preserve precision. A dot product over a long K dimension sums many products, and doing that in fp16 accumulates rounding error quickly. Triton accumulates in fp32 and casts to fp16 only at the end, right before the masked store, a standard mixed-precision pattern."
          }
        },
        {
          "@type": "Question",
          "name": "Can Triton really match cuBLAS?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "On common shapes, yes. Triton was explicitly designed to reach vendor-library performance from portable code. Hand-written CUDA can also get there, reaching about 95% of cuBLAS after roughly ten kernels, but Triton gets you most of the way with far less code."
          }
        }
      ]
    }
  ]
}
</script>
{% endraw %}
