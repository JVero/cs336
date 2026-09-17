"""Per-block memory of a Transformer from an nsys memory profile: what one block's forward keeps,
and what one block's backward frees and produces.

Reads the .sqlite that Nsight Systems writes next to a .nsys-rep when you open it in the desktop
app (or `nsys export --type sqlite <rep>`). The profile must come from a run with
--cuda-memory-usage=true, the PyTorch op labels from --pytorch=autograd-shapes-nvtx, and
PYTORCH_NO_CUDA_MEMORY_CACHING=1 so every tensor is its own cudaMalloc/cudaFree; see the
memory-profiling paragraph in modal_nsys.py. Without the last one the profile has no frees and
this script cannot tell a temporary from a saved tensor.

A block is the window between two consecutive occurrences of one NVTX range that happens once
per block. --range names it: either a range of your own ("Self-attention", the default, from
cs336_basics.model) or the op name at the front of a PyTorch label ("SigmoidBackward0" matches
"SigmoidBackward0, seq = 811, ..."). Blocks are numbered in time order, so with a backward marker
--block 1 is the last layer's backward.

Forward, from one Self-attention to the next: the window holds one attention sub-layer, the FFN
after it, and the *next* block's first RMSNorm instead of this block's own; every block is
identical so the totals equal a true block. An allocation that is still alive when the window
ends is a tensor saved for backward (a residual), plus the block's output.
    cd assignment2-systems && uv run python ../scripts/nsys_alloc_pivot.py results/<run>/<run>.sqlite --block 5

Backward, from one SigmoidBackward0 (the SiLU in SwiGLU, once per block) to the next: the window
holds the rest of one block's FFN backward, its attention backward, and the start of the previous
block's FFN backward, again one block's worth. "freed, allocated before the window" is the
residuals released; "still alive at end" is what the block's backward produced and kept, its
parameter gradients and the gradient handed to the previous block.
    cd assignment2-systems && uv run python ../scripts/nsys_alloc_pivot.py results/<run>/<run>.sqlite --range SigmoidBackward0 --block 5

Each surviving allocation is attributed to the innermost aten:: range open, on the marker's
thread, when its cudaMalloc ran. Ops nest (aten::linear calls aten::mm), so the innermost op is
the one that made the tensor. Times print in ms from the session start, the same clock as the
ruler in the Nsight timeline, so the window can be found and hovered there. "in use" is bytes
allocated minus bytes freed since the capture began, which is what the Memory Usage row's
tooltip shows.
"""

import argparse
import sqlite3
from collections import defaultdict

MiB = 1048576.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("sqlite", help="the .sqlite exported from the .nsys-rep")
    p.add_argument("--block", type=int, default=2, help="which block window, 1-based in time order (default 2)")
    p.add_argument("--range", default="Self-attention",
                   help="NVTX range that occurs once per block: exact text, or the op name a PyTorch label starts with")
    p.add_argument("--top", type=int, default=10, help="rows in the per-op table")
    p.add_argument("--allocs", type=int, default=10, help="rows in the largest-allocations table")
    args = p.parse_args()

    db = sqlite3.connect(args.sqlite)
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "CUDA_GPU_MEMORY_USAGE_EVENTS" not in tables:
        raise SystemExit("no CUDA_GPU_MEMORY_USAGE_EVENTS table: the profile recorded no cudaMalloc/cudaFree "
                         "(missing --cuda-memory-usage=true, or the caching allocator never grew the pool)")

    marks = db.execute(
        "SELECT start, globalTid FROM NVTX_EVENTS WHERE text=? OR text LIKE ? ORDER BY start",
        (args.range, args.range + ", %"),
    ).fetchall()
    if len(marks) < args.block + 1:
        raise SystemExit(f"{len(marks)} '{args.range}' ranges in the profile; --block {args.block} needs {args.block + 1}")
    t0, tid = marks[args.block - 1]
    t1 = marks[args.block][0]

    # (start, bytes, address, op) for every memory event, in time order. op 0 = alloc, 1 = free.
    events = db.execute(
        "SELECT start, bytes, address, memoryOperationType FROM CUDA_GPU_MEMORY_USAGE_EVENTS ORDER BY start"
    ).fetchall()
    # Pair each free with the latest still-open allocation at that address, and keep a running
    # total of bytes in use since the capture began.
    open_at = defaultdict(list)
    freed_at = {}  # alloc index -> time of its free
    frees = []  # (free time, bytes, alloc time or None if allocated before the capture)
    in_use = 0
    in_use_at = {}  # t0 / t1 -> bytes in use just before that time
    for i, (t, b, addr, op) in enumerate(events):
        for mark in (t0, t1):
            if mark not in in_use_at and t >= mark:
                in_use_at[mark] = in_use
        if op == 0:
            open_at[addr].append(i)
            in_use += b
        else:
            alloc_t = None
            if open_at[addr]:
                j = open_at[addr].pop()
                freed_at[j] = t
                alloc_t = events[j][0]
            frees.append((t, b, alloc_t))
            in_use -= b
    for mark in (t0, t1):
        in_use_at.setdefault(mark, in_use)

    window = [(i, e) for i, e in enumerate(events) if e[3] == 0 and t0 <= e[0] < t1]
    survivors = [(i, e) for i, e in window if freed_at.get(i, t1) >= t1]
    freed_old = sum(b for t, b, a in frees if t0 <= t < t1 and (a is None or a < t0))
    freed_new = sum(b for t, b, a in frees if t0 <= t < t1 and a is not None and a >= t0)

    # Innermost aten:: range on the marker's thread at each survivor's allocation time.
    ranges = db.execute(
        "SELECT start, end, text FROM NVTX_EVENTS WHERE globalTid=? AND text LIKE 'aten::%' AND end>=? AND start<? ORDER BY start",
        (tid, t0, t1),
    ).fetchall()

    def label(t: int) -> str:
        best = None
        for s, e, text in ranges:
            if s > t:
                break
            if e >= t:
                best = text
        return best or "(no aten:: range open)"

    labelled = [(e[1], label(e[0])) for _, e in survivors]
    by_op = defaultdict(lambda: [0, 0])
    for b, text in labelled:
        op = text.split(",", 1)[0]
        by_op[op][0] += 1
        by_op[op][1] += b
    allocated = sum(e[1] for _, e in window)
    survived = sum(b for b, _ in labelled)

    print(f"block {args.block}: {t0 / 1e6:.3f} ms to {t1 / 1e6:.3f} ms on the timeline ruler "
          f"(from '{args.range}' #{args.block} to #{args.block + 1})")
    print(f"in use at window start {in_use_at[t0] / MiB:9.1f} MiB")
    print(f"in use at window end   {in_use_at[t1] / MiB:9.1f} MiB   (change {(in_use_at[t1] - in_use_at[t0]) / MiB:+.1f} MiB)")
    print(f"allocated in window    {allocated / MiB:9.1f} MiB in {len(window)} cudaMallocs")
    print(f"freed in window        {freed_old / MiB:9.1f} MiB allocated before the window, "
          f"{freed_new / MiB:.1f} MiB allocated inside it")
    print(f"still alive at end     {survived / MiB:9.1f} MiB in {len(survivors)} tensors")
    print()
    print(f"{'op':<24}{'n':>4}{'MiB':>10}{'% of alive':>12}")
    for op, (n, b) in sorted(by_op.items(), key=lambda kv: -kv[1][1])[: args.top]:
        print(f"{op:<24}{n:>4}{b / MiB:>10.1f}{100 * b / survived:>11.1f}%")
    print()
    print("largest surviving allocations (label = op, autograd seq, input shapes):")
    for b, text in sorted(labelled, key=lambda x: -x[0])[: args.allocs]:
        print(f"{b / MiB:>8.1f} MiB  {text}")


if __name__ == "__main__":
    main()
