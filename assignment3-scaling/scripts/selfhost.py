"""Self-hosted stand-in for the course training API (tooling, not a graded solution).

The course runs the staff JAX trainer (`cs336_scaling/training/`) behind a Stanford-only
API. This file runs that same trainer on your own Modal account instead, without the
course server, its W&B team, or Tailscale. Training code, model and data pipeline are
the staff's, imported unchanged; nothing in `cs336_scaling/` is edited.

Run everything from `assignment3-scaling/`.

One-time data prep (tokenizes a prefix of DCLM, then shuffles it into training shards):

    uv run modal run scripts/selfhost.py::prep --measure-only   # tokenize 1 shard, print cost estimate
    uv run modal run scripts/selfhost.py::prep                  # full prep, aborts if estimate > --max-cost

Training (one config, or a JSON list of configs run in parallel):

    uv run modal run scripts/selfhost.py::train --config cfg.json --gpu h100

A config file holds the same `training_config` object as the handout's POST /submit body.
Every run appends one line to `experiments/results.jsonl` (config, val losses, timings, cost
estimate, W&B link) and logs to W&B project `cs336-a3-scaling` under your default entity.
A config that already completed on the same GPU is skipped unless you pass --force.

From your own Python scripts:

    from selfhost import run_experiments          # with scripts/ on sys.path
    results = run_experiments([cfg1, cfg2], gpu="h100")

Spend so far (local only, reads experiments/results.jsonl):

    uv run python scripts/selfhost.py budget

Each finished run also saves its record on the Modal volume, and run_experiments() launches
detached, so runs survive a dropped laptop connection. Pull any records the laptop missed
(for the CLI entrypoint, pass `modal run --detach ...` to get the same behavior):

    uv run python scripts/selfhost.py sync

Detached runs keep going after Ctrl-C. To stop them, find the app ID (state "ephemeral
(detached)") and stop it by ID; stopping by name only works for deployed apps:

    uv run modal app list
    uv run modal app stop ap-...
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import threading
import uuid
import time
import traceback
from pathlib import Path
from typing import Any

import modal

APP_NAME = "cs336-a3-selfhost"
WANDB_PROJECT = "cs336-a3-scaling"

# Size of the shuffled training set. Training and prep must agree on this number: it is
# part of the DCLMData artifact identity, so changing it means re-running prep.
DATA_TOTAL_TOKENS = 10_000_000_000
_SHARD_TOKENS = 2**29  # DCLMData.tokens_per_chunk and validation_tokens defaults
DATA_TRAIN_TOKENS = math.ceil(DATA_TOTAL_TOKENS / _SHARD_TOKENS) * _SHARD_TOKENS
DATA_REQUESTED_TOKENS = (
    DATA_TRAIN_TOKENS + _SHARD_TOKENS
)  # train shards + 1 validation shard

# Modal list prices, USD per second (modal.com/pricing, 2026-10-14).
GPU_PRICE_PER_S = {"b200": 0.001736, "h100": 0.001097, "a10": 0.000306, "l4": 0.000222}
CPU_CORE_PRICE_PER_S = 0.0000131
MEM_GIB_PRICE_PER_S = 0.00000222

TOKENIZE_CPU = 8.0
TOKENIZE_MEM_MIB = 32768
MAX_PARALLEL_RUNS = 8  # cap on concurrent training containers, a spending guard

RESULTS_PATH = Path("experiments/results.jsonl")
VOLUME_RESULTS_DIR = "a3_results"  # per-run records on furu_volume, read back by sync()

furu_volume = modal.Volume.from_name("cs336-a3-furu", create_if_missing=True, version=2)
# Furu stores artifacts under the relative path furu/data; Modal containers start in /root.
VOLUMES = {"/root/furu": furu_volume}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync("./", extras=["server"])
    .env({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90", "TOKENIZERS_PARALLELISM": "true"})
    .add_local_python_source("cs336_scaling")
)

app = modal.App(APP_NAME, image=image)


def _run_name(tc: Any, gpu: str) -> str:
    """Readable W&B run name, e.g. L9-d448-lr0.0001-bs128-tok1M-a1b2c3-h100.

    The hash suffix keeps names distinct when configs differ only in unlisted fields.
    """
    arch = tc.architecture_config
    lr = tc.optimizer_config.lr_scheduler.peak_value
    tokens = tc.total_train_tokens
    tok = f"{tokens / 1e9:.3g}B" if tokens >= 1e9 else f"{tokens / 1e6:.3g}M"
    return (
        f"L{arch.num_hidden_layers}-d{arch.hidden_size}-lr{lr:.3g}"
        f"-bs{tc.train_batch_size}-tok{tok}-{tc.unique_id[:6]}-{gpu}"
    )


def _check_cwd() -> None:
    pyproject = Path("pyproject.toml")
    if not pyproject.exists() or 'name = "cs336_scaling"' not in pyproject.read_text():
        raise SystemExit(
            "Run this from assignment3-scaling/ (the image is built from ./)."
        )


def _training_config_cls():
    # Importing training_config directly hits a circular import in the staff package
    # (schemas/__init__ imports it back); importing schemas first resolves it.
    import cs336_scaling.schemas  # noqa: F401
    from cs336_scaling.training.training_config import TrainingConfig

    return TrainingConfig


# --------------------------------------------------------------------------- data prep


@app.function(volumes=VOLUMES, timeout=30 * 60)
def list_shard_urls() -> list[str]:
    from cs336_scaling.tokenized_data import ShuffledDCLMShardUrls

    urls = ShuffledDCLMShardUrls().load_or_create()
    furu_volume.commit()
    return urls


@app.function(
    volumes=VOLUMES,
    cpu=TOKENIZE_CPU,
    memory=TOKENIZE_MEM_MIB,
    timeout=2 * 60 * 60,
    max_containers=16,
    retries=1,
)
def tokenize_shard(url: str) -> dict[str, Any]:
    """Same logic as the staff make_chunk: reuse a finished chunk, wipe a partial one."""
    import shutil

    from cs336_scaling.tokenized_data import TokenizedDCLMChunk

    started = time.perf_counter()
    chunk = TokenizedDCLMChunk(url=url)
    cached = chunk.is_completed()
    if cached:
        result = chunk.try_load()
        assert result is not None
    else:
        if chunk.data_dir.exists():
            shutil.rmtree(chunk.data_dir)
        result = chunk.load_or_create(use_lock=False)
        furu_volume.commit()
    return {
        "url": url,
        "n_tokens": int(result.n_tokens),
        "seconds": time.perf_counter() - started,
        "cached": cached,
    }


@app.function(
    volumes=VOLUMES,
    cpu=8.0,
    memory=65536,  # one shuffle run holds every token in RAM: ~21 GB of uint16 at 10.7B tokens
    # Raw copies + shuffle run + final shards need ~65 GB; 512 GiB is Modal's minimum request.
    ephemeral_disk=524288,
    timeout=6 * 60 * 60,
)
def build_dataset() -> dict[str, Any]:
    from cs336_scaling.tokenized_data import DCLMData

    started = time.perf_counter()
    result = DCLMData(total_tokens=DATA_TOTAL_TOKENS).load_or_create()
    furu_volume.commit()
    return {
        "n_train_tokens": int(result.n_train_tokens),
        "n_validation_tokens": int(result.n_validation_tokens),
        "n_train_files": len(result.train_filepaths),
        "n_source_chunks": int(result.n_source_chunks),
        "seconds": time.perf_counter() - started,
    }


def _tokenize_cost(seconds: float) -> float:
    per_s = (
        TOKENIZE_CPU * CPU_CORE_PRICE_PER_S
        + TOKENIZE_MEM_MIB / 1024 * MEM_GIB_PRICE_PER_S
    )
    return seconds * per_s


@app.local_entrypoint()
def prep(measure_only: bool = False, max_cost: float = 15.0) -> None:
    """Tokenize the DCLM prefix the training set needs, then build the shuffled shards."""
    _check_cwd()
    urls = list_shard_urls.remote()
    print(
        f"{len(urls):,} DCLM parquet shards listed; need {DATA_REQUESTED_TOKENS:,} tokens"
    )

    first = tokenize_shard.remote(urls[0])
    est_chunks = math.ceil(DATA_REQUESTED_TOKENS / first["n_tokens"] * 1.15)
    # Tokenize cost only; the one shuffle/build job adds well under a dollar at this size.
    est_cost = _tokenize_cost(first["seconds"]) * 1.2 * est_chunks
    print(
        f"shard 0: {first['n_tokens']:,} tokens in {first['seconds']:.0f}s "
        f"(cached={first['cached']}), ~${_tokenize_cost(first['seconds']):.3f}\n"
        f"estimate: ~{est_chunks} shards, ~${est_cost:.2f} to tokenize"
    )
    if measure_only:
        return
    if est_cost > max_cost:
        raise SystemExit(
            f"estimate ${est_cost:.2f} exceeds --max-cost {max_cost}; not continuing"
        )

    # DCLMData consumes shards in list order, so the tokenized set must be a prefix.
    results = [first]
    n_tokens = first["n_tokens"]
    next_idx = 1
    while n_tokens < DATA_REQUESTED_TOKENS:
        remaining = DATA_REQUESTED_TOKENS - n_tokens
        avg = n_tokens / len(results)
        batch = urls[next_idx : next_idx + max(1, math.ceil(remaining / avg * 1.1))]
        if not batch:
            raise SystemExit("ran out of DCLM shards")
        next_idx += len(batch)
        for r in tokenize_shard.map(batch):
            results.append(r)
            n_tokens += r["n_tokens"]
        spent = sum(_tokenize_cost(r["seconds"]) for r in results if not r["cached"])
        print(
            f"tokenized {len(results)} shards, {n_tokens:,} tokens, ~${spent:.2f} so far"
        )

    print("building shuffled training shards (one job, can take a while)...")
    summary = build_dataset.remote()
    print(json.dumps(summary, indent=2))
    if summary["n_train_tokens"] != DATA_TRAIN_TOKENS:
        raise SystemExit(
            f"expected {DATA_TRAIN_TOKENS:,} train tokens, got {summary['n_train_tokens']:,}"
        )
    print("data ready")


# ---------------------------------------------------------------------------- training


def _train_impl(config: dict[str, Any], gpu: str) -> dict[str, Any]:
    """The staff run() loop from cs336_scaling/training/run.py, minus the course server."""
    container_started = time.perf_counter()
    import equinox.nn as nn
    import jax
    import wandb
    from jax.sharding import AxisType, NamedSharding
    from jax.sharding import PartitionSpec as P

    TrainingConfig = _training_config_cls()
    from cs336_scaling.tokenized_data import DCLMData
    from cs336_scaling.training.data import Batch, download_data
    from cs336_scaling.training.loop import OuterLossResult, outer_loss
    from cs336_scaling.training.model.basic_model import BasicCausalLM
    from cs336_scaling.training.model.jax_utils import (
        count_params,
        show_memory_analysis,
        tree_rearrange,
    )

    tc = TrainingConfig.model_validate(config)
    record: dict[str, Any] = {
        "unique_id": tc.unique_id,
        "gpu": gpu,
        "status": "failed",
        "val_losses": [],
        "final_val_loss": None,
        "train_seconds": None,
        # Per eval chunk. Chunk 0 can include JIT compile time, so measure throughput from the rest.
        "chunk_seconds": [],
        "compile_seconds": None,
        "model_params": None,
        "estimated_memory_bytes": None,
        "memory_limit_bytes": None,
        "memory_in_use_bytes": None,
        "wandb_url": None,
        "error": None,
    }

    # Hard stop if a run hangs past its budget (the staff server cancels the call the same way).
    # Modal may reuse this container for the next input, so the watchdog must stop with its run.
    run_finished = threading.Event()

    def watchdog() -> None:
        while not run_finished.wait(15):
            if train_started is not None:
                if (
                    time.perf_counter() - train_started
                    > tc.max_runtime_seconds * 1.05 + 120
                ):
                    print(
                        "watchdog: run exceeded max_runtime_seconds, killing container"
                    )
                    os._exit(1)

    train_started: float | None = None
    threading.Thread(target=watchdog, daemon=True).start()

    try:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=_run_name(tc, gpu),
            config=tc.model_dump(mode="json") | {"gpu": gpu},
            dir="/tmp",
        )
        record["wandb_url"] = run.url
    except Exception as exc:  # results still land in results.jsonl
        print(f"wandb.init failed, continuing without W&B: {exc!r}")
        wandb.init(mode="disabled")

    try:
        jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
        data = DCLMData(total_tokens=DATA_TOTAL_TOKENS).try_load()
        if data is None:
            raise RuntimeError(
                "training data not built; run `modal run scripts/selfhost.py::prep`"
            )

        mesh = jax.make_mesh(
            (jax.device_count(),), ("fsdp",), axis_types=(AxisType.Explicit,)
        )
        jax.set_mesh(mesh)
        data_sharding = NamedSharding(mesh, P(None, "fsdp", None))

        def put_batched_data(batch, batch_size: int):
            return jax.device_put(
                tree_rearrange(
                    batch.to_jax(),
                    "(loops batch) ... -> loops batch ...",
                    batch=batch_size,
                ),
                data_sharding,
            )

        data_result = download_data(data, tc)

        def train_data(batch_idx: int):
            return put_batched_data(
                Batch.train_data(data_result, batch_idx=batch_idx, training_config=tc),
                batch_size=tc.train_batch_size,
            )

        val_data = put_batched_data(
            Batch.val_data(data_result, training_config=tc),
            batch_size=tc.val_batch_size,
        )

        @jax.jit
        def make_model():
            model, state = nn.make_with_state(BasicCausalLM)(
                tc.architecture_config, key=jax.random.PRNGKey(tc.model_seed)
            )
            return model.apply_sharding(mesh), state

        model, state = make_model()
        record["model_params"] = int(count_params(model))
        print(f"model params: {record['model_params']:,}")

        opt_state = tc.optimizer_config.build(tc).init(model)

        # JAX's GPU pool (XLA_PYTHON_CLIENT_MEM_FRACTION of the card) and what is already
        # resident before the step runs: weights, optimizer state, data buffers.
        stats = jax.devices()[0].memory_stats() or {}
        record["memory_limit_bytes"] = stats.get("bytes_limit")
        record["memory_in_use_bytes"] = stats.get("bytes_in_use")
        print(
            f"device memory: limit {(stats.get('bytes_limit') or 0) / 2**30:.1f} GiB, "
            f"in use {(stats.get('bytes_in_use') or 0) / 2**30:.1f} GiB"
        )

        compile_started = time.perf_counter()
        record["estimated_memory_bytes"] = int(
            show_memory_analysis(
                outer_loss, model, state, train_data(0), val_data, tc, opt_state
            )
        )
        record["compile_seconds"] = time.perf_counter() - compile_started

        res: OuterLossResult | None = None
        train_started = time.perf_counter()
        timed_out = False
        for i in range(tc.n_evals):
            if time.perf_counter() - train_started > tc.max_runtime_seconds:
                timed_out = True
                break
            chunk_started = time.perf_counter()
            res = outer_loss(model, state, train_data(i), val_data, tc, opt_state)
            model, opt_state = res.model, res.opt_state
            val_loss = res.val_loss.item()  # blocks, so chunk timing is real
            chunk_seconds = time.perf_counter() - chunk_started
            record["val_losses"].append(val_loss)
            record["chunk_seconds"].append(chunk_seconds)
            tokens_seen = (i + 1) * tc.eval_every_tokens
            wandb.log(
                {
                    "chunk": i,
                    "tokens": tokens_seen,
                    "optimizer_step": (i + 1) * tc.optimizer_steps_per_eval,
                    "train_loss": res.train_losses.mean().item(),
                    "val_loss": val_loss,
                    "chunk_seconds": chunk_seconds,
                }
            )
            print(f"eval {i + 1}/{tc.n_evals}: val_loss {val_loss:.4f}")

        record["train_seconds"] = time.perf_counter() - train_started
        if timed_out or record["train_seconds"] > tc.max_runtime_seconds:
            # Same rule as the course API: over budget means not completed.
            record["status"] = "timeout"
        else:
            record["status"] = "completed"
            record["final_val_loss"] = record["val_losses"][-1]
        wandb.log(
            {
                "train_seconds": record["train_seconds"],
                "final_val_loss": record["final_val_loss"],
            }
        )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        run_finished.set()
        wandb.finish()

    record["container_seconds"] = time.perf_counter() - container_started
    record["est_cost_usd"] = record["container_seconds"] * GPU_PRICE_PER_S[gpu]
    record["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    record["training_config"] = config
    record["record_id"] = uuid.uuid4().hex
    # Also save the record on the volume, so a run whose laptop session dropped can be
    # recovered later with `python scripts/selfhost.py sync`.
    try:
        out_dir = Path("/root/furu") / VOLUME_RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{record['record_id']}.json").write_text(json.dumps(record))
        furu_volume.commit()
    except Exception:
        traceback.print_exc()
    return record


_TRAIN_KW = dict(
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("wandb")],
    timeout=13 * 60 * 60,  # last-resort cap; the watchdog normally fires first
    max_containers=MAX_PARALLEL_RUNS,
)


@app.function(gpu="B200", **_TRAIN_KW)
def train_b200(config: dict[str, Any]) -> dict[str, Any]:
    return _train_impl(config, "b200")


@app.function(gpu="H100", **_TRAIN_KW)
def train_h100(config: dict[str, Any]) -> dict[str, Any]:
    return _train_impl(config, "h100")


@app.function(gpu="A10", **_TRAIN_KW)
def train_a10(config: dict[str, Any]) -> dict[str, Any]:
    return _train_impl(config, "a10")


@app.function(gpu="L4", **_TRAIN_KW)
def train_l4(config: dict[str, Any]) -> dict[str, Any]:
    return _train_impl(config, "l4")


TRAIN_FUNCTIONS = {
    "b200": train_b200,
    "h100": train_h100,
    "a10": train_a10,
    "l4": train_l4,
}


# ------------------------------------------------------------------- local-side helpers


def load_results() -> list[dict[str, Any]]:
    if not RESULTS_PATH.exists():
        return []
    return [
        json.loads(line)
        for line in RESULTS_PATH.read_text().splitlines()
        if line.strip()
    ]


def _append_result(record: dict[str, Any]) -> None:
    # A record may arrive twice (sync() pulled it from the volume first); keep one copy.
    rid = record.get("record_id")
    if rid is not None and any(r.get("record_id") == rid for r in load_results()):
        return
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _validate(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate locally so a bad config fails before any GPU time is paid for."""
    TrainingConfig = _training_config_cls()
    out = []
    for cfg in configs:
        tc = TrainingConfig.model_validate(cfg)
        if tc.total_train_tokens >= DATA_TRAIN_TOKENS:
            raise ValueError(
                f"total_train_tokens={tc.total_train_tokens:,} needs more than the "
                f"{DATA_TRAIN_TOKENS:,} prepared tokens; raise DATA_TOTAL_TOKENS and re-run prep"
            )
        out.append(tc.model_dump(mode="json"))
    return out


def _run(configs: list[dict[str, Any]], gpu: str, force: bool) -> list[dict[str, Any]]:
    TrainingConfig = _training_config_cls()
    gpu = gpu.lower()
    if gpu not in TRAIN_FUNCTIONS:
        raise ValueError(f"gpu must be one of {sorted(TRAIN_FUNCTIONS)}, got {gpu!r}")
    configs = _validate(configs)

    done = {
        (r["unique_id"], r["gpu"]) for r in load_results() if r["status"] == "completed"
    }
    todo = []
    for cfg in configs:
        uid = TrainingConfig.model_validate(cfg).unique_id
        if (uid, gpu) in done and not force:
            print(f"skip {uid[:10]} on {gpu}: already completed (use --force to rerun)")
        else:
            todo.append(cfg)
    if not todo:
        return []

    print(f"launching {len(todo)} run(s) on {gpu}")
    records = []
    started = time.perf_counter()
    outputs = TRAIN_FUNCTIONS[gpu].map(todo, return_exceptions=True, order_outputs=True)
    # Iterate the outputs to the end (not zip, which stops early and leaves Modal's generator open).
    for i, out in enumerate(outputs):
        cfg = todo[i]
        if isinstance(
            out, BaseException
        ):  # container died (watchdog, OOM kill, preemption)
            out = {
                "unique_id": TrainingConfig.model_validate(cfg).unique_id,
                "gpu": gpu,
                "status": "failed",
                "val_losses": [],
                "final_val_loss": None,
                "error": f"{type(out).__name__}: {out}",
            }
        out.setdefault("finished_at", dt.datetime.now(dt.timezone.utc).isoformat())
        out["local_wall_seconds"] = time.perf_counter() - started
        out["training_config"] = cfg
        _append_result(out)
        records.append(out)
        loss = out.get("final_val_loss")
        print(
            f"{out['unique_id'][:10]} {out['status']}: "
            f"final_val_loss={loss if loss is None else round(loss, 4)} "
            f"train={out.get('train_seconds') or 0:.0f}s cost~${out.get('est_cost_usd') or 0:.3f}"
            + (f"  error: {out['error']}" if out.get("error") else "")
        )
    return records


def run_experiments(
    configs: list[dict[str, Any]], gpu: str = "h100", force: bool = False
) -> list[dict[str, Any]]:
    """Run configs on Modal from a plain Python script and return their result records."""
    _check_cwd()
    # detach=True: if this laptop disconnects, the containers keep training and save their
    # records to the volume; recover them afterwards with `python scripts/selfhost.py sync`.
    with modal.enable_output(), app.run(detach=True):
        return _run(configs, gpu, force)


@app.local_entrypoint()
def train(config: str, gpu: str = "h100", force: bool = False) -> None:
    """Run the config (or list of configs) in a JSON file."""
    _check_cwd()
    loaded = json.loads(Path(config).read_text())
    _run(loaded if isinstance(loaded, list) else [loaded], gpu, force)


def budget() -> None:
    """Summarize estimated spend from experiments/results.jsonl (GPU time only)."""
    _check_cwd()
    records = load_results()
    by_gpu: dict[str, list[float]] = {}
    for r in records:
        by_gpu.setdefault(r["gpu"], []).append(r.get("est_cost_usd") or 0.0)
    for gpu, costs in sorted(by_gpu.items()):
        print(f"{gpu}: {len(costs)} runs, ~${sum(costs):.2f}")
    print(f"total: {len(records)} runs, ~${sum(sum(c) for c in by_gpu.values()):.2f}")


def sync() -> None:
    """Append records saved on the volume that are missing from experiments/results.jsonl.

    Runs launched before the volume copy existed have no record_id and are never touched.
    """
    _check_cwd()
    have = {r.get("record_id") for r in load_results()} - {None}
    try:
        entries = furu_volume.listdir(VOLUME_RESULTS_DIR)
    except Exception as exc:  # directory does not exist until the first run saves to it
        print(f"no records on the volume yet ({type(exc).__name__})")
        return
    added = 0
    for entry in sorted(entries, key=lambda e: e.path):
        if not entry.path.endswith(".json"):
            continue
        record = json.loads(b"".join(furu_volume.read_file(entry.path)))
        if record.get("record_id") in have:
            continue
        record.setdefault("local_wall_seconds", None)
        _append_result(record)
        have.add(record["record_id"])
        added += 1
        loss = record.get("final_val_loss")
        print(
            f"recovered {record['unique_id'][:10]} {record['status']} on {record['gpu']}: "
            f"final_val_loss={loss if loss is None else round(loss, 4)}"
        )
    print(f"sync: {added} record(s) added, {len(entries)} on the volume")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["budget"]:
        budget()
    elif sys.argv[1:] == ["sync"]:
        sync()
    else:
        raise SystemExit(__doc__)
