"""Flat table of experiments/results.jsonl, one row per run (tooling, no Modal import).

    from results import load_table      # with scripts/ on sys.path
    df = load_table()                   # completed runs only
    df = load_table(completed_only=False)

Or print it from the shell:

    uv run python scripts/results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS_PATH = Path(__file__).resolve().parent.parent / "experiments" / "results.jsonl"


def _gib(n: int | None) -> float | None:
    return None if n is None else n / 2**30


def _row(r: dict) -> dict:
    tc = r["training_config"]
    arch = tc["architecture_config"]
    sched = tc["optimizer_config"]["lr_scheduler"]
    return {
        "layers": arch["num_hidden_layers"],
        "d_model": arch["hidden_size"],
        "d_ff": arch["intermediate_size"],
        "heads": arch["num_attention_heads"],
        "head_dim": arch["head_dim"],
        "lr": sched["peak_value"],
        "beta1": tc["optimizer_config"].get("beta1"),
        "beta2": tc["optimizer_config"].get("beta2"),
        "weight_decay": tc["optimizer_config"].get("weight_decay"),
        "batch": tc["train_batch_size"],
        "tokens": tc["total_train_tokens"],
        "n_evals": tc["n_evals"],
        # Non-embedding params by the handout's estimate, 12 * n_layer * d_model^2.
        "params_nonembed": 12 * arch["num_hidden_layers"] * arch["hidden_size"] ** 2,
        "params_total": r.get("model_params"),  # as counted by the trainer
        "loss": r["final_val_loss"],
        "val_losses": r["val_losses"],
        "status": r["status"],
        "gpu": r["gpu"],
        "train_seconds": r.get("train_seconds"),
        # Memory in GiB: the step's compile-time estimate, JAX's pool limit, and what was
        # already resident (weights, optimizer state, data) before the step ran.
        "mem_est_gib": _gib(r.get("estimated_memory_bytes")),
        "mem_limit_gib": _gib(r.get("memory_limit_bytes")),
        "mem_in_use_gib": _gib(r.get("memory_in_use_bytes")),
        "cost_usd": r.get("est_cost_usd"),
        "id": r["unique_id"][:10],
        "wandb_url": r.get("wandb_url"),
        "finished_at": r.get("finished_at"),
        "training_config": tc,  # full nested config, for anything not flattened above
    }


def load_table(completed_only: bool = True) -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        return pd.DataFrame()
    records = [
        json.loads(line) for line in RESULTS_PATH.read_text().splitlines() if line.strip()
    ]
    df = pd.DataFrame([_row(r) for r in records])
    if completed_only and not df.empty:
        df = df[df["status"] == "completed"].reset_index(drop=True)
    return df


if __name__ == "__main__":
    cols = ["layers", "d_model", "lr", "beta1", "beta2", "batch", "tokens", "params_nonembed", "loss", "id"]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(load_table()[cols].to_string(index=False))
