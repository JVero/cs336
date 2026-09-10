"""Run assignment 1's training script on a Modal GPU.

Nothing in cs336_basics changes. This wraps `python -m cs336_basics.training_loop`
in a Modal function: the repo is mounted, deps come from assignment1-basics/uv.lock,
tokenized data is read from the `cs336-data` volume, and run directories are written
to the `cs336-runs` volume so they survive the container.

Prerequisites (once):
    uv tool install modal        # CLI, lives outside the course projects
    modal volume create cs336-data
    modal volume create cs336-runs
    modal volume put cs336-data assignment1-basics/data/TinyStoriesV2-GPT4-train.npy /TinyStoriesV2-GPT4-train.npy
    modal volume put cs336-data assignment1-basics/data/TinyStoriesV2-GPT4-valid.npy /TinyStoriesV2-GPT4-valid.npy

Launch (from the workspace root). --flags takes the training script's usual flags:
    modal run scripts/modal_train.py --flags "--train_data TinyStoriesV2-GPT4-train.npy --val_data TinyStoriesV2-GPT4-valid.npy --num_steps 60 --label smoke"
    modal run --detach scripts/modal_train.py --flags "... --num_steps 40000 --lr 1e-3 --label bs32"

--detach keeps the run going after Ctrl+C or closing the laptop. Follow it with:
    modal app logs cs336-a1

Results:
    modal volume ls cs336-runs
    modal volume get cs336-runs /<run-dir> assignment1-basics/runs/     # one run
    modal volume get cs336-runs / assignment1-basics/runs/              # everything
The destination must be an existing directory; the run dir is created inside it. Pointing at
a path that does not exist yet makes `get` write every file onto that one path, so you end up
with a single checkpoint-sized file named like the run dir.

Default GPU is B200. Override with --gpu: T4, L4, A10, L40S, A100, A100-80GB, H100, H200.
"""

import argparse
import pathlib
import shlex
import shutil
import subprocess

import modal

WORKSPACE = pathlib.Path(__file__).resolve().parent.parent
A1 = WORKSPACE / "assignment1-basics"
REMOTE_WS = "/root/cs336"
REMOTE_A1 = f"{REMOTE_WS}/assignment1-basics"
DATA_MOUNT = "/data"
RUNS_MOUNT = "/runs"
LOCAL_DATA = "/tmp/data"  # container-local NVMe; get_batch's random reads are faster here than on the volume

app = modal.App("cs336-a1")
data_vol = modal.Volume.from_name("cs336-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("cs336-runs", create_if_missing=True)

# What not to upload from the workspace. Tracked files must all be present, otherwise the
# training script's `git status` / `git diff HEAD` calls would report them as deleted.
IGNORE = [
    "**/.venv",
    "**/__pycache__",
    "**/.pytest_cache",
    "**/.ruff_cache",
    "**/.DS_Store",
    "assignment1-basics/data",
    "assignment1-basics/runs",
    "assignment1-basics/*.svg",
    "assignment1-basics/*.log",
    "assignment1-basics/*_merges.json",
    "assignment1-basics/*_vocab.json",
    "scratch",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands("git config --global --add safe.directory '*'")
    .uv_sync(uv_project_dir=str(A1))
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir(WORKSPACE, remote_path=REMOTE_WS, ignore=IGNORE)
)


@app.function(
    image=image,
    gpu="B300",
    volumes={DATA_MOUNT: data_vol, RUNS_MOUNT: runs_vol},
    timeout=24 * 3600,
)
def train(flags: list[str]) -> None:
    # Copy just the two data files this run needs onto local disk.
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--train_data")
    peek.add_argument("--val_data")
    known, _ = peek.parse_known_args(flags)
    pathlib.Path(LOCAL_DATA).mkdir(exist_ok=True)
    for name in {known.train_data, known.val_data} - {None}:
        src = pathlib.Path(DATA_MOUNT) / name
        if not src.exists():
            raise FileNotFoundError(f"{name} is not on the cs336-data volume; upload it with `modal volume put`")
        shutil.copyfile(src, pathlib.Path(LOCAL_DATA) / name)

    cmd = [
        "python", "-m", "cs336_basics.training_loop",
        "--device", "cuda",
        "--train_dir", LOCAL_DATA,
        "--val_dir", LOCAL_DATA,
        "--runs_dir", RUNS_MOUNT,
        *flags,
    ]
    print("$", " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
    try:
        subprocess.run(cmd, cwd=REMOTE_A1, check=True)
    finally:
        runs_vol.commit()


@app.local_entrypoint()
def main(flags: str, gpu: str = "B300", timeout_hours: float = 12.0) -> None:
    fn = train.with_options(gpu=gpu, timeout=int(timeout_hours * 3600))
    fn.remote(shlex.split(flags))
