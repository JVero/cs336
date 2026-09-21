"""Run assignment 2's attention benchmark (cs336_systems.attn_bench) on a Modal GPU.

Same image and mount as modal_bench.py. attn_bench sweeps its own grid and appends plain
text lines to --out, so this launcher just runs it and brings that file back. It does not
touch bench_log.csv, which is for bench_script's format.

Launch (from the workspace root). --flags is passed through; leave out --device and --out,
the launcher sets those.
    modal run scripts/modal_attn.py
    modal run scripts/modal_attn.py --gpu A100-80GB --flags "--warmup 5"

The output lands in assignment2-systems/results/attn_<gpu>_<timestamp>.txt (override with
--out) and is also printed. It is copied to the cs336-runs volume at
/a2/attn_<gpu>_<timestamp>.txt before the container exits, including after Ctrl+C or a
crash, so lines written before that are not lost. Fetch one with:
    modal volume get cs336-runs /a2/<name>.txt assignment2-systems/results/<name>.txt

Default GPU is H100 (80 GB, about $4/h). The sweep is 20 configs and should take a few minutes.
"""

import datetime
import pathlib
import shlex
import shutil
import subprocess

import modal

WORKSPACE = pathlib.Path(__file__).resolve().parent.parent
A2 = WORKSPACE / "assignment2-systems"
REMOTE_A2 = "/root/cs336/assignment2-systems"
REMOTE_OUT = "/tmp/attn.txt"
RUNS_MOUNT = "/runs"

app = modal.App("cs336-a2")
runs_vol = modal.Volume.from_name("cs336-runs", create_if_missing=True)

# What not to upload from assignment2-systems. Copied from modal_bench.py.
IGNORE = [
    "**/.venv",
    "**/__pycache__",
    "**/.pytest_cache",
    "**/.ruff_cache",
    "**/.DS_Store",
    "results",
    "cs336_systems/runs",
    "**/*.nsys-rep",
    "**/*.sqlite",
    "**/*.pkl",
    "**/*.pickle",
]

# See modal_bench.py for why cs336-basics is skipped at build time and put on PYTHONPATH.
# MPLBACKEND=Agg: attn_bench ends in plt.show(), and the container has no display.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(A2), extra_options="--no-install-package cs336-basics")
    .env({"PYTHONUNBUFFERED": "1", "PYTHONPATH": f"{REMOTE_A2}/cs336-basics", "MPLBACKEND": "Agg"})
    .add_local_dir(A2, remote_path=REMOTE_A2, ignore=IGNORE)
)


@app.function(image=image, gpu="H100", volumes={RUNS_MOUNT: runs_vol}, timeout=3600)
def attn(flags: list[str], name: str) -> tuple[int, str, str]:
    """Returns (exit code, the text attn_bench wrote to --out, the exact command run)."""
    cmd = ["python", "-m", "cs336_systems.attn_bench", "--device", "cuda", "--out", REMOTE_OUT, *flags]
    command = " ".join(shlex.quote(c) for c in cmd)
    print("$", command, flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
    out_path = pathlib.Path(REMOTE_OUT)
    returncode = -1
    try:
        returncode = subprocess.run(cmd, cwd=REMOTE_A2, check=False).returncode
    finally:
        # Runs on Ctrl+C and on a crash too, so lines already written survive on the volume.
        if out_path.exists():
            keep = pathlib.Path(RUNS_MOUNT) / "a2"
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out_path, keep / f"{name}.txt")
            runs_vol.commit()
    return returncode, out_path.read_text() if out_path.exists() else "", command


@app.local_entrypoint()
def main(flags: str = "", gpu: str = "H100", out: str = "", timeout_hours: float = 1.0) -> None:
    flag_list = shlex.split(flags)
    for owned in ("--device", "--out"):
        if owned in flag_list:
            raise SystemExit(f"{owned} is set by the launcher; drop it from --flags (use --out on the launcher for the local path)")
    name = f"attn_{gpu}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    path = pathlib.Path(out) if out else A2 / "results" / f"{name}.txt"
    fn = attn.with_options(gpu=gpu, timeout=int(timeout_hours * 3600))
    try:
        returncode, text, command = fn.remote(flag_list, name)
    except KeyboardInterrupt:
        print(f"\ninterrupted; lines written before that are on the volume. Fetch with:\n"
              f"  modal volume get cs336-runs /a2/{name}.txt {path}")
        raise
    if text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {command}\n{text}")
        print(f"\n{text}wrote {path}")
    else:
        print("\nno output file came back")
    if returncode != 0:
        raise SystemExit(f"attn_bench exited with code {returncode}")
