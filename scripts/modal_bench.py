"""Run assignment 2's benchmarking script on a Modal GPU.

Nothing in cs336_systems changes. This wraps `python -m cs336_systems.bench_script` in a
Modal function: assignment2-systems is mounted, deps come from its uv.lock, and the CSV the
script writes is copied back to the laptop when the run ends.

Launch (from the workspace root). --flags takes the bench script's usual flags. Leave out
--device and --of_name; the launcher sets those and refuses to start if they are present.
    modal run scripts/modal_bench.py --flags "--context_length 512 --models large xl --forward --forward_and_back --full_step"

The CSV lands in assignment2-systems/results/<gpu>_<timestamp>.csv (override with --out) and
is also printed. The same file is copied to the cs336-runs volume at /a2/<gpu>_<timestamp>.csv
before the container exits, including after Ctrl+C, so rows written before an interrupt or a
crash are not lost. Fetch one with:
    modal volume get cs336-runs /a2/<name>.csv assignment2-systems/results/<name>.csv
Use --detach for a run you do not want tied to the terminal; then the CSV only reaches the
volume, not the laptop, and `modal app logs cs336-a2` follows it.

Default GPU is H100 (80 GB, about $4/h). A100-80GB is the cheaper 80 GB option (about
$2.50/h). Anything with 24 GB (A10G, L4) fits at most the medium size.
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
REMOTE_CSV = "/tmp/bench.csv"
RUNS_MOUNT = "/runs"

app = modal.App("cs336-a2")
runs_vol = modal.Volume.from_name("cs336-runs", create_if_missing=True)

# What not to upload from assignment2-systems.
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
]

# uv.lock pins cs336-basics as an editable install from ./cs336-basics, but Modal's uv_sync
# copies only pyproject.toml and uv.lock into the image build, so that path does not exist
# there. That one package is skipped at build time and imported from the mounted source
# through PYTHONPATH instead.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(A2), extra_options="--no-install-package cs336-basics")
    .env({"PYTHONUNBUFFERED": "1", "PYTHONPATH": f"{REMOTE_A2}/cs336-basics"})
    .add_local_dir(A2, remote_path=REMOTE_A2, ignore=IGNORE)
)


@app.function(image=image, gpu="H100", volumes={RUNS_MOUNT: runs_vol}, timeout=3600)
def bench(flags: list[str], name: str) -> tuple[int, str]:
    cmd = [
        "python", "-m", "cs336_systems.bench_script",
        "--device", "cuda",
        "--of_name", REMOTE_CSV,
        *flags,
    ]
    print("$", " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
    csv = pathlib.Path(REMOTE_CSV)
    returncode = -1
    try:
        returncode = subprocess.run(cmd, cwd=REMOTE_A2, check=False).returncode
    finally:
        # Runs on Ctrl+C and on a crash too, so rows the script already wrote survive on the volume.
        if csv.exists():
            keep = pathlib.Path(RUNS_MOUNT) / "a2" / f"{name}.csv"
            keep.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(csv, keep)
            runs_vol.commit()
            print(f"\n{csv.read_text()}(also on volume cs336-runs at /a2/{name}.csv)", flush=True)
    return returncode, csv.read_text() if csv.exists() else ""


@app.local_entrypoint()
def main(flags: str, gpu: str = "H100", out: str = "", timeout_hours: float = 1.0) -> None:
    flag_list = shlex.split(flags)
    for owned in ("--device", "--of_name"):
        if owned in flag_list:
            raise SystemExit(f"{owned} is set by the launcher; drop it from --flags (use --out for the local CSV path)")
    name = f"{gpu}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    path = pathlib.Path(out) if out else A2 / "results" / f"{name}.csv"
    fn = bench.with_options(gpu=gpu, timeout=int(timeout_hours * 3600))
    try:
        returncode, csv = fn.remote(flag_list, name)
    except KeyboardInterrupt:
        print(f"\ninterrupted; rows written before that are on the volume. Fetch with:\n"
              f"  modal volume get cs336-runs /a2/{name}.csv {path}")
        raise
    if csv:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(csv)
        print(f"\n{csv}wrote {path}")
    else:
        print("\nno CSV came back")
    if returncode != 0:
        raise SystemExit(f"bench_script exited with code {returncode}")
