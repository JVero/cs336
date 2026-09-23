"""Run pytest, or any Python file, from assignment 2 on a Modal GPU.

This is the quick edit-run loop for kernel work: mount the current assignment2-systems, run
one command, print everything, done. Nothing is written back to the laptop or the volume.
Same image and mount as modal_bench.py.

Launch (from the workspace root):
    modal run scripts/modal_run.py --pytest "-k test_flash_forward_pass_triton -x"
    modal run scripts/modal_run.py --script scratch/flash_check.py   # any .py, path from assignment2-systems

Output is streamed as it runs. stderr is merged into stdout, so a Triton compile error shows
up in order: the one-line `error:` message comes first, then the IR dump.

Default GPU is A10G (Ampere, 24 GB, about $1.10/h). Ampere is what the fp32 tl.dot TF32 path assumes.
Each launch builds nothing new unless uv.lock changed; startup is typically under a minute.
"""

import pathlib
import shlex
import subprocess

import modal

WORKSPACE = pathlib.Path(__file__).resolve().parent.parent
A2 = WORKSPACE / "assignment2-systems"
REMOTE_A2 = "/root/cs336/assignment2-systems"

app = modal.App("cs336-a2")

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
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(A2), extra_options="--no-install-package cs336-basics")
    .env({"PYTHONUNBUFFERED": "1", "PYTHONPATH": f"{REMOTE_A2}/cs336-basics"})
    .add_local_dir(A2, remote_path=REMOTE_A2, ignore=IGNORE)
)


@app.function(image=image, gpu="A10G", timeout=900)
def run(cmd: list[str]) -> int:
    print("$", " ".join(shlex.quote(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=REMOTE_A2, stderr=subprocess.STDOUT, check=False).returncode


@app.local_entrypoint()
def main(pytest: str = "", script: str = "", gpu: str = "A10G") -> None:
    if not pytest and not script:
        raise ValueError("Both --pytest and --script can't be empty. For reverence, try '--pytest \"-k flash_forward\"")
    if script:
        cmd = ["python", script]
    else:
        cmd = ["python", "-m", "pytest", *shlex.split(pytest)]
    code = run.with_options(gpu=gpu).remote(cmd)
    if code != 0:
        raise SystemExit(f"exited with code {code}")
