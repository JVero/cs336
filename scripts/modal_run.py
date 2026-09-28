"""Run pytest, or any Python file, from assignment 2 on a Modal GPU.

This is the quick edit-run loop for kernel work: mount the current assignment2-systems, run
one command, print everything, done. Nothing is written to the volume. --fetch copies one
file the command wrote back to the same path under assignment2-systems on the laptop.
Same image and mount as modal_bench.py.

Launch (from the workspace root):
    modal run scripts/modal_run.py --pytest "-k test_flash_forward_pass_triton -x"
    modal run scripts/modal_run.py --script scratch/flash_check.py   # any .py, path from assignment2-systems
    modal run scripts/modal_run.py --script cs336_systems/naive_ddp.py --args "--backend nccl --modelsize xl" --gpu H100:2
    modal run scripts/modal_run.py --script tests/flash_benchmarking.py --fetch results/flash_bench.csv --timeout 3600

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
    .env({"PYTHONUNBUFFERED": "1", "PYTHONPATH": f"{REMOTE_A2}:{REMOTE_A2}/cs336-basics"})
    .add_local_dir(A2, remote_path=REMOTE_A2, ignore=IGNORE)
)


@app.function(image=image, gpu="A10G", timeout=900)
def run(cmd: list[str], fetch: str = "") -> tuple[int, bytes | None]:
    print("$", " ".join(shlex.quote(c) for c in cmd), flush=True)
    code = subprocess.run(cmd, cwd=REMOTE_A2, stderr=subprocess.STDOUT, check=False).returncode
    out = pathlib.Path(REMOTE_A2) / fetch
    # Return the file even on failure, so a partial sweep still comes back.
    return code, out.read_bytes() if fetch and out.is_file() else None


@app.local_entrypoint()
def main(pytest: str = "", script: str = "", args: str = "", gpu: str = "A10G", fetch: str = "", timeout: int = 900) -> None:
    if not pytest and not script:
        raise ValueError("Both --pytest and --script can't be empty. For reverence, try '--pytest \"-k flash_forward\"")
    if script:
        cmd = ["python", script, *shlex.split(args)]
    else:
        cmd = ["python", "-m", "pytest", *shlex.split(pytest)]
    code, data = run.with_options(gpu=gpu, timeout=timeout).remote(cmd, fetch)
    if fetch:
        if data is None:
            print(f"--fetch: {fetch} was not written on the remote")
        else:
            local = A2 / fetch
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
            print(f"fetched {fetch} -> {local}")
    if code != 0:
        raise SystemExit(f"exited with code {code}")
