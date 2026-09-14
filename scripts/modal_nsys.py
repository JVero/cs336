"""Profile assignment 2's benchmarking script with Nsight Systems on a Modal GPU.

Nothing in cs336_systems changes. This wraps `nsys profile -- python -m cs336_systems.bench_script`
in a Modal function: assignment2-systems is mounted, deps come from its uv.lock, nsys is installed
on the image from NVIDIA's CUDA apt repo, and the .nsys-rep and CSV the run writes are copied back
to the laptop when it ends.

Launch (from the workspace root). --flags takes the bench script's usual flags; --nsys takes the
flags for `nsys profile` (see handout section 2.1.4). Leave out --device and --of_name from --flags
and -o/--output and -f/--force-overwrite from --nsys; the launcher sets those and refuses to start
if they are present.
    modal run scripts/modal_nsys.py --gpu A10G --nsys "--trace=cuda,nvtx --sample=none" \
        --flags "--context_length 128 --models small --warmup_steps 1 --num_repeats 1 --forward"

Outputs land in assignment2-systems/results/<gpu>_<timestamp>.nsys-rep and .csv (override the
stem with --out) and the CSV is also printed. Both files are copied to the cs336-runs volume at
/a2/<gpu>_<timestamp>.* before the container exits, including after Ctrl+C, so a profile written
before an interrupt or a crash is not lost. Fetch one with:
    modal volume get cs336-runs /a2/<name>.nsys-rep assignment2-systems/results/<name>.nsys-rep
Use --detach for a run you do not want tied to the terminal; then the files only reach the volume,
not the laptop, and `modal app logs cs336-a2` follows it. Open .nsys-rep files in the Nsight
Systems desktop app.

Cheap checks before spending GPU time:
    modal run scripts/modal_nsys.py::check
builds the image if needed (CPU builder, cached per layer, the torch layer is reused) and prints
`nsys --version` from a container with no GPU, about a cent. Then do one short run on the
cheapest GPU (A10G) with the small model before switching to H100.

Modal sandboxes containers with gVisor, which does not hand out the permissions two nsys features
need: CPU sampling (perf_event_open) and --gpu-metrics-devices (hardware counters). CUDA API,
kernel, and NVTX tracing do not need them. If a run complains about sampling, add
--sample=none --cpuctxsw=none to --nsys.

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
REMOTE_REP_STEM = "/tmp/profile"  # nsys appends .nsys-rep
RUNS_MOUNT = "/runs"

# Version from the CUDA apt repo's Debian 12 tree. 2025.6.3 pairs with CUDA 13.2 and covers the
# 13.0 runtime torch 2.11 bundles; Modal hosts run driver 580.95 (CUDA 13.0), so newer works too.
NSYS_VERSION = "2025.6.3"
CUDA_KEYRING = "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb"

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
#
# nsys is not on PyPI or in Debian's archives; it comes from NVIDIA's CUDA apt repo. That layer
# sits after uv_sync so the cached torch install is reused and only this layer builds, on a CPU
# builder. The trailing `nsys --version` makes a broken install fail there, not on a GPU.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(A2), extra_options="--no-install-package cs336-basics")
    .apt_install("curl")
    .run_commands(
        f"curl -fsSL -o /tmp/cuda-keyring.deb {CUDA_KEYRING} && dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb",
        "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends"
        f" nsight-systems-{NSYS_VERSION} && rm -rf /var/lib/apt/lists/*",
        f"command -v nsys || ln -s /opt/nvidia/nsight-systems/{NSYS_VERSION}/bin/nsys /usr/local/bin/nsys",
        "nsys --version",
    )
    .env({"PYTHONUNBUFFERED": "1", "PYTHONPATH": f"{REMOTE_A2}/cs336-basics"})
    .add_local_dir(A2, remote_path=REMOTE_A2, ignore=IGNORE)
)


@app.function(image=image)
def check() -> str:
    """No GPU: prove the image builds and nsys runs. `modal run scripts/modal_nsys.py::check`."""
    version = subprocess.run(["nsys", "--version"], capture_output=True, text=True, check=True).stdout.strip()
    print(version, flush=True)
    return version


@app.function(image=image, gpu="H100", volumes={RUNS_MOUNT: runs_vol}, timeout=3600)
def profile(flags: list[str], nsys_flags: list[str], name: str) -> tuple[int, str, bytes]:
    cmd = [
        "nsys", "profile", *nsys_flags,
        "--output", REMOTE_REP_STEM, "--force-overwrite", "true",
        "--", "python", "-m", "cs336_systems.bench_script",
        "--device", "cuda",
        "--of_name", REMOTE_CSV,
        *flags,
    ]
    print("$", " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
    csv = pathlib.Path(REMOTE_CSV)
    rep = pathlib.Path(REMOTE_REP_STEM + ".nsys-rep")
    returncode = -1
    try:
        returncode = subprocess.run(cmd, cwd=REMOTE_A2, check=False).returncode
    finally:
        # Runs on Ctrl+C and on a crash too, so whatever nsys and the script already wrote survives
        # on the volume.
        kept = []
        for src in (csv, rep):
            if src.exists():
                dst = pathlib.Path(RUNS_MOUNT) / "a2" / f"{name}{src.suffix}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                kept.append(f"/a2/{dst.name}")
        if kept:
            runs_vol.commit()
        if csv.exists():
            print(f"\n{csv.read_text()}", end="", flush=True)
        if rep.exists():
            print(f"{rep.name}: {rep.stat().st_size / 1e6:.1f} MB", flush=True)
        if kept:
            print(f"(also on volume cs336-runs at {', '.join(kept)})", flush=True)
    return (
        returncode,
        csv.read_text() if csv.exists() else "",
        rep.read_bytes() if rep.exists() else b"",
    )


def _has_flag(args: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in args)


@app.local_entrypoint()
def main(flags: str, nsys: str = "", gpu: str = "H100", out: str = "", timeout_hours: float = 1.0) -> None:
    flag_list = shlex.split(flags)
    nsys_list = shlex.split(nsys)
    for owned in ("--device", "--of_name"):
        if _has_flag(flag_list, owned):
            raise SystemExit(f"{owned} is set by the launcher; drop it from --flags (use --out for the local path)")
    for owned in ("-o", "--output", "-f", "--force-overwrite"):
        if _has_flag(nsys_list, owned):
            raise SystemExit(f"{owned} is set by the launcher; drop it from --nsys (use --out for the local path)")
    name = f"{gpu}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    stem = pathlib.Path(out) if out else A2 / "results" / name
    csv_path = pathlib.Path(f"{stem}.csv")
    rep_path = pathlib.Path(f"{stem}.nsys-rep")
    fn = profile.with_options(gpu=gpu, timeout=int(timeout_hours * 3600))
    try:
        returncode, csv, rep = fn.remote(flag_list, nsys_list, name)
    except KeyboardInterrupt:
        print(f"\ninterrupted; files written before that are on the volume. Fetch with:\n"
              f"  modal volume get cs336-runs /a2/{name}.nsys-rep {rep_path}\n"
              f"  modal volume get cs336-runs /a2/{name}.csv {csv_path}")
        raise
    stem.parent.mkdir(parents=True, exist_ok=True)
    if csv:
        csv_path.write_text(csv)
        print(f"\n{csv}wrote {csv_path}")
    else:
        print("\nno CSV came back")
    if rep:
        rep_path.write_bytes(rep)
        print(f"wrote {rep_path} ({len(rep) / 1e6:.1f} MB)")
    else:
        print("no .nsys-rep came back")
    if returncode != 0:
        raise SystemExit(f"nsys/bench_script exited with code {returncode}")
