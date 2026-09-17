"""Profile assignment 2's benchmarking script with Nsight Systems on a Modal GPU.

Nothing in cs336_systems changes. This wraps `nsys profile -- python -m cs336_systems.bench_script`
in a Modal function: assignment2-systems is mounted, deps come from its uv.lock, nsys is installed
on the image from NVIDIA's CUDA apt repo, and the .nsys-rep and CSV the run writes are copied back
to the laptop when it ends.

Launch (from the workspace root). --flags takes the bench script's usual flags. Leave out --device
and --of_name; the launcher sets those and refuses to start if they are present.
    modal run scripts/modal_nsys.py --gpu H100 \
        --flags "--context_length 512 --model small --forward --forward_and_back --full_step"

`nsys profile` always runs with the NSYS_BASE flags below: trace CUDA and NVTX, no CPU sampling,
and record only the bench script's "Measurement" NVTX range. bench_script wraps its warmup in a
"Warmup" range and its timed loops in "Measurement", so the warmup never enters the profile and
the stats reports below are warmup-free without any filtering. Recording starts when Measurement
is pushed. With --capture-range-end=stop it ends when Measurement is popped and the script runs
on to write its timing CSV. (The nsys default, stop-shutdown, ends the nsys session at that point
and sends the script SIGTERM, since --kill defaults to sigterm, which can land before the CSV
write.) Nothing after the first Measurement range is recorded; bench_script profiles one model
per call, so that is the whole run. The capture only matches because the launcher also sets
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 in the script's environment (NSYS_ENV below): by default nsys
compares the capture-range name only against NVTX registered strings, and torch.cuda.nvtx pushes
plain ones. Without it the range never matches and nsys writes no report; the launcher then exits
non-zero with "no .nsys-rep came back".

--nsys takes extra flags for `nsys profile`, appended after NSYS_BASE, for example
--nsys "--pytorch=autograd-nvtx" (see handout section 2.1.4). Leave out the flags NSYS_BASE
already sets and -o/--output, -f/--force-overwrite and --env-var; the launcher refuses them.

--env sets extra environment variables for the bench script, as a comma-separated list of
NAME=VALUE pairs. They travel on the one --env-var flag nsys accepts, after NSYS_ENV. The case
that needs it is memory profiling (handout memory_profiling (f)):
    modal run scripts/modal_nsys.py --gpu H100 --env "PYTORCH_NO_CUDA_MEMORY_CACHING=1" \
        --nsys "--cuda-memory-usage=true --pytorch=autograd-shapes-nvtx" \
        --flags "--model xl --context_length 1024 --batch_size 1 --num_repeats 1 --warmup_steps 0 --full_step"
--cuda-memory-usage records cudaMalloc and cudaFree, and PyTorch's caching allocator only calls
those when its pool grows, never on a free. After a warmup every step reuses the pool, so a
profile that starts at "Measurement" holds no memory events at all (xl_1024_memory on
2026-09-17 came back with no memory table). --warmup_steps 0 puts the first, pool-growing step
inside the capture; PYTORCH_NO_CUDA_MEMORY_CACHING=1 makes every tensor allocation and free a
real cudaMalloc/cudaFree, so the memory line also drops where tensors die. That step runs
slower (each cudaFree synchronizes), so its timing row is not a benchmark.

Outputs land in one directory per run, assignment2-systems/results/<model>_<context_length>/,
named from --flags (small_512 for --model small --context_length 512). It holds <name>.nsys-rep,
<name>.csv (the timing CSV, also printed) and the stats CSVs described below. The launcher
refuses to start if that directory already has files in it; pass --out with the same directory
to overwrite them on purpose, or another directory to keep both. The same files are copied to
the cs336-runs volume at /a2/<name>/ before the container exits, including after Ctrl+C, so a
profile written before an interrupt or a crash is not lost. Fetch a run with:
    modal volume get cs336-runs /a2/<name> assignment2-systems/results/
Use --detach for a run you do not want tied to the terminal; then the files only reach the volume,
not the laptop, and `modal app logs cs336-a2` follows it. Open .nsys-rep files in the Nsight
Systems desktop app.

After the profile, `nsys stats` runs in the same container and its tables come back as CSVs, one
file per report: <name>_<report>.csv next to the .nsys-rep. These are the same reports as the
desktop app's Stats System View. nsys computes them by exporting the profile to SQLite and running
one query per report; the SQLite file stays in the container. The default set below can be
replaced with --reports, a comma-separated list of names from `nsys stats --help-reports`. Times
are nanoseconds and the column headers say so.
    nvtx_sum            wall-clock of each NVTX range on the CPU (the timeit counterpart)
    nvtx_gpu_proj_sum   each NVTX range projected onto the GPU: time the GPU was busy inside it
    nvtx_kern_sum       kernels grouped by the NVTX range that launched them
    cuda_gpu_kern_sum   kernels over the whole profile

Cheap checks before spending GPU time:
    modal run scripts/modal_nsys.py::check
builds the image if needed (CPU builder, cached per layer, the torch layer is reused) and prints
`nsys --version` from a container with no GPU, about a cent. Then do one short run on the
cheapest GPU (A10G) with the small model before switching to H100.

Modal sandboxes containers with gVisor, which does not hand out the permissions two nsys features
need: CPU sampling (perf_event_open) and --gpu-metrics-devices (hardware counters). CUDA API,
kernel, and NVTX tracing do not need them. That is why NSYS_BASE has --sample=none; if a run
still complains, add --cpuctxsw=none to --nsys.

Default GPU is H100 (80 GB, about $4/h). A100-80GB is the cheaper 80 GB option (about
$2.50/h). Anything with 24 GB (A10G, L4) fits at most the medium size.
"""

import pathlib
import shlex
import shutil
import subprocess

import modal

WORKSPACE = pathlib.Path(__file__).resolve().parent.parent
A2 = WORKSPACE / "assignment2-systems"
REMOTE_A2 = "/root/cs336/assignment2-systems"
REMOTE_CSV = "/tmp/bench.csv"
REMOTE_REP_STEM = "/tmp/profile"  # nsys appends .nsys-rep; `nsys stats` appends _<report>.csv
RUNS_MOUNT = "/runs"
DEFAULT_REPORTS = "nvtx_sum,nvtx_gpu_proj_sum,nvtx_kern_sum,cuda_gpu_kern_sum"
# Every profile: CUDA + NVTX tracing, no CPU sampling (gVisor), and record only the bench script's
# "Measurement" NVTX range so the warmup stays out of the file. See the module docstring.
# NSYS_NVTX_PROFILER_REGISTER_ONLY=0: without it nsys only matches the capture range against NVTX
# *registered* strings, torch.cuda.nvtx pushes plain strings, so the range never matches and nsys
# writes no report at all (six H100 runs on 2026-09-14 came back with timing CSVs and nothing else).
NSYS_BASE = [
    "--trace=cuda,nvtx",
    "--sample=none",
    "--capture-range=nvtx",
    "--nvtx-capture=Measurement",
    "--capture-range-end=stop",
]
# Environment for the bench script. nsys takes one --env-var flag holding a comma-separated list
# of NAME=VALUE pairs; --env appends to this list.
NSYS_ENV = ["NSYS_NVTX_PROFILER_REGISTER_ONLY=0"]

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
def profile(
    flags: list[str], nsys_flags: list[str], env: list[str], name: str, reports: str
) -> tuple[int, str, bytes, dict[str, str]]:
    """Returns (exit code, timing CSV text, .nsys-rep bytes, {"_<report>.csv": CSV text, ...}).

    `nsys_flags` go after NSYS_BASE; `env` is NAME=VALUE pairs added to NSYS_ENV for the bench
    script; `reports` is the comma-separated list for `nsys stats --report`.
    """
    cmd = [
        "nsys", "profile", *NSYS_BASE, "--env-var=" + ",".join(NSYS_ENV + env), *nsys_flags,
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
        if rep.exists():
            # Writes /tmp/profile_<report>.csv per report (and /tmp/profile.sqlite, the export the
            # report queries run over, which is not copied back).
            stats_cmd = [
                "nsys", "stats", "--report", reports, "--format", "csv",
                "--output", REMOTE_REP_STEM, "--force-overwrite", "true", str(rep),
            ]
            print("$", " ".join(shlex.quote(c) for c in stats_cmd), flush=True)
            stats_rc = subprocess.run(stats_cmd, check=False).returncode
            if stats_rc != 0:
                print(f"nsys stats exited with code {stats_rc}; see its output above", flush=True)
                if returncode == 0:
                    returncode = stats_rc
    finally:
        # Runs on Ctrl+C and on a crash too, so whatever nsys and the script already wrote survives
        # on the volume. Each file is (path, suffix it gets after the run name).
        stats_files = sorted(rep.parent.glob(f"{rep.stem}_*.csv"))
        outputs = [(csv, ".csv"), (rep, ".nsys-rep")] + [(p, p.name.removeprefix(rep.stem)) for p in stats_files]
        kept = []
        for src, suffix in outputs:
            if src.exists():
                dst = pathlib.Path(RUNS_MOUNT) / "a2" / name / f"{name}{suffix}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                kept.append(f"/a2/{name}/{dst.name}")
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
        {suffix: src.read_text() for src, suffix in outputs[2:]},
    )


def _has_flag(args: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in args)


def _flag_value(args: list[str], flag: str) -> str | None:
    """Value of `--flag value` or `--flag=value` in args, None if absent."""
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


@app.local_entrypoint()
def main(
    flags: str,
    nsys: str = "",
    env: str = "",
    gpu: str = "H100",
    out: str = "",
    timeout_hours: float = 1.0,
    reports: str = DEFAULT_REPORTS,
) -> None:
    flag_list = shlex.split(flags)
    nsys_list = shlex.split(nsys)
    env_list = [e.strip() for e in env.split(",") if e.strip()]
    for owned in ("--device", "--of_name"):
        if _has_flag(flag_list, owned):
            raise SystemExit(f"{owned} is set by the launcher; drop it from --flags (use --out for the local path)")
    for owned in ("-o", "--output", "-f", "--force-overwrite"):
        if _has_flag(nsys_list, owned):
            raise SystemExit(f"{owned} is set by the launcher; drop it from --nsys (use --out for the local path)")
    if _has_flag(nsys_list, "--env-var"):
        raise SystemExit("--env-var is set by the launcher; pass the variables with --env instead")
    for owned in (base.split("=")[0] for base in NSYS_BASE):
        if _has_flag(nsys_list, owned):
            raise SystemExit(f"{owned} is set by the launcher (NSYS_BASE); drop it from --nsys or edit NSYS_BASE")
    for pair in env_list:
        if "=" not in pair or pair.startswith("="):
            raise SystemExit(f"--env entries are NAME=VALUE, comma-separated; got {pair!r}")
    report_list = ",".join(r.strip() for r in reports.split(",") if r.strip())
    if not report_list:
        raise SystemExit("--reports needs at least one report name")
    model = _flag_value(flag_list, "--model")
    ctx = _flag_value(flag_list, "--context_length")
    if model is None or ctx is None:
        raise SystemExit("--flags must include --model and --context_length; they name the run")
    run_dir = pathlib.Path(out) if out else A2 / "results" / f"{model}_{ctx}"
    name = run_dir.name
    stem = run_dir / name
    if not out and run_dir.is_dir() and any(run_dir.iterdir()):
        raise SystemExit(
            f"{run_dir} already has files in it; pass --out {run_dir} to overwrite them "
            f"or --out <other directory> to keep both"
        )
    csv_path = pathlib.Path(f"{stem}.csv")
    rep_path = pathlib.Path(f"{stem}.nsys-rep")
    fn = profile.with_options(gpu=gpu, timeout=int(timeout_hours * 3600))
    try:
        returncode, csv, rep, stats_csvs = fn.remote(flag_list, nsys_list, env_list, name, report_list)
    except KeyboardInterrupt:
        print(f"\ninterrupted; files written before that are on the volume. Fetch the run with:\n"
              f"  modal volume get cs336-runs /a2/{name} {run_dir.parent}/")
        raise
    run_dir.mkdir(parents=True, exist_ok=True)
    if csv:
        csv_path.write_text(csv)
        print(f"\n{csv}wrote {csv_path}")
    else:
        print("\nno CSV came back")
    if rep:
        rep_path.write_bytes(rep)
        print(f"wrote {rep_path} ({len(rep) / 1e6:.1f} MB)")
    for suffix, text in sorted(stats_csvs.items()):
        path = pathlib.Path(f"{stem}{suffix}")
        path.write_text(text)
        print(f"wrote {path} ({text.count(chr(10))} rows)")
    if not stats_csvs:
        print("no stats CSVs came back")
    if returncode != 0:
        raise SystemExit(f"nsys/bench_script exited with code {returncode}")
    if not rep:
        raise SystemExit("no .nsys-rep came back: nsys wrote no profile, so the capture range never "
                         "triggered; check the nsys messages in the log above")
