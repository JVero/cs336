"""Run assignment 2's benchmarking script on a Modal GPU.

Nothing in cs336_systems changes. This wraps `python -m cs336_systems.bench_script` in a
Modal function: assignment2-systems is mounted, deps come from its uv.lock, and the CSV the
script writes is copied back to the laptop when the run ends.

Launch (from the workspace root). --flags takes the bench script's usual flags. Leave out
--device and --of_name; the launcher sets those and refuses to start if they are present.
    modal run scripts/modal_bench.py --flags "--context_length 512 --model large --forward --forward_and_back --full_step"

The CSV lands in assignment2-systems/results/<gpu>_<timestamp>.csv (override with --out) and
is also printed. It is the raw artifact of the run. The launcher then appends one row per timed
pass to assignment2-systems/results/bench_log.csv, the long-format log across launches: the
settings the run used (model, context_length, batch_size, ...), gpu, pass, mean, std, exit
code, the per-launch CSV, and the exact command the container ran. The settings come from the
bench script itself: the container parses the flags with the script's own argparse definitions
and reports every value, so defaults appear as their real values and nothing is mirrored here.
A run that dies before writing a row still gets a log line with blank numbers and its exit
code. If the log and a per-launch CSV ever disagree, the CSV wins.

The per-launch file is also copied to the cs336-runs volume at /a2/<gpu>_<timestamp>.csv
before the container exits, including after Ctrl+C, so rows written before an interrupt or a
crash are not lost. Fetch one with:
    modal volume get cs336-runs /a2/<name>.csv assignment2-systems/results/<name>.csv
Use --detach for a run you do not want tied to the terminal; then the CSV only reaches the
volume, not the laptop, and `modal app logs cs336-a2` follows it.

With --memory_profiling in --flags, the bench script also dumps one memory snapshot per timed
pass (forward.pkl, forward_and_back.pkl, full_step.pkl) into its working directory. The
launcher copies those to the volume at /a2/<gpu>_<timestamp>/<pass>.pkl and downloads them to
assignment2-systems/results/<gpu>_<timestamp>/<pass>.pkl, the directory named after the run's
CSV, ready to drop on pytorch.org/memory_viz. Fetch them by hand (after Ctrl+C or --detach) with:
    modal volume get cs336-runs /a2/<name> assignment2-systems/results/<name>
The timings such a run writes to the CSV and the log are not benchmarks: recording a stack
trace for every allocation slows each pass down. Tell those rows apart by the flag in the
log's command column.

Default GPU is H100 (80 GB, about $4/h). A100-80GB is the cheaper 80 GB option (about
$2.50/h). Anything with 24 GB (A10G, L4) fits at most the medium size.
"""

import csv
import datetime
import json
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
LOG = A2 / "results" / "bench_log.csv"
# memory_snapshot in cs336_basics.nn_utils dumps <pass>.pkl into the bench script's cwd.
SNAPSHOT_GLOB = "*.pkl"

# Settings copied into the log, by their name in the bench script's argparse namespace.
LOGGED_SETTINGS = (
    "model", "context_length", "batch_size", "num_steps", "num_repeats", "warmup_steps",
    "mixed_precision", "compile",
)
LOG_COLUMNS = ["timestamp", "gpu", *LOGGED_SETTINGS, "pass", "mean", "std", "exit_code", "run_csv", "command"]

# Run in the container with the bench script's own argv: prints the settings it would use.
SETTINGS_PROBE = """
import json
from cs336_systems.bench_script import model_parser, bench_parser
model_args, rest = model_parser.parse_known_args()
bench_args = bench_parser.parse_args(rest)
print(json.dumps({**vars(model_args), **vars(bench_args)}))
"""

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
    "**/*.pkl",  # so any .pkl in the container after a run came from that run
    "**/*.pickle",
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
def bench(flags: list[str], name: str) -> tuple[int, str, str, dict, list[str]]:
    """Returns (exit code, the CSV text the bench script wrote, the exact command run, the
    settings the script parsed from that command, or {} if parsing failed, and the file names of
    the memory snapshots it dumped, which are on the volume under /a2/<name>/)."""
    argv = ["--device", "cuda", "--of_name", REMOTE_CSV, *flags]
    cmd = ["python", "-m", "cs336_systems.bench_script", *argv]
    command = " ".join(shlex.quote(c) for c in cmd)
    print("$", command, flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
    probe = subprocess.run(["python", "-c", SETTINGS_PROBE, *argv], cwd=REMOTE_A2,
                           capture_output=True, text=True, check=False)
    settings = json.loads(probe.stdout) if probe.returncode == 0 else {}
    if not settings:
        print(f"could not read the script's settings (log columns will be blank):\n{probe.stderr}", flush=True)
    csv_path = pathlib.Path(REMOTE_CSV)
    returncode = -1
    snapshots: list[str] = []
    try:
        returncode = subprocess.run(cmd, cwd=REMOTE_A2, check=False).returncode
    finally:
        # Runs on Ctrl+C and on a crash too, so rows the script already wrote survive on the volume.
        keep = pathlib.Path(RUNS_MOUNT) / "a2"
        if csv_path.exists():
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(csv_path, keep / f"{name}.csv")
            print(f"\n{csv_path.read_text()}(also on volume cs336-runs at /a2/{name}.csv)", flush=True)
        # Snapshots dumped by --memory_profiling. IGNORE keeps .pkl out of the upload, so any
        # here came from this run. memory_snapshot dumps in a finally, so a pass that died
        # (OOM included) leaves a snapshot of the moment it died; a pass killed with the
        # container may not.
        for pkl in sorted(pathlib.Path(REMOTE_A2).glob(SNAPSHOT_GLOB)):
            (keep / name).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pkl, keep / name / pkl.name)
            snapshots.append(pkl.name)
        if snapshots:
            print(f"memory snapshots on volume cs336-runs at /a2/{name}/: {' '.join(snapshots)}", flush=True)
        if csv_path.exists() or snapshots:
            runs_vol.commit()
    return returncode, csv_path.read_text() if csv_path.exists() else "", command, settings, snapshots


def bench_rows(csv_text: str) -> list[tuple[str, str, str]]:
    """(pass, mean, std) per timed pass from the CSV bench_script writes. Empty if header-only.

    The header is `label,<pass> mean,<pass> std,...`; a data row can be shorter than the header
    when the script dropped a pass after writing it, so missing cells stay blank.
    """
    lines = [line for line in csv_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    header = lines[0].split(",")[1:]
    passes = [column[: -len(" mean")] for column in header if column.endswith(" mean")]
    rows = []
    for line in lines[1:]:
        cells = line.split(",")[1:]
        for i, name in enumerate(passes):
            pair = cells[2 * i : 2 * i + 2]
            rows.append((name, *pair, *[""] * (2 - len(pair))))
    return rows


def append_log(log: pathlib.Path, base: dict[str, str], rows: list[tuple[str, str, str]]) -> int:
    """Append one line per timed pass (or one blank-numbered line if there were none). Returns the count."""
    lines = rows or [("", "", "")]
    log.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log.exists()
    with log.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        for name, mean, std in lines:
            writer.writerow({**base, "pass": name, "mean": mean, "std": std})
    return len(lines)


def fetch_snapshots(name: str, snapshots: list[str], dest: pathlib.Path) -> None:
    """Download the run's memory snapshots from the volume (/a2/<name>/) into dest/."""
    dest.mkdir(parents=True, exist_ok=True)
    for snap in snapshots:
        with (dest / snap).open("wb") as f:
            for chunk in runs_vol.read_file(f"a2/{name}/{snap}"):
                f.write(chunk)
        print(f"wrote {dest / snap}")


@app.local_entrypoint()
def main(flags: str, gpu: str = "H100", out: str = "", timeout_hours: float = 1.0) -> None:
    flag_list = shlex.split(flags)
    for owned in ("--device", "--of_name"):
        if owned in flag_list:
            raise SystemExit(f"{owned} is set by the launcher; drop it from --flags (use --out for the local CSV path)")
    started = datetime.datetime.now()
    name = f"{gpu}_{started.strftime('%Y%m%d-%H%M%S')}"
    path = pathlib.Path(out) if out else A2 / "results" / f"{name}.csv"
    snapshot_dir = path.with_suffix("")
    fn = bench.with_options(gpu=gpu, timeout=int(timeout_hours * 3600))
    try:
        returncode, csv_text, command, settings, snapshots = fn.remote(flag_list, name)
    except KeyboardInterrupt:
        print(f"\ninterrupted; rows written before that are on the volume. Fetch with:\n"
              f"  modal volume get cs336-runs /a2/{name}.csv {path}")
        if "--memory_profiling" in flag_list:
            print(f"and the snapshots of the passes that finished with:\n"
                  f"  modal volume get cs336-runs /a2/{name} {snapshot_dir}")
        raise
    if csv_text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(csv_text)
        print(f"\n{csv_text}wrote {path}")
    else:
        print("\nno CSV came back")
    if snapshots:
        fetch_snapshots(name, snapshots, snapshot_dir)
    elif "--memory_profiling" in flag_list:
        print("no memory snapshots came back")
    resolved = path.resolve()
    base = {
        "timestamp": started.isoformat(timespec="seconds"),
        "gpu": gpu,
        **{key: settings.get(key, "") for key in LOGGED_SETTINGS},
        "exit_code": returncode,
        "run_csv": str(resolved.relative_to(WORKSPACE)) if resolved.is_relative_to(WORKSPACE) else str(resolved),
        "command": command,
    }
    n = append_log(LOG, base, bench_rows(csv_text))
    print(f"logged {n} row(s) to {LOG.relative_to(WORKSPACE)}")
    if returncode != 0:
        raise SystemExit(f"bench_script exited with code {returncode}")
