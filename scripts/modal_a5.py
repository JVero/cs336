"""Run pytest, or any Python file, from assignment 5 on a Modal GPU.

Same idea as modal_run.py for A2: mount the current assignment5-alignment, run one command,
stream everything, done. --fetch copies one file the command wrote back to the same path
under assignment5-alignment on the laptop.

Launch (from the workspace root):
    modal run scripts/modal_a5.py --script scripts/model_interface.py
    modal run scripts/modal_a5.py --script scripts/prompting_baselines.py --fetch results/baselines.jsonl
    modal run scripts/modal_a5.py --pytest "tests/test_grpo.py -x"
    modal run scripts/modal_a5.py --script scripts/grpo.py --gpu B200:2 --timeout 7200

Default GPU is A10G (24 GB, about $1.10/h). OLMo-2-0425-1B in bf16 plus a vLLM KV cache fits.

vLLM note: VLLMServer in cs336_alignment/vllm_utils.py defaults to gpu=1, the second GPU,
because the RL loop trains on GPU 0. On a single-GPU container pass gpu=0, or vLLM will
not find a device.

Image: the staff recipe from cs336_alignment/modal_utils.py (CUDA devel base, the `gpu`
extra for vLLM and flash-attn). Hugging Face downloads are cached on the cs336-a5-hf volume,
so the model weights download once, not every launch. The wandb secret is attached so
wandb.init works if a script calls it.
"""

import pathlib
import shlex
import subprocess

import modal

WORKSPACE = pathlib.Path(__file__).resolve().parent.parent
A5 = WORKSPACE / "assignment5-alignment"
REMOTE_A5 = "/root/cs336/assignment5-alignment"
HF_CACHE = "/root/.cache/huggingface"

app = modal.App("cs336-a5")
hf_volume = modal.Volume.from_name("cs336-a5-hf", create_if_missing=True)

IGNORE = [
    "**/.venv",
    "**/__pycache__",
    "**/.pytest_cache",
    "**/.ruff_cache",
    "**/.DS_Store",
    "**/*.pdf",
    "results",
    "wandb",
]

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .uv_sync(uv_project_dir=str(A5), extras=["gpu"])
    .env({"PYTHONUNBUFFERED": "1", "PYTHONPATH": REMOTE_A5, "HF_HOME": HF_CACHE})
    .add_local_dir(A5, remote_path=REMOTE_A5, ignore=IGNORE)
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=900,
    volumes={HF_CACHE: hf_volume},
    secrets=[modal.Secret.from_name("wandb")],
)
def run(cmd: list[str], fetch: str = "") -> tuple[int, bytes | None]:
    print("$", " ".join(shlex.quote(c) for c in cmd), flush=True)
    code = subprocess.run(cmd, cwd=REMOTE_A5, stderr=subprocess.STDOUT, check=False).returncode
    hf_volume.commit()
    out = pathlib.Path(REMOTE_A5) / fetch
    # Return the file even on failure, so a partial run still comes back.
    return code, out.read_bytes() if fetch and out.is_file() else None


@app.local_entrypoint()
def main(pytest: str = "", script: str = "", args: str = "", gpu: str = "A10G", fetch: str = "", timeout: int = 900) -> None:
    if not pytest and not script:
        raise ValueError("Pass --pytest or --script, e.g. --script scripts/model_interface.py")
    if script:
        cmd = ["python", script, *shlex.split(args)]
    else:
        cmd = ["python", "-m", "pytest", *shlex.split(pytest)]
    code, data = run.with_options(gpu=gpu, timeout=timeout).remote(cmd, fetch)
    if fetch:
        if data is None:
            print(f"--fetch: {fetch} was not written on the remote")
        else:
            local = A5 / fetch
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
            print(f"fetched {fetch} -> {local}")
    if code != 0:
        raise SystemExit(f"exited with code {code}")
