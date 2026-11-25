"""Run data preparation, Halo training and GPU scoring on Modal.

Train with a config from configs/; checkpoints are written to the Modal volume
under runs/<config>:
    modal run modal_app.py::train --config sft-qwen3-1.7b-lora

Score a split with a base model, or with a checkpoint from the volume. The
score file is written to results/ on the local machine:
    modal run modal_app.py::score --split test --name qwen3-1.7b_zeroshot_test --model Qwen/Qwen3-1.7B
    modal run modal_app.py::score --split test --name qwen3-1.7b_lora_test --model Qwen/Qwen3-1.7B \
        --adapter runs/sft-qwen3-1.7b-lora/checkpoint-908
"""

import shutil
import subprocess
from pathlib import Path

import modal

HALO_IMAGE = "public.ecr.aws/whitecircle/halo:hopper-1.3.0rc1"
PROJECT = "/root/project"
VOLUME = "/vol"
HERE = Path(__file__).parent

image = (
    modal.Image.from_registry(HALO_IMAGE)
    # The base image's entrypoint only prints a banner.
    .entrypoint([])
    .env({"HF_HOME": f"{VOLUME}/hf"})
    .add_local_dir(HERE, PROJECT, ignore=[".venv", ".git", "data", "results", "adapter", "**/__pycache__", ".pytest_cache"])
)
app = modal.App("halo-qwen3-tr-offensive", image=image)
volume = modal.Volume.from_name("halo-qwen3-tr-offensive", create_if_missing=True)


def sh(*cmd: str, cwd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


@app.function(gpu="H100", volumes={VOLUME: volume}, timeout=60 * 60)
def train(config: str, overrides: str = ""):
    """`overrides` is passed to Halo as is, for example "--max_steps=5 --save_strategy=no"."""
    sh("python", "prepare_data.py", "--out", "/data", cwd=PROJECT)
    output_dir = f"{VOLUME}/runs/{config}"
    # Checkpoints of an earlier run of the same config would otherwise stay next to the new ones.
    shutil.rmtree(output_dir, ignore_errors=True)
    # Halo resolves its scripts from its own checkout, which the image keeps in /workspace.
    sh(
        "halo", "launch", "sft", f"{PROJECT}/configs/{config}.yaml",
        f"--output_dir={output_dir}", *overrides.split(),
        cwd="/workspace",
    )  # fmt: skip
    volume.commit()


@app.function(gpu="H100", volumes={VOLUME: volume}, timeout=30 * 60)
def score_remote(split: str, model: str, adapter: str, prompt: str, chat_template: str) -> str:
    sh("python", "prepare_data.py", "--out", "/data", cwd=PROJECT)
    cmd = ["python", "score.py", "--data", f"/data/{split}.jsonl", "--out", "/tmp/scores.jsonl", "--model", model]
    cmd += ["--batch-size", "256"]
    if adapter:
        cmd += ["--adapter", f"{VOLUME}/{adapter}"]
    if prompt:
        cmd += ["--prompt", prompt]
    if chat_template:
        cmd += ["--chat-template", f"/workspace/{chat_template}"]
    sh(*cmd, cwd=PROJECT)
    return Path("/tmp/scores.jsonl").read_text(encoding="utf-8")


@app.local_entrypoint()
def score(split: str, name: str, model: str, adapter: str = "", prompt: str = "", chat_template: str = ""):
    """An empty `prompt` means the selected wording in prompts.py. `chat_template` is a
    path inside the Halo checkout of the image, for models trained with one of its templates."""
    out = HERE / "results" / f"{name}.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text(score_remote.remote(split, model, adapter, prompt, chat_template), encoding="utf-8")
    print(f"wrote {out}")
