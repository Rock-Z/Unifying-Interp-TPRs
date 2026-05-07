#!/usr/bin/env python3
"""Create W&B Bayesian sweeps for layerwise decoder-only TPE training."""

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_TMP = REPO_ROOT / ".tmp" / "layerwise_tpe_sweeps"

PARAMETER_SPECS: Dict[str, Dict[str, object]] = {
    "tpe/TrainingArguments.learning_rate": {
        "distribution": "log_uniform_values",
        "min": 1e-4,
        "max": 5e-3,
    },
    "tpe/TrainingArguments.weight_decay": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.05,
    },
    "tpe/TrainingArguments.warmup_ratio": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.3,
    },
    "tpe/TrainingArguments.per_device_train_batch_size": {
        "values": [4, 8, 16],
    },
    "tpe/TrainingArguments.num_train_epochs": {
        "values": [50, 75, 100],
    },
    "main.tpe_config.filler_dim": {
        "values": [128, 256, 512],
    },
    "main.tpe_config.role_dim": {
        "values": [2, 4, 8],
    },
}


def sanitize_param_name(name: str) -> str:
    return name.replace("/", "__")


def safe_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def write_override(
    tmp_dir: Path,
    model: str,
    sentences_path: str,
    cache_path: str,
    output_root: str,
    wandb_project: str,
    wandb_entity: str | None,
) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    safe = safe_name(model)
    out_dir = os.path.join(output_root, f"{safe}", "layerwise")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    tags_literal = "['decoder-only', 'layerwise', '%s']" % safe
    entity_literal = f"'{wandb_entity}'" if wandb_entity else 'None'

    override_path = tmp_dir / f"{safe}.gin"
    content = f"""
main.use_wandb = True
main.wandb_project = '{wandb_project}'
main.wandb_entity = {entity_literal}
main.wandb_group = 'layerwise-tpe'
main.wandb_run_name = None
main.wandb_tags = {tags_literal}
main.sentences_path = '{sentences_path}'
main.embedding_cache_path = '{cache_path}'
main.embedding_model_name = '{model}'
tpe/TrainingArguments.output_dir = '{out_dir}'
"""
    override_path.write_text(content.strip() + "\n")
    return override_path


def build_sweep_config(base_config: Path, override_path: Path, sweep_name: str) -> Dict[str, object]:
    command = [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "experiments" / "layerwise_decoder_tpe" / "train_layerwise_tpe.py"),
        str(base_config),
        str(override_path),
    ]
    parameters = {sanitize_param_name(k): v for k, v in PARAMETER_SPECS.items()}

    return {
        "name": sweep_name,
        "method": "bayes",
        "metric": {"name": "layer0_explained_variance", "goal": "maximize"},
        "parameters": parameters,
        "command": command,
    }


def run(args: argparse.Namespace) -> None:
    model = args.model
    safe = safe_name(model)
    tmp_dir = SWEEP_TMP / safe
    tmp_dir.mkdir(parents=True, exist_ok=True)

    base_config = Path(args.base_config)

    project_name = f"{args.project_prefix}-{safe}"
    override = write_override(
        tmp_dir=tmp_dir,
        model=model,
        sentences_path=args.sentences_path,
        cache_path=args.cache_path,
        output_root=args.output_root,
        wandb_project=project_name,
        wandb_entity=args.wandb_entity,
    )

    sweep_name = f"{safe}-layerwise-tpe"
    sweep_cfg = build_sweep_config(base_config, override, sweep_name)

    sweep_path = tmp_dir / "sweep.yaml"
    with sweep_path.open("w") as f:
        yaml.safe_dump(sweep_cfg, f, sort_keys=False)

    env = os.environ.copy()
    env.setdefault("WANDB_PROJECT", project_name)
    if args.wandb_entity:
        env.setdefault("WANDB_ENTITY", args.wandb_entity)

    sweep_cmd = [
        "wandb",
        "sweep",
        "--project",
        project_name,
        str(sweep_path),
    ]
    if args.wandb_entity:
        sweep_cmd[2:2] = ["--entity", args.wandb_entity]

    print("Creating sweep:", " ".join(shlex.quote(part) for part in sweep_cmd))
    sweep_proc = subprocess.run(sweep_cmd, check=True, capture_output=True, text=True, env=env)
    sweep_id = None
    for line in (sweep_proc.stdout or "").splitlines():
        if line.strip().startswith("wandb: Created sweep with ID:"):
            sweep_id = line.strip().split()[-1]
            break
    if sweep_id is None:
        raise SystemExit("Failed to parse sweep id from wandb output.")
    print(f"Created sweep: {sweep_id}")

    agent_cmd = [
        "wandb",
        "agent",
        sweep_id,
    ]
    print("Run an agent with:", " ".join(shlex.quote(part) for part in agent_cmd))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch layerwise TPE W&B sweep")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_config", default="experiments/layerwise_decoder_tpe/configs/layerwise_tpe_base.gin")
    parser.add_argument("--sentences_path", default="data/sentences")
    parser.add_argument("--cache_path", default="data/sentences")
    parser.add_argument("--output_root", default="checkpoints/layerwise_tpe")
    parser.add_argument("--project_prefix", default="layerwise-tpe")
    parser.add_argument("--wandb_entity", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
