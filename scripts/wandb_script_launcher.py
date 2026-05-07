"""Render a gin template from sweep params and launch a training script.

W&B workflow:
  1) wandb sweep <sweep.yaml>
  2) wandb agent <entity>/<project>/<sweep_id>

Usage:
  uv run python scripts/wandb_script_launcher.py <train_script.py> <sweep.yaml> \
    --param=value --other_param=123
"""

import argparse
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from string import Template
from typing import Dict, List, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_TMP = REPO_ROOT / ".tmp" / "wandb_sweeps"

ENV_PARAM_MAP = {
    "WANDB_RUN_ID": "wandb_run_id",
    "WANDB_RUN_NAME": "wandb_run_name",
    "WANDB_SWEEP_ID": "wandb_sweep_id",
    "WANDB_PROJECT": "wandb_project",
    "WANDB_ENTITY": "wandb_entity",
}


def parse_cli_params(unknown_args: List[str]) -> Dict[str, str]:
    params: Dict[str, str] = {}
    idx = 0
    while idx < len(unknown_args):
        token = unknown_args[idx]
        if not token.startswith("--"):
            raise SystemExit(f"Unexpected argument '{token}'. Use --key=value or --key value.")
        token = token[2:]
        if "=" in token:
            key, value = token.split("=", 1)
            params[key] = value
            idx += 1
            continue
        if idx + 1 >= len(unknown_args):
            raise SystemExit(f"Missing value for '--{token}'.")
        next_val = unknown_args[idx + 1]
        if next_val.startswith("--"):
            raise SystemExit(f"Missing value for '--{token}'.")
        params[token] = next_val
        idx += 2
    return params


def extract_placeholders(template: str) -> List[str]:
    placeholders: List[str] = []
    for match in Template.pattern.finditer(template):
        name = match.group("named") or match.group("braced")
        if name:
            placeholders.append(name)
    return placeholders


def load_template(sweep_path: Path) -> Tuple[str, Dict[str, object]]:
    data = yaml.safe_load(sweep_path.read_text())
    if not isinstance(data, dict):
        raise SystemExit("Sweep YAML must parse to a mapping.")
    template = data.get("gin_template")
    if template is None:
        parameters = data.get("parameters", {})
        if isinstance(parameters, dict):
            param_spec = parameters.get("gin_template")
            if isinstance(param_spec, dict):
                template = param_spec.get("value")
            elif isinstance(param_spec, str):
                template = param_spec
    if template is None:
        raise SystemExit(
            "Sweep YAML must include 'gin_template' or "
            "'parameters.gin_template.value'."
        )
    if not isinstance(template, str):
        raise SystemExit("gin_template must be a string.")
    return template, data


def build_param_map(cli_params: Dict[str, str]) -> Dict[str, str]:
    params = dict(cli_params)
    for env_key, param_key in ENV_PARAM_MAP.items():
        if param_key in params:
            continue
        env_val = os.environ.get(env_key)
        if env_val:
            params[param_key] = env_val
    return params


def load_wandb_sweep_params() -> Dict[str, object]:
    sweep_param_path = os.environ.get("WANDB_SWEEP_PARAM_PATH")
    if not sweep_param_path:
        return {}
    path = Path(sweep_param_path)
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        return {}
    params: Dict[str, object] = {}
    for key, value in data.items():
        if isinstance(value, dict) and "value" in value:
            params[key] = value["value"]
        else:
            params[key] = value
    return params


def is_placeholder_value(value: object, key: str) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip() == f"${{{key}}}"


def render_template(template: str, params: Dict[str, str]) -> str:
    try:
        rendered = Template(template).substitute(params)
    except KeyError as exc:
        missing = exc.args[0]
        raise SystemExit(f"Missing template variable: {missing}") from exc
    if not rendered.endswith("\n"):
        rendered += "\n"
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a gin template from sweep params and run a training script."
    )
    parser.add_argument("train_script", help="Training script to execute.")
    parser.add_argument("sweep_config", help="Sweep YAML containing gin_template.")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running.")
    parser.add_argument("--keep-temp", action="store_true", help="Do not delete rendered gin.")
    args, unknown_args = parser.parse_known_args()

    sweep_path = Path(args.sweep_config)
    template, _ = load_template(sweep_path)
    cli_params = parse_cli_params(unknown_args)
    params = build_param_map(cli_params)
    sweep_params = load_wandb_sweep_params()
    placeholders = set(extract_placeholders(template))
    extra_cli = sorted(set(cli_params) - placeholders)
    if extra_cli:
        raise SystemExit(
            "Unexpected CLI params not used in template: "
            + ", ".join(extra_cli)
        )
    if sweep_params:
        for key in placeholders:
            if key not in params or is_placeholder_value(params[key], key):
                if key in sweep_params:
                    params[key] = sweep_params[key]
    missing = sorted(placeholders - set(params))
    if missing:
        raise SystemExit(
            "Missing template parameters: "
            + ", ".join(missing)
        )
    rendered = render_template(template, params)

    SWEEP_TMP.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".gin",
        prefix="sweep_",
        dir=SWEEP_TMP,
        delete=False,
    ) as handle:
        handle.write(rendered)
        tmp_path = Path(handle.name)

    cmd = ["uv", "run", args.train_script, str(tmp_path)]
    print(f"Rendered gin: {tmp_path}")
    print("Launching:", " ".join(shlex.quote(part) for part in cmd))

    try:
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    finally:
        if not args.keep_temp and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
