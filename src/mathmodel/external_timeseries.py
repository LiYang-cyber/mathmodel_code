"""Adapters for running research time-series repositories without vendoring them."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import prepare_output, save_manifest
from .io import write_json


@dataclass(frozen=True)
class ExternalModel:
    repository: str
    entrypoint: str
    required_arguments: tuple[str, ...]
    task: str


MODELS = {
    "s_mamba": ExternalModel(
        repository="https://github.com/wzhwzhwzh0921/S-D-Mamba",
        entrypoint="run.py",
        required_arguments=("is_training", "model_id", "model", "data"),
        task="multivariate long-term forecasting",
    ),
    "timepro": ExternalModel(
        repository="https://github.com/xwmaxwma/TimePro",
        entrypoint="run.py",
        required_arguments=("is_training", "model_id", "model", "data"),
        task="multivariate long-term forecasting",
    ),
    "stm3": ExternalModel(
        repository="https://github.com/IfReasonable/STM3_KDD26",
        entrypoint="run.py",
        required_arguments=("dataset", "model", "lag", "horizon"),
        task="long-term spatio-temporal forecasting",
    ),
    "mambasl": ExternalModel(
        repository="https://github.com/yoom618/MambaSL",
        entrypoint="run.py",
        required_arguments=("task_name", "is_training", "model_id", "model", "data"),
        task="multivariate time-series classification",
    ),
}


def _argument_tokens(arguments: dict[str, Any]) -> list[str]:
    """Convert a mapping to argparse tokens without invoking a shell."""
    tokens: list[str] = []
    for name, value in arguments.items():
        flag = f"--{name}"
        if value is None or value is False:
            continue
        if value is True:
            tokens.append(flag)
        elif isinstance(value, (list, tuple)):
            tokens.extend([flag, *(str(item) for item in value)])
        else:
            tokens.extend([flag, str(value)])
    return tokens


def build_command(config: dict[str, Any]) -> tuple[list[str], Path, ExternalModel]:
    """Validate a config and return the upstream command and repository path."""
    adapter = str(config.get("adapter", "")).lower()
    if adapter not in MODELS:
        raise ValueError(f"未知外部时序模型: {adapter!r}；可选值: {', '.join(MODELS)}")
    spec = MODELS[adapter]
    repository = Path(config["repository_path"]).expanduser().resolve()
    entrypoint = repository / spec.entrypoint
    if not entrypoint.is_file():
        raise FileNotFoundError(
            f"未找到上游入口 {entrypoint}。请先克隆 {spec.repository}，"
            "并在 repository_path 中填写本地目录。"
        )
    arguments = dict(config.get("arguments", {}))
    missing = [name for name in spec.required_arguments if name not in arguments]
    if missing:
        raise ValueError(f"{adapter} 缺少上游参数: {', '.join(missing)}")
    python = str(config.get("python", sys.executable))
    return [python, str(entrypoint), *_argument_tokens(arguments)], repository, spec


def _git_revision(repository: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def run(config: dict[str, Any]) -> Path:
    """Run an upstream model while keeping logs and metadata in this project's outputs."""
    command, repository, spec = build_command(config)
    output = prepare_output(config).resolve()
    write_json(
        {
            "command": command,
            "cwd": str(repository),
            "upstream_repository": spec.repository,
            "upstream_revision": _git_revision(repository),
            "upstream_task": spec.task,
        },
        output / "command.json",
    )
    save_manifest(
        config,
        output,
        {"adapter": config["adapter"], "upstream_repository": spec.repository},
    )
    if config.get("dry_run", False):
        return output

    with (output / "stdout.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=repository,
            check=False,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    (output / "exit_code.txt").write_text(f"{process.returncode}\n", encoding="utf-8")
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    return output


def describe_models() -> str:
    """Return registry metadata as formatted JSON for lightweight discovery."""
    payload = {
        name: {
            "repository": spec.repository,
            "task": spec.task,
        }
        for name, spec in MODELS.items()
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
