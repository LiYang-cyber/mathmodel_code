import json
import sys

import pytest

from mathmodel.external_timeseries import build_command, run


def _config(tmp_path, **overrides):
    repository = tmp_path / "upstream"
    repository.mkdir()
    (repository / "run.py").write_text("print('placeholder')\n", encoding="utf-8")
    config = {
        "task": "external_timeseries",
        "adapter": "s_mamba",
        "repository_path": str(repository),
        "run_name": "external",
        "output_dir": str(tmp_path / "outputs"),
        "dry_run": True,
        "python": sys.executable,
        "arguments": {
            "is_training": 1,
            "model_id": "demo",
            "model": "S_Mamba",
            "data": "custom",
            "use_amp": True,
            "devices": [0, 1],
            "inverse": False,
        },
    }
    config.update(overrides)
    return config


def test_build_command_uses_argument_list_without_shell(tmp_path):
    command, repository, _ = build_command(_config(tmp_path))
    assert repository == (tmp_path / "upstream").resolve()
    assert command[:2] == [sys.executable, str(repository / "run.py")]
    assert "--model_id" in command
    assert "--use_amp" in command
    assert command[command.index("--devices") + 1 :] == ["0", "1"]
    assert "--inverse" not in command


def test_dry_run_writes_reproducibility_files(tmp_path):
    output = run(_config(tmp_path))
    command = json.loads((output / "command.json").read_text(encoding="utf-8"))
    assert command["upstream_repository"].endswith("S-D-Mamba")
    assert (output / "run.json").is_file()
    assert not (output / "stdout.log").exists()


def test_missing_required_argument_is_reported(tmp_path):
    config = _config(tmp_path)
    del config["arguments"]["model_id"]
    with pytest.raises(ValueError, match="model_id"):
        build_command(config)
