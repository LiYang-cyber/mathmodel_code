"""让各子目录脚本在未安装项目时也能导入 src/mathmodel。"""

from __future__ import annotations

import sys
from pathlib import Path

TEMPLATE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TEMPLATE_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"

for path in (str(SRC_ROOT), str(TEMPLATE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def repo_path(value: str | Path) -> Path:
    """相对路径统一以仓库根目录为基准。"""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path

