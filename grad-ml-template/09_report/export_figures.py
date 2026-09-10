"""汇总实验图片并生成 Markdown 图表清单。"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import repo_path


def export_figures(source: str, output: str) -> Path:
    source_path, output_path = repo_path(source), repo_path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    figures = sorted(path for path in source_path.rglob("*") if path.suffix.lower() in {".png", ".svg", ".pdf"})
    lines = ["# 实验图表清单", ""]
    used: set[str] = set()
    for index, figure in enumerate(figures, 1):
        name = figure.name
        if name in used:
            name = f"{figure.parent.name}_{name}"
        used.add(name)
        shutil.copy2(figure, output_path / name)
        lines.extend([f"## 图 {index}: {figure.stem}", "", f"![{figure.stem}]({name})", ""])
    (output_path / "figures.md").write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总输出目录中的论文图片")
    parser.add_argument("--source", default="outputs")
    parser.add_argument("--output", default="outputs/report_figures")
    args = parser.parse_args()
    print(export_figures(args.source, args.output).resolve())


if __name__ == "__main__":
    main()

