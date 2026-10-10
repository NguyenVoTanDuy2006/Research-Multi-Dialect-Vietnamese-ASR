#!/usr/bin/env python3
"""Convert Python percent-cell sources in nb/ into Jupyter .ipynb files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKER = re.compile(r"^# %%\s*(?:\[(markdown|code)\])?.*$", re.IGNORECASE)


def parse_percent(source: str) -> list[dict]:
    lines = source.splitlines(keepends=True)
    cells = []
    current_kind = "code"
    current = []
    found_marker = False
    for line in lines:
        match = MARKER.match(line.rstrip("\r\n"))
        if match:
            found_marker = True
            if current:
                cells.append(_cell(current_kind, current))
            current_kind = (match.group(1) or "code").lower()
            current = []
            continue
        current.append(line)
    if current:
        cells.append(_cell(current_kind, current))
    if not found_marker:
        return [_cell("code", lines)]
    return cells


def _cell(kind: str, lines: list[str]) -> dict:
    if kind == "markdown":
        source = [line[2:] if line.startswith("# ") else line[1:] if line.startswith("#") else line
                  for line in lines]
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}


def export_file(path: Path, *, overwrite: bool = False) -> Path | None:
    destination = path.with_suffix(".ipynb")
    if destination.exists() and not overwrite:
        return None
    notebook = {"cells": parse_percent(path.read_text(encoding="utf-8")),
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                              "name": "python3"},
                             "language_info": {"name": "python", "version": "3"}},
                "nbformat": 4, "nbformat_minor": 5}
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT / "nb"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    files = sorted(Path(args.root).rglob("*.py"))
    written, skipped = [], []
    for path in files:
        result = export_file(path, overwrite=args.overwrite)
        (written if result else skipped).append(str(result or path.with_suffix(".ipynb")))
    print(json.dumps({"written": written, "skipped_existing": skipped}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
