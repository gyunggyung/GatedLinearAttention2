#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def merge_lm_eval(paths: list[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {"results": {}}
    for path in paths:
        if not path.exists():
            continue
        data = load(path)
        if not merged.get("configs") and data.get("configs"):
            merged["configs"] = {}
        if not merged.get("versions") and data.get("versions"):
            merged["versions"] = {}
        if not merged.get("n-shot") and data.get("n-shot"):
            merged["n-shot"] = {}
        for key in ["results", "configs", "versions", "n-shot"]:
            if isinstance(data.get(key), dict):
                merged.setdefault(key, {}).update(data[key])
        for key in ["config", "git_hash", "date"]:
            if key in data and key not in merged:
                merged[key] = data[key]
    return merged


def merge_ruler(paths: list[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {"results": {}, "examples": {}}
    for path in paths:
        if not path.exists():
            continue
        data = load(path)
        for key in ["checkpoint", "tokenizer", "lengths", "limit"]:
            if key in data and key not in merged:
                merged[key] = data[key]
        merged["results"].update(data.get("results", {}))
        merged["examples"].update(data.get("examples", {}))
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge accelerated 10B split evaluation outputs.")
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--split_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = args.split_dir
    ckpt = args.checkpoint_dir
    dump(ckpt / "standard_lm_eval.json", merge_lm_eval(sorted(split.glob("standard_*.json"))))
    dump(ckpt / "real_world_lm_eval.json", merge_lm_eval(sorted(split.glob("real_*.json"))))
    dump(ckpt / "ruler_table3.json", merge_ruler(sorted(split.glob("ruler_*.json"))))


if __name__ == "__main__":
    main()
