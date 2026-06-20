#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder


def load_env_file(path: str) -> None:
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a model checkpoint folder to Hugging Face Hub.")
    parser.add_argument("--folder", required=True, help="Local checkpoint folder to upload.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face Hub repo id, e.g. user/model.")
    parser.add_argument("--path-in-repo", required=True, help="Destination folder inside the repo.")
    parser.add_argument("--env-file", default=".env", help="Env file containing HF_TOKEN.")
    parser.add_argument("--private", action="store_true", help="Create the repo as private if it does not exist.")
    args = parser.parse_args()

    load_env_file(args.env_file)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set.")

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(folder)

    create_repo(args.repo_id, token=token, private=args.private, exist_ok=True)

    readme = folder / "README.md"
    if readme.exists():
        upload_file(
            repo_id=args.repo_id,
            token=token,
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            commit_message=f"Update model card for {args.path_in_repo}",
        )

    upload_folder(
        repo_id=args.repo_id,
        token=token,
        folder_path=str(folder),
        path_in_repo=args.path_in_repo,
        commit_message=f"Upload {args.path_in_repo}",
    )

    api = HfApi(token=token)
    api.create_tag(
        repo_id=args.repo_id,
        tag=args.path_in_repo.replace("/", "-"),
        tag_message=f"Milestone {args.path_in_repo}",
        exist_ok=True,
    )
    print(f"Uploaded {folder} to {args.repo_id}/{args.path_in_repo}")


if __name__ == "__main__":
    main()
