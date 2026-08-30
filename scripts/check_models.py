#!/usr/bin/env python3
"""Report cache status for every model in configs/models.yaml.

Two provider kinds are supported. Hugging Face entries are checked against
the shared HF cache and their pinned revision resolved against the Hub.
InsightFace entries live in ~/.insightface and carry no revision, so only
presence on disk is checked.

Exit code is 1 if any active model is missing.
"""
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, scan_cache_dir
from huggingface_hub.errors import (
    CacheNotFound,
    HFValidationError,
    RepositoryNotFoundError,
)

CFG = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"
INSIGHTFACE_ROOT = Path.home() / ".insightface" / "models"


def check_huggingface(api, cached, repo, revision):
    present = repo in cached

    try:
        resolved = api.repo_info(repo, revision=revision).sha[:12]
    except (RepositoryNotFoundError, HFValidationError, OSError) as error:
        resolved = "<" + type(error).__name__ + ">"

    return present, resolved


def check_insightface(name):
    pack = INSIGHTFACE_ROOT / name

    if not pack.is_dir():
        return False, "not in ~/.insightface"

    models = sorted(p.name for p in pack.glob("*.onnx"))

    if not models:
        return False, "directory present, no .onnx files"

    return True, str(len(models)) + " onnx models"


def main() -> int:
    cfg = yaml.safe_load(CFG.read_text())
    api = HfApi()

    try:
        cached = {r.repo_id for r in scan_cache_dir().repos}
    except CacheNotFound:
        cached = set()

    rc = 0

    for role, entry in cfg.get("models", {}).items():
        repo = entry["repo_id"]
        revision = entry.get("revision", "main")
        provider = entry.get("provider", "huggingface")
        status = entry.get("status", "active")

        if provider == "insightface":
            present, detail = check_insightface(repo)
        else:
            present, detail = check_huggingface(api, cached, repo, revision)
            detail = revision[:12] + " -> " + detail

        if present:
            mark = "OK  "
        elif status == "planned":
            mark = "PLAN"
        else:
            mark = "MISS"
            rc = 1

        licence = entry.get("license", "?")

        print(mark + " " + role.ljust(26) + repo.ljust(34)
              + licence.ljust(30) + detail)

    return rc


if __name__ == "__main__":
    sys.exit(main())
