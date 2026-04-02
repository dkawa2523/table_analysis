from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import ArtifactBundle


def write_json_artifact(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_text_artifact(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_split_artifacts(output_dir: Path, payload: Mapping[str, Any]) -> ArtifactBundle:
    split_path = write_json_artifact(output_dir / "split.json", dict(payload))
    return ArtifactBundle(
        primary_path=split_path,
        paths={"split.json": split_path},
        metadata={"canonical_name": "split.json"},
    )
