from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelIdentity:
    label: str
    compatibility_key: str


def model_identity(
    model_root: Path, model_pack: str, *, model_files: list[Path] | None = None,
    preprocessing: dict | None = None,
) -> ModelIdentity:
    pack = model_root / "models" / model_pack
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    label = str(manifest.get("model_version", model_pack))
    contract = {key: value for key, value in manifest.items() if key not in {"model_version", "files"}}
    if preprocessing is not None:
        contract["runtime_preprocessing"] = preprocessing
    digest = hashlib.sha256()
    digest.update(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode())
    models = sorted(model_files if model_files is not None else pack.glob("*.onnx"),
                    key=lambda path: str(path.relative_to(model_root)) if path.is_relative_to(model_root) else path.name)
    if not models:
        raise FileNotFoundError(f"No ONNX models found for pack {model_pack}")
    for path in models:
        name = (str(path.relative_to(model_root)) if model_files is not None and path.is_relative_to(model_root)
                else path.name)
        digest.update(name.encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return ModelIdentity(label, f"{label}@sha256:{digest.hexdigest()}")
