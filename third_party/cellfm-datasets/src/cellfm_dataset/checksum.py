"""Checksum manifest utilities for dataset release bundles."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _normalize_algorithm(algorithm: str) -> str:
    normalized = str(algorithm).strip().lower()
    if normalized not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
    return normalized


def compute_file_checksum(
    path: str | Path,
    *,
    algorithm: str = "sha256",
    chunk_size: int = 1024 * 1024,
) -> str:
    """Hash one file with a streaming digest."""

    digest = hashlib.new(_normalize_algorithm(algorithm))
    file_path = Path(path).expanduser().resolve()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_checksum_manifest(
    input_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    algorithm: str = "sha256",
) -> dict[str, Any]:
    """Build a deterministic checksum manifest for all files under one directory."""

    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"input_dir must be a directory: {root}")

    normalized_algorithm = _normalize_algorithm(algorithm)
    output_file = None if output_path is None else Path(output_path).expanduser().resolve()

    files: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if output_file is not None and path == output_file:
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "checksum": compute_file_checksum(path, algorithm=normalized_algorithm),
            }
        )

    return {
        "root_dir": str(root),
        "algorithm": normalized_algorithm,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(files),
        "files": files,
    }


def write_checksum_manifest(
    input_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    algorithm: str = "sha256",
) -> dict[str, Any]:
    """Write a checksum manifest JSON next to a release bundle."""

    root = Path(input_dir).expanduser().resolve()
    resolved_output = (
        root / "checksum_manifest.json"
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_checksum_manifest(
        root,
        output_path=resolved_output,
        algorithm=algorithm,
    )
    resolved_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_checksum_manifest(
    manifest_json: str | Path,
    *,
    root_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a checksum manifest against a local directory tree."""

    manifest_path = Path(manifest_json).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    algorithm = _normalize_algorithm(payload["algorithm"])
    root = (
        Path(payload["root_dir"]).expanduser().resolve()
        if root_dir is None
        else Path(root_dir).expanduser().resolve()
    )
    missing: list[str] = []
    mismatched: list[dict[str, str]] = []

    for entry in payload.get("files", []):
        relative_path = Path(entry["path"])
        file_path = root / relative_path
        if not file_path.exists():
            missing.append(relative_path.as_posix())
            continue
        actual_checksum = compute_file_checksum(file_path, algorithm=algorithm)
        if actual_checksum != str(entry["checksum"]):
            mismatched.append(
                {
                    "path": relative_path.as_posix(),
                    "expected": str(entry["checksum"]),
                    "actual": actual_checksum,
                }
            )

    return {
        "manifest_json": str(manifest_path),
        "root_dir": str(root),
        "algorithm": algorithm,
        "expected_files": int(payload.get("n_files", len(payload.get("files", [])))),
        "missing_files": missing,
        "mismatched_files": mismatched,
        "ok": not missing and not mismatched,
    }
