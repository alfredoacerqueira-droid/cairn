"""Content-hash manifest for incremental indexing.

Tracks per-file SHA256 hashes so ``cairn reindex --mode quick`` can skip
files whose content hasn't changed since they were last indexed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.repo import RepoManager

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1


def sha256_of_file(filepath: Path) -> str:
    """Compute the SHA256 hex digest of a file's contents."""
    return hashlib.sha256(filepath.read_bytes()).hexdigest()


class IndexManifest:
    def __init__(self, path: Path, project_id: str):
        self.path = path
        self.project_id = project_id
        self.files: dict[str, dict] = {}
        self.version = MANIFEST_VERSION

    @classmethod
    def load(cls, repo: RepoManager, project_id: str) -> IndexManifest:
        path = repo.get_manifest_path()
        manifest = cls(path, project_id)
        if not path.exists():
            return manifest
        try:
            raw = path.read_text()
        except OSError:
            return manifest
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Corrupted manifest at %s, starting fresh", path)
            return manifest
        if not isinstance(data, dict):
            return manifest
        if data.get("project_id") != project_id:
            return manifest
        manifest.version = data.get("version", MANIFEST_VERSION)
        manifest.files = data.get("files", {})
        if not isinstance(manifest.files, dict):
            manifest.files = {}
        return manifest

    def has_same_hash(self, relpath: str, sha256: str) -> bool:
        entry = self.files.get(relpath)
        if not isinstance(entry, dict):
            return False
        return entry.get("sha256") == sha256

    def set_entry(self, relpath: str, sha256: str, blocks: int) -> None:
        self.files[relpath] = {
            "sha256": sha256,
            "blocks": blocks,
            "indexed_at": int(time.time()),
        }

    def remove_entry(self, relpath: str) -> None:
        self.files.pop(relpath, None)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.version,
            "project_id": self.project_id,
            "files": self.files,
        }
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp_path.write_text(json.dumps(data, indent=2))
            os.replace(str(tmp_path), str(self.path))
        except Exception as e:
            logger.warning("Failed to save manifest to %s: %s", self.path, e)
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
