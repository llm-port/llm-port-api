"""File store abstraction for chat attachment storage.

Ships with ``LocalFileStore`` — files on the local filesystem at a
configurable root path.  The ``FileStore`` protocol enables alternative
backends (e.g. MinIO) without changing consumers.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import aiofiles
import aiofiles.os

log = logging.getLogger(__name__)


@runtime_checkable
class FileStore(Protocol):
    """Minimal async file storage interface."""

    async def put_bytes(self, key: str, data: bytes) -> str:
        """Store *data* under *key*.  Returns the key."""
        ...

    async def get_bytes(self, key: str) -> bytes:
        """Retrieve previously stored bytes."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete a stored object.  Returns ``True`` if it existed."""
        ...

    async def exists(self, key: str) -> bool:
        """Check whether *key* exists."""
        ...


class LocalFileStore:
    """Store files on the local filesystem under *root*."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _resolve(self, key: str) -> Path:
        safe = Path(key)
        if safe.is_absolute() or ".." in safe.parts:
            raise ValueError(f"Invalid key: {key}")
        return self.root / safe

    async def put_bytes(self, key: str, data: bytes) -> str:
        path = self._resolve(key)
        await aiofiles.os.makedirs(path.parent, exist_ok=True)
        async with aiofiles.open(path, "wb") as fh:
            await fh.write(data)
        log.debug("Stored %d bytes at %s", len(data), path)
        return key

    async def get_bytes(self, key: str) -> bytes:
        path = self._resolve(key)
        async with aiofiles.open(path, "rb") as fh:
            return await fh.read()

    async def delete(self, key: str) -> bool:
        path = self._resolve(key)
        if not path.exists():
            return False
        await aiofiles.os.remove(path)
        parent = path.parent
        while parent != self.root:
            try:
                os.rmdir(parent)
            except OSError:
                break
            parent = parent.parent
        return True

    async def exists(self, key: str) -> bool:
        path = self._resolve(key)
        return await aiofiles.os.path.exists(path)
