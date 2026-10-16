"""Where uploaded PDFs, figure images and generated reports actually live.

Two backends. `local` writes to the data directory, which is correct whenever
the process owns its filesystem. `s3` writes to an S3-compatible bucket, for
hosts that hand you a container with no durable disk — the free tiers of most
managed platforms.

**Reads always end at a local file path**, whichever backend is active. That is
the design decision that keeps this small: PyMuPDF opens documents by path and
`FileResponse` streams by path, so handing callers bytes would have meant
rewriting every read site and inventing a temp-file dance at each one. Instead
`local_path()` materialises the object into a cache directory on demand and
returns the path, so existing code keeps working untouched.

The cache is disposable. Losing it costs one download, never data.
"""

from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class StorageError(RuntimeError):
    """Raised when an object cannot be stored or retrieved."""


# --- backends ----------------------------------------------------------------


class LocalStorage:
    """Files on this machine's disk, addressed by key."""

    name = "local"

    def _resolve(self, key: str) -> Path:
        return Path(settings.data_dir) / key

    def put(self, key: str, data: bytes) -> str:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def put_file(self, key: str, source: Path) -> str:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if Path(source).resolve() != path.resolve():
            shutil.copyfile(source, path)
        return str(path)

    def local_path(self, key: str) -> Path | None:
        path = self._resolve(key)
        return path if path.exists() else None

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def delete(self, key: str) -> None:
        self._resolve(key).unlink(missing_ok=True)


class S3Storage:
    """An S3-compatible bucket (Cloudflare R2, Backblaze B2, MinIO, AWS)."""

    name = "s3"

    def __init__(self) -> None:
        if not settings.s3_bucket:
            raise StorageError("STORAGE_BACKEND is 's3' but S3_BUCKET is not set.")
        try:
            import boto3  # imported lazily: the local backend must not need it
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise StorageError(
                "STORAGE_BACKEND is 's3' but boto3 is not installed."
            ) from exc

        self._bucket = settings.s3_bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url or None,
            region_name=settings.s3_region or None,
            aws_access_key_id=settings.s3_access_key_id or None,
            aws_secret_access_key=settings.s3_secret_access_key or None,
        )
        self._lock = threading.Lock()

    def _object_key(self, key: str) -> str:
        prefix = settings.s3_prefix.strip("/")
        return f"{prefix}/{key}" if prefix else key

    def _cache_path(self, key: str) -> Path:
        return Path(settings.storage_cache_dir) / key

    def put(self, key: str, data: bytes) -> str:
        try:
            self._client.put_object(Bucket=self._bucket, Key=self._object_key(key), Body=data)
        except Exception as exc:
            raise StorageError(f"Could not store {key}: {exc}") from exc

        # Seed the cache with what we already hold, so the write is immediately
        # readable without a round trip back out to the bucket.
        cached = self._cache_path(key)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
        return key

    def put_file(self, key: str, source: Path) -> str:
        return self.put(key, Path(source).read_bytes())

    def local_path(self, key: str) -> Path | None:
        cached = self._cache_path(key)
        if cached.exists():
            return cached

        # One download at a time per process: two concurrent requests for the
        # same paper would otherwise race on a half-written file.
        with self._lock:
            if cached.exists():
                return cached
            cached.parent.mkdir(parents=True, exist_ok=True)
            partial = cached.with_suffix(cached.suffix + ".part")
            try:
                self._client.download_file(self._bucket, self._object_key(key), str(partial))
            except Exception:
                partial.unlink(missing_ok=True)
                logger.warning("Object %s is not retrievable", key, exc_info=True)
                return None
            # Rename only once the bytes are all there, so a crash mid-download
            # cannot leave a truncated PDF looking like a complete one.
            partial.replace(cached)
        return cached

    def exists(self, key: str) -> bool:
        if self._cache_path(key).exists():
            return True
        try:
            self._client.head_object(Bucket=self._bucket, Key=self._object_key(key))
            return True
        except Exception:
            return False

    def delete(self, key: str) -> None:
        self._cache_path(key).unlink(missing_ok=True)
        try:
            self._client.delete_object(Bucket=self._bucket, Key=self._object_key(key))
        except Exception:  # pragma: no cover - deletion is best-effort
            logger.warning("Could not delete %s from the bucket", key, exc_info=True)


# --- selection ---------------------------------------------------------------

_storage: LocalStorage | S3Storage | None = None
_select_lock = threading.Lock()


def get_storage() -> LocalStorage | S3Storage:
    global _storage
    if _storage is None:
        with _select_lock:
            if _storage is None:
                if (settings.storage_backend or "local").lower() == "s3":
                    _storage = S3Storage()
                    logger.info("Object storage: s3 bucket %s", settings.s3_bucket)
                else:
                    _storage = LocalStorage()
    return _storage


def reset_storage() -> None:
    """Drop the cached backend — used by tests and after a settings change."""
    global _storage
    _storage = None


# --- key layout --------------------------------------------------------------
#
# Keys double as the local backend's paths under `data_dir`, so an existing
# installation keeps finding the files it already wrote.


def upload_key(paper_id: str) -> str:
    return f"uploads/{paper_id}.pdf"


def figure_key(paper_id: str, figure_id: str) -> str:
    return f"figures/{paper_id}/{figure_id}.png"


def report_key(report_id: str) -> str:
    return f"reports/{report_id}.pdf"


# --- convenience -------------------------------------------------------------


def publish_report(report_id: str, rendered: Path) -> str:
    """Take a freshly rendered report PDF into storage and return its key.

    ReportLab writes to a real path, so the file is always produced locally
    first and handed over afterwards. With the local backend that hand-off is
    a no-op, because the render target is already the stored location.
    """
    key = report_key(report_id)
    get_storage().put_file(key, Path(rendered))
    return key


def forget(stored: str | None) -> None:
    """Delete whatever a database column points at, key or legacy path."""
    if not stored:
        return
    if Path(stored).is_absolute():
        Path(stored).unlink(missing_ok=True)
    else:
        get_storage().delete(stored)


def path_for(stored: str | None) -> Path | None:
    """Resolve a value held in the database to a readable local path.

    Rows written before object storage existed hold absolute filesystem paths;
    rows written since hold keys. Accepting both means no migration, and no
    paper uploaded earlier stops opening.
    """
    if not stored:
        return None

    candidate = Path(stored)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    return get_storage().local_path(stored)
