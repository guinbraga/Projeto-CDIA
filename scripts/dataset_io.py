#!/usr/bin/env python3
"""Shared primitive layer for dataset splitting, merging and (future) registry.

This module holds the small, side-effecting building blocks that the split and
merge commands already duplicate: image discovery helpers, integrity survey,
output reset, collision-free naming, file materialisation and manifest I/O.
Higher-level planning and CLI code stay in their respective commands.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - friendly message
    raise SystemExit(
        "[erro] Pillow não está instalado. Instale com:  pip install pillow"
    ) from exc

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: Recognised image extensions (compared case-insensitively).
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
)

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

MANIFEST_FIELDS: tuple[str, ...] = (
    "split",
    "class",
    "source",
    "md5",
    "mode",
    "src_path",
    "dst_path",
)

#: Mapping used when a source image does not report a format.
_SUFFIX_FORMATS: dict[str, str] = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".bmp": "BMP",
    ".gif": "GIF",
    ".webp": "WEBP",
}


# --------------------------------------------------------------------------- #
# Discovery / survey
# --------------------------------------------------------------------------- #


def is_image(path: Path) -> bool:
    """Return True when ``path`` has a recognised image extension."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


@dataclass(frozen=True)
class ImageInfo:
    """Metadata gathered in a single read of each candidate image."""

    path: Path
    source: str
    cls: str
    md5: str
    mode: str
    width: int
    height: int


def survey_images(
    found: dict[str, dict[str, list[Path]]],
) -> tuple[dict[Path, ImageInfo], list[tuple[Path, str]]]:
    """Read each candidate once: compute md5, capture mode/size, verify integrity.

    Returns ``(infos, invalid)`` where ``infos`` maps path -> :class:`ImageInfo`
    for every readable image and ``invalid`` lists ``(path, reason)`` for the
    ones that were dropped.
    """
    infos: dict[Path, ImageInfo] = {}
    invalid: list[tuple[Path, str]] = []
    for source, classes in found.items():
        for cls, paths in classes.items():
            for path in paths:
                try:
                    data = path.read_bytes()
                except OSError as exc:
                    invalid.append((path, f"ilegível: {exc}"))
                    continue
                if not data:
                    invalid.append((path, "arquivo vazio (0 bytes)"))
                    continue
                digest = hashlib.md5(data).hexdigest()
                try:
                    with Image.open(BytesIO(data)) as im:
                        mode, size = im.mode, im.size
                        im.verify()
                except Exception as exc:  # noqa: BLE001 - report and drop
                    invalid.append((path, f"corrompido: {exc}"))
                    continue
                infos[path] = ImageInfo(
                    path, source, cls, digest, mode, size[0], size[1]
                )
    return infos, invalid


# --------------------------------------------------------------------------- #
# Sizing
# --------------------------------------------------------------------------- #


def split_sizes(n: int, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    """Split ``n`` items into (train, val, test), rounding val/test and fixing
    rounding overshoot so the counts are always non-negative and sum to ``n``."""
    n_val = int(round(n * val_ratio))
    n_test = int(round(n * test_ratio))
    while n_val + n_test > n:
        if n_test > 0 and n_test >= n_val:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


# --------------------------------------------------------------------------- #
# Materialisation
# --------------------------------------------------------------------------- #


def reset_output(output: Path, overwrite: bool) -> None:
    """Create ``output``, refusing to replace a non-empty directory by default."""
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise SystemExit(
                f"[erro] '{output}' já existe e não está vazio. "
                "Use --overwrite para substituí-lo."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def unique_name(base: str, used: set[str]) -> str:
    """Return ``base`` or ``<stem>_<n><ext>``, avoiding names already in ``used``."""
    if base not in used:
        return base
    stem, ext = os.path.splitext(base)
    counter = 1
    while f"{stem}_{counter}{ext}" in used:
        counter += 1
    return f"{stem}_{counter}{ext}"


def _format_for_suffix(dst_path: Path) -> str:
    return _SUFFIX_FORMATS.get(dst_path.suffix.lower(), "JPEG")


def write_image(
    src_path: Path,
    dst_path: Path,
    *,
    use_link: bool,
    convert_rgb: bool,
    mode: str | None,
    convert_format: str | None = None,
) -> bool:
    """Write one image to ``dst_path``. Returns True if it was re-encoded."""
    if convert_rgb and mode not in (None, "RGB"):
        with Image.open(src_path) as im:
            image_format = convert_format or im.format or _format_for_suffix(dst_path)
            save_kwargs: dict[str, object] = {}
            if image_format == "JPEG":
                save_kwargs["quality"] = 95
            im.convert("RGB").save(dst_path, format=image_format, **save_kwargs)
        return True

    if use_link:
        try:
            os.link(src_path, dst_path)
            return False
        except OSError:
            pass  # fall through to copy

    shutil.copy2(src_path, dst_path)
    return False


def materialize(
    records: dict[str, list[tuple[str, str, Path]]],
    output: Path,
    infos: dict[Path, ImageInfo],
    *,
    use_link: bool,
    convert_rgb: bool,
    convert_format: str | None = None,
    name_fn: Callable[[str, Path], str] | None = None,
) -> tuple[list[dict[str, str]], int]:
    """Write the planned images and return ``(manifest rows, re-encoded count)``."""
    manifest: list[dict[str, str]] = []
    used_names: dict[Path, set[str]] = {}
    converted = 0

    for name in SPLIT_NAMES:
        for cls, source, src_path in records[name]:
            dst_dir = output / name / cls
            dst_dir.mkdir(parents=True, exist_ok=True)

            base = name_fn(source, src_path) if name_fn is not None else src_path.name
            used = used_names.setdefault(dst_dir, set())
            filename = unique_name(base, used)
            used.add(filename)

            dst_path = dst_dir / filename
            if dst_path.exists():
                dst_path.unlink()

            info = infos.get(src_path)
            if write_image(
                src_path,
                dst_path,
                use_link=use_link,
                convert_rgb=convert_rgb,
                mode=info.mode if info else None,
                convert_format=convert_format,
            ):
                converted += 1

            manifest.append(
                {
                    "split": name,
                    "class": cls,
                    "source": source,
                    "md5": info.md5 if info else "",
                    "mode": info.mode if info else "",
                    "src_path": str(src_path),
                    "dst_path": str(dst_path),
                }
            )

    return manifest, converted


def write_manifest(manifest: list[dict[str, str]], output: Path) -> None:
    with open(output / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        writer.writerows(manifest)


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read ``manifest.csv`` from a directory or from an explicit file path."""
    manifest_path = path / "manifest.csv" if path.is_dir() else path
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
