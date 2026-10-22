#!/usr/bin/env python3
"""Reproducible content index over the raw source datasets.

One row per readable image, carrying the image's content identity::

    cluster_id, source, class, path, md5, dhash, phash, cluster_class,
    cluster_status

Near-duplicate clusters are the connected components of ``dHash <= 1 AND
pHash <= 1``.  That threshold and the hash functions are the project's existing
ones (``detect_duplicate_images.dhash``/``phash``, the default of
``analyze_duplicate_images.py``), so exact md5 duplicates are always members of
the same cluster.  ``cluster_id`` is the lexicographically smallest md5 of the
cluster, which makes it a content-derived, split-independent identity.

A cluster whose members disagree on the class label is a class-label conflict:
all of its rows carry ``cluster_status=ambiguous_class`` and an empty
``cluster_class`` instead of a silently chosen label.  Unreadable, empty or
undecodable images are excluded from the index and listed in a report.

``path`` is relative to ``--data-root`` and every row is sorted by
``(source, class, path)``, so re-running the command produces a byte-identical
artifact.  The index is a DVC pipeline stage (``content-index``), not a one-off
script run.

Usage
-----
    python3 scripts/content_index.py
    python3 scripts/content_index.py --data-root data --workers 4
    python3 scripts/content_index.py --out /tmp/index.csv --unreadable-report /tmp/bad.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image

import dataset_io
from analyze_duplicate_images import NJOINT_DEFAULT
from detect_duplicate_images import DEFAULT_DATASETS, dhash, phash

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

#: Raw source datasets, relative to ``--data-root``.
SOURCE_ROOTS: dict[str, Path] = {
    name: Path(relative) for name, relative in DEFAULT_DATASETS.items()
}

#: Near-duplicate threshold, shared with ``analyze_duplicate_images.py``.
CLUSTER_DHASH_MAX: int = NJOINT_DEFAULT
CLUSTER_PHASH_MAX: int = NJOINT_DEFAULT

INDEX_FIELDS: tuple[str, ...] = (
    "cluster_id",
    "source",
    "class",
    "path",
    "md5",
    "dhash",
    "phash",
    "cluster_class",
    "cluster_status",
)

UNREADABLE_FIELDS: tuple[str, ...] = ("source", "class", "path", "reason")

CLUSTER_STATUS_OK = "ok"
CLUSTER_STATUS_AMBIGUOUS = "ambiguous_class"


class ContentIndexError(Exception):
    """Raised when the configured sources cannot be indexed."""


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ImageRecord:
    source: str
    cls: str
    path: str
    md5: str
    dhash: str
    phash: str


@dataclass(frozen=True)
class UnreadableImage:
    source: str
    cls: str
    path: str
    reason: str


@dataclass(frozen=True)
class Cluster:
    cluster_id: str
    size: int
    classes: tuple[str, ...]

    @property
    def ambiguous(self) -> bool:
        return len(self.classes) > 1


@dataclass(frozen=True)
class IndexRow:
    cluster_id: str
    source: str
    cls: str
    path: str
    md5: str
    dhash: str
    phash: str
    cluster_class: str
    cluster_status: str


@dataclass(frozen=True)
class ContentIndex:
    rows: tuple[IndexRow, ...]
    unreadable: tuple[UnreadableImage, ...]
    clusters: tuple[Cluster, ...]

    @property
    def n_images(self) -> int:
        return len(self.rows)

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def n_redundant_members(self) -> int:
        return self.n_images - self.n_clusters

    @property
    def ambiguous_clusters(self) -> tuple[Cluster, ...]:
        return tuple(cluster for cluster in self.clusters if cluster.ambiguous)

    @property
    def n_ambiguous_images(self) -> int:
        return sum(cluster.size for cluster in self.ambiguous_clusters)


# --------------------------------------------------------------------------- #
# Discovery / hashing
# --------------------------------------------------------------------------- #


def discover_images(
    data_root: Path,
    sources: dict[str, Path] | None = None,
) -> list[tuple[str, str, Path, str]]:
    """Return ``(source, class, absolute path, relative path)`` for every image."""
    mapping = sources if sources is not None else SOURCE_ROOTS
    root = Path(data_root).resolve()
    jobs: list[tuple[str, str, Path, str]] = []

    for source, relative in sorted(mapping.items()):
        base = root / relative
        if not base.is_dir():
            raise ContentIndexError(f"diretório de origem não encontrado: {base}")
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(
                name for name in dirnames if not name.startswith(".")
            )
            cls = Path(dirpath).name
            for filename in sorted(filenames):
                path = Path(dirpath) / filename
                if not dataset_io.is_image(path):
                    continue
                jobs.append(
                    (source, cls, path, path.relative_to(root).as_posix())
                )
    return jobs


def _hash_one(
    job: tuple[str, str, Path, str],
) -> tuple[ImageRecord | None, UnreadableImage | None]:
    source, cls, path, relative = job
    try:
        payload = path.read_bytes()
    except OSError as exc:
        return None, UnreadableImage(source, cls, relative, f"ilegível: {exc}")
    if not payload:
        return None, UnreadableImage(source, cls, relative, "arquivo vazio (0 bytes)")
    try:
        with Image.open(BytesIO(payload)) as image:
            rgb = image.convert("RGB")
            dh = dhash(rgb).tobytes().hex()
            ph = phash(rgb).tobytes().hex()
    except Exception as exc:  # noqa: BLE001 - report and exclude
        return None, UnreadableImage(source, cls, relative, f"corrompido: {exc}")
    record = ImageRecord(
        source=source,
        cls=cls,
        path=relative,
        md5=hashlib.md5(payload).hexdigest(),
        dhash=dh,
        phash=ph,
    )
    return record, None


def hash_images(
    jobs: list[tuple[str, str, Path, str]],
    workers: int | None = None,
) -> tuple[list[ImageRecord], list[UnreadableImage]]:
    """Hash every job, returning ``(records, unreadable)`` in discovery order."""
    records: list[ImageRecord] = []
    unreadable: list[UnreadableImage] = []
    total = len(jobs)
    if not total:
        return records, unreadable

    def collect(result: tuple[ImageRecord | None, UnreadableImage | None]) -> None:
        record, bad = result
        if record is not None:
            records.append(record)
        if bad is not None:
            unreadable.append(bad)

    if workers == 1:
        for job in jobs:
            collect(_hash_one(job))
        return records, unreadable

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_hash_one, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            collect(future.result())
            if done % 2000 == 0 or done == total:
                print(
                    f"[content-index] hashed {done}/{total}",
                    file=sys.stderr,
                    flush=True,
                )
    return records, unreadable


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #


def _components(records: list[ImageRecord]) -> dict[int, list[int]]:
    """Connected components at dHash <= 1 AND pHash <= 1."""
    if CLUSTER_DHASH_MAX != 1:  # pragma: no cover - protects the lookup below
        raise ContentIndexError(
            "the neighbour lookup only implements dHash <= 1"
        )
    count = len(records)
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    dhashes = [int.from_bytes(bytes.fromhex(row.dhash), "big") for row in records]
    phashes = [int.from_bytes(bytes.fromhex(row.phash), "big") for row in records]

    # A pair at dHash distance <= 1 differs in at most one bit, so it is found
    # by looking up the hash itself and its 64 one-bit neighbours.
    by_dhash: dict[int, list[int]] = defaultdict(list)
    for index, value in enumerate(dhashes):
        by_dhash[value].append(index)

    for index, value in enumerate(dhashes):
        candidates = set(by_dhash[value])
        for bit in range(64):
            candidates.update(by_dhash.get(value ^ (1 << bit), ()))
        candidates.discard(index)
        for other in candidates:
            if (phashes[index] ^ phashes[other]).bit_count() <= CLUSTER_PHASH_MAX:
                union(index, other)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(count):
        grouped[find(index)].append(index)
    return grouped


def build_index(
    records: list[ImageRecord],
    unreadable: list[UnreadableImage],
) -> ContentIndex:
    """Assign cluster identities and class flags to already-hashed records."""
    ordered = sorted(records, key=lambda row: (row.source, row.cls, row.path))
    bad = sorted(unreadable, key=lambda row: (row.source, row.cls, row.path))

    components = _components(ordered)
    cluster_ids: dict[int, str] = {}
    cluster_classes: dict[str, tuple[str, ...]] = {}
    clusters: list[Cluster] = []
    for members in components.values():
        cluster_id = min(ordered[index].md5 for index in members)
        classes = tuple(sorted({ordered[index].cls for index in members}))
        clusters.append(Cluster(cluster_id, len(members), classes))
        cluster_classes[cluster_id] = classes
        for index in members:
            cluster_ids[index] = cluster_id
    clusters.sort(key=lambda cluster: cluster.cluster_id)

    rows: list[IndexRow] = []
    for index, record in enumerate(ordered):
        cluster_id = cluster_ids[index]
        ambiguous = len(cluster_classes[cluster_id]) > 1
        rows.append(
            IndexRow(
                cluster_id=cluster_id,
                source=record.source,
                cls=record.cls,
                path=record.path,
                md5=record.md5,
                dhash=record.dhash,
                phash=record.phash,
                cluster_class="" if ambiguous else record.cls,
                cluster_status=(
                    CLUSTER_STATUS_AMBIGUOUS if ambiguous else CLUSTER_STATUS_OK
                ),
            )
        )
    return ContentIndex(tuple(rows), tuple(bad), tuple(clusters))


def build_content_index(
    data_root: Path,
    *,
    sources: dict[str, Path] | None = None,
    workers: int | None = None,
) -> ContentIndex:
    """Discover, hash and cluster every image under ``data_root``."""
    jobs = discover_images(data_root, sources)
    records, unreadable = hash_images(jobs, workers)
    return build_index(records, unreadable)


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #


def _write_rows(
    path: Path,
    fields: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_content_index(
    index: ContentIndex,
    out: Path,
    unreadable_report: Path,
) -> None:
    _write_rows(
        out,
        INDEX_FIELDS,
        [
            {
                "cluster_id": row.cluster_id,
                "source": row.source,
                "class": row.cls,
                "path": row.path,
                "md5": row.md5,
                "dhash": row.dhash,
                "phash": row.phash,
                "cluster_class": row.cluster_class,
                "cluster_status": row.cluster_status,
            }
            for row in index.rows
        ],
    )
    _write_rows(
        unreadable_report,
        UNREADABLE_FIELDS,
        [
            {
                "source": item.source,
                "class": item.cls,
                "path": item.path,
                "reason": item.reason,
            }
            for item in index.unreadable
        ],
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the reproducible content index (md5, dHash, pHash and "
            "near-duplicate cluster identity) for the raw source datasets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data",
        help="directory containing the raw source datasets",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "content_index.csv",
        help="index artifact path",
    )
    parser.add_argument(
        "--unreadable-report",
        type=Path,
        default=REPO_ROOT / "reports" / "content_index" / "unreadable_images.csv",
        help="CSV listing unreadable/empty images that were excluded",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="parallel hashing processes",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        index = build_content_index(args.data_root, workers=args.workers)
    except ContentIndexError as exc:
        print(f"[erro] {exc}", file=sys.stderr)
        return 2

    write_content_index(index, args.out, args.unreadable_report)

    print(
        f"[content-index] {index.n_images} imagens, {index.n_clusters} clusters, "
        f"{index.n_redundant_members} membros redundantes"
    )
    print(
        f"[content-index] {len(index.ambiguous_clusters)} clusters ambíguos "
        f"({index.n_ambiguous_images} imagens)"
    )
    print(f"[content-index] {len(index.unreadable)} imagens ilegíveis excluídas")
    print(f"[content-index] índice -> {args.out}")
    print(f"[content-index] ilegíveis -> {args.unreadable_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
