#!/usr/bin/env python3
"""Durable global train/val/test registry built from the content index.

The content index already groups near-duplicate images into content clusters.
This module turns that index into the single authority on *class* and *split*
membership::

    cluster_id, class, split, canonical_source, n_images

Every non-ambiguous content cluster appears exactly once, with one canonical
class, one canonical source and one split.  A cluster flagged as ambiguous by the
content index (or whose members disagree on the class label) cannot be labelled
safely, so it is excluded from the registry and listed in a separate report.

Split assignment is deterministic from the seed and the content-derived
``cluster_id``: within each ``(class, canonical_source)`` stratum the cluster ids
are ordered by ``sha256(f"{seed}\\0{cluster_id}")`` and cut with
:func:`dataset_io.split_sizes`, the project's 70/15/15 rule.  Re-running with the
same seed therefore reproduces byte-identical artifacts, and the input row order
(worker count, discovery order) never changes the result.

Usage
-----
    python3 scripts/global_split_registry.py
    python3 scripts/global_split_registry.py --seed 42 --out /tmp/registry.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import dataset_io
from merge_datasets import TARGET_CLASSES

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

REGISTRY_FIELDS: tuple[str, ...] = (
    "cluster_id",
    "class",
    "split",
    "canonical_source",
    "n_images",
)

EXCLUDED_FIELDS: tuple[str, ...] = (
    "cluster_id",
    "classes",
    "n_images",
    "n_target_class_images",
    "sources",
)

DEFAULT_SEED: int = 42
DEFAULT_VAL_RATIO: float = 0.15
DEFAULT_TEST_RATIO: float = 0.15

AMBIGUOUS_STATUS = "ambiguous_class"

_TARGET_CLASSES_LOWER: frozenset[str] = frozenset(
    name.lower() for name in TARGET_CLASSES
)

_REQUIRED_INDEX_COLUMNS: tuple[str, ...] = (
    "cluster_id",
    "source",
    "class",
    "cluster_class",
    "cluster_status",
)

_SPLIT_ORDER: tuple[str, ...] = ("train", "val", "test")


class RegistryError(Exception):
    """Raised when the content index cannot be turned into a split registry."""


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IndexEntry:
    cluster_id: str
    source: str
    cls: str
    cluster_class: str
    cluster_status: str


@dataclass(frozen=True)
class RegistryEntry:
    cluster_id: str
    cls: str
    split: str
    canonical_source: str
    n_images: int


@dataclass(frozen=True)
class ExcludedCluster:
    cluster_id: str
    classes: tuple[str, ...]
    n_images: int
    n_target_class_images: int
    sources: tuple[str, ...]


@dataclass(frozen=True)
class StratumCount:
    cls: str
    source: str
    train: int
    val: int
    test: int

    @property
    def total(self) -> int:
        return self.train + self.val + self.test


@dataclass(frozen=True)
class Registry:
    entries: tuple[RegistryEntry, ...]
    excluded: tuple[ExcludedCluster, ...]
    counts: tuple[StratumCount, ...]

    @property
    def n_clusters(self) -> int:
        return len(self.entries)

    @property
    def n_images(self) -> int:
        return sum(entry.n_images for entry in self.entries)

    @property
    def n_excluded_clusters(self) -> int:
        return len(self.excluded)

    @property
    def n_excluded_images(self) -> int:
        return sum(cluster.n_images for cluster in self.excluded)

    @property
    def n_excluded_target_images(self) -> int:
        return sum(
            cluster.n_target_class_images for cluster in self.excluded
        )


# --------------------------------------------------------------------------- #
# Index reading
# --------------------------------------------------------------------------- #


def read_content_index(path: Path) -> list[IndexEntry]:
    """Read ``content_index.csv`` into :class:`IndexEntry` rows.

    Only the columns needed to build the registry are required; extra columns
    (``path``, ``md5``, hashes) are ignored.
    """
    entries: list[IndexEntry] = []
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            missing = [
                name
                for name in _REQUIRED_INDEX_COLUMNS
                if name not in fieldnames
            ]
            if missing:
                raise RegistryError(
                    "colunas obrigatórias ausentes no índice: "
                    + ", ".join(missing)
                )
            for line, row in enumerate(reader, start=2):
                cluster_id = (row.get("cluster_id") or "").strip()
                cls = (row.get("class") or "").strip()
                if not cluster_id:
                    raise RegistryError(f"cluster_id vazio na linha {line}")
                if not cls:
                    raise RegistryError(f"class vazia na linha {line}")
                entries.append(
                    IndexEntry(
                        cluster_id=cluster_id,
                        source=(row.get("source") or "").strip(),
                        cls=cls,
                        cluster_class=(row.get("cluster_class") or "").strip(),
                        cluster_status=(
                            row.get("cluster_status") or ""
                        ).strip(),
                    )
                )
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise RegistryError(
            f"não foi possível ler o índice {path}: {exc}"
        ) from exc
    return entries


# --------------------------------------------------------------------------- #
# Split assignment
# --------------------------------------------------------------------------- #


def canonical_source(sources: Sequence[str]) -> str:
    """Most frequent source in a cluster; ties pick the lexicographically first."""
    if not sources:
        raise RegistryError("cluster sem source para atribuição canônica")
    counts = Counter(sources)
    return min(counts, key=lambda source: (-counts[source], source))


def split_key(seed: int, cluster_id: str) -> bytes:
    """Deterministic, content-addressed ordering key for ``cluster_id``."""
    return hashlib.sha256(f"{seed}\0{cluster_id}".encode("utf-8")).digest()


def _validate_ratios(val_ratio: float, test_ratio: float) -> None:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise RegistryError(
            "razões de validação inválidas: exige 0 <= val, 0 <= test e "
            f"val + test < 1 (recebido val={val_ratio}, test={test_ratio})"
        )


def assign_splits(
    groups: Mapping[tuple[str, str], Sequence[str]],
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    """Assign one split to every cluster id, stratified by ``(class, source)``."""
    _validate_ratios(val_ratio, test_ratio)
    assignment: dict[str, str] = {}
    for key in sorted(groups):
        cluster_ids = sorted(
            groups[key], key=lambda cid: (split_key(seed, cid), cid)
        )
        n_train, n_val, _ = dataset_io.split_sizes(
            len(cluster_ids), val_ratio, test_ratio
        )
        for index, cluster_id in enumerate(cluster_ids):
            if index < n_train:
                split = "train"
            elif index < n_train + n_val:
                split = "val"
            else:
                split = "test"
            assignment[cluster_id] = split
    return assignment


# --------------------------------------------------------------------------- #
# Registry construction
# --------------------------------------------------------------------------- #


def build_registry(
    entries: Sequence[IndexEntry],
    seed: int = DEFAULT_SEED,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> Registry:
    """Build the global registry, excluding ambiguous multi-class clusters."""
    grouped: dict[str, list[IndexEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.cluster_id].append(entry)

    excluded: list[ExcludedCluster] = []
    cluster_classes: dict[str, str] = {}
    cluster_sources: dict[str, str] = {}
    cluster_sizes: dict[str, int] = {}
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)

    for cluster_id in sorted(grouped):
        rows = grouped[cluster_id]
        distinct_classes = tuple(sorted({row.cls for row in rows}))
        ambiguous = len(distinct_classes) > 1 or any(
            row.cluster_status == AMBIGUOUS_STATUS for row in rows
        )
        if ambiguous:
            excluded.append(
                ExcludedCluster(
                    cluster_id=cluster_id,
                    classes=distinct_classes,
                    n_images=len(rows),
                    n_target_class_images=sum(
                        1
                        for row in rows
                        if row.cls.lower() in _TARGET_CLASSES_LOWER
                    ),
                    sources=tuple(sorted({row.source for row in rows})),
                )
            )
            continue

        canonical_classes = {row.cluster_class for row in rows}
        if len(canonical_classes) != 1 or "" in canonical_classes:
            raise RegistryError(
                "cluster "
                f"{cluster_id} não possui uma cluster_class canônica única: "
                f"{sorted(canonical_classes)}"
            )
        cls = next(iter(canonical_classes))
        source = canonical_source([row.source for row in rows])

        cluster_classes[cluster_id] = cls
        cluster_sources[cluster_id] = source
        cluster_sizes[cluster_id] = len(rows)
        strata[(cls, source)].append(cluster_id)

    assignment = assign_splits(strata, seed, val_ratio, test_ratio)

    registry_entries = tuple(
        RegistryEntry(
            cluster_id=cluster_id,
            cls=cluster_classes[cluster_id],
            split=assignment[cluster_id],
            canonical_source=cluster_sources[cluster_id],
            n_images=cluster_sizes[cluster_id],
        )
        for cluster_id in sorted(cluster_classes)
    )

    split_counts: dict[tuple[str, str], dict[str, int]] = {}
    for entry in registry_entries:
        bucket = split_counts.setdefault(
            (entry.cls, entry.canonical_source),
            {split: 0 for split in _SPLIT_ORDER},
        )
        bucket[entry.split] += 1

    counts = tuple(
        StratumCount(
            cls=key[0],
            source=key[1],
            train=bucket["train"],
            val=bucket["val"],
            test=bucket["test"],
        )
        for key, bucket in sorted(split_counts.items())
    )

    return Registry(
        entries=registry_entries,
        excluded=tuple(sorted(excluded, key=lambda item: item.cluster_id)),
        counts=counts,
    )


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


def write_registry(
    registry: Registry,
    out: Path,
    excluded_report: Path,
) -> None:
    """Write the registry and the excluded-cluster report as deterministic CSVs."""
    _write_rows(
        out,
        REGISTRY_FIELDS,
        [
            {
                "cluster_id": entry.cluster_id,
                "class": entry.cls,
                "split": entry.split,
                "canonical_source": entry.canonical_source,
                "n_images": entry.n_images,
            }
            for entry in registry.entries
        ],
    )
    _write_rows(
        excluded_report,
        EXCLUDED_FIELDS,
        [
            {
                "cluster_id": cluster.cluster_id,
                "classes": ";".join(cluster.classes),
                "n_images": cluster.n_images,
                "n_target_class_images": cluster.n_target_class_images,
                "sources": ";".join(cluster.sources),
            }
            for cluster in registry.excluded
        ],
    )


def format_summary(registry: Registry) -> str:
    """Deterministic, human-readable summary of the registry."""
    split_clusters = {split: 0 for split in _SPLIT_ORDER}
    split_images = {split: 0 for split in _SPLIT_ORDER}
    for entry in registry.entries:
        split_clusters[entry.split] += 1
        split_images[entry.split] += entry.n_images

    lines = [
        f"[global-split] {registry.n_clusters} clusters incluídos, "
        f"{registry.n_images} imagens",
        f"[global-split] {registry.n_excluded_clusters} clusters excluídos, "
        f"{registry.n_excluded_images} imagens "
        f"({registry.n_excluded_target_images} de classes-alvo)",
    ]
    for split in _SPLIT_ORDER:
        lines.append(
            f"[global-split] {split}: {split_clusters[split]} clusters / "
            f"{split_images[split]} imagens"
        )
    for count in registry.counts:
        lines.append(
            f"[global-split] {count.cls} @ {count.source}: "
            f"n={count.total} train={count.train} val={count.val} "
            f"test={count.test}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the durable global split registry (one canonical class, "
            "source and split per content cluster) from the content index."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--content-index",
        type=Path,
        default=REPO_ROOT / "data" / "content_index.csv",
        help="content index CSV to read",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "global_split_registry.csv",
        help="registry artifact path",
    )
    parser.add_argument(
        "--excluded-report",
        type=Path,
        default=(
            REPO_ROOT
            / "reports"
            / "global_split_registry"
            / "excluded_ambiguous_clusters.csv"
        ),
        help="CSV listing ambiguous multi-class clusters that were excluded",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--val-ratio", type=float, default=DEFAULT_VAL_RATIO
    )
    parser.add_argument(
        "--test-ratio", type=float, default=DEFAULT_TEST_RATIO
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        entries = read_content_index(args.content_index)
        registry = build_registry(
            entries, args.seed, args.val_ratio, args.test_ratio
        )
    except RegistryError as exc:
        print(f"[erro] {exc}", file=sys.stderr)
        return 2

    write_registry(registry, args.out, args.excluded_report)
    print(format_summary(registry))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
