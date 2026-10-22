#!/usr/bin/env python3
"""Materialise the merged dataset as a pure projection of the global registry.

The global split registry (:mod:`global_split_registry`) is the only authority
on class and split membership. This command therefore does **no** discovery,
deduplication or re-splitting: it reads

* the registry (one canonical class/split per content cluster), and
* the four source split manifests, which are already registry projections
  (``cluster_id, source, class, split, ...``),

and writes one representative image per included cluster::

    <output>/
        train/<class>/*.jpg
        val/<class>/*.jpg
        test/<class>/*.jpg
        manifest.csv

Selection rule
--------------
* ``test`` is the **full union** of every registry ``test`` cluster of a target
  class: each one is materialised exactly once, with no sampling or balancing.
* ``train``/``val`` are sampled per ``(class, source)``. Every source contributes
  the same quota per class, capped by the smallest source for that class. A
  cluster that appears in several source datasets is credited to the scarcest
  source for its class, so shared clusters never inflate one source's quota.
  The manifest's ``source`` column is that credited source (the source of the
  representative image), which is not necessarily the registry's
  ``canonical_source``.
* ``tetra_pack`` is join-all: every available ``train``/``val`` cluster is kept.
* ``--balance-classes`` (optional) additionally caps every non-join-all class to
  the smallest class total, sub-sampling its sources proportionally. It only
  changes ``train``/``val``; ``test`` is untouched.

The remaining unique representatives stay in the source splits and are
intentionally not materialised here.

Reproducibility
---------------
Every input is sorted before use and sampling uses ``random.Random(seed)``, so
the same registry, manifests and seed produce the same output. ``--dry-run``
prints the plan without touching the output directory.

Usage
-----
::

    python3 scripts/merge_datasets.py --dry-run
    python3 scripts/merge_datasets.py --overwrite
    python3 scripts/merge_datasets.py --balance-classes --overwrite
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import dataset_io

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

#: Canonical classes for the merged dataset (the source datasets also contain
#: other labels such as ``battery``/``e-waste``; those stay in the sources).
TARGET_CLASSES: tuple[str, ...] = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "tetra_pack",
    "organic",
    "medical",
)

#: Classes that are joined in full instead of being balanced.
JOIN_ALL_CLASSES: frozenset[str] = frozenset({"tetra_pack"})

#: Source datasets whose split manifests are registry projections.
SOURCE_DATASETS: tuple[str, ...] = (
    "trashnet",
    "kaggle_dataset",
    "trashbox",
    "roboflow_convertido",
)

DEFAULT_SEED: int = 42

_TRAIN_VAL: frozenset[str] = frozenset({"train", "val"})

_REGISTRY_REQUIRED_COLUMNS: tuple[str, ...] = ("cluster_id", "class", "split")
_MANIFEST_REQUIRED_COLUMNS: tuple[str, ...] = (
    "split",
    "class",
    "source",
    "cluster_id",
)


class MergeError(Exception):
    """Raised when the registry and the source manifests cannot be merged."""


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RegistryCluster:
    """One registry row: the canonical class/split of a content cluster."""

    cluster_id: str
    cls: str
    split: str
    canonical_source: str = ""
    n_images: int = 0


@dataclass(frozen=True)
class ManifestImage:
    """One image of a source split manifest (a member of a content cluster)."""

    cluster_id: str
    source: str
    cls: str
    split: str
    md5: str
    mode: str
    path: Path


@dataclass(frozen=True)
class MergePlan:
    """The complete, validated plan consumed by :func:`dataset_io.materialize`.

    ``records`` maps ``split -> [(class, source, src_path), ...]``.
    ``infos``/``cluster_ids`` are keyed by the representative image path.
    ``availability`` is the per-``(class, source)`` train/val cluster count after
    shared-cluster attribution; ``quota`` is the number kept per source.
    """

    records: dict[str, list[tuple[str, str, Path]]]
    infos: dict[Path, dataset_io.ImageInfo]
    cluster_ids: dict[Path, str]
    availability: dict[str, dict[str, int]]
    quota: dict[str, int]
    balance_classes: bool
    totals: dict[str, int]
    class_target: int | None = None


# --------------------------------------------------------------------------- #
# Input parsing
# --------------------------------------------------------------------------- #


def _read_csv(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    manifest = path / "manifest.csv" if path.is_dir() else path
    try:
        with open(manifest, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise MergeError(f"cannot read {label} '{path}': {exc}") from exc


def read_registry(path: Path) -> list[RegistryCluster]:
    """Read the global split registry, rejecting malformed or duplicate rows."""
    fieldnames, rows = _read_csv(path, "registry")
    missing = [
        column for column in _REGISTRY_REQUIRED_COLUMNS if column not in fieldnames
    ]
    if missing:
        raise MergeError(
            f"registry '{path}' is missing required columns: {', '.join(missing)}"
        )

    registry: dict[str, RegistryCluster] = {}
    for line, raw in enumerate(rows, start=2):
        cluster_id = (raw.get("cluster_id") or "").strip()
        cls = (raw.get("class") or "").strip()
        split = (raw.get("split") or "").strip()
        if not cluster_id:
            raise MergeError(f"registry '{path}': empty cluster_id at line {line}")
        if not cls:
            raise MergeError(f"registry '{path}': empty class at line {line}")
        if split not in dataset_io.SPLIT_NAMES:
            raise MergeError(
                f"registry '{path}': invalid split {split!r} at line {line}"
            )
        if cluster_id in registry:
            raise MergeError(
                f"registry '{path}': duplicate cluster_id '{cluster_id}' "
                f"at line {line}"
            )
        raw_n_images = (raw.get("n_images") or "").strip()
        try:
            n_images = int(raw_n_images) if raw_n_images else 0
        except ValueError as exc:
            raise MergeError(
                f"registry '{path}': invalid n_images {raw_n_images!r} "
                f"at line {line}"
            ) from exc
        registry[cluster_id] = RegistryCluster(
            cluster_id=cluster_id,
            cls=cls,
            split=split,
            canonical_source=(raw.get("canonical_source") or "").strip(),
            n_images=n_images,
        )
    return [registry[cluster_id] for cluster_id in sorted(registry)]


def read_source_manifests(paths: Sequence[Path]) -> list[ManifestImage]:
    """Read the four registry-derived source split manifests.

    The same image path may only appear once. Rows are returned sorted by
    ``(cluster_id, source, path)`` so later sampling is deterministic.
    """
    images: dict[str, ManifestImage] = {}
    for path in paths:
        fieldnames, rows = _read_csv(path, "source manifest")
        missing = [
            column
            for column in _MANIFEST_REQUIRED_COLUMNS
            if column not in fieldnames
        ]
        if missing:
            raise MergeError(
                f"source manifest '{path}' is missing required columns: "
                f"{', '.join(missing)}"
            )
        if "dst_path" not in fieldnames and "src_path" not in fieldnames:
            raise MergeError(
                f"source manifest '{path}' needs a dst_path or src_path column"
            )

        for line, raw in enumerate(rows, start=2):
            split = (raw.get("split") or "").strip()
            cls = (raw.get("class") or "").strip()
            source = (raw.get("source") or "").strip()
            cluster_id = (raw.get("cluster_id") or "").strip()
            image_path = (
                (raw.get("dst_path") or "").strip()
                or (raw.get("src_path") or "").strip()
            )
            if split not in dataset_io.SPLIT_NAMES:
                raise MergeError(
                    f"source manifest '{path}': invalid split {split!r} "
                    f"at line {line}"
                )
            if not cls or not source or not cluster_id:
                raise MergeError(
                    f"source manifest '{path}': empty class/source/cluster_id "
                    f"at line {line}"
                )
            if not image_path:
                raise MergeError(
                    f"source manifest '{path}': empty image path at line {line}"
                )

            image = ManifestImage(
                cluster_id=cluster_id,
                source=source,
                cls=cls,
                split=split,
                md5=(raw.get("md5") or "").strip(),
                mode=(raw.get("mode") or "").strip(),
                path=Path(image_path),
            )
            previous = images.get(image_path)
            if previous is not None:
                if previous != image:
                    raise MergeError(
                        f"source manifest '{path}': image '{image_path}' is "
                        "listed with conflicting metadata"
                    )
                continue
            images[image_path] = image

    return sorted(
        images.values(),
        key=lambda item: (item.cluster_id, item.source, str(item.path)),
    )


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def _group_clusters(
    registry: Sequence[RegistryCluster],
    images: Sequence[ManifestImage],
) -> dict[str, list[ManifestImage]]:
    """Group images by cluster and check that manifests agree with the registry."""
    canonical = {entry.cluster_id: entry for entry in registry}
    if len(canonical) != len(registry):
        raise MergeError("registry contains duplicate cluster ids")

    grouped: dict[str, list[ManifestImage]] = defaultdict(list)
    for image in images:
        entry = canonical.get(image.cluster_id)
        if entry is None:
            raise MergeError(
                f"cluster '{image.cluster_id}' from '{image.path}' is not in "
                "the registry; the registry is stale"
            )
        if entry.cls != image.cls:
            raise MergeError(
                f"cluster '{image.cluster_id}': source manifest says class "
                f"'{image.cls}' but the registry says '{entry.cls}'"
            )
        if entry.split != image.split:
            raise MergeError(
                f"cluster '{image.cluster_id}': source manifest says split "
                f"'{image.split}' but the registry says '{entry.split}'"
            )
        grouped[image.cluster_id].append(image)

    for entry in registry:
        if entry.cls not in TARGET_CLASSES:
            continue
        if entry.cluster_id not in grouped:
            raise MergeError(
                f"target-class cluster '{entry.cluster_id}' ({entry.cls}, "
                f"{entry.split}) has no image in the source manifests"
            )
    return grouped


def _attribute_train_val(
    registry: Sequence[RegistryCluster],
    grouped: dict[str, list[ManifestImage]],
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Credit every train/val cluster to one source, balancing shared clusters.

    A cluster that only appears in one source belongs to that source. A shared
    cluster is credited to the source that currently has the fewest clusters for
    that class, so one source's copies do not inflate the quota of every source.
    """
    assigned: dict[str, str] = {}
    availability: dict[str, dict[str, int]] = {}

    for cls in TARGET_CLASSES:
        candidates = [
            entry
            for entry in registry
            if entry.cls == cls and entry.split in _TRAIN_VAL
        ]
        if not candidates:
            continue
        counts: Counter[str] = Counter()
        exclusive: list[tuple[str, str]] = []
        shared: list[tuple[str, tuple[str, ...]]] = []

        for entry in candidates:
            sources = tuple(
                sorted({image.source for image in grouped[entry.cluster_id]})
            )
            for source in sources:
                counts.setdefault(source, 0)
            if len(sources) == 1:
                exclusive.append((entry.cluster_id, sources[0]))
            else:
                shared.append((entry.cluster_id, sources))

        for cluster_id, source in sorted(exclusive):
            counts[source] += 1
            assigned[cluster_id] = source

        for cluster_id, sources in sorted(
            shared, key=lambda item: (len(item[1]), item[0])
        ):
            source = min(sources, key=lambda name: (counts[name], name))
            counts[source] += 1
            assigned[cluster_id] = source

        availability[cls] = dict(sorted(counts.items()))

    return assigned, availability


def _select_train_val(
    assigned: dict[str, str],
    registry: Sequence[RegistryCluster],
    rng: random.Random,
) -> tuple[dict[tuple[str, str], list[str]], dict[str, int]]:
    """Select the train/val clusters kept per ``(class, source)``."""
    pools: dict[tuple[str, str], list[str]] = defaultdict(list)
    cls_of = {entry.cluster_id: entry.cls for entry in registry}
    for cluster_id, source in assigned.items():
        pools[(cls_of[cluster_id], source)].append(cluster_id)

    selection: dict[tuple[str, str], list[str]] = {}
    quota: dict[str, int] = {}
    for cls in TARGET_CLASSES:
        keys = sorted(key for key in pools if key[0] == cls)
        if not keys:
            continue
        for key in keys:
            pools[key].sort()
        if cls in JOIN_ALL_CLASSES:
            keep = sum(len(pools[key]) for key in keys)
            for key in keys:
                selection[key] = list(pools[key])
            quota[cls] = keep
            continue

        keep = min(len(pools[key]) for key in keys)
        quota[cls] = keep
        for key in keys:
            pool = pools[key]
            selection[key] = (
                list(pool) if len(pool) <= keep else rng.sample(pool, keep)
            )

    return selection, quota


def _balance_class_totals(
    selection: dict[tuple[str, str], list[str]],
    rng: random.Random,
) -> tuple[dict[tuple[str, str], list[str]], int]:
    """Cap every non-join-all class to the smallest class total.

    Sources are sub-sampled proportionally with largest-remainder rounding, so
    the per-source proportions inside a class stay intact. ``tetra_pack`` is
    never touched.
    """
    totals: Counter[str] = Counter()
    for (cls, _source), cluster_ids in selection.items():
        totals[cls] += len(cluster_ids)

    eligible = [cls for cls in totals if cls not in JOIN_ALL_CLASSES]
    if not eligible:
        return selection, 0
    target = min(totals[cls] for cls in eligible)

    balanced: dict[tuple[str, str], list[str]] = {}
    for cls in sorted(totals):
        total = totals[cls]
        keys = sorted(key for key in selection if key[0] == cls)
        if cls in JOIN_ALL_CLASSES or total <= target:
            for key in keys:
                balanced[key] = list(selection[key])
            continue

        raw = {key: target * len(selection[key]) / total for key in keys}
        keep = {key: int(raw[key]) for key in keys}
        shortfall = target - sum(keep.values())
        for key in sorted(
            keys, key=lambda name: (raw[name] - keep[name], name), reverse=True
        ):
            if shortfall == 0:
                break
            if keep[key] < len(selection[key]):
                keep[key] += 1
                shortfall -= 1
        for key in keys:
            pool = sorted(selection[key])
            balanced[key] = (
                list(pool)
                if keep[key] >= len(pool)
                else rng.sample(pool, keep[key])
            )
    return balanced, target


def _representative(
    images: Sequence[ManifestImage], preferred_source: str
) -> ManifestImage:
    """Pick the representative image of a cluster, preferring ``preferred_source``."""
    return min(
        images,
        key=lambda image: (
            image.source != preferred_source,
            image.source,
            str(image.path),
        ),
    )


def plan_merge(
    registry: Sequence[RegistryCluster],
    images: Sequence[ManifestImage],
    seed: int = DEFAULT_SEED,
    balance_classes: bool = False,
) -> MergePlan:
    """Build and validate the merged dataset plan (no files are written)."""
    grouped = _group_clusters(registry, images)
    rng = random.Random(seed)

    assigned, availability = _attribute_train_val(registry, grouped)
    selection, quota = _select_train_val(assigned, registry, rng)
    class_target: int | None = None
    if balance_classes:
        selection, class_target = _balance_class_totals(selection, rng)

    split_of = {entry.cluster_id: entry.split for entry in registry}
    canonical_source = {
        entry.cluster_id: entry.canonical_source for entry in registry
    }

    records: dict[str, list[tuple[str, str, Path]]] = {
        split: [] for split in dataset_io.SPLIT_NAMES
    }
    infos: dict[Path, dataset_io.ImageInfo] = {}
    cluster_ids: dict[Path, str] = {}

    def remember(cluster_id: str, image: ManifestImage) -> None:
        records[split_of[cluster_id]].append(
            (image.cls, image.source, image.path)
        )
        infos[image.path] = dataset_io.ImageInfo(
            path=image.path,
            source=image.source,
            cls=image.cls,
            md5=image.md5,
            mode=image.mode,
            width=0,
            height=0,
        )
        cluster_ids[image.path] = cluster_id

    for (cls, source), selected_ids in sorted(selection.items()):
        for cluster_id in selected_ids:
            if split_of[cluster_id] not in _TRAIN_VAL:
                continue
            remember(
                cluster_id, _representative(grouped[cluster_id], source)
            )

    for entry in registry:
        if entry.cls not in TARGET_CLASSES or entry.split != "test":
            continue
        remember(
            entry.cluster_id,
            _representative(grouped[entry.cluster_id], entry.canonical_source),
        )

    for split in dataset_io.SPLIT_NAMES:
        records[split].sort(key=lambda row: (row[0], row[1], str(row[2])))

    totals = {split: len(rows) for split, rows in records.items()}
    plan = MergePlan(
        records=records,
        infos=infos,
        cluster_ids=cluster_ids,
        availability=availability,
        quota=quota,
        balance_classes=balance_classes,
        totals=totals,
        class_target=class_target,
    )
    validate_plan(plan, registry)
    return plan


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_plan(plan: MergePlan, registry: Sequence[RegistryCluster]) -> None:
    """Assert the merged invariants; raise :class:`MergeError` on any violation."""
    canonical = {entry.cluster_id: entry for entry in registry}
    seen: dict[str, str] = {}

    for split, rows in plan.records.items():
        for cls, _source, path in rows:
            cluster_id = plan.cluster_ids.get(path)
            if cluster_id is None:
                raise MergeError(f"image '{path}' has no cluster id in the plan")
            if cluster_id in seen:
                raise MergeError(
                    f"cluster '{cluster_id}' appears more than once in the "
                    "merged dataset"
                )
            seen[cluster_id] = split
            entry = canonical.get(cluster_id)
            if entry is None:
                raise MergeError(
                    f"cluster '{cluster_id}' is not in the registry"
                )
            if entry.cls != cls or entry.split != split:
                raise MergeError(
                    f"cluster '{cluster_id}' materialised as ({cls}, {split}) "
                    f"but the registry says ({entry.cls}, {entry.split})"
                )

    expected_test = {
        entry.cluster_id
        for entry in registry
        if entry.cls in TARGET_CLASSES and entry.split == "test"
    }
    actual_test = {
        plan.cluster_ids[path] for _cls, _src, path in plan.records["test"]
    }
    if actual_test != expected_test:
        missing = sorted(expected_test - actual_test)
        extra = sorted(actual_test - expected_test)
        raise MergeError(
            "merged test is not the full registry test union "
            f"(missing: {missing[:5]}, extra: {extra[:5]})"
        )

    per_class: dict[str, Counter[str]] = defaultdict(Counter)
    for split in _TRAIN_VAL:
        for cls, source, _path in plan.records[split]:
            per_class[cls][source] += 1

    if plan.balance_classes:
        for cls in TARGET_CLASSES:
            if cls in JOIN_ALL_CLASSES or cls not in plan.quota:
                continue
            total = sum(per_class[cls].values())
            if total != plan.class_target:
                raise MergeError(
                    f"balanced class '{cls}' has {total} rows, expected "
                    f"{plan.class_target}"
                )
        return

    for cls in TARGET_CLASSES:
        if cls in JOIN_ALL_CLASSES:
            continue
        if cls not in plan.quota:
            continue
        quota = plan.quota[cls]
        for source, available in plan.availability.get(cls, {}).items():
            count = per_class[cls].get(source, 0)
            expected = min(available, quota)
            if count != expected:
                raise MergeError(
                    f"class '{cls}', source '{source}' contributed {count} "
                    f"train/val rows, expected {expected}"
                )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _kept_counts(plan: MergePlan) -> dict[str, dict[str, int]]:
    kept: dict[str, dict[str, int]] = defaultdict(dict)
    for split in _TRAIN_VAL:
        for cls, source, _path in plan.records[split]:
            kept[cls][source] = kept[cls].get(source, 0) + 1
    return kept


def format_plan(plan: MergePlan) -> str:
    """Render the deterministic class/source plan printed by the CLI."""
    kept = _kept_counts(plan)
    sources = sorted(
        {
            source
            for cls in plan.availability
            for source in plan.availability[cls]
        }
    )
    lines = [
        "\n[plano] train/val por classe (mantidas/disponíveis por fonte; "
        "cota = menor fonte):"
    ]
    header = (
        "classe".ljust(12)
        + "".join(s.rjust(20) for s in sources)
        + "total".rjust(10)
    )
    lines.append(header)
    lines.append("-" * len(header))
    for cls in TARGET_CLASSES:
        if cls not in plan.availability:
            continue
        cells = []
        for source in sources:
            available = plan.availability[cls].get(source, 0)
            if not available and source not in kept.get(cls, {}):
                cells.append("-".rjust(20))
            else:
                cells.append(
                    f"{kept.get(cls, {}).get(source, 0)}/{available}".rjust(20)
                )
        total = sum(kept.get(cls, {}).values())
        suffix = " (join-all)" if cls in JOIN_ALL_CLASSES else ""
        lines.append(cls.ljust(12) + "".join(cells) + str(total).rjust(10) + suffix)
    lines.append(
        f"\n[plano] test: {plan.totals['test']} clusters (união completa do registro)"
    )
    lines.append(
        f"[plano] train/val: {plan.totals['train'] + plan.totals['val']} clusters"
    )
    lines.append(f"[plano] total a materializar: {sum(plan.totals.values())} imagens")
    if plan.balance_classes:
        lines.append(
            f"[balance] cada classe não join-all limitada a {plan.class_target}"
        )
    return "\n".join(lines)


def format_result(plan: MergePlan) -> str:
    lines = ["\n[resultado] imagens por split/classe:"]
    for split in dataset_io.SPLIT_NAMES:
        per_class: Counter[str] = Counter()
        per_source: Counter[str] = Counter()
        for cls, source, _path in plan.records[split]:
            per_class[cls] += 1
            per_source[source] += 1
        lines.append(
            f"  {split:<5} total={plan.totals[split]:<7} "
            f"classes={dict(sorted(per_class.items()))}"
        )
        lines.append(f"        fontes={dict(sorted(per_source.items()))}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materializa o dataset mesclado como projeção do registro global "
            "de splits (teste = união completa; train/val balanceados por fonte)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data",
        help=(
            "diretório que contém os datasets de origem, o registro e os "
            "manifests <dataset>_split"
        ),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help=(
            "registro global de splits (cluster_id,class,split,...); "
            "padrão: <data-root>/global_split_registry.csv"
        ),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        action="append",
        default=None,
        help=(
            "manifesto de um split de origem (pode repetir; padrão: os quatro "
            "<dataset>_split sob --data-root)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "merged_dataset",
        help="diretório de saída do dataset mesclado",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="iguala o total de cada classe não join-all à menor classe",
    )
    parser.add_argument(
        "--rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="re-codifica imagens não-RGB (RGBA/P/L/CMYK) para RGB",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="apaga a saída existente antes de gravar",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="só mostra o plano, sem gravar arquivos",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="tenta hardlink em vez de copiar (cai para cópia se falhar)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_manifests = args.source_manifest or [
        args.data_root / f"{name}_split" for name in SOURCE_DATASETS
    ]
    registry_path = args.registry or args.data_root / "global_split_registry.csv"

    try:
        registry = read_registry(registry_path)
        images = read_source_manifests(source_manifests)
        plan = plan_merge(
            registry,
            images,
            seed=args.seed,
            balance_classes=args.balance_classes,
        )
    except MergeError as exc:
        print(f"[erro] {exc}", file=sys.stderr)
        return 2

    print(format_plan(plan))
    print(format_result(plan))

    if args.dry_run:
        print("\n[dry-run] nada foi copiado.")
        return 0

    dataset_io.reset_output(args.output, args.overwrite)
    manifest, converted = dataset_io.materialize(
        plan.records,
        args.output,
        plan.infos,
        use_link=args.link,
        convert_rgb=args.rgb,
        convert_format="JPEG",
        name_fn=lambda source, path: f"{source}__{path.name}",
        cluster_ids=plan.cluster_ids,
    )
    dataset_io.write_manifest(
        manifest, args.output, dataset_io.MANIFEST_FIELDS_WITH_CLUSTER
    )
    print(f"\n[ok] {len(manifest)} imagens gravadas em {args.output}")
    if args.rgb:
        print(f"[ok] {converted} imagens re-codificadas para RGB")
    print(f"[ok] manifesto: {args.output / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
