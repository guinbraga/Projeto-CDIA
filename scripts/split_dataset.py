#!/usr/bin/env python3
"""Materialise source datasets as projections of the global split registry.

An image's split and class are not chosen here: both come from the row of the
content cluster that the image belongs to.  Every image of an included cluster
is kept, so near-duplicate members always land in the same split.  Images whose
cluster is ``ambiguous_class`` are excluded.

The low-level I/O primitives (discovery, survey, materialisation, manifest) live
in ``scripts/dataset_io.py``.

Output layout
-------------
::

    <data-root>/<dataset>_split/
        train/<class>/*.jpg
        val/<class>/*.jpg
        test/<class>/*.jpg
        manifest.csv

``train_efficientnet.py`` consumes this layout directly.  The manifest adds a
``cluster_id`` column to the historical provenance fields.

Usage
-----
Rebuild all four presets (default: ``./data``)::

    python scripts/split_dataset.py

Rebuild one preset::

    python scripts/split_dataset.py --dataset trashbox

Preview the projection without writing anything::

    python scripts/split_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import dataset_io

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

#: Preset source datasets, relative to ``--data-root``.
PRESETS: dict[str, Path] = {
    "trashnet": Path("trashnet/dataset-resized"),
    "kaggle_dataset": Path("kaggle_dataset/convertido"),
    "trashbox": Path("trashbox/TrashBox_train_set"),
    "roboflow_convertido": Path("roboflow_convertido/merged"),
}

DEFAULT_DATASETS: dict[str, Path] = PRESETS

_AMBIGUOUS_STATUS = "ambiguous_class"


class ProjectionError(Exception):
    """Raised when the registry or content index cannot drive the projection."""


@dataclass(frozen=True)
class RegistryProjection:
    cls: str
    split: str


@dataclass(frozen=True)
class IndexIdentity:
    cluster_id: str
    md5: str
    cluster_status: str


@dataclass
class Projection:
    """One source dataset projected through the registry.

    ``records``, ``infos`` and ``cluster_ids`` are consumed by
    :func:`dataset_io.materialize`; the remaining fields only drive reporting.
    """

    records: dict[str, list[tuple[str, str, Path]]]
    total_readable: int
    included: int
    excluded_ambiguous: int
    invalid: list[tuple[Path, str]]
    counts: dict[tuple[str, str], int]
    infos: dict[Path, dataset_io.ImageInfo]
    cluster_ids: dict[Path, str]


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def discover_classes(
    source: Path,
    exclude: set[Path] | None = None,
) -> dict[str, list[Path]]:
    """Group images by class, where a class is a directory containing images.

    Hidden directories and any directory resolving into ``exclude`` are skipped.
    Two different directories sharing the same name are rejected because they
    would collide in the output layout.
    """
    if not source.is_dir():
        raise SystemExit(f"[erro] diretório de origem não encontrado: {source}")

    excluded = {p.resolve() for p in (exclude or set())}
    classes: dict[str, list[Path]] = {}
    origins: dict[str, Path] = {}

    for root, dirs, files in os.walk(source):
        root_path = Path(root)

        if root_path.resolve() in excluded:
            dirs[:] = []
            continue

        dirs[:] = sorted(
            d
            for d in dirs
            if not d.startswith(".") and (root_path / d).resolve() not in excluded
        )

        images = sorted(
            root_path / name for name in files if dataset_io.is_image(root_path / name)
        )
        if not images:
            continue

        cls = root_path.name
        if cls in classes:
            raise SystemExit(
                f"[erro] classe '{cls}' encontrada em mais de um diretório: "
                f"{origins[cls]} e {root_path}"
            )
        classes[cls] = images
        origins[cls] = root_path

    if not classes:
        raise SystemExit(f"[erro] nenhuma imagem de classe encontrada em {source}")

    return dict(sorted(classes.items()))


# --------------------------------------------------------------------------- #
# Registry / index input
# --------------------------------------------------------------------------- #


def _open_csv(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)
    except OSError as exc:
        raise ProjectionError(
            f"não foi possível ler {label} '{path}': {exc}"
        ) from exc


def read_registry(path: Path) -> dict[str, RegistryProjection]:
    fieldnames, rows = _open_csv(path, "o registro global")
    required = ("cluster_id", "class", "split")
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise ProjectionError(
            f"registro '{path}' sem colunas obrigatórias: {', '.join(missing)}"
        )

    registry: dict[str, RegistryProjection] = {}
    for line, raw in enumerate(rows, start=2):
        cluster_id = (raw.get("cluster_id") or "").strip()
        cls = (raw.get("class") or "").strip()
        split = (raw.get("split") or "").strip()
        if not cluster_id:
            raise ProjectionError(
                f"registro '{path}': cluster_id vazio na linha {line}"
            )
        if not cls:
            raise ProjectionError(
                f"registro '{path}': classe vazia na linha {line}"
            )
        if split not in dataset_io.SPLIT_NAMES:
            raise ProjectionError(
                f"registro '{path}': split inválido {split!r} na linha {line}"
            )
        if cluster_id in registry:
            raise ProjectionError(
                f"registro '{path}': cluster_id duplicado '{cluster_id}'"
            )
        registry[cluster_id] = RegistryProjection(cls, split)
    return registry


def read_index_identities(path: Path) -> dict[tuple[str, str], IndexIdentity]:
    fieldnames, rows = _open_csv(path, "o índice de conteúdo")
    required = ("cluster_id", "source", "path", "md5", "cluster_status")
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise ProjectionError(
            f"índice de conteúdo '{path}' sem colunas obrigatórias: "
            f"{', '.join(missing)}"
        )

    identities: dict[tuple[str, str], IndexIdentity] = {}
    for line, raw in enumerate(rows, start=2):
        source = (raw.get("source") or "").strip()
        rel_path = (raw.get("path") or "").strip()
        key = (source, rel_path)
        if key in identities:
            raise ProjectionError(
                f"índice de conteúdo '{path}': (source, path) duplicado "
                f"({source}, {rel_path}) na linha {line}"
            )
        identities[key] = IndexIdentity(
            cluster_id=(raw.get("cluster_id") or "").strip(),
            md5=(raw.get("md5") or "").strip(),
            cluster_status=(raw.get("cluster_status") or "").strip(),
        )
    return identities


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #


def project_source(
    source_name: str,
    source_root: Path,
    data_root: Path,
    identities: dict[tuple[str, str], IndexIdentity],
    registry: dict[str, RegistryProjection],
) -> Projection:
    classes = discover_classes(source_root)
    infos, invalid = dataset_io.survey_images({source_name: classes})
    records: dict[str, list[tuple[str, str, Path]]] = {
        name: [] for name in dataset_io.SPLIT_NAMES
    }
    counts: dict[tuple[str, str], int] = {}
    cluster_ids: dict[Path, str] = {}
    included = 0
    excluded_ambiguous = 0
    root = data_root.resolve()

    for path, reason in invalid:
        rel = path.resolve().relative_to(root).as_posix()
        if (source_name, rel) in identities:
            raise ProjectionError(
                f"imagem '{rel}' ({source_name}) está no índice de conteúdo "
                f"mas não pôde ser lida ({reason}); índice desatualizado"
            )

    for path in sorted(infos):
        rel = path.resolve().relative_to(root).as_posix()
        identity = identities.get((source_name, rel))
        if identity is None:
            raise ProjectionError(
                f"imagem '{rel}' ({source_name}) não está no índice de conteúdo; "
                "índice desatualizado"
            )
        if infos[path].md5 != identity.md5:
            raise ProjectionError(
                f"md5 divergente para '{rel}' ({source_name}); "
                "índice de conteúdo desatualizado"
            )
        if identity.cluster_status == _AMBIGUOUS_STATUS:
            excluded_ambiguous += 1
            continue

        projection = registry.get(identity.cluster_id)
        if projection is None:
            raise ProjectionError(
                f"cluster '{identity.cluster_id}' de '{rel}' ({source_name}) "
                "não está no registro global; registro desatualizado"
            )

        records[projection.split].append(
            (projection.cls, source_name, path)
        )
        included += 1
        count_key = (projection.split, projection.cls)
        counts[count_key] = counts.get(count_key, 0) + 1
        cluster_ids[path] = identity.cluster_id

    for name in dataset_io.SPLIT_NAMES:
        records[name].sort(key=lambda item: (item[0], str(item[2])))

    return Projection(
        records=records,
        total_readable=len(infos),
        included=included,
        excluded_ambiguous=excluded_ambiguous,
        invalid=invalid,
        counts=counts,
        infos=infos,
        cluster_ids=cluster_ids,
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def print_dataset_plan(
    dataset: str,
    projection: Projection,
    output: Path,
) -> None:
    print(f"\n[{dataset}] projeção por classe:")
    header = (
        "classe".ljust(20)
        + "total".rjust(8)
        + "train".rjust(8)
        + "val".rjust(8)
        + "test".rjust(8)
    )
    print(header)
    print("-" * len(header))

    classes = sorted({cls for _, cls in projection.counts})
    for cls in classes:
        n_train = projection.counts.get(("train", cls), 0)
        n_val = projection.counts.get(("val", cls), 0)
        n_test = projection.counts.get(("test", cls), 0)
        print(
            cls.ljust(20)
            + str(n_train + n_val + n_test).rjust(8)
            + str(n_train).rjust(8)
            + str(n_val).rjust(8)
            + str(n_test).rjust(8)
        )

    split_totals = {
        split: sum(
            count
            for (count_split, _), count in projection.counts.items()
            if count_split == split
        )
        for split in dataset_io.SPLIT_NAMES
    }
    print(
        "TOTAL".ljust(20)
        + str(projection.included).rjust(8)
        + str(split_totals["train"]).rjust(8)
        + str(split_totals["val"]).rjust(8)
        + str(split_totals["test"]).rjust(8)
    )
    print(
        f"  lidas: {projection.total_readable} | incluídas: {projection.included} | "
        f"ambíguas excluídas: {projection.excluded_ambiguous}"
    )
    if projection.invalid:
        print(
            f"  [survey] {len(projection.invalid)} imagens ilegíveis/corrompidas "
            "descartadas"
        )
        for path, reason in projection.invalid[:10]:
            print(f"         - {path}: {reason}")
    print(f"  output: {output}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materializa os datasets de origem como projeções do registro "
            "global de splits."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data",
        help="diretório que contém os datasets predefinidos",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPO_ROOT / "data" / "global_split_registry.csv",
        help="registro global de splits (cluster_id,class,split,...)",
    )
    parser.add_argument(
        "--content-index",
        type=Path,
        default=REPO_ROOT / "data" / "content_index.csv",
        help="índice de conteúdo (cluster_id,source,path,md5,...)",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=list(PRESETS),
        default=None,
        help="dataset predefinido a materializar (pode repetir; padrão: todos)",
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
        help="só mostra a projeção, sem gravar arquivos",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="tenta hardlink em vez de copiar (cai para cópia se falhar)",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        registry = read_registry(args.registry)
        identities = read_index_identities(args.content_index)
        names = args.dataset or list(DEFAULT_DATASETS)

        for dataset in names:
            source = args.data_root / PRESETS[dataset]
            output = args.data_root / f"{dataset}_split"
            projection = project_source(
                dataset, source, args.data_root, identities, registry
            )
            print_dataset_plan(dataset, projection, output)

            if args.dry_run:
                print("  [dry-run] nada foi gravado para este dataset.")
                continue

            dataset_io.reset_output(output, args.overwrite)
            manifest, converted = dataset_io.materialize(
                projection.records,
                output,
                projection.infos,
                use_link=args.link,
                convert_rgb=args.rgb,
                cluster_ids=projection.cluster_ids,
            )
            dataset_io.write_manifest(
                manifest, output, dataset_io.MANIFEST_FIELDS_WITH_CLUSTER
            )

            print(f"\n[ok] {dataset}: {len(manifest)} imagens gravadas em {output}")
            if args.rgb:
                print(f"[ok] {converted} imagens re-codificadas para RGB")
            print(f"[ok] manifesto: {output / 'manifest.csv'}")
    except ProjectionError as exc:
        print(f"[erro] {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print("\n[dry-run] nada foi gravado.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
