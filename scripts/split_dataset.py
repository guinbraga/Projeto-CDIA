#!/usr/bin/env python3
"""Split a single class-folder dataset into train/val/test (70/15/15).

Each subdirectory that directly contains images is treated as a class.  The
split is **stratified by class**: every class is shuffled and divided
independently, using the same sizing rule as ``merge_datasets.py``
(``split_sizes``), so the class distribution is preserved in every split.

Equivalence with ``merged_dataset``
-----------------------------------
``scripts/merge_datasets.py`` builds ``data/merged_dataset`` with the very same
70/15/15 procedure, stratified by ``(class, source)``.  This script applies that
procedure to one already-aggregated dataset at a time.  Each dataset is split
with a fresh ``random.Random(seed)``, therefore ``--dataset trashbox`` produces
exactly the same assignment as running the split for all three presets.  Dedupe
and cleaning are upstream (``remove_exact_duplicates.py``); this script only
splits, it does not deduplicate.

Output layout
-------------
::

    <output>/
        train/<class>/*.jpg
        val/<class>/*.jpg
        test/<class>/*.jpg
        manifest.csv

``train_efficientnet.py`` consumes this layout directly.

By default each preset under ``<data-root>`` is written to
``<data-root>/<dataset>_split``:

* ``data/roboflow_convertido_split``
* ``data/kaggle_dataset_split``
* ``data/trashbox_split``

Usage
-----
Split all three presets (default: ``./data``)::

    python scripts/split_dataset.py

Split one preset::

    python scripts/split_dataset.py --dataset trashbox

Split an arbitrary class-folder directory::

    python scripts/split_dataset.py \\
        --source /path/to/dataset --output /path/to/dataset_split

Preview the plan without writing anything::

    python scripts/split_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import shutil
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - friendly message
    raise SystemExit(
        "[erro] Pillow não está instalado. Instale com:  pip install pillow"
    ) from exc

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

#: Preset source datasets, relative to ``--data-root``.
DEFAULT_DATASETS: dict[str, Path] = {
    "roboflow_convertido": Path("roboflow_convertido/merged"),
    "kaggle_dataset": Path("kaggle_dataset/convertido"),
    "trashbox": Path("trashbox/TrashBox_train_set"),
}

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
# Discovery
# --------------------------------------------------------------------------- #


def is_image(path: Path) -> bool:
    """Return True when ``path`` has a recognised image extension."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


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
            root_path / name for name in files if is_image(root_path / name)
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
# Sizing / planning (identical to merge_datasets.py)
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


def plan_split(
    classes: dict[str, list[Path]],
    rng: random.Random,
    val_ratio: float,
    test_ratio: float,
) -> tuple[
    dict[str, list[tuple[str, Path]]],
    dict[str, tuple[int, int, int]],
]:
    """Stratify by class: shuffle and split each class independently.

    Returns ``split_name -> [(class, path), ...]`` and
    ``class -> (n_train, n_val, n_test)``.
    """
    splits: dict[str, list[tuple[str, Path]]] = {name: [] for name in SPLIT_NAMES}
    counts: dict[str, tuple[int, int, int]] = {}

    for cls in sorted(classes):
        paths = list(classes[cls])
        rng.shuffle(paths)
        n_train, n_val, n_test = split_sizes(len(paths), val_ratio, test_ratio)
        counts[cls] = (n_train, n_val, n_test)
        splits["train"].extend((cls, path) for path in paths[:n_train])
        splits["val"].extend(
            (cls, path) for path in paths[n_train : n_train + n_val]
        )
        splits["test"].extend((cls, path) for path in paths[n_train + n_val :])

    return splits, counts


# --------------------------------------------------------------------------- #
# Survey / validation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ImageInfo:
    """Metadata gathered in a single read of each candidate image."""

    path: Path
    cls: str
    md5: str
    mode: str
    width: int
    height: int


def survey_images(
    classes: dict[str, list[Path]],
) -> tuple[dict[Path, ImageInfo], list[tuple[Path, str]]]:
    """Read each candidate once: compute md5, capture mode/size, verify integrity.

    Returns ``(infos, invalid)`` where ``infos`` maps path -> :class:`ImageInfo`
    for every readable image and ``invalid`` lists ``(path, reason)`` for the
    ones that could not be read or decoded.
    """
    infos: dict[Path, ImageInfo] = {}
    invalid: list[tuple[Path, str]] = []

    for cls in sorted(classes):
        for path in classes[cls]:
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
                path=path,
                cls=cls,
                md5=digest,
                mode=mode,
                width=size[0],
                height=size[1],
            )

    return infos, invalid


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


def _format_for_suffix(dst_path: Path) -> str:
    return _SUFFIX_FORMATS.get(dst_path.suffix.lower(), "JPEG")


def write_image(
    src_path: Path,
    dst_path: Path,
    *,
    use_link: bool,
    convert_rgb: bool,
    mode: str | None,
) -> bool:
    """Write one image to ``dst_path``. Returns True if it was re-encoded."""
    if convert_rgb and mode not in (None, "RGB"):
        with Image.open(src_path) as im:
            image_format = im.format or _format_for_suffix(dst_path)
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
    records: dict[str, list[tuple[str, Path]]],
    output: Path,
    dataset: str,
    infos: dict[Path, ImageInfo],
    *,
    use_link: bool,
    convert_rgb: bool,
) -> tuple[list[dict[str, str]], int]:
    """Write the planned images and return ``(manifest rows, re-encoded count)``."""
    manifest: list[dict[str, str]] = []
    used_names: dict[Path, set[str]] = {}
    converted = 0

    for name in SPLIT_NAMES:
        for cls, src_path in records[name]:
            dst_dir = output / name / cls
            dst_dir.mkdir(parents=True, exist_ok=True)

            filename = src_path.name
            used = used_names.setdefault(dst_dir, set())
            if filename in used:
                stem, ext = os.path.splitext(filename)
                counter = 1
                while f"{stem}_{counter}{ext}" in used:
                    counter += 1
                filename = f"{stem}_{counter}{ext}"
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
            ):
                converted += 1

            manifest.append(
                {
                    "split": name,
                    "class": cls,
                    "source": dataset,
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


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def print_dataset_plan(
    dataset: str,
    classes: dict[str, list[Path]],
    counts: dict[str, tuple[int, int, int]],
    output: Path,
) -> None:
    print(f"\n[{dataset}] imagens por classe:")
    header = (
        "classe".ljust(20)
        + "total".rjust(8)
        + "train".rjust(8)
        + "val".rjust(8)
        + "test".rjust(8)
    )
    print(header)
    print("-" * len(header))

    for cls in sorted(classes):
        n_train, n_val, n_test = counts[cls]
        print(
            cls.ljust(20)
            + str(len(classes[cls])).rjust(8)
            + str(n_train).rjust(8)
            + str(n_val).rjust(8)
            + str(n_test).rjust(8)
        )

    total = sum(len(paths) for paths in classes.values())
    total_train = sum(size[0] for size in counts.values())
    total_val = sum(size[1] for size in counts.values())
    total_test = sum(size[2] for size in counts.values())
    print(
        "TOTAL".ljust(20)
        + str(total).rjust(8)
        + str(total_train).rjust(8)
        + str(total_val).rjust(8)
        + str(total_test).rjust(8)
    )
    print(
        f"  total: {total} imagens -> "
        f"train={total_train} val={total_val} test={total_test}"
    )
    print(f"  output: {output}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Divide um dataset de pastas por classe em train/val/test (70/15/15)."
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
        "--dataset",
        action="append",
        choices=list(DEFAULT_DATASETS),
        default=None,
        help="dataset predefinido a dividir (pode repetir; padrão: todos)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="diretório arbitrário de classes (exige --output)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="saída para --source (obrigatório com --source)",
    )
    parser.add_argument("--seed", type=int, default=42, help="semente de aleatoriedade")
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="fração de validação"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15, help="fração de teste"
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="descarta imagens vazias/corrompidas/ilegíveis",
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
        "--dry-run", action="store_true", help="só mostra o plano, sem gravar arquivos"
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="tenta hardlink em vez de copiar (cai para cópia se falhar)",
    )

    args = parser.parse_args(argv)

    if (args.source is None) != (args.output is None):
        parser.error("--source e --output devem ser usados juntos")
    if args.source is not None and args.dataset is not None:
        parser.error("--source não pode ser combinado com --dataset")
    if (
        args.val_ratio < 0
        or args.test_ratio < 0
        or args.val_ratio + args.test_ratio >= 1
    ):
        parser.error("--val-ratio e --test-ratio devem ser >= 0 e somar menos que 1")

    return args


def _jobs(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    if args.source is not None:
        dataset = args.source.resolve().name or args.source.name or "source"
        return [(dataset, args.source, args.output)]

    names = args.dataset or list(DEFAULT_DATASETS)
    return [
        (
            name,
            args.data_root / DEFAULT_DATASETS[name],
            args.data_root / f"{name}_split",
        )
        for name in names
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    for dataset, source, output in _jobs(args):
        # Fresh RNG per dataset: each split is independent of the others.
        rng = random.Random(args.seed)
        classes = discover_classes(source, exclude={output.resolve()})

        infos: dict[Path, ImageInfo] = {}
        if args.validate or args.rgb:
            infos, invalid = survey_images(classes)
            status = "descartadas" if args.validate else "inválidas (mantidas)"
            print(
                f"\n[survey] {dataset}: {len(infos)} imagens válidas; "
                f"{len(invalid)} {status}"
            )
            for path, reason in invalid[:10]:
                print(f"         - {path}: {reason}")

            if args.validate:
                classes = {
                    cls: [path for path in paths if path in infos]
                    for cls, paths in classes.items()
                }
                classes = {cls: paths for cls, paths in classes.items() if paths}
                if not classes:
                    raise SystemExit(
                        f"[erro] nenhuma imagem válida encontrada em {source}"
                    )

        records, counts = plan_split(
            classes, rng, args.val_ratio, args.test_ratio
        )
        print_dataset_plan(dataset, classes, counts, output)

        if args.dry_run:
            print("  [dry-run] nada foi gravado para este dataset.")
            continue

        reset_output(output, args.overwrite)
        manifest, converted = materialize(
            records,
            output,
            dataset,
            infos,
            use_link=args.link,
            convert_rgb=args.rgb,
        )
        write_manifest(manifest, output)

        print(f"\n[ok] {dataset}: {len(manifest)} imagens gravadas em {output}")
        if args.rgb:
            print(f"[ok] {converted} imagens re-codificadas para RGB")
        print(f"[ok] manifesto: {output / 'manifest.csv'}")

    if args.dry_run:
        print("\n[dry-run] nada foi gravado.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
