#!/usr/bin/env python3
"""Merge the four trash source datasets into one clean, balanced dataset.

The output is meant to feed an EfficientNetB0 classifier (e.g. on Google Colab
via ``torchvision.datasets.ImageFolder`` + ``DataLoader``, or TensorFlow's
``image_dataset_from_directory``).

Balance rule
------------
* For every target class, images are sampled in **equal amounts from each source
  dataset that contains that class**, capped by the smallest source for that
  class (``min`` over the sources that have it). A source that does not contain
  the class simply contributes 0 for that class.
* Exception: ``tetra_pack`` is scarce (``trashnet`` has only a handful), so
  **all** ``tetra_pack`` images from every source are joined instead of being
  balanced.
* ``--balance-classes`` (optional) additionally caps every class to the
  smallest class total, sub-sampling each source proportionally, so all classes
  end up with the same number of images.

Cleaning (applied before balancing)
-----------------------------------
* ``--validate`` (default) drops unreadable/empty/corrupt images.
* ``--dedupe`` (default) removes exact content duplicates (md5). Each duplicate
  group keeps a single representative, credited to the source that balances the
  per-source counts (shared images go to the scarcest source for that class). A
  duplicate group that spans **more than one class** is dropped entirely,
  because its label is ambiguous. This removes leakage between train/val/test.
* ``--rgb`` (default) re-encodes non-RGB images (RGBA/P/L/CMYK) to RGB so the
  files work with default vision transforms.

Output layout
-------------
::

    <output>/
        train/<class>/*.jpg
        val/<class>/*.jpg
        test/<class>/*.jpg
        manifest.csv

The split is 70/15/15 and is **stratified by (class, source)**: each
``(class, source)`` group is split independently, so every split preserves both
the class distribution and the source distribution.

Copied files are prefixed with their source dataset name (``<source>__name``),
which keeps provenance and avoids name collisions between sources.

Usage
-----
Local (defaults to ``./data``)::

    python scripts/merge_datasets.py --dry-run
    python scripts/merge_datasets.py

Colab (after unzipping to ``/content/extracted_datasets``)::

    !python merge_datasets.py \\
        --data-root /content/extracted_datasets \\
        --output /content/merged_dataset
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import shutil
import sys
from collections import defaultdict
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

#: Canonical classes for the merged dataset.
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

#: Classes that are joined in full (all sources) instead of being balanced.
JOIN_ALL_CLASSES: frozenset[str] = frozenset({"tetra_pack"})

#: Recognised image extensions (compared case-insensitively).
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
)

#: Default split ratios (train, val, test).
DEFAULT_SPLIT: tuple[float, float, float] = (0.70, 0.15, 0.15)
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


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


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def discover_class_images(
    data_root: Path,
    exclude: set[Path] | None = None,
) -> dict[str, dict[str, list[Path]]]:
    """Group source images by ``(source, class)``.

    Each immediate subdirectory of ``data_root`` is treated as a *source
    dataset*. Inside it, every directory whose name matches a target class
    (case-insensitively) is collected; nesting is handled by walking the whole
    tree, so ``trashbox/TrashBox_train_set/<class>`` and
    ``roboflow_convertido/merged/<class>`` are found alike.

    Hidden directories and ``exclude`` (e.g. a previous output dir placed inside
    ``data_root``) are skipped so re-runs stay idempotent.
    """
    if not data_root.is_dir():
        raise SystemExit(f"[erro] data root não encontrado: {data_root}")

    exclude = {p.resolve() for p in (exclude or set())}
    canonical = {c.lower(): c for c in TARGET_CLASSES}
    found: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    seen: set[Path] = set()

    source_dirs = sorted(
        p
        for p in data_root.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.resolve() not in exclude
    )
    for source_dir in source_dirs:
        for root, dirs, files in os.walk(source_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            cls = canonical.get(Path(root).name.lower())
            if cls is None:
                continue
            for fname in files:
                path = Path(root) / fname
                if not is_image(path):
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                found[source_dir.name][cls].append(path)

    return {src: dict(classes) for src, classes in found.items()}


def pools_from_infos(infos: dict[Path, ImageInfo]) -> dict[str, dict[str, list[Path]]]:
    """Rebuild ``source -> class -> [path]`` from surveyed (valid) images."""
    pools: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for info in infos.values():
        pools[info.source][info.cls].append(info.path)
    return {src: dict(classes) for src, classes in pools.items()}


# --------------------------------------------------------------------------- #
# Survey / cleaning
# --------------------------------------------------------------------------- #


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
                infos[path] = ImageInfo(path, source, cls, digest, mode, size[0], size[1])
    return infos, invalid


def dedupe_images(
    infos: dict[Path, ImageInfo],
) -> tuple[dict[str, dict[str, list[Path]]], dict[str, int]]:
    """Remove exact content duplicates, keeping one representative per group.

    * Groups spanning multiple classes are dropped entirely (ambiguous label).
    * Otherwise exactly one image per group is kept, and its provenance is
      chosen to keep the per-source counts balanced: images unique to a source
      stay there, while images shared by several sources go to the source that
      currently has the fewest images for that class. This keeps the balanced
      per-source quota as high as possible.
    """
    groups: dict[str, list[ImageInfo]] = defaultdict(list)
    for info in infos.values():
        groups[info.md5].append(info)

    by_class: dict[str, list[list[ImageInfo]]] = defaultdict(list)
    dropped_conflicts = 0
    for members in groups.values():
        if len({m.cls for m in members}) > 1:
            dropped_conflicts += len(members)
            continue
        by_class[members[0].cls].append(members)

    kept: list[ImageInfo] = []
    dropped_dupes = 0
    for member_lists in by_class.values():
        counts: dict[str, int] = defaultdict(int)
        exclusive = [g for g in member_lists if len({m.source for m in g}) == 1]
        shared = [g for g in member_lists if len({m.source for m in g}) > 1]

        for group in exclusive:
            counts[group[0].source] += 1
            kept.append(group[0])
            dropped_dupes += len(group) - 1

        # Most-constrained groups first, so balancing has the most freedom last.
        shared.sort(key=lambda g: len({m.source for m in g}))
        for group in shared:
            source = min({m.source for m in group}, key=lambda s: (counts[s], s))
            representative = min(
                (m for m in group if m.source == source), key=lambda m: str(m.path)
            )
            counts[source] += 1
            kept.append(representative)
            dropped_dupes += len(group) - 1

    stats = {"dupes_removed": dropped_dupes, "conflicts_removed": dropped_conflicts}
    return pools_from_infos({info.path: info for info in kept}), stats


# --------------------------------------------------------------------------- #
# Planning (balance + sampling)
# --------------------------------------------------------------------------- #


def available_table(
    found: dict[str, dict[str, list[Path]]],
) -> dict[str, dict[str, int]]:
    """Return ``class -> source -> count`` for the requested classes."""
    table: dict[str, dict[str, int]] = {c: {} for c in TARGET_CLASSES}
    for source, classes in found.items():
        for cls, paths in classes.items():
            table.setdefault(cls, {})[source] = len(paths)
    return table


def plan_sampling(
    found: dict[str, dict[str, list[Path]]],
    rng: random.Random,
) -> dict[tuple[str, str], list[Path]]:
    """Build the list of images to keep, keyed by ``(class, source)``.

    For a normal class the per-source target is the minimum count across the
    sources that contain it. For a ``JOIN_ALL_CLASSES`` class every available
    image is kept.
    """
    plan: dict[tuple[str, str], list[Path]] = {}
    for cls in TARGET_CLASSES:
        by_source: dict[str, list[Path]] = {}
        for src, classes in found.items():
            paths = classes.get(cls)
            if paths:
                by_source[src] = paths
        if not by_source:
            continue
        if cls in JOIN_ALL_CLASSES:
            for src, paths in by_source.items():
                plan[(cls, src)] = list(paths)
        else:
            target = min(len(paths) for paths in by_source.values())
            for src, paths in by_source.items():
                plan[(cls, src)] = rng.sample(paths, target)
    return plan


def balance_class_totals(
    plan: dict[tuple[str, str], list[Path]],
    rng: random.Random,
) -> tuple[dict[tuple[str, str], list[Path]], int]:
    """Cap every class to the smallest class total.

    Each class is reduced by sub-sampling its sources proportionally
    (largest-remainder rounding), which keeps the per-source proportions of the
    class intact. Returns the balanced plan and the target used.
    """
    totals: dict[str, int] = defaultdict(int)
    for (cls, _src), paths in plan.items():
        totals[cls] += len(paths)
    if not totals:
        return plan, 0

    target = min(totals.values())
    balanced: dict[tuple[str, str], list[Path]] = {}
    for cls, total in totals.items():
        keys = [k for k in plan if k[0] == cls]
        if total <= target:
            for k in keys:
                balanced[k] = plan[k]
            continue

        raw = {k: target * len(plan[k]) / total for k in keys}
        keep = {k: int(raw[k]) for k in keys}
        shortfall = target - sum(keep.values())
        for k in sorted(keys, key=lambda k: raw[k] - keep[k], reverse=True):
            if shortfall == 0:
                break
            if keep[k] < len(plan[k]):
                keep[k] += 1
                shortfall -= 1
        for k in keys:
            balanced[k] = rng.sample(plan[k], keep[k])
    return balanced, target


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
    plan: dict[tuple[str, str], list[Path]],
    rng: random.Random,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[tuple[str, str, Path]]]:
    """Stratify by ``(class, source)``: split each group independently.

    Returns ``split_name -> [(class, source, path), ...]``.
    """
    splits: dict[str, list[tuple[str, str, Path]]] = {name: [] for name in SPLIT_NAMES}
    for (cls, src), paths in plan.items():
        shuffled = list(paths)
        rng.shuffle(shuffled)
        n_train, n_val, _n_test = split_sizes(len(shuffled), val_ratio, test_ratio)
        sections = {
            "train": shuffled[:n_train],
            "val": shuffled[n_train : n_train + n_val],
            "test": shuffled[n_train + n_val :],
        }
        for name, section in sections.items():
            splits[name].extend((cls, src, p) for p in section)
    return splits


# --------------------------------------------------------------------------- #
# Materialisation
# --------------------------------------------------------------------------- #


def reset_output(output: Path, overwrite: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise SystemExit(
                f"[erro] '{output}' já existe e não está vazio. "
                "Use --overwrite para substituí-lo."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def write_image(
    src_path: Path,
    dst_path: Path,
    *,
    use_link: bool,
    convert_rgb: bool,
    mode: str | None,
) -> bool:
    """Write one image to ``dst_path``. Returns True if it was re-encoded."""
    if convert_rgb and mode is not None and mode != "RGB":
        with Image.open(src_path) as im:
            im.convert("RGB").save(dst_path, format="JPEG", quality=95)
        return True
    if use_link:
        try:
            os.link(src_path, dst_path)
            return False
        except OSError:
            pass  # fall through to copy
    shutil.copy2(src_path, dst_path)
    return False


def copy_or_link(
    records: dict[str, list[tuple[str, str, Path]]],
    output: Path,
    infos: dict[Path, ImageInfo],
    *,
    use_link: bool,
    convert_rgb: bool,
) -> tuple[list[dict[str, str]], int]:
    """Write the selected images and return ``(manifest rows, re-encoded count)``."""
    manifest: list[dict[str, str]] = []
    used_names: dict[Path, set[str]] = defaultdict(set)
    converted = 0

    for name in SPLIT_NAMES:
        for cls, src, src_path in records[name]:
            dst_dir = output / name / cls
            dst_dir.mkdir(parents=True, exist_ok=True)

            base = f"{src}__{src_path.name}"
            dst_path = dst_dir / base
            if base in used_names[dst_dir]:
                stem, ext = os.path.splitext(base)
                counter = 1
                while f"{stem}_{counter}{ext}" in used_names[dst_dir]:
                    counter += 1
                dst_path = dst_dir / f"{stem}_{counter}{ext}"
            used_names[dst_dir].add(dst_path.name)

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
                    "source": src,
                    "md5": info.md5 if info else "",
                    "mode": info.mode if info else "",
                    "src_path": str(src_path),
                    "dst_path": str(dst_path),
                }
            )
    return manifest, converted


def write_manifest(manifest: list[dict[str, str]], output: Path) -> None:
    fields = ["split", "class", "source", "md5", "mode", "src_path", "dst_path"]
    with open(output / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def print_plan(
    table: dict[str, dict[str, int]],
    plan: dict[tuple[str, str], list[Path]],
) -> None:
    sources = sorted({src for counts in table.values() for src in counts})
    print("\n[plano] imagens por classe (balanceado por fonte; min = cota por fonte):")
    header = "classe".ljust(12) + "".join(s.rjust(20) for s in sources) + "total".rjust(10)
    print(header)
    print("-" * len(header))
    for cls in TARGET_CLASSES:
        cells = []
        for src in sources:
            avail = table.get(cls, {}).get(src, 0)
            kept = len(plan.get((cls, src), []))
            cells.append(f"{kept}/{avail}".rjust(20))
        total = sum(len(plan.get((cls, s), [])) for s in sources)
        print(cls.ljust(12) + "".join(cells) + str(total).rjust(10))
    grand = sum(len(v) for v in plan.values())
    print(f"\n[plano] total de imagens no dataset mesclado: {grand}")


def print_result(records: dict[str, list[tuple[str, str, Path]]]) -> None:
    print("\n[resultado] imagens por split/classe:")
    for name in SPLIT_NAMES:
        per_class: dict[str, int] = defaultdict(int)
        per_source: dict[str, int] = defaultdict(int)
        for cls, src, _ in records[name]:
            per_class[cls] += 1
            per_source[src] += 1
        total = len(records[name])
        print(f"  {name:<5} total={total:<7} classes={dict(sorted(per_class.items()))}")
        print(f"        fontes={dict(sorted(per_source.items()))}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Mescla os 4 datasets de lixo em um dataset balanceado (limpo).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo_root / "data",
        help="diretório com as pastas dos datasets de origem",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "merged_dataset",
        help="diretório de saída do dataset mesclado",
    )
    parser.add_argument("--seed", type=int, default=42, help="semente de aleatoriedade")
    parser.add_argument(
        "--val-ratio", type=float, default=DEFAULT_SPLIT[1], help="fração de validação"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=DEFAULT_SPLIT[2], help="fração de teste"
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="descarta imagens vazias/corrompidas/ilegíveis",
    )
    parser.add_argument(
        "--dedupe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="remove duplicatas exatas (md5); grupos de classes diferentes são descartados",
    )
    parser.add_argument(
        "--rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="re-codifica imagens não-RGB (RGBA/P/L/CMYK) para RGB",
    )
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="iguala o total de cada classe à menor classe (sub-amostra por fonte)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="só mostra o plano, sem copiar arquivos"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="apaga a saída existente antes de gravar"
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="tenta hardlink em vez de copiar (mais rápido; cai para cópia se falhar)",
    )
    args = parser.parse_args(argv)
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        parser.error("--val-ratio e --test-ratio devem ser >= 0 e somar menos que 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    found = discover_class_images(args.data_root, exclude={args.output.resolve()})
    if not found:
        raise SystemExit(f"[erro] nenhuma imagem de classe encontrada em {args.data_root}")

    infos: dict[Path, ImageInfo] = {}
    if args.validate or args.dedupe or args.rgb:
        infos, invalid = survey_images(found)
        found = pools_from_infos(infos)
        print(
            f"\n[survey] {len(infos)} imagens válidas; "
            f"{len(invalid)} descartadas (vazias/corrompidas/ilegíveis)"
        )
        for path, reason in invalid[:10]:
            print(f"         - {path}: {reason}")

    if args.dedupe:
        found, stats = dedupe_images(infos)
        print(
            f"[dedupe] {stats['dupes_removed']} duplicatas exatas removidas; "
            f"{stats['conflicts_removed']} imagens descartadas por conflito de classe"
        )

    table = available_table(found)
    plan = plan_sampling(found, rng)
    if args.balance_classes:
        plan, target = balance_class_totals(plan, rng)
        print(f"\n[balance] cada classe limitada a {target} imagens")
    print_plan(table, plan)

    records = plan_split(plan, rng, args.val_ratio, args.test_ratio)
    print_result(records)

    if args.dry_run:
        print("\n[dry-run] nada foi copiado.")
        return 0

    reset_output(args.output, args.overwrite)
    manifest, converted = copy_or_link(
        records, args.output, infos, use_link=args.link, convert_rgb=args.rgb
    )
    write_manifest(manifest, args.output)
    print(f"\n[ok] {len(manifest)} imagens gravadas em {args.output}")
    if args.rgb:
        print(f"[ok] {converted} imagens re-codificadas para RGB")
    print(f"[ok] manifesto: {args.output / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
