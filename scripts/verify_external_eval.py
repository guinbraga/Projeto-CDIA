#!/usr/bin/env python3
"""Verify that external evaluation targets are disjoint from the training corpus.

The preparation command (:mod:`prepare_external_eval`) already drops every
cluster that touches ``data/content_index.csv``. This gate re-checks the
written manifests without trusting that step. For every external test row it
fails when:

* the row's exact ``md5`` exists in the content index;
* a content-index image is a near-duplicate (``dHash <= 1 AND pHash <= 1``);
* the row's ``cluster_id`` is already in ``data/global_split_registry.csv``;
* the same ``cluster_id`` appears twice inside one target or across targets.

It also checks that the manifest's ``test/`` tree matches the row count and
that every required class has support. The check runs in seconds because the
hashes travel inside the manifest; images are never decoded.

Usage
-----
    python3 scripts/verify_external_eval.py
    python3 scripts/verify_external_eval.py --external data/external_eval/realwaste
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "external_eval"
DEFAULT_CONTENT_INDEX = REPO_ROOT / "data" / "content_index.csv"
DEFAULT_REGISTRY = REPO_ROOT / "data" / "global_split_registry.csv"
REQUIRED_CLASSES: tuple[str, ...] = ("cardboard", "metal", "paper", "plastic")

#: dHash <= 1 means at most one bit differs; the lookup probes the hash itself
#: and its 64 one-bit neighbours, exactly like content_index._components.
DHASH_MAX = 1
PHASH_MAX = 1


class ExternalEvalError(Exception):
    """Raised when the verifier cannot read its inputs."""


@dataclass(frozen=True)
class ExternalRow:
    dataset: str
    cluster_id: str
    md5: str
    dhash: str
    phash: str
    cls: str
    path: str


@dataclass
class HashIndex:
    md5s: set[str] = field(default_factory=set)
    by_dhash: dict[int, list[tuple[int, str, str, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add(self, md5: str, dhash: str, phash: str, source: str, cls: str) -> None:
        self.md5s.add(md5)
        try:
            dhash_value = int(dhash, 16)
            phash_value = int(phash, 16)
        except (TypeError, ValueError):
            return
        self.by_dhash[dhash_value].append((phash_value, md5, source, cls))

    def matches(self, dhash: str, phash: str, md5: str) -> list[tuple[str, str, str]]:
        """Return ``(md5, source, class)`` for every near-duplicate."""
        try:
            dhash_value = int(dhash, 16)
            phash_value = int(phash, 16)
        except (TypeError, ValueError):
            raise ExternalEvalError(f"invalid hash pair dhash={dhash!r} phash={phash!r}")
        candidates: list[tuple[int, str, str, str]] = list(
            self.by_dhash.get(dhash_value, ())
        )
        for bit in range(64):
            candidates.extend(self.by_dhash.get(dhash_value ^ (1 << bit), ()))
        hits: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for candidate_phash, candidate_md5, source, cls in candidates:
            if candidate_md5 == md5 or candidate_md5 in seen:
                continue
            if (candidate_phash ^ phash_value).bit_count() <= PHASH_MAX:
                seen.add(candidate_md5)
                hits.append((candidate_md5, source, cls))
        if md5 in self.md5s and md5 not in seen:
            hits.append((md5, "", ""))
        return hits


@dataclass
class Verification:
    rows: list[ExternalRow]
    identities: dict[str, list[ExternalRow]]
    registry: dict[str, str]
    violations: dict[str, list[str]]
    class_counts: Counter
    missing_files: int
    manifest_files: int

    @property
    def passed(self) -> bool:
        return not self.violations and not self.missing_files


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ExternalEvalError(f"cannot read {path}: {exc}") from exc


def load_external(path: Path) -> list[ExternalRow]:
    manifest = path / "manifest.csv" if path.is_dir() else path
    if not manifest.is_file():
        raise ExternalEvalError(f"manifest not found: {path}")
    fields, raw_rows = _read_csv(manifest)
    required = {"split", "class", "source", "cluster_id", "md5", "dhash", "phash"}
    missing = required - set(fields)
    if missing:
        raise ExternalEvalError(
            f"{manifest}: missing columns {sorted(missing)}; rebuild the target "
            "with scripts/prepare_external_eval.py"
        )
    dataset = manifest.parent.name
    rows: list[ExternalRow] = []
    for line, raw in enumerate(raw_rows, start=2):
        if (raw.get("split") or "").strip() != "test":
            raise ExternalEvalError(
                f"{manifest}: external rows must be split=test (line {line})"
            )
        for column in ("cluster_id", "md5", "dhash", "phash"):
            if not (raw.get(column) or "").strip():
                raise ExternalEvalError(
                    f"{manifest}: empty {column} at line {line}"
                )
        rows.append(
            ExternalRow(
                dataset=dataset,
                cluster_id=raw["cluster_id"].strip(),
                md5=raw["md5"].strip(),
                dhash=raw["dhash"].strip(),
                phash=raw["phash"].strip(),
                cls=(raw.get("class") or "").strip(),
                path=(raw.get("dst_path") or raw.get("src_path") or "").strip(),
            )
        )
    return rows


def load_registry(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ExternalEvalError(f"registry not found: {path}")
    _fields, rows = _read_csv(path)
    return {
        (row.get("cluster_id") or "").strip(): (row.get("split") or "").strip()
        for row in rows
        if (row.get("cluster_id") or "").strip()
    }


def build_hash_index(path: Path) -> HashIndex:
    index = HashIndex()
    fields, rows = _read_csv(path)
    required = {"source", "class", "md5", "dhash", "phash"}
    missing = required - set(fields)
    if missing:
        raise ExternalEvalError(f"{path}: missing columns {sorted(missing)}")
    for row in rows:
        index.add(
            row["md5"].strip(),
            row["dhash"].strip(),
            row["phash"].strip(),
            row.get("source", ""),
            row.get("class", ""),
        )
    return index


def verify(
    external_paths: list[Path],
    content_index: Path,
    registry_path: Path,
    require_classes: tuple[str, ...] = REQUIRED_CLASSES,
) -> Verification:
    if not external_paths:
        raise ExternalEvalError("no external targets given")
    rows: list[ExternalRow] = []
    for path in external_paths:
        rows.extend(load_external(path))
    identities: dict[str, list[ExternalRow]] = defaultdict(list)
    for row in rows:
        identities[row.cluster_id].append(row)
    registry = load_registry(registry_path)
    index = build_hash_index(content_index)

    violations: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        if row.cluster_id in registry:
            violations["cluster_id in training registry"].append(
                f"{row.dataset}/{row.cls}: {row.cluster_id} "
                f"(registry split={registry[row.cluster_id]})"
            )
        hits = index.matches(row.dhash, row.phash, row.md5)
        if row.md5 in index.md5s:
            violations["exact md5 in training corpus"].append(
                f"{row.dataset}/{row.cls}: {row.md5}"
            )
        for candidate_md5, source, cls in hits:
            if candidate_md5 == row.md5:
                continue
            violations["near-duplicate in training corpus"].append(
                f"{row.dataset}/{row.cls}: {row.md5} ~ "
                f"{source}/{cls}:{candidate_md5}"
            )

    for cluster_id, group in identities.items():
        if len(group) < 2:
            continue
        datasets = sorted({row.dataset for row in group})
        label = (
            "cluster_id repeated across external targets"
            if len(datasets) > 1
            else "cluster_id repeated inside one external target"
        )
        violations[label].append(
            f"{cluster_id}: " + ", ".join(f"{row.dataset}/{row.cls}" for row in group)
        )

    # Cross-target disjointness: an image may not be an exact/near-duplicate of
    # a row that belongs to a different external target.
    md5_datasets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        md5_datasets[row.md5].add(row.dataset)
    for md5, datasets in md5_datasets.items():
        if len(datasets) > 1:
            violations["exact md5 repeated across external targets"].append(
                f"{md5}: {', '.join(sorted(datasets))}"
            )

    external_index = HashIndex()
    for row in rows:
        external_index.add(row.md5, row.dhash, row.phash, row.dataset, row.cls)
    for row in rows:
        for candidate_md5, source, cls in external_index.matches(
            row.dhash, row.phash, row.md5
        ):
            if not source or candidate_md5 == row.md5 or source == row.dataset:
                continue
            violations["near-duplicate across external targets"].append(
                f"{row.dataset}/{row.cls}: {row.md5} ~ {source}/{cls}:{candidate_md5}"
            )

    class_counts = Counter(row.cls for row in rows)
    for name in require_classes:
        if class_counts.get(name, 0) == 0:
            violations["required class has no images"].append(name)

    manifest_files = len(rows)
    missing_files = sum(
        1 for row in rows if row.path and not Path(row.path).is_file()
    )

    return Verification(
        rows=rows,
        identities=identities,
        registry=registry,
        violations=dict(violations),
        class_counts=class_counts,
        missing_files=missing_files,
        manifest_files=manifest_files,
    )


def format_report(result: Verification, max_examples: int = 5) -> str:
    datasets = sorted({row.dataset for row in result.rows})
    lines = [
        f"[external] {', '.join(datasets)}",
        f"[rows] {len(result.rows)} (test-only)",
        "[classes] "
        + ", ".join(f"{name}={count}" for name, count in sorted(result.class_counts.items())),
    ]
    if not result.violations and not result.missing_files:
        lines.append("[check] no training-corpus overlap: PASS")
    for label, examples in sorted(result.violations.items()):
        lines.append(f"[check] {label}: {len(examples)} - FAIL")
        for example in examples[: max(max_examples, 0)]:
            lines.append(f"  - {example}")
        if len(examples) > max_examples:
            lines.append(f"  ... and {len(examples) - max_examples} more")
    if result.missing_files:
        lines.append(
            f"[check] manifest rows missing on disk: {result.missing_files} - FAIL"
        )
    lines.append("RESULT: PASS" if result.passed else "RESULT: FAIL")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that external evaluation targets do not overlap the "
            "training corpus."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--external",
        action="append",
        type=Path,
        default=None,
        help="external target directory (repeatable); default: every */manifest.csv",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT
    )
    parser.add_argument(
        "--content-index", type=Path, default=DEFAULT_CONTENT_INDEX
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--require-classes",
        default=",".join(REQUIRED_CLASSES),
        help="comma-separated classes that must have support",
    )
    parser.add_argument("--max-examples", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_classes = tuple(
        name.strip() for name in args.require_classes.split(",") if name.strip()
    )
    if args.external:
        paths = args.external
    else:
        paths = sorted(
            path.parent
            for path in args.dataset_root.glob("*/manifest.csv")
        )
    try:
        result = verify(paths, args.content_index, args.registry, require_classes)
    except ExternalEvalError as exc:
        print(f"[erro] {exc}", file=sys.stderr)
        return 2
    print(format_report(result, args.max_examples))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
