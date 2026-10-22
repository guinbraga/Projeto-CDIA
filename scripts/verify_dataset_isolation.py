#!/usr/bin/env python3
"""Dataset isolation gate: no content identity may leak across split roles.

A content identity is a fingerprint of one image (``md5`` for the current
pre-registry manifests, ``cluster_id`` once the registry exists). The invariant
is simple: an identity used for evaluation (``test``) must never appear in a
training split (``train``/``val``) of any dataset derived from it.

Two merged conditions are checked separately because a merged manifest can leak
in both directions: it may pull a source ``test`` image into its own ``train``
or ``val`` split, and it may place a source ``train``/``val`` image into its own
``test`` split. A third check rejects the same identity appearing twice inside
one merged manifest, and an optional registry check pins each row to its
canonical split.

Example usage::

    python3 scripts/verify_dataset_isolation.py \\
        --merged data/merged_dataset_balanced
    python3 scripts/verify_dataset_isolation.py --registry data/registry.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SPLIT_NAMES = ("train", "val", "test")

_IDENTITY_COLUMNS = ("cluster_id", "md5")
_TRAINVAL = frozenset({"train", "val"})
_CSV_PARSE_ERRORS = (OSError, UnicodeDecodeError, csv.Error)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SOURCE_NAMES = (
    "trashnet_split",
    "kaggle_dataset_split",
    "trashbox_split",
    "roboflow_convertido_split",
)
_DEFAULT_MERGED_NAMES = ("merged_dataset", "merged_dataset_balanced")


class ConfigError(Exception):
    """Raised when manifests or the registry cannot be used as supplied."""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    manifest: Path
    kind: str  # "source" or "merged"


@dataclass(frozen=True)
class ManifestRow:
    dataset: str
    kind: str
    split: str
    cls: str
    identity: str
    path: str


@dataclass(frozen=True)
class ViolationGroup:
    identity: str
    rows: tuple[ManifestRow, ...]
    note: str = ""


@dataclass
class Check:
    """One isolation check.

    ``groups``/``count`` count violating identities, while ``images`` is
    deliberately check-specific: all rows of a conflicting source identity,
    leaking merged rows for the merged-vs-source checks, duplicated rows for
    merged duplicates, and mismatched plus missing rows for registry consistency.
    """

    name: str
    groups: list[ViolationGroup]
    images: int
    missing: list[ManifestRow]

    @property
    def count(self) -> int:
        return len(self.groups)

    @property
    def passed(self) -> bool:
        return not self.groups and not self.missing


@dataclass
class VerificationResult:
    datasets: tuple[DatasetSpec, ...]
    identity_keys: dict[str, str]
    source_split_conflicts: Check
    merged_trainval_vs_source_test: Check
    merged_test_vs_source_trainval: Check
    merged_duplicate_identities: Check
    registry_label_consistency: Check | None = None
    identity_key: str = "md5"

    @property
    def passed(self) -> bool:
        checks = [
            self.source_split_conflicts,
            self.merged_trainval_vs_source_test,
            self.merged_test_vs_source_trainval,
            self.merged_duplicate_identities,
        ]
        if self.registry_label_consistency is not None:
            checks.append(self.registry_label_consistency)
        return all(check.passed for check in checks)

    def failures(self) -> list[str]:
        labeled = [
            (self.source_split_conflicts, "source split conflicts"),
            (self.merged_trainval_vs_source_test, "merged train/val leaks"),
            (self.merged_test_vs_source_trainval, "merged test leaks"),
            (self.merged_duplicate_identities, "merged duplicate identities"),
        ]
        failures = [
            f"{check.count} {label}"
            for check, label in labeled
            if not check.passed
        ]
        registry = self.registry_label_consistency
        if registry is not None and not registry.passed:
            failures.append(
                f"{registry.count} registry split mismatches"
                f" ({len(registry.missing)} missing from registry)"
            )
        return failures


# --------------------------------------------------------------------------- #
# Manifest / registry parsing
# --------------------------------------------------------------------------- #


def _resolve_manifest(path: Path) -> Path:
    candidate = path / "manifest.csv" if path.is_dir() else path
    if not candidate.is_file():
        raise ConfigError(f"manifest not found: {path}")
    return candidate


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)
    except _CSV_PARSE_ERRORS as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc


def _identity_key(
    manifest: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> str:
    present = [column for column in _IDENTITY_COLUMNS if column in fieldnames]
    if not present:
        raise ConfigError(
            f"{manifest}: missing identity column (expected md5 or cluster_id)"
        )
    first_empty: tuple[str, int] | None = None
    for column in present:
        empty_line = next(
            (
                line
                for line, row in enumerate(rows, start=2)
                if not (row.get(column) or "").strip()
            ),
            None,
        )
        if empty_line is None:
            return column
        if first_empty is None:
            first_empty = (column, empty_line)
    column, line = first_empty or (present[0], 2)
    raise ConfigError(
        f"{manifest}: column '{column}' is empty at line {line} "
        "and no identity column is fully populated"
    )


def _parse_manifest(spec: DatasetSpec) -> tuple[list[ManifestRow], str]:
    if spec.kind not in ("source", "merged"):
        raise ConfigError(f"dataset '{spec.name}': invalid kind '{spec.kind}'")
    manifest = _resolve_manifest(spec.manifest)
    fieldnames, raw_rows = _read_csv(manifest)
    if "split" not in fieldnames:
        raise ConfigError(f"{manifest}: missing required column 'split'")
    identity_key = _identity_key(manifest, fieldnames, raw_rows)
    rows: list[ManifestRow] = []
    for line, raw in enumerate(raw_rows, start=2):
        split = (raw.get("split") or "").strip()
        if split not in SPLIT_NAMES:
            raise ConfigError(
                f"{manifest}: invalid split {split!r} at line {line}"
            )
        rows.append(
            ManifestRow(
                dataset=spec.name,
                kind=spec.kind,
                split=split,
                cls=raw.get("class") or "",
                identity=(raw.get(identity_key) or "").strip(),
                path=raw.get("dst_path") or raw.get("src_path") or "",
            )
        )
    return rows, identity_key


def _resolve_registry_key(
    registry: Path, fieldnames: list[str], manifest_key: str | None
) -> str:
    if manifest_key is not None:
        if manifest_key in fieldnames:
            return manifest_key
        other_key = "md5" if manifest_key == "cluster_id" else "cluster_id"
        if other_key in fieldnames:
            raise ConfigError(
                f"registry {registry} uses '{other_key}' but manifests use "
                f"'{manifest_key}'"
            )
        raise ConfigError(
            f"registry {registry}: missing identity column '{manifest_key}'"
        )
    for column in _IDENTITY_COLUMNS:
        if column in fieldnames:
            return column
    raise ConfigError(
        f"registry {registry}: missing identity column (expected md5 or cluster_id)"
    )


def _load_registry(
    registry: Path, manifest_key: str | None
) -> tuple[dict[str, str], str]:
    if not registry.is_file():
        raise ConfigError(f"registry not found: {registry}")
    fieldnames, rows = _read_csv(registry)
    identity_key = _resolve_registry_key(registry, fieldnames, manifest_key)
    if "split" not in fieldnames:
        raise ConfigError(f"registry {registry}: missing required column 'split'")
    canonical: dict[str, str] = {}
    for line, raw in enumerate(rows, start=2):
        identity = (raw.get(identity_key) or "").strip()
        split = (raw.get("split") or "").strip()
        if not identity:
            raise ConfigError(
                f"registry {registry}: empty '{identity_key}' at line {line}"
            )
        if split not in SPLIT_NAMES:
            raise ConfigError(
                f"registry {registry}: invalid split {split!r} at line {line}"
            )
        if identity in canonical:
            raise ConfigError(
                f"registry {registry}: duplicate identity '{identity}'"
            )
        canonical[identity] = split
    return canonical, identity_key


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def _sorted_groups(groups: list[ViolationGroup]) -> list[ViolationGroup]:
    return sorted(groups, key=lambda group: (group.identity, group.note))


def _source_conflicts(rows: Sequence[ManifestRow]) -> Check:
    by_identity: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        by_identity[row.identity].append(row)
    groups = [
        ViolationGroup(identity, tuple(group_rows))
        for identity, group_rows in by_identity.items()
        if "test" in {row.split for row in group_rows}
        and _TRAINVAL & {row.split for row in group_rows}
    ]
    groups = _sorted_groups(groups)
    return Check(
        "source split conflicts (test vs train/val)",
        groups,
        sum(len(group.rows) for group in groups),
        [],
    )


def _merged_against_source(
    merged_rows: Sequence[ManifestRow],
    source_rows: Sequence[ManifestRow],
    merged_splits: frozenset[str],
    source_splits: frozenset[str],
    name: str,
) -> Check:
    merged_by: dict[str, list[ManifestRow]] = defaultdict(list)
    source_by: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in merged_rows:
        if row.split in merged_splits:
            merged_by[row.identity].append(row)
    for row in source_rows:
        if row.split in source_splits:
            source_by[row.identity].append(row)
    groups = [
        ViolationGroup(identity, tuple(merged_by[identity] + source_by[identity]))
        for identity in merged_by
        if identity in source_by
    ]
    groups = _sorted_groups(groups)
    return Check(
        name,
        groups,
        sum(len(merged_by[group.identity]) for group in groups),
        [],
    )


def _merged_duplicates(rows: Sequence[ManifestRow]) -> Check:
    by_key: dict[tuple[str, str], list[ManifestRow]] = defaultdict(list)
    for row in rows:
        by_key[(row.dataset, row.identity)].append(row)
    groups = [
        ViolationGroup(identity, tuple(group_rows), note=dataset)
        for (dataset, identity), group_rows in by_key.items()
        if len(group_rows) > 1
    ]
    groups = _sorted_groups(groups)
    return Check(
        "merged duplicate identities",
        groups,
        sum(len(group.rows) for group in groups),
        [],
    )


def _registry_consistency(
    all_rows: Sequence[ManifestRow],
    canonical: dict[str, str],
) -> Check:
    mismatched: dict[str, list[ManifestRow]] = defaultdict(list)
    mismatched_rows = 0
    missing: list[ManifestRow] = []
    for row in all_rows:
        expected = canonical.get(row.identity)
        if expected is None:
            missing.append(row)
        elif row.split != expected:
            mismatched[row.identity].append(row)
            mismatched_rows += 1
    groups = [
        ViolationGroup(
            identity,
            tuple(mismatched[identity]),
            note=f"expected split {canonical[identity]}",
        )
        for identity in mismatched
    ]
    return Check(
        "registry split labels",
        _sorted_groups(groups),
        mismatched_rows + len(missing),
        missing,
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def verify(
    sources: Sequence[DatasetSpec],
    merged: Sequence[DatasetSpec],
    registry: Path | None = None,
) -> VerificationResult:
    """Load every manifest and return all isolation checks.

    Passing an empty ``merged`` sequence checks the source splits alone, so a
    caller can gate before the merged rebuild exists.
    """
    specs = tuple(sources) + tuple(merged)
    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ConfigError(f"duplicate dataset names: {', '.join(duplicates)}")

    parsed: list[list[ManifestRow]] = []
    identity_keys: dict[str, str] = {}
    for spec in specs:
        rows, key = _parse_manifest(spec)
        parsed.append(rows)
        identity_keys[spec.name] = key

    distinct_keys = set(identity_keys.values())
    if len(distinct_keys) > 1:
        detail = ", ".join(f"{name}={key}" for name, key in identity_keys.items())
        raise ConfigError(f"mixed identity keys across manifests: {detail}")
    manifest_key = next(iter(distinct_keys), None)

    all_rows = [row for rows in parsed for row in rows]
    source_rows = [row for row in all_rows if row.kind == "source"]
    merged_rows = [row for row in all_rows if row.kind == "merged"]

    registry_check: Check | None = None
    identity_key = manifest_key or "md5"
    if registry is not None:
        canonical, identity_key = _load_registry(registry, manifest_key)
        if all_rows:
            registry_check = _registry_consistency(all_rows, canonical)

    return VerificationResult(
        datasets=specs,
        identity_keys=identity_keys,
        source_split_conflicts=_source_conflicts(source_rows),
        merged_trainval_vs_source_test=_merged_against_source(
            merged_rows,
            source_rows,
            _TRAINVAL,
            frozenset({"test"}),
            "merged train/val vs source test",
        ),
        merged_test_vs_source_trainval=_merged_against_source(
            merged_rows,
            source_rows,
            frozenset({"test"}),
            _TRAINVAL,
            "merged test vs source train/val",
        ),
        merged_duplicate_identities=_merged_duplicates(merged_rows),
        registry_label_consistency=registry_check,
        identity_key=identity_key,
    )


def _format_names(names: Sequence[str]) -> str:
    if not names:
        return "none"
    if len(names) == 1:
        return names[0]
    return f"{names[0]} (+{', +'.join(names[1:])})"


def _row_text(row: ManifestRow) -> str:
    return f"{row.dataset}/{row.split}/{row.cls}: {row.path}"


def _example_lines(
    check: Check, max_examples: int, registry: bool = False
) -> list[str]:
    limit = max(max_examples, 0)
    lines: list[str] = []
    for group in _sorted_groups(check.groups)[:limit]:
        if registry:
            detail = "; ".join(_row_text(row) for row in group.rows)
            lines.append(f"  - {group.identity}: {group.note} ({detail})")
        else:
            rows = sorted(
                group.rows, key=lambda row: (row.dataset, row.split, row.path)
            )
            lines.append(
                f"  - {group.identity}: "
                + " | ".join(_row_text(row) for row in rows)
            )
    remaining = len(check.groups) - limit
    if remaining > 0:
        lines.append(f"  ... and {remaining} more")
    for row in check.missing[:limit]:
        lines.append(f"  - MISSING {row.identity}: {_row_text(row)}")
    missing_remaining = len(check.missing) - limit
    if missing_remaining > 0:
        lines.append(f"  ... and {missing_remaining} missing more")
    return lines


def format_report(result: VerificationResult, max_examples: int = 5) -> str:
    """Render the deterministic plain-text report printed by the CLI."""
    source_names = [d.name for d in result.datasets if d.kind == "source"]
    merged_names = [d.name for d in result.datasets if d.kind == "merged"]
    lines = [
        f"[identity] {result.identity_key}",
        f"[sources] {_format_names(source_names)}",
        f"[merged] {_format_names(merged_names)}",
    ]
    checks: list[tuple[Check | None, bool]] = [
        (result.source_split_conflicts, False),
        (result.merged_trainval_vs_source_test, False),
        (result.merged_test_vs_source_trainval, False),
        (result.merged_duplicate_identities, False),
        (result.registry_label_consistency, True),
    ]
    for check, is_registry in checks:
        if check is None:
            if is_registry:
                lines.append("[check] registry split labels: skipped")
            continue
        status = "PASS" if check.passed else "FAIL"
        missing_suffix = (
            f" ({len(check.missing)} missing from registry)"
            if is_registry and check.missing
            else ""
        )
        lines.append(
            f"[check] {check.name}: {check.count} groups / "
            f"{check.images} images - {status}{missing_suffix}"
        )
        if not check.passed:
            lines.extend(_example_lines(check, max_examples, is_registry))
    if result.passed:
        lines.append("RESULT: PASS")
    else:
        lines.append("RESULT: FAIL (" + "; ".join(result.failures()) + ")")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _default_sources(data_root: Path) -> list[DatasetSpec]:
    missing = [
        name
        for name in _DEFAULT_SOURCE_NAMES
        if not (data_root / name / "manifest.csv").is_file()
    ]
    if missing:
        raise ConfigError(
            f"source manifests missing under {data_root}: {', '.join(missing)}"
        )
    return [
        DatasetSpec(name, data_root / name / "manifest.csv", "source")
        for name in _DEFAULT_SOURCE_NAMES
    ]


def _default_merged(data_root: Path) -> list[DatasetSpec]:
    for name in _DEFAULT_MERGED_NAMES:
        candidate = data_root / name / "manifest.csv"
        if candidate.is_file():
            return [DatasetSpec(name, candidate, "merged")]
    raise ConfigError(
        f"no merged manifest under {data_root}: expected one of "
        f"{', '.join(_DEFAULT_MERGED_NAMES)}"
    )


def _spec_for(path: Path, kind: str) -> DatasetSpec:
    manifest = _resolve_manifest(path)
    return DatasetSpec(manifest.parent.name, manifest, kind)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify that no dataset identity leaks across split roles."
    )
    parser.add_argument(
        "--data-root", type=Path, default=_REPO_ROOT / "data"
    )
    parser.add_argument("--source", action="append", type=Path, default=None)
    parser.add_argument("--merged", action="append", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=5)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    try:
        if args.registry is not None and not args.source and not args.merged:
            sources: list[DatasetSpec] = []
            merged: list[DatasetSpec] = []
        else:
            sources = (
                [_spec_for(path, "source") for path in args.source]
                if args.source
                else _default_sources(args.data_root)
            )
            merged = (
                [_spec_for(path, "merged") for path in args.merged]
                if args.merged
                else _default_merged(args.data_root)
            )
        result = verify(sources, merged, args.registry)
    except ConfigError as exc:
        print(f"[erro] {exc}", file=sys.stderr)
        return 2

    print(format_report(result, args.max_examples))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
