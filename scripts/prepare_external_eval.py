#!/usr/bin/env python3
"""Build leakage-free external evaluation targets from raw waste datasets.

The training pipeline in ``dvc.yaml`` owns the four *training* sources. This
command adds *evaluation-only* targets: datasets that no run was trained on and
that the cross-dataset notebook can score as an independent domain shift.

The command is deliberately not a DVC stage, because the raw archives are
large, licensed third-party downloads and are not versioned in this repo. It
still produces a reproducible artifact from a pinned archive: the catalog
records the URL and the SHA-256 of each archive, and the output carries its own
manifest and provenance card.

What it guarantees
------------------
Every candidate image is hashed with the project's own ``md5``/``dHash``/
``pHash`` functions, then clustered **together with** ``data/content_index.csv``
using the same connected-component rule the registry uses
(``dHash <= 1 AND pHash <= 1``). A new image whose cluster contains *any*
already-indexed image is dropped, not relabelled. This is stricter than the
training requirement: overlap with an existing *test* image is removed too, so
two external targets can never double-count the same content.

Only one representative per surviving cluster is written, matching the merged
dataset's one-image-per-cluster rule. A cluster whose members disagree on the
mapped class is dropped as ambiguous, mirroring the registry.

Usage
-----
    # inspect the catalog
    python3 scripts/prepare_external_eval.py --list

    # download (once) and build both targets
    python3 scripts/prepare_external_eval.py --download --overwrite

    # rebuild one target from an already-extracted tree
    python3 scripts/prepare_external_eval.py --dataset realwaste --overwrite

    # add a new dataset ad hoc (not in the catalog)
    python3 scripts/prepare_external_eval.py --name mydata \\
        --raw-root /path/to/class_dirs --class-map mydata.json --overwrite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

import dataset_io
from content_index import ImageRecord, build_index, hash_images

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG = REPO_ROOT / "scripts" / "external_eval_catalog.json"
DEFAULT_CONTENT_INDEX = REPO_ROOT / "data" / "content_index.csv"
DEFAULT_REGISTRY = REPO_ROOT / "data" / "global_split_registry.csv"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "external_eval"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "external_eval"
DATA_ROOT = REPO_ROOT / "data"

REQUIRED_CLASSES: tuple[str, ...] = ("cardboard", "metal", "paper", "plastic")

#: Manifest schema. The hashes let the verifier re-check near-duplicates
#: against the content index without decoding the images again.
EXTERNAL_MANIFEST_FIELDS: tuple[str, ...] = (
    "split",
    "class",
    "source",
    "cluster_id",
    "md5",
    "dhash",
    "phash",
    "mode",
    "src_path",
    "dst_path",
)


class ExternalEvalError(Exception):
    """Raised when a dataset cannot be prepared safely."""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    title: str
    url: str
    license: str
    citation: str
    archive_url: str | None
    archive_sha256: str | None
    extract_to: Path
    raw_root: Path
    class_map: dict[str, str]
    notes: str = ""

    @property
    def canonical_classes(self) -> set[str]:
        return set(self.class_map.values())


@dataclass(frozen=True)
class Candidate:
    cluster_id: str
    src_class: str
    cls: str
    path: Path
    relative: str
    md5: str
    dhash: str
    phash: str


# --------------------------------------------------------------------------- #
# Catalog / discovery
# --------------------------------------------------------------------------- #


def load_catalog(path: Path = DEFAULT_CATALOG) -> dict[str, dict]:
    with open(path, encoding="utf-8") as handle:
        catalog = json.load(handle)
    if not isinstance(catalog, dict):
        raise ExternalEvalError(f"{path}: catalog must be a JSON object")
    return catalog


def _repo_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def spec_from_catalog(
    name: str, entry: dict, *, base: Path = REPO_ROOT
) -> DatasetSpec:
    required = ("title", "raw_root", "class_map")
    missing = [key for key in required if not entry.get(key)]
    if missing:
        raise ExternalEvalError(
            f"catalog entry '{name}' is missing: {', '.join(missing)}"
        )
    class_map = {str(key): str(value) for key, value in entry["class_map"].items()}
    if len(set(class_map.values())) < 2:
        raise ExternalEvalError(
            f"catalog entry '{name}' maps every source class to one target class"
        )
    return DatasetSpec(
        name=name,
        title=str(entry.get("title", name)),
        url=str(entry.get("url", "")),
        license=str(entry.get("license", "")),
        citation=str(entry.get("citation", "")),
        archive_url=entry.get("archive_url"),
        archive_sha256=entry.get("archive_sha256"),
        extract_to=_repo_path(
            entry.get("extract_to", f"data/external_eval_raw/{name}"), base
        ),
        raw_root=_repo_path(entry["raw_root"], base),
        class_map=class_map,
        notes=str(entry.get("notes", "")),
    )


def spec_from_args(name: str, raw_root: Path, class_map_path: Path) -> DatasetSpec:
    with open(class_map_path, encoding="utf-8") as handle:
        raw_map = json.load(handle)
    if not isinstance(raw_map, dict) or not raw_map:
        raise ExternalEvalError(f"{class_map_path}: expected a non-empty JSON object")
    class_map = {str(key): str(value) for key, value in raw_map.items()}
    return DatasetSpec(
        name=name,
        title=name,
        url=str(raw_root),
        license="unspecified",
        citation="",
        archive_url=None,
        archive_sha256=None,
        extract_to=raw_root.parent,
        raw_root=raw_root.resolve(),
        class_map=class_map,
    )


def discover_raw_images(spec: DatasetSpec) -> list[tuple[str, str, Path, str]]:
    """Return ``(source, canonical class, absolute path, relative path)`` jobs.

    Every immediate sub-directory of ``raw_root`` must be named by the class
    map; an unknown directory is an error instead of a silent drop, because a
    typo in the map would otherwise quietly remove a whole class.
    """
    root = spec.raw_root
    if not root.is_dir():
        raise ExternalEvalError(
            f"raw root not found: {root}. Pass --download or check the catalog."
        )
    class_dirs = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    if not class_dirs:
        raise ExternalEvalError(f"raw root has no class directories: {root}")

    unknown = [
        path.name for path in class_dirs if path.name not in spec.class_map
    ]
    if unknown:
        raise ExternalEvalError(
            f"{root}: unmapped class directories {unknown}; add them to the "
            "class map or remove them from the raw tree"
        )
    empty = [
        path.name for path in class_dirs if not any(path.iterdir())
    ]
    if empty:
        raise ExternalEvalError(f"{root}: empty class directories {empty}")

    jobs: list[tuple[str, str, Path, str]] = []
    for class_dir in class_dirs:
        src_class = class_dir.name
        cls = spec.class_map[src_class]
        for dirpath, dirnames, filenames in os.walk(class_dir):
            dirnames.sort()
            for filename in sorted(filenames):
                path = Path(dirpath) / filename
                if not dataset_io.is_image(path):
                    continue
                relative = (
                    f"{spec.name}/{src_class}/{path.relative_to(class_dir).as_posix()}"
                )
                jobs.append((spec.name, cls, path, relative))
    return jobs


# --------------------------------------------------------------------------- #
# Existing corpus
# --------------------------------------------------------------------------- #


def load_existing_records(
    content_index: Path,
) -> tuple[list[ImageRecord], dict[tuple[str, str], str]]:
    """Load the existing index and the original cluster id of each row."""
    if not content_index.is_file():
        raise ExternalEvalError(f"content index not found: {content_index}")
    records: list[ImageRecord] = []
    original_cluster: dict[tuple[str, str], str] = {}
    with open(content_index, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"source", "class", "path", "md5", "dhash", "phash"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ExternalEvalError(
                f"{content_index}: missing columns {sorted(missing)}"
            )
        for row in reader:
            key = (row["source"], row["path"])
            records.append(
                ImageRecord(
                    source=row["source"],
                    cls=row["class"],
                    path=row["path"],
                    md5=row["md5"],
                    dhash=row["dhash"],
                    phash=row["phash"],
                )
            )
            original_cluster[key] = row.get("cluster_id", "")
    return records, original_cluster


def load_prepared_records(
    dataset_root: Path, exclude: str
) -> tuple[list[ImageRecord], dict[tuple[str, str], str]]:
    """Load every already-written external target except ``exclude``.

    Including earlier external targets keeps the external family mutually
    disjoint: a later dataset cannot reuse an image that an earlier target
    already owns. The rows are read from the manifests, so this needs no image
    decoding.
    """
    records: list[ImageRecord] = []
    originals: dict[tuple[str, str], str] = {}
    if not dataset_root.is_dir():
        return records, originals
    for manifest in sorted(dataset_root.glob("*/manifest.csv")):
        name = manifest.parent.name
        if name == exclude:
            continue
        with open(manifest, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not {"class", "md5", "dhash", "phash"}.issubset(
                set(reader.fieldnames or [])
            ):
                print(
                    f"[warn] {manifest}: missing hash columns; "
                    "cross-external dedupe skips it",
                    file=sys.stderr,
                )
                continue
            for row in reader:
                relative = f"{name}/{row['class']}/{Path(row['dst_path']).name}"
                key = (name, relative)
                records.append(
                    ImageRecord(
                        source=name,
                        cls=row["class"],
                        path=relative,
                        md5=row["md5"],
                        dhash=row["dhash"],
                        phash=row["phash"],
                    )
                )
                originals[key] = row.get("cluster_id", "")
    return records, originals


def load_registry(path: Path) -> dict[str, tuple[str, str, str]]:
    """Return ``cluster_id -> (split, class, canonical_source)``."""
    if not path.is_file():
        raise ExternalEvalError(f"registry not found: {path}")
    registry: dict[str, tuple[str, str, str]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cluster_id = (row.get("cluster_id") or "").strip()
            if not cluster_id:
                continue
            registry[cluster_id] = (
                row.get("split", ""),
                row.get("class", ""),
                row.get("canonical_source", ""),
            )
    return registry


# --------------------------------------------------------------------------- #
# Clustering + filtering
# --------------------------------------------------------------------------- #


@dataclass
class DedupePlan:
    kept: list[Candidate]
    contaminated: list[dict]
    ambiguous: list[dict]
    raw_count: int
    readable_count: int
    unreadable_count: int
    cluster_count: int


def plan_dedupe(
    spec: DatasetSpec,
    jobs: list[tuple[str, str, Path, str]],
    existing_records: list[ImageRecord],
    original_cluster: dict[tuple[str, str], str],
    registry: dict[str, tuple[str, str, str]],
    workers: int | None,
) -> DedupePlan:
    new_records, unreadable = hash_images(jobs, workers)
    if not new_records:
        raise ExternalEvalError(f"{spec.name}: no readable images in {spec.raw_root}")

    combined = build_index(existing_records + new_records, unreadable)
    # The content index is the identity authority for the existing corpus; the
    # recomputed component is the authority for cross-corpus overlap. Adding
    # new nodes can merge two existing clusters, so never compare new rows to
    # the old cluster id alone.
    new_keys = {(record.source, record.path) for record in new_records}
    row_by_key = {
        (row.source, row.path): row
        for row in combined.rows
        if row.source == spec.name
    }
    members: dict[str, list] = defaultdict(list)
    for row in combined.rows:
        members[row.cluster_id].append(row)

    abs_by_key = {(source, relative): path for source, _, path, relative in jobs}

    kept: list[Candidate] = []
    contaminated: list[dict] = []
    ambiguous: list[dict] = []
    for cluster_id, rows in members.items():
        new_rows = [row for row in rows if (row.source, row.path) in new_keys]
        if not new_rows:
            continue
        existing_rows = [row for row in rows if (row.source, row.path) not in new_keys]

        if existing_rows:
            splits = Counter()
            sources = Counter()
            for row in existing_rows:
                original = original_cluster.get((row.source, row.path), "")
                split, cls, _source = registry.get(
                    original, ("not_registered", row.cls, row.source)
                )
                splits[split] += 1
                sources[f"{row.source}:{cls}"] += 1
            # Attribute each dropped new image to the most severe role its
            # cluster touches, so the counts partition the dropped images.
            dropped_by_role: Counter = Counter()
            for row in new_rows:
                role = next(
                    (
                        name
                        for name in ("train", "val", "test", "not_registered")
                        if name in splits
                    ),
                    "not_registered",
                )
                dropped_by_role[role] += 1
            contaminated.append(
                {
                    "cluster_id": cluster_id,
                    "new_images": len(new_rows),
                    "new_classes": sorted({row.cls for row in new_rows}),
                    "new_images_by_role": dict(sorted(dropped_by_role.items())),
                    "existing_images": len(existing_rows),
                    "existing_sources": dict(sorted(sources.items())),
                    "registry_splits": dict(sorted(splits.items())),
                }
            )
            continue

        classes = sorted({row.cls for row in new_rows})
        if len(classes) > 1:
            ambiguous.append(
                {
                    "cluster_id": cluster_id,
                    "images": len(new_rows),
                    "classes": classes,
                }
            )
            continue

        # cluster_id is the smallest md5 in the component by construction.
        representative = min(
            new_rows, key=lambda row: (row.md5 != cluster_id, row.md5, row.path)
        )
        key = (representative.source, representative.path)
        kept.append(
            Candidate(
                cluster_id=cluster_id,
                src_class=representative.path.split("/")[1],
                cls=classes[0],
                path=abs_by_key[key],
                relative=representative.path,
                md5=representative.md5,
                dhash=representative.dhash,
                phash=representative.phash,
            )
        )

    kept.sort(key=lambda candidate: (candidate.cls, candidate.relative))
    return DedupePlan(
        kept=kept,
        contaminated=sorted(contaminated, key=lambda item: item["cluster_id"]),
        ambiguous=sorted(ambiguous, key=lambda item: item["cluster_id"]),
        raw_count=len(jobs),
        readable_count=len(new_records),
        unreadable_count=len(unreadable),
        cluster_count=sum(1 for rows in members.values() if any(
            (row.source, row.path) in new_keys for row in rows
        )),
    )


# --------------------------------------------------------------------------- #
# Materialisation
# --------------------------------------------------------------------------- #


def materialize_target(
    spec: DatasetSpec,
    plan: DedupePlan,
    dataset_root: Path,
    overwrite: bool,
) -> tuple[Path, list[dict[str, str]]]:
    output = dataset_root / spec.name
    dataset_io.reset_output(output, overwrite)
    test_dir = output / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str]] = []
    used_names: dict[Path, set[str]] = {}
    for candidate in plan.kept:
        dst_dir = test_dir / candidate.cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        used = used_names.setdefault(dst_dir, set())
        filename = dataset_io.unique_name(candidate.path.name, used)
        used.add(filename)
        dst_path = dst_dir / filename
        with Image.open(candidate.path) as image:
            mode = image.mode
        dataset_io.write_image(
            candidate.path,
            dst_path,
            use_link=False,
            convert_rgb=True,
            mode=mode,
        )
        manifest.append(
            {
                "split": "test",
                "class": candidate.cls,
                "source": spec.name,
                "cluster_id": candidate.cluster_id,
                "md5": candidate.md5,
                "dhash": candidate.dhash,
                "phash": candidate.phash,
                "mode": mode,
                "src_path": str(candidate.path),
                "dst_path": str(dst_path),
            }
        )

    dataset_io.write_manifest(manifest, output, fields=EXTERNAL_MANIFEST_FIELDS)
    return output, manifest


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def summarize(
    spec: DatasetSpec,
    plan: DedupePlan,
    manifest: list[dict[str, str]],
    content_index: Path,
    registry_path: Path,
    output: Path,
    require_classes: tuple[str, ...],
) -> dict:
    class_counts = Counter(row["class"] for row in manifest)
    overlap_split = Counter()
    dropped_split = Counter()
    overlap_source = Counter()
    for item in plan.contaminated:
        for split, count in item["registry_splits"].items():
            overlap_split[split] += count
        for split, count in item["new_images_by_role"].items():
            dropped_split[split] += count
        for source, count in item["existing_sources"].items():
            overlap_source[source] += count

    kept_clusters = {row["cluster_id"] for row in manifest if row["cluster_id"]}
    required_present = {
        name: class_counts.get(name, 0) for name in require_classes
    }
    report = {
        "dataset": spec.name,
        "title": spec.title,
        "url": spec.url,
        "license": spec.license,
        "citation": spec.citation,
        "notes": spec.notes,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sources": {
            "raw_root": str(spec.raw_root),
            "archive_url": spec.archive_url,
            "archive_sha256": spec.archive_sha256,
            "content_index": str(content_index),
            "registry": str(registry_path),
        },
        "class_map": spec.class_map,
        "identity": {
            "method": "combined md5/dHash/pHash connected components "
            "(dHash <= 1 AND pHash <= 1), same rule as content_index",
        },
        "counts": {
            "raw_images": plan.raw_count,
            "unreadable_images": plan.unreadable_count,
            "candidate_clusters": plan.cluster_count,
            "redundant_images": plan.readable_count - plan.cluster_count,
            "kept_images": len(manifest),
            "dropped_overlap_clusters": len(plan.contaminated),
            "dropped_overlap_images": sum(
                item["new_images"] for item in plan.contaminated
            ),
            "dropped_ambiguous_clusters": len(plan.ambiguous),
            "dropped_ambiguous_images": sum(
                item["images"] for item in plan.ambiguous
            ),
            "kept_clusters": len(kept_clusters),
        },
        "classes": dict(sorted(class_counts.items())),
        "required_classes": required_present,
        "missing_required_classes": [
            name for name, count in required_present.items() if count == 0
        ],
        "dropped_new_images_by_registry_role": dict(sorted(dropped_split.items())),
        "existing_images_by_registry_split": dict(sorted(overlap_split.items())),
        "existing_images_by_source": dict(sorted(overlap_source.items())),
        "overlap_examples": plan.contaminated[:20],
        "ambiguous_examples": plan.ambiguous[:20],
        "output": str(output),
    }
    return report


def write_reports(
    report: dict,
    plan: DedupePlan,
    report_root: Path,
) -> None:
    target = report_root / report["dataset"]
    target.mkdir(parents=True, exist_ok=True)
    with open(target / "dedupe_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
        handle.write("\n")

    with open(target / "overlap_clusters.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cluster_id",
                "new_images",
                "new_classes",
                "new_images_by_role",
                "existing_images",
                "existing_sources",
                "registry_splits",
            ],
        )
        writer.writeheader()
        for item in plan.contaminated:
            writer.writerow(
                {
                    "cluster_id": item["cluster_id"],
                    "new_images": item["new_images"],
                    "new_classes": "|".join(item["new_classes"]),
                    "new_images_by_role": ";".join(
                        f"{key}={value}"
                        for key, value in item["new_images_by_role"].items()
                    ),
                    "existing_images": item["existing_images"],
                    "existing_sources": ";".join(
                        f"{key}={value}"
                        for key, value in item["existing_sources"].items()
                    ),
                    "registry_splits": ";".join(
                        f"{key}={value}"
                        for key, value in item["registry_splits"].items()
                    ),
                }
            )

    with open(target / "ambiguous_clusters.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["cluster_id", "images", "classes"]
        )
        writer.writeheader()
        for item in plan.ambiguous:
            writer.writerow(
                {
                    "cluster_id": item["cluster_id"],
                    "images": item["images"],
                    "classes": "|".join(item["classes"]),
                }
            )

    with open(target / "kept_by_class.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "images"])
        writer.writeheader()
        for cls, count in sorted(report["classes"].items()):
            writer.writerow({"class": cls, "images": count})


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            resolved = (destination / member.filename).resolve()
            if (
                member.filename.startswith("/")
                or ".." in Path(member.filename).parts
                or destination.resolve() not in resolved.parents
                and resolved != destination.resolve()
            ):
                raise ExternalEvalError(
                    f"{archive}: unsafe member path {member.filename!r}"
                )
        bundle.extractall(destination)


def download_dataset(spec: DatasetSpec, *, force: bool = False) -> None:
    if spec.raw_root.is_dir() and any(spec.raw_root.iterdir()) and not force:
        print(f"[{spec.name}] raw tree already present: {spec.raw_root}")
        return
    if not spec.archive_url:
        raise ExternalEvalError(f"{spec.name}: no archive_url in the catalog")
    archive = DATA_ROOT / "external_eval_raw" / f"{spec.name}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if force or not archive.is_file():
        print(f"[{spec.name}] downloading {spec.archive_url}")
        request = urllib.request.Request(
            spec.archive_url, headers={"User-Agent": "trash-external-eval/1.0"}
        )
        with urllib.request.urlopen(request) as response, open(
            archive, "wb"
        ) as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
    if spec.archive_sha256:
        hasher = hashlib.sha256()
        with open(archive, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest != spec.archive_sha256:
            raise ExternalEvalError(
                f"{spec.name}: archive sha256 mismatch (got {digest}, "
                f"expected {spec.archive_sha256})"
            )
    print(f"[{spec.name}] extracting to {spec.extract_to}")
    _safe_extract(archive, spec.extract_to)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def prepare_one(
    spec: DatasetSpec,
    *,
    content_index: Path,
    registry_path: Path,
    dataset_root: Path,
    report_root: Path,
    workers: int | None,
    overwrite: bool,
    require_classes: tuple[str, ...],
    include_prepared: bool = True,
) -> int:
    jobs = discover_raw_images(spec)
    existing_records, original_cluster = load_existing_records(content_index)
    if include_prepared:
        prepared_records, prepared_clusters = load_prepared_records(
            dataset_root, spec.name
        )
        existing_records.extend(prepared_records)
        original_cluster.update(prepared_clusters)
    registry = load_registry(registry_path)

    plan = plan_dedupe(
        spec, jobs, existing_records, original_cluster, registry, workers
    )
    output, manifest = materialize_target(spec, plan, dataset_root, overwrite)
    report = summarize(
        spec,
        plan,
        manifest,
        content_index,
        registry_path,
        output,
        require_classes,
    )
    write_reports(report, plan, report_root)

    counts = report["counts"]
    print(
        f"[{spec.name}] raw={counts['raw_images']} "
        f"unreadable={counts['unreadable_images']} "
        f"clusters={counts['candidate_clusters']} "
        f"dups={counts['redundant_images']} "
        f"overlap={counts['dropped_overlap_images']} "
        f"ambiguous={counts['dropped_ambiguous_images']} "
        f"kept={counts['kept_images']}"
    )
    print(
        f"[{spec.name}] kept by class: "
        + ", ".join(f"{cls}={count}" for cls, count in sorted(report["classes"].items()))
    )
    print(f"[{spec.name}] target -> {output}")
    print(f"[{spec.name}] report -> {report_root / spec.name / 'dedupe_report.json'}")

    violations: list[str] = []
    contaminated_ids = {item["cluster_id"] for item in plan.contaminated}
    kept_clusters = [row["cluster_id"] for row in manifest]
    if any(not cluster_id for cluster_id in kept_clusters):
        violations.append("empty cluster_id in the written manifest")
    if contaminated_ids.intersection(kept_clusters):
        violations.append("a contaminated cluster reached the manifest")
    if len(set(kept_clusters)) != len(kept_clusters):
        violations.append("duplicate cluster ids in the written manifest")
    for name, count in report["required_classes"].items():
        if count == 0:
            violations.append(f"required class '{name}' has no kept images")
    if violations:
        print(f"[{spec.name}] FAIL: " + "; ".join(violations), file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-free external evaluation targets by clustering raw "
            "images together with data/content_index.csv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--catalog", type=Path, default=DEFAULT_CATALOG, help="dataset catalog"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="catalog entry to prepare (repeatable); default: all entries",
    )
    parser.add_argument("--name", help="ad-hoc dataset name (requires --raw-root)")
    parser.add_argument("--raw-root", type=Path, help="ad-hoc class-directory root")
    parser.add_argument("--class-map", type=Path, help="ad-hoc source->target JSON map")
    parser.add_argument(
        "--content-index", type=Path, default=DEFAULT_CONTENT_INDEX
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT
    )
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="hashing processes"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing target"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="download and extract catalog archives when raw_root is missing",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="re-download even when the archive/tree already exists",
    )
    parser.add_argument(
        "--require-classes",
        default=",".join(REQUIRED_CLASSES),
        help="comma-separated classes the target must contain",
    )
    parser.add_argument(
        "--no-include-prepared",
        action="store_true",
        help=(
            "do not dedupe against targets already under --dataset-root; "
            "only useful when deliberately rebuilding one target in isolation"
        ),
    )
    parser.add_argument(
        "--list", action="store_true", help="list catalog entries and exit"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_classes = tuple(
        name.strip() for name in args.require_classes.split(",") if name.strip()
    )
    try:
        catalog = load_catalog(args.catalog)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[erro] cannot read catalog: {exc}", file=sys.stderr)
        return 2

    if args.list:
        for name, entry in catalog.items():
            print(
                f"{name:<12} {entry.get('title', '')} | "
                f"{entry.get('license', '')} | {entry.get('raw_root', '')}"
            )
        return 0

    if args.name:
        if not args.raw_root or not args.class_map:
            print(
                "[erro] --name requires --raw-root and --class-map",
                file=sys.stderr,
            )
            return 2
        specs = [spec_from_args(args.name, args.raw_root, args.class_map)]
    else:
        names = args.dataset or list(catalog)
        missing = [name for name in names if name not in catalog]
        if missing:
            print(
                f"[erro] unknown catalog entries: {', '.join(missing)}",
                file=sys.stderr,
            )
            return 2
        specs = [spec_from_catalog(name, catalog[name]) for name in names]
        # Process in catalog order so a multi-dataset run allocates shared
        # content deterministically, regardless of --dataset order.
        catalog_order = {name: index for index, name in enumerate(catalog)}
        specs.sort(key=lambda spec: catalog_order[spec.name])

    status = 0
    for spec in specs:
        try:
            if args.download or args.force_download:
                download_dataset(spec, force=args.force_download)
            status = max(
                status,
                prepare_one(
                    spec,
                    content_index=args.content_index,
                    registry_path=args.registry,
                    dataset_root=args.dataset_root,
                    report_root=args.report_root,
                    workers=args.workers,
                    overwrite=args.overwrite,
                    require_classes=require_classes,
                    include_prepared=not args.no_include_prepared,
                ),
            )
        except (ExternalEvalError, OSError, zipfile.BadZipFile) as exc:
            print(f"[erro] {spec.name}: {exc}", file=sys.stderr)
            status = 2
    return status


if __name__ == "__main__":
    raise SystemExit(main())
