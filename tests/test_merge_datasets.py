"""Tests for the registry-driven merged dataset.

The merged dataset is a pure projection of the global split registry plus the
four source split manifests: it never discovers raw files, dedupes or re-splits.
These tests pin the contracts that make that safe:

* one representative per included cluster, at most once;
* ``test`` is the full union of registry ``test`` clusters;
* ``train``/``val`` are balanced per ``(class, source)`` with the smallest
  source as the quota, except ``tetra_pack`` which is join-all;
* ``--balance-classes`` only changes ``train``/``val``;
* rebuilding with the same seed is byte-for-byte reproducible.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import dataset_io  # noqa: E402
import merge_datasets  # noqa: E402
from merge_datasets import (  # noqa: E402
    ManifestImage,
    MergeError,
    MergePlan,
    RegistryCluster,
    plan_merge,
    validate_plan,
)

A = "source_a"
B = "source_b"
C = "source_c"


def registry_row(
    cluster_id: str,
    cls: str,
    split: str,
    canonical_source: str = A,
    n_images: int = 1,
) -> RegistryCluster:
    return RegistryCluster(cluster_id, cls, split, canonical_source, n_images)


def image(
    cluster_id: str,
    source: str,
    cls: str = "cardboard",
    split: str = "train",
    path: str | None = None,
) -> ManifestImage:
    return ManifestImage(
        cluster_id=cluster_id,
        source=source,
        cls=cls,
        split=split,
        md5=f"md5-{cluster_id}-{source}",
        mode="RGB",
        path=Path(path or f"/images/{source}/{cluster_id}.jpg"),
    )


def split_sizes(plan: merge_datasets.MergePlan) -> dict[str, int]:
    return {split: len(rows) for split, rows in plan.records.items()}


def source_counts(
    plan: merge_datasets.MergePlan, splits: tuple[str, ...] = ("train", "val")
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in splits:
        for _cls, source, _path in plan.records[split]:
            counts[source] = counts.get(source, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Split assignment / balancing
# --------------------------------------------------------------------------- #


def test_shared_clusters_are_credited_to_the_scarcest_source():
    """A shared cluster goes to the source with the fewest clusters so far.

    a1/a2 are exclusive to A, b1 is exclusive to B and a3 is shared. Crediting
    a3 to B (it has fewer clusters) leaves A=2, B=2, so the quota is 2 and each
    source keeps 2 clusters.
    """
    registry = [
        registry_row("a1", "cardboard", "train"),
        registry_row("a2", "cardboard", "val"),
        registry_row("a3", "cardboard", "train"),
        registry_row("b1", "cardboard", "train"),
    ]
    images = [
        image("a1", A),
        image("a2", A, split="val"),
        image("a3", A),
        image("a3", B),
        image("b1", B),
    ]

    plan = plan_merge(registry, images, seed=42)

    assert plan.quota["cardboard"] == 2
    assert source_counts(plan) == {A: 2, B: 2}
    assert len(plan.records["train"]) + len(plan.records["val"]) == 4


def test_source_whose_only_cluster_is_credited_elsewhere_contributes_zero():
    """An availability cap of 0 must not force the class quota to 0.

    B and C both only appear in the shared cluster s1. Greedy attribution gives
    s1 to the lexicographically first source (B), leaving C with 0 clusters.
    C contributes 0 (its cap); A and B still contribute the quota of 1.
    """
    registry = [
        registry_row("a1", "cardboard", "train"),
        registry_row("a2", "cardboard", "train"),
        registry_row("s1", "cardboard", "train"),
    ]
    images = [
        image("a1", A),
        image("a2", A),
        image("s1", B),
        image("s1", C),
    ]

    plan = plan_merge(registry, images, seed=42)

    assert plan.availability["cardboard"] == {A: 2, B: 1, C: 0}
    assert plan.quota["cardboard"] == 1
    assert source_counts(plan) == {A: 1, B: 1}


def test_train_val_sources_all_hit_the_same_quota_per_class():
    registry = [
        registry_row("a1", "cardboard", "train"),
        registry_row("a2", "cardboard", "train"),
        registry_row("a3", "cardboard", "train"),
        registry_row("b1", "cardboard", "train"),
        registry_row("b2", "cardboard", "train"),
    ]
    images = [image(cid, A) for cid in ("a1", "a2", "a3")] + [
        image(cid, B) for cid in ("b1", "b2")
    ]

    plan = plan_merge(registry, images, seed=42)

    assert plan.quota["cardboard"] == 2
    assert source_counts(plan) == {A: 2, B: 2}


def test_tetra_pack_is_join_all_and_ignores_the_quota():
    registry = [
        registry_row("p1", "tetra_pack", "train"),
        registry_row("p2", "tetra_pack", "train"),
        registry_row("p3", "tetra_pack", "val"),
    ]
    images = [
        image("p1", A, cls="tetra_pack"),
        image("p2", A, cls="tetra_pack"),
        image("p2", B, cls="tetra_pack"),
        image("p3", B, cls="tetra_pack", split="val"),
    ]

    plan = plan_merge(registry, images, seed=42)

    assert len(plan.records["train"]) + len(plan.records["val"]) == 3
    assert plan.quota["tetra_pack"] == 3


def test_test_split_is_the_full_union_of_registry_test_clusters():
    registry = [
        registry_row("t1", "cardboard", "test"),
        registry_row("t2", "cardboard", "test"),
        registry_row("tr", "cardboard", "train"),
    ]
    images = [
        image("t1", A, split="test"),
        image("t2", B, split="test"),
        image("tr", A),
    ]

    plan = plan_merge(registry, images, seed=42)

    test_ids = [plan.cluster_ids[path] for _cls, _src, path in plan.records["test"]]
    assert sorted(test_ids) == ["t1", "t2"]
    assert len(test_ids) == len(set(test_ids))


def test_test_selection_never_changes_with_class_balancing():
    registry = [
        registry_row("t1", "cardboard", "test"),
        registry_row("a1", "cardboard", "train"),
        registry_row("a2", "cardboard", "train"),
        registry_row("a3", "glass", "train"),
        registry_row("a4", "glass", "train"),
    ]
    images = [
        image("t1", A, split="test"),
        image("a1", A),
        image("a2", B),
        image("a3", A, cls="glass"),
        image("a4", B, cls="glass"),
    ]

    plain = plan_merge(registry, images, seed=42)
    balanced = plan_merge(registry, images, seed=42, balance_classes=True)

    assert plain.records["test"] == balanced.records["test"]


def test_balance_classes_is_opt_in_and_only_shrinks_train_val():
    cardboard_ids = [f"c{i}" for i in range(8)]
    glass_ids = [f"g{i}" for i in range(4)]
    registry = (
        [registry_row(cid, "cardboard", "train") for cid in cardboard_ids]
        + [registry_row(cid, "glass", "train") for cid in glass_ids]
        + [registry_row("t1", "cardboard", "test")]
    )
    images = [
        image(cid, A if i % 2 == 0 else B)
        for i, cid in enumerate(cardboard_ids)
    ] + [
        image(cid, A if i % 2 == 0 else B, cls="glass")
        for i, cid in enumerate(glass_ids)
    ] + [image("t1", A, split="test")]

    plain = plan_merge(registry, images, seed=42)
    balanced = plan_merge(registry, images, seed=42, balance_classes=True)

    plain_trainval = split_sizes(plain)["train"] + split_sizes(plain)["val"]
    balanced_trainval = split_sizes(balanced)["train"] + split_sizes(balanced)["val"]
    assert plain_trainval == 12
    assert balanced_trainval < plain_trainval

    per_class: dict[str, int] = {}
    for split in ("train", "val"):
        for cls, _source, _path in balanced.records[split]:
            per_class[cls] = per_class.get(cls, 0) + 1
    assert per_class == {"cardboard": 4, "glass": 4}
    assert split_sizes(balanced)["test"] == 1


def test_plan_is_reproducible_with_the_same_seed():
    ids = [f"c{i}" for i in range(10)]
    registry = [registry_row(cid, "cardboard", "train") for cid in ids]
    images = [image(cid, A if i < 6 else B) for i, cid in enumerate(ids)]

    first = plan_merge(registry, images, seed=7)
    second = plan_merge(registry, images, seed=7)

    assert first.records == second.records
    assert first.cluster_ids == second.cluster_ids


# --------------------------------------------------------------------------- #
# Registry authority
# --------------------------------------------------------------------------- #


def test_non_target_classes_are_not_materialised():
    registry = [
        registry_row("e1", "e-waste", "train"),
        registry_row("a1", "cardboard", "train"),
    ]
    images = [
        image("e1", A, cls="e-waste"),
        image("a1", A),
    ]

    plan = plan_merge(registry, images, seed=42)

    materialised = {
        plan.cluster_ids[path]
        for rows in plan.records.values()
        for _cls, _src, path in rows
    }
    assert materialised == {"a1"}


def test_manifest_split_must_match_the_registry():
    registry = [registry_row("a1", "cardboard", "val")]
    images = [image("a1", A, split="train")]

    with pytest.raises(MergeError, match="a1"):
        plan_merge(registry, images, seed=42)


def test_manifest_cluster_must_exist_in_the_registry():
    registry: list[RegistryCluster] = []
    images = [image("missing", A)]

    with pytest.raises(MergeError, match="missing"):
        plan_merge(registry, images, seed=42)


def test_validate_plan_rejects_a_duplicate_cluster():
    registry = [
        registry_row("a1", "cardboard", "train"),
        registry_row("a2", "cardboard", "train"),
    ]
    images = [image("a1", A), image("a2", A)]
    plan = plan_merge(registry, images, seed=42)

    duplicated = MergePlan(
        records={
            "train": list(plan.records["train"]) + list(plan.records["train"]),
            "val": [],
            "test": [],
        },
        infos=plan.infos,
        cluster_ids=plan.cluster_ids,
        availability=plan.availability,
        quota=plan.quota,
        balance_classes=False,
        totals={"train": 4, "val": 0, "test": 0},
    )
    with pytest.raises(MergeError, match="more than once"):
        validate_plan(duplicated, registry)


# --------------------------------------------------------------------------- #
# CLI / materialisation
# --------------------------------------------------------------------------- #


def write_registry(path: Path, rows: list[RegistryCluster]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["cluster_id", "class", "split", "canonical_source", "n_images"]
        )
        for row in rows:
            writer.writerow(
                [row.cluster_id, row.cls, row.split, row.canonical_source, 1]
            )


def write_source_manifest(
    directory: Path,
    images: list[ManifestImage],
    color: tuple[int, int, int],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in images:
        dst = directory / item.split / item.cls / item.path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2), color).save(dst, format="JPEG")
        rows.append(
            {
                "split": item.split,
                "class": item.cls,
                "source": item.source,
                "cluster_id": item.cluster_id,
                "md5": item.md5,
                "mode": item.mode,
                "src_path": str(dst),
                "dst_path": str(dst),
            }
        )
    with open(directory / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(dataset_io.MANIFEST_FIELDS_WITH_CLUSTER)
        )
        writer.writeheader()
        writer.writerows(rows)


def test_main_materialises_one_representative_per_cluster(tmp_path):
    registry_path = tmp_path / "registry.csv"
    source_a = tmp_path / "source_a_split"
    source_b = tmp_path / "source_b_split"
    output = tmp_path / "merged"

    registry = [
        registry_row("a1", "cardboard", "train"),
        registry_row("a2", "cardboard", "train"),
        registry_row("b1", "cardboard", "train", canonical_source=B),
        registry_row("t1", "cardboard", "test"),
        registry_row("p1", "tetra_pack", "train"),
    ]
    write_registry(registry_path, registry)
    write_source_manifest(
        source_a,
        [
            image("a1", A),
            image("a2", A),
            image("t1", A, split="test"),
            image("p1", A, cls="tetra_pack"),
        ],
        (255, 0, 0),
    )
    write_source_manifest(
        source_b,
        [
            image("b1", B),
            image("t1", B, split="test"),
        ],
        (0, 0, 255),
    )

    exit_code = merge_datasets.main(
        [
            "--registry",
            str(registry_path),
            "--source-manifest",
            str(source_a),
            "--source-manifest",
            str(source_b),
            "--output",
            str(output),
            "--seed",
            "42",
        ]
    )

    assert exit_code == 0
    rows = dataset_io.read_manifest(output)
    assert tuple(rows[0].keys()) == dataset_io.MANIFEST_FIELDS_WITH_CLUSTER
    ids = [row["cluster_id"] for row in rows]
    assert len(ids) == len(set(ids))
    assert "t1" in ids
    for row in rows:
        assert row["source"] in {A, B}
        assert Path(row["dst_path"]).is_file()
        entry = next(item for item in registry if item.cluster_id == row["cluster_id"])
        assert row["class"] == entry.cls
        assert row["split"] == entry.split


def test_main_dry_run_does_not_create_the_output(tmp_path):
    registry_path = tmp_path / "registry.csv"
    source = tmp_path / "source_split"
    output = tmp_path / "merged"

    write_registry(registry_path, [registry_row("a1", "cardboard", "train")])
    write_source_manifest(source, [image("a1", A)], (0, 255, 0))

    exit_code = merge_datasets.main(
        [
            "--registry",
            str(registry_path),
            "--source-manifest",
            str(source),
            "--output",
            str(output),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert not output.exists()


def _materialised_fixture(tmp_path: Path) -> tuple[Path, list[str]]:
    registry_path = tmp_path / "registry.csv"
    source = tmp_path / "source_split"
    output = tmp_path / "merged"
    write_registry(
        registry_path,
        [
            registry_row("a1", "cardboard", "train"),
            registry_row("a2", "cardboard", "train"),
            registry_row("t1", "cardboard", "test"),
        ],
    )
    write_source_manifest(
        source,
        [
            image("a1", A),
            image("a2", A),
            image("t1", A, split="test"),
        ],
        (10, 20, 30),
    )
    argv = [
        "--registry",
        str(registry_path),
        "--source-manifest",
        str(source),
        "--output",
        str(output),
        "--seed",
        "42",
    ]
    assert merge_datasets.main(argv) == 0
    return output, argv


def test_verify_existing_accepts_the_materialised_output(tmp_path):
    output, argv = _materialised_fixture(tmp_path)

    assert (
        merge_datasets.main([*argv, "--verify-existing", str(output)]) == 0
    )


def test_verify_existing_rejects_a_changed_manifest(tmp_path, capsys):
    output, argv = _materialised_fixture(tmp_path)
    rows = [
        row
        for row in dataset_io.read_manifest(output)
        if row["cluster_id"] != "a2"
    ]
    with open(output / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    code = merge_datasets.main([*argv, "--verify-existing", str(output)])

    assert code == 1
    assert "does not match the registry plan" in capsys.readouterr().err
