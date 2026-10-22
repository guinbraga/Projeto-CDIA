"""Tests for scripts/verify_dataset_isolation.py."""

import csv
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import verify_dataset_isolation as vdi  # noqa: E402


DEFAULT_FIELDS = [
    "split",
    "class",
    "source",
    "md5",
    "mode",
    "src_path",
    "dst_path",
]


def write_csv(
    path: Path, rows: list[dict[str, str]], fields: list[str]
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fields})
    return path


def write_manifest(
    path: Path,
    rows: list[dict[str, str]],
    fields: list[str] | None = None,
) -> Path:
    return write_csv(path, rows, fields or DEFAULT_FIELDS)


def make_row(identity: str, split: str, cls: str = "A") -> dict[str, str]:
    return {
        "split": split,
        "class": cls,
        "source": "synthetic",
        "md5": identity,
        "mode": "RGB",
        "src_path": "",
        "dst_path": f"/tmp/{split}/{identity}.jpg",
    }


def spec(path: Path, kind: str, name: str | None = None) -> vdi.DatasetSpec:
    return vdi.DatasetSpec(name or path.parent.name, path, kind)


def test_clean_synthetic_manifests_pass(tmp_path: Path, capsys) -> None:
    source = write_manifest(
        tmp_path / "src_a" / "manifest.csv",
        [make_row("m1", "train"), make_row("m2", "test")],
    )
    merged = write_manifest(
        tmp_path / "merged_a" / "manifest.csv",
        [make_row("m1", "train"), make_row("m2", "test")],
    )

    result = vdi.verify([spec(source, "source")], [spec(merged, "merged")])

    assert result.passed is True
    assert result.source_split_conflicts.count == 0
    assert result.merged_trainval_vs_source_test.count == 0
    assert result.merged_test_vs_source_trainval.count == 0
    assert result.merged_duplicate_identities.count == 0
    assert vdi.main(["--source", str(source), "--merged", str(merged)]) == 0
    assert "RESULT: PASS" in capsys.readouterr().out


def test_source_only_api(tmp_path: Path) -> None:
    source = write_manifest(
        tmp_path / "src_a" / "manifest.csv",
        [make_row("m1", "train"), make_row("m2", "test")],
    )

    result = vdi.verify([spec(source, "source")], [])

    assert result.passed is True
    assert result.merged_trainval_vs_source_test.count == 0
    assert result.merged_test_vs_source_trainval.count == 0
    assert result.merged_duplicate_identities.count == 0


def test_source_split_conflict(tmp_path: Path, capsys) -> None:
    source_a = write_manifest(
        tmp_path / "src_a" / "manifest.csv", [make_row("m1", "test")]
    )
    source_b = write_manifest(
        tmp_path / "src_b" / "manifest.csv", [make_row("m1", "train")]
    )
    merged = write_manifest(
        tmp_path / "merged_a" / "manifest.csv", [make_row("m9", "train")]
    )

    result = vdi.verify(
        [spec(source_a, "source"), spec(source_b, "source")],
        [spec(merged, "merged")],
    )

    assert result.source_split_conflicts.count == 1
    assert result.source_split_conflicts.images == 2
    assert result.source_split_conflicts.passed is False
    assert (
        vdi.main(
            [
                "--source",
                str(source_a),
                "--source",
                str(source_b),
                "--merged",
                str(merged),
            ]
        )
        == 1
    )
    assert "FAIL" in capsys.readouterr().out


def test_merged_trainval_vs_source_test(
    tmp_path: Path, capsys
) -> None:
    source = write_manifest(
        tmp_path / "src_a" / "manifest.csv", [make_row("m1", "test")]
    )
    merged = write_manifest(
        tmp_path / "merged_a" / "manifest.csv", [make_row("m1", "train")]
    )

    result = vdi.verify([spec(source, "source")], [spec(merged, "merged")])

    assert result.merged_trainval_vs_source_test.count == 1
    assert result.merged_trainval_vs_source_test.images == 1
    assert result.merged_test_vs_source_trainval.count == 0
    assert vdi.main(["--source", str(source), "--merged", str(merged)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_merged_test_vs_source_trainval(
    tmp_path: Path, capsys
) -> None:
    source = write_manifest(
        tmp_path / "src_a" / "manifest.csv", [make_row("m1", "val")]
    )
    merged = write_manifest(
        tmp_path / "merged_a" / "manifest.csv", [make_row("m1", "test")]
    )

    result = vdi.verify([spec(source, "source")], [spec(merged, "merged")])

    assert result.merged_test_vs_source_trainval.count == 1
    assert result.merged_test_vs_source_trainval.images == 1
    assert result.merged_trainval_vs_source_test.count == 0
    assert vdi.main(["--source", str(source), "--merged", str(merged)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_merged_duplicate_identity(tmp_path: Path) -> None:
    source = write_manifest(
        tmp_path / "src_a" / "manifest.csv", [make_row("m1", "train")]
    )
    merged = write_manifest(
        tmp_path / "merged_a" / "manifest.csv",
        [make_row("m1", "train"), make_row("m1", "val")],
    )

    result = vdi.verify([spec(source, "source")], [spec(merged, "merged")])

    assert result.merged_duplicate_identities.count == 1
    assert result.merged_duplicate_identities.images == 2
    assert result.merged_duplicate_identities.groups[0].note == "merged_a"


def write_registry(
    path: Path,
    rows: list[dict[str, str]],
    fields: list[str] | None = None,
) -> Path:
    return write_csv(path, rows, fields or ["md5", "split"])


def test_registry_consistency(tmp_path: Path) -> None:
    source = write_manifest(
        tmp_path / "src_a" / "manifest.csv",
        [make_row("m1", "train"), make_row("m2", "test")],
    )
    merged = write_manifest(
        tmp_path / "merged_a" / "manifest.csv",
        [make_row("m1", "train"), make_row("m2", "test")],
    )
    sources, merged_specs = [spec(source, "source")], [spec(merged, "merged")]

    good = write_registry(
        tmp_path / "registry.csv",
        [{"md5": "m1", "split": "train"}, {"md5": "m2", "split": "test"}],
    )
    result = vdi.verify(sources, merged_specs, good)
    assert result.registry_label_consistency is not None
    assert result.registry_label_consistency.passed is True
    assert result.passed is True

    wrong = write_registry(
        tmp_path / "wrong.csv",
        [{"md5": "m1", "split": "test"}, {"md5": "m2", "split": "test"}],
    )
    result = vdi.verify(sources, merged_specs, wrong)
    check = result.registry_label_consistency
    assert check is not None
    assert check.count == 1
    assert check.images == 2
    assert result.passed is False
    assert (
        vdi.main(
            [
                "--source",
                str(source),
                "--merged",
                str(merged),
                "--registry",
                str(wrong),
            ]
        )
        == 1
    )
    assert (
        vdi.main(
            [
                "--source",
                str(source),
                "--merged",
                str(merged),
                "--registry",
                str(good),
            ]
        )
        == 0
    )

    partial = write_registry(
        tmp_path / "partial.csv", [{"md5": "m2", "split": "test"}]
    )
    result = vdi.verify(sources, merged_specs, partial)
    check = result.registry_label_consistency
    assert check is not None
    assert check.missing
    assert check.passed is False


def test_cluster_id_keying(tmp_path: Path) -> None:
    fields = ["split", "class", "md5", "cluster_id", "dst_path"]
    separate_clusters = write_manifest(
        tmp_path / "src_a" / "manifest.csv",
        [
            {
                "split": "test",
                "class": "A",
                "md5": "same-md5",
                "cluster_id": "cluster-1",
                "dst_path": "/tmp/a.jpg",
            }
        ],
        fields,
    )
    other_cluster = write_manifest(
        tmp_path / "src_b" / "manifest.csv",
        [
            {
                "split": "train",
                "class": "A",
                "md5": "same-md5",
                "cluster_id": "cluster-2",
                "dst_path": "/tmp/b.jpg",
            }
        ],
        fields,
    )
    merged = write_manifest(
        tmp_path / "merged_a" / "manifest.csv",
        [
            {
                "split": "train",
                "class": "A",
                "md5": "ignored",
                "cluster_id": "cluster-9",
                "dst_path": "/tmp/merged.jpg",
            }
        ],
        fields,
    )

    result = vdi.verify(
        [spec(separate_clusters, "source"), spec(other_cluster, "source")],
        [spec(merged, "merged")],
    )
    assert set(result.identity_keys.values()) == {"cluster_id"}
    assert result.source_split_conflicts.count == 0

    same_cluster = write_manifest(
        tmp_path / "src_c" / "manifest.csv",
        [
            {
                "split": "train",
                "class": "A",
                "md5": "different-md5",
                "cluster_id": "cluster-1",
                "dst_path": "/tmp/c.jpg",
            }
        ],
        fields,
    )
    result = vdi.verify(
        [spec(separate_clusters, "source"), spec(same_cluster, "source")],
        [spec(merged, "merged")],
    )
    assert result.source_split_conflicts.count == 1

    registry = write_registry(
        tmp_path / "cluster_registry.csv",
        [
            {"cluster_id": "cluster-1", "split": "test"},
            {"cluster_id": "cluster-2", "split": "train"},
            {"cluster_id": "cluster-9", "split": "train"},
        ],
        ["cluster_id", "split"],
    )
    result = vdi.verify(
        [spec(separate_clusters, "source"), spec(other_cluster, "source")],
        [spec(merged, "merged")],
        registry,
    )
    assert result.registry_label_consistency is not None
    assert result.registry_label_consistency.passed is True
    assert result.passed is True


def test_mixed_identity_keys_are_config_error(tmp_path: Path, capsys) -> None:
    md5_source = write_manifest(
        tmp_path / "src_md5" / "manifest.csv", [make_row("m1", "train")]
    )
    cluster_source = write_manifest(
        tmp_path / "src_cluster" / "manifest.csv",
        [
            {
                "split": "train",
                "class": "A",
                "cluster_id": "c1",
                "dst_path": "/tmp/c.jpg",
            }
        ],
        ["split", "class", "cluster_id", "dst_path"],
    )
    merged = write_manifest(
        tmp_path / "merged_a" / "manifest.csv", [make_row("m1", "train")]
    )

    with pytest.raises(vdi.ConfigError):
        vdi.verify(
            [spec(md5_source, "source"), spec(cluster_source, "source")],
            [spec(merged, "merged")],
        )

    code = vdi.main(
        [
            "--source",
            str(md5_source),
            "--source",
            str(cluster_source),
            "--merged",
            str(merged),
        ]
    )
    assert code == 2
    assert "[erro]" in capsys.readouterr().err


def test_missing_manifest_and_invalid_split_exit_2(tmp_path: Path, capsys) -> None:
    merged = write_manifest(
        tmp_path / "merged_a" / "manifest.csv", [make_row("m1", "train")]
    )

    missing = vdi.main(
        ["--source", str(tmp_path / "nope"), "--merged", str(merged)]
    )
    assert missing == 2
    assert "[erro]" in capsys.readouterr().err

    invalid = write_manifest(
        tmp_path / "src_bad" / "manifest.csv", [make_row("m1", "bogus")]
    )
    code = vdi.main(["--source", str(invalid), "--merged", str(merged)])
    captured = capsys.readouterr()
    assert code == 2
    assert "[erro]" in captured.err
    assert "Traceback" not in captured.err


def test_format_report_contains_all_checks(tmp_path: Path) -> None:
    source = write_manifest(
        tmp_path / "src_a" / "manifest.csv", [make_row("m1", "test")]
    )
    merged = write_manifest(
        tmp_path / "merged_a" / "manifest.csv", [make_row("m1", "train")]
    )
    report = vdi.format_report(
        vdi.verify([spec(source, "source")], [spec(merged, "merged")])
    )

    assert "[check] source split conflicts (test vs train/val)" in report
    assert "[check] merged train/val vs source test" in report
    assert "[check] merged test vs source train/val" in report
    assert "[check] merged duplicate identities" in report
    assert "RESULT:" in report


# Ticket 07 replaces this pre-registry baseline once cluster ids exist.
def test_real_data_baseline_pre_registry() -> None:
    data = REPO / "data"
    names = (
        "trashnet_split",
        "kaggle_dataset_split",
        "trashbox_split",
        "roboflow_convertido_split",
    )
    manifests = [data / name / "manifest.csv" for name in names]
    merged_manifest = data / "merged_dataset_balanced" / "manifest.csv"
    for manifest in [*manifests, merged_manifest]:
        if not manifest.is_file():
            pytest.skip(f"missing baseline manifest: {manifest}")

    for manifest in [*manifests, merged_manifest]:
        with open(manifest, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "cluster_id" in (reader.fieldnames or []) and any(
                (row.get("cluster_id") or "").strip() for row in reader
            ):
                pytest.skip(
                    "post-registry state: pre-registry baseline no longer applies"
                )

    sources = [
        spec(manifest, "source", name)
        for name, manifest in zip(names, manifests)
    ]
    merged_specs = [
        spec(merged_manifest, "merged", "merged_dataset_balanced")
    ]
    result = vdi.verify(sources, merged_specs)

    assert result.passed is False
    assert result.source_split_conflicts.count == 659
    assert result.source_split_conflicts.images == 1396
    assert result.merged_trainval_vs_source_test.count == 1352
    assert result.merged_trainval_vs_source_test.images == 1352
    assert result.merged_test_vs_source_trainval.count == 1183
    assert result.merged_test_vs_source_trainval.images == 1183
    assert result.merged_duplicate_identities.count == 0

    argv = ["--merged", str(merged_manifest)]
    for manifest in manifests:
        argv.extend(["--source", str(manifest)])
    assert vdi.main(argv) == 1


def write_cluster_registry(
    path: Path,
    rows: list[dict[str, str]],
    fields: list[str] | None = None,
) -> Path:
    return write_csv(
        path, rows, fields or ["cluster_id", "class", "split", "n_images"]
    )


def test_registry_only_verify_passes(tmp_path: Path) -> None:
    registry = write_cluster_registry(
        tmp_path / "global_split_registry.csv",
        [
            {"cluster_id": "c1", "class": "A", "split": "train"},
            {"cluster_id": "c2", "class": "B", "split": "val"},
            {"cluster_id": "c3", "class": "A", "split": "test"},
        ],
    )

    result = vdi.verify([], [], registry)

    assert result.passed is True
    assert result.identity_key == "cluster_id"
    assert result.registry_label_consistency is None

    report = vdi.format_report(result)
    assert "[identity] cluster_id" in report
    assert "registry split labels: skipped" in report
    assert "RESULT: PASS" in report


def test_registry_only_main_passes(tmp_path: Path, capsys) -> None:
    registry = write_cluster_registry(
        tmp_path / "global_split_registry.csv",
        [
            {"cluster_id": "c1", "class": "A", "split": "train"},
            {"cluster_id": "c2", "class": "B", "split": "test"},
        ],
    )

    assert vdi.main(["--registry", str(registry)]) == 0
    assert "RESULT: PASS" in capsys.readouterr().out


def test_registry_only_duplicate_cluster_id_exits_2(
    tmp_path: Path, capsys
) -> None:
    registry = write_cluster_registry(
        tmp_path / "duplicate.csv",
        [
            {"cluster_id": "c1", "class": "A", "split": "train"},
            {"cluster_id": "c1", "class": "A", "split": "test"},
        ],
    )

    code = vdi.main(["--registry", str(registry)])
    captured = capsys.readouterr()
    assert code == 2
    assert "[erro]" in captured.err
    assert "Traceback" not in captured.err


def test_registry_only_invalid_split_exits_2(tmp_path: Path, capsys) -> None:
    registry = write_cluster_registry(
        tmp_path / "invalid_split.csv",
        [{"cluster_id": "c1", "class": "A", "split": "bogus"}],
    )

    code = vdi.main(["--registry", str(registry)])
    captured = capsys.readouterr()
    assert code == 2
    assert "[erro]" in captured.err
    assert "Traceback" not in captured.err


def test_registry_only_missing_identity_column_exits_2(
    tmp_path: Path, capsys
) -> None:
    registry = write_csv(
        tmp_path / "no_identity.csv",
        [{"split": "train"}],
        ["split"],
    )

    code = vdi.main(["--registry", str(registry)])
    captured = capsys.readouterr()
    assert code == 2
    assert "[erro]" in captured.err
    assert "Traceback" not in captured.err


def test_registry_only_md5_keying(tmp_path: Path) -> None:
    registry = write_registry(
        tmp_path / "md5_registry.csv",
        [
            {"md5": "m1", "split": "train"},
            {"md5": "m2", "split": "test"},
        ],
    )

    result = vdi.verify([], [], registry)

    assert result.passed is True
    assert result.identity_key == "md5"
    assert "[identity] md5" in vdi.format_report(result)


def test_registry_only_without_class_column_is_tolerated(
    tmp_path: Path,
) -> None:
    registry = write_csv(
        tmp_path / "no_class.csv",
        [
            {"cluster_id": "c1", "split": "train"},
            {"cluster_id": "c2", "split": "test"},
        ],
        ["cluster_id", "split"],
    )

    result = vdi.verify([], [], registry)

    assert result.passed is True
    assert result.identity_key == "cluster_id"
