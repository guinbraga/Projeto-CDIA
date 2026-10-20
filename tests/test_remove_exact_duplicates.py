import csv
import hashlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import remove_exact_duplicates as dedupe  # noqa: E402

FIELDS = ["dataset", "dup_group_id", "group_size", "md5", "cls",
          "keep", "redundant", "path"]


def file_md5(data):
    return hashlib.md5(data).hexdigest()


def make_file(root, rel, data=b"duplicate-bytes"):
    path = Path(root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def make_row(dataset, group_id, digest, cls, keep, path):
    return {
        "dataset": dataset,
        "dup_group_id": str(group_id),
        "group_size": "2",
        "md5": digest,
        "cls": cls,
        "keep": keep,
        "redundant": "no" if keep == "yes" else "yes",
        "path": str(path),
    }


def run(tmp_path, rows, *extra, data_root=None):
    csv_path = tmp_path / "duplicates.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    root = Path(data_root) if data_root else tmp_path / "data"
    manifest = tmp_path / "manifest.csv"
    args = ["--csv", str(csv_path), "--data-root", str(root),
            "--manifest", str(manifest), *extra]
    return dedupe.main(args), manifest


def read_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_shortest_keep_rule_keeps_shortest_basename(tmp_path):
    root = tmp_path / "data"
    keep = make_file(root, "kaggle_dataset/convertido/cardboard/a.jpg")
    drop = make_file(root, "kaggle_dataset/convertido/cardboard/long_name.jpg")
    digest = file_md5(b"duplicate-bytes")
    rows = [
        make_row("kaggle_dataset", 1, digest, "cardboard", "no", keep),
        make_row("kaggle_dataset", 1, digest, "cardboard", "yes", drop),
    ]

    code, manifest = run(tmp_path, rows, "--apply", "--keep-rule", "shortest")

    assert code == 0
    assert keep.exists()
    assert not drop.exists()
    quarantine = (root / "_duplicates_quarantine"
                  / "kaggle_dataset/convertido/cardboard/long_name.jpg")
    assert quarantine.exists()
    entries = read_manifest(manifest)
    assert len(entries) == 1
    assert entries[0]["action"] == "quarantined"
    assert entries[0]["kept_path"] == str(keep)
    assert entries[0]["reason"] == "same_class_redundant"


def test_cross_class_group_removes_all_copies(tmp_path):
    root = tmp_path / "data"
    cardboard = make_file(root, "kaggle_dataset/convertido/cardboard/x.jpg")
    paper = make_file(root, "kaggle_dataset/convertido/paper/x.jpg")
    digest = file_md5(b"duplicate-bytes")
    rows = [
        make_row("kaggle_dataset", 1, digest, "cardboard", "yes", cardboard),
        make_row("kaggle_dataset", 1, digest, "paper", "yes", paper),
    ]

    code, manifest = run(tmp_path, rows, "--apply")

    assert code == 0
    assert not cardboard.exists()
    assert not paper.exists()
    entries = read_manifest(manifest)
    assert len(entries) == 2
    assert {e["reason"] for e in entries} == {"cross_class_remove_all"}
    assert {e["kept_path"] for e in entries} == {""}


def test_keep_flag_rule_uses_the_flagged_row(tmp_path):
    root = tmp_path / "data"
    shortest = make_file(root, "roboflow_convertido/merged/glass/a.jpg")
    flagged = make_file(root, "roboflow_convertido/merged/glass/long_keep.jpg")
    digest = file_md5(b"duplicate-bytes")
    rows = [
        make_row("roboflow_convertido", 1, digest, "glass", "no", shortest),
        make_row("roboflow_convertido", 1, digest, "glass", "yes", flagged),
    ]

    code, manifest = run(tmp_path, rows, "--apply", "--keep-rule", "keep-flag")

    assert code == 0
    assert flagged.exists()
    assert not shortest.exists()
    entries = read_manifest(manifest)
    assert len(entries) == 1
    assert entries[0]["kept_path"] == str(flagged)


def test_path_outside_data_root_is_rejected_before_any_move(tmp_path):
    root = tmp_path / "data"
    inside_a = make_file(root, "kaggle_dataset/convertido/cardboard/a.jpg")
    inside_b = make_file(root, "kaggle_dataset/convertido/cardboard/b.jpg")
    outside_a = make_file(tmp_path, "outside_a.jpg")
    outside_b = make_file(tmp_path, "outside_b.jpg")
    digest = file_md5(b"duplicate-bytes")
    rows = [
        make_row("kaggle_dataset", 1, digest, "cardboard", "no", inside_a),
        make_row("kaggle_dataset", 1, digest, "cardboard", "yes", inside_b),
        make_row("kaggle_dataset", 2, digest, "cardboard", "no", outside_a),
        make_row("kaggle_dataset", 2, digest, "cardboard", "yes", outside_b),
    ]

    code, manifest = run(tmp_path, rows, "--apply")

    assert code != 0
    assert inside_a.exists() and inside_b.exists()
    assert outside_a.exists() and outside_b.exists()
    assert not (root / "_duplicates_quarantine").exists()
    assert not manifest.exists()


def test_path_inside_merged_dataset_is_rejected(tmp_path):
    root = tmp_path / "data"
    merged = make_file(root, "merged_dataset/cardboard/x.jpg")
    source = make_file(root, "kaggle_dataset/convertido/cardboard/x.jpg")
    digest = file_md5(b"duplicate-bytes")
    rows = [
        make_row("kaggle_dataset", 1, digest, "cardboard", "yes", merged),
        make_row("kaggle_dataset", 1, digest, "cardboard", "no", source),
    ]

    code, manifest = run(tmp_path, rows, "--apply")

    assert code != 0
    assert merged.exists() and source.exists()
    assert not (root / "_duplicates_quarantine").exists()
    assert not manifest.exists()


def test_md5_mismatch_is_skipped_and_file_stays(tmp_path):
    root = tmp_path / "data"
    keep = make_file(root, "trashbox/TrashBox_train_set/metal/a.jpg", b"aaa")
    changed = make_file(root, "trashbox/TrashBox_train_set/metal/b.jpg", b"bbb")
    digest = file_md5(b"aaa")
    rows = [
        make_row("trashbox", 1, digest, "metal", "no", keep),
        make_row("trashbox", 1, digest, "metal", "yes", changed),
    ]

    code, manifest = run(tmp_path, rows, "--apply")

    assert code == 0
    assert keep.exists() and changed.exists()
    entries = read_manifest(manifest)
    assert len(entries) == 1
    assert entries[0]["action"] == "skipped_md5_mismatch"
    assert entries[0]["removed_path"] == str(changed)


def test_second_apply_is_idempotent(tmp_path):
    root = tmp_path / "data"
    keep = make_file(root, "trashnet/dataset-resized/paper/a.jpg")
    drop = make_file(root, "trashnet/dataset-resized/paper/long_name.jpg")
    digest = file_md5(b"duplicate-bytes")
    rows = [
        make_row("trashnet", 1, digest, "paper", "no", keep),
        make_row("trashnet", 1, digest, "paper", "yes", drop),
    ]

    first_code, _ = run(tmp_path, rows, "--apply")
    second_code, manifest = run(tmp_path, rows, "--apply")

    assert first_code == 0 and second_code == 0
    assert keep.exists()
    assert not drop.exists()
    quarantined = list((root / "_duplicates_quarantine").rglob("*.jpg"))
    assert len(quarantined) == 1
    entries = read_manifest(manifest)
    assert len(entries) == 1
    assert entries[0]["action"] == "skipped_missing"


def test_quarantine_preserves_relative_structure(tmp_path):
    root = tmp_path / "data"
    keep = make_file(root, "trashnet/dataset-resized/paper/a.jpg")
    drop = make_file(root, "trashnet/dataset-resized/paper/long_name.jpg")
    digest = file_md5(b"duplicate-bytes")
    rows = [
        make_row("trashnet", 1, digest, "paper", "no", keep),
        make_row("trashnet", 1, digest, "paper", "yes", drop),
    ]

    code, _ = run(tmp_path, rows, "--apply")

    assert code == 0
    dest = (root / "_duplicates_quarantine"
            / "trashnet/dataset-resized/paper/long_name.jpg")
    assert dest.exists()
    assert dest.read_bytes() == b"duplicate-bytes"


def test_quarantine_does_not_overwrite_same_md5_destination(tmp_path):
    root = tmp_path / "data"
    keep = make_file(root, "kaggle_dataset/convertido/cardboard/a.jpg")
    drop = make_file(root, "kaggle_dataset/convertido/cardboard/long_name.jpg")
    dest = (root / "_duplicates_quarantine"
            / "kaggle_dataset/convertido/cardboard/long_name.jpg")
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"duplicate-bytes")
    digest = file_md5(b"duplicate-bytes")
    rows = [
        make_row("kaggle_dataset", 1, digest, "cardboard", "no", keep),
        make_row("kaggle_dataset", 1, digest, "cardboard", "yes", drop),
    ]

    code, manifest = run(tmp_path, rows, "--apply")

    assert code == 0
    assert drop.exists()
    assert dest.read_bytes() == b"duplicate-bytes"
    entries = read_manifest(manifest)
    assert len(entries) == 1
    assert entries[0]["action"] == "already_quarantined"


def test_quarantine_uses_dup_suffix_when_destination_differs(tmp_path):
    root = tmp_path / "data"
    keep = make_file(root, "kaggle_dataset/convertido/cardboard/a.jpg")
    drop = make_file(root, "kaggle_dataset/convertido/cardboard/long_name.jpg")
    dest = (root / "_duplicates_quarantine"
            / "kaggle_dataset/convertido/cardboard/long_name.jpg")
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"older-different-file")
    digest = file_md5(b"duplicate-bytes")
    rows = [
        make_row("kaggle_dataset", 1, digest, "cardboard", "no", keep),
        make_row("kaggle_dataset", 1, digest, "cardboard", "yes", drop),
    ]

    code, _ = run(tmp_path, rows, "--apply")

    assert code == 0
    assert not drop.exists()
    assert dest.read_bytes() == b"older-different-file"
    assert dest.with_name("long_name.dup1.jpg").read_bytes() == b"duplicate-bytes"


def test_dry_run_is_default_and_writes_nothing(tmp_path, capsys):
    root = tmp_path / "data"
    keep = make_file(root, "kaggle_dataset/convertido/cardboard/a.jpg")
    drop = make_file(root, "kaggle_dataset/convertido/cardboard/long_name.jpg")
    digest = file_md5(b"duplicate-bytes")
    rows = [
        make_row("kaggle_dataset", 1, digest, "cardboard", "no", keep),
        make_row("kaggle_dataset", 1, digest, "cardboard", "yes", drop),
    ]

    code, manifest = run(tmp_path, rows)

    assert code == 0
    assert keep.exists() and drop.exists()
    assert not (root / "_duplicates_quarantine").exists()
    assert not manifest.exists()
    output = capsys.readouterr().out
    assert "dry-run: 1 planned removals" in output
