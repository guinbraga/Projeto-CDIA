"""Tests for scripts/dataset_io.py."""

import csv
import hashlib
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import dataset_io  # noqa: E402


def make_jpeg(path: Path, color: tuple[int, int, int] = (255, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path, format="JPEG")


def test_split_sizes_known_values() -> None:
    assert dataset_io.split_sizes(20, 0.15, 0.15) == (14, 3, 3)
    assert dataset_io.split_sizes(10, 0.15, 0.15) == (6, 2, 2)
    assert dataset_io.split_sizes(1, 0.15, 0.15) == (1, 0, 0)
    assert dataset_io.split_sizes(0, 0.15, 0.15) == (0, 0, 0)


def test_split_sizes_zero_ratios() -> None:
    for n in (0, 1, 7, 100):
        assert dataset_io.split_sizes(n, 0.0, 0.0) == (n, 0, 0)


def test_split_sizes_rounding_overshoot_is_corrected() -> None:
    n_train, n_val, n_test = dataset_io.split_sizes(1, 0.5, 0.5)
    assert n_train + n_val + n_test == 1
    assert min(n_train, n_val, n_test) >= 0


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 10, 20, 33, 99, 100])
def test_split_sizes_non_negative_and_sum(n: int) -> None:
    for val_ratio, test_ratio in ((0.15, 0.15), (0.3, 0.2), (0.5, 0.5)):
        parts = dataset_io.split_sizes(n, val_ratio, test_ratio)
        assert min(parts) >= 0
        assert sum(parts) == n


def test_survey_images_reports_and_keeps(tmp_path: Path) -> None:
    valid = tmp_path / "src" / "A" / "valid.jpg"
    make_jpeg(valid)
    empty = tmp_path / "src" / "A" / "empty.jpg"
    empty.write_bytes(b"")
    corrupt = tmp_path / "src" / "A" / "corrupt.jpg"
    corrupt.write_bytes(b"not an image")
    unreadable = tmp_path / "src" / "A" / "unreadable.jpg"
    unreadable.symlink_to(tmp_path / "missing")

    found = {"src": {"A": [valid, empty, corrupt, unreadable]}}
    infos, invalid = dataset_io.survey_images(found)

    assert set(infos) == {valid}
    info = infos[valid]
    assert info.path == valid
    assert info.source == "src"
    assert info.cls == "A"
    assert info.mode == "RGB"
    assert (info.width, info.height) == (8, 8)
    assert len(info.md5) == 32
    assert info.md5 == hashlib.md5(valid.read_bytes()).hexdigest()

    reasons = dict(invalid)
    assert reasons[empty] == "arquivo vazio (0 bytes)"
    assert reasons[corrupt].startswith("corrompido")
    assert reasons[unreadable].startswith("ilegível")


def test_unique_name() -> None:
    assert dataset_io.unique_name("a.jpg", set()) == "a.jpg"

    used = {"a.jpg"}
    assert dataset_io.unique_name("a.jpg", used) == "a_1.jpg"

    used = {"a.jpg", "a_1.jpg"}
    assert dataset_io.unique_name("a.jpg", used) == "a_2.jpg"

    used = {"archive.tar.gz"}
    assert dataset_io.unique_name("archive.tar.gz", used) == "archive.tar_1.gz"
    assert dataset_io.unique_name("x.PNG", {"x.PNG"}) == "x_1.PNG"


def test_manifest_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    rows = [
        {
            "split": "train",
            "class": "A",
            "source": "src",
            "md5": "0" * 32,
            "mode": "RGB",
            "src_path": "/tmp/a.jpg",
            "dst_path": "/tmp/out/train/A/a.jpg",
        },
        {
            "split": "val",
            "class": "B",
            "source": "src",
            "md5": "1" * 32,
            "mode": "L",
            "src_path": "/tmp/b.jpg",
            "dst_path": "/tmp/out/val/B/b.jpg",
        },
    ]

    dataset_io.write_manifest(rows, output)

    assert dataset_io.read_manifest(output) == rows
    assert dataset_io.read_manifest(output / "manifest.csv") == rows


def test_manifest_round_trip_with_cluster_id(tmp_path: Path) -> None:
    source = tmp_path / "src" / "A" / "a.jpg"
    make_jpeg(source)

    output = tmp_path / "out"
    records = {
        "train": [("A", "src", source)],
        "val": [],
        "test": [],
    }
    manifest, _ = dataset_io.materialize(
        records,
        output,
        {},
        use_link=False,
        convert_rgb=False,
        cluster_ids={source: "cluster-1"},
    )
    dataset_io.write_manifest(
        manifest, output, dataset_io.MANIFEST_FIELDS_WITH_CLUSTER
    )

    with open(output / "manifest.csv", newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == [
        "split",
        "class",
        "source",
        "cluster_id",
        "md5",
        "mode",
        "src_path",
        "dst_path",
    ]

    rows = dataset_io.read_manifest(output)
    assert rows[0]["cluster_id"] == "cluster-1"

    old_output = tmp_path / "old_out"
    old_manifest, _ = dataset_io.materialize(
        records, old_output, {}, use_link=False, convert_rgb=False
    )
    dataset_io.write_manifest(old_manifest, old_output)
    with open(old_output / "manifest.csv", newline="", encoding="utf-8") as handle:
        old_header = next(csv.reader(handle))
    assert old_header == list(dataset_io.MANIFEST_FIELDS)
    assert "cluster_id" not in dataset_io.read_manifest(old_output)[0]


def test_materialize_resolves_collisions(tmp_path: Path) -> None:
    first = tmp_path / "one" / "x.jpg"
    second = tmp_path / "two" / "x.jpg"
    make_jpeg(first)
    make_jpeg(second, (0, 255, 0))

    output = tmp_path / "out"
    records = {
        "train": [("A", "src", first), ("A", "src", second)],
        "val": [],
        "test": [],
    }

    manifest, converted = dataset_io.materialize(
        records, output, {}, use_link=False, convert_rgb=False
    )

    assert converted == 0
    assert [Path(row["dst_path"]).name for row in manifest] == ["x.jpg", "x_1.jpg"]
    assert (output / "train" / "A" / "x.jpg").is_file()
    assert (output / "train" / "A" / "x_1.jpg").is_file()
    assert all(Path(row["dst_path"]).is_file() for row in manifest)
    assert [row["src_path"] for row in manifest] == [str(first), str(second)]
