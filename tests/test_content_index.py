"""Tests for scripts/content_index.py."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import content_index as ci  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def smooth_image(seed: int, size: int = 256) -> Image.Image:
    """A deterministic, smoothly-textured image (stable under resizing)."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((size, size), Image.BICUBIC)


def write_jpeg(
    data_root: Path,
    relative: Path,
    image: Image.Image,
    *,
    quality: int = 95,
) -> Path:
    path = data_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=quality)
    return path


def write_bytes(data_root: Path, relative: Path, payload: bytes) -> Path:
    path = data_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


SOURCES = {"alpha": Path("alpha/raw"), "beta": Path("beta/raw")}


def build(data_root: Path, out: Path, report: Path) -> ci.ContentIndex:
    index = ci.build_content_index(data_root, sources=SOURCES, workers=1)
    ci.write_content_index(index, out, report)
    return index


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------- #
# Perceptual identity
# --------------------------------------------------------------------------- #


def test_resized_and_recompressed_images_cluster_together(tmp_path: Path) -> None:
    base = smooth_image(1)
    write_jpeg(tmp_path, Path("alpha/raw/paper/base.jpg"), base)
    write_jpeg(
        tmp_path,
        Path("alpha/raw/paper/base_resized.jpg"),
        base.resize((128, 128)),
        quality=75,
    )
    write_jpeg(
        tmp_path,
        Path("beta/raw/paper/base_recompressed.jpg"),
        base,
        quality=50,
    )

    index = ci.build_content_index(tmp_path, sources=SOURCES, workers=1)

    assert index.n_images == 3
    assert index.n_clusters == 1
    assert index.n_redundant_members == 2
    assert len({row.cluster_id for row in index.rows}) == 1
    assert all(row.cluster_status == "ok" for row in index.rows)
    assert all(row.cluster_class == "paper" for row in index.rows)


def test_images_beyond_the_threshold_do_not_cluster(tmp_path: Path) -> None:
    write_jpeg(tmp_path, Path("alpha/raw/paper/one.jpg"), smooth_image(1))
    write_jpeg(tmp_path, Path("alpha/raw/paper/two.jpg"), smooth_image(2))
    write_jpeg(tmp_path, Path("alpha/raw/glass/three.jpg"), smooth_image(3))

    index = ci.build_content_index(
        tmp_path, sources={"alpha": Path("alpha/raw")}, workers=1
    )

    assert index.n_images == 3
    assert index.n_clusters == 3
    assert index.n_redundant_members == 0
    assert len({row.cluster_id for row in index.rows}) == 3


def test_exact_duplicates_share_a_cluster_and_one_identity(tmp_path: Path) -> None:
    image = smooth_image(1)
    first = write_jpeg(tmp_path, Path("alpha/raw/paper/a.jpg"), image)
    second = write_bytes(
        tmp_path, Path("beta/raw/paper/b.jpg"), first.read_bytes()
    )

    index = ci.build_content_index(tmp_path, sources=SOURCES, workers=1)

    expected_md5 = hashlib.md5(second.read_bytes()).hexdigest()
    assert index.n_images == 2
    assert index.n_clusters == 1
    row_md5 = {row.md5 for row in index.rows}
    assert row_md5 == {expected_md5}
    assert {row.cluster_id for row in index.rows} == {expected_md5}


def test_multi_class_cluster_is_flagged_ambiguous(tmp_path: Path) -> None:
    payload = write_jpeg(
        tmp_path, Path("alpha/raw/paper/a.jpg"), smooth_image(1)
    ).read_bytes()
    write_bytes(tmp_path, Path("beta/raw/cardboard/b.jpg"), payload)
    write_jpeg(tmp_path, Path("alpha/raw/paper/other.jpg"), smooth_image(2))

    index = ci.build_content_index(tmp_path, sources=SOURCES, workers=1)
    ambiguous = [cluster for cluster in index.clusters if cluster.ambiguous]

    assert len(ambiguous) == 1
    assert ambiguous[0].size == 2
    assert ambiguous[0].classes == ("cardboard", "paper")
    flags = {
        (row.path, row.cluster_status, row.cluster_class) for row in index.rows
    }
    assert (
        "alpha/raw/paper/a.jpg",
        "ambiguous_class",
        "",
    ) in flags
    assert (
        "beta/raw/cardboard/b.jpg",
        "ambiguous_class",
        "",
    ) in flags
    assert (
        "alpha/raw/paper/other.jpg",
        "ok",
        "paper",
    ) in flags


# --------------------------------------------------------------------------- #
# Unreadable images
# --------------------------------------------------------------------------- #


def test_unreadable_images_are_reported_and_excluded(tmp_path: Path) -> None:
    write_jpeg(tmp_path, Path("alpha/raw/paper/good.jpg"), smooth_image(1))
    write_bytes(tmp_path, Path("alpha/raw/paper/empty.jpg"), b"")
    write_bytes(tmp_path, Path("beta/raw/paper/corrupt.jpg"), b"not an image")

    index = ci.build_content_index(tmp_path, sources=SOURCES, workers=1)

    assert index.n_images == 1
    assert [row.path for row in index.rows] == ["alpha/raw/paper/good.jpg"]
    unreadable = {item.path: item for item in index.unreadable}
    assert set(unreadable) == {
        "alpha/raw/paper/empty.jpg",
        "beta/raw/paper/corrupt.jpg",
    }
    assert "vazio" in unreadable["alpha/raw/paper/empty.jpg"].reason
    assert unreadable["beta/raw/paper/corrupt.jpg"].reason
    assert index.n_clusters == 1


def test_cli_writes_artifacts_and_reports_unreadable(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    for source, relative in ci.SOURCE_ROOTS.items():
        write_jpeg(
            data_root,
            relative / "paper" / f"{source}.jpg",
            smooth_image(len(source)),
        )
    write_bytes(data_root, Path("kaggle_dataset/convertido/paper/broken.jpg"), b"")
    out = tmp_path / "index.csv"
    report = tmp_path / "unreadable.csv"

    exit_code = ci.main(
        [
            "--data-root",
            str(data_root),
            "--out",
            str(out),
            "--unreadable-report",
            str(report),
            "--workers",
            "1",
        ]
    )

    assert exit_code == 0
    rows = read_rows(out)
    assert len(rows) == len(ci.SOURCE_ROOTS)
    assert list(rows[0]) == list(ci.INDEX_FIELDS)
    unreadable = read_rows(report)
    assert list(unreadable[0]) == list(ci.UNREADABLE_FIELDS)
    assert len(unreadable) == 1
    assert unreadable[0]["path"] == "kaggle_dataset/convertido/paper/broken.jpg"


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def test_rerunning_the_index_is_byte_identical(tmp_path: Path) -> None:
    write_jpeg(tmp_path, Path("alpha/raw/paper/b.jpg"), smooth_image(1))
    write_jpeg(tmp_path, Path("alpha/raw/paper/a.jpg"), smooth_image(2))
    write_jpeg(tmp_path, Path("beta/raw/glass/c.jpg"), smooth_image(3))
    write_bytes(tmp_path, Path("beta/raw/glass/broken.jpg"), b"broken")

    out = tmp_path / "index.csv"
    report = tmp_path / "unreadable.csv"
    build(tmp_path, out, report)
    first_index = out.read_bytes()
    first_report = report.read_bytes()

    build(tmp_path, out, report)

    assert out.read_bytes() == first_index
    assert report.read_bytes() == first_report


def test_rows_are_sorted_by_source_class_and_path(tmp_path: Path) -> None:
    write_jpeg(tmp_path, Path("beta/raw/paper/z.jpg"), smooth_image(1))
    write_jpeg(tmp_path, Path("alpha/raw/paper/b.jpg"), smooth_image(2))
    write_jpeg(tmp_path, Path("alpha/raw/glass/a.jpg"), smooth_image(3))

    index = ci.build_content_index(tmp_path, sources=SOURCES, workers=1)

    keys = [(row.source, row.cls, row.path) for row in index.rows]
    assert keys == sorted(keys)


def test_clusters_require_both_hashes_within_threshold() -> None:
    def record(md5: str, dhash_hex: str, phash_hex: str) -> ci.ImageRecord:
        return ci.ImageRecord(
            source="alpha",
            cls="paper",
            path=f"alpha/raw/paper/{md5}.jpg",
            md5=md5,
            dhash=dhash_hex,
            phash=phash_hex,
        )

    zero = "00" * 8
    one_bit = "00" * 7 + "01"
    all_bits = "ff" * 8

    dhash_only = ci.build_index(
        [record("a", zero, zero), record("b", one_bit, all_bits)],
        [],
    )
    phash_only = ci.build_index(
        [record("c", zero, zero), record("d", all_bits, one_bit)],
        [],
    )
    both = ci.build_index(
        [record("e", zero, zero), record("f", one_bit, one_bit)],
        [],
    )

    assert dhash_only.n_clusters == 2
    assert phash_only.n_clusters == 2
    assert both.n_clusters == 1


def test_worker_count_does_not_change_output(tmp_path: Path) -> None:
    write_jpeg(tmp_path, Path("alpha/raw/paper/a.jpg"), smooth_image(1))
    write_jpeg(tmp_path, Path("beta/raw/paper/b.jpg"), smooth_image(2))
    write_jpeg(tmp_path, Path("beta/raw/glass/c.jpg"), smooth_image(1))

    outputs = []
    for workers in (1, 2):
        out = tmp_path / f"index_{workers}.csv"
        report = tmp_path / f"unreadable_{workers}.csv"
        index = ci.build_content_index(tmp_path, sources=SOURCES, workers=workers)
        ci.write_content_index(index, out, report)
        outputs.append((out.read_bytes(), report.read_bytes()))

    assert outputs[0] == outputs[1]


def test_cluster_identity_is_content_derived(tmp_path: Path) -> None:
    write_jpeg(tmp_path, Path("alpha/raw/paper/a.jpg"), smooth_image(1))
    write_jpeg(tmp_path, Path("beta/raw/glass/b.jpg"), smooth_image(1))

    index = ci.build_content_index(tmp_path, sources=SOURCES, workers=1)

    assert index.n_clusters == 1
    assert {row.cluster_id for row in index.rows} == {
        min(row.md5 for row in index.rows)
    }


def test_source_roots_must_exist(tmp_path: Path) -> None:
    write_jpeg(tmp_path, Path("alpha/raw/paper/a.jpg"), smooth_image(1))

    with pytest.raises(ci.ContentIndexError):
        ci.build_content_index(tmp_path, sources=SOURCES, workers=1)
