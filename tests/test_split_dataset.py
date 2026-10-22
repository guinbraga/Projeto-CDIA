"""Tests for scripts/split_dataset.py — registry projection, not shuffling."""

import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import split_dataset  # noqa: E402

TRASHNET = Path("trashnet/dataset-resized")
TRASHBOX = Path("trashbox/TrashBox_train_set")

INDEX_FIELDS = [
    "cluster_id",
    "source",
    "class",
    "path",
    "md5",
    "dhash",
    "phash",
    "cluster_class",
    "cluster_status",
]

REGISTRY_FIELDS = ["cluster_id", "class", "split", "canonical_source", "n_images"]


def make_image(
    data_root: Path,
    preset: Path,
    cls: str,
    name: str,
    color: tuple[int, int, int],
) -> Path:
    path = data_root / preset / cls / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path, format="JPEG")
    return path


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def index_row(
    data_root: Path,
    source: str,
    cls: str,
    path: Path,
    cluster_id: str,
    cluster_class: str | None = None,
    status: str = "ok",
) -> dict[str, str]:
    return {
        "cluster_id": cluster_id,
        "source": source,
        "class": cls,
        "path": path.relative_to(data_root).as_posix(),
        "md5": md5_of(path),
        "dhash": "",
        "phash": "",
        "cluster_class": cluster_class or cls,
        "cluster_status": status,
    }


def write_index(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_registry(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "cluster_id": row["cluster_id"],
                    "class": row["class"],
                    "split": row["split"],
                    "canonical_source": row.get("canonical_source", "trashnet"),
                    "n_images": row.get("n_images", "1"),
                }
            )
    return path


def read_manifest(output: Path) -> list[dict[str, str]]:
    with open(output / "manifest.csv", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def manifest_header(output: Path) -> list[str]:
    with open(output / "manifest.csv", newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def run(
    data_root: Path,
    registry: Path,
    index: Path,
    *datasets: str,
    extra: tuple[str, ...] = (),
) -> int:
    argv = [
        "--data-root",
        str(data_root),
        "--registry",
        str(registry),
        "--content-index",
        str(index),
    ]
    for dataset in datasets:
        argv.extend(["--dataset", dataset])
    argv.extend(extra)
    return split_dataset.main(argv)


def two_source_setup(tmp_path: Path):
    """One cluster spans two sources; another is single-source."""
    data_root = tmp_path / "data"
    first = make_image(data_root, TRASHNET, "cardboard", "n1.jpg", (255, 0, 0))
    second = make_image(data_root, TRASHNET, "cardboard", "n2.jpg", (0, 255, 0))
    third = make_image(data_root, TRASHBOX, "cardboard", "b1.jpg", (0, 0, 255))

    index = write_index(
        tmp_path / "content_index.csv",
        [
            index_row(data_root, "trashnet", "cardboard", first, "c-shared"),
            index_row(data_root, "trashnet", "cardboard", second, "c-solo"),
            index_row(data_root, "trashbox", "cardboard", third, "c-shared"),
        ],
    )
    registry = write_registry(
        tmp_path / "global_split_registry.csv",
        [
            {"cluster_id": "c-shared", "class": "cardboard", "split": "test"},
            {"cluster_id": "c-solo", "class": "cardboard", "split": "train"},
        ],
    )
    return data_root, index, registry, {"c-shared": "test", "c-solo": "train"}


def test_rows_follow_registry_split_and_class(tmp_path: Path) -> None:
    data_root, index, registry, canonical = two_source_setup(tmp_path)

    assert run(data_root, registry, index, "trashnet", "trashbox") == 0

    trashnet = read_manifest(data_root / "trashnet_split")
    trashbox = read_manifest(data_root / "trashbox_split")
    assert len(trashnet) == 2
    assert len(trashbox) == 1
    for row in [*trashnet, *trashbox]:
        assert row["split"] == canonical[row["cluster_id"]]
        assert row["class"] == "cardboard"


def test_cluster_stays_in_one_split_across_sources(tmp_path: Path) -> None:
    data_root, index, registry, _ = two_source_setup(tmp_path)

    assert run(data_root, registry, index, "trashnet", "trashbox") == 0

    rows = [
        *read_manifest(data_root / "trashnet_split"),
        *read_manifest(data_root / "trashbox_split"),
    ]
    shared = [row for row in rows if row["cluster_id"] == "c-shared"]
    assert len(shared) == 2
    assert {row["source"] for row in shared} == {"trashnet", "trashbox"}
    assert {row["split"] for row in shared} == {"test"}


def test_ambiguous_cluster_is_excluded_and_counted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root = tmp_path / "data"
    kept = make_image(data_root, TRASHNET, "glass", "ok.jpg", (255, 0, 0))
    dropped = make_image(data_root, TRASHNET, "glass", "ambiguous.jpg", (0, 255, 0))

    index = write_index(
        tmp_path / "content_index.csv",
        [
            index_row(data_root, "trashnet", "glass", kept, "c-ok"),
            index_row(
                data_root,
                "trashnet",
                "glass",
                dropped,
                "c-ambiguous",
                cluster_class="glass|metal",
                status="ambiguous_class",
            ),
        ],
    )
    registry = write_registry(
        tmp_path / "global_split_registry.csv",
        [{"cluster_id": "c-ok", "class": "glass", "split": "train"}],
    )

    assert run(data_root, registry, index, "trashnet") == 0

    rows = read_manifest(data_root / "trashnet_split")
    assert [row["cluster_id"] for row in rows] == ["c-ok"]
    printed = capsys.readouterr().out
    assert "1" in printed
    assert "ambígu" in printed.lower()


def test_manifest_has_fully_populated_cluster_id(tmp_path: Path) -> None:
    data_root, index, registry, canonical = two_source_setup(tmp_path)

    assert run(data_root, registry, index, "trashnet", "trashbox") == 0

    for name in ("trashnet_split", "trashbox_split"):
        rows = read_manifest(data_root / name)
        assert rows
        assert all(row["cluster_id"] for row in rows)
        assert {row["cluster_id"] for row in rows} <= set(canonical)


def test_overwrite_rebuild_is_byte_identical(tmp_path: Path) -> None:
    data_root, index, registry, _ = two_source_setup(tmp_path)

    assert run(data_root, registry, index, "trashnet", "trashbox") == 0
    first = (data_root / "trashnet_split" / "manifest.csv").read_bytes()
    first_box = (data_root / "trashbox_split" / "manifest.csv").read_bytes()

    assert (
        run(data_root, registry, index, "trashnet", "trashbox", extra=("--overwrite",))
        == 0
    )
    assert (data_root / "trashnet_split" / "manifest.csv").read_bytes() == first
    assert (data_root / "trashbox_split" / "manifest.csv").read_bytes() == first_box


def test_dry_run_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root, index, registry, _ = two_source_setup(tmp_path)

    assert (
        run(data_root, registry, index, "trashnet", "trashbox", extra=("--dry-run",))
        == 0
    )

    assert not (data_root / "trashnet_split").exists()
    assert not (data_root / "trashbox_split").exists()
    assert "dry-run" in capsys.readouterr().out


def test_readable_image_missing_from_index_is_stale_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root = tmp_path / "data"
    known = make_image(data_root, TRASHNET, "paper", "known.jpg", (255, 0, 0))
    make_image(data_root, TRASHNET, "paper", "unknown.jpg", (0, 255, 0))

    index = write_index(
        tmp_path / "content_index.csv",
        [index_row(data_root, "trashnet", "paper", known, "c1")],
    )
    registry = write_registry(
        tmp_path / "global_split_registry.csv",
        [{"cluster_id": "c1", "class": "paper", "split": "train"}],
    )

    code = run(data_root, registry, index, "trashnet")

    captured = capsys.readouterr()
    assert code == 2
    assert "[erro]" in captured.err
    assert "índice de conteúdo" in captured.err
    assert "Traceback" not in captured.err
    assert not (data_root / "trashnet_split" / "manifest.csv").exists()


def test_unreadable_image_listed_in_index_is_stale_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root = tmp_path / "data"
    corrupt = data_root / TRASHNET / "paper" / "bad.jpg"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not an image")

    index = write_index(
        tmp_path / "content_index.csv",
        [index_row(data_root, "trashnet", "paper", corrupt, "c1")],
    )
    registry = write_registry(
        tmp_path / "global_split_registry.csv",
        [{"cluster_id": "c1", "class": "paper", "split": "train"}],
    )

    code = run(data_root, registry, index, "trashnet")

    captured = capsys.readouterr()
    assert code == 2
    assert "[erro]" in captured.err
    assert "não pôde ser lida" in captured.err
    assert not (data_root / "trashnet_split" / "manifest.csv").exists()


def test_real_data_projection_matches_registry() -> None:
    data = REPO / "data"
    index = data / "content_index.csv"
    registry_path = data / "global_split_registry.csv"
    names = (
        "trashnet_split",
        "kaggle_dataset_split",
        "trashbox_split",
        "roboflow_convertido_split",
    )
    manifests = {name: data / name / "manifest.csv" for name in names}
    required = [index, registry_path, *manifests.values()]
    for path in required:
        if not path.is_file():
            pytest.skip(f"missing required data file: {path}")

    # Stale pre-registry manifests: skip until ticket 05 is materialised.
    for manifest in manifests.values():
        with open(manifest, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if "cluster_id" not in fieldnames:
                pytest.skip("source manifests not regenerated from the registry")
            if not all((row.get("cluster_id") or "").strip() for row in reader):
                pytest.skip("source manifests have empty cluster ids")

    with open(registry_path, newline="", encoding="utf-8") as handle:
        registry = {
            row["cluster_id"]: (row["class"], row["split"])
            for row in csv.DictReader(handle)
        }

    expected = {
        "trashnet_split": {
            "cardboard": 400,
            "glass": 498,
            "metal": 409,
            "paper": 563,
            "plastic": 480,
            "tetra_pack": 6,
        },
        "kaggle_dataset_split": {
            "cardboard": 310,
            "glass": 1397,
            "metal": 1600,
            "organic": 611,
            "paper": 995,
            "plastic": 4941,
        },
        "trashbox_split": {
            "cardboard": 1855,
            "cigarette_butt": 409,
            "e-waste": 2404,
            "glass": 1853,
            "medical": 1564,
            "metal": 2065,
            "paper": 1563,
            "plastic": 1708,
            "tetra_pack": 582,
        },
        "roboflow_convertido_split": {
            "battery": 877,
            "cardboard": 805,
            "e-waste": 800,
            "medical": 1358,
            "metal": 806,
            "organic": 1518,
            "paper": 804,
            "plastic": 1603,
            "tetra_pack": 736,
        },
    }
    total = 0
    for name, class_counts in expected.items():
        rows = read_manifest(data / name)
        assert len(rows) == sum(class_counts.values())
        assert Counter(row["class"] for row in rows) == class_counts
        total += len(rows)
        for row in rows:
            cluster_id = row["cluster_id"]
            assert cluster_id in registry
            cls, split = registry[cluster_id]
            assert row["split"] == split
            assert row["class"] == cls
    assert total == 35520
