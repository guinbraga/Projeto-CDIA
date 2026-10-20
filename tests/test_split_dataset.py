"""Tests for scripts/split_dataset.py."""

import csv
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import split_dataset  # noqa: E402

SPLITS = ("train", "val", "test")


def make_dataset(root: Path, counts: dict[str, int]) -> None:
    """Create ``root/<class>/<n>.jpg`` with tiny valid JPEGs."""
    for cls, count in counts.items():
        directory = root / cls
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            color = (index * 37 % 256, index * 91 % 256, index * 53 % 256)
            Image.new("RGB", (8, 8), color).save(
                directory / f"{index:03d}.jpg", format="JPEG"
            )


def run(source: Path, output: Path, *extra: str) -> int:
    return split_dataset.main(
        ["--source", str(source), "--output", str(output), *extra]
    )


def read_manifest(output: Path) -> list[dict[str, str]]:
    with open(output / "manifest.csv", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def counts_by_split_and_class(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["split"], row["class"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def test_split_is_stratified_and_complete(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    expected_counts = {"A": 20, "B": 10, "C": 1}
    make_dataset(source, expected_counts)

    code = run(source, output, "--seed", "42")

    assert code == 0
    rows = read_manifest(output)
    by_split = counts_by_split_and_class(rows)

    for cls, total in expected_counts.items():
        n_train, n_val, n_test = split_dataset.split_sizes(total, 0.15, 0.15)
        assert by_split.get(("train", cls), 0) == n_train
        assert by_split.get(("val", cls), 0) == n_val
        assert by_split.get(("test", cls), 0) == n_test

    # Same sizing rule as merge_datasets.py (banker's rounding on 10 * 0.15).
    assert split_dataset.split_sizes(20, 0.15, 0.15) == (14, 3, 3)
    assert split_dataset.split_sizes(10, 0.15, 0.15) == (6, 2, 2)
    assert split_dataset.split_sizes(1, 0.15, 0.15) == (1, 0, 0)

    sources = [Path(row["src_path"]).resolve() for row in rows]
    originals = {path.resolve() for path in source.rglob("*.jpg")}
    assert len(rows) == len(originals)
    assert len(sources) == len(set(sources))
    assert set(sources) == originals


def test_splits_do_not_overlap(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    make_dataset(source, {"A": 18, "B": 9})

    assert run(source, output) == 0
    rows = read_manifest(output)

    for cls in ("A", "B"):
        per_split = {
            split: {
                Path(row["src_path"]).resolve()
                for row in rows
                if row["split"] == split and row["class"] == cls
            }
            for split in SPLITS
        }
        all_paths = set().union(*per_split.values())
        assert sum(len(paths) for paths in per_split.values()) == len(all_paths)
        assert all_paths == {
            path.resolve() for path in (source / cls).glob("*.jpg")
        }


def test_deterministic_for_same_seed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    make_dataset(source, {"A": 40})

    first = tmp_path / "first"
    second = tmp_path / "second"
    different = tmp_path / "different"

    assert run(source, first, "--seed", "42") == 0
    assert run(source, second, "--seed", "42") == 0
    assert run(source, different, "--seed", "7") == 0

    def mapping(output: Path) -> dict[str, str]:
        return {
            row["src_path"]: row["split"] for row in read_manifest(output)
        }

    assert mapping(first) == mapping(second)
    assert mapping(first) != mapping(different)


def test_corrupt_image_is_dropped_by_default(tmp_path: Path) -> None:
    source = tmp_path / "source"
    make_dataset(source, {"A": 5})
    corrupt = source / "A" / "corrupt.jpg"
    corrupt.write_bytes(b"this is not a jpeg")

    validated = tmp_path / "validated"
    assert run(source, validated) == 0
    validated_rows = read_manifest(validated)
    validated_names = {Path(row["src_path"]).name for row in validated_rows}
    assert corrupt.name not in validated_names
    assert not list(validated.rglob(corrupt.name))

    unvalidated = tmp_path / "unvalidated"
    assert run(source, unvalidated, "--no-validate") == 0
    unvalidated_rows = read_manifest(unvalidated)
    unvalidated_names = {Path(row["src_path"]).name for row in unvalidated_rows}
    assert corrupt.name in unvalidated_names
    assert list(unvalidated.rglob(corrupt.name))


def test_dry_run_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    make_dataset(source, {"A": 12})

    assert run(source, output, "--dry-run") == 0

    assert not output.exists()
    printed = capsys.readouterr().out
    assert "A" in printed
    assert "dry-run" in printed


def test_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    make_dataset(source, {"A": 6})
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        run(source, output)

    assert excinfo.value.code not in (0, None)
    assert sentinel.exists()
    assert not (output / "manifest.csv").exists()

    assert run(source, output, "--overwrite") == 0
    assert not sentinel.exists()
    assert (output / "manifest.csv").exists()
