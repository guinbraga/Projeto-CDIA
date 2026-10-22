"""Tests for scripts/global_split_registry.py."""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import dataset_io  # noqa: E402
import global_split_registry as gsr  # noqa: E402

INDEX_FIELDS = ("cluster_id", "source", "class", "cluster_class", "cluster_status")
REAL_INDEX = REPO / "data" / "content_index.csv"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def index_row(
    cluster_id: str,
    source: str,
    cls: str,
    *,
    status: str = "ok",
    cluster_class: str | None = None,
) -> dict[str, str]:
    if cluster_class is None:
        cluster_class = cls if status == "ok" else ""
    return {
        "cluster_id": cluster_id,
        "source": source,
        "class": cls,
        "cluster_class": cluster_class,
        "cluster_status": status,
    }


def write_index(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(INDEX_FIELDS), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def build(tmp_path: Path, rows: list[dict[str, str]], **kwargs: object) -> gsr.Registry:
    path = write_index(tmp_path / "index.csv", rows)
    return gsr.build_registry(gsr.read_content_index(path), **kwargs)


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def split_map(registry: gsr.Registry) -> dict[str, str]:
    return {entry.cluster_id: entry.split for entry in registry.entries}


def stratum_rows(prefix: str, cls: str, source: str, n: int) -> list[dict[str, str]]:
    return [index_row(f"{prefix}{i:03d}", source, cls) for i in range(n)]


# --------------------------------------------------------------------------- #
# Registry invariants
# --------------------------------------------------------------------------- #


def test_every_included_cluster_has_one_class_and_one_split(tmp_path: Path) -> None:
    rows = [
        index_row("c1", "alpha", "paper"),
        index_row("c2", "alpha", "paper"),
        index_row("c3", "beta", "glass"),
        index_row("c3", "beta", "glass"),
    ]

    registry = build(tmp_path, rows)

    assert registry.n_clusters == 3
    assert [entry.cluster_id for entry in registry.entries] == ["c1", "c2", "c3"]
    for entry in registry.entries:
        assert entry.cls
        assert entry.split in {"train", "val", "test"}
        assert entry.canonical_source
        assert entry.n_images >= 1


def test_per_stratum_counts_follow_split_sizes(tmp_path: Path) -> None:
    rows = stratum_rows("paper", "paper", "alpha", 20) + stratum_rows(
        "glass", "glass", "beta", 7
    )

    registry = build(tmp_path, rows)

    by_stratum = {(count.cls, count.source): count for count in registry.counts}
    assert set(by_stratum) == {("paper", "alpha"), ("glass", "beta")}
    for count in registry.counts:
        expected = dataset_io.split_sizes(count.total, 0.15, 0.15)
        assert (count.train, count.val, count.test) == expected
    assert sum(count.total for count in registry.counts) == registry.n_clusters


def test_included_cluster_uses_single_cluster_class(tmp_path: Path) -> None:
    rows = [index_row("c1", "alpha", "paper")]
    registry = build(tmp_path, rows)

    assert registry.entries[0].cls == "paper"


def test_cluster_with_missing_cluster_class_is_an_error(tmp_path: Path) -> None:
    rows = [index_row("c1", "alpha", "paper", cluster_class="")]

    with pytest.raises(gsr.RegistryError):
        build(tmp_path, rows)


# --------------------------------------------------------------------------- #
# Ambiguous clusters
# --------------------------------------------------------------------------- #


def test_ambiguous_clusters_are_excluded_and_reported(tmp_path: Path) -> None:
    rows = [
        index_row("amb", "alpha", "battery", status="ambiguous_class"),
        index_row("amb", "alpha", "battery", status="ambiguous_class"),
        index_row("amb", "beta", "metal", status="ambiguous_class"),
        index_row("ok", "alpha", "paper"),
    ]
    out = tmp_path / "registry.csv"
    report = tmp_path / "excluded.csv"

    registry = build(tmp_path, rows)
    gsr.write_registry(registry, out, report)

    assert registry.n_clusters == 1
    assert registry.n_excluded_clusters == 1
    assert registry.n_excluded_images == 3
    assert registry.n_excluded_target_images == 1
    excluded = registry.excluded[0]
    assert excluded.cluster_id == "amb"
    assert excluded.classes == ("battery", "metal")
    assert excluded.sources == ("alpha", "beta")
    assert excluded.n_images == 3

    report_rows = read_rows(report)
    assert len(report_rows) == 1
    assert report_rows[0]["cluster_id"] == "amb"
    assert report_rows[0]["classes"] == "battery;metal"
    assert report_rows[0]["sources"] == "alpha;beta"
    assert report_rows[0]["n_images"] == "3"
    assert report_rows[0]["n_target_class_images"] == "1"


def test_excluded_clusters_are_sorted_by_cluster_id(tmp_path: Path) -> None:
    rows = [
        index_row("zzz", "alpha", "battery", status="ambiguous_class"),
        index_row("zzz", "alpha", "metal", status="ambiguous_class"),
        index_row("aaa", "alpha", "paper", status="ambiguous_class"),
        index_row("aaa", "alpha", "glass", status="ambiguous_class"),
    ]

    registry = build(tmp_path, rows)

    assert [item.cluster_id for item in registry.excluded] == ["aaa", "zzz"]


def test_cluster_flagged_ambiguous_is_excluded_even_with_one_class(
    tmp_path: Path,
) -> None:
    rows = [
        index_row("amb", "alpha", "paper", status="ambiguous_class"),
        index_row("ok", "alpha", "paper"),
    ]

    registry = build(tmp_path, rows)

    assert [entry.cluster_id for entry in registry.entries] == ["ok"]
    assert [item.cluster_id for item in registry.excluded] == ["amb"]
    assert registry.n_excluded_images == 1


def test_target_class_count_is_case_insensitive(tmp_path: Path) -> None:
    rows = [
        index_row("amb", "alpha", "Metal", status="ambiguous_class"),
        index_row("amb", "alpha", "battery", status="ambiguous_class"),
    ]

    registry = build(tmp_path, rows)

    assert registry.n_excluded_target_images == 1


# --------------------------------------------------------------------------- #
# Canonical source
# --------------------------------------------------------------------------- #


def test_canonical_source_majority_wins() -> None:
    assert gsr.canonical_source(["alpha", "alpha", "beta"]) == "alpha"
    assert gsr.canonical_source(["beta", "beta", "alpha"]) == "beta"


def test_canonical_source_tie_picks_lexicographically_smallest() -> None:
    assert gsr.canonical_source(["beta", "alpha"]) == "alpha"
    assert gsr.canonical_source(["gamma", "alpha", "gamma", "alpha"]) == "alpha"


def test_canonical_source_rejects_empty_input() -> None:
    with pytest.raises(gsr.RegistryError):
        gsr.canonical_source([])


def test_cluster_canonical_source_is_majority(tmp_path: Path) -> None:
    rows = [
        index_row("c1", "beta", "paper"),
        index_row("c1", "beta", "paper"),
        index_row("c1", "alpha", "paper"),
    ]

    registry = build(tmp_path, rows)

    assert registry.entries[0].canonical_source == "beta"
    assert registry.entries[0].n_images == 3


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_same_seed_is_byte_identical_and_different_seed_resplits(
    tmp_path: Path,
) -> None:
    rows = stratum_rows("c", "paper", "alpha", 20)
    index = write_index(tmp_path / "index.csv", rows)
    entries = gsr.read_content_index(index)

    def write(name: str, seed: int) -> gsr.Registry:
        registry = gsr.build_registry(entries, seed, 0.15, 0.15)
        gsr.write_registry(
            registry, tmp_path / f"{name}.csv", tmp_path / f"{name}_excluded.csv"
        )
        return registry

    first = write("first", 42)
    second = write("second", 42)
    assert (tmp_path / "first.csv").read_bytes() == (
        tmp_path / "second.csv"
    ).read_bytes()
    assert (tmp_path / "first_excluded.csv").read_bytes() == (
        tmp_path / "second_excluded.csv"
    ).read_bytes()

    other = write("other", 7)
    assert split_map(first) != split_map(other)
    assert [(c.train, c.val, c.test) for c in first.counts] == [
        (c.train, c.val, c.test) for c in other.counts
    ]


def test_input_row_order_does_not_change_output(tmp_path: Path) -> None:
    rows = stratum_rows("paper", "paper", "alpha", 12) + stratum_rows(
        "glass", "glass", "beta", 9
    )
    shuffled = list(rows)
    random.Random(1234).shuffle(shuffled)

    first_entries = gsr.read_content_index(write_index(tmp_path / "a.csv", rows))
    second_entries = gsr.read_content_index(
        write_index(tmp_path / "b.csv", shuffled)
    )
    gsr.write_registry(
        gsr.build_registry(first_entries),
        tmp_path / "a_registry.csv",
        tmp_path / "a_excluded.csv",
    )
    gsr.write_registry(
        gsr.build_registry(second_entries),
        tmp_path / "b_registry.csv",
        tmp_path / "b_excluded.csv",
    )

    assert (tmp_path / "a_registry.csv").read_bytes() == (
        tmp_path / "b_registry.csv"
    ).read_bytes()
    assert (tmp_path / "a_excluded.csv").read_bytes() == (
        tmp_path / "b_excluded.csv"
    ).read_bytes()


# --------------------------------------------------------------------------- #
# Index parsing
# --------------------------------------------------------------------------- #


def test_read_content_index_ignores_extra_columns(tmp_path: Path) -> None:
    path = tmp_path / "index.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(INDEX_FIELDS) + ["path", "md5", "dhash", "phash"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                **index_row("c1", "alpha", "paper"),
                "path": "alpha/paper/a.jpg",
                "md5": "x",
                "dhash": "y",
                "phash": "z",
            }
        )

    entries = gsr.read_content_index(path)

    assert len(entries) == 1
    assert entries[0].cluster_id == "c1"


def test_read_content_index_requires_columns(tmp_path: Path) -> None:
    path = tmp_path / "index.csv"
    path.write_text("cluster_id,source,class\nc1,alpha,paper\n", encoding="utf-8")

    with pytest.raises(gsr.RegistryError):
        gsr.read_content_index(path)


def test_read_content_index_rejects_empty_cluster_id(tmp_path: Path) -> None:
    path = write_index(
        tmp_path / "index.csv", [index_row("", "alpha", "paper")]
    )

    with pytest.raises(gsr.RegistryError):
        gsr.read_content_index(path)


def test_read_content_index_rejects_empty_class(tmp_path: Path) -> None:
    path = write_index(
        tmp_path / "index.csv", [index_row("c1", "alpha", "")]
    )

    with pytest.raises(gsr.RegistryError):
        gsr.read_content_index(path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_writes_registry_and_report(tmp_path: Path) -> None:
    index = write_index(
        tmp_path / "index.csv",
        [index_row("c1", "alpha", "paper"), index_row("c2", "alpha", "glass")],
    )
    out = tmp_path / "registry.csv"
    report = tmp_path / "excluded.csv"

    exit_code = gsr.main(
        [
            "--content-index",
            str(index),
            "--out",
            str(out),
            "--excluded-report",
            str(report),
        ]
    )

    assert exit_code == 0
    assert out.exists()
    assert report.exists()
    assert list(read_rows(out)[0]) == list(gsr.REGISTRY_FIELDS)
    assert list(read_rows(report)) == []


def test_cli_rejects_invalid_ratios(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index = write_index(
        tmp_path / "index.csv", [index_row("c1", "alpha", "paper")]
    )
    out = tmp_path / "registry.csv"
    report = tmp_path / "excluded.csv"

    exit_code = gsr.main(
        [
            "--content-index",
            str(index),
            "--out",
            str(out),
            "--excluded-report",
            str(report),
            "--val-ratio",
            "0.6",
            "--test-ratio",
            "0.5",
        ]
    )

    assert exit_code == 2
    assert "[erro]" in capsys.readouterr().err
    assert not out.exists()


def test_cli_missing_index_reports_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "registry.csv"

    exit_code = gsr.main(
        [
            "--content-index",
            str(tmp_path / "missing.csv"),
            "--out",
            str(out),
            "--excluded-report",
            str(tmp_path / "excluded.csv"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "[erro]" in captured.err
    assert "Traceback" not in captured.err
    assert not out.exists()


def test_cli_corrupt_index_reports_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index = tmp_path / "index.csv"
    index.write_bytes(b"\xff\xfe0")

    exit_code = gsr.main(
        [
            "--content-index",
            str(index),
            "--out",
            str(tmp_path / "registry.csv"),
            "--excluded-report",
            str(tmp_path / "excluded.csv"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "[erro]" in captured.err
    assert "Traceback" not in captured.err


# --------------------------------------------------------------------------- #
# Real data
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not REAL_INDEX.exists(), reason="data/content_index.csv not available"
)
def test_real_content_index_counts(tmp_path: Path) -> None:
    entries = gsr.read_content_index(REAL_INDEX)
    registry = gsr.build_registry(entries, 42, 0.15, 0.15)

    assert registry.n_clusters == 32_451
    assert registry.n_images == 35_520
    assert registry.n_excluded_clusters == 31
    assert registry.n_excluded_images == 83
    assert registry.n_excluded_target_images == 80

    for count in registry.counts:
        assert (count.train, count.val, count.test) == dataset_io.split_sizes(
            count.total, 0.15, 0.15
        )

    out = tmp_path / "registry.csv"
    report = tmp_path / "excluded.csv"
    gsr.write_registry(registry, out, report)
    first = (out.read_bytes(), report.read_bytes())
    gsr.write_registry(registry, out, report)
    assert (out.read_bytes(), report.read_bytes()) == first
