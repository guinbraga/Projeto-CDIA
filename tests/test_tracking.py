"""Tests for the MLflow/DVC data-version helpers in ``scripts/tracking.py``.

MLflow itself is not needed: the module imports it lazily, so these tests run
on the system interpreter without torch or mlflow installed. The one network
call (``/health`` probe) is exercised against a closed local port only.
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import tracking  # noqa: E402

LOCK_TEXT = """\
stages:
  merge:
    cmd: python3 scripts/merge_datasets.py --overwrite
    deps:
    - path: data/global_split_registry.csv
      hash: md5
      md5: 8bff93a05f9fd9de6e3ab1d3dcf6cc82
      size: 2019703
    outs:
    - path: data/merged_dataset
      hash: md5
      md5: 0d90cefd0d230110090207b13df8548f.dir
      size: 12
      nfiles: 2
"""


def make_repo(tmp_path: Path, lock_text: str = LOCK_TEXT) -> Path:
    (tmp_path / "dvc.lock").write_text(lock_text, encoding="utf-8")
    dataset = tmp_path / "data" / "merged_dataset"
    (dataset / "train" / "paper").mkdir(parents=True)
    (dataset / "train" / "paper" / "a.jpg").write_bytes(b"abc123")
    (dataset / "train" / "paper" / "b.jpg").write_bytes(b"def456")
    return tmp_path


# --------------------------------------------------------------------------- #
# Git and dvc.lock parsing
# --------------------------------------------------------------------------- #
def test_git_metadata_reads_the_checkout() -> None:
    git = tracking.git_metadata(REPO)

    assert isinstance(git["commit"], str)
    assert len(git["commit"]) == 40
    assert int(git["commit"], 16) >= 0
    assert isinstance(git["branch"], str)
    assert isinstance(git["dirty"], bool)


def test_load_dvc_lock_finds_pipeline_outputs() -> None:
    lock = tracking.load_dvc_lock(REPO)

    assert lock is not None
    assert lock["path"] == "dvc.lock"
    assert isinstance(lock["md5"], str) and len(lock["md5"]) == 32

    outs = {stage: data["outs"] for stage, data in lock["stages"].items()}
    assert "merge" in outs and "source-splits" in outs
    merged = next(out for out in outs["merge"] if out["path"] == "data/merged_dataset")
    assert merged["md5"].endswith(".dir")
    assert merged["nfiles"] > 0
    assert merged["size"] > 0


def test_lock_scanner_matches_the_yaml_parser() -> None:
    text = (REPO / "dvc.lock").read_text(encoding="utf-8")
    document = yaml.safe_load(text)
    from_yaml = {
        name: [
            {key: out.get(key) for key in ("path", "md5", "size", "nfiles")}
            for out in stage.get("outs") or []
        ]
        for name, stage in (document.get("stages") or {}).items()
    }

    from_scanner = {
        name: [
            {key: out.get(key) for key in ("path", "md5", "size", "nfiles")}
            for out in data["outs"]
        ]
        for name, data in tracking._parse_lock_outs(text).items()
    }

    assert from_scanner == from_yaml


def test_find_lock_out_matches_path_then_basename(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    lock = tracking.load_dvc_lock(repo)

    by_path = tracking.find_lock_out(
        lock, repo, repo / "data" / "merged_dataset"
    )
    assert by_path is not None
    assert by_path["match"] == "path"
    assert by_path["out"]["md5"].endswith(".dir")
    assert by_path["repo_relative_path"] == "data/merged_dataset"

    # Colab: the tree is extracted outside the clone, so only the name matches.
    by_name = tracking.find_lock_out(
        lock, repo, Path("/content/data/merged_dataset")
    )
    assert by_name is not None
    assert by_name["match"] == "basename"
    assert by_name["repo_relative_path"] is None

    assert tracking.find_lock_out(lock, repo, Path("/content/data/unknown")) is None


# --------------------------------------------------------------------------- #
# Dataset version
# --------------------------------------------------------------------------- #
def test_dataset_version_marks_a_tree_that_matches_the_lock(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)

    version = tracking.dataset_version(repo, repo / "data" / "merged_dataset")

    assert version["live"]["files"] == 2
    assert version["live"]["bytes"] == 12
    assert version["live"]["splits"]["train"] == {"files": 2, "bytes": 12}
    assert version["dvc"] is not None
    assert version["dvc"]["hash"] == "0d90cefd0d230110090207b13df8548f.dir"
    assert version["dvc"]["stage"] == "merge"
    assert version["dvc"]["match"] == "path"
    assert version["dvc"]["verified"] is True
    assert version["repo_relative_path"] == "data/merged_dataset"


def test_dataset_version_flags_a_stale_tree(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    dataset = repo / "data" / "merged_dataset"
    with (dataset / "train" / "paper" / "a.jpg").open("ab") as handle:
        handle.write(b"x")

    version = tracking.dataset_version(repo, dataset)

    assert version["live"]["bytes"] == 13
    assert version["dvc"]["verified"] is False


def test_dataset_version_without_a_lock_entry(tmp_path: Path) -> None:
    (tmp_path / "data" / "external" / "test" / "paper").mkdir(parents=True)
    (tmp_path / "data" / "external" / "test" / "paper" / "a.jpg").write_bytes(b"1")

    version = tracking.dataset_version(tmp_path, tmp_path / "data" / "external")

    assert version["dvc"] is None
    assert version["live"]["files"] == 1
    assert version["git"]["commit"] is None


def test_dataset_version_hashes_a_manifest(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    dataset = repo / "data" / "merged_dataset"
    (dataset / "manifest.csv").write_text("split,class\n", encoding="utf-8")

    version = tracking.dataset_version(repo, dataset)

    assert version["live"]["manifest_md5"] == tracking.md5_file(
        dataset / "manifest.csv"
    )
    assert version["live"]["other_files"] == 1


# --------------------------------------------------------------------------- #
# MLflow value shapes
# --------------------------------------------------------------------------- #
def test_flatten_params_scalarises_nested_payloads() -> None:
    flat = tracking.flatten_params(
        {
            "lr": 3e-4,
            "amp": True,
            "run_name": None,
            "classes": ["paper", "metal"],
            "resolved": {"batch_size": 32, "nested": {"depth": 2}},
        }
    )

    assert flat["lr"] == 3e-4
    assert flat["amp"] is True
    assert "run_name" not in flat
    assert json.loads(flat["classes"]) == ["paper", "metal"]
    assert flat["resolved.batch_size"] == 32
    assert flat["resolved.nested.depth"] == 2


def test_flatten_params_sanitises_keys() -> None:
    flat = tracking.flatten_params({"data root": "/tmp/x", "a/b": 1})

    assert set(flat) == {"data_root", "a_b"}
    assert flat["data_root"] == "/tmp/x"


def test_sanitize_metric_name_removes_unsupported_characters() -> None:
    assert tracking.sanitize_metric_name("val_f1_tetra pak") == "val_f1_tetra_pak"
    assert tracking.sanitize_metric_name("val/f1") == "val_f1"
    assert tracking.sanitize_metric_name("  ") == "metric"


def test_is_number_rejects_bools_and_strings() -> None:
    assert tracking.is_number(1)
    assert tracking.is_number(1.5)
    assert not tracking.is_number(True)
    assert not tracking.is_number("1")


# --------------------------------------------------------------------------- #
# Tracker fail-soft behaviour
# --------------------------------------------------------------------------- #
def test_disabled_tracker_is_a_noop(tmp_path: Path) -> None:
    tracker = tracking.MlflowTracker(enabled=False)

    assert tracker.start() is False
    assert tracker.active is False
    tracker.log_params({"lr": 1.0})
    tracker.log_metrics({"val_acc": 0.5}, step=1)
    tracker.log_artifacts([tmp_path / "missing.json"])
    tracker.log_tags({"a": "b"})
    tracker.end("FINISHED")

    assert tracker.run_id is None


def test_tracker_survives_a_missing_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenImport:
        @staticmethod
        def import_module(name: str):
            raise ImportError("mlflow is not installed")

    monkeypatch.setattr(tracking, "importlib", BrokenImport)
    tracker = tracking.MlflowTracker(tracking_uri="http://127.0.0.1:9")

    assert tracker.start() is False
    assert tracker.active is False
    tracker.log_metrics({"val_acc": 1.0})
    tracker.end("FINISHED")


def test_server_reachable_rejects_a_closed_port() -> None:
    assert tracking.server_reachable("http://127.0.0.1:9", timeout=0.25) is False
    assert tracking.server_reachable("file:///tmp/runs") is True
    assert tracking.server_reachable("") is False


def test_server_reachable_accepts_a_protected_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import urllib.error

    def unauthorized(*args, **kwargs):
        raise urllib.error.HTTPError(
            "http://mlflow.example/health", 401, "Unauthorized", None, None
        )

    monkeypatch.setattr(tracking.urllib.request, "urlopen", unauthorized)

    assert tracking.server_reachable("https://mlflow.example") is True
