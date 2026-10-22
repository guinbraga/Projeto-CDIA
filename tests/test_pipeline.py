"""Structural tests for the DVC pipeline definition in dvc.yaml."""

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DVC_FILE = REPO / "dvc.yaml"

SPLIT_DIRS = {
    "data/trashnet_split",
    "data/kaggle_dataset_split",
    "data/trashbox_split",
    "data/roboflow_convertido_split",
}


def load_stages() -> dict:
    with open(DVC_FILE, encoding="utf-8") as handle:
        return yaml.safe_load(handle)["stages"]


def test_stage_names() -> None:
    assert set(load_stages()) == {
        "content-index",
        "global-split-registry",
        "source-splits",
        "merge",
        "verify",
    }


def test_source_splits_stage() -> None:
    stage = load_stages()["source-splits"]
    assert set(stage["outs"]) == SPLIT_DIRS
    cmd = stage["cmd"]
    for token in (
        "split_dataset.py",
        "--registry",
        "--content-index",
        "--overwrite",
    ):
        assert token in cmd
    assert "data/global_split_registry.csv" in stage["deps"]
    assert "data/content_index.csv" in stage["deps"]


def test_merge_stage() -> None:
    stage = load_stages()["merge"]
    assert "data/global_split_registry.csv" in stage["deps"]
    assert SPLIT_DIRS <= set(stage["deps"])
    assert "merge_datasets.py" in stage["cmd"]
    assert "--registry" in stage["cmd"]


def test_verify_stage() -> None:
    stage = load_stages()["verify"]
    deps = set(stage["deps"])
    assert "scripts/merge_datasets.py" in deps
    assert "scripts/verify_dataset_isolation.py" in deps
    assert "data/global_split_registry.csv" in deps
    assert SPLIT_DIRS <= deps
    assert "data/merged_dataset" in deps
    assert "merge_datasets.py" in stage["cmd"]
    assert "--verify-existing" in stage["cmd"]
    assert "verify_dataset_isolation.py" in stage["cmd"]
    assert "--registry" in stage["cmd"]
    assert "--merged" in stage["cmd"]
    assert "merge.seed" in stage["params"]
    assert not (stage.get("outs") or [])


def test_no_stale_stage_references() -> None:
    text = yaml.safe_dump(load_stages(), sort_keys=False)
    for stale in (
        "merged_dataset_balanced",
        "merged_datasets_unbalanced",
        "_quarantine",
        "remove_exact_duplicates",
    ):
        assert stale not in text


def test_params_declared() -> None:
    stages = load_stages()
    assert "merge.seed" in stages["merge"]["params"]
    assert {
        "registry.seed",
        "registry.val_ratio",
        "registry.test_ratio",
    } <= set(stages["global-split-registry"]["params"])
