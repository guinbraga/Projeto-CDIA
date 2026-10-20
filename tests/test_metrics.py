"""Tests for scripts/metrics.py.

The helpers under test are pure numpy, so this file never imports torch.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import metrics  # noqa: E402


# --------------------------------------------------------------------------- #
# Confusion summarisation
# --------------------------------------------------------------------------- #
def test_perfect_confusion_scores_one_everywhere() -> None:
    confusion = np.array([[5, 0], [0, 5]])

    summary = metrics.summarize_confusion(confusion, ["a", "b"], top3_correct=10, total=10)

    assert summary["acc"] == pytest.approx(1.0)
    assert summary["macro_f1"] == pytest.approx(1.0)
    assert summary["balanced_acc"] == pytest.approx(1.0)
    assert summary["macro_precision"] == pytest.approx(1.0)
    assert summary["macro_recall"] == pytest.approx(1.0)
    assert summary["worst_class_f1"] == pytest.approx(1.0)
    assert summary["worst_class"] == "a"


def test_balanced_accuracy_is_the_mean_per_class_recall() -> None:
    # Class "a" is always right, class "b" is never recalled.
    confusion = np.array([[5, 0], [5, 0]])

    summary = metrics.summarize_confusion(confusion, ["a", "b"], top3_correct=5, total=10)

    assert summary["acc"] == pytest.approx(0.5)
    assert summary["per_class"]["a"]["recall"] == pytest.approx(1.0)
    assert summary["per_class"]["b"]["recall"] == pytest.approx(0.0)
    assert summary["balanced_acc"] == pytest.approx(0.5)
    assert summary["macro_recall"] == pytest.approx(0.5)
    # "a": tp=5, predicted=10 -> precision 0.5; "b": predicted=0 -> 0.0.
    assert summary["macro_precision"] == pytest.approx(0.25)


def test_worst_class_f1_names_the_collapsed_class() -> None:
    confusion = np.array([[5, 0], [5, 0]])

    summary = metrics.summarize_confusion(confusion, ["a", "b"], top3_correct=5, total=10)

    assert summary["worst_class"] == "b"
    assert summary["worst_class_f1"] == pytest.approx(0.0)


def test_zero_support_class_is_excluded_from_derived_scalars() -> None:
    confusion = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 0]])

    summary = metrics.summarize_confusion(confusion, ["a", "b", "c"], top3_correct=8, total=8)

    assert np.isfinite(summary["balanced_acc"])
    assert np.isfinite(summary["macro_precision"])
    assert np.isfinite(summary["macro_recall"])
    assert np.isfinite(summary["worst_class_f1"])
    assert summary["balanced_acc"] == pytest.approx(1.0)
    assert summary["macro_precision"] == pytest.approx(1.0)
    assert summary["macro_recall"] == pytest.approx(1.0)
    assert summary["worst_class_f1"] == pytest.approx(1.0)
    assert summary["worst_class"] == "a"
    # The absent class is still described in the per-class table, without NaN.
    assert summary["per_class"]["c"]["support"] == 0
    assert summary["per_class"]["c"]["f1"] == 0.0


def test_empty_confusion_does_not_divide_by_zero() -> None:
    confusion = np.zeros((2, 2), dtype=np.int64)

    summary = metrics.summarize_confusion(confusion, ["a", "b"], top3_correct=0, total=0)

    for key in ("acc", "macro_f1", "balanced_acc", "macro_precision", "macro_recall",
                "worst_class_f1", "top3"):
        assert summary[key] == 0.0
    assert summary["worst_class"] is None


def test_confusion_is_serialised_as_integer_lists() -> None:
    summary = metrics.summarize_confusion(np.array([[2, 1], [0, 3]]), ["a", "b"], 5, 6)

    assert summary["confusion"] == [[2, 1], [0, 3]]


# --------------------------------------------------------------------------- #
# History schema
# --------------------------------------------------------------------------- #
def test_history_fields_keep_the_existing_columns_first() -> None:
    fields = metrics.history_fieldnames(["a", "b"])

    assert fields[: len(metrics.HISTORY_BASE_FIELDS)] == metrics.HISTORY_BASE_FIELDS


def test_history_fields_carry_per_class_f1_and_recall_for_every_class() -> None:
    class_names = ["cardboard", "glass", "medical"]

    fields = metrics.history_fieldnames(class_names)

    for name in class_names:
        assert f"val_f1_{name}" in fields
        assert f"val_recall_{name}" in fields
    assert len(fields) == len(set(fields))


def test_history_fields_are_derived_per_run_not_assumed() -> None:
    six = metrics.history_fieldnames(["a", "b", "c", "d", "e", "f"])
    nine = metrics.history_fieldnames(["a", "b", "c", "d", "e", "f", "g", "h", "i"])

    assert len(nine) - len(six) == 6
    assert nine != six
