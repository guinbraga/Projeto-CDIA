"""Tests for scripts/imbalance.py.

The helpers under test are pure numpy, so this file never imports torch.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import imbalance  # noqa: E402


def labels_from_counts(counts: list[int]) -> list[int]:
    """Expand per-class counts into a contiguous label list."""
    out: list[int] = []
    for index, count in enumerate(counts):
        out.extend([index] * count)
    return out


# --------------------------------------------------------------------------- #
# Class counts
# --------------------------------------------------------------------------- #
def test_class_counts_counts_every_label() -> None:
    assert imbalance.class_counts([0, 0, 0, 1, 2, 2], 3) == [3, 1, 2]


def test_class_counts_reports_empty_classes_as_zero() -> None:
    assert imbalance.class_counts([0, 1], 3) == [1, 1, 0]


def test_class_counts_of_empty_labels_is_all_zero() -> None:
    assert imbalance.class_counts([], 2) == [0, 0]


# --------------------------------------------------------------------------- #
# Per-sample weights
# --------------------------------------------------------------------------- #
def test_alpha_zero_gives_uniform_per_sample_weights() -> None:
    counts = [100, 10, 1]
    weights = imbalance.per_sample_weights(labels_from_counts(counts), counts, 0.0)

    assert weights.shape == (sum(counts),)
    assert np.allclose(weights, 1.0)


def test_alpha_zero_keeps_the_natural_class_share() -> None:
    counts = [100, 10, 1]

    shares = imbalance.effective_class_shares(counts, 0.0)

    assert np.allclose(shares, np.array(counts) / sum(counts))


def test_per_sample_weights_scale_as_inverse_count_power() -> None:
    counts = [64, 4]
    weights = imbalance.per_sample_weights(labels_from_counts(counts), counts, 0.5)

    # 1 / n_c ** alpha per sample.
    assert weights[0] == pytest.approx(1 / 64 ** 0.5)
    assert weights[-1] == pytest.approx(1 / 4 ** 0.5)


def test_per_sample_weights_realise_the_effective_class_share() -> None:
    counts = [50, 5]
    labels = np.array(labels_from_counts(counts))
    weights = imbalance.per_sample_weights(labels, counts, 0.5)

    realised = np.array([weights[labels == c].sum() for c in range(2)]) / weights.sum()

    assert np.allclose(realised, imbalance.effective_class_shares(counts, 0.5))


# --------------------------------------------------------------------------- #
# Effective class shares
# --------------------------------------------------------------------------- #
def test_alpha_one_gives_a_uniform_class_share() -> None:
    counts = [100, 10, 1]

    shares = imbalance.effective_class_shares(counts, 1.0)

    assert np.allclose(shares, np.full(3, 1 / 3))


def test_partial_alpha_sits_between_natural_and_uniform() -> None:
    counts = [3465, 229]
    natural_majority = counts[0] / sum(counts)

    majority_share = imbalance.effective_class_shares(counts, 0.5)[0]

    assert natural_majority > majority_share > 0.5


def test_alpha_one_uses_n_class_raise_to_the_power_one_minus_alpha() -> None:
    counts = [16, 4]

    shares = imbalance.effective_class_shares(counts, 0.5)

    mass = np.array([16 ** 0.5, 4 ** 0.5])
    assert np.allclose(shares, mass / mass.sum())


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_effective_shares_sum_to_one(alpha: float) -> None:
    counts = [1299, 287, 1682, 1297, 1094, 1445, 1108, 1196, 408]

    assert imbalance.effective_class_shares(counts, alpha).sum() == pytest.approx(1.0)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_zero_support_class_produces_no_nan_or_infinity(alpha: float) -> None:
    counts = [10, 0, 3]

    weights = imbalance.per_sample_weights(labels_from_counts(counts), counts, alpha)
    shares = imbalance.effective_class_shares(counts, alpha)

    assert np.all(np.isfinite(weights))
    assert np.all(np.isfinite(shares))
    assert shares[1] == 0.0
    assert shares.sum() == pytest.approx(1.0)


def test_all_zero_counts_produce_no_nan_or_infinity() -> None:
    shares = imbalance.effective_class_shares([0, 0], 0.5)

    assert np.all(np.isfinite(shares))
    assert shares.sum() == 0.0


def test_per_sample_weights_reject_out_of_range_labels() -> None:
    with pytest.raises(ValueError):
        imbalance.per_sample_weights([0, 3], [1, 1, 1], 0.5)


def test_negative_counts_are_rejected() -> None:
    with pytest.raises(ValueError):
        imbalance.effective_class_shares([4, -1], 0.5)


# --------------------------------------------------------------------------- #
# Imbalance ratio
# --------------------------------------------------------------------------- #
def test_imbalance_ratio_is_largest_over_smallest_class() -> None:
    assert imbalance.imbalance_ratio([1299, 287]) == pytest.approx(1299 / 287)


def test_imbalance_ratio_ignores_empty_classes() -> None:
    assert imbalance.imbalance_ratio([10, 0, 5]) == pytest.approx(2.0)


def test_imbalance_ratio_of_a_single_class_is_one() -> None:
    assert imbalance.imbalance_ratio([7, 0]) == 1.0


def test_imbalance_ratio_without_data_is_zero() -> None:
    assert imbalance.imbalance_ratio([0, 0]) == 0.0


# --------------------------------------------------------------------------- #
# Alpha validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_validate_alpha_accepts_the_closed_unit_interval(alpha: float) -> None:
    assert imbalance.validate_alpha(alpha) == alpha


@pytest.mark.parametrize("alpha", [-0.01, 1.01])
def test_validate_alpha_rejects_values_outside_the_unit_interval(alpha: float) -> None:
    with pytest.raises(ValueError):
        imbalance.validate_alpha(alpha)


# --------------------------------------------------------------------------- #
# Audit table
# --------------------------------------------------------------------------- #
def test_share_table_prints_natural_and_effective_shares() -> None:
    table = imbalance.format_class_share_table(["paper", "glass"], [75, 25], 1.0)

    lines = table.splitlines()
    assert lines[0].split() == ["class", "train_n", "natural", "effective"]
    assert lines[1].split() == ["paper", "75", "0.7500", "0.5000"]
    assert lines[2].split() == ["glass", "25", "0.2500", "0.5000"]
