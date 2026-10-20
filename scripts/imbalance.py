"""Class-balance helpers for the training stream.

Pure numpy (no torch), so the arithmetic that decides how the training stream
is reweighted can be unit-tested without a GPU, a dataset fixture or even a
torch install.

Terminology
-----------
``n_c``
    Number of training samples in class ``c``.
``alpha`` (``--sampler-alpha``)
    Resampling intensity in ``[0, 1]``. Each sample of class ``c`` is drawn
    with weight ``1 / n_c**alpha``, so class ``c`` accounts for a share
    proportional to ``n_c**(1 - alpha)`` of the stream. ``0`` is the natural
    distribution, ``1`` is a fully balanced stream.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Class counts
# --------------------------------------------------------------------------- #
def class_counts(labels: Iterable[int], num_classes: int) -> List[int]:
    """Return the number of samples per class index ``0 .. num_classes - 1``.

    Classes with no samples are reported as ``0`` rather than omitted, so the
    returned list always lines up with the model's output dimension.
    """
    array = np.asarray(list(labels), dtype=np.int64)
    if array.size and (array.min() < 0 or array.max() >= num_classes):
        raise ValueError(
            f"labels outside [0, {num_classes}): min={int(array.min())} max={int(array.max())}"
        )
    counts = np.bincount(array, minlength=num_classes) if array.size else np.zeros(
        num_classes, dtype=np.int64)
    return [int(value) for value in counts[:num_classes]]


# --------------------------------------------------------------------------- #
# Alpha validation
# --------------------------------------------------------------------------- #
def validate_alpha(alpha: float) -> float:
    """Return *alpha* as a float, rejecting values outside ``[0, 1]``.

    Outside that interval the sampler stops being an interpolation between the
    natural distribution and a balanced one: ``alpha > 1`` would over-correct
    past uniformity, and ``alpha < 0`` would amplify the imbalance.
    """
    try:
        value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sampler alpha must be a number, got {alpha!r}") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"sampler alpha must lie within [0, 1], got {value}")
    return value


# --------------------------------------------------------------------------- #
# Shares and weights
# --------------------------------------------------------------------------- #
def _counts_array(counts: Sequence[int]) -> np.ndarray:
    array = np.asarray(counts, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"class counts must be one-dimensional, got shape {array.shape}")
    if np.any(array < 0):
        raise ValueError(f"class counts must be non-negative, got {counts!r}")
    return array


def _class_mass(counts: np.ndarray, alpha: float) -> np.ndarray:
    """Unnormalised share mass per class: ``n_c**(1 - alpha)``, ``0`` when empty."""
    mass = np.zeros_like(counts)
    present = counts > 0
    if alpha == 0.0:
        mass[present] = counts[present]
    else:
        mass[present] = np.power(counts[present], 1.0 - alpha)
    return mass


def natural_class_shares(counts: Sequence[int]) -> np.ndarray:
    """Share of each class in the untouched training set (``alpha = 0``)."""
    return effective_class_shares(counts, 0.0)


def effective_class_shares(counts: Sequence[int], alpha: float) -> np.ndarray:
    """Probability that a drawn training sample belongs to each class.

    Sums to ``1`` over the classes that have support; every entry is finite,
    including for classes with no samples (share ``0``). When there are no
    samples at all the result is all zeros rather than ``NaN``.
    """
    counts_array = _counts_array(counts)
    mass = _class_mass(counts_array, alpha)
    total = mass.sum()
    if total <= 0:
        return np.zeros_like(mass)
    return mass / total


def per_sample_weights(labels: Iterable[int], counts: Sequence[int],
                       alpha: float) -> np.ndarray:
    """Return the sampler weight of every sample: ``1 / n_c**alpha``.

    ``labels[i]`` is the class index of sample ``i``. Classes without support
    get weight ``0`` so that ``0 ** -alpha`` is never evaluated: no sample can
    carry that label all the same.
    """
    counts_array = _counts_array(counts)
    label_array = np.asarray(list(labels), dtype=np.int64)
    if label_array.size == 0:
        return np.zeros(0, dtype=np.float64)
    if label_array.min() < 0 or label_array.max() >= counts_array.size:
        raise ValueError(
            f"labels outside [0, {counts_array.size}): "
            f"min={int(label_array.min())} max={int(label_array.max())}"
        )
    class_weights = np.zeros_like(counts_array)
    present = counts_array > 0
    class_weights[present] = np.power(counts_array[present], -alpha)
    return class_weights[label_array]


# --------------------------------------------------------------------------- #
# Imbalance summary
# --------------------------------------------------------------------------- #
def imbalance_ratio(counts: Sequence[int]) -> float:
    """Largest over smallest non-empty class count.

    ``1`` means balanced (or a single populated class); ``0`` means no data.
    Empty classes are ignored: a class with no samples says nothing about how
    skewed the stream is, it just cannot be learned.
    """
    positive = [int(count) for count in counts if count > 0]
    if not positive:
        return 0.0
    if len(positive) == 1:
        return 1.0
    return float(max(positive)) / float(min(positive))


def format_class_share_table(class_names: Sequence[str], counts: Sequence[int],
                             alpha: float) -> str:
    """Render the startup audit table: natural vs effective per-class share."""
    natural = natural_class_shares(counts)
    effective = effective_class_shares(counts, alpha)
    width = max([len("class")] + [len(name) for name in class_names])
    lines = [f"{'class':<{width}}{'train_n':>10}{'natural':>10}{'effective':>11}"]
    for name, count, natural_share, effective_share in zip(
            class_names, counts, natural, effective):
        lines.append(f"{name:<{width}}{int(count):>10d}"
                     f"{natural_share:>10.4f}{effective_share:>11.4f}")
    return "\n".join(lines)
