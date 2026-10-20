"""Evaluation metrics and run-artifact schema helpers.

Pure numpy (no torch), so the numbers that end up in ``history.csv``,
``test_metrics.json`` and ``run_config.json`` can be unit-tested without a
dataset, a GPU, or even a torch install.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

# One row per epoch. Per-class columns are appended by ``history_fieldnames``
# because the class set differs between datasets (6 classes for kaggle, 9 for
# roboflow/trashbox): the header is derived from the run, never assumed.
HISTORY_BASE_FIELDS = [
    "stage", "epoch", "lr_head", "lr_backbone", "train_loss", "train_acc",
    "val_loss", "val_acc", "val_f1", "val_top3", "seconds",
]


def history_fieldnames(class_names: Sequence[str]) -> List[str]:
    """Return the ``history.csv`` header for a run with *class_names*.

    Every class contributes a per-class F1 and recall column so a minority
    class collapsing mid-training is visible while the run is still going,
    not only in the post-hoc confusion matrix.
    """
    return (list(HISTORY_BASE_FIELDS)
            + [f"val_f1_{name}" for name in class_names]
            + [f"val_recall_{name}" for name in class_names])


def summarize_confusion(confusion: np.ndarray, class_names: Sequence[str],
                        top3_correct: int, total: int) -> Dict[str, Any]:
    """Turn a confusion matrix into per-class and summary metrics.

    ``confusion[i, j]`` counts samples whose true class is ``i`` and predicted
    class is ``j``.

    Aggregation contract:

    * ``macro_f1`` averages over *all* classes, including absent ones (this is
      the historical behaviour and it drives ``--monitor f1``).
    * ``balanced_acc``, ``macro_precision``, ``macro_recall`` and
      ``worst_class_f1`` average/select over classes that actually have
      support in this split. A class with zero support has an undefined
      recall; counting it as ``0`` would invent a collapse that no sample can
      contradict. With no support at all the scalars are ``0`` and
      ``worst_class`` is ``None``. ``balanced_acc`` is the mean per-class
      recall, so it equals ``macro_recall`` by definition.
    """
    confusion = np.asarray(confusion, dtype=np.int64)
    num_classes = confusion.shape[0]
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    precision = np.divide(true_positive, predicted, out=np.zeros(num_classes), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros(num_classes), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros(num_classes),
                   where=denominator > 0)

    grand_total = confusion.sum()
    accuracy = float(true_positive.sum() / grand_total) if grand_total > 0 else 0.0
    present = np.flatnonzero(support > 0)
    if present.size:
        macro_precision = float(precision[present].mean())
        macro_recall = float(recall[present].mean())
        balanced_acc = macro_recall
        worst_index = int(present[int(np.argmin(f1[present]))])
        worst_class_f1 = float(f1[worst_index])
        worst_class: Any = class_names[worst_index]
    else:
        balanced_acc = macro_precision = macro_recall = worst_class_f1 = 0.0
        worst_class = None

    per_class = {
        name: {"precision": float(precision[i]), "recall": float(recall[i]),
               "f1": float(f1[i]), "support": int(support[i])}
        for i, name in enumerate(class_names)
    }
    return {
        "acc": accuracy,
        "top1": accuracy,
        "top3": float(top3_correct / total) if total > 0 else 0.0,
        "macro_f1": float(f1.mean()) if num_classes > 0 else 0.0,
        "balanced_acc": balanced_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "worst_class_f1": worst_class_f1,
        "worst_class": worst_class,
        "per_class": per_class,
        "confusion": confusion.tolist(),
    }
