"""Tests for the separated training and evaluation criteria.

torch is not installed in the development environment, so this module skips
cleanly; with torch available it locks the invariant that a future weighted
training loss cannot leak into validation/test loss, ``--monitor loss``
checkpoint selection or the ``val_loss`` column of ``history.csv``.
"""

import argparse
import csv
import inspect
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
pytest.importorskip("torchvision")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import train_efficientnet as train  # noqa: E402


class TinyModel(nn.Module):
    """Two-feature linear classifier with deterministic weights."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))
            self.linear.bias.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class TinyDataset(torch.utils.data.Dataset):
    """Six samples over two imbalanced classes, enough for a stage loop."""

    def __init__(self) -> None:
        self.inputs = torch.tensor(
            [[2.0, 0.0], [1.5, 0.5], [1.0, 0.0], [0.0, 2.0], [0.5, 1.5], [0.0, 1.0]]
        )
        self.targets = torch.tensor([0, 0, 0, 0, 0, 1])
        self.classes = ["majority", "minority"]
        self.class_to_idx = {"majority": 0, "minority": 1}

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.inputs[index], int(self.targets[index])


def tiny_loader() -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(TinyDataset(), batch_size=3, shuffle=False)


def stage_args() -> argparse.Namespace:
    return argparse.Namespace(
        channels_last=False, freeze_bn=False, monitor="loss", save_last=False)


def test_eval_criterion_is_always_unweighted_cross_entropy() -> None:
    weights = torch.tensor([0.1, 10.0])

    train_criterion = train.build_train_criterion(0.1, class_weights=weights)
    eval_criterion = train.build_eval_criterion(0.1)

    assert isinstance(train_criterion, nn.CrossEntropyLoss)
    assert isinstance(eval_criterion, nn.CrossEntropyLoss)
    assert train_criterion.weight is not None
    assert torch.equal(train_criterion.weight, weights)
    assert eval_criterion.weight is None
    assert train_criterion.label_smoothing == pytest.approx(0.1)
    assert eval_criterion.label_smoothing == pytest.approx(0.1)


def test_train_criterion_is_unweighted_by_default() -> None:
    criterion = train.build_train_criterion(0.1)

    assert criterion.weight is None
    assert criterion.label_smoothing == pytest.approx(0.1)


def test_unweighted_train_and_eval_criteria_are_equivalent() -> None:
    logits = torch.tensor([[2.0, -1.0], [-1.0, 3.0], [0.5, 0.5]])
    targets = torch.tensor([0, 1, 1])

    train_criterion = train.build_train_criterion(0.1)
    eval_criterion = train.build_eval_criterion(0.1)

    assert train_criterion(logits, targets).item() == pytest.approx(
        eval_criterion(logits, targets).item(), rel=1e-6)


def test_evaluation_path_takes_only_an_evaluation_criterion() -> None:
    evaluate_params = inspect.signature(train.evaluate).parameters
    stage_params = inspect.signature(train.train_stage).parameters

    assert "eval_criterion" in evaluate_params
    assert "train_criterion" not in evaluate_params
    assert "criterion" not in evaluate_params
    assert {"train_criterion", "eval_criterion"} <= set(stage_params)


def test_evaluate_reports_unweighted_cross_entropy() -> None:
    dataset = TinyDataset()
    model = TinyModel()
    eval_criterion = train.build_eval_criterion(0.2)

    metrics = train.evaluate(
        model, model, tiny_loader(), eval_criterion, torch.device("cpu"),
        dataset.classes, channels_last=False, use_amp=False, amp_dtype=None)

    with torch.inference_mode():
        expected = eval_criterion(model(dataset.inputs), dataset.targets).item()
        weighted_criterion = train.build_train_criterion(
            0.2, class_weights=torch.tensor([1.0, 50.0]))
        weighted_loss = weighted_criterion(model(dataset.inputs), dataset.targets).item()
    assert metrics["loss"] == pytest.approx(expected, rel=1e-6)
    assert weighted_loss != pytest.approx(expected, rel=1e-6)


def run_stage(tmp_path: Path, train_criterion: nn.Module,
              stage: str) -> tuple[list[dict[str, str]], object]:
    """Run one stage with a frozen model and return history rows plus run state."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset = TinyDataset()
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    history = train.HistoryWriter(run_dir / "history.csv",
                                  train.history_fieldnames(dataset.classes))
    state = train.RunState()

    train.train_stage(
        model, model, tiny_loader(), tiny_loader(), train_criterion,
        train.build_eval_criterion(0.1), optimizer, torch.device("cpu"), stage_args(),
        dataset.classes, dataset, use_amp=False, amp_dtype=None, scaler=None,
        history=history, run_dir=run_dir, state=state, stage=stage, epochs=3,
        epoch_scheduler=None, batch_scheduler=None, patience=0)

    with (run_dir / "history.csv").open(newline="") as handle:
        return list(csv.DictReader(handle)), state


@pytest.mark.parametrize("stage", ["warmup", "finetune"])
def test_weighted_training_criterion_does_not_touch_validation_loss(
        tmp_path: Path, stage: str) -> None:
    """A weighted training loss must not redefine reported held-out loss.

    ``lr=0`` keeps the model fixed, so any difference in ``val_loss`` can only
    come from the evaluation path reusing the training criterion. Both stages
    are covered because they share the same driver.
    """
    baseline_rows, baseline_state = run_stage(
        tmp_path / "baseline", train.build_train_criterion(0.1), stage)
    weighted_rows, weighted_state = run_stage(
        tmp_path / "weighted",
        train.build_train_criterion(0.1, class_weights=torch.tensor([1.0, 50.0])), stage)

    assert [row["val_loss"] for row in weighted_rows] == \
        [row["val_loss"] for row in baseline_rows]
    assert [row["val_acc"] for row in weighted_rows] == \
        [row["val_acc"] for row in baseline_rows]
    assert [row["train_loss"] for row in weighted_rows] != \
        [row["train_loss"] for row in baseline_rows]
    assert weighted_state.best_epoch == baseline_state.best_epoch
    assert weighted_state.best_metric == pytest.approx(baseline_state.best_metric)
