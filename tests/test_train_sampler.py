"""Integration tests for the imbalance-aware training stream.

torch is not installed in the development environment, so this module skips
cleanly; the pure arithmetic behind the sampler is covered by
``test_imbalance.py`` without torch.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import imbalance  # noqa: E402
import train_efficientnet as train  # noqa: E402


class LabelDataset(torch.utils.data.Dataset):
    """Minimal ImageFolder stand-in: only ``targets``/``classes`` are read."""

    def __init__(self, labels: list[int], num_classes: int) -> None:
        self.targets = list(labels)
        self.classes = [f"class_{index}" for index in range(num_classes)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(1), self.targets[index]


def balanced_dataset() -> LabelDataset:
    return LabelDataset([0] * 80 + [1] * 20, 2)


def loader_steps(sampler: object = None) -> int:
    dataset = balanced_dataset()
    train_loader, _, _ = train.build_loaders(
        dataset, dataset, dataset, 16, 16, 0, 42, False, sampler=sampler)
    return len(train_loader)


def test_alpha_zero_keeps_the_historical_shuffled_loader() -> None:
    dataset = balanced_dataset()

    assert train.build_train_sampler(dataset, 0.0) is None

    train_loader, val_loader, test_loader = train.build_loaders(
        dataset, dataset, dataset, 16, 16, 0, 42, False)

    assert isinstance(train_loader.sampler, torch.utils.data.RandomSampler)
    assert train_loader.drop_last is True
    for eval_loader in (val_loader, test_loader):
        assert isinstance(eval_loader.sampler, torch.utils.data.SequentialSampler)
        assert eval_loader.drop_last is False


def test_sampler_replaces_and_keeps_steps_per_epoch() -> None:
    dataset = balanced_dataset()
    sampler = train.build_train_sampler(dataset, 0.5)

    assert sampler is not None
    assert sampler.replacement is True
    assert sampler.num_samples == len(dataset)
    assert sampler.generator is None
    assert loader_steps(sampler) == loader_steps()


def test_sampler_realises_the_effective_class_share() -> None:
    labels = [0] * 800 + [1] * 200
    dataset = LabelDataset(labels, 2)
    sampler = train.build_train_sampler(dataset, 1.0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=100, sampler=sampler)

    torch.manual_seed(0)
    drawn = torch.cat([targets for _, targets in loader]).bincount(minlength=2)
    realised = (drawn / drawn.sum()).tolist()

    expected = imbalance.effective_class_shares([800, 200], 1.0).tolist()
    assert realised == pytest.approx(expected, abs=0.02)
