#!/usr/bin/env python3
"""Train an EfficientNetB0 classifier on a trash image dataset.

What this script does
---------------------
Standalone PyTorch / torchvision training: it builds ``ImageFolder`` datasets
from a ``train/val/test`` tree, applies ImageNet-style augmentation, then
fine-tunes ``efficientnet_b0`` in two stages (frozen-head warmup, then full
fine-tune). It writes a complete set of run artifacts: checkpoints
(``best.pt`` / ``last.pt`` / ``final.pt``), ``history.csv``, ``run_config.json``,
``class_mapping.json``, ``confusion_matrix.csv/.png`` and ``test_metrics.json``.

Expected layout::

    merged_dataset/train/<class>/*.jpg
    merged_dataset/val/<class>/*.jpg
    merged_dataset/test/<class>/*.jpg
    merged_dataset/manifest.csv          (not read at train time)

Class order and count are never hardcoded: they always come from
``dataset.classes`` (the datasets used here carry 6 or 9 classes).

Class imbalance
---------------
``--sampler-alpha`` (default ``0.0``) controls how the training stream is
resampled. At ``0.0`` the loader is the historical shuffled stream with the
natural class distribution. Above ``0`` samples are drawn with replacement,
``len(train)`` per epoch, with per-sample weight ``1 / n_c**alpha`` — class
``c`` then accounts for a share proportional to ``n_c**(1 - alpha)``.
``1.0`` makes every class equally likely per epoch. Validation and test
loaders are never sampled or shuffled.

Because the stream is resampled, the training loss is a
resampled-distribution quantity as soon as ``alpha > 0`` and is **not
comparable** to validation/test loss. That is why ``--monitor`` also offers
``balanced_acc`` and why the startup audit table prints natural vs effective
per-class share before any GPU time is spent.

The pure-numpy helpers live in ``imbalance.py`` and ``metrics.py`` next to
this file (they are unit-tested without torch, so keep all three files
together when uploading to Colab).

Runs in Google Colab on A100 40/80GB, L4 24GB and G4, and also on T4 and CPU.
Hard deps are ``torch``, ``torchvision``, ``PIL``, ``numpy``. ``matplotlib``
and ``tqdm`` are optional.

Colab usage
-----------
Drive must be mounted in a notebook CELL (``drive.mount`` needs the kernel and
cannot work inside a ``!python`` subprocess)::

    from google.colab import drive
    drive.mount('/content/drive')

    !python train_efficientnet.py \\
        --data-zip /content/drive/MyDrive/zips/merged_dataset.zip

Without Drive, upload the zip from a cell and pass its ``/content`` path::

    from google.colab import files
    files.upload()

    !python train_efficientnet.py --data-zip /content/merged_dataset.zip

Already-extracted data under ``/content`` is auto-discovered::

    !python train_efficientnet.py --epochs 30

CPU / T4 smoke test::

    !python train_efficientnet.py --epochs 2 --warmup-epochs 1 --no-amp

Making the dataset zip locally
------------------------------
From the repository root, so the archive contains a top-level
``merged_dataset/`` folder::

    cd data && zip -r merged_dataset.zip merged_dataset

Then upload ``data/merged_dataset.zip`` to Drive (or Colab files) and pass it
with ``--data-zip``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

# Pure-numpy helpers kept next to this file so they are unit-testable without
# torch (tests/test_imbalance.py, tests/test_metrics.py).
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from imbalance import (  # noqa: E402
    class_counts,
    effective_class_shares,
    format_class_share_table,
    imbalance_ratio,
    natural_class_shares,
    per_sample_weights,
    validate_alpha,
)
from metrics import history_fieldnames, summarize_confusion  # noqa: E402

try:  # optional dependency
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:  # pragma: no cover - optional
    HAS_TQDM = False
    tqdm = None  # type: ignore[assignment]

try:  # optional dependency
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:  # pragma: no cover - optional
    HAS_MPL = False
    plt = None  # type: ignore[assignment]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# monitor name -> (metrics key, higher is better). One table so a new monitor
# cannot be added with an accessor and a direction that disagree.
MONITORED_METRICS: Dict[str, Tuple[str, bool]] = {
    "f1": ("macro_f1", True),
    "acc": ("acc", True),
    "loss": ("loss", False),
    "balanced_acc": ("balanced_acc", True),
}


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def is_colab() -> bool:
    """Return True when running inside Google Colab."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def mount_drive() -> None:
    """Mount Google Drive at /content/drive in Colab.

    ``google.colab.drive.mount`` needs the notebook kernel, so it only works
    when called from a cell -- not from a plain ``!python script.py``
    subprocess. On failure this aborts with instructions rather than letting
    the run continue into a confusing "dataset not found" error.
    """
    if not is_colab():
        print("[drive] --mount-drive ignored: not running inside Google Colab.")
        return
    if Path("/content/drive/MyDrive").is_dir():
        print("[drive] Google Drive already mounted.")
        return
    try:
        from google.colab import drive

        drive.mount("/content/drive")
    except Exception as exc:  # noqa: BLE001 - surface any mount failure clearly
        raise SystemExit(
            f"[drive] Failed to mount Google Drive: {exc}\n"
            "[drive] Drive cannot be mounted from a '!python' subprocess. Run this in a "
            "notebook CELL first:\n"
            "            from google.colab import drive\n"
            "            drive.mount('/content/drive')\n"
            "[drive] Then rerun this script WITHOUT --mount-drive. Or upload the zip in a "
            "cell:\n"
            "            from google.colab import files; files.upload()\n"
            "[drive] Then pass its /content path, e.g. "
            "--data-zip /content/merged_dataset.zip"
        )


def seed_everything(seed: int) -> None:
    """Seed random, numpy and torch (all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def get_amp_dtype(device: torch.device) -> Optional[torch.dtype]:
    """Return the AMP dtype to use, or None when AMP is not applicable."""
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast_ctx(use_amp: bool, amp_dtype: Optional[torch.dtype]) -> Any:
    """Build a fresh modern autocast context manager."""
    dtype = amp_dtype if amp_dtype is not None else torch.float16
    return torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp)


def print_environment(device: torch.device) -> None:
    """Print interpreter, library versions and device information."""
    print("=" * 72)
    print(f"python={platform.python_version()} torch={torch.__version__} "
          f"torchvision={torchvision.__version__} numpy={np.__version__}")
    print(f"device={device.type}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"gpu={props.name} vram={props.total_memory / (1024 ** 3):.1f}GB")
    amp_dtype = get_amp_dtype(device)
    print(f"amp_dtype={amp_dtype if amp_dtype is not None else 'disabled'}")
    print("=" * 72)


def auto_batch_size(device: torch.device, num_train: int, min_steps_per_epoch: int) -> int:
    """Choose a batch size from VRAM while keeping enough steps per epoch.

    VRAM decides the largest feasible batch (A100 -> 512, L4/G4 -> 256,
    T4 -> 128). Because there are only ~6.3k training images, that value is
    capped so every epoch still has at least ``min_steps_per_epoch`` optimizer
    steps -- a 1024 batch would give only 6 steps/epoch and hurt convergence.
    Set ``min_steps_per_epoch=0`` to use the full VRAM budget instead.
    """
    if device.type != "cuda":
        print("[batch] CPU detected -> batch size 32")
        return 32
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    thresholds = [(35, 512), (20, 256), (14, 128), (0, 64)]
    vram_batch = next(size for floor, size in thresholds if vram_gb >= floor)
    if min_steps_per_epoch <= 0:
        batch = vram_batch
        cap_note = "uncapped"
    else:
        target = max(1, num_train // min_steps_per_epoch)
        step_cap = max(32, 2 ** int(math.floor(math.log2(target))))
        batch = min(vram_batch, step_cap)
        cap_note = f"step_cap={step_cap}"
    print(f"[batch] GPU {props.name} with {vram_gb:.1f}GB VRAM -> "
          f"vram_batch={vram_batch} {cap_note} -> batch size {batch}")
    return batch


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def alpha_arg(text: str) -> float:
    """argparse type for ``--sampler-alpha``: a float inside ``[0, 1]``."""
    try:
        return validate_alpha(float(text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train EfficientNetB0 on a trash image dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default=None, help="Dataset root.")
    parser.add_argument("--data-zip", default=None, help="Dataset zip (extracted first).")
    parser.add_argument("--extract-to", default="/content/merged_dataset", help="Zip target.")
    parser.add_argument("--mount-drive", action="store_true",
                        help="Mount Drive (cell/kernel only, not from !python).")
    parser.add_argument("--output-dir", default=None, help="Runs parent directory.")
    parser.add_argument("--run-name", default=None, help="Fixed run folder name.")
    parser.add_argument("--epochs", type=int, default=30, help="Total epochs.")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Frozen-head epochs.")
    parser.add_argument("--batch-size", type=int, default=-1, help="Train batch (-1=auto).")
    parser.add_argument("--min-steps-per-epoch", type=int, default=20,
                        help="Cap auto batch so an epoch keeps >=N steps (0=VRAM max).")
    parser.add_argument("--eval-batch-size", type=int, default=0, help="Eval batch (0=auto).")
    parser.add_argument("--img-size", type=int, default=224, help="Input resolution.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Backbone fine-tune lr.")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Classifier head lr.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing.")
    parser.add_argument("--patience", type=int, default=5, help="Early stop patience (0=off).")
    parser.add_argument("--num-workers", type=int, default=0, help="Workers (0=auto).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="Automatic mixed precision.")
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True,
                        help="channels_last memory format.")
    parser.add_argument("--freeze-bn", action="store_true", help="Keep BatchNorm in eval mode.")
    parser.add_argument("--sampler-alpha", type=alpha_arg, default=0.0,
                        help="Training-stream resampling strength in [0,1]: 0 keeps the "
                             "natural shuffled stream, 1 makes classes equally likely.")
    parser.add_argument("--monitor", choices=list(MONITORED_METRICS), default="f1",
                        help="Val metric that selects the best checkpoint.")
    parser.add_argument("--test", action=argparse.BooleanOptionalAction, default=True,
                        help="Evaluate the test split with best weights.")
    parser.add_argument("--save-last", action=argparse.BooleanOptionalAction, default=True,
                        help="Save last.pt every epoch.")
    parser.add_argument("--compile", action="store_true", help="torch.compile with fallback.")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Dataset resolution
# --------------------------------------------------------------------------- #
def is_valid_dataset(path: Path) -> bool:
    """Return True if *path* holds train/val/test splits with class subdirs."""
    if not path.is_dir():
        return False
    for split in ("train", "val", "test"):
        split_dir = path / split
        if not split_dir.is_dir() or not any(c.is_dir() for c in split_dir.iterdir()):
            return False
    return True


def find_dataset_root(base: Path, max_depth: int = 3) -> Optional[Path]:
    """Breadth-first search *base* up to *max_depth* for a valid dataset."""
    if is_valid_dataset(base):
        return base
    frontier = [base]
    for _ in range(max_depth):
        nxt: List[Path] = []
        for directory in frontier:
            if not directory.is_dir():
                continue
            for child in sorted(directory.iterdir()):
                if not child.is_dir():
                    continue
                if is_valid_dataset(child):
                    return child
                nxt.append(child)
        frontier = nxt
    return None


def dataset_candidates(repo_root: Path) -> List[Path]:
    """Return the ordered list of auto-discovery candidate roots."""
    return [
        Path("/content/merged_dataset"),
        Path("/content/data/merged_dataset"),
        repo_root / "data" / "merged_dataset",
        Path("/content/drive/MyDrive/merged_dataset"),
        Path("/content/drive/MyDrive/trash/merged_dataset"),
        Path("/content/drive/MyDrive/zips/merged_dataset"),
    ]


def _failure_message(candidates: List[Path], searched: Optional[Path]) -> str:
    lines = ["Could not locate a valid dataset (needs train/val/test dirs)."]
    if searched is not None:
        lines.append(f"  Given path is not a valid dataset: {searched}")
    lines.append("  Searched candidates:")
    lines += [f"    - {c}" for c in candidates]
    lines.append("  Build the zip locally from the repository root with:")
    lines.append("    cd data && zip -r merged_dataset.zip merged_dataset")
    if is_colab():
        lines.append("  Colab tip: mount Drive in a CELL (not via !python) or pass --data-zip "
                     "with a /content path.")
    return "\n".join(lines)


def _zip_not_found_message(zip_path: Path) -> str:
    search_dirs = [Path("/content"), Path("/content/drive/MyDrive"),
                   Path("/content/drive/MyDrive/zips"), Path.cwd()]
    found: List[Path] = []
    for directory in search_dirs:
        if directory.is_dir():
            found += sorted(directory.glob("*.zip"))
    lines = [f"Dataset zip not found: {zip_path}"]
    if found:
        lines.append("  Zip files currently visible in the runtime:")
        lines += [f"    - {path}" for path in dict.fromkeys(found)]
    lines.append("  Get the zip into the runtime, then retry --data-zip:")
    lines.append("    (cell) from google.colab import files; files.upload()")
    lines.append("    (cell) from google.colab import drive; drive.mount('/content/drive')")
    lines.append("    !wget -O /content/merged_dataset.zip <url>")
    return "\n".join(lines)


def extract_dataset_zip(zip_path: Path, extract_to: Path) -> Path:
    """Extract *zip_path* into *extract_to* and return the dataset root."""
    if not zip_path.is_file():
        raise SystemExit(_zip_not_found_message(zip_path))
    if extract_to.exists():
        existing = find_dataset_root(extract_to)
        if existing is not None:
            print(f"[data] Valid dataset already at {existing}; skipping extraction.")
            return existing
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"[data] Extracting {zip_path} -> {extract_to} ...")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_to)
    root = find_dataset_root(extract_to)
    if root is None:
        raise SystemExit(f"Extracted {zip_path} but no train/val/test found under {extract_to}.")
    print(f"[data] Dataset root: {root}")
    return root


def resolve_data_root(args: argparse.Namespace, repo_root: Path) -> Path:
    """Resolve the dataset root from CLI flags or auto-discovery."""
    if args.data_zip is not None:
        return extract_dataset_zip(Path(args.data_zip), Path(args.extract_to))
    if args.data_dir is not None:
        given = Path(args.data_dir)
        if given.is_file() and given.suffix.lower() == ".zip":
            return extract_dataset_zip(given, Path(args.extract_to))
        if is_valid_dataset(given):
            print(f"[data] Using --data-dir: {given}")
            return given
        raise SystemExit(_failure_message([given], given))
    candidates = dataset_candidates(repo_root)
    for candidate in candidates:
        if is_valid_dataset(candidate):
            print(f"[data] Auto-discovered dataset: {candidate}")
            return candidate
    raise SystemExit(_failure_message(candidates, None))


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """Resolve and create the timestamped run directory."""
    if args.output_dir is not None:
        base = Path(args.output_dir)
    elif Path("/content/drive/MyDrive").is_dir():
        base = Path("/content/drive/MyDrive/trash_efficientnet_b0")
    elif Path("/content").is_dir():
        base = Path("/content/runs")
    else:
        base = Path("./runs")
    name = args.run_name or f"efficientnet_b0_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] Run directory: {run_dir}")
    return run_dir


# --------------------------------------------------------------------------- #
# Transforms and dataloaders
# --------------------------------------------------------------------------- #
def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return (train, eval) transform pipelines."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.60, 1.0), ratio=(3 / 4, 4 / 3)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.2, inplace=True),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(round(img_size * 1.14))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def worker_init_fn(worker_id: int) -> None:
    """Seed python/numpy random per dataloader worker."""
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def build_train_sampler(train_ds: datasets.ImageFolder,
                        alpha: float) -> Optional[WeightedRandomSampler]:
    """Return the resampling sampler for *alpha* > 0, or None for the natural stream.

    ``alpha == 0`` returns ``None`` on purpose: the loader then keeps exactly
    the historical ``shuffle=True`` path, so defaults stay bit-identical and
    checkpoint selection is unchanged.

    Above zero every sample is drawn with replacement, ``len(train_ds)`` draws
    per epoch, with per-sample weight ``1 / n_c**alpha``. No generator is
    attached to the sampler, so its draws come from the global torch RNG and
    the stream differs from epoch to epoch.
    """
    if alpha == 0.0:
        return None
    counts = class_counts(train_ds.targets, len(train_ds.classes))
    weights = per_sample_weights(train_ds.targets, counts, alpha)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(train_ds), replacement=True)


def print_sampler_report(class_names: List[str], counts: List[int], alpha: float) -> None:
    """Print natural vs effective per-class share, before any GPU time is spent."""
    if alpha == 0.0:
        print("[sampler] alpha=0.000 -> natural shuffled stream (no resampling)")
    else:
        print(f"[sampler] alpha={alpha:.3f} -> weighted sampling WITH replacement; "
              f"class share scales as n_c^(1-alpha)")
    print(format_class_share_table(class_names, counts, alpha))
    print(f"[sampler] imbalance_ratio={imbalance_ratio(counts):.2f} "
          f"(largest/smallest train class)")
    if alpha > 0.0:
        print("[sampler] NOTE: train loss is a resampled-distribution quantity and is "
              "NOT comparable to validation/test loss.")


def build_loaders(train_ds: datasets.ImageFolder, val_ds: datasets.ImageFolder,
                  test_ds: datasets.ImageFolder, batch_size: int, eval_batch_size: int,
                  num_workers: int, seed: int, pin_memory: bool,
                  sampler: Optional[WeightedRandomSampler] = None
                  ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build the train/val/test dataloaders.

    Only the training loader can be resampled; validation and test are always
    sequential and unsampled so the reported metrics describe the real,
    imbalanced evaluation distribution. ``drop_last`` stays on and the sampler
    draws ``len(train_ds)`` samples, so ``steps_per_epoch`` is unchanged.
    """
    common: Dict[str, Any] = {"num_workers": num_workers, "pin_memory": pin_memory,
                              "persistent_workers": num_workers > 0}
    if num_workers > 0:
        common["prefetch_factor"] = 2
    generator = torch.Generator()
    generator.manual_seed(seed)
    if sampler is None:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                                  generator=generator, worker_init_fn=worker_init_fn, **common)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  drop_last=True, generator=generator,
                                  worker_init_fn=worker_init_fn, **common)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False,
                            worker_init_fn=worker_init_fn, **common)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False,
                             worker_init_fn=worker_init_fn, **common)
    return train_loader, val_loader, test_loader


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(num_classes: int, device: torch.device, channels_last: bool) -> nn.Module:
    """Build EfficientNetB0 with a fresh classifier head."""
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    assert in_features == 1280, f"Expected 1280 head features, got {in_features}"
    print(f"[model] head in_features={in_features} num_classes={num_classes}")
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] params total={total:,} trainable={trainable:,}")
    return model


def maybe_compile(model: nn.Module, enabled: bool, device: torch.device,
                  img_size: int) -> nn.Module:
    """Optionally wrap the model with torch.compile, falling back on failure.

    ``torch.compile`` is lazy, so a backend failure would otherwise surface in
    the middle of the first training step. A dummy forward here forces the
    compilation to happen up front, where a failure can fall back to eager.
    """
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        print("[compile] torch.compile not available; using eager model.")
        return model
    try:
        compiled = torch.compile(model)
        was_training = model.training
        model.eval()
        with torch.inference_mode():
            compiled(torch.zeros(2, 3, img_size, img_size, device=device))
        model.train(was_training)
        print("[compile] torch.compile enabled.")
        return compiled
    except Exception as exc:  # noqa: BLE001 - any compile/backend failure
        model.train()
        print(f"[compile] torch.compile failed ({exc}); using eager model.")
        return model


def set_bn_eval(model: nn.Module) -> None:
    """Put every BatchNorm2d module back into eval mode."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


# --------------------------------------------------------------------------- #
# Metrics (pure helpers live in metrics.py)
# --------------------------------------------------------------------------- #
def print_metrics_table(class_names: List[str], metrics: Dict[str, Any],
                        imbalance: Optional[float] = None) -> None:
    """Print a clean per-class metric table plus summary scalars."""
    print(f"{'class':<14}{'precision':>11}{'recall':>11}{'f1':>11}{'support':>10}")
    for name in class_names:
        row = metrics["per_class"][name]
        print(f"{name:<14}{row['precision']:>11.4f}{row['recall']:>11.4f}"
              f"{row['f1']:>11.4f}{row['support']:>10d}")
    print(f"summary: acc={metrics['acc']:.4f} balanced_acc={metrics['balanced_acc']:.4f} "
          f"top3={metrics['top3']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    print(f"macro: precision={metrics['macro_precision']:.4f} "
          f"recall={metrics['macro_recall']:.4f} | "
          f"worst_class_f1={metrics['worst_class_f1']:.4f} ({metrics['worst_class']})"
          + (f" | imbalance_ratio={imbalance:.2f}" if imbalance is not None else ""))


def print_confusion_matrix(class_names: List[str], confusion: np.ndarray) -> None:
    """Print an aligned integer confusion matrix."""
    width = max(9, max(len(name) for name in class_names))
    print("confusion matrix (rows=true, cols=pred)")
    print(" " * (width + 2) + "".join(f"{name:>{width + 1}}" for name in class_names))
    for i, name in enumerate(class_names):
        row = "".join(f"{int(confusion[i, j]):>{width + 1}}" for j in range(len(class_names)))
        print(f"{name:>{width}}: {row}")


def save_confusion_csv(path: Path, class_names: List[str], confusion: np.ndarray) -> None:
    """Write the confusion matrix to CSV with a class-name header."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred"] + class_names)
        for i, name in enumerate(class_names):
            writer.writerow([name] + [int(v) for v in confusion[i].tolist()])


def save_confusion_png(path: Path, class_names: List[str], confusion: np.ndarray) -> None:
    """Save the confusion matrix as PNG when matplotlib is importable."""
    if not HAS_MPL:
        return
    try:
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(confusion, interpolation="nearest", cmap="Blues")
        ax.set_title("Confusion matrix")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)), class_names)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 - plotting must never fail the run
        print(f"[warn] Could not write confusion_matrix.png: {exc}")


# --------------------------------------------------------------------------- #
# History logging
# --------------------------------------------------------------------------- #
class HistoryWriter:
    """Appends one row per epoch to history.csv, writing the header once.

    A run directory can be reused (``--run-name`` or a fixed ``--output-dir``).
    Appending under a header from a different class set would silently
    misalign every column, so a mismatch aborts before anything is written.
    """

    def __init__(self, path: Path, fieldnames: List[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self._reject_mismatched_header()

    def _reject_mismatched_header(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(newline="") as handle:
            existing = next(csv.reader(handle), None)
        if existing is None:
            return
        if existing != list(self.fieldnames):
            raise SystemExit(
                f"[history] Refusing to append to {self.path}: the existing header does not "
                f"match this run's schema.\n"
                f"  existing ({len(existing)} cols): {existing}\n"
                f"  expected ({len(self.fieldnames)} cols): {list(self.fieldnames)}\n"
                "  The dataset's class set or the history schema changed. Use a new "
                "--run-name / --output-dir, or delete the stale history.csv."
            )

    def append(self, row: Dict[str, Any]) -> None:
        """Append a row and flush to disk."""
        needs_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #
def train_one_epoch(net: nn.Module, raw_model: nn.Module, loader: DataLoader,
                    criterion: nn.Module, optimizer: torch.optim.Optimizer,
                    device: torch.device, channels_last: bool, use_amp: bool,
                    amp_dtype: Optional[torch.dtype], scaler: Optional[Any],
                    freeze_bn: bool, batch_scheduler: Optional[Any],
                    desc: str) -> Tuple[float, float]:
    """Run one training epoch, returning (average loss, accuracy)."""
    raw_model.train()
    if freeze_bn:
        set_bn_eval(raw_model)
    grad_params = [p for p in raw_model.parameters() if p.requires_grad]
    running_loss, correct, total = 0.0, 0, 0
    iterator = tqdm(loader, desc=desc, leave=False) if HAS_TQDM else loader
    for inputs, targets in iterator:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(use_amp, amp_dtype):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(grad_params, 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(grad_params, 5.0)
            optimizer.step()
        if batch_scheduler is not None:
            batch_scheduler.step()
        running_loss += loss.item() * targets.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)
    return (running_loss / total if total else 0.0, correct / total if total else 0.0)


def evaluate(net: nn.Module, raw_model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device, class_names: List[str], channels_last: bool, use_amp: bool,
             amp_dtype: Optional[torch.dtype]) -> Dict[str, Any]:
    """Evaluate the model, returning loss plus a full metric dictionary."""
    raw_model.eval()
    num_classes = len(class_names)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    running_loss, top3_correct, total = 0.0, 0, 0
    top_k = min(3, num_classes)
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if channels_last:
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            with autocast_ctx(use_amp, amp_dtype):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            running_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            indices = targets * num_classes + preds
            confusion += torch.bincount(indices, minlength=num_classes ** 2).reshape(
                num_classes, num_classes)
            topk = outputs.topk(top_k, dim=1).indices
            top3_correct += (topk == targets.unsqueeze(1)).any(dim=1).sum().item()
            total += targets.size(0)
    metrics = summarize_confusion(confusion.cpu().numpy(), class_names, top3_correct, total)
    metrics["loss"] = running_loss / total if total else 0.0
    return metrics


# --------------------------------------------------------------------------- #
# Checkpointing and stage driver
# --------------------------------------------------------------------------- #
@dataclass
class RunState:
    """Mutable bookkeeping shared across training stages."""

    best_metric: Optional[float] = None
    best_epoch: int = -1
    best_stage: str = "none"
    epoch: int = 0
    last_val: Optional[Dict[str, Any]] = None


def save_checkpoint(path: Path, raw_model: nn.Module, train_ds: datasets.ImageFolder,
                    epoch: int, stage: str, metrics: Dict[str, Any],
                    args: argparse.Namespace) -> None:
    """Save a checkpoint dictionary with model weights and metadata."""
    torch.save({"model": raw_model.state_dict(), "classes": train_ds.classes,
                "class_to_idx": train_ds.class_to_idx, "epoch": epoch, "stage": stage,
                "metrics": metrics, "args": vars(args)}, path)


def monitored_value(monitor: str, metrics: Dict[str, Any]) -> float:
    """Return the value of the monitored metric."""
    key, _ = MONITORED_METRICS[monitor]
    return float(metrics[key])


def is_improvement(monitor: str, value: float, best: Optional[float]) -> bool:
    """Return True when *value* beats *best* for the monitored metric."""
    if best is None:
        return True
    _, higher_is_better = MONITORED_METRICS[monitor]
    return value > best if higher_is_better else value < best


def train_stage(net: nn.Module, raw_model: nn.Module, train_loader: DataLoader,
                val_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer,
                device: torch.device, args: argparse.Namespace, class_names: List[str],
                train_ds: datasets.ImageFolder, use_amp: bool, amp_dtype: Optional[torch.dtype],
                scaler: Optional[Any], history: HistoryWriter, run_dir: Path, state: RunState,
                stage: str, epochs: int, epoch_scheduler: Optional[Any],
                batch_scheduler: Optional[Any], patience: int) -> None:
    """Run one training stage (warmup or fine-tune) for *epochs* epochs."""
    stale = 0
    for i in range(epochs):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            net, raw_model, train_loader, criterion, optimizer, device, args.channels_last,
            use_amp, amp_dtype, scaler, args.freeze_bn, batch_scheduler, f"{stage} {i + 1}")
        val = evaluate(net, raw_model, val_loader, criterion, device, class_names,
                       args.channels_last, use_amp, amp_dtype)
        state.last_val = val
        print(f"[{stage} {i + 1}/{epochs}] train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val loss={val['loss']:.4f} acc={val['acc']:.4f} f1={val['macro_f1']:.4f} "
              f"bal_acc={val['balanced_acc']:.4f} top3={val['top3']:.4f} "
              f"worst={val['worst_class_f1']:.4f} ({val['worst_class']})")
        lrs = [group["lr"] for group in optimizer.param_groups]
        if epoch_scheduler is not None:
            epoch_scheduler.step()
        state.epoch += 1
        row: Dict[str, Any] = {
            "stage": stage, "epoch": state.epoch, "lr_head": f"{lrs[0]:.8g}",
            "lr_backbone": f"{lrs[-1] if len(lrs) > 1 else 0.0:.8g}",
            "train_loss": f"{train_loss:.6f}", "train_acc": f"{train_acc:.6f}",
            "val_loss": f"{val['loss']:.6f}", "val_acc": f"{val['acc']:.6f}",
            "val_f1": f"{val['macro_f1']:.6f}", "val_top3": f"{val['top3']:.6f}",
            "seconds": f"{time.time() - start:.2f}"}
        for name in class_names:
            row[f"val_f1_{name}"] = f"{val['per_class'][name]['f1']:.6f}"
            row[f"val_recall_{name}"] = f"{val['per_class'][name]['recall']:.6f}"
        history.append(row)
        score = monitored_value(args.monitor, val)
        if is_improvement(args.monitor, score, state.best_metric):
            state.best_metric, state.best_epoch = score, state.epoch
            state.best_stage, stale = stage, 0
            save_checkpoint(run_dir / "best.pt", raw_model, train_ds, state.epoch, stage, val, args)
            print(f"[ckpt] new best ({args.monitor}={score:.4f}) at epoch {state.epoch}")
        else:
            stale += 1
        if args.save_last:
            save_checkpoint(run_dir / "last.pt", raw_model, train_ds, state.epoch, stage, val, args)
        if stage == "finetune" and patience > 0 and stale >= patience:
            print(f"[early-stop] no {args.monitor} improvement for {stale} epochs; "
                  f"stopping at epoch {state.epoch}.")
            break


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> None:
    """Entry point: resolve data, build the model, train, and write artifacts."""
    args = parse_args(argv)
    if args.mount_drive:
        mount_drive()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_environment(device)
    repo_root = Path(__file__).resolve().parent.parent
    data_root = resolve_data_root(args, repo_root)
    run_dir = resolve_output_dir(args)

    train_tf, eval_tf = build_transforms(args.img_size)
    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_root / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_root / "test", transform=eval_tf)
    if val_ds.classes != train_ds.classes:
        raise SystemExit(f"Class mismatch val vs train: {val_ds.classes} != {train_ds.classes}")
    if test_ds.classes != train_ds.classes:
        raise SystemExit(f"Class mismatch test vs train: {test_ds.classes} != {train_ds.classes}")
    class_names = list(train_ds.classes)
    print(f"[data] classes ({len(class_names)}): {train_ds.class_to_idx}")
    print(f"[data] counts: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    with (run_dir / "class_mapping.json").open("w") as handle:
        json.dump({"classes": train_ds.classes, "class_to_idx": train_ds.class_to_idx},
                  handle, indent=2)

    train_counts = class_counts(train_ds.targets, len(class_names))
    ratio = imbalance_ratio(train_counts)
    print_sampler_report(class_names, train_counts, args.sampler_alpha)
    empty_classes = [name for name, count in zip(class_names, train_counts) if count == 0]
    if empty_classes:
        print(f"[sampler] WARNING: no train samples for {empty_classes}; they get zero stream "
              f"share and are excluded from the imbalance ratio.")

    batch_size = (auto_batch_size(device, len(train_ds), args.min_steps_per_epoch)
                  if args.batch_size == -1 else args.batch_size)
    if args.batch_size != -1:
        print(f"[batch] Using --batch-size {batch_size}")
    eval_batch_size = (args.eval_batch_size if args.eval_batch_size > 0
                       else min(2 * batch_size, 1024))
    num_workers = args.num_workers if args.num_workers > 0 else min(8, os.cpu_count() or 2)
    print(f"[data] batch={batch_size} eval_batch={eval_batch_size} workers={num_workers} "
          f"img_size={args.img_size}")

    train_loader, val_loader, test_loader = build_loaders(
        train_ds, val_ds, test_ds, batch_size, eval_batch_size, num_workers, args.seed,
        pin_memory=device.type == "cuda",
        sampler=build_train_sampler(train_ds, args.sampler_alpha))
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise SystemExit(
            f"[data] {len(train_ds)} train samples with batch_size={batch_size} and "
            "drop_last=True gives zero steps per epoch. Set --batch-size "
            f"<= {len(train_ds)} or add more data.")
    print(f"[data] steps_per_epoch={steps_per_epoch}")

    raw_model = build_model(len(class_names), device, args.channels_last)
    net = maybe_compile(raw_model, args.compile, device, args.img_size)
    amp_dtype = get_amp_dtype(device)
    use_amp = bool(args.amp) and device.type == "cuda" and amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    print(f"[amp] enabled={use_amp} dtype={amp_dtype if use_amp else 'off'}")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    history = HistoryWriter(run_dir / "history.csv", history_fieldnames(class_names))
    state = RunState()
    warmup_epochs = max(0, min(args.warmup_epochs, args.epochs))
    fine_epochs = args.epochs - warmup_epochs
    print(f"[plan] warmup_epochs={warmup_epochs} fine_epochs={fine_epochs} monitor={args.monitor}")

    try:
        if warmup_epochs > 0:
            for param in raw_model.features.parameters():
                param.requires_grad = False
            optimizer = torch.optim.AdamW(list(raw_model.classifier.parameters()),
                                          lr=args.head_lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, warmup_epochs), eta_min=args.head_lr * 0.05)
            print(f"[stage] warmup: training head only for {warmup_epochs} epochs")
            train_stage(net, raw_model, train_loader, val_loader, criterion, optimizer, device,
                        args, class_names, train_ds, use_amp, amp_dtype, scaler, history, run_dir,
                        state, "warmup", warmup_epochs, scheduler, None, 0)

        if fine_epochs > 0:
            for param in raw_model.parameters():
                param.requires_grad = True
            head_params = list(raw_model.classifier.parameters())
            head_ids = {id(p) for p in head_params}
            decay, no_decay = [], []
            for _, param in raw_model.named_parameters():
                if id(param) in head_ids:
                    continue
                (decay if param.ndim > 1 else no_decay).append(param)
            optimizer = torch.optim.AdamW([
                {"params": head_params, "lr": args.head_lr, "weight_decay": args.weight_decay},
                {"params": decay, "lr": args.lr, "weight_decay": args.weight_decay},
                {"params": no_decay, "lr": args.lr, "weight_decay": 0.0}])
            warm = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=steps_per_epoch)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, (fine_epochs - 1) * steps_per_epoch), eta_min=1e-6)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warm, cosine], milestones=[steps_per_epoch])
            print(f"[stage] fine-tune: full model for {fine_epochs} epochs "
                  f"(head={len(head_params)} decay={len(decay)} no_decay={len(no_decay)})")
            train_stage(net, raw_model, train_loader, val_loader, criterion, optimizer, device,
                        args, class_names, train_ds, use_amp, amp_dtype, scaler, history, run_dir,
                        state, "finetune", fine_epochs, None, scheduler, args.patience)
        else:
            print("[stage] fine_epochs<=0 -> skipping fine-tune stage.")
    except KeyboardInterrupt:
        print("\n[interrupt] Training interrupted. The best checkpoint saved so far is preserved.")
        sys.exit(1)

    if state.last_val is not None:
        last = state.last_val
        print(f"[val] last epoch {state.epoch}: balanced_acc={last['balanced_acc']:.4f} "
              f"macro_precision={last['macro_precision']:.4f} "
              f"macro_recall={last['macro_recall']:.4f} "
              f"worst_class_f1={last['worst_class_f1']:.4f} ({last['worst_class']}) "
              f"imbalance_ratio={ratio:.2f}")

    best_path = run_dir / "best.pt"
    if not best_path.exists():
        print("[warn] No best checkpoint written; keeping current weights.")
        save_checkpoint(best_path, raw_model, train_ds, state.epoch, state.best_stage, {}, args)
    try:
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:  # torch without weights_only
        checkpoint = torch.load(best_path, map_location=device)
    raw_model.load_state_dict(checkpoint["model"])
    print(f"[ckpt] loaded best weights from epoch {checkpoint.get('epoch')} "
          f"({checkpoint.get('stage')}).")
    torch.save(raw_model.state_dict(), run_dir / "final.pt")

    test_summary: Optional[Dict[str, Any]] = None
    if args.test:
        test_metrics = evaluate(net, raw_model, test_loader, criterion, device, class_names,
                                args.channels_last, use_amp, amp_dtype)
        confusion_np = np.array(test_metrics["confusion"], dtype=np.int64)
        print("\n" + "=" * 72 + "\nTEST RESULTS (best checkpoint)\n" + "=" * 72)
        print_metrics_table(class_names, test_metrics, imbalance=ratio)
        print_confusion_matrix(class_names, confusion_np)
        save_confusion_csv(run_dir / "confusion_matrix.csv", class_names, confusion_np)
        save_confusion_png(run_dir / "confusion_matrix.png", class_names, confusion_np)
        test_summary = {
            "loss": float(test_metrics["loss"]), "acc": float(test_metrics["acc"]),
            "top1": float(test_metrics["top1"]), "top3": float(test_metrics["top3"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "balanced_acc": float(test_metrics["balanced_acc"]),
            "macro_precision": float(test_metrics["macro_precision"]),
            "macro_recall": float(test_metrics["macro_recall"]),
            "worst_class_f1": float(test_metrics["worst_class_f1"]),
            "worst_class": test_metrics["worst_class"],
            "imbalance_ratio": float(ratio),
            "per_class": test_metrics["per_class"],
            "split_counts": {"train": len(train_ds), "val": len(val_ds),
                             "test": len(test_ds)}}
        with (run_dir / "test_metrics.json").open("w") as handle:
            json.dump(test_summary, handle, indent=2)
        print(f"[test] loss={test_metrics['loss']:.4f} acc={test_metrics['acc']:.4f} "
              f"f1={test_metrics['macro_f1']:.4f} bal_acc={test_metrics['balanced_acc']:.4f} "
              f"top3={test_metrics['top3']:.4f} "
              f"worst_class_f1={test_metrics['worst_class_f1']:.4f} "
              f"({test_metrics['worst_class']}) imbalance_ratio={ratio:.2f}")
    else:
        print("[test] Skipped (--no-test).")

    run_config = {
        **vars(args),
        "resolved": {"data_root": str(data_root), "run_dir": str(run_dir),
                     "best_checkpoint": str(best_path)},
        "device": str(device),
        "gpu_name": torch.cuda.get_device_properties(0).name if device.type == "cuda" else None,
        "batch_size": batch_size, "eval_batch_size": eval_batch_size,
        "num_workers": num_workers, "steps_per_epoch": steps_per_epoch,
        "num_classes": len(class_names), "classes": class_names,
        "amp_used": use_amp, "amp_dtype": str(amp_dtype) if use_amp else None,
        "best_epoch": state.best_epoch, "best_stage": state.best_stage,
        "best_metric": state.best_metric,
        "imbalance": {
            "sampler_alpha": float(args.sampler_alpha),
            "train_class_counts": dict(zip(class_names, train_counts)),
            "natural_class_shares": {
                name: float(share) for name, share in
                zip(class_names, natural_class_shares(train_counts))},
            "effective_class_shares": {
                name: float(share) for name, share in
                zip(class_names, effective_class_shares(train_counts, args.sampler_alpha))},
            "imbalance_ratio": float(ratio),
            "empty_train_classes": empty_classes,
            "train_loss_comparable_to_val_loss": args.sampler_alpha == 0.0,
        },
        "last_validation": (
            None if state.last_val is None else {
                "loss": float(state.last_val["loss"]),
                "acc": float(state.last_val["acc"]),
                "macro_f1": float(state.last_val["macro_f1"]),
                "balanced_acc": float(state.last_val["balanced_acc"]),
                "macro_precision": float(state.last_val["macro_precision"]),
                "macro_recall": float(state.last_val["macro_recall"]),
                "worst_class_f1": float(state.last_val["worst_class_f1"]),
                "worst_class": state.last_val["worst_class"],
            }),
        "test_metrics": test_summary,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "numpy": np.__version__},
    }
    with (run_dir / "run_config.json").open("w") as handle:
        json.dump(run_config, handle, indent=2, default=str)
    print("=" * 72 + f"\nDone. Artifacts written to: {run_dir}\n" + "=" * 72)


if __name__ == "__main__":
    main()
