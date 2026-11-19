#!/usr/bin/env python3
"""One table of per-architecture facts for training, export and evaluation.

The training algorithm is backbone-agnostic: loss, metrics, samplers, AMP,
checkpoint selection and history only see ``(logits, targets)``. What is bound
to a specific network is a handful of facts — the factory, the pretrained
weights, where the classifier ``Linear`` lives, its input width and where the
last feature map comes from. They live here so that
``train_efficientnet.py``, ``export_onnx.py``, ``check_onnx_parity.py`` and the
cross-evaluation notebook all rebuild the same model from the same record.

See ``docs/architecture-swap.md`` for why the surface is this small and which
architectures were rejected (MobileNetV3 breaks the static Grad-CAM contract;
ResNet has no ``.features`` attribute and would need a feature-extractor hook).

Design notes
------------
* Only the architectures listed in :data:`ARCHS` are supported. Adding one is
  a registry entry plus, if it is not a ``features``/``classifier`` network,
  the extractor hook described in ``docs/architecture-swap.md`` §4.
* Checkpoints record ``arch``, ``arch_weights``, ``img_size``,
  ``feature_channels``, ``head_attr`` and ``head_index``. A checkpoint without
  ``arch`` (everything trained before this module existed) is read as
  :data:`DEFAULT_ARCH` with a printed warning, never silently as a guess.
* Pretrained weights download on first use. Pass ``pretrained=False`` to build
  an untrained network for export, parity and tests.

CLI usage::

    python scripts/train_efficientnet.py --arch mobilenet_v2 --data-dir data/merged_dataset
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    efficientnet_b0,
    mobilenet_v2,
)

#: Architecture used when ``--arch`` is not passed and by loaders reading a
#: checkpoint that predates the ``arch`` key.
DEFAULT_ARCH = "efficientnet_b0"


@dataclass(frozen=True)
class ArchSpec:
    """Every per-architecture fact the pipeline needs.

    ``head_attr``/``head_index`` point at the classifier ``Linear`` inside the
    model (``classifier[1]`` for EfficientNet and MobileNet, ``fc`` for a
    ResNet). ``feature_attr`` is the module that produces the last feature map
    used by Grad-CAM. ``gradcam_linear_head`` is True only when the pooled
    vector reaches the classifier through a single ``Linear`` with no
    activation in between, which is what makes the exported
    ``classifier_weights`` matrix an exact Grad-CAM weight.
    """

    name: str
    factory: Callable[..., nn.Module]
    weights: Any
    in_features: int
    feature_channels: int
    head_attr: str
    head_index: Optional[int]
    feature_attr: str = "features"
    gradcam_linear_head: bool = True
    default_img_size: int = 224

    @property
    def weights_name(self) -> Optional[str]:
        """Weights enum member name, or ``None`` for an untrained build."""
        return None if self.weights is None else str(self.weights.name)


#: Supported architectures, in the order shown by ``--arch``. The default is
#: first. ``IMAGENET1K_V2`` weights are used for MobileNetV2 (better accuracy
#: than V1 at the same FLOPs); the checkpoint records which enum was used.
ARCHS: Dict[str, ArchSpec] = {
    "efficientnet_b0": ArchSpec(
        name="efficientnet_b0",
        factory=efficientnet_b0,
        weights=EfficientNet_B0_Weights.IMAGENET1K_V1,
        in_features=1280,
        feature_channels=1280,
        head_attr="classifier",
        head_index=1,
        feature_attr="features",
    ),
    "mobilenet_v2": ArchSpec(
        name="mobilenet_v2",
        factory=mobilenet_v2,
        weights=MobileNet_V2_Weights.IMAGENET1K_V2,
        in_features=1280,
        feature_channels=1280,
        head_attr="classifier",
        head_index=1,
        feature_attr="features",
    ),
}


def arch_names() -> List[str]:
    """Return the supported architecture names in CLI order."""
    return list(ARCHS)


def get_arch(name: str) -> ArchSpec:
    """Return the spec for *name*, or exit with the valid choices."""
    try:
        return ARCHS[name]
    except KeyError:
        raise SystemExit(
            f"[arch] unknown architecture {name!r}; choose from {arch_names()}"
        ) from None


def get_head(model: nn.Module, spec: ArchSpec) -> nn.Linear:
    """Return the classifier ``Linear`` of *model* according to *spec*."""
    parent = getattr(model, spec.head_attr, None)
    if parent is None:
        raise AttributeError(
            f"{type(model).__name__} has no attribute {spec.head_attr!r} "
            f"(arch={spec.name})"
        )
    head = parent if spec.head_index is None else parent[spec.head_index]
    if not isinstance(head, nn.Linear):
        raise TypeError(
            f"{spec.name}: expected {spec.head_attr}[{spec.head_index}] to be "
            f"nn.Linear, got {type(head).__name__}"
        )
    return head


def set_head(model: nn.Module, spec: ArchSpec, head: nn.Linear) -> None:
    """Install *head* in the slot named by *spec*."""
    if spec.head_index is None:
        setattr(model, spec.head_attr, head)
    else:
        getattr(model, spec.head_attr)[spec.head_index] = head


def head_state_dict_key(spec: ArchSpec) -> str:
    """State-dict key of the classifier weight (``classifier.1.weight``)."""
    prefix = spec.head_attr
    if spec.head_index is not None:
        prefix = f"{prefix}.{spec.head_index}"
    return f"{prefix}.weight"


def head_parameters(model: nn.Module, spec: ArchSpec) -> List[nn.Parameter]:
    """Parameters of the classifier head only."""
    return list(get_head(model, spec).parameters())


def backbone_parameters(model: nn.Module, spec: ArchSpec) -> List[nn.Parameter]:
    """Every parameter that is not in the classifier head.

    Defined by exclusion rather than by ``.features`` so a backbone without
    that attribute (ResNet) does not break the warmup freeze.
    """
    head_ids = {id(p) for p in head_parameters(model, spec)}
    return [p for p in model.parameters() if id(p) not in head_ids]


def replace_head(model: nn.Module, spec: ArchSpec, num_classes: int) -> nn.Linear:
    """Replace the head with a fresh ``Linear(in_features, num_classes)``.

    The registered width is a real guard: if torchvision changes the backbone,
    this fails instead of silently training a head with the wrong shape.
    """
    head = get_head(model, spec)
    if head.in_features != spec.in_features:
        raise ValueError(
            f"{spec.name}: head in_features={head.in_features} != registered "
            f"in_features={spec.in_features}; update the registry in "
            f"scripts/architectures.py"
        )
    new_head = nn.Linear(head.in_features, num_classes)
    set_head(model, spec, new_head)
    return new_head


def build_model(
    spec: ArchSpec, num_classes: int, pretrained: bool = True
) -> nn.Module:
    """Build *spec*'s network with a fresh *num_classes*-way linear head."""
    model = spec.factory(weights=spec.weights if pretrained else None)
    replace_head(model, spec, num_classes)
    return model


def extract_features(
    model: nn.Module, spec: ArchSpec, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(last feature map, globally pooled features)``.

    Pooling goes through ``F.adaptive_avg_pool2d`` instead of ``model.avgpool``
    because torchvision's ``MobileNetV2`` has no ``avgpool`` attribute — it
    pools inline in ``forward``. For EfficientNet the two are identical.
    """
    features = getattr(model, spec.feature_attr)(images)
    return features, F.adaptive_avg_pool2d(features, (1, 1))


def checkpoint_metadata(spec: ArchSpec, img_size: int) -> Dict[str, Any]:
    """The architecture block written into every checkpoint."""
    return {
        "arch": spec.name,
        "arch_weights": spec.weights_name,
        "img_size": int(img_size),
        "feature_channels": int(spec.feature_channels),
        "head_attr": spec.head_attr,
        "head_index": spec.head_index,
        "torchvision": torchvision.__version__,
    }


def arch_from_checkpoint(payload: Any, source: str = "checkpoint") -> ArchSpec:
    """Resolve the architecture recorded in *payload*.

    A missing ``arch`` means the checkpoint predates the registry, so the
    default is used with a printed warning (backward compatibility with the
    runs already under ``runs/``). An unknown ``arch`` is an error: rebuilding
    a different network than the one trained would only be caught later, by
    ``load_state_dict(strict=True)``, with a much less useful message.
    """
    name = payload.get("arch") if isinstance(payload, dict) else None
    if not name:
        print(
            f"[arch] WARNING: {source} has no 'arch' key; assuming "
            f"{DEFAULT_ARCH}. Pass the architecture explicitly or retrain to "
            f"record it."
        )
        return ARCHS[DEFAULT_ARCH]
    return get_arch(str(name))


def checkpoint_img_size(
    payload: Any, spec: ArchSpec, default: Optional[int] = None
) -> int:
    """Resolve the input resolution recorded in *payload*.

    ``img_size`` is a top-level key in new checkpoints; older ones only carry
    it inside ``args``. Falls back to *default* or the spec's default.
    """
    fallback = int(default if default is not None else spec.default_img_size)
    if not isinstance(payload, dict):
        return fallback
    if payload.get("img_size"):
        return int(payload["img_size"])
    args = payload.get("args") or {}
    if isinstance(args, dict) and args.get("img_size"):
        return int(args["img_size"])
    return fallback
