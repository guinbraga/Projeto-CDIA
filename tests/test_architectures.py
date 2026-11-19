"""CPU tests for the architecture registry and checkpoint metadata.

torch is required; the rest of the suite runs without it and skips cleanly.
These tests are the contract that lets ``export_onnx.py``, ``check_onnx_parity.py``
and the cross-eval notebook rebuild a checkpoint without hardcoding a network:
every registered spec builds with ``weights=None``, the head is replaced in the
slot the spec names, the metadata round-trips, and a checkpoint without ``arch``
still loads as EfficientNet-B0.
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import architectures  # noqa: E402
import train_efficientnet as train  # noqa: E402


class FakeImageFolder:
    """The only attributes ``save_checkpoint`` reads from a dataset."""

    classes = ["alpha", "beta", "gamma"]
    class_to_idx = {"alpha": 0, "beta": 1, "gamma": 2}


def build_untrained(name: str, num_classes: int = 3):
    spec = architectures.get_arch(name)
    return spec, architectures.build_model(spec, num_classes, pretrained=False)


def test_default_arch_is_registered_and_first():
    assert architectures.DEFAULT_ARCH in architectures.ARCHS
    assert architectures.arch_names()[0] == architectures.DEFAULT_ARCH


@pytest.mark.parametrize("name", architectures.arch_names())
def test_build_replaces_head_with_requested_width(name):
    spec, model = build_untrained(name, num_classes=7)
    head = architectures.get_head(model, spec)
    assert isinstance(head, torch.nn.Linear)
    assert head.in_features == spec.in_features
    assert head.out_features == 7


@pytest.mark.parametrize("name", architectures.arch_names())
def test_feature_map_channels_match_the_registry(name):
    spec, model = build_untrained(name)
    model.eval()
    with torch.inference_mode():
        # A small input keeps the CPU test fast; the channel count is
        # resolution-independent and the padding/spatial size does not matter.
        features, pooled = architectures.extract_features(
            model, spec, torch.zeros(1, 3, 64, 64)
        )
    assert features.shape[1] == spec.feature_channels
    assert pooled.shape == (1, spec.feature_channels, 1, 1)


@pytest.mark.parametrize("name", architectures.arch_names())
def test_head_state_dict_key_exists(name):
    spec, model = build_untrained(name, num_classes=5)
    key = architectures.head_state_dict_key(spec)
    state = model.state_dict()
    assert key in state, f"{name}: {key} missing from state_dict"
    assert state[key].shape[0] == 5


@pytest.mark.parametrize("name", architectures.arch_names())
def test_head_and_backbone_parameters_partition_the_model(name):
    spec, model = build_untrained(name)
    head = architectures.head_parameters(model, spec)
    backbone = architectures.backbone_parameters(model, spec)
    assert {id(p) for p in head}.isdisjoint({id(p) for p in backbone})
    assert len(head) + len(backbone) == len(list(model.parameters()))
    # The warmup stage must actually freeze something: the backbone of every
    # registered net is far larger than its head.
    assert len(backbone) > len(head)


def test_replace_head_rejects_a_width_that_does_not_match_the_registry():
    spec, model = build_untrained("mobilenet_v2")
    wrong = dataclasses.replace(spec, in_features=999)
    with pytest.raises(ValueError, match="in_features"):
        architectures.replace_head(model, wrong, num_classes=3)


def test_head_key_for_the_registered_heads_is_stable():
    # The cross-eval notebook and older export code used this literal; the
    # registry must keep producing it.
    for name in architectures.arch_names():
        assert architectures.head_state_dict_key(architectures.get_arch(name)) == (
            "classifier.1.weight"
        )


def test_checkpoint_metadata_round_trips(tmp_path):
    spec, model = build_untrained("mobilenet_v2")
    args = argparse.Namespace(arch="mobilenet_v2", img_size=224)
    path = tmp_path / "best.pt"
    train.save_checkpoint(path, model, FakeImageFolder(), 3, "finetune", {}, args)

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    assert ckpt["arch"] == "mobilenet_v2"
    assert ckpt["arch_weights"] == "IMAGENET1K_V2"
    assert ckpt["img_size"] == 224
    assert ckpt["feature_channels"] == spec.feature_channels
    assert ckpt["head_attr"] == "classifier"
    assert ckpt["head_index"] == 1
    assert ckpt["classes"] == FakeImageFolder.classes

    assert architectures.arch_from_checkpoint(ckpt) is spec
    assert architectures.checkpoint_img_size(ckpt, spec) == 224


def test_checkpoint_without_arch_falls_back_to_efficientnet_b0(capsys):
    spec = architectures.arch_from_checkpoint({"classes": ["a"]}, source="old.pt")
    assert spec.name == "efficientnet_b0"
    assert "no 'arch' key" in capsys.readouterr().out


def test_unknown_arch_in_checkpoint_exits():
    with pytest.raises(SystemExit, match="unknown architecture"):
        architectures.arch_from_checkpoint({"arch": "resnet152"})


def test_legacy_checkpoint_loads_as_efficientnet_b0():
    _, model = build_untrained("efficientnet_b0")
    state = model.state_dict()
    spec = architectures.arch_from_checkpoint({"model": state}, source="legacy.pt")
    rebuilt = architectures.build_model(spec, 3, pretrained=False)
    rebuilt.load_state_dict(state, strict=True)  # must not raise


def test_rebuilding_the_wrong_arch_fails_loudly():
    _, model = build_untrained("mobilenet_v2")
    wrong_spec = architectures.get_arch("efficientnet_b0")
    rebuilt = architectures.build_model(wrong_spec, 3, pretrained=False)
    with pytest.raises(RuntimeError):
        rebuilt.load_state_dict(model.state_dict(), strict=True)


def test_img_size_prefers_the_top_level_key_then_args():
    spec = architectures.get_arch("efficientnet_b0")
    assert architectures.checkpoint_img_size({"img_size": 256}, spec) == 256
    assert architectures.checkpoint_img_size(
        {"args": {"img_size": 256}}, spec
    ) == 256
    # Top-level wins over the archived CLI namespace.
    assert architectures.checkpoint_img_size(
        {"img_size": 224, "args": {"img_size": 256}}, spec
    ) == 224
    assert architectures.checkpoint_img_size({}, spec) == spec.default_img_size


@pytest.mark.parametrize("name", architectures.arch_names())
def test_gradcam_export_wrapper_matches_the_classifier(name):
    """One forward must yield the same logits, the feature map and the head.

    This is the CPU-level guard for the MobileNetV2 trap: torchvision's
    MobileNetV2 has no ``.avgpool`` attribute, so a wrapper that calls
    ``model.avgpool`` raises here instead of only at export time.
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import export_onnx

    spec, model = build_untrained(name, num_classes=4)
    model.eval()
    wrapper = export_onnx.GradCamExportModel(model, spec).eval()
    images = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        logits, features, weights = wrapper(images)
        expected = model(images)
    assert torch.allclose(logits, expected, atol=1e-6)
    assert features.shape[1] == spec.feature_channels
    assert tuple(weights.shape) == (4, spec.in_features)
