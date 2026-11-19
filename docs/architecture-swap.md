# Training another backbone instead of EfficientNet-B0

Status: **implemented for MobileNetV2 and ResNet18** on 2026-11-19 (see §10).
The analysis below is kept as the design record: §1–§4 describe the coupling as
it was before the patch, §5 is annotated with what landed, and §9 records the
decisions. MobileNetV3 remains unimplemented by choice.
Scope: what in the training code and its consumers is bound to `efficientnet_b0`,
what breaks for MobileNet/ResNet, and the smallest seam that keeps one training
algorithm for several backbones.

## 1. Bottom line

The training *algorithm* is already backbone-agnostic. Loss, metrics, samplers,
AMP, checkpoint selection, early stopping, history and confusion matrices operate
on `(logits, targets)` and never mention the network. What is bound is a small
surface: **six call sites in `train_efficientnet.py`**, plus the fact that the
checkpoint does not record which architecture produced it, plus four downstream
artifacts that rebuild the model by hand.

Concrete count in `scripts/train_efficientnet.py` (1,798 lines):

| Coupling | Lines | Kind |
| --- | ---: | --- |
| import of model + weights enum | 134 | mechanical |
| `build_model`: factory, weights, `classifier[1]`, `assert 1280` | 835–850 | functional |
| warmup freeze via `.features` | 1492 | functional (breaks ResNet) |
| warmup optimizer via `.classifier` | 1494–1498 | functional (breaks ResNet) |
| fine-tune head split via `.classifier` | 1533–1539 | functional (breaks ResNet) |
| checkpoint payload lacks `arch` | 1140–1161 | structural |
| run naming / Drive dir / CLI text | 318, 585, 590 | cosmetic |
| `final.pt` is a bare `state_dict` | 1631 | structural |

Everything else — `train_one_epoch` (1033), `evaluate` (1082), `train_stage`
(1211), `main`'s schedule (1490–1592), `imbalance.py`, `metrics.py`,
`build_transforms` (684, ImageNet mean/std is shared by all torchvision
ImageNet-1k weights) — needs no change. `num_classes` already comes from
`dataset.classes`, so the 6/8/9-class split difference is free.

## 2. What breaks, per candidate

Verified against the environment's torchvision (`2.5.1` / `0.20.1`).

| | EfficientNet-B0 | MobileNetV2 | MobileNetV3-Large | ResNet18 | ResNet50 |
| --- | --- | --- | --- | --- | --- |
| factory | `efficientnet_b0` | `mobilenet_v2` | `mobilenet_v3_large` | `resnet18` | `resnet50` |
| weights (v0.20) | `..._B0_Weights.IMAGENET1K_V1` | `..._V2_Weights.IMAGENET1K_V1`/`V2` | `..._V3_Large_Weights.IMAGENET1K_V1`/`V2` | `ResNet18_Weights.IMAGENET1K_V1` | `ResNet50_Weights.IMAGENET1K_V1`/`V2` |
| head module | `classifier[1]` | `classifier[1]` | `classifier[3]` | `fc` | `fc` |
| head in_features | 1280 | 1280 | 1280 | 512 | 2048 |
| `.features` attr | yes | yes | yes | **no** | **no** |
| `.classifier` attr | yes | yes | yes | **no** | **no** |
| last feature map @224 | 1280×7×7 | 1280×7×7 | 960×7×7 | 512×7×7 | 2048×7×7 |
| Grad-CAM `weight/(H·W)` identity | holds | holds | **does not hold** | holds | holds |

Consequences for the current code:

- **MobileNetV2** is a near drop-in: same `classifier[1]` slot, same 1280
  in_features, same 1280×7×7 feature map, head is `Dropout → Linear`. Only the
  factory, the weights enum and the `assert in_features == 1280` need to change
  (the assert happens to pass, which is a trap: it would pass while the export
  script rebuilds the wrong network). The export wrapper is the one exception:
  torchvision's `MobileNetV2` has **no `.avgpool` attribute** (its forward calls
  `F.adaptive_avg_pool2d` inline), while `GradCamExportModel` calls
  `self.model.avgpool(features)`, so that call must become
  `F.adaptive_avg_pool2d`. The Grad-CAM identity itself is exact for this head
  (verified numerically: max error 0.0 against autograd).
- **MobileNetV3-Large** breaks the head index (`classifier[3]`) and, more
  importantly, the Grad-CAM contract: its head is
  `Linear(960→1280) → Hardswish → Dropout → Linear(1280→C)`, so the gradient of
  a class score with respect to the pooled feature vector is *not* the last
  linear's weights; it is `W₃[c] · diag(hardswish′(p)) · W₀`, which depends on
  the image. The exported static `classifier_weights` matrix would produce a
  wrong heatmap. The classification itself is fine.
- **ResNet18/50** fail in `main` at line 1492 with `AttributeError: 'ResNet'
  object has no attribute 'features'`, then again at 1495/1533 for
  `.classifier`. Their head is `fc`, their feature map is `layer4` output, and
  the Grad-CAM identity does hold (GAP → fc, no activation after the last conv).

One more trap that is independent of the architecture: `--img-size` is searched
as 224 or 256 by the tuner, and every hardcoded `7` (export `FEATURES_SHAPE`,
app docs, `IMG_SIZE = 224` in the parity/calibration scripts) assumes the 224
stride-32 output. Any new backbone trained at 256 yields 8×8.

## 3. The structural problem: checkpoints do not say what they are

`save_checkpoint` writes `model`, `classes`, `class_to_idx`, `epoch`, `stage`,
`metrics`, `args`. `args` carries `img_size`, `lr`, etc., but there is no
`arch`. `final.pt` is a bare `state_dict`, so it carries nothing at all.

Everything that loads a checkpoint therefore hardcodes the rebuild:

- `scripts/export_onnx.py:96` — `efficientnet_b0(weights=None)` +
  `classifier[1]`, inside `GradCamExportModel` it also hardcodes
  `model.features`, `model.avgpool`, `model.classifier`, and
  `FEATURES_SHAPE = (1, 1280, 7, 7)`.
- `scripts/check_onnx_parity.py:63` — same rebuild, plus `IMG_SIZE = 224` and
  `RESIZE_SHORT_SIDE = 255`.
- `runs/Avaliacao_Cruzada_Lixo.ipynb`, cell 8 — `head_key =
  "classifier.1.weight"` and `torchvision.models.efficientnet_b0(weights=None)`.
- `scripts/calibrate_threshold.py` and `scripts/make_golden_fixtures.py` only
  consume the ONNX graph, so they are architecture-agnostic *except* for the
  same `224`/`255` constants.

With `strict=True`, export and parity fail loudly when the rebuild is wrong,
which is the good case. The notebook fails with a misleading "checkpoint has no
classifier.1.weight", and `final.pt` (no metadata, no classes) would fail even
later. This metadata gap — not `build_model` — is what makes a second
architecture expensive today.

## 4. Two ways forward

### Option A — fork the script

Copy `train_efficientnet.py` to `train_resnet.py` and edit the six sites. Around
40–60 changed lines. Cheap today, but it duplicates 1,800 lines that carry hard
won behaviour (AMP fallback, sampler audit, DVC identity, MLflow fail-soft,
early stop). Every future fix must be applied twice, and the export path still
has to become architecture-aware, so the fork does not avoid the real work.

### Option B — a small architecture registry (recommended)

Keep one training script and put the per-architecture facts in one table. The
surface needed is exactly five facts per backbone: factory, pretrained weights,
how to find/replace the head `Linear`, its `in_features`, and (for export only)
where the last feature map comes from and whether the linear-head Grad-CAM
identity holds.

```python
# scripts/architectures.py  (torch/torchvision only, imported lazily by tests)
@dataclass(frozen=True)
class ArchSpec:
    name: str                          # "efficientnet_b0", "resnet50", ...
    factory: Callable[..., nn.Module]
    weights: Any                       # torchvision Weights enum or None
    in_features: int                   # head input width
    head_attr: str                     # "classifier" or "fc"
    head_index: Optional[int]          # 1 / 3 / None for a bare Linear
    feature_channels: int              # channels of the last feature map
    feature_attr: str = "features"     # module that emits the last feature map
    feature_mode: str = "module"       # "module" or "hook"
    gradcam_linear_head: bool          # False for mobilenet_v3_*
```

`feature_mode="hook"` runs the full model and captures `feature_attr`'s output
with a forward hook, which is what ResNet needs: `layer4` consumes the previous
stage's 512-channel map, so it cannot be called on an image. The hook is removed
in a `finally` block, and `extract_features` pools with
`F.adaptive_avg_pool2d` for both modes.

with three helpers that replace every hardcoded attribute access:

```python
def get_head(model, spec) -> nn.Linear          # classifier[i] or fc
def head_parameters(model, spec) -> list[nn.Parameter]
def backbone_parameters(model, spec) -> list[nn.Parameter]   # everything but the head
```

`backbone_parameters` is `[p for p in model.parameters() if id(p) not in
head_ids]`, which removes the `.features` dependency entirely — the warmup
freeze becomes "freeze all, unfreeze head" instead of "freeze `.features`".

Then the edits are:

1. `--arch` in `parse_args` (default `efficientnet_b0`), resolved to an
   `ArchSpec`. `--img-size` default derived from the spec.
2. `build_model(spec, num_classes, ...)`: call `spec.factory(weights=spec.weights)`,
   `get_head(...).in_features == spec.in_features` (a real guard now), replace
   with `nn.Linear(spec.in_features, num_classes)`.
3. Warmup: freeze `backbone_parameters`, optimizer over `head_parameters`.
4. Fine-tune: head vs rest split through the same helper (the existing
   `decay`/`no_decay` split stays untouched).
5. `save_checkpoint`: add `arch`, `arch_weights`, `img_size`, `feature_channels`,
   `head_attr`/`head_index`, `torchvision.__version__`. Old checkpoints without
   the key fall back to `efficientnet_b0` with a printed warning.
6. Naming: `run_name`, the Drive base directory, the CLI description and the
   MLflow experiment default become `spec.name`-derived.

Cost: ~120–180 lines added to the training script, ~40 lines of new module,
zero change to the algorithmic core.

## 5. Downstream work, file by file

| File | Change | Blocking? |
| --- | --- | --- |
| `scripts/train_efficientnet.py` | the six sites above + checkpoint metadata | yes |
| `scripts/architectures.py` (new) | the registry and head helpers | yes |
| `scripts/export_onnx.py` | rebuild from `ckpt["arch"]`; replace the hardcoded `features`/`avgpool`/`classifier` in `GradCamExportModel` with a spec-driven extractor; derive `FEATURES_SHAPE` and the dummy input from `ckpt["img_size"]`; per-family Grad-CAM | yes, for the app |
| `scripts/check_onnx_parity.py` | rebuild through the registry; constants from the checkpoint | yes, gate |
| `runs/Avaliacao_Cruzada_Lixo.ipynb` cell 8 | rebuild through the registry; drop `head_key = "classifier.1.weight"`; `IMG_SIZE_FALLBACK` from `run_config` | yes, for the matrix |
| `scripts/calibrate_threshold.py`, `make_golden_fixtures.py` | only the `224`/`255` constants become parameters | no |
| `app/` (Kotlin) | **none** if the ONNX output names and shapes are preserved: `ModelLoader.kt:54` requires the three names, `TrashClassifier.kt:82` derives channels from the graph, and `GradCam.kt:22 FEATURE_SIZE` is dead code. `RgbImage.INPUT_SIZE` is 224. | only if `img_size` changes |
| `app/README.md`, `docs/adr/0001`, `README.md`, `scripts/usage.md` | model-contract text ("EfficientNet-B0", "16 MB", "7×7×1280") | no |
| `scripts/tune_efficientnet.py` | `--arch` passthrough; study name; per-arch search ranges (see §6) | no |
| `scripts/tracking.py` | `DEFAULT_EXPERIMENT = "trash-efficientnet"` and tags | no |
| `tests/` | add a CPU test that, for every registered spec, `weights=None` builds and the head is replaced with the right width, and that checkpoint metadata round-trips | recommended |

`dvc.yaml` needs no change — training is deliberately outside the DVC graph. If
architecture sweeps should be reproducible, add `train.arch` to `params.yaml`
and record it in `run_config.json` (it lands there automatically once it is a
CLI argument).

## 6. Hyperparameters that are not transferable

- **Learning rates.** `--lr 3e-4`, `--head-lr 1e-3` were tuned for B0 (via the
  Optuna study). ResNet50 with ImageNet-1k V2 weights typically wants a lower
  backbone lr; keep the CLI but re-tune, do not reuse silently.
- **Batch size.** `auto_batch_size` (271–301) maps VRAM to 512/256/128/64 from
  the B0 memory profile (5.3M params). ResNet50 is ~5× the parameters (25.6M)
  and ~10× the FLOPs, so a 512 default on an A100 will OOM at 224. Either scale
  the table by a per-spec
  factor or require `--batch-size` for the new arch.
- **Warmup semantics.** For a ResNet the "head" is one randomly initialised
  `Linear`; for MobileNetV3 it is two (a 960→1280 projection plus the
  classifier), so the frozen stage trains ~1.2M parameters instead of ~8k. It
  still converges, but the stage is no longer "linear probe only" — worth
  saying in the paper.
- **BatchNorm.** All candidates are BN networks, so `set_bn_eval` and
  `--freeze-bn` keep working. With a frozen backbone, BN running statistics
  still update unless `--freeze-bn` is passed. That is current behaviour for B0
  too, but it matters more for BN-heavy MobileNets; consider making
  `--freeze-bn` the default during warmup.
- **Resolution.** Keep 224 for the first comparison so the feature map is 7×7
  and the app/export contract is unchanged. If 256 is kept in the tuner, every
  hardcoded 7 has to be derived.

## 7. Suggested landing order

1. `architectures.py` + the six training-script sites + checkpoint metadata.
   Verify with `--arch efficientnet_b0`: the run must reproduce an existing
   `run_config.json` bit-for-bit in the metric columns.
2. MobileNetV2 (contract-compatible). Full export → parity → golden fixtures →
   app smoke test, no Kotlin change.
3. ResNet18 or ResNet50. Same pipeline; only the registry entry plus the
   feature-extractor hook in the export wrapper.
4. MobileNetV3-Large only if the Grad-CAM story is worth changing: the app
   would need the two head matrices and the pooled vector to compute the
   effective per-channel weight, or the heatmap has to be dropped for that
   model.
5. Then the cross-eval notebook and the docs.

## 8. Verification checklist per new backbone

```bash
# 1. two-epoch smoke on a small split (CPU or GPU)
python3 scripts/train_efficientnet.py --arch mobilenet_v2 \
    --data-dir data/roboflow_convertido_split \
    --epochs 2 --warmup-epochs 1 --run-name smoke_mbv2 --no-test

# 2. the algorithmic core is untouched
python3 -m pytest tests/ -q

# 3. graph + app contract
python3 scripts/export_onnx.py --checkpoint runs/smoke_mbv2/best.pt
python3 scripts/check_onnx_parity.py --checkpoint runs/smoke_mbv2/best.pt
python3 scripts/make_golden_fixtures.py
# Android: gradlew assembleDebug + one inference + Grad-CAM

# 4. no silent model substitution: export must refuse a checkpoint whose
#    recorded arch differs from the one requested
```

The last point is the one that pays for the metadata work: today a wrong rebuild
is only caught by `strict=True`; after the change the loader can refuse by name.

## 9. Decisions (resolved 2026-11-19)

- MobileNet **V2**, not V3: drop-in head slot, identical 1280x7x7 feature map
  and an exact linear-head Grad-CAM identity.
- **Preserve the on-device contract**: same ONNX output names, same
  `[1, 1280, 7, 7]` feature shape, so the Kotlin app is unchanged.
- `img_size` stays **224** across architectures for a controlled comparison
  (the export/parity scripts now derive the feature shape from the checkpoint
  instead of hardcoding 7, so a 256 run fails loudly rather than silently).
- Weights: `MobileNet_V2_Weights.IMAGENET1K_V2`; the enum name is recorded in
  the checkpoint as `arch_weights`.
- Optuna search ranges stay the B0 ones for now; `--arch` only routes the trial
  into a per-backbone study (`runs/optuna_<arch>/`, study name
  `<arch>_macro_f1`). Re-tuning the ranges is future work.
- Keep the filename `train_efficientnet.py`; the backbone is the `--arch` flag.
- ResNet is deferred in this decision. It needs the feature-extractor hook and
  the head-as-`fc` entry described in §4; the registry already models both
  fields, so adding it is an entry plus that hook. *(Superseded: ResNet18 landed
  later the same day — see §10.)*

## 10. Implementation status (2026-11-19)

| Piece | State |
| --- | --- |
| `scripts/architectures.py` | new: `ArchSpec`, `ARCHS` (`efficientnet_b0`, `mobilenet_v2`, `resnet18`), head/backbone parameter helpers, `build_model`, `extract_features` (direct module call or forward hook), checkpoint metadata and `arch_from_checkpoint` (missing `arch` -> B0 with a warning) |
| `scripts/train_efficientnet.py` | `--arch`; `build_model`, warmup freeze, fine-tune param groups and `save_checkpoint` go through the registry; checkpoints add `arch`, `arch_weights`, `img_size`, `feature_channels`, `head_attr`, `head_index`, `torchvision`; run name, Drive base and MLflow tag follow the arch. No ResNet-specific branch is needed: warmup freezes `backbone_parameters` (everything but `fc`) |
| `scripts/export_onnx.py` | rebuilds from `ckpt["arch"]` / `ckpt["img_size"]`; the feature map comes from `F.adaptive_avg_pool2d` (MobileNetV2 has no `.avgpool`) and from `architectures.extract_features` (ResNet captures `layer4` with a hook); expected feature shape is derived from the model, and a non-checkpoint payload (bare `final.pt`) is refused by name |
| `scripts/check_onnx_parity.py` | rebuilds through the registry, derives the eval transform from the checkpoint resolution, and checks the ONNX input shape against it |
| `runs/Avaliacao_Cruzada_Lixo.ipynb` | cell 8 rebuilds from `ckpt["arch"]` (falling back to `run_config.json`), so a mixed run set cannot silently rebuild B0 for an MBv2/ResNet18 checkpoint |
| `scripts/tune_efficientnet.py` | `--arch` passthrough, per-arch study directory and study name |
| `tests/test_architectures.py` | CPU tests: every spec builds with `weights=None`, head width, feature channels, parameter partition, checkpoint round-trip, legacy checkpoint -> B0, wrong rebuild fails `strict=True`, Grad-CAM wrapper matches the classifier (the MobileNetV2 `.avgpool` trap), and the ResNet `layer4` hook captures the same map as a manual forward while leaving no hook behind |
| Verification done | `pytest tests/` (222 passed; one pre-existing `test_content_index` byte-identity failure unrelated to backbones), a two-epoch CPU smoke for ResNet18 at 64 px, and ONNX export + parity + Grad-CAM checks on a 224 px untrained ResNet18 checkpoint and on the smoke checkpoint (`features` derived as `512x7x7` and `512x2x2` respectively) |
| Not done | ResNet50, MobileNetV3, full 50-epoch runs (no GPU on this host), re-tuning the B0 learning-rate/batch-size ranges for ResNet |
