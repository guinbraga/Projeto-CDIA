# trash

This repository builds a reproducible image-classification dataset from four
waste-image sources and trains an ImageNet backbone on it — EfficientNet-B0 by
default, with MobileNetV2 and ResNet18 as comparison models, all through the
same script. The dataset side
runs as a five-stage DVC pipeline that indexes image content, assigns every
duplicate cluster to one split, materialises four per-source projections and a
merged training set, then gates the result on a hard isolation check. The
training side consumes those registry-derived directories directly. DVC tracks
the data outputs, so the exact state behind any split or merged dataset is
recoverable from `dvc.lock`.

## Pipeline

The five stages are:

1. `content-index` reads the raw sources and writes `data/content_index.csv`
   plus `reports/content_index/unreadable_images.csv`.
2. `global-split-registry` clusters the content index and writes
   `data/global_split_registry.csv` plus
   `reports/global_split_registry/excluded_ambiguous_clusters.csv`.
3. `source-splits` projects the registry onto each source, writing
   `data/trashnet_split`, `data/kaggle_dataset_split`, `data/trashbox_split`
   and `data/roboflow_convertido_split`.
4. `merge` writes `data/merged_dataset`.
5. `verify` checks `data/merged_dataset/manifest.csv` against a freshly
   computed registry plan (full-union test and per-source quotas) and runs
   `scripts/verify_dataset_isolation.py`. It exits non-zero on a balance or
   isolation violation.

```
raw sources -> content-index -> global-split-registry
                                        |
                                        v
                                source-splits -> data/*_split (x4)
                                        |
                                        v
                                     merge -> data/merged_dataset -> verify
```

Common commands:

```bash
dvc repro              # rebuild whatever changed
dvc repro verify       # run the isolation gate on its own
dvc status             # show stale stages
dvc dag                # show the stage graph
```

A clean `dvc repro --force` rebuilds every dataset output from the raw sources.
Parameters live in `params.yaml`; changing a seed or a ratio and rerunning
`dvc repro` rebuilds only the affected stages.

## Invariants and decisions

### Content identity threshold

Exact duplicates share an md5. Near-duplicates are connected components at
dHash <= 1 AND pHash <= 1, the project's existing thresholds. Every component
gets a `cluster_id`: the lexicographically smallest md5 in the cluster. One
image belongs to one cluster, and one cluster carries exactly one canonical
class, one canonical source and one split.

### Split registry authority

`data/global_split_registry.csv` is the only authority on class and split
membership. The four source splits and the merged dataset are projections of
it, not independent decisions. There is no second shuffle downstream.

### Isolation invariant

No cluster may be in test in any dataset while also in train or val in any
dataset. The verifier checks this in three separate directions:

- source test against source train/val conflicts;
- merged train/val against source test;
- merged test against source train/val.

It also rejects duplicate identities inside the merged manifest and, with
`--registry`, requires every manifest split to equal the registry split.

### Full-union test

The merged test split is the full union of every registry test cluster of a
target class, exactly once. There is no sampling, capping or balancing of test.
This keeps evaluation honest and makes the test set deterministic.

### Per-source train/val balance

Merged train/val is sampled per `(class, source)`: every source contributes the
same quota per class, capped by the smallest source for that class. A cluster
that appears in several sources is credited to the scarcest source, so shared
clusters never inflate one source's quota. `tetra_pack` is join-all.
`--balance-classes` is opt-in and changes only train/val.

### Ambiguous-cluster exclusion

A cluster whose members disagree on class is excluded from the registry instead
of being given a silent label. The measured cost is 31 clusters and 83 images,
80 of them target-class images. They are listed in
`reports/global_split_registry/excluded_ambiguous_clusters.csv`.

### Determinism

Split assignment is deterministic given the seed and `cluster_id`, stratified
by `(class, canonical source)`, using the 70/15/15 sizing rule implemented by
`dataset_io.split_sizes`. The same seed produces a byte-identical registry.
Seeds and ratios are `merge.seed` and `registry.seed` / `registry.val_ratio` /
`registry.test_ratio` in `params.yaml`.

## Current measured state

| Measure | Count |
| --- | --- |
| Indexed images | 35,603 |
| Content clusters | 32,482 |
| Redundant members | 3,121 |
| Ambiguous clusters / images | 31 / 83 (80 target-class) |
| Registry clusters | 32,451 |
| Registry images | 35,520 |
| Registry train / val / test clusters | 22,717 / 4,867 / 4,867 |
| Source split images, total | 35,520 |
| trashnet split | 2,356 |
| kaggle split | 9,854 |
| trashbox split | 14,003 |
| roboflow split | 9,307 |
| Merged dataset | 15,707 |
| Merged train/val + test | 11,495 + 4,212 |

## External evaluation targets

The cross-dataset notebook needs targets that **no run was trained on**. Two
third-party datasets are prepared as test-only trees under
`data/external_eval/<name>/test`:

| Target | Raw images | Kept after dedupe | Removed as overlap | cardboard / metal / paper / plastic |
| --- | ---: | ---: | ---: | --- |
| `realwaste` | 4,752 | 4,752 | 0 | 461 / 790 / 500 / 921 |
| `garbage12` | 15,520 | 12,450 | 3,024 (3,011 clusters) | 403 / 289 / 436 / 281 |

`realwaste` is [RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste)
(CC BY 4.0), photographed in a landfill. It has no content overlap with the
training corpus at all.

`garbage12` is
[Garbage Classification (12 classes)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
(ODbL 1.0 database). About 29% of its non-clothing classes came from the
6-class TrashNet-derived dataset, so 3,024 images were exact or near-duplicates
of the existing corpus (2,543 of them in train/val) and were removed before the
target was written. The three glass colours are collapsed to `glass`, and
`biological` is collapsed to `organic`; the class map is recorded in the
catalog and the report.

Both targets are built by one command:

```bash
# first run only: download and verify the pinned archives, then dedupe
python3 scripts/prepare_external_eval.py --download --overwrite

# rebuild from an already-extracted tree
python3 scripts/prepare_external_eval.py --dataset realwaste --dataset garbage12 --overwrite

# gate: no exact md5, no dHash/pHash near-duplicate, no registry cluster id
python3 scripts/verify_external_eval.py
dvc repro verify   # the training-side gate must still pass too
```

The preparation command hashes every candidate with the project's existing
`md5`/`dHash`/`pHash` primitives and clusters it **together with**
`data/content_index.csv` using the same `dHash <= 1 AND pHash <= 1` rule as the
registry. A candidate whose component contains any already-indexed image is
dropped, so the targets are disjoint from training, validation **and** existing
test splits. One representative is kept per surviving cluster, and a cluster
whose members disagree on the mapped class is dropped as ambiguous. The full
accounting is in `reports/external_eval/<name>/dedupe_report.json` and
`overlap_clusters.csv`.

The external targets are not DVC outputs: the raw archives are large,
third-party and licensed, and the targets are evaluation-only. The catalog
pins each archive URL and SHA-256, so the build is still reproducible. The
`runs/Avaliacao_Cruzada_Lixo.ipynb` notebook discovers them through
`EXTERNAL_DATA_ROOT` and runs `scripts/verify_external_eval.py` before scoring.
For Colab, package each target as `<name>_external.zip` (top-level `<name>/`)
and upload it to the Drive `zips/` folder; also upload `content_index.csv` and
`global_split_registry.csv` there so the notebook can copy them into
`/content/data`.

## Training

Train on the merged dataset:

```bash
python3 scripts/train_efficientnet.py --data-dir data/merged_dataset
```

Or train on any single source projection:

```bash
python3 scripts/train_efficientnet.py --data-dir data/trashbox_split
```

`--arch mobilenet_v2` (or `--arch resnet18`) switches the backbone without
touching the training algorithm; the per-architecture facts live in
`scripts/architectures.py` and every checkpoint records the architecture it was
trained with, so the ONNX export, the parity gate and the cross-eval notebook
rebuild it from the record rather than assuming B0. ResNet18 needs a
feature-extractor hook because its `layer4` consumes an intermediate map, not an
image; the registry models that as `feature_mode="hook"`.

The imbalance sampler, its `--sampler-alpha` guidance, the per-dataset ratios
and the `--arch` details are documented in [scripts/usage.md](scripts/usage.md).

Runs are mirrored to MLflow when the tracking server (`192.168.15.2:5000`,
SeaweedFS artifact store) answers, and stay local-only otherwise. A run carries
its hyperparameters, per-epoch and test metrics, artifacts, the model, and the
dataset's DVC identity: the `.dir` md5 from `dvc.lock`, the lock's own md5, the
git commit, and a live file-count/byte check against the lock
(`data.verified`). The recording logic lives in `scripts/tracking.py`; the
decision and its rationale are in
[docs/adr/0003-run-tracking-with-mlflow.md](docs/adr/0003-run-tracking-with-mlflow.md).
Colab cannot reach the LAN server, so tracking is skipped there by default;
[scripts/usage.md](scripts/usage.md#tracking-from-colab) covers the tunnel,
client install and `--mlflow` steps for logging from Colab.

## Superseded paths

The old workflow shuffled each source independently and then applied a manual
cross-dataset quarantine. The pipeline no longer produces or consumes any of
it:

- `scripts/remove_exact_duplicates.py` is superseded and retained only to
  reproduce historical reports.
- `data/merged_dataset_balanced` and `data/merged_datasets_unbalanced` are
  stale outputs.
- any pre-registry `data/merged_dataset.zip` is stale. If you need a Colab
  archive, rebuild it from the current `data/merged_dataset` as shown in
  `scripts/train_efficientnet.py`.
- `data/._cross_dataset_leak_quarantine` and `data/._duplicates_quarantine` are
  the retired manual quarantine directories.
- `reports/duplicates/` is historical evidence only; see
  [reports/duplicates/SUPERSEDED.md](reports/duplicates/SUPERSEDED.md).

## Repository layout

- `scripts/`: pipeline, verification and training code. The external-target
  builder and its independence gate are `prepare_external_eval.py` and
  `verify_external_eval.py`; `external_eval_catalog.json` pins the downloads.
- `app/`: the Android demo app that serves the trained classifier on a phone,
  plus the ONNX export and its parity gate. See [app/README.md](app/README.md).
- `tests/`: pytest coverage for the index, registry, splits, merge, verifier,
  training helpers and the external-target gate.
- `reports/`: generated diagnostics and historical duplicate reports;
  `reports/external_eval/` carries the per-target dedupe accounting.
- `dvc.yaml`: the five-stage pipeline definition.
- `params.yaml`: seeds and split ratios.
