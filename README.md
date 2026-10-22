# trash

This repository builds a reproducible image-classification dataset from four
waste-image sources and trains an EfficientNetB0 model on it. The dataset side
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

## Training

Train on the merged dataset:

```bash
python3 scripts/train_efficientnet.py --data-dir data/merged_dataset
```

Or train on any single source projection:

```bash
python3 scripts/train_efficientnet.py --data-dir data/trashbox_split
```

The imbalance sampler, its `--sampler-alpha` guidance and the per-dataset
ratios are documented in [scripts/usage.md](scripts/usage.md).

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

- `scripts/`: pipeline, verification and training code.
- `tests/`: pytest coverage for the index, registry, splits, merge, verifier
  and training helpers.
- `reports/`: generated diagnostics and historical duplicate reports.
- `dvc.yaml`: the five-stage pipeline definition.
- `params.yaml`: seeds and split ratios.
