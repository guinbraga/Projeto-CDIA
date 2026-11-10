## Usage

### Dataset pipeline
```bash
dvc repro              # rebuild whatever changed
dvc repro verify       # run the isolation gate on its own
dvc status             # show stale stages
```

The four source splits and the merged dataset are projections of the global
split registry, so nothing is split twice or cleaned up by hand after the fact.
Edit `params.yaml` (seeds and ratios) and rerun `dvc repro` to rebuild only the
affected stages.

### Train
Train per dataset with, e.g.:
```bash
python3 scripts/train_efficientnet.py --data-dir data/roboflow_convertido_split
```

`train_efficientnet.py` imports the pure-numpy helpers in `scripts/imbalance.py`
and `scripts/metrics.py`. When uploading to Colab, keep all three files
together.

### Class imbalance (`--sampler-alpha`)
`--sampler-alpha` resamples only the **training** stream, with intensity in
`[0, 1]`. `0.0` keeps the natural shuffled stream, `1.0` makes every class
equally likely per epoch, and `0.5` is a partial balance. Validation and test
loaders are never sampled or shuffled, so the reported metrics still describe
the real evaluation distribution.

Pick the value from the train split's imbalance ratio (largest ÷ smallest
class), printed at startup as `[sampler] imbalance_ratio=...`:

| `--data-dir`                    | `--sampler-alpha` | Ratio | Why |
| ------------------------------- | ----------------- | ----- | --- |
| `data/roboflow_convertido_split` | `0.0`            | 2.2x  | Mild skew; resampling would mostly add sampling noise. |
| `data/merged_dataset`            | `0.0`            | 2.3x  | Per-source balanced; keep the natural stream. |
| `data/trashbox_split`            | `0.5`            | 5.9x  | Partial balance lifts `cigarette_butt` without starving the large classes. |
| `data/kaggle_dataset_split`      | `0.75`–`1.0`     | 15.9x | `cardboard` is 15x rarer than `plastic`; only a near-uniform stream keeps it learnable. |

```bash
python3 scripts/train_efficientnet.py --data-dir data/trashbox_split --sampler-alpha 0.5
python3 scripts/train_efficientnet.py --data-dir data/kaggle_dataset_split \
    --sampler-alpha 0.75 --monitor balanced_acc
```

The startup audit table prints natural vs effective per-class share, and the
same numbers (plus the natural counts) are persisted in `run_config.json`.

Because the stream is resampled, **training loss is not comparable to
validation/test loss** once `alpha > 0`. Judge those runs on the held-out
metrics: `balanced_acc` (mean per-class recall), macro precision/recall,
worst-class F1 and the imbalance ratio appear in the console summary, in
`test_metrics.json` and in `run_config.json`. `--monitor balanced_acc` selects
the best checkpoint by balanced accuracy instead of macro F1, and every epoch's
`history.csv` row carries per-class F1 and recall so a collapsing minority
class is visible while training runs.

## Colab recipe

### Cell 0 — you, locally: make the zips

JPEGs don't compress, so `-0` is faster:

```bash
cd data
zip -0 -qr trashbox_split.zip trashbox_split
zip -0 -qr kaggle_dataset_split.zip kaggle_dataset_split
zip -0 -qr roboflow_convertido_split.zip roboflow_convertido_split
zip -0 -qr merged_dataset.zip merged_dataset
```

Upload all four to `MyDrive/zips/` (~2.1 G + 3.5 G + 191 M + 2.6 G).

The zips are rebuilt from the registry-derived `data/*_split` and
`data/merged_dataset` directories. The committed pipeline (`dvc.yaml` +
`params.yaml`), not the zips, is the source of truth: rerun `dvc repro` and
re-zip after changing a seed or ratio. Do not run `dvc repro` while zipping or
training — it deletes and rewrites those directories. Rebuild and run
`dvc repro verify` first, then zip, then train.

### Cell 1 — Colab: check the runtime

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch; print(torch.__version__, torch.cuda.get_device_name(0))
```

### Cell 2 — mount Drive (must be a cell, not `!python`)

```python
from google.colab import drive
drive.mount('/content/drive')
```

Mounting matters beyond data access: `resolve_output_dir` sends artifacts to `/content/drive/MyDrive/trash_efficientnet_b0/` when Drive is mounted, and to `/content/runs` (which dies with the session) when it isn't.

### Cell 3 — get the code (all three files)

```python
!git clone -q https://github.com/<you>/trash.git /content/trash
!ls /content/trash/scripts/{train_efficientnet,imbalance,metrics}.py
```

### Cell 4 — unzip once per session

```python
!mkdir -p /content/data
!unzip -q -o /content/drive/MyDrive/zips/trashbox_split.zip            -d /content/data
!unzip -q -o /content/drive/MyDrive/zips/kaggle_dataset_split.zip      -d /content/data
!unzip -q -o /content/drive/MyDrive/zips/roboflow_convertido_split.zip -d /content/data
!unzip -q -o /content/drive/MyDrive/zips/merged_dataset.zip            -d /content/data
!ls -d /content/data/*/train
```

### Cell 5 — smoke test first (smallest dataset, 2 epochs)

```python
!python /content/trash/scripts/train_efficientnet.py \
    --data-dir /content/data/roboflow_convertido_split \
    --sampler-alpha 0.0 --epochs 2 --warmup-epochs 1 \
    --run-name smoke_roboflow --no-test
```

This validates the 3-file import, the sampler, the history schema and the checkpoint write in ~2 minutes. Then the optional but honest check:

```python
!cd /content/trash && python -m pytest tests/test_imbalance.py tests/test_metrics.py \
    tests/test_train_sampler.py tests/test_train_criteria.py -q
```

### Cells 6/7/8/9 — the four real runs (one cell each)

```python
# merged: 8 classes, 2.3x, per-source balanced -> production path
!python /content/trash/scripts/train_efficientnet.py \
    --data-dir /content/data/merged_dataset \
    --sampler-alpha 0.0 \
    --run-name merged_a0.00

# kaggle: 15.9x, plastic 50.1% -> strongest correction
!python /content/trash/scripts/train_efficientnet.py \
    --data-dir /content/data/kaggle_dataset_split \
    --sampler-alpha 0.75 \
    --run-name kaggle_a0.75

# trashbox: 5.9x, cigarette_butt 2.9% -> sqrt sampling
!python /content/trash/scripts/train_efficientnet.py \
    --data-dir /content/data/trashbox_split \
    --sampler-alpha 0.5 \
    --run-name trashbox_a0.50

# roboflow: 2.2x -> leave the natural stream alone
!python /content/trash/scripts/train_efficientnet.py \
    --data-dir /content/data/roboflow_convertido_split \
    --sampler-alpha 0.0 \
    --run-name roboflow_a0.00
```

Everything else is the default (`--epochs 30 --warmup-epochs 3 --monitor f1 --patience 5 --amp`). Don't add `--compile` — recompilation overhead on a 30-epoch T4 run buys nothing.

**Use separate cells, not one `&&` chain.** If kaggle OOMs at epoch 20 you want the other three to still run.

If you'd rather skip Cell 4, this also works — but note the distinct target:

```python
!python .../train_efficientnet.py \
    --data-zip /content/drive/MyDrive/zips/kaggle_dataset_split.zip \
    --extract-to /content/data/kaggle \
    --sampler-alpha 0.75 --run-name kaggle_a0.75
```

Only safe because `--extract-to` is unique per dataset. Reusing `/content/data` for every dataset is the trap from #1.

### Cell 10 — verify each run started on the right data

The startup audit is your fingerprint. Check the class count and `imbalance_ratio` before leaving it alone:

| dataset | `[data] classes` | `imbalance_ratio` | expected effective share (natural → effective) |
|---|---|---|---|
| merged `a=0.00` | **8** | **2.31** | natural == effective |
| kaggle `a=0.75` | **6** | **15.87** | `cardboard` ~0.032 → ~0.12 |
| trashbox `a=0.50` | **9** | **5.88** | `cigarette_butt` ~0.029 → ~0.06; `cardboard` ~0.133 → ~0.12 |
| roboflow `a=0.00` | **9** | **2.17** | natural == effective |

For kaggle you should see `plastic 0.5014 → ~0.236`. If `imbalance_ratio` reads 5.88 during the kaggle run, you hit trap #1.

### Cell 11 — compare runs afterwards

```python
import json, glob, pandas as pd
rows = []
for p in sorted(glob.glob("/content/drive/MyDrive/trash_efficientnet_b0/*/run_config.json")):
    c = json.load(open(p)); t = c.get("test_metrics") or {}
    rows.append({"run": p.split("/")[-2],
                 "classes": c["num_classes"],
                 "alpha": c["imbalance"]["sampler_alpha"],
                 "macro_f1": round(t.get("macro_f1", 0), 4),
                 "bal_acc": round(t.get("balanced_acc", 0), 4),
                 "worst": f'{t.get("worst_class")}:{t.get("worst_class_f1", 0):.3f}',
                 "best_epoch": c["best_epoch"]})
pd.DataFrame(rows).sort_values("run")
```

And the per-class view — the minority-collapse check, which is the whole point of α:

```python
for p in sorted(glob.glob("/content/drive/MyDrive/trash_efficientnet_b0/*/run_config.json")):
    c = json.load(open(p)); t = c.get("test_metrics") or {}
    if not t.get("per_class"):
        continue
    print(p.split("/")[-2], {k: round(v["recall"], 3) for k, v in t["per_class"].items()})
```

`history.csv` in each run dir also carries `val_recall_<class>` per epoch, so you can watch `cigarette_butt` / `cardboard` recover mid-run rather than waiting for the test set.

## Artifacts per run

`/content/drive/MyDrive/trash_efficientnet_b0/<run-name>/` gets `best.pt`, `last.pt`, `final.pt`, `history.csv`, `run_config.json` (with the `imbalance` block: α, per-class counts, natural vs effective shares, `train_loss_comparable_to_val_loss`), `class_mapping.json`, `test_metrics.json`, `confusion_matrix.csv/.png`.

One last thing: `run_config.json` stores the *resolved* `train_class_counts`, so even if you reuse a `--run-name` the `HistoryWriter` header check (23 cols for kaggle vs 29 for trashbox) will abort loudly rather than append misaligned columns. That's a good guard — just means unique run names per dataset+α.

---

## Hyperparameter Optimization with Optuna

The EfficientNetB0 training pipeline supports hyperparameter
optimization using Optuna.

Install the optional dependency:

    pip install "optuna>=3.6,<5"

Run the optimization:

    python scripts/tune_efficientnet.py \
        --data-dir data/merged_dataset \
        --n-trials 20 \
        --epochs 12 \
        --warmup-epochs 3

The optimization searches for the best combination of:

- Learning rate
- Classifier learning rate
- Weight decay
- Label smoothing
- Sampler alpha
- Batch size
- Image resolution

Macro F1 on the validation set is used as the optimization objective.

Test evaluation is disabled during optimization.

Results are saved under:

    runs/optuna_efficientnet/
        study.db
        best_params.json
        trials.csv
        trials/

After optimization, use best_params.json to configure
the final training with train_efficientnet.py.

---