## Usage

### Splitting datasets
```bash
python3 scripts/split_dataset.py --dry-run              # preview
python3 scripts/split_dataset.py                        # all three
python3 scripts/split_dataset.py --dataset trashbox     # one
python3 scripts/split_dataset.py --source <dir> --output <dir>
```

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
| `data/merged_datasets_unbalanced`| `0.0`            | 2.4x  | Already close to balanced; keep the natural stream. |
| `data/trashbox_split`            | `0.5`            | 5.9x  | Partial balance lifts `cigarette_butt` without starving the large classes. |
| `data/kaggle_dataset_split`      | `0.75`–`1.0`     | 15.1x | `cardboard` is 15x rarer than `plastic`; only a near-uniform stream keeps it learnable. |

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
