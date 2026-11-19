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
and `scripts/metrics.py`, plus `scripts/tracking.py` for MLflow and
`scripts/architectures.py` for the backbone registry. When uploading to Colab,
keep all five files together.

### Switch backbone

All three architectures come from the registry in `scripts/architectures.py`:

```bash
# EfficientNet-B0 (default; reproduces the existing runs)
python3 scripts/train_efficientnet.py --data-dir data/merged_dataset

# MobileNetV2 — same training code, same 224 px input, same Grad-CAM contract
python3 scripts/train_efficientnet.py --data-dir data/merged_dataset --arch mobilenet_v2

# ResNet18 — 512x7x7 feature map, head is `fc`, layer4 captured by a forward hook
python3 scripts/train_efficientnet.py --data-dir data/merged_dataset --arch resnet18
```

`--arch` changes only the backbone: the two-stage schedule, the sampler, the
losses and the artifact set are shared, so the comparison is controlled.
MobileNetV2 uses `IMAGENET1K_V2` weights; ResNet18 uses `IMAGENET1K_V1`. Every
checkpoint records `arch`, `arch_weights`, `img_size`, `feature_channels` and
the head location; `export_onnx.py`, `check_onnx_parity.py` and the cross-eval
notebook rebuild from that record instead of assuming B0. Checkpoints written
before the registry carry no `arch` and are read as EfficientNet-B0 with a
printed warning. Run names and the Drive base follow the arch
(`trash_efficientnet_b0/`, `trash_mobilenet_v2/`, `trash_resnet18/`), and
`tune_efficientnet.py --arch resnet18` writes its own Optuna study
(`runs/optuna_resnet18/{study.db,best_params.json}`) so trials from different
backbones are never mixed. The Optuna search space is still the one tuned for
B0; re-tune it before trusting ResNet18's learning rates or batch size.

### Run tracking (MLflow)

Each training run can be mirrored to the project's MLflow server,
`http://192.168.15.2:5000`, whose artifact store is SeaweedFS. The default is
**auto**: tracking starts when the server answers `/health` and is skipped
otherwise, so a Colab session without the VPN runs exactly as before.

```bash
# auto (default): track when the server is reachable
python3 scripts/train_efficientnet.py --data-dir data/merged_dataset

# force / disable / point somewhere else
python3 scripts/train_efficientnet.py --data-dir data/merged_dataset --mlflow
python3 scripts/train_efficientnet.py --data-dir data/merged_dataset --no-mlflow
MLFLOW_TRACKING_URI=http://10.0.0.5:5000 python3 scripts/train_efficientnet.py ...
```

A run records the hyperparameters (all CLI flags plus the resolved batch size,
worker count and class list), one metric point per epoch (`train_loss`,
`val_loss`, `val_macro_f1`, `val_balanced_acc`, per-class `val_f1_*` and
`val_recall_*`, learning rates), the final `test_*` metrics, the run
directory's artifacts under `outputs/`, the checkpoints under `checkpoints/`,
`dvc.lock`/`dvc.yaml`/`params.yaml` under `reproducibility/`, and the model
itself with a tensor signature.

The dataset identity is the DVC hash, not the path: `data.dvc.hash` is the
`.dir` md5 from `dvc.lock`, `data.dvc.lock_md5` pins the whole pipeline,
`data.git.commit` pins the code, and `data.verified` says whether the tree on
disk still matches the lock's file count and byte size. See
[docs/adr/0003-run-tracking-with-mlflow.md](../docs/adr/0003-run-tracking-with-mlflow.md)
for why the lock is the authority.

Checkpoints are 16 MB each; `--no-mlflow-log-checkpoints` and
`--no-mlflow-log-model` skip them (the Optuna driver already does, since trial
weights are throwaway). To publish the model in the registry, add
`--mlflow-registered-model trash-efficientnet-b0`.

Tracking needs the MLflow client (`pip install mlflow`, pinned in
`requirements.txt` at the server's version); without it the run prints one
warning and stays local. Tracking is fail-soft: a dead server prints one
warning and the run continues to completion, with the local run directory
unchanged.

#### Tracking from Colab

Colab cannot route to `192.168.15.2`, so the default auto mode prints
`[mlflow] Tracking disabled: no server answered ...` after a 2 s probe and the
run stays local — nothing else changes. To actually log from Colab:

1. **Push the code.** `train_efficientnet.py` imports `scripts/tracking.py`
   and `scripts/architectures.py`; a clone from before either file existed
   fails at import. The Cell 3 check of all five files catches this.
2. **Install the client** in a cell; do not rely on Colab's preinstalled set:
   `!pip install -q mlflow==3.18.0`.
3. **Expose the tracking server** on a URL Colab can reach. Running the tunnel
   from the laptop that already sees the server over the VPN is enough, but it
   has to present a `Host` header the server accepts:

   ```bash
   cloudflared tunnel --url http://192.168.15.2:5000 \
       --http-host-header 192.168.15.2:5000
   ```

   MLflow 3 checks `Host` against `--allowed-hosts`, whose default covers
   localhost and the RFC 1918 ranges only. A tunnel hostname
   (`*.trycloudflare.com`, `*.ngrok-free.app`) is answered with
   `Invalid Host header - possible DNS rebinding attack`, while `/health` still
   returns `200` — so the run looks healthy and dies on its first logging
   call. Rewriting the header to the server's own address avoids a server
   restart; the alternative is starting the server with
   `--allowed-hosts "<tunnel-host>"`. Only port 5000 needs exposing: the server
   reports `artifacts_presigned_only=false` and multipart transfers disabled,
   so artifact bytes are proxied through the tracking server and SeaweedFS is
   never contacted by the client.
4. **Point the run at it** and force tracking:
   `%env MLFLOW_TRACKING_URI=https://<tunnel-host>` then add `--mlflow` to the
   training command. Forcing matters behind a tunnel that redirects `/health`
   or demands a password; a `401`/`403` already counts as reachable, so basic
   auth works by also exporting `MLFLOW_TRACKING_USERNAME` and
   `MLFLOW_TRACKING_PASSWORD`.

A public tunnel is unauthenticated — this server reports an empty
`tracking_auth`, so anyone holding the URL can write and delete runs. Keep the
tunnel up only for the training window, or front it with basic auth or a
private overlay network. If the tunnel dies mid-run, three failed calls disable
tracking and training continues.

A starter sequence for a tracked Colab run — assume Cells 1/2 of the Colab
recipe below already ran, so the GPU runtime is up and Drive is mounted:

```python
# Cell A — code, client, tracking target
!git clone -q https://github.com/guinbraga/Projeto-CDIA.git /content/trash
!pip install -q mlflow==3.18.0
%env MLFLOW_TRACKING_URI=https://<tunnel-host>
%env MLFLOW_HTTP_REQUEST_TIMEOUT=60
```

The client's own default is 10 s; raise it because each 16 MB checkpoint has to
cross the tunnel's upstream link. `MLFLOW_HTTP_REQUEST_MAX_RETRIES=2` is left
alone.

```python
# Cell B — start the run detached from the browser tab
!cd /content/trash && nohup python scripts/train_efficientnet.py \
    --data-dir /content/data/merged_dataset \
    --sampler-alpha 0.0 --run-name merged_a0.00 \
    --mlflow --mlflow-log-model \
    > /content/drive/MyDrive/trash_efficientnet_b0/merged_a0.00.log 2>&1 &
```

```python
# Cell C — follow it
!tail -n 20 /content/drive/MyDrive/trash_efficientnet_b0/merged_a0.00.log
```

Detaching matters when the browser is the fragile part — a phone screen
locking, a flaky connection: the cell returns immediately and the kernel keeps
training. Log and run directory live on Drive, so a disconnect costs console
output, not the run. Keep the tab foregrounded anyway: Colab reclaims a
runtime that stays disconnected, and the training process dies with it.

A quick tunnel gets a new hostname on every `cloudflared` restart, so the
`%env` line has to be redone after one; for an unattended multi-hour run use a
named tunnel (`cloudflared tunnel create` with a domain you control) so the URL
is stable.

The **browser** UI is a separate matter: MLflow rejects state-changing browser
requests whose `Origin` is not in `--cors-allowed-origins` (default: localhost
only) with `403 Cross-origin request blocked`. The run list then shows
`Request error / INTERNAL_ERROR` while the runs are perfectly healthy — the
Python client sends no `Origin`, so logging is unaffected. From the phone, open
a run directly (`#/experiments/<id>/runs/<run_id>?workflowType=mlflow`, which
only issues `GET`s) or restart the server with
`--cors-allowed-origins "http://192.168.15.2:5000,https://<tunnel-host>"`.
The experiment page also defaults to the GenAI overview; the runs table lives
under the **Model training** switch.

The DVC hash still works in Colab because the zips keep the directory names:
`/content/data/merged_dataset` matches the lock's `data/merged_dataset` by
basename, and `data.verified` then really compares the zip against the lock —
a stale zip shows `false`.

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

Mounting matters beyond data access: `resolve_output_dir` sends artifacts to `/content/drive/MyDrive/trash_<arch>/` (`trash_efficientnet_b0/` by default, `trash_mobilenet_v2/` for the comparison model) when Drive is mounted, and to `/content/runs` (which dies with the session) when it isn't.

### Cell 3 — get the code (all five module files)

```python
!git clone -q https://github.com/guinbraga/Projeto-CDIA.git /content/trash
!ls /content/trash/scripts/{train_efficientnet,imbalance,metrics,tracking,architectures}.py
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

This validates the module imports (including `tracking.py`), the sampler,
the history schema and the checkpoint write in ~2 minutes. Then the optional but honest check:

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

`/content/drive/MyDrive/trash_<arch>/<run-name>/` gets `best.pt`, `last.pt`, `final.pt`, `history.csv`, `run_config.json` (with the `imbalance` block: α, per-class counts, natural vs effective shares, `train_loss_comparable_to_val_loss`), `class_mapping.json`, `test_metrics.json`, `confusion_matrix.csv/.png`, and `data_version.json` (the DVC hash, lock md5, git commit and live file/byte counts of the input). When MLflow tracking is on, the same files are mirrored to the run plus `best.pt`/`final.pt` under `checkpoints/`.

One last thing: `run_config.json` stores the *resolved* `train_class_counts`, so even if you reuse a `--run-name` the `HistoryWriter` header check (23 cols for kaggle vs 29 for trashbox) will abort loudly rather than append misaligned columns. That's a good guard — just means unique run names per dataset+α.

---

## Hyperparameter Optimization with Optuna

The training pipeline supports hyperparameter optimization using Optuna for
either registered backbone; `--arch` is passed through to the trainer (the
search space is the one tuned for EfficientNet-B0).

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

Each trial is also logged to MLflow (when the tracking server is reachable),
tagged with its `optuna.trial` number and `optuna.study` name. Trials log
params, per-epoch metrics and the small run-directory artifacts, but not
checkpoints or the model: twenty trials would otherwise push roughly 600 MB of
throwaway weights into the artifact store.

Results are saved under `runs/optuna_efficientnet/` for EfficientNet-B0 and
`runs/optuna_<arch>/` for any other backbone:

    runs/optuna_mobilenet_v2/
        study.db
        best_params.json
        trials.csv
        trials/

Each architecture has its own study name (`<arch>_macro_f1`), so trials from
different backbones are never mixed. After optimization, use best_params.json
to configure the final training with train_efficientnet.py.

---