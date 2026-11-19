# Track runs in MLflow, with the DVC hash as the dataset identity

The project now produces dozens of training runs across four source splits and
the merged dataset: different seeds, sampler alphas, image sizes and learning
rates, run locally, in Colab and through Optuna. The run directories under
`runs/` are complete but local, and the Colab ones die with the session.
Comparing two runs meant reading two `history.csv` files side by side and
trusting that the `run_config.json` next to each one described the data that
actually went in. An MLflow tracking server already exists on the LAN
(`192.168.15.2:5000`, artifact store in SeaweedFS), so the missing piece was
the client side: what a run records, and what identifies its input.

Every training run is mirrored there: the hyperparameters, one metric point per
epoch, the final test metrics, the run directory's artifacts, and the model
with a tensor signature. The dataset is identified by DVC, not by a path or a
date: `data.dvc.hash` is the `.dir` md5 that `dvc.lock` records for the exact
output consumed, `data.dvc.lock_md5` pins the whole pipeline state behind that
output, `data.git.commit` pins the code, and `data.verified` reports whether the
tree on disk still matches the lock's file count and byte size. The hash comes
from the lock rather than from re-hashing images here because DVC is already
the authority on what the dataset is: it is the only component that knows about
`.dvcignore`, the cache layout and the pipeline that produced the tree.
Re-hashing 2.6 GB would be slower and could disagree with `dvc status` for
reasons no reader of the run would be able to explain. `data.verified` is
deliberately a cheap check — it catches a moved, stale or partially rebuilt
dataset, not a byte flipped inside one JPEG; content verification is what
`dvc status` and `dvc repro` are for.

Tracking is on by default only when the server answers `/health`, and every
MLflow call is fail-soft: a failure prints one warning, three consecutive
failures switch the run to local-only, and training finishes either way. Colab
is a primary runtime for this project and cannot reach the LAN, so a tracking
outage must not cost a 30-epoch fine-tune. The local run directory remains the
canonical artifact; MLflow mirrors it, so any run stays reproducible without
the server.

The consequence for tuning is that trial runs log their params, metrics and
small artifacts but not checkpoints or the model: twenty trials would otherwise
push roughly 600 MB of throwaway weights into SeaweedFS. The Optuna driver sets
`--no-mlflow-log-checkpoints --no-mlflow-log-model`, tags each run with its
trial number and study name, and the winning configuration is retrained as a
normal run with the full artifact set. Model registration is opt-in
(`--mlflow-registered-model`), so the registry holds promoted models rather
than every experiment.
