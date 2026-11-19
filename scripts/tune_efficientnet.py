
#!/usr/bin/env python3
"""Hyperparameter optimization for a registry backbone using Optuna.

The search space is the one tuned for EfficientNet-B0. ``--arch`` passes the
backbone through to ``train_efficientnet.py``; the study name includes the
architecture so MobileNetV2 or ResNet18 trials are not mixed into the B0 study.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import optuna
import torch

import architectures
import train_efficientnet as trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize a registry backbone's hyperparameters with Optuna."
    )

    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--arch",
        choices=architectures.arch_names(),
        default=architectures.DEFAULT_ARCH,
        help="Backbone to tune (passed through to train_efficientnet.py).",
    )
    parser.add_argument(
        "--study-dir",
        type=str,
        default=None,
        help="Optuna study directory (default: runs/optuna_efficientnet for "
        "efficientnet_b0, runs/optuna_<arch> otherwise).",
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)

    return parser.parse_args()


def objective(trial, args, study_dir):

    # 1. Hyperparameter search space

    params = {
        "lr": trial.suggest_float(
            "lr", 1e-5, 5e-4, log=True
        ),

        "head_lr": trial.suggest_float(
            "head_lr", 1e-4, 3e-3, log=True
        ),

        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-6, 1e-3, log=True
        ),

        "label_smoothing": trial.suggest_float(
            "label_smoothing", 0.0, 0.2
        ),

        "sampler_alpha": trial.suggest_float(
            "sampler_alpha", 0.0, 1.0
        ),

        "batch_size": trial.suggest_categorical(
            "batch_size", [16, 32, 64]
        ),

        "img_size": trial.suggest_categorical(
            "img_size", [224, 256]
        ),
    }

    print("\n" + "=" * 65)
    print(f"TRIAL {trial.number}")
    print(f"Parameters: {params}")
    print("=" * 65)

    # 2. Unique output directory for each trial

    run_name = f"trial_{trial.number:04d}"

    # 3. Arguments passed to the original training script

    train_args = [
        "--arch", args.arch,
        "--epochs", str(args.epochs),
        "--warmup-epochs", str(args.warmup_epochs),
        "--batch-size", str(params["batch_size"]),
        "--img-size", str(params["img_size"]),
        "--lr", str(params["lr"]),
        "--head-lr", str(params["head_lr"]),
        "--weight-decay", str(params["weight_decay"]),
        "--label-smoothing", str(params["label_smoothing"]),
        "--sampler-alpha", str(params["sampler_alpha"]),
        "--monitor", "f1",
        "--patience", "0",
        "--seed", str(args.seed),
        "--num-workers", str(args.num_workers),
        "--eval-batch-size", "64",
        "--output-dir", str(study_dir / "trials"),
        "--run-name", run_name,
        "--no-test",
        "--no-save-last",
        # Trial weights are throwaway: log params, metrics and the run
        # directory, but keep 16 MB checkpoints and the model out of MLflow.
        # The winning configuration is retrained as a normal run.
        "--no-mlflow-log-checkpoints",
        "--no-mlflow-log-model",
    ]

    if args.data_dir:
        train_args.extend(["--data-dir", args.data_dir])

    try:
        # 4. Run the training pipeline for the selected backbone

        best_macro_f1 = trainer.main(
            argv=train_args,
            trial=trial
        )

        if best_macro_f1 is None:
            raise RuntimeError(
                "Training did not produce a valid Macro F1."
            )

        print(
            f"\nTrial {trial.number} "
            f"Macro F1: {best_macro_f1:.4f}"
        )

        return best_macro_f1

    except torch.cuda.OutOfMemoryError:
        print(f"[trial {trial.number}] CUDA out of memory.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raise optuna.TrialPruned(
            "Insufficient GPU memory for this configuration."
        )

    finally:
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def default_study_dir(arch: str) -> str:
    """Keep the historical B0 study tree; give every other arch its own."""
    if arch == architectures.DEFAULT_ARCH:
        return "runs/optuna_efficientnet"
    return f"runs/optuna_{arch}"


def main():

    args = parse_args()

    study_dir = Path(args.study_dir or default_study_dir(args.arch))
    study_dir.mkdir(parents=True, exist_ok=True)

    # 5. Persistent study database

    db_path = (study_dir / "study.db").resolve()
    storage = f"sqlite:///{db_path.as_posix()}"

    # 6. Optuna sampler and pruner

    sampler = optuna.samplers.TPESampler(
        seed=args.seed
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
        interval_steps=1
    )

    study = optuna.create_study(
        study_name=f"{args.arch}_macro_f1",
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )

    print(f"\nStarting {args.arch} optimization...")
    print(f"Architecture: {args.arch}")
    print(f"Trials: {args.n_trials}")
    print(f"Epochs per trial: {args.epochs}")
    print(f"Study directory: {study_dir}")

    # 7. Execute optimization sequentially

    study.optimize(
        lambda trial: objective(trial, args, study_dir),
        n_trials=args.n_trials,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=True
    )

    # 8. Save results

    completed_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed_trials:
        print("\nNo completed trials were found.")
        return

    best_params = {
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "params": study.best_params
    }

    with (study_dir / "best_params.json").open("w") as file:
        json.dump(best_params, file, indent=4)

    study.trials_dataframe(
        attrs=("number", "value", "params", "state")
    ).to_csv(
        study_dir / "trials.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("OPTIMIZATION FINISHED")
    print("=" * 65)

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best Macro F1: {study.best_value:.4f}")
    print("\nBest parameters:")

    for name, value in study.best_params.items():
        print(f"{name}: {value}")

    print(f"\nResults saved in: {study_dir}")


if __name__ == "__main__":
    main()
