
#!/usr/bin/env python3
"""Hyperparameter optimization for EfficientNetB0 using Optuna."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import optuna
import torch

import train_efficientnet as trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize EfficientNetB0 hyperparameters with Optuna."
    )

    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--study-dir",
        type=str,
        default="runs/optuna_efficientnet"
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
    ]

    if args.data_dir:
        train_args.extend(["--data-dir", args.data_dir])

    try:
        # 4. Run the original EfficientNet training pipeline

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


def main():

    args = parse_args()

    study_dir = Path(args.study_dir)
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
        study_name="efficientnet_b0_macro_f1",
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )

    print("\nStarting EfficientNetB0 optimization...")
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
