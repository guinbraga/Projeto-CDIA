#!/usr/bin/env python3
"""MLflow tracking for training runs, versioned against DVC and git.

What this module is for
-----------------------
``train_efficientnet.py`` already writes a complete local run directory
(``history.csv``, ``run_config.json``, ``test_metrics.json``, checkpoints). That
is enough to reproduce a run you still have on disk, and useless once the disk
is gone or you have thirty runs to compare. This module mirrors a run to an
MLflow tracking server, adding the one thing the local artifacts cannot express
on their own: *which dataset* produced the numbers.

The dataset version is the DVC hash, not a re-hash computed here. ``dvc.lock``
is the pipeline's own record of every output it built, so the ``.dir`` md5 of
``data/merged_dataset`` (or of a per-source split) is the canonical identity of
the training input. Re-hashing 2.6 GB of JPEGs here would be slower, and would
disagree with DVC the moment ``.dvcignore`` or the cache layout matters. The
lock is read, the live tree is compared against the lock's ``nfiles``/``size``,
and both are logged:

- ``data.dvc.hash`` — the ``.dir`` md5 from ``dvc.lock``;
- ``data.dvc.lock_md5`` — md5 of ``dvc.lock`` itself, so the whole pipeline
  state (params, deps, every output) is pinned;
- ``data.git.commit`` / ``data.git.dirty`` — the code that ran;
- ``data.verified`` — whether the tree on disk still matches the lock's file
  count and byte size (``None`` when it cannot be checked).

``data.verified`` is a cheap check, not a content audit: it catches a moved,
stale or partially rebuilt dataset, not a byte flipped inside one image. Full
content verification is what ``dvc status`` is for.

Fail-soft contract
------------------
Tracking must never be able to kill a training run. Every MLflow call goes
through :meth:`MlflowTracker._call`, which downgrades any exception to a printed
warning and, after three consecutive failures, switches the run to local-only
for the rest of the process. That matters twice: on Colab, where the tracking
server is behind a VPN and unreachable, and on a long fine-tune, where the
server may go away mid-run. Callers use ``if tracker.active`` for optional
work and otherwise do not branch on tracking state.

Configuration
-------------
``--mlflow-tracking-uri`` (or ``MLFLOW_TRACKING_URI``) points at the server;
the default is this project's SeaweedFS-backed server, ``192.168.15.2:5000``.
The server's experiments use ``mlflow-artifacts:/`` locations, so artifacts are
proxied through the tracking server and no S3 credentials are needed on the
client.

Timeouts are set with ``setdefault`` before importing MLflow: a dead VPN link
should cost seconds, not minutes of retries. Override the ``MLFLOW_HTTP_*``
environment variables to change that.

This module deliberately has no torch import so it is unit-testable without
torch (``tests/test_tracking.py``), matching ``imbalance.py`` and ``metrics.py``.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib
import inspect
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

DEFAULT_TRACKING_URI = "http://192.168.15.2:5000"
DEFAULT_EXPERIMENT = "trash-efficientnet"

# Files train_efficientnet.py always writes. Logged under ``outputs/`` so the
# run's artifact layout mirrors the local run directory.
SUMMARY_ARTIFACTS = (
    "history.csv",
    "run_config.json",
    "class_mapping.json",
    "test_metrics.json",
    "confusion_matrix.csv",
    "confusion_matrix.png",
    "data_version.json",
)

# Checkpoints are ~16 MB each and optional (--mlflow-log-checkpoints).
CHECKPOINT_ARTIFACTS = ("best.pt", "final.pt")

# Files that pin the pipeline state; logged under ``reproducibility/``.
REPRODUCIBILITY_FILES = ("dvc.lock", "dvc.yaml", "params.yaml")

_KEY_RE = re.compile(r"[^A-Za-z0-9_.-]+")


# --------------------------------------------------------------------------- #
# Git and DVC provenance
# --------------------------------------------------------------------------- #
def _git(repo_root: Path, *args: str) -> Optional[str]:
    """Run a git command, returning stripped stdout or None on failure."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_metadata(repo_root: Path) -> Dict[str, Any]:
    """Return the commit, branch and dirty flag of *repo_root*."""
    commit = _git(repo_root, "rev-parse", "HEAD")
    branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git(repo_root, "status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": None if status is None else bool(status),
    }


def md5_file(path: Path, chunk_size: int = 1 << 20) -> Optional[str]:
    """Return the md5 hex digest of *path*, or None when it cannot be read."""
    digest = hashlib.md5()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _assign_lock_field(item: Dict[str, Any], line: str) -> None:
    """Assign one ``key: value`` line to a dvc.lock out entry."""
    key, _, value = line.partition(":")
    key, value = key.strip(), value.strip().strip("'\"")
    if not key:
        return
    if key in ("size", "nfiles"):
        try:
            item[key] = int(value)
            return
        except ValueError:
            return
    item[key] = value


def _parse_lock_outs(text: str) -> Dict[str, Dict[str, Any]]:
    """Extract ``stage -> {cmd, outs}`` from dvc.lock without a YAML parser.

    Backup for environments without PyYAML. The parser only understands the
    block layout DVC writes: a two-space stage name, a four-space ``outs:``
    section, four-space ``- path:`` items and six-space fields. Anything else
    is ignored, so a DVC format change degrades to "no hash", never to a wrong
    hash.
    """
    stages: Dict[str, Dict[str, Any]] = {}
    stage: Optional[Dict[str, Any]] = None
    section: Optional[str] = None
    current: Optional[Dict[str, Any]] = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 2 and stripped.endswith(":"):
            stage = stages.setdefault(stripped[:-1], {"cmd": None, "outs": []})
            section, current = None, None
        elif indent == 4 and stripped in ("outs:", "deps:"):
            section, current = stripped[:-1], None
        elif stage is not None and indent == 4 and section == "outs":
            if stripped.startswith("- "):
                current = {}
                stage["outs"].append(current)
                _assign_lock_field(current, stripped[2:])
        elif stage is not None and indent == 6 and section == "outs" and current:
            _assign_lock_field(current, stripped)
        elif stage is not None and indent == 4 and stripped.startswith("cmd: "):
            stage["cmd"] = stripped[len("cmd: ") :].strip()
    return stages


def load_dvc_lock(repo_root: Path) -> Optional[Dict[str, Any]]:
    """Return the parsed ``dvc.lock`` of *repo_root*, or None when absent.

    The result is ``{"path", "md5", "stages": {name: {"cmd", "outs": [...]}}}``
    where each out is a dict with ``path``, ``md5``, ``size`` and ``nfiles``.
    """
    lock_path = Path(repo_root) / "dvc.lock"
    if not lock_path.is_file():
        return None
    try:
        text = lock_path.read_text(encoding="utf-8")
    except OSError:
        return None

    stages: Optional[Dict[str, Dict[str, Any]]] = None
    try:
        import yaml  # optional: dvc and mlflow both depend on it

        document = yaml.safe_load(text) or {}
        stages = {}
        for name, stage in (document.get("stages") or {}).items():
            outs = [
                {
                    key: out.get(key)
                    for key in ("path", "md5", "size", "nfiles")
                }
                for out in (stage.get("outs") or [])
                if isinstance(out, dict)
            ]
            stages[name] = {"cmd": stage.get("cmd"), "outs": outs}
    except Exception:  # noqa: BLE001 - fall back to the scanner below
        stages = None

    return {
        "path": "dvc.lock",
        "md5": md5_file(lock_path),
        "stages": stages if stages is not None else _parse_lock_outs(text),
    }


def find_lock_out(
    lock: Dict[str, Any], repo_root: Path, data_root: Path
) -> Optional[Dict[str, Any]]:
    """Locate the dvc.lock output that describes *data_root*.

    Match order: the path relative to the repository (authoritative), then a
    unique basename match (the Colab case, where ``--data-dir`` points at
    ``/content/data/merged_dataset`` extracted from a zip). ``match`` records
    which one applied so the run does not overclaim.
    """
    outs = [
        (stage_name, out)
        for stage_name, stage in (lock.get("stages") or {}).items()
        for out in stage.get("outs") or []
    ]
    root = Path(data_root).resolve()
    repo = Path(repo_root).resolve()
    try:
        relative = root.relative_to(repo).as_posix()
    except ValueError:
        relative = None

    if relative is not None:
        for stage_name, out in outs:
            if str(out.get("path", "")).rstrip("/") == relative:
                return {
                    "stage": stage_name,
                    "out": out,
                    "match": "path",
                    "repo_relative_path": relative,
                }

    basename = root.name
    matches = [
        (stage_name, out)
        for stage_name, out in outs
        if Path(str(out.get("path", ""))).name == basename
    ]
    if len(matches) == 1:
        stage_name, out = matches[0]
        return {
            "stage": stage_name,
            "out": out,
            "match": "basename",
            "repo_relative_path": None,
        }
    return None


# --------------------------------------------------------------------------- #
# Live dataset fingerprint
# --------------------------------------------------------------------------- #
def live_snapshot(data_root: Path) -> Dict[str, Any]:
    """Count files and bytes under *data_root*, split by train/val/test.

    One walk, stat-only: the same cost class as starting a DataLoader worker,
    and it lets :func:`dataset_version` say whether the tree still matches the
    lock without reading 2.6 GB.
    """
    root = Path(data_root)
    splits: Dict[str, Dict[str, int]] = {
        name: {"files": 0, "bytes": 0} for name in ("train", "val", "test")
    }
    other_files, other_bytes = 0, 0
    try:
        entries = sorted(root.rglob("*"))
    except OSError:
        entries = []
    for path in entries:
        try:
            if not path.is_file():
                continue
            size = path.stat().st_size
        except OSError:
            continue
        relative = path.relative_to(root)
        split = relative.parts[0] if len(relative.parts) > 1 else None
        if split in splits:
            splits[split]["files"] += 1
            splits[split]["bytes"] += size
        else:
            other_files += 1
            other_bytes += size
    return {
        "files": sum(s["files"] for s in splits.values()) + other_files,
        "bytes": sum(s["bytes"] for s in splits.values()) + other_bytes,
        "splits": splits,
        "other_files": other_files,
        "other_bytes": other_bytes,
        "manifest_md5": md5_file(root / "manifest.csv"),
    }


def dataset_version(repo_root: Path, data_root: Path) -> Dict[str, Any]:
    """Describe the exact dataset and code state behind a run.

    Returns a JSON-serialisable dict: ``root``, ``repo_relative_path``,
    ``git``, ``live`` and ``dvc`` (with ``hash``, ``lock_md5``, ``stage``,
    ``match`` and ``verified``). ``dvc`` is None when no lock entry describes
    the tree, e.g. for an external evaluation directory.
    """
    repo = Path(repo_root)
    root = Path(data_root)
    snapshot = live_snapshot(root)
    version: Dict[str, Any] = {
        "root": str(root),
        "repo_relative_path": None,
        "git": git_metadata(repo),
        "live": snapshot,
        "dvc": None,
    }
    try:
        version["repo_relative_path"] = (
            root.resolve().relative_to(repo.resolve()).as_posix()
        )
    except ValueError:
        pass

    lock = load_dvc_lock(repo)
    located = find_lock_out(lock, repo, root) if lock else None
    if lock is None or located is None:
        return version

    out = located["out"]
    nfiles, size = out.get("nfiles"), out.get("size")
    verified: Optional[bool] = None
    if nfiles is not None and size is not None:
        verified = snapshot["files"] == nfiles and snapshot["bytes"] == size
    version["dvc"] = {
        "lock": lock["path"],
        "lock_md5": lock["md5"],
        "stage": located["stage"],
        "path": out.get("path"),
        "hash": out.get("md5"),
        "nfiles": nfiles,
        "size": size,
        "match": located["match"],
        "verified": verified,
        "verify": "nfiles+size",
    }
    return version


# --------------------------------------------------------------------------- #
# MLflow value shapes
# --------------------------------------------------------------------------- #
def sanitize_metric_name(name: str) -> str:
    """Return *name* using only the characters MLflow accepts in metric keys."""
    return _KEY_RE.sub("_", str(name)).strip("_") or "metric"


def _sanitize_param_key(name: str) -> str:
    return _KEY_RE.sub("_", str(name)).strip("_") or "param"


def is_number(value: Any) -> bool:
    """Return True for int/float values MLflow can log as a metric."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def flatten_params(payload: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested params into MLflow's scalar-only param values.

    Dicts become dotted keys, lists and other objects become JSON strings, and
    ``None`` entries are dropped (MLflow rejects null params). Booleans stay
    booleans, which MLflow stores and renders correctly.
    """
    flat: Dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, prefix=f"{name}."))
        elif isinstance(value, Path):
            flat[_sanitize_param_key(name)] = str(value)
        elif isinstance(value, (list, tuple)):
            flat[_sanitize_param_key(name)] = json.dumps(value, default=str)
        elif value is None:
            # MLflow rejects null params; an absent key is more honest than "".
            continue
        elif isinstance(value, (bool, int, float, str)):
            flat[_sanitize_param_key(name)] = value
        else:
            flat[_sanitize_param_key(name)] = json.dumps(value, default=str)
    return flat


# --------------------------------------------------------------------------- #
# Server reachability
# --------------------------------------------------------------------------- #
def server_reachable(tracking_uri: str, timeout: float = 2.0) -> bool:
    """Return True when an HTTP tracking server answers ``/health``.

    Non-HTTP URIs (``file:``, ``s3:``, ``sqlite:`` ...) are considered usable
    without a probe; only the network case is worth guarding, because that is
    the one that can hang a Colab run behind an unreachable VPN.
    """
    uri = str(tracking_uri or "").strip()
    if not uri:
        return False
    if not uri.startswith(("http://", "https://")):
        return True
    url = uri.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except urllib.error.HTTPError as exc:
        # 401/403 means a server is there and wants credentials; that is a
        # reachable server, and MLflow will authenticate with the
        # MLFLOW_TRACKING_USERNAME/PASSWORD it is given.
        return exc.code in (401, 403)
    except Exception:  # noqa: BLE001 - any other failure means "not reachable"
        return False


# --------------------------------------------------------------------------- #
# Tracker
# --------------------------------------------------------------------------- #
class MlflowTracker:
    """Fail-soft facade over the MLflow client.

    A tracker is created before training and ended exactly once. ``active`` is
    False when tracking is disabled or the first call failed; every logging
    method is then a no-op, so callers do not need try/except around tracking.
    """

    def __init__(
        self,
        tracking_uri: str = DEFAULT_TRACKING_URI,
        experiment: str = DEFAULT_EXPERIMENT,
        run_name: Optional[str] = None,
        tags: Optional[Mapping[str, Any]] = None,
        enabled: bool = True,
        verbose: bool = True,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        self.run_name = run_name
        self.tags = dict(tags or {})
        self.run_id: Optional[str] = None
        self._enabled = bool(enabled)
        self._active = False
        self._failures = 0
        self._warned: set[str] = set()
        self._verbose = verbose
        self._mlflow: Any = None
        # Set before importing mlflow; its REST client reads these at import.
        # Small JSON calls, so a short timeout bounds the damage of a flaky
        # VPN link to seconds; artifact uploads do not use this timeout.
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")

    # -- lifecycle -------------------------------------------------------- #
    @property
    def active(self) -> bool:
        """True while this tracker is recording to the server."""
        return self._active

    def start(self) -> bool:
        """Open the MLflow run, returning whether tracking is active."""
        if not self._enabled or self._active:
            return self._active
        try:
            mlflow = importlib.import_module("mlflow")
        except Exception as exc:  # noqa: BLE001 - missing or broken install
            self._warn(
                f"mlflow is not importable ({exc}); the run stays local-only."
            )
            self._enabled = False
            return False
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            experiment = mlflow.set_experiment(self.experiment)
            start_kwargs: Dict[str, Any] = {
                "run_name": self.run_name,
                "tags": self.tags or None,
            }
            # ``log_system_metrics`` exists from MLflow 2.8; a Colab image with
            # an older client should still open the run rather than fail.
            if "log_system_metrics" in inspect.signature(mlflow.start_run).parameters:
                start_kwargs["log_system_metrics"] = False
            run = mlflow.start_run(**start_kwargs)
        except Exception as exc:  # noqa: BLE001 - unreachable server, bad auth
            self._warn(
                f"could not start a run on {self.tracking_uri} ({exc}); "
                "training continues without tracking."
            )
            self._enabled = False
            return False
        self._mlflow = mlflow
        self.run_id = run.info.run_id
        self._active = True
        atexit.register(self._end_on_exit)
        if self._verbose:
            print(
                f"[mlflow] run {self.run_id} in experiment "
                f"'{experiment.name}' at {self.tracking_uri}"
            )
        return True

    def end(self, status: str = "FINISHED") -> None:
        """Close the run once; later calls are no-ops."""
        if not self._active:
            return
        self._active = False
        try:
            self._mlflow.end_run(status=status)
            if self._verbose:
                print(f"[mlflow] run {self.run_id} closed as {status}.")
        except Exception as exc:  # noqa: BLE001 - never fail the caller
            self._warn(f"could not close the run as {status}: {exc}")

    def _end_on_exit(self) -> None:
        """Close a run that was left open by a crash."""
        if self._active:
            self.end("FAILED")

    # -- logging ---------------------------------------------------------- #
    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log scalarised params, flattening nested dicts."""
        flattened = flatten_params(params)
        if not flattened:
            return
        self._call("log_params", lambda mlflow: mlflow.log_params(flattened))

    def log_metrics(
        self, metrics: Mapping[str, Any], step: Optional[int] = None
    ) -> None:
        """Log numeric metrics, sanitising per-class names for MLflow."""
        payload = {
            sanitize_metric_name(key): float(value)
            for key, value in metrics.items()
            if is_number(value)
        }
        if not payload:
            return
        if step is None:
            self._call("log_metrics", lambda mlflow: mlflow.log_metrics(payload))
        else:
            self._call(
                "log_metrics", lambda mlflow: mlflow.log_metrics(payload, step=step)
            )

    def log_tags(self, tags: Mapping[str, Any]) -> None:
        """Set run tags."""
        payload = {str(key): str(value) for key, value in tags.items()}
        if not payload:
            return
        self._call("log_tags", lambda mlflow: mlflow.set_tags(payload))

    def log_artifacts(
        self, paths: Iterable[Path], artifact_path: Optional[str] = None
    ) -> None:
        """Log existing files (or directories) under *artifact_path*."""
        existing = [Path(path) for path in paths if Path(path).exists()]
        if not existing:
            return

        def _do(mlflow: Any) -> None:
            for path in existing:
                if path.is_dir():
                    mlflow.log_artifacts(str(path), artifact_path=artifact_path)
                else:
                    mlflow.log_artifact(str(path), artifact_path=artifact_path)

        self._call("log_artifacts", _do)

    def log_reproducibility(self, repo_root: Path) -> None:
        """Log the small files that pin the pipeline and its params."""
        root = Path(repo_root)
        self.log_artifacts(
            [root / name for name in REPRODUCIBILITY_FILES],
            artifact_path="reproducibility",
        )

    def log_model(
        self,
        model: Any,
        input_example: Any = None,
        output_example: Any = None,
        registered_model_name: Optional[str] = None,
        artifact_path: str = "model",
    ) -> None:
        """Log a torch model with a signature, falling back to pickle.

        MLflow 3 defaults to ``torch.export`` (``pt2``); older torch or an
        unsupported graph falls back to pickle so a run still carries its
        weights.
        """

        def _do(mlflow: Any) -> None:
            pytorch = importlib.import_module("mlflow.pytorch")
            models = importlib.import_module("mlflow.models")
            signature = None
            if input_example is not None and output_example is not None:
                signature = models.infer_signature(input_example, output_example)
            common: Dict[str, Any] = {
                "signature": signature,
                "input_example": input_example,
                "registered_model_name": registered_model_name,
            }
            # MLflow 2.x calls the destination ``artifact_path``; 3.x ``name``.
            parameters = getattr(pytorch.log_model, "__wrapped__", pytorch.log_model)
            try:
                parameters_by_name = inspect.signature(parameters).parameters
            except (TypeError, ValueError):
                parameters_by_name = {}
            kwargs = (
                {"name": artifact_path}
                if "name" in parameters_by_name
                else {"artifact_path": artifact_path}
            )
            if "serialization_format" not in parameters_by_name:
                # Old client: whole-module pickling is the only format.
                pytorch.log_model(model, **kwargs, **common)
                return
            try:
                pytorch.log_model(model, serialization_format="pt2", **kwargs, **common)
            except Exception as exc:  # noqa: BLE001 - export can fail on some graphs
                self._warn(f"pt2 export failed ({exc}); retrying as pickle.")
                pytorch.log_model(
                    model, serialization_format="pickle", **kwargs, **common
                )

        self._call("log_model", _do)

    # -- internals -------------------------------------------------------- #
    def _call(self, what: str, action: Any) -> Any:
        """Run an MLflow action, downgrading failures to warnings."""
        if not self._active or self._mlflow is None:
            return None
        try:
            result = action(self._mlflow)
            self._failures = 0
            return result
        except Exception as exc:  # noqa: BLE001 - fail-soft by contract
            self._failures += 1
            self._warn(f"{what} failed: {exc}")
            if self._failures >= 3:
                self._warn(
                    "three consecutive MLflow failures; tracking is disabled for "
                    "the rest of this run."
                )
                self._active = False
                try:
                    self._mlflow.end_run(status="FAILED")
                except Exception:  # noqa: BLE001 - the server is already down
                    pass
            return None

    def _warn(self, message: str) -> None:
        """Print a warning once per distinct message."""
        if message in self._warned:
            return
        self._warned.add(message)
        print(f"[mlflow] WARNING: {message}")
