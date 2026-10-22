#!/usr/bin/env python3
"""Remove md5-identical duplicate images from the four source datasets.

This script is no longer part of the dataset pipeline. The content index plus
the global split registry replaced manual duplicate removal and cross-dataset
quarantine, and this script is retained only to reproduce historical reports.

Consumes ``exact_duplicates_by_dataset.csv`` (written by
analyze_duplicate_images.py) and groups its rows by ``(dataset, md5)``.

Selection rules
---------------
* CROSS-CLASS group (more than one distinct ``cls`` in the group):
  with ``--cross-class remove-all`` (default) EVERY file in the group is
  removed and no file is kept.  With ``--cross-class keep-one-per-class``
  the shortest file per ``(dataset, cls, md5)`` is kept and the rest go.
* SAME-CLASS group: exactly one file is kept and the rest are removed.
  The keeper is chosen with ``--keep-rule``:
    shortest       smallest basename length, ties by full path (default)
    lexicographic  smallest full path
    keep-flag      the single row with ``keep == "yes"``; if absent or
                   ambiguous, fall back to ``shortest``

Safety
------
* Every CSV path and ``--data-root`` are resolved with ``os.path.realpath``.
  A path outside ``--data-root`` aborts the run before anything is touched.
* Paths inside ``data/merged_dataset/`` or equal to ``data/merged_dataset.zip``
  also abort the run.
* Only paths present in the CSV are ever acted on.
* Every candidate is re-hashed immediately before the move; a mismatch is
  skipped (``skipped_md5_mismatch``) and the run continues.
* Quarantine is the default.  Files are moved to
  ``<data-root>/_duplicates_quarantine/<path relative to data-root>`` and an
  existing destination is never overwritten.  ``--delete`` hard-deletes
  instead.
* The default is a dry run; ``--apply`` is required to move or delete.

Usage
-----
    python3 scripts/remove_exact_duplicates.py --dry-run
    python3 scripts/remove_exact_duplicates.py --apply
    python3 scripts/remove_exact_duplicates.py --apply --keep-rule keep-flag
    python3 scripts/remove_exact_duplicates.py --apply --delete
"""
import argparse
import collections
import csv
import hashlib
import os
import shutil
import sys

# Repository root = parent of this script's directory.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset name -> sub-directory (relative to --data-root).
DEFAULT_DATASETS = {
    "kaggle_dataset": "kaggle_dataset/convertido",
    "roboflow_convertido": "roboflow_convertido/merged",
    "trashbox": "trashbox/TrashBox_train_set",
    "trashnet": "trashnet/dataset-resized",
}

MANIFEST_FIELDS = [
    "dataset", "md5", "dup_group_id", "cls", "reason", "action",
    "kept_path", "removed_path", "size_bytes",
]

REMOVED_ACTIONS = {"quarantined", "deleted", "planned"}


def md5_file(path):
    """Streaming md5 of a file."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_inside(path, root):
    """True when ``path`` is ``root`` or below it (both already real paths)."""
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives on Windows
        return False


def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def validate(rows, data_root, datasets):
    """Return a list of human-readable errors; an empty list means safe."""
    errors = []
    merged_dir = os.path.realpath(os.path.join(data_root, "merged_dataset"))
    merged_zip = os.path.realpath(os.path.join(data_root, "merged_dataset.zip"))

    unknown = sorted({r["dataset"] for r in rows if r["dataset"] not in datasets})
    for ds in unknown:
        errors.append(f"unknown dataset {ds!r} (known: {', '.join(sorted(datasets))})")

    for r in rows:
        path = os.path.realpath(r["path"])
        if not is_inside(path, data_root):
            errors.append(f"path outside data-root: {r['path']}")
        elif path == merged_zip or path == merged_dir or is_inside(path, merged_dir):
            errors.append(f"path inside merged dataset: {r['path']}")

    return sorted(set(errors))


def pick_keeper(rows, keep_rule):
    if keep_rule == "keep-flag":
        flagged = [r for r in rows if r["keep"].strip().lower() == "yes"]
        if len(flagged) == 1:
            return flagged[0]
    if keep_rule == "lexicographic":
        return min(rows, key=lambda r: r["path"])
    return min(rows, key=lambda r: (len(os.path.basename(r["path"])), r["path"]))


def build_actions(rows, keep_rule, cross_class):
    """Return removal actions as dicts with row, kept_path and reason."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["md5"])].append(r)

    actions = []
    for key in groups:
        group = groups[key]
        classes = {r["cls"] for r in group}
        if len(classes) > 1:
            if cross_class == "remove-all":
                for r in group:
                    actions.append({"row": r, "kept_path": "",
                                    "reason": "cross_class_remove_all"})
                continue
            for cls in sorted(classes):
                per_class = [r for r in group if r["cls"] == cls]
                keeper = pick_keeper(per_class, keep_rule)
                for r in per_class:
                    if r["path"] != keeper["path"]:
                        actions.append({"row": r, "kept_path": keeper["path"],
                                        "reason": "cross_class_remove_all"})
            continue

        keeper = pick_keeper(group, keep_rule)
        for r in group:
            if r["path"] != keeper["path"]:
                actions.append({"row": r, "kept_path": keeper["path"],
                                "reason": "same_class_redundant"})
    return actions


def count_before(data_root, datasets):
    """Files currently present per (dataset, class) directory."""
    counts = collections.Counter()
    for ds, rel in datasets.items():
        base = os.path.realpath(os.path.join(data_root, rel))
        if not os.path.isdir(base):
            continue
        for cls in sorted(os.listdir(base)):
            cdir = os.path.join(base, cls)
            if not os.path.isdir(cdir):
                continue
            counts[(ds, cls)] = sum(
                1 for entry in os.scandir(cdir) if entry.is_file())
    return counts


def quarantine_destination(src, data_root, quarantine_dir, expected_md5):
    """Return (destination, action) without ever overwriting a file."""
    rel = os.path.relpath(src, data_root)
    dest = os.path.join(quarantine_dir, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if not os.path.exists(dest):
        return dest, "quarantined"
    if md5_file(dest) == expected_md5:
        return dest, "already_quarantined"

    base, ext = os.path.splitext(dest)
    n = 1
    while True:
        candidate = f"{base}.dup{n}{ext}"
        if not os.path.exists(candidate):
            return candidate, "quarantined"
        if md5_file(candidate) == expected_md5:
            return candidate, "already_quarantined"
        n += 1


def execute(action, args):
    """Perform one removal; return (action_name, size_bytes)."""
    row = action["row"]
    src = os.path.realpath(row["path"])
    if not os.path.exists(src):
        return "skipped_missing", ""
    size = os.path.getsize(src)
    if md5_file(src) != row["md5"]:
        return "skipped_md5_mismatch", size
    if not args.apply:
        return "planned", size
    if args.delete:
        os.remove(src)
        return "deleted", size

    dest, result = quarantine_destination(
        src, args.data_root, args.quarantine_dir, row["md5"])
    if result == "quarantined":
        shutil.move(src, dest)
    return result, size


def manifest_row(action, result, size):
    row = action["row"]
    return {
        "dataset": row["dataset"],
        "md5": row["md5"],
        "dup_group_id": row["dup_group_id"],
        "cls": row["cls"],
        "reason": action["reason"],
        "action": result,
        "kept_path": action["kept_path"],
        "removed_path": row["path"],
        "size_bytes": size,
    }


def parse_args(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv",
                    default=os.path.join(REPO, "reports", "duplicates",
                                         "exact_duplicates_by_dataset.csv"),
                    help="duplicate CSV (relative paths resolve from the repo root)")
    ap.add_argument("--data-root", default=os.path.join(REPO, "data"),
                    help="root dir containing the source datasets")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="report planned removals without touching files (default)")
    mode.add_argument("--apply", action="store_true",
                      help="actually move/delete files and write the manifest")
    ap.add_argument("--keep-rule", choices=["shortest", "lexicographic", "keep-flag"],
                    default="shortest")
    ap.add_argument("--quarantine-dir", default=None,
                    help="default: <data-root>/_duplicates_quarantine")
    ap.add_argument("--delete", action="store_true",
                    help="hard-delete instead of quarantining (quarantine is default)")
    ap.add_argument("--manifest",
                    default=os.path.join(REPO, "reports", "duplicates",
                                         "removed_files.csv"),
                    help="manifest written only on --apply")
    ap.add_argument("--cross-class", choices=["remove-all", "keep-one-per-class"],
                    default="remove-all",
                    help="cross-class policy (default: remove every copy)")
    args = ap.parse_args(argv)

    if not os.path.isabs(args.csv):
        args.csv = os.path.join(REPO, args.csv)
    args.data_root = os.path.realpath(args.data_root)
    if args.quarantine_dir is None:
        args.quarantine_dir = os.path.join(args.data_root, "_duplicates_quarantine")
    args.quarantine_dir = os.path.realpath(args.quarantine_dir)
    args.apply = bool(args.apply)  # dry run is the default
    return args


def print_summary(before, removed, datasets):
    classes = set(before) | set(removed)
    for ds in [d for d in datasets if any(c[0] == d for c in classes)]:
        for key in sorted(c for c in classes if c[0] == ds):
            b = before.get(key, 0)
            r = removed.get(key, 0)
            print(f"  {ds}/{key[1]}: before={b} to_remove={r} after={b - r}")
    total_before = sum(before.values())
    total_removed = sum(removed.values())
    print(f"TOTAL: before={total_before} to_remove={total_removed} "
          f"after={total_before - total_removed}")


def main(argv=None):
    args = parse_args(argv)
    print(
        "[aviso] remove_exact_duplicates.py está supersedido: o índice de "
        "conteúdo + registro global substituem a quarentena manual; use dvc repro.",
        file=sys.stderr,
    )

    try:
        rows = load_rows(args.csv)
    except OSError as exc:
        print(f"error: cannot read CSV {args.csv}: {exc}", file=sys.stderr)
        return 2

    errors = validate(rows, args.data_root, DEFAULT_DATASETS)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        print("aborted before touching any file", file=sys.stderr)
        return 2

    actions = build_actions(rows, args.keep_rule, args.cross_class)
    before = count_before(args.data_root, DEFAULT_DATASETS)
    removed = collections.Counter()
    outcomes = collections.Counter()
    manifest = None
    writer = None
    if args.apply:
        os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
        manifest = open(args.manifest, "w", newline="")
        writer = csv.DictWriter(manifest, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()

    try:
        for action in actions:
            result, size = execute(action, args)
            outcomes[result] += 1
            if result in REMOVED_ACTIONS:
                row = action["row"]
                removed[(row["dataset"], row["cls"])] += 1
            if writer is not None:
                writer.writerow(manifest_row(action, result, size))
    finally:
        if manifest is not None:
            manifest.close()

    print_summary(before, removed, DEFAULT_DATASETS)
    if args.apply:
        print(f"apply: quarantined={outcomes['quarantined']} "
              f"deleted={outcomes['deleted']} "
              f"already_quarantined={outcomes['already_quarantined']} "
              f"skipped_missing={outcomes['skipped_missing']} "
              f"skipped_md5_mismatch={outcomes['skipped_md5_mismatch']}")
        print(f"manifest -> {args.manifest}")
    else:
        print(f"dry-run: {outcomes['planned']} planned removals "
              f"(skipped_missing={outcomes['skipped_missing']}, "
              f"skipped_md5_mismatch={outcomes['skipped_md5_mismatch']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
