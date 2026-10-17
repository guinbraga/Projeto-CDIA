#!/usr/bin/env python3
"""Find duplicate images among the source datasets from a hash table.

Consumes the CSV produced by detect_duplicate_images.py and reports:

* exact duplicates   - byte-identical files (same md5)
* perceptual dups    - dHash AND pHash both within NJOINT bits (default 1);
                       catches the same picture resized / recompressed.
                       Threshold 1 is used because it is the last value before
                       transitive chaining collapses unrelated images together
                       (see --sweep to reproduce).

Artifacts written to --out-dir:
    final_report.json               - all statistics
    duplicate_groups.csv            - every duplicate group (exact + perceptual)
    exact_duplicates_by_dataset.csv - intra-dataset md5 groups, per dataset

Usage
-----
    python3 scripts/analyze_duplicate_images.py
    python3 scripts/analyze_duplicate_images.py --sweep
"""
import argparse
import collections
import csv
import json
import os
import random

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int16)


def hb(h):
    return np.frombuffer(bytes.fromhex(h), dtype=np.uint8) if h else np.zeros(8, np.uint8)


def load(csv_path):
    with open(csv_path) as f:
        return [r for r in csv.DictReader(f) if not r["error"]]


def cluster(n, pairs):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    groups = collections.defaultdict(list)
    for k in range(n):
        groups[find(k)].append(k)
    return [g for g in groups.values() if len(g) > 1]


def candidate_pairs(dh, ph, dmax=3):
    """All (i,j,dhash_dist,phash_dist) with i<j and dhash_dist<=dmax."""
    n = len(dh)
    out = []
    B = 256
    for s in range(0, n, B):
        e = min(s + B, n)
        x = dh[s:e][:, None, :] ^ dh[None, :, :]
        pd = POP[x].sum(2).astype(np.int16)
        del x
        for li in range(e - s):
            pd[li, s + li] = 9999
        for li, j in np.argwhere(pd <= dmax):
            gi = s + li
            if gi < j:
                p = int(POP[ph[gi] ^ ph[j]].sum())
                out.append((int(gi), int(j), int(pd[li, j]), p))
        del pd
    return out


def group_stats(groups, datasets, classes):
    intra_same = intra_diff = cross = 0
    by_ds = collections.Counter()
    for g in groups:
        ds = {datasets[i] for i in g}
        for d in ds:
            by_ds[d] += 1
        if len(ds) == 1:
            if len({classes[i] for i in g}) == 1:
                intra_same += 1
            else:
                intra_diff += 1
        else:
            cross += 1
    return {
        "groups": len(groups),
        "images_involved": sum(len(g) for g in groups),
        "redundant_removable": sum(len(g) - 1 for g in groups),
        "intra_same_class": intra_same,
        "intra_cross_class": intra_diff,
        "cross_dataset": cross,
        "by_dataset": dict(by_ds),
        "largest_groups": sorted((len(g) for g in groups), reverse=True)[:10],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(REPO, "reports", "duplicates", "image_hashes.csv"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "reports", "duplicates"))
    ap.add_argument("--njoint", type=int, default=1,
                    help="max Hamming distance for BOTH dHash and pHash (default 1)")
    ap.add_argument("--sweep", action="store_true",
                    help="also print a joint-threshold sweep to justify --njoint")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load(args.csv)
    n = len(rows)
    datasets = [r["dataset"] for r in rows]
    classes = [r["cls"] for r in rows]
    paths = [r["path"] for r in rows]
    md5s = [r["md5"] for r in rows]
    dh = np.stack([hb(r["dhash"]) for r in rows])
    ph = np.stack([hb(r["phash"]) for r in rows])

    # exact duplicates
    by = collections.defaultdict(list)
    for i, m in enumerate(md5s):
        by[m].append(i)
    exact = [g for g in by.values() if len(g) > 1]

    # perceptual candidates -> clusters
    cand = candidate_pairs(dh, ph)
    near_pairs = [(i, j) for (i, j, d, p) in cand if d <= args.njoint and p <= args.njoint]
    near = cluster(n, near_pairs)

    rep = {
        "n_images": n,
        "njoint_threshold": args.njoint,
        "exact_md5": group_stats(exact, datasets, classes),
        "perceptual": group_stats(near, datasets, classes),
    }

    # per-dataset intra clusters
    per = {}
    for ds in sorted(set(datasets)):
        idx = [i for i in range(n) if datasets[i] == ds]
        par = {j: j for j in idx}

        def f2(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for i, j in near_pairs:
            if datasets[i] == ds and datasets[j] == ds:
                ri, rj = f2(i), f2(j)
                if ri != rj:
                    par[rj] = ri
        gg = collections.defaultdict(list)
        for j in idx:
            gg[f2(j)].append(j)
        grp = [g for g in gg.values() if len(g) > 1]
        per[ds] = {
            "n_images": len(idx),
            "intra_dup_groups": len(grp),
            "intra_images_involved": sum(len(g) for g in grp),
            "intra_redundant": sum(len(g) - 1 for g in grp),
            "largest_groups": sorted((len(g) for g in grp), reverse=True)[:8],
        }
    rep["per_dataset_intra"] = per

    # cross-dataset matrix
    mtx = collections.Counter()
    for g in near:
        ds = sorted({datasets[i] for i in g})
        for a in range(len(ds)):
            for b in range(a + 1, len(ds)):
                mtx[f"{ds[a]} <-> {ds[b]}"] += 1
    rep["cross_dataset_matrix"] = dict(mtx.most_common())

    # label conflicts (same image, different class)
    seen, conflicts = set(), []
    for g in sorted(exact + near, key=len, reverse=True):
        if len({classes[i] for i in g}) > 1:
            key = tuple(sorted(g))
            if key not in seen:
                seen.add(key)
                conflicts.append(g)
    rep["label_conflict_groups"] = len(conflicts)
    rep["label_conflict_examples"] = [
        {"size": len(g), "datasets": sorted({datasets[i] for i in g}),
         "classes": sorted({classes[i] for i in g}),
         "files": [(datasets[i], classes[i], paths[i]) for i in g[:6]]}
        for g in conflicts[:15]
    ]

    # write artifacts
    groups_csv = os.path.join(args.out_dir, "duplicate_groups.csv")
    with open(groups_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "group_id", "size", "dataset", "class", "path"])
        for kind, gs in (("exact", exact), ("perceptual", near)):
            for gid, g in enumerate(gs):
                for i in g:
                    w.writerow([kind, gid, len(g), datasets[i], classes[i], paths[i]])

    # per-dataset intra EXACT (md5) groups
    exact_ds_csv = os.path.join(args.out_dir, "exact_duplicates_by_dataset.csv")
    with open(exact_ds_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "dup_group_id", "group_size", "md5", "cls", "keep", "redundant", "path"])
        for ds in sorted(set(datasets)):
            md = collections.defaultdict(list)
            for i in range(n):
                if datasets[i] == ds:
                    md[md5s[i]].append(i)
            gid = 0
            for m, g in md.items():
                if len(g) < 2:
                    continue
                gid += 1
                for k, i in enumerate(g):
                    w.writerow([ds, gid, len(g), m, classes[i],
                                "yes" if k == 0 else "no",
                                "no" if k == 0 else "yes", paths[i]])

    with open(os.path.join(args.out_dir, "final_report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    if args.sweep:
        print("joint threshold sweep (d,p):")
        for D, P in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (6, 6)]:
            sel = [(i, j) for (i, j, d, p) in cand if d <= D and p <= P]
            gs = cluster(n, sel)
            print(f"  d<={D},p<={P}: pairs={len(sel):>7} groups={len(gs):>6} "
                  f"largest={sorted((len(g) for g in gs), reverse=True)[:4]}")

    print(json.dumps({k: v for k, v in rep.items()
                      if k in ("n_images", "njoint_threshold", "exact_md5",
                               "perceptual", "label_conflict_groups")}, indent=2))
    print(f"\nartifacts -> {args.out_dir}")


if __name__ == "__main__":
    main()
