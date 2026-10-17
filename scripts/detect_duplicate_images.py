#!/usr/bin/env python3
"""Hash every image in the source datasets (md5 + dHash + pHash + aHash).

Excludes merged_dataset by default (it is a concatenation of the sources).

Outputs a CSV with one row per image, consumed by analyze_duplicate_images.py.

Usage
-----
    python3 scripts/detect_duplicate_images.py
    python3 scripts/detect_duplicate_images.py --out reports/duplicates/image_hashes.csv
"""
import argparse
import csv
import hashlib
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

# Repository root = parent of this script's directory.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset name -> sub-directory (relative to --data-root) holding <class>/*.{{jpg,png,...}}
DEFAULT_DATASETS = {
    "kaggle_dataset": "kaggle_dataset/convertido",
    "roboflow_convertido": "roboflow_convertido/merged",
    "trashbox": "trashbox/TrashBox_train_set",
    "trashnet": "trashnet/dataset-resized",
}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


def collect(data_root, datasets):
    rows = []
    for ds, rel in datasets.items():
        base = os.path.join(data_root, rel)
        if not os.path.isdir(base):
            print(f"warning: missing dataset dir {base}", flush=True)
            continue
        for cls in sorted(os.listdir(base)):
            cdir = os.path.join(base, cls)
            if not os.path.isdir(cdir):
                continue
            for fn in sorted(os.listdir(cdir)):
                if os.path.splitext(fn)[1].lower() in IMG_EXT:
                    rows.append((ds, cls, os.path.join(cdir, fn)))
    return rows


def dhash(img, hash_size=8):
    """9x8 grayscale, compare horizontal neighbours -> hash_size*hash_size bits."""
    im = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def phash(img, hash_size=8, highfreq_factor=4):
    """DCT-based perceptual hash (numpy-only, no scipy/cv2 needed)."""
    size = hash_size * highfreq_factor
    im = img.convert("L").resize((size, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float64)
    k = np.arange(size)
    basis = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * size)) * np.sqrt(2.0 / size)
    basis[0, :] *= np.sqrt(0.5)
    dct = basis @ a @ basis.T
    low = dct[:hash_size, :hash_size]
    return np.packbits((low > np.median(low)).flatten())


def ahash(img, hash_size=8):
    im = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float64)
    return np.packbits((a > a.mean()).flatten())


def process(row):
    ds, cls, path = row
    base = dict(dataset=ds, cls=cls, path=path, filesize=-1, width=-1,
                height=-1, md5="", dhash="", phash="", ahash="", error="")
    try:
        md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                md5.update(chunk)
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            d = dhash(img).tobytes().hex()
            p = phash(img).tobytes().hex()
            a = ahash(img).tobytes().hex()
        base.update(filesize=os.path.getsize(path), width=w, height=h,
                    md5=md5.hexdigest(), dhash=d, phash=p, ahash=a)
    except Exception as e:  # noqa: BLE001 - record and continue
        base["error"] = repr(e)
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.path.join(REPO, "data"),
                    help="root dir containing the source datasets")
    ap.add_argument("--out", default=os.path.join(REPO, "reports", "duplicates", "image_hashes.csv"),
                    help="output CSV path")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = collect(args.data_root, DEFAULT_DATASETS)
    print(f"total images: {len(rows)}", flush=True)
    fields = ["dataset", "cls", "path", "filesize", "width", "height",
              "md5", "dhash", "phash", "ahash", "error"]
    t0 = time.time()
    done = 0
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process, r) for r in rows]
            for fut in as_completed(futs):
                w.writerow(fut.result())
                done += 1
                if done % 2000 == 0:
                    f.flush()
                    rate = done / (time.time() - t0)
                    print(f"  {done}/{len(rows)}  {rate:.0f} img/s  "
                          f"eta {(len(rows) - done) / rate:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
