"""Loader for the original Cheng .mat release. Adds tumor masks, which the
PNG copy lacks. Same subject-grouping convention as core/data.py."""
import glob, collections, re
import numpy as np, h5py

ROOT_DEFAULT = "/kaggle/input/datasets/rashashi/cheng-tumor-mat"
MAT_LABEL = {1: "meningioma", 2: "glioma", 3: "pituitary"}
CLASSES = ["glioma", "meningioma", "pituitary"]          # keep index order
MAT2IDX = {1: 1, 2: 0, 3: 2}                             # mat label -> CLASSES idx
_BASE = re.compile(r"^(.*?\d+)[A-Z]*$")
NATIVE = 224


def base_pid(p):
    m = _BASE.match(p)
    return m.group(1) if m else p


def build_cache(root=ROOT_DEFAULT, out_path="/kaggle/working/cache/cheng_mat224.npz"):
    import os
    from PIL import Image
    files = sorted(glob.glob(f"{root}/*/*.mat"))
    n = len(files)
    images = np.zeros((n, NATIVE, NATIVE), dtype=np.uint8)
    masks  = np.zeros((n, NATIVE, NATIVE), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)
    pids, raw_pids = [], []

    for i, fp in enumerate(files):
        with h5py.File(fp, "r") as h:
            d = h["cjdata"]
            lab = int(np.array(d["label"]).ravel()[0])
            pid = "".join(chr(c) for c in np.array(d["PID"]).ravel()).strip()
            img = np.array(d["image"]).astype(np.float64)
            msk = np.array(d["tumorMask"]).astype(np.uint8)

        lo, hi = img.min(), img.max()                    # README's own scaling
        img = np.uint8(255.0 / (hi - lo) * (img - lo)) if hi > lo else np.zeros_like(img, np.uint8)

        images[i] = np.array(Image.fromarray(img).resize((NATIVE, NATIVE), Image.BILINEAR))
        masks[i]  = (np.array(Image.fromarray(msk * 255).resize(
                        (NATIVE, NATIVE), Image.NEAREST)) > 127).astype(np.uint8)
        labels[i] = MAT2IDX[lab]
        raw_pids.append(pid); pids.append(base_pid(pid))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, images=images, masks=masks, labels=labels,
                        pids=np.array(pids), raw_pids=np.array(raw_pids))
    print(f"cached {n} -> {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB, "
          f"{len(set(pids))} subjects / {len(set(raw_pids))} sessions)")
    return out_path


def load_arrays(npz_path="/kaggle/working/cache/cheng_mat224.npz"):
    d = np.load(npz_path, allow_pickle=False)
    return d["images"], d["masks"], d["labels"], d["pids"]
