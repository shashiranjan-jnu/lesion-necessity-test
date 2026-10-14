"""HAM10000 melanoma-vs-nevus loader, mirroring core/data_mat.py.

Scope decisions, all deliberate:
  * dx in {mel, nv} only -- binary, and the two classes with enough lesions.
  * dx_type == "histo" only. In the full set melanoma is 1113/1113 histo while
    nevi are 3704 follow_up / 503 consensus / 2498 histo, so "not histo -> not
    melanoma" is a perfect rule on 4207 nevus images and provenance is
    confounded with class. Restricting to histo removes the confound at no cost
    in melanoma.
  * balanced at the LESION level (614 mel lesions, 614 nv lesions sampled with
    rng seed 0), not the image level: 1372 lesions carry 2-6 images, and
    sampling images would split a lesion across the balance.
  * grouping key is lesion_id. Never split a lesion across folds.

Cache: (N,224,224,3) uint8 images, (N,224,224) uint8 masks, int64 labels
(0=nv, 1=mel), and lesion ids as a string array. 600x450 is resized directly to
224x224; the distortion is identical for both classes.
"""
import os, glob
import numpy as np
import pandas as pd
from PIL import Image

CACHE = "/kaggle/working/ham_mel_nv_224.npz"
SIZE = 224
CLASSES = ["nv", "mel"]


def _paths():
    g = lambda pat: glob.glob(f"/kaggle/input/**/{pat}", recursive=True)
    md = g("HAM10000_metadata*.csv")
    imgs = {os.path.basename(p)[:-4]: p
            for p in g("*.jpg") if "HAM10000_images" in p}
    segs = {os.path.basename(p).replace("_segmentation.png", ""): p
            for p in g("*_segmentation.png")}
    assert md and imgs and segs, "attach kmader/skin-cancer-mnist-ham10000 " \
                                 "and tschandl/ham10000-lesion-segmentations"
    return md[0], imgs, segs


def select(seed=0):
    """Returns the selected metadata rows. Deterministic."""
    md_p, imgs, segs = _paths()
    md = pd.read_csv(md_p)
    md = md[md.dx.isin(CLASSES) & (md.dx_type == "histo")
            & md.image_id.isin(imgs) & md.image_id.isin(segs)]
    rng = np.random.default_rng(seed)
    mel = md[md.dx == "mel"].lesion_id.unique()
    nv_all = md[md.dx == "nv"].lesion_id.unique()
    nv = rng.choice(nv_all, size=min(len(mel), len(nv_all)), replace=False)
    sel = md[md.lesion_id.isin(set(mel) | set(nv))]
    return sel.sort_values("image_id").reset_index(drop=True), imgs, segs


def build_cache(path=CACHE, seed=0):
    sel, imgs, segs = select(seed)
    n = len(sel)
    X = np.zeros((n, SIZE, SIZE, 3), np.uint8)
    M = np.zeros((n, SIZE, SIZE), np.uint8)
    for i, iid in enumerate(sel.image_id):
        X[i] = np.array(Image.open(imgs[iid]).convert("RGB")
                        .resize((SIZE, SIZE), Image.BILINEAR))
        m = Image.open(segs[iid]).convert("L").resize((SIZE, SIZE), Image.NEAREST)
        M[i] = (np.array(m) > 127).astype(np.uint8)
    y = (sel.dx.values == "mel").astype(np.int64)
    g = sel.lesion_id.values.astype(str)
    np.savez_compressed(path, imgs=X, masks=M, labels=y, gids=g)
    print(f"cached {n} images, {len(np.unique(g))} lesions, "
          f"mel {int(y.sum())} / nv {int((1 - y).sum())} -> {path}")
    return path


def load_arrays(path=CACHE):
    """(imgs uint8 NHWC, masks uint8 NHW, labels int64, gids str) --- same
    contract as data_mat.load_arrays, except imgs are RGB and gids are lesions."""
    if not os.path.exists(path):
        build_cache(path)
    z = np.load(path, allow_pickle=False)
    return z["imgs"], z["masks"], z["labels"], z["gids"]
