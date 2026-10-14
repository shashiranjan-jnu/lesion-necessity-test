"""A6i — leakage control for the notumor geometry result.

A6h found four head-outline scalars separate `notumor` (Br35H) from each tumor
class at 0.699 / 0.722 / 0.704 macro-F1 (chance 0.50). But notumor has no patient
identifiers, so A6h grouped each image alone -- if Br35H contains several slices
per patient, that class gets leakage the tumor classes do not, and part of 0.70
could be patient memorisation rather than class signal.

Bassi (2026) hit the same wall and grouped notumor by greedy near-duplicate
cluster at cosine >= 0.99 (Methods 4.2). Same fix here.

Reported:
  1. cluster structure -- how much redundancy is actually in Br35H
  2. pairwise notumor-vs-tumor with clusters as CV groups
  3. cluster-level classification (one point per cluster) -- leakage-free
  4. sensitivity across cluster thresholds

Usage:
    python scripts/a6i_notumor_control.py --data_dir /kaggle/input/<...>
"""
import argparse, os, collections
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_predict)
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SIZE, EMB = 128, 32
TUMOR = ("glioma", "meningioma", "pituitary")


def otsu(im):
    h, _ = np.histogram(im.ravel(), bins=64, range=(0, 1))
    h = h.astype(float); p = h / max(h.sum(), 1)
    omega = np.cumsum(p); mu = np.cumsum(p * np.arange(64))
    den = omega * (1 - omega); den[den == 0] = 1e-12
    return (np.argmax((mu[-1] * omega - mu) ** 2 / den) + 0.5) / 64


def load_class(root, cls):
    """Returns geometry features, embedding vectors, and file paths."""
    feats, embs, paths = [], [], []
    for split in ("Training", "Testing"):
        d = os.path.join(root, split, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            p = os.path.join(d, f)
            im = Image.open(p).convert("L")
            a = np.asarray(im.resize((SIZE, SIZE), Image.BILINEAR),
                           dtype=np.float32) / 255.0
            fg = a > otsu(a)
            if fg.sum() < 50:
                continue
            ys, xs = np.nonzero(fg)
            feats.append([fg.mean(), ys.mean() / SIZE, xs.mean() / SIZE,
                          (ys.max() - ys.min()) / SIZE])
            v = np.asarray(im.resize((EMB, EMB), Image.BILINEAR),
                           dtype=np.float64).ravel()
            v -= v.mean(); n = np.linalg.norm(v)
            embs.append(v / n if n else v)
            paths.append(p)
    return np.array(feats), np.array(embs), paths


def greedy_clusters(E, thresh=0.99):
    """Greedy near-duplicate grouping. A representative cannot be absorbed into
    another cluster, matching Bassi's 'preventing two cluster representatives
    from merging'."""
    n = len(E)
    assigned = -np.ones(n, dtype=int)
    cid = 0
    for i in range(n):
        if assigned[i] >= 0:
            continue
        sims = E[i:] @ E[i]
        idx = np.nonzero(sims >= thresh)[0] + i
        idx = idx[assigned[idx] < 0]
        assigned[idx] = cid
        cid += 1
    return assigned, cid


def evaluate(Xa, Xb, ga, gb, name_a, name_b, grouped=True):
    X = np.vstack([Xa, Xb])
    y = np.r_[np.zeros(len(Xa), int), np.ones(len(Xb), int)]
    g = np.r_[np.asarray(ga, dtype=object), np.asarray(gb, dtype=object)]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000,
                                           class_weight="balanced"))
    if grouped:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
        p = cross_val_predict(clf, X, y, groups=g, cv=cv)
    else:
        p = cross_val_predict(clf, X, y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0))
    return f1_score(y, p, average="macro"), confusion_matrix(y, p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    a = ap.parse_args()

    print("A6i | notumor leakage control\n")
    print("loading...")
    data = {c: load_class(a.data_dir, c) for c in ("notumor",) + TUMOR}
    for c, (F, E, P) in data.items():
        print(f"  {c:<12} {len(F)} images")

    Fn, En, _ = data["notumor"]

    # ---------------- 1. cluster structure -------------------------------
    print("\n" + "=" * 62)
    print("1 | near-duplicate structure of notumor (Br35H)\n")
    print(f"  {'thresh':>7} {'clusters':>9} {'largest':>8} {'singletons':>11} "
          f"{'mean size':>10}")
    clusterings = {}
    for t in (0.999, 0.99, 0.98, 0.95):
        lab, k = greedy_clusters(En, t)
        sizes = np.bincount(lab)
        clusterings[t] = lab
        print(f"  {t:>7.3f} {k:>9} {sizes.max():>8} "
              f"{(sizes == 1).sum():>11} {sizes.mean():>10.2f}")
    print("\n  many small clusters -> little redundancy; a few large ones ->")
    print("  repeated slices that would leak across an ungrouped split")

    # ---------------- 2/3. pairwise, ungrouped vs clustered ---------------
    print("\n" + "=" * 62)
    print("2 | notumor vs each tumor class\n")
    print(f"  {'comparison':<26} {'ungrouped':>10} {'clustered':>10} "
          f"{'delta':>8}")
    print("  " + "-" * 56)
    lab99 = clusterings[0.99]
    for c in TUMOR:
        Ft, _, Pt = data[c]
        gt = [f"{c}_{i}" for i in range(len(Ft))]          # tumor: ungrouped
        gn_u = [f"nt_{i}" for i in range(len(Fn))]
        gn_c = [f"ntc_{v}" for v in lab99]
        f_u, _ = evaluate(Fn, Ft, gn_u, gt, "notumor", c)
        f_c, cm = evaluate(Fn, Ft, gn_c, gt, "notumor", c)
        print(f"  notumor vs {c:<15} {f_u:>10.4f} {f_c:>10.4f} "
              f"{100*(f_c-f_u):>+7.1f}")

    # ---------------- 4. cluster-level, leakage-free ----------------------
    print("\n" + "=" * 62)
    print("3 | cluster-level classification (one point per notumor cluster)\n")
    k = lab99.max() + 1
    Xn = np.array([Fn[lab99 == i].mean(axis=0) for i in range(k)])
    for c in TUMOR:
        Ft, _, _ = data[c]
        # subsample tumor to match cluster count so the comparison is balanced
        rng = np.random.default_rng(0)
        sel = rng.choice(len(Ft), size=min(len(Ft), len(Xn)), replace=False)
        X = np.vstack([Xn, Ft[sel]])
        y = np.r_[np.zeros(len(Xn), int), np.ones(len(sel), int)]
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000,
                                               class_weight="balanced"))
        p = cross_val_predict(clf, X, y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0))
        print(f"  notumor({len(Xn)} clusters) vs {c}({len(sel)}): "
              f"macro-F1 {f1_score(y, p, average='macro'):.4f}")

    # ---------------- 5. threshold sensitivity ----------------------------
    print("\n" + "=" * 62)
    print("4 | sensitivity to the clustering threshold (vs glioma)\n")
    Fg, _, _ = data["glioma"]
    gt = [f"gl_{i}" for i in range(len(Fg))]
    for t, lab in clusterings.items():
        f, _ = evaluate(Fn, Fg, [f"c_{v}" for v in lab], gt, "notumor", "glioma")
        print(f"  thresh {t:.3f}  ({lab.max()+1:>4} clusters)  macro-F1 {f:.4f}")

    print("\n  if macro-F1 holds near 0.70 across thresholds, the separation is")
    print("  class signal, not patient memorisation")
