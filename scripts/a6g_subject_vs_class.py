"""A6g — is head size a CLASS feature or a SUBJECT feature?

A logistic regression on four occluded-image geometry scalars (Otsu threshold)
reaches 0.596 macro-F1 (manuscript Section 3.4). Earlier drafts quoted 0.611 and
area values from a fixed-threshold pipeline; the manuscript uses Otsu throughout.
But each subject contributes ~13 slices from one acquisition, so head geometry is
partly a property of the person, not the pathology.

If most variance is BETWEEN subjects, then under a slice-level split (the
protocol distributed with this dataset -- 210/233 patients span folds, and Bassi
2026 finds 761/761 traceable test images share a patient with training) the same
feature identifies the PATIENT, not just the class. Leakage and shortcut would
then be two faces of one failure.

Tests:
  T1  variance decomposition of each geometry feature: between-subject vs within
  T2  can geometry alone re-identify the subject? (kNN, subject as the label)
  T3  logistic regression under SUBJECT-GROUPED vs SLICE-LEVEL CV -- the gap is
      the amount of the 0.611 that is patient memorisation
  T4  subject-mean geometry: does the class signal survive collapsing each
      subject to one point? (this is the leakage-free version of the claim)
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_predict)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway
from core import data_mat

CLASSES = ["glioma", "meningioma", "pituitary"]
FIELDS = ["head_area", "vert_centroid", "horiz_centroid", "vert_extent"]

imgs, masks, labels, pids = data_mat.load_arrays()


# ------------------------------------------------------------------ features
def otsu(im):
    """Otsu threshold on a normalised slice. Replaces the crude >0.08 cutoff:
    per-slice min-max normalisation makes a fixed threshold non-comparable."""
    h, _ = np.histogram(im.ravel(), bins=64, range=(0, 1))
    h = h.astype(float); p = h / max(h.sum(), 1)
    omega = np.cumsum(p); mu = np.cumsum(p * np.arange(64))
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    denom[denom == 0] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    return (np.argmax(sigma_b) + 0.5) / 64


def geometry(use_otsu=True, thresh=0.08):
    x = torch.from_numpy(imgs).float().div_(255)[:, None]
    m = torch.from_numpy(masks).float()[:, None]
    x = F.interpolate(x, (128, 128), mode="bilinear", align_corners=False)
    m = (F.interpolate(m, (128, 128), mode="nearest") > 0.5).float()
    occ = (x * (1 - m)).squeeze(1).numpy()

    feats = []
    for im in occ:
        t = otsu(im) if use_otsu else thresh
        fg = im > t
        if fg.sum() < 50:
            feats.append([0.0, 0.0, 0.0, 0.0]); continue
        ys, xs = np.nonzero(fg)
        feats.append([fg.mean(), ys.mean() / 128, xs.mean() / 128,
                      (ys.max() - ys.min()) / 128])
    return np.array(feats)


# ----------------------------------------------------------------------- T1
def variance_decomposition(X):
    """One-way random-effects MoM: between-subject vs within-subject variance.
    ICC = between / (between + within). High ICC -> the feature describes the
    person more than the slice."""
    print("T1 | variance decomposition of geometry features\n")
    print(f"  {'feature':<16} {'between sd':>11} {'within sd':>10} {'ICC':>7}   "
          f"{'class F':>9}")
    print("  " + "-" * 60)
    subs = np.unique(pids)
    for j, name in enumerate(FIELDS):
        groups = [X[pids == s, j] for s in subs]
        groups = [g for g in groups if len(g) >= 2]
        k = len(groups)
        n_bar = np.mean([len(g) for g in groups])
        grand = np.concatenate(groups).mean()
        ms_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
        ms_w = sum(((g - g.mean()) ** 2).sum() for g in groups) / \
               sum(len(g) - 1 for g in groups)
        var_b = max(0.0, (ms_b - ms_w) / n_bar)
        icc = var_b / max(var_b + ms_w, 1e-12)
        F_cls, _ = f_oneway(*[X[labels == c, j] for c in range(3)])
        print(f"  {name:<16} {np.sqrt(var_b):>11.4f} {np.sqrt(ms_w):>10.4f} "
              f"{icc:>7.3f}   {F_cls:>9.1f}")
    print("\n  ICC near 1 -> the feature is a property of the SUBJECT.")
    print("  Under a slice-level split it would identify the patient, not the class.\n")


# ----------------------------------------------------------------------- T2
def subject_reidentification(X, min_slices=8):
    """Can four scalars re-identify which subject a slice came from?
    Leave-one-slice-out 1NN over subjects with enough slices."""
    print("T2 | subject re-identification from geometry alone\n")
    keep = np.array([s for s in np.unique(pids) if (pids == s).sum() >= min_slices])
    mask = np.isin(pids, keep)
    Xs, ps = X[mask], pids[mask]
    Xs = StandardScaler().fit_transform(Xs)

    correct = 0
    for i in range(len(Xs)):
        d = np.linalg.norm(Xs - Xs[i], axis=1)
        d[i] = np.inf
        correct += (ps[np.argmin(d)] == ps[i])
    acc = correct / len(Xs)
    chance = 1.0 / len(keep)
    print(f"  subjects with >={min_slices} slices : {len(keep)}")
    print(f"  slices                       : {len(Xs)}")
    print(f"  1NN subject accuracy         : {acc:.4f}")
    print(f"  chance                       : {chance:.4f}  ({acc/chance:.0f}x)\n")
    return acc, chance


# ----------------------------------------------------------------------- T3
def grouped_vs_slice_cv(X):
    """The gap between subject-grouped and slice-level CV is the portion of the
    0.611 that is patient memorisation rather than class signal."""
    print("T3 | subject-grouped vs slice-level CV, same features, same model\n")
    clf = lambda: make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000,
                                                   class_weight="balanced"))
    out = {}
    p_g = cross_val_predict(clf(), X, labels, groups=pids,
                            cv=StratifiedGroupKFold(5, shuffle=True, random_state=0))
    p_s = cross_val_predict(clf(), X, labels,
                            cv=StratifiedKFold(5, shuffle=True, random_state=0))
    for tag, p in [("subject-grouped", p_g), ("slice-level (leaky)", p_s)]:
        f1 = f1_score(labels, p, average="macro")
        out[tag] = f1
        print(f"  {tag:<22} macro-F1 {f1:.4f}   acc {(p == labels).mean():.4f}")
    gap = out["slice-level (leaky)"] - out["subject-grouped"]
    print(f"\n  leakage inflation: {100*gap:+.1f} F1 points")
    print("  (Bassi 2026 report 2.73 pts from the grouping rule alone on the")
    print("   four-class aggregate with a ResNet-18)\n")
    print("  subject-grouped confusion (rows=true):")
    print(confusion_matrix(labels, p_g), "\n")
    return out


# ----------------------------------------------------------------------- T4
def subject_level_classification(X):
    """Collapse each subject to its mean geometry, then classify SUBJECTS.
    n=209. This is the leakage-free version: no slice can leak to another."""
    print("T4 | subject-level classification (one point per subject)\n")
    subs = np.unique(pids)
    Xs = np.array([X[pids == s].mean(axis=0) for s in subs])
    ys = np.array([labels[pids == s][0] for s in subs])
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    p = cross_val_predict(clf, Xs, ys,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0))
    print(f"  subjects {len(subs)}   macro-F1 {f1_score(ys, p, average='macro'):.4f}"
          f"   acc {(p == ys).mean():.4f}")
    print(f"  per-class n: " +
          "  ".join(f"{CLASSES[c]} {int((ys==c).sum())}" for c in range(3)))
    print("\n  confusion (rows=true):")
    print(confusion_matrix(ys, p), "\n")


if __name__ == "__main__":
    print("A6g | is the geometry shortcut class-level or subject-level?\n")
    X_otsu = geometry(use_otsu=True)
    X_fix = geometry(use_otsu=False)
    d = np.abs(X_otsu - X_fix).mean(axis=0)
    print("threshold sensitivity (Otsu vs fixed 0.08), mean |difference|:")
    for n, v in zip(FIELDS, d):
        print(f"  {n:<16} {v:.4f}")
    print()

    X = X_otsu
    print("=" * 66)
    variance_decomposition(X)
    print("=" * 66)
    subject_reidentification(X)
    print("=" * 66)
    grouped_vs_slice_cv(X)
    print("=" * 66)
    subject_level_classification(X)
