"""H0 -- assemble the HAM10000 arm and bound the mask-shape channel.

Melanoma masks cover 31.9% of the frame against 24.8% for nevi (~0.36 pooled SD
on area, 0.44 on width; centroids are identical). So the occluded image x- leaks
class information through the SHAPE OF THE HOLE, independently of any residual
skin. Before training anything, measure how much: fit the same four-scalar
logistic regression used on the brain collection to the mask alone, with no
pixels. That number is the floor any occluded-image result must clear before it
can be read as evidence about the background.

T1  mask-only geometry: 4 scalars -> class, lesion-grouped CV
T2  the same with area removed, to see whether position carries anything
T3  per-class mask statistics on the full selected set (not a sample)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway
from core import data_ham

FIELDS = ["area", "cy", "cx", "height", "width"]


def mask_geometry(masks):
    """Five scalars per mask. Same definitions as the brain-collection
    surrogates, applied to the annotation rather than the head."""
    out = np.zeros((len(masks), 5), np.float32)
    H, W = masks.shape[1:]
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        if len(ys) < 50:
            continue
        out[i] = [m.mean(), ys.mean() / H, xs.mean() / W,
                  (ys.max() - ys.min()) / H, (xs.max() - xs.min()) / W]
    return out


def cv(X, y, g, tag):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    cvs = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(clf, X, y, groups=g, cv=cvs)
    pr = cross_val_predict(clf, X, y, groups=g, cv=cvs, method="predict_proba")[:, 1]
    print(f"  {tag:<28} macro-F1 {f1_score(y, p, average='macro'):.4f}   "
          f"acc {(p == y).mean():.4f}   AUROC {roc_auc_score(y, pr):.4f}")
    return p


if __name__ == "__main__":
    imgs, masks, y, g = data_ham.load_arrays()
    print(f"\nH0 | {len(y)} images, {len(np.unique(g))} lesions, "
          f"mel {int(y.sum())} / nv {int((1 - y).sum())}\n")

    X = mask_geometry(masks)

    print("T3 | mask geometry by class (mean +- sd), full selected set")
    print(f"  {'field':<10} {'nv':>18} {'mel':>18} {'F':>10}")
    for j, f in enumerate(FIELDS):
        a, b = X[y == 0, j], X[y == 1, j]
        F, _ = f_oneway(a, b)
        sd = np.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 +
                      (len(b) - 1) * b.std(ddof=1) ** 2) / (len(a) + len(b) - 2))
        print(f"  {f:<10} {a.mean():>8.3f}+-{a.std():<8.3f} "
              f"{b.mean():>8.3f}+-{b.std():<8.3f} {F:>10.1f}   "
              f"d={abs(b.mean() - a.mean()) / max(sd, 1e-9):.2f}")

    print("\nT1/T2 | can the ANNOTATION alone predict the class? (no pixels)")
    cv(X, y, g, "5 scalars (incl. area)")
    cv(X[:, 1:3], y, g, "centroids only")
    cv(X[:, [0]], y, g, "area only")
    print("\n  Chance is 0.500. Whatever this reaches is the floor for the")
    print("  occluded condition: an occluded model can score at least this")
    print("  well from the hole alone, before using any skin.\n")
