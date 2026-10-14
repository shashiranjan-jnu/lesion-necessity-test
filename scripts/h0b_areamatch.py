"""H0b -- close the hole-size leak by matching the classes on mask geometry.

H0 showed the annotation alone predicts the class at 0.615 macro-F1 (AUROC
0.664), driven entirely by size: area alone 0.611, centroids 0.539. So under
plain occlusion the hole itself carries the label, and any occluded-image result
would be uninterpretable.

Fix: match melanoma and nevus LESIONS on mask geometry, so the hole distribution
is the same in both classes. Then delete the real lesion as usual. The check is
that the mask-only regression falls to chance on the matched subset -- if it
does not, the matching is not tight enough and the caliper must shrink.

Matching is greedy 1-1 nearest neighbour on standardised (area, height, width),
at the lesion level (a lesion's geometry is the mean over its images), with a
caliper. Lesions outside the caliper are dropped, not forced into a pair.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from core import data_ham
from scripts.h0_assemble import mask_geometry, FIELDS

OUT = "/kaggle/working/ham_matched_ids.npy"
CALIPER = 0.30          # in pooled SD units of the 3-vector
COLS = [0, 3, 4]        # area, height, width


def lesion_table(X, y, g):
    lesions = np.unique(g)
    Z = np.stack([X[g == L][:, COLS].mean(0) for L in lesions])
    yl = np.array([y[g == L][0] for L in lesions])
    return lesions, Z, yl


def match(Z, yl, caliper=CALIPER, seed=0):
    Zs = StandardScaler().fit_transform(Z)
    mel = np.where(yl == 1)[0]
    nv = np.where(yl == 0)[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(mel)           # order-independence: randomise
    taken, pairs = set(), []
    for i in order:
        d = np.linalg.norm(Zs[nv] - Zs[i], axis=1)
        d[[k for k, j in enumerate(nv) if j in taken]] = np.inf
        k = int(np.argmin(d))
        if d[k] <= caliper:
            taken.add(nv[k]); pairs.append((i, nv[k]))
    return pairs


def mask_only_cv(X, y, g, tag):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(clf, X, y, groups=g, cv=cv)
    pr = cross_val_predict(clf, X, y, groups=g, cv=cv, method="predict_proba")[:, 1]
    print(f"  {tag:<34} macro-F1 {f1_score(y, p, average='macro'):.4f}   "
          f"AUROC {roc_auc_score(y, pr):.4f}")


if __name__ == "__main__":
    imgs, masks, y, g = data_ham.load_arrays()
    X = mask_geometry(masks)
    lesions, Z, yl = lesion_table(X, y, g)
    print(f"\nH0b | {len(lesions)} lesions before matching "
          f"(mel {int(yl.sum())}, nv {int((1 - yl).sum())})\n")

    for cal in [0.40, 0.30, 0.20, 0.10, 0.05]:
        pairs = match(Z, yl, cal)
        keep = set(lesions[[i for p in pairs for i in p]])
        m = np.isin(g, list(keep))
        print(f"caliper {cal:.2f}: {len(pairs)} pairs, {m.sum()} images")
        mask_only_cv(X[m], y[m], g[m], f"  mask-only, caliper {cal:.2f}")
        for j, c in enumerate(COLS):
            a, b = X[m & (y == 0), c], X[m & (y == 1), c]
            sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            print(f"     {FIELDS[c]:<8} nv {a.mean():.3f}  mel {b.mean():.3f}  "
                  f"d={abs(b.mean() - a.mean()) / max(sd, 1e-9):.3f}")
        print()

    pairs = match(Z, yl, CALIPER)
    keep = lesions[[i for p in pairs for i in p]]
    np.save(OUT, keep)
    print(f"saved {len(keep)} matched lesion ids (caliper {CALIPER}) -> {OUT}")
    print("Pick the largest caliper whose mask-only macro-F1 is <= ~0.53 and")
    print("whose d values are all below ~0.10; that is the working subset.")
