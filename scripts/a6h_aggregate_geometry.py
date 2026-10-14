"""A6h — does the geometry shortcut extend to the four-class aggregate,
including the independently sourced `notumor` class?

Context. On Cheng, four scalars measured from the head outline of TUMOR-OCCLUDED
images separate the classes (manuscript Sec. 3.4: 0.596 macro-F1 with the Otsu
pipeline). NOTE: numbers printed by this script that refer to the tumor classes
used an earlier fixed-threshold pipeline and patient grouping that is invalid on
the aggregate; the manuscript reports only the no-tumor analysis, whose
leakage-controlled values come from a6i_notumor_control.py (Table 5).

The four-class aggregate is NOT an independent replication for the tumor classes:
4,586 of its 5,023 tumor images are the same figshare slices, JPEG re-encoded and
rotated (Bassi 2026). But its `notumor` images (1,800 in the release analysed) come from Br35H, a different
collection. If notumor also separates on head geometry, the shortcut spans a
source the tumor classes never touched.

No masks here, so geometry is measured on INTACT images. That is a weaker test
than the Cheng occlusion result -- the tumor contributes to the head mask -- but
tumors are ~1.7% of the frame and the class differences are ~19 points, so the
lesion cannot account for the effect. Reported both ways where possible.

Usage:
    python scripts/a6h_aggregate_geometry.py --data_dir /kaggle/input/<slug>
    (expects Training/ and Testing/ subfolders with class directories)
"""
import argparse, glob, json, os, collections
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_predict)
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway

SIZE = 128


def otsu(im):
    h, _ = np.histogram(im.ravel(), bins=64, range=(0, 1))
    h = h.astype(float); p = h / max(h.sum(), 1)
    omega = np.cumsum(p); mu = np.cumsum(p * np.arange(64))
    denom = omega * (1 - omega); denom[denom == 0] = 1e-12
    return (np.argmax((mu[-1] * omega - mu) ** 2 / denom) + 0.5) / 64


def geom(path):
    im = np.asarray(Image.open(path).convert("L").resize((SIZE, SIZE),
                                                         Image.BILINEAR),
                    dtype=np.float32) / 255.0
    fg = im > otsu(im)
    if fg.sum() < 50:
        return None
    ys, xs = np.nonzero(fg)
    return [fg.mean(), ys.mean() / SIZE, xs.mean() / SIZE,
            (ys.max() - ys.min()) / SIZE]


def load(data_dir, pmap):
    rows = []
    for split in ("Training", "Testing"):
        d = os.path.join(data_dir, split)
        if not os.path.isdir(d):
            continue
        for cls in sorted(os.listdir(d)):
            cd = os.path.join(d, cls)
            if not os.path.isdir(cd):
                continue
            for f in sorted(os.listdir(cd)):
                if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                g = geom(os.path.join(cd, f))
                if g is None:
                    continue
                key = f"data/{split}/{cls}/{f}"
                info = pmap.get(key, {})
                rows.append({"x": g, "cls": cls, "split": split, "file": f,
                             "pid": info.get("pid"),
                             "traced": key in pmap})
    return rows


FIELDS = ["head_area", "vert_centroid", "horiz_centroid", "vert_extent"]


def describe(rows, classes, title):
    print(f"\n{title}")
    print(f"  {'class':<12} {'n':>6} " + "".join(f"{f:>17}" for f in FIELDS))
    for c in classes:
        sub = np.array([r["x"] for r in rows if r["cls"] == c])
        line = f"  {c:<12} {len(sub):>6} "
        for j in range(4):
            line += f"{sub[:,j].mean():>9.3f}+-{sub[:,j].std():<6.3f}"
        print(line)
    print("\n  one-way ANOVA across classes:")
    for j, f in enumerate(FIELDS):
        groups = [np.array([r["x"][j] for r in rows if r["cls"] == c])
                  for c in classes]
        F_, p = f_oneway(*groups)
        print(f"    {f:<16} F={F_:>9.1f}  p={p:.2e}")


def classify(rows, classes, groups=None, label=""):
    X = np.array([r["x"] for r in rows])
    y = np.array([classes.index(r["cls"]) for r in rows])
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000,
                                           class_weight="balanced"))
    if groups is not None:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
        p = cross_val_predict(clf, X, y, groups=groups, cv=cv)
    else:
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        p = cross_val_predict(clf, X, y, cv=cv)
    f1 = f1_score(y, p, average="macro")
    print(f"\n  {label}")
    print(f"    macro-F1 {f1:.4f}   acc {(p == y).mean():.4f}   n={len(y)}")
    cm = confusion_matrix(y, p)
    print(f"    per-class recall: " +
          "  ".join(f"{c} {cm[i,i]/cm[i].sum():.2f}"
                    for i, c in enumerate(classes)))
    print("    confusion (rows=true):"); print(cm)
    return f1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--pmap", default="figshare_patient_map.json")
    a = ap.parse_args()

    pmap = json.load(open(a.pmap)) if os.path.exists(a.pmap) else {}
    print(f"A6h | four-class aggregate geometry | patient map: "
          f"{len(pmap)} entries\n")

    rows = load(a.data_dir, pmap)
    classes = sorted(set(r["cls"] for r in rows))
    print(f"loaded {len(rows)} images, classes {classes}")
    print(f"traced to figshare: {sum(r['traced'] for r in rows)}")

    describe(rows, classes, "GEOMETRY BY CLASS (intact images, Otsu threshold)")

    # --- 4-class, slice level (no patient IDs for notumor) ---
    classify(rows, classes, None, "4-class, slice-level CV (leaky for tumor classes)")

    # --- 4-class, grouped where possible: pid for tumor, filename for notumor ---
    groups = [r["pid"] if r["pid"] else f"nt_{r['file']}" for r in rows]
    classify(rows, classes, np.array(groups),
             "4-class, grouped by patient (notumor ungrouped -> optimistic)")

    # --- notumor vs each tumor class: is the Br35H class separable? ---
    if "notumor" in classes:
        print("\n" + "=" * 62)
        print("notumor (Br35H, independent source) vs each tumor class:")
        for c in [x for x in classes if x != "notumor"]:
            sub = [r for r in rows if r["cls"] in ("notumor", c)]
            g = np.array([r["pid"] if r["pid"] else f"nt_{r['file']}"
                          for r in sub])
            classify(sub, ["notumor", c], g, f"notumor vs {c}")

    # --- tumor-only, patient-grouped: the comparable number to Cheng 0.611 ---
    tum = [r for r in rows if r["cls"] != "notumor" and r["pid"]]
    if tum:
        print("\n" + "=" * 62)
        tcls = sorted(set(r["cls"] for r in tum))
        classify(tum, tcls, np.array([r["pid"] for r in tum]),
                 f"tumor-only, patient-grouped  (Cheng occluded: 0.611)")

        # subject-level, leakage-free
        subs = sorted(set(r["pid"] for r in tum))
        Xs = np.array([np.mean([r["x"] for r in tum if r["pid"] == s], axis=0)
                       for s in subs])
        ys = np.array([tcls.index(next(r["cls"] for r in tum if r["pid"] == s))
                       for s in subs])
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000,
                                               class_weight="balanced"))
        p = cross_val_predict(clf, Xs, ys,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0))
        print(f"\n  subject-level, one point per patient  (Cheng: 0.699)")
        print(f"    macro-F1 {f1_score(ys, p, average='macro'):.4f}   "
              f"acc {(p == ys).mean():.4f}   n={len(subs)}")
        print(confusion_matrix(ys, p))
