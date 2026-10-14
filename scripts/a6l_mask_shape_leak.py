"""A6l -- does the SHAPE OF THE HOLE explain the occluded brain result? 

Reviewer concern (M1). In the occluded condition x- = x * (1 - m), the black
region left by the tumor mask has the size and position of the tumor. Tumor
size and position differ by class (pituitary: small, low, central; glioma:
large). An occluded model could therefore read the class from the hole, and
"the tumor is not necessary" would really mean "the tumor's outline is enough".
The HAM10000 arm controls this channel (h0 / h0b: mask-only 0.615 macro-F1,
then size matching). The brain arm did not. This script measures it.

Part A (CPU, ~2 min) -- logistic regression on scalars, same folds as A6e
  A1  mask-only: 5 scalars of the ANNOTATION alone (area, cy, cx, height,
      width), no pixels -- identical definition to h0_assemble.py
  A2  head geometry on OCCLUDED images (A6g features; should reproduce the
      0.596 macro-F1 reported in the paper -- sanity check)
  A3  head geometry on INTACT images (no hole at all)
  A4  head (intact) + mask scalars -- does the hole add to the head outline?

Part B (GPU) -- the strongest version of the leak: a CNN that sees ONLY the
  binary mask, trained with the exact A6e recipe, folds and seeds. Compared
  with the occluded model (re-run in the same session so that per-image
  correctness is available for a paired, patient-clustered bootstrap).
  B1  mask-only CNN          input = m
  B2  occluded CNN (re-run)  input = x * (1 - m)     -> must reproduce A6e
  B3  paired contrast        acc(occluded) - acc(mask-only), 95% patient CI

Reading the result
  - If B1 is well below B2 (CI of B3 excludes zero, positive), the hole alone
    cannot account for the occluded performance: the residual signal is in the
    remaining image (head geometry), as the paper claims.
  - If A3 ~ A2, the head-geometry surrogate does not depend on the hole.
  - If B1 ~ B2, the brain arm needs size matching like the HAM arm before the
    "not necessary" verdict can stand.

Usage (Kaggle, after the usual clone + data_mat.build_cache() cells):
    !cd /tmp/code && PYTHONPATH=/tmp/code python scripts/a6l_mask_shape_leak.py
    # Part A only (no GPU needed):
    !cd /tmp/code && PYTHONPATH=/tmp/code python scripts/a6l_mask_shape_leak.py --part A
    # add the ResNet-18 arm to Part B (slower, ~15 extra runs per condition):
    !cd /tmp/code && PYTHONPATH=/tmp/code python scripts/a6l_mask_shape_leak.py --arch both
Output: console log + /kaggle/working/a6l_results.json
"""
import argparse, json, os, time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # needed by set_seed
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from core import data_mat, model as M
from core.seeding import set_seed

CLASSES = ["glioma", "meningioma", "pituitary"]
MASK_FIELDS = ["area", "cy", "cx", "height", "width"]
HEAD_FIELDS = ["head_area", "vert_centroid", "horiz_centroid", "vert_extent"]

# A6e recipe -- do not change
EPOCHS, BATCH, LR, WD = 30, 32, 1e-3, 1e-4
SEEDS = [1, 2, 3]
N_BOOT = 2000
ARCHS = {
    "smallcnn":     dict(name="smallcnn", size=128, kw=dict(in_size=128)),
    "resnet18-l4b": dict(name="resnet18", size=224,
                         kw=dict(pretrained=True, trainable="layer4b")),
}
# A6e per-fold macro-F1 for the occluded condition (results/brain/a6e_results.json)
A6E_OCC = {"smallcnn":     [0.8704, 0.7997, 0.7599, 0.7411, 0.6726],
           "resnet18-l4b": [0.8501, 0.8356, 0.8264, 0.6800, 0.7908]}
OUT = "/kaggle/working/a6l_results.json"

imgs, masks, labels, pids = data_mat.load_arrays()
SPLITS = list(StratifiedGroupKFold(5, shuffle=True, random_state=0)
              .split(imgs, labels, groups=pids))          # identical to A6e


# =============================================================== Part A
def mask_scalars():
    """Same five scalars as h0_assemble.mask_geometry, on the 224 cache."""
    H, W = masks.shape[1:]
    out = np.zeros((len(masks), 5), np.float64)
    n_small = 0
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            n_small += 1
            continue
        out[i] = [m.mean(), ys.mean() / H, xs.mean() / W,
                  (ys.max() - ys.min()) / H, (xs.max() - xs.min()) / W]
    if n_small:
        print(f"  note: {n_small} slices with an empty mask (features left at 0)")
    return out


def otsu(im):
    """Identical to A6g."""
    h, _ = np.histogram(im.ravel(), bins=64, range=(0, 1))
    h = h.astype(float); p = h / max(h.sum(), 1)
    omega = np.cumsum(p); mu = np.cumsum(p * np.arange(64))
    den = omega * (1 - omega); den[den == 0] = 1e-12
    return (np.argmax((mu[-1] * omega - mu) ** 2 / den) + 0.5) / 64


def head_scalars(occluded):
    """A6g head-outline features at 128x128, on occluded or intact images."""
    x = torch.from_numpy(imgs).float().div_(255)[:, None]
    x = F.interpolate(x, (128, 128), mode="bilinear", align_corners=False)
    if occluded:
        m = torch.from_numpy(masks).float()[:, None]
        m = (F.interpolate(m, (128, 128), mode="nearest") > 0.5).float()
        x = x * (1 - m)
    out = []
    for im in x.squeeze(1).numpy():
        fg = im > otsu(im)
        if fg.sum() < 50:
            out.append([0.0, 0.0, 0.0, 0.0]); continue
        ys, xs = np.nonzero(fg)
        out.append([fg.mean(), ys.mean() / 128, xs.mean() / 128,
                    (ys.max() - ys.min()) / 128])
    return np.array(out)


def lr_cv(X, tag):
    """Logistic regression on the A6e folds. Returns pooled predictions."""
    p = np.zeros_like(labels)
    fold_f1 = []
    for tr, te in SPLITS:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], labels[tr])
        p[te] = clf.predict(X[te])
        fold_f1.append(f1_score(labels[te], p[te], average="macro"))
    f1 = f1_score(labels, p, average="macro")
    rec = recall_score(labels, p, average=None)
    print(f"  {tag:<34} macro-F1 {f1:.4f}   acc {(p == labels).mean():.4f}   "
          f"recall g/m/p {rec[0]:.2f}/{rec[1]:.2f}/{rec[2]:.2f}   "
          f"fold-F1 {np.mean(fold_f1):.4f}+-{np.std(fold_f1):.4f}")
    return dict(macro_f1=float(f1), acc=float((p == labels).mean()),
                recall=[float(r) for r in rec],
                fold_f1=[float(v) for v in fold_f1],
                confusion=confusion_matrix(labels, p).tolist())


def part_a():
    print("=" * 78)
    print("PART A | logistic regression on scalars, A6e subject-grouped folds")
    print("  chance macro-F1 is about 0.33 (three classes)\n")
    Xm = mask_scalars()
    print("  mask scalars by class (mean, population SD):")
    for j, f in enumerate(MASK_FIELDS):
        s = "   ".join(f"{CLASSES[c][:4]} {Xm[labels == c, j].mean():.4f}"
                       f" ({Xm[labels == c, j].std():.4f})" for c in range(3))
        print(f"    {f:<7} {s}")
    print()
    Xho = head_scalars(occluded=True)
    Xhi = head_scalars(occluded=False)
    res = {}
    print("  A1 | annotation only (no pixels)")
    res["A1_mask_5"] = lr_cv(Xm, "mask: 5 scalars")
    res["A1_mask_area"] = lr_cv(Xm[:, [0]], "mask: area only")
    res["A1_mask_centroid"] = lr_cv(Xm[:, 1:3], "mask: centroids only")
    res["A1_mask_size"] = lr_cv(Xm[:, [0, 3, 4]], "mask: area+height+width")
    print("\n  A2 | head geometry, occluded images (paper: 0.596 macro-F1)")
    res["A2_head_occluded"] = lr_cv(Xho, "head: 4 scalars, occluded")
    print("\n  A3 | head geometry, intact images (no hole)")
    res["A3_head_intact"] = lr_cv(Xhi, "head: 4 scalars, intact")
    print("\n  A4 | does the hole add information to the head outline?")
    res["A4_head_intact+mask"] = lr_cv(np.hstack([Xhi, Xm]), "head (intact) + mask 5")
    d = res["A4_head_intact+mask"]["macro_f1"] - res["A3_head_intact"]["macro_f1"]
    print(f"\n  gain from adding mask scalars to head outline: {100 * d:+.1f} F1 points")
    print(f"  head occluded - head intact: "
          f"{100 * (res['A2_head_occluded']['macro_f1'] - res['A3_head_intact']['macro_f1']):+.1f}"
          " F1 points\n")
    return res


# =============================================================== Part B
def make_x(idx, mode, size):
    m = torch.from_numpy(masks[idx]).float()[:, None]
    m = (F.interpolate(m, (size, size), mode="nearest") > 0.5).float()
    if mode == "mask_only":
        return m
    x = torch.from_numpy(imgs[idx]).float().div_(255)[:, None]
    x = F.interpolate(x, (size, size), mode="bilinear", align_corners=False)
    if mode == "occlude":
        return x * (1 - m)
    raise ValueError(mode)


def train(tr, arch, mode, seed, dev):
    set_seed(seed)
    cfg = ARCHS[arch]
    x = make_x(tr, mode, cfg["size"]).to(dev)
    y = torch.from_numpy(labels[tr]).to(dev)
    net = M.build(cfg["name"], **cfg["kw"]).to(dev)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=LR, weight_decay=WD)
    cnt = torch.bincount(y, minlength=3).float()
    lossf = nn.CrossEntropyLoss(weight=(cnt.sum() / (3 * cnt.clamp(min=1))).to(dev))
    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH, shuffle=True)
    net.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
    del x, y
    return net


@torch.no_grad()
def predict(net, idx, arch, mode, dev):
    net.eval()
    size = ARCHS[arch]["size"]
    P = [net(make_x(idx[s:s + 64], mode, size).to(dev)).argmax(1).cpu()
         for s in range(0, len(idx), 64)]
    return torch.cat(P).numpy()


def run_condition(arch, mode, dev):
    """Per-fold macro-F1 (mean over 3 seeds), per-image correctness (mean over
    seeds), pooled predictions of seed 1 for per-class recall."""
    fold_f1, correct, pred1 = [], np.zeros(len(labels)), np.zeros_like(labels)
    for f, (tr, te) in enumerate(SPLITS):
        s_f1, s_c = [], []
        for sd in SEEDS:
            t0 = time.time()
            net = train(tr, arch, mode, sd, dev)
            p = predict(net, te, arch, mode, dev)
            s_f1.append(f1_score(labels[te], p, average="macro"))
            s_c.append((p == labels[te]).astype(float))
            if sd == SEEDS[0]:
                pred1[te] = p
            del net
            if dev == "cuda":
                torch.cuda.empty_cache()
            print(f"    {arch:<12} {mode:<9} fold {f} seed {sd}  "
                  f"F1 {s_f1[-1]:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        fold_f1.append(float(np.mean(s_f1)))
        correct[te] = np.mean(s_c, axis=0)
    return fold_f1, correct, pred1


def bootstrap_paired(a, b, n_boot=N_BOOT, seed=0):
    """Patient-clustered bootstrap of mean(a - b) over images, resampling
    subjects (identical to A6e.bootstrap_paired)."""
    rng = np.random.default_rng(seed)
    keys = np.unique(pids)
    diff = a - b
    means = np.array([diff[pids == k].mean() for k in keys])
    ns = np.array([(pids == k).sum() for k in keys], dtype=float)
    obs = float((means * ns).sum() / ns.sum())
    boots = []
    for _ in range(n_boot):
        s = rng.integers(0, len(keys), len(keys))
        boots.append((means[s] * ns[s]).sum() / ns[s].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


def part_b(archs):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("  WARNING: no GPU found -- Part B will be very slow. "
              "Enable a GPU accelerator in Kaggle settings.")
    print("=" * 78)
    print(f"PART B | CNN on the mask alone vs occluded image, A6e recipe ({dev})\n")
    res = {}
    for arch in archs:
        r = {}
        for mode in ("mask_only", "occlude"):
            f1s, corr, p1 = run_condition(arch, mode, dev)
            rec = recall_score(labels, p1, average=None)
            r[mode] = dict(fold_f1=f1s, mean_f1=float(np.mean(f1s)),
                           sd_f1_pop=float(np.std(f1s)),
                           pooled_acc=float(corr.mean()),
                           recall_seed1=[float(v) for v in rec])
            r[mode + "_correct"] = corr
        occ, mo = r["occlude"], r["mask_only"]
        obs, lo, hi = bootstrap_paired(r["occlude_correct"], r["mask_only_correct"])
        r["delta_acc_occ_minus_mask"] = dict(obs=obs, ci=[lo, hi])
        rep = np.array(occ["fold_f1"]) - np.array(A6E_OCC[arch])
        r["occlude_vs_a6e_max_abs_fold_diff"] = float(np.abs(rep).max())
        print(f"\n  {arch}")
        print(f"    mask-only  macro-F1 {mo['mean_f1']:.4f} +- {mo['sd_f1_pop']:.4f}"
              f"   pooled acc {mo['pooled_acc']:.4f}"
              f"   per-fold {[round(v, 4) for v in mo['fold_f1']]}")
        print(f"    occluded   macro-F1 {occ['mean_f1']:.4f} +- {occ['sd_f1_pop']:.4f}"
              f"   pooled acc {occ['pooled_acc']:.4f}"
              f"   per-fold {[round(v, 4) for v in occ['fold_f1']]}")
        print(f"    A6e occluded per-fold {A6E_OCC[arch]}  "
              f"(max |diff| {r['occlude_vs_a6e_max_abs_fold_diff']:.4f})")
        print(f"    recall g/m/p (seed 1): mask-only "
              f"{'/'.join(f'{v:.2f}' for v in mo['recall_seed1'])}   occluded "
              f"{'/'.join(f'{v:.2f}' for v in occ['recall_seed1'])}")
        print(f"    acc(occluded) - acc(mask-only) = {100 * obs:+.2f} points "
              f"(95% patient CI {100 * lo:+.2f}, {100 * hi:+.2f})")
        verdict = ("hole alone does NOT account for the occluded result"
                   if lo > 0 else
                   "hole alone may account for the occluded result -- size matching needed")
        print(f"    -> {verdict}\n")
        for k in ("mask_only_correct", "occlude_correct"):
            r.pop(k)
        res[arch] = r
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "AB"], default="AB")
    ap.add_argument("--arch", choices=["smallcnn", "resnet18-l4b", "both"],
                    default="smallcnn")
    a = ap.parse_args()
    print(f"A6l | {len(labels)} slices, {len(np.unique(pids))} subjects, "
          f"class counts {np.bincount(labels).tolist()}\n")
    out = {}
    if "A" in a.part:
        out["part_a"] = part_a()
    if "B" in a.part:
        archs = list(ARCHS) if a.arch == "both" else [a.arch]
        out["part_b"] = part_b(archs)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"saved -> {OUT}")
