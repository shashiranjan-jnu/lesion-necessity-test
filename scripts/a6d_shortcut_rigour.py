"""A6d — three controls to make the A6c shortcut result submittable.

A6c: intact 0.751, tumor-occluded 0.785, tumor-only 0.759 (5-fold, 1 seed).
Both signal paths are independently sufficient. Three things needed:

  E4  STATISTICS  3 seeds x 5 folds, paired Wilcoxon per fold vs intact.
                  The claim is "occlusion does not hurt", not "occlusion helps".
  E5  WHAT IS THE CONTEXT SIGNAL?  train occluded at 128/64/32/16/8 px.
                  If 16px still works, the shortcut is gross slice geometry,
                  not texture.
  E6  VALIDATE THE ATTRIBUTION PIPELINE.  Run the A6 XAI metrics on the
                  TUMOR-ONLY model. Energy-in-mask must jump from 0.007 toward
                  1.0 -- the model can see nothing else. If it does not, the
                  attribution code is broken and A6 must be rewritten.
"""
import os, json, itertools
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon
from core import data_mat, model as M, xai
from core.seeding import set_seed

DEV = "cuda"
EPOCHS, BATCH, LR, WD = 30, 32, 1e-3, 1e-4
SEEDS = [1, 2, 3]
CKPT = "/kaggle/working/ckpt/a6d"
os.makedirs(CKPT, exist_ok=True)

imgs, masks, labels, pids = data_mat.load_arrays()
SPLITS = list(StratifiedGroupKFold(5, shuffle=True, random_state=0)
              .split(imgs, labels, groups=pids))


def make_x(idx, mode, size=128, dilate_px=0):
    x = torch.from_numpy(imgs[idx]).float().div_(255)[:, None]
    x = F.interpolate(x, (size, size), mode="bilinear", align_corners=False)
    if mode != "intact":
        m = torch.from_numpy(masks[idx]).float()[:, None]
        if dilate_px:
            k = 2 * dilate_px + 1
            m = (F.max_pool2d(m, k, stride=1, padding=dilate_px) > 0).float()
        m = (F.interpolate(m, (size, size), mode="nearest") > 0.5).float()
        x = x * (1 - m) if mode == "occlude" else x * m
    return x


def train(tr_idx, mode, size=128, dilate_px=0, seed=1):
    set_seed(seed)
    x = make_x(tr_idx, mode, size, dilate_px).to(DEV)
    y = torch.from_numpy(labels[tr_idx]).to(DEV)
    net = M.build("smallcnn", in_size=size).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    cnt = torch.bincount(y, minlength=3).float()
    lossf = nn.CrossEntropyLoss(weight=(cnt.sum() / (3 * cnt.clamp(min=1))).to(DEV))
    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH, shuffle=True)
    net.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
    del x, y; torch.cuda.empty_cache()
    return net


@torch.no_grad()
def score(net, idx, mode, size=128, dilate_px=0):
    net.eval()
    P = []
    for s in range(0, len(idx), 128):
        b = idx[s:s + 128]
        P.append(net(make_x(b, mode, size, dilate_px).to(DEV)).argmax(1).cpu())
    p, y = torch.cat(P).numpy(), labels[idx]
    return f1_score(y, p, average="macro")


def run_cv(mode, size=128, dilate_px=0, seeds=SEEDS, save_tag=None):
    """Returns per-fold mean F1 across seeds (length 5)."""
    per_fold = []
    for f, (tr, te) in enumerate(SPLITS):
        s_vals = []
        for sd in seeds:
            net = train(tr, mode, size, dilate_px, sd)
            s_vals.append(score(net, te, mode, size, dilate_px))
            if save_tag and f == 2 and sd == 1:
                torch.save(net.state_dict(), f"{CKPT}/{save_tag}.pt")
            del net; torch.cuda.empty_cache()
        per_fold.append(float(np.mean(s_vals)))
    return per_fold


# ------------------------------------------------------------------ E6 helper
def xai_on(net, size, eval_idx, mode):
    """Attribution metrics for a model trained on `mode` inputs."""
    net.eval()
    layer = net.last_conv
    acc = {k: [] for k in ("point", "energy", "iou")}
    for i in eval_idx:
        x1 = make_x(np.array([i]), mode, size).to(DEV)
        x1.requires_grad_(True)
        with torch.no_grad():
            pred = int(net(x1).argmax(1).item())
        m = torch.from_numpy(masks[i].astype(np.float32))[None, None]
        m = (F.interpolate(m, (size, size), mode="nearest").squeeze().numpy() > 0.5).astype(np.uint8)
        a = xai.gradcam(net, x1, pred, layer)[0]
        acc["point"].append(xai.pointing_game(a, m))
        acc["energy"].append(xai.energy_in_mask(a, m))
        acc["iou"].append(xai.iou_at_mask_area(a, m))
    return {k: float(np.mean(v)) for k, v in acc.items()}


if __name__ == "__main__":
    res = {}

    # ---------------------------------------------------------------- E4
    print("E4 | STATISTICS: 3 seeds x 5 folds, paired Wilcoxon vs intact\n")
    conds = [("intact", "intact", 128, 0, None),
             ("occlude", "occlude", 128, 0, "occlude_f2"),
             ("occlude+10", "occlude", 128, 10, None),
             ("tumor_only", "tumor_only", 128, 0, "tumoronly_f2")]
    folds = {}
    for lab, mode, size, d, tag in conds:
        folds[lab] = run_cv(mode, size, d, save_tag=tag)
        print(f"  {lab:<12} F1 {np.mean(folds[lab]):.4f} +- {np.std(folds[lab]):.4f}   "
              f"per-fold {[f'{v:.3f}' for v in folds[lab]]}")
    res["E4"] = folds

    print("\n  paired Wilcoxon (n=5 folds) vs intact:")
    for lab in ["occlude", "occlude+10", "tumor_only"]:
        d = np.array(folds[lab]) - np.array(folds["intact"])
        try:
            stat, p = wilcoxon(folds[lab], folds["intact"])
            pstr = f"p={p:.3f}"
        except ValueError:
            pstr = "p=n/a"
        print(f"    {lab:<12} mean delta {100*d.mean():+.1f} pts   {pstr}   "
              f"{'NOT significant' if 'n/a' in pstr or p > 0.05 else 'significant'}")

    # ---------------------------------------------------------------- E5
    print("\n" + "=" * 62)
    print("E5 | WHAT IS THE CONTEXT SIGNAL? occluded models at falling resolution\n")
    res["E5"] = {}
    for size in [128, 64, 32, 16, 8]:
        v = run_cv("occlude", size=size, seeds=[1, 2])
        res["E5"][size] = v
        print(f"  {size:>4}px  F1 {np.mean(v):.4f} +- {np.std(v):.4f}")
    print("\n  (if 16px still works, the shortcut is gross slice geometry /")
    print("   anatomical level, not fine texture)")

    # ---------------------------------------------------------------- E6
    print("\n" + "=" * 62)
    print("E6 | VALIDATE ATTRIBUTION PIPELINE on the tumor-only model")
    print("    energy-in-mask MUST approach 1.0 -- the model sees nothing else.")
    print("    A6 intact reference: point 0.020, energy 0.007, chance 0.016\n")
    rng = np.random.default_rng(0)
    eval_idx = rng.choice(SPLITS[2][1], size=120, replace=False)
    for tag, mode in [("tumoronly_f2", "tumor_only"), ("occlude_f2", "occlude")]:
        p = f"{CKPT}/{tag}.pt"
        if not os.path.exists(p):
            print(f"  {tag}: checkpoint missing"); continue
        net = M.build("smallcnn", in_size=128).to(DEV)
        net.load_state_dict(torch.load(p))
        m = xai_on(net, 128, eval_idx, mode)
        res[f"E6_{tag}"] = m
        print(f"  {tag:<14} gradcam  point {m['point']:.3f}  "
              f"energy {m['energy']:.3f}  IoU {m['iou']:.3f}")
        del net; torch.cuda.empty_cache()

    with open("/kaggle/working/a6d_results.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print("\nsaved /kaggle/working/a6d_results.json")
