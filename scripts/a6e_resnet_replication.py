"""A6e — does the shortcut hold for a pretrained ResNet, and does it survive
the statistical bar Bassi (2026) set?

A6d used SmallCNN (0.24M, from scratch). A reviewer will call that a toy model.
This repeats the occlusion experiment with ImageNet-pretrained ResNet-18
(layer4b trainable, 4.72M) and adds the statistics Bassi's audit uses:

  - patient-clustered bootstrap CI on the paired difference (resample SUBJECTS)
  - one-way random-effects variance decomposition (partition vs seed)
  - paired Wilcoxon across folds

Single-split differences of one to two points are hard to interpret on this
benchmark. The paired design (identical folds and seeds, conditions differ only in
masking) and the patient-clustered interval are what allow small contrasts to be
judged. Output: a6e_results.json (manuscript Tables 1-2, Fig. 2).
"""
import os, json, itertools
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon
from core import data_mat, model as M
from core.seeding import set_seed

DEV = "cuda"
EPOCHS, BATCH, LR, WD = 30, 32, 1e-3, 1e-4
SEEDS = [1, 2, 3]
N_BOOT = 2000
OUT = "/kaggle/working/a6e_results.json"

imgs, masks, labels, pids = data_mat.load_arrays()
SPLITS = list(StratifiedGroupKFold(5, shuffle=True, random_state=0)
              .split(imgs, labels, groups=pids))

ARCHS = {
    "resnet18-l4b": dict(name="resnet18", size=224,
                         kw=dict(pretrained=True, trainable="layer4b")),
    "smallcnn":     dict(name="smallcnn", size=128, kw=dict(in_size=128)),
}
CONDS = [("intact", "intact", 0), ("occlude", "occlude", 0),
         ("occlude+10", "occlude", 10), ("tumor_only", "tumor_only", 0)]


def make_x(idx, mode, size, dilate_px=0):
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


def train(tr_idx, arch, mode, dilate_px, seed):
    set_seed(seed)
    cfg = ARCHS[arch]
    x = make_x(tr_idx, mode, cfg["size"], dilate_px).to(DEV)
    y = torch.from_numpy(labels[tr_idx]).to(DEV)
    net = M.build(cfg["name"], **cfg["kw"]).to(DEV)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=LR, weight_decay=WD)
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
def predict(net, idx, arch, mode, dilate_px):
    net.eval()
    size = ARCHS[arch]["size"]
    P = []
    for s in range(0, len(idx), 64):
        b = idx[s:s + 64]
        P.append(net(make_x(b, mode, size, dilate_px).to(DEV)).argmax(1).cpu())
    return torch.cat(P).numpy()


def run_condition(arch, mode, dilate_px):
    """Returns per-fold F1 (len 5) and per-image correctness keyed by subject."""
    fold_f1, correct, subj = [], {}, {}
    for f, (tr, te) in enumerate(SPLITS):
        s_f1, s_corr = [], []
        for sd in SEEDS:
            net = train(tr, arch, mode, dilate_px, sd)
            p = predict(net, te, arch, mode, dilate_px)
            s_f1.append(f1_score(labels[te], p, average="macro"))
            s_corr.append((p == labels[te]).astype(float))
            del net; torch.cuda.empty_cache()
        fold_f1.append(float(np.mean(s_f1)))
        for j, i in enumerate(te):                   # mean correctness over seeds
            correct[int(i)] = float(np.mean([c[j] for c in s_corr]))
            subj[int(i)] = str(pids[i])
    return fold_f1, correct, subj


def bootstrap_paired(corr_a, corr_b, subj, n_boot=N_BOOT, seed=0):
    """Patient-clustered bootstrap on the paired accuracy difference.
    Resamples SUBJECTS with replacement (Bassi 2026, Methods 4.5)."""
    rng = np.random.default_rng(seed)
    by_subj = {}
    for i in corr_a:
        by_subj.setdefault(subj[i], []).append(corr_a[i] - corr_b[i])
    keys = list(by_subj)
    means = np.array([np.mean(by_subj[k]) for k in keys])
    ns = np.array([len(by_subj[k]) for k in keys], dtype=float)
    obs = float((means * ns).sum() / ns.sum())
    boots = []
    for _ in range(n_boot):
        s = rng.integers(0, len(keys), len(keys))
        boots.append((means[s] * ns[s]).sum() / ns[s].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


def variance_components(f1_by_fold_seed):
    """One-way random-effects MoM decomposition of the fold x seed table.
    Negative partition estimates truncate at zero (boundary estimate)."""
    a = np.asarray(f1_by_fold_seed)          # (folds, seeds)
    k, n = a.shape
    grand = a.mean()
    ms_between = n * ((a.mean(1) - grand) ** 2).sum() / (k - 1)
    ms_within = ((a - a.mean(1, keepdims=True)) ** 2).sum() / (k * (n - 1))
    var_part = max(0.0, (ms_between - ms_within) / n)
    return float(np.sqrt(var_part)), float(np.sqrt(ms_within))


if __name__ == "__main__":
    res = {}
    for arch in ["resnet18-l4b", "smallcnn"]:
        print(f"\n{'='*66}\n{arch}  ({ARCHS[arch]['size']}px, {len(SEEDS)} seeds x 5 folds)\n")
        store = {}
        for lab, mode, d in CONDS:
            f1s, corr, subj = run_condition(arch, mode, d)
            store[lab] = (f1s, corr, subj)
            print(f"  {lab:<12} F1 {np.mean(f1s):.4f} +- {np.std(f1s):.4f}   "
                  f"per-fold {[f'{v:.3f}' for v in f1s]}")

        base_f1, base_corr, subj = store["intact"]
        print(f"\n  paired vs intact (patient-clustered bootstrap, {N_BOOT} reps):")
        res[arch] = {"intact_f1": base_f1}
        for lab in ["occlude", "occlude+10", "tumor_only"]:
            f1s, corr, _ = store[lab]
            obs, lo, hi = bootstrap_paired(corr, base_corr, subj)
            try:
                _, p = wilcoxon(f1s, base_f1); ps = f"{p:.3f}"
            except ValueError:
                ps = "n/a"
            brack = "brackets 0" if lo <= 0 <= hi else "excludes 0"
            print(f"    {lab:<12} dAcc {100*obs:+.2f} pts  "
                  f"95% CI [{100*lo:+.2f}, {100*hi:+.2f}]  {brack}   "
                  f"dF1 {100*(np.mean(f1s)-np.mean(base_f1)):+.2f}  wilcoxon p={ps}")
            res[arch][lab] = {"f1": f1s, "dacc": obs, "ci": [lo, hi]}

        # variance decomposition on the intact arm
        tab = []
        for f, (tr, te) in enumerate(SPLITS):
            row = []
            for sd in SEEDS:
                net = train(tr, arch, "intact", 0, sd)
                p = predict(net, te, arch, "intact", 0)
                row.append(f1_score(labels[te], p, average="macro"))
                del net; torch.cuda.empty_cache()
            tab.append(row)
        sp, ss = variance_components(tab)
        print(f"\n  variance (intact): partition sd {100*sp:.2f} pts, "
              f"seed sd {100*ss:.2f} pts, ratio {sp/max(ss,1e-9):.1f}x")
        res[arch]["variance"] = {"partition_sd": sp, "seed_sd": ss}

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"\nsaved {OUT}")
