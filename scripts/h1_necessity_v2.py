"""H1 -- lesion-necessity test on HAM10000 (melanoma vs nevus). Version 2.

v1 used the brain-collection recipe (GroupNorm ResNet, layer4b trainable, /255
scaling, no augmentation) and reached only 0.601 intact accuracy, so every
contrast was one near-chance model against another. H1a calibration on one fold:

  A  GN, layer4b, /255, no aug          acc 0.668  AUROC 0.724   (v1 recipe)
  B  BN, layer4, ImageNet norm          acc 0.731  AUROC 0.814
  C  BN, full fine-tune, lr 1e-4        acc 0.754  AUROC 0.822
  D  C + flips and rot90                acc 0.784  AUROC 0.861   <- adopted

The GroupNorm constraint exists for the DP-federated arm, not here, so stock
BatchNorm with its pretrained statistics is available. Hyperparameters differ
from the brain arm by design: the necessity test is a within-dataset paired
contrast, so what transfers is the protocol, not the learning rate.

Read the occluded row against the ~0.52 mask-only floor (h0b_areamatch.py),
not against 0.50.
"""
import json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import wilcoxon
from core import data_ham
from core.seeding import set_seed

DEV = "cuda" if torch.cuda.is_available() else "cpu"
assert DEV == "cuda", "no GPU: Session options -> Accelerator -> GPU"
EPOCHS, BATCH, WD = 30, 32, 1e-4
SEEDS = [1, 2, 3]
N_BOOT = 2000
MATCHED = "/kaggle/working/ham_matched_ids.npy"
OUT = "/kaggle/working/h1_results_v2.json"

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

imgs, masks, labels, gids = data_ham.load_arrays()
keep = np.isin(gids, np.load(MATCHED, allow_pickle=True).astype(str))
imgs, masks, labels, gids = imgs[keep], masks[keep], labels[keep], gids[keep]
print(f"matched subset: {len(labels)} images, {len(np.unique(gids))} lesions, "
      f"mel {int(labels.sum())} / nv {int((1 - labels).sum())}")

SPLITS = list(StratifiedGroupKFold(5, shuffle=True, random_state=0)
              .split(imgs, labels, groups=gids))

ARCHS = {
    "resnet18-bn": dict(size=224, lr=1e-4, norm=True),
    "smallcnn":    dict(size=128, lr=1e-3, norm=False),
}
CONDS = [("intact", "intact", 0), ("occlude", "occlude", 0),
         ("occlude+10", "occlude", 10), ("lesion_only", "tumor_only", 0)]


def build_net(arch):
    if arch == "resnet18-bn":
        from torchvision.models import resnet18, ResNet18_Weights
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(512, 2)          # full fine-tune, no freezing
        return m
    from core import model as M
    return M.build("smallcnn", n_classes=2, in_ch=3, in_size=128)


def make_x(idx, mode, arch, dilate_px=0):
    cfg = ARCHS[arch]
    x = torch.from_numpy(imgs[idx]).float().div_(255).permute(0, 3, 1, 2)
    x = F.interpolate(x, (cfg["size"],) * 2, mode="bilinear", align_corners=False)
    if mode != "intact":
        m = torch.from_numpy(masks[idx]).float()[:, None]
        if dilate_px:
            k = 2 * dilate_px + 1
            m = (F.max_pool2d(m, k, stride=1, padding=dilate_px) > 0).float()
        m = (F.interpolate(m, (cfg["size"],) * 2, mode="nearest") > 0.5).float()
        x = x * (1 - m) if mode == "occlude" else x * m
    if cfg["norm"]:
        x = (x - MEAN) / STD               # after masking: zeroed pixels shift too
    return x


def augment(xb):
    """Dermoscopy has no canonical orientation."""
    if torch.rand(1).item() < 0.5:
        xb = torch.flip(xb, [3])
    if torch.rand(1).item() < 0.5:
        xb = torch.flip(xb, [2])
    return torch.rot90(xb, int(torch.randint(0, 4, (1,)).item()), [2, 3])


def train(tr_idx, arch, mode, dilate_px, seed):
    set_seed(seed)
    x = make_x(tr_idx, mode, arch, dilate_px).to(DEV)
    y = torch.from_numpy(labels[tr_idx]).to(DEV)
    net = build_net(arch).to(DEV)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=ARCHS[arch]["lr"], weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    cnt = torch.bincount(y, minlength=2).float()
    lossf = nn.CrossEntropyLoss(weight=(cnt.sum() / (2 * cnt.clamp(min=1))).to(DEV))
    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH, shuffle=True)
    for _ in range(EPOCHS):
        net.train()
        for xb, yb in loader:
            xb = augment(xb)
            opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
        sched.step()
    del x, y; torch.cuda.empty_cache()
    return net


@torch.no_grad()
def predict(net, idx, arch, mode, dilate_px):
    net.eval()
    P = []
    x = make_x(idx, mode, arch, dilate_px)
    for s in range(0, len(idx), 64):
        P.append(torch.softmax(net(x[s:s + 64].to(DEV)), 1).cpu())
    return torch.cat(P).numpy()


def run_condition(arch, mode, dilate_px):
    fold_f1, auc, acc_tab, correct, grp = [], [], [], {}, {}
    for f, (tr, te) in enumerate(SPLITS):
        s_f1, s_auc, s_acc, s_corr = [], [], [], []
        for sd in SEEDS:
            net = train(tr, arch, mode, dilate_px, sd)
            pr = predict(net, te, arch, mode, dilate_px)
            p = pr.argmax(1)
            s_f1.append(f1_score(labels[te], p, average="macro"))
            s_auc.append(roc_auc_score(labels[te], pr[:, 1]))
            s_acc.append(float((p == labels[te]).mean()))
            s_corr.append((p == labels[te]).astype(float))
            del net; torch.cuda.empty_cache()
        fold_f1.append(float(np.mean(s_f1))); auc.append(float(np.mean(s_auc)))
        acc_tab.append(s_acc)
        for j, i in enumerate(te):
            correct[int(i)] = float(np.mean([c[j] for c in s_corr]))
            grp[int(i)] = str(gids[i])
    return fold_f1, auc, np.array(acc_tab), correct, grp


def bootstrap_paired(corr_a, corr_b, grp, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    by = {}
    for i in corr_a:
        by.setdefault(grp[i], []).append(corr_a[i] - corr_b[i])
    keys = list(by)
    means = np.array([np.mean(by[k]) for k in keys])
    ns = np.array([len(by[k]) for k in keys], float)
    obs = float((means * ns).sum() / ns.sum())
    boots = [(means[s] * ns[s]).sum() / ns[s].sum()
             for s in (rng.integers(0, len(keys), len(keys)) for _ in range(n_boot))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


def variance_components(tab):
    a = np.asarray(tab); k, n = a.shape
    ms_b = n * ((a.mean(1) - a.mean()) ** 2).sum() / (k - 1)
    ms_w = ((a - a.mean(1, keepdims=True)) ** 2).sum() / (k * (n - 1))
    return float(np.sqrt(max(0.0, (ms_b - ms_w) / n))), float(np.sqrt(ms_w))


if __name__ == "__main__":
    res = {}
    for arch in ARCHS:
        print(f"\n{'='*70}\n{arch}  ({ARCHS[arch]['size']}px, lr {ARCHS[arch]['lr']}, "
              f"{len(SEEDS)} seeds x 5 folds, flips+rot90)\n")
        store = {}
        for lab, mode, d in CONDS:
            f1s, auc, acc, corr, grp = run_condition(arch, mode, d)
            store[lab] = (f1s, auc, acc, corr, grp)
            print(f"  {lab:<12} F1 {np.mean(f1s):.4f} +- {np.std(f1s):.4f}   "
                  f"acc {acc.mean():.4f}   AUROC {np.mean(auc):.4f}   "
                  f"per-fold {[f'{v:.3f}' for v in f1s]}")

        base_f1, base_auc, base_acc, base_corr, grp = store["intact"]
        sp, ss = variance_components(base_acc)
        print(f"\n  variance (intact): partition sd {100*sp:.2f} pts, "
              f"seed sd {100*ss:.2f} pts")
        print(f"\n  paired vs intact (lesion-clustered bootstrap, {N_BOOT} reps):")
        res[arch] = {"intact_f1": base_f1, "intact_auc": base_auc,
                     "variance": {"partition_sd": sp, "seed_sd": ss}}
        for lab in ["occlude", "occlude+10", "lesion_only"]:
            f1s, auc, acc, corr, _ = store[lab]
            obs, lo, hi = bootstrap_paired(corr, base_corr, grp)
            pair = 100 * (acc - base_acc)
            try:
                _, p = wilcoxon(f1s, base_f1); ps = f"{p:.3f}"
            except ValueError:
                ps = "n/a"
            print(f"    {lab:<12} dAcc {100*obs:+.2f} pts  "
                  f"95% CI [{100*lo:+.2f}, {100*hi:+.2f}]  "
                  f"{'brackets 0' if lo <= 0 <= hi else 'excludes 0'}   "
                  f"paired sd {pair.std(ddof=1):.2f}  wilcoxon p={ps}")
            res[arch][lab] = {"f1": f1s, "auc": auc, "dacc": obs, "ci": [lo, hi],
                              "paired_sd": float(pair.std(ddof=1))}

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"\nsaved {OUT}")
