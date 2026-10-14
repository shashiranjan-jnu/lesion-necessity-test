"""H1 -- lesion-necessity test on HAM10000 (melanoma vs nevus).

Same protocol as a6e on the brain collection, with four differences forced by
the data:

  * RGB input (colour is diagnostic in dermoscopy), so the pretrained ResNet
    stem is kept rather than averaged to one channel.
  * Binary task, so accuracy and macro-F1 coincide more closely; AUROC is
    reported as well.
  * Grouping is by lesion_id (1.8 images per lesion) rather than by subject.
  * Restricted to the area-matched subset (scripts/h0b_areamatch.py). Without
    matching, the annotation alone predicts the class at 0.615 macro-F1, so the
    hole in the occluded image would carry the label. After matching the
    mask-only floor is ~0.52, and that is the number the occluded arm must be
    compared against -- not 0.50.

Expectation: unlike the brain collection, the lesion should be NECESSARY here,
i.e. the interval for the occluded contrast should exclude zero and the loss
should be large. That is the positive control for the protocol itself.

Also computes the paired per-fold/seed spread of the accuracy difference, which
the bootstrap interval does not cover (it averages over seeds before
resampling).
"""
import json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import wilcoxon
from core import data_ham, model as M
from core.seeding import set_seed

DEV = "cuda"
EPOCHS, BATCH, LR, WD = 30, 32, 1e-3, 1e-4
SEEDS = [1, 2, 3]
N_BOOT = 2000
MATCHED = "/kaggle/working/ham_matched_ids.npy"
OUT = "/kaggle/working/h1_results.json"

imgs, masks, labels, gids = data_ham.load_arrays()
keep = np.isin(gids, np.load(MATCHED, allow_pickle=True).astype(str))
imgs, masks, labels, gids = imgs[keep], masks[keep], labels[keep], gids[keep]
print(f"matched subset: {len(labels)} images, {len(np.unique(gids))} lesions, "
      f"mel {int(labels.sum())} / nv {int((1 - labels).sum())}")

SPLITS = list(StratifiedGroupKFold(5, shuffle=True, random_state=0)
              .split(imgs, labels, groups=gids))

ARCHS = {
    "resnet18-l4b": dict(name="resnet18", size=224,
                         kw=dict(pretrained=True, trainable="layer4b",
                                 n_classes=2, in_ch=3)),
    "smallcnn":     dict(name="smallcnn", size=128,
                         kw=dict(in_size=128, n_classes=2, in_ch=3)),
}
CONDS = [("intact", "intact", 0), ("occlude", "occlude", 0),
         ("occlude+10", "occlude", 10), ("lesion_only", "tumor_only", 0)]


def make_x(idx, mode, size, dilate_px=0):
    x = torch.from_numpy(imgs[idx]).float().div_(255).permute(0, 3, 1, 2)
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
    cnt = torch.bincount(y, minlength=2).float()
    lossf = nn.CrossEntropyLoss(weight=(cnt.sum() / (2 * cnt.clamp(min=1))).to(DEV))
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
        P.append(torch.softmax(net(make_x(b, mode, size, dilate_px).to(DEV)), 1).cpu())
    return torch.cat(P).numpy()


def run_condition(arch, mode, dilate_px):
    """fold F1 (5), fold x seed accuracy (5x3), per-image correctness, group map."""
    fold_f1, acc_tab, correct, grp = [], [], {}, {}
    for f, (tr, te) in enumerate(SPLITS):
        s_f1, s_acc, s_corr = [], [], []
        for sd in SEEDS:
            net = train(tr, arch, mode, dilate_px, sd)
            pr = predict(net, te, arch, mode, dilate_px)
            p = pr.argmax(1)
            s_f1.append(f1_score(labels[te], p, average="macro"))
            s_acc.append(float((p == labels[te]).mean()))
            s_corr.append((p == labels[te]).astype(float))
            del net; torch.cuda.empty_cache()
        fold_f1.append(float(np.mean(s_f1))); acc_tab.append(s_acc)
        for j, i in enumerate(te):
            correct[int(i)] = float(np.mean([c[j] for c in s_corr]))
            grp[int(i)] = str(gids[i])
    return fold_f1, np.array(acc_tab), correct, grp


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
    for arch in ["resnet18-l4b", "smallcnn"]:
        print(f"\n{'='*66}\n{arch}  ({ARCHS[arch]['size']}px, "
              f"{len(SEEDS)} seeds x 5 folds)\n")
        store = {}
        for lab, mode, d in CONDS:
            f1s, acc, corr, grp = run_condition(arch, mode, d)
            store[lab] = (f1s, acc, corr, grp)
            print(f"  {lab:<12} F1 {np.mean(f1s):.4f} +- {np.std(f1s):.4f}   "
                  f"acc {acc.mean():.4f}   per-fold {[f'{v:.3f}' for v in f1s]}")

        base_f1, base_acc, base_corr, grp = store["intact"]
        sp, ss = variance_components(base_acc)
        print(f"\n  variance (intact): partition sd {100*sp:.2f} pts, "
              f"seed sd {100*ss:.2f} pts")
        print(f"\n  paired vs intact (lesion-clustered bootstrap, {N_BOOT} reps):")
        res[arch] = {"intact_f1": base_f1,
                     "variance": {"partition_sd": sp, "seed_sd": ss}}
        for lab in ["occlude", "occlude+10", "lesion_only"]:
            f1s, acc, corr, _ = store[lab]
            obs, lo, hi = bootstrap_paired(corr, base_corr, grp)
            pair = 100 * (acc - base_acc)          # paired, same fold AND seed
            try:
                _, p = wilcoxon(f1s, base_f1); ps = f"{p:.3f}"
            except ValueError:
                ps = "n/a"
            print(f"    {lab:<12} dAcc {100*obs:+.2f} pts  "
                  f"95% CI [{100*lo:+.2f}, {100*hi:+.2f}]  "
                  f"{'brackets 0' if lo <= 0 <= hi else 'excludes 0'}   "
                  f"paired sd {pair.std(ddof=1):.2f}  wilcoxon p={ps}")
            res[arch][lab] = {"f1": f1s, "dacc": obs, "ci": [lo, hi],
                              "paired_sd": float(pair.std(ddof=1))}

        print("\n  mask-only floor after area matching: ~0.52 macro-F1 "
              "(scripts/h0b_areamatch.py). Read the occluded row against that.")

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"\nsaved {OUT}")
