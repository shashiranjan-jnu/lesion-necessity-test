"""A3 (revised) — centralized baseline via subject-grouped 5-fold CV.

Single-split results showed +-7% variance driven by which subjects landed in
the test set, not by the model. CV over all 209 subjects removes that.
Gate: >= 88% mean CV accuracy (lowered from 90%; see A3 notes).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix
from core import data, model as M
from core.seeding import set_seed

NPZ, DEV = "/kaggle/working/cache/cheng224.npz", "cuda"
EPOCHS, BATCH, LR, WD = 30, 32, 1e-3, 1e-4
N_FOLDS = 5

imgs, labels, pids = data.load_arrays(NPZ)


def augment(x):
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3])
    ang = (torch.rand(1).item() * 20 - 10) * np.pi / 180
    tx, ty = (torch.rand(2) * 0.1 - 0.05).tolist()
    c, s = np.cos(ang), np.sin(ang)
    th = torch.tensor([[c, -s, tx], [s, c, ty]], dtype=torch.float, device=x.device)
    th = th.unsqueeze(0).expand(x.size(0), -1, -1)
    grid = F.affine_grid(th, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)


def tensors(idx, size):
    x = torch.from_numpy(imgs[idx]).float().div_(255).unsqueeze(1)
    if size != data.NATIVE:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x.to(DEV), torch.from_numpy(labels[idx]).to(DEV)


@torch.no_grad()
def evaluate(net, x, y):
    net.eval()
    P = [net(x[i:i+128]).argmax(1) for i in range(0, len(x), 128)]
    p = torch.cat(P).cpu().numpy(); yt = y.cpu().numpy()
    return (p == yt).mean(), f1_score(yt, p, average="macro"), p, yt


def run_fold(name, size, tr_idx, te_idx, init_seed, aug, **kw):
    set_seed(init_seed)
    xtr, ytr = tensors(tr_idx, size)
    xte, yte = tensors(te_idx, size)
    net = M.build(name, **kw).to(DEV)
    info = M.assert_dp_safe(net)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=LR, weight_decay=WD)
    lossf = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=BATCH, shuffle=True)

    for _ in range(EPOCHS):
        net.train()
        for xb, yb in loader:
            if aug: xb = augment(xb)
            opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()

    acc, f1, p, yt = evaluate(net, xte, yte)
    return acc, f1, confusion_matrix(yt, p), info["trainable"]


def cross_validate(label, name, size, aug, init_seed=1, **kw):
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    accs, f1s, cms, t0 = [], [], None, time.time()
    for f, (tr, te) in enumerate(skf.split(imgs, labels, groups=pids)):
        assert not (set(pids[tr]) & set(pids[te])), "SUBJECT LEAKAGE"
        a, s, cm, ntr = run_fold(name, size, tr, te, init_seed, aug, **kw)
        accs.append(a); f1s.append(s)
        cms = cm if cms is None else cms + cm
        print(f"    fold {f}: acc {a:.4f}  F1 {s:.4f}  (n_test={len(te)})")
    print(f"  {label:<22} acc {np.mean(accs):.4f} +- {np.std(accs):.4f}   "
          f"F1 {np.mean(f1s):.4f}   ({time.time()-t0:.0f}s, {ntr/1e6:.2f}M)")
    print(f"  pooled confusion (rows=true {data.CLASSES}):\n{cms}\n")
    return np.mean(accs), np.std(accs), np.mean(f1s)


CONFIGS = [
    ("smallcnn 128 noaug", "smallcnn", 128, False, dict(in_size=128)),
    ("smallcnn 128 aug",   "smallcnn", 128, True,  dict(in_size=128)),
    ("r18 ft-head noaug",  "resnet18", 224, False, dict(pretrained=True, freeze_upto="layer3")),
    ("r18 ft-head aug",    "resnet18", 224, True,  dict(pretrained=True, freeze_upto="layer3")),
    ("r18 ft-full aug",    "resnet18", 224, True,  dict(pretrained=True, freeze_upto=None)),
]

if __name__ == "__main__":
    print(f"A3 revised | {N_FOLDS}-fold subject-grouped CV | {EPOCHS} epochs\n")
    res = {}
    for label, name, size, aug, kw in CONFIGS:
        print(label)
        res[label] = cross_validate(label, name, size, aug, **kw)
    print("=" * 60)
    for k, (a, s, f) in sorted(res.items(), key=lambda x: -x[1][0]):
        print(f"  {'PASS' if a >= 0.88 else '    '}  {k:<22} {a:.4f} +- {s:.4f}  F1 {f:.4f}")
