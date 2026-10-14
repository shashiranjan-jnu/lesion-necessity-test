"""H2 -- do the attributions localise the lesion on HAM10000?

Companion to the brain-collection attribution analysis. There, all three methods
sat at chance inside the tumor for an intact-trained model (pointing 0.020 vs
0.0156 chance) while the tumor-only control reached 0.733, so the methods were
exonerated and the model blamed. Here the lesion IS necessary (dAcc -6.41 pts,
CI excluding zero), so attributions should localise well. That completes the
parallel: same code, opposite verdict on both the model and its explanations.

One scale difference matters. Lesion masks cover ~28% of the frame here against
1.7% on the brain collection, so chance pointing accuracy is ~0.28, not 0.0156.
A pointing score of 0.6 is unimpressive at this mask size. Lead with energy in
mask and area-matched IoU, both of which are normalised against the same area,
and always print the chance line next to the result.

Three models, one fold, one seed:
  intact       trained on x        -- the question
  lesion_only  trained on x (+)    -- positive control: must localise
  untrained    no training         -- parameter randomisation check
"""
import json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import spearmanr
from captum.attr import LayerGradCam, IntegratedGradients, NoiseTunnel
from core import data_ham
from core.seeding import set_seed

DEV = "cuda" if torch.cuda.is_available() else "cpu"
assert DEV == "cuda", "no GPU: Session options -> Accelerator -> GPU"
EPOCHS, BATCH, LR, WD, SIZE = 30, 32, 1e-4, 1e-4, 224
SEED, N_EVAL = 1, 150
MATCHED = "/kaggle/working/ham_matched_ids.npy"
OUT = "/kaggle/working/h2_attribution.json"

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

imgs, masks, labels, gids = data_ham.load_arrays()
k = np.isin(gids, np.load(MATCHED, allow_pickle=True).astype(str))
imgs, masks, labels, gids = imgs[k], masks[k], labels[k], gids[k]
tr, te = next(iter(StratifiedGroupKFold(5, shuffle=True, random_state=0)
                   .split(imgs, labels, groups=gids)))
rng = np.random.default_rng(0)
te = rng.choice(te, size=min(N_EVAL, len(te)), replace=False)


def make_x(idx, mode="intact"):
    x = torch.from_numpy(imgs[idx]).float().div_(255).permute(0, 3, 1, 2)
    x = F.interpolate(x, (SIZE, SIZE), mode="bilinear", align_corners=False)
    if mode != "intact":
        m = get_masks(idx)
        x = x * m if mode == "lesion_only" else x * (1 - m)
    return (x - MEAN) / STD


def get_masks(idx):
    m = torch.from_numpy(masks[idx]).float()[:, None]
    return (F.interpolate(m, (SIZE, SIZE), mode="nearest") > 0.5).float()


def build():
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(512, 2)
    return m


def augment(xb):
    if torch.rand(1).item() < 0.5:
        xb = torch.flip(xb, [3])
    if torch.rand(1).item() < 0.5:
        xb = torch.flip(xb, [2])
    return torch.rot90(xb, int(torch.randint(0, 4, (1,)).item()), [2, 3])


def train(mode):
    set_seed(SEED)
    net = build().to(DEV)
    if mode == "untrained":
        return net.eval()
    x = make_x(tr, mode).to(DEV)
    y = torch.from_numpy(labels[tr]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    cnt = torch.bincount(y, minlength=2).float()
    lossf = nn.CrossEntropyLoss(weight=(cnt.sum() / (2 * cnt.clamp(min=1))).to(DEV))
    from torch.utils.data import TensorDataset, DataLoader
    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH, shuffle=True)
    for _ in range(EPOCHS):
        net.train()
        for xb, yb in loader:
            xb = augment(xb)
            opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
        sched.step()
    del x, y; torch.cuda.empty_cache()
    return net.eval()


# ------------------------------------------------------------------ metrics
def metrics(attr, mask):
    """attr (H,W) float, mask (H,W) 0/1. Positive part only, as in the brain
    analysis: negative attribution is evidence against the predicted class."""
    a = np.clip(attr, 0, None)
    tot = a.sum()
    if tot <= 0:
        return dict(point=0.0, energy=0.0, iou=0.0)
    p = np.unravel_index(np.argmax(a), a.shape)
    kk = int(mask.sum())
    thr_idx = np.argpartition(a.ravel(), -kk)[-kk:]      # area-matched top-k
    top = np.zeros(a.size, bool); top[thr_idx] = True
    top = top.reshape(a.shape)
    m = mask.astype(bool)
    return dict(point=float(m[p]),
                energy=float((a * mask).sum() / tot),
                iou=float((top & m).sum() / max((top | m).sum(), 1)))


def attribute(net, method, x, target):
    if method == "gradcam":
        cam = LayerGradCam(net, net.layer4[-1])
        a = cam.attribute(x, target=target, relu_attributions=False)
        a = F.interpolate(a, (SIZE, SIZE), mode="bilinear", align_corners=False)
        return a[:, 0].detach().cpu().numpy()
    ig = IntegratedGradients(net)
    if method == "smoothgrad":
        nt = NoiseTunnel(ig)
        a = nt.attribute(x, target=target, nt_type="smoothgrad",
                         nt_samples=8, stdevs=0.1, n_steps=16)
    else:
        a = ig.attribute(x, target=target, n_steps=32)
    return a.sum(1).detach().cpu().numpy()               # sum over channels


def evaluate(net, tag, store):
    rows = {m: [] for m in ["gradcam", "ig", "smoothgrad"]}
    maps = {m: [] for m in rows}
    for s in range(0, len(te), 16):
        idx = te[s:s + 16]
        x = make_x(idx).to(DEV).requires_grad_(True)
        with torch.no_grad():
            pred = net(make_x(idx).to(DEV)).argmax(1)
        mk = get_masks(idx)[:, 0].numpy()
        for meth in rows:
            A = attribute(net, meth, x, pred)
            maps[meth].append(A)
            rows[meth] += [metrics(A[i], mk[i]) for i in range(len(idx))]
    chance = float(np.mean([get_masks(te[i:i + 1])[0, 0].mean().item()
                            for i in range(len(te))]))
    print(f"\n  {tag}  (chance pointing = mean mask area = {chance:.4f})")
    print(f"  {'method':<12} {'pointing':>9} {'energy':>9} {'IoU':>9}")
    store[tag] = {"chance": chance}
    for meth, r in rows.items():
        v = {k: float(np.mean([d[k] for d in r])) for k in r[0]}
        store[tag][meth] = v
        print(f"  {meth:<12} {v['point']:>9.3f} {v['energy']:>9.3f} {v['iou']:>9.3f}")
    return {m: np.concatenate(v) for m, v in maps.items()}


if __name__ == "__main__":
    print(f"H2 | fold 0, seed {SEED}, {len(te)} test images")
    store, saved = {}, {}
    for tag in ["intact", "lesion_only", "untrained"]:
        net = train(tag)
        saved[tag] = evaluate(net, tag, store)
        del net; torch.cuda.empty_cache()

    print("\n  parameter randomisation (Spearman, trained vs untrained maps):")
    store["randomisation"] = {}
    for meth in saved["intact"]:
        rs = [spearmanr(saved["intact"][meth][i].ravel(),
                        saved["untrained"][meth][i].ravel()).correlation
              for i in range(len(te))]
        r = float(np.nanmean(rs))
        store["randomisation"][meth] = r
        print(f"    {meth:<12} {r:+.3f}")

    with open(OUT, "w") as fh:
        json.dump(store, fh, indent=2)
    print(f"\nsaved {OUT}")
    print("\n  Read against the brain collection: there the intact-trained model")
    print("  scored at chance and the tumor-only control at 0.733 pointing.")
    print("  Here the lesion is necessary, so the intact row should already")
    print("  localise, and the gap to the lesion_only control should be small.")
