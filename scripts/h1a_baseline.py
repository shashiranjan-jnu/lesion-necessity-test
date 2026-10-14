"""H1a -- get the intact baseline to a defensible level before running the
necessity conditions.

The first H1 run reached 0.564 macro-F1 / 0.601 accuracy intact, with seed sd
4.68 against partition sd 0.62. That is an underfit, not a finding, and a
positive control built on it proves nothing. Candidate causes, each tested here
on ONE fold and ONE seed:

  A  current setup: GroupNorm ResNet (BN stats discarded by ModuleValidator),
     layer4b trainable, /255 scaling, no augmentation
  B  stock BatchNorm ResNet + ImageNet normalisation, layer4 trainable
  C  B + full fine-tune at a lower learning rate
  D  C + flips and 90-degree rotations (dermoscopy is orientation-invariant)

The DP-safety constraint that forces GroupNorm applies to the federated arm,
not here, so BatchNorm is available for this dataset.

Target: >= ~0.78 accuracy / >= ~0.85 AUROC intact. Anything below that and the
necessity contrasts are noise against noise.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from core import data_ham
from core.seeding import set_seed

DEV = "cuda" if torch.cuda.is_available() else "cpu"
assert DEV == "cuda", "no GPU: Session options -> Accelerator -> GPU"
MATCHED = "/kaggle/working/ham_matched_ids.npy"
SIZE, SEED = 224, 1

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

imgs, masks, labels, gids = data_ham.load_arrays()
k = np.isin(gids, np.load(MATCHED, allow_pickle=True).astype(str))
imgs, masks, labels, gids = imgs[k], masks[k], labels[k], gids[k]
tr, te = next(iter(StratifiedGroupKFold(5, shuffle=True, random_state=0)
                   .split(imgs, labels, groups=gids)))
print(f"{len(labels)} images / {len(np.unique(gids))} lesions | "
      f"train {len(tr)}  test {len(te)}")


def tensor(idx, imagenet_norm):
    x = torch.from_numpy(imgs[idx]).float().div_(255).permute(0, 3, 1, 2)
    x = F.interpolate(x, (SIZE, SIZE), mode="bilinear", align_corners=False)
    if imagenet_norm:
        x = (x - MEAN) / STD
    return x


def net_gn():
    from core import model as M
    return M.build("resnet18", n_classes=2, in_ch=3, trainable="layer4b")


def net_bn(trainable):
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(512, 2)
    if trainable != "all":
        for n_, p in m.named_parameters():
            p.requires_grad = n_.startswith(("layer4", "fc"))
    return m


def augment(xb):
    if torch.rand(1).item() < 0.5:
        xb = torch.flip(xb, [3])
    if torch.rand(1).item() < 0.5:
        xb = torch.flip(xb, [2])
    return torch.rot90(xb, int(torch.randint(0, 4, (1,)).item()), [2, 3])


def run(tag, net_fn, lr, epochs, imagenet_norm, aug):
    set_seed(SEED)
    x, y = tensor(tr, imagenet_norm).to(DEV), torch.from_numpy(labels[tr]).to(DEV)
    net = net_fn().to(DEV)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    cnt = torch.bincount(y, minlength=2).float()
    lossf = nn.CrossEntropyLoss(weight=(cnt.sum() / (2 * cnt.clamp(min=1))).to(DEV))
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
    for _ in range(epochs):
        net.train()
        for xb, yb in loader:
            if aug:
                xb = augment(xb)
            opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
        sched.step()
    net.eval()
    P = []
    with torch.no_grad():
        xt = tensor(te, imagenet_norm)
        for s in range(0, len(te), 64):
            P.append(torch.softmax(net(xt[s:s + 64].to(DEV)), 1).cpu())
    pr = torch.cat(P).numpy()
    p, yt = pr.argmax(1), labels[te]
    print(f"  {tag:<44} acc {(p == yt).mean():.4f}  "
          f"macroF1 {f1_score(yt, p, average='macro'):.4f}  "
          f"AUROC {roc_auc_score(yt, pr[:, 1]):.4f}")
    del net, x, y; torch.cuda.empty_cache()


if __name__ == "__main__":
    print("\nH1a | one fold, one seed. Target: acc >= 0.78, AUROC >= 0.85\n")
    run("A  GN, layer4b, /255, no aug (current)",
        net_gn, 1e-3, 30, False, False)
    run("B  BN, layer4, ImageNet norm",
        lambda: net_bn("layer4"), 1e-3, 30, True, False)
    run("C  BN, full fine-tune, lr 1e-4",
        lambda: net_bn("all"), 1e-4, 30, True, False)
    run("D  BN, full fine-tune, lr 1e-4, flips + rot90",
        lambda: net_bn("all"), 1e-4, 30, True, True)
    print("\nPick the cheapest config that clears the target; that becomes the")
    print("training recipe for every condition in h1_necessity.py.\n")
