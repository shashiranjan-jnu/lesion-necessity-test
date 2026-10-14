"""A6k -- attribution controls on the brain collection (manuscript Table 7, Sec. 3.8).

SmallCNN, fold 2 of the subject-grouped split, seed 1, 120 test images.
Grad-CAM, integrated gradients and SmoothGrad are scored by pointing accuracy and
energy in the tumor mask for models trained on intact and on tumor-only images,
and for a randomly initialised network of the same architecture
(core.xai.randomized_copy, the extreme of the Adebayo et al. 2018 randomisation
test). Rank correlation between trained and random attributions is reported.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from core import data_mat, model as M, xai
from core.seeding import set_seed

DEV, SIZE = "cuda", 128
CKPT = "/kaggle/working/ckpt/a6d"
N_EVAL = 120

imgs, masks, labels, pids = data_mat.load_arrays()
tr, te = list(StratifiedGroupKFold(5, shuffle=True, random_state=0)
              .split(imgs, labels, groups=pids))[2]


def make_x(idx, mode):
    x = torch.from_numpy(imgs[idx]).float().div_(255)[:, None]
    x = F.interpolate(x, (SIZE, SIZE), mode="bilinear", align_corners=False)
    if mode != "intact":
        m = torch.from_numpy(masks[idx]).float()[:, None]
        m = (F.interpolate(m, (SIZE, SIZE), mode="nearest") > 0.5).float()
        x = x * (1 - m) if mode == "occlude" else x * m
    return x


def resize_mask(m):
    t = torch.from_numpy(m.astype(np.float32))[None, None]
    return (F.interpolate(t, (SIZE, SIZE), mode="nearest").squeeze().numpy()
            > 0.5).astype(np.uint8)


def train(mode, seed=1):
    set_seed(seed)
    x = make_x(tr, mode).to(DEV)
    y = torch.from_numpy(labels[tr]).to(DEV)
    net = M.build("smallcnn", in_size=SIZE).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    cnt = torch.bincount(y, minlength=3).float()
    lf = nn.CrossEntropyLoss(weight=(cnt.sum() / (3 * cnt.clamp(min=1))).to(DEV))
    dl = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
    net.train()
    for _ in range(30):
        for xb, yb in dl:
            opt.zero_grad(); lf(net(xb), yb).backward(); opt.step()
    del x, y; torch.cuda.empty_cache()
    return net


def load_or_train(tag, mode):
    os.makedirs(CKPT, exist_ok=True)
    p = f"{CKPT}/{tag}.pt"
    if os.path.exists(p):
        net = M.build("smallcnn", in_size=SIZE).to(DEV)
        net.load_state_dict(torch.load(p)); print(f"  loaded {tag}")
    else:
        print(f"  training {tag} (~2 min)")
        net = train(mode); torch.save(net.state_dict(), p)
    return net.eval()


def attributions(net, idx, mode, method="gradcam"):
    """Returns per-image attribution maps and the predicted classes."""
    fn = xai.METHODS[method]
    maps, preds = [], []
    for i in idx:
        x1 = make_x(np.array([i]), mode).to(DEV).requires_grad_(True)
        with torch.no_grad():
            p = int(net(x1).argmax(1).item())
        a = fn(net, x1, p, net.last_conv) if method == "gradcam" else fn(net, x1, p)
        maps.append(a[0]); preds.append(p)
    return maps, preds


def score(maps, idx):
    pg = [xai.pointing_game(a, resize_mask(masks[i])) for a, i in zip(maps, idx)]
    en = [xai.energy_in_mask(a, resize_mask(masks[i])) for a, i in zip(maps, idx)]
    return float(np.mean(pg)), float(np.mean(en))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    eval_idx = rng.choice(te, size=min(N_EVAL, len(te)), replace=False)
    chance = float(np.mean([(resize_mask(masks[i]) > 0).mean() for i in eval_idx]))

    print("A6k | model-parameter randomization check (Adebayo et al., 2018)")
    print(f"      {len(eval_idx)} test images | pointing chance = {chance:.4f}\n")

    build = lambda: M.build("smallcnn", in_size=SIZE)
    rnd = xai.randomized_copy(build, seed=0, device=DEV)

    models = [("intact-trained", load_or_train("intact_f2", "intact"), "intact"),
              ("tumor-only-trained", load_or_train("tumoronly_f2", "tumor_only"),
               "tumor_only")]

    print(f"\n  {'model':<20} {'method':<12} {'pointing':>9} {'energy':>8} "
          f"{'rho vs random':>14}")
    print("  " + "-" * 68)

    for name, net, mode in models:
        for method in ("gradcam", "ig", "smoothgrad"):
            m_tr, _ = attributions(net, eval_idx, mode, method)
            m_rd, _ = attributions(rnd, eval_idx, mode, method)
            rho = float(np.mean([xai.rank_correlation(a, b)
                                 for a, b in zip(m_tr, m_rd)]))
            pg, en = score(m_tr, eval_idx)
            print(f"  {name:<20} {method:<12} {pg:>9.3f} {en:>8.3f} {rho:>14.3f}")

        # the randomized model itself, for reference
        if name == "intact-trained":
            m_rd, _ = attributions(rnd, eval_idx, mode, "gradcam")
            pg, en = score(m_rd, eval_idx)
            print(f"  {'RANDOM (untrained)':<20} {'gradcam':<12} "
                  f"{pg:>9.3f} {en:>8.3f} {'--':>14}")

    print("\n  A method PASSES if rho is low: its output depends on the trained")
    print("  weights. High rho would mean the attribution is an edge detector.")
    print("  Grad-CAM maps are ReLU-sparse, so many pixels tie at zero and rho")
    print("  is attenuated; read it as a direction, not an effect size.")
