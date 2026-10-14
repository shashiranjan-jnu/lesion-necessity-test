"""Attribution methods and faithfulness metrics.

Mask-referenced metrics (pointing game, energy, IoU) measure LOCALISATION against
ground-truth tumor masks. Deletion/insertion measure FAITHFULNESS to the model's
own decision. The randomization check (Adebayo et al., NeurIPS 2018) asks whether
the attribution is reading the model at all: a method whose output is UNCHANGED
by weight randomization is not explaining anything, and must be dropped.
 
Chance level for the pointing game on this dataset is the mean mask area = 1.7%.
"""
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import LayerGradCam, IntegratedGradients, NoiseTunnel, Saliency


# ------------------------------------------------------------------ attribution
def _positive(a):
    return np.clip(a, 0, None)


def gradcam(model, x, target, layer):
    lg = LayerGradCam(model, layer)
    a = lg.attribute(x, target=target, relu_attributions=True)
    a = F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return a.squeeze(1).detach().cpu().numpy()


def integrated_gradients(model, x, target, steps=32):
    ig = IntegratedGradients(model)
    a = ig.attribute(x, target=target, n_steps=steps, baselines=torch.zeros_like(x))
    return a.squeeze(1).detach().cpu().numpy()


def smoothgrad(model, x, target, n=16, stdev=0.15):
    nt = NoiseTunnel(Saliency(model))
    a = nt.attribute(x, target=target, nt_type="smoothgrad",
                     nt_samples=n, stdevs=stdev, abs=True)
    return a.squeeze(1).detach().cpu().numpy()


METHODS = {"gradcam": gradcam, "ig": integrated_gradients, "smoothgrad": smoothgrad}


# ------------------------------------------------- mask-referenced localisation
def pointing_game(attr, mask):
    """Does the attribution's argmax fall inside the tumor mask?"""
    h, w = attr.shape
    i = np.unravel_index(np.argmax(attr), (h, w))
    return float(mask[i] > 0)


def energy_in_mask(attr, mask):
    """Fraction of positive attribution mass inside the mask."""
    p = _positive(attr)
    tot = p.sum()
    return float(p[mask > 0].sum() / tot) if tot > 0 else 0.0


def iou_at_mask_area(attr, mask):
    """Threshold the attribution to the same area as the mask, then IoU.
    Area-matched so the metric is not trivially inflated by a diffuse map."""
    k = int(max(1, (mask > 0).sum()))
    p = _positive(attr).ravel()
    if p.sum() == 0:
        return 0.0
    idx = np.argpartition(-p, k - 1)[:k]
    pred = np.zeros_like(p, dtype=bool)
    pred[idx] = True
    pred = pred.reshape(mask.shape)
    m = mask > 0
    inter = (pred & m).sum()
    union = (pred | m).sum()
    return float(inter / union) if union > 0 else 0.0


def dilate(mask, px=10):
    """Binary dilation via max-pool. Peritumoral edema may carry real signal, so
    a strict mask can understate a faithful attribution."""
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    k = 2 * px + 1
    d = F.max_pool2d(t, kernel_size=k, stride=1, padding=px)
    return (d.squeeze().numpy() > 0).astype(np.uint8)


# ----------------------------------------------------- deletion / insertion AUC
@torch.no_grad()
def deletion_insertion(model, x1, attr, target, steps=20, device="cuda"):
    """x1: (1,1,H,W). Progressively remove (deletion) or reveal (insertion) pixels
    in attribution order and track the target-class probability.
    Lower deletion AUC and higher insertion AUC both indicate a faithful map."""
    h, w = attr.shape
    order = np.argsort(-_positive(attr).ravel())
    n = h * w
    chunk = max(1, n // steps)

    del_curve, ins_curve = [], []
    x_del = x1.clone()
    x_ins = torch.zeros_like(x1)
    flat_src = x1.reshape(-1)

    for s in range(steps + 1):
        pd = torch.softmax(model(x_del), 1)[0, target].item()
        pi = torch.softmax(model(x_ins), 1)[0, target].item()
        del_curve.append(pd); ins_curve.append(pi)
        if s == steps:
            break
        sel = order[s * chunk:(s + 1) * chunk]
        if len(sel) == 0:
            continue
        t = torch.from_numpy(sel).long().to(device)
        x_del.reshape(-1)[t] = 0.0
        x_ins.reshape(-1)[t] = flat_src[t]

    return float(np.mean(del_curve)), float(np.mean(ins_curve))


# -------------------------------------------------- randomization sanity check

def randomized_copy(build_fn, seed=0, device="cuda"):
    """An untrained model of the same architecture.

    This is the extreme of the cascading randomization test of Adebayo et al.
    (2018): every learned parameter is replaced by its initialisation. A
    faithful attribution must change substantially; a method whose output is
    unchanged is not reading the trained weights.

    build_fn must construct a RANDOMLY initialised model. For a pretrained
    backbone, pass a builder with pretrained=False -- ImageNet weights are not
    a random initialisation.
    """
    from core.seeding import set_seed
    set_seed(seed)
    return build_fn().to(device).eval()




def rank_correlation(a, b):
    """Spearman correlation between two flattened attribution maps."""
    from scipy.stats import spearmanr
    r, _ = spearmanr(a.ravel(), b.ravel())
    return float(r) if np.isfinite(r) else 0.0
