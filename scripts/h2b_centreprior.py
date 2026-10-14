"""H2b -- why the faithfulness metrics do not discriminate on HAM10000.

H2 found that an UNTRAINED network scores as high as the trained ones
(Grad-CAM pointing 0.693 untrained vs 0.653 intact; IoU 0.406 vs 0.379), and
parameter-randomisation correlations are near zero for IG and SmoothGrad yet
the scores are unchanged. That is the Adebayo check failing: the metric is not
reading the weights.

The reason is geometric. Lesions cover 26.3% of the frame and sit centrally
(both centroids ~0.50), so any centrally biased map -- which is what an
untrained CNN produces -- falls inside the mask. Contrast the brain collection:
masks 1.7% of the frame, chance pointing 0.0156, untrained pointing 0.000.

This script removes the model entirely and scores three fixed maps against the
same masks:

  centre_disk   a disk at the frame centre with area equal to the mask's
  centre_gauss  an isotropic Gaussian at the frame centre, sigma = 0.25 * side
  uniform       constant map (pointing is then the mask area by definition)

Whatever these reach is the score a method gets for knowing nothing. Reported
beside the H2 table, it states the ceiling of the inference the metrics support
on this collection. No GPU, seconds to run.
"""
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from core import data_ham

OUT = "/kaggle/working/h2b_centreprior.json"
SIZE, N_EVAL = 224, 150

imgs, masks, labels, gids = data_ham.load_arrays()
k = np.isin(gids, np.load("/kaggle/working/ham_matched_ids.npy",
                          allow_pickle=True).astype(str))
masks, labels, gids = masks[k], labels[k], gids[k]
tr, te = next(iter(StratifiedGroupKFold(5, shuffle=True, random_state=0)
                   .split(np.zeros(len(labels)), labels, groups=gids)))
te = np.random.default_rng(0).choice(te, size=min(N_EVAL, len(te)), replace=False)

yy, xx = np.mgrid[0:SIZE, 0:SIZE]
r2 = (yy - SIZE / 2) ** 2 + (xx - SIZE / 2) ** 2


def metrics(attr, mask):
    a = np.clip(attr, 0, None)
    tot = a.sum()
    if tot <= 0:
        return dict(point=0.0, energy=0.0, iou=0.0)
    p = np.unravel_index(np.argmax(a), a.shape)
    kk = max(int(mask.sum()), 1)
    idx = np.argpartition(a.ravel(), -kk)[-kk:]
    top = np.zeros(a.size, bool); top[idx] = True
    top = top.reshape(a.shape)
    m = mask.astype(bool)
    return dict(point=float(m[p]), energy=float((a * mask).sum() / tot),
                iou=float((top & m).sum() / max((top | m).sum(), 1)))


def resize_mask(m):
    from PIL import Image
    return (np.array(Image.fromarray(m * 255).resize((SIZE, SIZE),
            Image.NEAREST)) > 127).astype(np.uint8)


if __name__ == "__main__":
    rows = {k: [] for k in ["centre_disk", "centre_gauss", "uniform"]}
    areas = []
    for i in te:
        m = resize_mask(masks[i])
        areas.append(m.mean())
        rad2 = m.sum() / np.pi                       # disk of equal area
        rows["centre_disk"].append(metrics((r2 <= rad2).astype(float), m))
        rows["centre_gauss"].append(
            metrics(np.exp(-r2 / (2 * (0.25 * SIZE) ** 2)), m))
        rows["uniform"].append(metrics(np.ones((SIZE, SIZE)), m))

    chance = float(np.mean(areas))
    print(f"\nH2b | {len(te)} images, mean mask area {chance:.4f}\n")
    print(f"  {'fixed map':<14} {'pointing':>9} {'energy':>9} {'IoU':>9}")
    store = {"chance": chance}
    for tag, r in rows.items():
        v = {kk: float(np.mean([d[kk] for d in r])) for kk in r[0]}
        store[tag] = v
        print(f"  {tag:<14} {v['point']:>9.3f} {v['energy']:>9.3f} {v['iou']:>9.3f}")
    with open(OUT, "w") as fh:
        json.dump(store, fh, indent=2)
    print(f"\nsaved {OUT}")
    print("\n  Compare with H2: intact Grad-CAM pointing 0.653, IoU 0.379;")
    print("  untrained 0.693 / 0.406. If a model-free centred map is in that")
    print("  range, the metrics are measuring the frame, not the model, and no")
    print("  attribution claim can be made on this collection.\n")
