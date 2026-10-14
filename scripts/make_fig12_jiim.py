"""Figures 1 and 2, journal-compliant.

Changes from a6f_mechanism_figure.py:
  - no titles or captions inside the artwork (JIIM: "Do not include titles or
    captions into your illustrations"). Row/class labels are retained -- those
    identify panel content and are not captions.
  - 400 dpi, and PDF alongside PNG so the figure is vector where possible
  - figure widths sized to JIIM's 174 mm single-column-spanning width
  - sans-serif lettering at 8-10 pt per the artwork guidelines

The caption text lives in the manuscript. The printed statistics that used to sit
in the suptitle ("a model trained on the 16px row reaches 0.762 macro-F1") are
now in the Fig. 1 caption.

Output: Fig1.pdf/.png, Fig2.pdf/.png in /kaggle/working/figures_jiim
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from core import data_mat

OUT = "/kaggle/working/figures_jiim"
os.makedirs(OUT, exist_ok=True)
CLASSES = ["glioma", "meningioma", "pituitary"]
MM = 1 / 25.4                      # JIIM widths: 39, 84, 129, 174 mm

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
    "savefig.dpi": 400,
    "pdf.fonttype": 42,            # embed fonts as TrueType
})

imgs, masks, labels, pids = data_mat.load_arrays()


def occluded(idx, size):
    x = torch.from_numpy(imgs[idx]).float().div_(255)[:, None]
    m = torch.from_numpy(masks[idx]).float()[:, None]
    x = F.interpolate(x, (size, size), mode="bilinear", align_corners=False)
    m = (F.interpolate(m, (size, size), mode="nearest") > 0.5).float()
    return (x * (1 - m)).squeeze(1).numpy()


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=400)
    plt.close(fig)
    print(f"saved {OUT}/{name}.pdf and .png")


# ------------------------------------------------------------------- Figure 1

def figure1(n_samples=4, seed=0):
    rng = np.random.default_rng(seed)
    picks = {k: rng.choice(np.where(labels == k)[0], n_samples, replace=False)
             for k in range(3)}
    sizes = [128, 32, 16]

    # one spacer column between each class block
    ncol = 3 * n_samples + 2
    col0 = lambda c: c * (n_samples + 1)          # first column of class c

    fig = plt.figure(figsize=(174 * MM, 155 * MM))
    gs = fig.add_gridspec(len(sizes) + 2, ncol, hspace=0.40, wspace=0.08)

    # rows 1-3: samples per class at each resolution
    for r, size in enumerate(sizes):
        for c in range(3):
            xs = occluded(picks[c], size)
            for j in range(n_samples):
                ax = fig.add_subplot(gs[r, col0(c) + j])
                ax.imshow(xs[j], cmap="gray", interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_linewidth(0.4)
                if r == 0 and j == n_samples // 2:
                    ax.set_title(CLASSES[c], fontsize=9, pad=4)
                if c == 0 and j == 0:
                    ax.set_ylabel(f"{size} px", fontsize=8.5)

    # row 4: per-class mean at 16 px
    means16 = [occluded(np.where(labels == c)[0], 16).mean(axis=0)
               for c in range(3)]
    vmin, vmax = min(m.min() for m in means16), max(m.max() for m in means16)
    for c in range(3):
        ax = fig.add_subplot(gs[len(sizes), col0(c):col0(c) + n_samples])
        ax.imshow(means16[c], cmap="gray", vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel("class mean", fontsize=8.5)

    # row 5: deviation from the grand mean
    grand = np.mean(means16, axis=0)
    devs = [m - grand for m in means16]
    lim = max(abs(d).max() for d in devs)
    for c in range(3):
        ax = fig.add_subplot(gs[len(sizes) + 1, col0(c):col0(c) + n_samples])
        ax.imshow(devs[c], cmap="RdBu_r", vmin=-lim, vmax=lim,
                  interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel("deviation", fontsize=8.5)

    save(fig, "Fig1")

    print("\npairwise L2 between 16 px class means:")
    for a in range(3):
        for b in range(a + 1, 3):
            print(f"  {CLASSES[a]:<11} vs {CLASSES[b]:<11} "
                  f"{np.linalg.norm(means16[a] - means16[b]):.4f}")



# ------------------------------------------------------------------- Figure 2
def otsu(im):
    h, _ = np.histogram(im.ravel(), bins=64, range=(0, 1))
    h = h.astype(float); p = h / max(h.sum(), 1)
    omega = np.cumsum(p); mu = np.cumsum(p * np.arange(64))
    den = omega * (1 - omega); den[den == 0] = 1e-12
    return (np.argmax((mu[-1] * omega - mu) ** 2 / den) + 0.5) / 64


def figure2():
    stats = {k: {"area": [], "cy": [], "cx": [], "height": []} for k in range(3)}
    for c in range(3):
        for x in occluded(np.where(labels == c)[0], 128):
            fg = x > otsu(x)
            if fg.sum() < 50:
                continue
            ys, xs_ = np.nonzero(fg)
            stats[c]["area"].append(fg.mean())
            stats[c]["cy"].append(ys.mean() / 128)
            stats[c]["cx"].append(xs_.mean() / 128)
            stats[c]["height"].append((ys.max() - ys.min()) / 128)

    fields = [("area", "fractional head area"),
              ("cy", "vertical centroid"),
              ("cx", "horizontal centroid"),
              ("height", "vertical extent")]

    fig, axes = plt.subplots(1, 4, figsize=(174 * MM, 52 * MM))
    for ax, (key, ylab) in zip(axes, fields):
        data = [stats[c][key] for c in range(3)]
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor("#9ecae1"); b.set_alpha(0.75)
            b.set_edgecolor("#3182bd"); b.set_linewidth(0.6)
        parts["cmeans"].set_color("#08519c"); parts["cmeans"].set_linewidth(1.0)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(CLASSES, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    save(fig, "Fig2")

    print("\nper-class geometry (mean +- sd), tumor occluded, Otsu threshold:")
    print(f"  {'class':<12}" + "".join(f"{n:>20}" for _, n in fields))
    for c in range(3):
        row = f"  {CLASSES[c]:<12}"
        for key, _ in fields:
            v = np.array(stats[c][key])
            row += f"{v.mean():>11.3f}+-{v.std():<8.3f}"
        print(row)
    print("\n  one-way ANOVA:")
    for key, name in fields:
        F_, p = f_oneway(*[stats[c][key] for c in range(3)])
        print(f"    {name:<24} F={F_:>9.1f}  p={p:.2e}")


if __name__ == "__main__":
    print("Figures 1 and 2, journal-compliant (no embedded titles, 400 dpi)\n")
    figure1()
    print()
    figure2()
