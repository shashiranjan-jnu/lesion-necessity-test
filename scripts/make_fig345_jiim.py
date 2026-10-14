"""Figures 3, 4 and 5, journal-compliant.

Changes from make_figures.py:
  - no titles inside the artwork; the text that was in each suptitle now lives in
    the manuscript captions
  - the "favours occluded / favours intact" annotations on Fig. 3 are retained:
    they are axis orientation labels, not a caption
  - sans-serif lettering at 8-10 pt, widths of 174 mm (Figs 3 and 5) and 84 mm
    (Fig. 4), per JIIM artwork guidelines
  - 400 dpi, PDF primary

Figures 3 and 4 need no GPU. Figure 5 needs the checkpoints written by
a6d_shortcut_rigour.py and trains any that are missing.

Output: Fig3.pdf, Fig4.pdf, Fig5.pdf in /kaggle/working/figures_jiim

Usage:
    python scripts/make_fig345_jiim.py            # all three
    python scripts/make_fig345_jiim.py --only 5
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/kaggle/working/figures_jiim"
os.makedirs(OUT, exist_ok=True)
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "savefig.bbox": "tight", "savefig.dpi": 400, "pdf.fonttype": 42,
})

C_RES, C_CNN, C_REF = "#1f4e79", "#c0504d", "#7f7f7f"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=400)
    plt.close(fig)
    print(f"saved {OUT}/{name}.pdf and .png")


# ------------------------------------------------------------------- Figure 3
FOREST = {
    "ResNet-18 (4.72M)": [
        ("Occluded",              -0.23, -1.79, +1.18),
        ("Occluded, 10 px dil.",  +0.34, -1.83, +2.30),
        ("Tumor only",            -4.80, -9.15, -0.36),
    ],
    "Small network (0.24M)": [
        ("Occluded",              -0.73, -2.62, +1.20),
        ("Occluded, 10 px dil.",  -2.71, -4.69, -0.77),
        ("Tumor only",            -0.28, -5.00, +4.21),
    ],
}


def figure3():
    fig, ax = plt.subplots(figsize=(174 * MM, 78 * MM))
    ypos, labels_, colors = [], [], []
    y = 0
    for arch, rows in FOREST.items():
        for lab, d, lo, hi in rows:
            ypos.append(y); labels_.append(lab)
            colors.append(C_RES if "ResNet" in arch else C_CNN)
            y += 1
        y += 0.8

    flat = [r for rows in FOREST.values() for r in rows]
    for yy, (lab, d, lo, hi), col in zip(ypos, flat, colors):
        crosses = lo <= 0 <= hi
        ax.plot([lo, hi], [yy, yy], color=col, lw=1.4, solid_capstyle="butt",
                zorder=2)
        for e in (lo, hi):
            ax.plot([e, e], [yy - 0.13, yy + 0.13], color=col, lw=1.4, zorder=2)
        ax.plot(d, yy, "o" if crosses else "D", color=col, ms=5,
                mfc="white" if crosses else col, mew=1.4, zorder=3)

    ax.axvline(0, color="black", lw=0.9, ls="--", zorder=1)
    ax.set_yticks(ypos); ax.set_yticklabels(labels_, fontsize=8.5)
    ax.set_xlabel("Change in accuracy vs. intact (percentage points)",
                  fontsize=9)
    ax.set_xlim(-11, 6)
    ax.tick_params(labelsize=8.5)

    # axis orientation labels -- these identify direction, not a caption
    ymax = max(ypos) + 0.9
    ax.text(-10.7, ymax, "favours occluded  \u2190", fontsize=7.5,
            color=C_REF, ha="left", va="center")
    ax.text(5.6, ymax, "\u2192  favours intact", fontsize=7.5,
            color=C_REF, ha="right", va="center")
    ax.set_ylim(ymax + 0.5, min(ypos) - 0.7)

    n_res = len(FOREST["ResNet-18 (4.72M)"])
    for tag, idx, col in [("ResNet-18\n(4.72M)", range(n_res), C_RES),
                          ("Small network\n(0.24M)", range(n_res, len(ypos)),
                           C_CNN)]:
        ax.text(6.4, np.mean([ypos[i] for i in idx]), tag, color=col,
                va="center", ha="left", fontsize=8, linespacing=1.3)

    save(fig, "Fig3")


# ------------------------------------------------------------------- Figure 4
RES = [(128, 0.7687, 0.0654), (64, 0.8008, 0.0422), (32, 0.7866, 0.0372),
       (16, 0.7616, 0.0057), (8, 0.6770, 0.0247)]
INTACT_F1, INTACT_SD = 0.777, 0.079


def figure4():
    fig, ax = plt.subplots(figsize=(84 * MM, 68 * MM))
    x = np.array([r[0] for r in RES], float)
    m = np.array([r[1] for r in RES]); s = np.array([r[2] for r in RES])
    o = np.argsort(x); x, m, s = x[o], m[o], s[o]

    ax.axhspan(INTACT_F1 - INTACT_SD, INTACT_F1 + INTACT_SD, color=C_REF,
               alpha=0.15, zorder=0)
    ax.axhline(INTACT_F1, color=C_REF, lw=1.0, ls="--", zorder=1)
    ax.text(8.6, INTACT_F1 + 0.008, "intact, 128 px", fontsize=7.5,
            color=C_REF)

    ax.fill_between(x, m - s, m + s, color=C_CNN, alpha=0.18, zorder=2)
    ax.plot(x, m, "-o", color=C_CNN, ms=4.5, lw=1.4, zorder=3)

    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64, 128])
    ax.set_xticklabels(["8", "16", "32", "64", "128"])
    ax.set_xlabel("Input resolution (pixels per side)", fontsize=9)
    ax.set_ylabel("Macro-F1", fontsize=9)
    ax.set_ylim(0.60, 0.88)
    ax.tick_params(labelsize=8.5)
    save(fig, "Fig4")


# ------------------------------------------------------------------- Figure 5
def figure5(n_show=4, seed=0):
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import StratifiedGroupKFold
    from core import data_mat, model as M, xai
    from core.seeding import set_seed

    DEV, SIZE = "cuda", 128
    CKPT = "/kaggle/working/ckpt/a6d"
    os.makedirs(CKPT, exist_ok=True)
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

    def train(mode, sd=1):
        set_seed(sd)
        x = make_x(tr, mode).to(DEV)
        y = torch.from_numpy(labels[tr]).to(DEV)
        net = M.build("smallcnn", in_size=SIZE).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        cnt = torch.bincount(y, minlength=3).float()
        lf = nn.CrossEntropyLoss(
            weight=(cnt.sum() / (3 * cnt.clamp(min=1))).to(DEV))
        dl = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
        net.train()
        for _ in range(30):
            for xb, yb in dl:
                opt.zero_grad(); lf(net(xb), yb).backward(); opt.step()
        del x, y; torch.cuda.empty_cache()
        return net

    def load_or_train(tag, mode):
        p = f"{CKPT}/{tag}.pt"
        if os.path.exists(p):
            net = M.build("smallcnn", in_size=SIZE).to(DEV)
            net.load_state_dict(torch.load(p)); print(f"  loaded {tag}")
        else:
            print(f"  training {tag} (~2 min)")
            net = train(mode); torch.save(net.state_dict(), p)
        return net.eval()

    print("figure 5: models")
    nets = [("intact-trained", load_or_train("intact_f2", "intact"), "intact"),
            ("tumor-only-trained", load_or_train("tumoronly_f2", "tumor_only"),
             "tumor_only")]

    rng = np.random.default_rng(seed)
    area = np.array([(masks[i] > 0).mean() for i in te])
    cand = te[area > np.percentile(area, 60)]
    pick, seen = [], set()
    for i in rng.permutation(cand):
        if labels[i] not in seen or len(pick) >= 3:
            pick.append(int(i)); seen.add(labels[i])
        if len(pick) == n_show:
            break

    CLASSES = data_mat.CLASSES
    fig, axes = plt.subplots(3, n_show, figsize=(174 * MM, 140 * MM))
    for c, i in enumerate(pick):
        m = torch.from_numpy(masks[i].astype(np.float32))[None, None]
        m = (F.interpolate(m, (SIZE, SIZE), mode="nearest").squeeze().numpy()
             > 0.5)
        base = make_x(np.array([i]), "intact")[0, 0].numpy()

        ax = axes[0, c]
        ax.imshow(base, cmap="gray")
        ax.contour(m, levels=[0.5], colors="#00d4ff", linewidths=1.0)
        ax.set_title(CLASSES[labels[i]], fontsize=9, pad=4)
        if c == 0:
            ax.set_ylabel("image and mask", fontsize=8.5)

        for r, (name, net, mode) in enumerate(nets, start=1):
            x1 = make_x(np.array([i]), mode).to(DEV).requires_grad_(True)
            with torch.no_grad():
                pred = int(net(x1).argmax(1).item())
            a = xai.gradcam(net, x1, pred, net.last_conv)[0]
            a = np.clip(a, 0, None)
            a = a / a.max() if a.max() > 0 else a
            ax = axes[r, c]
            ax.imshow(base, cmap="gray")
            ax.imshow(a, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
            ax.contour(m, levels=[0.5], colors="#00d4ff", linewidths=1.0)
            inside = bool(m[np.unravel_index(np.argmax(a), a.shape)])
            ax.text(0.04, 0.95, "in" if inside else "out",
                    transform=ax.transAxes, fontsize=8.5, fontweight="bold",
                    color="#00d4ff" if inside else "#ff5555",
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.16", fc="black", ec="none",
                              alpha=0.6))
            if c == 0:
                ax.set_ylabel(name, fontsize=8.5)

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
    fig.tight_layout()
    save(fig, "Fig5")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None, choices=[3, 4, 5])
    a = ap.parse_args()
    print("Figures 3-5, journal-compliant (no embedded titles, 400 dpi)\n")
    if a.only in (None, 3): figure3()
    if a.only in (None, 4): figure4()
    if a.only in (None, 5): figure5()
