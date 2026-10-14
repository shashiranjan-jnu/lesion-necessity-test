"""Fig. 7 -- schematic of the lesion-necessity protocol (Algorithm 1).

No data, no GPU, no model: a pure matplotlib diagram of the protocol. Shows the
three derived inputs, the shared fold-and-seed structure that makes the paired
contrasts resolvable, the two contrasts, the clustered bootstrap, the decision
rule of Eq. 3, and the positive control branching off f+.

Note: matplotlib mathtext needs \\leq, not \\le, which is why the strings here
differ slightly from the LaTeX source.
"""
import matplotlib
matplotlib.use("Agg")
import os
import matplotlib.pyplot as plt
from core.jiim_figstyle import apply, size, save
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


INK, MUT, ACC = "#1a1a1a", "#6b6b6b", "#0b5394"
apply()

fig, ax = plt.subplots(figsize=size(171, 110))
ax.set_xlim(0, 100); ax.set_ylim(0, 66.5); ax.axis("off")


def box(x, y, w, h, text, fc="white", ec=INK, fs=8, lw=0.9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.35,rounding_size=1.1",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, zorder=3, linespacing=1.45,
            fontweight=weight)


def arrow(x1, y1, x2, y2, ec=MUT, style="-|>", lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=9, lw=lw, color=ec,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


# ---------------------------------------------------------------- inputs
box(1.5, 41, 17, 12, "image $x$\nannotation $m$", fc="#f2f2f2")
ax.text(10, 56.8, "collection", ha="center", fontsize=8, color=MUT)

conds = [
    (21, 49.5, "intact\n$x$"),
    (21, 37.0, "occluded\n$x^- = x \\odot (1-m)$"),
    (21, 24.5, "finding only\n$x^+ = x \\odot m$"),
]
for x, y, t in conds:
    box(x, y, 22, 10.2, t)
    arrow(18.5, 47, x - 0.4, y + 5.1, rad=0.0 if abs(y - 44) < 6 else 0.12)

# ---------------------------------------------------------------- training
ax.add_patch(FancyBboxPatch((47, 22.6), 20.5, 39.2,
                            boxstyle="round,pad=0.4,rounding_size=1.2",
                            fc="#f7f9fc", ec=ACC, lw=0.9, ls=(0, (4, 2)),
                            zorder=0))
ax.text(57.25, 64.4, "identical folds and seeds", ha="center", fontsize=8,
        color=ACC)
for (x, y, _), name in zip(conds, ["$f^{\\,\\mathrm{intact}}$", "$f^-$", "$f^+$"]):
    box(48.5, y, 17.5, 10.2, f"train {name}\nsubject-grouped\n$K\\!\\times\\!S$ runs",
        fs=8)
    arrow(43.4, y + 5.1, 48.1, y + 5.1)

# ---------------------------------------------------------------- contrasts
box(71.5, 43.5, 27.5, 12.5,
    "$\\Delta^- = A(f^-)-A(f^{\\,\\mathrm{intact}})$\n"
    "$\\Delta^+ = A(f^+)-A(f^{\\,\\mathrm{intact}})$", fs=8)
arrow(66.4, 54.6, 71.0, 52.0, rad=-0.15)
arrow(66.4, 42.1, 71.0, 48.5, rad=0.15)
arrow(66.4, 29.6, 71.0, 45.2, rad=0.22)

box(71.5, 28.5, 27.5, 10.5,
    "patient-clustered\nbootstrap\n$95\\%$ interval", fs=8)
arrow(85.25, 43.2, 85.25, 39.3)

# ---------------------------------------------------------------- decision
box(47, 5.0, 52, 12.0,
    "necessary if  $u(\\Delta^-) < 0$\n"
    "sufficient if  $\\ell(\\Delta^+) \\leq 0 \\leq u(\\Delta^+)$",
    fc="#f7f9fc", ec=ACC, fs=8, lw=1.1)
arrow(85.25, 28.2, 85.25, 17.4, ec=ACC)

# ---------------------------------------------------------------- control
box(21, 5.0, 22, 12.0,
    "positive control:\nattributions of $f^+$\nmust fall inside $m$", fs=8,
    ec=MUT)
arrow(57.25, 22.4, 38, 17.4, ec=MUT, rad=0.22)

ax.text(50, 0.6, "one training run per condition", ha="center", fontsize=8,
        color=MUT, style="italic")

fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
save(fig, 6, "vector")
