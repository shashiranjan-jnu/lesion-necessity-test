"""JIIM figure style, one place for every figure script.

Implements the Springer/JIIM artwork rules:
  * widths 39, 84, 129 or 174 mm, height at most 234 mm
  * lettering Arial/Helvetica (Liberation Sans is metric-identical), 8 pt,
    minimal size variation, no titles inside the figure
  * lines at least 0.3 pt (0.1 mm); here 0.6-0.9 pt
  * fonts embedded (TrueType, fonttype 42)
  * vector figures  -> FigN.pdf (for LaTeX) + FigN.eps (preferred format)
  * figures with photographs and lettering (combination art)
                    -> FigN.pdf + FigN.tif at 600 dpi, RGB 8-bit, LZW

Usage in a figure script:
    from core.jiim_figstyle import apply, size, save
    apply()
    fig, ax = plt.subplots(figsize=size("double", 70))
    ...
    save(fig, 3, "vector")       # or "combination" for image panels
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4
WIDTHS = {"minimal": 39, "single": 84, "onehalf": 129, "double": 174}
MAX_H = 234
OUTDIR = os.environ.get("JIIM_FIGDIR", "/kaggle/working/figures_jiim")


def apply(base_pt=8):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans",
                            "Nimbus Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
        "font.size": base_pt, "axes.titlesize": base_pt, "axes.labelsize": base_pt,
        "xtick.labelsize": base_pt, "ytick.labelsize": base_pt,
        "legend.fontsize": base_pt, "figure.titlesize": base_pt,
        "lines.linewidth": 0.9, "axes.linewidth": 0.6, "patch.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 600, "figure.dpi": 150,
    })


def size(width="double", height_mm=None, aspect=0.6):
    """Figure size in inches for a named JIIM width and a height in mm."""
    w = WIDTHS[width] if isinstance(width, str) else float(width)
    h = height_mm if height_mm is not None else w * aspect
    if h > MAX_H:
        raise ValueError(f"height {h:.0f} mm exceeds the {MAX_H} mm limit")
    return (w * MM, h * MM)


def save(fig, n, kind="vector", outdir=OUTDIR):
    """Write FigN.pdf plus FigN.eps (vector) or FigN.tif at 600 dpi (combination)."""
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, f"Fig{n}")
    kw = dict(bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base + ".pdf", **kw)
    if kind == "vector":
        fig.savefig(base + ".eps", **kw)
    elif kind == "combination":
        fig.savefig(base + ".png", dpi=600, **kw)
        from PIL import Image
        Image.open(base + ".png").convert("RGB").save(
            base + ".tif", dpi=(600, 600), compression="tiff_lzw")
        os.remove(base + ".png")
    else:
        raise ValueError("kind must be 'vector' or 'combination'")
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = bb.width * 25.4 + 1.0, bb.height * 25.4 + 1.0   # size after cropping
    ok = w <= WIDTHS["double"] + 0.5 and h <= MAX_H
    print(f"saved {base}.pdf/{'eps' if kind == 'vector' else 'tif'}  "
          f"{w:.0f} x {h:.0f} mm  {'OK' if ok else 'TOO LARGE'}")
