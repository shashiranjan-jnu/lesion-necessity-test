"""Check FigN files against JIIM artwork rules. Run in the folder with the figures:
    python3 check_figures.py
Reports size in mm, embedded fonts (PDF/EPS) and resolution/mode (TIFF)."""
import glob, re, subprocess
from PIL import Image
MAXW, MAXH = 174.5, 234
def pdf_mm(f):
    out = subprocess.run(["pdfinfo", f], capture_output=True, text=True).stdout
    w, h = map(float, re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", out).groups())
    return w * 25.4 / 72, h * 25.4 / 72
def fonts(f):
    out = subprocess.run(["pdffonts", f], capture_output=True, text=True).stdout.splitlines()[2:]
    return {(l.split()[0].split("+")[-1], l.split()[-5]) for l in out if l.strip()}
for f in sorted(glob.glob("Fig*.pdf") + glob.glob("Fig*.tif") + glob.glob("Fig*.eps")):
    if f.endswith(".pdf"):
        w, h = pdf_mm(f); fs = fonts(f)
        emb = all(e == "yes" for _, e in fs)
        fam = sorted({n for n, _ in fs})
        ok = w <= MAXW and h <= MAXH and emb
        print(f"{f:10s} {w:5.0f} x {h:3.0f} mm  embedded={emb}  {fam}  {'OK' if ok else 'FIX'}")
    elif f.endswith(".tif"):
        im = Image.open(f); dpi = im.info.get("dpi", (0, 0))[0]
        w, h = im.size[0] / dpi * 25.4, im.size[1] / dpi * 25.4
        ok = dpi >= 600 and im.mode == "RGB" and w <= MAXW and h <= MAXH
        print(f"{f:10s} {w:5.0f} x {h:3.0f} mm  {dpi:.0f} dpi  {im.mode}  {'OK' if ok else 'FIX'}")
    else:
        print(f"{f:10s} EPS present")
