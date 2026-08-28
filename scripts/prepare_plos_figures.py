r"""
Export the six main-text figures to PLOS figure specification
==============================================================

PLOS requires TIFF (or EPS) at 300--600 dpi, 789--2250 px wide, at most 2625 px
tall, RGB or grayscale (no alpha), LZW-compressed and under 10 MB. The rendered
PNGs under ``figures/`` are 160--200 dpi RGBA and up to 2521 px wide, so each is
flattened onto white, downscaled only if it exceeds the width ceiling, and
written with a 300 dpi tag.

Downscaling is the only resampling done: nothing is upscaled, because that would
add pixels without adding detail. Every figure regenerates from the scripts in
``scripts/`` if a higher-resolution version is ever requested at revision.

Usage:
    python3 scripts/prepare_plos_figures.py
"""
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "figures"
DEST = ROOT / "manuscript" / "data_paper" / "figures"

def main_text_figures():
    """Main-text figures, in order, read from the .tex.

    The main/S1 split is positional on \\appendix, exactly as in make_plos_submission.py,
    so this cannot drift out of step with the manuscript the way a hand-kept list did.
    """
    tex = (ROOT / "manuscript_PRX.tex").read_text(encoding="utf-8")
    body = tex[:tex.index("\\appendix")]
    stems = re.findall(r"\\includegraphics\[[^\]]*\]\{figures/([^}]+?)\.png\}", body)
    return [(f"Fig{i}", s) for i, s in enumerate(stems, 1)]


FIGURES = main_text_figures()

MAX_W, MAX_H, MIN_W, DPI = 2250, 2625, 789, 300


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name, stem in FIGURES:
        im = Image.open(SRC / f"{stem}.png")
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            flat = Image.new("RGB", im.size, "white")
            flat.paste(im, mask=im.split()[-1])
            im = flat
        else:
            im = im.convert("RGB")

        w, h = im.size
        scale = min(MAX_W / w, MAX_H / h, 1.0)
        if scale < 1.0:
            im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

        out = DEST / f"{name}.tif"
        im.save(out, format="TIFF", compression="tiff_lzw", dpi=(DPI, DPI))
        mb = out.stat().st_size / 1e6
        flags = []
        if im.width < MIN_W:
            flags.append("TOO NARROW")
        if mb > 10:
            flags.append("OVER 10 MB")
        print(f"{name:<5} {stem:<15} {w}x{h} -> {im.width}x{im.height}  "
              f"{mb:5.1f} MB  {' '.join(flags) or 'ok'}")


if __name__ == "__main__":
    main()
