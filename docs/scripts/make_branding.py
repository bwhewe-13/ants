"""Render the logo, favicon and social preview into docs/_static.

The mark is a faceted ant drawn in the same inferno shades as the
NeutronDiffusion hex core: a yellow head, orange thorax and a gaster that
darkens to purple at the tip.  The wordmark is Inter SemiBold converted to
outlines, so the SVGs render the same without the font installed.  Inter is
only needed to regenerate them:

    python docs/scripts/make_branding.py [--font path/to/Inter-SemiBold.otf]
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path as MplPath

OUT = Path(__file__).resolve().parents[1] / "_static"

# The ant is drawn in its own units, y up, about 120 tall.
GAP = 1.5  # space between neighboring body segments
CORNER = 0.75  # corner rounding radius
LEG = 2.8  # leg and antenna stroke width

INK = {"light": "#1f2433", "dark": "#e8eaf0"}
DARK_BG = "#0f1117"
MUTED = "#9aa0b4"

# Logo layout, in viewBox units: the mark fills 112 of a 128 square, and the
# wordmark sits to its right.
BOX = 128.0
MARK = 112.0
WORD_X, WORD_BASELINE, WORD_SIZE, WORD_SPACING = 142.0, 84.0, 58.0, -1.0
WORD = "ants"


def mirror(points):
    return [(-x, y) for x, y in points]


def ant():
    """Body segments as (polygon, inferno level), and legs as polylines.

    The left half is spelled out and mirrored; the antennae count as legs.
    """
    head = [
        (-12.0, 40.0),
        (-6.0, 50.39),
        (6.0, 50.39),
        (12.0, 40.0),
        (6.0, 29.61),
        (-6.0, 29.61),
    ]
    thorax = [
        (-6.0, 29.61),
        (6.0, 29.61),
        (9.5, 19.0),
        (6.0, 8.0),
        (-6.0, 8.0),
        (-9.5, 19.0),
    ]
    petiole = [(-6.0, 8.0), (6.0, 8.0), (17.0, -7.0), (0.0, -14.0), (-17.0, -7.0)]
    gaster = [(-17.0, -7.0), (0.0, -14.0), (0.0, -31.0), (-20.0, -25.0)]
    tip = [(-20.0, -25.0), (0.0, -31.0), (0.0, -58.0), (-13.0, -46.0)]
    segments = [
        (head, 0.85),
        (thorax, 0.7),
        (petiole, 0.55),
        (gaster, 0.3),
        (mirror(gaster), 0.3),
        (tip, 0.0),
        (mirror(tip), 0.0),
    ]

    legs = [
        [(-6.0, 25.0), (-22.0, 32.0), (-30.0, 47.0)],
        [(-8.0, 18.0), (-29.0, 14.0), (-43.0, 0.0)],
        [(-6.0, 11.0), (-25.0, -6.0), (-33.0, -34.0)],
        [(-4.0, 49.0), (-12.0, 62.0)],
    ]
    legs += [mirror(leg) for leg in legs]

    polys = [np.array(p, dtype=float) for p, _ in segments]
    levels = np.array([v for _, v in segments])
    return polys, levels, [np.array(leg) for leg in legs]


def cross(a, b):
    """Z component of the cross product of two 2-D vectors."""
    return a[0] * b[1] - a[1] * b[0]


def inset(poly, d):
    """Move every edge of a convex polygon inward by d."""
    x, y = poly[:, 0], poly[:, 1]
    ccw = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0
    edge = np.roll(poly, -1, axis=0) - poly
    normal = np.column_stack([-edge[:, 1], edge[:, 0]])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    if not ccw:
        normal = -normal
    start = poly + d * normal
    out = []
    for i in range(len(poly)):
        # Corner i joins the offset edges i - 1 and i.
        p, r = start[i - 1], edge[i - 1]
        q, s = start[i], edge[i]
        t = cross(q - p, s) / cross(r, s)
        out.append(p + t * r)
    return np.array(out)


def shrink(poly):
    """Inset for rounded corners and gaps.

    Once the outline is stroked by CORNER with round joins, neighbors are GAP
    apart.
    """
    return inset(poly, 0.5 * GAP + CORNER)


def colors(levels):
    """Hex colors for inferno levels in [0, 1], on the NeutronDiffusion scale."""
    cmap = matplotlib.colormaps["inferno"]
    return [matplotlib.colors.to_hex(cmap(0.22 + 0.76 * v)) for v in levels]


def fmt(v):
    """Format a coordinate with at most two decimals."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


# ---------------------------------------------------------------------------
# Mark
# ---------------------------------------------------------------------------


def mark_transform(polys, legs, size, x0, y0):
    """Map ant coordinates into a size x size box at (x0, y0), y down."""
    pts = np.vstack(polys + legs)
    lo, hi = pts.min(axis=0) - 0.5 * LEG, pts.max(axis=0) + 0.5 * LEG
    span = (hi - lo).max()
    scale = size / span
    mid = 0.5 * (lo + hi)

    def to_box(p):
        q = (p - mid) * scale
        return np.column_stack([x0 + 0.5 * size + q[:, 0], y0 + 0.5 * size - q[:, 1]])

    return to_box, scale


def mark_svg(mark, size, x0, y0, ink):
    """SVG elements for the mark in a size x size box at (x0, y0)."""
    polys, fills, legs = mark
    to_box, scale = mark_transform(polys, legs, size, x0, y0)
    lines = [
        f'<g fill="none" stroke="{ink}" stroke-width="{fmt(LEG * scale)}" '
        f'stroke-linecap="round" stroke-linejoin="round">'
    ]
    for leg in legs:
        pts = " ".join(f"{fmt(x)},{fmt(y)}" for x, y in to_box(leg))
        lines.append(f'<polyline points="{pts}"/>')
    lines.append("</g>")
    lines.append(
        f'<g stroke-width="{fmt(2.0 * CORNER * scale)}" stroke-linejoin="round">'
    )
    for poly, color in zip(polys, fills):
        pts = " ".join(f"{fmt(x)},{fmt(y)}" for x, y in to_box(shrink(poly)))
        lines.append(f'<polygon points="{pts}" fill="{color}" stroke="{color}"/>')
    lines.append("</g>")
    return "\n".join(lines)


def draw_mark(ax, mark, size, x0, y0, ink):
    """Draw the mark onto a pixel-space matplotlib axis."""
    polys, fills, legs = mark
    to_box, scale = mark_transform(polys, legs, size, x0, y0)
    # Figure coordinates are in pixels with y down, so the linewidth in points
    # is pixels * 72 / dpi.
    pt = scale * 72.0 / ax.figure.dpi
    for leg in legs:
        q = to_box(leg)
        ax.plot(
            q[:, 0],
            q[:, 1],
            color=ink,
            linewidth=LEG * pt,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
    for poly, color in zip(polys, fills):
        ax.add_patch(
            Polygon(
                to_box(shrink(poly)),
                closed=True,
                facecolor=color,
                edgecolor=color,
                linewidth=2.0 * CORNER * pt,
                joinstyle="round",
                zorder=3,
            )
        )


# ---------------------------------------------------------------------------
# Wordmark
# ---------------------------------------------------------------------------


class OutlinePen(BasePen):
    """Collects glyph outlines as (code, points) for both SVG and matplotlib."""

    def __init__(self, glyph_set, transform):  # noqa: D107
        super().__init__(glyph_set)
        self.transform = transform
        self.segments = []

    def _xy(self, p):
        return self.transform(*p)

    def _moveTo(self, p):
        self.segments.append(("M", [self._xy(p)]))

    def _lineTo(self, p):
        self.segments.append(("L", [self._xy(p)]))

    def _curveToOne(self, p1, p2, p3):
        self.segments.append(("C", [self._xy(p) for p in (p1, p2, p3)]))

    def _qCurveToOne(self, p1, p2):
        self.segments.append(("Q", [self._xy(p) for p in (p1, p2)]))

    def _closePath(self):
        self.segments.append(("Z", []))


def wordmark(font_path, size, x, baseline, spacing):
    """Outline WORD; returns the segments and the x where the text ends."""
    font = TTFont(font_path)
    glyphs = font.getGlyphSet()
    cmap = font.getBestCmap()
    scale = size / font["head"].unitsPerEm
    segments = []
    pen_x = x
    for ch in WORD:
        name = cmap[ord(ch)]
        ox = pen_x

        def transform(gx, gy, ox=ox):
            return (ox + gx * scale, baseline - gy * scale)

        pen = OutlinePen(glyphs, transform)
        glyphs[name].draw(pen)
        segments += pen.segments
        pen_x += font["hmtx"][name][0] * scale + spacing
    return segments, pen_x - spacing


def wordmark_svg(segments, color):
    """SVG path for the outlined wordmark."""
    d = []
    for code, pts in segments:
        d.append(code + " ".join(f"{fmt(px)} {fmt(py)}" for px, py in pts))
    return f'<path fill="{color}" d="{"".join(d)}"/>'


def wordmark_path(segments):
    """Matplotlib path for the outlined wordmark."""
    codes_for = {
        "M": [MplPath.MOVETO],
        "L": [MplPath.LINETO],
        "C": [MplPath.CURVE4] * 3,
        "Q": [MplPath.CURVE3] * 2,
    }
    verts, codes = [], []
    start = None
    for code, pts in segments:
        if code == "Z":
            verts.append(start)
            codes.append(MplPath.CLOSEPOLY)
            continue
        if code == "M":
            start = pts[0]
        verts += pts
        codes += codes_for[code]
    return MplPath(verts, codes)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def svg(width, height, body, background=None):
    """Wrap body in an SVG document."""
    bg = (
        f'<rect width="{fmt(width)}" height="{fmt(height)}" fill="{background}"/>\n'
        if background
        else ""
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {fmt(width)} {fmt(height)}" '
        f'width="{fmt(width)}" height="{fmt(height)}">\n'
        f"<title>ants</title>\n{bg}{body}\n</svg>\n"
    )


def write_logos(mark, font):
    """Write the light and dark logos and the icon; return the wordmark."""
    margin = 0.5 * (BOX - MARK)
    segments, end = wordmark(font, WORD_SIZE, WORD_X, WORD_BASELINE, WORD_SPACING)
    width = np.ceil(end + margin)
    for theme, name in (("light", "logo.svg"), ("dark", "logo-dark.svg")):
        body = (
            mark_svg(mark, MARK, margin, margin, INK[theme])
            + "\n"
            + wordmark_svg(segments, INK[theme])
        )
        (OUT / name).write_text(svg(width, BOX, body))
    (OUT / "icon.svg").write_text(
        svg(BOX, BOX, mark_svg(mark, MARK, margin, margin, INK["light"]))
    )
    return segments


def write_favicon(mark, px=64):
    """Write the mark as a transparent px x px PNG."""
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, px)
    ax.set_ylim(px, 0)
    ax.axis("off")
    draw_mark(
        ax,
        mark,
        px * MARK / BOX,
        px * (BOX - MARK) / (2 * BOX),
        px * (BOX - MARK) / (2 * BOX),
        INK["light"],
    )
    fig.savefig(OUT / "favicon.png", transparent=True)
    plt.close(fig)


def write_social_preview(mark, segments, regular_font):
    """Write the 1280 x 640 social preview on the dark background."""
    from matplotlib.font_manager import FontProperties

    w, h = 1280, 640
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor=DARK_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    size, gutter, scale = 360.0, 64.0, 1.9
    tag = FontProperties(fname=regular_font, size=21)
    lines = [
        "Discrete ordinates neutron transport in 1-D and 2-D",
        "Cython solvers, Python interface",
    ]

    # The logo wordmark, scaled up, with its origin moved to (0, 0) so the
    # text block can be placed once its width is known.
    path = wordmark_path(segments)
    word = (path.vertices - [WORD_X, WORD_BASELINE]) * scale
    texts = [
        ax.text(0.0, 0.0, s, color=MUTED, fontproperties=tag, va="baseline")
        for s in lines
    ]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_w = max(t.get_window_extent(renderer).width for t in texts)
    block_w = max(word[:, 0].max() - word[:, 0].min(), text_w)

    x0 = 0.5 * (w - (size + gutter + block_w))
    draw_mark(ax, mark, size, x0, 0.5 * (h - size), INK["dark"])

    # Ascender of the wordmark to the last tagline baseline, centered on the mark.
    tx = x0 + size + gutter
    top = word[:, 1].min()
    spacing = 1.45 * tag.get_size_in_points() * fig.dpi / 72.0
    gap = 1.9 * spacing
    baseline = 0.5 * h - 0.5 * (gap + spacing * (len(lines) - 1) + top)
    ax.add_patch(
        PathPatch(
            MplPath(word + [tx - word[:, 0].min(), baseline], path.codes),
            facecolor=INK["dark"],
            edgecolor="none",
        )
    )
    for k, t in enumerate(texts):
        t.set_position((tx, baseline + gap + k * spacing))
    fig.savefig(OUT / "social-preview.png", facecolor=DARK_BG)
    plt.close(fig)


def find_font(style):
    """Find an installed Inter font file of the given style."""
    roots = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.local/share/fonts"),
        os.path.expanduser("~/.fonts"),
        "/Library/Fonts",
        os.path.expanduser("~/Library/Fonts"),
        "C:/Windows/Fonts",
    ]
    for root in roots:
        for ext in ("otf", "ttf"):
            hits = glob.glob(f"{root}/**/Inter-{style}.{ext}", recursive=True)
            if hits:
                return hits[0]
    raise SystemExit(f"Inter-{style} not found; pass --font")


def main():
    """Regenerate every branding asset."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--font", help="Inter SemiBold (.otf or .ttf)")
    parser.add_argument("--regular-font", help="Inter Regular, for the preview tagline")
    args = parser.parse_args()
    font = args.font or find_font("SemiBold")
    regular = args.regular_font or find_font("Regular")

    OUT.mkdir(parents=True, exist_ok=True)
    polys, levels, legs = ant()
    mark = (polys, colors(levels), legs)
    segments = write_logos(mark, font)
    write_favicon(mark)
    write_social_preview(mark, segments, regular)
    for name in (
        "logo.svg",
        "logo-dark.svg",
        "icon.svg",
        "favicon.png",
        "social-preview.png",
    ):
        print(OUT / name)


if __name__ == "__main__":
    main()
