# /// script
# requires-python = ">=3.12"
# dependencies = ["pillow>=10.1", "fonttools>=4.40"]
# ///
"""Draw the biotimbre logo, ((-)) in a monospace font, as the icon files the
package ships.

    uv run scripts/make_icons.py [FONT]

FONT is a TrueType or OpenType file, DejaVu Sans Mono Bold by default. Writes
src/biotimbre/assets/biotimbre.{svg,png,ico}. The text is drawn from the
font's glyph outlines rather than set as text, so every file looks the same
everywhere and the SVG needs no font installed.
"""

import sys
from pathlib import Path

from fontTools.pens.basePen import BasePen
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw

TEXT = "((-))"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
# The dark theme's header (header_bg and header_text of DEFAULT_DARK_THEME in
# src/biotimbre/config.py).
BACKGROUND = "#0d1a05"
FOREGROUND = "#fcfee5"

SIZE = 512  # design grid; everything below is in its units
CORNER = 104  # background corner radius
WIDTH = 410  # width of the text's ink, centred on the square

OUT = Path(__file__).resolve().parent.parent / "src" / "biotimbre" / "assets"
ICO_SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256]


class Polygons(BasePen):
    """Collects each contour as a list of points, curves cut into short lines."""

    STEPS = 24

    def __init__(self, glyph_set=None):
        super().__init__(glyph_set)
        self.contours = []

    def _moveTo(self, pt):
        self.contours.append([pt])

    def _lineTo(self, pt):
        self.contours[-1].append(pt)

    def _curveToOne(self, p1, p2, p3):
        p0 = self._getCurrentPoint()
        for i in range(1, self.STEPS + 1):
            t = i / self.STEPS
            u = 1 - t
            self.contours[-1].append(
                tuple(
                    u**3 * a + 3 * u * u * t * b + 3 * u * t * t * c + t**3 * d
                    for a, b, c, d in zip(p0, p1, p2, p3)
                )
            )

    def _qCurveToOne(self, p1, p2):
        p0 = self._getCurrentPoint()
        for i in range(1, self.STEPS + 1):
            t = i / self.STEPS
            u = 1 - t
            self.contours[-1].append(
                tuple(u * u * a + 2 * u * t * b + t * t * c for a, b, c in zip(p0, p1, p2))
            )


def draw_text(font, pen):
    """Draw TEXT into pen, in font units with y pointing down."""
    glyphs = font.getGlyphSet()
    cmap = font.getBestCmap()
    x = 0
    for char in TEXT:
        name = cmap[ord(char)]
        glyphs[name].draw(TransformPen(pen, (1, 0, 0, -1, x, 0)))
        x += glyphs[name].width


def placement(font):
    """The transform from font units (y down) to the design grid."""
    bounds = BoundsPen(None)
    draw_text(font, bounds)
    x0, y0, x1, y1 = bounds.bounds
    k = WIDTH / (x1 - x0)
    # The ink's centre goes to the square's centre.
    return (k, 0, 0, k, SIZE / 2 - k * (x0 + x1) / 2, SIZE / 2 - k * (y0 + y1) / 2)


def svg(font):
    path = SVGPathPen(None, ntos=lambda n: f"{n:.2f}".rstrip("0").rstrip("."))
    draw_text(font, TransformPen(path, placement(font)))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}">'
        f'<rect width="{SIZE}" height="{SIZE}" rx="{CORNER}" fill="{BACKGROUND}"/>'
        f'<path d="{path.getCommands()}" fill="{FOREGROUND}"/></svg>\n'
    )


def raster(font, size, scale=8):
    """The logo at size x size pixels, drawn large and scaled down for smooth edges."""
    k = size * scale / SIZE
    img = Image.new("RGBA", (size * scale,) * 2, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, SIZE * k - 1, SIZE * k - 1), CORNER * k, fill=BACKGROUND)
    a, b, c, d, e, f = placement(font)
    polygons = Polygons()
    draw_text(font, TransformPen(polygons, (a * k, b, c, d * k, e * k, f * k)))
    # Filled one by one, which is right for these glyphs: none has a hole.
    for contour in polygons.contours:
        draw.polygon(contour, fill=FOREGROUND)
    return img.resize((size, size), Image.LANCZOS)


def main():
    font = TTFont(sys.argv[1] if len(sys.argv) > 1 else FONT)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "biotimbre.svg").write_text(svg(font))
    raster(font, 512).save(OUT / "biotimbre.png", optimize=True)
    # Each size drawn on its own, so small ones are not blurred copies of 256.
    images = [raster(font, s) for s in ICO_SIZES]
    images[-1].save(
        OUT / "biotimbre.ico",
        sizes=[(s, s) for s in ICO_SIZES],
        append_images=images[:-1],
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
