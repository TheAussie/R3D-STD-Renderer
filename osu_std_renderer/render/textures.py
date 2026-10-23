"""Procedural "Argon-ish" default textures — the taiko `argon/` playbook
(PIL/numpy-baked) applied to std. Original ModIcon PNGs are the exception;
their provenance and licences are in assets/hud/mod-icons/.

Everything the Phase-1 scene draws is baked here at init:

  disc          filled hit-circle disc, near-white with a subtle radial
                lift toward the rim so combo-colour tinting reads with a
                little depth (tint = multiply, gl.py's u_color)
  ring          the hitcircleoverlay stand-in: a white border ring whose
                thickness fraction (0.128 of the radius) matches the
                slider-body border (slider_body.BASE_BORDER_PORTION x2
                visual family) so circles and slider borders read as one
                skin
  approach      thinner white ring for the approach circle (tinted combo)
  glow          radial-falloff dot (cursor glow / trail)
  digit_0..9    combo-number glyphs, DejaVu Sans Bold (system font; the
                repo bundles no fonts — same decision the taiko engine
                took before Torus landed), baked white for tinting
  miss_x        AA diagonal cross for the miss judgment popup (tinted red
                by the scene — the ruleset phase's judgment sprites)
  dot           small filled AA disc — the sliderscorepoint (slider tick)
                and followpoint procedural stand-in (white, untinted)
  arrow         right-pointing solid arrow — the reversearrow stand-in
                (white; the scene rotates it along the path tangent)

HUD PHASE additions (render/hud.py consumes these; §4.6 elements, drawn
in the §5.3 virtual-1080p UI space):

  glyph_<ch>    the HUD text set (mania text.py's PIL pattern, but baked
                ONCE per glyph instead of per-string): digits + the
                punctuation/letters the HUD composes (".%x,KMSABCDUR").
                All share one vertical extent so runs baseline-align;
                digit advance is uniform (mono) so the score roll and UR
                readout never jitter.
  key_square    rounded-square key-overlay cell (border + faint fill)
  vignette      edge-weighted red-pulse mask (combo-break feedback; the
                mania full-frame wash, reshaped to an edge vignette)
  pie_00..N-1   progress-pie fill masks (§4.6 Score.ProgressBar "Pie"),
                quantized fractions; a full pie reuses `disc`
  pie_ring      thin outline ring around the progress pie
  tri_down      small downward triangle (hit-error moving-average arrow)

All textures are white/greyscale so the sprite tint does the colouring —
the §3.2 default ComboColors rotate per combo set at draw time.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

DISC_SIZE = 256
RING_SIZE = 256
APPROACH_SIZE = 512          # drawn up to 4x scale (§2.5) — bake bigger
GLOW_SIZE = 128
DIGIT_HEIGHT = 128

RING_THICKNESS = 0.128       # of radius; matches lazer BORDER_PORTION
APPROACH_THICKNESS = 0.09
_AA_PX = 1.5                 # texture-space anti-alias band

# The HUD's own font, DejaVu Sans Bold, ships with the engine and is tried
# first, so every platform draws the same glyphs. It used to be looked up only
# in Linux system folders: a Mac or a Windows node found none, fell through to
# Pillow's built-in font (about 10 px whatever size is asked for), and drew
# every DejaVu label tiny. The Argon text shrank with it, to 65% (digits 55%),
# because ARGON_GLYPH_CAP_SCALE / ARGON_DIGIT_CAP_SCALE below are measured
# against this font at import. The bundled file is the one Ubuntu 24.04 ships
# (2.37, sha256 5c1247ac...2ce895), which is what the Linux nodes were already
# drawing with. Licence: assets/fonts/LICENSE_DEJAVU.txt.
BUNDLED_FONT_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "assets", "fonts", "DejaVuSans-Bold.ttf"))
_FONT_CANDIDATES = (
    BUNDLED_FONT_PATH,
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
)

# --- Argon-league text font (bundled, OFL) -----------------------------------
# lazer draws the skinless/Argon text (judgment GREAT/OK/MEH/MISS, HUD labels,
# hit-circle numbers, results) in Torus — a thin geometric-rounded sans. Torus
# is non-commercial, so we bundle Nunito instead: the closest OFL / free-for-
# commercial geometric-rounded stand-in (SIL OFL 1.1; see assets/fonts/OFL.txt).
# Shipped as the variable font, weight pinned to Medium at load — Torus reads
# Regular/SemiBold, NOT the DejaVu-Bold weight this replaces. The custom-skin /
# legacy path keeps DejaVu (_load_font / bake_glyphs / bake_digits) untouched,
# so those renders stay byte-identical.
ARGON_FONT_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "assets", "fonts", "Nunito[wght].ttf"))
ARGON_FONT_WEIGHT = 500          # Nunito Medium (variable-font `wght` axis)

# Cap-fill = cap('E') height / baked sprite vertical extent, MEASURED at the
# real bake (font px = int(DIGIT_HEIGHT*0.95), union bbox over the charset,
# pad=4). Nunito fills slightly less of the sprite than DejaVu, so an Argon run
# at the SAME requested (DejaVu-tuned) height would render a slightly smaller
# visible cap. Each Argon run multiplies its requested height by DejaVu/Argon
# so the VISIBLE text size is unchanged by the font swap (re-derive sizing):
#   glyph bank (HUD_CHARSET): DejaVu 0.7154, Nunito Medium 0.7025 → ×1.0184
#   digit bank ("0123456789"): DejaVu 0.8800, Nunito Medium 0.8854 → ×0.9939
# Judgment: requested size-20 × ~0.72 cap-fill ≈ 14.4 osu!px cap, matching
# lazer's Torus size-20 — preserved across the swap by ARGON_GLYPH_CAP_SCALE.
DEJAVU_GLYPH_CAP_FILL = 0.7154
ARGON_GLYPH_CAP_FILL = 0.7025
ARGON_GLYPH_CAP_SCALE = DEJAVU_GLYPH_CAP_FILL / ARGON_GLYPH_CAP_FILL
DEJAVU_DIGIT_CAP_FILL = 0.8800
ARGON_DIGIT_CAP_FILL = 0.8854
ARGON_DIGIT_CAP_SCALE = DEJAVU_DIGIT_CAP_FILL / ARGON_DIGIT_CAP_FILL


def _dist_grid(size: int) -> np.ndarray:
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    return np.hypot(xx - c, yy - c)


def bake_disc(size: int = DISC_SIZE, inner_shade: float = 0.88) -> np.ndarray:
    """Filled disc, AA edge; radial shade lifts rim to full white so a
    combo tint keeps a bright edge (flat tints look like paper cutouts)."""
    d = _dist_grid(size)
    radius = size / 2.0 - 2.0
    alpha = np.clip((radius - d) / _AA_PX, 0.0, 1.0)
    shade = inner_shade + (1.0 - inner_shade) * np.clip(d / radius, 0.0, 1.0)
    rgba = np.empty((size, size, 4), dtype=np.uint8)
    grey = np.round(shade * 255.0).astype(np.uint8)
    rgba[..., 0] = grey
    rgba[..., 1] = grey
    rgba[..., 2] = grey
    rgba[..., 3] = np.round(alpha * 255.0).astype(np.uint8)
    return rgba


def bake_ring(size: int = RING_SIZE,
              thickness_frac: float = RING_THICKNESS) -> np.ndarray:
    """White annulus (border inward from the texture edge), AA both edges."""
    d = _dist_grid(size)
    radius = size / 2.0 - 2.0
    thickness = max(radius * thickness_frac, 2.0)
    outer = np.clip((radius - d) / _AA_PX, 0.0, 1.0)
    inner = np.clip((d - (radius - thickness)) / _AA_PX, 0.0, 1.0)
    alpha = outer * inner
    rgba = np.full((size, size, 4), 255, dtype=np.uint8)
    rgba[..., 3] = np.round(alpha * 255.0).astype(np.uint8)
    return rgba


def bake_glow(size: int = GLOW_SIZE, power: float = 2.2) -> np.ndarray:
    """Soft radial falloff dot for the cursor glow / trail (additive)."""
    d = _dist_grid(size)
    radius = size / 2.0 - 1.0
    alpha = np.clip(1.0 - d / radius, 0.0, 1.0) ** power
    rgba = np.full((size, size, 4), 255, dtype=np.uint8)
    rgba[..., 3] = np.round(alpha * 255.0).astype(np.uint8)
    return rgba


def bake_smoke_puff(size: int = 64, power: float = 1.7,
                    core: float = 0.16) -> np.ndarray:
    """Soft round smoke dab for the replay-Smoke trail (additive)."""
    d = _dist_grid(size)
    radius = size / 2.0 - 1.0
    r = np.clip(d / radius, 0.0, 1.0)
    shoulder = np.clip((1.0 - r) / (1.0 - core), 0.0, 1.0) ** power
    alpha = np.where(r <= core, 1.0, shoulder)
    rgba = np.full((size, size, 4), 255, dtype=np.uint8)
    rgba[..., 3] = np.round(alpha * 255.0).astype(np.uint8)
    return rgba


def bake_flashlight(size: int = 1024, core: float = 0.45) -> np.ndarray:
    """OsuModFlashlight overlay texture: BLACK RGB with a radial alpha ramp —
    fully transparent in the lit core, smoothstep up to fully opaque at the
    texture mid-edge and beyond. Drawn as one cursor-centred quad whose uv is
    scaled so the mid-edge (normalised distance 1) lands on the flashlight
    radius; sampled with clamp-to-edge so everything past the radius is solid
    black (the playfield outside the flashlight is invisible). `core` is the
    fraction of the radius that stays fully lit before the soft falloff — a
    stand-in for Flashlight.cs's FlashlightSmoothness gradient."""
    d = _dist_grid(size)
    dn = d / (size / 2.0)                       # 1.0 at the mid-edge
    x = np.clip((dn - core) / (1.0 - core), 0.0, 1.0)
    alpha = x * x * (3.0 - 2.0 * x)             # smoothstep(core, 1, dn)
    rgba = np.zeros((size, size, 4), dtype=np.uint8)   # black RGB
    rgba[..., 3] = np.round(alpha * 255.0).astype(np.uint8)
    return rgba


def bake_miss_x(size: int = 128, thickness: float = 0.16,
                arm: float = 0.42) -> np.ndarray:
    """White AA diagonal cross (the miss popup, tinted red at draw time).
    thickness/arm are fractions of the texture size."""
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    u, v = xx - c, yy - c
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    d1 = np.abs(u - v) * inv_sqrt2          # distance to the / diagonal
    d2 = np.abs(u + v) * inv_sqrt2          # distance to the \ diagonal
    half_t = size * thickness / 2.0
    bar1 = np.clip((half_t - d1) / _AA_PX, 0.0, 1.0)
    bar2 = np.clip((half_t - d2) / _AA_PX, 0.0, 1.0)
    reach = np.clip((size * arm - np.hypot(u, v)) / _AA_PX, 0.0, 1.0)
    alpha = np.maximum(bar1, bar2) * reach
    rgba = np.full((size, size, 4), 255, dtype=np.uint8)
    rgba[..., 3] = np.round(alpha * 255.0).astype(np.uint8)
    return rgba


def bake_dot(size: int = 64) -> np.ndarray:
    """Small filled white disc, AA edge (slider tick / followpoint dot)."""
    d = _dist_grid(size)
    radius = size / 2.0 - 2.0
    alpha = np.clip((radius - d) / _AA_PX, 0.0, 1.0)
    rgba = np.full((size, size, 4), 255, dtype=np.uint8)
    rgba[..., 3] = np.round(alpha * 255.0).astype(np.uint8)
    return rgba


def bake_arrow(size: int = 256) -> np.ndarray:
    """White right-pointing arrow (shaft + head), supersampled AA — the
    reversearrow fallback, sized to sit inside the end circle when drawn
    at the circle diameter."""
    s4 = size * 4
    pts = [(0.16, 0.40), (0.50, 0.40), (0.50, 0.24), (0.84, 0.50),
           (0.50, 0.76), (0.50, 0.60), (0.16, 0.60)]
    img = Image.new("RGBA", (s4, s4), (0, 0, 0, 0))
    ImageDraw.Draw(img).polygon([(x * s4, y * s4) for x, y in pts],
                                fill=(255, 255, 255, 255))
    img = img.resize((size, size), Image.LANCZOS)   # cheap supersampled AA
    return np.asarray(img, dtype=np.uint8).copy()


def _load_font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for cand in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(cand, px)
        except OSError:
            continue
    return ImageFont.load_default()  # bitmap fallback; blurry but functional


def _load_argon_font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """The bundled OFL Argon-league font (Nunito). Variable font: pin the
    `wght` axis to ARGON_FONT_WEIGHT. Falls back to DejaVu (_load_font) if the
    bundled asset is missing, so a stripped checkout still renders."""
    try:
        font = ImageFont.truetype(ARGON_FONT_PATH, px)
    except OSError:
        return _load_font(px)
    try:
        font.set_variation_by_axes([ARGON_FONT_WEIGHT])
    except Exception:
        pass       # non-variable FreeType build → keep the default instance
    return font


def bake_glyphs(chars: str, height: int = DIGIT_HEIGHT,
                loader=_load_font) -> dict[str, np.ndarray]:
    """Each char as a white RGBA glyph, all sharing one vertical extent
    (union of the glyph bboxes) so runs baseline-align when centred."""
    font = loader(int(height * 0.95))
    boxes = {}
    for ch in chars:
        try:
            boxes[ch] = font.getbbox(ch)
        except AttributeError:  # ancient PIL bitmap font
            w, h = font.getsize(ch)  # type: ignore[attr-defined]
            boxes[ch] = (0, 0, w, h)
    top = min(b[1] for b in boxes.values())
    bottom = max(b[3] for b in boxes.values())
    pad = 4
    out: dict[str, np.ndarray] = {}
    for ch in chars:
        x0, _, x1, _ = boxes[ch]
        w = (x1 - x0) + 2 * pad
        h = (bottom - top) + 2 * pad
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(img).text((pad - x0, pad - top), ch, font=font,
                                 fill=(255, 255, 255, 255))
        out[ch] = np.asarray(img, dtype=np.uint8).copy()
    return out


def bake_digits(height: int = DIGIT_HEIGHT) -> dict[str, np.ndarray]:
    """0-9 as white RGBA glyphs (the combo-number set; bake_glyphs with the
    original digit-only vertical extent)."""
    return bake_glyphs("0123456789", height)


def bake_argon_glyphs(chars: str,
                      height: int = DIGIT_HEIGHT) -> dict[str, np.ndarray]:
    """bake_glyphs in the bundled Argon-league font (Nunito) — the skinless
    judgment text + HUD labels. Same union-extent layout as bake_glyphs."""
    return bake_glyphs(chars, height, loader=_load_argon_font)


def bake_argon_digits(height: int = DIGIT_HEIGHT) -> dict[str, np.ndarray]:
    """0-9 hit-circle/combo numbers in the bundled Argon-league font."""
    return bake_glyphs("0123456789", height, loader=_load_argon_font)


# --- HUD textures (render/hud.py) ------------------------------------------------

# full uppercase set: HUD labels (ACCURACY/COMBO), key names, grades,
# SPIN!/CLEAR!/RPM, UR, time readouts ("-0:00") — plus lowercase 'p'
# for the pp counter's "pp" suffix and '×' for the custom-rate mod pill
# (e.g. "DT 1.3×"). Appending '×' is baked as its own glyph_× / aglyph_×
# texture and — verified — sits inside the existing union vertical extent
# (its bbox is narrower than the ascenders/'p' descender), so every other
# glyph bakes byte-identically and legacy renders are unchanged. '/' is
# appended for the free-tier URL watermark ("https://renderer.r3dwolfie.com/");
# it spans cap-top→baseline, inside the existing A-Z union extent, so it
# likewise bakes without disturbing any other glyph.
HUD_CHARSET = "0123456789.%x,:-!ABCDEFGHIJKLMNOPQRSTUVWXYZp×/"

# --- Runtime-derive the cap-fill scales from the ACTUAL bundled fonts -------
# The hand-derived constants above are a frozen snapshot; Pillow/freetype can
# change variable-font (Nunito wght) metrics between versions, drifting them
# from what actually bakes. Measure the live fonts here (identical to the
# drift-guard test: <cap> height / union sprite extent over <chars>) so both
# scales track the real bake and cannot drift. Falls back to the hand-derived
# constants if measurement raises.
def _measure_cap_fill(loader, chars, cap) -> float:
    font = loader(int(DIGIT_HEIGHT * 0.95))
    boxes = [font.getbbox(c) for c in chars]
    top, bot = min(b[1] for b in boxes), max(b[3] for b in boxes)
    cb = font.getbbox(cap)
    return (cb[3] - cb[1]) / ((bot - top) + 8)


try:
    DEJAVU_GLYPH_CAP_FILL = _measure_cap_fill(_load_font, HUD_CHARSET, "E")
    ARGON_GLYPH_CAP_FILL = _measure_cap_fill(_load_argon_font, HUD_CHARSET, "E")
    ARGON_GLYPH_CAP_SCALE = DEJAVU_GLYPH_CAP_FILL / ARGON_GLYPH_CAP_FILL
    DEJAVU_DIGIT_CAP_FILL = _measure_cap_fill(_load_font, "0123456789", "8")
    ARGON_DIGIT_CAP_FILL = _measure_cap_fill(_load_argon_font, "0123456789", "8")
    ARGON_DIGIT_CAP_SCALE = DEJAVU_DIGIT_CAP_FILL / ARGON_DIGIT_CAP_FILL
except Exception:
    pass  # keep the hand-derived fallbacks defined near the top of the module
# ---------------------------------------------------------------------------

PIE_STEPS = 48          # quantized progress-pie fill masks
PIE_SIZE = 96
KEY_SQUARE_SIZE = 128
VIGNETTE_SIZE = 256


def bake_key_square(size: int = KEY_SQUARE_SIZE, radius_frac: float = 0.18,
                    border_frac: float = 0.07,
                    fill_alpha: float = 0.28) -> np.ndarray:
    """Rounded square: solid AA border + faint interior fill. White — the
    HUD tints it (idle grey / pressed key colour)."""
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    half = size / 2.0 - 2.0
    r = radius_frac * size
    # signed distance to a rounded square (box SDF)
    qx = np.abs(xx - c) - (half - r)
    qy = np.abs(yy - c) - (half - r)
    d = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) \
        + np.minimum(np.maximum(qx, qy), 0.0) - r
    border = border_frac * size
    inside = np.clip(-d / _AA_PX, 0.0, 1.0)
    in_border = inside * np.clip((d + border) / _AA_PX, 0.0, 1.0)
    alpha = np.maximum(in_border, inside * fill_alpha)
    rgba = np.full((size, size, 4), 255, dtype=np.uint8)
    rgba[..., 3] = np.round(alpha * 255.0).astype(np.uint8)
    return rgba


def bake_vignette(size: int = VIGNETTE_SIZE, inner: float = 0.52,
                  outer: float = 1.0, power: float = 1.6) -> np.ndarray:
    """Edge-weighted mask: transparent centre, alpha ramping toward the
    frame edges/corners (distance normalized by the half-diagonal). Drawn
    full-screen with a red tint for the combo-break pulse."""
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    d = np.hypot(xx - c, yy - c) / (c * math.sqrt(2.0))
    alpha = np.clip((d - inner) / (outer - inner), 0.0, 1.0) ** power
    rgba = np.full((size, size, 4), 255, dtype=np.uint8)
    rgba[..., 3] = np.round(alpha * 255.0).astype(np.uint8)
    return rgba


def bake_pie(fraction: float, size: int = PIE_SIZE) -> np.ndarray:
    """Pie-fill mask: disc sector from 12 o'clock, clockwise, covering
    `fraction` of the turn. Radial edge AA'd; the angular edge is hard
    (quantized steps at PIE_STEPS are well under a frame's worth of
    progress drift at any sane fps)."""
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    dx, dy = xx - c, yy - c
    radius = size / 2.0 - 2.0
    disc = np.clip((radius - np.hypot(dx, dy)) / _AA_PX, 0.0, 1.0)
    # angle from the top (12 o'clock), clockwise, in turns [0, 1)
    phi = (np.arctan2(dx, -dy) / (2.0 * math.pi)) % 1.0
    alpha = disc * (phi <= max(fraction, 1e-9))
    rgba = np.full((size, size, 4), 255, dtype=np.uint8)
    rgba[..., 3] = np.round(alpha * 255.0).astype(np.uint8)
    return rgba


def stable_sector_vertices(progress: float) -> list[tuple[float, float]]:
    """Stable CircularProgress's unit-radius, 40-division triangle fan.

    Negative sweeps start at the fractional endpoint and finish at noon;
    positive sweeps start at noon. Retain the exact endpoint, including
    the source's possible second intro revolution (Progress >= -2).
    """
    progress = min(1.0, max(-2.0, progress))
    if progress == 0:
        return []
    start = -math.pi / 2
    end = start + progress * math.tau
    lo, hi = sorted((start, end))
    step = math.tau / 40
    angles = [lo + i * step for i in range(math.ceil((hi-lo)/step))]
    angles.append(hi)
    return [(0.0, 0.0)] + [(math.cos(a), math.sin(a)) for a in angles]


def bake_stable_sector(progress: float, alpha: float = 0.6,
                       size: int = 128) -> np.ndarray:
    """Raster stand-in for the native fan, without progress quantization.

    Supersampling approximates native edge AA. Separate fan revolutions
    composite independently, retaining stable's overlapping intro fill.
    Opacity is baked here so a second revolution blends correctly.
    """
    vertices = stable_sector_vertices(progress)
    high = size * 4
    mask = Image.new('L', (high, high))
    arc = vertices[1:]
    for start in range(0, max(len(arc)-1, 0), 40):
        points = [(0.0, 0.0)] + arc[start:start+41]
        layer = Image.new('L', (high, high))
        ImageDraw.Draw(layer).polygon(
            [((x+1)*high/2, (y+1)*high/2) for x, y in points], fill=round(alpha*255))
        mask = ImageChops.screen(mask, layer)
    mask = mask.resize((size, size), Image.Resampling.LANCZOS)
    rgba = np.full((size, size, 4), 255, dtype=np.uint8)
    rgba[..., 3] = np.asarray(mask)
    return rgba


def bake_stable_circularmetre(size: int = 336) -> np.ndarray:
    """Independent procedural equivalent of stock's 42-unit overlay.

    The inspected stock PNG is 42x42 (84x84 @2x), with a ring centred at
    radius 16 and a small centre dot. Its glow/AA are approximated here;
    no upstream artwork is bundled. Overlay and sector sizes are separate.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    radius = np.hypot((xx+.5)*42/size-21, (yy+.5)*42/size-21)
    ring = np.exp(-0.5*((radius-16)/0.45)**2)
    glow = 0.20*np.exp(-0.5*((radius-16)/1.5)**2)
    dot = np.clip((1.75-radius)*size/42 + .5, 0, 1)
    rgba = np.full((size, size, 4), 255, dtype=np.uint8)
    rgba[..., 3] = np.round(np.maximum(dot, np.maximum(ring, glow))*255).astype(np.uint8)
    return rgba


# --- Argon HUD bakes (skinless default — ppy/osu Argon* components) -----------------

def bake_pill(width: int = 256, height: int = 64) -> np.ndarray:
    """Fully-rounded white bar (radius = height/2). Generic rounded-rect
    stand-in for the Argon boxes/bars/indicators (BoxElement with
    CornerRadius 0.5, ArgonSongProgressBar's RoundedBar, the ArgonKeyCounter
    input indicator). Scaled non-uniformly at draw time — the cap
    distortion is invisible at HUD bar sizes."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    r = height / 2.0 - 1.0
    qx = np.abs(xx - (width - 1) / 2.0) - (width / 2.0 - 1.0 - r)
    qy = np.abs(yy - (height - 1) / 2.0) - (height / 2.0 - 1.0 - r)
    d = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) \
        + np.minimum(np.maximum(qx, qy), 0.0) - r
    alpha = np.clip(-d / _AA_PX, 0.0, 1.0)
    rgba = np.full((height, width, 4), 255, dtype=np.uint8)
    rgba[..., 3] = np.round(alpha * 255.0).astype(np.uint8)
    return rgba


# --- Argon counter: 7-segment display (ArgonCounterTextComponent) -------------
# Lazer draws the score/acc/combo counters with the `argon-counter` sprite font
# and a same-font all-segments "wireframes" backing at WireframeOpacity 0.25
# (ArgonAccuracyCounter/ArgonComboCounter default 0.25). We draw the digits
# procedurally as rounded 7-segment bars: the LIT glyph and the UNLIT wireframe
# come from the SAME segment geometry on the SAME cell canvas, so lit+ghost
# align BY CONSTRUCTION (no more phantom-8 misregistration between a DejaVu
# digit and a hand-drawn wireframe). Zero font-license concern.
ARGON_SEG_W = 132               # one fixed-width digit cell (aspect 0.55)
ARGON_SEG_H = 240
# standard 7-segment map: A top, B upper-right, C lower-right, D bottom,
# E lower-left, F upper-left, G middle.
_ARGON_SEG_MAP = {
    "0": "ABCDEF", "1": "BC", "2": "ABGED", "3": "ABGCD", "4": "FGBC",
    "5": "AFGCD", "6": "AFGEDC", "7": "ABC", "8": "ABCDEFG", "9": "ABCDFG",
}
ARGON_SEG_CHARS = "0123456789"


def _argon_seg_fields(width: int, height: int, seg_frac: float,
                      gap_frac: float):
    """(xx, yy, seg_alpha dict) for one segment cell — each of the seven
    segments A..G as a rounded-bar alpha field (shared by the lit glyphs
    and the all-segments wireframe backing)."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    t = seg_frac * width                   # segment thickness
    gap = gap_frac * height
    m = t * 0.75                           # cell inset
    x0, x1 = m, width - m
    y0, ym, y1 = m, height / 2.0, height - m
    half_v = (ym - y0) / 2.0

    def hseg(cy: float) -> np.ndarray:
        qx = np.abs(xx - width / 2.0) - (x1 - x0 - 2 * t - 2 * gap) / 2.0
        qy = np.abs(yy - cy) - t / 2.0
        d = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) \
            + np.minimum(np.maximum(qx, qy), 0.0) - t * 0.18
        return np.clip(-d / _AA_PX, 0.0, 1.0)

    def vseg(cx: float, cy: float) -> np.ndarray:
        qx = np.abs(xx - cx) - t / 2.0
        qy = np.abs(yy - cy) - (half_v - t / 2.0 - gap)
        d = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) \
            + np.minimum(np.maximum(qx, qy), 0.0) - t * 0.18
        return np.clip(-d / _AA_PX, 0.0, 1.0)

    seg = {
        "A": hseg(y0 + t / 2.0), "G": hseg(ym), "D": hseg(y1 - t / 2.0),
        "F": vseg(x0 + t / 2.0, (y0 + ym) / 2.0),
        "B": vseg(x1 - t / 2.0, (y0 + ym) / 2.0),
        "E": vseg(x0 + t / 2.0, (ym + y1) / 2.0),
        "C": vseg(x1 - t / 2.0, (ym + y1) / 2.0),
    }
    return xx, yy, seg


def _argon_seg_char_alpha(char: str, xx, yy, seg,
                          width: int, height: int, seg_frac: float):
    """Alpha field for one glyph — a subset of the seven segments for a
    digit, or a procedural '.'/'%'/'x' in the same segment weight."""
    t = seg_frac * width
    if char in _ARGON_SEG_MAP:
        a = np.zeros((height, width))
        for s in _ARGON_SEG_MAP[char]:
            a = np.maximum(a, seg[s])
        return a
    if char in (".", "dot"):
        r = t * 0.85
        cx, cy = width / 2.0, height - t * 0.75 - r
        d = np.hypot(xx - cx, yy - cy)
        return np.clip((r - d) / _AA_PX, 0.0, 1.0)
    if char in ("%", "percent"):
        # A clean percent: two OPEN rings (upper-left + lower-right) joined by
        # a bold diagonal slash. Tuned to read as a percent (not a crossed box
        # or a "Z") even at HUD scale — a wider ring band keeps a clear centre
        # hole so the rings don't collapse to dots when the counter is small.
        rr = height * 0.135                # ring radius
        band = t * 0.5                     # ring HALF-thickness → open centre
        cxu, cyu = width * 0.31, height * 0.205
        cxl, cyl = width * 0.69, height * 0.795
        du = np.abs(np.hypot(xx - cxu, yy - cyu) - rr)
        dl = np.abs(np.hypot(xx - cxl, yy - cyl) - rr)
        ring = np.maximum(np.clip((band - du) / _AA_PX, 0.0, 1.0),
                          np.clip((band - dl) / _AA_PX, 0.0, 1.0))
        # diagonal slash from bottom-left to top-right
        m = t * 0.95
        dirx, diry = (width - 2 * m), -(height - 2 * m)
        L = math.hypot(dirx, diry)
        nx, ny = -diry / L, dirx / L
        px, py = xx - width / 2.0, yy - height / 2.0
        dperp = np.abs(px * nx + py * ny)
        dalong = np.abs(px * (dirx / L) + py * (diry / L))
        slash = (np.clip((t * 0.42 - dperp) / _AA_PX, 0.0, 1.0)
                 * np.clip((L / 2.0 - dalong) / _AA_PX, 0.0, 1.0))
        return np.maximum(ring, slash)
    if char in ("x",):
        th = t * 0.60
        arm = height * 0.24
        px, py = xx - width / 2.0, yy - height * 0.5
        s2 = 1.0 / math.sqrt(2.0)
        d1 = np.abs((px - py) * s2)
        d2 = np.abs((px + py) * s2)
        reach = np.clip((arm - np.hypot(px, py)) / _AA_PX, 0.0, 1.0)
        return np.maximum(np.clip((th - d1) / _AA_PX, 0.0, 1.0),
                          np.clip((th - d2) / _AA_PX, 0.0, 1.0)) * reach
    return np.zeros((height, width))


def bake_argon_segment(char: str, width: int = ARGON_SEG_W,
                       height: int = ARGON_SEG_H, seg_frac: float = 0.14,
                       gap_frac: float = 0.045) -> np.ndarray:
    """A single lit Argon-counter glyph (white, tintable): digit segments
    or the '.'/'%'/'x' symbols, on the shared cell canvas."""
    xx, yy, seg = _argon_seg_fields(width, height, seg_frac, gap_frac)
    a = _argon_seg_char_alpha(char, xx, yy, seg, width, height, seg_frac)
    rgba = np.full((height, width, 4), 255, dtype=np.uint8)
    rgba[..., 3] = np.round(np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgba


def bake_argon_seg_glyphs() -> dict[str, np.ndarray]:
    """{char: rgba} for the lit Argon-counter set (digits + . % x)."""
    return {ch: bake_argon_segment(ch) for ch in "0123456789.%x"}


def bake_wireframe_cell(width: int = ARGON_SEG_W, height: int = ARGON_SEG_H,
                        seg_frac: float = 0.14,
                        gap_frac: float = 0.045) -> np.ndarray:
    """The all-segments '8' wireframe backing (WireframeOpacity 0.25). Same
    segment geometry as the lit glyphs, so a lit digit registers exactly on
    top of its own wireframe cell."""
    xx, yy, seg = _argon_seg_fields(width, height, seg_frac, gap_frac)
    a = np.zeros((height, width))
    for s in "ABCDEFG":
        a = np.maximum(a, seg[s])
    rgba = np.full((height, width, 4), 255, dtype=np.uint8)
    rgba[..., 3] = np.round(a * 255.0).astype(np.uint8)
    return rgba


def bake_wireframe_dot(width: int = ARGON_SEG_W, height: int = ARGON_SEG_H,
                       seg_frac: float = 0.14) -> np.ndarray:
    """The '.' wireframe backing (a dot) — matches the lit '.' glyph."""
    xx, yy, seg = _argon_seg_fields(width, height, seg_frac, 0.045)
    a = _argon_seg_char_alpha(".", xx, yy, seg, width, height, seg_frac)
    rgba = np.full((height, width, 4), 255, dtype=np.uint8)
    rgba[..., 3] = np.round(np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgba


# --- Argon counter: REAL `argon-counter` sprites (pixel-exact to lazer) -------
# The procedural 7-segment bake above (bake_argon_segment / bake_argon_seg_glyphs
# / bake_wireframe_*) reads a touch too bold/round vs lazer's actual counter.
# Red OK'd ripping the Argon art, so when the sprite dir is present we load
# lazer's real `argon-counter-*.png` glyphs (CC-BY-NC — Argon default skin) and
# use them for the ARGON score/accuracy/combo counters. Missing dir (stripped
# checkout) → the procedural bake is the fallback (the `or bake_*()` sites in
# TextureBank). This is the ARGON path ONLY: the legacy / custom-skin score
# font (lg_* / bake_legacy_font) is a separate texture set and is untouched.
#
# lazer's argon-counter is a FIXED-WIDTH square font: every digit ships on a
# 240x240 canvas (~31 px pad, digit centred), '.' on a narrow 52x240 canvas,
# and '%'/'x'/'wireframes' on 240x240. We normalise every glyph onto one common
# square cell so the counter run stays fixed-width — the run advance derives
# from the reference-glyph aspect (argon_seg_advance = argon_seg_aspect["8"]),
# which becomes the real square metric so digits render undistorted at
# ARGON_DIGIT_H. Anchors/heights/right-alignment in hud.py are unchanged (it
# reads argon_seg_advance dynamically); only the glyph art + true fixed-width
# spacing change.
ARGON_COUNTER_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "argon_assets"))
ARGON_COUNTER_CELL = ARGON_SEG_H          # 240 — common square cell (glyph height)

_ARGON_COUNTER_FILES = {
    **{ch: f"argon-counter-{ch}.png" for ch in "0123456789"},
    ".": "argon-counter-dot.png",
    "%": "argon-counter-percentage.png",
    "x": "argon-counter-x.png",
}
_ARGON_WIREFRAME_FILE = "argon-counter-wireframes.png"
_ARGON_WIREFRAME_DOT_FILE = "argon-counter-dot.png"


def _load_argon_counter_png(name: str) -> np.ndarray | None:
    """Load one ripped argon-counter sprite as RGBA uint8, or None if the
    file is absent / unreadable (→ procedural fallback)."""
    path = os.path.join(ARGON_COUNTER_DIR, name)
    if not os.path.isfile(path):
        return None
    try:
        im = Image.open(path).convert("RGBA")
    except Exception:
        return None
    return np.asarray(im, dtype=np.uint8).copy()


def _argon_counter_cell(im: np.ndarray,
                        cell: int = ARGON_COUNTER_CELL) -> np.ndarray:
    """Normalise a real argon-counter sprite onto a `cell`x`cell` square:
    scaled by height, horizontally centred, RGB forced WHITE (tintable —
    the counters are drawn with a combo/flash colour), alpha preserved. Keeps
    every glyph the SAME cell aspect so the run stays fixed-width."""
    h, w = im.shape[:2]
    if h != cell:
        new_w = max(1, int(round(w * cell / h)))
        im = np.asarray(Image.fromarray(im).resize((new_w, cell),
                                                    Image.LANCZOS),
                        dtype=np.uint8)
        h, w = im.shape[:2]
    if w > cell:                                # defensive (real art is <= cell)
        new_h = max(1, int(round(h * cell / w)))
        im = np.asarray(Image.fromarray(im).resize((cell, new_h),
                                                    Image.LANCZOS),
                        dtype=np.uint8)
        h, w = im.shape[:2]
    canvas = np.zeros((cell, cell, 4), dtype=np.uint8)
    y0, x0 = (cell - h) // 2, (cell - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = im
    canvas[..., 0:3] = 255                      # white → tint does the colour
    return canvas


def load_argon_seg_glyphs() -> dict[str, np.ndarray] | None:
    """Real lit argon-counter glyph set {char: rgba} (digits + . % x) from
    the ripped sprite dir, each normalised onto the common square cell.
    Returns None if ANY sprite is missing → caller falls back to the
    procedural bake_argon_seg_glyphs()."""
    out: dict[str, np.ndarray] = {}
    for ch, name in _ARGON_COUNTER_FILES.items():
        im = _load_argon_counter_png(name)
        if im is None:
            return None
        out[ch] = _argon_counter_cell(im)
    return out


def load_argon_wireframe_cell() -> np.ndarray | None:
    """Real all-glyph 'wireframes' ghost backing (WireframeOpacity 0.25),
    normalised onto the common cell. None → fallback bake_wireframe_cell()."""
    im = _load_argon_counter_png(_ARGON_WIREFRAME_FILE)
    return None if im is None else _argon_counter_cell(im)


def load_argon_wireframe_dot() -> np.ndarray | None:
    """Real '.' ghost backing (the dot cell). None → bake_wireframe_dot()."""
    im = _load_argon_counter_png(_ARGON_WIREFRAME_DOT_FILE)
    return None if im is None else _argon_counter_cell(im)


WEDGE_W = 380.0                 # ArgonWedgePiece size in the default layout
WEDGE_H = 72.0
WEDGE_SHEAR = 0.8               # osu!framework Shear = (0.8, 0): x' = x - 0.8y
WEDGE_R = 10.0                  # CornerRadius (pre-shear space)
WEDGE_COLOR = (0x66, 0xCC, 0xFF)   # AccentColour #66CCFF


def bake_wedge(scale: float = 1.0) -> np.ndarray:
    """ArgonWedgePiece: a rounded rect sheared by (0.8, 0) with a vertical
    gradient of #66CCFF from alpha 0 (top) to 0.25 (bottom). Baked in the
    SHEARED frame (canvas width = W + 0.8·H) so the sprite draws axis-
    aligned; masking (corner radius) applies pre-shear, as upstream."""
    w = int(round((WEDGE_W + WEDGE_SHEAR * WEDGE_H) * scale))
    h = int(round(WEDGE_H * scale))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64) / scale
    # unshear: canvas x = u - 0.8·y + 0.8·H  (u = pre-shear x)
    u = xx + WEDGE_SHEAR * yy - WEDGE_SHEAR * WEDGE_H
    r = WEDGE_R
    qx = np.abs(u - WEDGE_W / 2.0) - (WEDGE_W / 2.0 - r)
    qy = np.abs(yy - WEDGE_H / 2.0) - (WEDGE_H / 2.0 - r)
    d = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) \
        + np.minimum(np.maximum(qx, qy), 0.0) - r
    shape = np.clip(-d / (_AA_PX / scale), 0.0, 1.0)
    grad = 0.25 * np.clip(yy / WEDGE_H, 0.0, 1.0)
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = WEDGE_COLOR[0]
    rgba[..., 1] = WEDGE_COLOR[1]
    rgba[..., 2] = WEDGE_COLOR[2]
    rgba[..., 3] = np.round(shape * grad * 255.0).astype(np.uint8)
    return rgba


# --- legacy-default HUD bakes (custom-skin fallback — classic osu! look) ------------
#
# Under a CUSTOM skin, missing elements fall back to the CLASSIC default
# skin's look (lazer's legacy-skin fallback chain), NEVER to Argon. These
# are procedural approximations of the classic assets (house pattern, no
# ppy asset files): the classic score/combo font (white digits with a dark
# outline), the classic scorebar (dark frame + warm green fill + round ki
# marker — LegacyHealthDisplay's old-style pieces), and the classic input
# overlay (translucent strip + light rounded key — LegacyKeyCounter[Display]).

LEGACY_FONT_HEIGHT = 45.0       # classic score-*.png logical height (768-space)
LEGACY_BAR_BG_SIZE = (695.0, 44.0)   # classic scorebar-bg logical size
LEGACY_BAR_FILL_SIZE = (672.0, 16.0)
LEGACY_KI_SIZE = 24.0
LEGACY_IO_KEY_SIZE = 43.0       # classic inputoverlay-key logical size
LEGACY_IO_BG_SIZE = (200.0, 50.0)


def bake_legacy_font(chars: str = "0123456789.,%x",
                     height: int = 128) -> dict[str, np.ndarray]:
    """Classic score-font stand-in: bold white glyphs with a dark outline
    (the default skin score-0..9/dot/comma/percent/x look). Used whenever
    a custom skin lacks its ScorePrefix/ComboPrefix set (or single chars
    of it — stable falls back per-file to the default skin)."""
    font = _load_font(int(height * 0.9))
    boxes = {}
    for ch in chars:
        try:
            boxes[ch] = font.getbbox(ch)
        except AttributeError:
            w, h = font.getsize(ch)  # type: ignore[attr-defined]
            boxes[ch] = (0, 0, w, h)
    top = min(b[1] for b in boxes.values())
    bottom = max(b[3] for b in boxes.values())
    pad = 8
    outline = max(2, height // 24)
    out: dict[str, np.ndarray] = {}
    for ch in chars:
        x0, _, x1, _ = boxes[ch]
        w = (x1 - x0) + 2 * pad
        h = (bottom - top) + 2 * pad
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        drw = ImageDraw.Draw(img)
        try:
            drw.text((pad - x0, pad - top), ch, font=font,
                     fill=(255, 255, 255, 255),
                     stroke_width=outline, stroke_fill=(40, 40, 48, 255))
        except TypeError:      # ancient PIL without stroke support
            drw.text((pad - x0, pad - top), ch, font=font,
                     fill=(255, 255, 255, 255))
        out[ch] = np.asarray(img, dtype=np.uint8).copy()
    return out


def _rounded_rect_alpha(w: int, h: int, radius: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    qx = np.abs(xx - (w - 1) / 2.0) - (w / 2.0 - 1.0 - radius)
    qy = np.abs(yy - (h - 1) / 2.0) - (h / 2.0 - 1.0 - radius)
    d = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) \
        + np.minimum(np.maximum(qx, qy), 0.0) - radius
    return d


def bake_legacy_scorebar_bg() -> np.ndarray:
    """Classic scorebar-bg stand-in: a dark translucent panel with a light
    rim around the fill groove (groove at the LegacyOldStyleFill offset
    (3,10)·1.6 so the classic fill lines up)."""
    w, h = int(LEGACY_BAR_BG_SIZE[0]), int(LEGACY_BAR_BG_SIZE[1])
    d = _rounded_rect_alpha(w, h, 10.0)
    panel = np.clip(-d / _AA_PX, 0.0, 1.0)
    rim = panel * np.clip((d + 2.5) / _AA_PX, 0.0, 1.0)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    base = panel * 0.55
    rgba[..., 0] = np.round(24 * panel + 200 * rim).astype(np.uint8)
    rgba[..., 1] = np.round(26 * panel + 205 * rim).astype(np.uint8)
    rgba[..., 2] = np.round(34 * panel + 215 * rim).astype(np.uint8)
    rgba[..., 3] = np.round(np.maximum(base, rim * 0.9) * 255).astype(np.uint8)
    return rgba


def bake_legacy_scorebar_colour() -> np.ndarray:
    """Classic scorebar-colour stand-in: the warm green-yellow fill bar."""
    w, h = int(LEGACY_BAR_FILL_SIZE[0]), int(LEGACY_BAR_FILL_SIZE[1])
    yy = np.mgrid[0:h, 0:w][0].astype(np.float64) / max(h - 1, 1)
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    top = np.array([214, 242, 120], dtype=np.float64)
    bot = np.array([158, 208, 60], dtype=np.float64)
    for c in range(3):
        rgba[..., c] = np.round(top[c] + (bot[c] - top[c]) * yy).astype(np.uint8)
    rgba[..., 3] = 255
    return rgba


def bake_legacy_ki(tint: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Classic ki marker stand-in: filled disc + ring, optionally tinted
    (kidanger → amber, kidanger2 → red)."""
    size = 96
    d = _dist_grid(size)
    radius = size / 2.0 - 2.0
    disc = np.clip((radius - d) / _AA_PX, 0.0, 1.0)
    ring = disc * np.clip((d - radius * 0.72) / _AA_PX, 0.0, 1.0)
    rgba = np.empty((size, size, 4), dtype=np.uint8)
    for c in range(3):
        inner = tint[c] * 0.92
        rgba[..., c] = np.round(inner + (255 - inner) * ring).astype(np.uint8)
    rgba[..., 3] = np.round(disc * 255.0).astype(np.uint8)
    return rgba


def bake_legacy_io_key() -> np.ndarray:
    """Classic inputoverlay-key stand-in: light rounded square with a soft
    border (tinted yellow/pink by the display while pressed)."""
    size = 128
    d = _rounded_rect_alpha(size, size, size * 0.22)
    inside = np.clip(-d / _AA_PX, 0.0, 1.0)
    border = inside * np.clip((d + size * 0.055) / _AA_PX, 0.0, 1.0)
    rgba = np.empty((size, size, 4), dtype=np.uint8)
    grey = 226 - 46 * border
    for c in range(3):
        rgba[..., c] = np.round(grey).astype(np.uint8)
    rgba[..., 3] = np.round(inside * 255.0).astype(np.uint8)
    return rgba


def bake_legacy_io_bg() -> np.ndarray:
    """Classic inputoverlay-background stand-in: translucent dark strip
    (drawn rotated 90° at the right screen edge, as stable does)."""
    w, h = int(LEGACY_IO_BG_SIZE[0]), int(LEGACY_IO_BG_SIZE[1])
    d = _rounded_rect_alpha(w, h, 12.0)
    inside = np.clip(-d / _AA_PX, 0.0, 1.0)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 18
    rgba[..., 1] = 18
    rgba[..., 2] = 24
    rgba[..., 3] = np.round(inside * 0.55 * 255.0).astype(np.uint8)
    return rgba


def bake_tri_down(size: int = 64) -> np.ndarray:
    """Downward-pointing AA triangle (hit-error moving-average arrow)."""
    img = Image.new("RGBA", (size * 4, size * 4), (0, 0, 0, 0))
    ImageDraw.Draw(img).polygon(
        [(4, 4), (size * 4 - 4, 4), (size * 2, size * 4 - 4)],
        fill=(255, 255, 255, 255))
    img = img.resize((size, size), Image.LANCZOS)   # cheap supersampled AA
    return np.asarray(img, dtype=np.uint8).copy()


def bake_triangle_up(size: int = 128) -> np.ndarray:
    """Upward-pointing equilateral AA triangle, white — the bg_triangles
    deco sprite (tinted grey per triangle, alpha'd subtle at draw)."""
    s4 = size * 4
    h = math.sqrt(3.0) / 2.0        # equilateral height/side
    pad = 6
    top = (s4 / 2.0, pad + s4 * (1 - h) / 2.0)
    by = pad + s4 * (1 + h) / 2.0 - 2 * pad
    img = Image.new("RGBA", (s4, s4), (0, 0, 0, 0))
    ImageDraw.Draw(img).polygon(
        [top, (pad, by), (s4 - pad, by)], fill=(255, 255, 255, 255))
    img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8).copy()


# the R3D logo tile — versus_splash.py's look (red rounded tile, heavy
# white R), baked procedurally so the engine stays asset-free
LOGO_TILE_RED = (216, 44, 54)
LOGO_TILE_RADIUS_FRAC = 0.18


def bake_logo_tile(size: int = 256) -> np.ndarray:
    """The R3D 'R' tile (show_logo intro splash): rounded red square with
    a heavy white R centred like the site logo (versus_splash.py measures
    the inner R at ~0.20..0.79 of the tile — matched here)."""
    # Prefer R3D's real logo asset (own IP, license-clean) over the procedural
    # draw so the splash matches the site icon; fall back to the bake if missing.
    try:
        _lp = os.path.join(os.path.dirname(__file__), "..", "assets", "logo.png")
        _im = Image.open(_lp).convert("RGBA").resize((size, size), Image.LANCZOS)
        return np.asarray(_im, dtype=np.uint8).copy()
    except Exception:
        pass
    d = _rounded_rect_alpha(size, size, size * LOGO_TILE_RADIUS_FRAC)
    tile = np.clip(-d / _AA_PX, 0.0, 1.0)
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., 0] = LOGO_TILE_RED[0]
    rgba[..., 1] = LOGO_TILE_RED[1]
    rgba[..., 2] = LOGO_TILE_RED[2]
    rgba[..., 3] = np.round(tile * 255.0).astype(np.uint8)
    img = Image.fromarray(rgba, "RGBA")
    drw = ImageDraw.Draw(img)
    # inner R spans ~0.59 of the tile height, centred (the splash's
    # measured 0.203..0.793 band)
    font = _load_font(int(size * 0.66))
    try:
        box = font.getbbox("R")
    except AttributeError:
        w, h = font.getsize("R")  # type: ignore[attr-defined]
        box = (0, 0, w, h)
    rw, rh = box[2] - box[0], box[3] - box[1]
    drw.text(((size - rw) / 2.0 - box[0], (size - rh) / 2.0 - box[1]),
             "R", font=font, fill=(255, 255, 255, 255))
    return np.asarray(img, dtype=np.uint8).copy()


# ============================================================================
# Argon GAMEPLAY bakes — the skinless "Argon league" pieces.
#
# Procedural port of ppy/osu master (MIT) osu.Game.Rulesets.Osu/Skinning/Argon/
# (ZERO ppy asset files; fonts stay DejaVu — Torus is non-commercial). These
# textures are drawn ONLY when NO custom skin is attached
# (scene.StdScene.skin is None → the "Argon league"). The classic procedural
# set above is left byte-for-byte for the custom-skin per-element fallback, so
# custom-skin renders are pixel-identical (league integrity).
#
# Geometry from ArgonMainCirclePiece.cs, as fractions of the object RADIUS R
# (lazer OBJECT_RADIUS=59, OBJECT_DIMENSIONS=118, BORDER_THICKNESS=D*2/58):
#   BORDER_THICKNESS   = 4/58 R  = 0.068966 R   (white RingPiece, at the rim)
#   GRADIENT_THICKNESS = 10/58 R = 0.172414 R   (slider/ball band thickness)
#   OUTER_GRADIENT r   = 50/58 R = 0.862069 R   (bright saturated band edge)
#   INNER_GRADIENT r   = 40/58 R = 0.689655 R
#   INNER_FILL r       = 30/58 R = 0.517241 R   (dark centre)
# osu!framework Color4.Darken(a) = channel/(1+a):
#   Darken(4)=0.20 (fills), Darken(0.5)=0.6667 / Darken(0.6)=0.625 (inner
#   gradient), Darken(0.1)=0.9091 (outer gradient bottom).  All Argon bakes
#   are grayscale so the combo-accent multiply (gl.py u_color) colours them;
#   the WHITE pieces (borders) are separate sprites so they stay white.
ARGON_BORDER_FRAC = 4.0 / 58.0     # 0.068966 R — the white circle border
ARGON_OUTER_GRAD_R = 50.0 / 58.0   # 0.862069 R
ARGON_INNER_GRAD_R = 40.0 / 58.0   # 0.689655 R
ARGON_INNER_FILL_R = 30.0 / 58.0   # 0.517241 R
ARGON_GRAD_FRAC = 10.0 / 58.0      # 0.172414 R — GRADIENT_THICKNESS
ARGON_BALL_BORDER_FRAC = 0.20      # GRADIENT_THICKNESS / (OUTER_GRADIENT/2)
ARGON_CIRCLE_SIZE = 320
ARGON_BALL_SIZE = 256

# ArgonCursor.cs palette (fixed — not combo-tinted):
ARGON_CURSOR_PINK = (0xFC, 0x61, 0x8F)
ARGON_CURSOR_DARKRED = (0xBB, 0x1A, 0x41)
ARGON_CURSOR_CGLOW = (171, 255, 255)   # centre EdgeEffect glow (cyan)


def _ss_inside(d: np.ndarray, edge_r: float, R: float) -> np.ndarray:
    """1 inside radius edge_r*R, smooth 1→0 across the AA band at the edge."""
    return np.clip((edge_r * R - d) / _AA_PX, 0.0, 1.0)


def _grey_rgba(val: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    grey = np.round(np.clip(val, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba = np.empty(val.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = grey
    rgba[..., 1] = grey
    rgba[..., 2] = grey
    rgba[..., 3] = np.round(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgba


def bake_argon_circle(size: int = ARGON_CIRCLE_SIZE) -> np.ndarray:
    """ArgonMainCirclePiece's fill stack (outerFill/outerGradient/
    innerGradient/innerFill) as ONE grayscale disc — the radial bands with
    lazer's vertical gradient baked in, tinted by the combo accent at draw.
    The white RingPiece border is bake_argon_border (drawn on top)."""
    d = _dist_grid(size)
    R = size / 2.0 - 2.0
    yn = np.mgrid[0:size, 0:size][0].astype(np.float64) / (size - 1)
    inner_fill = 0.20                                   # Darken(4)
    # fidelity pass 3 (owner: EXACT lazer — REVERT the pass-2 "make it pop"
    # push). ArgonMainCirclePiece.innerGradient =
    # ColourInfo.GradientVertical(AccentColour.Darken(0.5), Darken(0.6))
    # = 0.6667 (top) → 0.625 (bottom). Color4.Darken(a)=channel/(1+a).
    # Ground truth frame_26 (real lazer) reads a smooth dark-amber mid band,
    # NOT the exaggerated 0.50→0.46 step pass 2 used.
    inner_grad = 0.6667 + (0.625 - 0.6667) * yn         # Darken(0.5)→Darken(0.6)
    outer_grad = 1.0 + (0.90909 - 1.0) * yn             # 1.0→Darken(0.1)
    outer_fill = 0.20                                   # Darken(4)
    w_out = _ss_inside(d, ARGON_OUTER_GRAD_R, R)
    w_in = _ss_inside(d, ARGON_INNER_GRAD_R, R)
    w_fl = _ss_inside(d, ARGON_INNER_FILL_R, R)
    val = np.full_like(d, outer_fill)
    val = val * (1.0 - w_out) + outer_grad * w_out
    val = val * (1.0 - w_in) + inner_grad * w_in
    val = val * (1.0 - w_fl) + inner_fill * w_fl
    alpha = np.clip((R - d) / _AA_PX, 0.0, 1.0)
    return _grey_rgba(val, alpha)


def bake_argon_border(size: int = ARGON_CIRCLE_SIZE,
                      thickness_frac: float = ARGON_BORDER_FRAC) -> np.ndarray:
    """The white RingPiece(BORDER_THICKNESS): a bright thin ring at the rim
    (drawn untinted over the accent disc)."""
    return bake_ring(size, thickness_frac)


ARGON_APPROACH_THICKNESS = 0.066   # bright CORE thickness (fraction of the ring
                                   # radius). Pass-4 re-measure of frame_26 (real
                                   # lazer): the approach ring around the combo-1
                                   # circle has a SHARP bright core ~11 px on its
                                   # 166 px radius (11/166 = 0.066 — matching the
                                   # bare DefaultApproachCircle texture proportion
                                   # 0.064) PLUS a SOFT glow shoulder trailing to
                                   # ~19 px total FWHM. Pass-3 baked a FLAT solid
                                   # 0.11 band (the full FWHM at uniform bright),
                                   # so it read too THICK vs the game's crisp ring
                                   # (owner-flagged pass 4). We now bake the bright
                                   # thin core at 0.066 + a faint additive-style
                                   # glow shoulder, reproducing the measured
                                   # profile: crisp like the game, not a slab.
ARGON_APPROACH_GLOW_FRAC = 0.065   # glow shoulder sigma (fraction of radius) —
                                   # frame_26's shoulder sits ~0.6 of peak out
                                   # to ~19 px, so give the halo real weight
ARGON_APPROACH_GLOW_ALPHA = 0.55   # shoulder peak alpha vs the bright core


def bake_argon_approach(size: int = APPROACH_SIZE,
                        thickness_frac: float = ARGON_APPROACH_THICKNESS
                        ) -> np.ndarray:
    """Argon approach circle — a bright THIN core ring + a soft glow shoulder
    (base DrawableHitCircle ApproachCircle; Argon ships no bespoke texture).
    The core reproduces frame_26's crisp ~0.066·R bright band; the Gaussian
    shoulder gives the game's soft falloff so the ring reads crisp, not as the
    uniform slab pass-3's flat bake produced."""
    d = _dist_grid(size)
    R = size / 2.0 - 2.0
    core_half = max(0.5 * thickness_frac * R, 1.0)
    r0 = R - core_half - 1.0                       # ring centerline (near edge)
    sd = np.abs(d - r0)                            # radial dist from centerline
    core = np.clip((core_half - sd) / _AA_PX, 0.0, 1.0)
    sigma = max(ARGON_APPROACH_GLOW_FRAC * R, 1.0)
    glow = ARGON_APPROACH_GLOW_ALPHA * np.exp(-(sd / sigma) ** 2)
    alpha = np.maximum(core, glow)
    alpha = alpha * np.clip((R + 2.0 - d) / _AA_PX, 0.0, 1.0)   # clip at edge
    return _grey_rgba(np.ones_like(d), alpha)


# --- ArgonFollowPoint (chevrons) + RingExplosion (kiai bubbles) ----------------
ARGON_FP_PINK = (0xFC, 0x61, 0x8F)      # ArgonFollowPoint gradient top
ARGON_FP_DARKRED = (0xBB, 0x1A, 0x41)   # ArgonFollowPoint gradient bottom
ARGON_FP_GRAY = (51, 51, 51)            # OsuColour.Gray(0.2) → LEADING chevron


def bake_argon_followpoint(size: int = 96) -> np.ndarray:
    """ArgonFollowPoint (lazer): TWO ChevronRight (each Size 8), the second
    offset X=4 so they half-overlap, additive. The LEADING (left) chevron is
    OsuColour.Gray(0.2); the trailing (right) one carries the vertical pink→
    dark-red gradient (FromHex FC618F→BB1A41) — a subtle two-tone '>>' rather
    than pass-3's uniform pink pair. Fixed palette (not combo-tinted); the
    scene rotates it along the connection direction and draws it additive."""
    off = int(round(0.075 * size))                  # X=4 of Size-8 ≈ half a chevron
    base = _chevron_mask(size, hw=0.15, hh=0.26, thick=0.105, n=1)
    left = np.roll(base, -off, axis=1)              # leading chevron (gray)
    right = np.roll(base, off, axis=1)              # trailing chevron (pink)
    yn = np.mgrid[0:size, 0:size][0].astype(np.float64) / (size - 1)
    pink = np.array(ARGON_FP_PINK, dtype=np.float64) / 255.0
    dark = np.array(ARGON_FP_DARKRED, dtype=np.float64) / 255.0
    grad = (pink[None, None, :] * (1.0 - yn[..., None])
            + dark[None, None, :] * yn[..., None])
    gray = np.broadcast_to(np.array(ARGON_FP_GRAY, np.float64) / 255.0,
                           (size, size, 3))
    rgb, a = _over(gray, left, grad, right)         # pink chevron over gray one
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba[..., 3] = np.round(np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgba


def bake_argon_bubble(size: int = 64,
                      thickness_frac: float = 0.40) -> np.ndarray:
    """A RingExplosion ring "bubble" (RingPiece, thickness 4). White,
    additive, tinted by the judgement colour at draw."""
    return bake_ring(size, thickness_frac)


def _chevron_mask(size: int, hw: float, hh: float, thick: float,
                  n: int = 1, gap: float = 0.0) -> np.ndarray:
    """Right-pointing '>' (n=1) or '>>' (n=2) chevron mask in [0,1]."""
    s4 = 4
    img = Image.new("L", (size * s4, size * s4), 0)
    drw = ImageDraw.Draw(img)
    cx, cy = size * s4 / 2.0, size * s4 / 2.0
    for i in range(n):
        ox = (i - (n - 1) / 2.0) * gap * size * s4
        drw.line([(cx + ox - hw * size * s4, cy - hh * size * s4),
                  (cx + ox + hw * size * s4, cy),
                  (cx + ox - hw * size * s4, cy + hh * size * s4)],
                 fill=255, width=int(thick * size * s4), joint="curve")
    img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


def bake_argon_ball(size: int = ARGON_BALL_SIZE) -> np.ndarray:
    """ArgonSliderBall fill: a vertical accent gradient (accent→Darken(0.5))
    with the dark AngleRight '>' icon (accent.Darken(4)) baked in. Grayscale
    → tinted by the combo accent. The white ball border is bake_argon_ball_ring."""
    d = _dist_grid(size)
    R = size / 2.0 - 2.0
    yn = np.mgrid[0:size, 0:size][0].astype(np.float64) / (size - 1)
    val = 1.0 + (0.6667 - 1.0) * yn                     # accent→Darken(0.5)
    chev = _chevron_mask(size, hw=0.14, hh=0.20, thick=0.055)
    val = val * (1.0 - chev) + 0.20 * chev              # dark '>' (Darken 4)
    alpha = np.clip((R - d) / _AA_PX, 0.0, 1.0)
    return _grey_rgba(val, alpha)


def bake_argon_ball_ring(size: int = ARGON_BALL_SIZE) -> np.ndarray:
    """The ArgonSliderBall white border (BorderThickness=GRADIENT_THICKNESS
    → 0.2 of the ball radius)."""
    return bake_ring(size, ARGON_BALL_BORDER_FRAC)


def bake_argon_follow(size: int = 256, thickness_frac: float = 4 / 64,
                      fill: float = 0.30, gradient: bool = True) -> np.ndarray:
    """ArgonFollowCircle: additive accent ring (border) + faint accent fill
    (Alpha 0.3). BorderThickness 4 in the 128px follow container. Accent
    gradients vertically to accent.Darken(0.5), independent of ball sizing.
    Default uses the same bake with border 5, fill .2 and no gradient."""
    d = _dist_grid(size)
    R = size / 2.0 - 2.0
    outer = np.clip((R - d) / _AA_PX, 0.0, 1.0)
    inner_edge = R * (1.0 - thickness_frac)
    ring = outer * np.clip((d - inner_edge) / _AA_PX, 0.0, 1.0)
    val = np.maximum(ring, fill * outer)
    if gradient:
        yn = np.arange(size, dtype=float)[:, None] / max(1, size - 1)
        val *= 1 - yn / 3
    return _grey_rgba(val, outer)


def bake_argon_tick(size: int = 96,
                    thickness_frac: float = 0.42) -> np.ndarray:
    """ArgonSliderScorePoint: a hollow ring (BorderThickness 3 in Size 12 →
    ~0.42 of the radius), border = accent. Grayscale ring, tinted accent."""
    return bake_ring(size, thickness_frac)


def bake_argon_reverse(size: int = 256) -> np.ndarray:
    """ArgonReverseArrow: a white rounded pill (40×20 in the 118 object box)
    with the dark AngleDoubleRight '>>' icon. Drawn WHITE (untinted), rotated
    along the path tangent by the scene."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    drw = ImageDraw.Draw(img)
    pw = 40.0 / 118.0 * size
    ph = 20.0 / 118.0 * size
    cx = cy = size / 2.0
    r = ph / 2.0
    drw.rounded_rectangle([cx - pw / 2.0, cy - ph / 2.0,
                           cx + pw / 2.0, cy + ph / 2.0],
                          radius=r, fill=(255, 255, 255, 255))
    pill = np.asarray(img, dtype=np.uint8).copy()
    # dark AngleDoubleRight '>>' chevron (accent.Darken(4) ≈ near-black) baked
    # over the pill. lazer ArgonReverseArrow: SpriteIcon Size(16) inside the
    # Circle(40,20) pill → the glyph fills ~0.8 of the 20-tall pill and ~0.5
    # of its 40 width. Pass-3's chevron (hh 0.055 ≈ 0.65 pill-height, narrow)
    # read too small/bent against the game's bold '>>'; scale it up to match.
    chev = _chevron_mask(size, hw=0.052, hh=0.068, thick=0.028, n=2, gap=0.095)
    dark = np.array([36, 36, 44], dtype=np.float64)
    out = pill.astype(np.float64)
    for c in range(3):
        out[..., c] = out[..., c] * (1.0 - chev) + dark[c] * chev
    return np.clip(out, 0, 255).astype(np.uint8)


def _over(dst_rgb, dst_a, src_rgb, src_a):
    """Straight-alpha 'over' composite (float arrays), returns (rgb, a)."""
    out_a = src_a + dst_a * (1.0 - src_a)
    safe = np.where(out_a > 1e-6, out_a, 1.0)
    out_rgb = (src_rgb * src_a[..., None]
               + dst_rgb * dst_a[..., None] * (1.0 - src_a[..., None])) / safe[..., None]
    return out_rgb, out_a


def bake_argon_cursor(size: int = 192) -> np.ndarray:
    """ArgonCursor: the pink→dark-red ring (GradientVertical FC618F→BB1A41,
    BorderThickness 6) over a faint pink fill (FC618F.Darken(0.6), Alpha 0.4)
    with an inner white ring (Alpha 0.8). Fixed palette (colored RGBA). The
    white centre dot + cyan glow are drawn by the scene."""
    d = _dist_grid(size)
    R = size / 2.0 - 2.0
    yn = np.mgrid[0:size, 0:size][0].astype(np.float64) / (size - 1)
    pink = np.array(ARGON_CURSOR_PINK, dtype=np.float64) / 255.0
    dark = np.array(ARGON_CURSOR_DARKRED, dtype=np.float64) / 255.0
    fill_col = pink / 1.6                                # Darken(0.6)
    rgb = np.zeros((size, size, 3), dtype=np.float64)
    a = np.zeros((size, size), dtype=np.float64)
    disc = np.clip((R - d) / _AA_PX, 0.0, 1.0)
    # faint pink interior fill (bottom layer)
    rgb, a = _over(rgb, a, np.broadcast_to(fill_col, (size, size, 3)),
                   0.4 * disc)
    # outer pink→dark-red ring, thickness ~0.16 R
    ring = disc * np.clip((d - R * (1.0 - 0.16)) / _AA_PX, 0.0, 1.0)
    grad = pink[None, None, :] * (1.0 - yn[..., None]) + dark[None, None, :] * yn[..., None]
    rgb, a = _over(rgb, a, grad, ring)
    # inner white ring (Alpha 0.8) around r≈0.62
    winner = (np.clip((R * 0.66 - d) / _AA_PX, 0.0, 1.0)
              * np.clip((d - R * 0.58) / _AA_PX, 0.0, 1.0))
    rgb, a = _over(rgb, a, np.ones((size, size, 3)), 0.8 * winner)
    out = np.empty((size, size, 4), dtype=np.uint8)
    out[..., :3] = np.round(np.clip(rgb, 0, 1) * 255.0).astype(np.uint8)
    out[..., 3] = np.round(np.clip(a, 0, 1) * 255.0).astype(np.uint8)
    return out


def bake_argon_spin_ring(size: int = 512,
                         thickness_frac: float = 0.02) -> np.ndarray:
    """ArgonSpinnerRingArc pair (top+bottom arcs) → a thin white outer ring."""
    return bake_ring(size, thickness_frac)


def bake_argon_spin_center(size: int = 160) -> np.ndarray:
    """ArgonSpinnerDisc centre: RingPiece(10)@0.8 + RingPiece(3)@1.0 — a
    thick inner ring plus a thin outer ring, white."""
    d = _dist_grid(size)
    R = size / 2.0 - 2.0
    thin = (np.clip((R - d) / _AA_PX, 0.0, 1.0)
            * np.clip((d - R * 0.94) / _AA_PX, 0.0, 1.0))
    thick = (np.clip((R * 0.80 - d) / _AA_PX, 0.0, 1.0)
             * np.clip((d - R * 0.60) / _AA_PX, 0.0, 1.0))
    a = np.maximum(thin, thick)
    return _grey_rgba(np.ones_like(d), a)


class TextureBank:
    """Bakes the procedural set and uploads it into a SpriteRenderer.

    Texture keys: disc, ring, approach, glow, digit_0..digit_9, plus the
    HUD set (glyph_<ch>, key_square, vignette, pie_00.., pie_ring,
    tri_down). digit_aspect / glyph_aspect map char → width/height for
    layout math; glyph_mono_advance is the uniform digit advance (aspect
    units) the score/UR readouts use so rolling numbers don't jitter.
    """

    def __init__(self, renderer) -> None:
        from .mod_icons import LazerModIcons
        self.mod_icons = LazerModIcons(renderer)
        renderer.upload_texture("disc", bake_disc())
        renderer.upload_texture("ring", bake_ring())
        renderer.upload_texture("approach",
                                bake_ring(APPROACH_SIZE, APPROACH_THICKNESS))
        renderer.upload_texture("glow", bake_glow())
        # replay Smoke fallback dab (no skin cursor-smoke / Argon league)
        renderer.upload_texture("smoke", bake_smoke_puff())
        # OsuModFlashlight overlay: clamp-to-edge so uv beyond the cutout
        # samples solid black (see bake_flashlight / scene flashlight quad)
        renderer.upload_texture("flashlight", bake_flashlight(), clamp=True)
        renderer.upload_texture("miss_x", bake_miss_x())
        renderer.upload_texture("dot", bake_dot())
        renderer.upload_texture("arrow", bake_arrow())
        self.digit_aspect: dict[str, float] = {}
        for ch, rgba in bake_digits().items():
            renderer.upload_texture(f"digit_{ch}", rgba)
            self.digit_aspect[ch] = rgba.shape[1] / rgba.shape[0]

        # --- HUD set --------------------------------------------------------
        self.glyph_aspect: dict[str, float] = {}
        for ch, rgba in bake_glyphs(HUD_CHARSET).items():
            renderer.upload_texture(f"glyph_{ch}", rgba)
            self.glyph_aspect[ch] = rgba.shape[1] / rgba.shape[0]
        self.glyph_mono_advance = max(
            self.glyph_aspect[ch] for ch in "0123456789")

        # --- Argon-league text set (bundled OFL font; drawn only when the
        # scene is skinless — judgment text, HUD labels, hit-circle numbers).
        # Separate keys (aglyph_/adigit_) so the DejaVu glyph_/digit_ set the
        # legacy/custom-skin path draws stays byte-identical.
        self.argon_digit_aspect: dict[str, float] = {}
        for ch, rgba in bake_argon_digits().items():
            renderer.upload_texture(f"adigit_{ch}", rgba)
            self.argon_digit_aspect[ch] = rgba.shape[1] / rgba.shape[0]
        self.argon_glyph_aspect: dict[str, float] = {}
        for ch, rgba in bake_argon_glyphs(HUD_CHARSET).items():
            renderer.upload_texture(f"aglyph_{ch}", rgba)
            self.argon_glyph_aspect[ch] = rgba.shape[1] / rgba.shape[0]
        self.argon_glyph_mono_advance = max(
            self.argon_glyph_aspect[ch] for ch in "0123456789")
        renderer.upload_texture("key_square", bake_key_square())
        renderer.upload_texture("vignette", bake_vignette())
        renderer.upload_texture("pie_ring", bake_ring(PIE_SIZE, 0.10))
        renderer.upload_texture("tri_down", bake_tri_down())
        # settings-surface additions: bg triangles deco + the R3D logo tile
        renderer.upload_texture("triangle_up", bake_triangle_up())
        renderer.upload_texture("logo_tile", bake_logo_tile())
        for i in range(PIE_STEPS):
            renderer.upload_texture(f"pie_{i:02d}", bake_pie(i / PIE_STEPS))

        # --- Argon HUD set (skinless default) --------------------------------
        renderer.upload_texture("pill", bake_pill())
        # ARGON score/acc/combo counter font: prefer the REAL ripped
        # `argon-counter` sprites (pixel-exact to lazer); fall back to the
        # procedural 7-segment bake when argon_assets/ is absent (stripped
        # checkout). ARGON path ONLY — the legacy / custom-skin score font
        # (lg_*) below is a separate set and stays byte-identical.
        seg_glyphs = load_argon_seg_glyphs() or bake_argon_seg_glyphs()
        wf = load_argon_wireframe_cell()
        wf = bake_wireframe_cell() if wf is None else wf
        wf_dot = load_argon_wireframe_dot()
        wf_dot = bake_wireframe_dot() if wf_dot is None else wf_dot
        renderer.upload_texture("argon_wireframe", wf)
        renderer.upload_texture("argon_wireframe_dot", wf_dot)
        renderer.upload_texture("argon_wedge", bake_wedge())
        self.wireframe_aspect = wf.shape[1] / wf.shape[0]
        # lit glyphs (real argon-counter sprites, else procedural fallback);
        # lit + ghost share the same cell so they register by construction.
        self.argon_seg_aspect: dict[str, float] = {}
        for ch, rgba in seg_glyphs.items():
            key = {".": "dot", "%": "pct", "x": "x"}.get(ch, ch)
            renderer.upload_texture(f"aseg_{key}", rgba)
            self.argon_seg_aspect[ch] = rgba.shape[1] / rgba.shape[0]
        # fixed-width digit advance (argon-counter digits are monospace)
        self.argon_seg_advance = self.argon_seg_aspect["8"]

        # --- legacy-default HUD set (custom-skin fallback, classic look) -----
        self.legacy_aspect: dict[str, float] = {}
        for ch, rgba in bake_legacy_font().items():
            key = {"%": "percent", ".": "dot", ",": "comma"}.get(ch, ch)
            renderer.upload_texture(f"lg_{key}", rgba)
            self.legacy_aspect[ch] = rgba.shape[1] / rgba.shape[0]
        renderer.upload_texture("lg_scorebar_bg", bake_legacy_scorebar_bg())
        renderer.upload_texture("lg_scorebar_colour",
                                bake_legacy_scorebar_colour())
        renderer.upload_texture("lg_ki", bake_legacy_ki())
        renderer.upload_texture("lg_kidanger",
                                bake_legacy_ki((255, 196, 70)))
        renderer.upload_texture("lg_kidanger2",
                                bake_legacy_ki((255, 80, 80)))
        renderer.upload_texture("lg_io_key", bake_legacy_io_key())
        renderer.upload_texture("lg_io_bg", bake_legacy_io_bg())

        # --- Argon gameplay set (skinless "Argon league" only) ---------------
        renderer.upload_texture("argon_circle", bake_argon_circle())
        renderer.upload_texture("argon_border", bake_argon_border())
        renderer.upload_texture("argon_approach", bake_argon_approach())
        renderer.upload_texture("argon_ball", bake_argon_ball())
        renderer.upload_texture("argon_ball_ring", bake_argon_ball_ring())
        renderer.upload_texture("argon_follow", bake_argon_follow())
        renderer.upload_texture("default_follow", bake_argon_follow(
            thickness_frac=5 / 64, fill=0.2, gradient=False))
        renderer.upload_texture("argon_followpoint", bake_argon_followpoint())
        renderer.upload_texture("argon_bubble", bake_argon_bubble())
        renderer.upload_texture("argon_tick", bake_argon_tick())
        renderer.upload_texture("argon_reverse", bake_argon_reverse())
        renderer.upload_texture("argon_cursor", bake_argon_cursor())
        renderer.upload_texture("argon_spin_ring", bake_argon_spin_ring())
        renderer.upload_texture("argon_spin_center", bake_argon_spin_center())
