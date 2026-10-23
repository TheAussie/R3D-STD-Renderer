"""Argon league (skinless) gameplay port — ppy/osu master Argon* pieces.

Covers the procedural bakes (geometry/colour), the ArgonJudgementPiece text
selection + colours + transforms, the ArgonSliderBody border sizing, and the
LEAGUE-INTEGRITY split at the sprite level: skinless draws draw the Argon
pieces; a custom skin never touches them (custom-skin renders unchanged).
All pure CPU / numpy — no GL context.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from osu_std_renderer.render import textures as T
from osu_std_renderer.render.scene import (
    ARGON_BALL_DIAM_FRAC, ARGON_FLASH_LIFE_MS, ARGON_GRAD_FRAC,
    ARGON_JUDGE_COLOR, ARGON_JUDGE_TEXT, ARGON_OUTER_GRAD_R,
    ARGON_RING_FADE_MS, ARGON_SLIDER_BORDER_WIDTH, StdScene,
    argon_judgment_transform,
)
from osu_std_renderer.render.markers import FollowPointDot
from osu_std_renderer.render.slider_body import border_portion
from osu_std_renderer.render.skin_elements import SkinElements
from osu_std_renderer.ruleset import JudgmentKind
from osu_std_renderer.skin.skin import Skin


# --- texture bakes: geometry + colour -----------------------------------------

def test_argon_circle_radial_bands():
    """ArgonMainCirclePiece: dark centre (Darken 4 ≈0.2), a bright saturated
    band, a dark gap, grayscale (accent-tintable), opaque centre."""
    rgba = T.bake_argon_circle(320)
    assert rgba.shape == (320, 320, 4) and rgba.dtype == np.uint8
    c = 160
    R = 320 / 2.0 - 2.0
    grey = rgba[..., 0].astype(float)
    # grayscale (r == g == b everywhere)
    assert np.array_equal(rgba[..., 0], rgba[..., 1])
    assert np.array_equal(rgba[..., 1], rgba[..., 2])
    centre = grey[c, c]
    bright = grey[c, int(c + 0.77 * R)]           # outer gradient band
    gap = grey[c, int(c + 0.90 * R)]              # dark gap (under the border)
    assert centre < 80.0                          # dark centre
    assert bright > 180.0                         # saturated band
    assert gap < 80.0                             # dark gap before the border
    # opaque centre, transparent corner
    assert rgba[c, c, 3] == 255
    assert rgba[0, 0, 3] == 0
    # lazer vertical gradient: brighter at the top of the bright band
    top = grey[int(c - 0.77 * R), c]
    bot = grey[int(c + 0.77 * R), c]
    assert top > bot + 8.0


def test_argon_border_is_white_ring():
    rgba = T.bake_argon_border(320)
    assert (rgba[..., :3] == 255).all()           # white (untinted)
    c = 160
    assert rgba[c, c, 3] == 0                      # hollow centre
    assert rgba[c, 312, 3] > 150                   # opaque near the rim


def test_argon_ball_gradient_and_chevron():
    """ArgonSliderBall: vertical accent gradient (bright top) + a dark '>'."""
    rgba = T.bake_argon_ball(256)
    grey = rgba[..., 0].astype(float)
    c = 128
    left = 83                                      # column left of the chevron
    # vertical accent gradient: top brighter than bottom (off the chevron)
    assert grey[64, left] > grey[192, left] + 15.0
    # a dark '>' chevron stroke exists in the centre box (accent.Darken 4)
    assert grey[110:146, 120:166].min() < 90.0


def test_argon_reverse_is_white_pill_with_dark_chevron():
    rgba = T.bake_argon_reverse(256)
    c = 128
    # a pill point right of the chevron (still inside the 40×20 pill) is
    # bright + opaque
    px = rgba[c, 160]
    assert px[3] > 150 and px[0] > 180
    # corners transparent
    assert rgba[4, 4, 3] == 0
    # a dark '>>' chevron stroke exists in the centre box (accent.Darken 4)
    grey = rgba[..., 0].astype(float)
    assert grey[114:142, 118:150].min() < 120.0


def test_argon_cursor_is_pink_ring():
    """ArgonCursor: pink→dark-red ring — coloured, not grayscale."""
    rgba = T.bake_argon_cursor(192)
    c = 96
    R = 192 / 2.0 - 2.0
    px = rgba[c, int(c + 0.92 * R)]               # in the ring band
    assert px[3] > 100                            # visible
    assert int(px[0]) > int(px[1]) and int(px[0]) > int(px[2])   # pinkish (R>G,B)


def test_argon_cursor_trail_matches_lazer_fade():
    """ArgonCursorTrail over the base CursorTrail (UI/Cursor/CursorTrail.cs):
    additive WHITE, Alpha 0.8, FadeExponent 4, and the trail window must equal
    the base FadeDuration (300 ms) so the ribbon length + (1-age/dur)^exp
    falloff match. Pass-3's 120 ms truncated it into a stubby blob."""
    from osu_std_renderer.render import scene as S
    assert S.ARGON_TRAIL_WINDOW_MS == 300.0        # == base CursorTrail FadeDuration
    assert S.ARGON_TRAIL_FADE_EXP == 4.0           # ArgonCursorTrail.FadeExponent
    assert S.ARGON_TRAIL_ALPHA == 0.8              # ArgonCursorTrail Alpha
    assert tuple(S.ARGON_CURSOR_TRAIL) == (1.0, 1.0, 1.0)   # no Colour → white
    # strength fades 1→0 across the full 300 ms window (age-linear, ^exp at draw)
    trail = [(0.0, 0.0, 0.0), (10.0, 0.0, 150.0), (20.0, 0.0, 300.0)]
    pts = S.long_trail_points(trail, [0.0, 150.0, 300.0], 300.0,
                              window_ms=S.ARGON_TRAIL_WINDOW_MS)
    strengths = [round(s, 3) for _, _, s in pts]
    assert strengths == [0.0, 0.5, 1.0]            # oldest→newest, 0 at window edge


def test_argon_tick_is_hollow_ring():
    rgba = T.bake_argon_tick(96)
    c = 48
    assert rgba[c, c, 3] == 0                      # hollow (ScorePoint fill α=0)
    assert rgba[c, 88, 3] > 150                    # opaque ring band


def test_argon_follow_has_ring_and_faint_fill():
    rgba = T.bake_argon_follow(256)
    grey = rgba[..., 0].astype(float)
    c = 128
    assert 30.0 < grey[c, c] < 140.0              # faint interior (Alpha 0.3)
    assert grey[c, 252] > 180.0                   # bright ring


# --- ArgonJudgementPiece: text + colour + transforms --------------------------

def test_judgment_text_selection():
    assert ARGON_JUDGE_TEXT[JudgmentKind.HIT300] == "GREAT"
    assert ARGON_JUDGE_TEXT[JudgmentKind.HIT100] == "OK"
    assert ARGON_JUDGE_TEXT[JudgmentKind.HIT50] == "MEH"
    assert ARGON_JUDGE_TEXT[JudgmentKind.MISS] == "MISS"


def test_judgment_colours_match_osucolour_forhitresult():
    def hexrgb(h):
        return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0,
                int(h[4:6], 16) / 255.0)
    assert ARGON_JUDGE_COLOR[JudgmentKind.HIT300] == hexrgb("66ccff")  # Blue
    assert ARGON_JUDGE_COLOR[JudgmentKind.HIT100] == hexrgb("88b300")  # Green
    assert ARGON_JUDGE_COLOR[JudgmentKind.HIT50] == hexrgb("ffcc22")   # Yellow
    assert ARGON_JUDGE_COLOR[JudgmentKind.MISS] == hexrgb("ed1121")    # Red


def test_judgment_transform_hit_fades_and_grows():
    # fades in over 300, whole piece fades out by 800, text scales 1→~1.2
    a0, s0, dy0, r0 = argon_judgment_transform(JudgmentKind.HIT300, 0.0)
    assert a0 == 0.0 and dy0 == 0.0 and r0 == 0.0
    a1, s1, _, _ = argon_judgment_transform(JudgmentKind.HIT300, 150.0)
    assert a1 > 0.0 and s1 > 1.0
    aL, sL, _, _ = argon_judgment_transform(JudgmentKind.HIT300, 799.0)
    assert 0.0 < aL < 0.05                          # nearly faded out
    assert 1.15 < sL <= 1.2
    assert argon_judgment_transform(JudgmentKind.HIT300, 800.0) is None


def test_judgment_transform_miss_drifts_down_and_rotates():
    a1, s1, dy1, r1 = argon_judgment_transform(JudgmentKind.MISS, 200.0)
    a2, s2, dy2, r2 = argon_judgment_transform(JudgmentKind.MISS, 600.0)
    assert dy2 > dy1 > 0.0                          # MoveToOffset (0,100) InQuint
    assert r2 > r1 > 0.0                            # RotateTo 40° InQuint
    assert a2 < a1                                  # FadeOutFromOne
    assert argon_judgment_transform(JudgmentKind.MISS, 800.0) is None
    assert argon_judgment_transform(JudgmentKind.MISS, -1.0) is None


# --- ArgonSliderBody: border sizing + geometry --------------------------------

def test_slider_border_width_maps_to_gradient_thickness():
    # intended_thickness = GRADIENT_THICKNESS/path_radius = 0.2 of path radius
    intended = ARGON_GRAD_FRAC / ARGON_OUTER_GRAD_R
    assert abs(intended - 0.2) < 1e-9
    assert abs(border_portion(ARGON_SLIDER_BORDER_WIDTH) - 0.2) < 5e-3


def test_argon_geometry_constants():
    assert abs(ARGON_OUTER_GRAD_R - 50.0 / 58.0) < 1e-12
    assert abs(ARGON_BALL_DIAM_FRAC - 50.0 / 58.0) < 1e-12
    assert abs(ARGON_GRAD_FRAC - 10.0 / 58.0) < 1e-12


# --- league integrity: sprite-level skinless-vs-skin split --------------------

class _FakeRenderer:
    def __init__(self):
        self.uploaded = {}

    def upload_texture(self, key, rgba):
        self.uploaded[key] = rgba


class _FakeCam:
    screen_w = 1280
    screen_h = 720

    def len_to_screen(self, v):
        return float(v)

    def to_screen(self, x, y):
        return float(x), float(y)


def _bare_scene():
    """A StdScene with only the attributes the tested draw helpers touch."""
    sc = object.__new__(StdScene)
    sc.radius_px = 30.0
    sc.circle_k = 1.0
    sc.cam = _FakeCam()
    _asp = {ch: 0.6 for ch in "GREATOKMEHMISS0123456789"}
    sc.bank = type("B", (), {"glyph_aspect": _asp, "argon_glyph_aspect": _asp,
                             "argon_digit_aspect": _asp, "digit_aspect": _asp,
                             "glyph_mono_advance": 0.6,
                             "argon_glyph_mono_advance": 0.6})()
    return sc


def _skin(tmp: Path) -> SkinElements:
    from PIL import Image as _I
    for n in ("hitcircle", "hitcircleoverlay"):
        _I.new("RGBA", (128, 128), (200, 120, 40, 255)).save(tmp / f"{n}.png")
    for i in range(10):
        _I.new("RGBA", (50, 70), (255, 255, 255, 255)).save(
            tmp / f"default-{i}.png")
    return SkinElements(Skin(skin_dir=tmp), _FakeRenderer())


def test_argon_circle_sprites_keys():
    sc = _bare_scene()
    sc.skin = None
    keys = [s.texture_key for s in
            sc._argon_circle_sprites(100.0, 100.0, (1.0, 0.5, 0.2), 1.0)]
    assert keys == ["argon_circle", "argon_border"]


def test_skinless_circle_is_argon_and_skin_never_is():
    """League integrity: skinless → Argon pieces; a custom skin path emits NO
    argon_* gameplay keys (its render is untouched by this change)."""
    sc = _bare_scene()
    sc.skin = None
    keys = [s.texture_key for s in
            sc._plain_circle_sprites(100.0, 100.0, (1.0, 0.0, 0.0), 1.0)]
    assert any(k.startswith("argon_") for k in keys)
    assert "disc" not in keys and "ring" not in keys

    with tempfile.TemporaryDirectory() as tmp:
        sc.skin = _skin(Path(tmp))
        keys = [s.texture_key for s in
                sc._plain_circle_sprites(100.0, 100.0, (1.0, 0.0, 0.0), 1.0)]
        assert not any((k or "").startswith("argon_") for k in keys)


def test_argon_ball_sprites_keys_and_follow():
    sc = _bare_scene()
    sc.skin = None
    tracked = [s.texture_key for s in
               sc._argon_ball_sprites(50.0, 50.0, (0.2, 0.6, 1.0), 1.0, True)]
    assert tracked == ["argon_follow", "argon_ball", "argon_ball_ring"]
    # follow circle is additive; ball + ring straight-alpha
    sprites = sc._argon_ball_sprites(50.0, 50.0, (0.2, 0.6, 1.0), 1.0, True)
    assert sprites[0].additive and not sprites[1].additive
    # not tracking → no follow circle
    untracked = [s.texture_key for s in
                 sc._argon_ball_sprites(50.0, 50.0, (0.2, 0.6, 1.0), 0.35,
                                        False)]
    assert untracked == ["argon_ball", "argon_ball_ring"]


def test_argon_judgment_run_glyphs_additive_coloured():
    sc = _bare_scene()
    out = []
    col = ARGON_JUDGE_COLOR[JudgmentKind.HIT300]
    sc._argon_judgment_run(out, "GREAT", 200.0, 200.0, 1.0, col, 0.8, 0.0)
    assert [s.texture_key for s in out] == [f"aglyph_{c}" for c in "GREAT"]
    assert all(s.additive for s in out)
    assert all(s.color[:3] == col and abs(s.color[3] - 0.8) < 1e-9
               for s in out)
    # zero alpha → nothing drawn
    out2 = []
    sc._argon_judgment_run(out2, "MISS", 0.0, 0.0, 1.0, col, 0.0, 0.0)
    assert out2 == []


# --- grade: lazer-EXACT rank table (ScoreProcessor + OsuScoreProcessor) --------

def test_grade_boundary_table_and_miss_cap():
    """osu.Game/Rulesets/Scoring/ScoreProcessor.RankFromScore cutoffs
    (X=1 / S=.95 / A=.9 / B=.8 / C=.7 / D=0) plus the std override
    OsuScoreProcessor.RankFromScore (a Miss downgrades S/X to A)."""
    from osu_std_renderer.render.hud import grade_for, std_accuracy

    def acc_pct(c300, c100, c50, cm):
        return round(std_accuracy(c300, c100, c50, cm) * 100.0, 2)

    # (c300, c100, c50, cmiss, expected_grade, expected_acc%)
    cases = [
        (100, 0, 0, 0, "SS", 100.00),      # perfect
        (47, 0, 3, 0, "S", 95.00),         # exactly the S cutoff, clean
        (46, 3, 1, 0, "A", 94.33),         # just below S → A
        (98, 0, 0, 2, "A", 98.00),         # 98% but a miss caps S/X → A
        (22, 0, 3, 0, "A", 90.00),         # exactly the A cutoff, clean
        (126, 24, 0, 0, "B", 89.33),       # 89.33% clean → B (old rule: A)
        (125, 24, 0, 1, "B", 88.67),       # ~89% WITH a miss → B (owner case)
        (21, 0, 4, 0, "B", 86.67),
        (14, 0, 6, 0, "C", 75.00),         # ≥70 → C
        (10, 0, 6, 0, "D", 68.75),         # <70 → D
        (0, 0, 0, 1, "D", 0.00),
    ]
    for c300, c100, c50, cm, exp, exp_acc in cases:
        assert abs(acc_pct(c300, c100, c50, cm) - exp_acc) < 0.02, \
            (c300, c100, c50, cm, acc_pct(c300, c100, c50, cm))
        got = grade_for(c300, c100, c50, cm)
        assert got == exp, f"{c300}/{c100}/{c50}/{cm} ({exp_acc}%) -> {got}, want {exp}"

    # the miss-cap only touches S/X — an A/B/C is NOT pushed further down
    assert grade_for(90, 10, 0, 0) == "A"          # 93.33% clean → A
    assert grade_for(89, 10, 0, 1) == "A"          # 92.83% + miss → still A
    assert grade_for(100, 0, 0, 0) == "SS"         # only all-300 → SS
    assert grade_for(99, 0, 0, 1) != "SS"          # any miss is never SS


# --- item 4 (pass 3): circle bands = lazer-EXACT ArgonMainCirclePiece stack ----

def test_argon_circle_bands_are_lazer_exact_darken_stack():
    """Fidelity pass 3 REVERT of the pass-2 exaggeration: the concentric
    bands are the EXACT ArgonMainCirclePiece darken stack — outerGradient ≈
    full white, innerGradient = GradientVertical(Darken(0.5), Darken(0.6)) ≈
    0.646 grey mid-row, innerFill = Darken(4) = 0.20 dark centre. A bright
    outer band, a distinct dark-amber mid ring, a dark centre — NOT the
    pass-2 pushed 0.50→0.46 mid step."""
    grey = T.bake_argon_circle(320)[..., 0].astype(float)
    R = 320 / 2.0 - 2.0
    c = 160
    outer = grey[c, int(c + 0.77 * R)]     # outerGradient bright band
    mid = grey[c, int(c + 0.60 * R)]       # innerGradient mid band
    centre = grey[c, c]                    # innerFill dark centre
    assert outer > 220                     # saturated outer band
    assert centre < 60                     # Darken(4) dark centre
    # innerGradient mid-row value = (Darken(0.5)+Darken(0.6))/2 =
    # (0.6667+0.625)/2 = 0.646 → ~165/255. Locks the EXACT reverted value
    # (the pass-2 push put this at ~0.48 → ~122).
    assert 155 < mid < 175
    assert mid > centre + 80               # a distinct mid ring
    assert mid < outer - 40                # ...clearly below the outer band


def test_argon_approach_is_crisp_core_plus_soft_shoulder():
    """Pass-4 re-measure of frame_26 (real lazer, Argon): the approach ring is
    a SHARP bright core (~11 px / 166 px radius = 0.066, matching the bare
    DefaultApproachCircle texture proportion 0.064) PLUS a soft glow shoulder
    trailing to ~19 px total. Pass-3's FLAT solid 0.11 band read too THICK
    (owner-flagged, pass 4). We bake a thin bright core + a Gaussian shoulder:
    the SOLID-bright band must be thin (~0.066) while a wider, dimmer halo
    exists around it — crisp like the game, not a uniform slab."""
    assert abs(T.ARGON_APPROACH_THICKNESS - 0.066) < 1e-9
    a = T.bake_argon_approach(512)[..., 3].astype(float)
    R = 512 / 2.0 - 2.0
    c = 256
    row = a[c]

    def band(thresh):
        idx = np.where(row > thresh)[0]
        idx = idx[idx > c]                 # ring band on the right half
        assert idx.size > 0
        return (idx.max() - idx.min() + 1) / R

    core = band(230)                       # solid bright band (alpha > 0.9)
    halo = band(60)                        # incl. the soft glow shoulder
    assert 0.045 < core < 0.085            # crisp thin core, NOT the 0.11 slab
    assert halo > core + 0.04              # a real soft shoulder surrounds it
    assert halo < 0.20                     # but bounded — not a blob


# --- item 3: ArgonFollowPoint (pink chevrons) ---------------------------------

def test_argon_followpoint_texture_is_pink_chevron():
    rgba = T.bake_argon_followpoint(96)
    a = rgba[..., 3]
    assert a.max() > 150                   # a solid chevron exists
    lit = a > 100
    r = rgba[..., 0][lit].mean()
    g = rgba[..., 1][lit].mean()
    b = rgba[..., 2][lit].mean()
    assert r > g and r > b                  # pink→dark-red (R dominant)


def test_argon_followpoint_sprite_is_rotated_chevron():
    sc = _bare_scene()
    sc.skin = None
    sc.circle_k = 1.0
    sc.diff = type("D", (), {"time_fade_in": 400.0})()
    dot = FollowPointDot(fade_in=0.0, fade_out=1000.0, x_start=0.0,
                         y_start=0.0, x_end=32.0, y_end=32.0, rotation=0.7853)
    sc._fp_dots = [dot]
    sc._fp_idx = 0
    sc._fp_active = []
    out = sc._followpoint_sprites(200.0)
    assert out and out[0].texture_key == "argon_followpoint"
    assert out[0].additive
    assert abs(out[0].rotation - 0.7853) < 1e-9   # rotated along the path


# --- item 2: ArgonCursorTrail (thin white line) -------------------------------

def test_argon_cursor_trail_is_thin_white_additive():
    from types import SimpleNamespace
    sc = _bare_scene()
    sc.skin = None
    sc.cursor_scale = 1.0
    sc.trail_scale = 1.0
    sc.cursor_rainbow = False
    sc._argon_trail_pts = [(float(i), 0.0, float(i)) for i in range(0, 20)]
    sc._argon_trail_times = [p[2] for p in sc._argon_trail_pts]
    sc.frames = [SimpleNamespace(x=0.0, y=0.0, time_ms=0.0, keys=0),
                 SimpleNamespace(x=19.0, y=0.0, time_ms=19.0, keys=0)]
    out = sc._argon_cursor_sprites(19.0)
    trail = [s for s in out if s.texture_key == "glow" and s.additive
             and tuple(s.color[:3]) == (1.0, 1.0, 1.0)]
    assert len(trail) >= 3                  # a continuous white line, not a blob
    # every additive glow tint is white (trail) or cyan (centre) — the old
    # pink/red soft comet (R>B) is gone
    for s in out:
        if s.texture_key == "glow" and s.additive:
            r, _, b = s.color[:3]
            assert not (r > b + 0.1)


# --- item 1: ArgonMainCirclePiece flash + RingExplosion bubbles ----------------

class _Ev:
    def __init__(self, kind, x, y, oid, tm=0.0):
        self.kind = kind
        self.x = x
        self.y = y
        self.object_id = oid
        self.time_ms = tm


def test_argon_hit_flash_blooms_then_expires():
    sc = _bare_scene()
    sc.skin = None
    out = []
    sc._argon_hit_flash(out, 100.0, 100.0, (0.4, 0.8, 1.0), 80.0)
    assert out and all(s.additive for s in out)
    keys = {s.texture_key for s in out}
    assert "glow" in keys and "argon_circle" in keys   # accent glow + white bloom
    assert any(tuple(s.color[:3]) == (1.0, 1.0, 1.0) for s in out)  # white flash
    # expired outside the flash life
    out2 = []
    sc._argon_hit_flash(out2, 0.0, 0.0, (1, 1, 1), ARGON_FLASH_LIFE_MS + 1.0)
    assert out2 == []
    out3 = []
    sc._argon_hit_flash(out3, 0.0, 0.0, (1, 1, 1), -1.0)
    assert out3 == []


def test_ring_explosion_bubble_counts_scatter_and_fade():
    sc = _bare_scene()
    sc.skin = None
    counts = {JudgmentKind.HIT300: 8, JudgmentKind.HIT100: 4,
              JudgmentKind.HIT50: 3, JudgmentKind.MISS: 0}
    for kind, n in counts.items():
        out = []
        sc._argon_ring_explosion(out, _Ev(kind, 100.0, 120.0, 7), 50.0)
        assert len(out) == n, (kind, len(out))
        assert all(s.texture_key == "argon_bubble" and s.additive for s in out)
        if n:
            col = ARGON_JUDGE_COLOR[kind]
            assert all(tuple(s.color[:3]) == col for s in out)
    e = _Ev(JudgmentKind.HIT300, 100.0, 120.0, 7)

    def spread(out):
        return max(math.hypot(s.x - 100.0, s.y - 120.0) for s in out)

    early, late = [], []
    sc._argon_ring_explosion(early, e, 40.0)
    sc._argon_ring_explosion(late, e, 500.0)
    assert spread(late) > spread(early)            # bubbles scatter outward
    assert late[0].color[3] < early[0].color[3]    # group fades out
    gone = []
    sc._argon_ring_explosion(gone, e, ARGON_RING_FADE_MS + 1.0)
    assert gone == []


# --- item 6: Argon 7-segment counter glyphs (aligned lit + wireframe) ---------

def test_argon_segment_digits_and_wireframe_align():
    wire = T.bake_wireframe_cell()[..., 3]
    eight = T.bake_argon_segment("8")[..., 3]
    # the lit '8' covers every wireframe pixel (lit+ghost register exactly →
    # the phantom-8 fix)
    assert (eight[wire > 10] > 10).mean() > 0.99
    # '1' lights ONLY the right-side segments (B, C)
    one = T.bake_argon_segment("1")[..., 3]
    hcut = one.shape[1] // 2
    assert one[:, hcut + 8:].max() > 150
    assert one[:, :hcut - 8].max() < 40
    # '.' is a dot low in the cell; white & tintable
    dot = T.bake_argon_segment(".")
    assert (dot[..., :3] == 255).all()
    ys, _ = np.where(dot[..., 3] > 128)
    assert ys.mean() > dot.shape[0] * 0.6
    # digits are fixed-width (all share the cell canvas → monospace)
    ref = T.bake_argon_segment("5").shape
    for ch in "0123456789.%x":
        assert T.bake_argon_segment(ch).shape == ref


# --- fidelity pass 3: judgment text size + the '%' glyph ----------------------

def test_judgment_font_and_spacing_match_argon_source():
    """ppy/osu master ArgonJudgementPiece.CreateJudgementText (osu.Game.
    Rulesets.Osu/Skinning/Argon/ArgonJudgementPiece.cs):
        Font = OsuFont.Default.With(size: 20, weight: FontWeight.Bold)
        Spacing = new Vector2(5, 0)
    (Was 25/7 — a legibility fudge that read ~27 % too big. Text now renders
    in the bundled Argon font (Nunito, caps fill ~0.7025 of the sprite); the
    run scales its height by ARGON_GLYPH_CAP_SCALE so size-20 keeps the same
    ~14.4-osu!px visible cap, matching lazer's Torus size-20.)"""
    from osu_std_renderer.render.scene import (
        ARGON_JUDGE_FONT_OSU, ARGON_JUDGE_SPACING_OSU)
    assert ARGON_JUDGE_FONT_OSU == 20.0
    assert ARGON_JUDGE_SPACING_OSU == 5.0


def test_bundled_argon_font_loads_and_is_not_dejavu():
    """The Argon-league font is a real bundled OFL asset (not the DejaVu
    fallback / PIL default bitmap) with its OFL licence alongside."""
    import os
    from PIL import ImageFont
    from osu_std_renderer.render import textures as T
    assert os.path.exists(T.ARGON_FONT_PATH)
    assert os.path.exists(os.path.join(os.path.dirname(T.ARGON_FONT_PATH),
                                       "OFL.txt"))
    f = T._load_argon_font(64)
    assert isinstance(f, ImageFont.FreeTypeFont)     # not load_default() bitmap
    fam = (f.getname()[0] or "").lower()
    assert "nunito" in fam and "dejavu" not in fam
    # renders visible glyphs
    glyphs = T.bake_argon_glyphs("GREAT")
    assert set(glyphs) == set("GREAT")
    assert all(g[..., 3].max() > 0 for g in glyphs.values())


def test_hud_font_is_the_bundled_one_on_every_platform():
    """The DejaVu labels must never fall through to Pillow's built-in font:
    it ignores the size asked for, and the Argon scales are measured against
    this font (a Mac drew GREAT at 65% and the key counts tiny)."""
    import os
    from PIL import ImageFont
    from osu_std_renderer.render import textures as T
    assert os.path.isfile(T.BUNDLED_FONT_PATH), T.BUNDLED_FONT_PATH
    assert T._FONT_CANDIDATES[0] == T.BUNDLED_FONT_PATH
    f = T._load_font(64)
    assert isinstance(f, ImageFont.FreeTypeFont) and f.path == T.BUNDLED_FONT_PATH, getattr(f, "path", f)
    assert f.getname() == ("DejaVu Sans", "Bold")
    x0, y0, x1, y1 = f.getbbox("GREAT")
    assert 40 <= y1 - y0 <= 60 and x1 - x0 > 200, (x0, y0, x1, y1)   # really 64 px


def test_argon_cap_scale_holds_visible_size():
    """The glyph/digit cap-fill scales re-derive the size so the font swap
    does NOT change the visible cap size (measured cap-fill drives them)."""
    from osu_std_renderer.render import textures as T
    # DejaVu fills MORE of the glyph-bank sprite than Nunito → scale up >1;
    # the digit bank too now scales up >1 (Pillow 12 shrank Nunito).
    assert 1.0 < T.ARGON_GLYPH_CAP_SCALE < 1.05
    assert 1.0 < T.ARGON_DIGIT_CAP_SCALE < 1.05
    # the constants match the actual bakes within tolerance (guards drift if
    # the bundled font / weight changes).
    def fill(chars, loader, cap="E"):
        px = int(T.DIGIT_HEIGHT * 0.95)
        font = loader(px)
        boxes = [font.getbbox(c) for c in chars]
        top, bot = min(b[1] for b in boxes), max(b[3] for b in boxes)
        cb = font.getbbox(cap)
        return (cb[3] - cb[1]) / ((bot - top) + 8)
    gd = fill(T.HUD_CHARSET, T._load_font)
    ga = fill(T.HUD_CHARSET, T._load_argon_font)
    assert abs((gd / ga) - T.ARGON_GLYPH_CAP_SCALE) < 0.02
    dd = fill("0123456789", T._load_font, "8")
    da = fill("0123456789", T._load_argon_font, "8")
    assert abs((dd / da) - T.ARGON_DIGIT_CAP_SCALE) < 0.02


def test_argon_percent_glyph_reads_as_percent():
    """The Argon counter '%' is two OPEN rings (upper-left + lower-right)
    joined by a bottom-left→top-right slash — NOT a filled box / crossed 'Z'.
    Each ring has a lit band around a HOLLOW centre so it survives HUD scale
    (the old glyph collapsed to two dots + a box and read as a 'Z')."""
    pct = T.bake_argon_segment("%")[..., 3].astype(float) / 255.0
    h, w = pct.shape
    assert pct.max() > 0.9                                  # glyph present
    rr = h * 0.135                                          # ring radius
    for cx, cy in ((0.31 * w, 0.205 * h), (0.69 * w, 0.795 * h)):
        # hollow centre + a lit band all the way round (the ring is OPEN)
        assert pct[int(cy), int(cx)] < 0.35
        band = [pct[int(cy + dy), int(cx + dx)]
                for dx, dy in ((rr, 0), (-rr, 0), (0, rr), (0, -rr))]
        assert min(band) > 0.5
    # the diagonal slash crosses the cell centre
    assert pct[int(0.5 * h), int(0.5 * w)] > 0.6
    # NOT a box: the vertical side edges (a box's walls) are empty
    assert pct[h // 3:2 * h // 3, :w // 12].mean() < 0.15
    assert pct[h // 3:2 * h // 3, -(w // 12):].mean() < 0.15
    # distinct from a full '8' cell (rings + slash, not all segments)
    eight = T.bake_argon_segment("8")[..., 3].astype(float) / 255.0
    assert (eight[pct > 0.5] > 0.5).mean() < 0.6
