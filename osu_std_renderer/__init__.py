"""osu!standard renderer — the 4th in-house R3D mode engine (mode 0).

Stack (locked): Python 3.12 + moderngl (headless EGL, no X server) + numpy +
Pillow + osrparse + raw-RGB frames piped to ffmpeg. NO BASS (audio is
numpy/ffmpeg mixing, see record/audio.py), NO GLFW, NO Go.

Behavioral spec: docs/RENDER_PLAN.md — a complete architectural teardown of
the reference Go renderer. This package ports the plan's MATH and PIPELINE
SEMANTICS (formulas cited per-module by plan section number) onto the same
stack that ships the mania/catch/taiko engines.

Package map (module → RENDER_PLAN.md section):
    beatmap/parser.py        §2.1–2.2  3-pass .osu parse, combo numbering
    beatmap/beatmap.py       §2.3      Beatmap struct + ParsePoint + memoised stacking
    beatmap/difficulty.py    §2.4      DifficultyRate, radius/preempt/windows, mods
    beatmap/objects/base.py  §2.5      commonParse, HitObject base, ModifyPosition
    beatmap/objects/circle.py §2.5     hit circles
    beatmap/objects/slider.py §2.5     velocity/tick math, scorePath walk
    beatmap/objects/spinner.py §2.5    spinners
    beatmap/objects/timing.py §2.6     red/green lines, GetRatio SV clamp
    beatmap/stacking.py      §2.7      v6+ new / old stacking algorithms
    beatmap/pause.py         §2.8      breaks
    curves/sliderpath.py     §2.9      bit-exact PathApproximator (verbatim from osu-catch)
    curves/__init__.py       §2.9      StdSliderPath (2D head, no flip) wrapper
    replay/replay.py         §5.5      .osr decode + 2D cursor interpolation
    ruleset/ruleset.py       §5.5      STUB: hit windows/notelock/judging contract
    skin/skin_ini.py         §3.2      full std skin.ini surface
    skin/skin.py             §3.1/3.3  fallback chain, @2x, animation frames
    render/context.py        —         headless EGL context (adapted from catch)
    render/gl.py             —         sprite batch (adapted from catch)
    render/playfield.py      §5.4      exact 512×384 → screen camera
    render/slider_body.py    §2.5/A§4  STUB: distance-field slider body (Phase-0 spike)
    record/pipeline.py       §5.2      fixed-timestep deterministic record loop
    record/encode.py         §5.6      ffmpeg rawvideo pipe (adapted from mania v2)
    record/audio.py          §5.6      NO-BASS offline numpy mix → wav → ffmpeg
    settings.py              §4        settings surface ↔ R3D preset keys
    cli.py                   —         adapter-contract CLI (mirrors catch_renderer.py)
"""

import warnings

from PIL import Image

# Treat Pillow's warning-only decompression-bomb band as a failed asset, not a
# memory allocation.  Every untrusted skin/beatmap image passes through Pillow.
Image.MAX_IMAGE_PIXELS = 50_000_000
warnings.simplefilter("error", Image.DecompressionBombWarning)

__version__ = "0.1.0.dev0"
