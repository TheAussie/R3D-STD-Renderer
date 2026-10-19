"""Dependency-free test runner (pytest also works on these files):

    python -m tests.run_all
"""
from __future__ import annotations

import importlib
import sys
import traceback

MODULES = [
    "tests.test_difficulty",
    "tests.test_sliderpath",
    "tests.test_parser_synthetic",
    # osu!stable replay-clock lead-in: osrparse strips the leading
    # (256,-500) placeholder frames without accumulating their deltas,
    # dropping the intro-skip/audio lead-in; parse_replay recovers it so
    # the cursor clock matches the beatmap (fixes ~25%% stable mass-miss)
    "tests.test_leadin_offset",
    "tests.test_skin_ini",
    "tests.test_ruleset",
    "tests.test_slider_visual_events",
    "tests.test_slider_chronological",
    "tests.test_slider_visual_animation",
    "tests.test_slider_visual_render",
    "tests.test_health",
    "tests.test_slider_body",
    "tests.test_playfield_pipeline",
    "tests.test_scene",
    "tests.test_hud",
    "tests.test_background",
    "tests.test_skin_elements",
    "tests.test_markers",
    "tests.test_spinner",
    "tests.test_gameplay_polish",
    "tests.test_hitsounds",
    "tests.test_results",
    "tests.test_lazer_results",
    "tests.test_leaderboard",
    # settings-surface phase (2026-07)
    "tests.test_settings",
    "tests.test_encode",
    "tests.test_effects",
    "tests.test_nightcore",
    "tests.test_pp",
    "tests.test_video_bg",
    "tests.test_hud_extras",
    # Argon league (skinless) gameplay port (2026-07)
    "tests.test_argon",
    # mod visuals: OsuModHidden fades + OsuModFlashlight overlay (2026-07)
    "tests.test_mods",
    # osu!(lazer) Classic mod (CL) switchboard (2026-07)
    "tests.test_classic_mod",
    # osu!(lazer) Difficulty Adjust mod (DA): AR/CS/OD/HP override + extended
    # limits (2026-07)
    "tests.test_difficulty_adjust",
    # osu!(lazer) custom-rate mods (DT/NC/HT/DC speed_change): effective rate
    # over the bitmask + NC/DC pitch vs DT/HT tempo audio (2026-07)
    "tests.test_rate_mods",
    # Relax auto-tap synthesis (RX) + Autopilot path (2026-07)
    "tests.test_relax",
    # FAIL/DEATH handling: detection + animation transforms + F grade (2026-07)
    "tests.test_fail",
    # osu!(lazer) Wind Up / Wind Down (WU/WD, ModTimeRamp): a clock rate that
    # RAMPS over the map — ramp parse + the wall<->map time warp + audio warp
    # + byte-identical non-ramp path (2026-07)
    "tests.test_wind_mods",
    # position mods: Mirror (MR) + Random (RD), with the .NET Random port
    # verified vs dotnet/runtime vectors (2026-07)
    "tests.test_pos_mods",
    # transform-family "fun" mods: GR/DF/SI/WG/TR — per-object entrance
    # animation (visual only; reconcile/geometry untouched) (2026-07)
    "tests.test_transform_mods",
    # approach/circle-appearance mods: FR (Freeze Frame) / AD (Approach
    # Different) / TC (Traceable) — approach-circle scale/timing + circle-fill
    # gate (visual only; reconcile/geometry untouched) (2026-07)
    "tests.test_appearance_mods",
    # screen / cursor-effect visual mods: BR (Barrel Roll) / BM (Bloom) / SY
    # (Synesthesia) / BL (Blinds) / NS (No Scope) / DP (Depth) / BU (Bubbles) —
    # playfield spin, cursor scale/fade, snap colours, screen blinds, per-object
    # depth + hit-position bubbles (visual only; reconcile/geometry untouched)
    "tests.test_screen_mods",
    # cursor-driven object-movement mods: MG (Magnetised, objects pulled TO the
    # cursor) / RP (Repel, objects pushed AWAY) — a stateful per-frame easeTo
    # (Interpolation.DampContinuously) over the recorded cursor; both hide
    # follow points (visual only; reconcile/geometry untouched)
    "tests.test_repel_magnet",
    # Target Practice (TP): a CONVERSION mod — the map is DISCARDED and rebuilt
    # as seeded "target" hit circles on the beat (OsuModTargetPractice via
    # DotNetRandom). The layout is reproduced bit-for-bit from the .osr seed
    # (object 0 re-derived straight from the C#); non-TP renders untouched.
    "tests.test_target_practice",
    # storyboard engine phase 1: the .osu[Events]+.osb PARSER (data model only;
    # command execution + rendering are later phases) (2026-07)
    "tests.test_storyboard",
    # storyboard engine phase 2: the COMMAND ENGINE (time -> per-sprite transform
    # state). Ports the ~36 osu.Framework easings + Interpolation.ValueAt
    # (scalar/vector/linear-colour), per-property transform-overwrite timeline
    # semantics, loop unrolling, sprite lifetime (alpha-0 opt), alpha flicker,
    # flip/additive params; trigger FIRING is stubbed (machinery only) (2026-07)
    "tests.test_storyboard_engine",
    # storyboard engine phase 3: the RENDERER (per-time SpriteState -> drawn
    # frames). 640x480->output transform + widescreen letterbox, 9 origins x
    # flip origin-adjust, additive z-slot blend, animation frame index, lazy
    # LRU texture cache; GL draw proven by an EGL smoke test (2026-07)
    "tests.test_storyboard_render",
    # sub-1080p results outro on the GPU: integer LANCZOS == Pillow's
    "tests.test_ssaa_gpu",
    # several textures per draw call: the plan, and drawn bytes off vs on
    "tests.test_multitex",
    # GPU colour conversion == ffmpeg's own, sample for sample
    "tests.test_gpu_yuv",
    # inline preview on the Mac's media engine: the master's arguments never
    # change, the probe decides, a hardware failure switches it off
    "tests.test_preview_hw",
    # the Metal renderer against the GL one (skips off a Mac)
    "tests.test_metal",
    # audio prepared beside the frame loop (R3D_STD_AUDIO_LATE, default off)
    "tests.test_audio_late",
]


def main() -> int:
    passed = failed = 0
    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
                print(f"PASS {mod_name}.{name}")
            except Exception:  # noqa: BLE001
                failed += 1
                print(f"FAIL {mod_name}.{name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
