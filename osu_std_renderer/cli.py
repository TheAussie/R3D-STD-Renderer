"""CLI entrypoint — mirrors the adapter contract of the catch renderer
(mania_ordr/catch_renderer.py shells out to `python -m osu_catch_renderer
REPLAY BEATMAP_DIR -o out.mp4 …`; the future std_renderer.py adapter will
call this module identically):

    python -m osu_std_renderer REPLAY.osr BEATMAP_DIR -o out.mp4 \
        [--resolution 1920x1080] [--fps 60] [--encoder auto] [--skin DIR] …

CURRENT STATE (Phase-1 record path + the judgment phase): procedural
textures (render/textures.py), object lifecycle + scene (render/scene.py),
slider bodies (render/slider_body.py), cursor+trail from the replay,
fixed-timestep record loop (record/pipeline.py) into the single-process
ffmpeg pipe (record/encode.py) with offline-mixed music (record/audio.py).
Judgments ARE simulated
(ruleset/ruleset.py — ported ppy/osu logic, stable notelock + classic
sliders, reconciled to the .osr's authoritative counts; the pre-reconcile
sim-vs-real delta is printed as the honesty metric). Explosions fire at
real hit times, misses fade out, judgment popups show, sliderbreaks dim
the ball. The §4.6 gameplay HUD is live (render/hud.py — score/acc/grade/
progress/combo/hit-error+UR/key-overlay/break-flash in the §5.3 virtual
1080p UI space; a `hud:` final-values line prints after each render).
Progress lines match catch's `rendering… NN%` shape.

BACKGROUND + SKIN PHASE: the map's `[Events]` background renders under
everything with the §4.10 dim envelope (render/background.py; intro/game/
break dims from the R3D preset keys, missing bg fails soft to the dark
void). `--skin DIR` loads a real skin's core gameplay textures
(render/skin_elements.py) — per-element fallback to the procedural set;
a `skin:` report line lists what came from the skin.

ARROWS + TICKS + FOLLOW POINTS PHASE (render/markers.py + scene draws):
reverse arrows (beat-pulsed, white, tangent-oriented), slider ticks of
the active span, follow points on the lazer schedule (--follow-points
honors DrawFollowPoints), sliderstartcircle/sliderendcircle head/tail
specialisations, sliderb/followpoint AnimationFramerate cycling, and the
§3.3 two-mode skin cursor trail (cursormiddle → long connected trail,
else sparse 16.67 ms drops; --force-long-trail = §4.7 ForceLongTrail).

SPINNER + HIT LIGHTING PHASE: spinners render (render/spinner.py + scene
_draw_spinner — §3.3 style auto-detect old/new/procedural, replay-driven
rotation, metre/glow progress, approach circle, SPIN!/CLEAR!/RPM,
SpinnerFadePlayfield) and non-miss judgments flash a combo-tinted
`lighting` glow under the popup (--no-hit-lighting; preset default ON).
HITSOUND PHASE (record/hitsounds.py): §3.4 hitsounds mix into the same
offline audio track — the 3×7 sample grid through BEATMAP(custom index)
→ skin chain → synthesized defaults, one-shots at judged hit times
(layered/bit semantics, timing-point volumes floored at 0.08),
slidertick one-shots, sliderslide/sliderwhistle loops tiled over the
ruleset's tracking windows and spinnerspin over spins (--no-hitsounds /
--use-skin-hitsounds / --hitsound-volume control it). Hitsounds are
judgment-driven, so they require a replay; `--no-replay` stays a bare
debug flag (visuals only — no cursor/HUD/judgments/hitsounds).

OWNER PUNCH-LIST PHASE (2026-07): distance-based long cursor trail
(uniform spacing/brightness at any speed), a coherent HUD skin container
(2026-10 migration: extracted --skin selects Legacy; missing assets keep
its anchors). --legacy-defaults also selects Legacy skinless and the
legacy synth bank; --renderer-default-font-and-ranks changes glyph/rank
appearance only. Brighter-inside slider bodies (BodyStyle
defaults), the classic falling hit0 miss (--miss-fall), skin ranking-*
images with --renderer-default-font-and-ranks, multi-reverse arrows
gated on the head hit, the bottom-parked hit-error/UR block,
--playfield-borders, and RED'S results screen as the outro
(render/results.py — --results is now IMPLEMENTED, results_screen_time
honored, mania-card layout).

SETTINGS-SURFACE PHASE (2026-07 — the full R3D website preset surface):
every std setting mania_ordr/presets.py exposes now maps to a flag and
renders (settings.py's table is the authority): nightcore beat overlay,
mod pills, live pp counter (rosu-pp gradual), hit counter, aim-error
scatter, strain graph, warning arrows, bg blur/parallax/triangles/video
(video_bg.py)/flash-to-beat, seizure card + lead-in pre-roll, real
fade-to-black (video+audio), bloom (+to-beat), the R3D logo splash,
cursor trail-scale/rainbow/ripples, slider snaking-out + merge, beatmap
[Colours] vs skin combo colours, and -skip intro trimming.
StdRenderSettings.from_preset(dict) consumes the bot preset JSON
wholesale — the future std_renderer.py service adapter calls that.
Accepted + NO-OP (documented): show_scoreboard/scoreboard_avatars
(render/scoreboard.py holds the osu!API JSON hand-off stub).
load_storyboard now RENDERS (render/storyboard_render.py, phase 3);
the service still gates it off on the free tier.

Debug extras:
    --parse-only            parse map+replay+skin, print a summary, exit 0
    --start N               start the render N seconds into the map
    --max-seconds N         stop the render N seconds in (MVP clips;
                            --start 35 --max-seconds 45 → a 35s-45s clip)
    --dump-frames "a,b,c"   write single-frame PNGs at map times a,b,c (ms)
                            instead of encoding video
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from pathlib import Path

from .beatmap import load_full
from .render import perf
from .beatmap.difficulty import HIT_FADE_OUT
from .beatmap.objects import Slider, Spinner
from .replay import (KEY_SMOKE, is_relax_meta, parse_replay,
                     synthesize_relax_frames)
from .settings import StdRenderSettings
from .skin.skin_ini import load as load_skin_ini


def _resolution(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def _bitrate(s: str) -> int:
    """Video bitrate: '16M' / '24000k' / '16000000' -> bits per second."""
    s = s.strip().lower()
    mult = 1
    if s.endswith("m"):
        mult, s = 1_000_000, s[:-1]
    elif s.endswith("k"):
        mult, s = 1_000, s[:-1]
    return int(float(s) * mult)


def _ms_list(s: str) -> list[float]:
    return [float(tok) for tok in s.split(",") if tok.strip()]


# visual-only mod overrides (--add-mods): HD/FL are driven purely by the mod
# bitmask and don't alter geometry/timing, so they can be forced on ANY replay
# without re-loading the beatmap or touching judgment reconciliation.
_VISUAL_MOD_BITS = {"HD": 1 << 3, "FL": 1 << 10}


def _mod_list(s: str) -> int:
    bits = 0
    for tok in s.replace(",", " ").split():
        acr = tok.strip().upper()
        if acr not in _VISUAL_MOD_BITS:
            raise argparse.ArgumentTypeError(
                f"--add-mods only forces visual mods {sorted(_VISUAL_MOD_BITS)}"
                f" (got {acr!r})")
        bits |= _VISUAL_MOD_BITS[acr]
    return bits


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="osu_std_renderer")
    ap.add_argument("osr", type=Path, nargs="?", default=None,
                    help="replay .osr file (omit with --no-replay)")
    ap.add_argument("beatmap", type=Path, nargs="?", default=None,
                    help="dir with .osu + audio + bg, or a direct .osu path")
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--resolution", type=_resolution, default=(1920, 1080))
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--encoder", default="auto",
                    help="auto | h264_nvenc | h264_vaapi | libx264")
    ap.add_argument("--encoder-device", default=None)
    ap.add_argument("--video-bitrate", type=_bitrate, default=None,
                    help="override the auto resolution-scaled bitrate, e.g. "
                         "16M / 24M / 30000k. Default: the built-in ladder.")
    ap.add_argument("--skin", type=Path, default=None,
                    help="extracted skin dir (resolver-provided .osk)")
    ap.add_argument("--default-skin", type=Path, default=None)
    BA = argparse.BooleanOptionalAction
    ap.add_argument("--skip-intro", action=BA, default=True,
                    help="trim a long silent intro: start ~1 s before the "
                         "first object's approach (danser -skip semantics; "
                         "only when it saves > 2 s and --start is not given)")
    ap.add_argument("--lead-in", type=float, default=0.0, metavar="SEC",
                    help="§4.10 LeadInTime: extra pre-roll hold before the "
                         "map starts (wall seconds, on top of the intro)")
    ap.add_argument("--fade-out", type=float, default=None, metavar="SEC",
                    help="§4.10 FadeOutTime: video+audio fade to black "
                         "after the last object (wall seconds; default 1.5)")
    ap.add_argument("--results", action=BA, default=True)
    ap.add_argument("--leaderboard", action=BA, default=True,
                    help="per-map render leaderboard on the lazer results "
                         "screen (featured play flanked by other renders of "
                         "the same map, from the local render DB); default on")
    ap.add_argument("--leaderboard-source", choices=("r3d", "osu"),
                    default="r3d",
                    help="flank-card source: 'r3d' = the local render DB "
                         "(default), 'osu' = the map's osu! GLOBAL top scores "
                         "from --leaderboard-json (silently falls back to r3d "
                         "when that file is missing/empty/invalid)")
    ap.add_argument("--leaderboard-json", type=Path, default=None,
                    help="path to the bot-written osu! global scores JSON "
                         "(only read when --leaderboard-source osu)")
    ap.add_argument("--no-fail-animation", dest="no_fail_animation",
                    action="store_true", default=False,
                    help="debug escape hatch: ignore the .osr's death point "
                         "and render a FAILED replay as a pass (no fall/red "
                         "sequence, the replay's own grade instead of F)")
    ap.add_argument("--results-style", choices=("lazer", "r3d"),
                    default="lazer",
                    help="outro: 'lazer' = the ported osu!(lazer) ranking "
                         "screen (score panel → expanded statistics; std "
                         "default), 'r3d' = the in-house shared card")
    ap.add_argument("--letterbox-breaks", action=BA, default=True)
    ap.add_argument("--approach-circles", action=BA, default=True)
    ap.add_argument("--combo-numbers", action=BA, default=True)
    ap.add_argument("--follow-points", action=BA, default=True)
    ap.add_argument("--hit-animations", action=BA, default=True,
                    help="lazer hit-explosion (scale+fade) on hit; "
                         "--no-hit-animations = fast fade-out, no pop (#38371)")
    ap.add_argument("--snaking-in", action=BA, default=True)
    ap.add_argument("--snaking-out", action=BA, default=True)
    ap.add_argument("--slider-merge", action=BA, default=False,
                    help="§4.9 SliderMerge: draw all visible slider bodies "
                         "as ONE union pass (shared borders, no stacking)")
    ap.add_argument("--cursor", action=BA, default=True)
    ap.add_argument("--skin-cursor", action=BA, default=True,
                    help="use the skin's cursor when the skin ships one "
                         "(real osu has no toggle — this is the default; "
                         "--no-skin-cursor forces the procedural cursor)")
    ap.add_argument("--cursor-scale", type=float, default=1.0)
    ap.add_argument("--cursor-trail-scale", type=float, default=1.0,
                    help="§4.8 TrailScale: trail sprite size only")
    ap.add_argument("--cursor-rainbow", action=BA, default=False,
                    help="§4.8 rainbow: hue-cycle the cursor+trail tint")
    ap.add_argument("--cursor-ripples", action=BA, default=False,
                    help="§4.8 ripples: expanding ring per press edge")
    ap.add_argument("--force-long-trail", action=BA, default=False,
                    help="§4.7 ForceLongTrail: long connected skin trail "
                         "even without cursormiddle")
    ap.add_argument("--key-overlay", action=BA, default=True)
    ap.add_argument("--pp-counter", action=BA, default=False,
                    help="§4.6 PPCounter: live rosu-pp gradual pp (site "
                         "default OFF; hides itself if rosu is missing)")
    ap.add_argument("--pp", type=float, default=None,
                    help="EXACT final pp to show (osu's OFFICIAL pp). The "
                         "results card PP + the live counter's ENDPOINT are "
                         "pinned to this; the live curve keeps its rosu "
                         "shape. Omit to keep the rosu estimate.")
    ap.add_argument("--sr", type=float, default=None,
                    help="EXACT star rating to show (osu's OFFICIAL SR). The "
                         "results card's star-rating pill is pinned to this. "
                         "Omit to keep the rosu SR estimate.")
    ap.add_argument("--hit-counter", action=BA, default=False)
    ap.add_argument("--aim-error-meter", action=BA, default=False,
                    help="§4.6 AimErrorMeter: cursor-offset-at-click "
                         "scatter panel (site default OFF)")
    ap.add_argument("--strain-graph", action=BA, default=False,
                    help="§4.6 StrainGraph: bottom strain fill graph "
                         "(rosu strains, else the density proxy)")
    ap.add_argument("--scoreboard", action=BA, default=True,
                    help="§4.6 ScoreBoard: ACCEPTED + NO-OP — needs the "
                         "osu!API leaderboard hand-off (render/"
                         "scoreboard.py documents the JSON contract)")
    ap.add_argument("--scoreboard-avatars", action=BA, default=False,
                    help="ACCEPTED + NO-OP (rides the scoreboard hand-off)")
    ap.add_argument("--warning-arrows", action=BA, default=True,
                    help="§4.6 ShowWarningArrows: flashing arrows before "
                         "gameplay resumes after a break")
    ap.add_argument("--hit-error-meter", action=BA, default=True)
    ap.add_argument("--unstable-rate", action=BA, default=True)
    ap.add_argument("--show-combo", action=BA, default=True)
    ap.add_argument("--show-score", action=BA, default=True)
    ap.add_argument("--show-hp", action=BA, default=True)
    ap.add_argument("--show-grade", action=BA, default=True)
    ap.add_argument("--show-mods", action=BA, default=True)
    ap.add_argument("--show-progress", action=BA, default=False)  # R3D: pie off by default (clashes w/ modern skins; Red 2026-07-21)
    ap.add_argument("--progress-style", choices=("pie", "bar"), default="pie")
    ap.add_argument("--hud-scale", type=float, default=1.0)
    ap.add_argument("--hud-opacity", type=float, default=1.0)
    ap.add_argument("--break-flash", action=BA, default=True,
                    help="red edge-vignette pulse on combo breaks")
    ap.add_argument("--hit-lighting", action=BA, default=True,
                    help="§3.3 combo-tinted lighting flash under non-miss "
                         "judgments (R3D preset default ON)")
    ap.add_argument("--miss-fall", action=BA, default=True,
                    help="classic miss animation: the hit0 popup falls + "
                         "slightly rotates while it fades (stable's look; "
                         "owner default ON)")
    ap.add_argument("--renderer-default-font-and-ranks", action=BA,
                    default=False,
                    help="force the renderer's default (Argon) HUD numbers "
                         "+ procedural rank text even when the skin ships "
                         "fonts / ranking-* images (appearance only; "
                         "preserves the selected HUD layout)")
    ap.add_argument("--legacy-defaults", action=BA, default=False,
                    help="the ALL-LEGACY look: every HUD element uses its "
                         "legacy component (skin textures where shipped, "
                         "classic lg_* bakes otherwise — no Argon anywhere, "
                         "even skinless) and missing hitsound samples "
                         "synthesize in the LEGACY sound family; gameplay "
                         "elements already fall back skin-else-legacy")
    ap.add_argument("--playfield-borders",
                    choices=("none", "edges", "full"), default="none",
                    help="subtle white outline of the playfield bounds: "
                         "'full' = thin border box, 'edges' = corner "
                         "markers only")
    ap.add_argument("--featured-avatar-png", type=Path, default=None,
                    help="PNG of the FEATURED (current) player's REAL osu! "
                         "avatar for the results CENTRE card. Set -> that "
                         "image is shown; absent -> the procedural username "
                         "chip. The service passes the player's osu! pfp; the "
                         "old render-DB Discord-id lookup is GONE, so the "
                         "featured card can never show the owner's pfp.")
    ap.add_argument("--watermark", default="")
    ap.add_argument("--music-volume", type=int, default=100)
    ap.add_argument("--hitsound-volume", type=int, default=100)
    ap.add_argument("--hitsounds", action=BA, default=True,
                    help="mix §3.4 hitsounds into the audio track — the "
                         "use_replay_hitsounds preset key (one-shots at "
                         "judged hit times, slide/spin loops)")
    ap.add_argument("--nightcore-hitsounds", action=BA, default=False,
                    help="§4.4 PlayNightcoreSamples: clap each beat + "
                         "finish each bar downbeat (red timing points)")
    ap.add_argument("--use-skin-hitsounds", action=BA, default=False,
                    help="ignore beatmap-folder samples "
                         "(Audio.IgnoreBeatmapSamples)")
    ap.add_argument("--general-volume", type=int, default=100)
    ap.add_argument("--audio-offset", type=int, default=0, help="ms; -earlier")
    ap.add_argument("--bg-dim-intro", type=int, default=0)
    ap.add_argument("--bg-dim-game", type=int, default=90)
    ap.add_argument("--bg-dim-breaks", type=int, default=30)
    ap.add_argument("--bg-dim", type=int, default=None,
                    help="override --bg-dim-game (gameplay dim, 0-100)")
    ap.add_argument("--bg-blur", type=int, default=0,
                    help="§4.10 Blur 0-10: gaussian on the bg at load")
    ap.add_argument("--bg-parallax", action=BA, default=False,
                    help="§4.10 Parallax: bg ~1.02×, sliding opposite "
                         "the cursor")
    ap.add_argument("--bg-triangles", action=BA, default=False,
                    help="the osu triangles deco drifting up over the bg")
    ap.add_argument("--storyboard", action=BA, default=False,
                    help="§4.10 LoadStoryboards: parse the .osu/.osb and "
                         "render the storyboard (in-house engine); free "
                         "tier gates it off service-side")
    ap.add_argument("--video", action=BA, default=False,
                    help="§4.10 LoadVideos: play the map's [Events] Video "
                         "behind gameplay (fail-soft to the bg image)")
    ap.add_argument("--flash-to-beat", action=BA, default=False,
                    help="§4.10 FlashToTheBeat: subtle bg brightness "
                         "pulse on each red-line beat")
    ap.add_argument("--seizure-warning", action=BA, default=False,
                    help="§4.10 SeizureWarning: 5 s dark warning card "
                         "pre-roll at render start")
    ap.add_argument("--bloom", action=BA, default=False,
                    help="§4.10 Bloom: post-process bloom on the gameplay "
                         "layer (bright-pass + blur + additive; GPU cost)")
    ap.add_argument("--bloom-to-beat", action=BA, default=True,
                    help="pulse the bloom strength on beats (with --bloom)")
    ap.add_argument("--logo", action=BA, default=False,
                    help="show_logo: the R3D 'R' tile splash during the "
                         "intro, fading out as gameplay starts")
    ap.add_argument("--combo-colors",
                    choices=("skin", "beatmap", "beatmap-force"),
                    default="skin",
                    help="skin_combo_colors preset key: 'skin' = skin.ini "
                         "Combo1.. (default); 'beatmap' = the .osu "
                         "[Colours] when the map defines them UNLESS the "
                         "loaded USER skin ships its own Combo1.. — then "
                         "the skin wins (lazer's rule); 'beatmap-force' = "
                         "the old blunt rule (map [Colours] always)")
    ap.add_argument("--results-seconds", type=float, default=None)
    ap.add_argument("--no-replay", action="store_true",
                    help="render without a replay: perfect play at object "
                         "times, auto-spun spinners (§2.5 RPMS); pass the "
                         "beatmap as the only positional (debug: no "
                         "cursor/HUD/judgments — and thus no hitsounds)")
    ap.add_argument("--parse-only", action="store_true",
                    help="parse map+replay+skin, print a summary, exit 0")
    ap.add_argument("--start", type=float, default=None,
                    help="start the render this many seconds into the map")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop the render this many seconds in (debug/MVP)")
    ap.add_argument("--dump-frames", type=_ms_list, default=None,
                    metavar="MS,MS,…",
                    help="write single-frame PNGs at these map times "
                         "instead of encoding video")
    ap.add_argument("--add-mods", type=_mod_list, default=0, metavar="HD,FL",
                    help="force extra VISUAL mods (HD/FL) on top of the "
                         "replay's mods — for proofs when no replay has them")
    ap.add_argument("--force-lazer-mods", default="", metavar="BR,NS,FL",
                    help="R3D DEBUG: merge these lazer mod acronyms into the "
                         "replay's mod list (sets R3D_FORCE_LAZER_MODS, read by "
                         "the single read_lazer_mods funnel) so a fixture can "
                         "PRICE otherwise-unreachable mod paths — visual/transform "
                         "mods especially (BR/NS/WG/...). Off unless set; "
                         "configurable mods like DA still need a real replay.")
    return ap


def find_osu_file(beatmap: Path, replay_md5: str) -> Path:
    """A direct .osu path, or the md5-matching .osu inside the beatmap dir
    (cached sets hold every diff — the replay's beatmap hash picks the
    right one; first-sorted is only the no-hash fallback)."""
    if beatmap.is_file():
        return beatmap
    candidates = sorted(beatmap.glob("*.osu"))
    if not candidates:
        raise FileNotFoundError(f"no .osu in {beatmap}")
    if replay_md5:
        for cand in candidates:
            if hashlib.md5(cand.read_bytes()).hexdigest() == replay_md5.lower():
                return cand
    return candidates[0]


# the R3D render DB — read-only source for the LOCAL-ONLY PB card (owner
# decision: no osu!API). Overridable for tests/deployments.
PB_DB_PATH = os.environ.get("R3D_RENDER_DB",
                            "/home/red/.local/state/mania-ordr/db.sqlite")
LAZER_RESULTS_MIN_SECONDS = 4.5   # floor so both stages + a hold fit


def _header_is_standardised(meta) -> bool:
    """True when the .osr header 'score' is ALREADY the lazer-standardised
    (ScoreV3) total: a LAZER replay (game_version >= StdRuleset.LAZER_GAME_VERSION
    = 30_000_000) or a stable ScoreV2-mod play (bit 1<<29, stored 1M-standardised).
    A plain STABLE header is legacy ScoreV1 (a different scoring model) -> False;
    the displayed ScoreV3 then comes from the sim's own lazer-standardised
    trajectory (ruleset _build_events, counts reconciled to the .osr) instead of
    the raw header (#115, parity with the mania/catch renderers)."""
    if meta is None:
        return False
    return (int(getattr(meta, "game_version", 0) or 0) >= 30_000_000
            or bool(int(getattr(meta, "mods", 0) or 0) & (1 << 29)))


def _display_acc_pct(meta, judgments) -> float:
    """The results screen's accuracy percentage. Stable replays (and CL —
    the Classic mod is an explicit stable-semantics request): the
    .osr-header round(acc·100, 2) as before. LAZER replays with a
    ScoreInfo blob: lazer's canonical tick/tail-inclusive accuracy,
    TRUNCATED the way lazer's FormatAccuracy does (floor to 2 display
    decimals) — the exact digits the player saw in-game and the value the
    site card's DB row carries (forum bug 11 item 1: 98.88% shown for a
    play lazer called 98.91%)."""
    if (judgments is not None and getattr(judgments, "lazer", False)
            and getattr(meta, "lazer_accuracy", None) is not None):
        return math.floor(meta.lazer_accuracy * 10000.0) / 100.0
    return meta.accuracy


def _spinner_stats(meta, judgments) -> tuple[tuple[int, int],
                                             tuple[int, int]] | None:
    """((spins, max_spins), (bonus, max_bonus)) for the results screens'
    SPINNER SPIN / SPINNER BONUS rows, or None to omit them. Data exists
    only in a lazer .osr's ScoreInfo blob (small_bonus = spinner spins,
    large_bonus = bonus spins — OsuRuleset's display names); a stable
    .osr carries no spin statistics, so stable renders keep the previous
    screen exactly (honest gap, same rule as the PP line). Only shown
    when the map actually has spinners (a zero max on both = no
    spinners → None)."""
    st = getattr(meta, "lazer_statistics", None) if meta is not None else None
    if st is None:
        return None
    if (getattr(st, "max_small_bonus", 0) <= 0
            and getattr(st, "max_large_bonus", 0) <= 0):
        return None
    return ((st.small_bonus, st.max_small_bonus),
            (st.large_bonus, st.max_large_bonus))


def _build_lazer_results(spr, settings, beatmap, meta, judgments, hud, fv,
                         frames, osu_path, args, speed, frozen=None):
    """Assemble the lazer ranking screen (render/lazer_results.py) — gather
    stars/pp/perf-breakdown/slider-stats/aim-scatter/PB-card data, size the
    outro to fit both stages. Returns (screen, duration_wall_ms).

    `frozen` (a FAIL): a dict with grade/counts/max_combo/acc_pct/score
    tallied AT the death point — overrides the .osr's full-map totals, and
    pp/perf are dropped (meaningless for a partial play)."""
    from .render.hud import build_aim_points
    from .render.lazer_results import (LazerResultsScreen, ResultsData,
                                       query_pb, slider_stats)
    from .render.leaderboard import (build_board, query_leaderboard,
                                     rows_from_osu_json)
    from .render.pp import build_performance_breakdown, star_rating

    is_fail = frozen is not None
    if is_fail:
        counts = frozen["counts"]
        grade = frozen["grade"]
        acc_pct = frozen["acc_pct"]
        score = frozen["score"]
        max_combo = frozen["max_combo"]
    else:
        counts = (meta.count_300, meta.count_100, meta.count_50,
                  meta.count_miss)
        grade = meta.grade
        acc_pct = _display_acc_pct(meta, judgments)
        # #115: the standardised ScoreV3 endpoint, not the raw ScoreV1 header.
        score = (int(meta.score) if _header_is_standardised(meta)
                 else int(fv["score"] or meta.score))
        max_combo = meta.max_combo
    stars = star_rating(osu_path, meta.mods)
    # --sr: pin the results-card star-rating pill to the EXACT official SR
    # passed via --sr, overriding the rosu estimate. Static display value (no
    # live SR counter). None -> unchanged rosu SR. Mirrors the --pp override.
    if getattr(settings, "sr_override", None) is not None:
        stars = float(settings.sr_override)
    # pp/perf are pass-only: a failed play never earns pp
    if is_fail:
        perf = None
    else:
        perf = build_performance_breakdown(osu_path, meta.mods, judgments,
                                           counts, judgments.final_max_combo)
    pp_val = perf.achieved_pp if perf is not None else None
    # --pp: pin the results-card PP (and the "Achieved ...pp" subtitle) to the
    # EXACT official value passed via --pp. Pass-only: a failed play never
    # earns pp, so the override is ignored on a fail (same rule as rosu above).
    # The Maximum-pp figure and the aim/speed/acc breakdown bars stay rosu.
    if not is_fail and getattr(settings, "pp_override", None) is not None:
        pp_val = float(settings.pp_override)
        if perf is not None:
            perf.achieved_pp = pp_val
    tick_hit, tick_total, end_hit, end_total = slider_stats(
        judgments, before=frozen["fail_time"] if is_fail else None)
    aim_points = build_aim_points(judgments, frames,
                                  beatmap.diff.circle_radius)
    # PB card: this player's best PREVIOUS render of the map (exclude the
    # current replay by its md5), local DB, read-only. None → card omitted.
    replay_md5 = ""
    try:
        if args.osr is not None:
            replay_md5 = hashlib.md5(Path(args.osr).read_bytes()).hexdigest()
    except Exception:  # noqa: BLE001
        replay_md5 = ""
    pb = query_pb(PB_DB_PATH, meta.player_name, meta.beatmap_md5, replay_md5)
    if pb is not None:
        print(f"pb:     {pb['player_name']} {int(pb['score']):,} "
              f"{pb['grade']} (render DB) — PB card shown", file=sys.stderr)
    else:
        print("pb:     no prior render of this map for the player — PB card "
              "omitted", file=sys.stderr)

    # per-map render leaderboard: best-per-player OTHER renders of this map,
    # ranked around the current play. Off → None (the single PB card path
    # above stays exactly as it was). Local render DB, read-only, fail-soft.
    cur_score = (int(meta.score) if _header_is_standardised(meta)
                 else int(fv["score"] or meta.score))
    board = None
    if settings.show_leaderboard:
        prev_best = int(pb["score"]) if pb is not None else None
        # osu! GLOBAL source (opt-in): the bot hands us the map's osu top scores
        # as JSON (avatars pre-fetched). Fail-soft — a missing/empty/invalid file
        # falls back to the render DB so the default r3d path is never disturbed.
        osu_rows = None
        if settings.leaderboard_source == "osu":
            import json as _json
            try:
                _raw = (_json.loads(Path(settings.leaderboard_json).read_text(
                    encoding="utf-8")) if settings.leaderboard_json else None)
                if _raw:
                    osu_rows = rows_from_osu_json(_raw)
            except Exception as _e:  # noqa: BLE001 — never kill a render over this
                print(f"board:  osu! JSON unreadable ({_e}) — falling back",
                      file=sys.stderr)
                osu_rows = None
        if osu_rows:
            board = build_board(osu_rows, meta.player_name, cur_score,
                                prev_best_score=prev_best, max_per_side=3)
            print(f"board:  source: osu!global ({len(osu_rows)} players) — "
                  f"#{board.rank}/{board.n_players}, {len(board.left)} left + "
                  f"{len(board.right)} right", file=sys.stderr)
        else:
            rows = query_leaderboard(PB_DB_PATH, meta.beatmap_md5, replay_md5)
            board = build_board(rows, meta.player_name, cur_score,
                                prev_best_score=prev_best, max_per_side=3)
            moment = f" [{board.moment}]" if board.moment else ""
            _src = ("R3D DB (fallback)"
                    if settings.leaderboard_source == "osu" else "render DB")
            print(f"board:  #{board.rank}/{board.n_players} on this map — "
                  f"{len(board.left)} left + {len(board.right)} right"
                  f"{moment} ({_src})", file=sys.stderr)

    # featured (centre) card avatar: the FEATURED (current) player's REAL osu!
    # avatar PNG, supplied by the service via --featured-avatar-png (it resolves
    # the osu! user -> avatar_url -> PNG). None -> the procedural username chip.
    # The old render-DB Discord-id lookup was REMOVED: for a player whose name
    # maps (stale/colliding) to a linked Discord account it could resolve to the
    # SITE OWNER's pfp on the featured card (bug 2026-07-11). Fail-soft read.
    featured_avatar_png = None
    _fap = getattr(args, "featured_avatar_png", None)
    if _fap is not None:
        try:
            featured_avatar_png = Path(_fap).read_bytes()
        except OSError as _e:
            print(f"avatar: featured avatar PNG unreadable ({_e}) -- "
                  "procedural chip", file=sys.stderr)
            featured_avatar_png = None
    if featured_avatar_png:
        print(f"avatar: featured player osu! avatar "
              f"({len(featured_avatar_png)}B) -- real osu! pfp",
              file=sys.stderr)
    else:
        print("avatar: no osu! avatar for the featured player -- "
              "procedural chip", file=sys.stderr)

    # SPINNER SPIN / SPINNER BONUS rows (lazer's results statistics for
    # spinner maps — forum bug 11 item 3); None (stable replay / no
    # spinners) keeps the panel byte-identical to the previous layout.
    spin_stats = _spinner_stats(meta, judgments)
    data = ResultsData(
        player=meta.player_name, grade=grade, acc_pct=acc_pct,
        score=score, max_combo=max_combo,
        counts=counts, title=beatmap.name, artist=beatmap.artist,
        diff_name=beatmap.difficulty_name, creator=beatmap.creator,
        mods=meta.mods, stars=stars, pp=pp_val,
        date_str=(meta.played_at or "—"), ur=fv["ur"],
        slider_ticks=(tick_hit, tick_total),
        slider_ends=(end_hit, end_total),
        spinner_spins=spin_stats[0] if spin_stats else None,
        spinner_bonus=spin_stats[1] if spin_stats else None,
        err_deltas=list(hud.data.err_deltas),
        windows=(hud.hw.great, hud.hw.ok, hud.hw.meh),
        aim_points=aim_points, perf=perf, pb=pb, leaderboard=board,
        featured_avatar_png=featured_avatar_png,
        lazer_mods=meta.lazer_mods, rate_override=meta.rate_override)
    dur_wall_ms = max(settings.results_screen_time,
                      LAZER_RESULTS_MIN_SECONDS) * 1000.0
    screen = LazerResultsScreen(spr, data, dur_wall_ms, speed=speed,
                                argon_font=settings.skin_dir is None)
    # the screen floors its own timeline to MIN_TOTAL_MS (5.4 s since the
    # lazer-exact 3 s sweep, deaa1ec) — budget the outro from the SAME
    # floored length or end_ms lands before the stage-2 unfold + hold end.
    # total_ms = max(dur_wall_ms, 5400) so this can only extend, never cut.
    return screen, screen.total_ms


def _render(args, settings: StdRenderSettings, beatmap, frames,
            beatmap_dir: Path, skin_info, judgments=None, meta=None,
            osu_path: Path | None = None) -> int:
    """The Phase-1 record path: scene → record loop → ffmpeg."""
    # GL-touching imports live here so --parse-only works GL-less
    from .record.audio import AudioError, AudioMixer, decode_to_pcm
    from .record.encode import (FfmpegPipe, build_ffmpeg_cmd,
                                preview_on_media_engine, probe_encoder)
    from .record.pipeline import RecordPipeline
    from .render.background import (blur_background, build_dim_envelope,
                                    cover_size, load_background)
    from .render.bloom import BloomPass
    from .render.effects import SEIZURE_DURATION_S
    from .render import gl as gl_mod
    from .render.gl import SpriteRenderer
    from .render.hud import StdHud, build_aim_points, build_mod_pills
    from .render.playfield import PlayfieldCamera
    from .render.scene import (FAIL_DURATION_MS, ScenePlayer, StdScene,
                               ssaa_internal_size)
    from .render.skin_elements import SkinElements
    from .render.slider_body import SliderBodyRenderer
    from .render.textures import TextureBank
    from .render.video_bg import VideoBackground
    from .skin.skin import Skin
    perf.mark("gl_imports_done")

    w, h = settings.resolution
    # Metal instead of OpenGL (macOS only). Same scene code, same frames to
    # ffmpeg; anything that goes wrong while setting it up falls back to the GL
    # renderer below, and a failure after that re-runs the job on GL
    # (__main__). Asked for by R3D_STD_METAL=1, or by the node's own Metal
    # switch R3D_METAL=1 (the one catch acts on); R3D_STD_METAL=0 keeps std on
    # GL whatever the node says, and so does R3D_STD_STOCK=1.
    metal = False
    if (perf.envflag("R3D_STD_METAL", perf.envflag("R3D_METAL"))
            and not perf.STOCK and sys.platform == "darwin"):
        try:
            from .render.metal.renderer import MetalSpriteRenderer
            from .render.metal.slider_body import MetalSliderBodyRenderer
            if settings.bloom:
                raise RuntimeError("bloom is not on the Metal path yet")
            if not gl_mod._GPU_YUV:
                raise RuntimeError("the Metal path needs R3D_STD_GPU_YUV on")
            spr = MetalSpriteRenderer(w, h)
            bodies = MetalSliderBodyRenderer(spr, w, h)
            metal = True
            perf.METAL_IN_USE = True
            print(f"metal:  {spr.core.device}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 - never fail a render over this
            print(f"metal:  unavailable ({e}); using OpenGL", file=sys.stderr)
    if not metal:
        spr = SpriteRenderer(w, h)
    perf.mark("setup:gl_spriterenderer", leaf=True)
    bank = TextureBank(spr)
    perf.mark("setup:gl_texbank")
    if not metal:
        bodies = SliderBodyRenderer(spr.ctx, w, h)
    cam = PlayfieldCamera(w, h)
    perf.mark("setup:gl_init_done")

    # FAIL: the death point in MAP ms (None = pass or --no-fail-animation).
    # Gameplay freezes here, the fail sequence plays, then the F results.
    fail_time = (meta.fail_time if meta is not None
                 and not args.no_fail_animation else None)

    perf.mark("setup:music_decode_early")
    # --- music decode, EARLY + OFF-THREAD (perf; byte-identical) -------------
    # decode_to_pcm (an ffmpeg subprocess with the inline loudnorm pass) is
    # the single largest setup cost (~3 s for a full track). Its inputs are
    # all bound at entry — beatmap.diff.speed IS the `speed` local bound
    # later (read-only in between), and `meta.has_rate_ramp` decides the
    # warp branch before the warp object exists — so the subprocess starts
    # here on a worker thread and the WHOLE remaining setup (skin,
    # background, textures, health/hud, results bake, scene build) overlaps
    # it, GIL-free. Collected where the pcm is consumed; same call, same
    # args, same bytes; an AudioError from the worker surfaces at .result()
    # inside the existing try. Skipped for --dump-frames (it never reaches
    # the audio section).
    _audio_afile = beatmap.get_audio_file(beatmap_dir)
    _audio_fut = None
    if _audio_afile is not None and not args.dump_frames:
        from concurrent.futures import ThreadPoolExecutor
        _audio_pool = ThreadPoolExecutor(max_workers=1)
        if meta is not None and meta.has_rate_ramp:
            # WU/WD decodes NATIVE (rate 1) and warps piecewise at collect
            _audio_fut = _audio_pool.submit(decode_to_pcm, _audio_afile,
                                            rate=1.0, loudnorm=True)
        else:
            _audio_fut = _audio_pool.submit(
                decode_to_pcm, _audio_afile, rate=beatmap.diff.speed,
                pitch=(meta is not None and meta.rate_pitch), loudnorm=True)
        _audio_pool.shutdown(wait=False)

    perf.mark("setup:real_skin_core")
    # --- real-skin core textures (per-element procedural fallback) ---------------
    skin_elems = None
    if settings.skin_dir is not None:
        skin = Skin(skin_dir=settings.skin_dir,
                    fallback_dir=settings.default_skin_dir)
        skin_elems = SkinElements(skin, spr)
        for line in skin_elems.report_lines():
            print(line, file=sys.stderr)

    perf.mark("setup:4_10_background")
    # --- §4.10 background + dim envelope (fail-soft to the dark void) -------------
    bg_key = bg_size = dim_env = None
    bg_path = beatmap.get_related_file(beatmap_dir, beatmap.bg)
    if bg_path is not None:
        rgba = load_background(bg_path)
        if rgba is not None:
            if settings.bg_blur > 0:      # §4.10 Blur — once, at load
                rgba = blur_background(rgba, settings.bg_blur)
            spr.upload_texture("background", rgba)
            bg_key = "background"
            bg_size = cover_size(w, h, rgba.shape[1], rgba.shape[0])
        else:
            print(f"WARNING: background '{beatmap.bg}' failed to decode — "
                  "rendering without background", file=sys.stderr)
    elif beatmap.bg:
        print(f"WARNING: background '{beatmap.bg}' not found in the beatmap "
              "dir — rendering without background", file=sys.stderr)

    # dim envelope applies to image AND video backgrounds
    dim_env = build_dim_envelope(
        settings.bg_dim_intro / 100.0, settings.bg_dim_game / 100.0,
        settings.bg_dim_breaks / 100.0,
        [o.get_start_time() for o in beatmap.hit_objects],
        beatmap.diff.preempt, beatmap.pauses)

    perf.mark("setup:4_10_loadvideos")
    # --- §4.10 LoadVideos (render/video_bg.py — the mania v2 port) ----------------
    video_bg = None
    if settings.load_video:
        if beatmap.video:
            vpath = beatmap.get_related_file(beatmap_dir, beatmap.video)
            if vpath is not None:
                video_bg = VideoBackground(vpath, width=w, height=h,
                                           fps=settings.fps,
                                           start_ms=beatmap.video_offset)
                print(f"video:  {beatmap.video} from "
                      f"{beatmap.video_offset} ms", file=sys.stderr)
            else:
                print(f"WARNING: video '{beatmap.video}' not found in the "
                      "beatmap dir — static background", file=sys.stderr)
        else:
            print("note: load_video on, but the map has no [Events] Video "
                  "— static background", file=sys.stderr)

    # the HUD needs the judgment stream; --dump-frames without a replay
    # sim just renders HUD-less (the Phase-1 fallback). An explicit skin
    # selects a Legacy HUD container, otherwise Argon (legacy_defaults also
    # selects Legacy). Missing assets keep the layout. HP drain and the
    # .osr score pin are wired here.
    hud = None
    health = None
    if judgments is not None:
        from .ruleset import HealthTimeline
        health = HealthTimeline(beatmap, judgments)
        print(f"health: drain {health.drain_rate * 1000.0:.3f} hp/s "
              f"(target min {health.target_min:.2f}), "
              f"end {health.final_hp * 100.0:.0f}%", file=sys.stderr)

        # --- empty-lifebar death/quit fallback (see replay.detect_fail_time) --
        # detect_fail_time returns None when the .osr carries no life-bar graph
        # (the common case for these replays). Confirm a NON-completed play from
        # a hard fact instead: a completed play judges EVERY object, so the
        # replay's 300/100/50/miss sum equals the object count (the same
        # invariant reconcile trusts). judged < object-count => the player died
        # or quit, so FREEZE at their last input rather than auto-missing the
        # unreached tail (which tanks sim accuracy and false-trips the honesty
        # gate). A pass can NEVER judge fewer objects than the map holds, so this
        # cannot false-fail a pass. Fail-immune NoFail/Autopilot and
        # --no-fail-animation still render as a pass. Kept here (after judgments
        # is built) so the sim stays reconcile=ON — its slider-part combo
        # reconcile is what keeps the frozen max-combo exact.
        from .replay.replay import _FAIL_IMMUNE_MODS
        if (fail_time is None and not args.no_fail_animation
                and meta is not None and frames
                and not (meta.mods & _FAIL_IMMUNE_MODS)):
            _judged = (meta.count_300 + meta.count_100
                       + meta.count_50 + meta.count_miss)
            if 0 < _judged < len(beatmap.hit_objects):
                fail_time = float(frames[-1].time_ms)
                _how = "died (hp 0%)" if health.final_hp <= 1e-3 else "quit"
                print(f"fail:   no life-bar in .osr; {_how} — judged "
                      f"{_judged}/{len(beatmap.hit_objects)} objects; freezing "
                      f"at last input {fail_time / 1000.0:.2f}s (grade F)",
                      file=sys.stderr)

        # --- §4.6 pp counter / strain graph / aim error data (fail-soft) --
        pp_timeline = strain = None
        aim_points = None
        if settings.show_pp_counter:
            from .render.pp import build_pp_timeline
            res = build_pp_timeline(osu_path, meta.mods if meta else 0,
                                    judgments)
            if res is not None:
                pp_timeline, ppinfo = res
                # --pp: anchor the live-counter ENDPOINT to the EXACT
                # official value: scale every point by (override /
                # gradual_end) so the curve keeps its rosu SHAPE and only
                # the endpoint moves (mirror of the taiko _final_pp override).
                if (getattr(settings, "pp_override", None) is not None
                        and pp_timeline and ppinfo.gradual_end > 0.0):
                    _sc = float(settings.pp_override) / ppinfo.gradual_end
                    pp_timeline = [(t, p * _sc) for t, p in pp_timeline]
                delta = abs(ppinfo.gradual_end - ppinfo.full_calc)
                ok = "==" if delta < 0.05 else f"Δ{delta:.2f}"
                print(f"pp:     gradual end {ppinfo.gradual_end:.2f}pp "
                      f"(full calc {ppinfo.full_calc:.2f}pp {ok}, "
                      f"{ppinfo.n_points} points)", file=sys.stderr)
            else:
                print("pp:     unavailable — counter hidden (rosu-pp "
                      "missing or the map failed)", file=sys.stderr)
        if settings.show_strain_graph:
            from .render.pp import build_strain_series, strain_proxy
            starts = [o.get_start_time() for o in beatmap.hit_objects]
            ends = [o.get_end_time() for o in beatmap.hit_objects]
            strain = build_strain_series(osu_path,
                                         meta.mods if meta else 0,
                                         beatmap.diff.speed,
                                         min(starts) if starts else 0.0)
            if strain is None:
                strain = strain_proxy(starts, ends)
            if strain is not None:
                print(f"strain: {len(strain.values)} sections via "
                      f"{strain.source}", file=sys.stderr)
        if settings.show_aim_error_meter:
            aim_points = build_aim_points(judgments, frames,
                                          beatmap.diff.circle_radius)

        # full display mod set (lazer-only mods + custom-rate suffix) — falls
        # back to the legacy bitmask when there's no .osr / no ScoreInfo blob.
        mod_pills = (build_mod_pills(meta.mods, meta.lazer_mods,
                                     meta.rate_override)
                     if meta is not None else None)
        perf.mark("setup:hud_build")
        hud = StdHud(spr, bank, settings, judgments, frames, beatmap,
                     skin_elems=skin_elems, health=health,
                     mods=meta.mods if meta is not None else 0,
                     pp_timeline=pp_timeline, aim_points=aim_points,
                     strain=strain, mod_pills=mod_pills)
        # #115: DISPLAY the lazer-standardised ScoreV3, never the raw ScoreV1.
        # A lazer/ScoreV2 header IS already standardised -> pin the displayed
        # curve to it (endpoint-exact). A plain STABLE ScoreV1 header is a
        # different model -> do NOT pin; the sim's standardised trajectory
        # (matching the site score_v3) is shown instead. PASS only (a fail
        # freezes at the death tally already).
        if (meta is not None and meta.score > 0 and fail_time is None
                and _header_is_standardised(meta)):
            hud.pin_final_score(meta.score)

    perf.mark("setup:reds_results_screen")
    # --- RED'S results screen (render/results.py; §4.6 ShowResultsScreen) --------
    # fade_out_time is WALL seconds — × speed for map-ms (the same rate-mod
    # convention as results_screen_time; the pre-fix code forgot the ×speed)
    last_end = max(o.get_end_time() for o in beatmap.hit_objects)
    speed = beatmap.diff.speed
    fade_start_ms = last_end + HIT_FADE_OUT
    fade_len_ms = settings.fade_out_time * 1000.0 * speed
    gameplay_end_ms = fade_start_ms + fade_len_ms
    # FAIL: the sequence lasts FailAnimation.duration=2500 ms WALL → ×speed
    fail_anim_len_ms = FAIL_DURATION_MS * speed

    perf.mark("setup:wu_wd_rate")
    # --- WU/WD rate ramp (ModTimeRamp): build the wall<->map time warp ------------
    # A Wind Up / Wind Down replay ramps the clock rate LINEARLY across the map
    # (timewarp.TimeWarp). One warp drives the record clock, the audio warp and
    # every wall<->map conversion below. ``warp is None`` for every non-ramp
    # render, so the constant-`speed` lines stay exactly as they were
    # (byte-identical). Ramps are incompatible with DT/HT so base speed is 1.0;
    # the boundary spans (pre-roll before the first object = rate `initial`,
    # fade after 75% of the map = rate `final`) are constant-rate, so map_span
    # is exact there.
    warp = None
    if meta is not None and meta.has_rate_ramp:
        from .timewarp import build_time_warp
        _obj_starts = [o.get_start_time() for o in beatmap.hit_objects]
        warp = build_time_warp(meta.ramp_initial, meta.ramp_final,
                               min(_obj_starts), last_end, base_speed=speed)
        fade_len_ms = warp.map_span(fade_start_ms,
                                    settings.fade_out_time * 1000.0)
        gameplay_end_ms = fade_start_ms + fade_len_ms
        _fanchor = fail_time if fail_time is not None else last_end
        fail_anim_len_ms = warp.map_span(_fanchor, FAIL_DURATION_MS)

    # FAIL results: grade F + stats FROZEN at the death point (NOT the .osr's
    # full-map reconciled totals). Tally judgments up to fail_time from the
    # (un-reconciled) sim, peak combo up to death, accuracy from those counts.
    perf.mark("setup:frozen")
    frozen = None
    if fail_time is not None and hud is not None:
        fc = hud.data.counts_at(fail_time)
        ftot = sum(fc)
        facc = ((300 * fc[0] + 100 * fc[1] + 50 * fc[2]) / (300.0 * ftot)
                if ftot else 1.0)
        frozen = {
            "grade": "F",
            "counts": fc,
            "max_combo": hud.data.max_combo_upto(fail_time),
            "acc_pct": round(facc * 100.0, 2),
            "score": int(meta.score) if meta.score > 0
            else hud.data.score_upto(fail_time),
            "fail_time": fail_time,
        }
        print(f"fail:   frozen stats @ death — {fc[0]}/{fc[1]}/{fc[2]}/{fc[3]} "
              f"{frozen['acc_pct']}% combo {frozen['max_combo']}x "
              f"score {frozen['score']} grade F", file=sys.stderr)

    perf.mark("setup:results_build")
    results = results_start_ms = results_dur_wall_ms = None
    results_ssaa = None
    if settings.show_results and hud is not None and meta is not None:
        fv = hud.final_values()
        deltas = hud.data.err_deltas
        avg_ms = (sum(deltas) / len(deltas)) if deltas else 0.0
        # SSAA the results outro: bind the card to a >=1080p offscreen
        # renderer (shared GL context) and downscale each results frame to
        # the output res, so text stays crisp at sub-1080p outputs. No-op
        # at >=1080p (the card renders at output res as before).
        perf.mark("setup:results_ssaa")
        results_spr = spr
        iw, ih = ssaa_internal_size(w, h)
        if (iw, ih) != (w, h):
            results_spr = (MetalSpriteRenderer(iw, ih, core=spr.core) if metal
                           else SpriteRenderer(iw, ih, ctx=spr.ctx))
            results_ssaa = results_spr
            print(f"ssaa:   results supersampled at {iw}x{ih} → {w}x{h}",
                  file=sys.stderr)
        if settings.results_style == "lazer":
            # the outro sits past the map end where a WU/WD ramp is pinned at
            # `final`; the results screen converts its MAP age -> wall via this
            # rate (age_ms / speed), so hand it the end rate under a ramp.
            _results_speed = warp.final if warp is not None else speed
            perf.mark("setup:lazer_results", leaf=True)
            results, results_dur_wall_ms = _build_lazer_results(
                results_spr, settings, beatmap, meta, judgments, hud, fv,
                frames, osu_path, args, _results_speed, frozen=frozen)
        else:
            from .render.results import ResultsScreen
            r_counts = frozen["counts"] if frozen else (
                meta.count_300, meta.count_100, meta.count_50, meta.count_miss)
            r_grade = frozen["grade"] if frozen else meta.grade
            r_acc = (frozen["acc_pct"] if frozen
                     else _display_acc_pct(meta, judgments))
            r_score = (frozen["score"] if frozen
                       else (int(meta.score) if _header_is_standardised(meta)
                             else int(fv["score"] or meta.score)))
            r_combo = frozen["max_combo"] if frozen else meta.max_combo
            # a lazer replay shows the FULL mod set (incl. lazer-only mods +
            # custom-rate suffix) on the results subtitle; a pure-legacy
            # replay passes None so the bitmask string stays byte-identical.
            mods_display = (",".join(p.text for p in mod_pills)
                            if (meta is not None and meta.lazer_mods)
                            else None)
            results = ResultsScreen(
                results_spr, skin_elems,
                counts=r_counts,
                acc_pct=r_acc, score=r_score,
                max_combo=r_combo, grade=r_grade, ur=fv["ur"],
                avg_ms=avg_ms, err_deltas=deltas, meh_ms=hud.hw.meh,
                player=meta.player_name,
                map_line=f"{beatmap.artist} - {beatmap.name}",
                diff_name=beatmap.difficulty_name, mods=meta.mods,
                use_skin_ranks=not settings.renderer_default_font_and_ranks,
                argon_font=settings.skin_dir is None,
                mods_display=mods_display,
                spinner_stats=_spinner_stats(meta, judgments))
            results.set_windows(hud.hw.great, hud.hw.ok)
            results_dur_wall_ms = settings.results_screen_time * 1000.0
        # PASS → results after the map-end fade; FAIL → after the fall
        results_start_ms = (fail_time + fail_anim_len_ms
                            if fail_time is not None else gameplay_end_ms)

    # §3.5/§4.7 combo colour source — lazer precedence. "beatmap" mode:
    # the .osu [Colours] win UNLESS the loaded USER skin ships its own
    # Combo1..N in skin.ini — then the skin's palette wins (what osu!
    # really does: a skin that defines colours keeps them even with
    # beatmap colours enabled). The bundled service default skin
    # ("_default-source", Night05) DOES define Combo1..4 but is nobody's
    # choice, so it never blocks the map's palette. "beatmap-force"
    # restores the old blunt rule; "skin" still forces skin.ini.
    _skin_ships_colours = (
        skin_info.combo_colors_custom
        and settings.skin_dir is not None
        and Path(settings.skin_dir).name != "_default-source")
    color_src = skin_info.combo_colors
    combo_colors_from_beatmap = False
    if (settings.use_beatmap_colors and beatmap.combo_colors
            and (settings.beatmap_colors_forced or not _skin_ships_colours)):
        color_src = beatmap.combo_colors
        combo_colors_from_beatmap = True
        print(f"colors: beatmap [Colours] ({len(color_src)} combo colours)",
              file=sys.stderr)
    elif settings.use_beatmap_colors and _skin_ships_colours:
        print(f"colors: skin.ini Combo1..{len(color_src)} — user skin ships "
              "its own [Colours], wins over the beatmap's (lazer rule)",
              file=sys.stderr)
    combo_colors = [(r / 255.0, g / 255.0, b / 255.0)
                    for r, g, b in color_src]
    track_override = None
    if skin_info.slider_track_override is not None:
        track_override = tuple(c / 255.0
                               for c in skin_info.slider_track_override)

    perf.mark("setup:flow_end_start")
    # --- flow: end/start + the §4.10 pre-roll (lead-in + seizure card) ------------
    # FAIL: gameplay ends at the death frame + the fall, never at map end
    end_ms = (fail_time + fail_anim_len_ms if fail_time is not None
              else gameplay_end_ms)
    if results is not None:
        # results duration is WALL ms — scale by the rate mod so the outro
        # holds the same real time under DT/HT (the lazer style floors the
        # duration so both stages + a hold always fit — see _build_lazer).
        # Under WU/WD the results sit past the map end (rate pinned `final`).
        if warp is not None:
            end_ms = results_start_ms + warp.map_span(results_start_ms,
                                                      results_dur_wall_ms)
        else:
            end_ms = results_start_ms + results_dur_wall_ms * speed
    if args.max_seconds is not None:
        end_ms = min(end_ms, args.max_seconds * 1000.0)
    start_ms = (args.start or 0.0) * 1000.0
    if start_ms and start_ms >= end_ms:
        print(f"error: --start {args.start:g}s is at/after the render end "
              f"({end_ms / 1000.0:.1f}s)", file=sys.stderr)
        return 2
    # lead_in_time and the seizure card are WALL seconds of extra pre-roll
    # BEFORE start_ms: the map clock simply begins earlier — objects
    # can't spawn, the dim envelope holds the intro level, music is laid
    # correspondingly later into the wall timeline
    if warp is not None:
        # pre-roll is defined in WALL seconds; the map clock begins earlier by
        # the ramp-correct amount. The pre-roll lands before the first object
        # (rate = `initial`) for a map-start render, exact via the warp.
        seizure_wall = (SEIZURE_DURATION_S * 1000.0
                        if settings.seizure_warning else 0.0)
        lead_wall = settings.lead_in_time * 1000.0
        _start_wall = warp.to_wall(start_ms)
        render_start_ms = warp.to_map(_start_wall - seizure_wall - lead_wall)
        _post_seizure = warp.to_map(_start_wall - lead_wall)
        seizure_ms = _post_seizure - render_start_ms   # map span of seizure
        lead_ms = start_ms - _post_seizure             # map span of lead-in
        if seizure_wall or lead_wall:
            print(f"lead:   {(seizure_wall + lead_wall) / 1000.0:.1f}s "
                  f"pre-roll (seizure {seizure_wall / 1000.0:.1f}s + "
                  f"lead-in {lead_wall / 1000.0:.1f}s)", file=sys.stderr)
    else:
        seizure_ms = (SEIZURE_DURATION_S * 1000.0 * speed
                      if settings.seizure_warning else 0.0)
        lead_ms = settings.lead_in_time * 1000.0 * speed
        render_start_ms = start_ms - seizure_ms - lead_ms
        if seizure_ms or lead_ms:
            print(f"lead:   {(seizure_ms + lead_ms) / speed / 1000.0:.1f}s "
                  f"pre-roll (seizure {seizure_ms / speed / 1000.0:.1f}s + "
                  f"lead-in {lead_ms / speed / 1000.0:.1f}s)", file=sys.stderr)

    bloom_pass = BloomPass(spr.ctx, w, h) if settings.bloom else None

    # mods driving the playfield visuals (HD fades / FL overlay): the
    # replay's mods plus any forced visual-only --add-mods
    scene_mods = (meta.mods if meta is not None else 0) | getattr(
        args, "add_mods", 0)

    perf.mark("setup:storyboard_phase_3")
    # --- storyboard (phase-3 renderer; auto-discovers the map .osb) -------------
    storyboard_renderer = None
    if settings.load_storyboard and osu_path is not None:
        from .beatmap.storyboard import parse_storyboard
        from .render.storyboard_engine import StoryboardEngine
        from .render.storyboard_render import StoryboardRenderer
        try:
            sb_data = parse_storyboard(osu_path)
            sb_engine = StoryboardEngine(sb_data)
            if sb_engine.sprites:
                storyboard_renderer = StoryboardRenderer(
                    spr, sb_engine, beatmap_dir, w, h, sb_data.widescreen)
                c = sb_data.counts()
                print(f"storyboard: {len(sb_engine.sprites)} drawable sprites"
                      f" ({c['sprites']} sprite / {c['animations']} anim,"
                      f" {c['commands']} cmds), widescreen={sb_data.widescreen}",
                      file=sys.stderr)
                if c['samples']:
                    print(f"storyboard: NOTE {c['samples']} sample event(s) NOT "
                          "played (storyboard audio deferred to a later phase)",
                          file=sys.stderr)
            else:
                print("storyboard: no drawable sprites — nothing to render",
                      file=sys.stderr)
        except Exception as e:  # noqa: BLE001 — never fail a render over the SB
            print(f"WARNING: storyboard load failed ({e!r}) — rendering "
                  "without storyboard", file=sys.stderr)

    perf.mark("setup:scene_build")
    scene = StdScene(
        beatmap, frames, cam, spr, bodies, bank,
        combo_colors=combo_colors,
        combo_colors_from_beatmap=combo_colors_from_beatmap,
        snaking_in=settings.slider_snaking_in,
        snaking_out=settings.slider_snaking_out,
        slider_merge=settings.slider_merge,
        draw_approach_circles=settings.draw_approach_circles,
        draw_combo_numbers=settings.draw_combo_numbers,
        draw_follow_points=settings.draw_follow_points,
        hit_animations=settings.hit_animations,
        draw_cursor=settings.draw_cursor,
        cursor_scale=settings.cursor_scale,
        cursor_trail_scale=settings.cursor_trail_scale,
        cursor_rainbow=settings.cursor_rainbow,
        cursor_ripples=settings.cursor_ripples,
        force_long_trail=settings.cursor_long_trail,
        judgments=judgments,
        draw_hit_lighting=settings.show_hit_lighting,
        spinner_fade_playfield=skin_info.spinner_fade_playfield,
        spinner_no_blink=skin_info.spinner_no_blink,
        hud=hud,
        skin_elems=skin_elems,
        use_skin_cursor=settings.use_skin_cursor,
        border_color=tuple(c / 255.0 for c in skin_info.slider_border),
        track_override=track_override,
        bg_key=bg_key,
        bg_draw_size=bg_size,
        dim_envelope=dim_env,
        video_bg=video_bg,
        bg_parallax=settings.bg_parallax,
        bg_triangles=settings.bg_triangles,
        flash_to_beat=settings.flash_to_beat,
        show_warning_arrows=settings.show_warning_arrows,
        fade_start_ms=fade_start_ms,
        fade_len_ms=fade_len_ms,
        logo_start_ms=(render_start_ms + seizure_ms
                       if settings.show_logo else None),
        seizure_start_ms=(render_start_ms if settings.seizure_warning
                          else None),
        bloom_pass=bloom_pass,
        bloom_to_beat=settings.bloom_to_beat,
        miss_fall=settings.miss_fall,
        playfield_borders=settings.playfield_borders,
        results=results,
        results_start_ms=results_start_ms,
        results_ssaa=results_ssaa,
        mods=scene_mods,
        transform_mod=(meta.transform_acronym if meta is not None else ""),
        transform_start_scale=(meta.transform_start_scale
                               if meta is not None else 1.0),
        transform_strength=(meta.transform_strength
                            if meta is not None else 1.0),
        freeze_frame=(meta.freeze_frame if meta is not None else False),
        traceable=(meta.traceable if meta is not None else False),
        approach_scale=(meta.approach_scale if meta is not None else None),
        approach_style=(meta.approach_style if meta is not None else ""),
        barrel_roll=(meta.barrel_roll if meta is not None else None),
        bloom=(meta.bloom if meta is not None else None),
        synesthesia=(meta.synesthesia if meta is not None else False),
        blinds=(meta.blinds if meta is not None else False),
        no_scope=(meta.no_scope if meta is not None else None),
        depth=(meta.depth if meta is not None else None),
        bubbles=(meta.bubbles if meta is not None else False),
        repel_magnet=(meta.repel_magnet_acronym if meta is not None else ""),
        repel_magnet_strength=(meta.repel_magnet_strength
                               if meta is not None else 0.5),
        health=health,
        fail_time_ms=fail_time,
        fail_anim_len_ms=fail_anim_len_ms,
        storyboard=storyboard_renderer,
    )

    perf.mark("setup:keyframe_dump_mode")
    # --- keyframe dump mode -----------------------------------------------------
    if args.dump_frames:
        from PIL import Image
        out_dir = args.output.parent if args.output else Path.cwd()
        stem = args.output.stem if args.output else "frame"
        out_dir.mkdir(parents=True, exist_ok=True)
        for t in sorted(args.dump_frames):
            rgb = scene.frame_rgb(t)
            p = out_dir / f"{stem}_t{int(round(t))}ms.png"
            Image.fromarray(rgb).save(p)
            print(f"wrote {p}", file=sys.stderr)
        _print_hud_final_values(hud, judgments, meta, frozen=frozen)
        if video_bg is not None:
            video_bg.close()
        spr.release()
        return 0

    if args.output is None:
        print("error: -o/--output is required to render", file=sys.stderr)
        return 2
    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    perf.mark("setup:offline_audio_music")
    # --- offline audio: music bed + §3.4 hitsounds (NO-BASS design) ---------------
    # map-ms -> render-relative wall-ms. For a ramp this is the exact warp; with
    # no ramp it is the ORIGINAL constant expression (byte-identical audio).
    if warp is not None:
        _rswall = warp.to_wall(render_start_ms)

        def m2w(m: float) -> float:
            return warp.to_wall(m) - _rswall
    else:
        def m2w(m: float) -> float:
            return (m - render_start_ms) / speed

    def _mix_audio(_amark=perf.mark):
        """The offline mix: music bed + hitsounds, written to a wav. Returns
        its path, or None for a silent render. A function only so that
        R3D_STD_AUDIO_LATE can run it beside the frame loop; otherwise it is
        called right here, as the inline code it always was."""
        audio_path = None
        _amark("aud:mixer")
        mixer = AudioMixer(m2w(end_ms))
        have_audio = False
        afile = _audio_afile
        if afile is not None:
            try:
                vol = ((settings.music_volume / 100.0)
                       * (settings.general_volume / 100.0))
                if warp is not None:
                    # WU/WD: the rate ramps, so a single atempo/asetrate can't warp
                    # the track — decode NATIVE and warp piecewise (record/audio.
                    # warp_music_pcm). adjust_pitch True (WU/WD default) shifts
                    # pitch with the rate; False keeps pitch (tempo-only).
                    from .record.audio import warp_music_pcm
                    _amark("aud:pcm_join", leaf=True)
                    pcm = (_audio_fut.result() if _audio_fut is not None
                           else decode_to_pcm(afile, rate=1.0, loudnorm=True))
                    pcm = warp_music_pcm(pcm, warp,
                                         adjust_pitch=meta.ramp_pitch)
                else:
                    # NC/DC pitch the music with the rate; DT/HT (and every
                    # standard/bitmask rate, where rate_pitch is False) change
                    # tempo only. (Decoded off-thread above — same args.)
                    _amark("aud:pcm_join", leaf=True)
                    pcm = (_audio_fut.result() if _audio_fut is not None
                           else decode_to_pcm(
                               afile, rate=speed,
                               pitch=(meta is not None and meta.rate_pitch),
                               loudnorm=True))
                # the map-time render start lands at wall t=0: the (already
                # rate-adjusted / ramp-warped) music is laid at the wall position
                # of map time 0 — mix_at clips a negative head; a pre-roll delays
                # it instead
                _amark("aud:lay_music")
                mixer.lay_music(pcm, m2w(0.0), volume=vol)
                have_audio = True
            except AudioError as e:
                print(f"WARNING: music decode failed, mixing without the music "
                      f"bed: {e}", file=sys.stderr)
        else:
            print(f"WARNING: beatmap audio '{beatmap.audio}' not found — "
                  "mixing without the music bed", file=sys.stderr)

        # sample bank shared by judged hitsounds + the nightcore overlay(s);
        # the synth-default bank follows the visual league (skinless→argon,
        # custom-skin gaps→legacy, --legacy-defaults→legacy)
        # ModNightcore beat overlay is AUTOMATIC when the NC mod (bit 512) is on.
        _nc_mod = bool(int(getattr(meta, "mods", 0) or 0) & 512) if meta is not None else False
        sample_bank = None
        if settings.hitsound_volume > 0 and settings.general_volume > 0 and (
                (settings.use_replay_hitsounds and judgments is not None)
                or settings.nightcore_hitsounds or _nc_mod):
            from .record.hitsounds import SampleBank, synth_style_for
            from .skin.skin import Skin as SampleSkin
            sample_skin = SampleSkin(skin_dir=settings.skin_dir,
                                     fallback_dir=settings.default_skin_dir)
            _amark("aud:sample_bank")
            sample_bank = SampleBank(
                skin=sample_skin, beatmap_dir=beatmap_dir,
                use_beatmap_samples=not settings.use_skin_hitsounds,
                synth_style=synth_style_for(settings.skin_dir is not None,
                                            settings.legacy_defaults))
        # -8 LU hit ceiling (2026-07-31): std per-note hits had NO attenuation
        # (rode at full volume x1.0) — the loudest in-house engine. Scale them to
        # match the -18 LUFS music loudnorm drop (mirrors catch/mania 0.55->0.22).
        HIT_CEILING = 0.40
        hs_gain = (HIT_CEILING
                   * (settings.hitsound_volume / 100.0)
                   * (settings.general_volume / 100.0))

        # §3.4 hitsounds: one-shots at judged hit times + slide/spin loops,
        # resolved BEATMAP(custom index) → skin chain → synthesized defaults
        if (settings.use_replay_hitsounds and judgments is not None
                and sample_bank is not None):
            from .record.hitsounds import collect_hitsound_events, mix_hitsounds
            _amark("aud:collect_hs")
            oneshots, loops = collect_hitsound_events(
                beatmap, judgments, layered=skin_info.layered_hit_sounds)
            if fail_time is not None:
                # FAIL: objects after the death point were never played — drop
                # their hitsounds (loops clip at the death point)
                oneshots = [o for o in oneshots if o.time_ms < fail_time]
                loops = [l for l in loops if l.t0 < fail_time]
            _amark("aud:mix_hitsounds", leaf=True)
            stats = mix_hitsounds(mixer, sample_bank, oneshots, loops,
                                  speed=speed, start_ms=render_start_ms,
                                  gain=hs_gain,
                                  to_wall=(m2w if warp is not None else None))
            srcs = sample_bank.source_counts()
            _amark("aud:hs_report")
            print(f"hitsounds: {stats.oneshots} one-shots, "
                  f"{stats.loop_ms / 1000.0:.1f}s loops | samples: "
                  f"beatmap {srcs['beatmap']}, skin {srcs['skin']}, "
                  f"synth {srcs['synth']} ({sample_bank.synth_style} bank) | "
                  f"track peak "
                  f"{stats.peak_before:.2f}→{stats.peak_after:.2f}",
                  file=sys.stderr)
            have_audio = have_audio or stats.oneshots > 0 or stats.loop_ms > 0

        # Beat overlays stop at gameplay END, not into the fail-anim/results outro
        # (taiko fix ac73af2): on a FAIL gameplay ends at the death point, so cap the
        # overlay horizon at fail_time (the one-shot hits are already truncated there).
        # On a pass last_end (< the results screen) is already the correct bound.
        _overlay_end = fail_time if fail_time is not None else last_end

        # §4.4 general beat-overlay metronome: clap each beat + finish each downbeat
        # across [render start, gameplay end]. SUPPRESSED while NC is active (the NC
        # drum overlay below plays instead — osu! never plays both).
        if settings.nightcore_hitsounds and not _nc_mod and sample_bank is not None:
            from .record.hitsounds import mix_nightcore, nightcore_beats
            beats = nightcore_beats(beatmap.timings,
                                    max(render_start_ms, 0.0), _overlay_end)
            laid = mix_nightcore(mixer, sample_bank, beats, speed=speed,
                                 start_ms=render_start_ms, gain=hs_gain,
                                 to_wall=(m2w if warp is not None else None))
            downs = sum(1 for _, d in beats if d)
            print(f"nightcore: {laid} beats laid ({downs} downbeats)",
                  file=sys.stderr)
            have_audio = have_audio or laid > 0

        # ModNightcore beat overlay — AUTOMATIC when the Nightcore mod is active:
        # the kick(1,3)/clap(2,4)/hat(off-beats)/finish(every 4th bar) drum pattern
        # from the SKIN's nightcore-* samples on the (sped-up) beat grid. Distinct
        # from + independent of the nightcore_hitsounds general metronome above
        # (both lay if that toggle is on for an NC play). Hats gate on
        # SliderTickRate%2==0 (osu!).
        if _nc_mod and sample_bank is not None:
            from .record.hitsounds import mix_nightcore_mod, nightcore_mod_events
            _play_hats = (int(round(beatmap.timings.tick_rate)) % 2 == 0)
            nc_events = nightcore_mod_events(beatmap.timings,
                                             max(render_start_ms, 0.0), _overlay_end,
                                             play_hats=_play_hats)
            nc_laid = mix_nightcore_mod(mixer, sample_bank, nc_events, speed=speed,
                                        start_ms=render_start_ms, gain=hs_gain,
                                        to_wall=(m2w if warp is not None else None))
            print(f"nightcore-mod: {nc_laid} NC beat samples laid "
                  f"(hats {'on' if _play_hats else 'off'})", file=sys.stderr)
            have_audio = have_audio or nc_laid > 0

        # §4.10 pre-roll audio: the seizure card / lead-in region is SILENT
        # (danser LeadInTime semantics) — for a map-start render the region
        # is silent anyway; this also covers --start clips with a pre-roll
        if have_audio and (seizure_ms or lead_ms):
            _amark("aud:post_fx")
            mixer.silence_before(m2w(start_ms))

        # §4.10 FadeOutTime, audio side.
        if have_audio and fail_time is None:
            if results is None:
                # No outro: the video ends on the map-end fade, so the music
                # fades to black with it (§4.10 FadeOutTime — unchanged).
                if fade_len_ms > 0.0:
                    mixer.fade_out(m2w(fade_start_ms), m2w(gameplay_end_ms))
            else:
                # Results outro present: the song KEEPS PLAYING under the
                # results screen (osu!/lazer behaviour) up to its natural end,
                # with a short fade at the very end of the video so the cut is
                # clean (parity with mania v2's 600 ms tail fade).
                tail = m2w(end_ms)
                mixer.fade_out(max(0.0, tail - 600.0), tail)

        # FAIL audio: the FailAnimation bends the track frequency to 0 over the
        # 2500 ms fall (a slowdown + pitch drop). We can't pitch-bend offline
        # without BASS, so we APPROXIMATE with a linear music fade to silence
        # across the fall (honest gap: no pitch-bend), and play the synthesized
        # fail sample once at the death point (FailAnimation.failSample.Play).
        if fail_time is not None and settings.general_volume > 0:
            from .record.hitsounds import synth_failsound
            t0 = m2w(fail_time)
            t1 = m2w(fail_time + fail_anim_len_ms)
            if have_audio:
                mixer.fade_out(t0, t1)
            fs_vol = settings.general_volume / 100.0
            mixer.mix_at(t0, synth_failsound(), volume=fs_vol)
            have_audio = True
            print(f"fail:   music fades {t0 / 1000.0:.1f}→{t1 / 1000.0:.1f}s "
                  f"(wall) + fail sample at {t0 / 1000.0:.1f}s "
                  f"(pitch-bend approximated)", file=sys.stderr)

        if have_audio:
            _amark("aud:write_wav")
            audio_path = output.with_suffix(".audio.wav")
            mixer.write_wav(audio_path)
        else:
            print("WARNING: no audio mixed — rendering SILENT video",
                  file=sys.stderr)
        return audio_path

    # AUDIO OFF THE START (R3D_STD_AUDIO_LATE=1, default OFF): draw and encode
    # the video at once and prepare the audio beside the frame loop, instead
    # of holding the first frame until the song is decoded and mixed (18 s for
    # a 7.5 minute song this node has not rendered before). See
    # record/audio_late.py. The streamable master and the live preview are
    # uploaded while they are written, so they keep the stock order.
    audio_late = (perf.envflag("R3D_STD_AUDIO_LATE") and not perf.STOCK
                  and os.environ.get("R3D_STREAM_MASTER") != "1"
                  and os.environ.get("R3D_PREVIEW_LIVE") != "1"
                  and os.environ.get("R3D_STD_NULL_SINK") != "1"
                  and not settings.audio_offset)
    audio_path = None if audio_late else _mix_audio()

    total_wall_ms = m2w(end_ms)
    # INLINE PREVIEW (R3D_PREVIEW_INLINE=1, default OFF): have the SAME ffmpeg
    # that encodes the master also write the lean 720p30 preview embed as a
    # second output, so it is finished the moment the render is. Without it the
    # node re-encodes the finished master afterwards before anything can be
    # published. Single renders only (merge.py builds its own command).
    preview_path = None
    if os.environ.get("R3D_PREVIEW_INLINE") == "1":
        preview_path = output.parent / (output.stem + ".embed.mp4")
        print(f"[std] inline preview -> {preview_path.name}",
              file=sys.stderr, flush=True)
    # STREAMABLE MASTER (R3D_STREAM_MASTER=1, default OFF): write the master
    # front to back (no +faststart) with the final loudness pass already
    # applied, so the contributor client can upload it WHILE it renders. The
    # marker file tells the client this engine honoured the flag; without it
    # the client keeps its post-render loudness pass.
    # INLINE DISCORD COPY (R3D_COMPACT_INLINE=1, default OFF; needs the inline
    # preview): a third output encoded to the compact plan the node/bot use
    # for `-embed-sm.mp4`, so nothing is left to encode after the render.
    compact_path = None
    from .record.encode import compact_wanted
    if preview_path is not None and compact_wanted(
            total_wall_ms / 1000.0, w, h, settings.fps, 1.0):
        compact_path = output.parent / (output.stem + ".embed-sm.mp4")
        print(f"[std] inline discord copy -> {compact_path.name}",
              file=sys.stderr, flush=True)
    stream_master = os.environ.get("R3D_STREAM_MASTER") == "1"
    if stream_master:
        import json as _json
        (output.parent / (output.stem + ".stream.json")).write_text(_json.dumps(
            {"schema": 1, "faststart": False,
             "loudnorm": "loudnorm=I=-18:TP=-1.5:LRA=11",
             "compact": compact_path is not None}))
        print("[std] streamable master (no faststart, loudnorm in-engine)",
              file=sys.stderr, flush=True)
    perf.mark("aud:encoder_spawn")
    encoder = probe_encoder(settings.encoder, settings.encoder_device)
    _preview_hw = preview_path is not None and preview_on_media_engine()
    late = None
    if audio_late:
        from .record.audio_late import LateAudio, video_end_sample
        from .record.encode import compact_plan
        from .record.pipeline import predict_draw_times
        # the frame count is known before the first frame (the loop's own
        # arithmetic), and with it where `-shortest` would end each audio
        _n_pred = sum(1 for _ in predict_draw_times(
            settings.fps, render_start_ms, end_ms, speed,
            warp.rate_at if warp is not None else None))
        _pfps = min(30, int(round(float(settings.fps))))
        _ends = {"master": video_end_sample(_n_pred, settings.fps),
                 "preview": video_end_sample(_n_pred, settings.fps, _pfps)}
        if compact_path is not None:
            _ends["compact"] = video_end_sample(
                _n_pred, settings.fps,
                min(compact_plan(total_wall_ms / 1000.0)[3],
                    int(round(float(settings.fps)))))
        late = LateAudio(output=output, preview_path=preview_path,
                         compact_path=compact_path, mix=_mix_audio,
                         total_dur_s=total_wall_ms / 1000.0,
                         preview_lead=_preview_hw, loudnorm=False,
                         end_samples=_ends)
    cmd = build_ffmpeg_cmd(
        encoder=encoder, resolution=(w, h), fps=settings.fps,
        output_path=output if late is None else late.video_master,
        audio_path=audio_path,
        audio_offset_ms=settings.audio_offset, loudnorm=False,
        video_bitrate=settings.video_bitrate,
        encoder_device=settings.encoder_device,
        preview_path=preview_path if late is None else late.video_preview,
        total_dur_s=(total_wall_ms / 1000.0 if preview_path is not None
                     else None),
        pix_fmt="yuv420p" if gl_mod._GPU_YUV else "rgb24",
        stream_master=stream_master,
        compact_path=compact_path if late is None else late.video_compact,
        preview_hw=_preview_hw,
        faststart=late is None)

    last_pct = [-1]
    render_complete = False

    def progress(frac: float) -> None:
        pct = int(frac * 100)
        if late is not None and not render_complete:
            pct = min(pct, 99)
        if pct != last_pct[0]:
            last_pct[0] = pct
            print(f"\rrendering… {pct}%", end="", file=sys.stderr, flush=True)

    player = ScenePlayer(scene, end_ms, speed=speed,
                         start_ms=render_start_ms,
                         rate_fn=(warp.rate_at if warp is not None else None))
    if (results is not None and results_start_ms is not None
            and perf.envflag("R3D_STD_RESULTS_AHEAD", perf.FAST_DEFAULT)
            and hasattr(results, "prebake_ahead")):
        # Bake the outro's rolling score / stats / accuracy arc during
        # gameplay instead of on the render thread (byte-identical: a memo of
        # pure functions, any misprediction is just a miss).
        from .record.pipeline import predict_draw_times
        _rs = results_start_ms
        results.prebake_ahead(
            t - _rs for t in predict_draw_times(
                settings.fps, render_start_ms, end_ms, speed,
                warp.rate_at if warp is not None else None) if t >= _rs)
    perf.mark("setup_done")
    t0 = time.monotonic()
    try:
        if late is not None:
            # Start only after setup succeeds and inside the cleanup guard.
            late.start()
            print("[std] audio off the start: the video does not wait for the mix",
                  file=sys.stderr, flush=True)
        with FfmpegPipe(cmd, recycle=spr.recycle_frame) as pipe:
            n_frames = RecordPipeline(settings.fps, pipe.push,
                                      progress=progress).run(
                player, total_ms=total_wall_ms)
        # FfmpegPipe.__exit__ has returned: stdin closed, writer thread joined
        # and ffmpeg reaped. Everything between drain_end and here is the
        # encoder tail, which is invisible to a frames/second number.
        perf.mark("encoder_done")
        if late is not None:
            _late_s = late.finish()
            perf.mark("aud:late_joined")
            if last_pct[0] >= 0:
                print(file=sys.stderr, flush=True)
            print(f"[std] audio off the start: audio was ready "
                  f"{late.ready_s:.1f}s in, joined in {_late_s:.2f}s"
                  + ("" if n_frames == _n_pred else
                     f" (frames {n_frames}, expected {_n_pred}: the audio "
                     f"ends {abs(n_frames - _n_pred)} frame(s) off)"),
                  file=sys.stderr, flush=True)
        render_complete = True
        progress(1.0)
    finally:
        if last_pct[0] >= 0:
            print(file=sys.stderr, flush=True)
        if late is not None:
            late.cleanup()
        if video_bg is not None:
            video_bg.close()
        if audio_path is not None:
            try:
                audio_path.unlink()
            except OSError:
                pass
    wall = time.monotonic() - t0
    _print_hud_final_values(hud, judgments, meta, frozen=frozen)
    print(f"done: {n_frames} frames in {wall:.1f}s "
          f"({n_frames / wall:.1f} fps render, encoder {encoder}) → {output}",
          file=sys.stderr)
    # ── SCORE FIDELITY sidecar (parity with catch/taiko #115/#155) ────
    # std DISPLAYS the lazer-standardised ScoreV3 (the pinned header for a
    # lazer/ScoreV2 replay; the sim's own standardised trajectory for a plain
    # stable ScoreV1 header, #115). Persist that SAME total to
    # `<output>.score.json` so the bot (worker.py / cli/r3d_render.py) stores
    # it in renders.score_v3 on EVERY render — re-renders and unsubmitted
    # plays included — instead of depending on the off-band score_v3 backfill.
    # Same filename + `score_v3` key + schema the catch/taiko engines emit;
    # the value == the full-map standardised total (HudData.final_score(), ==
    # the backfill's score_at(1e12) and the number the in-video counter ends
    # on for a PASS). Best-effort: a failed sidecar never fails a done render.
    if hud is not None:
        try:
            import json as _json
            _sim_final = int(hud.data.final_score())
            if (meta is not None and meta.score > 0
                    and _header_is_standardised(meta)):
                _v3, _src = int(meta.score), "header_standardised"
            else:
                _v3, _src = _sim_final, "sim_standardised"
            if _v3 > 0:
                _entry = {
                    "score_v3": _v3,
                    "source": _src,
                    "sim_final": _sim_final,
                    "header_score": int(getattr(meta, "score", 0) or 0),
                    "player": (getattr(meta, "player_name", "") or "")
                              if meta is not None else "",
                }
                _sidecar = Path(str(output) + ".score.json")
                # Gameplay-start anchor for the YT versus HUD (all-mode sync):
                # video-seconds into THIS panel where map-time 0 lands (the
                # render lead-in), plus the rate-mods speed. Lets the overlay
                # map video-time -> map-time exactly instead of sniffing audio.
                _map0_video_s = round(m2w(0.0) / 1000.0, 6)
                _sidecar.write_text(_json.dumps(
                    {"schema": 1, "mode": 0,
                     "map0_video_s": _map0_video_s, "rate": float(speed),
                     **_entry, "players": [dict(_entry)]}, default=str))
                print(f"score:  fidelity sidecar → standardised {_v3:,} "
                      f"(source={_src}, sim_final={_sim_final:,})",
                      file=sys.stderr)
        except Exception as _sc_e:  # noqa: BLE001 — sidecar is best-effort
            print(f"score:  fidelity sidecar write failed: {_sc_e}",
                  file=sys.stderr)
    if storyboard_renderer is not None:
        s = storyboard_renderer.stats()
        print(f"storyboard cache: {s['uploads']} uploads, "
              f"{s['evictions']} evictions, {s['resident']} resident, "
              f"peak {s['peak_mb']:.0f} MB", file=sys.stderr)
    spr.release()
    return 0


def _print_hud_final_values(hud, judgments, meta=None, frozen=None) -> None:
    """The HUD phase's honesty line: the numbers the LAST frame displays,
    with the sim-vs-replay score/accuracy checks spelled out.

    A FAILED render freezes the HUD at the death point, so the whole-map
    end values (all post-death misses) are meaningless and the replay
    comparison would false-alarm — report the frozen death tally instead."""
    if hud is None:
        return
    if frozen is not None:
        c3, c1, c5, cm = frozen["counts"]
        print(f"hud[FAIL]: frozen @ death — score {frozen['score']} | "
              f"acc {frozen['acc_pct']:.2f}% | {c3}/{c1}/{c5}/{cm} | "
              f"max combo {frozen['max_combo']}x | grade F", file=sys.stderr)
        return
    fv = hud.final_values()
    score_line = f"final score {fv['score']}"
    if meta is not None and meta.score > 0:
        if _header_is_standardised(meta):
            ok = "==" if fv["score"] == meta.score else "!= MISMATCH"
            score_line += f" (replay {meta.score} {ok})"
        else:
            # stable ScoreV1 header: the shown value is the sim's lazer-
            # standardised ScoreV3 (#115), not comparable to the raw header.
            score_line += f" (ScoreV3; stable ScoreV1 header {meta.score})"
    acc_line = f"acc {fv['acc'] * 100.0:.2f}%"
    if judgments is not None and judgments.real_counts is not None:
        c3, c1, c5, cm = judgments.real_counts
        total = c3 + c1 + c5 + cm
        real_acc = ((300 * c3 + 100 * c1 + 50 * c5) / (300.0 * total)
                    if total else 1.0)
        # a LAZER replay's authoritative accuracy is the ScoreInfo blob's
        # tick/tail-inclusive value, not the header-count formula — the
        # ruleset pins the final displayed acc to it (forum bug 11 item 1)
        if (judgments.lazer and meta is not None
                and getattr(meta, "lazer_accuracy", None) is not None):
            real_acc = meta.lazer_accuracy
        ok = "==" if abs(real_acc - fv["acc"]) < 5e-5 else "!= MISMATCH"
        acc_line += f" (replay {real_acc * 100.0:.2f}% {ok})"
    hp_part = ""
    if fv.get("hp") is not None:
        hp_part = f" | hp {fv['hp'] * 100.0:.0f}%"
    print(f"hud: {score_line} | {acc_line} | "
          f"combo {fv['combo']}x (max {fv['max_combo']}x) | "
          f"UR {fv['ur']:.1f} | grade {fv['grade']}{hp_part}",
          file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    perf.mark("main_entry")
    args = build_parser().parse_args(argv)
    if getattr(args, "force_lazer_mods", ""):
        # R3D DEBUG: read by replay.lazer_mods.read_lazer_mods (the single funnel
        # every mod consumer uses) — set before any replay parse.
        os.environ["R3D_FORCE_LAZER_MODS"] = args.force_lazer_mods

    # --no-replay: the beatmap may be the only positional
    if args.no_replay and args.beatmap is None:
        args.osr, args.beatmap = None, args.osr
    if args.beatmap is None or (args.osr is None and not args.no_replay):
        print("error: expected REPLAY.osr BEATMAP (or --no-replay BEATMAP)",
              file=sys.stderr)
        return 2

    settings = StdRenderSettings(
        resolution=args.resolution, fps=args.fps, encoder=args.encoder,
        encoder_device=args.encoder_device, skin_dir=args.skin,
        video_bitrate=args.video_bitrate,
        default_skin_dir=args.default_skin, skip_intro=args.skip_intro,
        lead_in_time=max(args.lead_in, 0.0),
        show_results=args.results, results_style=args.results_style,
        show_leaderboard=args.leaderboard,
        leaderboard_source=args.leaderboard_source,
        leaderboard_json=args.leaderboard_json,
        letterbox_breaks=args.letterbox_breaks,
        draw_approach_circles=args.approach_circles,
        draw_combo_numbers=args.combo_numbers,
        draw_follow_points=args.follow_points,
        hit_animations=args.hit_animations,
        slider_snaking_in=args.snaking_in, slider_snaking_out=args.snaking_out,
        slider_merge=args.slider_merge,
        draw_cursor=args.cursor, use_skin_cursor=args.skin_cursor,
        cursor_scale=args.cursor_scale,
        cursor_trail_scale=args.cursor_trail_scale,
        cursor_rainbow=args.cursor_rainbow,
        cursor_ripples=args.cursor_ripples,
        cursor_long_trail=args.force_long_trail,
        show_key_overlay=args.key_overlay,
        show_pp_counter=args.pp_counter, show_hit_counter=args.hit_counter,
        pp_override=args.pp,
        sr_override=args.sr,
        show_aim_error_meter=args.aim_error_meter,
        show_strain_graph=args.strain_graph,
        show_scoreboard=args.scoreboard,
        scoreboard_avatars=args.scoreboard_avatars,
        show_warning_arrows=args.warning_arrows,
        show_hit_error_meter=args.hit_error_meter,
        show_unstable_rate=args.unstable_rate, show_combo=args.show_combo,
        show_score=args.show_score, show_hp_bar=args.show_hp,
        show_grade=args.show_grade, show_mods=args.show_mods,
        show_progress=args.show_progress, progress_style=args.progress_style,
        hud_scale=args.hud_scale, hud_opacity=args.hud_opacity,
        combo_break_flash=args.break_flash,
        show_hit_lighting=args.hit_lighting,
        miss_fall=args.miss_fall,
        renderer_default_font_and_ranks=args.renderer_default_font_and_ranks,
        legacy_defaults=args.legacy_defaults,
        playfield_borders=args.playfield_borders,
        watermark_text=args.watermark, music_volume=args.music_volume,
        hitsound_volume=args.hitsound_volume,
        use_replay_hitsounds=args.hitsounds,
        nightcore_hitsounds=args.nightcore_hitsounds,
        use_skin_hitsounds=args.use_skin_hitsounds,
        general_volume=args.general_volume, audio_offset=args.audio_offset,
        bg_dim_intro=args.bg_dim_intro,
        bg_dim_game=(args.bg_dim if args.bg_dim is not None
                     else args.bg_dim_game),
        bg_dim_breaks=args.bg_dim_breaks, bg_blur=args.bg_blur,
        bg_parallax=args.bg_parallax, bg_triangles=args.bg_triangles,
        load_storyboard=args.storyboard, load_video=args.video,
        flash_to_beat=args.flash_to_beat,
        seizure_warning=args.seizure_warning,
        bloom=args.bloom, bloom_to_beat=args.bloom_to_beat,
        show_logo=args.logo,
        use_beatmap_colors=(args.combo_colors
                            in ("beatmap", "beatmap-force")),
        skin_combo_colors=(args.combo_colors == "skin"),
        beatmap_colors_forced=(args.combo_colors == "beatmap-force"),
    )
    if args.fade_out is not None:
        settings.fade_out_time = max(args.fade_out, 0.0)
    if args.results_seconds is not None:
        settings.results_screen_time = args.results_seconds

    if args.no_replay:
        frames, meta = [], None
        osu_path = find_osu_file(args.beatmap, "")
        beatmap = load_full(osu_path, mods=0)
    else:
        frames, meta = parse_replay(args.osr)
        osu_path = find_osu_file(args.beatmap, meta.beatmap_md5)
        beatmap = load_full(osu_path, mods=meta.mods,
                            difficulty_adjust=meta.difficulty_adjust,
                            speed_override=meta.rate_override,
                            mirror_reflection=meta.mirror_reflection,
                            random_seed=meta.random_seed,
                            random_angle_sharpness=meta.random_angle_sharpness,
                            target_practice_seed=meta.target_practice_seed)
        if meta.has_mirror:
            print(f"mirror: MR reflection={meta.mirror_reflection} "
                  f"(objects flipped about the playfield centre)", file=sys.stderr)
        if meta.has_random:
            print(f"random: RD seed={meta.random_seed} "
                  f"angle_sharpness={meta.random_angle_sharpness:g} "
                  f"(objects repositioned)", file=sys.stderr)
        if meta.has_target_practice:
            print(f"target: TP seed={meta.target_practice_seed} "
                  f"(map rebuilt as {len(beatmap.hit_objects)} target circles)",
                  file=sys.stderr)
        if meta.has_repel_magnet:
            _rmname = ("Magnetised (objects pulled TO the cursor)"
                       if meta.repel_magnet_acronym == "MG"
                       else "Repel (objects pushed AWAY from the cursor)")
            print(f"move:   {meta.repel_magnet_acronym} {_rmname}; "
                  f"strength={meta.repel_magnet_strength:g}, follow points "
                  f"hidden (visual only — judgement/reconcile unchanged)",
                  file=sys.stderr)
        # Relax (RX): the .osr has cursor motion but NO key presses (the
        # mod auto-taps). Synthesize the presses OsuModRelax injects so the
        # judgment sim, combo, popups, key overlay, slider-follow tracking
        # and spinner spin-up all resolve exactly as for a normal replay
        # (reconcile still keeps the counts exact). RX-only — Autopilot
        # keeps its real presses and renders through the untouched path.
        if is_relax_meta(meta):
            frames = synthesize_relax_frames(frames, beatmap)
            _held = sum(1 for f in frames if f.keys)
            print(f"relax:  RX detected — synthesized OsuModRelax auto-taps "
                  f"({_held}/{len(frames)} frames key-held)", file=sys.stderr)
    skin_info = load_skin_ini(settings.skin_dir)

    print(f"map:    {beatmap.artist} - {beatmap.name} [{beatmap.difficulty_name}] "
          f"by {beatmap.creator} (v{beatmap.version})", file=sys.stderr)
    n_sliders = sum(isinstance(o, Slider) for o in beatmap.hit_objects)
    n_spinners = sum(isinstance(o, Spinner) for o in beatmap.hit_objects)
    n_circles = len(beatmap.hit_objects) - n_sliders - n_spinners
    print(f"objects: {n_circles}c/{n_sliders}s/{n_spinners}sp  "
          f"CS{beatmap.diff.cs:g}→r={beatmap.diff.circle_radius:.2f}px  "
          f"AR{beatmap.diff.ar:g}→preempt={beatmap.diff.preempt}ms",
          file=sys.stderr)
    if meta is not None:
        print(f"replay: {meta.player_name} {meta.count_300}/{meta.count_100}/"
              f"{meta.count_50}/{meta.count_miss} {meta.accuracy}% {meta.grade} "
              f"mods={meta.mods:#x} frames={len(frames)}", file=sys.stderr)
        _da = meta.difficulty_adjust
        if _da is not None:
            _parts = " ".join(f"{k.upper()}{_da[k]:g}"
                              for k in ("ar", "cs", "od", "hp")
                              if _da[k] is not None)
            print(f"DA:     override {_parts}"
                  f"{' [extended-limits]' if _da['extended'] else ''} "
                  f"-> CS{beatmap.diff.cs:g} AR{beatmap.diff.ar:g} "
                  f"OD{beatmap.diff.od:g} HP{beatmap.diff.hp:g}", file=sys.stderr)
        if meta.rate_override is not None:
            _audio = ("pitch-shifted" if meta.rate_pitch
                      else "tempo-only (pitch preserved)")
            print(f"rate:   custom {meta.rate_override:g}x "
                  f"(bitmask default was {beatmap.diff.base_mod_speed:g}x) "
                  f"-> speed {beatmap.diff.speed:g}x, "
                  f"AR{beatmap.diff.ar:g}->ar_real {beatmap.diff.ar_real:.2f}, "
                  f"OD{beatmap.diff.od:g}->od_real {beatmap.diff.od_real:.2f}; "
                  f"audio {_audio}", file=sys.stderr)
        if meta.has_rate_ramp:
            _audio = ("pitch follows rate" if meta.ramp_pitch
                      else "tempo-only (pitch preserved)")
            print(f"ramp:   {meta.ramp_acronym} clock rate "
                  f"{meta.ramp_initial:g}x -> {meta.ramp_final:g}x "
                  f"(linear over [first object, 75% of the map], then pinned); "
                  f"audio {_audio}", file=sys.stderr)
    else:
        print("replay: (none — --no-replay perfect play)", file=sys.stderr)
    print(f"skin:   \"{skin_info.name or 'default'}\" v{skin_info.version:g}",
          file=sys.stderr)

    # judgment simulation (pure CPU — runs in --parse-only too, so the
    # sim-vs-real honesty metric is checkable without GL)
    # a FAILED replay (life-bar hit 0, no fail-immune mod) renders osu's fail
    # sequence and its counts are the partial tally AT death — so the ruleset
    # must NOT reconcile the whole-map sim to the .osr's (partial) totals.
    failing = (meta is not None and meta.fail_time is not None
               and not args.no_fail_animation)
    judgments = None
    if frames and meta is not None and meta.mode == 0:
        from .ruleset import StdRuleset
        judgments = StdRuleset(beatmap, frames, meta,
                               reconcile=not failing).run()
        for line in judgments.report_lines():
            print(line, file=sys.stderr)
    if meta is not None and meta.fail_time is not None:
        if failing:
            print(f"fail:   life-bar hit 0 at {meta.fail_time / 1000.0:.2f}s "
                  f"— rendering osu fail sequence (grade F)", file=sys.stderr)
        else:
            print(f"fail:   life-bar hit 0 at {meta.fail_time / 1000.0:.2f}s "
                  f"but --no-fail-animation set — rendering as a pass",
                  file=sys.stderr)

    if args.parse_only:
        return 0

    if meta is not None and meta.mode != 0:
        print(f"error: replay mode {meta.mode} is not osu!standard",
              file=sys.stderr)
        return 2
    if not frames and not args.no_replay:
        print("error: replay has no cursor frames", file=sys.stderr)
        return 2

    # §4.15 -skip semantics: trim a long SILENT intro — start ~1 s before
    # the first object's approach begins, but only when that actually
    # saves more than 2 s (short intros render untouched) and no explicit
    # --start was given. --dump-frames uses absolute map times either way.
    if args.skip_intro and args.start is None and beatmap.hit_objects:
        first_t = min(o.get_start_time() for o in beatmap.hit_objects)
        t0 = (first_t - beatmap.diff.preempt - 1000.0) / 1000.0
        if t0 > 2.0:
            # smoke-aware trim: the player can DRAW with the Smoke key
            # (keys bit 16) during the lead-in BEFORE the trimmed window
            # — never cut a drawing off. Frame times are MAP time
            # (replay/replay.py), the same timebase as t0/first_t, so
            # they compare directly and the warp-aware start/pre-roll
            # math downstream of args.start (§4.10) is untouched under
            # DT/HT/WU/WD. The 300 ms pad is map-time, mirroring the
            # 1000 ms approach buffer above; clamped to 0.0 (frame
            # times are >= 0 and 0 == the untrimmed baseline, never
            # pre-audio). No smoke / smoke only inside the window ->
            # t0 untouched (silent intros stay trimmed).
            _smoke_t = min(
                (f.time_ms for f in frames if f.keys & KEY_SMOKE),
                default=None)
            if _smoke_t is not None and _smoke_t < t0 * 1000.0:
                t0 = max(0.0, (_smoke_t - 300.0) / 1000.0)
                if t0 > 2.0:
                    print(f"smoke:  drawn from {_smoke_t / 1000.0:.1f}s "
                          f"— intro trim pulled back to {t0:.1f}s",
                          file=sys.stderr)
                else:
                    print(f"smoke:  drawn from {_smoke_t / 1000.0:.1f}s "
                          f"— intro trim disabled (drawing starts too "
                          f"early)", file=sys.stderr)
        if t0 > 2.0:
            args.start = t0
            print(f"skip:   intro trimmed to {t0:.1f}s (first object at "
                  f"{first_t / 1000.0:.1f}s)", file=sys.stderr)

    beatmap_dir = args.beatmap if args.beatmap.is_dir() else args.beatmap.parent
    return _render(args, settings, beatmap, frames, beatmap_dir, skin_info,
                   judgments=judgments, meta=meta, osu_path=osu_path)


if __name__ == "__main__":
    raise SystemExit(main())
