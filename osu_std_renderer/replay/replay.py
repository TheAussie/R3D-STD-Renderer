"""std .osr decode + 2D cursor interpolation — adapted from the production
catch renderer's replay.py (osu-catch/osu_catch_renderer/replay.py), extended
from 1D catcher-x to the std (x, y, keys) frame.

RENDER_PLAN.md §5.5 — the replay→cursor contract the ruleset phase must obey
(documented here; the full ruleset ships in a later phase):

  * the .osr is parsed by osrparse (reference used rplpa); the RNG seed
    frame (time_delta == -12345) is stripped;
  * playback replays EVERY INTERMEDIATE INTEGER MILLISECOND so low draw
    rates never skip frames: per tick, while
    `replayTime + frames[idx].Time <= floor(nTime)` advance the index and
    apply cursor.SetPos(MouseX, MouseY) + the key bitfield (feeding
    UpdateClickFor/UpdateNormalFor/UpdatePostFor for judgments);
  * when no frame is consumed on a tick, the cursor position is
    INTERPOLATED between the neighbouring frames (cursor_at below);
  * Relax/Autopilot get synthesized clicks/aim.

Frame times are in MAP time (rate mods don't compress them), so they line
up 1:1 with beatmap object times — same finding as catch.
"""
from __future__ import annotations

import lzma
import struct
from dataclasses import dataclass
from pathlib import Path

from osrparse import Replay

from osu_std_renderer.security import bounded_lzma_decompress, validate_replay_payload

from .lazer_mods import (LAZER_GAME_VERSION,
                         approach_different_from_mods,
                         barrel_roll_from_mods,
                         blinds_from_mods,
                         bloom_from_mods,
                         bubbles_from_mods,
                         depth_from_mods,
                         difficulty_adjust_from_mods,
                         freeze_frame_from_mods,
                         mirror_from_mods,
                         no_scope_from_mods,
                         random_from_mods,
                         rate_adjust_from_mods,
                         rate_ramp_from_mods,
                         LazerStatistics,
                         read_lazer_accuracy,
                         read_lazer_mods,
                         read_lazer_statistics,
                         repel_magnet_from_mods,
                         synesthesia_from_mods,
                         target_practice_from_mods,
                         traceable_from_mods,
                         transform_from_mods)

# osrparse seeds the last frame with this sentinel time_delta (RNG seed).
_SEED_DELTA = -12345

# std key bitmask per frame (osu! wiki / osr format)
KEY_M1 = 1
KEY_M2 = 2
KEY_K1 = 4
KEY_K2 = 8
KEY_SMOKE = 16

# mod bits (osu! wiki / .osr Mods) relevant to fail immunity. A play under a
# mod that overrides failing NEVER reaches the fail screen — these implement
# osu.Game IApplicableFailOverride with PerformFail() => false:
#   * OsuModNoFail    (ModNoFail)        — bit 1
#   * OsuModAutopilot (Relax2)           — bit 8192
# (Relax does NOT override failing — you can still fail while relaxing, so it
# is deliberately absent. Cinema/Autoplay don't submit replays.)
MOD_NOFAIL = 0x1
MOD_AUTOPILOT = 0x2000
_FAIL_IMMUNE_MODS = MOD_NOFAIL | MOD_AUTOPILOT

# life-bar health at/under this counts as "dead" (osrparse LifeBarState.life
# is 0..1; use a tiny epsilon so float noise near 0 still trips).
_DEAD_HEALTH_EPS = 0.001


class ReplayParseError(RuntimeError):
    pass


def detect_fail_time(replay) -> float | None:
    """The death point of a replay, or None for a PASS.

    Ported from the osu! fail model: a play fails the instant its health
    (HP bar) reaches zero (osu.Game/Rulesets/Scoring/HealthProcessor.cs —
    ``HasFailed => Health.Value == HealthProcessor.MinimumHealth`` where the
    minimum is 0). We read the AUTHORITATIVE record of that curve — the
    .osr's life-bar graph (osrparse ``Replay.life_bar_graph`` = a list of
    ``LifeBarState(time, life)`` with life in 0..1) — and return the FIRST
    state whose life hits 0.

    Safeguards:
      * FAIL-IMMUNE MODS: NoFail / Autopilot override failing in osu!
        (IApplicableFailOverride.PerformFail() => false), so the play can
        never fail regardless of the HP curve — return None.
      * EMPTY / MISSING life-bar graph (some lazer .osr omit it): be
        CONSERVATIVE and return None (PASS). We NEVER fail a replay off our
        own HP simulation — a submitted replay demonstrably *finished the
        map* unless the lifebar explicitly records a zero, so only that
        explicit evidence trips a fail. This keeps false-fails impossible on
        the pass path.
    """
    try:
        mods = int(replay.mods)
    except (TypeError, ValueError, AttributeError):
        mods = 0
    if mods & _FAIL_IMMUNE_MODS:
        return None
    graph = getattr(replay, "life_bar_graph", None) or []
    for state in graph:
        try:
            if float(state.life) <= _DEAD_HEALTH_EPS:
                return float(int(state.time))
        except (TypeError, ValueError, AttributeError):
            continue
    return None


@dataclass(frozen=True)
class StdFrame:
    """One replay frame: cursor position in osu!px (512×384) + key bitmask."""
    time_ms: int
    x: float
    y: float
    keys: int

    @property
    def pressed(self) -> bool:
        return bool(self.keys & (KEY_M1 | KEY_M2 | KEY_K1 | KEY_K2))


@dataclass(frozen=True)
class ReplayMeta:
    mode: int
    beatmap_md5: str
    player_name: str
    mods: int
    score: int
    max_combo: int
    count_300: int
    count_100: int
    count_50: int
    count_geki: int
    count_katu: int
    count_miss: int
    accuracy: float
    grade: str
    game_version: int = 0    # <30000000 = osu!stable, else lazer
    death_ms: int | None = None
    played_at: str = ""      # .osr timestamp → "Played on <date>" (results)
    # lazer mod acronyms read from the .osr's appended ScoreInfo blob
    # (empty for stable replays / when the blob is absent);
    # has_classic_mod = the Classic mod ("CL") is in that list.
    lazer_mods: tuple[str, ...] = ()
    has_classic_mod: bool = False
    # lazer slider-part judgement counts (LargeTick / SliderTail) from the
    # ScoreInfo blob - the combo-relevant tick/tail results the legacy
    # 300/100/50/miss header can't express. None for stable replays / a
    # blob without slider-part stats; see lazer_mods.read_lazer_statistics.
    lazer_statistics: "LazerStatistics | None" = None
    # Lazer's canonical accuracy RATIO (0..1) rebuilt from the ScoreInfo
    # blob's statistics dicts — the tick/tail-inclusive number the player
    # saw on lazer's own HUD/results (ScoreProcessor.Accuracy), which on
    # slider maps differs from the header-count `accuracy` above. None for
    # stable replays / an absent blob; see lazer_mods.read_lazer_accuracy.
    lazer_accuracy: float | None = None
    # Difficulty Adjust (DA) overrides read from the ScoreInfo blob (lazer
    # only). Each is the custom AR/CS/OD/HP or None (= keep the beatmap's
    # value). da_extended_limits mirrors the mod's ExtendedLimits toggle
    # (allows AR/CS/OD/HP beyond 0..10). See lazer_mods.read_difficulty_adjust.
    da_ar: float | None = None
    da_cs: float | None = None
    da_od: float | None = None
    da_hp: float | None = None
    da_extended_limits: bool = False
    # Custom-rate mods (lazer DT/NC/HT/DC with a per-play ``speed_change`` that
    # differs from the fixed legacy bitmask rate). rate_override is the
    # effective clock rate (None = plain bitmask rate — DT 1.5 / HT 0.75 /
    # nomod 1.0 — so legacy replays are byte-identical). rate_pitch mirrors
    # ModNightcore/ModDaycore: NC/DC shift the audio pitch with the rate,
    # DT/HT change tempo only. See lazer_mods.rate_adjust_from_mods.
    rate_override: float | None = None
    rate_pitch: bool = False
    # Rate-RAMP mods (lazer Wind Up / Wind Down): the clock rate RAMPS linearly
    # from ramp_initial to ramp_final over the map (see replay/lazer_mods.py
    # RateRamp + timewarp.TimeWarp). ramp_initial is None for every non-ramp
    # replay (constant rate / bitmask / nomod) so those stay byte-identical.
    # ramp_pitch mirrors ModTimeRamp.AdjustPitch (True = pitch follows rate).
    ramp_acronym: str = ""
    ramp_initial: float | None = None
    ramp_final: float | None = None
    ramp_pitch: bool = False
    # Position mods (lazer-only). Mirror (MR): the reflection axis
    # ("horizontal"/"vertical"/"both", "" = no MR) — objects are flipped about
    # the playfield centre. Random (RD): the .NET System.Random seed that
    # scatters the objects (None = no RD) plus its AngleSharpness. Both are
    # empty/None for every non-MR/RD replay, so those renders are byte-identical.
    # See beatmap/mods_position.py.
    mirror_reflection: str = ""
    random_seed: int | None = None
    random_angle_sharpness: float = 7.0
    # Target Practice (TP): a CONVERSION mod (lazer-only) that replaces the map
    # with seeded "target" hit circles on the beat. target_practice_seed is the
    # .NET Random seed the layout is regenerated from (None = no reproducible TP,
    # so the map is left untouched). target_practice_metronome mirrors the
    # audio metronome-tick toggle. See beatmap/target_practice.py.
    target_practice_seed: int | None = None
    target_practice_metronome: bool = True
    # Transform-family "fun" mods (GR/DF/SI/WG/TR): per-object visual entrance
    # animation only — the beatmap geometry, the cursor and the judgement are
    # untouched (see render/transform_mods.py). transform_acronym is "" for
    # every non-transform replay (the gate that keeps those renders identical);
    # transform_start_scale is the GR/DF starting size (else 1.0) and
    # transform_strength the WG wiggle strength (else 1.0).
    transform_acronym: str = ""
    transform_start_scale: float = 1.0
    transform_strength: float = 1.0
    # Approach/circle-appearance mods (FR/AD/TC): purely visual — the beatmap
    # geometry, the cursor and the judgement/reconcile are UNTOUCHED (the scene
    # never mutates diff.preempt). Each is False/None for every replay that
    # doesn't carry it, so those renders stay byte-identical. See
    # lazer_mods.py + render/appearance_mods.py.
    #   freeze_frame (FR): a whole combo's approach circles appear together.
    #   traceable (TC): hit circles show only their approach circle; sliders
    #     become outline-only.
    #   approach_scale/approach_style (AD): the approach circle's custom initial
    #     size + shrink-easing (None/"" = no AD).
    freeze_frame: bool = False
    traceable: bool = False
    approach_scale: float | None = None
    approach_style: str = ""
    # Screen / cursor-effect visual mods (BR/BM/SY/BL/NS/DP/BU): purely visual —
    # the beatmap geometry, the cursor and the judgement/reconcile are UNTOUCHED
    # (see render/screen_mods.py). Each is None/False/default for every replay
    # that doesn't carry it, so those renders stay byte-identical.
    #   BR (barrel_roll): the whole gameplay layer spins (spin_speed rev/min +
    #     direction ±1). BM (bloom_*): the cursor grows with combo. SY
    #     (synesthesia): combo colours come from each object's beat-snap. BL
    #     (blinds): health-driven dark panels close from the screen edges. NS
    #     (no_scope_*): the cursor fades with combo. DP (depth_*): objects
    #     approach in 3D. BU (bubbles): expanding bubbles spawn on each hit.
    barrel_roll: "BarrelRoll | None" = None
    bloom: "Bloom | None" = None
    synesthesia: bool = False
    blinds: bool = False
    no_scope: "NoScope | None" = None
    depth: "Depth | None" = None
    bubbles: bool = False
    # Cursor-driven object-movement mods Magnetised (MG) / Repel (RP): every
    # alive object EASES toward (MG) / away from (RP) the recorded cursor each
    # frame (render/repel_magnet.py; the scene integrates it statefully). Visual
    # only — the beatmap geometry, the cursor and the judgement/reconcile are
    # UNTOUCHED (the object's own drawable rides the eased offset; the judgement
    # popup stays at the original position, matching lazer). Both hide follow
    # points. repel_magnet_acronym is "" for every non-MG/RP replay, so those
    # renders stay byte-identical; repel_magnet_strength is the Attraction /
    # Repulsion strength (0.5 default). See lazer_mods.repel_magnet_from_mods.
    repel_magnet_acronym: str = ""
    repel_magnet_strength: float = 0.5

    @property
    def has_barrel_roll(self) -> bool:
        """True when the replay carries the Barrel Roll mod (BR)."""
        return self.barrel_roll is not None

    @property
    def has_bloom(self) -> bool:
        """True when the replay carries the Bloom mod (BM)."""
        return self.bloom is not None

    @property
    def has_no_scope(self) -> bool:
        """True when the replay carries the No Scope mod (NS)."""
        return self.no_scope is not None

    @property
    def has_depth(self) -> bool:
        """True when the replay carries the Depth mod (DP)."""
        return self.depth is not None

    @property
    def has_transform(self) -> bool:
        """True when the replay carries a GR/DF/SI/WG/TR entrance-animation
        mod (purely visual — judgement/reconcile stay on the real positions)."""
        return bool(self.transform_acronym)

    @property
    def has_approach_different(self) -> bool:
        """True when the replay carries the Approach Different mod (AD)."""
        return self.approach_scale is not None

    @property
    def has_mirror(self) -> bool:
        """True when the replay carries a Mirror mod (objects reflected)."""
        return bool(self.mirror_reflection)

    @property
    def has_random(self) -> bool:
        """True when the replay carries a reproducible Random mod (a seed is
        present). The gate that keeps every non-RD render byte-identical."""
        return self.random_seed is not None

    @property
    def has_target_practice(self) -> bool:
        """True when the replay carries a reproducible Target Practice mod (a
        seed is present). The gate that swaps the map for the generated targets;
        every non-TP render stays byte-identical."""
        return self.target_practice_seed is not None

    @property
    def has_repel_magnet(self) -> bool:
        """True when the replay carries Magnetised (MG) or Repel (RP) — the
        cursor-driven object-movement mods (objects ease to/from the cursor)."""
        return bool(self.repel_magnet_acronym)

    @property
    def fail_time(self) -> float | None:
        """Death point in MAP ms, or None for a pass (alias of death_ms)."""
        return None if self.death_ms is None else float(self.death_ms)

    @property
    def has_difficulty_adjust(self) -> bool:
        """True when the replay carries DA overrides that actually change a
        stat (an all-absent DA is a no-op and leaves the beatmap untouched)."""
        return (self.da_ar is not None or self.da_cs is not None
                or self.da_od is not None or self.da_hp is not None)

    @property
    def difficulty_adjust(self) -> dict | None:
        """The DA override mapping for load_full, or None when absent (the
        gate that keeps non-DA renders on the untouched beatmap path)."""
        if not self.has_difficulty_adjust:
            return None
        return {"ar": self.da_ar, "cs": self.da_cs, "od": self.da_od,
                "hp": self.da_hp, "extended": self.da_extended_limits}

    @property
    def has_rate_override(self) -> bool:
        """True when the replay carries a custom clock rate (a lazer
        DT/NC/HT/DC whose speed_change differs from the fixed bitmask rate).
        The gate that keeps plain bitmask/nomod renders on the legacy path."""
        return self.rate_override is not None

    @property
    def has_rate_ramp(self) -> bool:
        """True when the replay carries a Wind Up / Wind Down ramp (WU/WD).
        The gate that keeps every constant-rate/nomod render byte-identical —
        the time-varying warp only activates when this is True."""
        return self.ramp_initial is not None

    @property
    def rate_ramp(self) -> tuple[float, float, bool] | None:
        """The ramp descriptor ``(initial, final, adjust_pitch)`` for building
        a timewarp.TimeWarp, or None when the replay carries no WU/WD mod."""
        if not self.has_rate_ramp:
            return None
        return (self.ramp_initial, self.ramp_final, self.ramp_pitch)


def _recover_leadin_offset(path: Path) -> int:
    """Recover the replay-clock lead-in that osrparse silently discards.

    osu!stable replays begin with up to two placeholder frames at the
    off-screen sentinel position (256, -500). osu!'s own LegacyScoreDecoder
    ACCUMULATES each frame's time delta into the running clock *before* it
    drops those placeholders (``lastTime += diff`` precedes the
    ``i < 2 && (256,-500)`` skip), so the second placeholder's delta carries
    the audio lead-in / intro-skip offset. osrparse instead ``continue``s past
    these frames (osrparse/replay.py: ``if i < 2 and float(x) == 256 and
    float(y) == -500: continue``), throwing their deltas away — the lead-in
    never reaches ``Replay.replay_data``. Accumulating that stream from 0 then
    starts the cursor clock too early by the whole lead-in, so every object is
    sampled before the player reached it (mass over-miss on ~25% of stable
    replays: intro-skips of 1.7-7.0 s collapse to all-miss).

    Reproduce osu!'s accumulation: read the raw replay-data blob and sum the
    deltas of exactly the leading placeholder frames osrparse strips. Returns 0
    when there are none (lazer replays carry no placeholder frames; clean
    stable plays carry a ~0 ms lead-in), so already-aligned replays are left
    byte-identical. Fail-soft: any decode problem returns 0 (the pre-fix
    behaviour) rather than raising.
    """
    try:
        data = Path(path).read_bytes()
        off = 0

        def _skip_string() -> None:
            nonlocal off
            tag = data[off]
            off += 1
            if tag == 0x00:
                return
            if tag != 0x0b:
                raise ValueError(f"bad string tag {tag}")
            length = shift = 0
            while True:
                byte = data[off]
                off += 1
                length |= (byte & 0x7F) << shift
                if not (byte & 0x80):
                    break
                shift += 7
            off += length

        off += 1                       # mode (byte)
        off += 4                       # game version (int32)
        _skip_string()                 # beatmap md5
        _skip_string()                 # player name
        _skip_string()                 # replay md5
        off += 2 * 6                   # 300/100/50/geki/katu/miss (6 shorts)
        off += 4                       # score (int32)
        off += 2                       # max combo (short)
        off += 1                       # perfect (byte)
        off += 4                       # mods (int32)
        _skip_string()                 # life-bar graph
        off += 8                       # timestamp (int64)
        rlen = struct.unpack_from("<i", data, off)[0]
        off += 4                       # replay-data length (int32)
        if rlen < 0 or rlen > len(data) - off:
            raise ValueError("invalid replay payload length")
        raw = bounded_lzma_decompress(
            data[off:off + rlen], format=lzma.FORMAT_AUTO
        ).decode("ascii", "replace")

        lead = 0
        for i, group in enumerate(raw.rstrip(",").split(",")):
            if not group:
                continue
            fields = group.split("|")
            delta = int(fields[0])
            if delta == _SEED_DELTA:   # RNG seed (never a leading frame) — stop
                break
            # osrparse strips only the first two frames, and only when they sit
            # at the (256, -500) sentinel; mirror that set exactly.
            if i < 2 and float(fields[1]) == 256.0 and float(fields[2]) == -500.0:
                lead += delta
                continue
            break                      # first real frame: nothing more to strip
        return lead
    except Exception:  # noqa: BLE001 - never let a header quirk break parsing
        return 0


def parse_replay(path: Path) -> tuple[list[StdFrame], ReplayMeta]:
    path = Path(path)
    if not path.exists():
        raise ReplayParseError(f"replay not found: {path}")
    try:
        validate_replay_payload(path)
        r = Replay.from_path(path)
    except Exception as e:  # noqa: BLE001 - osrparse raises bare exceptions
        raise ReplayParseError(f"osrparse failed: {e}") from e

    frames: list[StdFrame] = []
    # osu!-faithful clock origin. osrparse drops the leading (256,-500)
    # placeholder frames WITHOUT accumulating their deltas, discarding the
    # intro-skip / audio lead-in that osu!'s LegacyScoreDecoder folds into the
    # running clock (`lastTime += diff` precedes the skip). Seed the clock with
    # that recovered lead-in, then accumulate every gameplay delta verbatim -
    # exactly as osu! does. This replaces the old `-5000` first-frame guard,
    # which only *approximated* the missing lead-in by zeroing a large negative
    # first delta: that guard mass-missed replays whose lead-in sat on the
    # placeholder with a small first delta (e.g. intro-skips), and, once the
    # lead-in is seeded, it wrongly cancels the legitimate large-negative first
    # delta that returns the clock to ~0. The seed is 0 for lazer and ~0 for
    # clean stable plays, so those stay byte-identical.
    t = _recover_leadin_offset(path)
    for ev in r.replay_data or []:
        delta = int(getattr(ev, "time_delta", 0))
        if delta == _SEED_DELTA:
            continue  # RNG seed frame, not a real position
        t += delta
        x = getattr(ev, "x", None)
        y = getattr(ev, "y", None)
        if x is None or y is None:
            continue  # non-positional frame
        keys = getattr(ev, "keys", 0)
        try:
            keys = int(keys)
        except (TypeError, ValueError):
            keys = 0
        frames.append(StdFrame(time_ms=max(t, 0), x=float(x), y=float(y), keys=keys))
    frames.sort(key=lambda f: f.time_ms)

    # std accuracy: 300/100/50 weighted (300 = full weight)
    total = r.count_300 + r.count_100 + r.count_50 + r.count_miss
    if total > 0:
        acc = (300 * r.count_300 + 100 * r.count_100 + 50 * r.count_50) / (300 * total)
    else:
        acc = 1.0

    # fail detection from the life-bar graph — NoFail/Autopilot exempt,
    # empty-lifebar conservative (see detect_fail_time).
    _ft = detect_fail_time(r)
    death_ms: int | None = None if _ft is None else int(_ft)

    # play date from the .osr timestamp (osrparse → datetime); "" fail-soft
    played_at = ""
    ts = getattr(r, "timestamp", None)
    if ts is not None:
        try:
            played_at = ts.strftime("%d %b %Y")
        except Exception:  # noqa: BLE001 — odd/naive datetime → skip
            played_at = str(ts)[:10]

    # lazer Classic-mod detection: read the mod-acronym list from the
    # .osr's appended ScoreInfo blob (lazer replays only — see
    # lazer_mods.py; the legacy `mods` bitmask can't encode CL).
    gv = int(getattr(r, "game_version", 0) or 0)
    lazer_mods: tuple[str, ...] = ()
    has_classic = False
    lazer_statistics: "LazerStatistics | None" = None
    da_ar = da_cs = da_od = da_hp = None
    da_ext = False
    rate_override: float | None = None
    rate_pitch = False
    ramp_acronym = ""
    ramp_initial: float | None = None
    ramp_final: float | None = None
    ramp_pitch = False
    mirror_reflection = ""
    random_seed: int | None = None
    random_angle_sharpness = 7.0
    target_practice_seed: int | None = None
    target_practice_metronome = True
    transform_acronym = ""
    transform_start_scale = 1.0
    transform_strength = 1.0
    freeze_frame = False
    traceable = False
    approach_scale: float | None = None
    approach_style = ""
    barrel_roll = None
    bloom = None
    synesthesia = False
    blinds = False
    no_scope = None
    depth = None
    bubbles = False
    repel_magnet_acronym = ""
    repel_magnet_strength = 0.5
    lazer_accuracy: float | None = None
    if gv >= LAZER_GAME_VERSION:
        # slider-part judgement counts (LargeTick / SliderTail) - the
        # ruleset reconciles the sim's tick/tail combo outcomes to these.
        lazer_statistics = read_lazer_statistics(path)
        # canonical lazer accuracy (tick/tail-inclusive ratio) — the value
        # the displayed accuracy of a lazer render is pinned to.
        lazer_accuracy = read_lazer_accuracy(path)
        mlist = read_lazer_mods(path)
        if mlist is not None:
            lazer_mods = tuple(m["acronym"] for m in mlist)
            has_classic = any(a.upper() == "CL" for a in lazer_mods)
            # Difficulty Adjust: the legacy `mods` bitmask can't encode DA, so
            # its custom AR/CS/OD/HP live only in the ScoreInfo blob settings.
            da = difficulty_adjust_from_mods(mlist)
            if da is not None:
                da_ar, da_cs, da_od, da_hp = da.ar, da.cs, da.od, da.hp
                da_ext = da.extended_limits
            # Custom-rate DT/NC/HT/DC: surface an override ONLY when the
            # speed_change differs from the fixed bitmask rate. A default-rate
            # DT/NC/HT/DC (is_default) leaves rate_override None so it renders
            # exactly like the legacy bitmask path (byte-identical).
            ra = rate_adjust_from_mods(mlist)
            if ra is not None and not ra.is_default:
                rate_override = ra.speed
                rate_pitch = ra.pitch
            # Rate-RAMP (Wind Up / Wind Down): a continuously-varying clock
            # rate the constant `speed` can't represent. Incompatible with the
            # rate-adjust family above, so it never co-occurs with a rate
            # override. Surfaced as ramp_initial/final/pitch for the timewarp.
            rr = rate_ramp_from_mods(mlist)
            if rr is not None:
                ramp_acronym = rr.acronym
                ramp_initial = rr.initial
                ramp_final = rr.final
                ramp_pitch = rr.adjust_pitch
            # Mirror (MR): the reflection axis the objects are flipped about.
            mr = mirror_from_mods(mlist)
            if mr is not None:
                mirror_reflection = mr
            # Random (RD): the .NET Random seed that repositions the objects.
            # A replay persists the actual seed used; without one the positions
            # can't be reproduced, so leave random_seed None (renders untouched).
            rd = random_from_mods(mlist)
            if rd is not None and rd.seed is not None:
                random_seed = rd.seed
                random_angle_sharpness = rd.angle_sharpness
            # Target Practice (TP): the .NET Random seed that regenerates the
            # map as seeded target circles. A replay persists the actual seed;
            # without one the target layout can't be reproduced, so leave
            # target_practice_seed None (map rendered untouched). TP is
            # incompatible with RD, so the two seeds never co-occur.
            tp = target_practice_from_mods(mlist)
            if tp is not None and tp.seed is not None:
                target_practice_seed = tp.seed
                target_practice_metronome = tp.metronome
            # Transform-family (GR/DF/SI/WG/TR): a per-object entrance
            # animation. Visual only, so no geometry/judgement change — the
            # scene reads transform_acronym to drive the appear transform.
            tr = transform_from_mods(mlist)
            if tr is not None:
                transform_acronym = tr.acronym
                transform_start_scale = tr.start_scale
                transform_strength = tr.strength
            # Approach/circle-appearance mods (FR/AD/TC): visual only, so no
            # geometry/judgement change — the scene reads these flags to drive
            # the approach-circle scale/timing (FR/AD) and the circle-fill /
            # slider-outline gate (TC).
            freeze_frame = freeze_frame_from_mods(mlist)
            traceable = traceable_from_mods(mlist)
            ad = approach_different_from_mods(mlist)
            if ad is not None:
                approach_scale = ad.scale
                approach_style = ad.style
            # Screen / cursor-effect visual mods (BR/BM/SY/BL/NS/DP/BU): visual
            # only, so no geometry/judgement change — the scene reads these to
            # drive the playfield spin (BR), cursor scale/fade (BM/NS), combo
            # colour (SY), screen blinds (BL), per-object depth (DP) and the
            # hit-position bubbles (BU).
            barrel_roll = barrel_roll_from_mods(mlist)
            bloom = bloom_from_mods(mlist)
            synesthesia = synesthesia_from_mods(mlist)
            blinds = blinds_from_mods(mlist)
            no_scope = no_scope_from_mods(mlist)
            depth = depth_from_mods(mlist)
            bubbles = bubbles_from_mods(mlist)
            # Magnetised (MG) / Repel (RP): cursor-driven object movement. The
            # scene reads the acronym + strength to integrate the per-frame
            # easeTo (render/repel_magnet.py). Visual only — no geometry or
            # judgement change; "" keeps every non-MG/RP render byte-identical.
            rm = repel_magnet_from_mods(mlist)
            if rm is not None:
                repel_magnet_acronym = rm.acronym
                repel_magnet_strength = rm.strength

    meta = ReplayMeta(
        mode=int(r.mode.value if hasattr(r.mode, "value") else r.mode),
        beatmap_md5=str(getattr(r, "beatmap_hash", "") or ""),
        player_name=r.username,
        mods=int(r.mods),
        score=int(r.score),
        max_combo=int(r.max_combo),
        count_300=int(r.count_300),
        count_100=int(r.count_100),
        count_50=int(r.count_50),
        count_geki=int(getattr(r, "count_geki", 0) or 0),
        count_katu=int(getattr(r, "count_katu", 0) or 0),
        count_miss=int(r.count_miss),
        accuracy=round(acc * 100, 2),
        grade=_grade(r),
        game_version=gv,
        death_ms=death_ms,
        played_at=played_at,
        lazer_mods=lazer_mods,
        has_classic_mod=has_classic,
        lazer_statistics=lazer_statistics,
        lazer_accuracy=lazer_accuracy,
        da_ar=da_ar,
        da_cs=da_cs,
        da_od=da_od,
        da_hp=da_hp,
        da_extended_limits=da_ext,
        rate_override=rate_override,
        rate_pitch=rate_pitch,
        ramp_acronym=ramp_acronym,
        ramp_initial=ramp_initial,
        ramp_final=ramp_final,
        ramp_pitch=ramp_pitch,
        mirror_reflection=mirror_reflection,
        random_seed=random_seed,
        random_angle_sharpness=random_angle_sharpness,
        target_practice_seed=target_practice_seed,
        target_practice_metronome=target_practice_metronome,
        transform_acronym=transform_acronym,
        transform_start_scale=transform_start_scale,
        transform_strength=transform_strength,
        freeze_frame=freeze_frame,
        traceable=traceable,
        approach_scale=approach_scale,
        approach_style=approach_style,
        barrel_roll=barrel_roll,
        bloom=bloom,
        synesthesia=synesthesia,
        blinds=blinds,
        no_scope=no_scope,
        depth=depth,
        bubbles=bubbles,
        repel_magnet_acronym=repel_magnet_acronym,
        repel_magnet_strength=repel_magnet_strength,
    )
    return frames, meta


def cursor_at(frames: list[StdFrame], t_ms: float) -> tuple[float, float, int]:
    """Linearly interpolate the cursor at t_ms → (x, y, keys).

    2D extension of catch's catcher_x_at: binary search + lerp. Keys are NOT
    interpolated — the frame at-or-after t carries the held state (key
    edges are only meaningful at exact frame times; the §5.5 per-ms walk in
    the ruleset consumes frames directly, this is for DRAWING)."""
    if not frames:
        return 256.0, 192.0, 0
    if t_ms <= frames[0].time_ms:
        f = frames[0]
        return f.x, f.y, f.keys
    if t_ms >= frames[-1].time_ms:
        f = frames[-1]
        return f.x, f.y, f.keys
    lo, hi = 0, len(frames) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if frames[mid].time_ms < t_ms:
            lo = mid + 1
        else:
            hi = mid
    a, b = frames[lo - 1], frames[lo]
    span = b.time_ms - a.time_ms
    if span <= 0:
        return b.x, b.y, b.keys
    f = (t_ms - a.time_ms) / span
    return a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f, b.keys


def _grade(r) -> str:
    """osu!(lazer)-EXACT rank from the replay counts. Ported from
    ScoreProcessor.RankFromScore (accuracy cutoffs X=1/S=.95/A=.9/B=.8/
    C=.7/D=0) + OsuScoreProcessor's std override (a Miss caps S/X at A).
    HD/FL silver variants are a display concern, applied elsewhere."""
    c300, c100, c50, miss = (int(r.count_300), int(r.count_100),
                             int(r.count_50), int(r.count_miss))
    total = c300 + c100 + c50 + miss
    if total == 0:
        return "D"
    acc = (300 * c300 + 100 * c100 + 50 * c50) / (300.0 * total)
    if acc >= 1.0:
        rank = "SS"
    elif acc >= 0.95:
        rank = "S"
    elif acc >= 0.9:
        rank = "A"
    elif acc >= 0.8:
        rank = "B"
    elif acc >= 0.7:
        rank = "C"
    else:
        rank = "D"
    if miss > 0 and rank in ("S", "SS"):
        rank = "A"
    return rank
