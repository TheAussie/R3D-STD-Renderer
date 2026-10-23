"""Read the lazer mod-acronym list from a .osr's appended ScoreInfo blob.

osu!lazer's LegacyScoreEncoder writes the standard .osr fields plus a tail
extension: after the 8-byte online/score id, a 4-byte little-endian length
prefix followed by an LZMA1 (alone-format) compressed JSON blob carrying the
full lazer ScoreInfo — including the ``mods`` list, where each entry is a
``{"acronym": "..."}`` dict (optionally with a ``settings`` object). The
legacy 32-bit ``mods`` bitmask osrparse exposes CANNOT represent the Classic
mod (acronym ``"CL"``), so the ScoreInfo blob is the only reliable source.

The blob-locating walk is ported from the production bot's
``mania_ordr/lazer_extension.py`` (``parse_lazer_score_info`` — the technique
the live versus/telemetry pipeline uses to read the ScoreInfo blob), reused
here so the std renderer can auto-detect the Classic mod from a .osr without
depending on the bot package.
"""
from __future__ import annotations

import json
import logging
import lzma
import os
import struct
from dataclasses import dataclass
from pathlib import Path

from osu_std_renderer.security import bounded_lzma_decompress

log = logging.getLogger(__name__)

# .osr game_version cutoff: stable writes 8-digit YYYYMMDD dates (~20251128),
# lazer writes 9-digit identifiers (~30000016). At/above this is a lazer
# replay — the only kind that carries the ScoreInfo blob / a Classic mod.
LAZER_GAME_VERSION = 30_000_000


def _read_uleb128(buf: bytes, pos: int) -> tuple[int, int]:
    """Read a ULEB128 unsigned int at ``pos`` → (value, new_position)."""
    val = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("uleb128 truncated")
        byte = buf[pos]
        pos += 1
        val |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return val, pos
        shift += 7
        if shift > 63:
            raise ValueError("uleb128 too long")


def _skip_string(buf: bytes, pos: int) -> int:
    """Skip an osu! .osr string in place → new position.
    0x00 = null/empty; 0x0b = ULEB128 length then that many UTF-8 bytes."""
    if pos >= len(buf):
        raise ValueError("string marker EOF")
    marker = buf[pos]
    pos += 1
    if marker == 0x00:
        return pos
    if marker != 0x0B:
        raise ValueError(f"unknown string marker {marker:#x}")
    length, pos = _read_uleb128(buf, pos)
    return pos + length


def parse_lazer_score_info(osr_path: Path) -> dict | None:
    """The lazer ScoreInfo JSON as a dict, or None when the .osr carries no
    (parseable) blob — a genuine stable .osr, a truncated/corrupt tail, or a
    future lazer schema this walk can't locate. Never raises."""
    try:
        buf = Path(osr_path).read_bytes()
    except OSError:
        return None
    if len(buf) < 32:
        return None
    pos = 0
    try:
        pos += 1                       # mode (byte)
        pos += 4                       # game_version (int)
        pos = _skip_string(buf, pos)   # beatmap_hash
        pos = _skip_string(buf, pos)   # username
        pos = _skip_string(buf, pos)   # replay_hash
        pos += 6 * 2                   # 6 judgement counts (shorts)
        pos += 4                       # total score (int)
        pos += 2                       # max_combo (short)
        pos += 1                       # perfect (byte)
        pos += 4                       # legacy mods bitfield (int)
        pos = _skip_string(buf, pos)   # life_bar_graph
        pos += 8                       # timestamp (long)
        if pos + 4 > len(buf):
            return None
        (replay_data_length,) = struct.unpack_from("<i", buf, pos)
        pos += 4
        if replay_data_length < 0 or pos + replay_data_length > len(buf):
            return None
        pos += replay_data_length      # opaque LZMA replay data
        pos += 8                       # online/score id (long)
        # A trailing 4-byte LE length + LZMA1 blob = lazer's ScoreInfo.
        # Stable .osr have no trailing data → EOF here is the normal signal.
        if pos + 4 > len(buf):
            return None
        (length,) = struct.unpack_from("<i", buf, pos)
        pos += 4
        if length <= 0 or pos + length > len(buf):
            return None
        compressed = buf[pos:pos + length]
        try:
            decompressed = bounded_lzma_decompress(
                compressed, format=lzma.FORMAT_ALONE
            )
        except lzma.LZMAError:
            decompressed = bounded_lzma_decompress(
                compressed, format=lzma.FORMAT_AUTO
            )
        return json.loads(decompressed.decode("utf-8"))
    except (struct.error, ValueError, lzma.LZMAError,
            json.JSONDecodeError, UnicodeDecodeError, IndexError) as e:
        log.debug("lazer ScoreInfo parse failed path=%s err=%s", osr_path, e)
        return None


def _r3d_forced_lazer_mods() -> list[str]:
    """R3D DEBUG: extra lazer-mod acronyms from --force-lazer-mods /
    $R3D_FORCE_LAZER_MODS, merged into every read_lazer_mods() result so a
    fixture can PRICE an otherwise-unreachable mod render path (visual/transform
    mods especially: BR/NS/WG/... which only ride the lazer blob). Off unless the
    env var is set, so it never fires in a normal render."""
    raw = os.environ.get("R3D_FORCE_LAZER_MODS", "")
    return [a.strip().upper() for a in raw.replace(",", " ").split() if a.strip()]


def _r3d_merge_forced(out: "list[dict]", forced: "list[str]") -> "list[dict]":
    if not forced:
        return out
    have = {str(e.get("acronym", "")).upper() for e in out}
    for ac in forced:
        if ac not in have:
            out.append({"acronym": ac, "settings": {}})
            have.add(ac)
    return out


def read_lazer_mods(osr_path: Path) -> list[dict] | None:
    """The lazer mod list as normalised ``{"acronym": str, "settings": dict}``
    entries (``settings`` is ``{}`` when the mod carries none), or None when no
    ScoreInfo blob is present. Unlike :func:`read_lazer_mod_acronyms` this
    PRESERVES each mod's ``settings`` object — the per-mod configuration lazer
    serialises for configurable mods such as Difficulty Adjust (DA), custom-rate
    Double Time, etc. An empty list means the blob was read but carried no
    mods (a nomod lazer play)."""
    info = parse_lazer_score_info(osr_path)
    forced = _r3d_forced_lazer_mods()
    if info is None:
        # No ScoreInfo blob (stable/legacy .osr) -> normally None (legacy path).
        # With forced mods set, synthesise a list so a NoMod/stable fixture can
        # still exercise the forced mod paths. (R3D debug)
        return _r3d_merge_forced([], forced) if forced else None
    mods = info.get("mods")
    if not isinstance(mods, list):
        return _r3d_merge_forced([], forced)
    out: list[dict] = []
    for m in mods:
        if isinstance(m, dict):
            ac = m.get("acronym")
            if isinstance(ac, str):
                s = m.get("settings")
                out.append({"acronym": ac,
                            "settings": s if isinstance(s, dict) else {}})
        elif isinstance(m, str):
            out.append({"acronym": m, "settings": {}})
    return _r3d_merge_forced(out, forced)


def read_lazer_mod_acronyms(osr_path: Path) -> list[str] | None:
    """The lazer mod acronyms (e.g. ``["HD", "DT", "CL"]``) from a .osr, or
    None when no ScoreInfo blob is present. An empty list means the blob was
    read but carried no mods (a nomod lazer play)."""
    mods = read_lazer_mods(osr_path)
    if mods is None:
        return None
    return [m["acronym"] for m in mods]


# --- slider-part judgement counts (LazerStatistics) --------------------------
# osu.Game/Rulesets/Scoring/HitResult.cs — the classic-aggregate note counts
# (300/100/50/miss the legacy .osr header carries) can't express a lazer
# slider's tick/tail judgements, which drive combo INDEPENDENTLY of the note
# grade:
#   * SliderTick / SliderRepeat -> HitResult.LargeTickHit (default MinResult
#     LargeTickMiss). LargeTickHit.AffectsCombo && IsHit  => IncreasesCombo;
#     LargeTickMiss.AffectsCombo && !IsHit                => BreaksCombo.
#   * SliderTailCircle (non-classic TailJudgement) -> HitResult.SliderTailHit
#     (Judgement default MinResult for SliderTailHit is IgnoreMiss).
#     SliderTailHit.AffectsCombo && IsHit  => IncreasesCombo; a MISSED tail is
#     IgnoreMiss (AffectsCombo == false) => NO combo effect (not a break).
# These exact per-play counts live only in the ScoreInfo ``statistics`` /
# ``maximum_statistics`` dicts. ``max_*`` are the FC totals (every large tick /
# every slider tail) used to validate the sim generated the right part count.
@dataclass(frozen=True)
class LazerStatistics:
    large_tick_hit: int
    large_tick_miss: int
    slider_tail_hit: int
    max_large_tick: int          # maximum_statistics.large_tick_hit (FC total)
    max_slider_tail: int         # maximum_statistics.slider_tail_hit (FC total)
    small_bonus: int = 0         # statistics.small_bonus (achieved bonus spins)
    large_bonus: int = 0         # statistics.large_bonus (achieved bonus spins)
    max_small_bonus: int = 0     # maximum_statistics.small_bonus (map total)
    max_large_bonus: int = 0     # maximum_statistics.large_bonus (map total)


def lazer_statistics_from_info(info: "dict | None") -> "LazerStatistics | None":
    """Extract the combo-relevant slider-part counts (and the spinner bonus
    counts) from a parsed ScoreInfo dict, or None when the blob carries none
    of them (a genuine stable score, or a lazer score on a circle-only map,
    in which case the note reconcile alone already suffices). A slider-less
    SPINNER map still returns stats: the bonus counts feed the score's bonus
    portion and the results screen's spinner rows (the parts reconcile then
    no-ops over zero ticks/tails)."""
    if not isinstance(info, dict):
        return None
    stats = info.get("statistics")
    mx = info.get("maximum_statistics")
    if not isinstance(stats, dict) or not isinstance(mx, dict):
        return None

    def g(d: dict, k: str) -> int:
        v = d.get(k, 0)
        return int(v) if isinstance(v, (int, float)) else 0

    max_lt = g(mx, "large_tick_hit")
    max_st = g(mx, "slider_tail_hit")
    max_sb = g(mx, "small_bonus")
    max_lb = g(mx, "large_bonus")
    if max_lt == 0 and max_st == 0 and max_sb == 0 and max_lb == 0:
        return None
    return LazerStatistics(
        large_tick_hit=g(stats, "large_tick_hit"),
        large_tick_miss=g(stats, "large_tick_miss"),
        slider_tail_hit=g(stats, "slider_tail_hit"),
        max_large_tick=max_lt,
        max_slider_tail=max_st,
        small_bonus=g(stats, "small_bonus"),
        large_bonus=g(stats, "large_bonus"),
        max_small_bonus=max_sb,
        max_large_bonus=max_lb,
    )


def read_lazer_statistics(osr_path: Path) -> "LazerStatistics | None":
    """The lazer slider-part judgement counts from a .osr's ScoreInfo blob, or
    None for a stable replay / a blob without slider-part statistics."""
    return lazer_statistics_from_info(parse_lazer_score_info(osr_path))


# --- lazer accuracy (the canonical displayed value) -------------------------------
# osu.Game/Rulesets/Scoring/ScoreProcessor.cs: Accuracy = baseScore /
# maxBaseScore over every ACC-AFFECTING judgement — for osu!std that folds
# slider large ticks (LargeTickHit, 30) and slider tails (SliderTailHit, 150)
# in with the object judgements (Great 300 / Ok 100 / Meh 50), which is why a
# lazer score's accuracy differs from the stable 300/100/50-only formula on
# any map with sliders. Bonus results (small_bonus/large_bonus spinner spins)
# and ignore_* never affect accuracy. The .osr ScoreInfo blob stores no
# explicit accuracy field — lazer itself rebuilds it from the statistics
# dicts, and so do we (the weight table mirrors HitResult.cs numerics; it is
# byte-for-byte the R3D bot's _STD_WEIGHTS so the render and the site card
# can never disagree).
_LAZER_ACC_WEIGHTS: dict[str, int] = {
    "perfect":         320,
    "great":           300,
    "good":            200,
    "ok":              100,
    "meh":              50,
    "miss":              0,
    "large_tick_hit":   30,
    "large_tick_miss":   0,
    "small_tick_hit":   10,
    "small_tick_miss":   0,
    "slider_tail_hit": 150,
    # bonus + ignore tiers explicitly = 0 (they don't affect accuracy)
}


def lazer_accuracy_from_info(info: "dict | None") -> float | None:
    """Lazer's accuracy (a 0..1 RATIO) rebuilt from a parsed ScoreInfo
    dict's statistics/maximum_statistics, or None when the blob is absent
    or malformed (zero denominator). This is the number the player saw on
    lazer's own HUD/results screen — the honesty reconcile pins the final
    displayed accuracy of a lazer render to it."""
    if not isinstance(info, dict):
        return None
    stats = info.get("statistics")
    mx = info.get("maximum_statistics")
    if not isinstance(stats, dict) or not isinstance(mx, dict):
        return None
    num = 0
    for k, count in stats.items():
        w = _LAZER_ACC_WEIGHTS.get(k, 0)
        if w == 0 or not isinstance(count, (int, float)):
            continue
        num += w * int(count)
    den = 0
    for k, max_count in mx.items():
        w = _LAZER_ACC_WEIGHTS.get(k, 0)
        if w == 0 or not isinstance(max_count, (int, float)):
            continue
        den += w * int(max_count)
    if den <= 0:
        return None
    return num / den


def read_lazer_accuracy(osr_path: Path) -> float | None:
    """The canonical lazer accuracy ratio from a .osr's ScoreInfo blob, or
    None for a stable replay / an unreadable blob."""
    return lazer_accuracy_from_info(parse_lazer_score_info(osr_path))


# --- Difficulty Adjust (DA) --------------------------------------------------
# osu.Game.Rulesets.Osu/Mods/OsuModDifficultyAdjust.cs +
# osu.Game/Rulesets/Mods/ModDifficultyAdjust.cs — the DA mod overrides the
# beatmap's AR/CS/OD/HP with per-setting values. Each SettingSource serialises
# to a snake_case key in the ScoreInfo ``settings`` object:
#   approach_rate  (ApproachRate,  MinValue 0 / MaxValue 10, Extended -10..11)
#   circle_size    (CircleSize,    0..10, ExtendedMaxValue 11)
#   overall_difficulty (OverallDifficulty, 0..10, ExtendedMaxValue 11)
#   drain_rate     (DrainRate,     0..10, ExtendedMaxValue 11)
#   extended_limits (ExtendedLimits BindableBool — ApplyLimits(extended))
# An ABSENT key means "keep the beatmap's value" (lazer's
# ReadCurrentFromDifficulty baseline: ``if (X.Value != null) difficulty.X =
# X.Value.Value``). DA sets the BASE stat, so rate mods (DT/HT) still scale it
# on top exactly as they would a beatmap stat.
DA_ACRONYM = "DA"
_DA_AR_KEY = "approach_rate"
_DA_CS_KEY = "circle_size"
_DA_OD_KEY = "overall_difficulty"
_DA_HP_KEY = "drain_rate"
_DA_EXT_KEY = "extended_limits"


@dataclass(frozen=True)
class DifficultyAdjust:
    """The DA overrides read from a .osr. Each of ar/cs/od/hp is the custom
    value or None (= keep the beatmap's stat). ``extended_limits`` mirrors the
    mod's ExtendedLimits toggle, which lifts the 0..10 clamp to 0..11 (AR to
    -10) so beyond-limits stats keep osu's linear AR->preempt extrapolation."""
    ar: float | None = None
    cs: float | None = None
    od: float | None = None
    hp: float | None = None
    extended_limits: bool = False

    @property
    def is_empty(self) -> bool:
        """True when the mod carried NO stat overrides (all four absent) — a
        no-op DA that must leave the beatmap difficulty untouched."""
        return (self.ar is None and self.cs is None
                and self.od is None and self.hp is None)


def _da_num(settings: dict, key: str) -> float | None:
    """A numeric DA setting or None (absent / non-numeric). JSON booleans are
    rejected so a stray ``true`` never reads back as a difficulty of 1.0."""
    v = settings.get(key)
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def difficulty_adjust_from_mods(mods: list[dict]) -> "DifficultyAdjust | None":
    """The Difficulty Adjust overrides from an already-parsed
    :func:`read_lazer_mods` list, or None when no DA mod is present."""
    for m in mods:
        if isinstance(m, dict) and str(m.get("acronym", "")).upper() == DA_ACRONYM:
            s = m.get("settings") or {}
            return DifficultyAdjust(
                ar=_da_num(s, _DA_AR_KEY),
                cs=_da_num(s, _DA_CS_KEY),
                od=_da_num(s, _DA_OD_KEY),
                hp=_da_num(s, _DA_HP_KEY),
                extended_limits=bool(s.get(_DA_EXT_KEY, False)),
            )
    return None


def read_difficulty_adjust(osr_path: Path) -> "DifficultyAdjust | None":
    """The Difficulty Adjust overrides from a .osr, or None when the replay
    carries no DA mod (or no ScoreInfo blob at all). Never raises."""
    mods = read_lazer_mods(osr_path)
    if not mods:
        return None
    return difficulty_adjust_from_mods(mods)


def has_classic_mod(osr_path: Path) -> bool:
    """True iff the .osr's lazer mod list contains the Classic mod (CL)."""
    acs = read_lazer_mod_acronyms(osr_path)
    return bool(acs) and any(a.upper() == "CL" for a in acs)


# --- Rate-adjust mods: custom-rate DT/NC/HT/DC (speed_change) -----------------
# osu.Game/Rulesets/Mods/ModRateAdjust.cs — the abstract base every rate mod
# derives from. It exposes a configurable ``SpeedChange`` BindableNumber and
# ``ApplyToRate(time, rate) => rate * SpeedChange.Value`` — the ONE clock rate
# that scales the whole play (gameplay clock + AR/OD windows + audio).
#
#   * ModDoubleTime.cs — SpeedChange ``new BindableDouble(1.5){MinValue=1.01,
#     MaxValue=2, Precision=0.01}``. ApplyToTrack via RateAdjustModHelper =
#     AdjustableProperty.Tempo → tempo change, pitch PRESERVED.
#   * ModHalfTime.cs  — SpeedChange ``new BindableDouble(0.75){MinValue=0.5,
#     MaxValue=0.99, Precision=0.01}``. Tempo-only, pitch preserved.
#   * ModNightcore.cs / ModDaycore.cs — subclass DT/HT (same SpeedChange
#     default+range) but override ApplyToTrack to add BOTH a Frequency and a
#     Tempo adjustment: ``freqAdjust = SpeedChange.Default`` (pitch pinned to
#     the CLASSIC 1.5×/0.75× nightcore/daycore frequency), ``tempoAdjust =
#     value / SpeedChange.Default`` (tempo carries the remainder). Net rate =
#     freq×tempo = value, but the audio PITCHES (whereas DT/HT do not).
#
# The legacy 32-bit `mods` bitmask can encode DT/HT/NC as fixed 1.5×/0.75×
# ONLY — a per-play custom rate lives solely in the ScoreInfo blob settings
# under the snake_case key ``speed_change`` (SettingSource "Speed increase"/
# "Speed decrease"). Absent = the mod's default.
_SPEED_CHANGE_KEY = "speed_change"

# acronym -> default SpeedChange (== the fixed legacy bitmask rate)
RATE_ADJUST_DEFAULTS = {"DT": 1.5, "NC": 1.5, "HT": 0.75, "DC": 0.75}
# acronym -> (MinValue, MaxValue) the SettingSource clamps SpeedChange to
RATE_ADJUST_RANGE = {"DT": (1.01, 2.0), "NC": (1.01, 2.0),
                     "HT": (0.5, 0.99), "DC": (0.5, 0.99)}
# NC/DC change PITCH (Frequency); DT/HT are tempo-only (pitch preserved)
PITCH_RATE_MODS = frozenset({"NC", "DC"})
RATE_ADJUST_MODS = frozenset(RATE_ADJUST_DEFAULTS)


@dataclass(frozen=True)
class RateAdjust:
    """A rate-adjust mod (DT/NC/HT/DC) read from a .osr.

    ``speed`` is the effective clock rate (the mod's default when the play
    carries no custom ``speed_change``, else the setting clamped to the mod's
    [Min,Max]). ``pitch`` is True for NC/DC (their audio shifts pitch) and
    False for DT/HT (tempo-only). ``is_default`` is True when ``speed`` equals
    the mod's default — i.e. the play is a plain fixed-rate DT/NC/HT/DC that
    the legacy bitmask already renders, so no override is needed."""
    acronym: str
    speed: float
    pitch: bool
    is_default: bool


def rate_adjust_from_mods(mods: list[dict]) -> "RateAdjust | None":
    """The rate-adjust mod from an already-parsed :func:`read_lazer_mods`
    list, or None when no DT/NC/HT/DC mod is present. Only one rate mod can be
    active in a play, so the first match wins. ``speed_change`` is clamped to
    the mod's SettingSource range; a non-numeric value (or an absent key) falls
    back to the default."""
    for m in mods:
        if not isinstance(m, dict):
            continue
        ac = str(m.get("acronym", "")).upper()
        if ac in RATE_ADJUST_MODS:
            s = m.get("settings") or {}
            default = RATE_ADJUST_DEFAULTS[ac]
            raw = _da_num(s, _SPEED_CHANGE_KEY)  # numeric-or-None (bools -> None)
            if raw is None:
                speed = default
            else:
                lo, hi = RATE_ADJUST_RANGE[ac]
                speed = min(hi, max(lo, raw))
            return RateAdjust(acronym=ac, speed=speed,
                              pitch=ac in PITCH_RATE_MODS,
                              is_default=(speed == default))
    return None


def read_rate_adjust(osr_path: Path) -> "RateAdjust | None":
    """The rate-adjust mod (DT/NC/HT/DC + effective speed_change) from a .osr,
    or None when the replay carries no rate mod (or no ScoreInfo blob). Never
    raises."""
    mods = read_lazer_mods(osr_path)
    if not mods:
        return None
    return rate_adjust_from_mods(mods)


# --- Rate-RAMP mods: Wind Up / Wind Down (ModTimeRamp) -----------------------
# osu.Game/Rulesets/Mods/ModTimeRamp.cs + ModWindUp.cs / ModWindDown.cs — the
# ramp mods CONTINUOUSLY vary the clock rate over the map (linear interp of
# InitialRate -> FinalRate across [firstObject, 75% of the map]), unlike the
# constant SpeedChange of ModRateAdjust above. Settings serialise to the
# snake_case keys ``initial_rate`` / ``final_rate`` / ``adjust_pitch``:
#
#   * ModWindUp   (acronym "WU"): InitialRate  BindableDouble(1.0){Min 0.5,
#       Max 1.99, Precision 0.01}; FinalRate BindableDouble(1.5){Min 0.51,
#       Max 2.0}; AdjustPitch new BindableBool(TRUE).
#   * ModWindDown (acronym "WD"): InitialRate  BindableDouble(1.0){Min 0.51,
#       Max 2.0}; FinalRate BindableDouble(0.75){Min 0.5, Max 1.99};
#       AdjustPitch new BindableBool(TRUE).
#
# WU and WD are mutually incompatible and incompatible with the constant-rate
# ModRateAdjust family (DT/NC/HT/DC), so a play carries AT MOST one of these
# and never alongside a rate_override. ``adjust_pitch`` DEFAULTS TO TRUE for
# both (pitch follows the rate — the classic wind-up "chipmunk"/wind-down
# "slowdown" sound); an absent key therefore reads back True.
_RAMP_INITIAL_KEY = "initial_rate"
_RAMP_FINAL_KEY = "final_rate"
_RAMP_PITCH_KEY = "adjust_pitch"

# acronym -> (initial_default, final_default)
RATE_RAMP_DEFAULTS = {"WU": (1.0, 1.5), "WD": (1.0, 0.75)}
# acronym -> ((initial_min, initial_max), (final_min, final_max))
RATE_RAMP_RANGE = {"WU": ((0.5, 1.99), (0.51, 2.0)),
                   "WD": ((0.51, 2.0), (0.5, 1.99))}
RATE_RAMP_MODS = frozenset(RATE_RAMP_DEFAULTS)


@dataclass(frozen=True)
class RateRamp:
    """A rate-ramp mod (WU/WD) read from a .osr. ``initial`` / ``final`` are
    the effective clock rates at the ramp endpoints (each clamped to the mod's
    SettingSource range, or the mod default when the setting is absent).
    ``adjust_pitch`` mirrors the mod's AdjustPitch toggle: True (the default)
    pitches the audio with the rate, False keeps pitch constant (tempo-only)."""
    acronym: str
    initial: float
    final: float
    adjust_pitch: bool


def rate_ramp_from_mods(mods: list[dict]) -> "RateRamp | None":
    """The rate-ramp mod (WU/WD) from an already-parsed :func:`read_lazer_mods`
    list, or None when no WU/WD mod is present. Only one ramp can be active, so
    the first match wins. ``initial_rate`` / ``final_rate`` are clamped to the
    mod's range; a non-numeric value (or an absent key) falls back to the mod
    default. ``adjust_pitch`` defaults to True (absent -> True)."""
    for m in mods:
        if not isinstance(m, dict):
            continue
        ac = str(m.get("acronym", "")).upper()
        if ac in RATE_RAMP_MODS:
            s = m.get("settings") or {}
            di, df = RATE_RAMP_DEFAULTS[ac]
            (ilo, ihi), (flo, fhi) = RATE_RAMP_RANGE[ac]
            ri = _da_num(s, _RAMP_INITIAL_KEY)     # numeric-or-None (bools->None)
            rf = _da_num(s, _RAMP_FINAL_KEY)
            initial = di if ri is None else min(ihi, max(ilo, ri))
            final = df if rf is None else min(fhi, max(flo, rf))
            ap = s.get(_RAMP_PITCH_KEY, True)
            adjust_pitch = ap if isinstance(ap, bool) else True
            return RateRamp(acronym=ac, initial=initial, final=final,
                            adjust_pitch=adjust_pitch)
    return None


def read_rate_ramp(osr_path: Path) -> "RateRamp | None":
    """The rate-ramp mod (WU/WD + initial/final/adjust_pitch) from a .osr, or
    None when the replay carries no ramp mod (or no ScoreInfo blob). Never
    raises."""
    mods = read_lazer_mods(osr_path)
    if not mods:
        return None
    return rate_ramp_from_mods(mods)


# --- Position mods: Mirror (MR) and Random (RD) ------------------------------
# osu.Game.Rulesets.Osu/Mods/OsuModMirror.cs — ``Reflection`` is a
# ``Bindable<MirrorType>()`` whose default (enum value 0) is Horizontal, so an
# ABSENT setting means Horizontal. The SettingSource "Flipped axes" serialises
# under the snake_case key ``reflection``; MirrorType.Horizontal=0 / Vertical=1
# / Both=2 (Newtonsoft writes the enum's integer value; a stringified name is
# tolerated too). Handled in beatmap/mods_position.apply_mirror.
MIRROR_ACRONYM = "MR"
_MIRROR_KEY = "reflection"
_MIRROR_BY_INT = {0: "horizontal", 1: "vertical", 2: "both"}


def mirror_from_mods(mods: list[dict]) -> "str | None":
    """The MR reflection axis (``"horizontal"``/``"vertical"``/``"both"``) from
    an already-parsed :func:`read_lazer_mods` list, or None when no MR mod is
    present. An absent/unrecognised ``reflection`` falls back to the enum
    default, Horizontal."""
    for m in mods:
        if isinstance(m, dict) and str(m.get("acronym", "")).upper() == MIRROR_ACRONYM:
            s = m.get("settings") or {}
            v = s.get(_MIRROR_KEY)
            if isinstance(v, bool):
                v = None
            if isinstance(v, (int, float)):
                return _MIRROR_BY_INT.get(int(v), "horizontal")
            if isinstance(v, str):
                key = v.strip().lower()
                if key in ("horizontal", "vertical", "both"):
                    return key
            return "horizontal"   # setting absent -> enum default (Horizontal)
    return None


def read_mirror(osr_path: Path) -> "str | None":
    """The MR reflection axis from a .osr, or None when the replay carries no
    Mirror mod (or no ScoreInfo blob). Never raises."""
    mods = read_lazer_mods(osr_path)
    if not mods:
        return None
    return mirror_from_mods(mods)


# osu.Game.Rulesets.Osu/Mods/OsuModRandom.cs + osu.Game/Rulesets/Mods/ModRandom.cs
# RD seeds .NET's ``System.Random`` with the ``Seed`` setting (Bindable<int?>);
# a replay ALWAYS persists the actual seed used (``Seed.Value ??= RNG.Next()``),
# so it can be reproduced. ``AngleSharpness`` (BindableFloat, default 7, [1,10])
# tunes the jump-angle distribution. Both live under the snake_case keys
# ``seed`` / ``angle_sharpness`` in the ScoreInfo blob settings.
RANDOM_ACRONYM = "RD"
_RANDOM_SEED_KEY = "seed"
_RANDOM_ANGLE_KEY = "angle_sharpness"
RANDOM_ANGLE_DEFAULT = 7.0


@dataclass(frozen=True)
class RandomMod:
    """The RD mod read from a .osr. ``seed`` is the .NET Random seed (None only
    when a replay somehow omitted it — then the positions can't be reproduced).
    ``angle_sharpness`` mirrors the AngleSharpness setting (default 7)."""
    seed: int | None
    angle_sharpness: float = RANDOM_ANGLE_DEFAULT


def random_from_mods(mods: list[dict]) -> "RandomMod | None":
    """The RD mod (seed + angle_sharpness) from an already-parsed
    :func:`read_lazer_mods` list, or None when no RD mod is present."""
    for m in mods:
        if isinstance(m, dict) and str(m.get("acronym", "")).upper() == RANDOM_ACRONYM:
            s = m.get("settings") or {}
            sv = s.get(_RANDOM_SEED_KEY)
            seed = (int(sv) if isinstance(sv, (int, float))
                    and not isinstance(sv, bool) else None)
            av = _da_num(s, _RANDOM_ANGLE_KEY)
            angle = av if av is not None else RANDOM_ANGLE_DEFAULT
            return RandomMod(seed=seed, angle_sharpness=angle)
    return None


def read_random(osr_path: Path) -> "RandomMod | None":
    """The RD mod (seed + angle_sharpness) from a .osr, or None when the replay
    carries no Random mod (or no ScoreInfo blob). Never raises."""
    mods = read_lazer_mods(osr_path)
    if not mods:
        return None
    return random_from_mods(mods)


# --- Transform-family "fun" mods: GR/DF/SI/WG/TR -----------------------------
# osu.Game.Rulesets.Osu/Mods/OsuModGrow.cs / OsuModDeflate.cs /
# OsuModSpinIn.cs / OsuModWiggle.cs / OsuModTransform.cs — a batch of per-object
# ENTRANCE-animation mods (ModType.Fun). They are purely VISUAL (the drawable's
# appear transform); none change object geometry, so a play only needs the
# acronym plus, for the two configurable ones, a single setting:
#   * GR / DF: ``StartScale`` (SettingSource "Starting Size") → key ``start_scale``.
#       GR default 0.5 [0, 0.99]; DF default 2.0 [1, 25].
#   * WG:      ``Strength`` → key ``strength``. Default 1.0 [0.1, 2.0].
#   * SI / TR: no settings (fixed animation constants).
# An absent/non-numeric setting falls back to the mod default (clamped to the
# SettingSource range), exactly like the DA / rate helpers above.
TRANSFORM_GROW = "GR"
TRANSFORM_DEFLATE = "DF"
TRANSFORM_SPIN_IN = "SI"
TRANSFORM_WIGGLE = "WG"
TRANSFORM_TRANSFORM = "TR"
TRANSFORM_MODS = frozenset({TRANSFORM_GROW, TRANSFORM_DEFLATE,
                            TRANSFORM_SPIN_IN, TRANSFORM_WIGGLE,
                            TRANSFORM_TRANSFORM})

_TRANSFORM_START_SCALE_KEY = "start_scale"
_TRANSFORM_STRENGTH_KEY = "strength"

# acronym -> StartScale default (OsuModGrow / OsuModDeflate BindableFloat)
TRANSFORM_START_SCALE_DEFAULT = {TRANSFORM_GROW: 0.5, TRANSFORM_DEFLATE: 2.0}
# acronym -> (MinValue, MaxValue) the SettingSource clamps StartScale to
TRANSFORM_START_SCALE_RANGE = {TRANSFORM_GROW: (0.0, 0.99),
                               TRANSFORM_DEFLATE: (1.0, 25.0)}
WIGGLE_STRENGTH_DEFAULT = 1.0
WIGGLE_STRENGTH_RANGE = (0.1, 2.0)


@dataclass(frozen=True)
class TransformMod:
    """A transform-family mod read from a .osr. ``acronym`` is one of
    GR/DF/SI/WG/TR. ``start_scale`` is the GR/DF starting size multiplier (1.0 —
    an identity — for SI/WG/TR, which have no scale setting). ``strength`` is
    the WG wiggle-strength multiplier (1.0 default; unused by the others)."""
    acronym: str
    start_scale: float = 1.0
    strength: float = WIGGLE_STRENGTH_DEFAULT


def transform_from_mods(mods: list[dict]) -> "TransformMod | None":
    """The transform-family mod from an already-parsed :func:`read_lazer_mods`
    list, or None when no GR/DF/SI/WG/TR mod is present. These five are
    mutually incompatible, so the first match wins. ``start_scale`` (GR/DF) and
    ``strength`` (WG) are clamped to the SettingSource range; an absent or
    non-numeric value falls back to the mod default."""
    for m in mods:
        if not isinstance(m, dict):
            continue
        ac = str(m.get("acronym", "")).upper()
        if ac not in TRANSFORM_MODS:
            continue
        s = m.get("settings") or {}
        start_scale = 1.0
        if ac in TRANSFORM_START_SCALE_DEFAULT:
            default = TRANSFORM_START_SCALE_DEFAULT[ac]
            raw = _da_num(s, _TRANSFORM_START_SCALE_KEY)   # numeric-or-None
            if raw is None:
                start_scale = default
            else:
                lo, hi = TRANSFORM_START_SCALE_RANGE[ac]
                start_scale = min(hi, max(lo, raw))
        strength = WIGGLE_STRENGTH_DEFAULT
        if ac == TRANSFORM_WIGGLE:
            raw = _da_num(s, _TRANSFORM_STRENGTH_KEY)
            if raw is not None:
                lo, hi = WIGGLE_STRENGTH_RANGE
                strength = min(hi, max(lo, raw))
        return TransformMod(acronym=ac, start_scale=start_scale,
                            strength=strength)
    return None


def read_transform(osr_path: Path) -> "TransformMod | None":
    """The transform-family mod (GR/DF/SI/WG/TR + its setting) from a .osr, or
    None when the replay carries none (or no ScoreInfo blob). Never raises."""
    mods = read_lazer_mods(osr_path)
    if not mods:
        return None
    return transform_from_mods(mods)


# --- Approach/circle-appearance mods: FR / AD / TC ---------------------------
# A batch of purely VISUAL osu!std mods that alter the approach circle or the
# hit-circle appearance without touching object geometry, the replay cursor or
# the judgement — so a play only needs the acronym (+ AD's two settings). Each
# is cited from osu.Game.Rulesets.Osu/Mods/ and ported in render/appearance_mods.py.
#
#   osu.Game.Rulesets.Osu/Mods/OsuModFreezeFrame.cs  (acronym "FR", ModType.Fun)
#       No settings. IApplicableToBeatmap extends every non-Spinner object's
#       TimePreempt by ``StartTime - lastNewComboTime`` (so a whole combo's
#       objects appear TOGETHER at the combo's first-object appear time and stay
#       "frozen"), and rescales each hit-circle approach circle to
#       ``4 * TimePreempt/originalPreempt`` shrinking linearly to 1 over the new
#       TimePreempt. Incompatible with AD/TR/Depth/HD.
#
#   osu.Game.Rulesets.Osu/Mods/OsuModApproachDifferent.cs  (acronym "AD")
#       ``Scale`` BindableFloat(4){MinValue=1.5, MaxValue=10, Precision=0.1}
#       (SettingSource "Initial size" → key ``scale``) sets the approach
#       circle's initial size; ``Style`` Bindable<AnimationStyle>(Gravity)
#       (SettingSource "Style" → key ``style``) picks the easing the circle
#       scales Scale→1 with over TimePreempt. AnimationStyle enum (order fixes
#       the serialised integer): Linear, Gravity, InOut1, InOut2, Accelerate1,
#       Accelerate2, Accelerate3, Decelerate1, Decelerate2, Decelerate3.
#
#   osu.Game.Rulesets.Osu/Mods/OsuModTraceable.cs  (acronym "TC",
#       ModType.DifficultyIncrease, IRequiresApproachCircles). No settings. Its
#       ApplyNormalVisibilityState hides each hit circle's whole CirclePiece
#       ("we only want to see the approach circle" — fill + ring + number all
#       gone; only the approach circle traces the position), hides the slider
#       tail, and turns the slider body outline-only (body AccentColour opacity
#       0, BorderColour = AccentColour). Slider heads (a DrawableHitCircle) and
#       repeat circle-pieces are hidden too; the repeat ARROW is kept.
FREEZE_FRAME_ACRONYM = "FR"
TRACEABLE_ACRONYM = "TC"
APPROACH_DIFFERENT_ACRONYM = "AD"


def _has_acronym(mods: list[dict], acronym: str) -> bool:
    """True iff ``mods`` (a :func:`read_lazer_mods` list) carries ``acronym``."""
    return any(isinstance(m, dict)
               and str(m.get("acronym", "")).upper() == acronym
               for m in mods)


def freeze_frame_from_mods(mods: list[dict]) -> bool:
    """True when the mod list carries Freeze Frame (FR). FR has no settings."""
    return _has_acronym(mods, FREEZE_FRAME_ACRONYM)


def read_freeze_frame(osr_path: Path) -> bool:
    """True iff the .osr's lazer mod list contains Freeze Frame (FR)."""
    mods = read_lazer_mods(osr_path)
    return bool(mods) and freeze_frame_from_mods(mods)


def traceable_from_mods(mods: list[dict]) -> bool:
    """True when the mod list carries Traceable (TC). TC has no settings."""
    return _has_acronym(mods, TRACEABLE_ACRONYM)


def read_traceable(osr_path: Path) -> bool:
    """True iff the .osr's lazer mod list contains Traceable (TC)."""
    mods = read_lazer_mods(osr_path)
    return bool(mods) and traceable_from_mods(mods)


# --- Approach Different (AD) settings ----------------------------------------
APPROACH_DIFFERENT_SCALE_DEFAULT = 4.0
APPROACH_DIFFERENT_SCALE_RANGE = (1.5, 10.0)   # BindableFloat(4){Min 1.5, Max 10}
_AD_SCALE_KEY = "scale"
_AD_STYLE_KEY = "style"
# The AnimationStyle enum, in declaration order — the index IS the integer
# Newtonsoft serialises for a Bindable<AnimationStyle>. Names are lower-cased
# so a stringified value ("Gravity"/"InOut1"/…) is accepted too.
APPROACH_DIFFERENT_STYLES = (
    "linear", "gravity", "inout1", "inout2",
    "accelerate1", "accelerate2", "accelerate3",
    "decelerate1", "decelerate2", "decelerate3",
)
APPROACH_DIFFERENT_STYLE_DEFAULT = "gravity"   # Bindable<AnimationStyle>(Gravity)


@dataclass(frozen=True)
class ApproachDifferent:
    """The AD config read from a .osr. ``scale`` is the approach circle's
    initial size multiplier (clamped to [1.5, 10]; default 4). ``style`` is one
    of :data:`APPROACH_DIFFERENT_STYLES` — the easing the approach circle
    shrinks Scale→1 with (default ``"gravity"`` == Easing.InBack)."""
    scale: float = APPROACH_DIFFERENT_SCALE_DEFAULT
    style: str = APPROACH_DIFFERENT_STYLE_DEFAULT


def _ad_style(settings: dict) -> str:
    """The AD AnimationStyle from the blob settings. Accepts the serialised
    enum integer (index into declaration order) or a string name; anything
    absent/unrecognised falls back to the enum default (Gravity)."""
    v = settings.get(_AD_STYLE_KEY)
    if isinstance(v, bool):
        return APPROACH_DIFFERENT_STYLE_DEFAULT
    if isinstance(v, (int, float)):
        i = int(v)
        if 0 <= i < len(APPROACH_DIFFERENT_STYLES):
            return APPROACH_DIFFERENT_STYLES[i]
        return APPROACH_DIFFERENT_STYLE_DEFAULT
    if isinstance(v, str):
        key = v.strip().lower()
        if key in APPROACH_DIFFERENT_STYLES:
            return key
    return APPROACH_DIFFERENT_STYLE_DEFAULT


def approach_different_from_mods(mods: list[dict]) -> "ApproachDifferent | None":
    """The AD config (scale + style) from an already-parsed
    :func:`read_lazer_mods` list, or None when no AD mod is present.
    ``scale`` is clamped to the SettingSource range; an absent/non-numeric
    value falls back to the default (4)."""
    for m in mods:
        if isinstance(m, dict) and str(m.get("acronym", "")).upper() == APPROACH_DIFFERENT_ACRONYM:
            s = m.get("settings") or {}
            raw = _da_num(s, _AD_SCALE_KEY)
            if raw is None:
                scale = APPROACH_DIFFERENT_SCALE_DEFAULT
            else:
                lo, hi = APPROACH_DIFFERENT_SCALE_RANGE
                scale = min(hi, max(lo, raw))
            return ApproachDifferent(scale=scale, style=_ad_style(s))
    return None


def read_approach_different(osr_path: Path) -> "ApproachDifferent | None":
    """The AD config (scale + style) from a .osr, or None when the replay
    carries no Approach Different mod (or no ScoreInfo blob). Never raises."""
    mods = read_lazer_mods(osr_path)
    if not mods:
        return None
    return approach_different_from_mods(mods)


# --- Screen / cursor-effect visual mods: BR/BM/SY/BL/NS/DP/BU ----------------
# A batch of purely VISUAL osu!std mods that alter the SCREEN, the CURSOR or a
# per-object DRAW transform without touching object geometry, the replay cursor
# or the judgement — so every one is reconcile-exact (the ruleset simulates the
# untouched beatmap + untouched replay frames). Each is cited from
# osu.Game.Rulesets.Osu/Mods/ (BR/NS also from the osu.Game base) and ported in
# render/screen_mods.py + the scene. Unlike the transform family these do not
# share one hook, so each carries its own reader + settings below.
#
#   osu.Game.Rulesets.Osu/Mods/OsuModBarrelRoll.cs + osu.Game/Rulesets/Mods/
#   ModBarrelRoll.cs  (acronym "BR", ModType.Fun)
#       SpinSpeed BindableDouble(0.5){Min 0.02, Max 12, Precision 0.01} (rev/min,
#       SettingSource "Roll speed" -> key ``spin_speed``); Direction
#       Bindable<RotationDirection>() default Clockwise (SettingSource
#       "Direction" -> key ``direction``; RotationDirection.Clockwise=0 /
#       Counterclockwise=1). Update: playfield rotation (deg) = CurrentRotation =
#       (Direction==CCW ? -1 : 1) * 360 * (time/60000 * SpinSpeed); the whole
#       playfield container spins, the cursor + circle number are counter-rotated
#       to stay upright. ApplyToDrawableRuleset scales the playfield by
#       minSide/maxSide so rotated objects stay on-screen. Incompatible w/ BU.
#
#   osu.Game.Rulesets.Osu/Mods/OsuModBloom.cs  (acronym "BM", ModType.Fun)
#       MaxSizeComboCount BindableInt(50){Min 5, Max 100} (SettingSource "Max
#       size at combo" -> key ``max_size_combo_count``); MaxCursorSize
#       BindableFloat(10){Min 5, Max 15, Precision 0.5} (SettingSource "Final
#       size multiplier" -> key ``max_cursor_size``). currentSize = clamp(
#       MaxCursorSize * combo/MaxSizeComboCount, MIN_SIZE=1, MaxCursorSize); the
#       cursor's ModScaleAdjust lerps toward it over TRANSITION_DURATION=100ms
#       (reset to 1 during breaks). Incompatible w/ FL, NS, TouchDevice.
#
#   osu.Game/Rulesets/Mods/ModSynesthesia.cs + osu.Game.Rulesets.Osu/Mods/
#   OsuModSynesthesia.cs  (acronym "SY", ModType.Fun). No settings. Overrides
#       each object's AccentColour from its beat-snap divisor at StartTime
#       (slider tail uses its slider's end time): BindableBeatDivisor.
#       GetColourFor(ControlPointInfo.GetClosestBeatDivisor(t)). Palette ported
#       in render/screen_mods.py.
#
#   osu.Game.Rulesets.Osu/Mods/OsuModBlinds.cs  (acronym "BL",
#   ModType.DifficultyIncrease). No settings. Two black panels close in from the
#       left/right screen edges; the covered width is health-driven (higher HP =
#       MORE closed) with a break/start widening. DrawableOsuBlinds constants +
#       curve ported in render/screen_mods.py. Incompatible w/ FL.
#
#   osu.Game/Rulesets/Mods/ModNoScope.cs + osu.Game.Rulesets.Osu/Mods/
#   OsuModNoScope.cs  (acronym "NS", ModType.Fun)
#       HiddenComboCount BindableInt(10){Min 0, Max 50} (SettingSource "Hidden at
#       combo" -> key ``hidden_combo_count``; 0 = always hidden). ComboBasedAlpha
#       = max(MIN_ALPHA=0.0002, 1 - combo/HiddenComboCount) fades ONLY the cursor
#       (lerp over 100ms), forced to 1 during breaks + spinners. Incompatible w/ BM.
#
#   osu.Game.Rulesets.Osu/Mods/OsuModDepth.cs  (acronym "DP", ModType.Fun)
#       MaxDepth BindableFloat(100){Min 50, Max 200, Precision 10} (SettingSource
#       "Maximum depth" -> key ``max_depth``); ShowApproachCircles
#       BindableBool(true) (SettingSource "Show Approach Circles" -> key
#       ``show_approach_circles``). Objects approach from far (small, offset
#       toward the playfield-centre vanishing point) growing to real size/pos at
#       the hit. Incompatible w/ MG/RP/FR/other visibility mods.
#
#   osu.Game.Rulesets.Osu/Mods/OsuModBubbles.cs  (acronym "BU", ModType.Fun). No
#       settings. On each successful/missed hit of a circle / slider head /
#       spinner an expanding, fading bubble spawns at the hit position, tinted by
#       the object's combo colour; maxSize = min(1.75, 1.25 + 0.005*combo).
#       Incompatible w/ BR/MG/RP.

BARREL_ROLL_ACRONYM = "BR"
BLOOM_ACRONYM = "BM"
SYNESTHESIA_ACRONYM = "SY"
BLINDS_ACRONYM = "BL"
NO_SCOPE_ACRONYM = "NS"
DEPTH_ACRONYM = "DP"
BUBBLES_ACRONYM = "BU"


# --- BR: Barrel Roll ---------------------------------------------------------
_BR_SPEED_KEY = "spin_speed"
_BR_DIRECTION_KEY = "direction"
BARREL_ROLL_SPEED_DEFAULT = 0.5             # BindableDouble(0.5) rev/min
BARREL_ROLL_SPEED_RANGE = (0.02, 12.0)      # {Min 0.02, Max 12}
_BR_DIR_CCW = 1                             # RotationDirection.Counterclockwise


@dataclass(frozen=True)
class BarrelRoll:
    """The BR config read from a .osr. ``spin_speed`` is rev/min (clamped to
    [0.02, 12]; default 0.5). ``direction`` is +1 for Clockwise (the enum
    default) or -1 for Counterclockwise — the sign of CurrentRotation in
    ModBarrelRoll.Update."""
    spin_speed: float = BARREL_ROLL_SPEED_DEFAULT
    direction: int = 1     # +1 clockwise, -1 counterclockwise


def barrel_roll_from_mods(mods: list[dict]) -> "BarrelRoll | None":
    """The BR config (spin_speed + direction) from an already-parsed
    :func:`read_lazer_mods` list, or None when no BR mod is present.
    ``spin_speed`` is clamped to the SettingSource range; an absent/non-numeric
    value falls back to the default (0.5). ``direction`` reads the
    RotationDirection enum (0/"clockwise" -> +1, 1/"counterclockwise" -> -1)."""
    for m in mods:
        if isinstance(m, dict) and str(m.get("acronym", "")).upper() == BARREL_ROLL_ACRONYM:
            s = m.get("settings") or {}
            raw = _da_num(s, _BR_SPEED_KEY)
            if raw is None:
                spin = BARREL_ROLL_SPEED_DEFAULT
            else:
                lo, hi = BARREL_ROLL_SPEED_RANGE
                spin = min(hi, max(lo, raw))
            dv = s.get(_BR_DIRECTION_KEY)
            direction = 1
            if isinstance(dv, bool):
                direction = 1
            elif isinstance(dv, (int, float)):
                direction = -1 if int(dv) == _BR_DIR_CCW else 1
            elif isinstance(dv, str):
                k = dv.strip().lower()
                if k in ("counterclockwise", "ccw", "anticlockwise"):
                    direction = -1
            return BarrelRoll(spin_speed=spin, direction=direction)
    return None


def read_barrel_roll(osr_path: Path) -> "BarrelRoll | None":
    """The BR config from a .osr, or None when the replay carries no Barrel
    Roll mod (or no ScoreInfo blob). Never raises."""
    mods = read_lazer_mods(osr_path)
    if not mods:
        return None
    return barrel_roll_from_mods(mods)


# --- BM: Bloom ---------------------------------------------------------------
_BM_COMBO_KEY = "max_size_combo_count"
_BM_SIZE_KEY = "max_cursor_size"
BLOOM_COMBO_DEFAULT = 50                    # BindableInt(50)
BLOOM_COMBO_RANGE = (5, 100)               # {Min 5, Max 100}
BLOOM_SIZE_DEFAULT = 10.0                  # BindableFloat(10)
BLOOM_SIZE_RANGE = (5.0, 15.0)            # {Min 5, Max 15}


@dataclass(frozen=True)
class Bloom:
    """The BM config read from a .osr. ``max_size_combo_count`` is the combo at
    which the cursor reaches its maximum (default 50, [5,100]).
    ``max_cursor_size`` is that maximum cursor-scale multiplier (default 10,
    [5,15])."""
    max_size_combo_count: int = BLOOM_COMBO_DEFAULT
    max_cursor_size: float = BLOOM_SIZE_DEFAULT


def bloom_from_mods(mods: list[dict]) -> "Bloom | None":
    """The BM config from an already-parsed :func:`read_lazer_mods` list, or
    None when no BM mod is present. Both settings clamp to their SettingSource
    range; an absent/non-numeric value falls back to the default."""
    for m in mods:
        if isinstance(m, dict) and str(m.get("acronym", "")).upper() == BLOOM_ACRONYM:
            s = m.get("settings") or {}
            rc = _da_num(s, _BM_COMBO_KEY)
            if rc is None:
                combo = BLOOM_COMBO_DEFAULT
            else:
                lo, hi = BLOOM_COMBO_RANGE
                combo = int(min(hi, max(lo, rc)))
            rz = _da_num(s, _BM_SIZE_KEY)
            if rz is None:
                size = BLOOM_SIZE_DEFAULT
            else:
                lo, hi = BLOOM_SIZE_RANGE
                size = min(hi, max(lo, rz))
            return Bloom(max_size_combo_count=combo, max_cursor_size=size)
    return None


def read_bloom(osr_path: Path) -> "Bloom | None":
    """The BM config from a .osr, or None when the replay carries no Bloom mod
    (or no ScoreInfo blob). Never raises."""
    mods = read_lazer_mods(osr_path)
    if not mods:
        return None
    return bloom_from_mods(mods)


# --- SY: Synesthesia (no settings) -------------------------------------------
def synesthesia_from_mods(mods: list[dict]) -> bool:
    """True when the mod list carries Synesthesia (SY). SY has no settings."""
    return _has_acronym(mods, SYNESTHESIA_ACRONYM)


def read_synesthesia(osr_path: Path) -> bool:
    """True iff the .osr's lazer mod list contains Synesthesia (SY)."""
    mods = read_lazer_mods(osr_path)
    return bool(mods) and synesthesia_from_mods(mods)


# --- BL: Blinds (no settings) ------------------------------------------------
def blinds_from_mods(mods: list[dict]) -> bool:
    """True when the mod list carries Blinds (BL). BL has no settings."""
    return _has_acronym(mods, BLINDS_ACRONYM)


def read_blinds(osr_path: Path) -> bool:
    """True iff the .osr's lazer mod list contains Blinds (BL)."""
    mods = read_lazer_mods(osr_path)
    return bool(mods) and blinds_from_mods(mods)


# --- NS: No Scope ------------------------------------------------------------
_NS_COMBO_KEY = "hidden_combo_count"
NO_SCOPE_COMBO_DEFAULT = 10                 # OsuModNoScope BindableInt(10)
NO_SCOPE_COMBO_RANGE = (0, 50)             # {Min 0, Max 50}


@dataclass(frozen=True)
class NoScope:
    """The NS config read from a .osr. ``hidden_combo_count`` is the combo at
    which the cursor becomes fully hidden (default 10, [0,50]); 0 means the
    cursor is ALWAYS hidden (ModNoScope's ApplyToScoreProcessor early-out)."""
    hidden_combo_count: int = NO_SCOPE_COMBO_DEFAULT


def no_scope_from_mods(mods: list[dict]) -> "NoScope | None":
    """The NS config from an already-parsed :func:`read_lazer_mods` list, or
    None when no NS mod is present. ``hidden_combo_count`` clamps to [0,50]; an
    absent/non-numeric value falls back to the default (10)."""
    for m in mods:
        if isinstance(m, dict) and str(m.get("acronym", "")).upper() == NO_SCOPE_ACRONYM:
            s = m.get("settings") or {}
            rc = _da_num(s, _NS_COMBO_KEY)
            if rc is None:
                combo = NO_SCOPE_COMBO_DEFAULT
            else:
                lo, hi = NO_SCOPE_COMBO_RANGE
                combo = int(min(hi, max(lo, rc)))
            return NoScope(hidden_combo_count=combo)
    return None


def read_no_scope(osr_path: Path) -> "NoScope | None":
    """The NS config from a .osr, or None when the replay carries no No Scope
    mod (or no ScoreInfo blob). Never raises."""
    mods = read_lazer_mods(osr_path)
    if not mods:
        return None
    return no_scope_from_mods(mods)


# --- DP: Depth ---------------------------------------------------------------
_DP_DEPTH_KEY = "max_depth"
_DP_APPROACH_KEY = "show_approach_circles"
DEPTH_MAX_DEFAULT = 100.0                   # BindableFloat(100)
DEPTH_MAX_RANGE = (50.0, 200.0)           # {Min 50, Max 200}


@dataclass(frozen=True)
class Depth:
    """The DP config read from a .osr. ``max_depth`` is how far away objects
    appear (default 100, [50,200]). ``show_approach_circles`` mirrors the
    BindableBool(true) toggle (approach rings hidden when False)."""
    max_depth: float = DEPTH_MAX_DEFAULT
    show_approach_circles: bool = True


def depth_from_mods(mods: list[dict]) -> "Depth | None":
    """The DP config from an already-parsed :func:`read_lazer_mods` list, or
    None when no DP mod is present. ``max_depth`` clamps to [50,200]; an
    absent/non-numeric value falls back to the default (100).
    ``show_approach_circles`` defaults to True (absent -> True)."""
    for m in mods:
        if isinstance(m, dict) and str(m.get("acronym", "")).upper() == DEPTH_ACRONYM:
            s = m.get("settings") or {}
            rd = _da_num(s, _DP_DEPTH_KEY)
            if rd is None:
                depth = DEPTH_MAX_DEFAULT
            else:
                lo, hi = DEPTH_MAX_RANGE
                depth = min(hi, max(lo, rd))
            av = s.get(_DP_APPROACH_KEY, True)
            show = av if isinstance(av, bool) else True
            return Depth(max_depth=depth, show_approach_circles=show)
    return None


def read_depth(osr_path: Path) -> "Depth | None":
    """The DP config from a .osr, or None when the replay carries no Depth mod
    (or no ScoreInfo blob). Never raises."""
    mods = read_lazer_mods(osr_path)
    if not mods:
        return None
    return depth_from_mods(mods)


# --- BU: Bubbles (no settings) -----------------------------------------------
def bubbles_from_mods(mods: list[dict]) -> bool:
    """True when the mod list carries Bubbles (BU). BU has no settings."""
    return _has_acronym(mods, BUBBLES_ACRONYM)


def read_bubbles(osr_path: Path) -> bool:
    """True iff the .osr's lazer mod list contains Bubbles (BU)."""
    mods = read_lazer_mods(osr_path)
    return bool(mods) and bubbles_from_mods(mods)


# --- MG / RP: Magnetised + Repel (cursor-driven object movement) -------------
# osu.Game.Rulesets.Osu/Mods/OsuModMagnetised.cs / OsuModRepel.cs — the two
# "fun" mods that EASE every alive object toward (MG) / away from (RP) the
# cursor each frame. Each carries a single strength setting:
#   * MG: ``AttractionStrength`` (SettingSource "Attraction strength") →
#     snake_case key ``attraction_strength``.
#   * RP: ``RepulsionStrength``  (SettingSource "Repulsion strength")  →
#     snake_case key ``repulsion_strength``.
# Both BindableFloat(0.5){MinValue 0.05, MaxValue 1.0, Precision 0.05}. An
# absent/non-numeric setting falls back to the default (clamped to the range),
# exactly like the DA / rate / transform helpers above. MG and RP are mutually
# incompatible, so a play carries at most one. The per-frame easing math is
# ported in render/repel_magnet.py; the scene integrates it statefully.
MAGNETISED_ACRONYM = "MG"
REPEL_ACRONYM = "RP"
REPEL_MAGNET_MODS = frozenset({MAGNETISED_ACRONYM, REPEL_ACRONYM})

_MG_STRENGTH_KEY = "attraction_strength"
_RP_STRENGTH_KEY = "repulsion_strength"
REPEL_MAGNET_STRENGTH_DEFAULT = 0.5             # BindableFloat(0.5)
REPEL_MAGNET_STRENGTH_RANGE = (0.05, 1.0)       # {Min 0.05, Max 1.0}
# acronym -> the setting key its strength serialises under
_REPEL_MAGNET_KEY = {MAGNETISED_ACRONYM: _MG_STRENGTH_KEY,
                     REPEL_ACRONYM: _RP_STRENGTH_KEY}


@dataclass(frozen=True)
class RepelMagnet:
    """The MG/RP mod read from a .osr. ``acronym`` is ``"MG"`` (Magnetised —
    objects pulled TO the cursor) or ``"RP"`` (Repel — objects pushed AWAY).
    ``strength`` is the Attraction/Repulsion strength (default 0.5, clamped to
    [0.05, 1.0]); higher = a stronger, snappier pull/push."""
    acronym: str
    strength: float = REPEL_MAGNET_STRENGTH_DEFAULT


def repel_magnet_from_mods(mods: list[dict]) -> "RepelMagnet | None":
    """The MG/RP mod from an already-parsed :func:`read_lazer_mods` list, or
    None when neither is present. MG and RP are mutually incompatible, so the
    first match wins. ``strength`` is clamped to [0.05, 1.0]; an absent or
    non-numeric value falls back to the default (0.5)."""
    for m in mods:
        if not isinstance(m, dict):
            continue
        ac = str(m.get("acronym", "")).upper()
        if ac not in REPEL_MAGNET_MODS:
            continue
        s = m.get("settings") or {}
        raw = _da_num(s, _REPEL_MAGNET_KEY[ac])     # numeric-or-None (bools->None)
        if raw is None:
            strength = REPEL_MAGNET_STRENGTH_DEFAULT
        else:
            lo, hi = REPEL_MAGNET_STRENGTH_RANGE
            strength = min(hi, max(lo, raw))
        return RepelMagnet(acronym=ac, strength=strength)
    return None


def read_repel_magnet(osr_path: Path) -> "RepelMagnet | None":
    """The MG/RP mod (acronym + strength) from a .osr, or None when the replay
    carries neither Magnetised nor Repel (or no ScoreInfo blob). Never raises."""
    mods = read_lazer_mods(osr_path)
    if not mods:
        return None
    return repel_magnet_from_mods(mods)


# osu.Game.Rulesets.Osu/Mods/OsuModTargetPractice.cs
# TP is a CONVERSION mod (ModType.Conversion): it discards the map's objects and
# rebuilds it as seeded "target" hit circles placed on the beat. Like OsuModRandom
# it is IHasSeed — ``ApplyToBeatmap`` does ``Seed.Value ??= RNG.Next()`` before
# ``new Random(Seed.Value.Value)``, so a replay ALWAYS persists the actual seed
# under the snake_case ``seed`` key in the ScoreInfo blob settings; without it the
# target layout can't be reproduced. ``Metronome`` (BindableBool, default true) is
# an audio-only tick overlay — read for completeness, it never moves a target.
TARGET_PRACTICE_ACRONYM = "TP"
_TP_SEED_KEY = "seed"
_TP_METRONOME_KEY = "metronome"


@dataclass(frozen=True)
class TargetPractice:
    """The TP mod read from a .osr. ``seed`` is the .NET Random seed (None only
    when a replay somehow omitted it — then the target positions can't be
    reproduced and TP must not be rendered). ``metronome`` mirrors the audio
    metronome-tick toggle (default True)."""
    seed: int | None
    metronome: bool = True


def target_practice_from_mods(mods: list[dict]) -> "TargetPractice | None":
    """The TP mod (seed + metronome) from an already-parsed
    :func:`read_lazer_mods` list, or None when no TP mod is present."""
    for m in mods:
        if isinstance(m, dict) and str(m.get("acronym", "")).upper() == TARGET_PRACTICE_ACRONYM:
            s = m.get("settings") or {}
            sv = s.get(_TP_SEED_KEY)
            seed = (int(sv) if isinstance(sv, (int, float))
                    and not isinstance(sv, bool) else None)
            mv = s.get(_TP_METRONOME_KEY)
            metronome = bool(mv) if isinstance(mv, bool) else True
            return TargetPractice(seed=seed, metronome=metronome)
    return None


def read_target_practice(osr_path: Path) -> "TargetPractice | None":
    """The TP mod (seed + metronome) from a .osr, or None when the replay
    carries no Target Practice mod (or no ScoreInfo blob). Never raises."""
    mods = read_lazer_mods(osr_path)
    if not mods:
        return None
    return target_practice_from_mods(mods)
