"""Per-video-frame control curves derived from the audio.

Everything here resamples from the analysis frame rate (about 43 Hz) to the video
frame rate and returns a plain list of floats, one per frame, so it can be wired
into any node that takes a float list: denoise, guidance, controlnet strength,
IPAdapter weight, a zoom amount.

Curves are always exactly ``total_frames`` long. A curve one frame short of the
render silently misaligns everything after the first shot.
"""

from __future__ import annotations

import numpy as np

from .analysis import AudioAnalysis, envelope_follower, _normalise


def _resample_to_frames(v: np.ndarray, src_fps: float, n_frames: int, dst_fps: float) -> np.ndarray:
    """Linear resample from analysis rate to video rate, exact length."""
    if v.size == 0 or n_frames <= 0:
        return np.zeros(max(n_frames, 0), dtype=np.float32)
    src_t = np.arange(v.size) / src_fps
    dst_t = np.arange(n_frames) / dst_fps
    return np.interp(dst_t, src_t, v).astype(np.float32)


def band_curve(
    analysis: AudioAnalysis,
    total_frames: int,
    video_fps: float,
    band: str = "kick",
    attack_s: float = 0.01,
    release_s: float = 0.18,
    floor: float = 0.0,
    ceiling: float = 1.0,
    gamma: float = 1.0,
) -> list[float]:
    """Smoothed energy of one frequency band, mapped to [floor, ceiling].

    ``gamma`` above 1 sharpens the peaks, which is usually what you want when the
    curve drives something visible: a linear energy curve reads as a constant
    wobble rather than as hits.
    """
    env = analysis.bands.get(band)
    if env is None:
        raise KeyError(f"unknown band {band!r}, have {sorted(analysis.bands)}")
    smooth = envelope_follower(env, analysis.fps, attack_s, release_s)
    smooth = _normalise(smooth)
    if gamma != 1.0:
        smooth = np.power(np.clip(smooth, 0.0, 1.0), gamma)
    out = _resample_to_frames(smooth, analysis.fps, total_frames, video_fps)
    return (floor + (ceiling - floor) * np.clip(out, 0.0, 1.0)).astype(float).tolist()


def pulse_curve(
    analysis: AudioAnalysis,
    total_frames: int,
    video_fps: float,
    source: str = "beat",
    decay_s: float = 0.18,
    floor: float = 0.0,
    ceiling: float = 1.0,
    weight_by_strength: bool = True,
) -> list[float]:
    """A decaying spike on every beat, bar line, or detected transient.

    This is the curve to reach for when something should punch on the beat rather
    than follow the music's level. ``source`` is "beat", "downbeat", or the name
    of a transient band such as "kick" or "high".
    """
    if source == "beat":
        times, strength = analysis.beats, analysis.beat_strength
    elif source in ("downbeat", "bar"):
        times = analysis.downbeats
        strength = np.ones(len(times))
    elif source in analysis.transients:
        times = analysis.transients[source]
        strength = np.ones(len(times))
    else:
        raise KeyError(f"unknown pulse source {source!r}")

    out = np.zeros(max(total_frames, 0), dtype=np.float32)
    if out.size == 0 or len(times) == 0:
        return out.astype(float).tolist()

    if not weight_by_strength or len(strength) != len(times):
        strength = np.ones(len(times))
    else:
        strength = _normalise(np.asarray(strength, dtype=np.float64)) * 0.7 + 0.3

    decay_frames = max(decay_s * video_fps, 1e-6)
    for t, s in zip(times, strength):
        i = int(round(t * video_fps))
        if i >= out.size or i < 0:
            continue
        n = min(int(decay_frames * 5) + 1, out.size - i)
        tail = float(s) * np.exp(-np.arange(n) / decay_frames)
        np.maximum(out[i : i + n], tail, out=out[i : i + n])

    return (floor + (ceiling - floor) * np.clip(out, 0.0, 1.0)).astype(float).tolist()


def section_curve(
    analysis: AudioAnalysis,
    total_frames: int,
    video_fps: float,
    floor: float = 0.0,
    ceiling: float = 1.0,
) -> list[float]:
    """A step per section, held at that section's energy. Good for slow drifts."""
    out = np.zeros(max(total_frames, 0), dtype=np.float32)
    for i, (a, b) in enumerate(zip(analysis.sections[:-1], analysis.sections[1:])):
        ia = max(0, int(a * video_fps))
        ib = min(out.size, int(b * video_fps))
        if ib > ia and i < analysis.section_energy.size:
            out[ia:ib] = float(analysis.section_energy[i])
    return (floor + (ceiling - floor) * np.clip(out, 0.0, 1.0)).astype(float).tolist()


def zoom_punch_curve(
    analysis: AudioAnalysis,
    total_frames: int,
    video_fps: float,
    amount: float = 0.06,
    decay_s: float = 0.16,
    source: str = "kick",
) -> list[float]:
    """Scale factor per frame: 1.0 at rest, 1.0 + amount on a hit.

    Feed straight into an image scale. Kept separate from ``pulse_curve`` because
    a scale factor centred on 1.0 is a different thing from a 0..1 weight, and
    conflating them is how you get a shot that scales to zero on the downbeat.
    """
    pulse = np.asarray(
        pulse_curve(analysis, total_frames, video_fps, source=source, decay_s=decay_s)
    )
    return (1.0 + amount * pulse).astype(float).tolist()
