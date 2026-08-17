"""Planning for the AnimateDiff text-to-video stack.

This is the other half of the pack, and it makes a different kind of video from
the shot-based one. There, each shot is an independent clip and the music decides
where to cut. Here the picture is one continuous diffusion over the whole track,
and the music drives how much it moves and how tightly it holds together.

The mapping, which is the whole idea:

    energy  ->  motion scale        louder moves more
    energy  ->  context reuse       quiet holds still, loud is free to change
    bars    ->  prompt keyframes    the picture turns over on the bar

AnimateDiff has no notion of a cut. A hard cut is expressed as a prompt keyframe
with interpolation disabled, which reads as a fast turn rather than a splice.
That is the honest limit of the technique, not a bug in this file.

Frame arithmetic follows the same rule as the shot planner: segment lengths are
differences between cumulative frame positions, never a sum of rounded durations.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from .analysis import AudioAnalysis, envelope_follower, _normalise
from .plan import Shot, ShotPlan, _stable_seed


# AnimateDiff's motion modules are trained on 16-frame windows. Straying far from
# that costs coherence, so the default is to pick a generation frame rate that
# makes one musical bar land on 16 frames rather than to stretch the window.
NATIVE_CONTEXT = 16


def suggest_gen_fps(analysis: AudioAnalysis, context_length: int = NATIVE_CONTEXT) -> float:
    """Frame rate at which one bar is exactly ``context_length`` frames.

    Aligning the temporal window to the bar means the model's attention window
    and the music's phrase boundary are the same thing, so motion tends to land
    on the bar instead of drifting across it.
    """
    bar_seconds = 60.0 * analysis.beats_per_bar / max(analysis.tempo, 1e-6)
    return context_length / max(bar_seconds, 1e-6)


def plan_segments(
    analysis: AudioAnalysis,
    gen_fps: float,
    width: int = 512,
    height: int = 512,
    max_segment_seconds: float = 24.0,
    min_segment_seconds: float = 4.0,
    cut_on_sections: bool = True,
    seed: int = 0,
) -> ShotPlan:
    """Split the track into render segments, on section boundaries where possible.

    Segments exist because a whole song will not fit in one sampling pass: the
    latent batch and the decoded frames both scale with length. They are placed on
    section boundaries so the seams land where the music already changes.
    """
    duration = analysis.duration
    total_frames = int(round(duration * gen_fps))
    bar_seconds = 60.0 * analysis.beats_per_bar / max(analysis.tempo, 1e-6)

    bars = np.asarray(analysis.downbeats, dtype=np.float64)
    bars = bars[(bars >= 0.0) & (bars < duration)]
    if bars.size == 0 or bars[0] > 1e-6:
        bars = np.concatenate([[0.0], bars])

    # Candidate cut times: section boundaries, else every max_segment_seconds,
    # snapped to the nearest bar so a seam never lands mid-phrase.
    if cut_on_sections and len(analysis.sections) > 2:
        candidates = list(analysis.sections[1:-1])
    else:
        candidates = list(np.arange(max_segment_seconds, duration, max_segment_seconds))

    cuts = [0.0]
    for t in candidates:
        snapped = float(bars[int(np.argmin(np.abs(bars - t)))])
        if snapped - cuts[-1] >= min_segment_seconds:
            cuts.append(snapped)

    # Split anything still longer than the cap, again on bar lines.
    split: list[float] = []
    for a, b in zip(cuts, cuts[1:] + [duration]):
        split.append(a)
        span = b - a
        if span > max_segment_seconds:
            n = int(np.ceil(span / max_segment_seconds))
            step_bars = max(1, int(round((span / n) / bar_seconds)))
            t = a
            while True:
                t = t + step_bars * bar_seconds
                if b - t < min_segment_seconds:
                    break
                split.append(float(bars[int(np.argmin(np.abs(bars - t)))]))
    cuts = sorted(set(split))

    # Cumulative frames, then differences.
    edges = [int(round(t * gen_fps)) for t in cuts] + [total_frames]
    edges[0] = 0
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1
    edges[-1] = max(total_frames, edges[-2] + 1)

    overall = _combined_energy(analysis)
    a_fps = analysis.fps

    shots: list[Shot] = []
    for idx, (f0, f1) in enumerate(zip(edges[:-1], edges[1:])):
        t0, t1 = f0 / gen_fps, f1 / gen_fps
        i0, i1 = int(t0 * a_fps), max(int(t1 * a_fps), int(t0 * a_fps) + 1)
        seg = overall[i0:i1]
        shots.append(
            Shot(
                index=idx,
                start_time=t0,
                end_time=t1,
                start_frame=f0,
                frame_count=f1 - f0,
                # No 8n+1 rule here; AnimateDiff takes any batch size.
                gen_frames=f1 - f0,
                bars=(t1 - t0) / max(bar_seconds, 1e-6),
                section=int(np.searchsorted(analysis.sections, t0, side="right") - 1),
                energy=float(seg.mean()) if seg.size else 0.0,
                peak_energy=float(seg.max()) if seg.size else 0.0,
                seed=_stable_seed(seed, idx),
            )
        )

    return ShotPlan(
        shots=shots,
        video_fps=gen_fps,
        total_frames=total_frames,
        duration=duration,
        width=width,
        height=height,
        tempo=analysis.tempo,
        beats_per_bar=analysis.beats_per_bar,
    )


def _combined_energy(analysis: AudioAnalysis) -> np.ndarray:
    return _normalise(
        0.5 * analysis.bands["kick"]
        + 0.3 * analysis.bands["mid"]
        + 0.2 * analysis.bands["high"]
    )


def _resample(v: np.ndarray, src_fps: float, n: int, dst_fps: float) -> np.ndarray:
    if v.size == 0 or n <= 0:
        return np.zeros(max(n, 0), dtype=np.float32)
    src_t = np.arange(v.size) / src_fps
    dst_t = np.arange(n) / dst_fps
    return np.interp(dst_t, src_t, v).astype(np.float32)


def motion_curve(
    analysis: AudioAnalysis,
    n_frames: int,
    gen_fps: float,
    band: str = "kick",
    low: float = 0.75,
    high: float = 1.35,
    attack_s: float = 0.02,
    release_s: float = 0.25,
    gamma: float = 1.4,
) -> list[float]:
    """Per-frame motion scale. Feeds scale_multival on the AnimateDiff model.

    1.0 is the model's own amount of movement. Below it the picture settles,
    above it moves more. The useful range is narrow: past about 1.5 the motion
    module stops producing motion and starts producing smear.
    """
    env = analysis.bands.get(band)
    if env is None:
        raise KeyError(f"unknown band {band!r}")
    smooth = _normalise(envelope_follower(env, analysis.fps, attack_s, release_s))
    if gamma != 1.0:
        smooth = np.power(np.clip(smooth, 0.0, 1.0), gamma)
    out = _resample(smooth, analysis.fps, n_frames, gen_fps)
    return (low + (high - low) * np.clip(out, 0.0, 1.0)).astype(float).tolist()


def coherence_curve(
    analysis: AudioAnalysis,
    n_frames: int,
    gen_fps: float,
    low: float = 0.82,
    high: float = 1.0,
    smooth_s: float = 0.6,
) -> list[float]:
    """Per-frame temporal coherence. Feeds effect_multival on the motion model.

    effect_multival scales how strongly the motion module binds frames to each
    other. At 1.0 the temporal attention is at full strength and the picture is
    smooth; lower it and frames drift apart, which reads as churn. It is the
    honest per-frame coherence control on this stack.

    Deliberately the inverse of energy: quiet passages hold together, loud ones
    are free to change. ``low`` applies at peak energy.

    The obvious alternative, a per-frame strength_multival on
    ADE_ContextExtras_NaiveReuse, does not work in AnimateDiff-Evolved as of this
    writing. resize_multival returns a 3D [batch, height, width] mask while
    NaiveReuse's arithmetic needs a 4D one, so broadcasting lines the batch
    dimension up against the latent's channels and it dies with a size mismatch.
    Pass NaiveReuse a plain float instead.

    Keep the range narrow. Below about 0.7 the motion module stops holding the
    picture together and the result is flicker rather than energy.
    """
    energy = _combined_energy(analysis)
    w = max(1, int(round(smooth_s * analysis.fps)))
    kernel = np.ones(w) / w
    smoothed = np.convolve(np.pad(energy, (w // 2, w // 2), mode="edge"), kernel, mode="valid")
    smoothed = _normalise(smoothed[: energy.size])
    inverted = 1.0 - smoothed
    out = _resample(inverted, analysis.fps, n_frames, gen_fps)
    return (low + (high - low) * np.clip(out, 0.0, 1.0)).astype(float).tolist()


def build_prompt_schedule(
    plan: ShotPlan,
    analysis: AudioAnalysis,
    master_prompt: str,
    look: str = "",
    variations: Optional[list[str]] = None,
    bars_per_keyframe: int = 2,
    hard_cuts: bool = False,
    seed: int = 0,
) -> list[str]:
    """One ADE prompt-schedule string per segment, indexed from that segment's
    frame zero.

    ``hard_cuts`` writes each index with a trailing colon, which tells ADE to hold
    the prompt rather than interpolate toward the next one. That is as close to a
    cut as this stack gets: the picture turns over in a frame or two instead of
    dissolving across the bar. Without it, prompts crossfade, which is the look
    the technique is actually good at.
    """
    from .prompts import FRAMINGS, LIGHTING, _pick

    variations = variations or FRAMINGS
    bar_seconds = 60.0 * analysis.beats_per_bar / max(analysis.tempo, 1e-6)
    keyframe_seconds = max(bar_seconds * max(bars_per_keyframe, 1), 1e-6)

    master = master_prompt.strip().rstrip(",")
    look = look.strip().rstrip(",")

    energy = _combined_energy(analysis)
    half = 0.3

    def energy_at(t: float) -> float:
        i = int(t * analysis.fps)
        return float(energy[i]) if 0 <= i < energy.size else 0.5

    out: list[str] = []
    for shot in plan.shots:
        entries: list[str] = []
        t = shot.start_time
        n = 0
        while t < shot.end_time or n == 0:
            local_frame = int(round((t - shot.start_time) * plan.video_fps))
            local_frame = max(0, min(local_frame, shot.frame_count - 1))
            key = _stable_seed(seed, shot.index * 1000 + n)
            e = energy_at(t)
            framing = _pick(variations, key, "framing", max(0.0, e - half), min(1.0, e + half))
            lighting = _pick(LIGHTING, key, "lighting", max(0.0, e - half), min(1.0, e + half))
            parts = [p for p in (framing, master, lighting, look) if p]
            # Double quotes would terminate the schedule entry early.
            text = ", ".join(parts).replace('"', "'")
            idx = f"{local_frame}:" if hard_cuts else f"{local_frame}"
            entries.append(f'"{idx}": "{text}"')
            n += 1
            t += keyframe_seconds
        out.append(",\n".join(entries))
    return out


def schedule_summary(schedules: list[str], plan: ShotPlan) -> str:
    lines = []
    for shot, sched in zip(plan.shots, schedules):
        n = sched.count("\n") + 1 if sched else 0
        lines.append(
            f"segment {shot.index:02d}  {shot.start_time:7.2f}s  "
            f"{shot.frame_count:4d}f  {shot.bars:5.1f} bars  "
            f"energy {shot.energy:.2f}  {n} prompt keyframes"
        )
    return "\n".join(lines)
