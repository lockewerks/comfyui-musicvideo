"""Nodes that turn audio into an analysis, and an analysis into control curves."""

from __future__ import annotations

import numpy as np

from ..musicvideo import analysis as A
from ..musicvideo import curves as C

CATEGORY = "music video"


def audio_to_numpy(audio: dict) -> tuple[np.ndarray, int]:
    """ComfyUI AUDIO is {'waveform': (batch, channels, samples), 'sample_rate': int}."""
    wf = audio["waveform"]
    if hasattr(wf, "detach"):
        wf = wf.detach().cpu().numpy()
    return np.asarray(wf), int(audio["sample_rate"])


class MVAnalyzeAudio:
    """Beat grid, transients, bands and sections from an audio input."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "beats_per_bar": ("INT", {"default": 4, "min": 1, "max": 16}),
                "grid_mode": (["rigid", "adaptive", "auto"], {"default": "rigid"}),
                "bpm_min": ("FLOAT", {"default": 60.0, "min": 20.0, "max": 300.0, "step": 0.5}),
                "bpm_max": ("FLOAT", {"default": 200.0, "min": 30.0, "max": 400.0, "step": 0.5}),
                "prior_bpm": ("FLOAT", {"default": 120.0, "min": 30.0, "max": 300.0, "step": 1.0}),
                "downbeat_offset": ("INT", {"default": 0, "min": 0, "max": 15}),
                # Fractional, unlike downbeat_offset. 0.5 moves the grid a half
                # beat later, which is the fix when cuts land on the pickup.
                "phase_offset_beats": ("FLOAT", {"default": 0.0, "min": -4.0, "max": 4.0, "step": 0.125}),
                "min_section_seconds": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 60.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("MV_ANALYSIS", "STRING", "FLOAT", "INT")
    RETURN_NAMES = ("analysis", "report", "tempo", "beat_count")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(
        self,
        audio,
        beats_per_bar,
        grid_mode,
        bpm_min,
        bpm_max,
        prior_bpm,
        downbeat_offset,
        phase_offset_beats,
        min_section_seconds,
    ):
        wf, sr = audio_to_numpy(audio)
        result = A.analyse(
            wf,
            sr,
            beats_per_bar=int(beats_per_bar),
            bpm_min=float(bpm_min),
            bpm_max=float(bpm_max),
            prior_bpm=float(prior_bpm),
            min_section_s=float(min_section_seconds),
            grid_mode=grid_mode,
            phase_offset_beats=float(phase_offset_beats),
        )

        # Manual downbeat override. No phase detector is right every time and the
        # cost of being one beat out is every cut in the video landing on the
        # wrong beat, so this stays a knob rather than a hidden decision.
        if downbeat_offset:
            phase = (result.downbeat_phase + int(downbeat_offset)) % int(beats_per_bar)
            result.downbeat_phase = phase
            result.downbeats = result.beats[phase :: int(beats_per_bar)]

        d = result.diagnostics
        report = "\n".join([
            result.summary(),
            "",
            f"rigid {d['rigid_tempo']:.3f} BPM lock {d['rigid_lock']:.2f}",
            f"dp    {d['dp_tempo']:.3f} BPM lock {d['dp_lock']:.2f} "
            f"residual {d['dp_residual_ms']:.0f} ms "
            f"{'coherent' if d['dp_coherent'] else 'slipping beats'}",
            "",
            "lock is onset strength on the grid over onset strength overall.",
            "1.0 is chance. Below 1.6 the cuts will look arbitrary.",
        ])
        return (result, report, float(result.tempo), int(len(result.beats)))


class MVAudioCurve:
    """A per-frame control curve from the audio, as a FLOAT list."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analysis": ("MV_ANALYSIS",),
                "total_frames": ("INT", {"default": 0, "min": 0, "max": 10_000_000}),
                "video_fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "mode": (
                    ["band", "pulse", "section", "zoom_punch"],
                    {"default": "band"},
                ),
                "source": (
                    ["kick", "sub", "bass", "mid", "high", "air", "beat", "downbeat"],
                    {"default": "kick"},
                ),
                "attack_seconds": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 2.0, "step": 0.001}),
                "release_seconds": ("FLOAT", {"default": 0.18, "min": 0.01, "max": 5.0, "step": 0.01}),
                "floor": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "ceiling": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 8.0, "step": 0.1}),
            },
            "optional": {
                "plan": ("MV_PLAN",),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("curve", "report")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(
        self,
        analysis,
        total_frames,
        video_fps,
        mode,
        source,
        attack_seconds,
        release_seconds,
        floor,
        ceiling,
        gamma,
        plan=None,
    ):
        # A curve that is not exactly as long as the render misaligns everything
        # after the first shot, so prefer the plan's own frame count when wired.
        if plan is not None:
            total_frames = plan.total_frames
            video_fps = plan.video_fps
        if not total_frames:
            total_frames = int(round(analysis.duration * video_fps))

        if mode == "band":
            curve = C.band_curve(
                analysis, total_frames, video_fps, band=source,
                attack_s=attack_seconds, release_s=release_seconds,
                floor=floor, ceiling=ceiling, gamma=gamma,
            )
        elif mode == "pulse":
            curve = C.pulse_curve(
                analysis, total_frames, video_fps, source=source,
                decay_s=release_seconds, floor=floor, ceiling=ceiling,
            )
        elif mode == "section":
            curve = C.section_curve(analysis, total_frames, video_fps, floor, ceiling)
        else:
            curve = C.zoom_punch_curve(
                analysis, total_frames, video_fps,
                amount=(ceiling - floor), decay_s=release_seconds, source=source,
            )

        arr = np.asarray(curve)
        report = (
            f"{mode}/{source}: {len(curve)} frames at {video_fps:g} fps, "
            f"min {arr.min():.3f} mean {arr.mean():.3f} max {arr.max():.3f}"
        )
        return (curve, report)


class MVBeatTimes:
    """Beat, downbeat or transient times as a FLOAT list of seconds."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analysis": ("MV_ANALYSIS",),
                "which": (["beat", "downbeat", "kick", "high", "section"], {"default": "downbeat"}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "STRING")
    RETURN_NAMES = ("times", "count", "report")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, analysis, which):
        if which == "beat":
            t = analysis.beats
        elif which == "downbeat":
            t = analysis.downbeats
        elif which == "section":
            t = analysis.sections
        else:
            t = analysis.transients.get(which, np.zeros(0))
        t = np.asarray(t, dtype=float)
        rate = len(t) / max(analysis.duration, 1e-9)
        return (t.tolist(), int(len(t)), f"{which}: {len(t)} events, {rate:.2f}/s")
