"""Turn an audio analysis into a shot list.

The one rule that matters here: cuts land on bar lines, and shot lengths are
computed as differences between cumulative frame positions, never by adding up
per-shot durations. Rounding each shot independently and summing accumulates
error until the picture drifts off the music, and it drifts slowly enough that
it looks fine for the first thirty seconds.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Optional

import numpy as np

from .analysis import AudioAnalysis


@dataclasses.dataclass
class Shot:
    index: int
    start_time: float
    end_time: float
    start_frame: int
    frame_count: int  # exact frames this shot occupies in the final cut
    gen_frames: int  # frames to actually generate, model-legal, >= frame_count
    bars: float
    section: int
    energy: float  # mean 0..1 over the shot
    peak_energy: float
    seed: int
    prompt: str = ""
    motion: str = ""

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclasses.dataclass
class ShotPlan:
    shots: list[Shot]
    video_fps: float
    total_frames: int
    duration: float
    width: int
    height: int
    tempo: float
    beats_per_bar: int

    def summary(self) -> str:
        if not self.shots:
            return "empty plan"
        lens = np.array([s.frame_count for s in self.shots])
        secs = lens / self.video_fps
        gen = sum(s.gen_frames for s in self.shots)
        return "\n".join([
            f"shots         {len(self.shots):7d}",
            f"total frames  {self.total_frames:7d}  ({self.duration:.2f} s at {self.video_fps:g} fps)",
            f"frames to gen {gen:7d}  ({gen / max(self.total_frames, 1) - 1:+.1%} over, trimmed on assembly)",
            f"shot length   {secs.min():7.2f} s min, {secs.mean():.2f} s mean, {secs.max():.2f} s max",
            f"resolution    {self.width}x{self.height}",
        ])

    def verify(self) -> list[str]:
        """Structural checks. An empty list means the plan is sound."""
        problems = []
        if not self.shots:
            return ["plan has no shots"]

        if self.shots[0].start_frame != 0:
            problems.append(f"first shot starts at frame {self.shots[0].start_frame}, not 0")

        covered = sum(s.frame_count for s in self.shots)
        if covered != self.total_frames:
            problems.append(
                f"shot frames sum to {covered}, expected {self.total_frames} "
                f"(off by {covered - self.total_frames})"
            )

        for a, b in zip(self.shots, self.shots[1:]):
            if a.start_frame + a.frame_count != b.start_frame:
                problems.append(
                    f"gap or overlap between shot {a.index} and {b.index}: "
                    f"{a.start_frame}+{a.frame_count} != {b.start_frame}"
                )
        for s in self.shots:
            if s.frame_count <= 0:
                problems.append(f"shot {s.index} has {s.frame_count} frames")
            if s.gen_frames < s.frame_count:
                problems.append(
                    f"shot {s.index} generates {s.gen_frames} but needs {s.frame_count}"
                )
        return problems


def round_up_to_8n1(n: int) -> int:
    """LTX and most latent video models want 8k+1 frames."""
    if n <= 9:
        return 9
    return ((n - 1 + 7) // 8) * 8 + 1


def round_up_to_4n1(n: int) -> int:
    """Wan wants 4k+1."""
    if n <= 5:
        return 5
    return ((n - 1 + 3) // 4) * 4 + 1


FRAME_QUANTISERS = {
    "ltx": round_up_to_8n1,
    "wan": round_up_to_4n1,
    "none": lambda n: max(1, int(n)),
}


def _section_of(t: float, sections: np.ndarray) -> int:
    i = int(np.searchsorted(sections, t, side="right") - 1)
    return max(0, min(i, len(sections) - 2))


def _stable_seed(base: int, index: int) -> int:
    """Per-shot seed that is reproducible and not correlated between shots.

    Sequential seeds produce visibly related images on some samplers, which
    defeats the point of a cut.
    """
    h = hashlib.sha256(f"{base}:{index}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**63)


def plan_shots(
    analysis: AudioAnalysis,
    video_fps: float = 25.0,
    width: int = 1280,
    height: int = 704,
    min_bars: float = 1.0,
    max_bars: float = 4.0,
    energy_drives_length: bool = True,
    cut_on_sections: bool = True,
    max_shot_seconds: float = 12.0,
    min_shot_seconds: float = 1.0,
    quantiser: str = "ltx",
    seed: int = 0,
) -> ShotPlan:
    """Build the shot list.

    ``min_bars`` applies at peak energy and ``max_bars`` at minimum energy, so a
    chorus cuts fast and a breakdown holds. Shot boundaries are always bar lines
    regardless, which is what keeps it feeling like an edit rather than a
    slideshow on a timer.
    """
    duration = analysis.duration
    total_frames = int(round(duration * video_fps))
    quant = FRAME_QUANTISERS.get(quantiser, FRAME_QUANTISERS["none"])

    bars = np.asarray(analysis.downbeats, dtype=np.float64)
    bars = bars[(bars >= 0.0) & (bars < duration)]
    if bars.size == 0 or bars[0] > 1e-6:
        bars = np.concatenate([[0.0], bars])
    bar_len = (
        float(np.median(np.diff(bars)))
        if bars.size > 1
        else 60.0 * analysis.beats_per_bar / max(analysis.tempo, 1e-6)
    )

    # Energy per bar, used to decide how many bars each shot holds.
    a_fps = analysis.fps
    overall = (
        0.5 * analysis.bands["kick"]
        + 0.3 * analysis.bands["mid"]
        + 0.2 * analysis.bands["high"]
    )

    def energy_between(t0: float, t1: float) -> tuple[float, float]:
        i0, i1 = int(t0 * a_fps), max(int(t1 * a_fps), int(t0 * a_fps) + 1)
        seg = overall[i0:i1]
        if seg.size == 0:
            return 0.0, 0.0
        return float(seg.mean()), float(seg.max())

    section_cuts = set()
    if cut_on_sections:
        for b in analysis.sections[1:-1]:
            # snap each section boundary to its nearest bar line
            j = int(np.argmin(np.abs(bars - b)))
            section_cuts.add(j)

    # Walk the bar grid choosing a shot length at each step.
    max_bars_by_limit = max(1.0, max_shot_seconds / max(bar_len, 1e-6))
    hi_bars = min(max_bars, max_bars_by_limit)
    lo_bars = min(min_bars, hi_bars)

    # Shot lengths snap to powers of two bars. Three-bar shots in 4/4 read as a
    # mistake however well they line up with the grid, and the ear counts in
    # fours whatever the picture does.
    allowed = [s for s in (1, 2, 4, 8, 16) if lo_bars <= s <= hi_bars]
    if not allowed:
        allowed = [max(1, int(round(lo_bars)))]

    # Raw energy clusters in a narrow band, so mapping it straight onto shot
    # length gives every shot the same length. Rank against the track's own
    # distribution instead, which spreads it over the full range by construction.
    bar_energies = np.asarray(
        [energy_between(bars[i], bars[i] + bar_len)[0] for i in range(len(bars))]
    )
    sorted_energies = np.sort(bar_energies)

    def energy_rank(e: float) -> float:
        if sorted_energies.size < 2:
            return 0.5
        return float(np.searchsorted(sorted_energies, e) / sorted_energies.size)

    cut_bars: list[int] = [0]
    i = 0
    n_bars = len(bars)
    while i < n_bars - 1:
        if energy_drives_length:
            t0 = bars[i]
            sec = _section_of(t0, analysis.sections)
            e_sec = (
                float(analysis.section_energy[sec])
                if analysis.section_energy.size > sec
                else 0.5
            )
            # Section energy carries the structure and is already spread across
            # the track; local rank supplies variation inside a section.
            e = 0.7 * e_sec + 0.3 * energy_rank(float(bar_energies[i]))
            span = hi_bars - e * (hi_bars - lo_bars)  # high energy -> short shots
        else:
            span = hi_bars
        step = min(allowed, key=lambda s: abs(s - span))

        nxt = i + step
        # Land on a section boundary if one falls inside this shot, so a new
        # section never starts mid-shot.
        for j in range(i + 1, min(nxt, n_bars)):
            if j in section_cuts:
                nxt = j
                break
        if nxt >= n_bars:
            break
        cut_bars.append(nxt)
        i = nxt

    cut_times = [float(bars[j]) for j in cut_bars] + [duration]
    # Cumulative frame positions, then differences. This is the whole point:
    # rounding happens once per boundary and never accumulates.
    cut_frames = [int(round(t * video_fps)) for t in cut_times]
    cut_frames[0] = 0
    cut_frames[-1] = total_frames
    # enforce monotonic, at least 1 frame per shot
    for k in range(1, len(cut_frames)):
        if cut_frames[k] <= cut_frames[k - 1]:
            cut_frames[k] = cut_frames[k - 1] + 1
    if cut_frames[-1] != total_frames and len(cut_frames) > 1:
        cut_frames[-1] = max(total_frames, cut_frames[-2] + 1)

    # Drop cuts that would leave a runt. A half-second shot at the end of a track
    # is a glitch, not an edit, and it costs a whole extra generation to produce.
    min_frames = max(1, int(round(min_shot_seconds * video_fps)))
    if len(cut_frames) > 2:
        pruned = [cut_frames[0]]
        for k in range(1, len(cut_frames) - 1):
            if cut_frames[k] - pruned[-1] >= min_frames:
                pruned.append(cut_frames[k])
        pruned.append(cut_frames[-1])
        # If the final shot is now a runt, remove the cut that opened it.
        while len(pruned) > 2 and pruned[-1] - pruned[-2] < min_frames:
            pruned.pop(-2)
        cut_frames = pruned

    shots: list[Shot] = []
    for idx, (f0, f1) in enumerate(zip(cut_frames[:-1], cut_frames[1:])):
        count = f1 - f0
        t0, t1 = f0 / video_fps, f1 / video_fps
        e, pk = energy_between(t0, t1)
        shots.append(
            Shot(
                index=idx,
                start_time=t0,
                end_time=t1,
                start_frame=f0,
                frame_count=count,
                gen_frames=int(quant(count)),
                bars=(t1 - t0) / max(bar_len, 1e-6),
                section=_section_of(t0, analysis.sections),
                energy=e,
                peak_energy=pk,
                seed=_stable_seed(seed, idx),
            )
        )

    return ShotPlan(
        shots=shots,
        video_fps=video_fps,
        total_frames=total_frames,
        duration=duration,
        width=width,
        height=height,
        tempo=analysis.tempo,
        beats_per_bar=analysis.beats_per_bar,
    )


def to_dict(plan: ShotPlan) -> dict:
    return {
        "video_fps": plan.video_fps,
        "total_frames": plan.total_frames,
        "duration": plan.duration,
        "width": plan.width,
        "height": plan.height,
        "tempo": plan.tempo,
        "beats_per_bar": plan.beats_per_bar,
        "shots": [dataclasses.asdict(s) for s in plan.shots],
    }


def from_dict(d: dict) -> ShotPlan:
    return ShotPlan(
        shots=[Shot(**s) for s in d["shots"]],
        video_fps=d["video_fps"],
        total_frames=d["total_frames"],
        duration=d["duration"],
        width=d["width"],
        height=d["height"],
        tempo=d["tempo"],
        beats_per_bar=d["beats_per_bar"],
    )
