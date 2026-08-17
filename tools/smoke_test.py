"""End-to-end check of analysis then planning, with the structural asserts.

Run this after touching anything in musicvideo/. It does not render, so it
finishes in about a second and catches the class of bug that is otherwise only
visible as picture drifting off the music two minutes into a render.

    python tools/smoke_test.py /path/to/song.mp3
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from musicvideo import analysis, plan as planning  # noqa: E402
from probe_song import decode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--min-bars", type=float, default=1.0)
    ap.add_argument("--max-bars", type=float, default=4.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=704)
    args = ap.parse_args()

    audio, sr = decode(args.song)
    a = analysis.analyse(audio, sr)
    print(a.summary())
    d = a.diagnostics
    print(f"  rigid {d['rigid_tempo']:7.3f} BPM lock {d['rigid_lock']:.2f}   |   "
          f"dp {d['dp_tempo']:7.3f} BPM lock {d['dp_lock']:.2f} "
          f"residual {d['dp_residual_ms']:.0f} ms "
          f"{'coherent' if d['dp_coherent'] else 'SLIPPING BEATS'}")

    bars_in_track = a.duration / (60.0 * a.beats_per_bar / a.tempo)
    off = abs(bars_in_track - round(bars_in_track))
    print(f"track length  {bars_in_track:7.2f} bars   "
          f"({'integer bar count, tempo confirmed' if off < 0.06 else 'not a round bar count'})")
    print()

    p = planning.plan_shots(
        a,
        video_fps=args.fps,
        width=args.width,
        height=args.height,
        min_bars=args.min_bars,
        max_bars=args.max_bars,
    )
    print(p.summary())

    problems = p.verify()
    print()
    if problems:
        print("PLAN INVALID:")
        for x in problems:
            print("  -", x)
    else:
        print("plan verified: no gaps, no overlaps, frames sum exactly")

    # Cuts must land on bar lines. Measure against the detected downbeats, not a
    # grid anchored at zero: the bar grid starts at the downbeat phase, and
    # assuming otherwise makes a correct plan look broken.
    errs = []
    for s in p.shots[1:]:
        j = int(np.argmin(np.abs(a.downbeats - s.start_time)))
        errs.append(abs(a.downbeats[j] - s.start_time) * 1000.0)
    if errs:
        frame_ms = 1000.0 / args.fps
        worst = max(errs)
        print(f"cut-to-bar error: max {worst:.1f} ms, mean {np.mean(errs):.1f} ms "
              f"(one video frame is {frame_ms:.1f} ms) "
              f"{'OK' if worst <= frame_ms else 'CUTS ARE OFF THE BAR GRID'}")

    # And the bar grid itself has to be regular, or cuts drift over the track.
    if a.downbeats.size > 2:
        d = np.diff(a.downbeats)
        print(f"bar length: {d.mean():.4f} s mean, {d.std()*1000:.1f} ms sd, "
              f"drift over track {abs(d.max()-d.min())*1000:.1f} ms spread")

    print(f"\nfirst 12 shots of {len(p.shots)}:")
    print(f"{'#':>3} {'start':>8} {'frames':>7} {'gen':>6} {'bars':>5} {'sec':>4} {'energy':>7}")
    for s in p.shots[:12]:
        print(f"{s.index:3d} {s.start_time:8.2f} {s.frame_count:7d} {s.gen_frames:6d} "
              f"{s.bars:5.1f} {s.section:4d} {s.energy:7.3f}")

    lens = np.array([s.frame_count for s in p.shots])
    print(f"\nshot length histogram (frames):")
    hist, edges = np.histogram(lens, bins=8)
    for c, lo, hi in zip(hist, edges[:-1], edges[1:]):
        print(f"  {lo:6.0f} - {hi:6.0f}  {'#' * int(40 * c / max(hist.max(), 1))} {c}")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
