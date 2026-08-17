"""Check AnimateDiff segment planning, curves and prompt schedules without rendering.

Same purpose as smoke_test.py: catch the arithmetic bugs in a second rather than
twenty minutes into a render.

    python tools/smoke_test_animatediff.py /path/to/song.mp3
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from musicvideo import analysis, animatediff as AD  # noqa: E402
from probe_song import decode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--max-segment", type=float, default=24.0)
    args = ap.parse_args()

    audio, sr = decode(args.song)
    a = analysis.analyse(audio, sr)
    print(a.summary())

    gen_fps = AD.suggest_gen_fps(a, args.context)
    bar_s = 60.0 * a.beats_per_bar / a.tempo
    print(f"\nbar {bar_s:.4f} s -> generation fps {gen_fps:.4f} "
          f"puts one bar on exactly {args.context} frames")

    plan = AD.plan_segments(a, gen_fps, width=args.width, height=args.height,
                            max_segment_seconds=args.max_segment)
    print()
    print(plan.summary())

    problems = plan.verify()
    print()
    if problems:
        print("PLAN INVALID:")
        for p in problems:
            print("  -", p)
    else:
        print("plan verified: no gaps, no overlaps, frames sum exactly")

    total = sum(s.frame_count for s in plan.shots)
    print(f"total generated frames {total} vs {plan.total_frames} planned")
    print(f"compare: the shot-based stack at 25 fps would generate "
          f"{int(round(a.duration * 25))}")

    motion = AD.motion_curve(a, plan.total_frames, plan.video_fps)
    coh = AD.coherence_curve(a, plan.total_frames, plan.video_fps)
    m, c = np.asarray(motion), np.asarray(coh)
    print(f"\nmotion    len {len(motion):5d}  min {m.min():.3f} mean {m.mean():.3f} max {m.max():.3f}")
    print(f"coherence len {len(coh):5d}  min {c.min():.3f} mean {c.mean():.3f} max {c.max():.3f}")
    corr = float(np.corrcoef(m, c)[0, 1])
    print(f"correlation {corr:+.3f}  (should be strongly negative: "
          f"{'ok' if corr < -0.3 else 'NOT INVERSE, check coherence_curve'})")

    scheds = AD.build_prompt_schedule(
        plan, a,
        master_prompt="a chrome sedan through neon rain",
        look="cinematic, volumetric light",
        bars_per_keyframe=2,
    )
    print()
    print(AD.schedule_summary(scheds, plan))

    print("\nsegment 0 schedule, first 3 entries:")
    for line in scheds[0].split("\n")[:3]:
        print("   ", line[:120])

    # Every keyframe index must be inside its segment or ADE drops it.
    bad = 0
    for seg, sched in zip(plan.shots, scheds):
        for line in sched.split("\n"):
            idx = line.split(":")[0].strip().strip('"')
            try:
                v = int(idx.rstrip(":"))
            except ValueError:
                continue
            if v < 0 or v >= seg.frame_count:
                bad += 1
                print(f"   OUT OF RANGE: segment {seg.index} frame {v} of {seg.frame_count}")
    print(f"\nkeyframe index check: {'all in range' if not bad else f'{bad} out of range'}")
    return 1 if (problems or bad) else 0


if __name__ == "__main__":
    sys.exit(main())
