"""Validate the analyser against a real track.

Beat detection fails quietly: a grid locked to the wrong tempo, or a downbeat one
beat late, both produce output that looks entirely reasonable in a printout. So
this does two things a printout cannot.

It scores the grid objectively, by asking how close the independently detected
kick transients land to the nearest beat. A correct grid pulls them under about
30 ms. A wrong one scatters them uniformly across the beat period.

And it renders a click track, because the only real test of a beat grid is
listening to it against the music.

    python tools/probe_song.py /path/to/song.mp3 [--click out.wav]
"""

import argparse
import os
import subprocess
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from musicvideo import analysis  # noqa: E402


def decode(path: str, sr: int = 48000) -> tuple[np.ndarray, int]:
    """Decode anything ffmpeg can read into mono float32."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le",
         "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32).copy(), sr


def alignment_score(events: np.ndarray, grid: np.ndarray, tol_ms: float = 30.0):
    """How tightly a set of event times snaps to a grid.

    Returns (fraction within tolerance, median absolute error in ms). The chance
    baseline for a uniformly scattered set is tol / (period/2), so compare against
    that rather than against zero.
    """
    if events.size == 0 or grid.size == 0:
        return 0.0, float("nan")
    idx = np.searchsorted(grid, events)
    idx = np.clip(idx, 1, len(grid) - 1)
    left, right = grid[idx - 1], grid[idx]
    err = np.minimum(np.abs(events - left), np.abs(events - right))
    return float((err < tol_ms / 1000.0).mean()), float(np.median(err) * 1000.0)


def render_click(path_out, audio, sr, beats, downbeats, dur, start=0.0, length=None):
    """Music at -6 dB with a click on every beat and a higher one on bar lines.

    ``start`` and ``length`` cut an excerpt. Judging a grid by ear takes about
    fifteen seconds of a busy section, not three minutes of a whole track, and
    a short file is far more likely to actually get listened to.
    """
    end = dur if length is None else min(dur, start + length)
    i0, i1 = int(start * sr), int(end * sr)
    mix = audio[i0:i1].astype(np.float32) * 0.5

    def ping(t, freq, amp, ms=28):
        i = int((t - start) * sr)
        n = int(sr * ms / 1000)
        if i < 0 or i + n > len(mix):
            return
        env = np.exp(-np.linspace(0, 9, n))
        mix[i:i + n] += (amp * env * np.sin(2 * np.pi * freq * np.arange(n) / sr)).astype(np.float32)

    db = set(np.round(downbeats, 4).tolist())
    for b in beats:
        if not (start <= b <= end):
            continue
        # Bar lines get an octave up and a two-tone stack so they stand out even
        # when the music is dense.
        if round(float(b), 4) in db:
            ping(b, 1600.0, 0.60)
            ping(b, 2400.0, 0.35, ms=20)
        else:
            ping(b, 900.0, 0.28)

    peak = float(np.abs(mix).max()) or 1.0
    pcm = (np.clip(mix / max(peak, 1.0), -1, 1) * 32767).astype("<i2")
    with wave.open(path_out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def busiest_window(analysis_result, length=16.0):
    """Start time of the most energetic ``length`` seconds.

    A grid error is easiest to hear where the music is dense, and an intro of
    pad and reverb tells you nothing.
    """
    a = analysis_result
    env = 0.5 * a.bands["kick"] + 0.3 * a.bands["mid"] + 0.2 * a.bands["high"]
    w = max(1, int(length * a.fps))
    if env.size <= w:
        return 0.0
    csum = np.concatenate([[0.0], np.cumsum(env)])
    means = (csum[w:] - csum[:-w]) / w
    return float(int(np.argmax(means)) / a.fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("--click", default=None, help="write a click-track wav here")
    ap.add_argument("--click-start", type=float, default=-1.0,
                    help="excerpt start in seconds, or -1 for the busiest part")
    ap.add_argument("--click-seconds", type=float, default=16.0,
                    help="excerpt length, 0 for the whole track")
    ap.add_argument("--phase-variants", action="store_true",
                    help="also write a click track for every other bar phase")
    ap.add_argument("--prior-bpm", type=float, default=120.0)
    ap.add_argument("--beats-per-bar", type=int, default=4)
    ap.add_argument("--grid-mode", default="rigid",
                    choices=["rigid", "adaptive", "auto"])
    ap.add_argument("--phase-offset", type=float, default=0.0,
                    help="slide the grid by this many beats, fractions allowed")
    args = ap.parse_args()

    audio, sr = decode(args.song)
    print(f"decoded {len(audio)/sr:.2f} s at {sr} Hz\n")

    import time
    t0 = time.time()
    a = analysis.analyse(
        audio, sr, beats_per_bar=args.beats_per_bar, prior_bpm=args.prior_bpm,
        grid_mode=args.grid_mode, phase_offset_beats=args.phase_offset,
    )
    elapsed = time.time() - t0

    print(a.summary())
    print(f"analysis took {elapsed:.2f} s\n")

    cands = analysis.tempo_candidates(a.onset_env, a.fps, prior_bpm=args.prior_bpm)
    print("tempo candidates (bpm / raw autocorr / prior-weighted):")
    for bpm, raw, sc in cands:
        print(f"   {bpm:7.2f}   {raw:6.3f}   {sc:6.3f}")
    print()

    ibi = np.diff(a.beats)
    print(f"inter-beat interval  mean {ibi.mean()*1000:7.2f} ms   "
          f"sd {ibi.std()*1000:6.2f} ms   "
          f"({60/ibi.mean():.2f} BPM)")

    period = float(np.median(ibi))
    beat_rate = 1.0 / period
    print(f"\nbeat rate {beat_rate:.2f}/s. A band detector firing much faster than "
          f"this is picking up noise,\nand any alignment score against it is "
          f"meaningless.\n")
    for name, ev in a.transients.items():
        rate = len(ev) / a.duration
        line = f"{name:>6s}: {len(ev):4d} events, {rate:4.2f}/s ({rate/beat_rate:4.2f}x beat rate)"
        for tol in (30.0, 50.0):
            frac, med = alignment_score(ev, a.beats, tol)
            chance = min(1.0, (tol / 1000.0) / (period / 2))
            line += f" | {tol:.0f}ms {frac*100:5.1f}% (chance {chance*100:4.1f}%)"
        frac30, med = alignment_score(ev, a.beats, 30.0)
        line += f" | median err {med:5.1f} ms"
        print(line)

    # Which beat of the bar won, and by how much.
    scores = a.diagnostics.get("downbeat_scores", [])
    conf = a.diagnostics.get("downbeat_confidence", 0.0)
    print(f"\ndownbeat phase scores (novelty + low end + onset), "
          f"confidence {conf:.1%}:")
    for p, s in enumerate(scores):
        mark = "  <- chosen bar line" if p == a.downbeat_phase else ""
        print(f"   beat {p+1}: {s:.4f}{mark}")
    if conf < 0.06:
        print("   margin is thin. If the bars sound wrong, add 1 to downbeat_offset.")

    print(f"\nsections ({len(a.sections)-1}):")
    for i, (s, e) in enumerate(zip(a.sections[:-1], a.sections[1:])):
        bars = (e - s) / (period * a.beats_per_bar)
        print(f"   {i:2d}  {s:7.2f} -> {e:7.2f} s  ({e-s:6.2f} s, {bars:5.1f} bars)  "
              f"energy {a.section_energy[i]:.2f}")

    if args.click:
        start = args.click_start
        if start < 0:
            start = busiest_window(a, args.click_seconds)
        length = None if args.click_seconds <= 0 else args.click_seconds

        render_click(args.click, audio, sr, a.beats, a.downbeats,
                     a.duration, start, length)
        print(f"\nclick track -> {args.click}")
        print(f"   {start:.1f}s to {start + (length or a.duration - start):.1f}s, "
              f"{a.tempo:.2f} BPM, bar line on beat {a.downbeat_phase + 1}")

        # Alternative bar phases, so a wrong downbeat can be settled in one
        # sitting instead of one round trip per guess.
        if args.phase_variants:
            base = os.path.splitext(args.click)[0]
            for off in range(1, a.beats_per_bar):
                p = (a.downbeat_phase + off) % a.beats_per_bar
                out = f"{base}_downbeat_offset{off}.wav"
                render_click(out, audio, sr, a.beats,
                             a.beats[p::a.beats_per_bar], a.duration, start, length)
                print(f"   variant downbeat_offset={off} -> {out}")


if __name__ == "__main__":
    main()
