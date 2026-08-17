"""Beat, transient and structure analysis for driving a music video edit.

Everything here runs on numpy and scipy alone. librosa, madmom, aubio and essentia
are all excellent and all absent from a stock ComfyUI venv, and asking a user to
install a compiled audio stack to render a video is a bad trade. The algorithms
below are the standard ones, written out.

The pipeline, in order:

    waveform -> mel spectrogram -> spectral flux onset envelope
             -> tempo by autocorrelation under a log-normal prior
             -> beat grid by dynamic programming (Ellis 2007)
             -> downbeat phase by low-band energy vote
             -> sections by self-similarity novelty

Frame rate through the whole thing is ``sr / hop``, called ``fps`` in this module
and kept distinct from video fps, which appears nowhere in this file. Analysis
does not know what a video is.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

try:
    from scipy.signal import resample_poly
except ImportError:  # pragma: no cover - scipy ships with ComfyUI
    resample_poly = None


# Analysis runs at 22050 Hz. Onsets live well below 11 kHz and halving the rate
# halves every FFT in the file for no measurable loss in beat accuracy.
ANALYSIS_SR = 22050
N_FFT = 2048
HOP = 512
N_MELS = 128


# --------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------


@dataclasses.dataclass
class AudioAnalysis:
    """Everything the downstream planner needs, in seconds and plain arrays."""

    duration: float
    sample_rate: int
    fps: float  # analysis frames per second, sr / hop

    onset_env: np.ndarray  # (n_frames,) normalised spectral flux
    times: np.ndarray  # (n_frames,) frame centre times in seconds

    tempo: float  # BPM
    beats: np.ndarray  # (n_beats,) beat times in seconds
    beat_strength: np.ndarray  # (n_beats,) onset envelope sampled at each beat

    downbeats: np.ndarray  # (n_bars,) bar-line times in seconds
    beats_per_bar: int
    downbeat_phase: int  # index into beats where bar 1 starts

    bands: dict[str, np.ndarray]  # name -> (n_frames,) normalised band energy
    transients: dict[str, np.ndarray]  # name -> times in seconds

    sections: np.ndarray  # (n_sections + 1,) boundary times, starts at 0
    section_energy: np.ndarray  # (n_sections,) mean energy per section, 0..1

    lock: float  # onset strength on the grid over onset strength overall
    method: str  # which estimator won, "comb" or "dp"
    diagnostics: dict  # per-estimator scores, for working out why it chose that

    def bar_times(self) -> np.ndarray:
        return self.downbeats

    @property
    def confident(self) -> bool:
        """Whether the beat grid is worth trusting.

        1.0 is chance. Below about 1.6 the grid is not locked to the music and
        cutting on it will look arbitrary, however tidy the numbers appear.
        """
        return self.lock >= 1.6

    def summary(self) -> str:
        lines = [
            f"duration      {self.duration:7.2f} s",
            f"tempo         {self.tempo:7.2f} BPM  ({self.beats_per_bar}/4)",
            f"grid lock     {self.lock:7.2f}   via {self.method}"
            f"   {'LOCKED' if self.confident else 'NOT LOCKED - do not trust these cuts'}",
            f"beats         {len(self.beats):7d}   ({len(self.beats) / max(self.duration, 1e-9) * 60:.1f}/min)",
            f"bars          {len(self.downbeats):7d}",
            f"sections      {len(self.sections) - 1:7d}",
        ]
        for name, t in self.transients.items():
            lines.append(f"transients {name:>6s} {len(t):7d}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# signal helpers
# --------------------------------------------------------------------------


def to_mono(waveform: np.ndarray) -> np.ndarray:
    """Accept (n,), (c, n) or (b, c, n) and return (n,) float32."""
    x = np.asarray(waveform, dtype=np.float32)
    while x.ndim > 1:
        # average whichever leading axis is smaller than the sample axis
        x = x.mean(axis=0)
    return x


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    if resample_poly is not None:
        g = np.gcd(int(sr_in), int(sr_out))
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    # linear fallback, good enough for an onset envelope
    n_out = int(round(len(x) * sr_out / sr_in))
    return np.interp(
        np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x
    ).astype(np.float32)


def _frame_signal(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Centre-padded framing, so frame t is centred on sample t * hop."""
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad), mode="reflect" if len(x) > pad else "constant")
    n_frames = 1 + (len(xp) - n_fft) // hop
    if n_frames < 1:
        return np.zeros((0, n_fft), dtype=np.float32)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return xp[idx]


def stft_magnitude(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """(n_frames, n_fft // 2 + 1) magnitude spectrogram."""
    frames = _frame_signal(x, n_fft, hop)
    if frames.shape[0] == 0:
        return np.zeros((0, n_fft // 2 + 1), dtype=np.float32)
    window = np.hanning(n_fft).astype(np.float32)
    return np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sr: int, n_fft: int, n_mels: int, fmin: float = 30.0, fmax: Optional[float] = None
) -> np.ndarray:
    """(n_mels, n_fft // 2 + 1) triangular filterbank, area-normalised."""
    fmax = fmax if fmax is not None else sr / 2.0
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_bins)

    mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)

    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(n_mels):
        lo, ctr, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        if hi <= lo:
            continue
        rising = (fft_freqs - lo) / max(ctr - lo, 1e-9)
        falling = (hi - fft_freqs) / max(hi - ctr, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(rising, falling))
        # Slaney-style area normalisation keeps low bands from dominating flux
        fb[i] *= 2.0 / max(hi - lo, 1e-9)
    return fb


def mel_frequencies(
    sr: int, n_mels: int, fmin: float = 30.0, fmax: Optional[float] = None
) -> np.ndarray:
    """Centre frequency of each mel band, matching mel_filterbank's layout."""
    fmax = fmax if fmax is not None else sr / 2.0
    mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    return _mel_to_hz(mel_pts)[1:-1]


def _normalise(v: np.ndarray) -> np.ndarray:
    """Scale to 0..1 by robust range, so one cymbal crash doesn't flatten the rest."""
    if v.size == 0:
        return v
    lo = float(np.percentile(v, 2))
    hi = float(np.percentile(v, 98))
    if hi - lo < 1e-9:
        return np.zeros_like(v)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


# --------------------------------------------------------------------------
# onsets
# --------------------------------------------------------------------------


def onset_envelope(
    x: np.ndarray, sr: int = ANALYSIS_SR, hop: int = HOP, n_mels: int = N_MELS
) -> tuple[np.ndarray, np.ndarray]:
    """Spectral flux onset envelope and the dB mel spectrogram it came from.

    Half-wave rectified difference of a dB-scaled mel power spectrogram, which is
    what makes it respond to note onsets rather than to loudness.

    Deliberately *not* detrended here. Subtracting a local mean on the order of a
    beat period removes the periodicity the tempo estimator exists to find, which
    is a quiet way to make every downstream number look plausible and be wrong.
    Peak picking does its own detrending, on its own window.
    """
    spec = stft_magnitude(x, N_FFT, hop)
    if spec.shape[0] == 0:
        return np.zeros(0, dtype=np.float32), np.zeros((0, n_mels), dtype=np.float32)

    fb = mel_filterbank(sr, N_FFT, n_mels)
    mel_power = (spec.astype(np.float64) ** 2) @ fb.T  # (n_frames, n_mels)

    db = 10.0 * np.log10(np.maximum(mel_power, 1e-10))
    db = np.maximum(db, db.max() - 80.0)  # floor 80 dB below peak

    diff = np.diff(db, axis=0, prepend=db[:1])
    flux = np.maximum(diff, 0.0).mean(axis=1)

    # Scale by standard deviation. The DP tracker's tightness penalty is in the
    # same units as the envelope, so the envelope needs a stable scale.
    sd = float(flux.std())
    flux = flux / sd if sd > 1e-9 else flux

    return flux.astype(np.float32), db.astype(np.float32)


def _moving_average(v: np.ndarray, w: int) -> np.ndarray:
    if w < 2 or v.size == 0:
        return np.zeros_like(v)
    w = min(w, v.size if v.size % 2 else v.size - 1)
    if w < 2:
        return np.zeros_like(v)
    pad = w // 2
    vp = np.pad(v, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(vp, kernel, mode="valid")[: v.size].astype(v.dtype)


def pick_peaks(
    env: np.ndarray,
    fps: float,
    min_gap_s: float = 0.10,
    delta: float = 0.6,
    mean_window_s: float = 0.6,
) -> np.ndarray:
    """Onset peak picking after Böck: local maximum, above a local mean plus
    delta, and at least ``min_gap_s`` after the previous accepted peak.

    ``delta`` is in units of the envelope's own standard deviation. It is the
    knob that decides how many events you get, and it wants to be well above
    zero: a permissive threshold on a band envelope produces several times the
    real hit rate, which then looks like a broken beat grid when you score
    anything against it.
    """
    if env.size < 3:
        return np.zeros(0, dtype=int)

    sd = float(env.std())
    if sd <= 1e-9:
        return np.zeros(0, dtype=int)

    local = _moving_average(env, int(round(mean_window_s * fps)) | 1)
    is_local_max = np.zeros(env.size, dtype=bool)
    is_local_max[1:-1] = (env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:])
    cand = np.flatnonzero(is_local_max & (env > local + delta * sd))
    if cand.size == 0:
        return cand

    min_gap = max(1, int(round(min_gap_s * fps)))
    kept = [int(cand[0])]
    for c in cand[1:]:
        if c - kept[-1] >= min_gap:
            kept.append(int(c))
        elif env[c] > env[kept[-1]]:
            kept[-1] = int(c)
    return np.asarray(kept, dtype=int)


def subband_flux(db: np.ndarray, mel_hz: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Spectral flux restricted to the mel bands inside [lo, hi) Hz.

    Onsets in a band are a rise in that band's spectrum, not a rise in its
    smoothed RMS level. Differentiating an RMS envelope responds to every wobble
    in a sustained note and buries the actual hits.
    """
    sel = (mel_hz >= lo) & (mel_hz < hi)
    if not sel.any() or db.shape[0] == 0:
        return np.zeros(db.shape[0], dtype=np.float32)
    sub = db[:, sel]
    diff = np.diff(sub, axis=0, prepend=sub[:1])
    flux = np.maximum(diff, 0.0).mean(axis=1)
    sd = float(flux.std())
    return (flux / sd if sd > 1e-9 else flux).astype(np.float32)


# --------------------------------------------------------------------------
# tempo
# --------------------------------------------------------------------------


def estimate_tempo(
    onset_env: np.ndarray,
    fps: float,
    bpm_min: float = 60.0,
    bpm_max: float = 200.0,
    prior_bpm: float = 120.0,
    prior_width: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Tempo in BPM by autocorrelation of the onset envelope.

    Autocorrelation alone is octave-ambiguous: a 4/4 track at 128 correlates just
    as well at 64 and 256. The log-normal prior around ``prior_bpm`` is what picks
    between them, and it is the single knob most worth touching if a track comes
    back at half or double time.
    """
    n = onset_env.size
    if n < 4:
        return float(prior_bpm), np.zeros(0, dtype=np.float32)

    env = onset_env - onset_env.mean()
    # FFT autocorrelation, zero-padded to avoid circular wrap
    size = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(env, size)
    ac = np.fft.irfft(spec * np.conj(spec), size)[:n].real
    if ac[0] > 0:
        ac = ac / ac[0]

    lag_min = max(1, int(np.floor(60.0 * fps / bpm_max)))
    lag_max = min(n - 1, int(np.ceil(60.0 * fps / bpm_min)))
    if lag_max <= lag_min:
        return float(prior_bpm), ac.astype(np.float32)

    lags = np.arange(lag_min, lag_max + 1)
    bpms = 60.0 * fps / lags
    prior = np.exp(-0.5 * (np.log2(bpms / prior_bpm) / prior_width) ** 2)
    score = ac[lags] * prior

    best = lags[int(np.argmax(score))]
    return float(60.0 * fps / best), ac.astype(np.float32)


def tempo_candidates(
    onset_env: np.ndarray,
    fps: float,
    bpm_min: float = 60.0,
    bpm_max: float = 200.0,
    prior_bpm: float = 120.0,
    prior_width: float = 1.0,
    top: int = 8,
) -> list[tuple[float, float, float]]:
    """Top (bpm, raw autocorrelation, prior-weighted score) peaks, for diagnosis.

    When a track comes back at half or double time this is the thing to look at:
    it shows whether the true tempo was a close second or was never a peak.
    """
    n = onset_env.size
    if n < 4:
        return []
    env = onset_env - onset_env.mean()
    size = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(env, size)
    ac = np.fft.irfft(spec * np.conj(spec), size)[:n].real
    if ac[0] > 0:
        ac = ac / ac[0]

    lag_min = max(1, int(np.floor(60.0 * fps / bpm_max)))
    lag_max = min(n - 1, int(np.ceil(60.0 * fps / bpm_min)))
    if lag_max <= lag_min:
        return []
    lags = np.arange(lag_min, lag_max + 1)
    bpms = 60.0 * fps / lags
    prior = np.exp(-0.5 * (np.log2(bpms / prior_bpm) / prior_width) ** 2)
    score = ac[lags] * prior

    local = np.zeros(len(lags), dtype=bool)
    if len(lags) > 2:
        local[1:-1] = (score[1:-1] >= score[:-2]) & (score[1:-1] > score[2:])
    peaks = np.flatnonzero(local)
    peaks = peaks[np.argsort(score[peaks])[::-1][:top]]
    return [(float(bpms[p]), float(ac[lags[p]]), float(score[p])) for p in peaks]


def fourier_tempo(
    onset_env: np.ndarray,
    fps: float,
    bpm_min: float = 60.0,
    bpm_max: float = 200.0,
    prior_bpm: float = 120.0,
    prior_width: float = 1.0,
    n_fft: int = 1 << 21,
) -> tuple[float, float]:
    """Rigid (tempo, phase) from a heavily zero-padded Fourier tempogram.

    The DFT of the onset envelope at the beat frequency has magnitude
    proportional to how periodic the track is at that tempo, and argument
    -2*pi*f*phi, so one complex number gives both the score and the exact beat
    offset. Zero padding to 2**21 interpolates the transform finely enough to
    place the peak to about 0.001 BPM, which over three minutes is well under one
    video frame of drift.

    Two properties make this the right primitive here, and both were learned by
    getting them wrong:

    Scoring by mean onset strength over grid points, which is the obvious comb
    filter, is biased toward slow tempos. Half as many grid points means twice
    the freedom to land them all on strong onsets, so the score rises as tempo
    falls and the search happily returns 60 BPM for a 123 BPM track.

    A DFT has no such bias, and it suppresses subharmonics rather than rewarding
    them: sampling an impulse train at half its rate alternates sign and cancels.
    Remaining ambiguity is toward integer multiples of the true tempo, which is
    what the log-normal prior is for.
    """
    n = onset_env.size
    if n < 8:
        return float(prior_bpm), 0.0

    env = (onset_env - onset_env.mean()).astype(np.float64)
    n_fft = max(n_fft, 1 << int(np.ceil(np.log2(max(n * 2, 2)))))
    X = np.fft.rfft(env, n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fps)

    with np.errstate(divide="ignore", invalid="ignore"):
        bpms = freqs * 60.0
    sel = (bpms >= bpm_min) & (bpms <= bpm_max)
    if not sel.any():
        return float(prior_bpm), 0.0

    b = bpms[sel]
    Xs = X[sel]
    prior = np.exp(-0.5 * (np.log2(b / prior_bpm) / prior_width) ** 2)
    score = np.abs(Xs) * prior

    k = int(np.argmax(score))
    f = float(freqs[sel][k])
    if f <= 0:
        return float(prior_bpm), 0.0
    period = 1.0 / f
    phase = float(-np.angle(Xs[k]) / (2.0 * np.pi * f)) % period
    return float(b[k]), phase


def rigid_grid(
    onset_env: np.ndarray,
    fps: float,
    duration: float,
    bpm_min: float = 60.0,
    bpm_max: float = 200.0,
    prior_bpm: float = 120.0,
    span: float = 2.0,
    n_steps: int = 801,
) -> tuple[float, float, float]:
    """Best constant-tempo grid. Returns (bpm, phase_s, lock).

    Two stages. The Fourier tempogram picks the octave and gets within a BPM or
    so. Then a fine sweep around it scores each candidate by the metric that
    actually matters, onset strength landing on the grid, taking the phase for
    each candidate from the argument of its own DFT bin rather than searching it.

    Scoring the fine stage by lock rather than by DFT magnitude matters: magnitude
    rewards any strong periodicity in the envelope, including the half-bar pulse
    of a shuffle, while lock only rewards beats that actually land on onsets.
    """
    coarse, _ = fourier_tempo(onset_env, fps, bpm_min, bpm_max, prior_bpm)

    n = onset_env.size
    if n < 8:
        return coarse, 0.0, 0.0

    env = (onset_env - onset_env.mean()).astype(np.float64)
    t = np.arange(n, dtype=np.float64) / fps
    bpms = np.linspace(max(coarse - span, bpm_min), min(coarse + span, bpm_max), n_steps)

    best = (coarse, 0.0, 0.0)
    for a in range(0, len(bpms), 64):
        chunk = bpms[a : a + 64]
        freqs = chunk / 60.0
        X = env @ np.exp(-2j * np.pi * np.outer(t, freqs))
        for j, f in enumerate(freqs):
            period = 1.0 / f
            phase = float(-np.angle(X[j]) / (2.0 * np.pi * f)) % period
            grid = np.arange(phase, duration, period)
            lk = grid_lock(grid, onset_env, fps)
            if lk > best[2]:
                best = (float(chunk[j]), phase, lk)
    return best


def refine_tempo_phase(
    onset_env: np.ndarray,
    fps: float,
    bpm0: float,
    span: float = 1.0,
    n_steps: int = 1201,
) -> tuple[float, float]:
    """Exact tempo and phase near ``bpm0`` from the Fourier tempogram.

    For an impulse train at times phi + k/f, the DFT of the onset envelope at
    frequency f has magnitude N and argument -2*pi*f*phi. So one complex value
    per candidate frequency gives both how well that tempo explains the track and
    exactly where its beats sit, at arbitrary sub-frame precision.

    This matters more than it looks. A coarse grid search resolves tempo to about
    0.05 BPM, which over a three minute track is a second of accumulated drift:
    the cuts start on the beat and end well off it.
    """
    n = onset_env.size
    if n < 8:
        return bpm0, 0.0

    env = (onset_env - onset_env.mean()).astype(np.float64)
    t = np.arange(n, dtype=np.float64) / fps
    bpms = np.linspace(max(bpm0 - span, 1.0), bpm0 + span, n_steps)
    freqs = bpms / 60.0

    best_bpm, best_phase, best_mag = bpm0, 0.0, -1.0
    # Chunked so the complex exponential matrix stays small.
    for a in range(0, len(freqs), 64):
        fchunk = freqs[a : a + 64]
        X = env @ np.exp(-2j * np.pi * np.outer(t, fchunk))
        mags = np.abs(X)
        k = int(np.argmax(mags))
        if mags[k] > best_mag:
            f = float(fchunk[k])
            phase = float(-np.angle(X[k]) / (2.0 * np.pi * f))
            period = 1.0 / f
            best_bpm = f * 60.0
            best_phase = phase % period
            best_mag = float(mags[k])
    return best_bpm, best_phase


def grid_lock(
    beats_s: np.ndarray, onset_env: np.ndarray, fps: float, tol_frames: int = 1
) -> float:
    """Mean onset strength on a beat grid over mean onset strength overall.

    1.0 is chance; a locked grid on percussive material runs 2 to 4.

    ``tol_frames`` takes the best value within a frame either side of each beat.
    Without it the metric is dominated by quantisation: an analysis frame is 23 ms
    and a grid point landing one frame off a hard onset scores nothing, which
    penalises a correct rigid grid against a tracker free to snap onto frames.
    """
    if beats_s.size == 0 or onset_env.size == 0:
        return 0.0
    baseline = float(onset_env.mean())
    if baseline <= 1e-9:
        return 0.0
    n = onset_env.size
    idx = np.clip(np.rint(beats_s * fps).astype(np.int64), 0, n - 1)
    if tol_frames <= 0:
        return float(onset_env[idx].mean() / baseline)
    stack = [onset_env[np.clip(idx + d, 0, n - 1)] for d in range(-tol_frames, tol_frames + 1)]
    return float(np.max(np.stack(stack), axis=0).mean() / baseline)


# --------------------------------------------------------------------------
# beat tracking
# --------------------------------------------------------------------------


def track_beats(
    onset_env: np.ndarray, fps: float, tempo: float, tightness: float = 100.0
) -> np.ndarray:
    """Ellis dynamic-programming beat tracker. Returns beat frame indices.

    Maximises total onset strength along the beat path minus a penalty for
    deviating from the tempo period, which is what keeps the grid regular through
    a bar of silence instead of collapsing onto whatever noise is there.
    """
    n = onset_env.size
    period = 60.0 * fps / max(tempo, 1e-6)
    if n < 2 or period < 1.0:
        return np.zeros(0, dtype=int)

    # Search window for the previous beat: half a period to twice a period back.
    lo = max(1, int(np.floor(period * 0.5)))
    hi = max(lo + 1, int(np.ceil(period * 2.0)))

    # Penalty indexed by offset value, not by position in a slice. Indexing it by
    # slice position is subtly wrong: the score window runs from the largest
    # offset down to the smallest, so a positional penalty is applied reversed
    # and every transition gets scored against the wrong period.
    pen_by_offset = np.full(hi + 1, -np.inf, dtype=np.float64)
    offsets = np.arange(lo, hi + 1)
    pen_by_offset[lo : hi + 1] = -tightness * (np.log(offsets / period) ** 2)

    cumscore = np.array(onset_env, dtype=np.float64)
    backlink = np.full(n, -1, dtype=int)

    for t in range(1, n):
        hi_t = min(hi, t)
        if hi_t < lo:
            continue
        offs = np.arange(lo, hi_t + 1)
        prev_idx = t - offs
        scores = cumscore[prev_idx] + pen_by_offset[offs]
        k = int(np.argmax(scores))
        best = float(scores[k])
        if best > 0:
            cumscore[t] = onset_env[t] + best
            backlink[t] = int(prev_idx[k])

    # Backtrace start: cumscore rises over the track, so the last beat is the
    # final local maximum that is still comparable to the typical one.
    is_max = np.zeros(n, dtype=bool)
    if n > 2:
        is_max[1:-1] = (cumscore[1:-1] >= cumscore[:-2]) & (cumscore[1:-1] > cumscore[2:])
    peaks = np.flatnonzero(is_max)
    if peaks.size:
        med = float(np.median(cumscore[peaks]))
        strong = peaks[cumscore[peaks] * 2 > med]
        end = int(strong[-1]) if strong.size else int(np.argmax(cumscore))
    else:
        end = int(np.argmax(cumscore))

    beats = []
    t = end
    guard = 0
    while t >= 0 and guard < n + 4:
        beats.append(t)
        t = backlink[t]
        guard += 1
    beats.reverse()
    return np.asarray(beats, dtype=int)


def extend_beat_grid(beats_s: np.ndarray, duration: float, period_s: float) -> np.ndarray:
    """Extend a beat grid to cover the whole track at a constant period.

    The DP tracker only emits beats where it found evidence, so an intro of pure
    pad or an outro fade leaves the grid short. A music video still needs cuts
    there, so both ends get extrapolated at the measured period.
    """
    if beats_s.size == 0:
        n = int(np.floor(duration / period_s)) + 1
        return np.arange(n) * period_s

    out = list(beats_s)
    t = out[0] - period_s
    while t > 0:
        out.insert(0, t)
        t -= period_s
    t = out[-1] + period_s
    while t < duration:
        out.append(t)
        t += period_s
    return np.asarray([t for t in out if 0.0 <= t <= duration], dtype=np.float64)


# --------------------------------------------------------------------------
# downbeats
# --------------------------------------------------------------------------


def find_downbeat_phase(
    beats_s: np.ndarray,
    times: np.ndarray,
    low_band: np.ndarray,
    onset_env: np.ndarray,
    db_spec: Optional[np.ndarray] = None,
    beats_per_bar: int = 4,
) -> tuple[int, np.ndarray]:
    """Pick which beat of every ``beats_per_bar`` is the bar line.

    Returns (phase, per-phase scores) so callers can report how close the call
    was. A marginal win here is worth surfacing: being one beat out puts every
    cut in the video on the wrong beat, and it is not visible in any other number.

    Three cues, because no single one survives contact with real material:

    Spectral novelty carries the most weight. Bar lines are where the chord
    changes, where a new element enters, where the bass note moves. This is the
    only cue that works on four-on-the-floor, where the kick is on every beat and
    low-band energy therefore says nothing about which beat is first.

    Low-band energy still helps on syncopated material, where the kick genuinely
    marks the bar.

    Onset strength is weighted least on purpose. On anything with a backbeat it
    votes confidently for the snare on 2 and 4.
    """
    n_phase = max(1, int(beats_per_bar))
    if beats_s.size < n_phase * 2:
        return 0, np.zeros(n_phase)

    idx = np.clip(np.searchsorted(times, beats_s), 0, len(times) - 1)
    low_at_beat = _normalise(low_band[idx])
    onset_at_beat = _normalise(onset_env[idx])

    # Spectral novelty per beat: how different the bar sounds either side of it.
    novelty = np.zeros(len(idx))
    if db_spec is not None and db_spec.shape[0] > 0 and len(idx) > 2:
        feats = []
        for a, b in zip(idx[:-1], idx[1:]):
            seg = db_spec[a:b] if b > a else db_spec[a : a + 1]
            feats.append(seg.mean(axis=0))
        feats = np.stack(feats)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.maximum(norms, 1e-9)
        # novelty at beat i is the change from the beat before it to the one after
        sim = np.sum(feats[:-1] * feats[1:], axis=1)
        novelty[1:-1] = 1.0 - sim[: len(novelty) - 2]
        novelty = _normalise(novelty)

    score_at_beat = 0.45 * novelty + 0.35 * low_at_beat + 0.20 * onset_at_beat

    scores = np.zeros(n_phase)
    for phase in range(n_phase):
        sel = score_at_beat[phase::n_phase]
        scores[phase] = float(sel.mean()) if sel.size else 0.0
    return int(np.argmax(scores)), scores


# --------------------------------------------------------------------------
# bands and transients
# --------------------------------------------------------------------------


BAND_EDGES = {
    "sub": (20.0, 60.0),
    "kick": (40.0, 140.0),
    "bass": (60.0, 250.0),
    "mid": (250.0, 2000.0),
    "high": (2000.0, 8000.0),
    "air": (8000.0, 11025.0),
}


def band_envelopes(
    x: np.ndarray, sr: int = ANALYSIS_SR, hop: int = HOP
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """RMS energy per named frequency band, one value per analysis frame."""
    spec = stft_magnitude(x, N_FFT, hop)
    if spec.shape[0] == 0:
        return {k: np.zeros(0, dtype=np.float32) for k in BAND_EDGES}, np.zeros(0)

    freqs = np.linspace(0.0, sr / 2.0, spec.shape[1])
    power = spec.astype(np.float64) ** 2

    out = {}
    for name, (lo, hi) in BAND_EDGES.items():
        sel = (freqs >= lo) & (freqs < hi)
        if not sel.any():
            out[name] = np.zeros(spec.shape[0], dtype=np.float32)
            continue
        rms = np.sqrt(power[:, sel].mean(axis=1))
        out[name] = _normalise(np.log1p(1000.0 * rms)).astype(np.float32)

    times = np.arange(spec.shape[0]) * hop / sr
    return out, times


def band_transients(
    db: np.ndarray,
    mel_hz: np.ndarray,
    fps: float,
    which=("kick", "high"),
    delta: float = 0.8,
) -> dict[str, np.ndarray]:
    """Onset times within individual bands: kick hits, hat and snare ticks."""
    out = {}
    for name in which:
        lo, hi = BAND_EDGES[name]
        flux = subband_flux(db, mel_hz, lo, hi)
        # A kick cannot repeat faster than about 8 Hz; hats can go quicker.
        min_gap = 0.12 if name in ("sub", "kick", "bass") else 0.06
        peaks = pick_peaks(flux, fps, min_gap_s=min_gap, delta=delta)
        out[name] = (peaks / fps).astype(np.float64)
    return out


def envelope_follower(
    env: np.ndarray, fps: float, attack_s: float = 0.01, release_s: float = 0.18
) -> np.ndarray:
    """Asymmetric smoothing: snap up on a hit, ease down after it.

    A raw band envelope is too twitchy to drive a visual parameter. This is the
    same one-pole follower a compressor uses, and it is what makes a curve read as
    "punch" rather than as noise.
    """
    if env.size == 0:
        return env
    a_att = float(np.exp(-1.0 / max(attack_s * fps, 1e-6)))
    a_rel = float(np.exp(-1.0 / max(release_s * fps, 1e-6)))
    out = np.empty_like(env, dtype=np.float32)
    y = float(env[0])
    for i, v in enumerate(env):
        coef = a_att if v > y else a_rel
        y = coef * y + (1.0 - coef) * float(v)
        out[i] = y
    return out


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------


def segment_sections(
    log_mel: np.ndarray,
    times: np.ndarray,
    beats_s: np.ndarray,
    duration: float,
    min_section_s: float = 8.0,
    max_sections: int = 12,
) -> np.ndarray:
    """Section boundaries by checkerboard novelty on a beat-synchronous SSM.

    Averaging features over each beat before building the self-similarity matrix
    is what makes this cheap: a 3 minute track goes from ~8000 frames to ~400
    beats, and the SSM from 64M cells to 160k.
    """
    if log_mel.shape[0] == 0 or beats_s.size < 8:
        return np.asarray([0.0, duration])

    idx = np.clip(np.searchsorted(times, beats_s), 0, log_mel.shape[0] - 1)
    feats = np.stack(
        [log_mel[a:b].mean(axis=0) if b > a else log_mel[a] for a, b in zip(idx[:-1], idx[1:])]
    )

    # cosine self-similarity
    norm = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.maximum(norm, 1e-9)
    ssm = feats @ feats.T

    n = ssm.shape[0]
    k = max(4, min(32, n // 8))
    kernel = _checkerboard(k)
    novelty = np.zeros(n)
    for i in range(k, n - k):
        novelty[i] = float((ssm[i - k : i + k, i - k : i + k] * kernel).sum())
    novelty = _normalise(np.maximum(novelty, 0.0))

    beat_period = float(np.median(np.diff(beats_s))) if beats_s.size > 1 else 0.5
    min_gap_beats = max(2, int(round(min_section_s / max(beat_period, 1e-6))))

    order = np.argsort(novelty)[::-1]
    chosen: list[int] = []
    for i in order:
        if novelty[i] <= 0.15:
            break
        if len(chosen) >= max_sections - 1:
            break
        if all(abs(i - c) >= min_gap_beats for c in chosen):
            chosen.append(int(i))

    bounds = sorted(float(beats_s[c]) for c in chosen if 0 <= c < beats_s.size)
    return np.asarray([0.0] + bounds + [duration])


def _checkerboard(k: int) -> np.ndarray:
    """Gaussian-tapered checkerboard kernel (Foote 2000)."""
    g = np.outer(
        np.exp(-0.5 * (np.linspace(-2, 2, 2 * k) ** 2)),
        np.exp(-0.5 * (np.linspace(-2, 2, 2 * k) ** 2)),
    )
    sign = np.ones((2 * k, 2 * k))
    sign[:k, k:] = -1.0
    sign[k:, :k] = -1.0
    return g * sign


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------


def analyse(
    waveform: np.ndarray,
    sample_rate: int,
    beats_per_bar: int = 4,
    bpm_min: float = 60.0,
    bpm_max: float = 200.0,
    prior_bpm: float = 120.0,
    tightness: float = 100.0,
    min_section_s: float = 8.0,
    grid_mode: str = "rigid",
    dp_margin: float = 1.25,
    phase_offset_beats: float = 0.0,
) -> AudioAnalysis:
    """Run the whole chain. ``waveform`` may be (n,), (c, n) or (b, c, n).

    ``grid_mode`` is "rigid" for one tempo across the whole track, "adaptive" to
    follow drift, or "auto" to take whichever scores better. Rigid is the default
    because cuts that stay locked for three minutes matter more than cuts that
    track a wobble, and most material fed to this is machine-timed.

    ``dp_margin`` only applies in "auto": how much better the drifting tracker
    must score before its drift is accepted.

    ``phase_offset_beats`` slides the whole grid, in beats, and takes fractions.
    It exists because a tempo estimator locks onto whichever pulse carries the
    most spectral flux, and on funk that is often the offbeat hats rather than
    the downbeat. The result is a grid half a beat early: musically it reads as
    cutting on the pickup, which is a real edit choice, but it should be a choice
    rather than an accident. Use 0.5 to move a half beat later, -0.25 for a
    sixteenth earlier. ``downbeat_offset`` only rotates whole beats within the
    bar and cannot express this.
    """
    mono = to_mono(waveform)
    x = resample(mono, sample_rate, ANALYSIS_SR)
    duration = len(x) / ANALYSIS_SR
    fps = ANALYSIS_SR / HOP

    onset_env, log_mel = onset_envelope(x, ANALYSIS_SR, HOP)
    bands, times = band_envelopes(x, ANALYSIS_SR, HOP)
    mel_hz = mel_frequencies(ANALYSIS_SR, N_MELS)

    # keep every array the same length; rfft framing can differ by one
    n = min(len(onset_env), len(times), *(len(v) for v in bands.values()))
    onset_env = onset_env[:n]
    times = times[:n]
    log_mel = log_mel[:n]
    bands = {k: v[:n] for k, v in bands.items()}

    # Two estimators. The rigid grid holds one tempo for the whole track; the DP
    # tracker follows drift.
    rigid_tempo, rigid_phase, rigid_lock = rigid_grid(
        onset_env, fps, duration, bpm_min, bpm_max, prior_bpm
    )
    rigid_beats = extend_beat_grid(
        np.asarray([rigid_phase]), duration, 60.0 / max(rigid_tempo, 1e-6)
    )

    ac_tempo, _ = estimate_tempo(onset_env, fps, bpm_min, bpm_max, prior_bpm)
    dp_beats = track_beats(onset_env, fps, ac_tempo, tightness) / fps
    dp_tempo = ac_tempo
    dp_resid = 0.0
    if dp_beats.size > 4:
        measured = float(np.median(np.diff(dp_beats)))
        if measured > 1e-6:
            dp_tempo = 60.0 / measured
        # Residual from a straight line separates real tempo drift from the
        # tracker dropping a beat. A slip shows up as a residual on the order of
        # a whole beat period; genuine rubato stays far smaller.
        k = np.arange(dp_beats.size)
        A = np.vstack([k, np.ones_like(k)]).T
        coef, *_ = np.linalg.lstsq(A, dp_beats, rcond=None)
        dp_resid = float(np.std(dp_beats - (coef[0] * k + coef[1])))
    dp_beats = extend_beat_grid(dp_beats, duration, 60.0 / max(dp_tempo, 1e-6))
    dp_lock = grid_lock(dp_beats, onset_env, fps)

    # Rigid by default. Comparing the two on lock alone is rigged in the DP
    # tracker's favour: a grid free to move raises its own score by snapping to
    # whichever onset is nearest, including on a track with a dead steady tempo,
    # and it collects that bonus even while slipping a beat. The cost is drift,
    # invisible in the score and very visible three minutes into a render. So the
    # flexible grid has to win by a margin, and has to be tracking a real line
    # rather than skipping beats, before its drift is worth accepting.
    # Genuine rubato keeps residuals to a small fraction of a beat. Anything
    # approaching half a period is the tracker having dropped or doubled a beat,
    # and a slipped grid must never be preferred however well it scores.
    dp_is_coherent = dp_resid < 0.15 * (60.0 / max(dp_tempo, 1e-6))
    if grid_mode == "adaptive" or (
        grid_mode == "auto" and dp_lock > rigid_lock * dp_margin and dp_is_coherent
    ):
        beats_s, tempo, lock, method = dp_beats, dp_tempo, dp_lock, "dp"
    else:
        beats_s, tempo, lock, method = rigid_beats, rigid_tempo, rigid_lock, "rigid"

    diagnostics = {
        "rigid_lock": rigid_lock,
        "rigid_tempo": rigid_tempo,
        "rigid_phase": rigid_phase,
        "dp_lock": dp_lock,
        "dp_tempo": dp_tempo,
        "dp_residual_ms": dp_resid * 1000.0,
        "dp_coherent": dp_is_coherent,
        "autocorr_tempo": ac_tempo,
    }

    # Slide the whole grid before anything is derived from it, so downbeats,
    # sections and every cut inherit the shift.
    if phase_offset_beats:
        beat_period = 60.0 / max(tempo, 1e-6)
        beats_s = extend_beat_grid(
            beats_s + phase_offset_beats * beat_period, duration, beat_period
        )
        lock = grid_lock(beats_s, onset_env, fps)

    # The tracker wants the std-scaled envelope; anything mixing it with band
    # energies wants it on a common 0..1 scale.
    onset_norm = _normalise(onset_env)

    phase, phase_scores = find_downbeat_phase(
        beats_s, times, bands["kick"], onset_norm, log_mel, beats_per_bar
    )
    downbeats = beats_s[phase::beats_per_bar]

    # How decisively the winning phase beat the runner-up, 0 to 1. Low means the
    # bar line is a coin toss and wants checking by ear.
    ordered = np.sort(phase_scores)[::-1]
    downbeat_confidence = (
        float((ordered[0] - ordered[1]) / max(ordered[0], 1e-9)) if ordered.size > 1 else 1.0
    )
    diagnostics["downbeat_scores"] = phase_scores.tolist()
    diagnostics["downbeat_confidence"] = downbeat_confidence

    bidx = np.clip(np.searchsorted(times, beats_s), 0, n - 1)
    beat_strength = onset_norm[bidx] if n else np.zeros(0, dtype=np.float32)

    transients = band_transients(log_mel, mel_hz, fps)
    sections = segment_sections(log_mel, times, beats_s, duration, min_section_s)

    # per-section mean energy, used later to set how fast that section cuts
    overall = _normalise(
        0.5 * bands["kick"] + 0.3 * bands["mid"] + 0.2 * bands["high"]
    )
    sec_energy = []
    for a, b in zip(sections[:-1], sections[1:]):
        ia, ib = int(a * fps), int(b * fps)
        seg = overall[ia:ib]
        sec_energy.append(float(seg.mean()) if seg.size else 0.0)
    sec_energy = np.asarray(sec_energy)
    # np.ptp, not sec_energy.ptp: NumPy 2.0 removed the ndarray method.
    if sec_energy.size and np.ptp(sec_energy) > 1e-9:
        sec_energy = (sec_energy - sec_energy.min()) / np.ptp(sec_energy)

    return AudioAnalysis(
        duration=duration,
        sample_rate=sample_rate,
        fps=fps,
        onset_env=onset_env,
        times=times,
        tempo=tempo,
        beats=beats_s,
        beat_strength=beat_strength,
        downbeats=downbeats,
        beats_per_bar=beats_per_bar,
        downbeat_phase=phase,
        bands=bands,
        transients=transients,
        sections=sections,
        section_energy=sec_energy,
        lock=lock,
        method=method,
        diagnostics=diagnostics,
    )
