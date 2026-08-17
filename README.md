# comfyui-musicvideo

Cut a music video to the beat, inside ComfyUI.

Give it a song and one master prompt. It analyses the audio, plans an edit that
cuts on bar lines, expands the prompt into a different shot for every cut,
renders them and assembles the result against the original audio. One queue
press, a video the full length of the track.

Shot lengths follow the music: loud sections cut fast, quiet ones hold. Cuts land
on bars, never on a timer.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/lockewerks/comfyui-musicvideo
```

Restart ComfyUI. That is the whole installation.

**No Python dependencies.** All the analysis runs on numpy and scipy, which
ComfyUI already ships, and `ffmpeg` is called as a subprocess. librosa, madmom,
aubio and essentia are all excellent and all absent from a stock ComfyUI venv;
requiring a compiled audio stack to render a video is a bad trade.

You need `ffmpeg` and `ffprobe` on PATH.

## Getting started

`example_workflows/music-video-beat-cut.json` is a complete, working graph. Drag
it onto the canvas. `examples/superfunk.mp3` is a demo track: copy it into
`ComfyUI/input/` and the workflow runs as-is.

Set `limit` to 8 on both render nodes first. That renders the opening eight shots
in about three minutes so you can judge the look before committing to a full
song. Set both to 0 for the whole track.

## The nodes

| Node | What it does |
| --- | --- |
| **Analyse audio** | Tempo, beat grid, bar lines, per-band transients, section boundaries. Reports a lock score so you know whether to trust it. |
| **Shot plan** | Cuts the song into shots on bar lines. Shot length follows section energy. |
| **Prompt book** | Expands one master prompt into a per-shot prompt, varying framing, lighting and camera move by the shot's energy. |
| **Render start frames** | One still per shot, as a single image batch. |
| **Render shots** | Animates each still to exactly the planned length. LTX backend. |
| **Assemble music video** | Concatenates the shots and muxes the original audio. |
| **Audio curve** | Per-frame FLOAT curve from any band, beat pulse or section, for wiring into schedulers. |
| **Beat times** | Beat, downbeat or transient times as a FLOAT list. |
| **Shot plan to JSON** | Serialises the edit, for driving another tool. |

## How the beat detection works

Mel spectral-flux onset envelope, then a Fourier tempogram to find the tempo and
its phase, then a fine sweep scored by how much onset energy actually lands on
the grid. A dynamic-programming tracker is available for material that drifts.

The analyser reports a **lock score**: onset strength on the beat grid over onset
strength overall. 1.0 is chance, 2 to 4 is a real lock, and below about 1.6 the
grid is not locked and the cuts will look arbitrary. It says so rather than
returning a confident wrong answer.

`grid_mode` defaults to `rigid`: one tempo across the whole track, zero drift. A
tracker free to follow the music will always score better, because it snaps to
whichever onset is nearest, and it collects that bonus even while slipping a
beat. The cost is drift, which is invisible in the score and very visible three
minutes into a render.

### When the cuts feel off

Two different knobs, and they are not interchangeable:

- `downbeat_offset` rotates whole beats within the bar. Use it when the beat feels
  right but the bar starts in the wrong place.
- `phase_offset_beats` slides the entire grid and takes fractions. Use it when the
  cuts sit slightly before or after the beat.

A tempo estimator locks onto whichever pulse carries the most spectral flux, and
on funk that is often the offbeat hats rather than the downbeat. The result is a
grid half a beat early: it reads as cutting on the pickup, which is a real edit
choice, but it should be a choice. `phase_offset_beats` of 0.5 moves it onto the
downbeat.

The lock score can prefer the wrong answer here, because it measures where the
transients are rather than where beat 1 is. Trust your ear over the number.

## Checking the analysis before you render

```
python tools/probe_song.py path/to/song.mp3 --click check.wav --phase-variants
```

Prints the diagnostics and writes a click track: the music at -6 dB with a click
on every beat and a higher one on bar lines, cut to a 16 second excerpt from the
busiest part of the track. `--phase-variants` writes one file per candidate bar
phase, so a wrong downbeat is settled in one sitting instead of one guess per
render.

Listening is not optional. Downbeat phase is frequently a near tie, and no metric
settles it.

```
python tools/smoke_test.py path/to/song.mp3
```

Runs analysis and planning without rendering, verifies the frame arithmetic, and
finishes in about a second. Run it after changing anything in `musicvideo/`.

```
python tools/run_musicvideo.py --audio superfunk.mp3 --limit 8
```

Builds and submits the whole graph to a running ComfyUI, headless.

## Notes on the design

**Cuts land on bar lines, and shot lengths are differences between cumulative
frame positions.** Rounding each shot independently and summing accumulates
error until the picture drifts off the music, and it drifts slowly enough to look
fine for the first thirty seconds. The planner asserts that shot frames sum
exactly to the track length and refuses a plan that does not.

**Shot count is only known after the audio is analysed,** which a static graph
cannot express. The render nodes use ComfyUI's node expansion to fan out into one
chain per shot at runtime.

**Shots are chained through their writers, not fanned out in parallel,** and the
frames input on each writer is lazy. Non-lazy inputs resolve before
`check_lazy_status` runs, so shot N-1 is written and released before shot N's
sampler is requested. Fifty shots of decoded frames held at once is about 50 GB.

**Every shot is written to disk as it finishes,** so a failure at shot 50 does not
cost the run.

## Requirements

- ComfyUI recent enough to support node expansion and lazy inputs
- `ffmpeg` and `ffprobe` on PATH
- For the example workflow: an SDXL checkpoint for start frames, plus
  `ltxv-2b-0.9.8-distilled.safetensors` and `t5xxl_fp8_e4m3fn_scaled.safetensors`

Measured on an RTX 4090 24 GB: a 195 second track planned to 58 shots and 4876
frames at 1280x704, and rendered in about 17 minutes, peaking at 22.0 GB of VRAM
during the start-frame pass.

## Known limits

**Identity is not held across cuts.** Each start frame is an independent
generation, so the subject's face and hair change from shot to shot. The world
stays consistent, the person does not. An IPAdapter reference feeding every start
frame is the fix and is not yet wired in.

**LTX only.** The `backend` selector has a Wan path, but it is untested.

**A `success` status proves nothing.** Ask a video model for a clip longer than it
can hold and it returns a valid file of the right length containing mud or black.
Check the last frame of a shot, never the first: in image-to-video the first frame
is baked in from the start frame and is guaranteed to look right.

## The demo track

`examples/superfunk.mp3` is included so the example workflow runs without
sourcing audio first. It analyses to 121.03 BPM with a lock of 2.70.

## Licence

MIT. See LICENSE.
