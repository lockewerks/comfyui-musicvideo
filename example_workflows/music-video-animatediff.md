# A music video as one continuous diffusion, driven by the music

Takes a song and a master prompt and returns a video the full length of the
track. Unlike the shot-based workflow next to it, this one never cuts. The
picture is a single continuous AnimateDiff generation and the audio drives how
much it moves and how tightly it coheres, frame by frame.

Reach for it when you want flow rather than an edit: the subject persists across
the whole track instead of changing at every cut, and the picture turns over on
the bar rather than being spliced.

Runs in one queue press. The segment count is only known after the audio has been
analysed, so the render node expands into one sampling chain per segment.

## Requires

- Custom nodes: [comfyui-musicvideo](https://github.com/lockewerks/comfyui-musicvideo),
  branch `animatediff-t2v`, for every `MV*` node here
- Custom nodes: [AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)
- Custom nodes: [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite),
  for `VHS_VAEDecodeBatched`
- Motion module: `AnimateLCM_sd15_t2v.ckpt` in `models/animatediff_models`
  (https://huggingface.co/wangfuyun/AnimateLCM, 1.8 GB)
- LoRA: `AnimateLCM_sd15_t2v_lora.safetensors` in `models/loras`, same source.
  Not optional; it is what makes eight steps work.
- Checkpoint: any SD1.5 model in `models/checkpoints`. Ships pointed at
  `DreamShaper_8_pruned.safetensors`.
- An audio file in `input/`. Ships pointed at `superfunk.mp3`, included with the
  node pack.

## Cost

Measured on an RTX 4090 24 GB against a 195.04 s track at 512x512.

The analysis planned 13 segments and 1574 frames at 8.068 fps, which is the rate
that puts one bar on exactly 16 frames.

- Whole track, 13 segments and 1574 frames: **12 minutes**, about 0.46 s per
  output frame.
- Output: 512x512 at 8.068 fps, H.264 with the original audio muxed as AAC.
  195.09 s of video against 195.04 s of audio. 87 MB.

Generating 1574 frames rather than the shot stack's 4876 is where the saving is.
Deliver at a higher frame rate by interpolating the finished video, as its own
pass, never inside this graph.

## Knobs that matter

- **`limit` on Render segments.** 2 renders the first two segments in about three
  minutes to check a look. 0 renders the whole song.
- **`coherence_mode`.** `effect_multival` is the one to use: it scales how
  strongly the motion module binds frames, per frame, and is the honest coherence
  control here. `naive_reuse` is scalar only, for the reason in Known limits.
- **`coherence_low` applies at peak energy.** Keep it above about 0.7. Below
  that the motion module stops holding the picture together and the result is
  flicker rather than energy.
- **`motion_high` past about 1.5 produces smear, not motion.** 1.0 is the motion
  module's own amount of movement; the useful range around it is narrow.
- **`hard_cuts` on Prompt travel schedule.** Off, prompts crossfade, which is what
  this technique does well. On, each prompt is held and the picture turns over in
  a frame or two. There is no real cut available either way.
- **`frame_rate_mode` bar_aligned** picks the generation rate so one bar is
  exactly `context_length` frames, putting the model's attention window and the
  musical phrase on the same span.
- **`beta_schedule` must be `sqrt_linear (AnimateDiff)`** for an SD1.5 motion
  module.
- `decode_batch` is where this runs out of memory, not sampling.
- Sampling is AnimateLCM: 8 steps, cfg 1.8, `lcm` sampler, `sgm_uniform`.

## Known limits

**No cuts.** AnimateDiff has no splice. If you want real cuts on the beat, use
`music-video-beat-cut` instead.

**Segment seams.** Each segment samples independently, so the picture does not
carry across a seam. Seams are placed on section boundaries to hide this and
mostly do, but a slow section change can show a jump.

**Prompt travel is not camera direction.** You can say what the picture is at
each bar, not where the camera goes. Motion LoRAs would give directional control.

**A per-frame multival on NaiveReuse is broken upstream.** In AnimateDiff-Evolved
as of this writing `resize_multival` returns a 3D mask where
`NaiveReuseHandler.apply_cached` needs 4D, so the batch dimension broadcasts
against the latent's channels and sampling dies with `The size of tensor a (16)
must match the size of tensor b (4)`. Hence scalar only in that mode.

**Squarish is safer.** 512x512 is near what the motion module was trained on.
Widescreen works and coherence degrades as you move away from it.
