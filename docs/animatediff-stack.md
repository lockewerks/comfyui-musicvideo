# The AnimateDiff stack

A second way to make a music video from the same audio analysis. The stack on
`main` cuts: it generates an independent clip per shot and the music decides
where the cuts land. This one does not cut. It generates one continuous
diffusion across the track and lets the music drive how much the picture moves
and how tightly it holds together.

They make different films. Pick by what you want, not by which is newer.

## What the music actually drives

```
energy  ->  scale_multival    on the motion model      louder moves more
energy  ->  effect_multival   on the motion model      quiet holds, loud churns
bars    ->  prompt keyframes  via ADE_PromptScheduling the picture turns over on the bar
```

Both curves are one float per frame, sliced per segment and handed to the
AnimateDiff model. That is the whole mechanism; there is no per-frame trickery
beyond it.

`scale_multival` is the amount of movement. 1.0 is the motion module's own
amount. The useful range is narrow: past about 1.5 it stops producing motion and
starts producing smear.

`effect_multival` is how strongly the motion module binds frames to each other,
which is the honest per-frame coherence control on this stack. It is the inverse
of energy on purpose, so quiet passages hold together and loud ones are free to
change. Keep it above about 0.7; below that the picture stops cohering and you
get flicker rather than energy.

## Frame rate is chosen, not assumed

`bar_aligned` mode picks a generation frame rate that puts one musical bar on
exactly `context_length` frames, so the model's temporal attention window and the
musical phrase are the same span. At 121.03 BPM with a 16 frame window that is
**8.068 fps**, and the whole 195 second track is **1574 frames** against 4876 for
the shot stack at 25 fps. Three times less to generate.

Deliver at a higher frame rate by interpolating the assembled video, not by
generating more frames. Generation cost is linear in frames; interpolation is
close to free and does not fight the diffusion model for VRAM. Do it as its own
pass over the finished video, never inside the render graph.

## There is no cut

AnimateDiff has no concept of a splice. `hard_cuts` writes each prompt index with
a trailing colon, which tells ADE to hold that prompt rather than interpolate
toward the next, and the picture turns over in a frame or two. That is as close
as this technique gets.

Left off, prompts crossfade into each other, which is the thing AnimateDiff is
actually good at. If you want real cuts, use the stack on `main`.

## Segments

A whole song does not fit in one sampling pass: the latent batch and the decoded
frames both scale with length. The track is split into segments on section
boundaries, snapped to bar lines, so the seams land where the music already
changes. Each segment is written to disk as it completes.

`decode_batch` is where this runs out of memory, not sampling. A whole segment of
frames decoded at once is the thing that will not fit.

## Known limits

**Segment seams are real.** Each segment is sampled independently, so the picture
does not carry across a seam. They are placed on section boundaries to hide this,
and it mostly works, but a slow section change will show a jump.

**Prompt travel is not direction.** You can say what the picture is at each bar.
You cannot say where the camera goes. Motion LoRAs (PanLeft, ZoomIn and so on)
would give directional control and none are installed here.

**Squarish is safer.** 512x512 is what the motion module was trained near.
Widescreen works but coherence degrades as you get further from that.

**A per-frame multival on NaiveReuse does not work.** In AnimateDiff-Evolved as
of this writing, `resize_multival` returns a 3D `[batch, height, width]` mask
while `NaiveReuseHandler.apply_cached` needs a 4D one. Broadcasting then lines the
batch dimension up against the latent's four channels and sampling dies with
`The size of tensor a (16) must match the size of tensor b (4)`. The
`naive_reuse` coherence mode therefore passes a plain float. This is why
coherence rides on `effect_multival` instead, which was the better control
anyway.

## Compared with the shot stack

| | shot stack (`main`) | AnimateDiff (`animatediff-t2v`) |
| --- | --- | --- |
| Structure | independent clip per cut | one continuous diffusion |
| Cuts | real, on bar lines | crossfade or fast turn only |
| Subject identity | changes every cut | holds across the track |
| Frames for a 195 s track | 4876 at 25 fps | 1574 at 8.07 fps |
| Base model | SDXL start frames, LTX video | SD1.5 with AnimateLCM |
| Resolution | 1280x704 | 512x512 |

The identity difference is the interesting one. The shot stack generates each
start frame independently, so the subject's face changes at every cut. This stack
generates one continuous latent, so the subject persists. Whether that matters
depends on whether your video has a character in it.
