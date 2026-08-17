"""Render nodes: fan a shot plan out into per-shot chains, then assemble.

ComfyUI graphs are static, but a shot count is only known after the audio is
analysed. Node expansion closes that gap: a node returns a subgraph and the
executor runs it, so one queue run covers however many shots the song turned out
to need.

Two things here are deliberate and both are about not falling over at shot 50.

Shots are chained through their writers rather than fanned out in parallel. Each
writer takes the previous writer's output, so the executor cannot start shot N
until shot N-1 has been written and released. Fifty shots of decoded frames held
at once is about 50 GB.

The frames input on each writer is lazy. Non-lazy inputs resolve before
check_lazy_status runs, so the chain position is settled before the sampler for
that shot is even requested. Without it the executor is free to evaluate every
sampler first and hold the lot.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np
import torch

from comfy_execution.graph_utils import GraphBuilder

import comfy.samplers
import folder_paths

from ..musicvideo import plan as P

CATEGORY = "music video"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _images_to_uint8(images: torch.Tensor) -> np.ndarray:
    """ComfyUI IMAGE is (batch, height, width, channel) float 0..1."""
    arr = images.detach().cpu().numpy()
    return (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _encode_frames(frames: np.ndarray, path: str, fps: float, crf: int, codec: str) -> None:
    """Pipe raw RGB frames into ffmpeg. Frames are (n, h, w, 3) uint8."""
    n, h, w, _ = frames.shape
    if codec == "ffv1":
        vcodec = ["-c:v", "ffv1", "-level", "3"]
    elif codec == "prores":
        vcodec = ["-c:v", "prores_ks", "-profile:v", "3"]
    else:
        vcodec = ["-c:v", "libx264", "-preset", "medium", "-crf", str(int(crf)),
                  "-pix_fmt", "yuv420p"]

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
        "-r", f"{fps:.6f}", "-i", "-",
        *vcodec, path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        proc.stdin.write(frames.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode("utf-8", "replace")
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}:\n{err}")


def _shot_dir(prefix: str) -> str:
    base = folder_paths.get_output_directory()
    d = os.path.join(base, prefix)
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# accumulators
# --------------------------------------------------------------------------


class MVShotsBegin:
    """Empty shot collection. Seeds the head of the writer chain."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"output_prefix": ("STRING", {"default": "musicvideo/shots"})}}

    RETURN_TYPES = ("MV_SHOTS",)
    RETURN_NAMES = ("shots",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, output_prefix):
        # Always re-run. This node's output carries a wall-clock start time, and
        # its inputs are identical between runs, so ComfyUI's cache happily
        # returns the previous run's timestamp and every ETA after it is
        # computed against a start that already happened. NaN never compares
        # equal to itself, which is how a node declares itself always dirty.
        return float("nan")

    def run(self, output_prefix):
        return ({"dir": _shot_dir(output_prefix), "files": [], "started": time.time()},)


class MVWriteShot:
    """Trim one shot to length, encode it to disk, append it to the collection."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "previous": ("MV_SHOTS",),
                # Lazy so the previous shot is fully written before this shot's
                # sampler is even asked for. See module docstring.
                "images": ("IMAGE", {"lazy": True}),
                "plan": ("MV_PLAN",),
                "index": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "crf": ("INT", {"default": 12, "min": 0, "max": 51}),
                "codec": (["h264", "ffv1", "prores"], {"default": "h264"}),
            }
        }

    RETURN_TYPES = ("MV_SHOTS",)
    RETURN_NAMES = ("shots",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def check_lazy_status(self, previous, plan, index, crf, codec, images=None):
        if images is None:
            return ["images"]
        return []

    def run(self, previous, plan, index, crf, codec, images=None):
        i = max(0, min(int(index), len(plan.shots) - 1))
        shot = plan.shots[i]

        frames = _images_to_uint8(images)
        got = frames.shape[0]
        # Models quantise length (LTX wants 8n+1), so a shot generates at least
        # what it needs and the tail is discarded here. This trim is what keeps
        # the picture aligned to the music.
        if got < shot.frame_count:
            raise RuntimeError(
                f"shot {i} generated {got} frames but the plan needs "
                f"{shot.frame_count}; check the length wired into the video node"
            )
        frames = frames[: shot.frame_count]

        out_dir = previous.get("dir") or _shot_dir("musicvideo/shots")
        ext = {"h264": "mp4", "ffv1": "mkv", "prores": "mov"}[codec]
        path = os.path.join(out_dir, f"shot_{i:04d}.{ext}")

        t0 = time.time()
        _encode_frames(frames, path, plan.video_fps, crf, codec)
        dt = time.time() - t0

        files = list(previous.get("files", []))
        files.append({
            "index": i,
            "path": path,
            "frames": int(shot.frame_count),
            "start_frame": int(shot.start_frame),
        })
        done, total = len(files), len(plan.shots)
        elapsed = time.time() - previous.get("started", time.time())
        rate = elapsed / max(done, 1)
        print(
            f"[musicvideo] shot {i + 1}/{total}  {shot.frame_count} frames  "
            f"encode {dt:.1f}s  elapsed {elapsed / 60:.1f}m  "
            f"eta {(total - done) * rate / 60:.1f}m"
        )

        out = dict(previous)
        out["files"] = files
        return (out,)


class MVCollectImage:
    """Append one image to a growing batch. Head of the stills chain."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"lazy": True}),
                "index": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "total": ("INT", {"default": 1, "min": 1, "max": 100000}),
            },
            "optional": {"previous": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def check_lazy_status(self, index, total, previous=None, image=None):
        if image is None:
            return ["image"]
        return []

    def run(self, index, total, previous=None, image=None):
        print(f"[musicvideo] still {int(index) + 1}/{int(total)}")
        if previous is None:
            return (image,)
        return (torch.cat([previous, image], dim=0),)


# --------------------------------------------------------------------------
# expansion: stills
# --------------------------------------------------------------------------


class MVRenderStills:
    """Generate one start frame per shot, as a single image batch.

    Runs the image model to completion across every shot before the video stage
    begins, so the two models swap in VRAM once rather than once per shot.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("MV_PLAN",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "negative_prompt": ("STRING", {"multiline": True, "default": "blurry, low quality, watermark, text"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                # Render the first N shots only. The way to see whether a look
                # works without waiting for the whole song.
                "limit": ("INT", {"default": 0, "min": 0, "max": 100000}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stills",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, plan, model, clip, vae, negative_prompt, steps, cfg,
            sampler_name, scheduler, limit):
        shots = plan.shots[: limit] if limit else plan.shots
        if not shots:
            raise RuntimeError("shot plan is empty")

        g = GraphBuilder()
        neg = g.node("CLIPTextEncode", "neg", text=negative_prompt, clip=clip)
        latent = g.node(
            "EmptyLatentImage", "latent",
            width=int(plan.width), height=int(plan.height), batch_size=1,
        )

        tail = None
        for n, shot in enumerate(shots):
            pos = g.node("CLIPTextEncode", f"pos{n}", text=shot.prompt, clip=clip)
            samp = g.node(
                "KSampler", f"sample{n}",
                model=model, seed=int(shot.seed), steps=int(steps), cfg=float(cfg),
                sampler_name=sampler_name, scheduler=scheduler,
                positive=pos.out(0), negative=neg.out(0),
                latent_image=latent.out(0), denoise=1.0,
            )
            dec = g.node("VAEDecode", f"decode{n}", samples=samp.out(0), vae=vae)
            coll = g.node(
                "MVCollectImage", f"collect{n}",
                image=dec.out(0), index=n, total=len(shots),
            )
            if tail is not None:
                coll.set_input("previous", tail.out(0))
            tail = coll

        return {"result": (tail.out(0),), "expand": g.finalize()}


# --------------------------------------------------------------------------
# expansion: video
# --------------------------------------------------------------------------


class MVRenderShots:
    """Animate each start frame into a shot of exactly the planned length.

    The LTX backend is the sampling stack that model actually wants:
    ModelSamplingLTXV for the sigma shift, LTXVScheduler for the curve,
    KSamplerSelect and SamplerCustom. KSampler is not it.

    cfg stays at 1.0 for a distilled checkpoint. It is trained to run without
    classifier-free guidance, so raising cfg degrades the image rather than
    adding control, and with no unconditional pass the negative prompt does
    nothing at all. It is wired for structure and for the day a non-distilled
    checkpoint goes in.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("MV_PLAN",),
                "stills": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "backend": (["ltx", "wan"], {"default": "ltx"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "static camera, frozen, jerky, stuttering, morphing, warping, blurry, low quality"}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "max_shift": ("FLOAT", {"default": 2.05, "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_shift": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step": 0.01}),
                "terminal": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.99, "step": 0.01}),
                "image_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "output_prefix": ("STRING", {"default": "musicvideo/shots"}),
                "crf": ("INT", {"default": 12, "min": 0, "max": 51}),
                "codec": (["h264", "ffv1", "prores"], {"default": "h264"}),
                "limit": ("INT", {"default": 0, "min": 0, "max": 100000}),
            }
        }

    RETURN_TYPES = ("MV_SHOTS",)
    RETURN_NAMES = ("shots",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, plan, stills, model, clip, vae, backend, negative_prompt, steps,
            cfg, sampler_name, max_shift, base_shift, terminal, image_strength,
            output_prefix, crf, codec, limit):
        shots = plan.shots[: limit] if limit else plan.shots
        if not shots:
            raise RuntimeError("shot plan is empty")
        have = int(stills.shape[0])
        if have < len(shots):
            raise RuntimeError(
                f"got {have} start frames for {len(shots)} shots; "
                f"render stills with the same plan and limit"
            )

        g = GraphBuilder()
        begin = g.node("MVShotsBegin", "begin", output_prefix=output_prefix)
        neg = g.node("CLIPTextEncode", "neg", text=negative_prompt, clip=clip)
        sampler = g.node("KSamplerSelect", "sampler", sampler_name=sampler_name)

        tail = begin
        for n, shot in enumerate(shots):
            start = g.node(
                "ImageFromBatch", f"start{n}",
                image=stills, batch_index=n, length=1,
            )
            pos = g.node("CLIPTextEncode", f"motion{n}", text=shot.motion, clip=clip)

            if backend == "ltx":
                i2v = g.node(
                    "LTXVImgToVideo", f"i2v{n}",
                    positive=pos.out(0), negative=neg.out(0), vae=vae,
                    image=start.out(0),
                    width=int(plan.width), height=int(plan.height),
                    length=int(shot.gen_frames), batch_size=1,
                    strength=float(image_strength),
                )
                cond = g.node(
                    "LTXVConditioning", f"cond{n}",
                    positive=i2v.out(0), negative=i2v.out(1),
                    frame_rate=float(plan.video_fps),
                )
                msl = g.node(
                    "ModelSamplingLTXV", f"shift{n}",
                    model=model, max_shift=float(max_shift),
                    base_shift=float(base_shift), latent=i2v.out(2),
                )
                sigmas = g.node(
                    "LTXVScheduler", f"sigmas{n}",
                    steps=int(steps), max_shift=float(max_shift),
                    base_shift=float(base_shift), stretch=True,
                    terminal=float(terminal), latent=i2v.out(2),
                )
                samp = g.node(
                    "SamplerCustom", f"sample{n}",
                    model=msl.out(0), add_noise=True, noise_seed=int(shot.seed),
                    cfg=float(cfg), positive=cond.out(0), negative=cond.out(1),
                    sampler=sampler.out(0), sigmas=sigmas.out(0),
                    latent_image=i2v.out(2),
                )
                latent_out = samp.out(0)
            else:
                i2v = g.node(
                    "WanImageToVideo", f"i2v{n}",
                    positive=pos.out(0), negative=neg.out(0), vae=vae,
                    width=int(plan.width), height=int(plan.height),
                    length=int(shot.gen_frames), batch_size=1,
                    start_image=start.out(0),
                )
                samp = g.node(
                    "KSampler", f"sample{n}",
                    model=model, seed=int(shot.seed), steps=int(steps),
                    cfg=float(cfg), sampler_name=sampler_name, scheduler="simple",
                    positive=i2v.out(0), negative=i2v.out(1),
                    latent_image=i2v.out(2), denoise=1.0,
                )
                latent_out = samp.out(0)

            dec = g.node("VAEDecode", f"decode{n}", samples=latent_out, vae=vae)
            writer = g.node(
                "MVWriteShot", f"write{n}",
                previous=tail.out(0), images=dec.out(0), plan=plan,
                index=n, crf=int(crf), codec=codec,
            )
            tail = writer

        return {"result": (tail.out(0),), "expand": g.finalize()}


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


class MVAssembleVideo:
    """Concatenate the shots in order and mux the original audio over them."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shots": ("MV_SHOTS",),
                "audio": ("AUDIO",),
                "plan": ("MV_PLAN",),
                "filename_prefix": ("STRING", {"default": "musicvideo/final"}),
                "audio_bitrate": ("STRING", {"default": "320k"}),
                "reencode": ("BOOLEAN", {"default": False}),
                "crf": ("INT", {"default": 16, "min": 0, "max": 51}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("path", "report")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, shots, audio, plan, filename_prefix, audio_bitrate, reencode, crf):
        files = sorted(shots.get("files", []), key=lambda f: f["index"])
        if not files:
            raise RuntimeError("no shots to assemble")

        expected = sum(f["frames"] for f in files)
        out_dir = folder_paths.get_output_directory()
        stem = os.path.join(out_dir, filename_prefix)
        os.makedirs(os.path.dirname(stem), exist_ok=True)

        n = 0
        while os.path.exists(f"{stem}_{n:04d}.mp4"):
            n += 1
        out_path = f"{stem}_{n:04d}.mp4"

        work = shots.get("dir") or out_dir
        list_path = os.path.join(work, "concat.txt")
        with open(list_path, "w", encoding="utf-8") as fh:
            for f in files:
                p = f["path"].replace("\\", "/").replace("'", r"'\''")
                fh.write(f"file '{p}'\n")

        # Raw float audio straight to ffmpeg, so nothing is quantised on the way
        # through and the muxed track is exactly what came in.
        wf = audio["waveform"]
        if hasattr(wf, "detach"):
            wf = wf.detach().cpu().numpy()
        wf = np.asarray(wf)
        while wf.ndim > 2:
            wf = wf[0]
        if wf.ndim == 1:
            wf = wf[None, :]
        channels = wf.shape[0]
        interleaved = wf.T.astype(np.float32).tobytes()
        raw_path = os.path.join(work, "audio.f32")
        with open(raw_path, "wb") as fh:
            fh.write(interleaved)

        vcodec = (
            ["-c:v", "libx264", "-preset", "medium", "-crf", str(int(crf)), "-pix_fmt", "yuv420p"]
            if reencode
            else ["-c:v", "copy"]
        )
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", list_path,
            "-f", "f32le", "-ar", str(int(audio["sample_rate"])), "-ac", str(channels),
            "-i", raw_path,
            *vcodec,
            "-c:a", "aac", "-b:a", str(audio_bitrate),
            "-shortest", "-movflags", "+faststart",
            out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "ffmpeg assembly failed:\n" + proc.stderr.decode("utf-8", "replace")
            )

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames,width,height,r_frame_rate",
             "-show_entries", "format=duration", "-of", "json", out_path],
            capture_output=True,
        )
        info = json.loads(probe.stdout or b"{}")

        report = "\n".join([
            f"wrote {out_path}",
            f"shots {len(files)}, frames {expected}, planned {plan.total_frames}",
            f"probe: {json.dumps(info.get('streams', [{}])[0])}",
            f"duration: {info.get('format', {}).get('duration')} s "
            f"(audio {plan.duration:.2f} s)",
        ])
        print("[musicvideo] " + report.replace("\n", "\n[musicvideo] "))
        return (out_path, report)
