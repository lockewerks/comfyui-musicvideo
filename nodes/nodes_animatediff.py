"""AnimateDiff text-to-video nodes: one continuous picture driven by the music.

The shot-based stack in nodes_render.py generates an independent clip per cut.
This one generates a single continuous diffusion and lets the audio drive how
much it moves and how tightly it coheres, which is what AnimateDiff is actually
good at. It cannot cut. A prompt keyframe with interpolation disabled turns the
picture over in a frame or two, and that is the closest thing available.

Everything audio-facing is shared with the other stack: same analysis, same
`MV_ANALYSIS`, same `MV_PLAN` container, same writer chain and assembler. Only
the render stage differs.
"""

from __future__ import annotations

import numpy as np

from comfy_execution.graph_utils import GraphBuilder

import comfy.samplers

from ..musicvideo import animatediff as AD
from ..musicvideo import plan as P

CATEGORY = "music video/animatediff"


class MVADSegmentPlan:
    """Split the track into AnimateDiff render segments and pick a frame rate."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analysis": ("MV_ANALYSIS",),
                "frame_rate_mode": (["bar_aligned", "manual"], {"default": "bar_aligned"}),
                "context_length": ("INT", {"default": 16, "min": 4, "max": 128}),
                "manual_fps": ("FLOAT", {"default": 12.5, "min": 1.0, "max": 60.0, "step": 0.01}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "max_segment_seconds": ("FLOAT", {"default": 24.0, "min": 2.0, "max": 120.0, "step": 1.0}),
                "min_segment_seconds": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 60.0, "step": 0.5}),
                "cut_on_sections": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("MV_PLAN", "STRING", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("plan", "report", "gen_fps", "context_length", "segment_count")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, analysis, frame_rate_mode, context_length, manual_fps, width, height,
            max_segment_seconds, min_segment_seconds, cut_on_sections, seed):
        if frame_rate_mode == "bar_aligned":
            gen_fps = AD.suggest_gen_fps(analysis, int(context_length))
        else:
            gen_fps = float(manual_fps)

        plan = AD.plan_segments(
            analysis, gen_fps, width=int(width), height=int(height),
            max_segment_seconds=float(max_segment_seconds),
            min_segment_seconds=float(min_segment_seconds),
            cut_on_sections=bool(cut_on_sections), seed=int(seed),
        )
        problems = plan.verify()
        if problems:
            raise RuntimeError("segment plan failed verification:\n  " + "\n  ".join(problems))

        bar_seconds = 60.0 * analysis.beats_per_bar / max(analysis.tempo, 1e-6)
        frames_per_bar = bar_seconds * gen_fps
        report = "\n".join([
            plan.summary(),
            "",
            f"generation fps {gen_fps:.3f}   one bar = {frames_per_bar:.1f} frames",
            f"context length {int(context_length)}"
            + ("  (bar aligned)" if frame_rate_mode == "bar_aligned" else ""),
            "",
            "Deliver at a higher frame rate by interpolating the assembled video,",
            "not by generating more frames. Generation cost is linear in frames.",
        ])
        return (plan, report, float(gen_fps), int(context_length), len(plan.shots))


class MVADPromptSchedule:
    """Build one ADE prompt-travel schedule per segment from a master prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("MV_PLAN",),
                "analysis": ("MV_ANALYSIS",),
                "master_prompt": ("STRING", {"multiline": True, "default": ""}),
                "look": ("STRING", {"multiline": True, "default": "cinematic, 35mm film grain, volumetric light"}),
                "bars_per_keyframe": ("INT", {"default": 2, "min": 1, "max": 32}),
                # False crossfades between prompts, which is what this technique
                # does well. True holds each prompt, which reads as a fast turn.
                "hard_cuts": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "framings": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("MV_PLAN", "STRING")
    RETURN_NAMES = ("plan", "schedule_preview")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, plan, analysis, master_prompt, look, bars_per_keyframe,
            hard_cuts, seed, framings=""):
        import copy
        from ..musicvideo.prompts import _parse_vocab, FRAMINGS

        p = copy.deepcopy(plan)
        vocab = _parse_vocab(framings, FRAMINGS)
        schedules = AD.build_prompt_schedule(
            p, analysis, master_prompt=master_prompt, look=look,
            variations=vocab, bars_per_keyframe=int(bars_per_keyframe),
            hard_cuts=bool(hard_cuts), seed=int(seed),
        )
        # The schedule rides on the shot's prompt field so it travels with the plan.
        for shot, sched in zip(p.shots, schedules):
            shot.prompt = sched

        preview = AD.schedule_summary(schedules, p)
        if schedules:
            preview += "\n\nsegment 0 schedule:\n" + schedules[0]
        return (p, preview)


class MVADCurvePreview:
    """The motion and coherence curves, for inspection and for wiring by hand."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analysis": ("MV_ANALYSIS",),
                "plan": ("MV_PLAN",),
                "motion_low": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 3.0, "step": 0.01}),
                "motion_high": ("FLOAT", {"default": 1.35, "min": 0.0, "max": 3.0, "step": 0.01}),
                "coherence_low": ("FLOAT", {"default": 0.82, "min": 0.0, "max": 1.0, "step": 0.01}),
                "coherence_high": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "band": (["kick", "sub", "bass", "mid", "high", "air"], {"default": "kick"}),
            }
        }

    RETURN_TYPES = ("FLOATS", "FLOATS", "STRING")
    RETURN_NAMES = ("motion", "coherence", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, analysis, plan, motion_low, motion_high,
            coherence_low, coherence_high, band):
        n = plan.total_frames
        motion = AD.motion_curve(analysis, n, plan.video_fps, band=band,
                                 low=float(motion_low), high=float(motion_high))
        coh = AD.coherence_curve(analysis, n, plan.video_fps,
                                 low=float(coherence_low), high=float(coherence_high))
        m, c = np.asarray(motion), np.asarray(coh)
        report = (
            f"{n} frames at {plan.video_fps:.3f} fps\n"
            f"motion    min {m.min():.3f} mean {m.mean():.3f} max {m.max():.3f}\n"
            f"coherence min {c.min():.3f} mean {c.mean():.3f} max {c.max():.3f}\n"
            f"correlation {float(np.corrcoef(m, c)[0, 1]):+.3f} (should be negative)\n"
            "coherence drives effect_multival: how strongly the motion module\n"
            "binds frames. Inverse of energy on purpose, so quiet passages hold\n"
            "together and loud ones are free to change."
        )
        return (motion, coh, report)


class MVADRenderSegments:
    """Render every segment with AnimateDiff, driven by the audio curves.

    Expands into one sampling chain per segment, chained through the writers so
    segments run strictly in order and only one segment's frames are resident.

    Per segment the expansion wires:
      ADE_LoadAnimateDiffModel -> ADE_ApplyAnimateDiffModel(scale_multival)
      ADE_ContextExtras_NaiveReuse(strength_multival) -> ADE_ContextExtras_Set
      ADE_UseEvolvedSampling -> KSampler -> VHS_VAEDecodeBatched -> MVWriteShot

    scale_multival and strength_multival carry a float per frame, sliced from the
    track-wide curves to that segment's range. That is where the music actually
    reaches the picture.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("MV_PLAN",),
                "analysis": ("MV_ANALYSIS",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "motion_model_name": ("STRING", {"default": "AnimateLCM_sd15_t2v.ckpt"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "watermark, text, blurry, low quality, deformed, extra limbs"}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.8, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "lcm"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "sgm_uniform"}),
                "beta_schedule": (
                    ["autoselect", "use existing", "sqrt_linear (AnimateDiff)",
                     "linear (AnimateDiff-SDXL)", "linear (HotshotXL/default)"],
                    {"default": "sqrt_linear (AnimateDiff)"},
                ),
                "context_length": ("INT", {"default": 16, "min": 4, "max": 128}),
                "context_overlap": ("INT", {"default": 4, "min": 0, "max": 64}),
                "fuse_method": (["pyramid", "flat", "overlap-linear"], {"default": "pyramid"}),
                "motion_low": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 3.0, "step": 0.01}),
                "motion_high": ("FLOAT", {"default": 1.35, "min": 0.0, "max": 3.0, "step": 0.01}),
                "coherence_mode": (
                    ["effect_multival", "naive_reuse", "none"],
                    {"default": "effect_multival"},
                ),
                # Applies at peak energy. Below about 0.7 the motion module stops
                # holding the picture together and you get flicker, not energy.
                "coherence_low": ("FLOAT", {"default": 0.82, "min": 0.0, "max": 1.0, "step": 0.01}),
                "coherence_high": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "naive_reuse_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "energy_band": (["kick", "sub", "bass", "mid", "high", "air"], {"default": "kick"}),
                "decode_batch": ("INT", {"default": 16, "min": 1, "max": 256}),
                "output_prefix": ("STRING", {"default": "musicvideo/ad_segments"}),
                "crf": ("INT", {"default": 12, "min": 0, "max": 51}),
                "limit": ("INT", {"default": 0, "min": 0, "max": 100000}),
            }
        }

    RETURN_TYPES = ("MV_SHOTS",)
    RETURN_NAMES = ("segments",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, plan, analysis, model, clip, vae, motion_model_name, negative_prompt,
            steps, cfg, sampler_name, scheduler, beta_schedule, context_length,
            context_overlap, fuse_method, motion_low, motion_high, coherence_mode,
            coherence_low, coherence_high, naive_reuse_strength, energy_band,
            decode_batch, output_prefix, crf, limit):

        segments = plan.shots[:limit] if limit else plan.shots
        if not segments:
            raise RuntimeError("segment plan is empty")
        if not any(s.prompt for s in segments):
            raise RuntimeError(
                "no prompt schedules on the plan; wire MVADPromptSchedule before this node"
            )

        # Track-wide curves, sliced per segment below.
        motion = AD.motion_curve(analysis, plan.total_frames, plan.video_fps,
                                 band=energy_band, low=float(motion_low), high=float(motion_high))
        coherence = AD.coherence_curve(analysis, plan.total_frames, plan.video_fps,
                                       low=float(coherence_low), high=float(coherence_high))

        g = GraphBuilder()
        begin = g.node("MVShotsBegin", "begin", output_prefix=output_prefix)
        motion_model = g.node("ADE_LoadAnimateDiffModel", "motion",
                              model_name=motion_model_name)
        neg = g.node("CLIPTextEncode", "neg", text=negative_prompt, clip=clip)

        tail = begin
        for n, seg in enumerate(segments):
            a, b = seg.start_frame, seg.start_frame + seg.frame_count
            seg_motion = motion[a:b]
            seg_coherence = coherence[a:b]
            # A multival shorter than the batch silently misaligns the curve
            # against the picture, so pad rather than let it slide.
            if len(seg_motion) < seg.frame_count:
                seg_motion = seg_motion + [seg_motion[-1] if seg_motion else 1.0] * (
                    seg.frame_count - len(seg_motion))
            if len(seg_coherence) < seg.frame_count:
                seg_coherence = seg_coherence + [seg_coherence[-1] if seg_coherence else 0.0] * (
                    seg.frame_count - len(seg_coherence))

            scale_mv = g.node("ADE_MultivalDynamicFloats", f"scale{n}", floats=seg_motion)
            coh_mv = g.node("ADE_MultivalDynamicFloats", f"coh{n}", floats=seg_coherence)

            apply_kwargs = dict(
                motion_model=motion_model.out(0),
                start_percent=0.0, end_percent=1.0,
                scale_multival=scale_mv.out(0),
            )
            if coherence_mode == "effect_multival":
                # How strongly the motion module binds frames together, per
                # frame. Same code path as scale_multival, which is why this is
                # the coherence control rather than NaiveReuse.
                apply_kwargs["effect_multival"] = coh_mv.out(0)
            m_models = g.node("ADE_ApplyAnimateDiffModel", f"apply{n}", **apply_kwargs)

            ctx = g.node("ADE_StandardUniformContextOptions", f"ctx{n}",
                         context_length=int(context_length), context_stride=1,
                         context_overlap=int(context_overlap), fuse_method=fuse_method)
            context_out = ctx.out(0)

            if coherence_mode == "naive_reuse":
                # Scalar only. A per-frame multival here hits a shape bug in
                # AnimateDiff-Evolved: resize_multival hands back a 3D mask where
                # the arithmetic needs 4D, so the batch dimension broadcasts
                # against the latent channels and sampling dies.
                reuse = g.node("ADE_ContextExtras_NaiveReuse", f"reuse{n}",
                               strength_multival=float(naive_reuse_strength),
                               start_percent=0.0, end_percent=0.15,
                               weighted_mean=0.95)
                ctx_set = g.node("ADE_ContextExtras_Set", f"ctxset{n}",
                                 context_opts=ctx.out(0), context_extras=reuse.out(0))
                context_out = ctx_set.out(0)

            evolved = g.node("ADE_UseEvolvedSampling", f"evolved{n}",
                             model=model, beta_schedule=beta_schedule,
                             m_models=m_models.out(0), context_options=context_out)

            pos = g.node("ADE_PromptScheduling", f"pos{n}",
                         prompts=seg.prompt, clip=clip, max_length=int(seg.frame_count))

            latent = g.node("ADE_EmptyLatentImageLarge", f"latent{n}",
                            width=int(plan.width), height=int(plan.height),
                            batch_size=int(seg.frame_count))

            samp = g.node("KSampler", f"sample{n}",
                          model=evolved.out(0), seed=int(seg.seed), steps=int(steps),
                          cfg=float(cfg), sampler_name=sampler_name, scheduler=scheduler,
                          positive=pos.out(0), negative=neg.out(0),
                          latent_image=latent.out(0), denoise=1.0)

            # Batched decode: a whole segment of frames decoded at once is where
            # this stack runs out of memory, not sampling.
            dec = g.node("VHS_VAEDecodeBatched", f"decode{n}",
                         samples=samp.out(0), vae=vae, per_batch=int(decode_batch))

            writer = g.node("MVWriteShot", f"write{n}",
                            previous=tail.out(0), images=dec.out(0), plan=plan,
                            index=n, crf=int(crf), codec="h264")
            tail = writer

        return {"result": (tail.out(0),), "expand": g.finalize()}
