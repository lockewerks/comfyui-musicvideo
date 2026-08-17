"""Nodes that turn an analysis into a shot list and a prompt book."""

from __future__ import annotations

import json

from ..musicvideo import plan as P
from ..musicvideo import prompts as PR

CATEGORY = "music video"


class MVShotPlan:
    """Cut the song into shots on bar lines."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analysis": ("MV_ANALYSIS",),
                "video_fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "width": ("INT", {"default": 1280, "min": 64, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 704, "min": 64, "max": 16384, "step": 32}),
                "min_bars": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 64.0, "step": 0.25}),
                "max_bars": ("FLOAT", {"default": 4.0, "min": 0.25, "max": 64.0, "step": 0.25}),
                "energy_drives_length": ("BOOLEAN", {"default": True}),
                "cut_on_sections": ("BOOLEAN", {"default": True}),
                "max_shot_seconds": ("FLOAT", {"default": 12.0, "min": 0.5, "max": 120.0, "step": 0.5}),
                "min_shot_seconds": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 30.0, "step": 0.1}),
                "frame_quantiser": (["ltx", "wan", "none"], {"default": "ltx"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("MV_PLAN", "STRING", "INT", "INT")
    RETURN_NAMES = ("plan", "report", "shot_count", "total_frames")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(
        self, analysis, video_fps, width, height, min_bars, max_bars,
        energy_drives_length, cut_on_sections, max_shot_seconds,
        min_shot_seconds, frame_quantiser, seed,
    ):
        p = P.plan_shots(
            analysis,
            video_fps=float(video_fps),
            width=int(width),
            height=int(height),
            min_bars=float(min_bars),
            max_bars=float(max_bars),
            energy_drives_length=bool(energy_drives_length),
            cut_on_sections=bool(cut_on_sections),
            max_shot_seconds=float(max_shot_seconds),
            min_shot_seconds=float(min_shot_seconds),
            quantiser=frame_quantiser,
            seed=int(seed),
        )

        problems = p.verify()
        if problems:
            # These are arithmetic invariants, not preferences. A plan with a gap
            # in it produces a video that drifts off the music, so refuse it here
            # rather than after an hour of rendering.
            raise RuntimeError(
                "shot plan failed verification:\n  " + "\n  ".join(problems)
            )

        report = p.summary() + "\n\nplan verified: no gaps, no overlaps, frames sum exactly"
        return (p, report, len(p.shots), p.total_frames)


class MVPromptBook:
    """Expand one master prompt into a per-shot prompt for every shot."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("MV_PLAN",),
                "analysis": ("MV_ANALYSIS",),
                "master_prompt": ("STRING", {"multiline": True, "default": ""}),
                "look": ("STRING", {"multiline": True, "default": "cinematic, 35mm film, shallow depth of field"}),
                "energy_influence": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "framings": ("STRING", {"multiline": True, "default": ""}),
                "lightings": ("STRING", {"multiline": True, "default": ""}),
                "camera_moves": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("MV_PLAN", "STRING")
    RETURN_NAMES = ("plan", "prompt_book")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(
        self, plan, analysis, master_prompt, look, energy_influence, seed,
        framings="", lightings="", camera_moves="",
    ):
        import copy

        # Copy: ComfyUI caches node outputs, and mutating an upstream plan in
        # place makes a re-run with a changed prompt silently keep the old one.
        p = copy.deepcopy(plan)
        PR.build_prompt_book(
            p, analysis,
            master_prompt=master_prompt,
            look=look,
            framings=framings,
            lightings=lightings,
            moves=camera_moves,
            energy_influence=float(energy_influence),
            seed=int(seed),
        )
        return (p, PR.prompt_book_text(p))


class MVShotInfo:
    """Everything about one shot, by index. Used inside the render expansion."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("MV_PLAN",),
                "index": ("INT", {"default": 0, "min": 0, "max": 100000}),
            }
        }

    RETURN_TYPES = (
        "STRING", "STRING", "INT", "INT", "INT", "INT", "INT", "FLOAT", "FLOAT", "FLOAT",
    )
    RETURN_NAMES = (
        "prompt", "motion", "gen_frames", "frame_count", "start_frame",
        "seed", "width", "height", "energy", "start_time",
    )
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, plan, index):
        i = max(0, min(int(index), len(plan.shots) - 1))
        s = plan.shots[i]
        return (
            s.prompt, s.motion, int(s.gen_frames), int(s.frame_count),
            int(s.start_frame), int(s.seed), int(plan.width), int(plan.height),
            float(s.energy), float(s.start_time),
        )


class MVPlanToJSON:
    """Serialise the plan, for saving next to the render or driving another tool."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"plan": ("MV_PLAN",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, plan):
        return (json.dumps(P.to_dict(plan), indent=2),)
