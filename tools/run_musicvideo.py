"""Build and submit the whole music video graph to a running ComfyUI.

Headless counterpart to the canvas workflow. Useful for long renders and for
proving a change end to end without clicking.

    python tools/run_musicvideo.py --audio superfunk.mp3 --limit 3
"""

import argparse
import json
import time
import urllib.error
import urllib.request

SERVER = "http://127.0.0.1:8188"


def post(path, payload):
    req = urllib.request.Request(
        SERVER + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen(SERVER + path) as r:
        return json.loads(r.read())


def build(args):
    """API-format prompt. Titles double as lookup keys, per the repo conventions."""
    g = {}

    def node(nid, class_type, title, **inputs):
        g[nid] = {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}
        return nid

    node("1", "LoadAudio", "Load audio", audio=args.audio)

    node("2", "MVAnalyzeAudio", "Analyse audio",
         audio=["1", 0], beats_per_bar=4, grid_mode=args.grid_mode,
         bpm_min=60.0, bpm_max=200.0, prior_bpm=args.prior_bpm,
         downbeat_offset=args.downbeat_offset, min_section_seconds=8.0)

    node("3", "MVShotPlan", "Shot plan",
         analysis=["2", 0], video_fps=args.fps,
         width=args.width, height=args.height,
         min_bars=args.min_bars, max_bars=args.max_bars,
         energy_drives_length=True, cut_on_sections=True,
         max_shot_seconds=12.0, min_shot_seconds=1.0,
         frame_quantiser="ltx", seed=args.seed)

    node("4", "MVPromptBook", "Prompt book",
         plan=["3", 0], analysis=["2", 0],
         master_prompt=args.prompt, look=args.look,
         energy_influence=0.7, seed=args.seed,
         framings="", lightings="", camera_moves="")

    node("5", "CheckpointLoaderSimple", "Load image checkpoint",
         ckpt_name=args.image_ckpt)

    node("6", "MVRenderStills", "Render start frames",
         plan=["4", 0], model=["5", 0], clip=["5", 1], vae=["5", 2],
         negative_prompt=args.negative,
         steps=args.still_steps, cfg=args.still_cfg,
         sampler_name=args.still_sampler, scheduler=args.still_scheduler,
         limit=args.limit)

    node("7", "CheckpointLoaderSimple", "Load LTX checkpoint",
         ckpt_name=args.video_ckpt)
    node("8", "CLIPLoader", "Load T5 text encoder",
         clip_name=args.text_encoder, type="ltxv", device="default")

    node("9", "MVRenderShots", "Render shots",
         plan=["4", 0], stills=["6", 0], model=["7", 0], clip=["8", 0],
         vae=["7", 2], backend="ltx",
         negative_prompt="static camera, frozen, jerky, stuttering, morphing, warping, blurry, low quality",
         steps=args.video_steps, cfg=1.0, sampler_name="euler",
         max_shift=2.05, base_shift=0.95, terminal=0.1, image_strength=1.0,
         output_prefix=args.prefix, crf=12, codec="h264", limit=args.limit)

    node("10", "MVAssembleVideo", "Assemble music video",
         shots=["9", 0], audio=["1", 0], plan=["4", 0],
         filename_prefix=args.out_prefix, audio_bitrate="320k",
         reencode=False, crf=16)

    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", default="superfunk.mp3")
    p.add_argument("--prompt", default=(
        "a lone dancer in a derelict 1970s ballroom, gold sequins, "
        "dust in the air, peeling wallpaper"))
    p.add_argument("--look", default="cinematic, 35mm film grain, shallow depth of field, warm analogue colour")
    p.add_argument("--negative", default="blurry, low quality, watermark, text, deformed hands")
    p.add_argument("--image-ckpt", default="Juggernaut_X_RunDiffusion_Hyper.safetensors")
    p.add_argument("--video-ckpt", default="ltxv-2b-0.9.8-distilled.safetensors")
    p.add_argument("--text-encoder", default="t5xxl_fp8_e4m3fn_scaled.safetensors")
    p.add_argument("--still-steps", type=int, default=8)
    p.add_argument("--still-cfg", type=float, default=2.0)
    p.add_argument("--still-sampler", default="dpmpp_sde")
    p.add_argument("--still-scheduler", default="karras")
    p.add_argument("--video-steps", type=int, default=8)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--min-bars", type=float, default=1.0)
    p.add_argument("--max-bars", type=float, default=4.0)
    p.add_argument("--grid-mode", default="rigid")
    p.add_argument("--prior-bpm", type=float, default=120.0)
    p.add_argument("--downbeat-offset", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="render only the first N shots")
    p.add_argument("--prefix", default="musicvideo/shots")
    p.add_argument("--out-prefix", default="musicvideo/final")
    p.add_argument("--dump", action="store_true", help="print the graph and exit")
    args = p.parse_args()

    graph = build(args)
    if args.dump:
        print(json.dumps(graph, indent=2))
        return 0

    try:
        r = post("/prompt", {"prompt": graph})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        print(f"submission rejected ({e.code}):\n{body}")
        return 1

    pid = r["prompt_id"]
    print(f"queued {pid}" + (f"  (first {args.limit} shots)" if args.limit else "  (whole song)"))

    t0 = time.time()
    last = ""
    while True:
        time.sleep(3)
        hist = get(f"/history/{pid}")
        if pid in hist:
            h = hist[pid]
            status = h.get("status", {})
            ok = status.get("status_str") == "success"
            print(f"\n{'finished' if ok else 'FAILED'} in {time.time()-t0:.0f}s")
            for out in h.get("outputs", {}).values():
                if "report" in out:
                    print(out["report"])
            if not ok:
                for m in status.get("messages", []):
                    print(m)
            return 0 if ok else 1
        q = get("/queue")
        running = len(q.get("queue_running", []))
        msg = f"running={running} pending={len(q.get('queue_pending', []))} t={time.time()-t0:.0f}s"
        if msg != last:
            print(msg, end="\r", flush=True)
            last = msg


if __name__ == "__main__":
    raise SystemExit(main())
