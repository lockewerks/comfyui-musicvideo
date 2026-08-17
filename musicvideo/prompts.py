"""Turn one master prompt into a per-shot prompt book.

The master prompt is the thing that must not change: subject, world, look. Each
shot then varies framing, lighting and camera move, chosen deterministically from
the shot's seed and its energy so the same song and seed always cut the same way.

Two constraints from experience shape this:

Prompt weighting has a fixed attention budget. Piling four requirements into one
prompt means one of them loses, and more prompt engineering only rotates which.
So a shot prompt is the master plus one framing and one lighting, and nothing
else fights for room.

A start frame anchors frame one and argues for nothing after it. Anything that
must hold for the whole clip belongs in the motion prompt, which is why the
motion prompt restates the subject rather than only naming a camera move.
"""

from __future__ import annotations

import hashlib
from typing import Optional, Sequence

from .analysis import AudioAnalysis
from .plan import ShotPlan


# Ordered low energy to high. A shot's energy picks a window into these, so a
# breakdown gets wide and static and a chorus gets close and dynamic.
FRAMINGS = [
    "extreme wide establishing shot",
    "wide shot",
    "full shot",
    "medium wide shot",
    "medium shot",
    "medium close-up",
    "close-up",
    "extreme close-up",
    "low angle hero shot",
    "dutch angle close-up",
]

LIGHTING = [
    "soft diffused daylight",
    "overcast flat light",
    "golden hour backlight",
    "hard directional sunlight",
    "deep shadow with a single practical",
    "neon rim light, cyan and magenta",
    "harsh top light, heavy contrast",
    "strobing coloured light",
]

# Camera moves, again ordered by how much they move.
MOVES = [
    "locked off, no camera movement",
    "very slow push in",
    "slow drift to the left",
    "slow pull back",
    "steady dolly forward",
    "handheld sway",
    "fast push in",
    "whip pan",
]


def _pick(options: Sequence[str], seed: int, salt: str, lo: float, hi: float) -> str:
    """Deterministic choice from the slice of ``options`` between lo and hi (0..1)."""
    if not options:
        return ""
    a = max(0, min(len(options) - 1, int(lo * len(options))))
    b = max(a + 1, min(len(options), int(hi * len(options)) + 1))
    window = options[a:b]
    h = hashlib.sha256(f"{seed}:{salt}".encode()).digest()
    return window[int.from_bytes(h[:4], "big") % len(window)]


def _parse_vocab(text: str, fallback: list[str]) -> list[str]:
    """One option per line. Blank input means use the built-in list."""
    if not text or not text.strip():
        return list(fallback)
    items = [ln.strip().rstrip(",") for ln in text.splitlines()]
    return [i for i in items if i and not i.startswith("#")] or list(fallback)


def build_prompt_book(
    plan: ShotPlan,
    analysis: AudioAnalysis,
    master_prompt: str,
    look: str = "",
    framings: str = "",
    lightings: str = "",
    moves: str = "",
    energy_influence: float = 0.7,
    seed: int = 0,
) -> ShotPlan:
    """Fill in ``prompt`` and ``motion`` on every shot. Mutates and returns plan.

    ``energy_influence`` is how strongly a shot's energy narrows the vocabulary
    window. At 0 every shot draws from the whole list and the edit has no arc; at
    1 the quiet parts can only be wide and static.
    """
    fr = _parse_vocab(framings, FRAMINGS)
    li = _parse_vocab(lightings, LIGHTING)
    mv = _parse_vocab(moves, MOVES)

    master = master_prompt.strip().rstrip(",")
    look = look.strip().rstrip(",")

    for shot in plan.shots:
        e = max(0.0, min(1.0, shot.energy))
        # Narrow the window around the shot's energy. Width shrinks as influence
        # rises, but never to nothing, or every shot in a section looks identical.
        half = max(0.15, 0.5 * (1.0 - energy_influence))
        lo, hi = max(0.0, e - half), min(1.0, e + half)

        framing = _pick(fr, shot.seed, "framing", lo, hi)
        lighting = _pick(li, shot.seed, "lighting", lo, hi)
        move = _pick(mv, shot.seed, "move", lo, hi)

        parts = [p for p in (framing, master, lighting, look) if p]
        shot.prompt = ", ".join(parts)

        # The motion prompt restates the subject because the start frame does not
        # argue for anything past frame one.
        shot.motion = f"{move}. {master}. The subject and the setting stay the same throughout."

    return plan


def prompt_book_text(plan: ShotPlan) -> str:
    """Human-readable dump, for pasting into a note or eyeballing before a run."""
    lines = []
    for s in plan.shots:
        lines.append(
            f"[{s.index:03d}] {s.start_time:7.2f}s  {s.frame_count:4d}f  "
            f"{s.bars:.0f} bar  energy {s.energy:.2f}  seed {s.seed}"
        )
        lines.append(f"      still : {s.prompt}")
        lines.append(f"      motion: {s.motion}")
    return "\n".join(lines)
