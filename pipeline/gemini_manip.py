"""Gemini 2.5 Flash — manipulable-object detector.

Programmatic alternative to Simify's GPT-4o filter. Given an RGB image,
returns labeled bounding boxes for every physical object a robot arm could
pick up. Used to prompt SAM-2 so we only segment+reconstruct real objects
(no table fragments, no shadows).

Requires env var:   GEMINI_API_KEY

Usage:
    from gemini_manip import get_manipulable_boxes
    boxes = get_manipulable_boxes("images/raw_images/IMG_3199_fixed.jpeg")
    # [{"label": "pink plate", "box_px": (x0, y0, x1, y1)}, ...]
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from PIL import Image


GEMINI_MODEL = "gemini-2.5-flash"

_PROMPT = """\
This is a close-up, near-overhead view of a robot's tabletop workspace. \
A RealSense camera sits ~15 cm above the table surface, pointing down. \
The table occupies most of the frame.

Identify ONLY objects that are ALL of:
  (a) resting on or directly above the visible table surface,
  (b) in the central/foreground of the image (not near the image border \
      or behind other objects in the far distance),
  (c) small enough to be picked up and manipulated by a robot arm \
      (typical sizes: 2–30 cm).

Return EACH distinct, whole object a robot arm would deliberately pick up and
relocate. Do NOT restrict to a fixed list of categories — identify whatever
graspable objects are actually present (these are only examples: cups, plates,
bowls, fruit, bottles, tools, phones, toys, markers, small boxes, containers).

Exclude anything that is NOT a deliberately-manipulable tabletop object:
- The table surface itself, the tablecloth, the floor.
- Walls, cabinets, doors, furniture beyond the table.
- Other tables, chairs, monitors, or rooms visible in the background.
- Hands, arms, or body parts.
- Shadows, reflections, cables, power strips.
- Large pieces of equipment (tripods, lamps, robot bases).
- Tiny debris, crumbs, seeds, pits, stains, marks, or fragments too small or
  insignificant to grasp on purpose (only count a small item if it is a clear,
  whole object like a single fruit or bottle cap).

If nothing qualifies, return an empty array [].

For EACH qualifying object, return a TIGHT bounding box that closely \
fits only that object. Use Gemini's standard normalized format: \
[y_min, x_min, y_max, x_max] with coordinates in 0..1000 (1000 = image \
height for y, image width for x).

If an object sits ON or INSIDE another (a lemon on a plate), return \
BOTH as separate entries.

Return ONLY a JSON array, no prose, no code fences. Example:
[
  {"label": "pink plate", "box_2d": [450, 120, 820, 560]},
  {"label": "lemon", "box_2d": [520, 280, 640, 410]}
]
"""


def _parse_json_response(text: str) -> list[dict[str, Any]]:
    """Extract JSON array from model response, tolerating code fences."""
    # Strip ```json ... ``` or ``` ... ``` wrappers if present
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    text = text.strip()
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"expected JSON array, got {type(data).__name__}")
    return data


def _targeted_prompt(targets: list) -> str:
    items = ", ".join(targets)
    return f"""\
This is a close-up, near-overhead view of a robot's tabletop workspace.

Find ONLY these specific objects of interest, if present: {items}.

Rules:
- For each requested object type, return the SINGLE BEST instance: the most
  prominent, most fully-visible, most central one. Do not return duplicates.
- Use the requested object name as the "label" (e.g. "cup", "bowl", "plate",
  "lemon"), even if a more specific name exists.
- IGNORE everything else: other objects, debris, crumbs, pits/seeds, cables,
  the table, hands, background, monitors, furniture.
- If a requested object type is not clearly present, skip it (do not guess).
- Return a TIGHT bounding box fitting only that object.

Coordinates: Gemini normalized format [y_min, x_min, y_max, x_max] in 0..1000
(1000 = image height for y, image width for x).

Return ONLY a JSON array, no prose. Example:
[
  {{"label": "plate", "box_2d": [450, 120, 820, 560]}},
  {{"label": "lemon", "box_2d": [520, 280, 640, 410]}}
]
"""


def get_manipulable_boxes(image_path: str, debug: bool = False,
                          targets: list = None) -> list[dict[str, Any]]:
    """Ask Gemini for manipulable objects in the image.

    If `targets` (or the MANIP_TARGETS env var, comma-separated) is given, only
    those specific object types are detected (single best instance each) and
    everything else is ignored. Otherwise the general manipulable-object prompt
    is used.

    Returns a list of dicts, each with:
        label:    short object name (str)
        box_px:   (x0, y0, x1, y1) in pixel coordinates of the FULL image
        box_norm: (y0, x0, y1, x1) raw Gemini 0..1000 normalized box (for debug)
    """
    if targets is None:
        _env = os.environ.get("MANIP_TARGETS", "").strip()
        targets = [t.strip() for t in _env.split(",") if t.strip()] if _env else None
    prompt = _targeted_prompt(targets) if targets else _PROMPT
    if targets:
        print(f"[gemini_manip] TARGETED detection for: {targets}")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Export it before running:\n"
            "    export GEMINI_API_KEY='your-key-here'"
        )

    from google import genai
    from google.genai import types

    # Load image and read raw bytes (Gemini accepts inline bytes)
    img = Image.open(image_path)
    w, h = img.size

    # Convert to JPEG bytes for the API call (Gemini handles JPEG/PNG)
    import io

    buf = io.BytesIO()
    rgb = img.convert("RGB")
    rgb.save(buf, format="JPEG", quality=90)
    img_bytes = buf.getvalue()

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema={
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "label": {"type": "STRING"},
                        "box_2d": {
                            "type": "ARRAY",
                            "items": {"type": "INTEGER"},
                        },
                    },
                    "required": ["label", "box_2d"],
                },
            },
        ),
    )

    raw = resp.text or ""
    if debug:
        print(f"[gemini_manip] raw response:\n{raw}\n---")

    entries = _parse_json_response(raw)

    results: list[dict[str, Any]] = []
    for e in entries:
        box = e.get("box_2d") or e.get("bbox") or e.get("box")
        if not box or len(box) < 4:
            continue
        # Gemini sometimes returns >4 values — take only the first 4
        box = box[:4]
        label = str(e.get("label") or e.get("name") or "object")
        # Gemini returns [y_min, x_min, y_max, x_max] in 0..1000
        y0n, x0n, y1n, x1n = [float(v) for v in box]
        x0 = int(round(x0n / 1000.0 * w))
        y0 = int(round(y0n / 1000.0 * h))
        x1 = int(round(x1n / 1000.0 * w))
        y1 = int(round(y1n / 1000.0 * h))
        x0, x1 = sorted((max(0, x0), min(w, x1)))
        y0, y1 = sorted((max(0, y0), min(h, y1)))
        if x1 <= x0 or y1 <= y0:
            continue
        results.append({
            "label": label,
            "box_px": (x0, y0, x1, y1),
            "box_norm": (y0n, x0n, y1n, x1n),
        })

    return results


if __name__ == "__main__":
    # Quick smoke test: python gemini_manip.py <image_path>
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gemini_manip.py <image_path>")
        sys.exit(1)
    boxes = get_manipulable_boxes(sys.argv[1], debug=True)
    print(f"\nFound {len(boxes)} manipulable objects:")
    for b in boxes:
        x0, y0, x1, y1 = b["box_px"]
        print(f"  {b['label']:20s}  px=({x0},{y0})-({x1},{y1})")
