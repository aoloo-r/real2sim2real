"""TAMP — the TASK-planning layer of the real2sim2real digital twin.

Turns a natural-language instruction + the reconstructed scene into a GROUNDED
symbolic goal and an ORDERED pick-and-place action plan over the ACTUAL detected
objects. This is the task half of TiPToP-style open-vocabulary TAMP (Gemini does
the grounding); the motion half (grasp sampling + cuRobo + the sim twin) consumes
the emitted actions and turns each into an EE trajectory to validate in parallel.

It does NOT move the robot — it produces `task_plan.json`:
    {
      "instruction": "...",
      "frame": "ur5e_base_link",
      "objects": [{id,label,size_m,pos_base}, ...],   # scene the plan is over
      "goal_predicates": [{relation,object,target}, ...],
      "actions": [{type:"pick_place", object, object_label,
                   target, target_label, relation, reason}, ...],
      "note": "..."
    }

Usage:
  GEMINI_API_KEY=... python tamp_plan.py \
    --scene_dir   outputs/robot_20260604_174345 \
    --capture_dir captures/robot_20260604_174345 \
    --instruction "put the blue cup in the pink plate" \
    --out /tmp/task_plan.json
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

GEMINI_MODEL = "gemini-2.5-flash"


def load_scene(scene_dir: str, capture_dir: str,
               base_frame: str = "ur5e_base_link") -> list[dict]:
    """Read scene_layout.json + extrinsics → objects with base-frame positions."""
    layout = json.load(open(os.path.join(scene_dir, "scene_layout.json")))
    T = None
    ex_path = os.path.join(capture_dir, "extrinsics.json")
    if os.path.isfile(ex_path):
        ex = json.load(open(ex_path))
        tf = ex.get("transforms", {}).get(base_frame)
        if tf:
            T = np.asarray(tf["T_base_cam"], float)
    objs = []
    for o in layout.get("objects", []):
        di = o.get("depth_info", {}) or {}
        pc = di.get("position_cam") or o.get("icp_pose", {}).get("position_cam")
        pb = None
        if T is not None and pc is not None:
            p = T @ np.array([pc[0], pc[1], pc[2], 1.0])
            pb = [round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)]
        objs.append({
            "id": o["id"],
            "label": o["label"],
            "size_m": round(float(o.get("physical_size_m") or di.get("physical_size_m") or 0.0), 3),
            "pos_base": pb,
        })
    return objs


_GROUND_PROMPT = """You are the TASK PLANNER for a single-arm tabletop robot \
(one parallel-jaw gripper). You decompose a natural-language instruction into an \
ordered sequence of pick-and-place actions over the objects actually present.

OBJECTS PRESENT (id, label, size_m, position in robot base frame [x,y,z] meters):
{obj_list}

INSTRUCTION: "{instruction}"

Return ONLY a JSON object (no prose, no code fences) with these keys:
  "goal_predicates": list of the symbolic goal relations the instruction implies,
     each {{"relation": "in|on|beside", "object": <id>, "target": <id>}}.
  "actions": the ORDERED list of pick-and-place steps that achieve the goal,
     each {{"type": "pick_place", "object": <id>, "object_label": <label>,
            "target": <id or null>, "target_label": <label or null>,
            "relation": "in|on|beside", "reason": <short string>}}.
  "note": short string — explain any assumption, or why actions is empty.

RULES:
- Reference objects ONLY by their integer id from the list above.
- Resolve descriptive / open-vocabulary references ("the fruit", "a container",
  "something to drink from") to the best-matching object by label and size.
- Physical ordering matters: to put A in/on B, B must be clear and A must be free
  (not under or inside another object). If not, insert the clearing steps FIRST.
- One arm: exactly one object moved per action; always pick before place.
- "in" = place inside a container (bowl/plate/cup); "on" = place on top of;
  "beside" = place next to on the table.
- If the instruction is already satisfied or physically impossible, return an
  empty "actions" list and explain in "note".
"""


def _parse_json(text: str) -> dict:
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    return json.loads(text.strip())


def ground(instruction: str, objects: list[dict]) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    obj_list = "\n".join(
        f"  id={o['id']}  label='{o['label']}'  size_m={o['size_m']}  pos_base={o['pos_base']}"
        for o in objects)
    prompt = _GROUND_PROMPT.format(obj_list=obj_list, instruction=instruction)

    from google import genai
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=[prompt])
    plan = _parse_json(resp.text)
    return plan


def validate(plan: dict, objects: list[dict]) -> list[str]:
    """Sanity-check the plan against the scene; return list of problems."""
    ids = {o["id"] for o in objects}
    problems = []
    for i, a in enumerate(plan.get("actions", [])):
        if a.get("object") not in ids:
            problems.append(f"action {i}: object id {a.get('object')} not in scene")
        tgt = a.get("target")
        if tgt is not None and tgt not in ids:
            problems.append(f"action {i}: target id {tgt} not in scene")
        if a.get("object") is not None and a.get("object") == tgt:
            problems.append(f"action {i}: object and target are the same id")
    for g in plan.get("goal_predicates", []):
        if g.get("object") not in ids or g.get("target") not in ids:
            problems.append(f"goal predicate references unknown id: {g}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--base_frame", default="ur5e_base_link")
    ap.add_argument("--out", default="/tmp/task_plan.json")
    args = ap.parse_args()

    objects = load_scene(args.scene_dir, args.capture_dir, args.base_frame)
    print(f"[TAMP] scene: {len(objects)} objects")
    for o in objects:
        print(f"        id={o['id']} '{o['label']}' size={o['size_m']} pos_base={o['pos_base']}")
    print(f"[TAMP] instruction: \"{args.instruction}\"")

    plan = ground(args.instruction, objects)
    problems = validate(plan, objects)

    label_by_id = {o["id"]: o["label"] for o in objects}
    print("\n[TAMP] GROUNDED GOAL:")
    for g in plan.get("goal_predicates", []):
        print(f"        {label_by_id.get(g.get('object'),'?')} "
              f"{g.get('relation')} {label_by_id.get(g.get('target'),'?')}")
    print("\n[TAMP] ACTION PLAN:")
    for i, a in enumerate(plan.get("actions", []), 1):
        print(f"   {i}. pick '{a.get('object_label')}' (id {a.get('object')}) "
              f"-> {a.get('relation')} '{a.get('target_label')}' (id {a.get('target')})"
              f"   [{a.get('reason','')}]")
    if plan.get("note"):
        print(f"\n[TAMP] note: {plan['note']}")
    if problems:
        print("\n[TAMP] !! VALIDATION PROBLEMS:")
        for p in problems:
            print(f"        - {p}")

    out = {
        "instruction": args.instruction,
        "frame": args.base_frame,
        "objects": objects,
        "goal_predicates": plan.get("goal_predicates", []),
        "actions": plan.get("actions", []),
        "note": plan.get("note", ""),
        "validation_problems": problems,
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[TAMP] wrote {args.out}  ({len(out['actions'])} actions, "
          f"{len(problems)} problems)")


if __name__ == "__main__":
    main()
