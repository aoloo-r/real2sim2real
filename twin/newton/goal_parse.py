"""Natural-language goal -> grounded goal predicates for the sled planner (tamp_sym.py).

A VLM (Gemini) reads the instruction together with the scene's objects and writes the goal in a
fixed predicate vocabulary; this file then CHECKS that goal against the scene and the domain
before any planning happens. The model only translates; it does not decide feasibility.

Vocabulary (what tamp_sym.py can plan for and twin_sled.py can audit):
    in_box(o)          object o ends in the box, upright
    folded(towel, k)   the towel is folded at least k times (k = 1 or 2, far edge onto near edge)
    towel_in_box       the towel ends in the box
    before(a, b)       object a is packed before object b
Anything in the instruction that the vocabulary cannot express goes to `unsupported` (the model is
told to put it there rather than approximate it), and the run stops unless --allow-unsupported.

Checks after the parse:
    every name is a scene object; every predicate is in the vocabulary
    before(a, b) needs in_box(a) and in_box(b); the order has no cycle
    folding or packing the towel needs it clear, so objects left out of in_box would have to be
        packed anyway: reported as a side effect the instruction did not ask for (not an error)

Usage:
    goal_parse.py --scene scene.json --text "pack everything, then fold the towel and pack it" --out goal.json
    goal_parse.py --scene scene.json --predicates "in_box(cup); towel_in_box" --out goal.json   # no VLM: check only
Needs GEMINI_API_KEY and google-genai (the sam3d-objects env has it); tamp_sym.py itself does not.
"""
import argparse
import json
import os
import re
import sys

MODEL = os.environ.get("GOAL_MODEL", "gemini-2.5-flash")
MAX_FOLDS = 2

PROMPT = """You turn a person's instruction for a two-arm robot into a goal written in a fixed predicate language.

The scene: a towel lies on a table in front of the robot, an open box stands next to it, and these objects stand on the towel:
{objects}

Predicates you may use (nothing else):
  in_box(o)          object o ends up in the box
  folded(towel, k)   the towel is folded at least k times, k = 1 or 2 (each fold halves it, far edge onto near edge)
  towel_in_box       the towel ends up in the box
  before(a, b)       object a is put in the box before object b

Rules:
- Use only the object names listed above. Map each description in the instruction to a listed name ("the mug" -> the cup, "the blocks" -> every block, "everything" -> every object).
- Write only what the instruction asks for. Do not add goals it does not state, and do not add orderings it does not state.
- "fold the towel" with no number means folded(towel, 1); "fold it twice" or "fold it in quarters" means folded(towel, 2).
- If part of the instruction cannot be written with these predicates (a different place, a pose, a person, an action like "wipe" or "hand me"), copy that part word for word into "unsupported". Never approximate it with a predicate that means something else.
- If a description matches no listed object, or could mean more than one, say so in "notes" and leave it out of the predicates.

Instruction: "{text}"
"""

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "predicates": {"type": "ARRAY", "items": {"type": "STRING"}},
        "grounding": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "phrase": {"type": "STRING"}, "objects": {"type": "ARRAY", "items": {"type": "STRING"}}}, "required": ["phrase", "objects"]}},
        "unsupported": {"type": "ARRAY", "items": {"type": "STRING"}},
        "notes": {"type": "STRING"},
    },
    "required": ["predicates", "grounding", "unsupported", "notes"],
}

PAT = {
    "in_box": re.compile(r"^in_box\(\s*([A-Za-z0-9_]+)\s*\)$"),
    "folded": re.compile(r"^folded\(\s*towel\s*,\s*(\d+)\s*\)$"),
    "towel_in_box": re.compile(r"^towel_in_box(\(\s*\))?$"),
    "before": re.compile(r"^before\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)$"),
}


def describe(scene):
    lines = []
    for o in scene["objects"]:
        d = o["dims_cm"]
        lines.append(f"  - {o['name']}: {o.get('shape', 'box')}, {d[0]:g} x {d[1]:g} x {d[2]:g} cm, {o['mass_kg']:g} kg")
    return "\n".join(lines)


def ask_vlm(scene, text):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("[goal] GEMINI_API_KEY is not set")
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key)
    try:
        resp = client.models.generate_content(
            model=MODEL, contents=[PROMPT.format(objects=describe(scene), text=text)],
            config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json", response_schema=SCHEMA,
                                               thinking_config=types.ThinkingConfig(thinking_budget=0)))
    except Exception as e:                                         # no silent fallback: a goal is never guessed
        raise SystemExit(f"[goal] the VLM call failed ({MODEL}): {e}")
    return json.loads(resp.text)


def check(scene, preds):
    """Parse and check predicate strings. Returns (goal, errors, warnings); goal is what tamp_sym.py reads."""
    names = [o["name"] for o in scene["objects"]]
    goal, errors, warnings = dict(in_box=[], folds_min=0, towel_in_box=False, before=[]), [], []
    for p in preds:
        p = p.strip()
        if m := PAT["in_box"].match(p):
            o = m.group(1)
            if o == "towel":
                goal["towel_in_box"] = True
            elif o not in names:
                errors.append(f"{p}: no object '{o}' in the scene ({', '.join(names)})")
            elif o not in goal["in_box"]:
                goal["in_box"].append(o)
        elif m := PAT["folded"].match(p):
            k = int(m.group(1))
            if not 1 <= k <= MAX_FOLDS:
                errors.append(f"{p}: the domain folds the towel 1 or {MAX_FOLDS} times")
            goal["folds_min"] = max(goal["folds_min"], min(k, MAX_FOLDS))
        elif PAT["towel_in_box"].match(p):
            goal["towel_in_box"] = True
        elif m := PAT["before"].match(p):
            a, b = m.group(1), m.group(2)
            bad = [x for x in (a, b) if x not in names]
            if bad:
                errors.append(f"{p}: no object {bad} in the scene")
            elif a == b:
                errors.append(f"{p}: an object cannot come before itself")
            elif [a, b] not in goal["before"]:
                goal["before"].append([a, b])
        else:
            errors.append(f"{p}: not in the vocabulary (in_box, folded, towel_in_box, before)")
    for a, b in goal["before"]:
        for x in (a, b):
            if x not in goal["in_box"]:
                errors.append(f"before({a}, {b}) orders packing, but in_box({x}) is not in the goal")
    # the order must have no cycle
    succ = {}
    for a, b in goal["before"]:
        succ.setdefault(a, set()).add(b)
    state = {}

    def cyc(n):
        state[n] = 1
        for m_ in succ.get(n, ()):
            if state.get(m_) == 1 or (state.get(m_) is None and cyc(m_)):
                return True
        state[n] = 2
        return False
    if any(state.get(n) is None and cyc(n) for n in list(succ)):
        errors.append(f"the before() order has a cycle: {goal['before']}")
    if not goal["in_box"] and not goal["towel_in_box"] and not goal["folds_min"]:
        errors.append("the goal is empty: nothing to do")
    if goal["towel_in_box"] or goal["folds_min"]:
        left = [n for n in names if n not in goal["in_box"]]
        if left:
            warnings.append(f"the towel must be clear to fold or pack it, so {left} will be packed too (the instruction did not ask for it)")
    goal["in_box"].sort(key=names.index)
    return goal, errors, warnings


def predicates(goal):
    """The goal written back as predicates (for logs and the audit)."""
    out = [f"in_box({o})" for o in goal["in_box"]]
    if goal["folds_min"]:
        out.append(f"folded(towel, {goal['folds_min']})")
    if goal["towel_in_box"]:
        out.append("towel_in_box")
    out += [f"before({a}, {b})" for a, b in goal["before"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--text", help="the instruction, in words")
    ap.add_argument("--predicates", help="skip the VLM: ';'-separated predicates, checked the same way")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-unsupported", action="store_true", help="plan the expressible part even if some of the instruction is not")
    a = ap.parse_args()
    scene = json.load(open(a.scene))
    if bool(a.text) == bool(a.predicates):
        raise SystemExit("[goal] give exactly one of --text or --predicates")
    if a.text:
        raw = ask_vlm(scene, a.text)
        src = dict(source="vlm", model=MODEL, text=a.text, raw=raw)
        preds = raw["predicates"]
    else:
        preds = [p for p in a.predicates.split(";") if p.strip()]
        src = dict(source="given", text=None, raw=None)
    goal, errors, warnings = check(scene, preds)
    unsupported = (src["raw"] or {}).get("unsupported", [])
    out = dict(src, goal=goal, predicates=predicates(goal), errors=errors, warnings=warnings, unsupported=unsupported,
               ok=not errors and (not unsupported or a.allow_unsupported))
    json.dump(out, open(a.out, "w"), indent=1)
    if a.text:
        print(f"[goal] \"{a.text}\"")
        for g in src["raw"].get("grounding", []):
            print(f"[goal]   '{g['phrase']}' -> {g['objects']}")
        if src["raw"].get("notes"):
            print(f"[goal]   notes: {src['raw']['notes']}")
    print(f"[goal] predicates: {out['predicates']}")
    for w in warnings:
        print(f"[goal] warning: {w}")
    for u in unsupported:
        print(f"[goal] UNSUPPORTED: '{u}'")
    for e in errors:
        print(f"[goal] ERROR: {e}")
    print(f"[goal] {'OK' if out['ok'] else 'REJECTED'} -> {a.out}")
    sys.exit(0 if out["ok"] else 2)


if __name__ == "__main__":
    main()
