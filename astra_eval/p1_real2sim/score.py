"""Score a scene.json against the tape measurements (gt.json).

    score.py --scene OUT/scene.json --gt inputs/gt.json [--downstream OUT/down]

Per object (matched by name or alias, else by nearest position): detected, shape right, size error
(max over the three extents, cm), position error (cm, relative to the towel centre). Towel: size and
centre error. Box: inner size, height, position. Mass is reported, not scored (not visible).
--downstream: plan with tamp_sym.py and execute with twin_sled.py from this scene (about 8 min) and
report the goal audit. Writes score.json next to the scene."""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NEWTON = os.path.join(HERE, "..", "..", "twin", "newton")
PY = os.environ.get("NEWTON_PY", os.path.expanduser("~/miniforge3/envs/newton-1.5/bin/python"))
SYN = {"mug": "cup", "cup": "cup", "bottle": "bottle", "block": "box", "cube": "box"}


def match(gt_objs, objs):
    """gt name -> scene object (by name/alias, then nearest unmatched position)."""
    used, out = set(), {}
    for g in gt_objs:
        names = {g["name"].lower()} | {a.lower() for a in g.get("aliases", [])}
        for i, o in enumerate(objs):
            if i not in used and o.get("name", "").lower() in names:
                out[g["name"]], _ = o, used.add(i)
                break
    for g in gt_objs:
        if g["name"] in out:
            continue
        best = None
        for i, o in enumerate(objs):
            if i in used or "on_towel_xy" not in o:
                continue
            d = float(np.linalg.norm(np.array(o["on_towel_xy"], float) - np.array(g["on_towel_xy"], float)))
            if d < 6.0 and (best is None or d < best[0]):
                best = (d, i, o)
        if best:
            out[g["name"]], _ = best[2], used.add(best[1])
    return out, [o for i, o in enumerate(objs) if i not in used]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--downstream", help="dir: also plan + execute from this scene and report the audit")
    a = ap.parse_args()
    sc, gt = json.load(open(a.scene)), json.load(open(a.gt))
    m, extra = match(gt["objects"], sc.get("objects", []))
    rows, res = [], dict(objects={}, extra_objects=[o.get("name") for o in extra])
    for g in gt["objects"]:
        o = m.get(g["name"])
        if not o:
            res["objects"][g["name"]] = dict(detected=False)
            rows.append(f"  {g['name']:8s} MISSING")
            continue
        gd, od = np.array(g["dims_cm"], float), np.array(o.get("dims_cm", [0, 0, 0]), float)
        # a cuboid may be reported with x and y swapped if it is square-ish; take the better of the two
        size_err = min(float(np.abs(od - gd).max()), float(np.abs(od[[1, 0, 2]] - gd).max()))
        pos_err = float(np.linalg.norm(np.array(o.get("on_towel_xy", [1e3, 1e3]), float) - np.array(g["on_towel_xy"], float)))
        shape_ok = SYN.get(o.get("shape", ""), o.get("shape", "")) == g["shape"]
        r = dict(detected=True, name_given=o.get("name"), shape_ok=shape_ok, size_err_cm=round(size_err, 2), pos_err_cm=round(pos_err, 2),
                 mass_given=o.get("mass_kg"), mass_gt=g.get("mass_kg"))
        if g["shape"] == "cup":
            r["handle_err_cm"] = round(abs(float(o.get("handle_cm", 0)) - float(g.get("handle_cm", 0))), 2)
        res["objects"][g["name"]] = r
        rows.append(f"  {g['name']:8s} as '{o.get('name')}' shape {'ok' if shape_ok else 'WRONG'}  size err {size_err:4.1f} cm  pos err {pos_err:4.1f} cm  mass {o.get('mass_kg')} (gt {g.get('mass_kg')})")
    tw, gtw = sc.get("towel", {}), gt["towel"]
    res["towel"] = dict(size_err_cm=round(float(np.abs(np.array(tw.get("size_cm", [0, 0]), float) - np.array(gtw["size_cm"], float)).max()), 2),
                        centre_err_cm=round(float(np.linalg.norm(np.array(tw.get("centre_xy", [1e3, 1e3]), float) - np.array(gtw["centre_xy"], float))), 2))
    bx, gbx = sc.get("box", {}), gt["box"]
    res["box"] = dict(inner_err_cm=round(float(np.abs(np.array(bx.get("inner_cm", [0, 0]), float) - np.array(gbx["inner_cm"], float)).max()), 2),
                      h_err_cm=round(abs(float(bx.get("h_cm", 0)) - float(gbx["h_cm"])), 2),
                      xy_err_cm=round(float(np.linalg.norm(np.array(bx.get("xy", [1e3, 1e3]), float) - np.array(gbx["xy"], float))), 2))
    det = [r for r in res["objects"].values() if r.get("detected")]
    res["summary"] = dict(detected=f"{len(det)}/{len(gt['objects'])}", shapes_ok=sum(r["shape_ok"] for r in det),
                          size_err_max_cm=max([r["size_err_cm"] for r in det] + [0]), pos_err_max_cm=max([r["pos_err_cm"] for r in det] + [0]),
                          within_1cm=sum(r["size_err_cm"] <= 1 and r["pos_err_cm"] <= 1 for r in det),
                          within_2cm=sum(r["size_err_cm"] <= 2 and r["pos_err_cm"] <= 2 for r in det))
    print("objects:\n" + "\n".join(rows))
    print(f"towel: size err {res['towel']['size_err_cm']} cm, centre err {res['towel']['centre_err_cm']} cm; box: inner err {res['box']['inner_err_cm']} cm, "
          f"h err {res['box']['h_err_cm']} cm, position err {res['box']['xy_err_cm']} cm; extra objects: {res['extra_objects']}")
    print(f"summary: {res['summary']}")
    if a.downstream:
        os.makedirs(a.downstream, exist_ok=True)
        plan = os.path.join(a.downstream, "plan.json")
        r1 = subprocess.run([PY, "tamp_sym.py", "--scene", os.path.abspath(a.scene), "--out", plan], cwd=NEWTON, capture_output=True, text=True)
        open(os.path.join(a.downstream, "plan.log"), "w").write(r1.stdout + r1.stderr)
        res["downstream"] = dict(planned=r1.returncode == 0)
        if r1.returncode == 0:
            r2 = subprocess.run([PY, "twin_sled.py", "--plan", plan, "--out", a.downstream], cwd=NEWTON, capture_output=True, text=True)
            open(os.path.join(a.downstream, "execute.log"), "w").write(r2.stdout + r2.stderr)
            au = os.path.join(a.downstream, "sled_audit.json")
            if os.path.exists(au):
                A = json.load(open(au))
                res["downstream"].update(success=A["success"], goal_check=A.get("goal_check"), objects={n: v["status"] for n, v in A["objects"].items()})
        print(f"downstream: {res['downstream']}")
    out = os.path.join(os.path.dirname(os.path.abspath(a.scene)), "score.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
