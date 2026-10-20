"""Write inputs/robot.json and inputs/actions.json: every motion problem our layer poses for the sled
plan (each action, each hand/jaw/flip variant it tries), as exported by SledMotion.run (MOTION_EXPORT).

    SLED_BOX_Y=36 SLED_BASE_LIFT_CM=14 export_inputs.py [--scene runs_2026-10/sled/scene.json]

Runs the cuRobo layer once (about 3 min) so the worlds and segments are exactly what it checked."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "twin", "newton"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=os.path.join(HERE, "..", "..", "runs_2026-10", "sled", "scene.json"))
    a = ap.parse_args()
    runs = os.path.join(HERE, "..", "runs", "p2_motion")                 # our results and trajectories: never in inputs/
    os.makedirs(runs, exist_ok=True)
    exp = os.path.join(runs, "export.jsonl")
    os.environ["MOTION_EXPORT"] = exp
    if os.path.exists(exp):
        os.remove(exp)
    from tamp_sym import Domain, S
    from motion import SledMotion
    sc = json.load(open(a.scene))
    D = Domain(sc)
    st = S(on_towel=frozenset(D.objs), in_box=frozenset(), toppled=frozenset(), folds=0, towel_in_box=False, towel_dy=0.0,
           pos={n: (D.cx + float(o["on_towel_xy"][0]), D.cy + float(o["on_towel_xy"][1])) for n, o in D.objs.items()}, pulls=0)
    M = SledMotion(D)
    # the baseline plan's actions, in order: pack the bottle, pull, pack the rest, fold twice, pack the towel
    for h in ("right", "left"):
        M.pick_place(st, "bottle", h)
    st = st.copy(on_towel=st.on_towel - {"bottle"}, in_box=frozenset({"bottle"}), pos={n: p for n, p in st.pos.items() if n != "bottle"})
    M.pull(st, 34.0)
    st = st.copy(towel_dy=33.3, pos={n: (p[0], p[1] + 33.3) for n, p in st.pos.items()}, pulls=1)
    for o in ("block1", "block2", "cup"):
        for h in ("left", "right"):
            M.pick_place(st, o, h)
        st = st.copy(on_towel=st.on_towel - {o}, in_box=st.in_box | {o}, pos={n: p for n, p in st.pos.items() if n != o})
    M.fold(st, 1)
    st = st.copy(folds=1)
    M.fold(st, 2)
    st = st.copy(folds=2)
    M.pack_towel(st, 2)
    m = M.m
    robot = dict(urdf="dual_ur5e.urdf", joints=list(m.jn), home=[float(v) for v in m.home_q], limits=m.limits,
                 bases={h: [float(v) for v in b] for h, b in m.bases.items()},
                 tcp_links={"right": "tcp", "left": "tcp_1"},
                 tool_down_quat_wxyz={h: [float(v) for v in m.pose(h, [0, 0, 0], 0.0)[3:]] for h in ("right", "left")}, units="m, rad")
    M.m.close()
    probs, seen, ours = [], {}, {}
    for i, line in enumerate(open(exp)):
        r = json.loads(line)
        head = r["label"].split("(")[0].strip()                     # "pick_place(bottle, right arm, ...)" -> pick_place:bottle; "fold 1" -> fold 1
        aid = head + (":" + r["label"].split("(")[1].split(",")[0].split(")")[0] if "(" in r["label"] else "")
        seen.setdefault(aid, 0)
        segs = []
        for sg in r["segs"]:
            if sg[0] == "world":
                segs.append(dict(kind="world", world=sg[1]))
            else:
                d = dict(kind=sg[0], goals=sg[1], attach=sg[2] or {}, off=sg[3], contact=bool(sg[4]) if len(sg) > 4 else False)
                if len(sg) > 5:
                    d["pair_tol"] = sg[5]
                segs.append(d)
        probs.append(dict(id=aid, variant=r["label"], problem=i, world=r["world"], segments=segs))
        ours[str(i)] = dict(id=aid, variant=r["label"], ok=r["result"]["ok"], why=r["result"]["why"], trajs=r["trajs"])
    json.dump(robot, open(os.path.join(HERE, "inputs", "robot.json"), "w"), indent=1)
    json.dump(probs, open(os.path.join(HERE, "inputs", "actions.json"), "w"))
    json.dump(ours, open(os.path.join(runs, "ours.json"), "w"))
    ids = sorted({p["id"] for p in probs})
    print(f"[export] {len(probs)} problems for {len(ids)} actions: {ids}")


if __name__ == "__main__":
    main()
