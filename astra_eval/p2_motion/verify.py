"""Verify delivered joint trajectories with our cuRobo collision checker (motion_server.py `check`).

    SLED_BOX_Y=36 SLED_BASE_LIFT_CM=14 verify.py --traj OUT/trajectories.json [--actions inputs/actions.json]

Per problem and segment: continuity with the previous segment, joint limits, max joint step, collisions
(arm links and carried objects vs obstacles not in `off`, arm vs arm; the gripper may touch the world
only when contact is true), and goal reaching (path: every pose in order; plan: the last). A problem
passes when every segment passes. Reports per action id whether any variant passed, and compares with
our layer's result on the same problems. Writes score.json next to the trajectories."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "twin", "newton"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--actions", default=os.path.join(HERE, "inputs", "actions.json"))
    ap.add_argument("--ours", default=os.path.join(HERE, "..", "runs", "p2_motion", "ours.json"), help="our layer's results on the same problems")
    a = ap.parse_args()
    from motion import Motion
    from twin_sled import BASES_XY
    probs = {str(p["problem"]): p for p in json.load(open(a.actions))}
    ours = json.load(open(a.ours)) if os.path.exists(a.ours) else {}
    for pid, p in probs.items():
        p["our_result"] = ours.get(pid, {})
    home = json.load(open(os.path.join(os.path.dirname(a.actions), "robot.json")))["home"]   # the home the agent was given (the
    T = json.load(open(a.traj))                                                               # server's own park IK is not deterministic)
    m = Motion(BASES_XY, verbose=False, world=probs[next(iter(probs))]["world"])
    res, by_action = {}, {}
    for pid, p in probs.items():
        by_action.setdefault(p["id"], dict(ours=False, agent=False, variants=[]))
        by_action[p["id"]]["ours"] |= bool((p.get("our_result") or {}).get("ok"))
        if pid not in T:
            continue
        segs = [s for s in p["segments"]]
        trajs = T[pid]["segments"]
        nonw = [s for s in segs if s["kind"] != "world"]
        out = dict(ok=True, why="", segments=[])
        if len(trajs) != len(nonw):
            out = dict(ok=False, why=f"{len(trajs)} trajectories for {len(nonw)} segments", segments=[])
        else:
            m.call(cmd="world", cuboids=p["world"])
            q, k, held = list(home), 0, {}
            cur = [dict(c) for c in p["world"]]
            for s in segs:
                if s["kind"] == "world":
                    cur = [dict(c) for c in s["world"]]; m.call(cmd="world", cuboids=cur); continue
                tr = trajs[k]; k += 1
                if s.get("attach"):
                    # the carried object rides with the hand, as in SledMotion.run
                    tcp = m.call(cmd="fk", q=q)["tcp"]
                    moved = False
                    for hand, names in s["attach"].items():
                        for c in cur:
                            if c["name"] in names:
                                if c["name"] not in held:
                                    held[c["name"]] = [c["pose"][i] - tcp[hand][i] for i in range(3)]
                                else:
                                    c["pose"] = [tcp[hand][i] + held[c["name"]][i] for i in range(3)] + c["pose"][3:]; moved = True
                    if moved:
                        m.call(cmd="world", cuboids=cur)
                r = m.call(cmd="check", q0=q, traj=tr, kind=s["kind"], goals=s["goals"], attach=s.get("attach") or {}, off=s.get("off", []),
                           gripper_contact=bool(s.get("contact")), pair_tol=s.get("pair_tol"))
                out["segments"].append(r)
                if not r["ok"]:
                    out.update(ok=False, why=f"segment {k - 1} ({s['kind']}): {r['why']}"); break
                q = list(tr[-1])
        res[pid] = dict(id=p["id"], variant=p["variant"], **out)
        by_action[p["id"]]["variants"].append(dict(problem=pid, ok=out["ok"], why=out["why"]))
        by_action[p["id"]]["agent"] |= out["ok"]
        print(f"[verify] {pid:3s} {p['variant']:60s} {'OK' if out['ok'] else 'FAIL: ' + out['why']}")
    m.close()
    print("\naction                      ours   agent")
    for aid, v in by_action.items():
        print(f"{aid:28s}{'OK' if v['ours'] else '-':7s}{'OK' if v['agent'] else '-'}")
    summ = dict(actions=len(by_action), ours_ok=sum(v["ours"] for v in by_action.values()), agent_ok=sum(v["agent"] for v in by_action.values()),
                problems_delivered=len([p for p in probs if p in T]), problems_ok=sum(r["ok"] for r in res.values()))
    print(f"summary: {summ}")
    json.dump(dict(problems=res, actions=by_action, summary=summ), open(os.path.join(os.path.dirname(os.path.abspath(a.traj)), "score.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
