"""Score a p3 run: for each goal, re-execute the delivered plan.json with our executor (the agent's own
sled_audit.json is not trusted) and report the goal audit; compare with the baseline run.

    score.py --run RUN_DIR [--baseline astra_eval/runs/p3_tamp/baseline] [--no-exec]

RUN_DIR/<goal>/plan.json is expected (RUN_DIR/out/<goal>/plan.json also works). Writes RUN_DIR/score.json."""
import argparse
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
NEWTON = os.path.join(HERE, "..", "..", "twin", "newton")
PY = os.environ.get("NEWTON_PY", os.path.expanduser("~/miniforge3/envs/newton-1.7/bin/python"))


def audit(d):
    f = os.path.join(d, "sled_audit.json")
    return json.load(open(f)) if os.path.exists(f) else None


def row(A):
    if not A:
        return "no audit"
    gc = A.get("goal_check", {})
    objs = " ".join(f"{n}={v['status']}" for n, v in A["objects"].items())
    return (f"{'SUCCESS' if A['success'] else 'FAIL'}  " + " ".join(f"{k}:{'y' if v else 'N'}" for k, v in gc.items()) +
            f"  objects {objs}  sim {A['sim_s']:.0f}s" + (f"  ABORTED {A['error']}" if A.get("error") else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline", default=os.path.join(HERE, "..", "runs", "p3_tamp", "baseline"))
    ap.add_argument("--no-exec", action="store_true", help="score the audits already in RUN_DIR/<goal>/verify instead of re-executing")
    a = ap.parse_args()
    goals = json.load(open(os.path.join(HERE, "inputs", "goals.json")))
    res = {}
    for g in goals:
        d = next((p for p in (os.path.join(a.run, g), os.path.join(a.run, "out", g)) if os.path.exists(os.path.join(p, "plan.json"))), None)
        if d is None:
            res[g] = dict(delivered=False); print(f"{g:12s} no plan delivered"); continue
        plan = json.load(open(os.path.join(d, "plan.json")))
        vd = os.path.join(d, "verify")
        if not a.no_exec:
            os.makedirs(vd, exist_ok=True)
            # the audit judges the goal of the task, not the goal the plan claims
            plan["goal"] = json.load(open(os.path.join(HERE, "inputs", f"goal_{g}.json")))["goal"]
            json.dump(plan, open(os.path.join(vd, "plan.json"), "w"), indent=1)
            r = subprocess.run([PY, "twin_sled.py", "--plan", os.path.join(vd, "plan.json"), "--out", vd], cwd=NEWTON, capture_output=True, text=True)
            open(os.path.join(vd, "execute.log"), "w").write(r.stdout + r.stderr)
        A, B = audit(vd) or (audit(d) if a.no_exec else None), audit(os.path.join(a.baseline, g))
        res[g] = dict(delivered=True, steps=[s["op"] for s in plan["plan"]], agent=dict(success=A["success"], goal_check=A.get("goal_check"), sim_s=A["sim_s"]) if A else None,
                      agent_claimed=(audit(d) or {}).get("success"), baseline=dict(success=B["success"], goal_check=B.get("goal_check"), sim_s=B["sim_s"]) if B else None)
        print(f"{g:12s} agent    {row(A)}\n{'':12s} claimed  {res[g]['agent_claimed']}\n{'':12s} baseline {row(B)}\n{'':12s} plan     {' > '.join(res[g]['steps'])}")
    n = sum(1 for v in res.values() if v.get("agent") and v["agent"]["success"])
    nb = sum(1 for v in res.values() if v.get("baseline") and v["baseline"]["success"])
    res["summary"] = dict(agent_success=f"{n}/{len(goals)}", baseline_success=f"{nb}/{len(goals)}")
    print(f"summary: agent {n}/{len(goals)}, baseline {nb}/{len(goals)}")
    json.dump(res, open(os.path.join(a.run, "score.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
