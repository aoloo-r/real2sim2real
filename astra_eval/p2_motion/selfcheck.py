"""Self-test of the verifier: our own layer's trajectories (runs/p2_motion/ours.json, written by
export_inputs.py) must pass verify.py. Writes runs/p2_motion/ours_trajectories.json in the delivered
format; then: verify.py --traj runs/p2_motion/ours_trajectories.json"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs", "p2_motion")
ours = json.load(open(os.path.join(RUNS, "ours.json")))
out = {k: dict(segments=v["trajs"]) for k, v in ours.items() if v["ok"]}
json.dump(out, open(os.path.join(RUNS, "ours_trajectories.json"), "w"))
print(f"[selfcheck] {len(out)} solved problems written to runs/p2_motion/ours_trajectories.json")
