"""Append a run record. Usage:
    runlog.py start RUN_DIR --agent codex --model gpt-6-astra --packet p1_real2sim
    runlog.py end   RUN_DIR [--tokens N] [--cost USD] [--interventions N] [--note "..."]
Writes RUN_DIR/run.json (one record per run; `end` fills in wall_s)."""
import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["start", "end"])
    ap.add_argument("run_dir")
    ap.add_argument("--agent")
    ap.add_argument("--model")
    ap.add_argument("--packet")
    ap.add_argument("--tokens", type=int)
    ap.add_argument("--cost", type=float)
    ap.add_argument("--interventions", type=int, default=0)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    f = os.path.join(a.run_dir, "run.json")
    rec = json.load(open(f)) if os.path.exists(f) else {}
    if a.what == "start":
        rec.update(agent=a.agent, model=a.model, packet=a.packet, t_start=time.time(),
                   start=time.strftime("%Y-%m-%d %H:%M:%S"), interventions=[], tokens=None, cost_usd=None, note="")
    else:
        rec.update(t_end=time.time(), end=time.strftime("%Y-%m-%d %H:%M:%S"), wall_s=round(time.time() - rec.get("t_start", time.time())),
                   tokens=a.tokens, cost_usd=a.cost, n_interventions=a.interventions, note=a.note)
    os.makedirs(a.run_dir, exist_ok=True)
    json.dump(rec, open(f, "w"), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
