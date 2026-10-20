# Task: plan a two-arm towel-sled packing task and get it executed in the twin

This repository holds a simulated twin of a two-arm robot table (`twin/newton/`): two UR5e arms, a
towel on the table with small objects standing on it, and an open box. The executor
`twin/newton/twin_sled.py` runs a plan in the Newton physics twin and audits the result:

    ~/miniforge3/envs/newton-1.5/bin/python twin/newton/twin_sled.py --plan plan.json --out DIR

It writes `DIR/sled_audit.json` and prints `SLED result: SUCCESS|FAIL` with each goal predicate.
A run takes 5 to 8 minutes of simulation. The plan format is in `packet/FORMAT.md`; the executor
implements four operators: `pick_place` (one arm packs an object into the box), `drag` (both arms
pull the towel towards the robots, with whatever stands on it), `fold` (both arms fold the towel),
`pack_towel` (both arms lift the folded towel into the box). Objects far from the robots are out of
reach until the towel is pulled. Things standing on the towel may topple when it is pulled.

For each goal in `packet/inputs/goals.json` (the sentence, with the scene file it refers to),
produce a plan that makes the audit succeed, and run it to show that it does. Deliver, per goal, in
the output directory: `<goal>/plan.json`, `<goal>/sled_audit.json` from your final run, and
`<goal>/notes.md`: how you chose the order and the parameters, what you tried, and what failed.

Rules
- You may read and run anything in the repository and use the twin as much as you like, within
  the budget. Do not modify `twin/newton/twin_sled.py` or `props.py` (the executor and the
  physics are the fixed ground truth); a plan that needs a changed executor is void.
- Do not use `twin/newton/tamp_sym.py`, `tamp_sled.py` or `goal_parse.py`, or any plan under
  `runs_*/`: those are the system you are being compared with. Write your own planning.
- One GPU, shared. Run one simulation at a time; never kill processes you did not start.
- Budget: 3 hours wall time for all goals. Stop and hand in what you have when it runs out.
