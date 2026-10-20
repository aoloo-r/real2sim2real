# Task: collision-free joint trajectories for two UR5e arms

`inputs/robot.json` and `inputs/dual_ur5e.urdf` define two UR5e arms with grippers mounted on a
table (12 joints: right arm first, then left; a fixed tcp frame at each gripper's pinch point).
`inputs/actions.json` lists the motion problems of a packing task: for each, the obstacles (axis-
aligned cuboids: table, box floor and walls, objects, the towel as a slab) and a sequence of
segments the hands must perform. A `plan` segment asks for any collision-free motion from the
current configuration to the given tcp pose(s); a `path` segment asks the tcp(s) to pass through
the listed poses in order (a straight descent, lift, pull or arc); a `world` segment changes the
obstacles (an object was put down). `attach` names objects carried by a hand from that segment on:
they move with the tcp and must not collide either. `off` names obstacles the segment may touch
(the object being grasped, the towel it stands on). `contact: true` lets the gripper itself touch
the world (pinching cloth on the table); arm links never may.

Several problems are variants of one action (a different arm, jaw direction or wrist turn). Solve
at least one variant per action id; more is better. A hand without a goal in a segment stays where
it is (it may move to make room, but must end where it was).

Deliver `trajectories.json` (format in `FORMAT.md`): for every problem you solved, one joint
trajectory per non-`world` segment, continuous from segment to segment, starting at
`robot.json.home`, within joint limits, with no collision between any arm link or carried object
and any obstacle not in `off`, and no collision between the two arms. Also `notes.md`: your method,
what failed, and anything about the layout you think is wrong (say so if a problem is infeasible
and why).

Rules
- Use any planner or library you like (`python3` here has numpy and scipy; you may pip-install
  CPU packages). The verifier is ours: a sphere model of the arms checked every 0.05 rad along
  your trajectory; see `FORMAT.md` for its tolerances.
- Budget: 2 hours wall time. Stop and hand in what you have when it runs out.
