# Inputs

`robot.json`
```json
{"urdf": "dual_ur5e.urdf", "joints": ["shoulder_pan_joint", ..., "wrist_3_joint_1"],
 "home": [12 joint values, rad], "limits": [[lo, hi] x 12],
 "bases": {"right": [x, y, z, yaw], "left": [...]},  "tcp_links": {"right": "tcp", "left": "tcp_1"},
 "tool_down_quat_wxyz": {"right": [w, x, y, z], "left": [...]},  "units": "m, rad"}
```
`tool_down_quat_wxyz` is the tcp orientation that points the gripper straight down with the jaws
along x; every goal pose in `actions.json` is that rotated about z by the jaw yaw.

`actions.json`: a list of problems
```json
{"id": "pick_place:bottle", "variant": "right arm, jaws +1,+0", "problem": 0,
 "world": [{"name": "table", "pose": [x, y, z, 1, 0, 0, 0], "dims": [dx, dy, dz]}, ...],
 "segments": [
   {"kind": "plan", "goals": {"right": [x, y, z, qw, qx, qy, qz]}, "attach": {}, "off": [], "contact": false},
   {"kind": "path", "goals": {"right": [[pose], [pose], ...]}, "attach": {}, "off": ["bottle", "towel"], "contact": false},
   {"kind": "path", "goals": {"right": [...]}, "attach": {"right": ["bottle"]}, "off": ["towel", "table"], "contact": false},
   {"kind": "world", "world": [...new obstacle list...]},
   ...
 ]}
```
Poses are `[x, y, z, qw, qx, qy, qz]` in metres, world frame (z up; the table top is at z 0.20).
Cuboid `pose` is its centre with identity rotation; `dims` its full extents. Two-hand segments give
a goal for both hands; `plan` segments with `pair_tol` (m) ask the hand-to-hand distance to change
by at most that much (the towel is held by both).

# Output: trajectories.json

```json
{"0": {"segments": [[[q x 12], [q x 12], ...], [[q x 12], ...], ...]},
 "3": {"segments": [...]}}
```
Keyed by `problem`. One trajectory per non-`world` segment, in order. The first segment starts
from `home` and each later one from the last point of the previous; that start point may be
included or left implicit, and the step from it to the first point counts like any other.
Consecutive points differ by at most 0.6 rad in any joint; the verifier interpolates linearly
between them and checks every 0.05 rad with a sphere model of the arms, grippers and carried
objects, and fails any sphere penetrating an obstacle by more than 0.5 cm or a sphere of the other
arm by more than 1.0 cm (the spheres enclose the meshes with margin; the clearances are reported). For a `path` segment every goal pose must be met
in order (tcp position within 1 cm, orientation within 5 degrees); for a `plan` segment, the last
point. A hand without a goal must end within 1 cm of where it started.
