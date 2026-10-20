# plan.json for twin_sled.py

```json
{
 "task": "sled",
 "planner": "<your name for it>",
 "scene": { ...the scene file, verbatim... },
 "goal": {"in_box": ["block1", "block2", "cup", "bottle"], "folds_min": 2, "towel_in_box": true, "before": []},
 "plan": [ ...steps, executed in order... ],
 "riders": ["block1", "block2", "cup"],
 "towel_place": {"rect": [-15, 15, 32, 47], "size_cm": [30, 15], "folds": 2}
}
```

- `goal`: what the audit checks (the predicates of the goal sentence). `in_box`: objects that must
  end inside the box, upright. `folds_min`: the towel must have been folded at least this many
  times and be at most 1.25 x its folded length. `towel_in_box`: 98 % of the towel inside the box
  and 90 % within `towel_place.rect`. `before`: pairs [a, b], a packed before b.
- `riders`: the objects that are still on the towel when it is dragged (they ride along).
- `towel_place` (only if the towel is packed): the rectangle `[x0, x1, y0, y1]` on the box floor
  where the folded towel should lie, its folded `size_cm` and `folds`.

Frame: cm, origin on the table between the arm bases, x to the right arm, y away from the robots.
The box floor is centred at `(0, 36)`, inner 38 x 38 cm, walls 12 cm; the arm bases are at
`(+-30, -2)`, reach about 92 cm from a base. The towel's `centre_xy` and `size_cm` (x extent, y
extent) are in the scene; objects are at `towel.centre_xy + on_towel_xy`.

## Steps

```json
{"op": "pick_place", "name": "bottle", "to_xy": [10.7, 18.0]}
```
One arm picks `name` where it stands and places it at `to_xy` on the box floor (box frame: the
box floor's centre is at `(0, 36)`; give the target in the table frame). Objects must not overlap in
the box. Optional: `"hand": "left"|"right"`, `"jaw": [dx, dy]` (jaw closing direction).

```json
{"op": "drag", "dir": [0.0, 1.0], "dist_cm": 33.6, "accel": 300.0, "vmax": 25.0, "lift_cm": 1.0,
 "grasp_uv": {"left": [0.25, 0.98], "right": [0.75, 0.98]}}
```
Both arms pinch the towel at `grasp_uv` (u along x from the left edge, v along y from the far
edge, both 0..1), lift `lift_cm`, and pull `dist_cm` along `dir` (unit vector; `[0, 1]` is
towards the robots) with acceleration `accel` cm/s^2 up to `vmax` cm/s. Everything on the towel
rides along, or topples.

```json
{"op": "fold", "name": "far edge onto near edge (1)",
 "moves": [{"hand": "left", "from": [0.05, 0.02], "to": [0.05, 0.98]}, {"hand": "right", "from": [0.95, 0.02], "to": [0.95, 0.98]}],
 "to_cm": [30, 30]}
```
Each hand pinches the towel at `from` (uv of the towel as it lies now) and carries it in an arc to
`to`. `to_cm` is the towel's size after the fold. The towel must be clear of objects.

```json
{"op": "pack_towel", "grasp_uv": {"left": [0.05, 0.875], "right": [0.95, 0.875]},
 "band_half_cm": 4.0, "to_xy": [0.0, 39.5], "clear_cm": 10.0}
```
Both arms pinch the folded towel at `grasp_uv`, lift it `clear_cm` above the box walls, carry it
over the box and lower it so its centre lands at `to_xy`. `band_half_cm`: half-width of the pinched band.
