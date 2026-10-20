# scene.json

All lengths in **cm**, masses in kg. Frame: the table frame of `inputs/frame.json` (origin on the
table surface, x to the right, y away from the robots towards the far edge, z up).

```json
{
 "box":   {"inner_cm": [38, 38], "h_cm": 12, "xy": [0, 36]},
 "towel": {"centre_xy": [0, -90], "size_cm": [30, 60], "mass_kg": 0.25, "mu": 0.5},
 "mu_table": 0.5,
 "pack_towel": true,
 "objects": [
  {"name": "block1", "shape": "box",    "dims_cm": [8, 8, 6],      "mass_kg": 0.35, "mu": 0.5, "on_towel_xy": [-6, -22]},
  {"name": "cup",    "shape": "cup",    "dims_cm": [8, 8, 9],      "mass_kg": 0.30, "mu": 0.5, "on_towel_xy": [7, -20], "handle_cm": 3.0},
  {"name": "bottle", "shape": "bottle", "dims_cm": [6.4, 6.4, 20], "mass_kg": 0.05, "mu": 0.5, "on_towel_xy": [6, 22], "fill": 0.05}
 ]
}
```

- `box.inner_cm` = inner width (x) and depth (y); `h_cm` = wall height; `xy` = centre of the box
  floor in the table frame.
- `towel.centre_xy` = centre of the towel in the table frame; `size_cm` = [extent along x, extent
  along y] as it lies.
- `objects[].name`: `block1`, `block2`, ... for cuboids; `cup` for a mug; `bottle` for a bottle.
  Unique names.
- `shape`: one of `box` (cuboid), `cup` (mug with a handle), `bottle` (narrow neck), `cylinder`.
- `dims_cm` = [x extent, y extent, height] of the object's body as it stands (the cup's handle is
  not included: give it as `handle_cm`, the handle's reach beyond the body).
- `on_towel_xy` = object centre relative to `towel.centre_xy`, in the same frame.
- `fill` (bottle only) = liquid level as a fraction of the height, if visible; omit otherwise.
- `mu`, `mu_table`: friction estimates; `pack_towel`: leave `true`.
