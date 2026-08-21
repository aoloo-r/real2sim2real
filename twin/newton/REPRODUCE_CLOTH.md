# Reproducing the cloth folding run

The run in the video is Newton's own `cloth_franka` example. Nothing of ours is
needed to reproduce it — start at §2. Our modified copy is described in §4 only
in case it is useful for comparison.

## 1. Environment

| | |
|---|---|
| newton | 1.5.0 |
| warp | 1.16.0 |
| python | 3.11.15 |
| driver | 580.173.02 (RTX 4090) |
| platform | Ubuntu, X display on `:1` |

```bash
conda create -n newton-1.5 python=3.11
conda activate newton-1.5
pip install newton==1.5.0        # pulls warp 1.16.0
```

## 2. The reproduction

```bash
python -m newton.examples cloth_franka
```

That is the whole repro. The example folds `unisex_shirt.usd` with a Franka using
its own 33-pose choreography; the scene runs in centimetre units.

## 3. What we observe

**The corner grasps close above the cloth rather than on it.** The grasp poses sit
at `z = 20.0`, which is exactly the table top, so the jaws meet at the surface
with the fabric alongside rather than between them. The drag that follows then
moves little or no cloth, and the shirt ends up crumpled rather than folded.

Source of the poses — `newton/examples/cloth/example_cloth_franka.py`, the
`robot_key_poses` table starting at **line 391**:

```python
# top left
[2, 31.0, -60.0, 20.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
[2, 31.0, -60.0, 20.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
```

**Evidence that the grasp is the cause.** Lowering only the grasp-height poses —
every row whose `z <= 21`, leaving the 26–40 cm lift and drag heights untouched —
changes the outcome of the example's own fold sequence:

| grasp height | shirt after the stock 5-corner fold |
|---|---|
| `z = 20.0` (as shipped) | 25 x 54 cm — barely folded |
| `z = 19.0` (−1.0 cm) | **26 x 31 cm** — genuinely folded |

Same choreography, same cloth, same solver settings; only the grasp depth
differs. At −1.0 cm there is a second effect worth knowing about: the jaws then
close *through* the fabric and it rides the gripper on release instead of
dropping. −0.5 cm catches the cloth without wedging it, and is what we settled on.

A one-line way to try it, applied after the table is built:

```python
z = self.robot_key_poses[:, 3]
z[z <= 21.0] -= 0.5     # grasp heights only; lifts and drags untouched
```

**Second observation — frictional folding plateaus.** Repeated fold passes stop
making progress:

```
54 -> 49 -> 45 -> 46 -> 45 -> 44 cm      (six passes)
```

After two folds the pile is roughly 2 cm of layered fabric and a parallel jaw at
table level only catches the top layer: the drag peels that layer and the pile
springs back. Related: a grasped bundle cannot be lifted at all — the arm arrives
at the target empty every time, so carrying the folded shirt anywhere needs a
kinematic attachment rather than a frictional grasp.

## 4. Our modified copy (optional, for comparison only)

| | |
|---|---|
| repo | https://github.com/aoloo-r/real2sim2real |
| branch | `main` |
| file | `twin/newton/twin_stockfold.py` |

```bash
DISPLAY=:1 TWIN_PLACE=stock python twin_stockfold.py --viewer gl
```

`TWIN_PLACE=stock` keeps the example's own scene layout and needs no data from
us. The cloth is untouched relative to upstream — same `unisex_shirt.usd`, same
`add_cloth_mesh` call, same solver settings; only the choreography differs
(`TWIN_GRASP_DZ_CM` grasp depth, `TWIN_FOLD_STYLE=shirt` three-move garment fold,
`TWIN_FOLD=assist|physical`, `TWIN_UNFOLD`, `TWIN_PACK`).

## 5. Unit-scale note

The scene simulates in centimetres and renders in metres. Newton's GL viewer
bakes primitive dimensions at `set_model` time and ignores the later
`shape_scale` rescale, so an added `add_shape_box` draws 100x oversized unless
its `VISIBLE` flag is cleared and it is drawn manually — the example's table
already does this. Porting the cloth constants into a metre-scale scene
(gravity -981, `tri_ke` 1e4, `particle_radius` 0.8, URDF `scale=100`) produced
holes and shredding for us; running in the units they were tuned for fixed it.
