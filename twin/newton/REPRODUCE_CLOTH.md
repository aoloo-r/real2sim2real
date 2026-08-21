# Reproducing the cloth folding run

Everything below runs on a single RTX 4090. The cloth simulation is Newton's own
`cloth_franka` example; our script is a modified copy of it.

## Environment

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

## 1. Upstream baseline (no code of ours)

This is the stock example our script is derived from. Start here — if the
behaviour of interest shows up already, nothing of ours is involved:

```bash
python -m newton.examples cloth_franka
```

## 2. Our script

| | |
|---|---|
| repo | https://github.com/aoloo-r/real2sim2real |
| branch | `main` |
| file | `twin/newton/twin_stockfold.py` |

```bash
git clone https://github.com/aoloo-r/real2sim2real
cd real2sim2real/twin/newton
DISPLAY=:1 TWIN_PLACE=stock python twin_stockfold.py --viewer gl
```

`TWIN_PLACE=stock` is the important flag: it keeps the example's own scene layout
and needs **no** data from us. Without it the script expects an RGBD capture and
SAM 3D reconstruction that only exist on our machine. The reconstructed box is
loaded inside a `try`, so its absence just prints `box skipped` and the run
continues — verified on a clean path.

Console output is one line per phase, e.g.

```
[twin-stock] flat sheet 55x60cm
[twin-stock] t=4s  phase 'fold 1 left panel'  queued (cloth 55x60cm)
[twin-stock] fold carries 2123 particles across the line
[twin-stock] t=21s phase 'fold 2 right panel' queued (cloth 36x60cm)
```

## 3. What we changed relative to the stock example

The cloth itself is untouched: same `unisex_shirt.usd`, same `add_cloth_mesh`
call, same solver settings. Our changes are the choreography around it.

- **Fold style** (`TWIN_FOLD_STYLE`, default `shirt`) — three garment folds on the
  flat shirt (left panel at 1/3 width, right panel at 2/3, hem to collar), instead
  of the stock 5-corner sequence. `TWIN_FOLD_STYLE=stock` restores the original.
- **Grasp depth** (`TWIN_GRASP_DZ_CM`, default `0.5`) — every grasp closes 0.5 cm
  below the authored height. At 0.0 the first corner grasp visibly misses the
  cloth; at 1.0 the jaws close through it and the fabric rides the gripper on
  release.
- **Fold assist** (`TWIN_FOLD`, default `assist`) — the half beyond the fold line
  is pinned and mirrored across it, tracked by how far the gripper has travelled.
  `TWIN_FOLD=physical` uses purely frictional folding.
- **Unfold** (`TWIN_UNFOLD=1`) — replays each fold with grab/drop swapped.
- **Pack** (`TWIN_PACK=1`) — carries the folded bundle into a box (needs capture
  data; skip for a clean-room repro).

## 4. Why the fold assist exists — the observation worth looking at

Purely frictional folding **plateaus**. Measured over six passes with
`TWIN_FOLD=physical`:

```
54 -> 49 -> 45 -> 46 -> 45 -> 44 cm
```

After two folds the pile is roughly 2 cm of layered fabric, and a parallel jaw
closing at table level only catches the top layer: the drag peels that layer and
the pile springs back. Grasping nearer the hem (3% inside the edge instead of 8%)
and dragging lower (8 cm instead of 11 cm) improved the early passes but not the
plateau. A single grasped bundle also cannot be lifted — the arm arrives at the
target empty every time — which is why the carry is kinematic rather than
frictional.

To watch this directly:

```bash
DISPLAY=:1 TWIN_PLACE=stock TWIN_FOLD=physical TWIN_FOLD_STYLE=stock \
    python twin_stockfold.py --viewer gl
```

## 5. Unit-scale note

The scene simulates in **centimetres** and renders in **metres**. Newton's GL
viewer bakes primitive dimensions at `set_model` time and ignores the later
`shape_scale` rescale, so any `add_shape_box` we add draws 100x oversized unless
its `VISIBLE` flag is cleared and it is drawn manually — the table and our
collision cage both do this. Porting the stock cloth constants into a metre-scale
scene (gravity -981, `tri_ke` 1e4, `particle_radius` 0.8, URDF `scale=100`) is
what produced holes and shredding for us earlier; running in the units the
constants were tuned for fixed it.
