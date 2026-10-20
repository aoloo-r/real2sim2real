# Capture protocol for the sled scene (one session gives every input)

Lay out the real sled scene on the robot table: towel flat, the open box, and on the towel two
blocks, a mug and a bottle. Mark the table frame with tape: origin on the table surface at the
midpoint between the arm bases (one real arm: the point the twin's right base maps to, 30 cm to
its left), x-axis towards the right arm, y-axis away from the robot.

1. **RGB-D** (robot camera, ROS1): `perception/capture_rgbd_ros1.py --out inputs/capture`. Gives
   `color_raw.npy`, `depth.npy` (m), `intrinsics.json`, `extrinsics.json` (camera -> ur5e base),
   `capture_meta.json`. Add `color.png` (`python -c "import numpy,cv2;cv2.imwrite('color.png',numpy.load('color_raw.npy'))"`).
2. **Phone video** (optional; for agents that prefer video): a slow orbit of the table, then
   5 s static from the robot's side. Save as `inputs/capture/orbit.mp4`.
3. **Frame**: `inputs/frame.json` = `{"origin_in_base_m": [x, y, z], "x_axis_in_base": [x, y, z], "z_axis_in_base": [0, 0, 1]}`,
   measured from the robot base to the taped origin (teach-pendant jog to the mark, read the TCP).
4. **Measurements** (`inputs/gt.json`, from `gt_template.json`): tape-measure, to the mm:
   - each object: x/y/z extents, centre position relative to the towel centre, mass on a kitchen scale;
   - towel: extent along x and y as it lies, centre in the table frame, mass;
   - box: inner width and depth, wall height, centre of the floor in the table frame.
   Measure positions from the towel's centre, which is marked on the towel with tape.
5. Photograph the ruler against each object (`inputs/gt_photos/`) so a measurement can be re-read.

The baseline (`baseline.sh`) runs our own stage on the same `inputs/capture`.
