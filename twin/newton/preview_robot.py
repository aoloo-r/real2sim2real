#!/usr/bin/env python3
"""Quick PREVIEW of the UR5e + gripper at the home pose in Newton (no cloth, no motion) so we can
see the arm and gripper clearly. --gripper jaw (2-jaw physical) or 2f85 (Robotiq)."""
import argparse
import numpy as np
import warp as wp
import newton
from newton import ModelBuilder, eval_fk
from newton.viewer import ViewerGL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gripper", default="jaw", choices=["jaw", "2f85"])
    ap.add_argument("--cam", default="120,-70,55,-16,28")
    ap.add_argument("--screenshot", default="/tmp/robot_preview.png")
    ap.add_argument("--hold", type=float, default=60.0)
    args = ap.parse_args()

    wp.init()
    # kill the GL shader's world-unit distance fog (cm scene sits inside 20-200 -> grey haze)
    try:
        from newton._src.viewer.gl import shaders as _sh
        _sh.shape_fragment_shader = (_sh.shape_fragment_shader
            .replace("float fog_start = 20.0;", "float fog_start = 100000.0;")
            .replace("float fog_end   = 200.0;", "float fog_end   = 200000.0;"))
    except Exception:
        pass

    builder = ModelBuilder()
    base = wp.transform(wp.vec3(0.0, 0.0, 20.0), wp.quat_identity())
    if args.gripper == "jaw":
        from ur5e_jaw_gripper import add_ur5e_jaw_gripper
        w3, dof0, n_arm, pinch, gdofs, op, cp = add_ur5e_jaw_gripper(builder, base, scale=100.0)
        print(f"[preview] UR5e + 2-JAW gripper: {n_arm} arm dof, grip dofs {gdofs}, open {op} close {cp:.1f}cm")
    else:
        from ur5e_gripper import add_ur5e_gripper
        w3, dof0, n_arm, pinch = add_ur5e_gripper(builder, base, scale=100.0)
        print(f"[preview] UR5e + Robotiq 2F-85: {n_arm} arm dof")

    builder.add_ground_plane()
    model = builder.finalize()
    state = model.state()
    eval_fk(model, model.joint_q, model.joint_qd, state)

    viewer = ViewerGL()
    viewer.set_model(model)
    r = getattr(viewer, "renderer", None)
    if r is not None:
        try:
            r.draw_sky = False
            r.sky_upper = (0.55, 0.57, 0.60); r.sky_lower = (0.32, 0.33, 0.35)
            r._light_color = (1.0, 1.0, 0.97)
        except Exception:
            pass
    cx, cy, cz, cp_, cyaw = (float(v) for v in args.cam.split(","))
    viewer.set_camera(wp.vec3(cx, cy, cz), cp_, cyaw)

    import time
    from PIL import Image
    for i in range(6):
        viewer.begin_frame(0.0); viewer.log_state(state); viewer.end_frame()
    Image.fromarray(viewer.get_frame().numpy()).save(args.screenshot)
    print(f"[preview] shot -> {args.screenshot}", flush=True)
    t_end = time.time() + args.hold
    while time.time() < t_end and viewer.is_running():
        viewer.begin_frame(0.0); viewer.log_state(state); viewer.end_frame()


if __name__ == "__main__":
    main()
