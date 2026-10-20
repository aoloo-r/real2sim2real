"""Turn the perception stage's layout (SAM 3D meshes with metric poses in the robot base frame) into the
sled scene JSON. Reads the layout generically: an object's position from the first of
position_base / position / icp_pose.position_cam, its size from the mesh's axis-aligned extents.
Validate on the first real capture: the keys below are our best reading of perception/sam3d_to_scene.py.

    layout_to_sled.py --recon RECON_DIR --frame frame.json --out scene.json"""
import argparse
import glob
import json
import os

import numpy as np

CLASS = {"mug": "cup", "cup": "cup", "bottle": "bottle", "block": "box", "cube": "box", "box": "box", "towel": "towel", "cloth": "towel"}


def to_table(p_base, fr):
    """Base-frame point (m) -> table frame (cm)."""
    o, x = np.array(fr["origin_in_base_m"], float), np.array(fr["x_axis_in_base"], float)
    z = np.array(fr.get("z_axis_in_base", [0, 0, 1]), float)
    x = x / np.linalg.norm(x); y = np.cross(z, x)
    d = np.array(p_base, float) - o
    return [float(d @ x) * 100, float(d @ y) * 100, float(d @ z) * 100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import trimesh
    fr = json.load(open(a.frame))
    lay = None
    for f in glob.glob(os.path.join(a.recon, "**", "*.json"), recursive=True):
        L = json.load(open(f))
        if isinstance(L, dict) and "objects" in L:
            lay = L; print(f"[layout] {f}: {len(L['objects'])} objects"); break
    if lay is None:
        raise SystemExit(f"[layout] no layout JSON with 'objects' under {a.recon}")
    objs, towel, box, nblk = [], None, None, 0
    for i, r in enumerate(lay["objects"]):
        label = str(r.get("label") or r.get("name") or "object").lower()
        cls = next((v for k, v in CLASS.items() if k in label), "box")
        pos = r.get("position_base") or r.get("position") or (r.get("icp_pose") or {}).get("position_cam")
        mesh = r.get("mesh") or r.get("mesh_path") or os.path.join(a.recon, f"object_{r.get('id', i)}", "mesh.obj")
        ext = None
        if mesh and os.path.exists(mesh):
            m = trimesh.load(mesh, force="mesh", process=False)
            ext = (m.bounds[1] - m.bounds[0]) * 100
            if ext.max() > 500:                                  # mm mesh
                ext = ext / 10
        print(f"[layout]   {label}: class {cls}, pos {pos}, extents {None if ext is None else ext.round(1).tolist()} (mesh {mesh})")
        if pos is None or ext is None:
            print("[layout]   skipped: no pose or mesh"); continue
        p = to_table(pos, fr)
        if cls == "towel":
            towel = dict(centre_xy=[round(p[0], 1), round(p[1], 1)], size_cm=[round(float(ext[0]), 1), round(float(ext[1]), 1)], mass_kg=0.25, mu=0.5)
        elif "box" in label and ext.max() > 25:
            box = dict(inner_cm=[round(float(ext[0]) - 2, 1), round(float(ext[1]) - 2, 1)], h_cm=round(float(ext[2]), 1), xy=[round(p[0], 1), round(p[1], 1)])
        else:
            nblk += cls == "box"
            o = dict(name={"cup": "cup", "bottle": "bottle"}.get(cls, f"block{nblk}"), shape=cls, dims_cm=[round(float(v), 1) for v in ext],
                     mass_kg={"cup": 0.3, "bottle": 0.5}.get(cls, 0.3), mu=0.5, xy=[round(p[0], 1), round(p[1], 1)])
            if cls == "cup":
                o["handle_cm"] = 3.0
            objs.append(o)
    if towel is None:
        raise SystemExit("[layout] no towel found; cannot express object positions relative to it")
    for o in objs:
        o["on_towel_xy"] = [round(o["xy"][0] - towel["centre_xy"][0], 1), round(o["xy"][1] - towel["centre_xy"][1], 1)]; del o["xy"]
    scene = dict(box=box or dict(inner_cm=[38, 38], h_cm=12, xy=[0, 36]), towel=towel, mu_table=0.5, pack_towel=True, objects=objs)
    json.dump(scene, open(a.out, "w"), indent=1)
    print(f"[layout] wrote {a.out}" + ("" if box else " (box not found: default box)"))


if __name__ == "__main__":
    main()
