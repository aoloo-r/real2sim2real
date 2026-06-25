# -*- coding: utf-8 -*-
"""Adapt SAM 3D output into a real2sim-ready scene.

SAM 3D (demo.py) produces clean canonical-scale meshes + vertex colors but no
metric size / depth positions. This wraps them through the SAME metric-scaling,
focus-filter, orientation, and scene_layout assembly used by the Hunyuan
pipeline — preserving SAM 3D's superior geometry while giving real2sim_franka.py
a baked, calibrated-placement-ready scene (no destructive voxel remesh).

Usage:
  python sam3d_to_scene.py --sam3d_dir outputs/sam3d_test \
      --capture_dir captures/robot_XXXX --output_dir outputs/sam3d_scene
"""
import argparse, json, os, shutil
import numpy as np, cv2, trimesh
from PIL import Image

from depth_scale import compute_physical_sizes
from object_focus import select_target_objects
from hunyuan_demo import (prepare_mesh, save_object, save_scene_layout,
                          sample_object_color, estimate_physical_size)
from render_compare import (up_in_camera, _rotation_z_to, rotation_to_quat_wxyz,
                            score_reconstruction)
from primitive_fit import category_for_label, build_primitive, make_lofted_dish


def _attach_colors(mesh, vc):
    """Attach per-vertex colors (Nx3, 0..1 or 0..255) to a trimesh."""
    v = np.asarray(vc, dtype=float)
    if v.max() <= 1.5:
        v = v * 255.0
    rgba = np.ones((len(mesh.vertices), 4), dtype=np.uint8) * 255
    rgba[:, :3] = np.clip(v[:, :3], 0, 255).astype(np.uint8)
    mesh.visual.vertex_colors = rgba


def _orient_open_up(m):
    """Flip a real reconstructed mesh so its concave/open side faces UP (e.g. a
    bowl/plate that came out open-side-down). Compares the height of the rim
    (outer ring) vs the center: if the center is higher than the rim, it's a
    dome/upside-down container -> rotate 180deg about X. No predefining — just
    reorienting the actual mesh."""
    v = np.asarray(m.vertices, dtype=float)
    z = v[:, 2]
    cx, cy = v[:, 0].mean(), v[:, 1].mean()
    r = np.sqrt((v[:, 0] - cx) ** 2 + (v[:, 1] - cy) ** 2)
    rmax = r.max()
    if rmax < 1e-6:
        return m
    inner = z[r < 0.40 * rmax]
    outer = z[r > 0.75 * rmax]
    height = float(z.max() - z.min())
    if len(inner) > 8 and len(outer) > 8 and height > 1e-4:
        # margin = 10% of height to avoid flipping near-flat/ambiguous objects
        if inner.mean() > outer.mean() + 0.10 * height:
            m.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
            m.apply_translation([0.0, 0.0, -m.bounds[0][2]])  # reseat bottom at z=0
            print("    [ORIENT] flipped open-side up")
    return m


def _to_local(m, up):
    """Bring a camera-frame (already-metric) mesh into the canonical local frame:
    up-axis -> +Z, centered in XY, bottom at z=0. Used for the depth-carve
    fallback so it matches the pipeline's mesh convention without rescaling."""
    R = np.asarray(_rotation_z_to(up)).T  # maps up_cam -> +Z
    out = m.copy()
    v = np.asarray(out.vertices, dtype=float)
    v = (R @ (v - v.mean(axis=0)).T).T
    out.vertices = v
    out.apply_translation([0.0, 0.0, -out.bounds[0][2]])
    out._r2s_scale_factor = 1.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3d_dir", required=True)
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--robot_base_frame", default="ur5e_base_link")
    ap.add_argument("--max_objects", type=int, default=6)
    ap.add_argument("--no_focus", dest="focus", action="store_false", default=True)
    # Render-compare QA gate on the SAM 3D meshes
    ap.add_argument("--no_qa", dest="qa", action="store_false", default=True,
                    help="Disable the render-compare QA gate.")
    ap.add_argument("--qa_min_iou", type=float, default=0.5,
                    help="Silhouette IoU below this flags an object as failed.")
    ap.add_argument("--fallbacks", type=str, default="depthcarve,primitive",
                    help="Comma-separated self-repair fallbacks tried (in order) "
                         "when SAM 3D fails QA: depthcarve, primitive. Empty=none.")
    ap.add_argument("--drop_failed", action="store_true", default=False,
                    help="Drop objects that still fail QA after all fallbacks "
                         "(default: keep the best candidate + flag needs_recapture).")
    args = ap.parse_args()
    fallbacks = [f.strip() for f in args.fallbacks.split(",") if f.strip()]

    cap, sd, out = args.capture_dir, args.sam3d_dir, args.output_dir
    rgb = cv2.cvtColor(cv2.imread(os.path.join(cap, "rgb.png")), cv2.COLOR_BGR2RGB)
    depth = np.load(os.path.join(cap, "depth.npy"))
    intr = json.load(open(os.path.join(cap, "intrinsics.json")))
    sam = json.load(open(os.path.join(sd, "scene_layout.json")))
    objs = sam["objects"]
    labels = [o["label"] for o in objs]
    boxes = [{"label": o["label"], "box_px": o["box_px"]} for o in objs]
    masks = [np.array(Image.open(os.path.join(sd, "masks", f"{o['id']}.png")).convert("L"))
             for o in objs]
    image_size = tuple(sam["image_size_px"])

    # calibrated camera->robot extrinsic
    T_base_cam = None
    ep = os.path.join(cap, "extrinsics.json")
    if os.path.isfile(ep):
        tf = (json.load(open(ep)).get("transforms") or {}).get(args.robot_base_frame)
        if tf:
            T_base_cam = tf["T_base_cam"]

    # focus filter (central/reachable)
    dr_all = compute_physical_sizes(depth, intr, masks, boxes)
    positions = [(r.get("position_cam") if r else None) for r in dr_all]
    keep = (select_target_objects(boxes, positions, image_size, T_base_cam,
                                  max_objects=args.max_objects)
            if args.focus else list(range(len(objs))))
    print(f"[SAM3D->scene] keeping {len(keep)}/{len(objs)}: {[labels[i] for i in keep]}")

    os.makedirs(out, exist_ok=True)
    up = up_in_camera(T_base_cam)
    metadatas, out_labels, out_boxes = [], [], []
    for i in keep:
        oj = len(metadatas)  # contiguous output id (matches save_scene_layout)
        oid = objs[i]["id"]
        objdir = os.path.join(out, f"object_{oj}")
        os.makedirs(objdir, exist_ok=True)
        dri = compute_physical_sizes(depth, intr, [masks[i]], [boxes[i]])
        dri = dri[0] if dri else {}
        physical_size = dri.get("physical_size_m") or estimate_physical_size(
            tuple(boxes[i]["box_px"]), image_size)
        pos = dri.get("position_cam")
        color = sample_object_color(rgb, masks[i])

        # ---- Candidate builders (each returns (prepared_mesh, vertex_colors|None)) ----
        def cand_sam3d():
            m = trimesh.load(os.path.join(sd, f"object_{oid}", "mesh.obj"),
                             force="mesh", process=False)
            nv = len(m.vertices)
            p = prepare_mesh(m, physical_size, label=labels[i], obj_id=oj)
            _orient_open_up(p)  # flip real mesh so a bowl/plate opens UP (no predefining)
            vcol = None
            vsrc = os.path.join(sd, f"object_{oid}", "vertex_colors.npy")
            if os.path.exists(vsrc):
                v = np.load(vsrc)
                if len(v) == nv == len(p.vertices):
                    vcol = v
                    _attach_colors(p, v)
            return p, vcol

        def cand_depthcarve():
            from depth_mesh import build_mesh_from_depth
            m = build_mesh_from_depth(depth, masks[i], intr, rgb)
            if m is None or len(m.vertices) == 0:
                return None
            vcol = None
            if getattr(m.visual, "vertex_colors", None) is not None:
                vcol = np.asarray(m.visual.vertex_colors)[:, :3]
            p = _to_local(m, up)
            if vcol is not None and len(vcol) == len(p.vertices):
                _attach_colors(p, vcol)
            else:
                vcol = None
            return p, vcol

        def cand_primitive():
            cat = category_for_label(labels[i])
            dw = dri.get("physical_width_m") or physical_size
            dh = dri.get("physical_height_m") or physical_size
            if cat is None:
                # Infer a SMALL, roughly-round unknown object (fruit/olive/ball)
                # as a small ellipsoid sized to its real measurement — a clean
                # ovoid, not the exaggerated lumpy blob depth-carve produces.
                aspect = (min(dw, dh) / max(dw, dh)) if max(dw, dh) > 1e-6 else 1.0
                if physical_size <= 0.10 and aspect >= 0.6:
                    cat = "ellipsoid"
                else:
                    return None  # no primitive sensibly approximates it -> don't fake a box
            m = None
            if cat == "dish":
                # Clean rounded-RECTANGLE open dish sized to the real footprint
                # (square-ish like a tray, but smooth) — keeps the shape without
                # the noisy/angular look of lofting the raw mask contour.
                try:
                    from primitive_fit import rounded_rect_contour
                    cm = rounded_rect_contour(dw, dh)
                    ll = (labels[i] or "").lower()
                    dfrac = 0.42 if ("bowl" in ll or "container" in ll) else 0.16
                    m = make_lofted_dish(cm, height=max(0.02, max(dw, dh) * dfrac))
                except Exception as e:
                    print(f"  [PRIM] rounded-rect dish failed ({e!r}); using round dish")
                    m = None
                if m is None:
                    m = build_primitive("dish", (physical_size, physical_size, 0.0), labels[i])
            elif cat == "cup":
                # hollow open cup sized to footprint width x tallest dim -> rim-graspable
                m = build_primitive("cup", (dw, dw, physical_size), labels[i])
            elif cat == "cylinder":
                m = build_primitive(cat, (dw, dh, physical_size), labels[i])
            else:                                # ellipsoid / box
                m = build_primitive(cat, (dw, dh, min(dw, dh)), labels[i])
            m._r2s_scale_factor = 1.0
            vcu = np.tile(np.asarray(color, dtype=float), (len(m.vertices), 1))
            _attach_colors(m, vcu)             # colored (not gray) primitive
            return m, vcu

        def cand_cylinder():
            # Geometry-fit a CLEAN cylinder to a thin/elongated object (pen, tube,
            # marker) that SAM 3D can't reconstruct — dimensions + axis come from
            # the masked depth points (not the label), so it's data-derived, and it
            # replaces the amorphous depth-carve blob with a correctly-sized tube.
            ys2, xs2 = np.where(masks[i]); z2 = depth[ys2, xs2]; sel = z2 > 0.05
            if sel.sum() < 30:
                return None
            fx, fy, cxi, cyi = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
            pc = np.stack([(xs2[sel]-cxi)*z2[sel]/fx, (ys2[sel]-cyi)*z2[sel]/fy, z2[sel]], 1)
            c = np.median(pc, 0); d = pc - c
            evals, evecs = np.linalg.eigh(d.T @ d / len(d))
            axis = evecs[:, 2]                       # long axis (max variance)
            proj = d @ evecs
            L = float(np.percentile(proj[:, 2], 98) - np.percentile(proj[:, 2], 2))
            diam = max(float(np.percentile(proj[:, 1], 98) - np.percentile(proj[:, 1], 2)),
                       float(np.percentile(proj[:, 0], 98) - np.percentile(proj[:, 0], 2)))
            if L < 0.02 or diam / max(L, 1e-6) > 0.55:
                return None                          # not actually elongated
            cyl = trimesh.creation.cylinder(radius=max(diam/2.0, 5e-3),
                                             height=max(L, 1e-3), sections=40)
            zc = np.array([0, 0, 1.0]); v = np.cross(zc, axis)
            s = float(np.linalg.norm(v)); cc = float(zc.dot(axis))
            if s > 1e-8:
                vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
                R = np.eye(3) + vx + vx @ vx * ((1-cc)/(s*s))
            else:
                R = np.eye(3) if cc > 0 else np.diag([1.0, -1.0, -1.0])
            Tm = np.eye(4); Tm[:3, :3] = R; Tm[:3, 3] = c
            cyl.apply_transform(Tm)
            p = _to_local(cyl, up)
            vcu = np.tile(np.asarray(color, dtype=float), (len(p.vertices), 1))
            _attach_colors(p, vcu)
            print(f"  [CYL] thin object -> cylinder L={L:.3f}m diam={diam:.3f}m")
            return p, vcu

        builders = {"sam3d": cand_sam3d, "depthcarve": cand_depthcarve,
                    "primitive": cand_primitive, "cylinder": cand_cylinder}
        # Reconstruction choice. SAM 3D's SILHOUETTE can pass QA while its 3D shape is
        # wrong (solid cup/bowl with no cavity, fruit blown into a blob) -> for the
        # round/hollow categories it reliably botches, PREFER the clean primitive sized
        # to the depth measurement (recognizable AND graspable: hollow cup/bowl, ellipsoid
        # fruit). SAM 3D stays first for everything else (boxes, complex/unknown shapes).
        _dw = dri.get("physical_width_m") or physical_size
        _dh = dri.get("physical_height_m") or physical_size
        _elong = (max(_dw, _dh) >= 0.08 and min(_dw, _dh)/max(_dw, _dh, 1e-6) <= 0.45)
        _fb = (["cylinder"] if _elong else fallbacks)
        _cat = category_for_label(labels[i])
        _prefer_prim = _cat in ("ellipsoid", "cup", "dish")
        if _prefer_prim:
            order = ["primitive", "sam3d"] + (_fb if (args.qa and pos is not None) else [])
        else:
            order = ["sam3d"] + (_fb if (args.qa and pos is not None) else [])

        # ---- Self-repair: try candidates in order, re-QA, keep best; accept first pass ----
        best = None
        for kind in order:
            try:
                built = builders[kind]()
            except Exception as e:
                print(f"  [REPAIR] {kind} failed: {e!r}")
                continue
            if not built or built[0] is None:
                continue
            cm, cvc = built
            if args.qa and pos is not None:
                s = score_reconstruction(cm, masks[i], intr, pos,
                                         T_base_cam=T_base_cam, rgb_image=rgb)
                iou = s["iou"]
            else:
                s, iou = None, 1.0
            print(f"  [REPAIR] object_{oj} {labels[i]} via {kind}: IoU={iou:.3f}")
            if best is None or iou > best["iou"]:
                best = {"mesh": cm, "kind": kind, "vc": cvc, "iou": iou, "qa": s}
            if s is None or iou >= args.qa_min_iou:
                break  # accepted — prefer earliest (SAM 3D > depthcarve > primitive)

        if best is None:
            print(f"  [REPAIR] object_{oj} ({labels[i]}): no candidate; skipping")
            shutil.rmtree(objdir, ignore_errors=True)
            continue

        accepted = (best["qa"] is None) or (best["iou"] >= args.qa_min_iou)
        if not accepted and args.drop_failed:
            print(f"  [QA] dropping ({labels[i]}) — all candidates < {args.qa_min_iou}")
            shutil.rmtree(objdir, ignore_errors=True)
            continue

        mesh, vc = best["mesh"], best["vc"]
        qa = best["qa"]
        if qa is not None:  # re-score chosen candidate to save its overlay
            qa = score_reconstruction(mesh, masks[i], intr, pos, T_base_cam=T_base_cam,
                                      rgb_image=rgb,
                                      out_overlay=os.path.join(objdir, "qa_overlay.png"))
            qa["accepted"] = accepted
            qa["source_kind"] = best["kind"]
            if not accepted:
                qa["needs_recapture"] = True
        print(f"  [QA] object_{oj} ({labels[i]}): chosen={best['kind']} "
              f"IoU={best['iou']:.3f} -> {'OK' if accepted else 'NEEDS_RECAPTURE'}")

        meta = save_object(mesh, out, oj, physical_size, color,
                           label=labels[i], prepared=True)
        if vc is not None:
            vv = np.asarray(vc, dtype=float)
            if vv.max() > 1.5:           # normalize to 0..1 for real2sim displayColor
                vv = vv / 255.0
            np.save(os.path.join(objdir, "vertex_colors.npy"),
                    vv[:, :3].astype(np.float32))
        if pos:
            R = _rotation_z_to(up)
            meta["icp_pose"] = {"position_cam": pos,
                                "rotation_cam": rotation_to_quat_wxyz(R),
                                "scale": 1.0, "source": "depth_centroid"}
            meta["depth_info"] = dri
        if qa is not None:
            meta["qa"] = qa
        metadatas.append(meta); out_labels.append(labels[i]); out_boxes.append(boxes[i])
        print(f"  object_{oj} ({labels[i]}): size={physical_size*100:.1f}cm "
              f"src={best['kind']} pos_cam={[round(x,3) for x in (pos or [])]}")

    layout = save_scene_layout(metadatas, out, out_labels, out_boxes, image_size)
    # tag source
    L = json.load(open(layout)); L["source"] = "sam3d"; json.dump(L, open(layout, "w"), indent=2)
    print(f"[SAM3D->scene] wrote {layout}")


if __name__ == "__main__":
    main()
