# -*- coding: utf-8 -*-
"""Clean parametric meshes for simple-shaped objects.

Single-view Hunyuan is unreliable for small round fruit (blocky cubes) and
clipped open dishes (warped slivers). For those known categories we fit a smooth
primitive sized from the depth measurements instead — giving clean, real-looking
geometry. Complex objects keep using Hunyuan.
"""
from __future__ import annotations

import numpy as np
import trimesh


# label keyword -> primitive category. Each category is a shape that APPROXIMATES
# the real object (never a generic stand-in): fruit->ellipsoid, cup->cylinder,
# plate/bowl->dish, box->box. Unknown labels return None (no sensible primitive
# -> caller should depth-carve or flag for recapture, NOT invent a box).
ELLIPSOID_KW = ("lemon", "lime", "ball", "apple", "orange", "fruit", "egg",
                "tomato", "onion", "potato", "sphere", "round", "peach", "plum",
                "kiwi", "avocado", "grape", "mango", "pear")
# hollow open vessels -> grasped at the RIM (need a real cavity, not a solid)
CUP_KW = ("cup", "mug", "glass", "tumbler", "jar", "vase", "beaker")
# solid-ish round columns -> grasped on the OUTSIDE wall (solid is fine)
CYLINDER_KW = ("can", "bottle", "tin", "thermos", "flask", "tube", "roll", "bin")
DISH_KW = ("bowl", "plate", "dish", "saucer", "tray", "container", "platter", "lid")
BOX_KW = ("box", "case", "carton", "block", "book", "phone", "tablet", "wallet",
          "pack", "brick")


def category_for_label(label: str):
    """Return 'ellipsoid' | 'cylinder' | 'dish' | 'box', or None if no primitive
    sensibly approximates the object (caller must depth-carve / recapture)."""
    ll = (label or "").lower()
    if any(k in ll for k in ELLIPSOID_KW):
        return "ellipsoid"
    if any(k in ll for k in CUP_KW):
        return "cup"
    if any(k in ll for k in CYLINDER_KW):
        return "cylinder"
    if any(k in ll for k in DISH_KW):
        return "dish"
    if any(k in ll for k in BOX_KW):
        return "box"
    return None


def make_cylinder(diameter, height, sections=48):
    """Upright cylinder (cup/can/bottle approximation), bottom at z=0."""
    m = trimesh.creation.cylinder(radius=max(diameter, 1e-3) / 2.0,
                                  height=max(height, 1e-3), sections=sections)
    m.apply_translation([0, 0, -m.bounds[0][2]])
    return m


def make_cup(diameter, height, wall=0.004, sections=48):
    """Hollow open cup/mug (surface of revolution), bottom at z=0. A real cavity so
    the gripper can straddle the rim wall — NOT a solid cylinder."""
    R = max(diameter, 1e-3) / 2.0
    H = max(height, 1e-3)
    w = min(max(wall, 0.003), 0.4 * R)      # wall thickness
    floor = min(0.10 * H, 0.02)             # inner floor height
    profile = [
        (0.0, 0.0), (R, 0.0), (R, H),                 # outer: bottom -> rim
        (R - w, H), (R - w, floor), (0.0, floor),     # inner: rim -> raised floor
    ]
    m = _revolve(profile, sections)
    m.apply_translation([0, 0, -m.bounds[0][2]])
    return m


def make_box(ex, ey, ez):
    """Box, bottom at z=0."""
    m = trimesh.creation.box(extents=[max(ex, 1e-3), max(ey, 1e-3), max(ez, 1e-3)])
    m.apply_translation([0, 0, -m.bounds[0][2]])
    return m


def make_ellipsoid(ex, ey, ez, subdivisions=4):
    """Smooth ellipsoid with the given full extents (m), bottom at z=0."""
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=0.5)
    m.apply_scale([max(ex, 1e-3), max(ey, 1e-3), max(ez, 1e-3)])
    m.apply_translation([0, 0, -m.bounds[0][2]])  # bottom at z=0
    return m


def _revolve(profile, segments=72):
    """Surface of revolution of a 2D (r, z) profile about the z-axis.

    Pure-numpy (no shapely/boolean backend needed). The profile should start
    and end on or near the axis so the shape closes.
    """
    profile = np.asarray(profile, dtype=float)
    n = len(profile)
    angs = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    verts = np.empty((segments * n, 3), dtype=float)
    for i, a in enumerate(angs):
        ca, sa = np.cos(a), np.sin(a)
        for j in range(n):
            r, z = profile[j]
            verts[i * n + j] = (r * ca, r * sa, z)
    faces = []
    for i in range(segments):
        i2 = (i + 1) % segments
        for j in range(n - 1):
            a = i * n + j; b = i * n + j + 1
            c = i2 * n + j; d = i2 * n + j + 1
            faces.append([a, b, d]); faces.append([a, d, c])
    m = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    return m


def make_dish(diameter, height, concave=True):
    """Open dish/bowl as a surface of revolution, bottom at z=0.

    Profile traces the outer wall up to the rim, then back down the inner wall
    to a raised inner floor — giving a real concave bowl/plate (not a lens).
    """
    R = max(diameter, 1e-3) / 2.0
    H = max(height, 1e-3)
    if concave:
        wall = 0.12 * R                 # wall thickness
        floor = 0.22 * H                # inner floor height
        profile = [
            (0.0, 0.0), (R, 0.0), (R, H),               # outer: bottom -> rim
            (R - wall, H), (R - wall, floor), (0.0, floor),  # inner: rim -> floor
        ]
    else:
        profile = [(0.0, 0.0), (R, 0.0), (R, H), (0.0, H)]  # solid disc
    m = _revolve(profile)
    m.apply_translation([0, 0, -m.bounds[0][2]])  # bottom at z=0
    return m


def rounded_rect_contour(w, h, r=None, n=64):
    """Smooth rounded-rectangle outline (Nx2, centered, CCW) sized w x h. Gives a
    clean tray/bowl footprint — square-ish but with rounded corners — instead of
    a noisy mask contour."""
    w = max(float(w), 1e-3); h = max(float(h), 1e-3)
    if r is None:
        r = 0.30 * min(w, h)
    r = min(r, 0.5 * min(w, h) - 1e-4)
    ax, ay = w / 2 - r, h / 2 - r          # corner-arc centers
    per = max(2, n // 4)
    corners = [(ax, ay, 0.0), (-ax, ay, np.pi / 2),
               (-ax, -ay, np.pi), (ax, -ay, 1.5 * np.pi)]
    pts = []
    for cx, cy, a0 in corners:
        for k in range(per):
            a = a0 + (np.pi / 2) * (k / (per - 1))
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.asarray(pts, dtype=float)


def make_lofted_dish(contour_m, height, wall_frac=0.14, floor_frac=0.25):
    """Open container that KEEPS the real footprint outline (e.g. a rounded-
    square tray), built by lofting the object's actual mask contour into walls +
    rim + inner cavity + floor. Pure numpy. Bottom at z=0, opening up.

    contour_m: (N,2) metric XY points, ordered around the outline, centered.
    """
    C = np.asarray(contour_m, dtype=float)
    N = len(C)
    if N < 3:
        raise ValueError("contour too small")
    H = max(float(height), 1e-3)
    inner = C * (1.0 - wall_frac)          # shrink toward centroid for the cavity
    floor_z = floor_frac * H
    ob = [(C[i, 0], C[i, 1], 0.0) for i in range(N)]        # 0..N-1 outer bottom
    ot = [(C[i, 0], C[i, 1], H) for i in range(N)]          # N..2N-1 outer rim
    it = [(inner[i, 0], inner[i, 1], H) for i in range(N)]  # 2N..3N-1 inner rim
    ifl = [(inner[i, 0], inner[i, 1], floor_z) for i in range(N)]  # 3N..4N-1 floor ring
    V = ob + ot + it + ifl + [(0.0, 0.0, 0.0), (0.0, 0.0, floor_z)]
    cb, cf = 4 * N, 4 * N + 1
    F = []
    for i in range(N):
        j = (i + 1) % N
        F += [[i, j, N + j], [i, N + j, N + i]]                 # outer wall
        F += [[N + i, N + j, 2 * N + j], [N + i, 2 * N + j, 2 * N + i]]  # rim
        F += [[2 * N + i, 2 * N + j, 3 * N + j], [2 * N + i, 3 * N + j, 3 * N + i]]  # inner wall
        F += [[3 * N + i, 3 * N + j, cf]]                       # inner floor fan
        F += [[i, cb, j]]                                       # outer bottom fan
    m = trimesh.Trimesh(vertices=np.array(V, dtype=float),
                        faces=np.array(F, dtype=int), process=True)
    trimesh.repair.fix_normals(m)
    m.apply_translation([0.0, 0.0, -m.bounds[0][2]])
    return m


def build_primitive(category, extents, label=""):
    """Build a primitive mesh for a category given depth extents (ex,ey,ez)."""
    ex, ey, ez = float(extents[0]), float(extents[1]), float(extents[2])
    if category == "ellipsoid":
        # roundish fruit: use footprint for x/y, the smaller of them for depth
        ez = ez if ez > 0.01 else min(ex, ey)
        return make_ellipsoid(ex, ey, ez)
    if category == "dish":
        diameter = max(ex, ey)
        # bowls are deeper than plates
        ll = (label or "").lower()
        depth_frac = 0.42 if ("bowl" in ll or "container" in ll) else 0.16
        height = max(ez, diameter * depth_frac)
        return make_dish(diameter, height, concave=True)
    if category == "cup":
        diameter = min(ex, ey)          # rim diameter (footprint)
        height = max(ez, max(ex, ey))   # tall axis
        return make_cup(diameter, height)
    if category == "cylinder":
        diameter = min(ex, ey)          # rim diameter (footprint)
        height = max(ez, max(ex, ey))   # tall axis
        return make_cylinder(diameter, height)
    if category == "box":
        return make_box(ex, ey, ez if ez > 0.01 else min(ex, ey))
    raise ValueError("unknown category %r" % category)


if __name__ == "__main__":
    # Render samples to /tmp for visual validation.
    import open3d as o3d

    def render(mesh, path):
        v = np.asarray(mesh.vertices) - np.asarray(mesh.vertices).mean(0)
        rad = float(np.linalg.norm(v, axis=1).max()) or 0.1
        om = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(v),
            o3d.utility.Vector3iVector(np.asarray(mesh.faces)))
        om.compute_vertex_normals()
        r = o3d.visualization.rendering.OffscreenRenderer(320, 320)
        r.scene.set_background([1, 1, 1, 1])
        mat = o3d.visualization.rendering.MaterialRecord(); mat.shader = "defaultLit"
        r.scene.add_geometry("m", om, mat)
        imgs = []
        for eye in [(0, -rad*3, rad*1.6), (rad*3, -rad*2, rad*1.6)]:
            r.setup_camera(60.0, [0, 0, 0], list(eye), [0, 0, 1])
            imgs.append(np.asarray(r.render_to_image()))
        del r
        import cv2; cv2.imwrite(path, cv2.cvtColor(np.hstack(imgs), cv2.COLOR_RGB2BGR))
        print("wrote", path)

    render(build_primitive("ellipsoid", (0.055, 0.055, 0.050), "green lemon"), "/tmp/prim_lemon.png")
    render(build_primitive("dish", (0.148, 0.148, 0.05), "yellow bowl"), "/tmp/prim_bowl.png")
    render(build_primitive("dish", (0.22, 0.22, 0.03), "yellow plate"), "/tmp/prim_plate.png")
