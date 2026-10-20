"""Household props for the twin scenes: a mug and a water bottle.

Each prop is described once here and used three ways:
  parts(spec)       render meshes (lathed profiles + a swept handle) with materials, for the USD recording
  add_shapes(...)   collision shapes and mass on a Newton body
  footprint/grasp   what the planner and the executor need: floor area taken, where the hand grasps

Object frame: origin at the centre of the bounding cylinder dims_cm = [diameter, diameter, height],
z up. The mug's handle points along +x.

What is and is not physical:
  bottle  collides as body cylinder + neck cylinder; the water's mass sits in the filled part, so a
          full bottle's centre of mass is low (harder to tip than a uniform cylinder) and an empty
          one's is near mid-height. The water is rigid: it does not slosh.
  mug     collides as its OUTER cylinder plus a box for the handle. The hollow is drawn and is used
          for the grasp (a pinch on the rim wall), but nothing can fall into the mug in the physics.
"""
import math

import numpy as np

CUP_WALL = 0.5          # mug wall thickness [cm]
NECK_R = 1.35           # bottle neck radius [cm]
BODY_FRAC = 0.60        # bottle: straight body up to this fraction of the height
NECK_FRAC = 0.78        # bottle: shoulder ends / neck starts here
FILL_FRAC = 0.70        # default water level, as a fraction of the height (scene: "fill")
SHELL_KG = 0.03         # the empty plastic bottle


def kind(spec):
    return spec.get("shape", "box")


def handle_len(spec):
    return float(spec.get("handle_cm", 3.0)) if kind(spec) == "cup" else 0.0


def footprint(spec):
    """Floor area the object takes with its handle along +x: (width x, depth y, body-centre x offset
    from the footprint's centre)."""
    w, d = float(spec["dims_cm"][0]), float(spec["dims_cm"][1])
    hl = handle_len(spec)
    return w + hl, d, -hl / 2.0


def grasp(spec):
    """Top grasp in the object frame: (tip xy offset, jaw direction xy, tip depth below the top [cm])."""
    w = float(spec["dims_cm"][0])
    if kind(spec) == "cup":                                     # pinch the rim wall on the side away from the handle
        return (-(w / 2.0 - CUP_WALL / 2.0), 0.0), (1.0, 0.0), 2.0
    if kind(spec) == "bottle":                                  # close on the neck, below the cap
        return (0.0, 0.0), (1.0, 0.0), 3.0
    return (0.0, 0.0), ((0.0, 1.0) if w <= float(spec["dims_cm"][1]) else (1.0, 0.0)), 2.5


def grasp_jaws(spec):
    """Jaw directions the top grasp may use, preferred first: a square block or the round bottle neck
    either way; the mug across its rim wall only; a long block across its narrow side only."""
    off, jaw, depth = grasp(spec)
    w, d = float(spec["dims_cm"][0]), float(spec["dims_cm"][1])
    if kind(spec) == "bottle" or (kind(spec) == "box" and abs(w - d) < 1.5):
        return [tuple(jaw), (-jaw[1], jaw[0])]
    return [tuple(jaw)]


def bottle_segments(spec):
    """The bottle as stacked cylinders (z_lo, z_hi, radius, mass) from the base: the filled part of
    the body (water + its share of the plastic), the empty part of the body, the neck with the cap.
    `fill` is the water level as a fraction of the height; mass_kg is the total."""
    w, _, h = [float(v) for v in spec["dims_cm"]]
    m, r, hb = float(spec["mass_kg"]), w / 2.0, BODY_FRAC * h
    hf = min(max(float(spec.get("fill", FILL_FRAC)), 0.05) * h, hb - 0.5)   # the collider's water level stays in the straight body
    shell = min(SHELL_KG, 0.5 * m)
    return [(0.0, hf, r, (m - shell) + 0.75 * shell * hf / hb), (hf, hb, r, 0.75 * shell * (hb - hf) / hb), (hb, h, NECK_R + 0.2, 0.25 * shell)]


def com_height(spec):
    """Height of the centre of mass above the base [cm] (for the planner's tipping estimate)."""
    if kind(spec) == "bottle":
        seg = bottle_segments(spec)
        return sum(0.5 * (a + c) * mm for a, c, _, mm in seg) / sum(mm for *_, mm in seg)
    return 0.5 * float(spec["dims_cm"][2])


# ---- physics -----------------------------------------------------------------------------
def add_shapes(b, body, spec, mu, ke=1.0e5):
    import warp as wp
    from newton import ModelBuilder
    w, d, h = [float(v) for v in spec["dims_cm"]]
    m, r = float(spec["mass_kg"]), w / 2.0
    cfg = lambda rho: ModelBuilder.ShapeConfig(density=rho, ke=ke, kd=0.0, mu=mu)
    at = lambda x, z: wp.transform((x, 0.0, z), wp.quat_identity())
    k = kind(spec)
    if k == "bottle":
        for z_lo, z_hi, rad, mm in bottle_segments(spec):
            b.add_shape_cylinder(body, xform=at(0.0, -h / 2 + 0.5 * (z_lo + z_hi)), radius=rad, half_height=0.5 * (z_hi - z_lo),
                                 cfg=cfg(mm / (math.pi * rad * rad * (z_hi - z_lo))))
    elif k == "cup":
        hl = handle_len(spec)
        b.add_shape_cylinder(body, radius=r, half_height=h / 2, cfg=cfg(0.93 * m / (math.pi * r * r * h)))
        if hl > 0:
            hx, hy, hz = hl / 2, 0.55, 0.30 * h
            b.add_shape_box(body, xform=at(r + hx - 0.2, 0.0), hx=hx, hy=hy, hz=hz, cfg=cfg(0.07 * m / (8 * hx * hy * hz)))
    elif k == "cylinder":
        b.add_shape_cylinder(body, radius=r, half_height=h / 2, cfg=cfg(m / (math.pi * r * r * h)))
    else:
        b.add_shape_box(body, hx=w / 2, hy=d / 2, hz=h / 2, cfg=cfg(m / (w * d * h)))


# ---- render meshes -----------------------------------------------------------------------
def lathe(profile, n=40):
    """Surface of revolution about z of a polyline of (radius, z). Returns points, face counts, indices."""
    pts, cnt, idx = [], [], []
    rings = []
    for r, z in profile:
        if r < 1e-6:
            rings.append([len(pts)]); pts.append((0.0, 0.0, z))
        else:
            rings.append(list(range(len(pts), len(pts) + n)))
            pts += [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), z) for k in range(n)]
    for a, c in zip(rings[:-1], rings[1:]):
        for k in range(n):
            k1 = (k + 1) % n
            if len(a) == 1 and len(c) == 1:
                continue
            if len(a) == 1:
                cnt.append(3); idx += [a[0], c[k1], c[k]]
            elif len(c) == 1:
                cnt.append(3); idx += [a[k], a[k1], c[0]]
            else:
                cnt.append(4); idx += [a[k], a[k1], c[k1], c[k]]
    return np.array(pts, float), cnt, idx


def sweep(path, rad, n=12):
    """Tube of radius `rad` along an open polyline lying in the x-z plane."""
    path = np.asarray(path, float)
    pts, cnt, idx = [], [], []
    for i, p in enumerate(path):
        t = path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)]
        t /= np.linalg.norm(t)
        nrm = np.array([-t[2], 0.0, t[0]])                      # in-plane normal
        for k in range(n):
            a = 2 * math.pi * k / n
            pts.append(p + rad * (math.cos(a) * nrm + math.sin(a) * np.array([0.0, 1.0, 0.0])))
    for i in range(len(path) - 1):
        for k in range(n):
            k1 = (k + 1) % n
            cnt.append(4); idx += [i * n + k, i * n + k1, (i + 1) * n + k1, (i + 1) * n + k]
    return np.array(pts, float), cnt, idx


def _arc(c, r0, r1, a0, a1, n=8):
    return [(c[0] + r0 * math.cos(a0 + (a1 - a0) * i / n), c[1] + r1 * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]


def parts(spec, color):
    """Render parts in the object frame: list of dict(name, mesh=(pts, counts, idx), color, opacity, rough)."""
    w, d, h = [float(v) for v in spec["dims_cm"]]
    r, z0, z1 = w / 2.0, -h / 2.0, h / 2.0
    k = kind(spec)
    if k == "cup":
        t, f = CUP_WALL, 0.7                                    # wall, foot fillet
        prof = ([(0.0, z0), (r - f, z0)] + _arc((r - f, z0 + f), f, f, -math.pi / 2, 0.0, 6)[1:] + [(r, z1 - t / 2)]
                + _arc((r - t / 2, z1 - t / 2), t / 2, t / 2, 0.0, math.pi, 6)[1:]                      # rounded rim
                + [(r - t, z0 + t + f)] + [(r - t - f + f * math.cos(a), z0 + t + f - f * math.sin(a)) for a in np.linspace(0, math.pi / 2, 6)[1:]]
                + [(0.0, z0 + t)])
        out = [dict(name="body", mesh=lathe(prof), color=color, opacity=1.0, rough=0.25)]
        hl = handle_len(spec)
        if hl > 0:
            hz = 0.30 * h
            path = [(r - 0.25 + (hl - 0.3) * math.sin(a), 0.0, 0.3 + hz * math.cos(a)) for a in np.linspace(0.0, math.pi, 22)]
            out.append(dict(name="handle", mesh=sweep(path, 0.55), color=color, opacity=1.0, rough=0.25))
        return out
    if k == "bottle":
        # Drawn opaque: the lower shell is tinted as the water seen through clear plastic, the upper
        # shell as empty plastic (translucent materials did not render reliably in the viewer).
        zb, zn, zc = z0 + BODY_FRAC * h, z0 + NECK_FRAC * h, z1 - 1.9
        zw = z0 + min(max(float(spec.get("fill", FILL_FRAC)), 0.05), NECK_FRAC - 0.02) * h
        sh = [(r + (NECK_R - r) * 0.5 * (1 - math.cos(math.pi * i / 10)), zb + (zn - zb) * i / 10) for i in range(11)]
        rw = float(np.interp(zw, [p[1] for p in sh], [p[0] for p in sh])) if zw > zb else r
        low = [(0.0, z0 + 0.5), (0.45 * r, z0 + 0.5), (0.7 * r, z0)] + _arc((r - 0.6, z0 + 0.6), 0.6, 0.6, -math.pi / 2, 0.0, 5) \
            + ([(r, zb)] + [p for p in sh[1:] if p[1] < zw - 0.05] if zw > zb else []) + [(rw, zw)]
        top = [(rw, zw)] + ([(r, zb)] if zw < zb else []) + [p for p in sh if p[1] > max(zw, zb) + 0.05] \
            + [(NECK_R, zc), (NECK_R + 0.25, zc), (NECK_R + 0.25, zc + 0.3), (NECK_R, zc + 0.3), (NECK_R, z1 - 0.1), (0.0, z1 - 0.1)]
        label = [(r + 0.04, z0 + 0.22 * h), (r + 0.04, z0 + 0.50 * h)]
        cap = [(0.0, z1), (NECK_R + 0.3, z1), (NECK_R + 0.35, z1 - 0.15), (NECK_R + 0.35, zc + 0.45), (NECK_R, zc + 0.45)]
        return [dict(name="water", mesh=lathe(low), color=(0.30, 0.60, 0.88), opacity=1.0, rough=0.08),
                dict(name="shell", mesh=lathe(top), color=(0.88, 0.93, 0.96), opacity=1.0, rough=0.08),
                dict(name="label", mesh=lathe(label), color=color, opacity=1.0, rough=0.6),
                dict(name="cap", mesh=lathe(cap), color=(0.15, 0.35, 0.75), opacity=1.0, rough=0.4)]
    return []


def define(stage, path, spec, color):
    """Author the prop under `path` in a USD stage. Returns the transform op to time-sample, or None
    if this shape has no prop mesh (boxes, plain cylinders)."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    ps = parts(spec, color)
    if not ps:
        return None
    root = UsdGeom.Xform.Define(stage, path)
    for p in ps:
        pts, cnt, idx = p["mesh"]
        m = UsdGeom.Mesh.Define(stage, f"{path}/{p['name']}")
        m.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in pts])
        m.CreateFaceVertexCountsAttr(cnt); m.CreateFaceVertexIndicesAttr(idx)
        m.CreateDoubleSidedAttr(True)
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.catmullClark if p["name"] in ("body", "handle") else UsdGeom.Tokens.none)
        m.CreateDisplayColorAttr([Gf.Vec3f(*p["color"])])
        mat = UsdShade.Material.Define(stage, f"{path}/mat_{p['name']}")
        sh = UsdShade.Shader.Define(stage, f"{path}/mat_{p['name']}/s")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*p["color"]))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(p["rough"]))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        if p["opacity"] < 1.0:
            sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(p["opacity"]))
            sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.45)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return root.AddTransformOp()
