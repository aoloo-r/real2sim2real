"""Robot-link penetration into the Task-1 box walls over a recording: which link, how deep, when."""
import sys, numpy as np
from pxr import Usd, UsdGeom, Gf
st = Usd.Stage.Open(sys.argv[1]); fps = st.GetTimeCodesPerSecond(); end = int(st.GetEndTimeCode())
# static boxes = walls/floor: cube prims under /World/static with transform + scale
walls = []
for p in st.GetPrimAtPath("/World/static").GetChildren():
    xf = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    m = np.array(xf).T   # row-major 4x4 -> column convention
    c = m[:3, 3]; half = np.array([np.linalg.norm(m[:3, i]) for i in range(3)]) / 2.0
    walls.append((p.GetName(), c, half))
print("boxes:", [(n, c.round(0).tolist(), (2 * h).round(1).tolist()) for n, c, h in walls])
links = []
for b in st.GetPrimAtPath("/World/robot").GetChildren():
    for m in b.GetChildren():
        if m.IsA(UsdGeom.Mesh) and m.GetName().startswith("shape_"):
            pts = np.array(UsdGeom.Mesh(m).GetPointsAttr().Get()); pts = pts[::max(1, len(pts) // 400)]
            lx = np.array(UsdGeom.Xformable(m).GetLocalTransformation(Usd.TimeCode.Default())).T
            links.append((b.GetName() + "/" + m.GetName(), b, lx, np.c_[pts, np.ones(len(pts))]))
worst = {}
t0, t1, dt = (float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0.0, end / fps, 1.0)
for tc in [int(round(t * fps)) for t in np.arange(t0, t1, dt)]:
    for name, b, lx, pts in links:
        bx = np.array(UsdGeom.Xformable(b).GetLocalTransformation(Usd.TimeCode(tc))).T
        w = (bx @ lx @ pts.T).T[:, :3]
        for wn, c, half in walls:
            if wn.endswith("box_0") and False: continue
            d = half - np.abs(w - c)              # positive in all 3 axes = inside
            inside = d.min(1) > 0
            if inside.any():
                depth = float(inside.mean())            # FRACTION of the link's sample points inside the wall
                key = (name, wn)
                if depth > worst.get(key, (0, 0))[0]:
                    worst[key] = (depth, tc / fps)
rows = sorted(worst.items(), key=lambda kv: -kv[1][0])
for (name, wn), (depth, t) in rows[:12]:
    print(f"{name:28s} in {wn:8s} {depth:4.0%} of the link inside  at t={t:5.1f}s")
print("none" if not rows else f"{len(rows)} link/box pairs penetrate")
