"""Minimum distance between the two arms' links over a recording (sampled link-mesh points).
Reports the worst moments; < 0 is impossible to measure with points, so 'closest points' < 1.5 cm = interpenetration."""
import sys, numpy as np
from pxr import Usd, UsdGeom
st = Usd.Stage.Open(sys.argv[1]); fps = st.GetTimeCodesPerSecond(); end = int(st.GetEndTimeCode())
dt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
links = []
for b in st.GetPrimAtPath("/World/robot").GetChildren():
    bi = int(b.GetName().split("_")[1])
    for m in b.GetChildren():
        if m.IsA(UsdGeom.Mesh) and m.GetName().startswith("shape_"):
            pts = np.array(UsdGeom.Mesh(m).GetPointsAttr().Get()); pts = pts[::max(1, len(pts) // 150)]
            lx = np.array(UsdGeom.Xformable(m).GetLocalTransformation(Usd.TimeCode.Default())).T
            links.append((bi, b, lx, np.c_[pts, np.ones(len(pts))]))
bodies = sorted({l[0] for l in links})
# the two arms = the two halves of the body list (each UR5e+2F-85 is 14 bodies, in order)
# each UR5e + 2F-85 is 14 bodies, in order; anything after body 27 is a block
armA, armB = set(bodies[:14]), set(bodies[14:28])
worst = []
for tc in [int(round(t * fps)) for t in np.arange(0, end / fps, dt)]:
    P = {}
    for bi, b, lx, pts in links:
        bx = np.array(UsdGeom.Xformable(b).GetLocalTransformation(Usd.TimeCode(tc))).T
        P.setdefault(bi, []).append((bx @ lx @ pts.T).T[:, :3])
    A = np.concatenate([np.concatenate(P[b]) for b in armA if b in P]); B = np.concatenate([np.concatenate(P[b]) for b in armB if b in P])
    # skip the bases (first 2 bodies of each arm) - they never move
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min()
    worst.append((float(d), tc / fps))
    if abs((tc / fps) % 4.0) < 1e-6:
        # which pair
        best = (1e9, None)
        for a in armA:
            for c in armB:
                if a in P and c in P:
                    dd = np.sqrt(((np.concatenate(P[a])[:, None, :] - np.concatenate(P[c])[None, :, :]) ** 2).sum(-1)).min()
                    if dd < best[0]: best = (dd, (a, c))
        print(f"t={tc / fps:5.1f}s  min {d:5.1f} cm  pair {best[1]}", flush=True)
worst.sort()
print("closest the two ARMS came (any link to any link):")
for d, t in worst[:6]:
    print(f"  {d:5.1f} cm at t={t:6.1f}s")
n_bad = sum(1 for d, _ in worst if d < 1.5)
print(f"{n_bad} of {len(worst)} samples closer than 1.5 cm (interpenetrating)")
