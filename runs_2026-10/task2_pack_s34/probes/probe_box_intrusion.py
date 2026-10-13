"""Per arm: which links enter the box interior (inside the walls, below the rim), when, and how often both arms are in at once.
Also wall hits: link points inside the wall slabs."""
import sys, numpy as np
from pxr import Usd, UsdGeom
st = Usd.Stage.Open(sys.argv[1]); fps = st.GetTimeCodesPerSecond(); end = int(st.GetEndTimeCode())
walls = []
for p in st.GetPrimAtPath("/World/static").GetChildren():
    m = np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    c = m[:3, 3]; half = np.array([np.linalg.norm(m[:3, i]) for i in range(3)]) / 2.0
    walls.append((p.GetName(), c, half))
floor = max(walls, key=lambda w: w[2][0] * w[2][1]); side = [w for w in walls if w is not floor]
cx, cy = floor[1][0], floor[1][1]; hin = floor[2][0] - 1.0; rim = max(w[1][2] + w[2][2] for w in side); zf = floor[1][2] + floor[2][2]
print(f"box centre ({cx:.0f},{cy:.0f}) inner half {hin:.1f} floor z {zf:.1f} rim z {rim:.1f}")
links = []
for b in st.GetPrimAtPath("/World/robot").GetChildren():
    bi = int(b.GetName().split("_")[1])
    if bi >= 28: continue
    for m in b.GetChildren():
        if m.IsA(UsdGeom.Mesh) and m.GetName().startswith("shape_"):
            pts = np.array(UsdGeom.Mesh(m).GetPointsAttr().Get()); pts = pts[::max(1, len(pts) // 200)]
            lx = np.array(UsdGeom.Xformable(m).GetLocalTransformation(Usd.TimeCode.Default())).T
            links.append((bi, b, lx, np.c_[pts, np.ones(len(pts))]))
rows = []; wall_hits = {}
for tc in range(0, end, int(fps // 2) or 1):
    inside = {0: set(), 1: set()}
    for bi, b, lx, pts in links:
        bx = np.array(UsdGeom.Xformable(b).GetLocalTransformation(Usd.TimeCode(tc))).T
        w = (bx @ lx @ pts.T).T[:, :3]
        arm = 0 if bi < 14 else 1
        if ((np.abs(w[:, 0] - cx) < hin) & (np.abs(w[:, 1] - cy) < hin) & (w[:, 2] < rim) & (w[:, 2] > zf)).any():
            inside[arm].add(bi - 14 * arm)
        for wn, c, half in side:
            d = half - np.abs(w - c)
            f = float((d.min(1) > 0).mean())
            if f > 0.05: wall_hits[(arm, bi - 14 * arm)] = max(wall_hits.get((arm, bi - 14 * arm), 0), f)
    rows.append((tc / fps, inside[0], inside[1]))
n = len(rows)
for arm, name in ((0, "right"), (1, "left")):
    t_in = [r for r in rows if r[1 + arm]]
    arm_links = [r for r in t_in if any(l <= 5 for l in r[1 + arm])]
    print(f"{name} arm: something inside the box in {len(t_in)}/{n} samples; ARM links (not just the gripper) inside in {len(arm_links)}/{n}")
both = [r for r in rows if r[1] and r[2]]
print(f"both arms inside the box at once: {len(both)}/{n} samples" + (f" (t={both[0][0]:.0f}..{both[-1][0]:.0f}s rec)" if both else ""))
print("deepest wall hits (fraction of a link inside a wall):", sorted(((f"{'RL'[a]}{l}", round(f, 2)) for (a, l), f in wall_hits.items()), key=lambda x: -x[1])[:6])
