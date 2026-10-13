"""Per garment: its shape at the top of its carry (drape) and at the end (in the box)."""
import sys, numpy as np
from pxr import Usd, UsdGeom
st = Usd.Stage.Open(sys.argv[1]); end = int(st.GetEndTimeCode())
m = UsdGeom.Mesh(st.GetPrimAtPath('/World/garment'))
G = {"shirt": (0, 6436), "towelA": (6436, 11377), "towelB": (11377, 14758)}
best = {}
for fr in range(0, end, 3):
    q = np.array(m.GetPointsAttr().Get(Usd.TimeCode(fr)))
    for n, (a, b) in G.items():
        p = q[a:b]
        if p[:, 2].max() > 34.0 and (n not in best or p[:, 2].max() > best[n][0]):
            best[n] = (float(p[:, 2].max()), fr, float(np.percentile(p[:, 2], 99) - np.percentile(p[:, 2], 1)), float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1])))
q = np.array(m.GetPointsAttr().Get(Usd.TimeCode(end)))
for n, (zmax, fr, drape, dx, dy) in best.items():
    a, b = G[n]; p = q[a:b]
    print(f"{n}: in the air it hangs {drape:.1f} cm top to bottom (extent {dx:.0f}x{dy:.0f}); at the end {np.ptp(p[:,0]):.0f}x{np.ptp(p[:,1]):.0f} cm")
print("frame_of_carry", {n: v[1] for n, v in best.items()})
