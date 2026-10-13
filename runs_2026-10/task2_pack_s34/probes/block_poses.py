"""block body poses over a recording (bodies after the 28 arm bodies)"""
import sys, numpy as np
from pxr import Usd, UsdGeom
st = Usd.Stage.Open(sys.argv[1]); fps = st.GetTimeCodesPerSecond(); end = int(st.GetEndTimeCode())
bodies = [p for p in st.GetPrimAtPath("/World/robot").GetChildren() if int(p.GetName().split("_")[1]) >= 28]
for t in [0, 1, 2, 4, 6, end / fps - 0.1]:
    tc = int(round(t * fps)); row = []
    for b in bodies:
        m = np.array(UsdGeom.Xformable(b).GetLocalTransformation(Usd.TimeCode(min(tc, end)))).T
        row.append(f"{b.GetName()} ({m[0,3]:.1f},{m[1,3]:.1f},{m[2,3]:.1f})")
    print(f"t={t:5.1f}s  " + "  ".join(row))
