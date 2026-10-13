"""Hide the box wall nearest the viewer (-y) in a recorded USD so the front of the box is visible."""
import sys, numpy as np
from pxr import Usd, UsdGeom
st = Usd.Stage.Open(sys.argv[1])
ws = []
for p in st.GetPrimAtPath("/World/static").GetChildren():
    m = np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    ws.append((float(m[1, 3]), p))
y, p = min(ws, key=lambda t: t[0])
UsdGeom.Imageable(p).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
st.GetRootLayer().Save(); print(f"hid {p.GetName()} (wall at y={y:.0f})")
