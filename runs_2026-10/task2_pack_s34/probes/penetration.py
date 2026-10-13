"""True inter-garment penetration at the end of a recording: particles of one garment lying BELOW the
top surface of another garment, counted only well inside the other's footprint (2 cm from its edge),
so cloth draping down the side of a pile is not counted."""
import sys, numpy as np
from pxr import Usd, UsdGeom
st = Usd.Stage.Open(sys.argv[1]); end = int(st.GetEndTimeCode())
m = UsdGeom.Mesh(st.GetPrimAtPath('/World/garment'))
G = {"shirt": (0, 6436), "towelA": (6436, 11377), "towelB": (11377, 14758)}
q = np.array(m.GetPointsAttr().Get(Usd.TimeCode(end)))
def surf(p):
    d = {}
    for x, y, z in p:
        k = (int(np.floor(x)), int(np.floor(y))); lo, hi = d.get(k, (9e9, -9e9)); d[k] = (min(lo, z), max(hi, z))
    inner = {k: v for k, v in d.items() if all((k[0] + i, k[1] + j) in d for i in (-2, 0, 2) for j in (-2, 0, 2))}
    return inner
S = {n: surf(q[a:b]) for n, (a, b) in G.items()}
names = list(G)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        pa, pb = q[G[a][0]:G[a][1]], q[G[b][0]:G[b][1]]
        def under(p, s):   # particles of p inside s's inner footprint; how many are below s's top / strictly between s's bottom and top
            n = mid = 0
            for x, y, z in p:
                k = (int(np.floor(x)), int(np.floor(y)))
                if k in s:
                    n += 1; lo, hi = s[k]
                    if lo + 0.4 < z < hi - 0.4: mid += 1
            return n, mid
        na, ma = under(pa, S[b]); nb, mb = under(pb, S[a])
        tot = na + nb
        print(f"{a} / {b}: {tot} particles over the other's interior; {ma + mb} sit INSIDE the other's thickness ({(ma + mb) / tot:.0%})" if tot else f"{a} / {b}: not stacked")
