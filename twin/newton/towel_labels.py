"""Exact grid labels for one or more towels (local particle indices), prefixed by garment name.
Usage: towel_labels.py out.json name:w,h[:spacing] [name:w,h ...]"""
import json, sys
out = {"garment": "towels", "labels": {}, "pairs": []}
for spec in sys.argv[2:]:
    name, wh = spec.split(":")[:2]; sp = float(spec.split(":")[2]) if spec.count(":") >= 2 else 0.5
    w, h = [float(v) for v in wh.split(",")]
    nx, ny = int(round(w / sp)) + 1, int(round(h / sp)) + 1
    at = lambda i, j: int(i * ny + j)
    names = {"near_left_corner": (at(0, 0), "corner"), "near_right_corner": (at(nx - 1, 0), "corner"),
             "far_left_corner": (at(0, ny - 1), "corner"), "far_right_corner": (at(nx - 1, ny - 1), "corner"),
             "near_edge_midpoint": (at(nx // 2, 0), "edge_midpoint"), "far_edge_midpoint": (at(nx // 2, ny - 1), "edge_midpoint"),
             "left_edge_midpoint": (at(0, ny // 2), "edge_midpoint"), "right_edge_midpoint": (at(nx - 1, ny // 2), "edge_midpoint"),
             "centre": (at(nx // 2, ny // 2), "region_center"),
             # quarter points of the near/far edges: after a GATE fold (both sides to the centre)
             # these are the ends of the folded edge, where two hands take it without crossing
             "near_left_quarter": (at(nx // 4, 0), "edge_point"), "near_right_quarter": (at(3 * nx // 4, 0), "edge_point"),
             "far_left_quarter": (at(nx // 4, ny - 1), "edge_point"), "far_right_quarter": (at(3 * nx // 4, ny - 1), "edge_point")}
    for nm, (pi, kd) in names.items():
        out["labels"][f"{name}/{nm}"] = {"kind": kd, "particle": pi, "snap_cm": 0.0, "garment": name}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(f"wrote {sys.argv[1]}: {len(out['labels'])} labels")
