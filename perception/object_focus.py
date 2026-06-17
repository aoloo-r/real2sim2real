# -*- coding: utf-8 -*-
"""Select which detected objects to reconstruct.

Instead of meshing every manipulable object Gemini finds, keep only the ones in
the robot's *current, central, reachable* view and drop objects that are:
  - clipped against / near the image edge (peripheral, partially out of frame),
  - too far from the camera (beyond max_depth_m),
  - out of the arm's reach (when a calibrated camera->robot extrinsic is given).
Finally cap to max_objects, preferring the most reachable / central ones.

This focuses all the reconstruction + QA effort on the few objects that matter.
"""

from __future__ import annotations

import numpy as np


def _reach_xy(T_base_cam, pos_cam):
    """Planar distance (m) from the robot base to the object, using the
    calibrated extrinsic. Returns None if no extrinsic / position."""
    if T_base_cam is None or pos_cam is None:
        return None
    T = np.asarray(T_base_cam, dtype=float)
    p = T @ np.array([pos_cam[0], pos_cam[1], pos_cam[2], 1.0])
    return float(np.hypot(p[0], p[1])), (float(p[0]), float(p[1]))


def select_target_objects(boxes, positions_cam, image_size, T_base_cam=None,
                          edge_margin_frac=0.04, max_depth_m=1.2,
                          max_reach_m=0.85, max_objects=4, workspace=None,
                          verbose=True):
    """Return the indices of objects to keep, in original order.

    Args:
        boxes: list of dicts with "label" and "box_px"=[x0,y0,x1,y1].
        positions_cam: list (parallel to boxes) of [x,y,z] in camera frame or None.
        image_size: (W, H) in pixels.
        T_base_cam: 4x4 list/array mapping camera-frame points to the robot base
            frame (p_base = T_base_cam @ p_cam), or None to skip reach checks.
        edge_margin_frac: drop boxes whose extent lies within this fraction of any
            image border.
        max_depth_m: drop objects whose camera-frame z exceeds this.
        max_reach_m: drop objects whose planar base distance exceeds this.
        max_objects: cap kept objects to this many (closest/most-central first).
        workspace: optional (x0, x1, y0, y1) base-frame box; drop objects outside.
        verbose: print a keep/drop line per object with the reason.
    """
    W, H = image_size
    mx, my = edge_margin_frac * W, edge_margin_frac * H

    cand = []
    for i, b in enumerate(boxes):
        x0, y0, x1, y1 = b["box_px"]
        reasons = []
        # Edge test is CENTER-based: only drop objects whose center is near the
        # border (i.e. mostly out of frame). An object that merely touches the
        # edge but is largely visible (e.g. a bowl with its front lip clipped)
        # is kept.
        cxb0, cyb0 = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        if cxb0 < mx or cyb0 < my or cxb0 > (W - mx) or cyb0 > (H - my):
            reasons.append("edge")

        pos = positions_cam[i] if i < len(positions_cam) else None
        if pos is not None and pos[2] is not None and pos[2] > max_depth_m:
            reasons.append("far(z=%.2f)" % pos[2])

        reach_dist = None
        rr = _reach_xy(T_base_cam, pos)
        if rr is not None:
            reach_dist, (px, py) = rr
            if reach_dist > max_reach_m:
                reasons.append("unreach(d=%.2f)" % reach_dist)
            if workspace is not None:
                wx0, wx1, wy0, wy1 = workspace
                if not (wx0 <= px <= wx1 and wy0 <= py <= wy1):
                    reasons.append("outside_ws")

        cxb, cyb = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        central = ((cxb - W / 2.0) / W) ** 2 + ((cyb - H / 2.0) / H) ** 2
        # rank key: prefer closest reach; fall back to image centrality
        rank = reach_dist if reach_dist is not None else central
        cand.append({"i": i, "label": b.get("label", "obj%d" % i),
                     "keep": not reasons, "reasons": reasons,
                     "central": central, "rank": rank})

    eligible = sorted([c for c in cand if c["keep"]], key=lambda c: c["rank"])
    capped = set(c["i"] for c in eligible[:max_objects])

    if verbose:
        for c in sorted(cand, key=lambda c: c["i"]):
            if c["i"] in capped:
                print("  [FOCUS] keep  %-18s (rank=%.3f central=%.3f)"
                      % (c["label"], c["rank"], c["central"]))
            else:
                why = c["reasons"] if c["reasons"] else ["capped(>%d)" % max_objects]
                print("  [FOCUS] drop  %-18s (%s)" % (c["label"], ",".join(why)))

    return sorted(capped)
