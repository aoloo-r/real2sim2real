"""Decode captured Meshy viewer GPU buffers (meshy_shirt_buffers.bin, written by the
in-page WebGL bufferData hook) into a standard textured GLB.

Free-tier Meshy gates file export, but the web viewer necessarily holds the decoded
geometry — this rebuilds the mesh from the captured vertex buffers + the full-res
texture_0.png fetched from the signed CDN URL. Buffer layout observed (textured task):
  ARRAY  Int8    x4/vert  -> normals (snorm8, ignored — recomputed)
  ARRAY  Uint16  x8/vert  -> positions (half-float or unorm16-quantized; auto-detected)
  ARRAY  Int8    x4/vert  -> tangents (ignored)
  ARRAY  Uint16  x4/vert  -> UV (half-float or unorm16; auto-detected)
  ELEM   Uint32           -> triangle indices

Run (newton-spike env):
  python meshy_buffers_to_glb.py --buffers ~/Downloads/meshy_shirt_buffers.bin \
      --texture .../meshy_dl/texture_0.png --out .../gen3d_shirt/shirt_gen.glb
"""
import argparse, json, struct

import numpy as np


def load_buffers(path):
    raw = open(path, "rb").read()
    n = struct.unpack("<I", raw[:4])[0]
    manifest = json.loads(raw[4:4 + n].decode())
    bufs, off = [], 4 + n
    for m in manifest:
        bufs.append((m, raw[off:off + m["bytes"]]))
        off += m["bytes"]
    return bufs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffers", required=True)
    ap.add_argument("--texture", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bufs = load_buffers(args.buffers)
    idx = pos = uv = None
    for m, b in bufs:
        if m["target"] == 34963:                       # ELEMENT_ARRAY_BUFFER
            idx = np.frombuffer(b, np.uint32).reshape(-1, 3)
    nvert = None
    if idx is not None:
        nvert = int(idx.max()) + 1
    for m, b in bufs:
        if m["target"] != 34962 or m["kind"] != "Uint16Array":
            continue
        u16 = np.frombuffer(b, np.uint16)
        stride = len(u16) // nvert
        if stride == 4:                                 # positions (x,y,z,pad)
            h = u16.reshape(nvert, 4)[:, :3]
            # CONFIRMED encoding (vs the untextured task's raw float32 in [-1,1]^3):
            # meshopt-style 14-bit quantization over the per-axis-normalized unit cube.
            # NOTE per-axis normalization means the TRUE aspect (esp. thickness) is lost —
            # the importer must re-calibrate xy to measured size and clamp thickness.
            pos = h.astype(np.float32) / 16383.0 * 2.0 - 1.0
        elif stride == 2:                               # uv
            h = u16.reshape(nvert, 2)
            f16 = np.frombuffer(h.tobytes(), np.float16).reshape(nvert, 2).astype(np.float32)
            un16 = h.astype(np.float32) / 65535.0
            # f16 reinterpretation of unorm data collapses to ~0 — require a sane spread
            if np.isfinite(f16).all() and -0.1 <= f16.min() and f16.max() <= 1.1 and f16.max() > 0.5:
                uv = f16
            elif h.max() <= 4100:                       # observed: 12-bit quantized UV
                uv = h.astype(np.float32) / 4095.0
            else:
                uv = un16
    if pos is None or idx is None:
        raise SystemExit("positions or indices not found in capture")
    print(f"[decode] {nvert} verts, {len(idx)} tris, pos bbox {pos.min(0)} .. {pos.max(0)}, "
          f"uv range {None if uv is None else (uv.min(), uv.max())}")
    import trimesh
    from PIL import Image
    tex = Image.open(args.texture).convert("RGB")
    visual = None
    if uv is not None:
        visual = trimesh.visual.TextureVisuals(uv=np.column_stack([uv[:, 0], 1.0 - uv[:, 1]]),
                                               image=tex)     # glTF v-down -> obj v-up
    tm = trimesh.Trimesh(vertices=pos.astype(np.float64), faces=idx.astype(np.int64),
                         visual=visual, process=False)
    tm.export(args.out)
    print(f"[decode] wrote {args.out}")


if __name__ == "__main__":
    main()
