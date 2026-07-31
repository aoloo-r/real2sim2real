"""LOCAL image-to-3D garment generation via Hunyuan3D-2 (open weights, runs on the 4090)
— the zero-cost alternative to Meshy/Rodin in the gen3d cloth experiment. Same output
convention as gen3d_cloth.py: <out_dir>/shirt_gen.glb + gen3d_meta.json.

Run (hunyuan3d env):
  /home/aoloo/miniforge3/envs/hunyuan3d/bin/python gen3d_local_hunyuan.py \
      --image .../shirt_crop_masked.png --out_dir .../gen3d_shirt_hunyuan
Texture stage (Hunyuan3D-Paint) is attempted and skipped on failure — for the fold
experiment topology is the point; colours can come from the capture.
"""
import argparse, json, os, sys

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--no_texture", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from hy3dgen.rembg import BackgroundRemover
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    image = Image.open(args.image).convert("RGB")
    image = BackgroundRemover()(image)              # -> RGBA, subject isolated

    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
    mesh = pipe(image=image)[0]
    print(f"[gen3d-local] shape: {len(mesh.vertices)} verts, {len(mesh.faces)} faces", flush=True)
    # Hunyuan's OWN postprocessors — raw marching-cubes output has floaters + degenerate
    # faces that NaN cloth solvers; their FaceReducer gives a cleaner low-poly than
    # generic quadric decimation
    try:
        from hy3dgen.shapegen import FloaterRemover, DegenerateFaceRemover, FaceReducer
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=20000)
        print(f"[gen3d-local] postprocessed: {len(mesh.vertices)} verts, {len(mesh.faces)} faces", flush=True)
    except Exception as e:
        print(f"[gen3d-local] postprocess skipped: {e}", flush=True)

    textured = False
    if not args.no_texture:
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            mesh = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")(mesh, image=image)
            textured = True
        except Exception as e:
            print(f"[gen3d-local] texture stage skipped: {e}", flush=True)

    out = os.path.join(args.out_dir, "shirt_gen.glb")
    mesh.export(out)
    json.dump({"provider": "hunyuan3d-2-local", "image": args.image, "textured": textured,
               "files": {"glb": out}},
              open(os.path.join(args.out_dir, "gen3d_meta.json"), "w"), indent=2)
    print(f"[gen3d-local] saved {out} (textured={textured})", flush=True)


if __name__ == "__main__":
    main()
