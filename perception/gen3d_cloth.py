"""Image-to-3D garment reconstruction via Meshy (or Rodin) for the cloth real2sim path.

SAM3D's weak link for garments is TOPOLOGY (sleeveless solid blobs). This sends the
masked shirt crop from the real capture to a generative image-to-3D service to recover
a complete garment mesh (sleeves, front/back surfaces), then the Newton side rescales
it to the DEPTH-MEASURED size — generation recovers shape, the capture stays the
metric + texture ground truth.

Run (any env with `requests`):
  export MESHY_API_KEY=msy_...        # https://www.meshy.ai -> API keys (free tier ok)
  python gen3d_cloth.py --image .../shirt_crop_masked.png --out_dir .../gen3d_shirt

Then load the GLB in Newton with twin/newton/newton_gen3d_cloth.py.
"""
import argparse, base64, json, os, sys, time

import requests

MESHY_BASE = "https://api.meshy.ai/openapi/v1"


def data_uri(path):
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "png"
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return f"data:image/{ext};base64,{b64}"


def meshy(image_path, out_dir, key, poll_s=10, timeout_s=1200):
    hdr = {"Authorization": f"Bearer {key}"}
    body = {
        "image_url": data_uri(image_path),
        "should_texture": True,
        "enable_pbr": False,
        "should_remesh": True,
        "topology": "triangle",
        "target_polycount": 12000,  # VBD-friendly; service remesh is clean (local quadric decimation of closed garments makes sliver tris -> NaN)
    }
    r = requests.post(f"{MESHY_BASE}/image-to-3d", headers=hdr, json=body, timeout=60)
    r.raise_for_status()
    task_id = r.json()["result"]
    print(f"[gen3d] meshy task {task_id}")
    t0 = time.time()
    while True:
        time.sleep(poll_s)
        r = requests.get(f"{MESHY_BASE}/image-to-3d/{task_id}", headers=hdr, timeout=60)
        r.raise_for_status()
        d = r.json()
        st, prog = d.get("status"), d.get("progress", 0)
        print(f"[gen3d] {st} {prog}%", flush=True)
        if st == "SUCCEEDED":
            break
        if st in ("FAILED", "CANCELED"):
            raise SystemExit(f"meshy task {st}: {d.get('task_error')}")
        if time.time() - t0 > timeout_s:
            raise SystemExit("meshy task timed out")
    os.makedirs(out_dir, exist_ok=True)
    saved = {}
    for kind, url in (d.get("model_urls") or {}).items():
        if kind not in ("glb", "obj") or not url:
            continue
        p = os.path.join(out_dir, f"shirt_gen.{kind}")
        open(p, "wb").write(requests.get(url, timeout=300).content)
        saved[kind] = p
        print(f"[gen3d] saved {p}")
    tex = d.get("texture_urls") or []
    if tex and tex[0].get("base_color"):
        p = os.path.join(out_dir, "shirt_gen_basecolor.png")
        open(p, "wb").write(requests.get(tex[0]["base_color"], timeout=300).content)
        saved["texture"] = p
        print(f"[gen3d] saved {p}")
    json.dump({"provider": "meshy", "task_id": task_id, "image": image_path,
               "files": saved}, open(os.path.join(out_dir, "gen3d_meta.json"), "w"), indent=2)
    return saved


RODIN_BASE = "https://api.hyper3d.ai/api/v2"


def rodin(image_path, out_dir, key, poll_s=10, timeout_s=1200):
    """Deemos Rodin (hyper3d.ai). NOTE: endpoint schema written from docs, unverified
    until the first real call — if it 404s, check https://developer.hyper3d.ai."""
    hdr = {"Authorization": f"Bearer {key}"}
    with open(image_path, "rb") as f:
        r = requests.post(f"{RODIN_BASE}/rodin", headers=hdr,
                          files={"images": (os.path.basename(image_path), f, "image/png")},
                          data={"tier": "Regular", "geometry_file_format": "glb",
                                "quality": "medium", "mesh_mode": "Raw"},
                          timeout=120)
    r.raise_for_status()
    d = r.json()
    task_uuid = d["uuid"]
    sub_key = d["jobs"]["subscription_key"]
    print(f"[gen3d] rodin task {task_uuid}")
    t0 = time.time()
    while True:
        time.sleep(poll_s)
        r = requests.post(f"{RODIN_BASE}/status", headers=hdr,
                          json={"subscription_key": sub_key}, timeout=60)
        r.raise_for_status()
        jobs = r.json().get("jobs", [])
        stats = [j.get("status") for j in jobs]
        print(f"[gen3d] {stats}", flush=True)
        if jobs and all(s == "Done" for s in stats):
            break
        if any(s == "Failed" for s in stats):
            raise SystemExit("rodin job failed")
        if time.time() - t0 > timeout_s:
            raise SystemExit("rodin task timed out")
    r = requests.post(f"{RODIN_BASE}/download", headers=hdr,
                      json={"task_uuid": task_uuid}, timeout=60)
    r.raise_for_status()
    os.makedirs(out_dir, exist_ok=True)
    saved = {}
    for item in r.json().get("list", []):
        name, url = item.get("name", ""), item.get("url")
        if not url:
            continue
        p = os.path.join(out_dir, name)
        open(p, "wb").write(requests.get(url, timeout=300).content)
        saved[name] = p
        print(f"[gen3d] saved {p}")
    json.dump({"provider": "rodin", "task_uuid": task_uuid, "image": image_path,
               "files": saved}, open(os.path.join(out_dir, "gen3d_meta.json"), "w"), indent=2)
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="masked subject crop (white bg)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--provider", default="meshy", choices=["meshy", "rodin"])
    args = ap.parse_args()
    if args.provider == "meshy":
        key = os.environ.get("MESHY_API_KEY")
        if not key:
            sys.exit("MESHY_API_KEY not set — create one at https://www.meshy.ai (free tier)")
        meshy(args.image, args.out_dir, key)
    else:
        key = os.environ.get("RODIN_API_KEY")
        if not key:
            sys.exit("RODIN_API_KEY not set — create one at https://hyper3d.ai")
        rodin(args.image, args.out_dir, key)


if __name__ == "__main__":
    main()
