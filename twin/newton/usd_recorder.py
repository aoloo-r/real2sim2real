"""Record a twin run (robot + cloth) as a time-sampled USD for Isaac Sim playback.

The twin already computes every body pose and cloth particle each frame; this just writes them
down. Robot links become Xforms with time-sampled world transforms, each carrying its visual
mesh as a static child (local shape offset baked into the child prim). The garment is one mesh
with time-sampled points. Open the result in Isaac Sim and press play - Newton physics,
RTX pictures.

Used by twin_stockfold when TWIN_REC_USD=<out.usd> is set; TWIN_REC_EVERY (default 4) thins
the 60fps sim to a 15fps animation.
"""

from __future__ import annotations

import numpy as np


class UsdRecorder:
    def __init__(self, model, out_path: str, every: int = 4, fps: float = 60.0,
                 table=None, vertex_colors=None, uvs=None, texture_path=None):
        import newton
        from pxr import Gf, Usd, UsdGeom, UsdLux

        self._Gf, self._Usd, self._UsdGeom = Gf, Usd, UsdGeom
        self.every = max(1, int(every))
        self.out = out_path
        self.frame = 0
        self.t = 0

        stage = Usd.Stage.CreateNew(out_path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)               # the twin runs in centimetres
        stage.SetTimeCodesPerSecond(fps / self.every)
        stage.SetStartTimeCode(0)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        self.stage = stage

        # --- garment: one deforming mesh -------------------------------------------------
        tris = model.tri_indices.numpy().reshape(-1, 3)
        g = UsdGeom.Mesh.Define(stage, "/World/garment")
        g.CreateFaceVertexIndicesAttr(tris.flatten().tolist())
        g.CreateFaceVertexCountsAttr([3] * len(tris))
        g.CreateDoubleSidedAttr(True)
        g.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        if vertex_colors is not None and len(vertex_colors) == int(model.particle_count):
            # per-garment colours from the reconstruction (a purple shirt next to a red one)
            pv = g.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
            pv.Set([Gf.Vec3f(*map(float, c)) for c in np.asarray(vertex_colors, float)])
        else:
            g.CreateDisplayColorAttr([Gf.Vec3f(0.83, 0.90, 0.28)])
        if uvs is not None and texture_path is not None and len(uvs) == int(model.particle_count):
            # a textured surface: st per vertex + a UsdPreviewSurface reading the texture. The
            # display colour above stays as the fallback for viewers without materials.
            from pxr import Sdf, UsdShade, Vt
            st_pv = UsdGeom.PrimvarsAPI(g).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                         UsdGeom.Tokens.vertex)
            st_pv.Set(Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in np.asarray(uvs, float)]))
            mat = UsdShade.Material.Define(stage, "/World/Looks/garment")
            sh = UsdShade.Shader.Define(stage, "/World/Looks/garment/surface")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.95)
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            tex = UsdShade.Shader.Define(stage, "/World/Looks/garment/tex")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
            tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            rd = UsdShade.Shader.Define(stage, "/World/Looks/garment/stReader")
            rd.CreateIdAttr("UsdPrimvarReader_float2")
            rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
            rd.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            UsdShade.MaterialBindingAPI(g).Bind(mat)
        self.garment_pts = g.GetPointsAttr()
        # VISUAL THICKNESS (TWIN_REC_THICK_CM): the cloth is a zero-thickness surface in USD, so
        # a folded single-layer towel renders as a sheet of paper. A second mesh with the same
        # topology offset along the vertex normals by the simulated contact thickness gives the
        # fabric its volume in the viewer; it changes nothing in the simulation.
        import os as _os
        self.thick = float(_os.environ.get("TWIN_REC_THICK_CM", "0"))
        self.tris = tris
        self.under_pts = None
        if self.thick > 0:
            u = UsdGeom.Mesh.Define(stage, "/World/garment_under")
            u.CreateFaceVertexIndicesAttr(tris[:, ::-1].flatten().tolist())
            u.CreateFaceVertexCountsAttr([3] * len(tris))
            u.CreateDoubleSidedAttr(True)
            u.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            if vertex_colors is not None and len(vertex_colors) == int(model.particle_count):
                u.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set([Gf.Vec3f(*map(float, c)) for c in np.asarray(vertex_colors, float)])
            else:
                u.CreateDisplayColorAttr([Gf.Vec3f(0.83, 0.90, 0.28)])
            self.under_pts = u.GetPointsAttr()
            if uvs is not None and texture_path is not None and len(uvs) == int(model.particle_count):
                from pxr import Sdf, UsdShade, Vt
                UsdGeom.PrimvarsAPI(u).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
                    Vt.Vec2fArray([Gf.Vec2f(float(a), float(b)) for a, b in np.asarray(uvs, float)]))
                UsdShade.MaterialBindingAPI(u).Bind(UsdShade.Material(stage.GetPrimAtPath("/World/Looks/garment")))

        # --- robot links: static meshes under per-body Xforms ----------------------------
        shape_body = model.shape_body.numpy()
        shape_type = model.shape_type.numpy()
        shape_scale = model.shape_scale.numpy()
        shape_xf = model.shape_transform.numpy()
        self.body_ops = {}
        n_link_meshes = 0
        n_boxes = 0
        for s in range(len(shape_body)):
            b = int(shape_body[s])
            if int(shape_type[s]) == int(newton.GeoType.BOX) and s in getattr(model, "_rec_box_shapes", ()):
                # box primitives: the task box walls (static) and the rigid block (its body is animated
                # like a robot link). Newton stores a box's HALF extents in shape_scale.
                if b >= 0 and b not in self.body_ops:
                    x = UsdGeom.Xform.Define(stage, f"/World/robot/body_{b}")
                    self.body_ops[b] = x.AddTransformOp()
                path = f"/World/robot/body_{b}/box_{s}" if b >= 0 else f"/World/static/box_{s}"
                c = UsdGeom.Cube.Define(stage, path)
                c.CreateSizeAttr(1.0)
                c.CreateDisplayColorAttr([Gf.Vec3f(0.82, 0.62, 0.38) if b < 0 else Gf.Vec3f(0.85, 0.30, 0.25)])
                xf = UsdGeom.Xformable(c)
                xf.AddTransformOp().Set(self._mat(shape_xf[s]))
                xf.AddScaleOp().Set(Gf.Vec3f(*[2.0 * float(v) for v in shape_scale[s]]))
                n_boxes += 1
                continue
            if b < 0 or int(shape_type[s]) != int(newton.GeoType.MESH):
                continue
            src = model.shape_source[s]
            if src is None or not hasattr(src, "vertices"):
                continue
            if b not in self.body_ops:
                x = UsdGeom.Xform.Define(stage, f"/World/robot/body_{b}")
                self.body_ops[b] = x.AddTransformOp()
            v = np.asarray(src.vertices, float) * np.asarray(shape_scale[s], float)[None, :]
            f = np.asarray(src.indices, np.int32).reshape(-1, 3)
            m = UsdGeom.Mesh.Define(stage, f"/World/robot/body_{b}/shape_{s}")
            m.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in v])
            m.CreateFaceVertexIndicesAttr(f.flatten().tolist())
            m.CreateFaceVertexCountsAttr([3] * len(f))
            m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            m.CreateDisplayColorAttr([Gf.Vec3f(0.92, 0.92, 0.92)])
            UsdGeom.Xformable(m).AddTransformOp().Set(self._mat(shape_xf[s]))
            n_link_meshes += 1

        # --- table + lights --------------------------------------------------------------
        if table is not None:
            pos, hx, hy, hz = table
            c = UsdGeom.Cube.Define(stage, "/World/table")
            c.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(c)
            xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
            xf.AddScaleOp().Set(Gf.Vec3f(2.0 * hx, 2.0 * hy, 2.0 * hz))
            c.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.85, 0.85)])
        from pxr import UsdLux
        sun = UsdLux.DistantLight.Define(stage, "/World/sun")
        sun.CreateIntensityAttr(3000.0)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 20.0, 0.0))
        UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(400.0)

        # A recording that only writes at plan-completion is lost the moment the process is
        # killed - measured: a full 3-fold run left a 492-byte file. Save periodically and at
        # interpreter exit; Save() is cheap and idempotent.
        import atexit
        atexit.register(self.save)
        print(f"[usd-rec] recording to {out_path}: {len(self.body_ops)} robot bodies "
              f"({n_link_meshes} link meshes, {n_boxes} boxes), garment {len(tris)} tris, "
              f"{fps / self.every:.0f}fps", flush=True)

    def _mat(self, tf):
        """Newton transform [x y z qx qy qz qw] -> Gf.Matrix4d."""
        Gf = self._Gf
        x, y, z, qx, qy, qz, qw = (float(v) for v in tf)
        m = Gf.Matrix4d().SetRotate(Gf.Quatd(qw, qx, qy, qz))
        m.SetTranslateOnly(Gf.Vec3d(x, y, z))
        return m

    def sample(self, state, n_cloth: int):
        """Call once per sim frame; records every Nth."""
        self.frame += 1
        if self.frame % self.every:
            return
        Gf, Usd = self._Gf, self._Usd
        tc = Usd.TimeCode(self.t)
        q = state.particle_q.numpy()[:n_cloth]
        if np.isfinite(q).all():
            self.garment_pts.Set([Gf.Vec3f(*map(float, p)) for p in q], tc)
            if self.under_pts is not None:
                t = self.tris[(self.tris < len(q)).all(1)]
                fn = np.cross(q[t[:, 1]] - q[t[:, 0]], q[t[:, 2]] - q[t[:, 0]])
                n = np.zeros_like(q)
                for k in range(3):
                    np.add.at(n, t[:, k], fn)
                n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
                self.under_pts.Set([Gf.Vec3f(*map(float, p)) for p in q - n * self.thick], tc)
        bq = state.body_q.numpy()
        for b, op in self.body_ops.items():
            op.Set(self._mat(bq[b]), tc)
        self.t += 1
        if self.t % 150 == 0:
            self.stage.SetEndTimeCode(self.t - 1)
            self.stage.GetRootLayer().Save()

    def save(self):
        self.stage.SetEndTimeCode(max(self.t - 1, 1))
        self.stage.GetRootLayer().Save()
        print(f"[usd-rec] saved {self.out}: {self.t} frames "
              f"({self.t / max(self.stage.GetTimeCodesPerSecond(), 1):.0f}s of animation) - "
              f"open in Isaac Sim and press PLAY", flush=True)
