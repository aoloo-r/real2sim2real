"""Black-box cloth-folding rollout environment: headless, no robot, batched worlds.

For planning we only care what a fold does to the CLOTH. The arm costs a Featherstone
step plus a tape-Jacobian IK every frame and does not change the fold outcome (the fold
is applied kinematically either way), so it is left out of the inner loop entirely.
Robot feasibility — reach, collision, whether a real gripper could catch that edge — is a
separate check run once on a chosen plan, in the full twin.

Cloth parameters are copied from newton's cloth_franka example so results transfer to
the full sim. Units are CENTIMETRES, matching the example (its constants are cm-tuned).

    env = FoldEnv(n_worlds=16)
    env.reset()                                  # flat shirts, optional random SE(2)
    env.step(FoldAction(axis=1, frac=0.5))       # fold every world in half
    ext = env.extents()                          # (n_worlds, 2) cloth bbox per world

An action is a fold LINE, not a trajectory: the half of the sheet beyond the line is
reflected across it, which is what a fold does geometrically. The simulator then settles
the result, so drape, self-collision and thickness are the black box's business.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton import ModelBuilder
from newton.solvers import SolverVBD


@wp.kernel
def _fold_kernel(
    start: wp.array(dtype=wp.vec3),
    target: wp.array(dtype=wp.vec3),
    sel: wp.array(dtype=wp.int32),
    arc: wp.array(dtype=wp.float32),
    t: float,
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
):
    """Carry the PINCHED PATCH toward its mirrored target on the SAME path the twin's arm
    takes: a semicircle about the crease (cosine profile along the fold axis, sine lift), so
    the flap swings over like a page instead of sliding on a low bump. The twin rotates the
    flap about the crease with the apex at the flap length (capped at 26cm); matching it here
    makes the physics grid and the robot execution the same motion.

    Only the few particles the gripper holds are prescribed. The rest of the garment is left
    to VBD and follows through the cloth itself, so it drapes and sags instead of being
    teleported to a reflection.

    Doing this on device is the whole point: the previous version read every particle
    back to the CPU and uploaded it again on every frame, which is what made batching
    get *worse* past 8 worlds.
    """
    i = wp.tid()
    if sel[i] == 0:
        return
    if sel[i] == 2:
        # HELD DOWN, not clamped: a hand's weight keeps the bundle on the table but lets the
        # fabric slide. Clamping xyz against a moving pinch drew strands and blew up.
        q[i] = wp.vec3(q[i][0], q[i][1], start[i][2])
        qd[i] = wp.vec3(qd[i][0], qd[i][1], 0.0)
        return
    s = start[i]
    d = target[i] - s
    u = 0.5 * (1.0 - wp.cos(3.14159265358979 * t))      # 0 -> 1 along the semicircle
    pz = s[2] + arc[i] * wp.sin(3.14159265358979 * t)
    q[i] = wp.vec3(s[0] + d[0] * u, s[1] + d[1] * u, pz)
    qd[i] = wp.vec3(0.0, 0.0, 0.0)


@dataclass
class FoldAction:
    """One fold. `axis` 0=x 1=y; `frac` is the fold line as a fraction of the current
    extent along that axis; `side` picks which half moves (+1 = the high side)."""

    axis: int = 1
    frac: float = 0.5
    side: int = 1
    # Where along the fold edge the gripper takes hold, as a fraction of the OTHER axis.
    # 0.5 (the middle) carries the middle and lets the corners lag, which is why a "half"
    # fold lands ~10cm short. 0.75 / 0.25 grab the upper and lower parts so a side fold can
    # be done as two moves that each complete, instead of one that half-completes.
    grab_at: float = 0.5
    # "left" / "right" turn this into a SLEEVE fold: the grasp targets that sleeve's tip and the
    # fold line is the garment midline (right sleeve, full mirror) or the shoulder seam (left
    # sleeve, half travel) - the geometry the twin uses, so plans transfer.
    sleeve: str | None = None
    # For a sleeve fold: "seam" (default) lays the tip on its own shoulder seam; "across"
    # carries the tip to the FAR body edge - one hand taking a sleeve and moving it to the
    # other side of the shirt. Only the sleeve is pinned, so the panel comes along through the
    # cloth and the crease forms where the fabric puts it (a person's fold, not a mirror).
    travel: str = "seam"
    # UNFOLD: undo this world's most recent fold record (fold_records) - pin the flap at its
    # loose edge and swing it back over the crease as it is now. hold_rest additionally pins
    # every non-flap particle in place during the swing: a second hand on the bundle.
    unfold: bool = False
    hold_rest: bool = False
    # overshoot > 1 swings the loose edge past its mirror point (flaps land short, like folds);
    # finish=True re-pins the loose edge of the LAST unfolded flap and drags it out to the
    # mirror point on a low arc - a second pass to complete the opening.
    overshoot: float = 1.0
    finish: bool = False
    # two_point pinches the loose edge at BOTH corners so the flap opens flat like a page (one
    # mid-edge pinch carries the middle and the corners lag - the same shortfall split folds
    # fix). hold_margin_cm leaves non-flap fabric within this distance of the crease free, so
    # a held bundle can still rotate at the crease instead of being drawn into a strand.
    two_point: bool = False
    hold_margin_cm: float = 6.0
    hold_z_only: bool = True     # weight-like hold (z only); False clamps xyz
    # A pinched fold needs longer than a prescribed one: the flap has to be carried over, then
    # actually fall and come to rest. 0.8s of settle was enough when the half was teleported.
    #
    # 5.0s, not 2.0s: at 2.0s the SECOND fold diverged to NaN - on the flat sheet control as
    # well as the shirt, so it was never a garment-geometry problem. The prescribed patch was
    # being dragged into the doubled-over stack faster than VBD could resolve the self-collision
    # it was creating. At 5.0s three folds run clean on both cloths. Costs ~1.8x per rollout.
    duration_s: float = 7.0     # the twin's arc time at carry x1
    settle_s: float = 3.0
    # LABEL MOVES (2026-09-18): [(grab_particle, drop_particle_or_(x,y)), ...]. Pinch the fabric
    # around each grab particle and carry it onto the drop particle's CURRENT position; several
    # pairs = a two-handed move. No fold line is prescribed - the crease forms where the fabric
    # puts it. This is the garment-agnostic operator the semantic labels feed.
    moves: list | None = None
    # band_cm > 0: a p2p pinch grips a BAND of fabric along the edge (across the travel direction),
    # pin_radius across it - the edge-spanning grip the twin executes with (TWIN_FOLD_BAND_CM).
    # 0 keeps the 4cm point pinch. A plan validated with a point pinch did not transfer to the
    # band grip (exec_base diverged at a 1316-particle band on a doubled region).
    band_cm: float = 0.0


class FoldEnv:
    # cloth constants: newton cloth_franka, cm scale
    # same knobs as the twin (TWIN_TRI_KE / TWIN_BEND_KE ...) so validation and execution share one fabric
    TRI_KE = float(__import__("os").environ.get("TWIN_TRI_KE", "1e4"))
    TRI_KA = float(__import__("os").environ.get("TWIN_TRI_KA", str(TRI_KE)))
    TRI_KD = float(__import__("os").environ.get("TWIN_TRI_KD", "1.5e-2"))
    BEND_KE = float(__import__("os").environ.get("TWIN_BEND_KE", "5"))
    BEND_KD = float(__import__("os").environ.get("TWIN_BEND_KD", "5e-1"))
    DENSITY = 0.02
    PARTICLE_RADIUS = 0.8
    SUBSTEPS, ITERATIONS, FPS = 10, 5, 60
    SETTLE_V = 5.0        # cm/s: below this the cloth is at rest, stop settling

    def __init__(self, n_worlds: int = 1, spacing_cm: float = 120.0, device: str | None = None,
                 collide_every: int = -1, view: bool = False, pin_radius_cm: float = 4.0,
                 cloth: str = "shirt", scale_xy=(1.0, 1.0)):
        self.n = n_worlds
        # Per-axis footprint scale for the 'shirt' asset, applied to the REST shape about its
        # centroid (the same warp the twin's TWIN_CLOTH=fit does). A state dumped from a shirt
        # fitted at 1.08x and loaded into the stock rest shape is pre-stretched 8% and contracts
        # when it settles - the planner and the twin must agree on the rest shape.
        self.scale_xy = (float(scale_xy[0]), float(scale_xy[1]))
        # GRID layout, the way Isaac Lab arranges parallel envs: a line of N worlds is
        # N*spacing wide, so the camera has to sit far back and every shirt becomes a
        # speck. A near-square grid keeps the whole set close and readable.
        self.grid_cols = max(1, int(math.ceil(math.sqrt(n_worlds))))
        self.world_xy = [((i % self.grid_cols) * spacing_cm,
                          (i // self.grid_cols) * spacing_cm) for i in range(n_worlds)]
        self.cloth_kind = cloth
        self.pin_radius = float(pin_radius_cm)   # a fingertip grip, not the whole half
        self.last_grip = np.zeros(n_worlds, dtype=np.int32)   # fabric found at the last grab
        # Where the ARM is, so a fold can be rejected for being out of reach rather than
        # silently turning into a different fold when the twin clamps the waypoint.
        self.base_cm = None
        self.reach_cm = 0.0
        self.last_reach = np.zeros(n_worlds, dtype=np.float32)   # grasp radius from the base
        self.spacing = spacing_cm
        self.frame_dt = 1.0 / self.FPS
        self.sim_dt = self.frame_dt / self.SUBSTEPS

        scene = ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0, 0.0, -981.0))

        # NOTE: `unisex_shirt.usd` IS the cloth from newton's own cloth_franka example - that
        # example loads the same asset from the same prim. The 'sheet' option below is not a
        # different example, it is a FLAT CONTROL: same material constants, same particle
        # count, no sleeves/seams/two-layer structure. It exists to separate "our fold operator
        # misbehaves" from "a shirt is hard to fold with one pinch".
        # 'mask:<TS>' builds the garment from that capture's own segmentation mask + depth, the
        # same sheet twin_stockfold makes with TWIN_CLOTH=mask. The planner and the twin MUST
        # agree on topology - a state dumped from a 915-particle trouser sheet cannot be loaded
        # into a 6436-particle shirt - so the source has to be selectable here too.
        _mesh_indices = None
        if self.cloth_kind.startswith("mask:"):
            from cloth_mesh import cloth_from_mask, garment_mask_path
            # mask:<TS> or mask:<TS>:<garment id | label substring> (multi-garment captures)
            _parts = self.cloth_kind.split(":", 2)
            _ts = _parts[1]
            _sel = _parts[2] if len(_parts) > 2 else None
            _v_base, _f, _uv, _tex = cloth_from_mask(
                garment_mask_path(_ts, _sel),
                f"/home/aoloo/sam-3d-objects/captures/{_ts}",
                step=int(__import__("os").environ.get("TWIN_MASK_STEP", "4")),
                two_layer=__import__("os").environ.get("TWIN_CLOTH_LAYERS", "2") == "2",
                layer_gap=float(__import__("os").environ.get("TWIN_LAYER_GAP_CM", "1.6")))
            # The twin places these base-frame (forward, left, up) cm coords into its world as
            # base + FWD*f + LEFT*l + UP*u with FWD=-y, LEFT=+x. Same mapping here, minus the
            # robot: only the garment's shape and relative layout matter to the rollout.
            _vw = np.stack([_v_base[:, 1], -_v_base[:, 0], _v_base[:, 2]], 1)
            _vw[:, 2] += 6.0 - _vw[:, 2].min()      # keep the layer gap, just rest it on the table
            verts = [wp.vec3(v) for v in _vw.astype(np.float32)]
            _mesh_indices = np.asarray(_f, dtype=np.int32).reshape(-1)
            print(f"[fold_env] cloth from mask {_ts}"
                  f"{' garment ' + _sel if _sel else ''}: {len(verts)} particles", flush=True)
        elif self.cloth_kind == "shirt":
            stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
            mesh = newton.usd.get_mesh(stage.GetPrimAtPath("/root/shirt"))
            _sv = np.asarray(mesh.vertices, np.float32).copy()
            if self.scale_xy != (1.0, 1.0):
                _c = _sv[:, :2].mean(0)
                _sv[:, 0] = (_sv[:, 0] - _c[0]) * self.scale_xy[0] + _c[0]
                _sv[:, 1] = (_sv[:, 1] - _c[1]) * self.scale_xy[1] + _c[1]
                print(f"[fold_env] shirt rest shape scaled ({self.scale_xy[0]:.3f}, "
                      f"{self.scale_xy[1]:.3f})", flush=True)
            verts = [wp.vec3(v) for v in _sv]
            _mesh_indices = mesh.indices

        for i in range(n_worlds):
            w = ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0, 0.0, -981.0))
            if _mesh_indices is not None:
                _is_mask = self.cloth_kind.startswith("mask:")
                w.add_cloth_mesh(
                    vertices=verts,
                    indices=_mesh_indices,
                    rot=wp.quat_identity() if _is_mask
                        else wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
                    pos=wp.vec3(self.world_xy[i][0],
                                (0.0 if _is_mask else 70.0) + self.world_xy[i][1],
                                0.0 if _is_mask else 6.0),
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    density=self.DENSITY,
                    scale=1.0,
                    tri_ke=self.TRI_KE, tri_ka=self.TRI_KA, tri_kd=self.TRI_KD,
                    edge_ke=self.BEND_KE, edge_kd=self.BEND_KD,
                    particle_radius=self.PARTICLE_RADIUS,
                )
            elif self.cloth_kind == "towel":
                # THE TWIN'S TOWEL, built the same way (twin_stockfold TWIN_CLOTH=towel): a flat
                # rectangle, vertex i*ny + j, spawned at table height, so the grid-index labels the
                # twin dumps (TWIN_TOWEL_LABELS) address the same particles here. Size, spacing and
                # density come from the same environment variables, so the planner and the twin
                # simulate one object.
                _e = __import__("os").environ
                _tw, _th = float(_e.get("TWIN_TOWEL_W_CM", "40")), float(_e.get("TWIN_TOWEL_H_CM", "30"))
                _sp = float(_e.get("TWIN_TOWEL_SPACING_CM", "0.5"))
                _nx, _ny = int(round(_tw / _sp)) + 1, int(round(_th / _sp)) + 1
                _gx, _gy = np.meshgrid(np.linspace(-_tw / 2, _tw / 2, _nx), np.linspace(-_th / 2, _th / 2, _ny), indexing="ij")
                _pts = np.stack([_gx.reshape(-1) + self.world_xy[i][0], _gy.reshape(-1) - 50.0 + self.world_xy[i][1],
                                 np.full(_gx.size, 20.5)], 1)
                _ix = np.arange(_nx * _ny).reshape(_nx, _ny)
                _a, _b = _ix[:-1, :-1].reshape(-1), _ix[1:, :-1].reshape(-1)
                _c, _d = _ix[1:, 1:].reshape(-1), _ix[:-1, 1:].reshape(-1)
                _tris = np.stack([_a, _b, _c, _a, _c, _d], 1).reshape(-1).astype(np.int32)
                w.add_cloth_mesh(
                    vertices=[wp.vec3(*v) for v in _pts], indices=_tris,
                    rot=wp.quat_identity(), pos=wp.vec3(0.0, 0.0, 0.0), vel=wp.vec3(0.0, 0.0, 0.0),
                    density=float(_e.get("TWIN_CLOTH_DENSITY", self.DENSITY)), scale=1.0,
                    tri_ke=self.TRI_KE, tri_ka=self.TRI_KA, tri_kd=self.TRI_KD,
                    edge_ke=self.BEND_KE, edge_kd=self.BEND_KD,
                    particle_radius=float(_e.get("TWIN_PARTICLE_RADIUS_CM", self.PARTICLE_RADIUS)),
                )
                self.FLAT_CM = (_tw, _th)
                if i == 0:
                    print(f"[fold_env] towel {_tw:.0f}x{_th:.0f}cm, {_nx}x{_ny} = {_nx * _ny} particles at {_sp:.1f}cm, "
                          f"density {float(_e.get('TWIN_CLOTH_DENSITY', self.DENSITY)):g}", flush=True)
            else:
                # 79x79 cells at 0.7cm = 55.3x55.3cm and 6400 particles, matching the shirt's
                # 55x59cm / 6436 as closely as a grid can, so only the GEOMETRY differs.
                w.add_cloth_grid(
                    pos=wp.vec3(self.world_xy[i][0] - 27.6,
                                -77.6 + self.world_xy[i][1], 6.0),
                    rot=wp.quat_identity(),
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    dim_x=79, dim_y=79, cell_x=0.7, cell_y=0.7,
                    mass=self.DENSITY * 0.7 * 0.7,
                    tri_ke=self.TRI_KE, tri_ka=self.TRI_KA, tri_kd=self.TRI_KD,
                    edge_ke=self.BEND_KE, edge_kd=self.BEND_KD,
                    particle_radius=self.PARTICLE_RADIUS,
                )
            w.color()
            scene.add_world(w)

        scene.add_ground_plane()
        self.model = scene.finalize(requires_grad=False)
        self.solver = SolverVBD(
            self.model,
            iterations=self.ITERATIONS,
            particle_self_contact_radius=0.2,
            particle_self_contact_margin=0.2,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.5,
            particle_enable_self_contact=True,
            particle_vertex_contact_buffer_size=16,
            particle_edge_contact_buffer_size=20,
            particle_collision_detection_interval=collide_every,
        )
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 1.0e1
        self.model.soft_contact_mu = 0.25

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model, soft_contact_margin=self.PARTICLE_RADIUS)
        self.contacts = self.collision_pipeline.contacts()

        # Optional viewer. The env works in CENTIMETRES, so the camera is placed in cm
        # too - no metre rescale here, which keeps the render honest about the units.
        self.viewer = None
        if view:
            self.viewer = newton.viewer.ViewerGL()
            self.viewer.set_model(self.model)
            # The viewer AUTO-COMPUTES a grid offset per world and adds it to the layout the
            # model already has, scattering worlds far outside the camera - which is why one
            # world rendered fine and four showed an empty screen.
            self.viewer.set_world_offsets((0.0, 0.0, 0.0))
            span = max(120.0, self.n * self.spacing)
            self.viewer.camera.pos = self.viewer.camera._as_vec3(
                wp.vec3(0.5 * (self.n - 1) * self.spacing, -60.0 - 0.9 * span, 0.75 * span))
            self.viewer.camera.look_at(
                wp.vec3(0.5 * (self.n - 1) * self.spacing, -50.0, 0.0))
            self.viewer.show_particles = False
            self._view_t = 0.0

        self.per_world = self.model.particle_count // self.n
        # per-world list of the folds applied since the last reset: what was grabbed where,
        # where it landed, about which line - the same record the twin keeps to UNFOLD
        self.fold_records = [[] for _ in range(n_worlds)]
        self.home = self.state_0.particle_q.numpy().copy()
        self.landmark_idx = None
        self._pick_landmarks()

    # ---- landmarks -------------------------------------------------------------
    # Named points on the garment (sleeve tips, collar, hem corners). These are just
    # INDICES into the existing particle set - the cloth itself is untouched. Extents
    # alone cannot tell "sleeves folded in" from "body folded in half"; landmarks can,
    # because particle identity persists through every fold.
    LANDMARK_NAMES = ("sleeve_l", "sleeve_r", "collar", "hem_l", "hem_r")

    def _pick_landmarks(self):
        """Locate landmarks on the flat garment, from its own geometry.

        The sleeve band is the widest slice across the garment; the narrower of the two
        ends is the collar, the other the hem. Derived from the mesh rather than
        hardcoded, so it survives the asset being placed at any rotation.
        """
        q = self.home.reshape(self.n, self.per_world, 3)[0]
        y = q[:, 1]
        edges = np.linspace(y.min(), y.max(), 11)
        widths, mids = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (y >= a) & (y < b)
            widths.append(float(np.ptp(q[m, 0])) if m.sum() else 0.0)
            mids.append(0.5 * (a + b))
        widths = np.asarray(widths)

        band = (y >= edges[widths.argmax()]) & (y < edges[widths.argmax() + 1])
        band_idx = np.where(band)[0]
        sleeve_l = band_idx[q[band_idx, 0].argmin()]
        sleeve_r = band_idx[q[band_idx, 0].argmax()]

        # the end with the smaller width is the collar
        collar_at_high_y = widths[-1] < widths[0]
        c_edge, h_edge = (y.max(), y.min()) if collar_at_high_y else (y.min(), y.max())
        c_mask = np.abs(y - c_edge) < 0.08 * float(np.ptp(y))
        h_mask = np.abs(y - h_edge) < 0.08 * float(np.ptp(y))
        c_idx, h_idx = np.where(c_mask)[0], np.where(h_mask)[0]
        collar = c_idx[np.abs(q[c_idx, 0] - q[c_idx, 0].mean()).argmin()]
        hem_l = h_idx[q[h_idx, 0].argmin()]
        hem_r = h_idx[q[h_idx, 0].argmax()]

        self.landmark_idx = np.array([sleeve_l, sleeve_r, collar, hem_l, hem_r], dtype=np.int64)
        self.sleeve_tip_idx = {"left": int(sleeve_l), "right": int(sleeve_r)}

        # Torso vs sleeves, decided on the flat garment: the body is the narrow band, the
        # sleeves are what sticks out past it. Membership is by particle index, so it
        # survives every fold - unlike a test against the garment's current bounding box,
        # which moves with the fold and so cannot say whether a sleeve came in.
        body_w = float(np.median([w for w in widths if w < 0.75 * widths.max()]))
        cx = float(np.median(q[:, 0]))
        self.body_idx = np.where(np.abs(q[:, 0] - cx) <= 0.5 * body_w)[0]
        self.body_half_w = 0.5 * body_w
        self.body_cx = cx
        # the sleeve as a particle SET, so a sleeve fold carries sleeve and not torso
        self.sleeve_set = {
            "left": np.where(q[:, 0] < cx - self.body_half_w)[0],
            "right": np.where(q[:, 0] > cx + self.body_half_w)[0],
        }
        return self.landmark_idx

    def landmarks(self) -> np.ndarray:
        """(n_worlds, 5, 3) landmark positions, ordered as LANDMARK_NAMES."""
        if getattr(self, "landmark_idx", None) is None:
            self._pick_landmarks()
        return self.particles()[:, self.landmark_idx, :]

    def landmark_report(self, world: int = 0) -> str:
        lm = self.landmarks()[world]
        q = self.particles()[world]
        c = q[:, :2].mean(axis=0)
        parts = []
        for name, p in zip(self.LANDMARK_NAMES, lm):
            # how far out the landmark sits relative to the garment's current spread
            r = float(np.linalg.norm(p[:2] - c))
            parts.append(f"{name} ({p[0]:.0f},{p[1]:.0f}) r={r:.0f}")
        return "  ".join(parts)

    def sleeve_overhang(self, world: int = 0) -> tuple[float, float]:
        """How far each sleeve tip sticks out past the TORSO, in cm (0 = inside).

        Measured against the body particles' current span rather than the garment's
        bounding box, so folding the body does not silently "tuck" a sleeve.
        Continuous rather than boolean: search needs a gradient, not a step.
        """
        return self.sleeve_overhang_of(self.particles()[world])

    def sleeve_overhang_of(self, q: np.ndarray) -> tuple[float, float]:
        """Same measure, but for a SAVED state rather than whatever is live in a world.

        The search scores candidates it has already read back and stashed, so asking the env
        for "world 0's overhang" gave every candidate the same number - whatever the last
        rollout chunk happened to leave there. Landmarks are fixed INDICES, so they apply to
        any state of this garment.
        """
        if self.cloth_kind != "shirt":
            return 0.0, 0.0        # a flat sheet has no sleeves; the predicate is vacuous
        if getattr(self, "landmark_idx", None) is None:
            self._pick_landmarks()
        lm = q[self.landmark_idx, :]
        body = q[self.body_idx, 0]
        lo, hi = float(np.percentile(body, 2.0)), float(np.percentile(body, 98.0))
        # side-agnostic: a fold can mirror a sleeve to the opposite side, so measure the
        # distance outside the body span in either direction, not "past its own edge"
        out = lambda x: max(0.0, lo - x, x - hi)
        return (out(float(lm[0, 0])), out(float(lm[1, 0])))

    def sleeves_tucked(self, world: int = 0, margin_cm: float = 2.0) -> tuple[bool, bool]:
        """Boolean form of `sleeve_overhang`, with a tolerance."""
        l, r = self.sleeve_overhang(world)
        return (l <= margin_cm, r <= margin_cm)

    # ---- state ----------------------------------------------------------------
    def particles(self) -> np.ndarray:
        """(n_worlds, per_world, 3) particle positions."""
        return self.state_0.particle_q.numpy().reshape(self.n, self.per_world, 3)

    def set_particles(self, q: np.ndarray):
        flat = q.reshape(-1, 3).astype(np.float32)
        self.state_0.particle_q.assign(flat)
        self.state_0.particle_qd.zero_()

    # The garment's flat extent, measured once on this asset. Symbolic size classes are
    # fractions of this, so they mean the same thing whatever the shirt is doing.
    # The FLAT footprint this garment started at. Measured from the cloth itself the first time
    # it settles, NOT a t-shirt constant: "half folded" means half of THIS garment. With the
    # shirt's 55.6x59.2 hard-coded, a 34x44cm pair of trousers read as already half-folded
    # before anything had touched it, and the goal was scored against the wrong denominator.
    FLAT_CM = (55.6, 59.2)

    def measure_flat(self, world: int = 0):
        """Adopt the garment's current extent as its flat reference."""
        q = self.particles()[world]
        lo, hi = np.percentile(q, 3.0, axis=0), np.percentile(q, 97.0, axis=0)
        ext = (hi - lo)[:2]
        if np.isfinite(ext).all() and ext[0] > 1.0 and ext[1] > 1.0:
            if self.cloth_kind == "towel":
                # the towel's flat size is KNOWN (it was built to it); the percentile extents
                # read 38x29 for a 40x30 towel and would make an ideal quarter fold score 0.27
                pass
            else:
                self.FLAT_CM = (float(ext[0]), float(ext[1]))
            print(f"[fold_env] flat reference: {ext[0]:.0f}x{ext[1]:.0f}cm "
                  f"({ext[0] * ext[1]:.0f}cm2) measured from the garment", flush=True)
        return self.FLAT_CM

    def symbolic_state(self, world: int = 0, q: np.ndarray | None = None) -> dict:
        """The garment's SYMBOLIC state - a few discrete predicates, not raw centimetres.

        Planning over continuous extents makes every rollout its own special case. These are
        the states a garment is actually IN: how much of it is still spread out, and which
        sleeves are tucked. A shirt that arrives half-folded and rotated has the same symbolic
        state as any other half-folded shirt, which is what lets plans generalise across
        placements instead of being fitted to one.
        """
        q = self.particles()[world] if q is None else q
        lo, hi = np.percentile(q, 3.0, axis=0), np.percentile(q, 97.0, axis=0)
        ext = (hi - lo)[:2]
        if not np.isfinite(ext).all():
            return {"size": "diverged", "sleeves": "unknown", "label": "diverged"}
        frac = float(ext[0] * ext[1]) / float(self.FLAT_CM[0] * self.FLAT_CM[1])
        size = ("spread" if frac > 0.80 else
                "half" if frac > 0.45 else
                "quarter" if frac > 0.22 else
                "packed")
        oh = self.sleeve_overhang_of(q)
        l_in, r_in = oh[0] <= 2.0, oh[1] <= 2.0
        sleeves = ("both_in" if l_in and r_in else
                   "left_in" if l_in else
                   "right_in" if r_in else
                   "both_out")
        return {"size": size, "sleeves": sleeves, "label": f"{size}/{sleeves}",
                "extent_cm": (round(float(ext[0]), 1), round(float(ext[1]), 1)),
                "area_frac": round(frac, 3)}

    def grip_counts(self) -> np.ndarray:
        """(n_worlds,) fabric particles a fixed-height gripper would find at the last grab.

        The planner's blind spot until now: extent and sleeve overhang say nothing about
        whether the garment can be TAKEN HOLD OF at the point the next fold needs.
        """
        return self.last_grip

    def extents(self) -> np.ndarray:
        """(n_worlds, 2) robust cloth extent in x and y — the planner's state summary."""
        q = self.particles()
        lo = np.percentile(q, 3.0, axis=1)
        hi = np.percentile(q, 97.0, axis=1)
        return (hi - lo)[:, :2]

    def reset(self, q: np.ndarray | None = None, randomize: bool = False, seed: int = 0,
              settle_s: float = 1.0):
        """Flat shirts (or a supplied state), optionally with random SE(2) placement."""
        q = self.home.copy() if q is None else q.reshape(-1, 3).astype(np.float32).copy()
        if randomize:
            rng = np.random.default_rng(seed)
            q = q.reshape(self.n, self.per_world, 3)
            for i in range(self.n):
                c = q[i, :, :2].mean(axis=0)
                th = rng.uniform(-np.pi, np.pi)
                R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                shift = rng.uniform(-8.0, 8.0, size=2)
                q[i, :, :2] = (q[i, :, :2] - c) @ R.T + c + shift
            q = q.reshape(-1, 3)
        self.set_particles(q)
        self.fold_records = [[] for _ in range(self.n)]
        self._advance(int(settle_s * self.FPS))
        return self.particles()

    # ---- dynamics -------------------------------------------------------------
    def hold(self, seconds: float = 1.0):
        """Keep the window alive WITHOUT stepping the sim.

        The viewer's hold loop used to call _advance(30) forever, i.e. it ran a full-rate cloth
        solve just to keep a finished pose on screen - a whole GPU's worth of work competing
        with every search running alongside it. Rendering the same state costs nothing.
        """
        if self.viewer is None:
            time.sleep(seconds)
            return
        self._view_t += seconds
        self.viewer.begin_frame(self._view_t)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()
        time.sleep(seconds)

    def set_reach(self, base_cm, reach_cm: float):
        """Tell the env where the robot base is (sim world cm) and how far it can reach.

        Worlds are laid out on a grid, so each world's cloth is offset from world 0. The base is
        given in world-0 coordinates and the same offset is applied per world, which is what
        makes a grasp radius comparable across the batch.
        """
        self.base_cm = np.asarray(base_cm, float)
        self.reach_cm = float(reach_cm)
        print(f"[fold_env] reach model: base ({self.base_cm[0]:.0f},{self.base_cm[1]:.0f})cm, "
              f"limit {self.reach_cm:.0f}cm", flush=True)

    def frame_scene(self, margin: float = 2.1):
        """Aim the camera so the WHOLE grid fits, computed from the field of view.

        Guessing distance multipliers put the near row in the lens one time and the whole scene
        outside the far clip the next. The camera's fov is known (45 deg), so the distance that
        fits a scene of a given size is arithmetic: d = (size/2) / tan(fov/2), times a margin.
        """
        if self.viewer is None:
            return
        q = self.state_0.particle_q.numpy()
        finite = q[np.isfinite(q).all(axis=1)]
        if not len(finite):
            return
        lo, hi = finite.min(axis=0), finite.max(axis=0)
        cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
        size = max(float(hi[0] - lo[0]), float(hi[1] - lo[1]), 60.0)
        fov = float(getattr(self.viewer.camera, "fov", 45.0))
        dist = margin * (0.5 * size) / math.tan(math.radians(fov) * 0.5)
        # 35 deg above the table: high enough to separate the rows, low enough to read a fold
        elev = math.radians(72.0)
        self.viewer.camera.pos = self.viewer.camera._as_vec3(
            wp.vec3(cx, cy - dist * math.cos(elev), dist * math.sin(elev)))
        self.viewer.camera.look_at(wp.vec3(cx, cy, 0.0))
        self.viewer.camera.near = 1.0
        self.viewer.camera.far = max(1000.0, 4.0 * dist)     # cm units: the default 1000 clips
        print(f"[fold_env] camera: scene {size:.0f}cm at ({cx:.0f},{cy:.0f}), fov {fov:.0f}deg, "
              f"dist {dist:.0f}cm, far {self.viewer.camera.far:.0f}cm", flush=True)

    # Optional per-frame observer, for recording a rollout (e.g. to a time-sampled USD that
    # Isaac Sim / Isaac Lab can play back with RTX rendering). Physics is untouched.
    frame_cb = None

    def _advance(self, frames: int):
        for _ in range(frames):
            if self.frame_cb is not None:
                self.frame_cb(self)
            self.solver.rebuild_bvh(self.state_0)
            for _s in range(self.SUBSTEPS):
                self.state_0.clear_forces()
                self.collision_pipeline.collide(self.state_0, self.contacts)
                self.solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0
            if self.viewer is not None:
                self._view_t += self.frame_dt
                self.viewer.begin_frame(self._view_t)
                # Draw the cloth as a lit LIME mesh rather than the viewer's default material -
                # the same thing the twin does. Flat tan on dark blue is hard to read a fold in.
                if not hasattr(self, "_tri_idx"):
                    self._tri_idx = wp.array(
                        self.model.tri_indices.numpy().reshape(-1).astype(np.int32),
                        dtype=wp.int32)
                    self.viewer.show_triangles = False
                self.viewer.log_state(self.state_0)
                self.viewer.log_mesh("cloth", self.state_0.particle_q, self._tri_idx,
                                     color=wp.vec3(0.66, 0.74, 0.36), backface_culling=False)
                self.viewer.end_frame()

    def step(self, action: FoldAction | list[FoldAction]):
        """Apply one fold per world, then settle. Actions may differ per world."""
        acts = action if isinstance(action, list) else [action] * self.n
        q = self.particles()

        lo = np.percentile(q, 3.0, axis=1)
        hi = np.percentile(q, 97.0, axis=1)

        sel, lines, axes, grip = [], [], [], []
        sleeve_targets = {}
        p2p_disp = {}
        hold_masks = {}
        unfold_over = {}
        for i, a in enumerate(acts):
            if a.unfold or a.finish:
                recs = self.fold_records[i]
                if a.finish:
                    rec = getattr(self, "last_unfold", {}).get(i)
                elif recs:
                    rec = recs.pop()
                else:
                    rec = None
                if rec is None:
                    sel.append(np.zeros(self.per_world, dtype=bool))
                    lines.append(0.0); axes.append(1); grip.append(0)
                    continue
                if not hasattr(self, "last_unfold"):
                    self.last_unfold = {}
                self.last_unfold[i] = rec
                ax = int(rec["axis"])
                dx, dy = self.world_xy[i]
                off = dx if ax == 0 else dy
                flap = np.asarray(rec.get("flap_set", []), int)
                if len(flap) < 40:
                    flap = np.arange(self.per_world)
                qf = q[i, flap]
                drop_ax = float(rec["drop"][ax]) + off
                line_ax = float(rec["line"]) + off
                side = 1.0 if drop_ax >= line_ax else -1.0          # loose edge lies this side
                zmid = 0.5 * (np.percentile(q[i, :, 2], 5) + np.percentile(q[i, :, 2], 95))
                upper = qf[qf[:, 2] > zmid] if (qf[:, 2] > zmid).sum() > 20 else qf
                edge = upper[int(np.argmax(upper[:, ax] * side))]
                crease = float(np.percentile(qf[:, ax], 3.0 if side > 0 else 97.0))
                m = np.zeros(self.per_world, dtype=bool)
                if a.two_point:
                    other = 1 - ax
                    band = upper[np.abs(upper[:, ax] - edge[ax]) < 4.0]     # the loose edge strip
                    if len(band) < 10:
                        band = upper
                    lo_c = band[int(np.argmin(band[:, other]))]
                    hi_c = band[int(np.argmax(band[:, other]))]
                    for corner in (lo_c, hi_c):
                        c_in = corner.copy()
                        c_in[other] += 3.0 if corner is lo_c else -3.0      # 3cm in from the corner
                        d2 = (q[i, :, 0] - c_in[0]) ** 2 + (q[i, :, 1] - c_in[1]) ** 2
                        m[flap] |= d2[flap] < self.pin_radius ** 2
                else:
                    d2 = ((q[i, :, 0] - edge[0]) ** 2 + (q[i, :, 1] - edge[1]) ** 2)
                    m[flap] = d2[flap] < self.pin_radius ** 2
                # Open to the flap's FLAT length, not to the mirror of where its loose edge
                # happens to be: a fold that landed short leaves the flap compressed (19cm of
                # edge-to-crease for 28cm of fabric on the 2026-09-03 bundle), and mirroring
                # the observed edge leaves that difference folded. The flat length comes from
                # the asset's home state. The pin target is set through `overshoot`.
                qh = self.home.reshape(self.n, self.per_world, 3)[i]
                L_flat = float(np.ptp(qh[flap, ax])) if len(flap) > 40 else 0.0
                L_now = max(abs(float(edge[ax]) - crease), 1.0)
                ov = float(a.overshoot)
                if a.overshoot == 1.0 and L_flat > L_now:
                    ov = min(L_flat / L_now, 1.6)
                sel.append(m), lines.append(crease), axes.append(ax)
                grip.append(int(m.sum()))
                unfold_over[i] = (ov, bool(a.finish))
                if a.hold_rest:
                    hm = np.ones(self.per_world, dtype=bool); hm[flap] = False
                    hm &= np.abs(q[i, :, ax] - crease) > float(a.hold_margin_cm)   # crease stays free
                    hold_masks[i] = hm
                continue
            if a.moves:
                m = np.zeros(self.per_world, dtype=bool)
                disp = np.zeros((self.per_world, 2), np.float32)
                dx, dy = self.world_xy[i]
                rec_moves = []
                for gi, di in a.moves:
                    tip = q[i, int(gi)]
                    tgt = q[i, int(di), :2] if isinstance(di, (int, np.integer)) else np.array([di[0] + dx, di[1] + dy], float)
                    if a.band_cm > 0.0:
                        # band along the edge: wide across the travel direction, pin_radius along it
                        _tv = tgt - tip[:2]; _tn = float(np.linalg.norm(_tv))
                        _u = _tv / _tn if _tn > 1e-6 else np.array([1.0, 0.0])
                        _rel = q[i, :, :2] - tip[:2]
                        _along = _rel @ _u; _across = _rel @ np.array([-_u[1], _u[0]])
                        near = (np.abs(_along) < self.pin_radius) & (np.abs(_across) < a.band_cm) \
                               & (np.abs(q[i, :, 2] - tip[2]) < self.pin_radius)
                    else:
                        near = np.linalg.norm(q[i, :, :2] - tip[:2], axis=1) < self.pin_radius
                    m |= near
                    disp[near] = (tgt - tip[:2]).astype(np.float32)
                    rec_moves.append({"grab": [float(tip[0]) - dx, float(tip[1]) - dy],
                                      "drop": [float(tgt[0]) - dx, float(tgt[1]) - dy]})
                sel.append(m); lines.append(0.0); axes.append(0)
                p2p_disp[i] = disp
                floor = float(np.percentile(q[i, :, 2], 2.0))
                tip0 = q[i, int(a.moves[0][0])]
                near0 = (np.linalg.norm(q[i, :, :2] - tip0[:2], axis=1) < self.pin_radius) & (q[i, :, 2] < floor + self.pin_radius)
                grip.append(int(near0.sum()))
                if self.base_cm is not None:
                    self.last_reach[i] = max(float(np.hypot(mv["grab"][0] - self.base_cm[0], mv["grab"][1] - self.base_cm[1])) for mv in rec_moves)
                self.fold_records[i].append({"kind": "p2p", "moves": rec_moves, "axis": 0, "line": 0.0,
                                             "grab": rec_moves[0]["grab"], "drop": rec_moves[0]["drop"],
                                             "flap_set": [int(v) for v in np.where(m)[0]]})
                continue
            if a.sleeve is None and a.side == 0:
                # NO-OP (a world whose sequence has finished). This used to fall through to the
                # geometric fold: side=0 picked the LOW edge (percentile 3), pinned a patch there
                # and mirrored it across the line - so every finished world got an extra hem
                # fold and its final extent was scored on a fold its plan never contained.
                sel.append(np.zeros(self.per_world, dtype=bool))
                lines.append(0.0); axes.append(a.axis); grip.append(0)
                continue
            if a.sleeve is not None:
                if getattr(self, "landmark_idx", None) is None:
                    self._pick_landmarks()
                tip = q[i, self.sleeve_tip_idx[a.sleeve]]
                # Midline and seam from the body particles' CURRENT positions, not the values
                # measured on the flat shirt at its stock spot. Those were absolute world x;
                # once a start state was loaded elsewhere (fold_state_fit moves the shirt 20cm
                # and folds it) the "seam" sat on empty table and the pinched tip was dragged
                # 60cm off the garment - the taffy strands in the validation grid, and 4 of 5
                # sleeve worlds diverging.
                bx = q[i, self.body_idx, 0]
                cx0 = float(np.median(bx))
                seam = float(np.percentile(bx, 3.0 if a.sleeve == "left" else 97.0))
                far_seam = float(np.percentile(bx, 97.0 if a.sleeve == "left" else 3.0))
                if getattr(a, "travel", "seam") == "across":
                    # to the far edge of the body; the panel follows, so no stretch cap
                    line_s = 0.5 * (float(tip[0]) + far_seam)
                    tgt_x = far_seam
                else:
                    line_s = cx0 if a.sleeve == "right" else seam
                    tgt_x = (2.0 * line_s - float(tip[0])) if a.sleeve == "right" else line_s
                    # A sleeve can fold at most twice its own length (a full 180 about the
                    # seam). Cap the travel there so a mis-located seam can stretch nothing.
                    sleeve_len = max(abs(float(tip[0]) - seam), 4.0)
                    travel = tgt_x - float(tip[0])
                    if abs(travel) > 2.0 * sleeve_len:
                        tgt_x = float(tip[0]) + np.sign(travel) * 2.0 * sleeve_len
                only = self.sleeve_set[a.sleeve]
                m = np.zeros(self.per_world, dtype=bool)
                near_tip = np.linalg.norm(q[i] - tip, axis=1) < self.pin_radius
                m[only] = near_tip[only]              # hold sleeve fabric only
                sel.append(m), lines.append(float(line_s)), axes.append(0)
                grip.append(int(m.sum()))
                sleeve_targets[i] = float(tgt_x)
                dx, dy = self.world_xy[i]
                self.fold_records[i].append({
                    "kind": "sleeve", "sleeve": a.sleeve,
                    "sleeve_particle": int(self.sleeve_tip_idx[a.sleeve]),
                    "sleeve_home": [float(tip[0]) - dx, float(tip[1]) - dy],
                    "sleeve_set": [int(v) for v in self.sleeve_set[a.sleeve]],
                    "axis": 0, "line": float(line_s) - dx,
                    "grab": [float(tip[0]) - dx, float(tip[1]) - dy],
                    "drop": [float(tgt_x) - dx, float(tip[1]) - dy]})
                continue
            ax = a.axis
            other = 1 - ax
            line = lo[i, ax] + (hi[i, ax] - lo[i, ax]) * a.frac
            # where a gripper would take hold: 3cm inside the moving half's outer edge, at the
            # mid of the other axis - the same point twin_stockfold._plan_poses drives to.
            gy = lo[i, other] + (hi[i, other] - lo[i, other]) * a.grab_at
            # The edge AT THIS SLICE, not the garment's global bounding box. A shirt narrows
            # toward the collar, so a grab 0.75 along the edge that used the global box reached
            # past the shoulder into empty table and caught 8 particles.
            band = np.abs(q[i, :, other] - gy) < 4.0
            if int(band.sum()) > 20:
                # Percentile, not max/min - a stray particle would define the edge and put the
                # grasp in empty table (measured on the twin: pinch fell to 18-57 particles).
                col = q[i, band, ax]
                gx = float(np.percentile(col, 97.0 if a.side > 0 else 3.0)) - 3.0 * a.side
            else:
                gx = (hi[i, ax] if a.side > 0 else lo[i, ax]) - 3.0 * a.side
            d2 = (q[i, :, ax] - gx) ** 2 + (q[i, :, other] - gy) ** 2
            anchor = q[i, int(np.argmin(d2))]             # nearest actual fabric to that column
            m = np.linalg.norm(q[i] - anchor, axis=1) < self.pin_radius
            sel.append(m), lines.append(line), axes.append(ax)
            dx, dy = self.world_xy[i]
            _drop = [float(anchor[0]) - dx, float(anchor[1]) - dy]
            _drop[ax] = 2.0 * float(line) - float(anchor[ax]) - (dx if ax == 0 else dy)
            # the FLAP: every particle on the moving side of the line at fold time. Unfolding
            # must pinch only these (the top layer), never the stack under them.
            _flap = np.where((q[i, :, ax] > line) if a.side > 0 else (q[i, :, ax] < line))[0]
            self.fold_records[i].append({
                "kind": "axis", "axis": int(ax), "line": float(line) - (dx if ax == 0 else dy),
                "grab": [float(anchor[0]) - dx, float(anchor[1]) - dy], "drop": _drop,
                "flap_set": [int(v) for v in _flap]})
            if self.base_cm is not None:
                # This world's cloth sits at its own grid offset; the arm does not move with it.
                dx, dy = self.world_xy[i]
                self.last_reach[i] = float(np.hypot(anchor[0] - dx - self.base_cm[0],
                                                    anchor[1] - dy - self.base_cm[1]))

            # How much fabric a REAL gripper would find here. It descends to a fixed height
            # rather than snapping to the nearest particle, so it catches only what lies near
            # the table at (gx, gy). A crease left by an earlier fold is a tight rounded edge
            # with very little there - measured on the twin as 2 and 21 particles against a
            # normal 70-480. Recorded, never enforced: the rollout still folds normally.
            floor = float(np.percentile(q[i, :, 2], 2.0))
            near = (d2 < self.pin_radius ** 2) & (q[i, :, 2] < floor + self.pin_radius)
            grip.append(int(near.sum()))

        start = q.copy()
        target = q.copy()
        for i in range(self.n):
            ax, m = axes[i], sel[i]
            if i in p2p_disp:
                target[i, m, 0] = start[i, m, 0] + p2p_disp[i][m, 0]
                target[i, m, 1] = start[i, m, 1] + p2p_disp[i][m, 1]
            elif i in sleeve_targets:
                dx = sleeve_targets[i] - float(np.mean(start[i, m, 0])) if int(m.sum()) else 0.0
                target[i, m, 0] = start[i, m, 0] + dx
            else:
                target[i, m, ax] = 2.0 * lines[i] - start[i, m, ax]   # patch only
                if i in unfold_over:
                    ov, fin = unfold_over[i]
                    target[i, m, ax] = start[i, m, ax] + (target[i, m, ax] - start[i, m, ax]) * ov

        # pack selection / start / target / arc once, then stay on device
        sel_flat = np.zeros(self.n * self.per_world, np.int32)
        arc_flat = np.zeros(self.n * self.per_world, np.float32)
        for i in range(self.n):
            base = i * self.per_world
            code = sel[i].astype(np.int32)
            if i in hold_masks:                        # held particles
                hm = hold_masks[i] & ~sel[i]
                target[i, hm] = start[i, hm]
                code[hm] = 2 if getattr(acts[i], "hold_z_only", True) else 1
            sel_flat[base:base + self.per_world] = code
            # semicircle about the crease: radius = half the travel, apex capped like the
            # twin's TWIN_FOLD_ARC_MAX_CM (26cm)
            if i in p2p_disp:
                travel = np.linalg.norm(target[i, :, :2] - start[i, :, :2], axis=1)
            else:
                travel = np.abs(target[i, :, axes[i]] - start[i, :, axes[i]])
            R = 0.5 * travel
            if i in unfold_over and unfold_over[i][1]:
                R = 0.15 * travel                       # finishing pass: drag low, do not swing
            arc_flat[base:base + self.per_world] = np.minimum(R, 26.0).astype(np.float32)

        dev = self.model.device
        d_start = wp.array(start.reshape(-1, 3).astype(np.float32), dtype=wp.vec3, device=dev)
        d_target = wp.array(target.reshape(-1, 3).astype(np.float32), dtype=wp.vec3, device=dev)
        d_sel = wp.array(sel_flat, dtype=wp.int32, device=dev)
        d_arc = wp.array(arc_flat, dtype=wp.float32, device=dev)

        n_move = int(max(a.duration_s for a in acts) * self.FPS)
        for f in range(1, n_move + 1):
            wp.launch(
                _fold_kernel,
                dim=self.n * self.per_world,
                inputs=[d_start, d_target, d_sel, d_arc, float(f) / n_move],
                outputs=[self.state_0.particle_q, self.state_0.particle_qd],
                device=dev,
            )
            self._advance(1)

        # Settle only as long as the cloth is still moving. A fixed 3.0s was 37% of every
        # rollout and the flap is usually at rest well before then; the cap still applies, so
        # this never settles LESS thoroughly, it just stops paying for stillness.
        budget = int(max(a.settle_s for a in acts) * self.FPS)
        chunk = max(1, self.FPS // 4)                 # check four times a second
        spent = 0
        while spent < budget:
            self._advance(min(chunk, budget - spent))
            spent += chunk
            v = self.state_0.particle_qd.numpy()
            if not np.isfinite(v).all():
                break                                  # diverged; scoring will reject it
            if float(np.abs(v).max()) < self.SETTLE_V:
                break
        self.last_settle_frames = spent
        self.last_grip = np.array(grip, dtype=np.int32)
        return self.particles()


def _bench():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--folds", type=int, default=1)
    ap.add_argument("--view", action="store_true", help="watch it in the GL viewer")
    ap.add_argument("--collide-every", type=int, default=-1,
                    help="VBD self-collision detection interval in substeps; -1 = every substep")
    a = ap.parse_args()

    t0 = time.time()
    env = FoldEnv(n_worlds=a.worlds, collide_every=a.collide_every, view=a.view)
    t_build = time.time() - t0

    t0 = time.time()
    env.reset()
    t_reset = time.time() - t0
    print(f"[fold_env] {a.worlds} worlds x {env.per_world} particles  collide_every={a.collide_every}  "
          f"build {t_build:.1f}s  reset {t_reset:.1f}s", flush=True)
    print(f"[fold_env] extent after reset: {env.extents()[0].round(1)} cm", flush=True)
    print(f"[fold_env] landmarks: {env.landmark_report()}", flush=True)
    print(f"[fold_env] sleeve overhang (l,r): "
          f"{tuple(round(v,1) for v in env.sleeve_overhang())} cm  tucked={env.sleeves_tucked()}", flush=True)

    for k in range(a.folds):
        t0 = time.time()
        env.step(FoldAction(axis=1 if k % 2 == 0 else 0, frac=0.5))
        dt = time.time() - t0
        e = env.extents()
        print(f"[fold_env] fold {k + 1}: {dt:.1f}s for {a.worlds} worlds "
              f"({dt / a.worlds:.2f}s per world)  extent {e[0].round(1)} cm", flush=True)
        print(f"[fold_env]   landmarks: {env.landmark_report()}", flush=True)
        print(f"[fold_env]   sleeve overhang (l,r): "
              f"{tuple(round(v,1) for v in env.sleeve_overhang())} cm  tucked={env.sleeves_tucked()}", flush=True)


if __name__ == "__main__":
    _bench()
