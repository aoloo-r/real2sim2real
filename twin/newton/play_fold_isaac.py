"""Open a recorded fold USD in Isaac Sim, aim the camera, and play on loop.

    python play_fold.py <stage.usd> [cx cy cz tx ty tz]
"""
import sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": False})
import omni.usd, omni.timeline
from pxr import Gf

path = sys.argv[1]
cam = [float(v) for v in sys.argv[2:8]] if len(sys.argv) >= 8 else [110, -150, 85, -5, -45, 22]
# SPEED: 317s of fold animation at the recorded 15fps takes 5min to watch, and a viewer opened
# while the file is still being written loops only the frames it had at open ("the robot never
# finishes the fold"). Pass a multiplier to play it faster; the range is set EXPLICITLY below
# from the stage, so a file that grew after open is still played in full on reopen.
speed = float(sys.argv[8]) if len(sys.argv) >= 9 else 1.0
omni.usd.get_context().open_stage(path)
app.update(); app.update()

# deterministic camera: no mouse gymnastics, just put it where the scene is
try:
    from omni.kit.viewport.utility import get_active_viewport
    from omni.kit.viewport.utility.camera_state import ViewportCameraState
    vs = ViewportCameraState(viewport=get_active_viewport())
    vs.set_position_world(Gf.Vec3d(*cam[:3]), True)
    vs.set_target_world(Gf.Vec3d(*cam[3:]), True)
    print(f"[play] camera at {cam[:3]} -> {cam[3:]}", flush=True)
except Exception as e:
    print(f"[play] camera setup failed: {e!r}", flush=True)

stage = omni.usd.get_context().get_stage()
fps = stage.GetTimeCodesPerSecond() or 30.0
t0, t1 = stage.GetStartTimeCode(), stage.GetEndTimeCode()
tl = omni.timeline.get_timeline_interface()
try:
    tl.set_time_codes_per_second(fps * speed)
    tl.set_start_time(float(t0) / fps)
    tl.set_end_time(float(t1) / fps)
except Exception as e:
    print(f"[play] could not set the timeline range: {e!r}", flush=True)
print(f"[play] {t1 - t0 + 1:.0f} frames = {(t1 - t0 + 1) / fps:.0f}s of animation, "
      f"playing at x{speed:g} ({(t1 - t0 + 1) / fps / speed:.0f}s per loop)", flush=True)
tl.set_looping(True)
tl.play()
while app.is_running():
    app.update()
app.close()
