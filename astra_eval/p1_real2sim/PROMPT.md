# Task: build the scene description of a table-top from one RGB-D capture

You are given one capture of a real table in `inputs/capture/`: a colour image, an aligned depth map
in metres, the camera intrinsics, and the calibrated camera-to-robot transform. On the table lie a
towel, an open cardboard box, and a few small objects standing on the towel. `inputs/frame.json`
defines the table frame the output must use.

Produce `scene.json` in exactly the format of `FORMAT.md`: every object on the towel with its name,
shape class, size in cm and position relative to the towel centre; the towel's centre, size and
mass; the box's inner size, wall height and position. Metric accuracy matters: sizes and positions
within 1 cm are what the downstream task needs. Use the depth; a pixel alone gives a ray, not a point.

Also write `notes.md`: how you measured each number, what you assumed, and which numbers you could
not settle from the data. Mass and friction cannot be seen; estimate them from the object class and
say so.

Rules
- Work only from `inputs/`. Do not search the web for this scene and do not ask for more captures.
- You may write and run any code you like here (Python 3 with numpy, opencv, trimesh are available
  in `python3`). Do not install packages that need the GPU.
- Budget: 90 minutes wall time. Stop and hand in what you have when it runs out.
- Output: `scene.json`, `notes.md`, and any code you wrote, in the output directory named at the end
  of this prompt.
