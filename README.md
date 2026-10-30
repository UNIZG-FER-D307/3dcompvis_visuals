# RealSense RGBD capture & camera matrix demo

A web GUI that captures one RGBD frame from a connected Intel RealSense camera
and shows how the camera matrix `P = K[R|t]` shapes the rendered image: first
the intrinsics **K**, then the extrinsics **[R|t]**, then two-view geometry
(the **essential matrix**).

If no RealSense is available, the server falls back to a **synthetic scene**: an
orange ball resting on a checkerboard floor. It is ray-cast once with D435-like
intrinsics (`app/synthetic.py`); sky pixels have no depth. The live view's heading
then says "synthetic scene", and `GET /camera` reports `{"name": "synthetic"}`
(otherwise `"realsense"`).

## Features

- **Live RGB feed** (MJPEG) before capture.
- **Capture** grabs colour and depth together (depth is aligned to the colour
  stream) and shows:
  - the RGB image,
  - the depth in the `plasma` colormap (black = no depth),
    with a colour scale in metres underneath. The scale spans the 2nd–98th
    percentile of the captured depth; values outside it are clipped. The
    re-rendered depth uses the same scale,
  - an interactive point cloud (`<model-viewer>`, drag to orbit), similar to the
    [FoundationStereo](https://nvlabs.github.io/FoundationStereo/) project page.
    After capture, the view uses the RealSense's vertical field of view and is
    turned 30° to the side, orbiting the centre of the framed content. It is
    zoomed so the part of the frustum between the nearest and farthest depth, and
    the image plane, just fit the pane. You can
    zoom right into it.
    It also shows the camera frustum out to the far end of the depth scale: white
    for the RealSense K, cyan for the edited camera (slider K, plus the pose from
    `[R|t]` in the Extrinsics tab). The cyan frustum updates with the
    sliders, so you can watch the field of view narrow (larger `fx`/`fy`), shift
    (`cx`/`cy`) or shear (`s`). Your 3D view is kept while it reloads.
    An image plane (75% opaque) shows the RGB seen through the slider K, i.e. the
    re-rendered RGB, and follows the **Interpolate** checkbox. Like a physical
    sensor, it sits **behind the optical centre**, at distance `f` (30% of the
    frustum depth at the captured `fx`). The pixel width is fixed, so
    `f = fx · pixel width`: a larger `fx` moves the plane further back while its
    width stays the same. Changing `fy` alone changes the pixel height (`fy = f /
    pixel height`), so the plane gets shorter or taller but doesn't move.
    Each pixel's ray continues through the centre to the plane (drawn as the rear
    pyramid in the frustum's colour), so the image appears **inverted** (rotated
    180°), as in a pinhole camera. After capture or **Reset** the two frustums coincide
    and the plane shows the captured RGB (black where there is no depth).
- **Intrinsics playground**: the captured depth is back-projected **once** with
  the RealSense colour intrinsics into a fixed 3D point cloud. Sliders for
  `fx`, `fy`, `cx`, `cy` and skew `s` start at the RealSense values. About
  100 ms after you stop sliding, the cloud is re-projected through the edited K
  into a new RGB and depth image (z-buffered, nearest point wins). Extrinsics
  stay fixed at identity.
  - `fx`, `fy`: zoom (focal length) along each axis.
  - `cx`, `cy`: move the principal point, which shifts the image.
  - `s`: skew, which shears rows in proportion to their height.
  - **Reset** restores the RealSense K. The re-render then matches the capture
    at every pixel with valid depth.
  - **Interpolate** switches from point projection to an inverse warp with
    bilinear sampling, which fills the holes left by zooming in (see below).
- **Extrinsics playground** (second tab after capture): K is taken from the
  Intrinsics tab (shown read-only), and 3 rotation + 3 translation sliders set the
  extrinsics. The cloud is rendered as `K(R·X + t)`, z-buffered.
  - Convention: raw `[R|t]`, world→camera, `X_cam = R·X + t`. The world frame is
    the capture camera's frame. `R = Rz·Ry·Rx` with `rx`, `ry`, `rz` in degrees
    (±45°); `tx`, `ty`, `tz` in metres (±1 m). The readout shows the 3×4 `[R|t]`.
  - So `+tx` shifts the scene right (the camera moves left), `tz < 0` brings the
    scene closer, and nearer objects shift more (parallax). Areas the capture
    camera never saw appear as black "shadows" behind objects.
  - In 3D, the cyan frustum and image plane sit at the camera pose:
    centre `−Rᵀt`, `X_world = Rᵀ(X_cam − t)`. A second readout shows this
    camera→world transform `[Rᵀ | −Rᵀt]` and the centre `C = −Rᵀt`.
  - **Coordinate frames**: all frames are right-handed. The camera frame is
    OpenCV/RealSense: x right, y down, z forward. The world frame is the capture
    camera's frame. Positive angles turn by the right-hand rule about each axis.
    The 3D view shows the same frames turned 180° about x (glTF: y up, z towards
    the viewer), which keeps them right-handed.
  - A readout shows `P = K[R|t]` with its 3×4 values.
  - The extrinsics apply only in the Extrinsics tab; the slider values are kept
    when you switch tabs. **Reset extrinsics** sets them back to identity.
- **Essential tab** (third tab): two views of the captured cloud.
  - View 1 is `P1 = K[I|0]`. View 2 is `P2 = K[R|t]`, with the same K from the
    Intrinsics tab and its own R, t sliders. They default to `tx = −0.2 m`
    (camera 2 is 0.2 m to the right), so there is a baseline from the start.
  - The top row shows the two rendered views and the point cloud with both
    frustums (white: view 1, cyan: view 2).
  - The bottom row is split: the R, t sliders on the left; on the right, the
    matrices. Readouts: `P1`, `P2`, `E = [t]× R` (so `x2ₙᵀ E x1ₙ = 0` in normalised
    coordinates), `F = K⁻ᵀ E K⁻¹` (so `x2ᵀ F x1 = 0` in pixels), and the
    epipoles `e1 = K(−Rᵀt)` (camera 2 seen from view 1) and `e2 = K t`. An
    epipole far outside the image or at infinity is reported as such.
    `[R|t]` maps camera-1 to camera-2 coordinates (`X2 = R·X1 + t`); the
    camera 2→camera 1 transform `[Rᵀ | −Rᵀt]` and `C2 = −Rᵀt` are shown too.
  - **Hover** over either view at pixel x to draw yellow epipolar lines in both
    images: `l = F x` (or `Fᵀ x`) in the other view, and `l = e × x` through the
    epipole in the hovered view. The epipoles are marked with a magenta disc
    whenever they fall inside the image, with or without hovering. About 100 ms after the pointer pauses, the 3D view reloads with the
    **epipolar plane**: a translucent yellow triangle through both camera centres
    and the hovered pixel's ray, out to the far end of the depth scale. Leaving
    the image removes both.
  - With `t = 0`, E and F vanish and the epipolar geometry is undefined. The
    readout says so and nothing is drawn.

### Expected artefacts

Each 3D point covers exactly one pixel. Zooming in (larger `fx`/`fy`)
therefore leaves holes between points, and zooming out leaves black borders
where the camera saw nothing. Pixels without depth stay black in the re-render.

With **Interpolate** on, holes are filled. Because the camera centre never moves,
reprojecting the cloud is exactly the 2D homography `K' K⁻¹`, and depth drops out.
Each output pixel is mapped back to `K K'⁻¹ p` in the captured image and
sampled bilinearly from its 4 neighbours. Only neighbours with valid depth
contribute, and the same weights apply to RGB and depth, so interpolation fills
holes without revealing pixels that had no depth. Bilinear depth can produce
in-between ("flying") values at object edges.

Once extrinsics are non-zero, that warp is no longer exact: the camera centre
moves, so there is parallax and new occlusion. **Interpolate** then **splats**
instead. Each point is drawn as an s×s square with
`s = ceil(max(fx'/fx, fy'/fy) · z_capture / z_new)`, clipped to 1–8, which is how
much its 1-pixel capture footprint is magnified. The z-buffer is kept, so
occlusion stays correct. At exactly zero extrinsics the exact bilinear warp is
used, so switching on the first slider movement is expected.

## Reconstruction demos

The **Reconstruction demos (no camera)** button on the start page opens four
more tabs. They follow the notation of the lecture notes (`mvg.tex`) and use
synthetic checkerboard targets seen through the synthetic K (640×480); the
RealSense and the captured frame are not used.

Common to all three tabs:

- **Picking points**: click near a checkerboard corner and it snaps to the
  nearest selectable corner within 15 px. A corner is selectable if it is in
  front of the camera(s) and inside the image(s); in the two-view tabs it must be
  seen by both cameras, and its correspondence in the other view is known from
  the target. Click a picked corner again to remove it; **Clear points** removes
  all. Picked points that leave the view when the pose changes are dropped.
- **Noise σ** (0–3 px): each corner has a fixed random offset per view, scaled
  by σ, added to its picked pixel. The results then change smoothly with σ and
  are exact at σ = 0.
- **Pose sliders** move a camera (`rx ry rz` in degrees, `tx ty tz` in metres,
  `R = Rz·Ry·Rx`, `X_cam = R·X + t`, right-handed OpenCV frames as above).
- The equations are typeset with KaTeX, with the numbers filled in. A 3D view
  shows the target (as points), the cameras and the solutions.

The tabs:

- **Triangulation** (linear structure reconstruction): a two-board target (an
  open book with perpendicular pages, world = camera A). `P_A = K[I|0]` is fixed
  and the sliders set `P_B = K[R|t]`. Click a corner in either view to get
  `λ_A q_A = P_A Q`, `λ_B q_B = P_B Q`, and `[q_i]× P_i Q = 0`.
  - Two rows per view give `A_4×4 Q = 0`, with rows `y_A P_A[3,:] − P_A[2,:]`,
    `x_A P_A[3,:] − P_A[1,:]` and the same for B.
  - Q is the right singular vector of the smallest singular value. The panel
    shows it next to the true corner, with the error in mm and the reprojection
    errors.
  - 3D: both cameras, the rays through `q_A` and `q_B` (yellow), the triangulated
    Q (green cross) and the true corner (white cross).
- **Plane pose** (homography between a plane and its image): a calibration board
  at Z = 0 in its own frame. The sliders set the target→camera pose, so
  `λq = K[R|t](X_Q, Y_Q, 0, 1)ᵀ = K[R_:,1 R_:,2 t] q_R = H q_R`.
  - With at least 4 picked corners, `q × (H q_R) = 0` gives two independent
    equations per pair: `A_2n×9 h_9 = 0`. h is the last right singular vector
    (the plain DLT of the notes, without Hartley normalisation; 3 or more
    collinear points are reported as degenerate).
  - The estimated H is shown next to the true one.
  - Then `K⁻¹H = μ[r1 r2 t]` with `μ = ±‖K⁻¹h1‖`, `r3 = r1 × r2`, and R the
    nearest rotation `UVᵀ` (with noise, `‖r2‖ ≠ 1` and `r1·r2 ≠ 0`). There are
    two candidates, one per sign of μ.
  - **Validity test**: every picked target point must be in front of the camera,
    `(R Q + t)_z = r31 X_Q + r32 Y_Q + t_z > 0`. Exactly one candidate passes.
  - 3D: the true camera (white), the valid candidate (green) and the invalid one
    (red, mirrored behind the target). The true camera's image plane (its
    frustum's far rectangle, 0.3 m in front of the centre, filled translucent)
    shows, per picked point, a + where the true projection ray crosses it (the
    ray runs from the centre through the true pixel to the target corner), an ×
    at the noisy pixel, and a line between them. The noise is to scale (1 px ≈
    0.5 mm on the image plane), so zoom in to see it.
  - Each picked point's ray and markers in 3D, and its circle and number in the
    image, share one colour, evenly spaced over matplotlib's `rainbow` colormap
    in picking order.
- **Two views of a plane**: the same board, seen by camera 1 at a fixed pose
  (the board parallel to its image plane, 1 m away: `n = e3`, `d = 1 m`) and
  by camera 2, which the sliders set relative to camera 1: `X2 = R·X1 + t`.
  - With at least 4 corners picked in both views, `H1` and `H2` are each
    estimated by the DLT from target ↔ image pairs, and
    `H21 = H2 H1⁻¹` (shown next to the true one).
  - Calibrated: with `P1 = [I|0]`, `P2 = [R|t]` and the plane `nᵀQ1 = d`, the
    homography is `H21 = R + t nᵀ/d`. The panel shows `K⁻¹ H21 K`, scaled to a
    middle singular value of 1 (sign chosen so that `m2ᵀ H m1 > 0`), next to the
    true `R + t nᵀ/d`.
  - **Decomposition** into R, t/d and n: the SVD method (Faugeras; Ma et al.,
    *An Invitation to 3-D Vision*, Alg. 5.2) gives four candidates, two pairs
    `(t/d, n)` and `(−t/d, −n)`. The panel lists them all.
  - **Validity test**: every picked point must be in front of camera 1,
    `nᵀ m1 > 0` with `m1 = K⁻¹ q1` (since d > 0). Testing only the optical axis
    (`n3 > 0`) always leaves two; testing every point can rule out one more.
    When two pass, the decomposition is ambiguous. ★ marks the candidate closest
    to the truth.
  - With `t = 0` (pure rotation) there is one candidate, `R = H`, and n is
    undefined.
  - 3D, in camera 1's frame: camera 1 (white), the true camera 2 (cyan) and, per
    candidate, its camera 2 and plane in its colour (greens valid, reds invalid).
    Both true cameras show their image plane with the true (+) and noisy (×)
    pixels and the projection rays through the true pixels, coloured per point as
    in Plane pose.
  - Each candidate has its own colour and number (#1–#4) in the table and in the
    3D legend. Its plane `nᵀX = d` is drawn as a translucent square near the
    target, so the valid one lies on the board, while the invalid ones are tilted
    or mirrored (`−n`) planes that fail `nᵀm1 > 0`.
    H only gives t/d, so the candidates are drawn with the true d.

- **Metric scale**: the same scene and picking as "Two views of a plane". It
  shows why H21 gives no metric scale, and how a known target restores it.
  - `H21 = R + t nᵀ/d = R + (k t) nᵀ/(k d)` for any k > 0: scaling the baseline
    and the plane distance together gives the same images, so the decomposition
    only gives t/d.
  - H1 and H2 map the target, whose coordinates are in metres. So each gives its
    camera's metric pose, as in Plane pose. Then `R = R2 R1ᵀ`, `t = t2 − R t1`,
    and the plane in camera 1 is `n = R1 e3`, `d = nᵀ t1`.
  - The metric n also settles the decomposition's ambiguity: the valid H21
    candidate with the nearest n is chosen, and `t = (t/d)·d`.
  - A table compares d, t (with its error in mm), n and R from H21 and the
    recovered d, from H1 and H2, and the truth.
  - 3D, in camera 1's frame:
    - camera 1 (white) and the true camera 2 (cyan);
    - camera 2 from the metric poses (green), and from H21 with t = (t/d)·d
      (orange), with the plane at the recovered d (orange);
    - the distance d as a yellow segment from camera 1 to the plane along n.
  - Without a known target (a floor or a wall), any one known length (the
    distance to the plane, the baseline, or a length on the plane) fixes the
    scale the same way.

## Usage

All code runs in podman. Host-specific settings (port, memory limit, USB
device) are in `vars.env`.

```bash
./run.sh build                    # build runtime (rs-demo), test (rs-demo-test) and e2e (rs-demo-e2e) images
./run.sh serve                    # start on http://localhost:$PORT (synthetic scene if no RealSense is plugged in)
./run.sh test                     # full test suite
./run.sh test tests/test_geometry.py::test_zbuffer_keeps_nearest   # single test
./run.sh e2e                      # browser tests of the UI (headless Chromium; needs internet for model-viewer and KaTeX)
```

The source is bind-mounted at `/app`, so it is never baked into the images.
`serve` runs uvicorn with `--reload` (watching `app/`), and the page and script
are sent with `Cache-Control: no-cache`, so code changes take effect on a browser
reload without restarting the server.
If the camera is not accessible inside the container, check the host udev
rules for RealSense, or add `--privileged` to the `serve` command in `run.sh`.

## Layout

| Path | Purpose |
| --- | --- |
| `app/camera.py` | `RGBDFrame` and the `Camera` protocol |
| `app/realsense.py` | `RealSenseCamera`: pyrealsense2 pipeline, aligned depth, colour intrinsics |
| `app/synthetic.py` | `SyntheticCamera`: ray-cast ball-on-floor scene, the fallback without a RealSense |
| `app/mvg.py` | Reconstruction demos: `checkerboard`, `book_target`, `triangulate`, `homography_dlt`, `pose_from_homography`, `decompose_homography`, and the per-tab scenes (`triangulation`, `plane`, `twoview`, `metric_scale`) and their GLBs |
| `app/geometry.py` | `backproject`, `project` (z-buffer, optional splats), `warp`, `splat_sizes`, `rotation_matrix`, `skew`, `essential_matrix`, `fundamental_matrix`, `epipoles`, `epipolar_plane`, `frustum_segments`, `colorize_depth`, `to_glb` |
| `app/server.py` | FastAPI app: `/camera`, `/stream`, `/capture`, `/colorbar.png`, `/capture/{rgb,depth}.png`, `/capture/cloud.glb`, `/render.png`, `/matrices.json`. The last three take K (`fx fy cx cy s`) and extrinsics (`rx ry rz tx ty tz`, default 0). `/render.png` and `cloud.glb` also take `interp`. `/matrices.json` returns `P1 P2 E F e1 e2`. `cloud.glb?pair=true` shows the two Essential-tab views, and `hover_view hover_u hover_v` add the epipolar plane. `/mvg/{triangulation,plane,twoview,scale}.{json,glb}` serve the reconstruction demos: they take the pose (`rx … tz`), `noise` and the picked corner(s), `point` (triangulation) or comma-separated `ids` |
| `app/static/` | Frontend (plain HTML + JS): `app.js` for the capture tabs, `mvg.js` for the reconstruction demos (KaTeX from a CDN) |
| `tests/` | pytest suite using synthetic frames, so no camera is needed |
| `tests/e2e/` | Playwright browser tests of the point cloud pane, the tabs, epipolar hover and the reconstruction demos (skipped outside the e2e image) |
