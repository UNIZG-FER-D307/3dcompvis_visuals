const $ = (id) => document.getElementById(id);
const PARAMS = ["fx", "fy", "cx", "cy", "s"];
const EXT = ["rx", "ry", "rz", "tx", "ty", "tz"]; // degrees, metres
const ES_DEFAULT = { rx: 0, ry: 0, rz: 0, tx: -0.2, ty: 0, tz: 0 }; // camera 2 0.2 m right
let tab = "intr";
let size = null; // {w, h} of the capture
let epi = null; // {F, e1, e2} of the Essential tab's current R, t
let hover = null; // {view, u, v} hovered pixel in the Essential tab
let hoverTimer = null;
let original = null; // {fx, fy, cx, cy, s}
let timer = null;
let captureId = null; // cache-busts per-capture assets
let keepView = false; // keep the user's 3D view when the cloud reloads
let initialView = null; // {vfov, target: [x,y,z], corners}: framing for a new capture
const INITIAL_ORBIT = "30deg 75deg"; // off-axis: shows depth and the image plane behind the centre

function showLive() {
  $("result").hidden = true;
  $("mvg").hidden = true;
  $("live").hidden = false;
  $("stream").src = "stream";
}

function current() {
  return Object.fromEntries(PARAMS.map((p) => [p, parseFloat($(`sl-${p}`).value)]));
}

// Pose slider values; prefix "" is the Extrinsics tab, "es-" the Essential tab.
function currentPose(prefix = "") {
  return Object.fromEntries(EXT.map((p) => [p, parseFloat($(`sl-${prefix}${p}`).value)]));
}

const fmtMat = (label, M, fmt) => M.map((row, i) =>
  `${i ? " ".repeat(label.length) : label}[${row.map(fmt).join(" ")}]`).join("\n");
const fixed = (d, w) => (x) => x.toFixed(d).padStart(w);
const sci = (x) => x.toExponential(2).padStart(10);

// Camera->world, the inverse of X_cam = R X + t: X = R^T (X_cam - t), centre C = -R^T t.
function cameraToWorld(R, t, from, to) {
  const Rt = R[0].map((_, j) => R.map((row) => row[j]));
  const C = Rt.map((row) => -dot(row, t));
  return `${from} -> ${to}: X = R^T (X_cam - t)\n` +
    `${fmtMat("[R^T | -R^T t] = ", Rt.map((row, i) => [...row, C[i]]), fixed(3, 7))}\n` +
    `C = -R^T t = (${C.map((x) => x.toFixed(3)).join(", ")})`;
}

async function matrices(k, pose) {
  const resp = await fetch(`matrices.json?${new URLSearchParams({ ...k, ...pose })}`);
  if (!resp.ok) throw new Error(`matrices.json: HTTP ${resp.status}`);
  return resp.json();
}

const MATRICES_ERROR = (err) => `Could not load the matrices (${err.message}).\nIf the server predates this page, restart it.`;

// R = Rz @ Ry @ Rx, angles in degrees (same as app/geometry.py rotation_matrix).
function rotation({ rx, ry, rz }) {
  const [a, b, c] = [rx, ry, rz].map((d) => (d * Math.PI) / 180);
  const [ca, sa, cb, sb, cc, sc] = [Math.cos(a), Math.sin(a), Math.cos(b), Math.sin(b), Math.cos(c), Math.sin(c)];
  return [
    [cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa],
    [sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa],
    [-sb, cb * sa, cb * ca],
  ];
}

function rerender() {
  const k = current();
  const e = currentPose();
  for (const p of PARAMS) $(`val-${p}`).textContent = k[p].toFixed(1);
  for (const prefix of ["", "es-"]) {
    const pose = currentPose(prefix);
    for (const p of EXT) {
      $(`val-${prefix}${p}`).textContent = p[0] === "r" ? `${pose[p].toFixed(1)}°` : `${pose[p].toFixed(2)} m`;
    }
  }
  const f = (x) => x.toFixed(1).padStart(8);
  const kText =
    `K = [${f(k.fx)} ${f(k.s)} ${f(k.cx)}]\n` +
    `    [${f(0)} ${f(k.fy)} ${f(k.cy)}]\n` +
    `    [${f(0)} ${f(0)} ${f(1)}]`;
  $("kmat").textContent = kText;
  $("kmat-ro").textContent = `From the Intrinsics tab:\n${kText}`;
  const R = rotation(e), t = [e.tx, e.ty, e.tz];
  const g = (x) => x.toFixed(3).padStart(7);
  $("rtmat").textContent = R.map((row, i) =>
    `${i ? "        " : "[R|t] = "}[${row.map(g).join(" ")} | ${g(t[i])}]`).join("\n");
  $("c2w").textContent = cameraToWorld(R, t, "camera", "world");
  const interp = $("interp").checked;
  if (tab === "ess") return rerenderEssential(k, interp);
  const ext = tab === "extr" ? e : {};
  const q = new URLSearchParams({ ...k, ...ext, interp });
  $("rrgb").src = `render.png?${q}&kind=rgb`;
  $("rdepth").src = `render.png?${q}&kind=depth`;
  setCloud(`capture/cloud.glb?${q}&t=${captureId}`);
  if (tab === "extr") {
    matrices(k, e).then(({ P2 }) => {
      $("pmat").textContent = `P = K[R|t] =\n${fmtMat("", P2, fixed(2, 10))}`;
    }).catch((err) => { $("pmat").textContent = MATRICES_ERROR(err); });
  }
}

function rerenderEssential(k, interp) {
  const pose = currentPose("es-");
  $("view1").src = `render.png?${new URLSearchParams({ ...k, interp })}&kind=rgb`;
  $("view2").src = `render.png?${new URLSearchParams({ ...k, ...pose, interp })}&kind=rgb`;
  setCloud(essentialCloudUrl());
  matrices(k, pose).then((m) => {
    epi = m;
    const ep = (e) => (Math.abs(e[2]) < 1e-12 ? "at infinity" : `(${(e[0] / e[2]).toFixed(1)}, ${(e[1] / e[2]).toFixed(1)}) px`);
    const degenerate = m.E.flat().every((x) => x === 0);
    $("es-E").textContent = `E = [t]x R =\n${fmtMat("", m.E, fixed(4, 9))}`;
    $("es-F").textContent = `F = K^-T E K^-1 =\n${fmtMat("", m.F, sci)}`;
    $("es-epi").textContent = degenerate
      ? "t = 0: E = F = 0, epipolar geometry undefined"
      : `Epipoles:\ne1 = ${ep(m.e1)}  (camera 2 seen in view 1)\ne2 = ${ep(m.e2)}  (camera 1 seen in view 2)`;
    $("es-P").textContent = `${fmtMat("P1 = K[I|0] = ", m.P1, fixed(1, 8))}\n\n${fmtMat("P2 = K[R|t] = ", m.P2, fixed(1, 8))}` +
      `\n\n${cameraToWorld(rotation(pose), [pose.tx, pose.ty, pose.tz], "camera 2", "camera 1 (world)")}`;
    drawEpipolar();
  }).catch((err) => {
    epi = null;
    $("es-E").textContent = MATRICES_ERROR(err);
    for (const id of ["es-F", "es-epi", "es-P"]) $(id).textContent = "";
    drawEpipolar();
  });
}

function essentialCloudUrl() {
  const q = new URLSearchParams({ ...current(), ...currentPose("es-"), pair: true });
  if (hover) {
    q.set("hover_view", hover.view);
    q.set("hover_u", hover.u);
    q.set("hover_v", hover.v);
  }
  return `capture/cloud.glb?${q}&t=${captureId}`;
}

// Epipoles that fall inside the image are always marked. Epipolar lines for the
// hovered pixel x in view i: l = F x (or F^T x) in the other view, l = e_i x x
// (through the epipole) in the same view.
function drawEpipolar() {
  for (const i of [1, 2]) {
    const ctx = $(`epi${i}`).getContext("2d");
    ctx.clearRect(0, 0, size.w, size.h);
  }
  if (!epi || epi.F.flat().every((x) => x === 0)) return;
  if (hover) drawHover();
  for (const i of [1, 2]) {
    const e = i === 1 ? epi.e1 : epi.e2;
    if (Math.abs(e[2]) < 1e-12) continue;
    const [u, v] = [e[0] / e[2], e[1] / e[2]];
    if (u < 0 || u > size.w || v < 0 || v > size.h) continue;
    const ctx = $(`epi${i}`).getContext("2d");
    ctx.beginPath();
    ctx.arc(u, v, 7, 0, 2 * Math.PI);
    ctx.fillStyle = "#ff00ff";
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();
  }
}

function drawHover() {
  const x = [hover.u, hover.v, 1];
  const F = epi.F;
  const Ft = F[0].map((_, j) => F.map((row) => row[j]));
  const mul = (M, v) => M.map((row) => dot(row, v));
  const other = hover.view === 1 ? 2 : 1;
  const lines = { [other]: mul(hover.view === 1 ? F : Ft, x), [hover.view]: cross(hover.view === 1 ? epi.e1 : epi.e2, x) };
  for (const i of [1, 2]) {
    const ctx = $(`epi${i}`).getContext("2d");
    ctx.strokeStyle = ctx.fillStyle = "#ffdc00";
    ctx.lineWidth = 2;
    drawLine(ctx, lines[i]);
  }
  const ctx = $(`epi${hover.view}`).getContext("2d");
  ctx.beginPath();
  ctx.arc(hover.u, hover.v, 5, 0, 2 * Math.PI);
  ctx.fill();
}

// Line a·u + b·v + c = 0 across the whole image.
function drawLine(ctx, [a, b, c]) {
  ctx.beginPath();
  if (Math.abs(b) > Math.abs(a)) {
    ctx.moveTo(0, -c / b);
    ctx.lineTo(size.w, -(a * size.w + c) / b);
  } else {
    ctx.moveTo(-c / a, 0);
    ctx.lineTo(-(b * size.h + c) / a, size.h);
  }
  ctx.stroke();
}

function onHover(event) {
  const view = Number(event.currentTarget.dataset.view);
  const r = event.currentTarget.getBoundingClientRect();
  hover = { view, u: ((event.clientX - r.left) / r.width) * size.w, v: ((event.clientY - r.top) / r.height) * size.h };
  drawEpipolar();
  clearTimeout(hoverTimer);
  hoverTimer = setTimeout(() => setCloud(essentialCloudUrl()), 100);
}

function onLeave() {
  hover = null;
  drawEpipolar();
  clearTimeout(hoverTimer);
  hoverTimer = setTimeout(() => setCloud(essentialCloudUrl()), 100);
}

function setCloud(url) {
  const mv = $("cloud");
  if (mv.src === url) return;
  if (keepView) {
    // The field-of-view attribute persists; re-applying the read-back FOV would
    // compound model-viewer's aspect adjustment on every reload.
    const o = mv.getCameraOrbit(), t = mv.getCameraTarget();
    mv.addEventListener("load", () => {
      mv.cameraOrbit = `${o.theta}rad ${o.phi}rad ${o.radius}m`;
      mv.cameraTarget = `${t.x}m ${t.y}m ${t.z}m`;
      mv.jumpCameraToGoal();
    }, { once: true });
  } else {
    mv.addEventListener("load", fitView, { once: true });
  }
  keepView = true;
  mv.src = url;
}

// Frame the cloud to fill the pane: orbiting the centre of the corners' bounding
// box at the initial angles, find the smallest radius at which all corners (the
// frustum slab between the nearest and farthest depth, and the image plane;
// glTF axes) project inside the view.
function fitView() {
  const mv = $("cloud");
  const { vfov, target, corners } = initialView;
  const [theta, phi] = INITIAL_ORBIT.split(" ").map((a) => (parseFloat(a) * Math.PI) / 180);
  const tanV = Math.tan(vfov / 2);
  const tanH = tanV * (mv.clientWidth / mv.clientHeight);
  const fits = (r) => {
    const dir = [Math.sin(phi) * Math.sin(theta), Math.cos(phi), Math.sin(phi) * Math.cos(theta)];
    const pos = target.map((x, i) => x + r * dir[i]);
    const fwd = norm(sub(target, pos));
    const right = norm(cross(fwd, [0, 1, 0]));
    const up = cross(right, fwd);
    return corners.every((c) => {
      const d = sub(c, pos), z = dot(d, fwd);
      return z > 0 && Math.abs(dot(d, up)) <= z * tanV && Math.abs(dot(d, right)) <= z * tanH;
    });
  };
  let lo = 0, hi = 100 * Math.max(...corners.map((c) => Math.hypot(...sub(c, target))));
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2;
    if (fits(mid)) hi = mid; else lo = mid;
  }
  mv.cameraOrbit = `${INITIAL_ORBIT} ${hi}m`;
  mv.jumpCameraToGoal();
}

const sub = (a, b) => a.map((x, i) => x - b[i]);
const dot = (a, b) => a.reduce((acc, x, i) => acc + x * b[i], 0);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => a.map((x) => x / Math.sqrt(dot(a, a)));

// Corners of the frustum between depths z0 and z1, camera K, in glTF axes.
function frustumCorners(k, w, h, z0, z1) {
  const out = [];
  for (const z of [z0, z1]) {
    for (const [u, v] of [[0, 0], [w, 0], [w, h], [0, h]]) {
      const y = (v - k.cy) / k.fy;
      const x = (u - k.cx - k.s * y) / k.fx;
      out.push([x * z, -y * z, -z]);
    }
  }
  return out;
}

function buildSliders(w, h) {
  const fmax = Math.max(original.fx, original.fy);
  const ranges = {
    fx: [0.25 * original.fx, 3 * original.fx],
    fy: [0.25 * original.fy, 3 * original.fy],
    cx: [0, w],
    cy: [0, h],
    s: [-0.5 * fmax, 0.5 * fmax],
  };
  $("sliders").innerHTML = PARAMS.map((p) =>
    `<label class="slider">${p}<input type="range" id="sl-${p}" step="any"
       min="${ranges[p][0]}" max="${ranges[p][1]}"><span id="val-${p}"></span></label>`
  ).join("");
  for (const p of PARAMS) {
    $(`sl-${p}`).addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(rerender, 100);
    });
  }
}

function buildPoseSliders(container, prefix, defaults,
  range = (p) => (p[0] === "r" ? [-45, 45] : [-1, 1]),
  onInput = () => {
    clearTimeout(timer);
    timer = setTimeout(rerender, 100);
  }) {
  $(container).innerHTML = EXT.map((p) =>
    `<label class="slider">${p}<input type="range" id="sl-${prefix}${p}" step="any" value="${defaults[p]}"
       min="${range(p)[0]}" max="${range(p)[1]}"><span id="val-${prefix}${p}"></span></label>`
  ).join("");
  for (const p of EXT) $(`sl-${prefix}${p}`).addEventListener("input", onInput);
}

function resetExt() {
  for (const p of EXT) $(`sl-${p}`).value = 0;
  rerender();
}

function resetEs() {
  for (const p of EXT) $(`sl-es-${p}`).value = ES_DEFAULT[p];
  rerender();
}

function selectTab(name) {
  tab = name;
  for (const b of document.querySelectorAll(".tab[data-tab]")) b.classList.toggle("active", b.dataset.tab === name);
  for (const t of ["intr", "extr", "ess"]) $(`tab-${t}`).hidden = name !== t;
  const ess = name === "ess";
  for (const id of ["panel-rgb", "panel-depth", "panel-rrgb", "panel-rdepth", "legend-single"]) $(id).hidden = ess;
  for (const id of ["panel-view1", "panel-view2", "legend-pair", "panel-ess-mats"]) $(id).hidden = !ess;
  hover = null;
}

function buildTicks(dmin, dmax, n = 5) {
  $("ticks").innerHTML = Array.from({ length: n }, (_, i) => {
    const t = i / (n - 1);
    return `<span style="left:${100 * t}%">${(dmin + t * (dmax - dmin)).toFixed(2)}</span>`;
  }).join("");
}

function reset() {
  for (const p of PARAMS) $(`sl-${p}`).value = original[p];
  rerender();
}

async function capture() {
  const resp = await fetch("capture", { method: "POST" });
  if (!resp.ok) return alert(`Capture failed: ${await resp.text()}`);
  const { K, width, height, depth_min, depth_max, image_plane_depth } = await resp.json();
  original = { fx: K[0][0], fy: K[1][1], cx: K[0][2], cy: K[1][2], s: K[0][1] };
  size = { w: width, h: height };
  for (const i of [1, 2]) Object.assign($(`epi${i}`), { width, height });
  $("stream").src = ""; // stop the MJPEG connection
  captureId = Date.now();
  $("rgb").src = `capture/rgb.png?t=${captureId}`;
  $("depth").src = `capture/depth.png?t=${captureId}`;
  // A new capture starts from the RealSense's own view (see fitView).
  // The depth slab, plus the image plane behind the optical centre.
  const corners = frustumCorners(original, width, height, depth_min, depth_max).concat(
    frustumCorners(original, width, height, -image_plane_depth, -image_plane_depth).slice(0, 4));
  const lo = [0, 1, 2].map((i) => Math.min(...corners.map((c) => c[i])));
  const hi = [0, 1, 2].map((i) => Math.max(...corners.map((c) => c[i])));
  initialView = {
    vfov: 2 * Math.atan(height / 2 / original.fy),
    target: lo.map((x, i) => (x + hi[i]) / 2),
    corners,
  };
  const mv = $("cloud");
  mv.cameraTarget = initialView.target.map((x) => `${x}m`).join(" ");
  mv.fieldOfView = `${(initialView.vfov * 180) / Math.PI}deg`;
  keepView = false;
  buildTicks(depth_min, depth_max);
  buildSliders(width, height);
  for (const p of EXT) $(`sl-${p}`).value = 0;
  for (const p of EXT) $(`sl-es-${p}`).value = ES_DEFAULT[p];
  selectTab("intr");
  reset();
  $("live").hidden = true;
  $("result").hidden = false;
}

$("capture").addEventListener("click", capture);
$("back").addEventListener("click", showLive);
$("reset").addEventListener("click", reset);
$("interp").addEventListener("change", rerender);
$("reset-ext").addEventListener("click", resetExt);
$("reset-es").addEventListener("click", resetEs);
for (const el of document.querySelectorAll(".stack")) {
  el.addEventListener("mousemove", onHover);
  el.addEventListener("mouseleave", onLeave);
}
for (const b of document.querySelectorAll(".tab[data-tab]")) {
  b.addEventListener("click", () => {
    selectTab(b.dataset.tab);
    rerender();
  });
}
buildPoseSliders("ext-sliders", "", { rx: 0, ry: 0, rz: 0, tx: 0, ty: 0, tz: 0 });
buildPoseSliders("es-sliders", "es-", ES_DEFAULT);
fetch("camera").then((r) => r.json()).then(({ name }) => {
  if (name === "synthetic") $("live-title").textContent = "Live RGB (synthetic scene: no RealSense found)";
});
showLive();
