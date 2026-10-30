// Reconstruction demos (app/mvg.py) on synthetic targets, in the notation of the
// lecture notes: linear triangulation, plane pose from a homography, two views of
// a plane. Uses $, EXT, currentPose and buildPoseSliders from app.js.
const MVG_TABS = {
  tri: {
    url: "mvg/triangulation", prefix: "tri-", views: 2,
    defaults: { rx: 0, ry: 10, rz: 0, tx: -0.2, ty: 0, tz: 0 },
    titles: ["View A: P<sub>A</sub> = K[I|0]", "View B: P<sub>B</sub> = K[R|t]"],
    pose: "Camera B: X<sub>B</sub> = R&middot;X<sub>A</sub> + t (world = camera A)",
    help: "Click a corner in either view (it snaps to the nearest corner seen by both cameras); its correspondence in the other view is known. Click it again to clear.",
    legend: `<span class="swatch" style="background:#fff"></span> camera A
      <span class="swatch" style="background:#0ff"></span> camera B
      <span class="swatch" style="background:#ffdc00"></span> rays through q<sub>A</sub>, q<sub>B</sub>
      <span class="swatch" style="background:#00dc50"></span> triangulated Q (cross)
      <span class="swatch" style="background:#fff"></span> true corner (cross)`,
  },
  plane: {
    url: "mvg/plane", prefix: "pl-", views: 1,
    defaults: { rx: 25, ry: -15, rz: 5, tx: 0, ty: 0, tz: 1 },
    range: (p) => (p[0] === "r" ? [-45, 45] : p === "tz" ? [0.4, 2] : [-0.5, 0.5]),
    titles: ["Camera image: &lambda;q = H q<sub>R</sub>"],
    pose: "Target &rarr; camera: X<sub>cam</sub> = R&middot;X + t (target frame: Z normal to the board)",
    help: "Click at least 4 corners (snaps to the nearest); click a picked corner again to remove it.",
    legend: `<span class="swatch" style="background:#fff"></span> true camera and its image plane.
      Per picked point, in rainbow colours (picking order): the projection ray through the true pixel to the
      corner; on the image plane, + true pixel, &times; noisy pixel and the line between them (to scale: 1 px &asymp; 0.5 mm, zoom in).`,
  },
  two: {
    url: "mvg/twoview", prefix: "two-", views: 2,
    defaults: { rx: 0, ry: 10, rz: 0, tx: -0.2, ty: 0, tz: 0 },
    titles: ["View 1: P<sub>1</sub> = K[I|0] (fixed, board parallel to its image plane)", "View 2: P<sub>2</sub> = K[R|t]"],
    pose: "Camera 2: X<sub>2</sub> = R&middot;X<sub>1</sub> + t (world = camera 1)",
    help: "Click at least 4 corners in either view (snaps to corners seen by both cameras); click a picked corner again to remove it.",
    legend: `<span class="swatch" style="background:#fff"></span> camera 1
      <span class="swatch" style="background:#0ff"></span> true camera 2, each with its image plane.
      Per picked point, in rainbow colours (picking order): the projection rays through the true pixels to the
      corner; on the image planes, + true pixel, &times; noisy pixel and the line between them (to scale: 1 px &asymp; 0.5 mm, zoom in).`,
  },
  scale: {
    url: "mvg/scale", prefix: "sc-", views: 2,
    defaults: { rx: 0, ry: 10, rz: 0, tx: -0.2, ty: 0, tz: 0 },
    titles: ["View 1: P<sub>1</sub> = K[I|0] (fixed, board parallel to its image plane)", "View 2: P<sub>2</sub> = K[R|t]"],
    pose: "Camera 2: X<sub>2</sub> = R&middot;X<sub>1</sub> + t (world = camera 1)",
    help: "Click at least 4 corners in either view (snaps to corners seen by both cameras); click a picked corner again to remove it.",
    legend: `<span class="swatch" style="background:#fff"></span> camera 1
      <span class="swatch" style="background:#0ff"></span> true camera 2
      <span class="swatch" style="background:#00dc50"></span> camera 2 from the metric poses (H<sub>1</sub>, H<sub>2</sub>)
      <span class="swatch" style="background:#ff8c00"></span> camera 2 from H<sub>21</sub>: t = (t/d)&middot;d, and the plane at the recovered d
      <span class="swatch" style="background:#ffdc00"></span> d: camera 1 to the plane along n`,
  },
};
const SNAP_PX = 15; // image pixels
let mtab = "tri";
const picked = { tri: null, plane: [], two: [], scale: [] }; // corner id(s) per tab
let mdata = null; // last JSON of the current tab
let mvgSeq = 0; // drops responses to outdated requests
let mvgTimer = null;
let mvgKeepView = false;

function showMvg() {
  $("live").hidden = true;
  $("stream").src = ""; // stop the MJPEG connection
  $("mvg").hidden = false;
  selectMvgTab(mtab);
}

function selectMvgTab(name) {
  mtab = name;
  mdata = null;
  const t = MVG_TABS[name];
  for (const b of document.querySelectorAll(".mtab")) b.classList.toggle("active", b.dataset.mtab === name);
  for (const k of Object.keys(MVG_TABS)) $(`mvg-sliders-${k}`).hidden = k !== name;
  $("mvg-panel2").hidden = t.views === 1;
  t.titles.forEach((title, i) => { $(`mvg-v${i + 1}-title`).innerHTML = title; });
  $("mvg-pose-title").innerHTML = t.pose;
  $("mvg-help").textContent = t.help;
  $("mvg-legend").innerHTML = t.legend;
  mvgKeepView = false;
  refreshMvg();
}

function mvgQuery() {
  const t = MVG_TABS[mtab];
  const q = new URLSearchParams({ ...currentPose(t.prefix), noise: $("sl-noise").value });
  if (mtab === "tri") {
    if (picked.tri !== null) q.set("point", picked.tri);
  } else {
    q.set("ids", picked[mtab].join(","));
  }
  return q;
}

async function refreshMvg() {
  const t = MVG_TABS[mtab];
  const pose = currentPose(t.prefix);
  for (const p of EXT) {
    $(`val-${t.prefix}${p}`).textContent = p[0] === "r" ? `${pose[p].toFixed(1)}°` : `${pose[p].toFixed(2)} m`;
  }
  $("val-noise").textContent = `${parseFloat($("sl-noise").value).toFixed(1)} px`;
  const q = mvgQuery();
  setMvgCloud(`${t.url}.glb?${q}`);
  const seq = ++mvgSeq;
  const resp = await fetch(`${t.url}.json?${q}`);
  if (seq !== mvgSeq) return;
  if (!resp.ok) {
    $("mvg-math").textContent = `Could not load ${t.url}.json (HTTP ${resp.status}). If the server predates this page, restart it.`;
    return;
  }
  mdata = await resp.json();
  // The server drops points that are no longer visible.
  if (mtab === "tri") picked.tri = mdata.point;
  else picked[mtab] = mdata.ids;
  drawMvg();
  $("mvg-legend").innerHTML = t.legend + candidateLegend(mdata);
  $("mvg-math").innerHTML = { tri: mathTri, plane: mathPlane, two: mathTwo, scale: mathScale }[mtab](mdata);
}

// Each decomposition candidate in its colour, as in the tables.
function candidateLegend(d) {
  if (!d.candidates || mtab === "scale") return "";
  const what = mtab === "plane"
    ? "Pose candidates (camera frustums)"
    : "Decomposition candidates: each one's camera 2 and its plane n<sup>T</sup>X = d (drawn with the true d, since H only gives t/d), near the target";
  return `<br>${what}: ${d.candidates.map((c, i) =>
    `${swatch(c.rgb)} #${i + 1} ${c.valid ? "valid" : "invalid"}`).join(" ")}.
    Green = passes the test (target in front of the camera${mtab === "two" ? " 1" : ""}), red = fails.`;
}

function setMvgCloud(url) {
  const mv = $("mvg-cloud");
  if (mv.src === url) return;
  if (mvgKeepView) {
    const o = mv.getCameraOrbit(), t = mv.getCameraTarget();
    mv.addEventListener("load", () => {
      mv.cameraOrbit = `${o.theta}rad ${o.phi}rad ${o.radius}m`;
      mv.cameraTarget = `${t.x}m ${t.y}m ${t.z}m`;
      mv.jumpCameraToGoal();
    }, { once: true });
  } else {
    mv.addEventListener("load", () => {
      mv.cameraOrbit = "25deg 70deg auto";
      mv.cameraTarget = "auto auto auto";
      mv.jumpCameraToGoal();
    }, { once: true });
  }
  mvgKeepView = true;
  mv.src = url;
}

// Picked points per view (noisy pixels), in the order they were picked.
function pickedPixels(d) {
  if (mtab === "tri") return d.point === null ? [[], []] : [[d.q_A], [d.q_B]];
  if (mtab === "plane") return [d.q];
  return [d.q1, d.q2];
}

function drawMvg() {
  const pts = pickedPixels(mdata);
  mdata.views.forEach((view, i) => {
    const c = $(`mvg-c${i + 1}`);
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#8aa4c0";
    ctx.fillRect(0, 0, c.width, c.height);
    view.squares.forEach((sq, k) => {
      ctx.beginPath();
      sq.forEach((id, j) => (j ? ctx.lineTo : ctx.moveTo).call(ctx, ...view.q[id]));
      ctx.closePath();
      ctx.fillStyle = view.dark[k] ? "#3c3c3c" : "#ebebeb";
      ctx.fill();
    });
    ctx.fillStyle = "#1e90ff";
    view.q.forEach(([u, v], id) => {
      if (!mdata.selectable[id]) return;
      ctx.beginPath();
      ctx.arc(u, v, 2.5, 0, 2 * Math.PI);
      ctx.fill();
    });
    ctx.font = "bold 16px system-ui, sans-serif";
    ctx.lineWidth = 3;
    pts[i].forEach(([u, v], k) => {
      // Same rainbow colour as the point's ray in 3D (yellow in triangulation).
      const colour = mdata.colors ? `rgb(${mdata.colors[k]})` : "#ffdc00";
      ctx.beginPath();
      ctx.arc(u, v, 6, 0, 2 * Math.PI);
      ctx.strokeStyle = "#000";
      ctx.stroke();
      ctx.strokeStyle = colour;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.lineWidth = 3;
      if (mtab !== "tri") {
        ctx.strokeStyle = "#000";
        ctx.strokeText(k + 1, u + 8, v - 8);
        ctx.fillStyle = colour;
        ctx.fillText(k + 1, u + 8, v - 8);
      }
    });
  });
}

function onMvgClick(event) {
  if (!mdata) return;
  const c = event.currentTarget;
  const view = mdata.views[Number(c.dataset.view) - 1];
  const r = c.getBoundingClientRect();
  const u = ((event.clientX - r.left) / r.width) * c.width;
  const v = ((event.clientY - r.top) / r.height) * c.height;
  let best = null, bestDist = SNAP_PX;
  view.q.forEach(([x, y], id) => {
    const dist = Math.hypot(x - u, y - v);
    if (mdata.selectable[id] && dist < bestDist) [best, bestDist] = [id, dist];
  });
  if (best === null) return;
  if (mtab === "tri") {
    picked.tri = picked.tri === best ? null : best;
  } else {
    const i = picked[mtab].indexOf(best);
    if (i >= 0) picked[mtab].splice(i, 1);
    else picked[mtab].push(best);
  }
  refreshMvg();
}

// --- KaTeX -----------------------------------------------------------------
const tex = (s) => katex.renderToString(s, { displayMode: true, throwOnError: false });
const itex = (s) => katex.renderToString(s, { throwOnError: false });
function num(x, d = 3) {
  const a = Math.abs(x);
  if (a < 1e-12) return "0";
  if (a >= 1e4 || a < 1e-3) {
    const [m, e] = x.toExponential(2).split("e");
    return `${m}{\\cdot}10^{${parseInt(e)}}`;
  }
  return x.toFixed(d);
}
const bmat = (M, d) => `\\begin{bmatrix}${M.map((r) => r.map((x) => num(x, d)).join("&")).join("\\\\")}\\end{bmatrix}`;
const col = (v, d) => bmat(v.map((x) => [x]), d);
const row = (v, d) => `(${v.map((x) => num(x, d)).join(",\\ ")})`;
// The first `max` rows, then a row of dots.
const bmatRows = (M, d, max) => (M.length <= max ? bmat(M, d)
  : bmat(M.slice(0, max), d).replace("\\end{bmatrix}", `\\\\${M[0].map(() => "\\vdots").join("&")}\\end{bmatrix}`));
const mm = (a, b) => 1000 * Math.hypot(...a.map((x, i) => x - b[i]));
const check = (ok) => (ok ? "<b style=\"color:#080\">&#10004; valid</b>" : "<b style=\"color:#c00\">&#10008; invalid</b>");
const swatch = (rgb) => `<span class="swatch" style="background:rgb(${rgb})"></span>`;
const message = (d) => (d.message ? `<p><b>${d.message}</b></p>` : "");

function mathTri(d) {
  const out = [
    "<h2>Structure reconstruction: linear method</h2>",
    tex(String.raw`\lambda_A\mathbf{q}_A=\mathbf{P}_A\mathbf{Q},\qquad \lambda_B\mathbf{q}_B=\mathbf{P}_B\mathbf{Q}`),
    "<p>Known: q<sub>A</sub>, q<sub>B</sub>, P<sub>A</sub> and P<sub>B</sub>. How do we find Q?</p>",
  ];
  if (d.point === null) return out.join("") + "<p><b>Click a corner in either view.</b></p>";
  const err = mm(d.Q, d.Q_true);
  out.push(
    tex(`\\mathbf{q}_A=(x_A,y_A)=${row(d.q_A, 2)},\\qquad \\mathbf{q}_B=(x_B,y_B)=${row(d.q_B, 2)}`),
    tex(`\\mathbf{P}_A=\\mathbf{K}[\\mathbf{I}|\\mathbf{0}]=${bmat(d.P_A, 1)},\\quad \\mathbf{P}_B=\\mathbf{K}[\\mathbf{R}|\\mathbf{t}]=${bmat(d.P_B, 1)}`),
    "<p>The cross product removes the unknown scale &lambda; (only two of the three constraints are independent):</p>",
    tex(String.raw`[\mathbf{q}_i]_\times\mathbf{P}_i\mathbf{Q}=\mathbf{0}\ \Rightarrow\ \mathbf{A}_{4\times 4}\mathbf{Q}_{4\times 1}=\mathbf{0},\quad
      \mathbf{A}_{4\times 4}=\begin{bmatrix}y_A\mathbf{P}_{A[3,:]}-\mathbf{P}_{A[2,:]}\\x_A\mathbf{P}_{A[3,:]}-\mathbf{P}_{A[1,:]}\\
      y_B\mathbf{P}_{B[3,:]}-\mathbf{P}_{B[2,:]}\\x_B\mathbf{P}_{B[3,:]}-\mathbf{P}_{B[1,:]}\end{bmatrix}`),
    tex(`\\mathbf{A}_{4\\times 4}=${bmat(d.A, 2)}`),
    "<p>Solution: the right singular vector of the smallest singular value (&sigma;<sub>4</sub> = 0 without noise).</p>",
    tex(`\\mathbf{A}=\\mathbf{U}\\boldsymbol{\\Sigma}\\mathbf{V}^\\top,\\ \\ \\sigma=${row(d.s, 3)},\\qquad \\mathbf{Q}\\propto\\mathbf{V}_{:,4}=${col(d.Q_h, 5)}`),
    tex(`\\mathbf{Q}=${col(d.Q, 4)}\\,\\mathrm{m},\\qquad \\mathbf{Q}_\\mathrm{true}=${col(d.Q_true, 4)}\\,\\mathrm{m},\\qquad \\|\\mathbf{Q}-\\mathbf{Q}_\\mathrm{true}\\|=${num(err, 2)}\\,\\mathrm{mm}`),
    `<p>Reprojection error: ${d.reprojection.map((e) => e.toFixed(3)).join(" px (A), ")} px (B). The linear method is not the maximum-likelihood estimate; in practice it initialises an optimisation (bundle adjustment).</p>`,
  );
  return out.join("");
}

function pointsTable(heads, rows) {
  return `<table><tr>${heads.map((h) => `<th>${h}</th>`).join("")}</tr>${rows.map((r) =>
    `<tr>${r.map((x) => `<td>${x}</td>`).join("")}</tr>`).join("")}</table>`;
}

const fmtPt = (p, d) => `(${p.map((x) => x.toFixed(d)).join(", ")})`;

function homographyRows(d) {
  return [
    "<p>The cross product removes the unknown scale:</p>",
    tex(String.raw`\mathbf{q}\times(\mathbf{H}\mathbf{q}_R)=
      \begin{bmatrix}\mathbf{0}_3^\top&-\mathbf{q}_R^\top&y_q\mathbf{q}_R^\top\\\mathbf{q}_R^\top&\mathbf{0}_3^\top&-x_q\mathbf{q}_R^\top\\-y_q\mathbf{q}_R^\top&x_q\mathbf{q}_R^\top&\mathbf{0}_3^\top\end{bmatrix}
      \begin{bmatrix}\mathbf{H}_{1,:}^\top\\\mathbf{H}_{2,:}^\top\\\mathbf{H}_{3,:}^\top\end{bmatrix}=\mathbf{0}_3`),
    "<p>Two independent equations per pair (the first two rows), so for all pairs:</p>",
    tex(`\\mathbf{A}_{${d.A.length}\\times 9}\\,\\mathbf{h}_9=\\mathbf{0},\\qquad \\mathbf{A}=${bmatRows(d.A, 2, 8)}`),
    tex(`\\hat{\\mathbf{h}}_{alg}=\\arg\\min_{|\\mathbf{h}|=1}\\|\\mathbf{A}\\mathbf{h}\\|=\\mathbf{V}_{:,9},\\qquad \\sigma=${row(d.s, 3)}`),
  ];
}

function mathPlane(d) {
  const out = [
    "<h2>Homography: a plane and its image</h2>",
    tex(String.raw`\lambda\begin{bmatrix}x_q\\y_q\\1\end{bmatrix}=\mathbf{K}[\mathbf{R}|\mathbf{t}]\begin{bmatrix}X_Q\\Y_Q\\0\\1\end{bmatrix}
      =\mathbf{K}[\mathbf{R}_{:,1}\ \mathbf{R}_{:,2}\ \mathbf{t}]\begin{bmatrix}X_Q\\Y_Q\\1\end{bmatrix},\qquad \lambda\mathbf{q}=\mathbf{H}\mathbf{q}_R`),
  ];
  if (d.ids.length) {
    out.push(pointsTable(["i", "q<sub>R</sub> = (X<sub>Q</sub>, Y<sub>Q</sub>) [m]", "q = (x<sub>q</sub>, y<sub>q</sub>) [px]"],
      d.ids.map((_, i) => [i + 1, fmtPt(d.q_R[i], 3), fmtPt(d.q[i], 2)])));
  }
  out.push(message(d));
  if (!d.A) return out.join("");
  out.push(...homographyRows(d));
  if (!d.H) return out.join(""); // degenerate points
  out.push(tex(`\\mathbf{H}=${bmat(d.H, 4)}\\ (H_{33}=1),\\qquad \\mathbf{H}_\\mathrm{true}=${bmat(d.H_true, 4)}`));
  out.push(
    "<h2>From H to R and t</h2>",
    tex(String.raw`\mathbf{K}^{-1}\mathbf{H}=\mu\,[\mathbf{r}_1\ \mathbf{r}_2\ \mathbf{t}],\quad \mu=\pm\|\mathbf{K}^{-1}\mathbf{h}_1\|,\quad
      \mathbf{r}_3=\mathbf{r}_1\times\mathbf{r}_2,\quad \mathbf{R}=\mathbf{U}\mathbf{V}^\top\ \text{(nearest rotation to } [\mathbf{r}_1\ \mathbf{r}_2\ \mathbf{r}_3])`),
    `<p>H has an unknown scale and sign, so &mu; has two signs. Test: every picked target point must lie in front of the camera,
      ${itex(String.raw`(\mathbf{R}\mathbf{Q}+\mathbf{t})_z=r_{31}X_Q+r_{32}Y_Q+t_z>0`)}. Without noise ${itex(String.raw`\|\mathbf{r}_2\|=1`)} and
      ${itex(String.raw`\mathbf{r}_1\cdot\mathbf{r}_2=0`)}; with noise they are not, hence the nearest rotation.</p>`,
    pointsTable(["", "&mu;", "&#8214;r<sub>2</sub>&#8214;", "r<sub>1</sub>&middot;r<sub>2</sub>", "depths [m]", "test", "R", "t [m]"],
      d.candidates.map((c, i) => [`${swatch(c.rgb)} #${i + 1}`, itex(num(c.mu, 4)), itex(num(c.r2_norm, 4)), itex(num(c.r1_dot_r2, 4)),
        `${Math.min(...c.depths).toFixed(3)} &hellip; ${Math.max(...c.depths).toFixed(3)}`, check(c.valid),
        itex(bmat(c.R, 3)), itex(col(c.t, 3))])),
    tex(`\\mathbf{R}_\\mathrm{true}=${bmat(d.R_true, 3)},\\qquad \\mathbf{t}_\\mathrm{true}=${col(d.t_true, 3)}`),
  );
  return out.join("");
}

function mathTwo(d) {
  const out = [
    "<h2>Planar scene: the homography between two views</h2>",
    tex(String.raw`\mathbf{q}_1=\mathbf{H}_1\mathbf{q}_\pi,\quad \mathbf{q}_2=\mathbf{H}_2\mathbf{q}_\pi\ \Rightarrow\
      \mathbf{q}_2=\mathbf{H}_2\mathbf{H}_1^{-1}\mathbf{q}_1=\mathbf{H}_{21}\mathbf{q}_1`),
  ];
  if (d.ids.length) {
    out.push(pointsTable(["i", "q<sub>&pi;</sub> [m]", "q<sub>1</sub> [px]", "q<sub>2</sub> [px]"],
      d.ids.map((_, i) => [i + 1, fmtPt(d.q_R[i], 3), fmtPt(d.q1[i], 2), fmtPt(d.q2[i], 2)])));
  }
  out.push(message(d));
  if (!d.candidates) return out.join("");
  out.push(
    "<p>H<sub>1</sub> and H<sub>2</sub> are each estimated from target &harr; image pairs by the DLT (A<sub>2n&times;9</sub> h = 0), scaled to H<sub>33</sub> = 1:</p>",
    tex(`\\mathbf{H}_1=${bmat(d.H1, 4)},\\quad \\mathbf{H}_2=${bmat(d.H2, 4)}`),
    tex(`\\mathbf{H}_{21}=\\mathbf{H}_2\\mathbf{H}_1^{-1}=${bmat(d.H21, 4)},\\quad \\mathbf{H}_{21,\\mathrm{true}}=${bmat(d.H21_true, 4)}`),
    "<h2>Calibrated homography</h2>",
    tex(String.raw`\mathbf{P}_1=[\mathbf{I}|\mathbf{0}],\quad \mathbf{P}_2=[\mathbf{R}|\mathbf{t}],\quad \pi\equiv\mathbf{n}^\top\mathbf{Q}_1=d`),
    tex(String.raw`\mathbf{Q}_2=\mathbf{R}\mathbf{Q}_1+\mathbf{t}=\mathbf{R}\mathbf{Q}_1+\mathbf{t}\,\frac{\mathbf{n}^\top\mathbf{Q}_1}{d}
      =\Big(\mathbf{R}+\frac{\mathbf{t}\mathbf{n}^\top}{d}\Big)\mathbf{Q}_1\ \Rightarrow\ \mathbf{H}_{21}=\mathbf{R}+\frac{\mathbf{t}\mathbf{n}^\top}{d}`),
    "<p>In normalised coordinates, scaled so its middle singular value is 1 (the sign chosen so that m<sub>2</sub><sup>T</sup>&thinsp;H&thinsp;m<sub>1</sub> &gt; 0):</p>",
    tex(`\\hat{\\mathbf{H}}=\\frac{\\mathbf{K}^{-1}\\mathbf{H}_{21}\\mathbf{K}}{\\sigma_2}=${bmat(d.Hc, 4)},\\qquad
      \\Big(\\mathbf{R}+\\frac{\\mathbf{t}\\mathbf{n}^\\top}{d}\\Big)_\\mathrm{true}=${bmat(d.H_true, 4)}`),
    `<h2>Decomposition: R, t/d, n</h2>
     <p>SVD method (Faugeras; Ma et al., <i>An Invitation to 3-D Vision</i>, Alg. 5.2): up to four candidates, two pairs with
      (t/d, n) and (&minus;t/d, &minus;n). Test: every picked point must lie in front of camera 1,
      ${itex(String.raw`\mathbf{n}^\top\mathbf{m}_1>0`)} with ${itex(String.raw`\mathbf{m}_1=\mathbf{K}^{-1}\mathbf{q}_1`)} (since d &gt; 0).
      Testing only the optical axis (n<sub>3</sub> &gt; 0) always leaves two; testing every picked point can rule out one more.
      Here ${d.candidates.filter((c) => c.valid).length} of ${d.candidates.length} pass; when two pass, the decomposition is ambiguous.
      &#9733; = closest to the truth.</p>`,
    pointsTable(["", "R", "t/d", "n", "min n<sup>T</sup>m<sub>1</sub>", "test", ""],
      d.candidates.map((c, i) => [`${swatch(c.rgb)} #${i + 1}`, itex(bmat(c.R, 3)), itex(col(c.t, 3)),
        c.n ? itex(col(c.n, 3)) : "undefined (t = 0)",
        c.n ? Math.min(...c.n_m1).toFixed(3) : "&ndash;",
        check(c.valid), i === d.truth ? "&#9733;" : ""])),
    tex(`\\mathbf{R}_\\mathrm{true}=${bmat(d.R_true, 3)},\\quad (\\mathbf{t}/d)_\\mathrm{true}=${col(d.t_d_true, 3)},\\quad
      \\mathbf{n}_\\mathrm{true}=${col(d.n_true, 3)},\\quad d=${num(d.d_true, 3)}\\,\\mathrm{m}`),
  );
  return out.join("");
}

function mathScale(d) {
  const out = [
    "<h2>Metric scale: what H<sub>21</sub> cannot give</h2>",
    tex(String.raw`\mathbf{H}_{21}=\mathbf{R}+\frac{\mathbf{t}\mathbf{n}^\top}{d}=\mathbf{R}+\frac{(k\mathbf{t})\,\mathbf{n}^\top}{k\,d}\quad\forall k>0`),
    `<p>Scaling the baseline and the plane distance together gives the same images and the same H<sub>21</sub>, so its
      decomposition only gives t/d.</p>`,
  ];
  if (d.ids.length) {
    out.push(pointsTable(["i", "q<sub>&pi;</sub> [m]", "q<sub>1</sub> [px]", "q<sub>2</sub> [px]"],
      d.ids.map((_, i) => [i + 1, fmtPt(d.q_R[i], 3), fmtPt(d.q1[i], 2), fmtPt(d.q2[i], 2)])));
  }
  out.push(message(d));
  if (d.d_metric === undefined) return out.join("");
  const c = d.candidates[d.chosen];
  const err = (a, b) => `${num(mm(a, b), 1)}\\,\\mathrm{mm}`;
  out.push(
    "<h2>The known target fixes the scale</h2>",
    `<p>H<sub>1</sub> and H<sub>2</sub> map the target, whose coordinates q<sub>&pi;</sub> are in metres. So each gives its camera's
      metric pose, as in the Plane pose tab (the sign of &mu; with the target in front):</p>`,
    tex(String.raw`\mathbf{K}^{-1}\mathbf{H}_i=\mu_i[\mathbf{r}_1\ \mathbf{r}_2\ \mathbf{t}_i],\quad \mu_i=\pm\|\mathbf{K}^{-1}\mathbf{h}_1\|,\quad \mathbf{R}_i=[\mathbf{r}_1\ \mathbf{r}_2\ \mathbf{r}_1\times\mathbf{r}_2]`),
    tex(d.poses.map((p, i) => `\\mathbf{R}_${i + 1}=${bmat(p.R, 3)},\\ \\mathbf{t}_${i + 1}=${col(p.t, 3)}`).join(",\\qquad ")),
    "<p>The relative pose and the plane in camera 1 then follow in metres:</p>",
    tex(`\\mathbf{R}=\\mathbf{R}_2\\mathbf{R}_1^\\top=${bmat(d.R_metric, 3)},\\quad \\mathbf{t}=\\mathbf{t}_2-\\mathbf{R}\\mathbf{t}_1=${col(d.t_metric, 3)}\\,\\mathrm{m}`),
    tex(`\\mathbf{n}=\\mathbf{R}_1\\mathbf{e}_3=${col(d.n_metric, 3)},\\quad d=\\mathbf{n}^\\top\\mathbf{t}_1=${num(d.d_metric, 4)}\\,\\mathrm{m}`),
    "<h2>Back to H<sub>21</sub></h2>",
    `<p>The metric n also settles the decomposition's ambiguity: the valid candidate with the nearest n is
      ${swatch(c.rgb)} #${d.chosen + 1}${d.chosen === d.truth ? " (the true one, &#9733;)" : " (not the true one: too much noise)"}.
      Its t/d times the recovered d gives the metric translation:</p>`,
    tex(`\\mathbf{t}=\\frac{\\mathbf{t}}{d}\\cdot d=${col(c.t, 4)}\\cdot ${num(d.d_metric, 4)}=${col(d.t_scaled, 4)}\\,\\mathrm{m}`),
    pointsTable(["", "from H<sub>21</sub> and the recovered d", "from H<sub>1</sub>, H<sub>2</sub>", "true"], [
      ["d [m]", "&ndash;", itex(num(d.d_metric, 4)), itex(num(d.d_true, 4))],
      ["t [m]", itex(col(d.t_scaled, 4)), itex(col(d.t_metric, 4)), itex(col(d.t_true, 4))],
      ["error of t", itex(err(d.t_scaled, d.t_true)), itex(err(d.t_metric, d.t_true)), ""],
      ["n", c.n ? itex(col(c.n, 3)) : "undefined", itex(col(d.n_metric, 3)), itex(col(d.n_true, 3))],
      ["R", itex(bmat(c.R, 3)), itex(bmat(d.R_metric, 3)), itex(bmat(d.R_true, 3))],
    ]),
    `<p>Without a known target (e.g. a floor or a wall), any one known length fixes the scale the same way: the distance to
      the plane, the baseline &#8214;t&#8214;, or a length on the plane.</p>`,
  );
  return out.join("");
}

for (const [name, t] of Object.entries(MVG_TABS)) {
  buildPoseSliders(`mvg-sliders-${name}`, t.prefix, t.defaults, t.range, () => {
    clearTimeout(mvgTimer);
    mvgTimer = setTimeout(refreshMvg, 100);
  });
}
$("sl-noise").addEventListener("input", () => {
  clearTimeout(mvgTimer);
  mvgTimer = setTimeout(refreshMvg, 100);
});
$("mvg-open").addEventListener("click", showMvg);
$("mvg-back").addEventListener("click", showLive);
$("mvg-clear").addEventListener("click", () => {
  picked[mtab] = mtab === "tri" ? null : [];
  refreshMvg();
});
for (const b of document.querySelectorAll(".mtab")) b.addEventListener("click", () => selectMvgTab(b.dataset.mtab));
for (const c of document.querySelectorAll("canvas.pick")) c.addEventListener("click", onMvgClick);
