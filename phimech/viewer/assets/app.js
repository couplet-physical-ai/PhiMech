/*
 * PhiMech viewer.
 *
 * One module, no build step. Loads /api/manifest, loads every part's STL, and presents the
 * assembly with gates, measurements and an exploded slider.
 *
 * Two things here are load-bearing and easy to break:
 *   1. CAD is Z-up. `camera.up` is (0,0,1) and every helper is rotated into the XY ground plane.
 *      Leave three.js at its Y-up default and the enclosure lies on its side.
 *   2. Parts are exported in ONE shared coordinate frame (each part shares the assembly origin,
 *      the base at 0..22.1). Centring each mesh on its own bbox would stack them all at the
 *      origin and the assembly would fall apart, so the *assembly* is centred once, as a whole.
 */

import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { STLLoader } from './vendor/STLLoader.js';

// --------------------------------------------------------------------------- state

const EXPLODE_STEPS = [0, 35, 70, 100];
const BED_DEFAULT = [360, 360];

const state = {
  manifest: null,
  parts: [],          // { name, mesh, edges, color, baseZ, height, visible, explodeDir, tris }
  group: null,        // holds the parts; centred as a whole
  assemblyBox: null,  // THREE.Box3 of the un-exploded assembly, in group-local space
  explode: 0,
  wireframe: false,
  showGrid: true,
  showBed: false,
  showShadow: true,
  showEdges: false,
  spin: false,
  currentBuild: null,
  tris: 0,
};

/** The agent/test contract. Kept in sync by `publish()` — never assign it piecemeal. */
window.__phimech = { ready: false, parts: [], tris: 0, error: null };

function publish(extra) {
  window.__phimech = Object.assign({
    ready: false,
    parts: state.parts.map((p) => ({
      name: p.name,
      visible: p.visible,
      triangles: p.tris,
      color: p.color,
      size_mm: p.size_mm || null,
    })),
    tris: state.tris,
    error: null,
    build: state.currentBuild,
    project: state.manifest ? state.manifest.project : null,
    title: state.manifest ? state.manifest.title : null,
    ok: state.manifest ? state.manifest.ok : null,
    explode: state.explode,
    wireframe: state.wireframe,
    grid: state.showGrid,
    bed: state.showBed,
    gates: state.manifest ? state.manifest.gates_summary : null,
    warnings: state.manifest ? state.manifest.warnings || [] : [],
  }, extra || {});
}

// --------------------------------------------------------------------------- dom helpers

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function fmt(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  return n.toFixed(digits).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1');
}

function fmtSize(size, digits = 1) {
  if (!size || size.length !== 3) return '—';
  return size.map((v) => fmt(v, digits)).join(' × ');
}

function clearNode(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

// --------------------------------------------------------------------------- three.js setup

const canvas = $('#canvas');
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  // Required for a reliable toDataURL() screenshot: without it the drawing buffer may already
  // be cleared by the time we read it back.
  preserveDrawingBuffer: true,
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
// Under 1.0: the lighting rig below is deliberately bright so curved walls and countersinks read,
// and a lower exposure keeps the part from blowing out to white against the near-black backdrop.
renderer.toneMappingExposure = 0.9;
renderer.outputColorSpace = THREE.SRGBColorSpace;

/**
 * Vertical gradient backdrop. A flat fill is what makes a viewer look like a debug harness; a
 * subtle studio falloff (lighter near the horizon, near-black overhead) costs one 2x256 texture
 * and does most of the work of making the part look photographed rather than dumped on screen.
 */
function makeBackdrop() {
  const cv = document.createElement('canvas');
  cv.width = 2;
  cv.height = 256;
  const ctx = cv.getContext('2d');
  const gradient = ctx.createLinearGradient(0, 0, 0, 256);
  gradient.addColorStop(0.0, '#161922');
  gradient.addColorStop(0.55, '#150101');
  gradient.addColorStop(1.0, '#20232D');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 2, 256);
  const texture = new THREE.CanvasTexture(cv);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

const scene = new THREE.Scene();
scene.background = makeBackdrop();
// A little depth fog so the grid fades out instead of ending in a hard line.
scene.fog = new THREE.Fog(0x150101, 420, 1500);

const camera = new THREE.PerspectiveCamera(38, 1, 0.5, 8000);
camera.up.set(0, 0, 1); // CAD is Z-up.

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.075;
controls.rotateSpeed = 0.85;
controls.zoomSpeed = 0.9;
controls.panSpeed = 0.85;
controls.minDistance = 5;
controls.maxDistance = 4000;
controls.autoRotateSpeed = 1.1;

// -- lighting: a three-point rig plus a hemisphere fill. Tuned so a grey print reads as a solid
// -- object with visible edges, not a flat silhouette.
scene.add(new THREE.HemisphereLight(0xcdd0d5, 0x150101, 0.34));
scene.add(new THREE.AmbientLight(0xffffff, 0.09));

// A soft light almost straight down gives the flat top face a gradient to catch, which is the
// difference between "CAD model" and "grey polygon soup". Deliberately a DirectionalLight and
// not a RectAreaLight: the latter renders black unless three's RectAreaLightUniformsLib is also
// vendored, and one more vendor file is not worth it.
const topLight = new THREE.DirectionalLight(0xf6f8fa, 0.5);
topLight.position.set(-60, -90, 500);
scene.add(topLight);

const keyLight = new THREE.DirectionalLight(0xffffff, 1.45);
keyLight.position.set(180, -260, 340);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.bias = -0.0012;
keyLight.shadow.normalBias = 0.7;
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0xff7212, 0.24);
fillLight.position.set(-260, 140, 130);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0xff4111, 0.72);
rimLight.position.set(-90, 300, -180);
scene.add(rimLight);

// Headlight tracked to the camera in `tick()`. Without it, an exploded tray interior — the very
// thing the explode slider exists to show — falls into near-black shadow: standoff posts and
// screw bosses become invisible exactly when the user wants to inspect them.
const headLight = new THREE.DirectionalLight(0xffffff, 0.42);
scene.add(headLight);
scene.add(headLight.target); // a DirectionalLight's target must be in the graph to be honoured

// -- helper layers -----------------------------------------------------------------------

const helpers = new THREE.Group();
scene.add(helpers);

const gridGroup = new THREE.Group();
const bedGroup = new THREE.Group();
const shadowGroup = new THREE.Group();
helpers.add(gridGroup, bedGroup, shadowGroup);

state.group = new THREE.Group();
scene.add(state.group);

/** Sprite label so the axes are readable as X/Y/Z rather than three coloured sticks. */
function makeLabel(text, color) {
  const size = 128;
  const cv = document.createElement('canvas');
  cv.width = size;
  cv.height = size;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = color;
  ctx.font = 'bold 84px -apple-system, Helvetica, Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, size / 2, size / 2 + 4);
  const texture = new THREE.CanvasTexture(cv);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture, transparent: true, depthTest: false, depthWrite: false,
  }));
  sprite.renderOrder = 5;
  return sprite;
}

/**
 * Grid in the XY plane + a labelled X/Y/Z triad.
 *
 * `origin` is where the triad is planted. It is deliberately NOT the model centre: axes drawn
 * through the middle of the assembly skewer the geometry and read as a defect. Like every CAD
 * package, the triad sits at the near-front-left corner of the footprint, outside the part.
 */
function buildGrid(span, origin, bedSpan = 0) {
  clearGroup(gridGroup);
  const step = span > 300 ? 50 : span > 120 ? 20 : 10;
  // Reach past the build plate too, so the grid still gives the eye a ground plane when the
  // 360 mm envelope is shown around a 67 mm part.
  const reach = Math.max(span * 2.2, bedSpan * 1.06);
  const extent = Math.ceil(reach / step) * step;

  const grid = new THREE.GridHelper(extent, extent / step, 0x525866, 0x20232d);
  grid.rotateX(Math.PI / 2); // GridHelper is XZ by default; PhiMech's ground plane is XY.
  grid.material.transparent = true;
  grid.material.opacity = 0.5;
  gridGroup.add(grid);

  const base = origin ? origin.clone() : new THREE.Vector3(0, 0, 0);
  const axisLen = Math.max(span * 0.3, 14);
  const axes = [
    { dir: new THREE.Vector3(1, 0, 0), color: 0xff1010, label: 'X' },
    { dir: new THREE.Vector3(0, 1, 0), color: 0xff7212, label: 'Y' },
    { dir: new THREE.Vector3(0, 0, 1), color: 0xcdd0d5, label: 'Z' },
  ];
  for (const axis of axes) {
    const end = base.clone().addScaledVector(axis.dir, axisLen);
    const geometry = new THREE.BufferGeometry().setFromPoints([base.clone(), end]);
    gridGroup.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({
      color: axis.color, transparent: true, opacity: 0.95,
    })));
    const label = makeLabel(axis.label, '#' + new THREE.Color(axis.color).getHexString());
    label.position.copy(end).addScaledVector(axis.dir, axisLen * 0.13);
    label.scale.setScalar(Math.max(span * 0.055, 5));
    gridGroup.add(label);
  }
  gridGroup.visible = state.showGrid;
}

/** The real Max4 envelope: a 360 × 360 mm plate outline the part can be judged against. */
function buildBed(sizeMM) {
  clearGroup(bedGroup);
  const [bx, by] = sizeMM && sizeMM.length >= 2 ? sizeMM : BED_DEFAULT;
  const hx = bx / 2;
  const hy = by / 2;

  const outline = new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-hx, -hy, 0), new THREE.Vector3(hx, -hy, 0),
      new THREE.Vector3(hx, hy, 0), new THREE.Vector3(-hx, hy, 0),
    ]),
    new THREE.LineBasicMaterial({ color: 0xff4111, transparent: true, opacity: 0.95 }),
  );
  bedGroup.add(outline);

  const plate = new THREE.Mesh(
    new THREE.PlaneGeometry(bx, by),
    new THREE.MeshBasicMaterial({
      color: 0xff4111, transparent: true, opacity: 0.05, side: THREE.DoubleSide,
      depthWrite: false,
    }),
  );
  plate.position.z = -0.05;
  bedGroup.add(plate);

  bedGroup.add(makeTextPlane(`${fmt(bx, 0)} × ${fmt(by, 0)} mm bed`, hx, hy));

  bedGroup.visible = state.showBed;
}

/** A flat text plane laid in the XY plane (sprites always face the camera; this should not). */
function makeTextPlane(text, hx, hy) {
  const cv = document.createElement('canvas');
  cv.width = 512;
  cv.height = 64;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = 'rgba(255,65,17,0.95)';
  ctx.font = 'bold 34px ui-monospace, Menlo, monospace';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 6, 34);
  const texture = new THREE.CanvasTexture(cv);
  texture.colorSpace = THREE.SRGBColorSpace;
  const width = Math.max(hx * 0.9, 60);
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(width, width / 8),
    new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false }),
  );
  mesh.position.set(-hx + width / 2 + 4, hy + width / 14, 0.1);
  return mesh;
}

/** Soft ground shadow so the assembly does not float in a void. */
function buildShadowCatcher(span) {
  clearGroup(shadowGroup);
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(span * 6, span * 6),
    new THREE.ShadowMaterial({ color: 0x000000, opacity: 0.55 }),
  );
  plane.position.z = -0.08; // just under z=0 so it never z-fights the part's own floor
  plane.receiveShadow = true;
  shadowGroup.add(plane);
  shadowGroup.visible = state.showShadow;
}

function clearGroup(group) {
  for (let i = group.children.length - 1; i >= 0; i -= 1) {
    const child = group.children[i];
    group.remove(child);
    disposeObject(child);
  }
}

function disposeObject(object) {
  object.traverse?.((node) => {
    if (node.geometry) node.geometry.dispose();
    const materials = Array.isArray(node.material) ? node.material : [node.material];
    for (const material of materials) {
      if (!material) continue;
      if (material.map) material.map.dispose();
      material.dispose();
    }
  });
}

// --------------------------------------------------------------------------- loading

const loader = new STLLoader();

function loadSTL(url) {
  return new Promise((resolve, reject) => {
    loader.load(
      url,
      (geometry) => resolve(geometry),
      undefined,
      () => reject(new Error(`could not load ${url}`)),
    );
  });
}

async function fetchJSON(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${url} → HTTP ${response.status}`);
  return response.json();
}

/** Tear down the current scene contents so a build switch does not leak GPU memory. */
function clearParts() {
  clearGroup(state.group);
  state.parts = [];
  state.tris = 0;
  state.assemblyBox = null;
}

async function loadBuild(buildId) {
  showLoading('Loading manifest…');
  hideError();
  clearParts();
  publish({ ready: false });

  let manifest;
  const query = buildId && buildId !== '.' ? `?build=${encodeURIComponent(buildId)}` : '';
  try {
    manifest = await fetchJSON(`/api/manifest${query}`);
  } catch (err) {
    fail('Could not load the manifest', String(err.message || err), [
      'Is PhiMech still running? Check the terminal that started it.',
      'Try /api/manifest directly in a new tab to see the raw error.',
    ]);
    return;
  }

  state.manifest = manifest;
  state.currentBuild = manifest.build_id || buildId || '.';
  renderChrome(manifest);
  renderBuildErrors(manifest);
  renderGates(manifest);
  renderStats(manifest);
  renderBuildInfo(manifest);
  renderWarnings(manifest.warnings);

  const parts = (manifest.parts || []).filter((p) => p.stl);
  if (!parts.length) {
    const hints = [
      `Build directory: ${manifest.build_dir || '?'}`,
      'Run: bin/phimech synth build examples/parametric-box.json -o out/final/box',
    ];
    if (manifest.previews && manifest.previews.length) {
      hints.push(`Only preview-only parts were found: ${
        manifest.previews.map((p) => p.name).join(', ')}. Those have no STL to render.`);
    }
    fail('No 3D geometry in this build',
      (manifest.warnings || []).join(' · ') || 'The manifest lists no part with an STL file.',
      hints);
    return;
  }

  // Load every STL, then decide the assembly transform once all bboxes are known.
  const failures = [];
  for (let i = 0; i < parts.length; i += 1) {
    const part = parts[i];
    showLoading(`Loading ${part.name} (${i + 1}/${parts.length})…`);
    try {
      const geometry = await loadSTL(part.stl);
      addPart(part, geometry);
    } catch (err) {
      failures.push(`${part.name}: ${err.message || err}`);
    }
  }

  if (!state.parts.length) {
    fail('Every STL failed to load', failures.join(' · '), [
      'The manifest referenced files that the server would not serve.',
      'Check the build directory still holds the .stl files it reported.',
    ]);
    return;
  }
  if (failures.length) {
    renderWarnings((manifest.warnings || []).concat(failures));
  }

  layoutAssembly();
  buildPartList();
  renderPartStats(manifest);
  applyExplode(state.explode);
  setView('iso');
  hideLoading();
  publish({ ready: true });
}

/** Add one part mesh. Geometry keeps its authored coordinates — the group carries the offset. */
function addPart(part, geometry) {
  geometry.computeBoundingBox();
  geometry.computeVertexNormals();

  const color = new THREE.Color(part.color || '#FF4111');
  // Physical-ish FDM plastic: matte, faintly sheened, no metal. `clearcoat` on the standard
  // material is what stops a flat top face from reading as a solid untextured blob.
  const material = new THREE.MeshPhysicalMaterial({
    color,
    metalness: 0.0,
    roughness: 0.55,
    clearcoat: 0.22,
    clearcoatRoughness: 0.6,
    // No `sheen`: it lifts the mid-tones enough to wash a mid-grey out to near-white under this
    // three-point rig, which is exactly the "untextured blob" look to avoid.
    flatShading: false,
    side: THREE.DoubleSide, // STLs from CSG can have an inconsistent winding on cut faces
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.name = part.name;

  const edgeGeometry = new THREE.EdgesGeometry(geometry, 24);
  const edges = new THREE.LineSegments(edgeGeometry, new THREE.LineBasicMaterial({
    color: 0x150101, transparent: true, opacity: 0.34,
  }));
  edges.visible = state.showEdges;
  mesh.add(edges);

  const box = geometry.boundingBox;
  const tris = geometry.index
    ? geometry.index.count / 3
    : geometry.attributes.position.count / 3;

  state.group.add(mesh);
  state.parts.push({
    name: part.name,
    mesh,
    edges,
    material,
    color: '#' + color.getHexString(),
    visible: true,
    tris,
    size_mm: part.size_mm || [box.max.x - box.min.x, box.max.y - box.min.y, box.max.z - box.min.z],
    minZ: box.min.z,
    maxZ: box.max.z,
    centerZ: (box.min.z + box.max.z) / 2,
    height: Math.max(box.max.z - box.min.z, 0.001),
    manifestPart: part,
    explodeOffset: 0,
  });
  state.tris += tris;
}

/**
 * Position the assembly, size the helpers, and work out each part's explode direction.
 *
 * The assembly is centred in X/Y and dropped so its lowest point sits on z=0 (the bed). Parts
 * keep their relative positions, which is the whole point — the lid must stay above the tray.
 */
function layoutAssembly() {
  state.group.position.set(0, 0, 0);
  for (const part of state.parts) part.mesh.position.set(0, 0, 0);

  const box = new THREE.Box3().setFromObject(state.group);
  state.assemblyBox = box.clone();
  const center = box.getCenter(new THREE.Vector3());
  // Centre X/Y, rest on the plate in Z.
  state.group.position.set(-center.x, -center.y, -box.min.z);

  const size = box.getSize(new THREE.Vector3());
  const span = Math.max(size.x, size.y, size.z, 20);

  // Triad origin in scene space: front-left corner of the footprint, backed off a little so the
  // lines never touch the part.
  const margin = Math.max(span * 0.14, 8);
  const triadOrigin = new THREE.Vector3(
    -size.x / 2 - margin,
    -size.y / 2 - margin,
    0,
  );

  const bed = state.manifest?.bed?.size_mm || BED_DEFAULT;
  buildGrid(span, triadOrigin, Math.max(bed[0] || 0, bed[1] || 0));
  buildBed(bed);
  buildShadowCatcher(span);

  keyLight.position.set(span * 1.4, -span * 2.0, span * 2.6);
  const shadowCam = keyLight.shadow.camera;
  const reach = span * 2.4;
  shadowCam.left = -reach;
  shadowCam.right = reach;
  shadowCam.top = reach;
  shadowCam.bottom = -reach;
  shadowCam.near = 1;
  shadowCam.far = span * 12;
  shadowCam.updateProjectionMatrix();

  scene.fog.near = span * 5;
  scene.fog.far = span * 22;

  computeExplodeOffsets();
  updateViewportBadge();
}

/**
 * Per-part travel for a 100% explode.
 *
 * Parts are ordered bottom-up by their centre Z; each successive part moves further than the one
 * below it, by a distance proportional to the two parts' heights. That keeps a thin lid clearly
 * clear of a tall tray without flinging a 3 mm plate into the next postcode.
 */
function computeExplodeOffsets() {
  const ordered = state.parts.slice().sort((a, b) => a.centerZ - b.centerZ);
  let cumulative = 0;
  ordered.forEach((part, index) => {
    if (index === 0) {
      part.explodeOffset = 0;
      return;
    }
    const below = ordered[index - 1];
    cumulative += Math.max(below.height, part.height) * 0.85 + 6;
    part.explodeOffset = cumulative;
  });
}

function applyExplode(percent) {
  state.explode = Math.max(0, Math.min(100, Number(percent) || 0));
  const factor = state.explode / 100;
  for (const part of state.parts) {
    part.mesh.position.z = part.explodeOffset * factor;
  }
  const value = $('#explode-value');
  if (value) value.textContent = `${Math.round(state.explode)}%`;
  const slider = $('#explode');
  if (slider) {
    slider.value = String(Math.round(state.explode));
    slider.style.setProperty('--fill', `${state.explode}%`);
  }
  publish({ ready: window.__phimech.ready });
}

// --------------------------------------------------------------------------- camera

function assemblyRadius() {
  if (!state.assemblyBox) return 100;
  const size = state.assemblyBox.getSize(new THREE.Vector3());
  return Math.max(size.length() / 2, 12);
}

/** Target the centre of what is actually on screen, including any explode offset. */
function currentCenter() {
  const box = new THREE.Box3().setFromObject(state.group);
  if (box.isEmpty()) return new THREE.Vector3(0, 0, 0);
  return box.getCenter(new THREE.Vector3());
}

/**
 * Camera distance that fits `radius` on screen, honouring BOTH the vertical FOV and the derived
 * horizontal one. Using the vertical FOV alone crops a wide part off the sides of the canvas.
 */
/** Width of canvas not hidden behind the floating panel, as a fraction of the whole. */
function visibleFraction() {
  const panel = document.querySelector('#panel');
  const width = window.innerWidth;
  if (!panel || !width) return 1;
  const right = panel.getBoundingClientRect().right;
  if (right <= 0 || right >= width * 0.75) return 1;
  return (width - right) / width;
}

function fitDistance(radius, margin = 1.16) {
  const fovRad = (camera.fov * Math.PI) / 180;
  // Fit against the *usable* aspect: the panel covers ~30% of the canvas, so fitting to the full
  // width leaves the part crammed against the top and right edges.
  const aspect = Math.max(camera.aspect * visibleFraction(), 0.35);
  const vertical = radius / Math.sin(fovRad / 2);
  const horizontal = radius / Math.sin(Math.atan(Math.tan(fovRad / 2) * aspect));
  return Math.max(vertical, horizontal) * margin;
}

/**
 * Slide the camera + target right so the model sits centred in the *visible* canvas rather than
 * behind the floating info panel. Without this the part is pushed toward the right edge and
 * clips, because the canvas is full-bleed under a ~336 px opaque panel.
 */
function offsetForPanel(distance) {
  const panel = document.querySelector('#panel');
  if (!panel) return;
  const panelRight = panel.getBoundingClientRect().right;
  if (panelRight <= 0 || panelRight >= window.innerWidth * 0.75) return;
  const fovRad = (camera.fov * Math.PI) / 180;
  const worldPerPixel = (2 * Math.tan(fovRad / 2) * distance) / Math.max(window.innerHeight, 1);
  const shift = (panelRight / 2) * worldPerPixel;
  // Translating camera AND target together keeps the target at screen centre and slides the
  // *world* the opposite way, so the shift is applied away from the panel, not toward it.
  const lateral = new THREE.Vector3()
    .subVectors(camera.position, controls.target)
    .cross(camera.up)
    .normalize()
    .multiplyScalar(shift);
  camera.position.add(lateral);
  controls.target.add(lateral);
}

function setView(preset) {
  const center = currentCenter();
  const box = new THREE.Box3().setFromObject(state.group);
  const radius = box.isEmpty()
    ? assemblyRadius()
    : Math.max(box.getSize(new THREE.Vector3()).length() / 2, 10);
  const distance = fitDistance(radius);

  const directions = {
    iso: new THREE.Vector3(0.78, -1, 0.72),
    top: new THREE.Vector3(0, -0.0001, 1),
    front: new THREE.Vector3(0, -1, 0),
    right: new THREE.Vector3(1, 0, 0),
    left: new THREE.Vector3(-1, 0, 0),
    back: new THREE.Vector3(0, 1, 0),
  };
  const dir = (directions[preset] || directions.iso).clone().normalize();
  camera.position.copy(center).addScaledVector(dir, distance);
  controls.target.copy(center);
  offsetForPanel(distance);
  camera.near = Math.max(distance / 800, 0.1);
  camera.far = distance * 40;
  camera.updateProjectionMatrix();
  controls.update();
  markActiveView(preset);
}

function fitAll() {
  const box = new THREE.Box3().setFromObject(state.group);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.length() / 2, 10);
  const fit = fitDistance(radius, 1.08);
  const dir = camera.position.clone().sub(controls.target);
  if (dir.lengthSq() < 1e-6) dir.set(0.78, -1, 0.72);
  dir.normalize();
  camera.position.copy(center).addScaledVector(dir, fit);
  controls.target.copy(center);
  offsetForPanel(fit);
  camera.near = Math.max(fit / 800, 0.1);
  camera.far = fit * 40;
  camera.updateProjectionMatrix();
  controls.update();
  markActiveView('fit');
}

function resetView() {
  state.explode = 0;
  applyExplode(0);
  controls.autoRotate = false;
  const spin = $('#t-spin');
  if (spin) spin.checked = false;
  state.spin = false;
  setView('iso');
}

function markActiveView(preset) {
  $$('[data-view]').forEach((btn) => {
    btn.classList.toggle('is-active', btn.dataset.view === preset);
  });
}

// --------------------------------------------------------------------------- panel rendering

function renderChrome(manifest) {
  const title = manifest.title || manifest.project || 'Untitled build';
  $('#build-title').textContent = title;
  document.title = `PhiMech · ${title}`;
  const rev = $('#build-rev');
  if (manifest.rev) {
    rev.textContent = `rev ${manifest.rev}`;
    rev.hidden = false;
  } else {
    rev.hidden = true;
  }

  const bits = [];
  if (manifest.generated) bits.push(manifest.generated.replace('T', ' '));
  if (manifest.backend) bits.push(`${manifest.backend} backend`);
  if (manifest.elapsed_s) bits.push(`built in ${fmt(manifest.elapsed_s, 1)} s`);
  if (manifest.phimech_version) bits.push(`PhiMech ${manifest.phimech_version}`);
  $('#build-stamp').textContent = bits.join('  ·  ');

  const summary = manifest.gates_summary || { failed: 0, total: 0 };
  const errorCount = (manifest.errors || []).length;
  const verdict = $('#verdict');
  const text = $('#verdict-text');
  const icon = verdict.querySelector('.verdict-icon');
  verdict.classList.remove('verdict-pass', 'verdict-fail', 'verdict-unknown');
  if (manifest.ok === true) {
    verdict.classList.add('verdict-pass');
    icon.textContent = '✓';
    text.textContent = `${summary.passed}/${summary.total} gates pass`;
  } else if (manifest.ok === false) {
    verdict.classList.add('verdict-fail');
    icon.textContent = '✕';
    // A build can fail with every gate green (e.g. a backend render error). Saying
    // "0 gates failed" in that case is worse than saying nothing, so name the real cause.
    if (summary.failed) {
      text.textContent = `${summary.failed} gate${summary.failed === 1 ? '' : 's'} failed`;
    } else if (errorCount) {
      text.textContent = `${errorCount} build error${errorCount === 1 ? '' : 's'}`;
    } else {
      text.textContent = 'build failed';
    }
  } else {
    verdict.classList.add('verdict-unknown');
    icon.textContent = '·';
    text.textContent = 'no gate data';
  }

  const badge = $('#tab-gate-badge');
  const badgeCount = summary.failed + errorCount;
  if (badgeCount) {
    badge.textContent = String(badgeCount);
    badge.hidden = false;
  } else {
    badge.hidden = true;
  }

  // Failed gates and build errors are the thing you must not miss: jump straight to them.
  if (manifest.ok === false) activateTab('gates');

  buildSwitcher(manifest);
}

function buildSwitcher(manifest) {
  const select = $('#build-select');
  const builds = manifest.builds || [];
  clearNode(select);
  if (!builds.length) {
    const option = el('option', null, manifest.project || 'build');
    option.value = manifest.build_id || '.';
    select.appendChild(option);
    select.disabled = true;
    return;
  }
  select.disabled = builds.length < 2;
  for (const build of builds) {
    const mark = build.ok === false ? '  ✕' : build.ok === true ? '' : '  ·';
    const option = el('option', null, `${build.name}${mark}`);
    option.value = build.id;
    if (build.id === (manifest.build_id || '.')) option.selected = true;
    select.appendChild(option);
  }
}

function buildPartList() {
  const list = $('#part-list');
  clearNode(list);
  state.parts.forEach((part, index) => {
    const row = el('li', 'part-row');
    row.dataset.part = part.name;
    row.tabIndex = 0;
    row.setAttribute('role', 'button');
    row.setAttribute('aria-pressed', 'true');

    const swatch = el('span', 'swatch');
    swatch.style.background = part.color;
    swatch.style.color = part.color;

    const name = el('span', 'part-name', part.name);
    const dims = el('span', 'part-dims', fmtSize(part.size_mm, 1));

    const eye = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    eye.setAttribute('class', 'eye');
    eye.setAttribute('viewBox', '0 0 20 20');
    eye.innerHTML = '<path d="M1.8 10S5 4.8 10 4.8 18.2 10 18.2 10 15 15.2 10 15.2 1.8 10 1.8 10z"'
      + ' fill="none" stroke="currentColor" stroke-width="1.4"/>'
      + '<circle cx="10" cy="10" r="2.3" fill="currentColor"/>';

    row.append(swatch, name, dims, eye);
    row.title = `${part.name} — ${part.tris.toLocaleString()} triangles. Click to toggle`
      + ` (shift+${index + 1}).`;
    row.addEventListener('click', () => togglePart(part.name));
    row.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        togglePart(part.name);
        event.preventDefault();
      }
    });
    list.appendChild(row);
  });
  syncPartRows();
}

function togglePart(name, force) {
  const part = state.parts.find((p) => p.name === name);
  if (!part) return;
  part.visible = force === undefined ? !part.visible : Boolean(force);
  part.mesh.visible = part.visible;
  syncPartRows();
  publish({ ready: window.__phimech.ready });
}

function syncPartRows() {
  $$('#part-list .part-row').forEach((row) => {
    const part = state.parts.find((p) => p.name === row.dataset.part);
    row.classList.toggle('is-hidden', part ? !part.visible : false);
    row.setAttribute('aria-pressed', String(part ? part.visible : false));
  });
}

/** First meaningful line of a multi-KB backend traceback, for a one-line summary. */
function firstLine(text, limit = 190) {
  const line = String(text).split('\n').map((s) => s.trim()).find(Boolean) || '';
  return line.length > limit ? `${line.slice(0, limit)}…` : line;
}

function renderBuildErrors(manifest) {
  const errors = manifest.errors || [];
  const group = $('#error-group');
  const list = $('#build-errors');
  clearNode(list);
  group.hidden = !errors.length;
  $('#error-count').textContent = errors.length
    ? `${errors.length} error${errors.length === 1 ? '' : 's'}` : '';

  for (const error of errors) {
    const li = el('li', 'build-error');
    // Backend errors are multi-KB qhull/OCC dumps. Show the first line; keep the rest one click
    // away rather than destroying the panel layout.
    const head = el('div', 'build-error-head', firstLine(error));
    li.appendChild(head);
    if (String(error).includes('\n')) {
      const details = document.createElement('details');
      details.appendChild(el('summary', null, 'full backend output'));
      details.appendChild(el('pre', 'build-error-pre', error));
      li.appendChild(details);
    }
    list.appendChild(li);
  }
}

function renderGates(manifest) {
  const list = $('#gate-list');
  clearNode(list);
  const gates = manifest.gates || [];
  const summary = manifest.gates_summary || {};
  $('#gate-summary').textContent = gates.length
    ? `${summary.passed || 0} pass · ${summary.failed || 0} fail · ${summary.skipped || 0} skipped`
    : 'no gate data';

  if (!gates.length) {
    list.appendChild(el('li', 'muted', 'This build has no gate results in report.json.'));
    return;
  }

  // Failures first — you should never have to scroll to find out what broke.
  const ordered = gates.slice().sort((a, b) => {
    const rank = (g) => (g.passed === false ? 0 : g.passed === true ? 1 : 2);
    return rank(a) - rank(b);
  });

  for (const gate of ordered) {
    const cls = gate.passed === true ? 'gate gate-pass'
      : gate.passed === false ? 'gate gate-fail' : 'gate gate-skip';
    const row = el('li', cls);
    row.appendChild(el('span', 'gate-mark',
      gate.passed === true ? '✓' : gate.passed === false ? '✕' : '–'));

    const body = el('div', 'gate-body');
    body.appendChild(el('div', 'gate-name', gate.name));
    if (gate.message) body.appendChild(el('div', 'gate-msg', gate.message));
    row.appendChild(body);

    let numeric = '';
    if (gate.value !== null && gate.value !== undefined) {
      numeric = fmt(gate.value, 2);
      if (gate.threshold !== null && gate.threshold !== undefined) {
        numeric += ` / ${fmt(gate.threshold, 2)}`;
      }
    } else if (gate.passed === null) {
      numeric = 'skipped';
    }
    row.appendChild(el('span', 'gate-num', numeric));
    list.appendChild(row);
  }
}

function kv(dl, key, value, accent) {
  if (value === null || value === undefined || value === '' || value === '—') return;
  dl.appendChild(el('dt', null, key));
  const dd = el('dd', accent ? 'accent' : null, value);
  dl.appendChild(dd);
}

function renderStats(manifest) {
  const totals = manifest.totals || {};
  const material = manifest.material || {};
  const dl = $('#totals');
  clearNode(dl);
  kv(dl, 'Parts', `${totals.parts ?? 0} (${totals.printable_parts ?? 0} printable)`);
  if (totals.volume_cm3 !== null && totals.volume_cm3 !== undefined) {
    kv(dl, 'Volume', `${fmt(totals.volume_cm3, 2)} cm³`, true);
  }
  if (totals.mass_g !== null && totals.mass_g !== undefined) {
    kv(dl, 'Est. mass', `${fmt(totals.mass_g, 1)} g`, true);
  }
  kv(dl, 'Triangles', (totals.triangles || state.tris || 0).toLocaleString());
  kv(dl, 'Density', material.mass_basis);
  kv(dl, 'Process', material.process ? String(material.process).toUpperCase() : null);
  if (material.nozzle_mm) kv(dl, 'Nozzle', `${fmt(material.nozzle_mm, 2)} mm`);
  if (material.layer_height_mm) kv(dl, 'Layer', `${fmt(material.layer_height_mm, 2)} mm`);
  kv(dl, 'Bed', manifest.bed
    ? `${fmtSize([...(manifest.bed.size_mm || []), 0].slice(0, 2).concat([0]), 0)
      .split(' × ').slice(0, 2).join(' × ')} mm — ${manifest.bed.name}`
    : null);

  const printGroup = $('#print-group');
  const printDl = $('#print-stats');
  clearNode(printDl);
  const print = manifest.print;
  if (print && (print.print_time_text || print.filament_g || print.files)) {
    printGroup.hidden = false;
    kv(printDl, 'Print time', print.print_time_text, true);
    if (print.filament_g) kv(printDl, 'Filament', `${fmt(print.filament_g, 1)} g`, true);
    for (const file of print.files || []) {
      printDl.appendChild(el('dt', null, 'File'));
      const dd = el('dd');
      const link = el('a', null, file.name);
      link.href = file.url;
      link.download = file.name;
      link.style.color = 'var(--accent)';
      dd.appendChild(link);
      printDl.appendChild(dd);
    }
  } else {
    printGroup.hidden = true;
  }
}

function renderPartStats(manifest) {
  const host = $('#part-stats');
  clearNode(host);
  for (const part of state.parts) {
    const meta = part.manifestPart || {};
    const card = el('div', 'stat-card');
    card.style.setProperty('--card-color', part.color);

    const heading = el('h3', null, part.name);
    if (meta.printable) heading.appendChild(el('span', 'badge', 'printable'));
    card.appendChild(heading);

    const dl = el('dl', 'kv');
    kv(dl, 'Size', `${fmtSize(part.size_mm, 2)} mm`);
    if (meta.volume_cm3 !== null && meta.volume_cm3 !== undefined) {
      kv(dl, 'Volume', `${fmt(meta.volume_cm3, 2)} cm³`);
    }
    if (meta.mass_g !== null && meta.mass_g !== undefined) {
      kv(dl, 'Est. mass', `${fmt(meta.mass_g, 2)} g`, true);
    }
    if (meta.area_mm2) kv(dl, 'Surface', `${fmt(meta.area_mm2 / 100, 1)} cm²`);
    kv(dl, 'Triangles', part.tris.toLocaleString());
    if (meta.solids !== null && meta.solids !== undefined) kv(dl, 'Solids', String(meta.solids));
    if (meta.bytes) kv(dl, 'STL', `${fmt(meta.bytes / 1048576, 2)} MB`);
    if (meta.formats && meta.formats.length) kv(dl, 'Formats', meta.formats.join(', '));
    card.appendChild(dl);
    host.appendChild(card);
  }
}

function renderBuildInfo(manifest) {
  const encl = manifest.enclosure || {};
  const dl = $('#encl-stats');
  clearNode(dl);
  kv(dl, 'Style', encl.style);
  kv(dl, 'Fastening', encl.fastening);
  if (encl.outer_size_mm) kv(dl, 'Outer', `${fmtSize(encl.outer_size_mm, 1)} mm`, true);
  if (encl.cavity_size_mm) kv(dl, 'Cavity', `${fmtSize(encl.cavity_size_mm, 1)} mm`);
  if (encl.wall_mm) kv(dl, 'Wall', `${fmt(encl.wall_mm, 2)} mm`);
  if (encl.floor_mm) kv(dl, 'Floor', `${fmt(encl.floor_mm, 2)} mm`);
  if (encl.standoffs !== null && encl.standoffs !== undefined) {
    kv(dl, 'Standoffs', String(encl.standoffs));
  }
  if (encl.screws !== null && encl.screws !== undefined) kv(dl, 'Screws', String(encl.screws));
  if (encl.cutouts !== null && encl.cutouts !== undefined) kv(dl, 'Cutouts', String(encl.cutouts));
  if (encl.estimated_material_mm3) {
    kv(dl, 'Material est.', `${fmt(encl.estimated_material_mm3 / 1000, 1)} cm³`);
  }
  if (!dl.children.length) dl.appendChild(el('dt', 'muted', 'No enclosure summary in the report.'));

  const comps = manifest.components || [];
  const compList = $('#comp-list');
  clearNode(compList);
  $('#comp-group').hidden = !comps.length;
  for (const comp of comps) {
    const li = el('li');
    li.appendChild(el('span', 'comp-ref', comp.ref));
    li.appendChild(document.createTextNode(comp.name || ''));
    const detail = [];
    if (comp.size_mm) detail.push(`${fmtSize(comp.size_mm, 1)} mm`);
    if (comp.source) detail.push(comp.source);
    if (detail.length) li.appendChild(el('span', 'comp-dims', detail.join('  ·  ')));
    compList.appendChild(li);
  }

  // Errors live in their own red group on the Gates tab; notes stay advisory here.
  const notes = manifest.notes || [];
  const noteList = $('#notes-list');
  clearNode(noteList);
  $('#notes-group').hidden = !notes.length;
  for (const note of notes) noteList.appendChild(el('li', null, note));

  const stages = manifest.stages || [];
  const stageList = $('#stage-list');
  clearNode(stageList);
  $('#stages-group').hidden = !stages.length;
  const slowest = stages.reduce((max, s) => Math.max(max, s.seconds || 0), 0) || 1;
  for (const stage of stages) {
    const li = el('li', 'stage');
    const left = el('div');
    const nameRow = el('div', 'stage-name');
    const dot = el('span', `stage-dot${stage.ok === false ? ' bad' : ''}`);
    nameRow.append(dot, el('span', null, stage.stage));
    left.appendChild(nameRow);
    const bar = el('div', 'stage-bar');
    const fill = el('i');
    fill.style.width = `${Math.max(((stage.seconds || 0) / slowest) * 100, 1.5)}%`;
    bar.appendChild(fill);
    left.appendChild(bar);
    if (stage.note) {
      const note = el('div', 'muted', stage.note);
      note.style.fontSize = '10px';
      left.appendChild(note);
    }
    li.appendChild(left);
    li.appendChild(el('span', 'stage-sec', `${fmt(stage.seconds, 2)}s`));
    stageList.appendChild(li);
  }

  const links = $('#file-links');
  clearNode(links);
  const files = manifest.files || {};
  const named = [
    ['REPORT.md', files.report_md],
    ['report.json', files.report_json],
    ['design.json', files.design_json],
  ];
  let any = false;
  for (const [label, url] of named) {
    if (!url) continue;
    any = true;
    const link = el('a', 'btn btn-sm', label);
    link.href = url;
    link.target = '_blank';
    link.rel = 'noopener';
    links.appendChild(link);
  }
  for (const part of state.parts) {
    const meta = part.manifestPart || {};
    if (!meta.stl) continue;
    any = true;
    const link = el('a', 'btn btn-sm', `${part.name}.stl`);
    link.href = meta.stl;
    link.download = '';
    links.appendChild(link);
  }
  $('#links-group').hidden = !any;
}

function renderWarnings(warnings) {
  const host = $('#warnings');
  const items = (warnings || []).filter(Boolean);
  clearNode(host);
  if (!items.length) {
    host.hidden = true;
    return;
  }
  host.hidden = false;
  host.appendChild(el('b', null, `${items.length} warning${items.length === 1 ? '' : 's'}`));
  const ul = el('ul');
  for (const warning of items) ul.appendChild(el('li', null, warning));
  host.appendChild(ul);
}

function updateViewportBadge() {
  const badge = $('#viewport-dims');
  if (!badge) return;
  if (!state.assemblyBox) {
    badge.textContent = '—';
    return;
  }
  const size = state.assemblyBox.getSize(new THREE.Vector3());
  badge.textContent = `${fmt(size.x, 1)} × ${fmt(size.y, 1)} × ${fmt(size.z, 1)} mm`
    + `   ·   ${state.tris.toLocaleString()} tris`
    + `   ·   ${state.parts.length} part${state.parts.length === 1 ? '' : 's'}`;
}

// --------------------------------------------------------------------------- overlays

function showLoading(text) {
  const overlay = $('#loading');
  $('#loading-text').textContent = text || 'Loading…';
  overlay.hidden = false;
}

function hideLoading() {
  $('#loading').hidden = true;
}

function hideError() {
  $('#error').hidden = true;
}

/** Any dead end lands here: a titled, explained, actionable card — never a blank black screen. */
function fail(title, body, hints) {
  hideLoading();
  $('#error-title').textContent = title;
  $('#error-body').textContent = body || '';
  const list = $('#error-hints');
  clearNode(list);
  for (const hint of hints || []) list.appendChild(el('li', null, hint));
  $('#error').hidden = false;
  publish({ ready: false, error: `${title}: ${body || ''}`.trim() });
}

// --------------------------------------------------------------------------- toggles

function setWireframe(on) {
  state.wireframe = Boolean(on);
  for (const part of state.parts) {
    part.material.wireframe = state.wireframe;
    // The dark edge overlay is noise on top of a wireframe.
    part.edges.visible = state.showEdges && !state.wireframe;
  }
  $('#t-wire').checked = state.wireframe;
  publish({ ready: window.__phimech.ready });
}

function setGrid(on) {
  state.showGrid = Boolean(on);
  gridGroup.visible = state.showGrid;
  $('#t-grid').checked = state.showGrid;
  publish({ ready: window.__phimech.ready });
}

function setBed(on) {
  const wasOn = state.showBed;
  state.showBed = Boolean(on);
  bedGroup.visible = state.showBed;
  $('#t-bed').checked = state.showBed;
  // The plate is 360 x 360 mm against a ~67 mm part: at the part's own fitted zoom the envelope
  // is entirely off-screen, so turning it on has to pull the camera back to show it. Turning it
  // off returns to the part. Without this the toggle looks broken.
  if (state.showBed && !wasOn) frameBed();
  else if (!state.showBed && wasOn) fitAll();
  publish({ ready: window.__phimech.ready });
}

/** Pull back far enough to show the whole build plate with the part sitting on it. */
function frameBed() {
  const bed = (state.manifest && state.manifest.bed && state.manifest.bed.size_mm) || BED_DEFAULT;
  const radius = Math.max(bed[0], bed[1]) * 0.72;
  const center = new THREE.Vector3(0, 0, 0);
  const distance = fitDistance(radius, 1.05);
  const dir = camera.position.clone().sub(controls.target);
  if (dir.lengthSq() < 1e-6) dir.set(0.78, -1, 0.72);
  dir.normalize();
  camera.position.copy(center).addScaledVector(dir, distance);
  controls.target.copy(center);
  offsetForPanel(distance);
  camera.near = Math.max(distance / 800, 0.1);
  camera.far = distance * 40;
  camera.updateProjectionMatrix();
  controls.update();
}

function setShadow(on) {
  state.showShadow = Boolean(on);
  shadowGroup.visible = state.showShadow;
  renderer.shadowMap.enabled = state.showShadow;
  for (const part of state.parts) part.material.needsUpdate = true;
  $('#t-shadow').checked = state.showShadow;
}

function setEdges(on) {
  state.showEdges = Boolean(on);
  for (const part of state.parts) part.edges.visible = state.showEdges && !state.wireframe;
  $('#t-edges').checked = state.showEdges;
}

function setSpin(on) {
  state.spin = Boolean(on);
  controls.autoRotate = state.spin;
  $('#t-spin').checked = state.spin;
}

function cycleExplode() {
  const current = state.explode;
  const next = EXPLODE_STEPS.find((step) => step > current + 0.5);
  applyExplode(next === undefined ? 0 : next);
}

// --------------------------------------------------------------------------- screenshot

function screenshot() {
  // Render synchronously first: with preserveDrawingBuffer the buffer is valid, but rendering
  // here guarantees the PNG matches exactly what is on screen right now.
  renderer.render(scene, camera);
  const url = renderer.domElement.toDataURL('image/png');
  const name = [
    'PhiMech',
    (state.manifest && state.manifest.project) || 'build',
    new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19),
  ].join('-') + '.png';
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// --------------------------------------------------------------------------- tabs

function activateTab(name) {
  $$('.tab').forEach((tab) => tab.classList.toggle('is-active', tab.dataset.tab === name));
  $$('.tabpane').forEach((pane) => pane.classList.toggle('is-active', pane.dataset.pane === name));
}

// --------------------------------------------------------------------------- wiring

function wire() {
  $('#explode').addEventListener('input', (event) => applyExplode(event.target.value));

  $('#t-wire').addEventListener('change', (e) => setWireframe(e.target.checked));
  $('#t-grid').addEventListener('change', (e) => setGrid(e.target.checked));
  $('#t-bed').addEventListener('change', (e) => setBed(e.target.checked));
  $('#t-shadow').addEventListener('change', (e) => setShadow(e.target.checked));
  $('#t-edges').addEventListener('change', (e) => setEdges(e.target.checked));
  $('#t-spin').addEventListener('change', (e) => setSpin(e.target.checked));

  $$('[data-view]').forEach((btn) => {
    btn.addEventListener('click', () => {
      if (btn.dataset.view === 'fit') fitAll();
      else setView(btn.dataset.view);
    });
  });
  $('#btn-reset').addEventListener('click', resetView);
  $('#btn-shot').addEventListener('click', screenshot);

  $('#btn-all-parts').addEventListener('click', () => {
    const anyHidden = state.parts.some((p) => !p.visible);
    for (const part of state.parts) togglePart(part.name, anyHidden);
  });

  $$('.tab').forEach((tab) => tab.addEventListener('click', () => activateTab(tab.dataset.tab)));

  $('#build-select').addEventListener('change', (event) => {
    loadBuild(event.target.value);
  });

  $('#error-retry').addEventListener('click', () => loadBuild(state.currentBuild));

  const help = $('#help');
  $('#btn-help').addEventListener('click', () => { help.hidden = !help.hidden; });
  $('#help-close').addEventListener('click', () => { help.hidden = true; });
  help.addEventListener('click', (event) => {
    if (event.target === help) help.hidden = true;
  });

  window.addEventListener('keydown', onKey);
  window.addEventListener('resize', resize);
}

function onKey(event) {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  const tag = (event.target && event.target.tagName) || '';
  if (tag === 'INPUT' && event.target.type !== 'range') return;
  if (tag === 'SELECT' || tag === 'TEXTAREA') return;

  const key = event.key;

  // shift+1..9 toggles part visibility, so the digits stay free for the view presets.
  if (event.shiftKey && /^[1-9!@#$%^&*(]$/.test(key)) {
    const index = '123456789'.indexOf(key) >= 0
      ? '123456789'.indexOf(key)
      : '!@#$%^&*('.indexOf(key);
    if (index >= 0 && state.parts[index]) {
      togglePart(state.parts[index].name);
      event.preventDefault();
    }
    return;
  }

  switch (key) {
    case 'r': case 'R': resetView(); break;
    case 'w': case 'W': setWireframe(!state.wireframe); break;
    case 'g': case 'G': setGrid(!state.showGrid); break;
    case 'b': case 'B': setBed(!state.showBed); break;
    case 'e': case 'E': cycleExplode(); break;
    case 's': case 'S': screenshot(); break;
    case 'f': case 'F': fitAll(); break;
    case '1': setView('iso'); break;
    case '2': setView('top'); break;
    case '3': setView('front'); break;
    case '4': setView('right'); break;
    case '?': $('#help').hidden = !$('#help').hidden; break;
    case 'Escape': $('#help').hidden = true; break;
    default: return;
  }
  event.preventDefault();
}

function resize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
}

// --------------------------------------------------------------------------- main loop

function syncHeadLight() {
  headLight.position.copy(camera.position);
  headLight.target.position.copy(controls.target);
  headLight.target.updateMatrixWorld();
}

function tick() {
  controls.update();
  syncHeadLight();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

/**
 * Scriptable handle for agents and tests (`forca eval`).
 *
 * `renderNow()` matters more than it looks: browsers pause requestAnimationFrame in a tab that
 * is not being composited, so a headless/backgrounded capture otherwise reads a stale or empty
 * drawing buffer. Forcing one synchronous render makes `grab()` reliable regardless of window
 * visibility.
 */
window.__phimechApi = {
  renderNow() {
    controls.update();
    syncHeadLight();
    renderer.render(scene, camera);
    return true;
  },
  setExplode(percent) {
    applyExplode(percent);
    this.renderNow();
    return state.explode;
  },
  setView(preset) {
    if (preset === 'fit') fitAll();
    else setView(preset);
    this.renderNow();
    return preset;
  },
  reset() {
    resetView();
    this.renderNow();
    return true;
  },
  toggle(name, on) {
    const map = {
      wireframe: setWireframe, grid: setGrid, bed: setBed, shadow: setShadow,
      edges: setEdges, spin: setSpin,
    };
    const fn = map[name];
    if (!fn) return null;
    const current = {
      wireframe: state.wireframe, grid: state.showGrid, bed: state.showBed,
      shadow: state.showShadow, edges: state.showEdges, spin: state.spin,
    }[name];
    fn(on === undefined ? !current : Boolean(on));
    this.renderNow();
    return true;
  },
  togglePart(name, on) {
    togglePart(name, on);
    this.renderNow();
    return true;
  },
  tab(name) {
    activateTab(name);
    return name;
  },
  /** JPEG/PNG data URL of the current view, downscaled to `maxWidth` for cheap transport. */
  grab(maxWidth = 1100, type = 'image/jpeg', quality = 0.85) {
    this.renderNow();
    const src = renderer.domElement;
    const scale = Math.min(1, maxWidth / src.width);
    const copy = document.createElement('canvas');
    copy.width = Math.round(src.width * scale);
    copy.height = Math.round(src.height * scale);
    copy.getContext('2d').drawImage(src, 0, 0, copy.width, copy.height);
    return copy.toDataURL(type, quality);
  },
  state() {
    return window.__phimech;
  },
};

// --------------------------------------------------------------------------- boot

async function boot() {
  wire();
  resize();
  tick();

  // Deep-link support: ?build=box opens straight into that build.
  const params = new URLSearchParams(window.location.search);
  const requested = params.get('build');

  try {
    await loadBuild(requested || undefined);
  } catch (err) {
    fail('PhiMech failed to start', String((err && err.message) || err), [
      'This is a PhiMech application error, not a problem with your build.',
      'The browser console has the stack trace.',
    ]);
  }
}

window.addEventListener('error', (event) => {
  // A module-level failure (bad import map, missing vendor file) would otherwise leave the
  // loading spinner up forever with no explanation.
  if (!window.__phimech.ready) {
    fail('A script error stopped the viewer', String(event.message || event.error || event.type), [
      'Most likely a missing vendor file under assets/vendor/.',
      'Check the import map in index.html maps "three" to ./vendor/three.module.js.',
    ]);
  }
});

boot();
