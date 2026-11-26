// Leaf diagnosis: VisionX model (DINOv2-small backbone + VisionX head) in ONNX, run in the browser.
const MODEL_URL = new URL('../models/visionx-leaf.onnx', import.meta.url).href;
const LABELS_URL = new URL('../models/labels.json', import.meta.url).href;
const ORT_URL = new URL('../vendor/ort/ort.wasm.min.mjs', import.meta.url).href;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const SIZE = 224, RESIZE = 256;

let sessionPromise = null;
let labels = null;
let ort = null;

export function loadModel(onProgress) {
  if (sessionPromise) return sessionPromise;
  sessionPromise = (async () => {
    ort = await import(ORT_URL);
    ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    labels = await (await fetch(LABELS_URL)).json();
    const buf = await fetchWithProgress(MODEL_URL, onProgress);
    return ort.InferenceSession.create(buf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  })();
  sessionPromise.catch(() => { sessionPromise = null; });
  return sessionPromise;
}

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error('Model yüklənmədi: ' + res.status);
  const total = +res.headers.get('content-length') || 0;
  if (!res.body || !total || !onProgress) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const out = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, got); got += value.length;
    onProgress(got / total);
  }
  return out.subarray(0, got);
}

// resize shorter side to 256, centre-crop 224 (same as training)
function toTensor(img, flip = false) {
  const w0 = img.naturalWidth || img.width, h0 = img.naturalHeight || img.height;
  const k = RESIZE / Math.min(w0, h0);
  const w = Math.round(w0 * k), h = Math.round(h0 * k);
  // two-step downscale for quality on large photos
  let src = img, sw = w0, sh = h0;
  while (sw / 2 > w) {
    const c = document.createElement('canvas'); c.width = Math.round(sw / 2); c.height = Math.round(sh / 2);
    const cx = c.getContext('2d'); cx.imageSmoothingQuality = 'high'; cx.drawImage(src, 0, 0, c.width, c.height);
    src = c; sw = c.width; sh = c.height;
  }
  const c = document.createElement('canvas'); c.width = SIZE; c.height = SIZE;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  if (flip) { ctx.translate(SIZE, 0); ctx.scale(-1, 1); }
  ctx.drawImage(src, 0, 0, sw, sh, -(w - SIZE) / 2, -(h - SIZE) / 2, w, h);
  const d = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const t = new Float32Array(3 * SIZE * SIZE);
  for (let i = 0, p = 0; i < SIZE * SIZE; i++, p += 4) {
    t[i] = (d[p] / 255 - MEAN[0]) / STD[0];
    t[SIZE * SIZE + i] = (d[p + 1] / 255 - MEAN[1]) / STD[1];
    t[2 * SIZE * SIZE + i] = (d[p + 2] / 255 - MEAN[2]) / STD[2];
  }
  return t;
}

export async function classify(img) {
  const session = await loadModel();
  const inName = session.inputNames[0], outName = session.outputNames[0];
  const run = async (flip) => {
    const x = new ort.Tensor('float32', toTensor(img, flip), [1, 3, SIZE, SIZE]);
    const out = await session.run({ [inName]: x });
    return out[outName].data;
  };
  // light test-time augmentation: original + mirrored
  const a = await run(false), b = await run(true);
  const probs = Array.from(a, (v, i) => (v + b[i]) / 2);
  const ranked = probs.map((p, i) => ({ cls: labels[i], p })).sort((x, y) => y.p - x.p);
  const healthyP = ranked.filter((r) => r.cls.endsWith(':healthy')).reduce((s, r) => s + r.p, 0);
  const otherP = ranked.find((r) => r.cls === 'other')?.p ?? 0;
  const byCrop = {};
  for (const r of ranked) { if (r.cls === 'other') continue; const c = r.cls.split(':')[0]; byCrop[c] = (byCrop[c] || 0) + r.p; }
  const topCrop = Object.entries(byCrop).sort((x, y) => y[1] - x[1])[0];
  return {
    top: ranked[0], ranked: ranked.slice(0, 5), healthyP, otherP,
    crop: topCrop ? { key: topCrop[0], p: topCrop[1] } : null,
  };
}
