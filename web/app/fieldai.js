// VisionX field AI (research demo): DINOv2-small + segmentation head trained on Agriculture-Vision (RGB).
// A) per-patch probabilities for 6 field anomaly classes
// B) label-free anomaly: how different each patch is from the rest of the same photo
const MODEL_URL = new URL('../models/visionx-field.onnx', import.meta.url).href;
const META_URL = new URL('../models/field_labels.json', import.meta.url).href;
const ORT_URL = new URL('../vendor/ort/ort.wasm.min.mjs', import.meta.url).href;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

export const FIELD_CLASSES = {
  drydown: { name: 'Quruma', color: [200, 117, 44], advice: 'Bitki vaxtından əvvəl quruyur. Suvarmanın bu zonaya çatdığını və torpaq nəmliyini yoxlayın.' },
  nutrient_deficiency: { name: 'Qida çatışmazlığı', color: [232, 197, 71], advice: 'Saralma qida çatışmazlığına bənzəyir. Torpaq və yarpaq analizi edin, azot və mikroelementləri yoxlayın.' },
  weed_cluster: { name: 'Alaq', color: [162, 89, 217], advice: 'Alaq toplusu ehtimalı var. Yerində baxın, yalnız bu zonaya lokal mübarizə tətbiq edin.' },
  water: { name: 'Su yığılması', color: [59, 130, 246], advice: 'Su yığılır və ya gölməçə var. Drenajı və suvarma normasını yoxlayın, kök çürüməsinə diqqət edin.' },
  planter_skip: { name: 'Əkin boşluğu', color: [239, 68, 68], advice: 'Cərgədə bitki çıxmayıb. Toxum səpən aparatı yoxlayın, lazım olarsa yenidən əkin edin.' },
  storm_damage: { name: 'Külək/fırtına zədəsi', color: [244, 114, 182], advice: 'Bitkilər yatıb və ya zədələnib. Zərərin həcmini qeyd edin, sığorta üçün şəkil saxlayın.' },
};

let sessionP = null, meta = null, ort = null;
export function loadFieldModel(onProgress) {
  if (sessionP) return sessionP;
  sessionP = (async () => {
    ort = await import(ORT_URL);
    ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    meta = await (await fetch(META_URL)).json();
    const res = await fetch(MODEL_URL);
    if (!res.ok) throw new Error('Sahə modeli yüklənmədi: ' + res.status);
    const total = +res.headers.get('content-length') || 0;
    let buf;
    if (res.body && total && onProgress) {
      const reader = res.body.getReader(); buf = new Uint8Array(total); let got = 0;
      for (;;) { const { done, value } = await reader.read(); if (done) break; buf.set(value, got); got += value.length; onProgress(got / total); }
    } else buf = new Uint8Array(await res.arrayBuffer());
    return ort.InferenceSession.create(buf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  })();
  sessionP.catch(() => { sessionP = null; });
  return sessionP;
}

function tileTensor(src, sx, sy, sw, sh, S) {
  const c = document.createElement('canvas'); c.width = S; c.height = S;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, sx, sy, sw, sh, 0, 0, S, S);
  const d = ctx.getImageData(0, 0, S, S).data, n = S * S, t = new Float32Array(3 * n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    t[i] = (d[p] / 255 - MEAN[0]) / STD[0];
    t[n + i] = (d[p + 1] / 255 - MEAN[1]) / STD[1];
    t[2 * n + i] = (d[p + 2] / 255 - MEAN[2]) / STD[2];
  }
  return t;
}

// src: canvas or image. Returns overlay canvases (W×H of src) and per-class area shares.
export async function analyzeFieldAI(src, onStep) {
  const session = await loadFieldModel();
  const W = src.width, H = src.height, S = meta.size, G = meta.grid, C = meta.classes.length;
  // 2 tiles along each side, square, overlapping when the photo is not square
  const tile = Math.ceil(Math.max(W, H) / 2 > Math.min(W, H) ? Math.min(W, H) : Math.max(W, H) / 2);
  const nx = Math.max(1, Math.ceil(W / tile)), ny = Math.max(1, Math.ceil(H / tile));
  const tiles = [];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const x = nx === 1 ? 0 : Math.round((i * (W - tile)) / (nx - 1));
    const y = ny === 1 ? 0 : Math.round((j * (H - tile)) / (ny - 1));
    tiles.push({ x, y, w: Math.min(tile, W), h: Math.min(tile, H) });
  }
  const results = [];
  for (let k = 0; k < tiles.length; k++) {
    onStep?.(k, tiles.length);
    const t = tiles[k];
    const out = await session.run({ pixel_values: new ort.Tensor('float32', tileTensor(src, t.x, t.y, t.w, t.h, S), [1, 3, S, S]) });
    results.push({ t, probs: out.probs.data, tokens: out.tokens.data });
  }
  // B) label-free anomaly: whitened distance from the photo's mean token
  const D = 384, N = G * G;
  const mu = new Float64Array(D), sq = new Float64Array(D); let cnt = 0;
  for (const r of results) for (let i = 0; i < N; i++) { for (let d = 0; d < D; d++) { const v = r.tokens[i * D + d]; mu[d] += v; sq[d] += v * v; } cnt++; }
  const sd = new Float64Array(D);
  for (let d = 0; d < D; d++) { mu[d] /= cnt; sd[d] = Math.sqrt(Math.max(1e-6, sq[d] / cnt - mu[d] * mu[d])); }
  const scores = [];
  for (const r of results) {
    const s = new Float32Array(N);
    for (let i = 0; i < N; i++) { let a = 0; for (let d = 0; d < D; d++) { const z = (r.tokens[i * D + d] - mu[d]) / sd[d]; a += z * z; } s[i] = a / D; }
    scores.push(s);
  }
  const all = scores.flatMap((s) => Array.from(s)).sort((a, b) => a - b);
  const med = all[Math.floor(all.length / 2)], p95 = all[Math.floor(all.length * 0.95)];

  // Render overlays (smooth: 32×32 per tile, scaled with bilinear filtering)
  const clsCanvas = document.createElement('canvas'); clsCanvas.width = W; clsCanvas.height = H;
  const anoCanvas = document.createElement('canvas'); anoCanvas.width = W; anoCanvas.height = H;
  const cctx = clsCanvas.getContext('2d'), actx = anoCanvas.getContext('2d');
  cctx.imageSmoothingEnabled = true; actx.imageSmoothingEnabled = true;
  const area = new Array(C).fill(0); let tokTotal = 0, anoCount = 0;
  results.forEach((r, k) => {
    const small = document.createElement('canvas'); small.width = G; small.height = G;
    const sctx = small.getContext('2d'); const id = sctx.createImageData(G, G);
    const asmall = document.createElement('canvas'); asmall.width = G; asmall.height = G;
    const asctx = asmall.getContext('2d'); const aid = asctx.createImageData(G, G);
    for (let i = 0; i < N; i++) {
      let best = -1, bp = 0;
      for (let c = 0; c < C; c++) {
        const p = r.probs[i * C + c], th = meta.thr[c];
        if (p >= th && p - th > bp) { bp = p - th; best = c; }
      }
      tokTotal++;
      if (best >= 0) {
        area[best]++;
        const col = FIELD_CLASSES[meta.classes[best]].color;
        id.data[i * 4] = col[0]; id.data[i * 4 + 1] = col[1]; id.data[i * 4 + 2] = col[2];
        id.data[i * 4 + 3] = Math.round(255 * Math.min(0.9, 0.45 + bp * 2));
      }
      const z = (scores[k][i] - med) / Math.max(1e-6, p95 - med);
      if (z > 1) anoCount++;
      const a = Math.max(0, Math.min(1, (z - 0.6) / 0.8));
      aid.data[i * 4] = 255; aid.data[i * 4 + 1] = 80; aid.data[i * 4 + 2] = 40; aid.data[i * 4 + 3] = Math.round(220 * a);
    }
    sctx.putImageData(id, 0, 0); asctx.putImageData(aid, 0, 0);
    cctx.drawImage(small, r.t.x, r.t.y, r.t.w, r.t.h);
    actx.drawImage(asmall, r.t.x, r.t.y, r.t.w, r.t.h);
  });
  const shares = meta.classes.map((c, i) => ({ key: c, ...FIELD_CLASSES[c], pct: Math.round((1000 * area[i]) / tokTotal) / 10 }))
    .sort((a, b) => b.pct - a.pct);
  return { classes: clsCanvas, anomaly: anoCanvas, shares, anomalyPct: Math.round((1000 * anoCount) / tokTotal) / 10, tiles: tiles.length };
}
