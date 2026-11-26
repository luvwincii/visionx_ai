// VisionX field AI (research demo): DINOv2-small + segmentation head trained on Agriculture-Vision (RGB only).
// Per 14×14-pixel patch of a 448×448 tile the model gives a probability for each field anomaly class.
// Held-out test IoU (int8, demo thresholds): water 0.58, drydown 0.38, nutrient deficiency 0.31,
// weed 0.19, planter skip 0.14; precision about 40-50%. Storm damage is disabled (too few training samples).
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
    const res = await fetch(meta.model ? new URL('../models/' + meta.model, import.meta.url).href : MODEL_URL);
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

// src: canvas or image. Returns an overlay canvas (W×H of src) and per-class area shares.
export async function analyzeFieldAI(src, onStep) {
  const session = await loadFieldModel();
  const W = src.width, H = src.height, S = meta.size, C = meta.classes.length;
  // two model formats: patch head (grid 32, probs [N, C]) and SegFormer (stride 4, probs [C, G, G])
  const G = meta.grid || S / meta.out_stride, N = G * G;
  const chw = !!meta.out_stride;
  // Square tiles as large as the short side (the model was trained on whole field tiles, not zoomed crops);
  // a 3:2 photo becomes 2 overlapping tiles, a square photo 1.
  const tile = Math.min(W, H);
  const nx = Math.max(1, Math.ceil(W / tile)), ny = Math.max(1, Math.ceil(H / tile));
  const tiles = [];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const x = nx === 1 ? 0 : Math.round((i * (W - tile)) / (nx - 1));
    const y = ny === 1 ? 0 : Math.round((j * (H - tile)) / (ny - 1));
    tiles.push({ x, y, w: Math.min(tile, W), h: Math.min(tile, H) });
  }

  const overlay = document.createElement('canvas'); overlay.width = W; overlay.height = H;
  const octx = overlay.getContext('2d'); octx.imageSmoothingEnabled = true;
  const area = new Array(C).fill(0); let tokTotal = 0;
  for (let k = 0; k < tiles.length; k++) {
    onStep?.(k, tiles.length);
    const t = tiles[k];
    const out = await session.run({ pixel_values: new ort.Tensor('float32', tileTensor(src, t.x, t.y, t.w, t.h, S), [1, 3, S, S]) });
    const raw = out.probs.data;
    // normalise to [N, C] layout
    let probs = raw;
    if (chw) { probs = new Float32Array(N * C); for (let c = 0; c < C; c++) for (let i = 0; i < N; i++) probs[i * C + c] = raw[c * N + i]; }
    // area shares on the output grid
    for (let i = 0; i < N; i++) {
      let best = -1, bp = 0;
      for (let c = 0; c < C; c++) { const p = probs[i * C + c], th = meta.thr[c]; if (p >= th && p - th >= bp) { bp = p - th; best = c; } }
      tokTotal++; if (best >= 0) area[best]++;
    }
    // smooth rendering: bilinearly upsample the probabilities (G → R), then threshold per pixel
    const R = Math.max(256, Math.min(512, G * 4)), sc = G / R, id = octx.createImageData(R, R);
    const pv = new Float32Array(C);
    for (let y = 0; y < R; y++) {
      const fy = Math.min(G - 1, Math.max(0, (y + 0.5) * sc - 0.5)), y0 = Math.floor(fy), y1 = Math.min(G - 1, y0 + 1), wy = fy - y0;
      for (let x = 0; x < R; x++) {
        const fx = Math.min(G - 1, Math.max(0, (x + 0.5) * sc - 0.5)), x0 = Math.floor(fx), x1 = Math.min(G - 1, x0 + 1), wx = fx - x0;
        const a = (y0 * G + x0) * C, b = (y0 * G + x1) * C, c2 = (y1 * G + x0) * C, d = (y1 * G + x1) * C;
        let best = -1, bp = 0;
        for (let c = 0; c < C; c++) {
          pv[c] = (probs[a + c] * (1 - wx) + probs[b + c] * wx) * (1 - wy) + (probs[c2 + c] * (1 - wx) + probs[d + c] * wx) * wy;
          const m = pv[c] - meta.thr[c];
          if (m >= 0 && m >= bp) { bp = m; best = c; }
        }
        if (best >= 0) {
          const col = FIELD_CLASSES[meta.classes[best]].color, o = (y * R + x) * 4;
          id.data[o] = col[0]; id.data[o + 1] = col[1]; id.data[o + 2] = col[2];
          id.data[o + 3] = Math.round(255 * Math.min(0.85, 0.55 + bp));
        }
      }
    }
    const small = document.createElement('canvas'); small.width = R; small.height = R;
    small.getContext('2d').putImageData(id, 0, 0);
    octx.drawImage(small, t.x, t.y, t.w, t.h);
  }
  const shares = meta.classes.map((c, i) => ({ key: c, ...FIELD_CLASSES[c], pct: Math.round((1000 * area[i]) / tokTotal) / 10 }))
    .filter((x, i) => meta.thr[i] <= 1)
    .sort((a, b) => b.pct - a.pct);
  return { classes: overlay, shares, tiles: tiles.length };
}
