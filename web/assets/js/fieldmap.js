// VisionX field health map — works on ordinary RGB photos (drone or phone, top-down).
//
// Every pixel is analysed (no blocks):
//   1. Masks: water, deep shadow and grey surfaces (roads, roofs) are "not analysed".
//   2. Vegetation: normalised Excess Green, ExG = (2G − R − B) / (R + G + B).
//      Green and yellowing leaves score high, brown soil scores ~0.
//   3. Health of vegetation: VARI = (G − R) / (G + R − B). Dense green canopy is
//      high; yellowing, browning or wilting leaves drop towards 0 and below.
//   4. A small blur (a few pixels) turns single-pixel noise into a local reading,
//      then each pixel gets a class:
//        0 healthy (green) · 1 risk (yellow) · 2 problem (red) · 3 bare soil · 4 not analysed
//   5. The map is shaded with the photo's own brightness, so rows and canopy stay visible.

export const CLASS = { OK: 0, RISK: 1, BAD: 2, SOIL: 3, VOID: 4 };
export const PALETTE = [
  [46, 160, 67],   // ok
  [227, 179, 65],  // risk
  [218, 54, 51],   // problem
  [122, 104, 88],  // soil
  [52, 56, 54],    // void
];

const DEFAULTS = {
  maxSide: 1400,     // analysis resolution (longest side, px)
  radius: null,      // blur radius in px; default ≈ longest side / 350
  tHealthy: 0.07,    // local VARI >= → healthy
  tRisk: -0.01,      // local VARI >= → risk, else problem
  minVegFrac: 0.2,   // local (weighted) vegetation share needed to grade a pixel
  shade: true,       // modulate colours with photo brightness
  zoneGrid: 6,       // zones for the report (6 × 6, labelled A1..F6)
};

export function imageToCanvas(img, maxSide) {
  const w0 = img.naturalWidth || img.width, h0 = img.naturalHeight || img.height;
  const k = Math.min(1, maxSide / Math.max(w0, h0));
  const w = Math.max(1, Math.round(w0 * k)), h = Math.max(1, Math.round(h0 * k));
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, 0, 0, w, h);
  return c;
}

// Separable box blur with running sums, applied twice (close to a Gaussian).
function blur(src, W, H, r) {
  const tmp = new Float32Array(W * H), out = new Float32Array(W * H);
  const pass = (a, b) => {
    const n = 2 * r + 1;
    for (let y = 0; y < H; y++) {           // horizontal
      const o = y * W; let acc = 0;
      for (let x = -r; x <= r; x++) acc += a[o + Math.min(W - 1, Math.max(0, x))];
      for (let x = 0; x < W; x++) {
        tmp[o + x] = acc / n;
        acc += a[o + Math.min(W - 1, x + r + 1)] - a[o + Math.max(0, x - r)];
      }
    }
    for (let x = 0; x < W; x++) {           // vertical
      let acc = 0;
      for (let y = -r; y <= r; y++) acc += tmp[Math.min(H - 1, Math.max(0, y)) * W + x];
      for (let y = 0; y < H; y++) {
        b[y * W + x] = acc / n;
        acc += tmp[Math.min(H - 1, y + r + 1) * W + x] - tmp[Math.max(0, y - r) * W + x];
      }
    }
  };
  pass(src, out);
  pass(out, out);
  return out;
}

export function analyzeField(img, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const src = imageToCanvas(img, o.maxSide);
  const W = src.width, H = src.height, N = W * H;
  const px = src.getContext('2d').getImageData(0, 0, W, H).data;
  const r = o.radius ?? Math.max(1, Math.round(Math.max(W, H) / 350));

  const inv = new Float32Array(N), veg = new Float32Array(N), vv = new Float32Array(N), lum = new Float32Array(N);
  for (let i = 0, p = 0; i < N; i++, p += 4) {
    const R = px[p] / 255, G = px[p + 1] / 255, B = px[p + 2] / 255;
    const s = R + G + B + 1e-6, mx = Math.max(R, G, B), mn = Math.min(R, G, B);
    lum[i] = 0.299 * R + 0.587 * G + 0.114 * B;
    const water = (B >= 0.88 * G && G >= 0.95 * R && mx - mn < 0.25) || (B > G && B > R);
    if (water || s < 0.2 || (mx - mn < 0.06 && mx > 0.45)) { inv[i] = 1; continue; }
    const exg = (2 * G - R - B) / s;
    if (exg > 0.06) {
      let v = (G - R) / Math.max(G + R - B, 0.05);
      v = v < -1 ? -1 : v > 1 ? 1 : v;
      // weight by how clearly the pixel is foliage: soil/leaf edge pixels count little
      const w = Math.min(1, (exg - 0.06) / 0.12);
      veg[i] = w; vv[i] = v * w;
    }
  }
  const bInv = blur(inv, W, H, r), bVeg = blur(veg, W, H, r), bVV = blur(vv, W, H, r);

  const cls = new Uint8Array(N);
  const counts = [0, 0, 0, 0, 0];
  let sumV = 0;
  for (let i = 0; i < N; i++) {
    let c;
    if (bInv[i] > 0.5) c = 4;
    else if (bVeg[i] < o.minVegFrac) c = 3;
    else {
      const v = bVV[i] / bVeg[i];
      c = v >= o.tHealthy ? 0 : v >= o.tRisk ? 1 : 2;
      sumV += v;
    }
    cls[i] = c; counts[c]++;
  }

  // Render
  const map = document.createElement('canvas');
  map.width = W; map.height = H;
  const mctx = map.getContext('2d');
  const id = mctx.createImageData(W, H);
  const d = id.data;
  for (let i = 0, p = 0; i < N; i++, p += 4) {
    const col = PALETTE[cls[i]];
    const k = o.shade && cls[i] !== 4 ? 0.72 + 0.56 * Math.min(1, lum[i] * 1.6) : 1;
    d[p] = Math.min(255, col[0] * k); d[p + 1] = Math.min(255, col[1] * k); d[p + 2] = Math.min(255, col[2] * k); d[p + 3] = 255;
  }
  mctx.putImageData(id, 0, 0);

  // Stats
  const vegPx = counts[0] + counts[1] + counts[2];
  const pct = (n, dd) => (dd ? Math.round((1000 * n) / dd) / 10 : 0);

  // Zones (grid A1..), ranked by problem share
  const G = o.zoneGrid, zones = [];
  for (let zy = 0; zy < G; zy++) for (let zx = 0; zx < G; zx++) {
    const x0 = Math.floor((zx * W) / G), x1 = Math.floor(((zx + 1) * W) / G);
    const y0 = Math.floor((zy * H) / G), y1 = Math.floor(((zy + 1) * H) / G);
    const c = [0, 0, 0, 0, 0];
    for (let y = y0; y < y1; y++) { const row = y * W; for (let x = x0; x < x1; x++) c[cls[row + x]]++; }
    const v = c[0] + c[1] + c[2], area = (x1 - x0) * (y1 - y0);
    zones.push({
      id: String.fromCharCode(65 + zx) + (zy + 1),
      x0: x0 / W, x1: x1 / W, y0: y0 / H, y1: y1 / H,
      veg: Math.round((100 * v) / Math.max(1, area)), // % of the zone covered by vegetation
      ok: pct(c[0], v), risk: pct(c[1], v), bad: pct(c[2], v), soil: pct(c[3], area),
      score: v ? (c[2] + c[1] * 0.4) / v : 0,
    });
  }
  const hot = zones.filter((z) => z.veg >= 10 && (z.bad >= 25 || z.risk + z.bad >= 50))
    .sort((a, b) => b.score - a.score).slice(0, 5);

  return {
    width: W, height: H, map, photo: src,
    stats: {
      ok: pct(counts[0], vegPx), risk: pct(counts[1], vegPx), bad: pct(counts[2], vegPx),
      soil: pct(counts[3], N), void: pct(counts[4], N), vegCover: pct(vegPx, N),
      index: vegPx ? Math.round((sumV / vegPx) * 100) / 100 : 0,
    },
    zones, hot, grid: G,
  };
}

// Draw a zone grid with labels onto a 2D context (for result view)
export function drawZoneGrid(ctx, w, h, G, hot = [], scale = 1) {
  ctx.save();
  ctx.strokeStyle = 'rgba(233,238,235,0.35)';
  ctx.lineWidth = Math.max(1, scale);
  for (let i = 1; i < G; i++) {
    const x = Math.round((i * w) / G) + 0.5, y = Math.round((i * h) / G) + 0.5;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  const fs = Math.round(12 * scale);
  ctx.font = `500 ${fs}px "Geist Mono", ui-monospace, monospace`;
  ctx.textBaseline = 'top';
  for (const z of hot) {
    const x = z.x0 * w, y = z.y0 * h, zw = (z.x1 - z.x0) * w, zh = (z.y1 - z.y0) * h;
    ctx.strokeStyle = '#e9eeeb'; ctx.lineWidth = Math.max(2, Math.round(2 * scale));
    ctx.strokeRect(x + 1, y + 1, zw - 2, zh - 2);
    const pad = Math.round(fs * 0.4);
    const tw = ctx.measureText(z.id).width + pad * 2;
    ctx.fillStyle = '#0c0f0e'; ctx.fillRect(x + 4, y + 4, tw, fs + pad * 1.4);
    ctx.fillStyle = '#e9eeeb'; ctx.fillText(z.id, x + 4 + pad, y + 4 + pad * 0.7);
  }
  ctx.restore();
}
