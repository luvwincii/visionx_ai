// VisionX field health map — works on ordinary RGB photos (drone or phone, top-down).
// Per pixel: vegetation via normalised Excess Green (ExG), health via VARI
// (Visible Atmospherically Resistant Index). Pixels are aggregated into blocks,
// smoothed 3x3, then each block gets one class:
//   0 healthy (green) · 1 risk (yellow) · 2 problem (red) · 3 bare soil · 4 not analysed (water, shadow, roads, roofs)

export const CLASS = { OK: 0, RISK: 1, BAD: 2, SOIL: 3, VOID: 4 };
export const PALETTE = [
  [46, 160, 67],   // ok
  [227, 179, 65],  // risk
  [218, 54, 51],   // problem
  [122, 104, 88],  // soil
  [52, 56, 54],    // void
];

const DEFAULTS = {
  maxSide: 960,      // analysis resolution
  blocksAcross: 110, // block grid density along the longest side
  tHealthy: 0.07,    // VARI >= -> healthy
  tRisk: -0.01,      // VARI >= -> risk, else problem
  minVegFrac: 0.18,  // block needs this share of vegetation pixels to be graded
  zoneGrid: 6,       // zones for the report (6 x 6, labelled A1..F6)
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

export function analyzeField(img, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const src = imageToCanvas(img, o.maxSide);
  const W = src.width, H = src.height;
  const px = src.getContext('2d').getImageData(0, 0, W, H).data;
  const bs = Math.max(4, Math.round(Math.max(W, H) / o.blocksAcross));
  const wb = Math.floor(W / bs), hb = Math.floor(H / bs);
  const nb = wb * hb;
  const vegCnt = new Float32Array(nb), varSum = new Float32Array(nb), invCnt = new Float32Array(nb);

  for (let y = 0; y < hb * bs; y++) {
    const by = (y / bs) | 0;
    for (let x = 0; x < wb * bs; x++) {
      const i = (y * W + x) * 4;
      const r = px[i] / 255, g = px[i + 1] / 255, b = px[i + 2] / 255;
      const s = r + g + b + 1e-6;
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      const water = (b >= 0.88 * g && g >= 0.95 * r && mx - mn < 0.25) || (b > g && b > r);
      const dark = s < 0.2;
      const grey = mx - mn < 0.06 && mx > 0.45;
      const k = by * wb + ((x / bs) | 0);
      if (water || dark || grey) { invCnt[k]++; continue; }
      const exg = (2 * g - r - b) / s;
      if (exg > 0.06) {
        let vari = (g - r) / Math.max(g + r - b, 0.05);
        vari = vari < -1 ? -1 : vari > 1 ? 1 : vari;
        vegCnt[k]++; varSum[k] += vari;
      }
    }
  }

  // 3x3 smoothing of block sums
  const area = bs * bs;
  const sm = (arr) => {
    const out = new Float32Array(nb);
    for (let y = 0; y < hb; y++) for (let x = 0; x < wb; x++) {
      let acc = 0, n = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const yy = y + dy, xx = x + dx;
        if (yy < 0 || xx < 0 || yy >= hb || xx >= wb) continue;
        const w = dx === 0 && dy === 0 ? 2 : 1;
        acc += arr[yy * wb + xx] * w; n += w;
      }
      out[y * wb + x] = acc / n;
    }
    return out;
  };
  const vegS = sm(vegCnt), varS = sm(varSum);

  const cls = new Uint8Array(nb);
  const blockVari = new Float32Array(nb);
  for (let k = 0; k < nb; k++) {
    if (invCnt[k] / area > 0.5) { cls[k] = 4; continue; }
    const vf = vegS[k] / area;
    if (vf < o.minVegFrac) { cls[k] = 3; continue; }
    const v = varS[k] / Math.max(vegS[k], 1e-6);
    blockVari[k] = v;
    cls[k] = v >= o.tHealthy ? 0 : v >= o.tRisk ? 1 : 2;
  }

  // Map canvas at the analysis resolution
  const map = document.createElement('canvas');
  map.width = wb * bs; map.height = hb * bs;
  const small = document.createElement('canvas');
  small.width = wb; small.height = hb;
  const sctx = small.getContext('2d');
  const id = sctx.createImageData(wb, hb);
  for (let k = 0; k < nb; k++) {
    const p = PALETTE[cls[k]];
    id.data[k * 4] = p[0]; id.data[k * 4 + 1] = p[1]; id.data[k * 4 + 2] = p[2]; id.data[k * 4 + 3] = 255;
  }
  sctx.putImageData(id, 0, 0);
  const mctx = map.getContext('2d');
  mctx.imageSmoothingEnabled = false;
  mctx.drawImage(small, 0, 0, map.width, map.height);

  // Stats
  const counts = [0, 0, 0, 0, 0];
  for (let k = 0; k < nb; k++) counts[cls[k]]++;
  const vegBlocks = counts[0] + counts[1] + counts[2];
  const pct = (n, d) => (d ? Math.round((1000 * n) / d) / 10 : 0);
  let meanVari = 0;
  for (let k = 0; k < nb; k++) if (cls[k] < 3) meanVari += blockVari[k];
  meanVari = vegBlocks ? meanVari / vegBlocks : 0;

  // Zones (grid A1..), ranked by problem share
  const G = o.zoneGrid, zones = [];
  for (let zy = 0; zy < G; zy++) for (let zx = 0; zx < G; zx++) {
    const x0 = Math.floor((zx * wb) / G), x1 = Math.floor(((zx + 1) * wb) / G);
    const y0 = Math.floor((zy * hb) / G), y1 = Math.floor(((zy + 1) * hb) / G);
    const c = [0, 0, 0, 0, 0];
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) c[cls[y * wb + x]]++;
    const veg = c[0] + c[1] + c[2];
    zones.push({
      id: String.fromCharCode(65 + zx) + (zy + 1),
      x0: x0 / wb, x1: x1 / wb, y0: y0 / hb, y1: y1 / hb,
      veg, ok: pct(c[0], veg), risk: pct(c[1], veg), bad: pct(c[2], veg),
      soil: pct(c[3], Math.max(1, (x1 - x0) * (y1 - y0))),
      score: veg ? (c[2] * 1 + c[1] * 0.4) / veg : 0,
    });
  }
  const hot = zones.filter((z) => z.veg >= 4 && (z.bad >= 25 || z.risk + z.bad >= 50))
    .sort((a, b) => b.score - a.score).slice(0, 5);

  return {
    width: W, height: H, map, photo: src,
    stats: {
      ok: pct(counts[0], vegBlocks), risk: pct(counts[1], vegBlocks), bad: pct(counts[2], vegBlocks),
      soil: pct(counts[3], nb), void: pct(counts[4], nb), vegCover: pct(vegBlocks, nb),
      index: Math.round(meanVari * 100) / 100,
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
