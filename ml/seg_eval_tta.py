# One-pass evaluation of a field-seg ONNX model, with and without flip TTA (mean of identity, h-flip, v-flip probs).
# Tile sets (same as seg_eval_compare / seg_thr_sweep / seg_eval_big):
#   tune : val-00000 every 3rd tile, first half  (thresholds are chosen here only)
#   c240 : val-00000 every 3rd tile, second half, first 240
#   t800 : val-00002 every 2nd tile, first 800
# Stores pixel TP/FP/FN and 4x4-zone TP/FP/FN per class per threshold -> npz. Run from /home/user/work.
import io, sys, time, numpy as np, onnxruntime as ort, pyarrow.parquet as pq
from PIL import Image
C = ['drydown', 'nutrient_deficiency', 'weed_cluster', 'water', 'planter_skip']
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
T = np.round(np.arange(0.2, 0.951, 0.05), 2); nT = len(T)
model, out = sys.argv[1], sys.argv[2]
import os; NOTTA = os.environ.get('NOTTA') == '1'   # plain only (the 'tta' slot stays zero)
s = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
def load(f, keep):
    rows, n = [], 0
    for b in pq.ParquetFile(f).iter_batches(batch_size=16, columns=['image_rgb'] + ['mask_' + c for c in C]):
        for r in b.to_pylist():
            if keep(n): rows.append(r)
            n += 1
    return rows
v0 = load('data/agv/val-00000-of-00008.parquet', lambda n: n % 3 == 0)
sets = {'tune': v0[:len(v0) // 2], 'c240': v0[len(v0) // 2:][:240],
        't800': load('data/hf/data/val-00002-of-00008.parquet', lambda n: n % 2 == 0)[:800]}
if len(sys.argv) > 3: sets = {k: v[:int(sys.argv[3])] for k, v in sets.items()}
run = lambda x: s.run(['probs'], {'pixel_values': x})[0][0]
R = {}
t0 = time.time()
for name, rows in sets.items():
    A = np.zeros((2, 2, 3, len(C), len(T)))   # [variant plain/tta][pixel/zone][tp,fp,fn][class][thr]
    for i, r in enumerate(rows):
        im = Image.open(io.BytesIO(r['image_rgb']['bytes'])).convert('RGB').resize((512, 512), Image.BICUBIC)
        x = ((np.asarray(im, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
        M = [np.asarray(Image.open(io.BytesIO(r['mask_' + c]['bytes'])), np.uint8) > 0 for c in C]
        p0 = run(x)
        if NOTTA: variants = [p0]
        else:
            ph = run(np.ascontiguousarray(x[..., ::-1]))[..., ::-1]
            pv = run(np.ascontiguousarray(x[..., ::-1, :]))[..., ::-1, :]
            variants = [p0, (p0 + ph + pv) / 3]
        for v, p in enumerate(variants):
            for j, c in enumerate(C):
                m = M[j]
                up = np.asarray(Image.fromarray(p[j].astype(np.float32)).resize((512, 512), Image.BILINEAR))
                b = np.searchsorted(T, up, side='left')            # pixel exceeds thresholds 0..b-1
                hp = np.bincount(b[m], minlength=nT + 1); ha = np.bincount(b.ravel(), minlength=nT + 1)
                gt = lambda h: h[::-1].cumsum()[::-1][1:]           # count with b > k, k = 0..nT-1
                tp = gt(hp); pp = gt(ha)
                A[v, 0, 0, j] += tp; A[v, 0, 1, j] += pp - tp; A[v, 0, 2, j] += m.sum() - tp
                zb = b.reshape(4, 128, 4, 128).transpose(0, 2, 1, 3).reshape(16, -1)
                zc = np.stack([gt(np.bincount(z, minlength=nT + 1)) for z in zb])   # (16, nT) pixels > t per zone
                zq = zc >= 0.1 * 128 * 128; zy = (m.reshape(4, 128, 4, 128).mean((1, 3)) >= 0.1).reshape(16, 1)
                A[v, 1, 0, j] += (zq & zy).sum(0); A[v, 1, 1, j] += (zq & ~zy).sum(0); A[v, 1, 2, j] += (~zq & zy).sum(0)
        if i % 50 == 0: print(name, i, len(rows), f'{time.time() - t0:.0f}s', flush=True)
    R[name] = A
np.savez(out, T=T, C=np.array(C), **R)
print('saved', out)
