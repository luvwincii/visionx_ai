# Threshold sweep: tune on val-00000 (first half, every 3rd tile; not the 240 compare tiles), report on val-00002 (800 tiles).
import io, json, sys, numpy as np, onnxruntime as ort, pyarrow.parquet as pq
from PIL import Image
C = ['drydown', 'nutrient_deficiency', 'weed_cluster', 'water', 'planter_skip']
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
T = np.round(np.arange(0.2, 0.95, 0.05), 2)
s = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider'])
def load(f, keep):
    rows, n = [], 0
    for b in pq.ParquetFile(f).iter_batches(batch_size=16, columns=['image_rgb'] + ['mask_' + c for c in C]):
        for r in b.to_pylist():
            if keep(n): rows.append(r)
            n += 1
    return rows
def sweep(rows):
    I = np.zeros((len(C), len(T))); U = np.zeros((len(C), len(T)))
    for r in rows:
        im = Image.open(io.BytesIO(r['image_rgb']['bytes'])).convert('RGB').resize((512, 512), Image.BICUBIC)
        x = ((np.asarray(im, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
        p = s.run(['probs'], {'pixel_values': x})[0][0]
        for j, c in enumerate(C):
            m = np.asarray(Image.open(io.BytesIO(r['mask_' + c]['bytes'])), np.uint8)[::2, ::2] > 0
            up = np.asarray(Image.fromarray(p[j]).resize((256, 256), Image.BILINEAR))
            for k, t in enumerate(T):
                q = up > t; I[j, k] += (q & m).sum(); U[j, k] += (q | m).sum()
    return I / np.maximum(1, U)
v0 = load('/home/user/work/data/agv/val-00000-of-00008.parquet', lambda n: n % 3 == 0)
tune = sweep(v0[:len(v0) // 2])
test = sweep(load('/home/user/work/data/hf/data/val-00002-of-00008.parquet', lambda n: n % 2 == 0)[:800])
best = tune.argmax(1)
out = {c: {'thr': float(T[best[j]]), 'tune_IoU': round(float(tune[j, best[j]]), 3), 'test_IoU': round(float(test[j, best[j]]), 3)} for j, c in enumerate(C)}
out['mean_test_IoU_5'] = round(float(np.mean([out[c]['test_IoU'] for c in C])), 3)
print(json.dumps(out, indent=1)); json.dump({'tune': tune.round(3).tolist(), 'test': test.round(3).tolist(), 'T': T.tolist(), 'chosen': out}, open('seg_thr_sweep.json', 'w'))
