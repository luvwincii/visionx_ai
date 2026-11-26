# End-to-end check of the exported field model on held-out val tiles (the half not used to pick thresholds)
import io, json, sys, numpy as np, onnxruntime as ort, pyarrow.parquet as pq
from PIL import Image
meta = json.load(open('field_labels.json')); C = meta['classes']; S = 448; G = 32
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
rows = []
n = 0
for b in pq.ParquetFile('data/agv/val-00000-of-00008.parquet').iter_batches(batch_size=16, columns=['image_rgb'] + ['mask_' + c for c in C]):
    for r in b.to_pylist():
        if n % 3 == 0: rows.append(r)
        n += 1
rows = rows[len(rows) // 2:][:240]
for path in sys.argv[1:]:
    s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    inter = np.zeros(len(C)); union = np.zeros(len(C))
    for r in rows:
        im = Image.open(io.BytesIO(r['image_rgb']['bytes'])).convert('RGB').resize((S, S), Image.BICUBIC)
        x = ((np.asarray(im, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
        p = s.run(['probs'], {'pixel_values': x})[0][0]
        for j, c in enumerate(C):
            y = np.asarray(Image.open(io.BytesIO(r['mask_' + c]['bytes'])).resize((G, G), Image.BOX), np.float32).reshape(-1) / 255 > 0.5
            pr = p[:, j] > meta['thr'][j]
            inter[j] += (pr & y).sum(); union[j] += (pr | y).sum()
    print(path, {c: round(float(inter[j] / max(1, union[j])), 3) for j, c in enumerate(C)}, flush=True)
