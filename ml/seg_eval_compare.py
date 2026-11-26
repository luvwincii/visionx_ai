# Compare old (frozen DINOv2 patch head) and new (SegFormer) field models on the SAME held-out tiles:
# val-00000, every 3rd tile, second half, first 240 (never used for training or threshold tuning of either model).
import io, json, sys, numpy as np, onnxruntime as ort, pyarrow.parquet as pq
from PIL import Image
C = ['drydown', 'nutrient_deficiency', 'weed_cluster', 'water', 'planter_skip', 'storm_damage']
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
rows = []; n = 0
for b in pq.ParquetFile('data/agv/val-00000-of-00008.parquet').iter_batches(batch_size=16, columns=['image_rgb'] + ['mask_' + c for c in C]):
    for r in b.to_pylist():
        if n % 3 == 0: rows.append(r)
        n += 1
rows = rows[len(rows) // 2:][:240]

def run(model, meta):
    s = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
    S = meta['size']; thr = meta['thr']
    I = {k: np.zeros(len(C)) for k in ['pix_i', 'pix_u', 'pix_p', 'pix_y', 'pat_i', 'pat_u']}
    for r in rows:
        im = Image.open(io.BytesIO(r['image_rgb']['bytes'])).convert('RGB').resize((S, S), Image.BICUBIC)
        x = ((np.asarray(im, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
        p = s.run(['probs'], {'pixel_values': x})[0][0]
        if 'out_stride' in meta: P = p                      # (C, g, g)
        else: P = p.T.reshape(len(C), 32, 32)              # (N, C) -> (C, 32, 32)
        for j, c in enumerate(C):
            if thr[j] > 1: continue
            m = np.asarray(Image.open(io.BytesIO(r['mask_' + c]['bytes'])), np.uint8) > 0          # 512x512
            # pixel level: upsample probability map to 512 (bilinear), threshold
            up = np.asarray(Image.fromarray(P[j].astype(np.float32)).resize((512, 512), Image.BILINEAR)) > thr[j]
            I['pix_i'][j] += (up & m).sum(); I['pix_u'][j] += (up | m).sum(); I['pix_p'][j] += up.sum(); I['pix_y'][j] += m.sum()
            # patch level 32x32
            pp = np.asarray(Image.fromarray(P[j].astype(np.float32)).resize((32, 32), Image.BOX)) > thr[j]
            yy = np.asarray(Image.fromarray(m.astype(np.float32)).resize((32, 32), Image.BOX)) > 0.5
            I['pat_i'][j] += (pp & yy).sum(); I['pat_u'][j] += (pp | yy).sum()
    out = {}
    for j, c in enumerate(C):
        if thr[j] > 1: continue
        out[c] = {'pixel_IoU': round(I['pix_i'][j] / max(1, I['pix_u'][j]), 3),
                  'precision': round(I['pix_i'][j] / max(1, I['pix_p'][j]), 3),
                  'recall': round(I['pix_i'][j] / max(1, I['pix_y'][j]), 3),
                  'patch32_IoU': round(I['pat_i'][j] / max(1, I['pat_u'][j]), 3)}
    five = [c for c in C[:5] if c in out]
    out['mean_pixel_IoU_5'] = round(float(np.mean([out[c]['pixel_IoU'] for c in five])), 3)
    return out

res = {}
for name, model, metaf in [(a.split('=')[0], a.split('=')[1].split(',')[0], a.split('=')[1].split(',')[1]) for a in sys.argv[1:]]:
    res[name] = run(model, json.load(open(metaf)))
    print(name, json.dumps(res[name]), flush=True)
json.dump(res, open('seg_compare.json', 'w'), indent=1)
