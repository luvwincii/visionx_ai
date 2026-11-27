# Build the full-resolution held-out test crops once (same preprocessing as ml/eval_onnx.py / web/app/classifier.js:
# resize short side 256 bicubic, centre crop 224). Sets: plantdoc/cotton/grape (test_rows.json), other (feats npz split), pv_test.
# Run from /home/user/work.
import json, io, numpy as np, pyarrow.parquet as pq, sys
from PIL import Image
sys.path.insert(0, '.'); from labels import canon
t = json.load(open('test_rows.json')); pvn = json.load(open('pv_names.json'))
def crop(b):
    im = Image.open(io.BytesIO(b)).convert('RGB'); w, h = im.size; k = 256 / min(w, h)
    im = im.resize((round(w*k), round(h*k)), Image.BICUBIC); w, h = im.size; l = (w-224)//2; tp = (h-224)//2
    return np.asarray(im.crop((l, tp, l+224, tp+224)), np.uint8)
X = []; Y = []; S = []
for name in ['plantdoc', 'cotton', 'grape']:
    T = set(t[name]['test_rows']); L = t[name]['labels']; i = 0
    for f in t[name]['files']:
        for b in pq.ParquetFile(f).iter_batches(batch_size=32, columns=['image']):
            for im in b.column('image').to_pylist():
                if i in T: X.append(crop(im['bytes'])); Y.append(L[i]); S.append(name)
                i += 1
s_other = [str(x) for x in np.load('feats_dinov2-small_other.npz')['s'] if x != 'train_aug']; i = 0
for b in pq.ParquetFile('data/other/train.parquet').iter_batches(batch_size=32, columns=['image']):
    for im in b.column('image').to_pylist():
        if i < 1600 and s_other[i] == 'test': X.append(crop(im['bytes'])); Y.append('other'); S.append('other')
        i += 1
cnt = {}
for b in pq.ParquetFile('data/pv/test.parquet').iter_batches(batch_size=32, columns=['image', 'label']):
    for im, l in zip(b.column('image').to_pylist(), b.column('label').to_pylist()):
        y = canon(pvn[l])
        if cnt.get(y, 0) >= 60: continue
        cnt[y] = cnt.get(y, 0) + 1; X.append(crop(im['bytes'])); Y.append(y); S.append('pv_test')
np.savez('leaf_ft/test_crops.npz', X=np.stack(X), y=np.array(Y), s=np.array(S))
from collections import Counter; print(Counter(S))
