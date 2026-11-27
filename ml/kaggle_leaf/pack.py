# Package the exact VisionX train/test rows (same split as ml/feats.py / test_rows.json) into a Kaggle dataset.
# Run from /home/user/work. Images downscaled to short side 288, JPEG q90. split in {train,val,test};
# val is carved from TRAIN rows only (seeded); test rows are exactly those of test_rows.json / feats npz.
import io, json, random, numpy as np, pyarrow as pa, pyarrow.parquet as pq, sys
from PIL import Image
sys.path.insert(0, '.')
from labels import canon
OUT = sys.argv[1] if len(sys.argv) > 1 else 'leaf_ft/ds'
t = json.load(open('test_rows.json'))
classes = json.load(open('labels.json'))
COT = ['cotton:bacterial_blight','cotton:curl_virus','cotton:healthy','cotton:herbicide_damage','cotton:jassids','cotton:leaf_reddening','cotton:leaf_variegation']
GRA = ['grape:bacterial_leaf_spot','grape:downy_mildew','grape:healthy','grape:powdery_mildew']
pdn = json.load(open('plantdoc_names.json')); pvn = json.load(open('pv_names.json'))
def small(b, S=288):
    im = Image.open(io.BytesIO(b)); im.draft('RGB', (2*S, 2*S)); im = im.convert('RGB')
    w, h = im.size; k = S / min(w, h)
    if k < 1: im = im.resize((round(w*k), round(h*k)), Image.BICUBIC)
    o = io.BytesIO(); im.save(o, 'JPEG', quality=90); return o.getvalue()
def splits_from_npz(name):
    s = np.load(f'feats_dinov2-small_{name}.npz')['s']; return [str(x) for x in s if x != 'train_aug']
VALF = {'plantdoc': 0.12, 'cotton': 0.1, 'grape': 0.1, 'other': 0.1, 'pv': 0.05}
SETS = [
 ('plantdoc', sorted(__import__('glob').glob('data/plantdoc/data/*.parquet')), lambda l: canon(pdn[l]), None),
 ('cotton', ['data/cotton/raw/train-00000-of-00001.parquet'], lambda l: COT[l], None),
 ('grape', ['data/grape/data/train-00000-of-00001.parquet'], lambda l: GRA[l], None),
 ('other', ['data/other/train.parquet'], lambda l: 'other', 1600),
 ('pv_test', ['data/pv/test.parquet'], lambda l: canon(pvn[l]), 60),
 ('pv', ['data/pv/train0.parquet', 'data/pv/train1.parquet'], lambda l: canon(pvn[l]), 350),
]
summary = {}
for name, files, lf, cap in SETS:
    if name in t: sp = ['test' if i in set(t[name]['test_rows']) else 'train' for i in range(len(t[name]['labels']))]; T = set(t[name]['test_rows'])
    else: sp = splits_from_npz(name)
    rng = random.Random(1234 + len(name)); rows = {'img': [], 'label': [], 'source': [], 'split': [], 'row': []}
    cnt = {}; base = 0
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64, columns=['image', 'label']):
            for img, l in zip(batch.column('image').to_pylist(), batch.column('label').to_pylist()):
                y = lf(l)
                if cap and cnt.get(y, 0) >= cap: continue
                cnt[y] = cnt.get(y, 0) + 1
                s = sp[base]
                if name in t: assert t[name]['labels'][base] == str(y)
                if s == 'train' and rng.random() < VALF.get(name, 0): s = 'val'
                if y is not None:
                    assert y in classes, y
                    rows['img'].append(small(img['bytes'])); rows['label'].append(y); rows['source'].append(name.replace('pv_test', 'pv'))
                    rows['split'].append(s); rows['row'].append(base)
                base += 1
    assert base == len(sp), (name, base, len(sp))
    pq.write_table(pa.table(rows), f'{OUT}/{name}.parquet')
    from collections import Counter; summary[name] = dict(Counter(rows['split'])); print(name, summary[name], flush=True)
json.dump(classes, open(f'{OUT}/labels.json', 'w'))
json.dump(summary, open(f'{OUT}/summary.json', 'w'))
