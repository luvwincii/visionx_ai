import sys, glob, io, json, time, torch, pandas as pd, numpy as np
from PIL import Image
from transformers import AutoModelForImageClassification
import torchvision.transforms as T
def make_proc(path):
    c=json.load(open(path+'/preprocessor_config.json'))
    size=c.get('size'); mean=c['image_mean']; std=c['image_std']
    interp = T.InterpolationMode.BICUBIC if c.get('resample')==3 else T.InterpolationMode.BILINEAR
    if isinstance(size,dict) and 'shortest_edge' in size:
        se=size['shortest_edge']
        if c.get('do_center_crop'): ops=[T.Resize(se,interpolation=interp),T.CenterCrop(c['crop_size']['height'])]
        elif 'crop_pct' in c: ops=[T.Resize(int(se/c['crop_pct']),interpolation=interp),T.CenterCrop(se)]
        else: ops=[T.Resize((se,se),interpolation=interp)]
    else:
        h = size['height'] if isinstance(size,dict) else size
        ops=[T.Resize((h,h),interpolation=interp)]
    tf=T.Compose(ops+[T.ToTensor(),T.Normalize(mean,std)])
    return lambda ims, return_tensors=None: {'pixel_values': torch.stack([tf(i) for i in ims])}
from labels import canon
torch.set_num_threads(4)
ds = sys.argv[1]  # parquet glob
names = json.load(open(sys.argv[2]))  # list of class names for dataset
models = sys.argv[3].split(',')
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(ds))])
if len(sys.argv)>4: df = df.sample(n=min(int(sys.argv[4]),len(df)), random_state=0)
def load(b):
    im=Image.open(io.BytesIO(b)); im.draft('RGB',(512,512)); im=im.convert('RGB'); im.thumbnail((384,384)); return im
imgs = [load(r['bytes']) for r in df['image']]
del df['image']
gts = [canon(names[l]) for l in df['label']]
for m in models:
    proc = make_proc('models/'+m); model = AutoModelForImageClassification.from_pretrained('models/'+m).eval()
    id2 = {int(k):canon(v) for k,v in model.config.id2label.items()}
    cov = set(id2.values())
    idx=[i for i,g in enumerate(gts) if g in cov]
    preds=[]; t=time.time()
    for b in range(0,len(idx),32):
        batch=[imgs[i] for i in idx[b:b+32]]
        with torch.no_grad(): lo=model(**proc(batch,return_tensors='pt')).logits
        preds += [id2[int(j)] for j in lo.argmax(-1)]
    g=[gts[i] for i in idx]
    acc=np.mean([p==q for p,q in zip(preds,g)])
    crop=np.mean([p and p.split(':')[0]==q.split(':')[0] for p,q in zip(preds,g)])
    hd=np.mean([(p is not None and p.endswith('healthy'))==q.endswith('healthy') for p,q in zip(preds,g)])
    print(f'{m:55s} n={len(idx):4d} exact={acc:.3f} crop={crop:.3f} healthy/diseased={hd:.3f} {1000*(time.time()-t)/len(idx):.0f}ms/img', flush=True)
