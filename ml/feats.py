import sys, io, json, glob, numpy as np, torch, pyarrow.parquet as pq, random
from PIL import Image
import torchvision.transforms as T
from transformers import AutoModel
from labels import canon
torch.set_num_threads(4); random.seed(0); torch.manual_seed(0)
BACK = sys.argv[1] if len(sys.argv)>1 else 'dinov2-small'
model = AutoModel.from_pretrained('models/'+BACK).eval()
norm = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
center = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), norm])
aug = T.Compose([T.RandomResizedCrop(224, scale=(0.45,1.0), interpolation=T.InterpolationMode.BICUBIC), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                 T.ColorJitter(0.3,0.3,0.3,0.03), T.RandomApply([T.GaussianBlur(5,(0.1,1.5))],p=0.3), T.ToTensor(), norm])
@torch.no_grad()
def embed(tensors):
    out = model(pixel_values=torch.stack(tensors))
    h = out.last_hidden_state
    return torch.cat([h[:,0], h[:,1:].mean(1)], 1).numpy().astype(np.float16)

COT = ['cotton:bacterial_blight','cotton:curl_virus','cotton:healthy','cotton:herbicide_damage','cotton:jassids','cotton:leaf_reddening','cotton:leaf_variegation']
GRA = ['grape:bacterial_leaf_spot','grape:downy_mildew','grape:healthy','grape:powdery_mildew']
pdn = json.load(open('plantdoc_names.json')); pvn = json.load(open('pv_names.json'))
# name, files, labelfn, test_frac, n_aug_train, max_per_class
SETS = [
 ('plantdoc', sorted(glob.glob('data/plantdoc/data/*.parquet')), lambda l: canon(pdn[l]), 0.25, 4, None),
 ('cotton', ['data/cotton/raw/train-00000-of-00001.parquet'], lambda l: COT[l], 0.2, 3, None),
 ('grape', ['data/grape/data/train-00000-of-00001.parquet'], lambda l: GRA[l], 0.2, 2, None),
 ('other', ['data/other/train.parquet'], lambda l: 'other', 0.2, 1, 1600),
 ('pv_test', ['data/pv/test.parquet'], lambda l: canon(pvn[l]), 1.0, 0, 60),
 ('pv', ['data/pv/train0.parquet','data/pv/train1.parquet'], lambda l: canon(pvn[l]), 0.0, 1, 350),
]
only = sys.argv[2].split(',') if len(sys.argv)>2 else None
for name, files, lf, tf, naug, cap in SETS:
    if only and name not in only: continue
    X=[]; Y=[]; S=[]; buf=[]; meta=[]; cnt={}
    rng = random.Random(hash(name)%1000)
    def flush():
        global buf, meta
        if buf:
            X.append(embed(buf)); Y.extend([m[0] for m in meta]); S.extend([m[1] for m in meta]); buf=[]; meta=[]
    n=0
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64, columns=['image','label']):
            for img, l in zip(batch.column('image').to_pylist(), batch.column('label').to_pylist()):
                y = lf(l)
                if cap and cnt.get(y,0) >= cap: continue
                cnt[y]=cnt.get(y,0)+1
                im = Image.open(io.BytesIO(img['bytes'])); im.draft('RGB',(640,640)); im = im.convert('RGB'); im.thumbnail((640,640))
                split = 'test' if rng.random() < tf else 'train'
                buf.append(center(im)); meta.append((y, split))
                if split=='train':
                    for _ in range(naug-1): buf.append(aug(im)); meta.append((y,'train_aug'))
                if len(buf)>=48: flush()
                n+=1
                if n%2000==0: print(name, n, flush=True)
    flush()
    np.savez(f'feats_{BACK}_{name}.npz', X=np.concatenate(X), y=np.array(Y), s=np.array(S))
    print('saved', name, sum(len(x) for x in X), flush=True)
