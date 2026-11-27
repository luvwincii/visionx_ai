# VisionX leaf classifier: end-to-end fine-tuning of DINOv2 on Kaggle (GPU T4 x2).
# Two runs in parallel, one per GPU: dinov2-small (GPU0) and dinov2-base (GPU1).
# Data: private dataset elxanaakiiyev/visionx-leaf-data (built by ml/kaggle_leaf/pack.py) with the exact
# VisionX train/test rows. Model selection uses split=='val' (carved from TRAIN rows only).
# Test rows are only evaluated once, after training, for reporting.
import os, io, sys, json, time, math, random, glob, subprocess, copy

MODE = 'full'          # 'smoke' | 'full'
TRAIN_MIN = 4 if MODE == 'smoke' else 85
ARCHS = {'small': ('facebook/dinov2-small', 0, 4e-5, 0.8, 3), 'base': ('facebook/dinov2-base', 1, 3e-5, 0.8, 2)}
WORK = '/kaggle/working'
T0 = time.time()
ROLE = os.environ.get('VX_ROLE')

def log(*a):
    print(f'[{ROLE or "main"} {(time.time() - T0) / 60:6.1f} min]', *a, flush=True)

if ROLE is None:
    me = os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else None
    if not me or not os.path.exists(me):
        cand = glob.glob('/kaggle/src/*.py') + glob.glob('/kaggle/working/*.py')
        me = cand[0] if cand else None
    log('script', me, 'files', glob.glob('/kaggle/input/*'))
    import torch
    ng = torch.cuda.device_count(); log('gpus', ng)
    procs = []
    for name, (_, gpu, *_r) in ARCHS.items():
        env = dict(os.environ, VX_ROLE=name, CUDA_VISIBLE_DEVICES=str(gpu % max(ng, 1)))
        f = open(f'{WORK}/log_{name}.txt', 'w')
        procs.append((name, subprocess.Popen([sys.executable, me], env=env, stdout=f, stderr=subprocess.STDOUT), f))
    for name, p, f in procs:
        p.wait(); f.close(); log(name, 'exit', p.returncode)
        print(open(f'{WORK}/log_{name}.txt').read()[-6000:], flush=True)
    sys.exit(0)

# ---------------- worker ----------------
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pyarrow.parquet as pq
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoModel
from collections import Counter
torch.manual_seed(0); random.seed(0); np.random.seed(0)
HF, _, LR_BB, LLRD, NW = ARCHS[ROLE]
DEV = 'cuda'
D = os.path.dirname(glob.glob('/kaggle/input/**/labels.json', recursive=True)[0])
classes = json.load(open(f'{D}/labels.json')); cid = {c: i for i, c in enumerate(classes)}
rows = []
for f in sorted(glob.glob(f'{D}/*.parquet')):
    tb = pq.read_table(f).to_pydict()
    name = os.path.basename(f)[:-8]
    for img, y, src, sp in zip(tb['img'], tb['label'], tb['source'], tb['split']):
        rows.append(dict(img=img, y=cid[y], src=src, split=sp, set=name))
tr = [r for r in rows if r['split'] == 'train']; va = [r for r in rows if r['split'] == 'val']
log('rows', len(rows), 'train', len(tr), 'val', len(va), Counter(r['split'] for r in rows))

norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
center = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), norm])
aug = T.Compose([T.RandomResizedCrop(224, scale=(0.3, 1.0), ratio=(0.75, 1.333), interpolation=T.InterpolationMode.BICUBIC),
                 T.RandomHorizontalFlip(), T.RandomVerticalFlip(p=0.3), T.RandomApply([T.RandomRotation(20)], p=0.3),
                 T.RandomApply([T.ColorJitter(0.35, 0.35, 0.3, 0.03)], p=0.8), T.RandomGrayscale(0.03),
                 T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.2), T.ToTensor(), norm, T.RandomErasing(p=0.2, scale=(0.02, 0.15))])

class DS(Dataset):
    def __init__(s, rs, tf): s.rs, s.tf = rs, tf
    def __len__(s): return len(s.rs)
    def __getitem__(s, i):
        r = s.rs[i]; im = Image.open(io.BytesIO(r['img'])).convert('RGB')
        return s.tf(im), r['y']

cnt = Counter(r['y'] for r in tr)
w = np.array([1.0 / math.sqrt(cnt[r['y']]) * (0.5 if r['src'] == 'pv' else 1.0) for r in tr])
BS = 64
sampler = WeightedRandomSampler(torch.tensor(w / w.sum(), dtype=torch.double), num_samples=10 ** 7, replacement=True)
dl = DataLoader(DS(tr, aug), batch_size=BS, sampler=sampler, num_workers=NW, pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4)

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.bb = AutoModel.from_pretrained(HF, drop_path_rate=0.1)
        Dm = 2 * s.bb.config.hidden_size
        s.head = nn.Sequential(nn.LayerNorm(Dm), nn.Dropout(0.2), nn.Linear(Dm, len(classes)))
    def forward(s, x):
        h = s.bb(pixel_values=x).last_hidden_state
        return s.head(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1))
model = Net().to(DEV)
L = model.bb.config.num_hidden_layers
groups = [{'params': list(model.head.parameters()), 'lr': 1e-3, 'weight_decay': 0.01}]
for n_, p in model.bb.named_parameters():
    if 'encoder.layer.' in n_: lid = int(n_.split('encoder.layer.')[1].split('.')[0]) + 1
    elif 'layernorm' in n_ and 'encoder' not in n_: lid = L + 1
    else: lid = 0
    nd = p.ndim == 1 or 'position_embeddings' in n_ or 'cls_token' in n_ or 'mask_token' in n_ or 'register' in n_
    groups.append({'params': [p], 'lr': LR_BB * LLRD ** (L + 1 - lid), 'weight_decay': 0.0 if nd else 0.05})
for g in groups: g['base_lr'] = g['lr']
opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
scaler = torch.amp.GradScaler()
ema = copy.deepcopy(model).eval()
for p in ema.parameters(): p.requires_grad_(False)
lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

@torch.no_grad()
def predict(m, rs, flip=False):
    m.eval(); out = []
    for x, _ in DataLoader(DS(rs, center), batch_size=128, num_workers=NW):
        x = x.to(DEV)
        with torch.autocast('cuda', dtype=torch.float16):
            p = F.softmax(m(x).float(), 1)
            if flip: p = (p + F.softmax(m(torch.flip(x, [3])).float(), 1)) / 2
        out.append(p.cpu())
    return torch.cat(out).numpy()

def val_score(m):
    p = predict(m, va); y = np.array([r['y'] for r in va]); ok = p.argmax(1) == y
    pdm = np.array([r['set'] == 'plantdoc' for r in va])
    return 0.5 * ok[pdm].mean() + 0.5 * ok.mean(), ok[pdm].mean(), ok.mean()

budget = TRAIN_MIN * 60; WARM = 300; total = None; step = 0; tstart = time.time(); best = (-1, None, None)
n_evals = 3 if MODE == 'smoke' else 10; next_eval = None; ema_decay = 0.999
it = iter(dl); model.train(); tl = []
while True:
    x, y = next(it); x = x.to(DEV, non_blocking=True); y = y.to(DEV, non_blocking=True)
    if total is None and step == 60:
        sps = (time.time() - t60) / 40; total = int(budget * 0.93 / sps); next_eval = total // n_evals
        log(f'{sps:.3f}s/step -> total steps {total} (~{total * BS / len(tr):.1f} epochs)')
    if step == 20: t60 = time.time()
    T_ = total or 10 ** 9
    f = min(1.0, (step + 1) / WARM) * (0.5 * (1 + math.cos(math.pi * min(1.0, step / T_))))
    for g in opt.param_groups: g['lr'] = g['base_lr'] * f
    with torch.autocast('cuda', dtype=torch.float16):
        loss = lossf(model(x).float(), y)
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
    d = min(ema_decay, (step + 1) / (step + 10))
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(d).add_(pm.detach(), alpha=1 - d)
    tl.append(loss.item()) if step % 10 == 0 else None
    step += 1
    if total and (step % next_eval == 0 or step >= total):
        for tag, m in [('raw', model), ('ema', ema)]:
            sc = val_score(m)
            log(f'step {step}/{total} loss {np.mean(tl[-50:]):.3f} {tag} val score {sc[0]:.4f} plantdoc_val {sc[1]:.4f} all_val {sc[2]:.4f}')
            if sc[0] > best[0]: best = (sc[0], {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, (tag, step, sc))
        model.train()
        if step >= total: break
log('best', best[2])
model.load_state_dict(best[1]); model.eval()
torch.save({'state': {k: v.half() if v.is_floating_point() else v for k, v in best[1].items()}, 'classes': classes, 'hf': HF,
            'best': str(best[2])}, f'{WORK}/leaf_{ROLE}.pt')

# ---- one-shot test report (not used for any selection) ----
res = {'best': str(best[2])}
for setname in ['plantdoc', 'cotton', 'grape', 'other', 'pv_test']:
    rs = [r for r in rows if r['split'] == 'test' and r['set'] == setname]
    if not rs: continue
    y = np.array([r['y'] for r in rs])
    for flip in [False, True]:
        p = predict(model, rs, flip); pr = p.argmax(1)
        top3 = np.mean([yy in np.argsort(-pp)[:3] for pp, yy in zip(p, y)])
        hd = np.mean([classes[a].endswith('healthy') == classes[b].endswith('healthy') for a, b in zip(pr, y) if classes[b] != 'other'] or [float('nan')])
        k = f'{setname}{"_tta" if flip else ""}'
        res[k] = dict(n=len(y), top1=float((pr == y).mean()), top3=float(top3), hd=float(hd))
        log(k, res[k])
json.dump(res, open(f'{WORK}/res_{ROLE}.json', 'w'), indent=1)
log('done')
