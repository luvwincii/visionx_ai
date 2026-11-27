# VisionX field anomaly segmentation — CONTINUED training from the 3-epoch checkpoint (GPU T4 x2).
# Fixes: encoder stages are called directly (no output_hidden_states recorder -> no accumulation/leak),
# streaming evaluation (no full-res maps kept in RAM).
# Data: Agriculture-Vision 2021 (RGB only), research / non-commercial use.
# Model: SegFormer (ImageNet-pretrained MiT encoder), fine-tuned end to end, multi-label sigmoid head.
import os, io, sys, json, time, math, random, glob, shutil, subprocess

MODE = 'smoke'           # 'smoke' = short pipeline check, 'full' = real run
BACKBONE = 'nvidia/mit-b3'
TRAIN_HOURS = 0.06 if MODE == 'smoke' else 2.0   # resume run: kernel total must stay under ~2.5 h
CLASSES = ['drydown', 'nutrient_deficiency', 'weed_cluster', 'water', 'planter_skip', 'storm_damage']
HF_REPO = 'Vadim21221/agri-vision-2021-segmentation'
N_TRAIN_SHARDS = 2 if MODE == 'smoke' else 26
S = 512
T0 = time.time()
def log(*a):
    print(f'[{(time.time() - T0) / 60:6.1f} min]', *a, flush=True)

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'onnx', 'onnxruntime', 'onnxscript'], check=False)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from huggingface_hub import hf_hub_download
from transformers import SegformerModel

WORK = '/kaggle/working'
DATA = '/tmp/agv'
os.makedirs(DATA, exist_ok=True); os.makedirs(WORK, exist_ok=True)
log('disk', shutil.disk_usage('/kaggle' if os.path.isdir('/kaggle') else '.'), 'gpus', torch.cuda.device_count())

# ---------- data ----------
def shard(name):
    return hf_hub_download(HF_REPO, f'data/{name}.parquet', repo_type='dataset', local_dir=DATA)
train_files = [shard(f'train-{i:05d}-of-00026') for i in (range(N_TRAIN_SHARDS) if MODE == 'full' else [0, 13])]
tune_files = [shard('val-00001-of-00008')]
test_files = [shard('val-00002-of-00008'), shard('val-00003-of-00008')]
log('downloaded', len(train_files), 'train shards')

MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
COLS = ['image_rgb'] + ['mask_' + c for c in CLASSES]

def decode(r):
    img = np.asarray(Image.open(io.BytesIO(r['image_rgb']['bytes'])).convert('RGB'), np.float32) / 255
    m = np.stack([np.asarray(Image.open(io.BytesIO(r['mask_' + c]['bytes'])), np.uint8) > 0 for c in CLASSES])
    return img, m

def repeats(m, rng):
    p = m.reshape(len(CLASSES), -1).mean(1)
    if p[5] > 0.005 or p[4] > 0.005: return 3          # storm damage, planter skip (rarest)
    if p[3] > 0.01 or p[2] > 0.01: return 2            # water, weed
    if p[1] > 0.01: return 1                           # nutrient deficiency
    return 1 if rng.random() < 0.6 else 0              # drydown-only / background tiles: downsample

def augment(img, m, rng):
    k = rng.randint(0, 3); img = np.rot90(img, k, (0, 1)); m = np.rot90(m, k, (1, 2))
    if rng.random() < 0.5: img = img[:, ::-1]; m = m[:, :, ::-1]
    img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
    m = torch.from_numpy(np.ascontiguousarray(m)).float()
    if rng.random() < 0.5:   # random scale crop: robustness to other ground resolutions (drones)
        sc = rng.uniform(0.5, 1.0); cs = int(S * sc)
        y, x = rng.randint(0, S - cs), rng.randint(0, S - cs)
        img = F.interpolate(img[None, :, y:y + cs, x:x + cs], size=(S, S), mode='bilinear', align_corners=False)[0]
        m = F.interpolate(m[None, :, y:y + cs, x:x + cs], size=(S, S), mode='nearest')[0]
    if rng.random() < 0.5:
        img = (img * rng.uniform(0.85, 1.15) + rng.uniform(-0.06, 0.06)).clamp(0, 1)
    return img, m

def norm(img):  # CHW tensor in [0,1]
    return (img - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]

class TrainStream(IterableDataset):
    def __init__(self, files): self.files = files; self.epoch = 0
    def __iter__(self):
        wi = get_worker_info(); wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = random.Random(1000 * self.epoch + wid)
        files = self.files[:]; rng.shuffle(files)
        for f in files[wid::nw] if len(files) >= nw else files:
            pf = pq.ParquetFile(f); rgs = list(range(pf.num_row_groups)); rng.shuffle(rgs)
            for g in rgs:
                rows = pf.read_row_group(g, columns=COLS).to_pylist(); rng.shuffle(rows)
                for r in rows:
                    img, m = decode(r)
                    for _ in range(repeats(m, rng)):
                        a, b = augment(img, m, rng)
                        yield norm(a), b

def eval_rows(files, every=1, limit=None):
    n = 0
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=32, columns=COLS):
            for r in b.to_pylist():
                if n % every == 0:
                    yield r
                n += 1
                if limit and n >= limit * every: return

# ---------- model ----------
class Net(nn.Module):
    def __init__(self, D=768):
        super().__init__()
        self.enc = SegformerModel.from_pretrained(BACKBONE)
        hs = self.enc.config.hidden_sizes
        self.proj = nn.ModuleList([nn.Conv2d(h, D, 1) for h in hs])
        self.fuse = nn.Sequential(nn.Conv2d(D * len(hs), D, 1, bias=False), nn.BatchNorm2d(D), nn.ReLU(inplace=True))
        self.drop = nn.Dropout2d(0.1)
        self.cls = nn.Conv2d(D, len(CLASSES), 1)
    def stages(self, x):
        # Walk the MiT stages by hand: patch-embed -> blocks -> layer norm -> (B,C,H,W).
        # Nothing is recorded or kept between calls, so it is safe under DataParallel and in ONNX export.
        feats = []; h = x
        for st in self.enc.stages:
            h, H, W = st.patch_embeddings(h)
            for blk in st.blocks:
                o = blk(h, H, W); h = o[0] if isinstance(o, (tuple, list)) else o
            h = st.layer_norm(h)
            h = h.reshape(h.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()
            feats.append(h)
        return feats
    def forward(self, x):
        feats = self.stages(x)
        size = feats[0].shape[-2:]
        ups = [F.interpolate(p(f), size=size, mode='bilinear', align_corners=False) for p, f in zip(self.proj, feats)]
        return self.cls(self.drop(self.fuse(torch.cat(ups[::-1], 1))))   # (B, C, S/4, S/4)

dev = 'cuda'
net = Net()
CK = glob.glob('/kaggle/input/**/ckpt.pt', recursive=True)
log('resume from', CK)
print(net.load_state_dict(torch.load(CK[0], map_location='cpu')), flush=True)
net = net.to(dev)
model = nn.DataParallel(net) if torch.cuda.device_count() > 1 else net
enc = [p for n, p in net.named_parameters() if n.startswith('enc.')]
dec = [p for n, p in net.named_parameters() if not n.startswith('enc.')]
LR_ENC, LR_DEC = 3e-5, 3e-4
opt = torch.optim.AdamW([{'params': enc, 'lr': LR_ENC}, {'params': dec, 'lr': LR_DEC}], weight_decay=0.01)
scaler = torch.amp.GradScaler('cuda')
pos_w = torch.tensor([2.4, 3.3, 4.4, 5.7, 10.0, 10.0], device=dev)[None, :, None, None]

def loss_fn(logits, target):
    t = F.adaptive_avg_pool2d(target, logits.shape[-2:])        # soft targets at 1/4 resolution
    bce = F.binary_cross_entropy_with_logits(logits, t, pos_weight=pos_w)
    p = torch.sigmoid(logits)
    inter = (p * t).sum((0, 2, 3)); den = (p + t).sum((0, 2, 3))
    dice = 1 - ((2 * inter + 1) / (den + 1))
    return bce + dice.mean()

BATCH = 16 if torch.cuda.device_count() > 1 else 8
ds = TrainStream(train_files)
budget = TRAIN_HOURS * 3600
train_start = time.time(); step = 0; WARM = 200 if MODE == 'full' else 20
last_ckpt = time.time(); ep = 0; running = None; win_t = time.time()
log('training', BACKBONE, 'batch', BATCH, 'budget h', TRAIN_HOURS)
done = False
while not done:
    ds.epoch = ep + 7          # different shuffles from the first run
    dl = DataLoader(ds, batch_size=BATCH, num_workers=4, pin_memory=True, persistent_workers=False, prefetch_factor=4)
    for x, y in dl:
        frac = (time.time() - train_start) / budget
        if frac >= 1: done = True; break
        sc = min(1.0, (step + 1) / WARM) * (0.5 * (1 + math.cos(math.pi * min(1.0, frac))))
        for i, g in enumerate(opt.param_groups): g['lr'] = (LR_ENC if i == 0 else LR_DEC) * max(sc, 0.01)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(x)
        loss = loss_fn(logits.float(), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt); scaler.update(); step += 1
        running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
        if step % 200 == 0:
            log(f'ep {ep} step {step} loss {running:.4f} lr {opt.param_groups[0]["lr"]:.2e} img/s window {200 * BATCH / (time.time() - win_t):.1f} '
                f'gpu mem GB {torch.cuda.memory_allocated() / 1e9:.2f}')
            win_t = time.time()
        if time.time() - last_ckpt > 3600:
            torch.save(net.state_dict(), f'{WORK}/ckpt.pt'); last_ckpt = time.time(); log('checkpoint saved')
    ep += 1
torch.save(net.state_dict(), f'{WORK}/ckpt.pt')
log('training done, steps', step, 'epochs', ep)

# ---------- evaluation (streaming: only TP/FP/FN counters are kept) ----------
net.eval()
TH = torch.tensor(np.round(np.arange(0.1, 0.951, 0.05), 2), device=dev)
@torch.no_grad()
def batches(rows):
    batch, ys = [], []
    for r in rows:
        img, m = decode(r)
        batch.append(norm(torch.from_numpy(img).permute(2, 0, 1))); ys.append(torch.from_numpy(m))
        if len(batch) == 16:
            yield torch.stack(batch), torch.stack(ys); batch, ys = [], []
    if batch: yield torch.stack(batch), torch.stack(ys)
@torch.no_grad()
def probs(x):
    with torch.autocast('cuda', dtype=torch.float16):
        lg = net(x.to(dev)).float()
    return torch.sigmoid(F.interpolate(lg, size=(S, S), mode='bilinear', align_corners=False))
@torch.no_grad()
def sweep(rows):   # counts per class per threshold
    tp = torch.zeros(len(CLASSES), len(TH), device=dev, dtype=torch.float64); fp = tp.clone(); fn = tp.clone(); n = 0
    for x, y in batches(rows):
        P = probs(x); Y = y.to(dev); n += len(x)
        for j in range(len(CLASSES)):
            yj = Y[:, j].bool()
            for k, t in enumerate(TH):
                pj = P[:, j] > t
                tp[j, k] += (pj & yj).sum(); fp[j, k] += (pj & ~yj).sum(); fn[j, k] += (~pj & yj).sum()
    return tp.cpu().numpy(), fp.cpu().numpy(), fn.cpu().numpy(), n

nt = 100 if MODE == 'smoke' else 1200
tp, fp, fn, _ = sweep(eval_rows(tune_files, every=1, limit=nt))
iou = tp / np.maximum(1, tp + fp + fn)
thr = [round(float(TH[int(iou[j].argmax())]), 2) for j in range(len(CLASSES))]
log('thresholds', dict(zip(CLASSES, thr)))
nte = 100 if MODE == 'smoke' else 800
tp, fp, fn, ntiles = sweep(eval_rows(test_files, every=1, limit=nte))
pix = {}
for j, c in enumerate(CLASSES):
    k = int(np.argmin(np.abs(TH.cpu().numpy() - thr[j])))
    a, b, d = tp[j, k], fp[j, k], fn[j, k]
    pix[c] = {'IoU': round(a / max(1, a + b + d), 4), 'precision': round(a / max(1, a + b), 4), 'recall': round(a / max(1, a + d), 4), 'thr': thr[j]}
pix['mean_IoU_5'] = round(float(np.mean([pix[c]['IoU'] for c in CLASSES[:5]])), 4)
res = {'backbone': BACKBONE, 'mode': MODE, 'resumed_from': 'visionx-seg-ckpt1 (3 epochs, 9896 steps)', 'steps': step, 'epochs': ep,
       'train_hours': round((time.time() - train_start) / 3600, 2), 'test_tiles': ntiles, 'pixel_level': pix}
json.dump(res, open(f'{WORK}/metrics.json', 'w'), indent=1)
json.dump({'classes': CLASSES, 'thr': thr, 'size': S, 'out_stride': 4, 'backbone': BACKBONE}, open(f'{WORK}/field_labels.json', 'w'), indent=1)
log(json.dumps(res, indent=1))

# ---------- export ----------
class Export(nn.Module):
    def __init__(self, n): super().__init__(); self.n = n
    def forward(self, pixel_values): return torch.sigmoid(self.n(pixel_values))
ex = Export(net.float().cpu()).eval()
torch.onnx.export(ex, (torch.randn(1, 3, S, S),), f'{WORK}/visionx-field-seg.onnx', input_names=['pixel_values'],
                  output_names=['probs'], opset_version=17, dynamo=False)
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(f'{WORK}/visionx-field-seg.onnx', f'{WORK}/visionx-field-seg-int8.onnx', weight_type=QuantType.QUInt8)
import onnxruntime as ort
xx = torch.randn(1, 3, S, S)
with torch.no_grad(): ref = ex(xx).numpy()
for f in ['visionx-field-seg.onnx', 'visionx-field-seg-int8.onnx']:
    o = ort.InferenceSession(f'{WORK}/{f}', providers=['CPUExecutionProvider']).run(None, {'pixel_values': xx.numpy()})[0]
    log(f, 'shape', o.shape, 'size MB', round(os.path.getsize(f'{WORK}/{f}') / 1e6, 1), 'max abs diff', float(np.abs(o - ref).max()))
log('ALL DONE')
