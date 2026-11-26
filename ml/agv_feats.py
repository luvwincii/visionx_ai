# DINOv2-small patch features for Agriculture-Vision tiles (RGB only).
# Train: all positive tokens + a sample of background tokens. Val: all tokens of a subset of tiles.
import sys, io, glob, numpy as np, torch, pyarrow.parquet as pq
from PIL import Image
from transformers import AutoModel
torch.set_num_threads(4); rng = np.random.default_rng(0)
CLASSES = ['drydown', 'nutrient_deficiency', 'weed_cluster', 'water', 'planter_skip', 'storm_damage']
S = 448; G = S // 14
model = AutoModel.from_pretrained('models/dinov2-small').eval()
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

def prep(b):
    im = Image.open(io.BytesIO(b)).convert('RGB').resize((S, S), Image.BICUBIC)
    return ((np.asarray(im, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)

def labels(row):
    Y = np.zeros((G * G, len(CLASSES)), np.uint8)
    for j, c in enumerate(CLASSES):
        m = np.asarray(Image.open(io.BytesIO(row['mask_' + c]['bytes'])).resize((G, G), Image.BOX), np.float32) / 255
        Y[:, j] = (m.reshape(-1) > 0.5)
    return Y

@torch.no_grad()
def tokens(batch):
    h = model(pixel_values=torch.tensor(np.stack(batch))).last_hidden_state[:, 1:]
    return h.numpy().astype(np.float16)

def run(files, out, mode, max_tiles=None, neg_frac=0.06):
    X, Y, T = [], [], []; n = 0; buf = []; lab = []
    cols = ['image_rgb'] + ['mask_' + c for c in CLASSES]
    def flush():
        nonlocal buf, lab
        if not buf: return
        F = tokens(buf)
        for f, y in zip(F, lab):
            if mode == 'train':
                pos = y.any(1); keep = pos | (rng.random(len(y)) < neg_frac)
                X.append(f[keep]); Y.append(y[keep])
            else:
                X.append(f); Y.append(y)
        buf, lab = [], []
    for fn in files:
        for b in pq.ParquetFile(fn).iter_batches(batch_size=16, columns=cols):
            for row in b.to_pylist():
                if max_tiles and n >= max_tiles: break
                if mode == 'val' and n % 3: n += 1; continue  # every 3rd tile for val
                buf.append(prep(row['image_rgb']['bytes'])); lab.append(labels(row)); n += 1
                if len(buf) == 16: flush()
                if n % 500 == 0: print(out, n, flush=True)
    flush()
    X = np.concatenate(X); Y = np.concatenate(Y)
    np.savez(out, X=X, Y=Y)
    print('saved', out, X.shape, Y.sum(0), flush=True)

if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'val':
        run(['data/agv/val-00000-of-00008.parquet'], 'agv_val.npz', 'val')
    else:
        run(sorted(glob.glob('data/agv/train-*.parquet')), 'agv_train.npz', 'train')
