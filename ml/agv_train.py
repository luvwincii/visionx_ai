# A) supervised patch segmentation head on frozen DINOv2 tokens
# B) label-free per-image anomaly score (whitened distance from the image's own mean token)
import numpy as np, torch, torch.nn as nn, json
from sklearn.metrics import roc_auc_score
torch.manual_seed(0); np.random.seed(0)
CLASSES = ['drydown', 'nutrient_deficiency', 'weed_cluster', 'water', 'planter_skip', 'storm_damage']
G = 32
tr = np.load('agv_train.npz'); va = np.load('agv_val.npz')
Y0 = tr['Y']
rare = Y0[:, 1:].any(1)                      # any class other than drydown
dry_only = (Y0[:, 0] > 0) & ~rare
bg = ~Y0.any(1)
rs = np.random.default_rng(0)
keep = rare | (dry_only & (rs.random(len(Y0)) < 0.25)) | (bg & (rs.random(len(Y0)) < 0.6))
idx = np.flatnonzero(keep)
Xtr = torch.tensor(tr['X'][idx].astype(np.float32)); Ytr = torch.tensor(Y0[idx].astype(np.float32))
del Y0
Xva = va['X'].astype(np.float32); Yva = va['Y'].astype(np.float32)
print('train tokens', Xtr.shape, 'pos per class', Ytr.sum(0).tolist())
print('val tokens', Xva.shape, 'pos per class', Yva.sum(0).tolist())
mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
pos = Ytr.sum(0); neg = len(Ytr) - pos
head = nn.Sequential(nn.Linear(384, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, len(CLASSES)))
opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.05)
E = 25; steps = E * (len(Xtr) // 2048 + 1)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
lossf = nn.BCEWithLogitsLoss(pos_weight=torch.clamp(neg / pos.clamp(min=1), max=20))
Xn = (Xtr - mu) / sd
for ep in range(E):
    head.train(); perm = torch.randperm(len(Xn))
    for i in range(0, len(Xn), 2048):
        b = perm[i:i + 2048]
        loss = lossf(head(Xn[b] + 0.05 * torch.randn_like(Xn[b])), Ytr[b])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
head.eval()
with torch.no_grad():
    P = torch.sigmoid(head((torch.tensor(Xva) - mu) / sd)).numpy()

# pick per-class threshold on half of val, report on the other half (no peeking)
ntile = len(Xva) // (G * G); half = (ntile // 2) * G * G
res = {'A_supervised': {}, 'B_label_free': {}}
ths = []
for j, c in enumerate(CLASSES):
    best = (0, 0.5)
    for t in np.arange(0.2, 0.96, 0.05):
        p = P[:half, j] > t; y = Yva[:half, j] > 0
        iou = (p & y).sum() / max(1, (p | y).sum()); best = max(best, (iou, t))
    t = best[1]; ths.append(float(t))
    p = P[half:, j] > t; y = Yva[half:, j] > 0
    iou = (p & y).sum() / max(1, (p | y).sum())
    auc = roc_auc_score(y, P[half:, j]) if y.any() and (~y).any() else float('nan')
    res['A_supervised'][c] = {'IoU': round(float(iou), 3), 'AUROC': round(float(auc), 3), 'thr': round(float(t), 2), 'val_pos_tokens': int(y.sum())}
any_y = Yva[half:].max(1) > 0
res['A_supervised']['any_anomaly_AUROC'] = round(float(roc_auc_score(any_y, P[half:].max(1))), 3)
# B: per image whitened distance
scores = np.zeros(len(Xva), np.float32)
for k in range(ntile):
    F = Xva[k * G * G:(k + 1) * G * G]
    m = F.mean(0); s = F.std(0) + 1e-3
    scores[k * G * G:(k + 1) * G * G] = (((F - m) / s) ** 2).mean(1)
res['B_label_free']['any_anomaly_AUROC'] = round(float(roc_auc_score(any_y, scores[half:])), 3)
for j, c in enumerate(CLASSES):
    y = Yva[half:, j] > 0
    if y.any(): res['B_label_free'][c + '_AUROC'] = round(float(roc_auc_score(y, scores[half:])), 3)
print(json.dumps(res, indent=1))
json.dump(res, open('agv_results.json', 'w'), indent=1)
torch.save({'state': head.state_dict(), 'mu': mu, 'sd': sd, 'classes': CLASSES, 'thr': ths}, 'agv_head.pt')
