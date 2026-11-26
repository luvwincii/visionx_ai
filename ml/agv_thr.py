# Precision / recall / IoU per class vs threshold on the threshold-tuning half of val (first half)
import numpy as np, torch, torch.nn as nn, json
ck = torch.load('agv_head.pt', weights_only=False); C = ck['classes']
head = nn.Sequential(nn.Linear(384, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, len(C)))
head.load_state_dict(ck['state']); head.eval()
va = np.load('agv_val.npz'); X = va['X'].astype(np.float32); Y = va['Y'] > 0
half = (len(X) // 1024 // 2) * 1024
with torch.no_grad(): P = torch.sigmoid(head((torch.tensor(X[:half]) - ck['mu']) / ck['sd'])).numpy()
Yh = Y[:half]
best = {}
for j, c in enumerate(C):
    print('==', c)
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p = P[:, j] > t; y = Yh[:, j]
        tp = (p & y).sum(); prec = tp / max(1, p.sum()); rec = tp / max(1, y.sum()); iou = tp / max(1, (p | y).sum())
        print(f'  t={t:.1f} precision={prec:.2f} recall={rec:.2f} IoU={iou:.3f} predicted={p.mean()*100:.1f}%')
