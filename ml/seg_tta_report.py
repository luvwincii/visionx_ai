# Summarize npz files from seg_eval_tta.py (local ONNX) or the Kaggle kernel (torch fp32).
# Thresholds chosen per class on the 'tune' set (pixel IoU), optionally overridden with --thr a,b,c,d,e.
import sys, json, numpy as np
args = [a for a in sys.argv[1:] if not a.startswith('--')]
fixed = [a for a in sys.argv[1:] if a.startswith('--thr=')]
out = {}
for f in args:
    d = np.load(f); T = d['T']; C = list(d['C'])
    for v, vn in enumerate(['plain', 'tta']):
        if fixed:
            th = [float(x) for x in fixed[0][6:].split(',')]; idx = [int(np.argmin(np.abs(T - t))) for t in th]
        else:
            tu = d['tune'][v, 0]; idx = list((tu[0] / np.maximum(1, tu.sum(0))).argmax(1))
        r = {'thr': [float(T[i]) for i in idx]}
        for sname in ['c240', 't800']:
            for lvl, ln in [(0, 'pixel'), (1, 'zone')]:
                if sname == 'c240' and lvl == 1: continue
                A = d[sname][v, lvl]; o = {}
                for j, c in enumerate(C):
                    tp, fp, fn = A[:, j, idx[j]]
                    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
                    o[c] = {'IoU': round(tp / max(1, tp + fp + fn), 3), 'P': round(P, 3), 'R': round(R, 3), 'F1': round(2 * P * R / max(1e-9, P + R), 3)}
                for k in ['IoU', 'P', 'R', 'F1']: o['mean_' + k] = round(float(np.mean([o[c][k] for c in C])), 3)
                r[f'{sname}_{ln}'] = o
        out[f'{f}:{vn}'] = r
        print(f, vn, 'thr', r['thr'], 'c240', r['c240_pixel']['mean_IoU'], 't800', r['t800_pixel']['mean_IoU'], 'zoneF1', r['t800_zone']['mean_F1'])
if '--json' in ''.join(sys.argv): print(json.dumps(out, indent=1))
