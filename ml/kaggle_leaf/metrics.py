# Metrics from eval_onnx_full.py outputs. usage: metrics.py name=probs.npz ...  (run from /home/user/work)
import numpy as np, json, sys
d = np.load('leaf_ft/test_crops.npz'); y = d['y']; S = d['s']; classes = json.load(open('labels.json'))
yi = np.array([classes.index(c) for c in y]); healthy = np.array([c.endswith('healthy') for c in classes])
out = {}
for arg in sys.argv[1:]:
    name, f = arg.split('='); z = np.load(f); r = {}
    for tta in [False, True]:
        P = (z['p'] + z['pf']) / 2 if tta else z['p']
        for s in ['plantdoc', 'cotton', 'grape', 'other', 'pv_test']:
            m = S == s; p = P[m]; t = yi[m]; pr = p.argmax(1)
            top3 = np.mean([a in np.argsort(-q)[:3] for q, a in zip(p, t)])
            nm = t != classes.index('other')
            hd = float(np.mean(healthy[pr[nm]] == healthy[t[nm]])) if nm.any() else None
            e = dict(n=int(m.sum()), top1=float((pr == t).mean()), top3=float(top3), healthy_vs_diseased=hd)
            if s == 'plantdoc':
                conf = p.max(1); e['confident'] = {str(th): dict(coverage=float((conf >= th).mean()), acc=float((pr[conf >= th] == t[conf >= th]).mean()) if (conf >= th).any() else None) for th in [0.5, 0.6, 0.7, 0.8, 0.9]}
            r[s + ('_tta' if tta else '')] = e
    out[name] = r
    for k, e in r.items():
        print(f"{name:12s} {k:13s} n={e['n']:4d} top1={e['top1']:.3f} top3={e['top3']:.3f} hd={e['healthy_vs_diseased'] if e['healthy_vs_diseased'] is None else round(e['healthy_vs_diseased'],3)}" +
              ('  conf: ' + ' '.join(f"{th}:{c['coverage']*100:.0f}%/{(c['acc'] or 0)*100:.1f}%" for th, c in e['confident'].items()) if 'confident' in e else ''))
json.dump(out, open('leaf_ft/metrics_' + '_'.join(a.split('=')[0] for a in sys.argv[1:]) + '.json', 'w'), indent=1)
