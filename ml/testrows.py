# recover which parquet rows ended up in the held-out test split (order-preserving)
import numpy as np, glob, json
out={}
for name, files in [('plantdoc', sorted(glob.glob('data/plantdoc/data/*.parquet'))), ('cotton',['data/cotton/raw/train-00000-of-00001.parquet']), ('grape',['data/grape/data/train-00000-of-00001.parquet'])]:
    d=np.load(f'feats_dinov2-small_{name}.npz'); s=d['s']; y=d['y']
    base=[(a,b) for a,b in zip(s,y) if a!='train_aug']
    out[name]={'files':files,'test_rows':[i for i,(a,b) in enumerate(base) if a=='test'],'labels':[str(b) for a,b in base]}
    print(name, len(base), 'test', len(out[name]['test_rows']))
json.dump(out, open('test_rows.json','w'))
