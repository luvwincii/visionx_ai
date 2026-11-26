# End-to-end check of the exported model on held-out real-world images (PlantDoc / cotton / grape test rows)
import onnxruntime as ort, numpy as np, json, io, sys, pyarrow.parquet as pq
from PIL import Image
t = json.load(open('test_rows.json')); classes = json.load(open('labels.json'))
MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)
def prep(b):
    im=Image.open(io.BytesIO(b)).convert('RGB'); w,h=im.size; k=256/min(w,h)
    im=im.resize((round(w*k),round(h*k)), Image.BICUBIC); w,h=im.size; l=(w-224)//2; tp=(h-224)//2
    a=(np.asarray(im.crop((l,tp,l+224,tp+224)),np.float32)/255-MEAN)/STD
    return a.transpose(2,0,1)[None]
res={}
for path in sys.argv[1:]:
    s=ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    for name in ['plantdoc','cotton','grape']:
        T=set(t[name]['test_rows']); L=t[name]['labels']; ok=hd=n=0; i=0
        for f in t[name]['files']:
            for b in pq.ParquetFile(f).iter_batches(batch_size=32,columns=['image']):
                for im in b.column('image').to_pylist():
                    if i in T:
                        x=prep(im['bytes']); p=(s.run(None,{'pixel_values':x})[0][0]+s.run(None,{'pixel_values':x[...,::-1].copy()})[0][0])/2
                        pr=classes[int(p.argmax())]; g=L[i]; ok+=pr==g; hd+=pr.endswith('healthy')==g.endswith('healthy'); n+=1
                    i+=1
        res[(path,name)]=(ok/n,hd/n,n); print(path, name, f'top1={ok/n:.3f} healthy/diseased={hd/n:.3f} n={n}', flush=True)
