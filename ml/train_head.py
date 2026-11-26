import numpy as np, json, sys, torch, torch.nn as nn
from collections import Counter
torch.manual_seed(0); np.random.seed(0)
B = sys.argv[1] if len(sys.argv)>1 else 'dinov2-small'
def load(n):
    d=np.load(f'feats_{B}_{n}.npz'); return d['X'].astype(np.float32), d['y'], d['s']
sets={n:load(n) for n in ['plantdoc','cotton','grape','other','pv_test','pv']}
tr_X=[];tr_y=[];te={}
for n,(X,y,s) in sets.items():
    if n=='pv_test': te[n]=(X,y); continue
    m = s!='test'; tr_X.append(X[m]); tr_y.append(y[m])
    if (s=='test').any(): te[n]=(X[s=='test'],y[s=='test'])
Xtr=np.concatenate(tr_X); ytr=np.concatenate(tr_y)
classes=sorted(set(ytr)); cid={c:i for i,c in enumerate(classes)}
print('train', Xtr.shape, 'classes', len(classes))
mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6
norm=lambda X:(X-mu)/sd
Xt=torch.tensor(norm(Xtr)); yt=torch.tensor([cid[c] for c in ytr])
# class-balanced weights: real-world sets upweighted vs lab PlantVillage
src=np.concatenate([np.full(len(t),n) for n,t in zip([k for k in sets if k!='pv_test'],tr_y)])
cnt=Counter(ytr)
w=np.array([1.0/np.sqrt(cnt[c]) for c in ytr])*np.where(src=='pv',0.5,1.0)
w=torch.tensor(w/w.mean(),dtype=torch.float32)
D=Xt.shape[1]; H=int(sys.argv[2]) if len(sys.argv)>2 else 768
model=nn.Sequential(nn.Dropout(0.2), nn.Linear(D,H), nn.GELU(), nn.Dropout(0.3), nn.Linear(H,len(classes)))
opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.05)
E=40; steps=E*(len(Xt)//512+1); sched=torch.optim.lr_scheduler.OneCycleLR(opt,1e-3,total_steps=steps)
lossf=nn.CrossEntropyLoss(reduction='none',label_smoothing=0.1)
for ep in range(E):
    model.train(); perm=torch.randperm(len(Xt))
    for i in range(0,len(Xt),512):
        b=perm[i:i+512]; x=Xt[b]+0.05*torch.randn_like(Xt[b])
        loss=(lossf(model(x),yt[b])*w[b]).mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
model.eval()
def evaluate(X,y,name):
    with torch.no_grad(): p=torch.softmax(model(torch.tensor(norm(X))),1).numpy()
    pred=[classes[i] for i in p.argmax(1)]
    top1=np.mean([a==b for a,b in zip(pred,y)])
    top3=np.mean([b in [classes[j] for j in np.argsort(-pi)[:3]] for pi,b in zip(p,y)])
    crop=np.mean([a.split(':')[0]==b.split(':')[0] for a,b in zip(pred,y)])
    hd=np.mean([a.endswith('healthy')==b.endswith('healthy') for a,b in zip(pred,y) if b!='other'])
    print(f'{name:10s} n={len(y):5d} top1={top1:.3f} top3={top3:.3f} crop={crop:.3f} healthy/diseased={hd:.3f}')
    return dict(n=int(len(y)),top1=float(top1),top3=float(top3),crop=float(crop),healthy_vs_diseased=float(hd)), p, pred
res={}
for n,(X,y) in te.items(): res[n]=evaluate(X,y,n)[0]
torch.save({'state':model.state_dict(),'mu':mu,'sd':sd,'classes':classes,'H':H}, f'head_{B}.pt')
json.dump(res, open(f'res_{B}.json','w'), indent=1)
