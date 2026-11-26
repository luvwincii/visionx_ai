import numpy as np, sys
from PIL import Image, ImageFilter
def analyze(path, th=0.07, tr=-0.01, bs=None):
    im = Image.open(path).convert('RGB'); im.thumbnail((960,960))
    a = np.asarray(im).astype(np.float32)/255
    H,W,_ = a.shape; bs = bs or max(4, round(max(H,W)/110))
    r,g,b = a[...,0],a[...,1],a[...,2]; s = r+g+b+1e-6
    exg = (2*g-r-b)/s
    vari = np.clip((g-r)/np.maximum(g+r-b, 0.05), -1, 1)
    mx, mn = a.max(-1), a.min(-1)
    water = (b >= 0.88*g) & (g >= r*0.95) & (mx-mn < 0.25) | ((b > g) & (b > r))
    dark = s < 0.20
    grey = ((mx-mn) < 0.06) & (mx > 0.45)
    invalid = water | dark | grey
    veg = (exg > 0.06) & ~invalid
    # block aggregation
    hb, wb = H//bs, W//bs
    def blk(x): return x[:hb*bs,:wb*bs].reshape(hb,bs,wb,bs).mean((1,3))
    vfrac = blk(veg.astype(np.float32)); inv = blk(invalid.astype(np.float32))
    vsum = blk(np.where(veg, vari, 0)); vmean = vsum/np.maximum(vfrac,1e-6)
    cls = np.full((hb,wb), 3)   # 3 = soil / no vegetation
    has = vfrac >= 0.18
    cls[has] = np.where(vmean[has]>=th, 0, np.where(vmean[has]>=tr, 1, 2))
    cls[inv > 0.5] = 4
    pal = np.array([[46,160,67],[227,179,65],[218,54,51],[120,104,88],[52,56,54]],np.uint8)
    out = Image.fromarray(pal[cls]).resize((wb*bs,hb*bs), Image.NEAREST)
    n = (cls<3).sum()
    print(path.split('/')[-1], 'of vegetation: green %.0f%% yellow %.0f%% red %.0f%%' % tuple(100*(cls==k).sum()/max(n,1) for k in range(3)), '| soil %.0f%% invalid %.0f%%' % (100*(cls==3).mean(), 100*(cls==4).mean()))
    base = np.asarray(im)[:hb*bs,:wb*bs]
    return Image.fromarray(np.concatenate([base, np.asarray(out)],1))
ims=[analyze(p) for p in sys.argv[1:]]
for i in ims: i.thumbnail((1400,420))
s=Image.new('RGB',(1400,sum(i.height for i in ims)),'white');y=0
for i in ims: s.paste(i,(0,y)); y+=i.height
s.save('maps2.png')
