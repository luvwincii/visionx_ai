import re
def canon(lbl):
    s = lbl.lower().replace('_',' ')
    s = re.sub(r'\s+',' ',s)
    crops = [('bell pepper','pepper'),('pepper','pepper'),('corn','corn'),('maize','corn'),('apple','apple'),('blueberry','blueberry'),('cherry','cherry'),
             ('grape','grape'),('orange','orange'),('peach','peach'),('potato','potato'),('raspberry','raspberry'),('soy','soybean'),('squash','squash'),
             ('strawberry','strawberry'),('tomato','tomato'),('wheat','wheat'),('rice','rice')]
    crop=None
    for k,v in crops:
        if k in s: crop=v; break
    if crop is None: return None
    rules=[('scab','scab'),('black rot','black_rot'),('cedar','rust'),('rust','rust'),('powdery','powdery_mildew'),('gray leaf spot','gray_leaf_spot'),('cercospora','gray_leaf_spot'),
           ('northern','leaf_blight'),('esca','esca'),('isariopsis','leaf_blight'),('greening','hlb'),('haunglongbing','hlb'),('bacterial','bacterial_spot'),
           ('early blight','early_blight'),('late blight','late_blight'),('scorch','leaf_scorch'),('mold','leaf_mold'),('septoria','septoria'),('spider','spider_mites'),
           ('target','target_spot'),('yellow leaf curl','ylcv'),('yellow virus','ylcv'),('mosaic','mosaic'),('leaf blight','leaf_blight'),('blight','leaf_blight'),('leaf spot','bacterial_spot'),('healthy','healthy')]
    dis='healthy'
    for k,v in rules:
        if k in s: dis=v; break
    if crop=='corn' and dis=='rust': dis='common_rust'
    return f'{crop}:{dis}'
