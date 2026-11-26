import torch, torch.nn as nn, numpy as np, json, sys
from transformers import AutoModel
B = sys.argv[1] if len(sys.argv)>1 else 'dinov2-small'
ck = torch.load(f'head_{B}.pt', weights_only=False)
classes = ck['classes']; H = ck['H']
bb = AutoModel.from_pretrained('models/'+B).eval()
D = 2*bb.config.hidden_size
head = nn.Sequential(nn.Dropout(0.2), nn.Linear(D,H), nn.GELU(), nn.Dropout(0.3), nn.Linear(H,len(classes)))
head.load_state_dict(ck['state']); head.eval()
class Full(nn.Module):
    def __init__(s):
        super().__init__(); s.bb=bb; s.head=head
        s.register_buffer('mu', torch.tensor(ck['mu'],dtype=torch.float32)); s.register_buffer('sd', torch.tensor(ck['sd'],dtype=torch.float32))
    def forward(s, pixel_values):
        h = s.bb(pixel_values=pixel_values).last_hidden_state
        f = torch.cat([h[:,0], h[:,1:].mean(1)],1)
        return torch.softmax(s.head((f-s.mu)/s.sd), -1)
m = Full().eval()
x = torch.randn(1,3,224,224)
torch.onnx.export(m, (x,), f'visionx-leaf-fp32.onnx', input_names=['pixel_values'], output_names=['probs'], opset_version=17, dynamo=False)
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('visionx-leaf-fp32.onnx', 'visionx-leaf-int8.onnx', weight_type=QuantType.QUInt8)
json.dump(classes, open('labels.json','w'))
import os; print({f: round(os.path.getsize(f)/1e6,1) for f in ['visionx-leaf-fp32.onnx','visionx-leaf-int8.onnx']})
