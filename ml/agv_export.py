import torch, torch.nn as nn, json, os
from transformers import AutoModel
ck = torch.load('agv_head.pt', weights_only=False)
C = len(ck['classes'])
bb = AutoModel.from_pretrained('models/dinov2-small').eval()
head = nn.Sequential(nn.Linear(384, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, C))
head.load_state_dict(ck['state']); head.eval()
class Seg(nn.Module):
    def __init__(s):
        super().__init__(); s.bb = bb; s.head = head
        s.register_buffer('mu', ck['mu'].float()); s.register_buffer('sd', ck['sd'].float())
    def forward(s, pixel_values):
        t = s.bb(pixel_values=pixel_values).last_hidden_state[:, 1:]      # (1,1024,384)
        p = torch.sigmoid(s.head((t - s.mu) / s.sd))                        # (1,1024,C)
        return p, t
m = Seg().eval()
torch.onnx.export(m, (torch.randn(1, 3, 448, 448),), 'visionx-field-fp32.onnx', input_names=['pixel_values'],
                  output_names=['probs', 'tokens'], opset_version=17, dynamo=False)
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('visionx-field-fp32.onnx', 'visionx-field-int8.onnx', weight_type=QuantType.QUInt8)
json.dump({'classes': ck['classes'], 'thr': ck['thr'], 'size': 448, 'grid': 32}, open('field_labels.json', 'w'), indent=1)
print({f: round(os.path.getsize(f) / 1e6, 1) for f in ['visionx-field-fp32.onnx', 'visionx-field-int8.onnx']})
