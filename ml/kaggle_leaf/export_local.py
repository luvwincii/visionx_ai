# Export a fine-tuned checkpoint (leaf_<arch>.pt from the Kaggle kernel) to ONNX fp32 + int8, drop-in for web/app/classifier.js:
# input 'pixel_values' [1,3,224,224] (ImageNet-normalised), output 'probs' [1,49] softmax, same labels.json order.
# usage: export_local.py leaf_small.pt models/dinov2-small outprefix   (run from /home/user/work)
import torch, torch.nn as nn, json, sys, os
from transformers import AutoModel
ck = torch.load(sys.argv[1], weights_only=False); classes = ck['classes']
assert classes == json.load(open('labels.json'))
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.bb = AutoModel.from_pretrained(sys.argv[2])
        Dm = 2 * s.bb.config.hidden_size
        s.head = nn.Sequential(nn.LayerNorm(Dm), nn.Dropout(0.2), nn.Linear(Dm, len(classes)))
    def forward(s, pixel_values):
        h = s.bb(pixel_values=pixel_values).last_hidden_state
        return torch.softmax(s.head(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)), -1)
m = Net(); missing = m.load_state_dict({k: v.float() for k, v in ck['state'].items()}, strict=True); m.eval()
out = sys.argv[3]
torch.onnx.export(m, (torch.randn(1, 3, 224, 224),), out + '-fp32.onnx', input_names=['pixel_values'], output_names=['probs'], opset_version=17, dynamo=False)
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(out + '-fp32.onnx', out + '-int8.onnx', weight_type=QuantType.QUInt8)
print({f: round(os.path.getsize(f) / 1e6, 1) for f in [out + '-fp32.onnx', out + '-int8.onnx']})
