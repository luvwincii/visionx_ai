# Export the Kaggle checkpoint to ONNX (fp32 + int8) locally; same Net as ml/kaggle/train_seg.py.
import sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import SegformerModel
CK, OUT = sys.argv[1], sys.argv[2]; S = 512
class Net(nn.Module):
    def __init__(self, D=768):
        super().__init__()
        self.enc = SegformerModel.from_pretrained('nvidia/mit-b3')
        hs = self.enc.config.hidden_sizes
        self.proj = nn.ModuleList([nn.Conv2d(h, D, 1) for h in hs])
        self.fuse = nn.Sequential(nn.Conv2d(D * len(hs), D, 1, bias=False), nn.BatchNorm2d(D), nn.ReLU(inplace=True))
        self.drop = nn.Dropout2d(0.1)
        self.cls = nn.Conv2d(D, 6, 1)
    def forward(self, x):
        hs = list(self.enc(pixel_values=x, output_hidden_states=True).hidden_states)
        self.nhs = len(hs)
        for i, f in enumerate(hs):
            if f.dim() == 3:
                B, N, Cc = f.shape; h = int(round(N ** 0.5)); hs[i] = f.transpose(1, 2).reshape(B, Cc, h, N // h)
        feats = [[f for f in hs if f.dim() == 4 and f.shape[1] == p.in_channels][-1] for p in self.proj]
        size = feats[0].shape[-2:]
        ups = [F.interpolate(p(f), size=size, mode='bilinear', align_corners=False) for p, f in zip(self.proj, feats)]
        return torch.sigmoid(self.cls(self.drop(self.fuse(torch.cat(ups[::-1], 1)))))
net = Net(); print(net.load_state_dict(torch.load(CK, map_location='cpu'))); net.eval()
x = torch.randn(1, 3, S, S)
with torch.no_grad():
    a = net(x); n1 = net.nhs; b = net(x); print('hidden_states per call', n1, net.nhs, 'repeat diff', float((a - b).abs().max()))
torch.onnx.export(net, (x,), f'{OUT}/visionx-field-seg.onnx', input_names=['pixel_values'], output_names=['probs'], opset_version=17, dynamo=False)
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(f'{OUT}/visionx-field-seg.onnx', f'{OUT}/visionx-field-seg-int8.onnx', weight_type=QuantType.QUInt8)
import onnxruntime as ort
for f in ['visionx-field-seg.onnx', 'visionx-field-seg-int8.onnx']:
    o = ort.InferenceSession(f'{OUT}/{f}', providers=['CPUExecutionProvider']).run(None, {'pixel_values': x.numpy()})[0]
    print(f, round(os.path.getsize(f'{OUT}/{f}') / 1e6, 1), 'MB, max abs diff', float(np.abs(o - a.numpy()).max()))
