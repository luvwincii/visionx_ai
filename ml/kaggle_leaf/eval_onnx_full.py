# Evaluate ONNX leaf models (batch-1, as in the browser) on leaf_ft/test_crops.npz with and without flip TTA.
# usage: eval_onnx_full.py out.npz model.onnx   (run from /home/user/work)
import onnxruntime as ort, numpy as np, sys, time
d = np.load('leaf_ft/test_crops.npz'); X = d['X']
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
o = ort.SessionOptions(); o.intra_op_num_threads = int(sys.argv[3]) if len(sys.argv) > 3 else 4
s = ort.InferenceSession(sys.argv[2], o, providers=['CPUExecutionProvider']); inn = s.get_inputs()[0].name
P = []; PF = []; t0 = time.time()
for i, im in enumerate(X):
    x = ((im.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
    P.append(s.run(None, {inn: x})[0][0]); PF.append(s.run(None, {inn: np.ascontiguousarray(x[..., ::-1])})[0][0])
    if i % 500 == 0: print(i, f'{time.time()-t0:.0f}s', flush=True)
np.savez(sys.argv[1], p=np.array(P), pf=np.array(PF))
print('done', time.time() - t0)
