"""
Lightweight inference script. Loads a checkpoint saved by `train_perfect.py` or compatible checkpoint
and runs inference over a dataset or a list of files. Writes out a CSV with file path and predicted label
and probabilities.

Usage example (PowerShell):
python predict_perfect.py --data_root "F:\deepfake\backup\LAV_df_fixed" --json_path "F:\deepfake\backup\LAV_df_fixed\metadata.json" --checkpoint "F:\deepfake\backup\Models\server_outputs\checkpoint_epoch_10_valacc_0.75.pth" --out predictions.csv
"""
import os
import sys
import argparse
import csv
import json

import torch
import torch.nn as nn

repo_dir = os.path.dirname(__file__)
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from dataset_loader import MultiModalDeepfakeDataset
import importlib
mm_mod = importlib.import_module('multi_modal_model')

# find model class
import inspect
ModelClass = None
for name, obj in inspect.getmembers(mm_mod, inspect.isclass):
    try:
        if issubclass(obj, nn.Module) and obj.__module__ == mm_mod.__name__:
            ModelClass = obj
            break
    except Exception:
        continue
if ModelClass is None:
    raise RuntimeError('No model class found in multi_modal_model.py')

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', required=True)
parser.add_argument('--json_path', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--out', default='predictions.csv')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

# dataset
ds = MultiModalDeepfakeDataset(args.data_root, args.json_path, phase='test')
loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=getattr(ds, 'collate_fn_skip_none', None))

# model
model = ModelClass()
ck = torch.load(args.checkpoint, map_location='cpu')
if 'state_dict' in ck:
    sd = ck['state_dict']
elif 'model_state_dict' in ck:
    sd = ck['model_state_dict']
else:
    sd = ck
try:
    model.load_state_dict(sd, strict=False)
except Exception:
    # try flexible key rename
    sd2 = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd2, strict=False)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

rows = []
with torch.no_grad():
    for batch in loader:
        if isinstance(batch, dict):
            inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            file_paths = inputs.get('file_path', None)
            labels = inputs.get('label', None)
        else:
            inputs, labels = batch
            file_paths = None
    # Filter out metadata-derived fields so predictions rely only on raw modalities
    forbidden = {'metadata_features', 'file_path', 'transcript', 'timestamps',
             'original_audio', 'original_video_frames', 'fake_periods', 'deepfake_type'}
    filtered_inputs = {k: v for k, v in inputs.items() if torch.is_tensor(v) and k not in forbidden}
    out = model(**filtered_inputs)
        logits = out['logits'] if isinstance(out, dict) and 'logits' in out else out
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        for i in range(probs.shape[0]):
            fp = file_paths[i] if isinstance(file_paths, list) else None
            rows.append([fp, int(preds[i]), float(probs[i,0]), float(probs[i,1])])

# write CSV
with open(args.out, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['file_path','pred','prob_real','prob_fake'])
    w.writerows(rows)

print('Wrote predictions to', args.out)
