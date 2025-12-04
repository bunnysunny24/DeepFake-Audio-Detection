"""
Minimal safe collate function for DataLoader that tolerates missing/None keys.

Behavior:
- Accepts a batch (list of dict-like samples).
- Drops keys where any sample has `None` to avoid collate errors.
- For tensor values: attempts to `torch.stack` them when shapes match.
- For strings and scalars: returns a list of values.
- For lists/tuples: returns a list-of-lists.

Usage:
  from Models.safe_collate import safe_collate
  dl = DataLoader(ds, batch_size=8, collate_fn=safe_collate)

This is a conservative helper to avoid changing `__getitem__` semantics.
If you prefer placeholders instead, consider updating `MultiModalDeepfakeDataset.__getitem__`.
"""
from typing import List, Dict, Any
import torch


def safe_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        return {}

    # Collect keys present in samples
    keys = set()
    for s in batch:
        if isinstance(s, dict):
            keys.update(s.keys())

    out = {}
    for k in sorted(keys):
        vals = [s.get(k, None) if isinstance(s, dict) else None for s in batch]

        # If any value is None, skip this key to avoid collate errors.
        if any(v is None for v in vals):
            # skip optional/missing key
            continue

        # All present; handle common types
        first = vals[0]
        if isinstance(first, torch.Tensor):
            # verify shapes align; if not, return as list
            shapes_match = all((hasattr(v, 'shape') and tuple(v.shape) == tuple(first.shape)) for v in vals)
            if shapes_match:
                try:
                    out[k] = torch.stack(vals, dim=0)
                except Exception:
                    out[k] = vals
            else:
                out[k] = vals
        elif isinstance(first, (int, float, bool)):
            out[k] = torch.tensor(vals)
        elif isinstance(first, str):
            out[k] = vals
        elif isinstance(first, (list, tuple)):
            out[k] = vals
        else:
            # fallback: return list of values
            out[k] = vals

    return out
