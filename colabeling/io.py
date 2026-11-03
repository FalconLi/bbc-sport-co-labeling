"""NPZ interchange for aligned views and task-specific labels, without pickle."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from scipy import sparse

from .kernels import validate_views


def load_dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        indices = sorted({int(m[1]) for key in archive.files
                          if (m := re.fullmatch(r"view_(\d+)(?:_data)?", key))})
        if indices != list(range(len(indices))) or len(indices) < 2:
            raise ValueError("NPZ needs contiguous view_0, view_1, ... matrices (at least two views).")
        views = []
        for i in indices:
            key = f"view_{i}"
            if key in archive:
                views.append(archive[key].copy())
            else:
                required = [f"{key}_{suffix}" for suffix in ["data", "indices", "indptr", "shape"]]
                if not all(name in archive for name in required):
                    raise ValueError(f"Incomplete CSR matrix for {key}.")
                shape = tuple(archive[f"{key}_shape"])
                x = sparse.csr_matrix((archive[f"{key}_data"], archive[f"{key}_indices"], archive[f"{key}_indptr"]), shape=shape)
                x.check_format(full_check=True)
                views.append(x)
        data = {key: archive[key].copy() for key in ["y", "known_mask", "bag_ids", "known_normal_mask", "sample_ids"] if key in archive}
    return validate_views(views), data


def save_dataset(path, views, **arrays):
    views = validate_views(views)
    payload = dict(arrays)
    for i, x in enumerate(views):
        if sparse.issparse(x):
            x = sparse.csr_matrix(x)
            for name in ["data", "indices", "indptr"]:
                payload[f"view_{i}_{name}"] = getattr(x, name)
            payload[f"view_{i}_shape"] = np.array(x.shape)
        else:
            payload[f"view_{i}"] = x
    if any(np.asarray(a).dtype.kind == "O" for a in payload.values()):
        raise ValueError("Use numeric/string arrays rather than object arrays in NPZ files.")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)
