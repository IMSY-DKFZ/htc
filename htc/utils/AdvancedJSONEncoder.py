# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import numpy as np
import torch


# Based on: https://github.com/hmallen/numpyencoder/blob/master/numpyencoder/numpyencoder.py
class AdvancedJSONEncoder(json.JSONEncoder):
    """Custom encoder for numpy/tensor data types."""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        elif isinstance(obj, torch.Tensor):
            return self.default(obj.cpu().numpy())

        elif isinstance(obj, Path):
            return str(obj)

        elif hasattr(obj, "to_json"):
            # E.g. LabelMapping class
            return obj.to_json()

        return json.JSONEncoder.default(self, obj)
