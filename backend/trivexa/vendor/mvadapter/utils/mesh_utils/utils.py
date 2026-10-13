# Trivexa: trimmed to the one alias camera.py needs; the upstream module also pulls in nvdiffrast.
from typing import Union

import numpy as np
import torch

LIST_TYPE = Union[list, np.ndarray, torch.Tensor]
