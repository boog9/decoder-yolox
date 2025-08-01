# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Post-processing utilities for detector heatmaps."""

from __future__ import annotations

from typing import List

import numpy as np


def refine_kps(heatmaps: np.ndarray) -> np.ndarray:
    """Refine keypoints based on heatmap maxima."""
    if heatmaps.ndim != 3:
        raise ValueError("heatmaps must have shape (C, H, W)")
    kps: List[List[float]] = []
    for idx in range(heatmaps.shape[0]):
        y, x = divmod(int(np.argmax(heatmaps[idx])), heatmaps.shape[2])
        kps.append([float(x), float(y)])
    return np.asarray(kps, dtype=np.float32)


__all__ = ["refine_kps"]
