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
"""Homography computation helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .court_reference import COURT_POINTS


def compute_homography(points: np.ndarray) -> Optional[np.ndarray]:
    """Compute a homography from detected points to ``COURT_POINTS``."""
    if points.shape != COURT_POINTS.shape:
        return None

    try:
        import cv2
    except Exception:  # pragma: no cover - optional dependency
        return None

    try:
        H, _ = cv2.findHomography(points, COURT_POINTS)
        return H
    except Exception:  # pragma: no cover - openCV may fail
        return None


__all__ = ["compute_homography"]
