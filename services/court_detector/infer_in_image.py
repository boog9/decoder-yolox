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
"""Court detection utilities for a single image."""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
import torch


def extract_bounding_boxes(
    output: torch.Tensor | np.ndarray, threshold: float = 0.5
) -> List[List[int]]:
    """Extract bounding boxes from a heatmap output.

    Args:
        output: Raw output tensor from ``BallTrackerNet``.
        threshold: Heatmap threshold for detections.

    Returns:
        List of bounding boxes ``[x1, y1, x2, y2]``.
    """
    if isinstance(output, torch.Tensor):
        arr = output.detach().cpu().numpy()
    else:
        arr = np.asarray(output)

    heatmap = arr[0, 0] if arr.ndim == 4 else arr[0]
    mask = (heatmap > threshold).astype("uint8")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[List[int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([int(x), int(y), int(x + w), int(y + h)])
    return boxes


class CourtDetector:
    """Lightweight wrapper for ``BallTrackerNet``."""

    def __init__(self, weights_path: str, device: str = "auto") -> None:
        from .tracknet import BallTrackerNet

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")

        self.device = torch.device(device)
        # Load network weights directly on the target device
        self.model = BallTrackerNet().to(self.device)
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state)
        # ➜ inference only: disable gradients and ensure eval mode
        self.model.eval()

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Return raw heatmaps for a BGR image.

        Args:
            frame: Image in BGR format as a NumPy array.

        Returns:
            Array of shape ``(15, H, W)`` containing heatmaps.
        """
        from PIL import Image
        from torchvision.transforms import ToTensor

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = ToTensor()(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)

        return output.squeeze(0).detach().cpu().numpy()


__all__ = ["extract_bounding_boxes", "CourtDetector"]
