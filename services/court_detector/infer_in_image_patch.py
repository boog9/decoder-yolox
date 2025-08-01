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
"""Patch for TennisCourtDetector with bbox extraction utilities."""

from typing import Dict, List

import cv2
import numpy as np
import torch


def extract_bounding_boxes(
    output: torch.Tensor | np.ndarray, threshold: float = 0.5
) -> List[List[int]]:
    """Extract bounding boxes from model output.

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
    """Wrapper around ``BallTrackerNet`` providing detection results."""

    def __init__(self, weights_path: str, device: str = "auto") -> None:
        from .tracknet import BallTrackerNet

        self.device = torch.device(
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.model = BallTrackerNet()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, frame_path: str) -> List[Dict[str, object]]:
        """Run detection on a frame and return ByteTrack-compatible results."""
        import torchvision.transforms as transforms
        from PIL import Image

        img = Image.open(frame_path).convert("RGB")
        tensor = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)

        heatmap = output[0, 0].detach().cpu().numpy()
        boxes = extract_bounding_boxes(output, threshold=0.5)

        results: List[Dict[str, object]] = []
        for x1, y1, x2, y2 in boxes:
            region = heatmap[y1:y2, x1:x2]
            score = float(region.max()) if region.size else 0.0
            results.append({"class": 100, "score": score, "bbox": [x1, y1, x2, y2]})
        return results


__all__ = ["extract_bounding_boxes", "CourtDetector"]
