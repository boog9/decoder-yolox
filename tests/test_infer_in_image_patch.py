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
"""Tests for infer_in_image_patch utilities."""

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from services.court_detector.infer_in_image import extract_bounding_boxes


def test_extract_bounding_boxes_simple() -> None:
    heatmap = np.zeros((1, 1, 5, 5), dtype=np.float32)
    heatmap[0, 0, 1:3, 2:4] = 0.6
    boxes = extract_bounding_boxes(heatmap, threshold=0.5)
    assert boxes == [[2, 1, 4, 3]]
