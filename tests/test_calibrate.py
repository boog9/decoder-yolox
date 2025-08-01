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
"""Tests for court detection calibration."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")
import numpy as np

from services.court_detector.calibrate import calibrate


class DummyDetector:
    """Mock CourtDetector used during tests."""

    def detect(self, image):  # type: ignore[no-untyped-def]
        return np.zeros((15, 2, 2), dtype=np.float32)


@pytest.fixture(autouse=True)
def mock_detector(monkeypatch):
    module = types.SimpleNamespace(CourtDetector=DummyDetector)
    monkeypatch.setitem(sys.modules, "tennis_court_detector", module)
    yield
    monkeypatch.delitem(sys.modules, "tennis_court_detector", raising=False)


def test_calibrate(tmp_path: Path) -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    frame = tmp_path / "frame.png"
    cv2.imwrite(str(frame), img)

    weights = tmp_path / "model.pt"
    weights.write_bytes(b"dummy")

    out = tmp_path / "meta.json"
    meta = calibrate(str(frame), str(out), str(weights), device="cpu")

    assert out.exists()
    data = json.loads(out.read_text())
    assert data == meta
    assert isinstance(data["heatmaps"], list)
    assert len(data["heatmaps"]) == 15
    assert "frame_id" in data
    assert "timestamp_ms" in data
    assert "model_sha" in data
    assert data["device"] == "cpu"


def test_missing_weights(tmp_path: Path) -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    frame = tmp_path / "frame.png"
    cv2.imwrite(str(frame), img)

    out = tmp_path / "meta.json"
    with pytest.raises(FileNotFoundError):
        calibrate(str(frame), str(out), str(tmp_path / "no.pt"), device="cpu")
