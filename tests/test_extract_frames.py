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
"""Tests for extract_frames module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from services.extractor.extract_frames import extract_frames


def create_test_video(path: Path, frame_count: int = 10, fps: int = 10) -> None:
    """Create a simple test video with blank frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 64))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(frame_count):
        writer.write(frame)
    writer.release()


def test_extract_frames(tmp_path: Path) -> None:
    video = tmp_path / "test.mp4"
    out_dir = tmp_path / "out"
    create_test_video(video, frame_count=10, fps=10)

    count = extract_frames(str(video), str(out_dir), fps=5)
    assert count == 5
    extracted = sorted(p.name for p in out_dir.glob("*.png"))
    assert len(extracted) == 5
    assert extracted[0] == "000000.png"
