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
"""Tests for run_tcd CLI wrapper."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_run_tcd_module():
    """Load run_tcd module from file path."""
    path = Path("services/court_detector/run_tcd.py")
    spec = importlib.util.spec_from_file_location("run_tcd", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_tcd"] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def test_run_tcd_creates_json(tmp_path, monkeypatch):
    """Ensure wrapper writes JSON with expected keys."""
    run_tcd = load_run_tcd_module()

    frame = tmp_path / "frame.png"
    frame.write_bytes(b"fake")
    out_json = tmp_path / "out.json"

    def dummy_run_infer(frame_path: str, device: str, tmp_png_out: str) -> None:
        Path(tmp_png_out).write_bytes(b"png")

    monkeypatch.setattr(run_tcd, "run_upstream_infer", dummy_run_infer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tcd",
            "--frame",
            str(frame),
            "--out",
            str(out_json),
            "--device",
            "cpu",
        ],
    )
    run_tcd.main()
    data = json.loads(out_json.read_text())
    assert data["device"] == "cpu"
    assert data["weights_path"] == "/app/TennisCourtDetector/model.pt"
    assert "keypoints" in data
    assert "homography" in data

