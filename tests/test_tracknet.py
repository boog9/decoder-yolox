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
"""Tests for the BallTrackerNet model."""

import pytest

torch = pytest.importorskip("torch")

from services.court_detector.tracknet import BallTrackerNet


def test_state_dict_keys() -> None:
    """Verify parameter names match the upstream format."""
    model = BallTrackerNet()
    keys = model.state_dict().keys()
    assert "conv1.block.0.weight" in keys
    assert model.state_dict()["conv18.block.0.weight"].shape[0] == 15


def test_state_dict_loading(tmp_path) -> None:
    """Ensure a saved state dict loads without missing or unexpected keys."""
    model = BallTrackerNet()
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)

    new_model = BallTrackerNet()
    result = new_model.load_state_dict(torch.load(path))
    assert not result.missing_keys
    assert not result.unexpected_keys
