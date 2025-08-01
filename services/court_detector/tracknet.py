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
"""Simplified BallTrackerNet network used for tennis court detection."""

from __future__ import annotations

from typing import List

import torch
from torch import nn


class ConvBNReLU(nn.Module):
    """Basic convolution block with batch norm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class BallTrackerNet(nn.Module):
    """Lightweight model for ball tracking."""

    def __init__(self) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_c = 3
        for _ in range(17):
            layers.append(ConvBNReLU(in_c, 64, 3, 1, 1))
            in_c = 64
        self.features = nn.Sequential(*layers)
        self.conv18 = ConvBNReLU(64, 15, 3, 1, 1)  # changed to 15 for model.pt compatibility

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        x = self.features(x)
        return self.conv18(x)


__all__ = ["BallTrackerNet"]
