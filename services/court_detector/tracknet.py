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
"""Original BallTrackerNet network for tennis court detection."""

from __future__ import annotations



import torch
from torch import nn


class ConvBlock(nn.Module):
    """Convolution, batch norm and ReLU block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.block(x)


class BallTrackerNet(nn.Module):
    """Ball tracking network used by TennisCourtDetector."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = ConvBlock(3, 64)
        for i in range(2, 18):
            setattr(self, f"conv{i}", ConvBlock(64, 64))
        self.conv18 = ConvBlock(64, 15)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        for i in range(1, 19):
            conv = getattr(self, f"conv{i}")
            x = conv(x)
        return x


__all__ = ["BallTrackerNet"]
