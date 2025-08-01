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
"""Canonical BallTrackerNet model definition."""

from __future__ import annotations

# NOTE: This file is *verbatim* from yastrebksv/TennisCourtDetector @unknown
# Do not modify without retraining weights.

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Convolution followed by ReLU and batch normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.block(x)


class BallTrackerNet(nn.Module):
    """Tennis court line detector network."""

    def __init__(self) -> None:  # noqa: D401 - short init
        super().__init__()
        # Layer widths double at each stage to match the pretrained weights.
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)
        self.conv8 = ConvBlock(256, 256)
        self.conv9 = ConvBlock(256, 512)
        self.conv10 = ConvBlock(512, 512)
        self.conv11 = ConvBlock(512, 512)
        self.conv12 = ConvBlock(512, 512)
        self.conv13 = ConvBlock(512, 512)
        self.conv14 = ConvBlock(512, 512)
        self.conv15 = ConvBlock(512, 512)
        self.conv16 = ConvBlock(512, 512)
        self.conv17 = ConvBlock(512, 512)
        self.conv18 = ConvBlock(512, 15, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        return x


__all__ = ["BallTrackerNet"]
