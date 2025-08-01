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

        # The original model defines each block explicitly so that the
        # state_dict keys match the upstream checkpoint exactly.
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = ConvBlock(64, 64)
        self.conv5 = ConvBlock(64, 64)
        self.conv6 = ConvBlock(64, 64)
        self.conv7 = ConvBlock(64, 64)
        self.conv8 = ConvBlock(64, 64)
        self.conv9 = ConvBlock(64, 64)
        self.conv10 = ConvBlock(64, 64)
        self.conv11 = ConvBlock(64, 64)
        self.conv12 = ConvBlock(64, 64)
        self.conv13 = ConvBlock(64, 64)
        self.conv14 = ConvBlock(64, 64)
        self.conv15 = ConvBlock(64, 64)
        self.conv16 = ConvBlock(64, 64)
        self.conv17 = ConvBlock(64, 64)
        self.conv18 = ConvBlock(64, 15)

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
