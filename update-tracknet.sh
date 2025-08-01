#!/usr/bin/env bash
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

# This script synchronizes tracknet.py with the official TennisCourtDetector
# repository and rebuilds the court-detector Docker image with the updated
# files. It then performs a smoke test to verify correct loading of the
# weights and validates the convolution channel widths.

set -euo pipefail

##############################################
# 0. Context
# Repository: decoder-yolox
# Service:    services/court_detector
# Base image: decoder/base-cuda
##############################################

############### 1. Sync tracknet.py ###############
curl -sSL https://raw.githubusercontent.com/yastrebksv/TennisCourtDetector/main/tracknet.py \
     -o services/court_detector/tracknet.py

############### 2. Patch channel CFG ###############
apply_patch <<'PATCH'
*** Begin Patch
*** Update File: services/court_detector/tracknet.py
@@
-        self.conv8 = ConvBlock(256, 256)
-        self.conv9 = ConvBlock(256, 512)
-        self.conv10 = ConvBlock(512, 512)
-        self.conv11 = ConvBlock(512, 512)
-        self.conv12 = ConvBlock(512, 512)
-        self.conv13 = ConvBlock(512, 512)
-        self.conv14 = ConvBlock(512, 512)
-        self.conv15 = ConvBlock(512, 512)
-        self.conv16 = ConvBlock(512, 512)
-        self.conv17 = ConvBlock(512, 512)
-        self.conv18 = ConvBlock(512, 15)
+        # === Official weight channel widths ===
+        self.conv8  = ConvBlock(256, 512)
+        self.conv9  = ConvBlock(512, 512)
+        self.conv10 = ConvBlock(512, 512)
+        self.conv11 = ConvBlock(512, 256)
+        self.conv12 = ConvBlock(256, 256)
+        self.conv13 = ConvBlock(256, 256)
+        self.conv14 = ConvBlock(256, 128)
+        self.conv15 = ConvBlock(128, 128)
+        self.conv16 = ConvBlock(128,  64)
+        self.conv17 = ConvBlock( 64,  64)
+        self.conv18 = ConvBlock( 64,  15)  # 3×3, padding=1
*** End Patch
PATCH

############### 3. Fix relative imports ###############
sed -i 's/^from utils /from .utils /'     services/court_detector/postprocess.py
sed -i 's/^from court_reference /from .court_reference /' \
       services/court_detector/homography.py

############### 4. Rebuild Docker (NO CACHE!) ###############
DOCKER_BUILDKIT=1 docker build --no-cache \
  -t decoder/court-detector \
  -f services/court_detector/Dockerfile .

############### 5. Smoke-test ###############
docker run --gpus all --rm -v $(pwd)/data:/data decoder/court-detector \
  --frame data/frames_min/000000.png \
  --out   data/court_meta.json

############### 6. Channel-list sanity check ###############
docker run --rm --entrypoint "" decoder/court-detector \
  python - <<'PY'
import tennis_court_detector.tracknet as t
ch = [getattr(t.BallTrackerNet(), f'conv{i}').block[0].out_channels for i in range(1,19)]
print('Conv outs \u2192', ch)
assert ch == [64,64,128,128,256,256,256,512,512,512,256,256,256,128,128,64,64,15], "Channel list mismatch!"
PY
##############################################
# If the script prints no AssertionError and court_meta.json is created,
# the integration is up to date with the official TennisCourtDetector.
##############################################
