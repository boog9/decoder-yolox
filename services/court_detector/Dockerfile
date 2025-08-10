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

# Modern, supported base with PyTorch+CUDA ready (no legacy base-cuda)
# torch 2.4.1 + cuda12.1 + cudnn9 runtime image
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1

# System deps (git, ffmpeg for video frames if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ca-certificates ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone upstream TennisCourtDetector (ref can be branch, tag, or commit)
ARG TCD_REF=main
RUN git clone https://github.com/yastrebksv/TennisCourtDetector.git /app/TennisCourtDetector \
    && cd /app/TennisCourtDetector && git checkout ${TCD_REF}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    MPLBACKEND=Agg
RUN python -m pip install --upgrade pip && \
    pip install \
      "numpy>=1.26,<2" \
      "scipy>=1.10,<1.14" \
      "matplotlib>=3.7,<3.9" \
      "tqdm>=4.66,<5" \
      "sympy>=1.12,<1.13" \
      "Pillow>=10,<11" \
      "opencv-python-headless>=4.8,<5" \
      "gdown>=5" || \
    (pip install "torchvision==0.19.*"; exit 0)

# Download upstream pretrained weights to expected path in repo root (per README).
# README points to Google Drive; using gdown by file id is stable.
# id: 1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG
RUN gdown --fuzzy "https://drive.google.com/uc?id=1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG" \
    -O /app/TennisCourtDetector/model.pt \
    && test -s /app/TennisCourtDetector/model.pt

# Our thin wrapper provides a stable CLI that:
#  - takes a single frame
#  - runs upstream inference
#  - emits JSON with 14+1 keypoints (if available) and homography (if upstream returns it)
COPY run_tcd.py /app/run_tcd.py
COPY entrypoint.sh /usr/local/bin/tennis-court-detect
RUN chmod +x /usr/local/bin/tennis-court-detect

WORKDIR /app
ENTRYPOINT ["tennis-court-detect"]

# Basic runtime health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c 'import torch, cv2; print("ok")' || exit 1
