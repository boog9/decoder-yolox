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

FROM decoder/base-cuda

ARG WEIGHTS_URL="https://drive.google.com/uc?id=1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG"

# Install gdown, matplotlib and pin NumPy below 2 for PyTorch compatibility
RUN pip install --no-cache-dir 'numpy<2' gdown matplotlib && rm -rf /root/.cache/pip

RUN git clone --depth 1 https://github.com/boog9/TennisCourtDetector /tmp/TennisCourtDetector

# TennisCourtDetector repository ships all code in the root directory.
# Package it under /app/tennis_court_detector so it can be imported.

RUN mkdir -p /app/tennis_court_detector && \
    cp /tmp/TennisCourtDetector/tracknet.py /app/tennis_court_detector/ && \
    cp /tmp/TennisCourtDetector/postprocess.py /app/tennis_court_detector/ && \
    cp /tmp/TennisCourtDetector/court_reference.py /app/tennis_court_detector/ && \
    cp /tmp/TennisCourtDetector/homography.py /app/tennis_court_detector/ && \
    cp /tmp/TennisCourtDetector/infer_in_image.py /app/tennis_court_detector/ && \
    sed -i '1s;^;from __future__ import annotations\n;' /app/tennis_court_detector/infer_in_image.py && \
    sed -i \
        -e 's/^from tracknet /from .tracknet /' \
        -e 's/^from postprocess /from .postprocess /' \
        -e 's/^from court_reference /from .court_reference /' \
        -e 's/^from homography /from .homography /' \
        -e 's/^from utils /from .utils /' \
        /app/tennis_court_detector/infer_in_image.py && \
    cp /tmp/TennisCourtDetector/infer_in_video.py /app/tennis_court_detector/ && \
    cp /tmp/TennisCourtDetector/utils.py /app/tennis_court_detector/ && \
    cp /tmp/TennisCourtDetector/base_trainer.py /app/tennis_court_detector/ && \
    cp /tmp/TennisCourtDetector/base_validator.py /app/tennis_court_detector/ && \
    printf 'from .infer_in_image import CourtDetector\n' > /app/tennis_court_detector/__init__.py

# Patch imports in postprocess.py to use package-relative utils module
RUN sed -i 's/^from utils /from .utils /' /app/tennis_court_detector/postprocess.py

# Patch imports in homography.py to use package-relative court_reference module
RUN sed -i 's/^from court_reference /from .court_reference /' /app/tennis_court_detector/homography.py

# Download pretrained weights.
RUN mkdir -p /opt/weights \
    && gdown "$WEIGHTS_URL" -O /opt/weights/model.pt \
    && test -s /opt/weights/model.pt || (echo "Weights download failed" && exit 1) \
    && python3 -c 'import torch; torch.load("/opt/weights/model.pt", map_location="cpu")' \
    || (echo "Weights file not usable as PyTorch model." && exit 1)

ENV TCD_WEIGHTS=/opt/weights/model.pt
ENV PYTHONPATH=/app

COPY services/court_detector/calibrate.py /app/calibrate.py

ENTRYPOINT ["python3", "/app/calibrate.py"]
