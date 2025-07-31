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

# Install gdown and clone TennisCourtDetector (no setup.py available)
RUN pip install --no-cache-dir gdown && rm -rf /root/.cache/pip

RUN git clone --depth 1 https://github.com/boog9/TennisCourtDetector /tmp/TennisCourtDetector

RUN cp -r /tmp/TennisCourtDetector/tennis_court_detector /app/

# Download pretrained weights.
RUN mkdir -p /opt/weights \
    && gdown "$WEIGHTS_URL" -O /opt/weights/model.pt \
    && test -s /opt/weights/model.pt || (echo "Weights download failed" && exit 1) \
    && python3 -c 'import torch; torch.load("/opt/weights/model.pt")' \
    || (echo "Weights file not usable as PyTorch model." && exit 1)

ENV TCD_WEIGHTS=/opt/weights/model.pt
ENV PYTHONPATH=/app

COPY services/court_detector/calibrate.py /app/calibrate.py

ENTRYPOINT ["python3", "/app/calibrate.py"]
