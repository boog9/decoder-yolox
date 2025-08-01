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

RUN pip install --no-cache-dir 'numpy<2' gdown matplotlib opencv-python torch torchvision \
    && rm -rf /root/.cache/pip

COPY services/court_detector /app/tennis_court_detector

RUN sed -i 's/^from utils /from .utils /' /app/tennis_court_detector/postprocess.py
RUN sed -i 's/^from court_reference /from .court_reference /' /app/tennis_court_detector/homography.py

RUN mkdir -p /opt/weights \
    && gdown "https://drive.google.com/uc?id=1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG" -O /opt/weights/model.pt \
    && test -s /opt/weights/model.pt

COPY services/court_detector/calibrate.py /app/calibrate.py

ENTRYPOINT ["python", "/app/calibrate.py"]
