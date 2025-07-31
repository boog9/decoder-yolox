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

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        python3.10 python3.10-venv python3-pip git ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip wheel setuptools
# Torch + базові наукові пакети
RUN pip install torch==2.3.*+cu122 torchvision==0.18.*+cu122 --index-url https://download.pytorch.org/whl/cu122
RUN pip install opencv-python-headless==4.* numpy scipy tqdm typer==0.9
CMD ["bash"]
