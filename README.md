# decoder-yolox

This repository contains Docker build files for the video-processing pipeline described in the project documentation. The images target Ubuntu 22.04 with NVIDIA GPUs and Python 3.10.

## Docker Images

### `decoder/base-cuda`
- **Purpose:** Provides CUDA 12.2 runtime along with Python 3.10 and basic scientific libraries for GPU-accelerated video processing.
- **GPU required:** Yes. Enable with `--gpus all` when running.

#### Build example
```bash
make base-cuda
# or
docker build -t decoder/base-cuda -f docker/base/Dockerfile .
```

#### Run example
```bash
docker run --gpus all -it decoder/base-cuda bash
```

#### Dependencies / volumes
- Optional: mount a working directory with `-v $(pwd):/data` to access local files inside the container.

#### Parameters
This base image exposes no additional parameters. It is intended as a foundation for other services in the pipeline.
