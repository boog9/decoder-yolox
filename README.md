# decoder-yolox

This repository contains Docker build files for the video-processing pipeline described in the project documentation. The images target Ubuntu 22.04 with NVIDIA GPUs and Python 3.10.

## Docker Images

### `decoder/base-cuda`
- **Purpose:** Provides CUDA 12.2 runtime along with Python 3.10 and basic scientific libraries for GPU-accelerated video processing.
- **GPU required:** Yes. Enable with `--gpus all` when running.
- **Packages:** Python 3.10 with PyTorch 2.3.0 + CUDA 12.2, torchvision 0.18.0, OpenCV, NumPy, SciPy, tqdm and typer.

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

### `decoder/extractor`
- **Purpose:** Extracts PNG frames from a video file.
- **GPU required:** No.

#### Build example
```bash
make extractor
# or
docker build -t decoder/extractor -f services/extractor/Dockerfile .
```

#### Run example
```bash
docker run --rm -v $(pwd)/data:/data decoder/extractor \
    --video /data/match.mp4 --out /data/frames --fps 30
```

#### Dependencies / volumes
- Mount a working directory with `-v $(pwd)/data:/data` to supply video
  files and collect frames.

#### Parameters
- `--video` (required) – path to the source video.
- `--out` (required) – directory where frames will be saved.
- `--fps` (default: `30`) – output frame rate.
