# decoder-yolox

This repository contains Docker build files for the video-processing pipeline described in the project documentation. The images target Ubuntu 22.04 with NVIDIA GPUs and Python 3.10+. The court-detector service is built on Python 3.11.

## Docker Images

### `decoder/base-cuda`
- **Purpose:** Provides CUDA 12.1 runtime along with Python 3.10 and basic scientific libraries for GPU-accelerated video processing.
- **GPU required:** Yes. Enable with `--gpus all` when running.
- **Packages:** Python 3.10 with PyTorch 2.2.2 + CUDA 12.1, torchvision 0.17.2, OpenCV, NumPy, SciPy, tqdm and typer.

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

## Court Detector (TennisCourtDetector)

This service packages the upstream [TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector) with original pretrained weights and a thin CLI wrapper.  
Base image uses modern **PyTorch 2.4.1 + CUDA 12.1 + cuDNN 9** (no legacy CUDA images).  
Upstream README describes 14 keypoints (+1 center) and provides Google Drive weights.  
PyTorch Docker tags for 2.4.1 CUDA 12.1 are available.

- **GPU required:** Yes. Enable with `--gpus all`. The CLI can run on CPU via `--device cpu`.

### Build
```bash
make court-detector
docker build --build-arg TCD_REF=<branch_or_commit> -t decoder/court-detector services/court_detector
```
To build from a different upstream reference, replace `<branch_or_commit>` with the desired branch, tag, or commit hash.

### Run (single frame)

```bash
docker run --rm --gpus all \
  -v $PWD/data:/data \
  decoder/court-detector \
  --frame /data/frames_min/000000.png \
  --out   /data/court_meta.json \
  --device cuda
```

#### Host paths (relative)
If you prefer to use host-relative paths (e.g., `data/frames_min/000000.png`), mirror your working directory inside the container:
```bash
docker run --rm --gpus all \
  -v "$PWD:$PWD" -w "$PWD" \
  decoder/court-detector \
  --frame data/frames_min/000000.png \
  --out   data/court_meta.json \
  --device cuda
```

#### Dependencies / volumes
- Mount a working directory with `-v $PWD/data:/data` for input and output files.

#### Parameters
- `--frame` (required) – path to the input frame image.
- `--out` (required) – destination JSON file containing court metadata.
- `--device` (default: `cuda`) – inference device, `cuda` or `cpu`.

* `court_meta.json` contains keypoints and homography when upstream modules are available.
* If modules are missing, the file still records that inference completed and which device and weights were used.

#### Weights
The build step downloads pretrained weights from Google Drive. If `gdown` hits rate limits (HTTP 403/429), download the file manually and mount it at runtime:

```bash
# Download once on the host
gdown --fuzzy "https://drive.google.com/uc?id=1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG" \
  -O weights/model.pt

# Run container with mounted weights
docker run --rm --gpus all \
  -v $PWD/data:/data \
  -v $PWD/weights/model.pt:/app/TennisCourtDetector/model.pt:ro \
  decoder/court-detector \
  --frame /data/frames_min/000000.png \
  --out   /data/court_meta.json \
  --device cuda
```


### Smoke-test
```bash
# 1) Build image
make court-detector

# 2) Run inference on GPU
docker run --rm --gpus all -v $PWD/data:/data decoder/court-detector \
  --frame /data/frames_min/000000.png \
  --out   /data/court_meta.json \
  --device cuda

# 3) Inspect JSON output
jq . /data/court_meta.json
```
If the JSON contains approximately 14–15 keypoints and a `homography` matrix, the upstream detector is fully utilized.
