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
"""Calibrate the tennis court location in a frame."""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import time
from pathlib import Path

import cv2
import torch
from tennis_court_detector.infer_in_image import CourtDetector

# Default location of pretrained weights inside the Docker image.
DEFAULT_WEIGHTS = "/opt/weights/model.pt"


def calibrate(frame: str, out: str, weights: str, device: str = "auto") -> dict:
    """Run court detection on a single frame and save metadata.

    Args:
        frame: Path to the input frame image.
        out: Path where the JSON metadata will be written.
        weights: Path to the detector weights.
        device: "cpu", "cuda" or "auto" to select runtime device.

    Returns:
        Metadata dictionary including detection results and extra fields.
    """
    image = cv2.imread(frame)
    if image is None:
        raise FileNotFoundError(f"Unable to read frame: {frame}")

    if not weights or not Path(weights).is_file():
        raise FileNotFoundError(f"Weights file not found: {weights}")
    if Path(weights).stat().st_size == 0:
        raise FileNotFoundError(f"Weights file is empty: {weights}")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        detector = CourtDetector(weights_path=weights, device=device)
    except RuntimeError as e:  # handle cuda-related errors
        if "cuda" in str(e).lower():
            raise RuntimeError("CUDA requested but not available") from e
        raise

    heatmaps = detector.detect(image)
    meta = {
        "frame_id": Path(frame).name,
        "timestamp_ms": int(time.time() * 1000),
        "model_sha": hashlib.sha256(Path(weights).read_bytes()).hexdigest()[:8]
        if Path(weights).stat().st_size > 0 else "unknown",
        "device": device,
        "heatmaps": heatmaps.tolist(),
    }

    try:
        from tennis_court_detector.postprocess import refine_kps, compute_homography
    except Exception:  # pragma: no cover - postprocess optional
        refine_kps = None
        compute_homography = None

    if refine_kps and compute_homography:
        try:
            kps = refine_kps(heatmaps)
            H = compute_homography(kps)
            if H is not None:
                meta["homography"] = [[float(x) for x in row] for row in H.tolist()]
        except Exception:  # pragma: no cover - homography is optional
            pass

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Calibrate tennis court position")
    parser.add_argument("--frame", required=True, help="Path to the input frame")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help=f"Path to model weights (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device: cpu, cuda or auto",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)
    meta = calibrate(args.frame, args.out, args.weights, args.device)
    # DEBUG-print disabled — use logger if needed
    # print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
