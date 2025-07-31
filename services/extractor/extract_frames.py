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
"""Frame extraction utility for video files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
from tqdm import tqdm


def extract_frames(video: str, out_dir: str, fps: int) -> int:
    """Extract frames from a video at the desired FPS.

    Args:
        video: Path to the input video file.
        out_dir: Directory where frames will be written as PNGs.
        fps: Target frames per second for extraction.

    Returns:
        Number of frames written to ``out_dir``.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video}")

    os.makedirs(out_dir, exist_ok=True)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = fps
    step = max(int(round(video_fps / fps)), 1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    written = 0
    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                filename = Path(out_dir) / f"{written:06d}.png"
                cv2.imwrite(str(filename), frame)
                written += 1
            frame_idx += 1
            pbar.update(1)

    cap.release()
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract frames from a video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", required=True, help="Directory for output frames")
    parser.add_argument("--fps", type=int, default=30, help="Output frames per second")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)
    count = extract_frames(args.video, args.out, args.fps)
    print(f"Saved {count} frames to {args.out}")


if __name__ == "__main__":
    main()
