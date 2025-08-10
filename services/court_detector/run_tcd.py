#!/usr/bin/env python3
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
"""Thin CLI wrapper around upstream TennisCourtDetector."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import tempfile
import sys
from typing import List, Optional

REPO_DIR = "/app/TennisCourtDetector"
WEIGHTS = os.path.join(REPO_DIR, "model.pt")

def run_upstream_infer(frame_path: str, device: str, tmp_png_out: str) -> None:
    """Invoke upstream inference script with fallback strategies.

    Args:
        frame_path: Path to input image.
        device: Target device ("cuda" or "cpu").
        tmp_png_out: Path to temporary output PNG to verify execution.

    Raises:
        RuntimeError: If upstream inference fails.
    """
    infer_py = os.path.join(REPO_DIR, "infer_in_image.py")

    # Upstream expects --model_path / --input_path / --output_path; no device flag.
    cmds = [
        ["python", infer_py,
         "--model_path", WEIGHTS,
         "--input_path", frame_path,
         "--output_path", tmp_png_out,
         "--use_refine_kps", "--use_homography"],
        ["python", infer_py,
         "--model_path", WEIGHTS,
         "--input_path", frame_path,
         "--output_path", tmp_png_out],
        ["python", infer_py,
         "--input_path", frame_path,
         "--output_path", tmp_png_out],
    ]

    last_exc: Optional[subprocess.CalledProcessError] = None
    for cmd in cmds:
        try:
            subprocess.run(cmd, cwd=REPO_DIR, check=True)
            return
        except subprocess.CalledProcessError as exc:
            last_exc = exc

    # Fallback: import module directly and call known entrypoints.
    import importlib.util
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    spec = importlib.util.spec_from_file_location("tcd_infer", infer_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Upstream inference failed and import fallback unavailable: {last_exc}"
        ) from last_exc
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tcd_infer"] = mod
    spec.loader.exec_module(mod)  # type: ignore[assignment]

    if hasattr(mod, "infer"):
        try:
            mod.infer(
                model_path=WEIGHTS,
                input_path=frame_path,
                output_path=tmp_png_out,
                use_refine_kps=True,
                use_homography=True,
            )
            return
        except TypeError:
            mod.infer(model_path=WEIGHTS, input_path=frame_path, output_path=tmp_png_out)
            return
    raise RuntimeError("Upstream module lacks known entrypoint 'infer'")

def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Thin CLI wrapper around upstream TennisCourtDetector."
    )
    parser.add_argument("--frame", required=True, help="Path to input frame (PNG/JPG).")
    parser.add_argument("--out", required=True, help="Path to JSON output file.")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference.",
    )
    args = parser.parse_args()

    frame = os.path.abspath(args.frame)
    out_json = os.path.abspath(args.out)
    pathlib.Path(os.path.dirname(out_json)).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        vis_path = os.path.join(td, "vis.png")
        run_upstream_infer(frame, args.device, vis_path)

    # Not parsing upstream outputs here; return empty structures for now.
    keypoints_list: List[List[float]] = []
    homography_mat: Optional[List[List[float]]] = None

    result = {
        "source_frame": args.frame,
        "weights_path": WEIGHTS,
        "device": args.device,
        "notes": "Inference OK. keypoints/homography populated when upstream modules are present.",
        "keypoints": keypoints_list,
        "homography": homography_mat,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_json}")

if __name__ == "__main__":
    main()
