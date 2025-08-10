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

    def _run(cmd: List[str]) -> bool:
        debug = os.getenv("TCD_DEBUG")
        res = subprocess.run(
            cmd, cwd=REPO_DIR, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if res.returncode == 0:
            return True
        if debug:
            print(f"[upstream stderr] {' '.join(cmd)}\n{res.stderr}")
        return False

    # Try minimal first (avoids homography crash), then add model_path, then homography flags.
    cmds: List[List[str]] = [
        ["python", infer_py, "--input_path", frame_path, "--output_path", tmp_png_out],
        ["python", infer_py, "--model_path", WEIGHTS, "--input_path", frame_path, "--output_path", tmp_png_out],
        ["python", infer_py, "--model_path", WEIGHTS, "--input_path", frame_path, "--output_path", tmp_png_out, "--use_refine_kps", "--use_homography"],
    ]

    last_exc: Optional[subprocess.CalledProcessError] = None
    for cmd in cmds:
        if _run(cmd):
            return

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
            mod.infer(model_path=WEIGHTS, input_path=frame_path, output_path=tmp_png_out,
                      use_refine_kps=False, use_homography=False)
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

    # Try to populate keypoints (and optional homography) directly via upstream modules.
    keypoints_list: List[List[float]] = []
    homography_mat: Optional[List[List[float]]] = None

    try:
        import importlib.util
        import numpy as np
        import cv2
        import torch
        import traceback

        # Ensure upstream is importable.
        if REPO_DIR not in sys.path:
            sys.path.insert(0, REPO_DIR)
        infer_path = os.path.join(REPO_DIR, "infer_in_image.py")
        spec_i = importlib.util.spec_from_file_location("tcd_infer", infer_path)
        if spec_i and spec_i.loader:
            mod_i = importlib.util.module_from_spec(spec_i)
            spec_i.loader.exec_module(mod_i)  # type: ignore[attr-defined]

            # Load image.
            img = cv2.imread(frame, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {frame}")

            dev = (
                "cuda"
                if (args.device == "cuda" and torch.cuda.is_available())
                else "cpu"
            )

            # Build detector robustly across API variants.
            Detector = getattr(mod_i, "CourtDetector", None)
            if Detector is None:
                raise RuntimeError("Upstream module has no CourtDetector")
            det = None
            for ctor in (
                lambda: Detector(model_path=WEIGHTS, device=dev),
                lambda: Detector(WEIGHTS, dev),
                lambda: Detector(WEIGHTS),
            ):
                try:
                    det = ctor()
                    break
                except TypeError:
                    continue
            if det is None:
                det = Detector(WEIGHTS)

            out = det.detect(img)

            # Prefer upstream postprocess if available.
            kps = None
            try:
                spec_pp = importlib.util.spec_from_file_location(
                    "tcd_post", os.path.join(REPO_DIR, "postprocess.py")
                )
                if spec_pp and spec_pp.loader:
                    mod_pp = importlib.util.module_from_spec(spec_pp)
                    spec_pp.loader.exec_module(mod_pp)  # type: ignore[attr-defined]
                    if hasattr(mod_pp, "refine_kps"):
                        heatmaps = out if isinstance(out, np.ndarray) else None
                        if isinstance(heatmaps, np.ndarray) and heatmaps.ndim == 3:
                            kps = mod_pp.refine_kps(heatmaps)
            except Exception:
                if os.getenv("TCD_DEBUG"):
                    traceback.print_exc()

            # If no upstream refinement, derive kps by argmax per channel.
            if kps is None:
                if isinstance(out, np.ndarray):
                    if out.ndim == 3:
                        c, h, w = out.shape
                        kps = []
                        for ch in range(c):
                            idx = int(np.argmax(out[ch]))
                            y, x = divmod(idx, w)
                            kps.append([float(x), float(y)])
                        kps = np.asarray(kps, dtype=np.float32)
                    elif out.ndim == 2 and out.shape[1] == 2:
                        kps = out.astype(np.float32)
                elif (
                    isinstance(out, (list, tuple))
                    and len(out) > 0
                    and len(out[0]) == 2
                ):
                    kps = np.asarray(out, dtype=np.float32)

            if kps is not None:
                keypoints_list = [
                    [float(x), float(y)] for x, y in np.asarray(kps).tolist()
                ]

            # Homography (best-effort, optional).
            try:
                spec_h = importlib.util.spec_from_file_location(
                    "tcd_h", os.path.join(REPO_DIR, "homography.py")
                )
                if spec_h and spec_h.loader and len(keypoints_list) >= 4:
                    mod_h = importlib.util.module_from_spec(spec_h)
                    spec_h.loader.exec_module(mod_h)  # type: ignore[attr-defined]
                    if hasattr(mod_h, "compute_homography"):
                        h_mat = mod_h.compute_homography(
                            np.asarray(keypoints_list, dtype=np.float32)
                        )
                        if h_mat is not None:
                            homography_mat = [
                                [float(a) for a in row]
                                for row in np.asarray(h_mat).tolist()
                            ]
            except Exception:
                if os.getenv("TCD_DEBUG"):
                    traceback.print_exc()
    except Exception:
        if os.getenv("TCD_DEBUG"):
            import traceback

            traceback.print_exc()

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
