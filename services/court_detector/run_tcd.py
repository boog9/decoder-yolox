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
"""Thin CLI wrapper around upstream TennisCourtDetector that returns keypoints.
Runs the upstream `tracknet` model directly, computes heatmap argmax keypoints,
optionally refines them and estimates homography if upstream helpers exist.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import List, Optional

REPO_DIR = "/app/TennisCourtDetector"
WEIGHTS = os.path.join(REPO_DIR, "model.pt")


def _load_module(path: str, name: str):
    """Load a python module from file path safely; return None on failure."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def run_upstream_infer(frame_path: str, device: str, tmp_png_out: str) -> None:
    """Optional: call upstream script for its visualization (best-effort only).

    Args:
        frame_path: Path to input image.
        device: Target device ("cuda" or "cpu").
        tmp_png_out: Path to temporary output PNG to verify execution.
    """

    infer_py = os.path.join(REPO_DIR, "infer_in_image.py")

    def _run(cmd: List[str]) -> bool:
        debug = os.getenv("TCD_DEBUG")
        res = subprocess.run(
            cmd,
            cwd=REPO_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if res.returncode == 0:
            return True
        if debug:
            print(f"[upstream stderr] {' '.join(cmd)}\n{res.stderr}")
        return False

    # Always provide model_path; enable optional flags progressively.
    cmds: List[List[str]] = [
        [
            "python",
            infer_py,
            "--model_path",
            WEIGHTS,
            "--input_path",
            frame_path,
            "--output_path",
            tmp_png_out,
        ],
        [
            "python",
            infer_py,
            "--model_path",
            WEIGHTS,
            "--input_path",
            frame_path,
            "--output_path",
            tmp_png_out,
            "--use_refine_kps",
        ],
        [
            "python",
            infer_py,
            "--model_path",
            WEIGHTS,
            "--input_path",
            frame_path,
            "--output_path",
            tmp_png_out,
            "--use_refine_kps",
            "--use_homography",
        ],
    ]

    for cmd in cmds:
        if _run(cmd):
            return

    # Fallback: import module directly and call known entrypoints.
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    mod = _load_module(infer_py, "tcd_infer")
    if mod and hasattr(mod, "infer"):
        try:
            mod.infer(
                model_path=WEIGHTS,
                input_path=frame_path,
                output_path=tmp_png_out,
                use_refine_kps=False,
                use_homography=False,
            )
        except TypeError:
            mod.infer(model_path=WEIGHTS, input_path=frame_path, output_path=tmp_png_out)


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(
        description="Thin CLI wrapper around upstream TennisCourtDetector.",
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
        try:
            run_upstream_infer(frame, args.device, vis_path)
        except Exception:
            if os.getenv("TCD_DEBUG"):
                import traceback

                traceback.print_exc()

    # === Local inference → stable keypoints (and optional homography) ===
    keypoints_list: List[List[float]] = []
    homography_mat: Optional[List[List[float]]] = None
    result_note_fallback = False
    try:
        import cv2  # type: ignore
        import numpy as np
        import torch

        if REPO_DIR not in sys.path:
            sys.path.insert(0, REPO_DIR)
        from tracknet import BallTrackerNet, ConvBlock  # type: ignore

        dev = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
        img = cv2.imread(frame, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {frame}")

        # BGR -> RGB, CHW, contiguous to avoid negative strides
        rgb_chw = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
        x = torch.from_numpy(rgb_chw).float().unsqueeze(0) / 255.0
        x = x.to(dev)

        # Build model on CPU first, maybe replace conv18, then move to device
        model = BallTrackerNet().eval()
        try:
            oc = getattr(getattr(model, "conv18").block[0], "out_channels", None)
            if oc != 15:
                model.conv18 = ConvBlock(64, 15)  # type: ignore[attr-defined]
        except Exception:
            pass
        model = model.to(dev)
        state = torch.load(WEIGHTS, map_location=dev)
        model.load_state_dict(state, strict=False)
        try:
            with torch.no_grad():
                heat = model(x).squeeze(0).detach().cpu().numpy()
        except Exception:
            if os.getenv("TCD_DEBUG"):
                import traceback

                traceback.print_exc()
            if dev == "cuda":
                dev = "cpu"
                x = x.to(dev)
                model = model.to(dev)
                state = torch.load(WEIGHTS, map_location="cpu")
                model.load_state_dict(state, strict=False)
                with torch.no_grad():
                    heat = model(x).squeeze(0).detach().cpu().numpy()
                result_note_fallback = True
            else:
                raise

        # Prefer upstream postprocess.refine_kps if available
        kps = None
        post_path = os.path.join(REPO_DIR, "postprocess.py")
        mod_pp = _load_module(post_path, "tcd_post") if os.path.exists(post_path) else None
        if mod_pp and hasattr(mod_pp, "refine_kps"):
            try:
                kps = mod_pp.refine_kps(heat)
            except Exception:
                kps = None

        if kps is None:
            c, h, w = heat.shape
            pts = []
            for ch in range(c):
                idx = int(heat[ch].argmax())
                y, x_ = divmod(idx, w)
                pts.append([float(x_), float(y)])
            kps = np.asarray(pts, dtype=np.float32)

        keypoints_list = [[float(x), float(y)] for x, y in np.asarray(kps).tolist()]

        # Homography best-effort (only if safe)
        try:
            hom_path = os.path.join(REPO_DIR, "homography.py")
            mod_h = _load_module(hom_path, "tcd_h") if os.path.exists(hom_path) else None
            if mod_h and len(keypoints_list) >= 4:
                pts_np = np.asarray(keypoints_list, dtype=np.float32)
                H = None
                if hasattr(mod_h, "compute_homography"):
                    H = mod_h.compute_homography(pts_np)
                elif hasattr(mod_h, "get_trans_matrix"):
                    try:
                        H = mod_h.get_trans_matrix(pts_np)
                    except Exception:
                        H = None
                H = np.asarray(H) if H is not None else None
                if H is not None and H.ndim == 2 and H.shape == (3, 3):
                    homography_mat = [[float(a) for a in row] for row in H.tolist()]
        except Exception:
            if os.getenv("TCD_DEBUG"):
                import traceback

                traceback.print_exc()
    except Exception:
        if os.getenv("TCD_DEBUG"):
            import traceback

            traceback.print_exc()

    note = (
        "Inference OK. keypoints/homography populated when upstream modules are present."
    )
    if result_note_fallback:
        note += " (fallback to cpu)"
    result = {
        "source_frame": args.frame,
        "weights_path": WEIGHTS,
        "device": args.device,
        "notes": note,
        "keypoints": keypoints_list,
        "homography": homography_mat,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
