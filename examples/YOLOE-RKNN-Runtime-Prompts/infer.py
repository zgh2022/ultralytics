#!/usr/bin/env python3
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Measure one- or two-input YOLOE RKNN models and summarize raw detection scores."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    """Resize and pad an image to a square while preserving its aspect ratio."""
    height, width = image.shape[:2]
    gain = min(size / height, size / width)
    new_width, new_height = round(width * gain), round(height * gain)
    pad_width, pad_height = size - new_width, size - new_height
    left, top = round(pad_width / 2 - 0.1), round(pad_height / 2 - 0.1)
    right, bottom = round(pad_width / 2 + 0.1), round(pad_height / 2 + 0.1)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, help="optional runtime-prompt .npz file")
    parser.add_argument("--classes", type=int, help="class count for a prompt-free model")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--core", choices=("0", "1", "2", "all"), default="0")
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    return parser.parse_args()


def main() -> None:
    """Load an RKNN model, run it repeatedly, and report output-score statistics."""
    args = parse_args()
    if (args.prompt is None) == (args.classes is None):
        raise ValueError("provide exactly one of --prompt or --classes")

    from rknnlite.api import RKNNLite

    masks = {
        "0": RKNNLite.NPU_CORE_0,
        "1": RKNNLite.NPU_CORE_1,
        "2": RKNNLite.NPU_CORE_2,
        "all": RKNNLite.NPU_CORE_0_1_2,
    }
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(args.image)
    rgb = cv2.cvtColor(letterbox(image, args.imgsz), cv2.COLOR_BGR2RGB)[None]
    inputs = [rgb]
    if args.prompt:
        with np.load(args.prompt, allow_pickle=False) as data:
            embeddings = data["embeddings"].astype(np.float32)
            classes = int(data["count"])
        inputs.append(embeddings)
    else:
        classes = args.classes

    runtime = RKNNLite()
    if runtime.load_rknn(str(args.model)) != 0:
        raise RuntimeError("load_rknn failed")
    if runtime.init_runtime(core_mask=masks[args.core]) != 0:
        raise RuntimeError("init_runtime failed")
    try:
        for _ in range(args.warmup):
            runtime.inference(inputs=inputs)
        timings = []
        outputs = None
        for _ in range(args.runs):
            started = time.perf_counter()
            outputs = runtime.inference(inputs=inputs)
            timings.append((time.perf_counter() - started) * 1000)
    finally:
        runtime.release()

    detections = next(np.asarray(item) for item in outputs if np.asarray(item).ndim == 3)
    if detections.shape[1] < 4 + classes:
        raise ValueError(f"detection output {detections.shape} has fewer than {classes} classes")
    scores = detections[:, 4 : 4 + classes]
    best = scores.max(axis=1)
    _, maximum_class, maximum_anchor = np.unravel_index(np.argmax(scores), scores.shape)
    result = {
        "model": args.model.name,
        "mode": "runtime-prompt" if args.prompt else "prompt-free",
        "classes": classes,
        "core": args.core,
        "runs": args.runs,
        "npu_ms": {
            "min": round(float(np.min(timings)), 2),
            "mean": round(float(np.mean(timings)), 2),
            "max": round(float(np.max(timings)), 2),
        },
        "outputs": [
            {
                "shape": list(np.asarray(item).shape),
                "dtype": str(np.asarray(item).dtype),
                "finite": bool(np.isfinite(item).all()),
                "min": float(np.min(item)),
                "max": float(np.max(item)),
            }
            for item in outputs
        ],
        "scores": {
            "nonzero": int(np.count_nonzero(scores)),
            "maximum": float(np.max(scores)),
            "maximum_class": int(maximum_class),
            "maximum_anchor": int(maximum_anchor),
            "candidates_at_conf": int(np.count_nonzero(best >= args.conf)),
            "confidence": args.conf,
        },
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
