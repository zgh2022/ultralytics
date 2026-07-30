#!/usr/bin/env python3
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Create padded runtime text or visual prompt embeddings for the RKNN detector."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLOE


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def find_clip_directory(weights: Path, requested: Path | None) -> Path:
    candidates = [requested, weights.parent, Path.cwd(), Path(sys.prefix).parent]
    for candidate in candidates:
        if candidate and (candidate / "mobileclip2_b.ts").is_file():
            return candidate.resolve()
    raise FileNotFoundError("mobileclip2_b.ts not found; pass --clip-dir or place it beside the YOLOE weights")


def parse_boxes(value: str) -> np.ndarray:
    boxes = [[float(number) for number in item.split(",")] for item in value.split(";") if item]
    if not boxes or any(len(box) != 4 for box in boxes):
        raise ValueError("--boxes must look like: x1,y1,x2,y2;x1,y1,x2,y2")
    return np.asarray(boxes, dtype=np.float32)


def save_prompt(output: Path, embeddings: torch.Tensor, names: list[str], max_prompts: int) -> None:
    embeddings = embeddings.detach().float().cpu().numpy()
    if embeddings.shape[0] != 1 or embeddings.shape[2] != 512:
        raise ValueError(f"unexpected embedding shape: {embeddings.shape}")
    count = embeddings.shape[1]
    if count > max_prompts:
        raise ValueError(f"got {count} prompts, but detector supports at most {max_prompts}")
    padded = np.zeros((1, max_prompts, 512), dtype=np.float32)
    padded[:, :count] = embeddings
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, embeddings=padded, names=np.asarray(names), count=np.int64(count))
    print(f"saved: {output.resolve()}")
    print(f"active prompts ({count}/{max_prompts}): {names}")
    print(f"active embedding norms: {np.linalg.norm(padded[0, :count], axis=1).tolist()}")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    text = subparsers.add_parser("text")
    add_common(text)
    text.add_argument("--prompts", nargs="+", required=True)
    text.add_argument("--clip-dir", type=Path)

    visual = subparsers.add_parser("visual")
    add_common(visual)
    visual.add_argument("--image", type=Path, required=True)
    visual.add_argument("--boxes", required=True)
    visual.add_argument(
        "--class-ids",
        help="comma-separated IDs; default gives every box a separate class",
    )
    visual.add_argument("--names", nargs="+")
    visual.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    model = YOLOE(str(weights))

    if args.mode == "text":
        clip_directory = find_clip_directory(weights, args.clip_dir)
        with working_directory(clip_directory), torch.inference_mode():
            embeddings = model.get_text_pe(args.prompts)
        names = args.prompts
    else:
        boxes = parse_boxes(args.boxes)
        class_ids = [int(item) for item in args.class_ids.split(",")] if args.class_ids else list(range(len(boxes)))
        if len(class_ids) != len(boxes):
            raise ValueError("--class-ids must contain one ID per box")
        unique_ids = sorted(set(class_ids))
        if unique_ids != list(range(len(unique_ids))):
            raise ValueError("class IDs must be contiguous and start at 0")
        names = args.names or [f"object{i}" for i in unique_ids]
        if len(names) != len(unique_ids):
            raise ValueError("--names must contain one name per unique class ID")

        # Stop after get_vpe instead of running a redundant detection on the
        # reference image. This also avoids a tuple postprocessing bug in 8.4.69.
        from ultralytics.models.yolo.yoloe.predict import YOLOEVPSegPredictor

        predictor = YOLOEVPSegPredictor(
            overrides={
                "task": "segment",
                "mode": "predict",
                "save": False,
                "verbose": False,
                "batch": 1,
                "device": "cpu",
                "half": False,
                "imgsz": args.imgsz,
            },
            _callbacks=model.callbacks,
        )
        predictor.set_prompts({"bboxes": boxes, "cls": np.asarray(class_ids)})
        predictor.setup_model(model=model.model, verbose=False)
        embeddings = predictor.get_vpe(str(args.image))

    save_prompt(args.output, embeddings, names, args.max_prompts)


if __name__ == "__main__":
    main()
