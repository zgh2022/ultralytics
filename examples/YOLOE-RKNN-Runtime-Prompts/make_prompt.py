#!/usr/bin/env python3
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Create padded runtime text or visual prompt embeddings for the RKNN detector."""

from __future__ import annotations

import argparse
import json
import os
import re
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


def save_prompt(model: YOLOE, output: Path, embeddings: torch.Tensor, names: list[str], max_prompts: int) -> None:
    model.save_prompt_embeddings(output, embeddings, names, max_prompts=max_prompts)
    active = embeddings.detach().float().cpu().numpy()[0]
    print(f"saved: {output.resolve()}")
    print(f"active prompts ({len(names)}/{max_prompts}): {names}")
    print(f"active embedding norms: {np.linalg.norm(active, axis=1).tolist()}")


def get_visual_embeddings(model: YOLOE, image: Path, boxes: np.ndarray, class_ids: list[int], imgsz: int):
    """Extract one embedding for each class present in an image."""
    unique_ids = sorted(set(class_ids))
    embeddings = model.get_visual_prompt_pe(
        image,
        {"bboxes": boxes, "cls": np.asarray(class_ids)},
        imgsz=imgsz,
        device="cpu",
    )
    if embeddings.shape[1] != len(unique_ids):
        raise RuntimeError(f"expected {len(unique_ids)} visual embeddings, got {embeddings.shape[1]}")
    return unique_ids, embeddings


def load_visual_manifest(path: Path) -> tuple[list[str], list[dict]]:
    """Load a multi-reference visual prompt manifest."""
    data = json.loads(path.read_text(encoding="utf-8"))
    names, images = data.get("names"), data.get("images")
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError("manifest 'names' must be a non-empty list of class names")
    if not isinstance(images, list) or not images:
        raise ValueError("manifest 'images' must be a non-empty list")
    return names, images


def make_manifest_embeddings(model: YOLOE, manifest: Path, imgsz: int) -> tuple[torch.Tensor, list[str]]:
    """Average normalized visual embeddings for each class across reference images."""
    names, entries = load_visual_manifest(manifest)
    sums = [None] * len(names)
    counts = [0] * len(names)
    for entry in entries:
        image = Path(entry["image"])
        if not image.is_absolute():
            image = manifest.parent / image
        boxes = np.asarray(entry["boxes"], dtype=np.float32)
        class_ids = [int(value) for value in entry["class_ids"]]
        if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) != len(class_ids):
            raise ValueError(f"{image}: 'boxes' must contain one xyxy box per class ID")
        if any(class_id < 0 or class_id >= len(names) for class_id in class_ids):
            raise ValueError(f"{image}: class IDs must index manifest names 0..{len(names) - 1}")
        unique_ids, embeddings = get_visual_embeddings(model, image, boxes, class_ids, imgsz)
        for row, class_id in enumerate(unique_ids):
            value = embeddings[0, row].detach().float()
            sums[class_id] = value if sums[class_id] is None else sums[class_id] + value
            counts[class_id] += 1
    if any(count == 0 for count in counts):
        missing = [names[i] for i, count in enumerate(counts) if count == 0]
        raise ValueError(f"manifest has no reference boxes for classes: {missing}")
    embeddings = torch.stack([value / count for value, count in zip(sums, counts)])
    return torch.nn.functional.normalize(embeddings, dim=-1).unsqueeze(0), names


def safe_stem(value: str) -> str:
    """Return a portable filename stem for a class name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "class"


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
    source = visual.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--manifest", type=Path, help="JSON manifest containing multiple reference images")
    visual.add_argument("--boxes")
    visual.add_argument(
        "--class-ids",
        help="comma-separated IDs; default gives every box a separate class",
    )
    visual.add_argument("--names", nargs="+")
    visual.add_argument("--imgsz", type=int, default=640)
    visual.add_argument("--per-class-dir", type=Path, help="also save one NPZ per visual class")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.expanduser().absolute()
    model = YOLOE(str(weights))

    if args.mode == "text":
        clip_directory = find_clip_directory(weights, args.clip_dir)
        with working_directory(clip_directory), torch.inference_mode():
            embeddings = model.get_text_pe(args.prompts)
        names = args.prompts
    else:
        if args.manifest:
            embeddings, names = make_manifest_embeddings(model, args.manifest.resolve(), args.imgsz)
        else:
            if not args.boxes:
                raise ValueError("--image requires --boxes")
            boxes = parse_boxes(args.boxes)
            class_ids = [int(item) for item in args.class_ids.split(",")] if args.class_ids else list(range(len(boxes)))
            if len(class_ids) != len(boxes):
                raise ValueError("--class-ids must contain one ID per box")
            unique_ids, embeddings = get_visual_embeddings(model, args.image.resolve(), boxes, class_ids, args.imgsz)
            if unique_ids != list(range(len(unique_ids))):
                raise ValueError("class IDs must be contiguous and start at 0")
            names = args.names or [f"object{i}" for i in unique_ids]
            if len(names) != len(unique_ids):
                raise ValueError("--names must contain one name per unique class ID")

        if args.per_class_dir:
            for index, name in enumerate(names):
                save_prompt(
                    model,
                    args.per_class_dir / f"{index}_{safe_stem(name)}.npz",
                    embeddings[:, index : index + 1],
                    [name],
                    args.max_prompts,
                )

    save_prompt(model, args.output, embeddings, names, args.max_prompts)


if __name__ == "__main__":
    main()
