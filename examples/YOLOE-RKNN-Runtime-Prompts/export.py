#!/usr/bin/env python3
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Export YOLOE-26 with runtime prompt embeddings as a second input."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLOE


class MatMulContrastiveHead(nn.Module):
    """RKNN-friendly equivalent of Ultralytics BNContrastiveHead."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.norm = source.norm
        self.bias = source.bias
        self.logit_scale = source.logit_scale

    def forward(self, image_features: torch.Tensor, prompts: torch.Tensor) -> torch.Tensor:
        image_features = self.norm(image_features)
        prompts = prompts / torch.clamp(torch.linalg.vector_norm(prompts, dim=-1, keepdim=True), min=1e-6)
        batch, _, height, width = image_features.shape
        scores = torch.matmul(prompts, image_features.flatten(2))
        return scores.reshape(batch, prompts.shape[1], height, width) * self.logit_scale.exp() + self.bias


class DynamicPromptDetector(nn.Module):
    def __init__(self, core: nn.Module, boxes_only: bool = False) -> None:
        super().__init__()
        self.core = core
        self.boxes_only = boxes_only

    def forward(self, images: torch.Tensor, prompt_embeddings: torch.Tensor):
        prediction = self.core.predict(images, vpe=prompt_embeddings)
        if self.boxes_only:
            return prediction[0]
        return prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("build/yoloe26n_dynamic_q8.onnx"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--boxes-only",
        action="store_true",
        help="drop the prototype output used for masks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    yolo = YOLOE(str(args.weights))
    core = yolo.model.eval().float()
    head = core.model[-1]

    # TopK in the end-to-end branch is not supported on RKNN. Keep decoded boxes
    # and class scores, then do NMS on the CPU.
    head.end2end = False
    core.fuse(verbose=False)
    head.cv4 = nn.ModuleList(MatMulContrastiveHead(module) for module in head.cv4)
    head.export = True
    head.format = "onnx"
    head.dynamic = False
    wrapper = DynamicPromptDetector(core, boxes_only=args.boxes_only).eval()

    generator = torch.Generator().manual_seed(0)
    images = torch.rand(1, 3, args.imgsz, args.imgsz, generator=generator)
    prompts = torch.randn(1, args.max_prompts, head.embed, generator=generator)
    prompts = torch.nn.functional.normalize(prompts, dim=-1)

    # ONNX tracing reuses the cached anchors produced by this dry run. Use
    # no_grad rather than inference_mode so those tensors remain traceable.
    with torch.no_grad():
        prediction = wrapper(images, prompts)
        tensors = (prediction,) if args.boxes_only else tuple(prediction)
        reference = tuple(t.cpu().numpy() for t in tensors)

    torch.onnx.export(
        wrapper,
        (images, prompts),
        str(args.output),
        input_names=["images", "prompt_embeddings"],
        output_names=["detections"] if args.boxes_only else ["detections", "prototypes"],
        opset_version=args.opset,
        do_constant_folding=True,
    )

    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"images": images.numpy(), "prompt_embeddings": prompts.numpy()})
    errors = [float(np.max(np.abs(a - b))) for a, b in zip(actual, reference)]
    inputs = {item.name: [d.dim_value for d in item.type.tensor_type.shape.dim] for item in graph.graph.input}
    outputs = {item.name: [d.dim_value for d in item.type.tensor_type.shape.dim] for item in graph.graph.output}
    print(f"saved: {args.output.resolve()}")
    print(f"inputs: {inputs}")
    print(f"outputs: {outputs}")
    print(f"PyTorch/ONNX max abs errors: {errors}")


if __name__ == "__main__":
    main()
