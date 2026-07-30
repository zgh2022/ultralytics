#!/usr/bin/env python3
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Convert the two-input YOLOE ONNX model to an FP16 or INT8 RKNN model."""

from __future__ import annotations

import argparse
from pathlib import Path

from rknn.api import RKNN


def check(ret: int, operation: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{operation} failed with code {ret}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("build/yoloe26n_dynamic_q8_fp16.rknn"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--quantize", action="store_true", help="build a W8A8 INT8 model")
    parser.add_argument(
        "--dataset",
        type=Path,
        help="multi-input calibration dataset used with --quantize",
    )
    parser.add_argument(
        "--quant-algorithm",
        choices=("normal", "kl_divergence", "mmse"),
        default="normal",
    )
    parser.add_argument("--quant-method", choices=("channel", "layer"), default="channel")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.quantize and (args.dataset is None or not args.dataset.is_file()):
        parser.error("--quantize requires an existing --dataset file")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=args.verbose)
    try:
        check(
            rknn.config(
                target_platform="rk3588",
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                quantized_algorithm=args.quant_algorithm,
                quantized_method=args.quant_method,
            ),
            "config",
        )
        check(
            rknn.load_onnx(
                model=str(args.onnx),
                inputs=["images", "prompt_embeddings"],
                input_size_list=[
                    [1, 3, args.imgsz, args.imgsz],
                    [1, args.max_prompts, 512],
                ],
            ),
            "load_onnx",
        )
        check(
            rknn.build(
                do_quantization=args.quantize,
                dataset=str(args.dataset.resolve()) if args.dataset else None,
                rknn_batch_size=args.batch_size if args.batch_size > 1 else None,
            ),
            "build",
        )
        check(rknn.export_rknn(str(args.output)), "export_rknn")
    finally:
        rknn.release()
    print(f"saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
