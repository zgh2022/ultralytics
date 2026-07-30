# YOLOE Runtime Prompts on RKNN

This example exports promptable YOLOE models to RKNN for RK3588/RK3588S. The exported graph has two fixed-shape
inputs: the image and precomputed text or visual prompt embeddings. Prompt encoders remain on the host, while the
Rockchip NPU runs the detector.

```text
images:            uint8   [1, 3, 640, 640]
prompt_embeddings: float32 [1, max_prompts, 512]
```

## Requirements

Export and prompt generation require Ultralytics with PyTorch. Text prompts also require `mobileclip2_b.ts`. RKNN
conversion runs on x86 Linux with `rknn-toolkit2>=2.3.2`; board inference requires `rknn-toolkit-lite2`.

FP16 is the recommended precision. An RK3588S W8A8 experiment reduced YOLOE-26L latency from about 394 ms to 159 ms,
but detections on a six-image small-object sample fell from 2/6 images to 0/6. Runtime-prompt INT8 export is therefore
disabled until a representative calibration and accuracy benchmark establish that it is usable.

## Export

YOLOE-26N/S/M/L/X text/visual checkpoints use the same API. `prompt_boxes_only=True` omits mask coefficients and
prototypes, which avoids expensive CPU mask postprocessing when only detection boxes are needed.

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-26l-seg.pt")
model.export(
    format="rknn",
    name="rk3588",
    quantize=16,
    runtime_prompts=True,
    prompt_boxes_only=True,
    max_prompts=8,
)
```

`max_prompts` is the static class capacity compiled into the graph, not an annotation limit. Choose it for the largest
text or visual vocabulary that this RKNN model must accept. The exporter produces `yoloe-26l-seg_rknn_model/` with the
RKNN file and its metadata.

## Generate Prompts

Prompt generation is required after export. RKNN does not encode text or inspect reference-image boxes itself; it
accepts the resulting small NPZ tensor. An NPZ is tied to the source checkpoint scale and cannot be shared between
N/S/M/L/X models.

Generate text prompts with the framework API:

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-26l-seg.pt")
names = ["plastic bottle", "drink can"]
embeddings = model.get_text_pe(names)
model.save_prompt_embeddings("waste_text.npz", embeddings, names, max_prompts=8)
```

Generate a visual prompt from one reference image and pixel-space `xyxy` boxes:

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-26l-seg.pt")
visual = {"bboxes": [[120, 80, 220, 210]], "cls": [0]}
embeddings = model.get_visual_prompt_pe("reference.jpg", visual)
model.save_prompt_embeddings("bottle_visual.npz", embeddings, ["bottle"], max_prompts=8)
```

The helper supports multiple reference images. Each image is processed independently; embeddings for the same class
are averaged and L2-normalized. This avoids introducing artificial object positions by compositing crops into a scene.

```json
{
    "names": ["bottle", "can"],
    "images": [
        {"image": "refs/bottle_1.jpg", "boxes": [[120, 80, 220, 210]], "class_ids": [0]},
        {"image": "refs/bottle_can.jpg", "boxes": [[25, 40, 90, 180], [240, 95, 330, 210]], "class_ids": [0, 1]}
    ]
}
```

```bash
python examples/YOLOE-RKNN-Runtime-Prompts/make_prompt.py visual \
    --weights yoloe-26l-seg.pt \
    --manifest references.json \
    --output waste_visual_all.npz \
    --per-class-dir prompts/by_class \
    --max-prompts 8
```

The combined multi-class NPZ is supported, but visual classes can interfere with one another. For best accuracy,
especially on small targets, use the files in `prompts/by_class/`: run one class at a time and merge the standard
post-NMS boxes. Multiple boxes and reference images may still be used to build each single-class embedding.

```python
from pathlib import Path

from ultralytics import YOLO

model = YOLO("yoloe-26l-seg_rknn_model")
per_class = [
    model.predict("test.jpg", prompt_embeddings=prompt, conf=0.25)[0]
    for prompt in sorted(Path("prompts/by_class").glob("*.npz"))
]
for result in per_class:
    print(result.names, result.boxes.xyxy, result.boxes.conf)
```

## Predict on RK3588

Copy the RKNN model directory, prompt NPZ, and source media to the board. The normal Ultralytics API returns standard
`Results`; `speed["inference"]` reports model inference time in milliseconds.

```python
from ultralytics import YOLO

model = YOLO("yoloe-26l-seg_rknn_model")
results = model.predict(
    "test.jpg",
    prompt_embeddings="waste_text.npz",
    imgsz=640,
    conf=0.25,
    iou=0.7,
)
for result in results:
    print(result.names, result.boxes.xyxy, result.boxes.conf, result.speed["inference"])
```

For raw NPU-only timing and output diagnostics, use `infer.py`. Its `--core` option selects one RK3588 NPU core.
Using all three cores for one model does not reduce latency materially. For concurrent requests, create three
independent RKNNLite contexts pinned to Core 0, 1, and 2; measured YOLOE-26L throughput increased from 2.54 FPS for one
context to 6.68 FPS for three contexts.

## Measured RK3588S Results

These FP16 numbers use RKNN Toolkit 2.3.2, RKNNLite 2.3.0, NPU driver 0.8.8, 640 x 640 input, and `person,bus` text
embeddings. Each result is the mean of five Core 0 runs after one warmup.

| Scale     | Mean NPU time | Maximum score | RKNN size |
| --------- | ------------: | ------------: | --------: |
| YOLOE-26N |      75.96 ms |        0.8950 |   8.2 MiB |
| YOLOE-26S |     143.32 ms |        0.9067 |  23.4 MiB |
| YOLOE-26M |     348.29 ms |        0.9233 |  49.9 MiB |
| YOLOE-26L |     393.89 ms |        0.9199 |  59.5 MiB |
| YOLOE-26X |     887.71 ms |        0.9131 | 127.9 MiB |

Prompt-free checkpoints are intentionally rejected. Their fixed 4585-class classifier produces a
`[1,4621,8400]` output at 640 x 640; YOLOE-26N measured 3043.61 ms on the same board. The large output and CPU transfer
cost make prompt-free RKNN deployment unsuitable for this target compared with a small runtime vocabulary.

## Limitations

- Input size, batch size, and prompt capacity are static.
- NMS runs on the CPU outside the RKNN graph.
- Visual and text prompt generation requires the PyTorch checkpoint on a host.
- Runtime-prompt NPZ files are source-model-specific.
- The tested board runtime and NPU driver are older than the compiler and should be updated for production validation.
