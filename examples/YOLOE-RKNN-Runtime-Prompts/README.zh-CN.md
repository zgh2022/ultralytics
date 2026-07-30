# 在 RKNN 上使用 YOLOE 运行时提示

[English](README.md) | [简体中文](README.zh-CN.md)

本示例将可提示的 YOLOE 模型导出到 RK3588/RK3588S 使用的 RKNN。导出图包含两个固定形状输入：待检测图像和
预先计算的文字或视觉提示 embedding。提示编码器运行在主机或板端 CPU，Rockchip NPU 只运行检测器。

```text
images:            uint8   [1, 3, 640, 640]
prompt_embeddings: float32 [1, max_prompts, 512]
```

## 环境要求

模型导出和提示生成需要带 PyTorch 的 Ultralytics 环境。文字提示还需要 `mobileclip2_b.ts`。RKNN 转换需要在
x86 Linux 上安装 `rknn-toolkit2>=2.3.2`，RK3588 板端推理需要 `rknn-toolkit-lite2`。

推荐使用 FP16。RK3588S 上的 W8A8 实验把 YOLOE-26L 延迟从约 394 ms 降到 159 ms，但在 6 张小目标样本中，
存在有效检测的图片从 2/6 降为 0/6。因此，在有代表性的标定集和精度评测证明 INT8 可用之前，运行时提示模型
暂不允许 INT8 导出。

## 导出

YOLOE-26N/S/M/L/X 文字/视觉提示 checkpoint 使用同一套 API。只需要检测框时应设置
`prompt_boxes_only=True`，它会删除 mask coefficient 和 prototype，避免在 CPU 上执行昂贵的 mask 后处理。

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

`max_prompts` 是编译进静态图的最大类别容量，不是参考框数量上限。应根据这个 RKNN 模型需要支持的最大文字或
视觉类别数设置它。导出结果是包含 RKNN 文件和 metadata 的 `yoloe-26l-seg_rknn_model/` 目录。

## 生成提示 NPZ

导出后必须生成提示 NPZ。RKNN 不会自己编码文字，也不会读取参考图中的框；它只接收生成好的 embedding。NPZ 与
生成它的原始 checkpoint 尺度绑定，N/S/M/L/X 之间不能混用。

NPZ 既可以在性能较强的电脑上生成，也可以像板端网页一样，在 RK3588 CPU 上加载原始 `.pt` 后现场生成。后者
并不表示 RKNN 模型或 NPU 能生成提示，只是板子同时保存了原始 `.pt`、PyTorch 和提示编码器。文字提示需要加载
PyTorch checkpoint 和 MobileCLIP，在板端通常较慢，因此更适合提前生成并缓存。

使用框架 API 生成文字提示：

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-26l-seg.pt")
names = ["塑料瓶", "易拉罐"]
embeddings = model.get_text_pe(names)
model.save_prompt_embeddings("waste_text.npz", embeddings, names, max_prompts=8)
```

使用一张参考图和原图像素坐标系中的 `xyxy` 框生成视觉提示：

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-26l-seg.pt")
visual = {"bboxes": [[120, 80, 220, 210]], "cls": [0]}
embeddings = model.get_visual_prompt_pe("reference.jpg", visual)
model.save_prompt_embeddings("bottle_visual.npz", embeddings, ["塑料瓶"], max_prompts=8)
```

辅助脚本支持多张参考图。每张图片独立提取 embedding，同一类别跨图片求平均后重新进行 L2 归一化。这样不会像
拼接裁剪目标那样引入虚假的目标位置、边界和场景关系。

```json
{
  "names": ["塑料瓶", "易拉罐"],
  "images": [
    { "image": "refs/bottle_1.jpg", "boxes": [[120, 80, 220, 210]], "class_ids": [0] },
    {
      "image": "refs/mixed.jpg",
      "boxes": [
        [25, 40, 90, 180],
        [240, 95, 330, 210]
      ],
      "class_ids": [0, 1]
    }
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

模型支持联合多类别 NPZ，但不同视觉类别可能互相干扰。对小目标等精度敏感场景，推荐使用
`prompts/by_class/` 中的文件逐类推理，再合并经过 NMS 的检测框。每个单类别 embedding 仍然可以由任意数量的
参考框和参考图片生成。

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

## RK3588 板端推理

将 RKNN 模型目录、提示 NPZ 和待检测媒体复制到板子。标准 Ultralytics API 会返回 `Results`，其中
`speed["inference"]` 是模型推理耗时，单位为毫秒。

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

需要测量纯 NPU 延迟和原始输出时可使用 `infer.py`，`--core` 用于选择 RK3588 NPU 核。把一个请求分配到三个
NPU 核不会显著降低单张延迟。并发请求应创建三个独立 RKNNLite context，分别绑定 Core 0、1、2。实测
YOLOE-26L 从单 context 的 2.54 FPS 提升到三 context 的 6.68 FPS。

## RK3588S 实测结果

以下 FP16 数据使用 RKNN Toolkit 2.3.2、RKNNLite 2.3.0、NPU driver 0.8.8、640 x 640 输入以及
`person,bus` 文字 embedding。每项数据均为一次预热后在 Core 0 上运行 5 次的均值。

| 尺寸      | NPU 平均耗时 | 最大分数 | RKNN 大小 |
| --------- | -----------: | -------: | --------: |
| YOLOE-26N |     75.96 ms |   0.8950 |   8.2 MiB |
| YOLOE-26S |    143.32 ms |   0.9067 |  23.4 MiB |
| YOLOE-26M |    348.29 ms |   0.9233 |  49.9 MiB |
| YOLOE-26L |    393.89 ms |   0.9199 |  59.5 MiB |
| YOLOE-26X |    887.71 ms |   0.9131 | 127.9 MiB |

不支持 prompt-free checkpoint。它的固定 4585 类分类器在 640 x 640 下产生 `[1,4621,8400]` 输出，
YOLOE-26N 在同一板子实测为 3043.61 ms。相比小型运行时词表，巨大的输出和 CPU 传输开销使其不适合在该平台
部署。

## 已知限制

- 输入尺寸、batch 和提示类别容量都是静态的。
- NMS 在 RKNN 图外由 CPU 执行。
- 生成文字或视觉提示需要主机或板端保留 PyTorch checkpoint。
- 运行时提示 NPZ 与生成它的源模型绑定。
- 实测使用的板端 runtime 和 NPU driver 早于编译器版本，正式部署前应升级并重新验证。
