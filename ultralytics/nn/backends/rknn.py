# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements, is_rockchip

from .base import BaseBackend


class RKNNBackend(BaseBackend):
    """Rockchip RKNN inference backend for Rockchip NPU hardware.

    Loads and runs inference with RKNN models (.rknn files) using the RKNN-Toolkit-Lite2 runtime. Only supported on
    Rockchip devices with NPU hardware (e.g., RK3588, RK3566).
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a Rockchip RKNN model from a .rknn file or model directory.

        Args:
            weight (str | Path): Path to the .rknn file or directory containing the model.

        Raises:
            OSError: If not running on a Rockchip device.
            RuntimeError: If model loading or runtime initialization fails.
        """
        if not is_rockchip():
            raise OSError("RKNN inference is only supported on Rockchip devices.")

        LOGGER.info(f"Loading {weight} for RKNN inference...")
        check_requirements("rknn-toolkit-lite2")
        from rknnlite.api import RKNNLite

        w = Path(weight)
        if not w.is_file():
            w = next(w.rglob("*.rknn"))

        self.model = RKNNLite()
        ret = self.model.load_rknn(str(w))
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")

        ret = self.model.init_runtime()
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime: {ret}")

        # Load metadata
        metadata_file = w.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))
        self._prompt_key = None
        self._prompt_input = None
        self._prompt_count = 0

    def _load_prompt_embeddings(self, value: str | Path | np.ndarray | torch.Tensor) -> np.ndarray:
        """Load, validate, pad, and cache a runtime YOLOE prompt tensor."""
        key = None
        names = None
        count = None
        if isinstance(value, (str, Path)):
            path = Path(value).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            key = (str(path), path.stat().st_mtime_ns)
            if key == self._prompt_key:
                return self._prompt_input
            with np.load(path, allow_pickle=False) as data:
                embeddings = data["embeddings"].astype(np.float32)
                count = int(data["count"]) if "count" in data.files else embeddings.shape[1]
                names = [str(x) for x in data["names"]] if "names" in data.files else None
                prompt_model = str(data["model"].item()) if "model" in data.files else None
            expected_model = getattr(self, "prompt_model", None)
            if prompt_model and expected_model and prompt_model != expected_model:
                raise ValueError(f"Prompt file is for '{prompt_model}', but the RKNN model expects '{expected_model}'.")
        else:
            embeddings = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            embeddings = embeddings.astype(np.float32)

        if embeddings.ndim != 3 or embeddings.shape[0] != 1 or embeddings.shape[2] != 512:
            raise ValueError(f"Expected prompt embeddings shaped [1,N,512], got {embeddings.shape}.")
        max_prompts = int(getattr(self, "max_prompts", embeddings.shape[1]))
        count = embeddings.shape[1] if count is None else count
        if not 0 < count <= embeddings.shape[1] <= max_prompts:
            raise ValueError(f"Prompt count {count} and shape {embeddings.shape} exceed max_prompts={max_prompts}.")
        if embeddings.shape[1] < max_prompts:
            embeddings = np.pad(embeddings, ((0, 0), (0, max_prompts - embeddings.shape[1]), (0, 0)))
        if names is not None and len(names) != count:
            raise ValueError(f"Prompt file contains {len(names)} names for {count} active prompts.")

        self.names = {i: names[i] if names is not None else f"prompt{i}" for i in range(count)}
        self._prompt_key = key
        self._prompt_input = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._prompt_count = count
        return self._prompt_input

    def forward(self, im: torch.Tensor, prompt_embeddings=None) -> list:
        """Run inference on the Rockchip NPU.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].
            prompt_embeddings (str | Path | np.ndarray | torch.Tensor, optional): Runtime YOLOE prompt NPZ or tensor.

        Returns:
            (list): Model predictions as a list of output arrays.
        """
        im = (im.cpu().numpy() * 255).astype("uint8")
        inputs = [im]
        if getattr(self, "runtime_prompts", False):
            if prompt_embeddings is None:
                raise ValueError("This RKNN model requires prompt_embeddings='path/to/prompts.npz'.")
            inputs.append(self._load_prompt_embeddings(prompt_embeddings))
        elif prompt_embeddings is not None:
            raise ValueError(
                "prompt_embeddings was provided, but this RKNN model was not exported with runtime prompts."
            )

        outputs = self.model.inference(inputs=inputs)
        if getattr(self, "runtime_prompts", False) and self._prompt_count < self.max_prompts:
            for i, output in enumerate(outputs):
                if output.ndim == 3 and output.shape[1] >= 4 + self.max_prompts:
                    outputs[i] = np.concatenate(
                        (output[:, : 4 + self._prompt_count], output[:, 4 + self.max_prompts :]), axis=1
                    )
                    break
        return outputs
