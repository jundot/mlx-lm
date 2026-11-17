# Copyright © 2026 Apple Inc.

import mlx.core as mx
import numpy as np


class Processor:
    image_token = "<|image_pad|>"

    def __init__(self, image_processor):
        self.image_processor = image_processor

    @classmethod
    def from_pretrained(cls, model_path):
        from transformers import AutoImageProcessor

        if model_path is None:
            raise ValueError(
                "Qwen3.5 needs the checkpoint's image processor configuration."
            )
        return cls(AutoImageProcessor.from_pretrained(model_path, backend="pil"))

    def __call__(self, text, *, images):
        parts = text.split(self.image_token)
        if len(parts) - 1 != len(images):
            raise ValueError(
                f"The text has {len(parts) - 1} image placeholders for {len(images)} images."
            )
        pixels, expanded = [], [parts[0]]
        merge = self.image_processor.merge_size
        for image, part in zip(images, parts[1:], strict=True):
            processed = self.image_processor(images=image, return_tensors="np")
            t, h, w = map(int, processed["image_grid_thw"][0])
            if t != 1:
                raise ValueError("Only still images are supported.")
            # Preserve the patch grid without changing the processor's patch order.
            pixels.append(
                mx.array(np.asarray(processed["pixel_values"]).reshape(h, w, -1))
            )
            expanded.extend([self.image_token * (h * w // merge**2), part])
        return "".join(expanded), pixels
