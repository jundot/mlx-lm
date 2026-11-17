# Copyright © 2026 Apple Inc.

import mlx.core as mx
import numpy as np


class Processor:
    image_token = "<|image_pad|>"

    def __init__(self, image_processor):
        self.image_processor = image_processor

    @classmethod
    def from_config(cls, config):
        from transformers import Qwen2VLImageProcessorPil

        vision = config["vision_config"]
        return cls(
            Qwen2VLImageProcessorPil(
                size={"shortest_edge": 65536, "longest_edge": 16777216},
                patch_size=vision.get("patch_size", 16),
                temporal_patch_size=vision.get("temporal_patch_size", 2),
                merge_size=vision.get("spatial_merge_size", 2),
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        )

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
