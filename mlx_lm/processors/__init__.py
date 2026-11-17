# Copyright © 2026 Apple Inc.

"""Image processors of multimodal models, one module per model."""

from mlx import nn

from . import inkling2, muse_spark, qwen3_5

# The processor of each model module.
PROCESSORS = {
    "qwen3_5": qwen3_5.Processor,
    "qwen3_5_moe": qwen3_5.Processor,
    "inkling2": inkling2.Processor,
    "muse_spark": muse_spark.Processor,
}


def load_processor(
    model: nn.Module, config: dict, model_path=None
) -> inkling2.Processor | muse_spark.Processor | qwen3_5.Processor | None:
    """The processor of a model, or None."""
    if not config.get("vision_config"):
        return None
    model_module = type(model).__module__.rsplit(".", 1)[-1]
    name = config.get("processor_type", model_module)
    processor = PROCESSORS.get(name)
    if processor is qwen3_5.Processor:
        return processor.from_pretrained(model_path)
    return processor.from_config(config) if processor else None


def prepare_inputs(model, inputs, *, pixel_values, cache=None):
    prepared = model.embed_inputs(inputs, pixel_values=pixel_values, cache=cache)
    return prepared if isinstance(prepared, dict) else {"input_embeddings": prepared}
