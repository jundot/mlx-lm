# Copyright © 2026 Apple Inc.

from .qwen3_5 import Model as Qwen3_5Model
from .qwen3_5 import ModelArgs


class Model(Qwen3_5Model):

    def sanitize(self, weights):
        new_weights = super().sanitize(weights)

        for l in range(self.language_model.args.num_hidden_layers):
            prefix = f"language_model.model.layers.{l}.mlp"
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            if gate_up_key in new_weights:
                gate_up = new_weights.pop(gate_up_key)
                mid = gate_up.shape[-2] // 2
                new_weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[
                    ..., :mid, :
                ]
                new_weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[
                    ..., mid:, :
                ]
                new_weights[f"{prefix}.switch_mlp.down_proj.weight"] = new_weights.pop(
                    f"{prefix}.experts.down_proj"
                )

        return new_weights
