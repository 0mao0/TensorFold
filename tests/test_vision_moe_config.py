"""The Qwen3.6 MoE vision gates: the model_type passes config validation and the CUDA tower gate, and CUDA-only."""
from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from tensorfold.vision.config import validate_vision_config


def vision_tower() -> dict:
    return {"model_type": "qwen3_5_moe", "hidden_size": 1152, "out_hidden_size": 2048, "depth": 27,
            "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2, "in_channels": 3,
            "intermediate_size": 4304, "num_heads": 16, "num_position_embeddings": 2304,
            "deepstack_visual_indexes": []}


def checkpoint_config() -> dict:
    return {"model_type": "qwen3_5_moe", "image_token_id": 248056,
            "text_config": {"hidden_size": 2048, "head_dim": 256,
                            "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                                                "partial_rotary_factor": 0.25}},
            "vision_config": vision_tower()}


def test_moe_vision_config_accepts_the_native_checkpoint():
    assert validate_vision_config(checkpoint_config(), "qwen3_5_moe")["hidden_size"] == 1152


def test_moe_vision_tower_gate_accepts_the_model_type(tmp_path):
    from tensorfold.vision.qwen_cuda import vision_config

    (tmp_path / "config.json").write_text(json.dumps(checkpoint_config()))
    assert vision_config(tmp_path)["out_hidden_size"] == 2048


def test_moe_vision_is_cuda_only(tmp_path):
    from tensorfold import serve_options

    (tmp_path / "config.json").write_text(json.dumps(checkpoint_config()))
    family = SimpleNamespace(model_type="qwen3_5_moe", title="Qwen3.6 MoE", package=SimpleNamespace())
    with pytest.raises(ValueError, match="MLX path has no image wiring"):
        serve_options.check(Namespace(vision=True), family, "mlx", tmp_path)
    assert serve_options.check(Namespace(vision=True), family, "cuda", tmp_path) is None
