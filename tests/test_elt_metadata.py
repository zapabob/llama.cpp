from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("convert_hf_to_gguf", ROOT / "convert_hf_to_gguf.py")
assert SPEC is not None
assert SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


class FakeWriter:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def __getattr__(self, name: str):
        if not name.startswith("add_"):
            raise AttributeError(name)

        def add_value(key: str, value: object) -> None:
            self.values[key] = value

        return add_value


def make_model(elt_config: dict[str, object] | None):
    model = object.__new__(converter.ModelBase)
    model.gguf_writer = FakeWriter()
    model.hparams = {
        "num_attention_heads": 4,
        "hidden_size": 512,
    }
    if elt_config is not None:
        model.hparams["elt_config"] = elt_config
    model.block_count = 3
    model.remote_hf_model_id = None
    model.model_name = None
    model.dir_model = Path("elt-qwen35")
    model.model_arch = converter.gguf.MODEL_ARCH.QWEN35
    model.turboquant_mode = "research-kv-split"
    model.turboquant_rotation_policy = "triality_vector"
    model.turboquant_rotation_seed = 7
    model.turboquant_triality_mix = None
    model.turboquant_artifact = None
    model.turboquant_weight_enabled = True
    model.turboquant_weight_source_ftype = "q8_0"
    model.turboquant_weight_policy = None
    model.turboquant_weight_protected_roles = None
    model.turboquant_weight_protected_layers = None
    model.turboquant_weight_modality_scope = None
    return model


def test_elt_config_emits_stable_metadata_keys() -> None:
    model = make_model(
        {
            "enabled": True,
            "required": True,
            "L_min": 2,
            "L_max": 6,
            "L_default": 4,
            "loop_unit": "decoder_layer",
            "backbone_kind": "qwen35",
            "source_model_id": "Qwen/Qwen3.5-9B-Instruct",
        }
    )

    model.add_elt_metadata()

    assert model.gguf_writer.values["elt.loop.enabled"] is True
    assert model.gguf_writer.values["elt.loop.required"] is True
    assert model.gguf_writer.values["elt.loop.L_min"] == 2
    assert model.gguf_writer.values["elt.loop.L_max"] == 6
    assert model.gguf_writer.values["elt.loop.L_default"] == 4
    assert model.gguf_writer.values["elt.loop_unit"] == "decoder_layer"
    assert model.gguf_writer.values["elt.backbone_kind"] == "qwen35"
    assert model.gguf_writer.values["elt.source_model_id"] == "Qwen/Qwen3.5-9B-Instruct"
    assert model.gguf_writer.values["elt.gguf.runtime_status"] == "requires_looped_qwen35_runtime"
    assert model.gguf_writer.values["elt.model_family"] == "ELT/Qwen3.5-looped"
    assert model.gguf_writer.values["elt.loop.model_family"] == "ELT/Qwen3.5-looped"


def test_plain_qwen_has_no_elt_metadata_or_turboquant_payload() -> None:
    model = make_model(None)

    model.add_elt_metadata()
    model.add_hypura_turboquant_metadata()

    assert not any(key.startswith("elt.") for key in model.gguf_writer.values)
    payload = json.loads(model.gguf_writer.values["hypura.turboquant.weight.payload_json"])
    assert payload["model_family"] == "elt-qwen35"
    assert "elt" not in payload


def test_turboquant_payload_carries_looped_elt_model_family() -> None:
    model = make_model(
        {
            "L_min": 1,
            "L_max": 4,
            "L_default": 4,
            "loop_unit": "decoder_layer",
            "backbone_kind": "qwen35",
            "source_model_id": "Qwen/Qwen3.5-9B-Instruct",
        }
    )

    model.add_hypura_turboquant_metadata()

    payload = json.loads(model.gguf_writer.values["hypura.turboquant.weight.payload_json"])
    assert payload["model_family"] == "ELT/Qwen3.5-looped"
    expected_elt = {
        "loop_enabled": True,
        "loop_required": True,
        "L_min": 1,
        "L_max": 4,
        "L_default": 4,
        "loop_unit": "decoder_layer",
        "backbone_kind": "qwen35",
        "source_model_id": "Qwen/Qwen3.5-9B-Instruct",
    }
    for key, value in expected_elt.items():
        assert payload["elt"][key] == value
    assert payload["elt"]["turboquant_model_family"] == "ELT/Qwen3.5-looped"
