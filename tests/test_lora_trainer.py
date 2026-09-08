from training.lora_trainer import LoRAFinetuner, TrainConfig, LoRAConfig


class TestToPromptCompletion:
    def test_prompt_completion_shape_passthrough(self):
        p, c = LoRAFinetuner._to_prompt_completion({"prompt": "P", "completion": "C"})
        assert (p, c) == ("P", "C")

    def test_alpaca_with_input(self):
        p, c = LoRAFinetuner._to_prompt_completion(
            {"instruction": "do it", "input": "ctx", "output": "done"}
        )
        assert "do it" in p and "ctx" in p
        assert p.rstrip().endswith("### Response:")
        assert c == "done"

    def test_alpaca_without_input_omits_input_section(self):
        p, c = LoRAFinetuner._to_prompt_completion(
            {"instruction": "do it", "input": "", "output": "done"}
        )
        assert "### Input:" not in p
        assert c == "done"


class TestTrainConfigDefaults:
    def test_lora_defaults_populated(self):
        cfg = TrainConfig()
        assert isinstance(cfg.lora, LoRAConfig)
        assert cfg.lora.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]

    def test_explicit_lora_target_modules_respected(self):
        cfg = TrainConfig(lora=LoRAConfig(target_modules=["q_proj"]))
        assert cfg.lora.target_modules == ["q_proj"]

    def test_four_bit_off_by_default(self):
        assert TrainConfig().load_in_4bit is False
