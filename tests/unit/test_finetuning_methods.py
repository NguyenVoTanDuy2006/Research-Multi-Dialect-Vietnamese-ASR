"""Small CPU-only verification. No pretrained downloads or dataset training."""
from copy import deepcopy
from dataclasses import replace
import json
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from peft import PeftModel

from mdv_asr_module.configs import ConfigLoader, PiSSASettings, AdaLoraSettings
from mdv_asr_module.runtime.distributed import DistributedContext
from mdv_asr_module.training.methods import PiSSAMethod, AdaLoRAMethod, ModuleTargetSelector
from mdv_asr_module.training.optimization import OptimizerStepController, TrainProgress
from mdv_asr_module.training.checkpoints import CheckpointManager


class TinyAudioModel(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.config = PretrainedConfig()
        self.config.model_type = "tiny_audio_test"
        self.name_or_path = "local/tiny-test"
        self.speech = nn.ModuleDict({"q_proj": nn.Linear(8, 8), "v_proj": nn.Linear(8, 8)})
        self.to(dtype=dtype)

    def forward(self, input_features, labels=None):
        logits = self.speech["v_proj"](torch.tanh(self.speech["q_proj"](input_features)))
        loss = ((logits - labels) ** 2).mean() if labels is not None else None
        return CausalLMOutput(loss=loss, logits=logits)


@pytest.fixture(autouse=True)
def small_cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(12)
    yield
    torch.set_num_threads(old)


def new_pissa(**kwargs):
    return PiSSAMethod(PiSSASettings(r=2, alpha=4, **kwargs))


def one_step(model, method):
    controller = OptimizerStepController(torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.01))
    x, target = torch.randn(3, 8), torch.randn(3, 8)
    controller.backward(model(input_features=x, labels=target).loss, 1)
    result = controller.step(model, method)
    assert result.successful
    return controller


def test_generic_selection_no_qwen_or_text_input_assumptions():
    base = TinyAudioModel()
    assert ModuleTargetSelector(("q_proj",)).select(base, rank=2) == ["speech.q_proj"]
    assert ModuleTargetSelector(r"speech\.[qv]_proj").select(base, rank=2) == ["speech.q_proj", "speech.v_proj"]
    with pytest.raises(ValueError, match="No Linear"):
        ModuleTargetSelector(("missing",)).select(base, rank=2)
    with pytest.raises(ValueError, match="exceeds"):
        ModuleTargetSelector(("q_proj",)).select(base, rank=9)


def test_pissa_config_roundtrip_and_validation(tmp_path):
    loader = ConfigLoader()
    cfg = loader.from_dict({"model": {"key": "unit", "backend": "qwen3_asr", "model_id": "test"},
                            "train": {"method": "pissa", "pissa": {"r": 2, "target_modules": ["q_proj"]}}})
    loader.save_resolved(cfg, tmp_path / "config.json")
    restored = loader.load(tmp_path / "config.json")
    assert restored.train.method == "pissa"
    assert restored.train.pissa.target_modules == ("q_proj",)
    with pytest.raises(ValueError):
        replace(restored.train.pissa, init_lora_weights="random").validate()


def test_pissa_initial_output_frozen_base_and_shared_seed():
    base = TinyAudioModel()
    other = deepcopy(base)
    x = torch.randn(2, 8)
    expected = base(input_features=x).logits.detach()
    rng = torch.get_rng_state().clone()
    method = new_pissa()
    model = method.attach(base)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.allclose(model(input_features=x).logits, expected, atol=2e-6)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    torch.manual_seed(999)  # Simulate a different per-rank training RNG.
    second = new_pissa()
    second.attach(other)
    assert all(torch.equal(v, second.initial_adapter_state[k]) for k, v in method.initial_adapter_state.items())
    one_step(model, method)
    assert method.successful_steps == 1
    assert all(torch.equal(p, frozen[n]) for n, p in model.named_parameters() if n in frozen)


def test_pissa_export_is_portable_and_does_not_mutate_live_model(tmp_path):
    original = TinyAudioModel()
    method = new_pissa()
    model = method.attach(deepcopy(original))
    one_step(model, method)
    # Snapshot may be written after a step; must still contain the ORIGINAL factors.
    method.save_initialization(tmp_path / "pissa_initial")
    state = {n: v.clone() for n, v in model.state_dict().items()}
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    settings_before = deepcopy(model.peft_config["default"].to_dict())
    method.save_adapter(tmp_path / "export")
    assert torch.equal(rng, torch.get_rng_state())
    assert model.peft_config["default"].to_dict() == settings_before
    assert flags == {n: p.requires_grad for n, p in model.named_parameters()}
    assert model.training
    assert all(torch.equal(v, model.state_dict()[n]) for n, v in state.items())
    cfg = json.loads((tmp_path / "export/adapter_config.json").read_text())
    assert cfg["r"] == 4 and cfg["init_lora_weights"] is True
    loaded = PeftModel.from_pretrained(original, tmp_path / "export").eval()
    from mdv_asr_module.models.audit import AdapterAuditor
    audit = AdapterAuditor().compare_loaded_state(tmp_path / "export", loaded)
    assert audit["compared_tensor_count"] == len(method.initial_adapter_state)
    x = torch.randn(4, 8)
    model.eval()
    assert torch.allclose(loaded(input_features=x).logits, model(input_features=x).logits, atol=3e-6)
    # A second export must also work, without doubling live rank/alpha.
    method.save_adapter(tmp_path / "export2")
    assert model.peft_config["default"].r == 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pissa_resume_restores_exact_residual_without_another_svd(dtype):
    original = TinyAudioModel(dtype=dtype)
    method = new_pissa()
    method.attach(deepcopy(original))
    state = method.state_dict()
    resumed = new_pissa()
    with patch("peft.tuners.lora.layer.svd_lowrank", side_effect=AssertionError("Resume must not repeat SVD")):
        model = resumed.attach(deepcopy(original), initialization_state=state)
    resumed.load_state_dict(state)
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in method.residual_state.items())
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in method.initial_adapter_state.items())


def test_checkpoint_resume_continues_identically(tmp_path):
    original = TinyAudioModel()
    method = new_pissa()
    model = method.attach(deepcopy(original))
    controller = one_step(model, method)
    progress = TrainProgress(attempted_steps=1, successful_steps=1)
    checkpoints = CheckpointManager(tmp_path)
    path = checkpoints.save_training_state(
        epoch=1, model=model, optimizer_controller=controller, method=method,
        progress=progress, history=[{"epoch": 1}],
        rng_by_rank=[checkpoints.capture_rng_state()], config_fingerprint="unit", model_revision="rev")
    payload = checkpoints.read_training_state(path)
    method2 = new_pissa()
    model2 = method2.attach(deepcopy(original), initialization_state=payload["method_state"])
    opt2 = OptimizerStepController(torch.optim.AdamW([p for p in model2.parameters() if p.requires_grad], lr=0.01))
    restored = checkpoints.load_training_state(path, model=model2, optimizer_controller=opt2,
                                             method=method2, context=DistributedContext(),
                                             config_fingerprint="unit", model_revision="rev")
    assert restored["epoch"] == 1 and method2.successful_steps == 1
    x, y = torch.randn(2, 8), torch.randn(2, 8)
    for m, meth, opt in [(model, method, controller), (model2, method2, opt2)]:
        opt.backward(m(input_features=x, labels=y).loss, 1)
        opt.step(m, meth)
    assert all(torch.equal(v, model2.state_dict()[k]) for k, v in model.state_dict().items())


def test_resume_rejects_method_mismatch(tmp_path):
    from mdv_asr_module.core.errors import ResumeMismatchError
    method = new_pissa()
    model = method.attach(TinyAudioModel())
    controller = one_step(model, method)
    checkpoints = CheckpointManager(tmp_path)
    path = checkpoints.save_training_state(epoch=1, model=model, optimizer_controller=controller,
        method=method, progress=TrainProgress(attempted_steps=1, successful_steps=1), history=[],
        rng_by_rank=[checkpoints.capture_rng_state()], config_fingerprint="unit", model_revision=None)
    ada = AdaLoRAMethod(AdaLoraSettings(init_r=4, target_r=2))
    ada_model = ada.attach(TinyAudioModel(), total_steps=10)
    ada_opt = OptimizerStepController(torch.optim.AdamW([p for p in ada_model.parameters() if p.requires_grad]))
    with pytest.raises(ResumeMismatchError, match="method"):
        checkpoints.load_training_state(path, model=ada_model, optimizer_controller=ada_opt,
            method=ada, context=DistributedContext(), config_fingerprint="unit", model_revision=None)


def test_skipped_update_does_not_call_method_or_scheduler():
    method = new_pissa()
    model = method.attach(TinyAudioModel())
    controller = one_step(model, method)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.full_like(p, float("inf"))
    result = controller.step(model, method)
    assert result.skipped and method.successful_steps == 1
    assert all(p.grad is None for p in model.parameters())


def test_generic_adalora_keeps_peft_orthogonality_regularization():
    method = AdaLoRAMethod(AdaLoraSettings(init_r=4, target_r=2, dropout=0))
    model = method.attach(TinyAudioModel(), total_steps=5)
    inputs = dict(input_features=torch.randn(2, 8), labels=torch.randn(2, 8))
    raw = model.get_base_model()(**inputs).loss.detach()
    regularized = model(**inputs).loss.detach()
    assert regularized > raw


def test_adalora_export_has_no_empty_tensors_and_is_equivalent(tmp_path):
    from safetensors.torch import load_file
    original = TinyAudioModel()
    method = AdaLoRAMethod(AdaLoraSettings(init_r=4, target_r=2, alpha=4, dropout=0, delta_t=1))
    model = method.attach(deepcopy(original), total_steps=5)
    for _ in range(5):
        one_step(model, method)
    pattern = deepcopy(model.peft_config["default"].rank_pattern)
    method.save_adapter(tmp_path / "ada")
    tensors = load_file(str(tmp_path / "ada/adapter_model.safetensors"))
    assert all(t.numel() > 0 for t in tensors.values())
    assert model.peft_config["default"].rank_pattern == pattern
    assert json.loads((tmp_path / "ada/adapter_config.json").read_text())["rank_pattern"] is None
    loaded = PeftModel.from_pretrained(original, tmp_path / "ada").eval()
    model.eval()
    x = torch.randn(4, 8)
    assert torch.allclose(loaded(input_features=x).logits, model(input_features=x).logits, atol=1e-6)


@pytest.mark.parametrize("method_name", ["pissa", "adalora"])
def test_native_tiny_qwen_audio_decoder_backward_and_generation(method_name):
    # Real Transformers Qwen3-ASR class with tiny RANDOM weights. This validates
    # the multimodal call path, not pretrained accuracy or T4 memory usage.
    from transformers import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration
    cfg = Qwen3ASRConfig(
        audio_config={"d_model": 16, "encoder_layers": 1, "encoder_attention_heads": 2,
                      "encoder_ffn_dim": 32, "num_mel_bins": 8, "output_dim": 16,
                      "downsample_hidden_size": 8},
        text_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
                     "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8,
                     "vocab_size": 32, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 5},
        audio_token_id=2, pad_token_id=0, bos_token_id=1, eos_token_id=5,
    )
    base = Qwen3ASRForConditionalGeneration(cfg)
    method = new_pissa() if method_name == "pissa" else AdaLoRAMethod(
        AdaLoraSettings(init_r=4, target_r=2, dropout=0, delta_t=1))
    model = method.attach(base, total_steps=5)
    inputs = {"input_ids": torch.tensor([[1, 2, 2, 3, 4]]),
              "input_features": torch.randn(1, 8, 100),
              "input_features_mask": torch.cat([torch.ones(1, 16, dtype=torch.long),
                                                 torch.zeros(1, 84, dtype=torch.long)], dim=1),
              "attention_mask": torch.ones(1, 5, dtype=torch.long)}
    labels = torch.tensor([[-100, -100, -100, 3, 4]])
    out = model(**inputs, labels=labels, use_cache=False)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    names = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
    assert any("audio_tower" in n for n in names)
    assert any("language_model" in n for n in names)
    controller = OptimizerStepController(torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]))
    assert controller.step(model, method).successful
    model.eval()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=2, do_sample=False)
    assert generated.shape[1] > inputs["input_ids"].shape[1]
