import torch
import torch.nn as nn
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mistral_kv import MistralForCausalLM as KVMistralForCausalLM

from transformers import PreTrainedModel, PretrainedConfig
from .utils import *
from .kv_cache import initialize_past_key_values
from .medusa_choices import *
from transformers import AutoTokenizer, AutoConfig
import os
import json
from huggingface_hub import hf_hub_download
import warnings

class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.
    """
    def __init__(
        self,
        medusa_num_heads=5,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path


def _ensure_medusa_fields(config, pretrained_model_name_or_path):
    if not hasattr(config, "medusa_num_heads"):
        config.medusa_num_heads = 5
    if not hasattr(config, "medusa_num_layers"):
        config.medusa_num_layers = 1
    if not getattr(config, "base_model_name_or_path", None):
        config.base_model_name_or_path = getattr(
            config, "_name_or_path", None
        ) or pretrained_model_name_or_path
    return config


def _load_medusa_config_fallback(pretrained_model_name_or_path):
    """Load Medusa config even if config.json lacks model_type."""
    try:
        config_path = hf_hub_download(pretrained_model_name_or_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as handle:
            config_dict = json.load(handle)
        config = MedusaConfig(**config_dict)
        return _ensure_medusa_fields(config, pretrained_model_name_or_path)
    except Exception:
        return MedusaConfig(base_model_name_or_path=pretrained_model_name_or_path)

class ResBlock(nn.Module):
    """A Residual Block module with SiLU activation."""
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        # [MODIFIED] Upcast to float32 for residual addition to prevent overflow on T4 GPUs
        input_dtype = x.dtype
        x = x.to(torch.float32)
        res = self.linear(x.to(self.linear.weight.dtype))
        res = self.act(res).to(torch.float32)
        return (x + res).to(input_dtype)


class MedusaModelABC(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.medusa = config.medusa_num_heads
        self.medusa_num_layers = config.medusa_num_layers
        self.base_model_name_or_path = config._name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        self.medusa_head = None

    def _add_medusa_heads(self, config):
        """Dynamically add the Medusa heads."""
        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * config.medusa_num_layers),
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                )
                for _ in range(config.medusa_num_heads)
            ]
        )
        self.medusa = config.medusa_num_heads

    @property
    def base_model(self):
        return self

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # Implementation in subclasses
        pass

    def get_tokenizer(self):
        return self.tokenizer

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        medusa_forward=False,
        **kwargs,
    ):
        if not medusa_forward:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
            if output_orig:
                # [MODIFIED] lm_head is now float32; cast hidden states to float32
                orig = self.base_model.lm_head(
                    outputs[0].to(torch.float32)
                )
        # Clone the output hidden states and upcast to float32 for Medusa heads
        hidden_states = outputs[0].to(torch.float32)
        medusa_logits = []
        for i in range(self.medusa):
            # [MODIFIED] Medusa heads are now float32
            medusa_logits.append(self.medusa_head[i](hidden_states))
        if output_orig:
            return torch.stack(medusa_logits, dim=0), outputs, orig
        return torch.stack(medusa_logits, dim=0)

    def get_medusa_choice(self, model_name):
        if 'vicuna' in model_name:
            if '7b' in model_name:
                choices = vicuna_7b_stage2
            elif '13b' in model_name:
                choices = vicuna_13b_stage2
            elif '33b' in model_name:
                choices = vicuna_33b_stage2
            else:
                choices = mc_sim_7b_63
        elif 'zephyr' in model_name:
            choices = zephyr_stage2
        else:
            warnings.warn('Please specify medusa choice configuration!')
            choices = mc_sim_7b_63

        filtered = [c for c in choices if len(c) <= self.medusa]
        if len(filtered) < len(choices):
            warnings.warn(
                f"Medusa choice tree was filtered from {len(choices)} to "
                f"{len(filtered)} paths to match medusa_num_heads={self.medusa}."
            )
        return filtered

    def medusa_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        medusa_choices=None,
        posterior_threshold=0.09,
        posterior_alpha=0.3,
        top_p=0.8, 
        sampling = 'typical', 
        fast = True
    ):
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        input_ids = input_ids.clone()

        if medusa_choices is None:
            medusa_choices = self.get_medusa_choice(self.base_model_name_or_path)

        if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
            medusa_buffers = self.medusa_buffers
        else:
            medusa_buffers = generate_medusa_buffers(
                medusa_choices, device=self.base_model.device
            )
        self.medusa_buffers = medusa_buffers
        self.medusa_choices = medusa_choices

        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_medusa_mode(self)
        medusa_logits, logits = initialize_medusa(
            input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values
        )

        new_token = 0
        for idx in range(max_steps):
            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_buffers["tree_indices"],
                medusa_buffers["retrieve_indices"],
                temperature=temperature,
                posterior_alpha=posterior_alpha,
                posterior_threshold=posterior_threshold,
                top_p=top_p,
                sampling=sampling,
                fast=fast,
            )

            medusa_logits, logits, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
            )

            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast
            )

            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                outputs,
                logits,
                medusa_logits,
                new_token,
                past_key_values_data,
                current_length_data,
            )

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break

def _load_medusa_head_state_dict(pretrained_model_name_or_path):
    """Helper to find and load medusa_lm_head.pt."""
    filename = None
    local_path = os.path.join(pretrained_model_name_or_path, "medusa_lm_head.pt")
    if os.path.exists(local_path):
        filename = local_path
    else:
        try:
            filename = hf_hub_download(pretrained_model_name_or_path, "medusa_lm_head.pt", local_files_only=True)
        except Exception:
            try:
                filename = hf_hub_download(pretrained_model_name_or_path, "medusa_lm_head.pt")
            except Exception:
                pass
    
    if filename:
        return torch.load(filename, map_location="cpu", weights_only=True)
    return None

def _apply_medusa_upcasting(model):
    """Helper to safely upcast heads to float32 for T4 stability."""
    model.medusa_head.to(device="cuda:0", dtype=torch.float32)
    if hasattr(model, "lm_head"):
        # Untie weights to prevent side effects on embeddings in 8-bit/4-bit mode
        if hasattr(model.lm_head, "weight") and model.lm_head.weight is not None:
            # We clone and upcast only the lm_head parameters
            old_weight = model.lm_head.weight
            model.lm_head.weight = nn.Parameter(old_weight.detach().clone().float())
        if hasattr(model.lm_head, "bias") and model.lm_head.bias is not None:
            model.lm_head.bias = nn.Parameter(model.lm_head.bias.detach().clone().float())
        model.lm_head.to(dtype=torch.float32)

def _remove_accelerate_hooks(model):
    def remove_hooks(module):
        if hasattr(module, "_hf_hook"):
            delattr(module, "_hf_hook")
        for child in module.children():
            remove_hooks(child)
    model.medusa_head.apply(remove_hooks)

class MedusaModelLlama(MedusaModelABC, KVLlamaForCausalLM):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # 1. Handle Config
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            if not hasattr(config, "model_type") or config.model_type != "llama":
                raise ValueError
        except Exception:
            config = _load_medusa_config_fallback(pretrained_model_name_or_path)
            config.model_type = "llama"
        
        config = _ensure_medusa_fields(config, pretrained_model_name_or_path)
        config._flash_attn_2_enabled = False

        # 2. Load Base Model
        model = super().from_pretrained(
            config.base_model_name_or_path if hasattr(config, "base_model_name_or_path") else pretrained_model_name_or_path,
            *args, **kwargs, config=config
        )

        # 3. Load Medusa Weights & Detect Heads
        state_dict = _load_medusa_head_state_dict(pretrained_model_name_or_path)
        if state_dict:
            head_indices = set()
            max_layer_idx = 0
            for key in state_dict.keys():
                parts = key.split(".")
                if parts[0].isdigit():
                    head_indices.add(int(parts[0]))
                    if parts[1].isdigit():
                        max_layer_idx = max(max_layer_idx, int(parts[1]))
            config.medusa_num_heads = len(head_indices)
            config.medusa_num_layers = max_layer_idx
            print(f"Detected {config.medusa_num_heads} heads and {config.medusa_num_layers} layers.")
            
            model._add_medusa_heads(config)
            missing, unexpected = model.medusa_head.load_state_dict(state_dict, strict=False)
            if missing: print(f"Missing Medusa keys: {missing}")
        else:
            warnings.warn("Medusa weights not found! Random initialization.")
            model._add_medusa_heads(config)

        _remove_accelerate_hooks(model)
        _apply_medusa_upcasting(model)
        return model

class MedusaModelMistral(MedusaModelABC, KVMistralForCausalLM):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # 1. Handle Config
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            if not hasattr(config, "model_type") or config.model_type != "mistral":
                raise ValueError
        except Exception:
            config = _load_medusa_config_fallback(pretrained_model_name_or_path)
            config.model_type = "mistral"
        
        config = _ensure_medusa_fields(config, pretrained_model_name_or_path)
        config._flash_attn_2_enabled = False

        # 2. Load Base Model
        model = super().from_pretrained(
            config.base_model_name_or_path if hasattr(config, "base_model_name_or_path") else pretrained_model_name_or_path,
            *args, **kwargs, config=config
        )

        # 3. Load Medusa Weights
        state_dict = _load_medusa_head_state_dict(pretrained_model_name_or_path)
        if state_dict:
            head_indices = set()
            max_layer_idx = 0
            for key in state_dict.keys():
                parts = key.split(".")
                if parts[0].isdigit():
                    head_indices.add(int(parts[0]))
                    if parts[1].isdigit():
                        max_layer_idx = max(max_layer_idx, int(parts[1]))
            config.medusa_num_heads = len(head_indices)
            config.medusa_num_layers = max_layer_idx
            
            model._add_medusa_heads(config)
            model.medusa_head.load_state_dict(state_dict, strict=False)
        else:
            model._add_medusa_heads(config)

        _remove_accelerate_hooks(model)
        _apply_medusa_upcasting(model)
        return model

class MedusaModel():
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        except Exception:
            config = _load_medusa_config_fallback(pretrained_model_name_or_path)
            base_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            config.model_type = base_config.model_type

        if config.model_type == "llama":
            return MedusaModelLlama.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        elif config.model_type == "mistral":
            return MedusaModelMistral.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        raise ValueError(f"Unsupported model type: {config.model_type}")

# Patch GenerationMixin
try:
    from transformers import GenerationMixin
    for model_class in [MedusaModelLlama, MedusaModelMistral]:
        if not hasattr(model_class, "generate"):
            model_class.generate = GenerationMixin.generate
except Exception:
    pass
