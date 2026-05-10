import torch
import torch.nn as nn
from .modeling_llama_kv import KVLlamaForCausalLM
from .modeling_mistral_kv import KVMistralForCausalLM

from transformers import PreTrainedModel, PretrainedConfig, LlamaConfig, MistralConfig
from .utils import *
from .kv_cache import initialize_past_key_values
from .medusa_choices import *
from transformers import AutoTokenizer, AutoConfig
import os
import json
from huggingface_hub import hf_hub_download
import warnings

class MedusaConfig(PretrainedConfig):
    def __init__(self, medusa_num_heads=5, medusa_num_layers=1, base_model_name_or_path="lmsys/vicuna-7b-v1.3", **kwargs):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path

def _ensure_medusa_fields(config, pretrained_model_name_or_path):
    if not hasattr(config, "medusa_num_heads"): config.medusa_num_heads = 5
    if not hasattr(config, "medusa_num_layers"): config.medusa_num_layers = 1
    if not getattr(config, "base_model_name_or_path", None):
        config.base_model_name_or_path = getattr(config, "_name_or_path", None) or pretrained_model_name_or_path
    return config

def _load_medusa_config_fallback(pretrained_model_name_or_path):
    try:
        config_path = hf_hub_download(pretrained_model_name_or_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as handle:
            config_dict = json.load(handle)
        config = MedusaConfig(**config_dict)
        return _ensure_medusa_fields(config, pretrained_model_name_or_path)
    except Exception:
        return MedusaConfig(base_model_name_or_path=pretrained_model_name_or_path)

class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()
    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        res = self.linear(x.to(self.linear.weight.dtype))
        res = self.act(res).to(torch.float32)
        return (x + res).to(input_dtype)

class MedusaModelABC(nn.Module):
    def _init_medusa(self, config):
        self.medusa_config = config
        self.medusa = config.medusa_num_heads
        self.medusa_num_layers = config.medusa_num_layers
        base_path = config.base_model_name_or_path if hasattr(config, "base_model_name_or_path") else config._name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False)
        self.base_model_name_or_path = base_path
        self.medusa_head = nn.ModuleList([
            nn.Sequential(*([ResBlock(self.config.hidden_size)] * config.medusa_num_layers),
            nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False))
            for _ in range(config.medusa_num_heads)
        ])

    def get_tokenizer(self): return self.tokenizer

    def fuse_medusa_heads(self):
        self.fused_res_weights = nn.ParameterList()
        self.fused_res_biases = nn.ParameterList()
        for layer_idx in range(self.medusa_num_layers):
            weights, biases = [], []
            for head_idx in range(self.medusa):
                res_block = self.medusa_head[head_idx][layer_idx]
                weights.append(res_block.linear.weight.data)
                biases.append(res_block.linear.bias.data)
            self.fused_res_weights.append(nn.Parameter(torch.stack(weights)))
            self.fused_res_biases.append(nn.Parameter(torch.stack(biases)))
            
        weights = []
        for head_idx in range(self.medusa):
            linear = self.medusa_head[head_idx][-1]
            weights.append(linear.weight.data)
        self.fused_final_weight = nn.Parameter(torch.stack(weights))
        self.is_fused = True
        
        # Free original un-batched heads to save VRAM
        del self.medusa_head
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def medusa_forward(self, input_ids=None, attention_mask=None, past_key_values=None, output_orig=False, position_ids=None, **kwargs):
        # [MODIFIED] Inject the medusa_mask if not provided, padded for past_key_values
        if attention_mask is None and hasattr(self.model, "medusa_mask") and self.model.medusa_mask is not None:
            medusa_mask = self.model.medusa_mask
            past_length = past_key_values[0][0].current_length.item() if past_key_values is not None and hasattr(past_key_values[0][0], "current_length") else 0
            if past_length > 0:
                padding = torch.zeros(
                    (medusa_mask.size(0), medusa_mask.size(1), medusa_mask.size(2), past_length),
                    device=medusa_mask.device, dtype=medusa_mask.dtype
                )
                attention_mask = torch.cat([padding, medusa_mask], dim=-1)
            else:
                attention_mask = medusa_mask

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, position_ids=position_ids, **kwargs)
            if output_orig:
                orig = self.lm_head(outputs[0])
        
        hidden_states = outputs[0]
        
        if getattr(self, "is_fused", False):
            # Batched execution for Medusa Heads
            input_dtype = hidden_states.dtype
            B, L, D = hidden_states.shape
            
            # Expand for K heads
            h_float = hidden_states.unsqueeze(0).expand(self.medusa, -1, -1, -1).reshape(self.medusa, B*L, D).to(torch.float32)
            
            for layer_idx in range(self.medusa_num_layers):
                w = self.fused_res_weights[layer_idx].transpose(1, 2)
                b = self.fused_res_biases[layer_idx]
                
                res = torch.bmm(h_float.to(w.dtype), w)
                res = res + b.unsqueeze(1)
                import torch.nn.functional as F
                res = F.silu(res).to(torch.float32)
                h_float = h_float + res
                
            w = self.fused_final_weight.transpose(1, 2)
            logits = torch.bmm(h_float.to(w.dtype), w)
            
            medusa_logits = logits.reshape(self.medusa, B, L, -1).to(input_dtype)
        else:
            hidden_states = hidden_states.to(torch.float32)
            medusa_logits = [self.medusa_head[i](hidden_states).to(outputs[0].dtype) for i in range(self.medusa)]
            medusa_logits = torch.stack(medusa_logits, dim=0)
            
        if output_orig: return medusa_logits, outputs, orig
        return medusa_logits

    def get_medusa_choice(self, model_name):
        if 'vicuna' in model_name:
            if '7b' in model_name: choices = vicuna_7b_stage2
            elif '13b' in model_name: choices = vicuna_13b_stage2
            elif '33b' in model_name: choices = vicuna_33b_stage2
            else: choices = mc_sim_7b_63
        elif 'zephyr' in model_name: choices = zephyr_stage2
        else:
            warnings.warn('Please specify medusa choice configuration!')
            choices = mc_sim_7b_63
        filtered = [c for c in choices if len(c) <= self.medusa]
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
            # Use robust device extraction
            device = self.lm_head.weight.device
            medusa_buffers = generate_medusa_buffers(medusa_choices, device=device)
        self.medusa_buffers = medusa_buffers
        self.medusa_choices = medusa_choices
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            (past_key_values, past_key_values_data, current_length_data) = initialize_past_key_values(self)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data
        input_len = input_ids.shape[1]
        reset_medusa_mode(self)
        medusa_logits, logits = initialize_medusa(input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values)
        new_token = 0
        for idx in range(max_steps):
            candidates, tree_candidates = generate_candidates(medusa_logits, logits, medusa_buffers["tree_indices"], medusa_buffers["retrieve_indices"], temperature=temperature, posterior_alpha=posterior_alpha, posterior_threshold=posterior_threshold, top_p=top_p, sampling=sampling, fast=fast)
            medusa_logits, logits, outputs = tree_decoding(self, tree_candidates, past_key_values, medusa_buffers["medusa_position_ids"], input_ids, medusa_buffers["retrieve_indices"])
            best_candidate, accept_length = evaluate_posterior(logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast)
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(input_ids, candidates, best_candidate, accept_length, medusa_buffers["retrieve_indices"], outputs, logits, medusa_logits, new_token, past_key_values_data, current_length_data)
            # [MODIFIED] Let the tokenizer handle spacing naturally
            decoded_text = self.tokenizer.decode(input_ids[0, input_len:], skip_special_tokens=True)
            yield {"text": decoded_text, "token_ids": input_ids[0, input_len:]}
            if self.tokenizer.eos_token_id in input_ids[0, input_len:]: break

def _load_medusa_head_state_dict(pretrained_model_name_or_path):
    filename = None
    local_path = os.path.join(pretrained_model_name_or_path, "medusa_lm_head.pt")
    if os.path.exists(local_path): filename = local_path
    else:
        try: filename = hf_hub_download(pretrained_model_name_or_path, "medusa_lm_head.pt")
        except Exception: pass
    if filename: return torch.load(filename, map_location="cpu", weights_only=True)
    return None

def _remove_accelerate_hooks(model):
    def remove_hooks(module):
        if hasattr(module, "_hf_hook"): delattr(module, "_hf_hook")
        for child in module.children(): remove_hooks(child)
    if hasattr(model, "medusa_head") and model.medusa_head is not None: model.medusa_head.apply(remove_hooks)

def patch_lm_head(model):
    if hasattr(model, "lm_head"):
        model.lm_head.to(dtype=torch.float32)
        old_forward = model.lm_head.forward
        def new_forward(x):
            return old_forward(x.to(torch.float32)).to(x.dtype)
        model.lm_head.forward = new_forward

class MedusaModelLlama(KVLlamaForCausalLM, MedusaModelABC):
    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, medusa_forward=False, **kwargs):
        if medusa_forward:
            return self.medusa_forward(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        try: m_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        except Exception: m_config = _load_medusa_config_fallback(pretrained_model_name_or_path)
        m_config = _ensure_medusa_fields(m_config, pretrained_model_name_or_path)
        base_path = m_config.base_model_name_or_path
        model = KVLlamaForCausalLM.from_pretrained(base_path, *args, **kwargs)
        model.__class__ = cls
        state_dict = _load_medusa_head_state_dict(pretrained_model_name_or_path)
        if state_dict:
            head_indices = {int(k.split(".")[0]) for k in state_dict.keys() if k.split(".")[0].isdigit()}
            m_config.medusa_num_heads = len(head_indices)
            model._init_medusa(m_config)
            model.medusa_head.load_state_dict(state_dict, strict=False)
        else: model._init_medusa(m_config)
        if hasattr(model, "lm_head"):
            model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone().float())
            patch_lm_head(model)
        
        model.fuse_medusa_heads()
        target_device = model.lm_head.weight.device
        for i in range(len(model.fused_res_weights)):
            model.fused_res_weights[i].data = model.fused_res_weights[i].data.to(dtype=torch.float32, device=target_device)
            model.fused_res_biases[i].data = model.fused_res_biases[i].data.to(dtype=torch.float32, device=target_device)
        model.fused_final_weight.data = model.fused_final_weight.data.to(dtype=torch.float32, device=target_device)
        
        _remove_accelerate_hooks(model)
        return model

class MedusaModelMistral(KVMistralForCausalLM, MedusaModelABC):
    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, medusa_forward=False, **kwargs):
        if medusa_forward:
            return self.medusa_forward(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        try: m_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        except Exception: m_config = _load_medusa_config_fallback(pretrained_model_name_or_path)
        m_config = _ensure_medusa_fields(m_config, pretrained_model_name_or_path)
        base_path = m_config.base_model_name_or_path
        model = KVMistralForCausalLM.from_pretrained(base_path, *args, **kwargs)
        model.__class__ = cls
        state_dict = _load_medusa_head_state_dict(pretrained_model_name_or_path)
        if state_dict:
            head_indices = {int(k.split(".")[0]) for k in state_dict.keys() if k.split(".")[0].isdigit()}
            m_config.medusa_num_heads = len(head_indices)
            model._init_medusa(m_config)
            model.medusa_head.load_state_dict(state_dict, strict=False)
        else: model._init_medusa(m_config)
        if hasattr(model, "lm_head"):
            model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone().float())
            patch_lm_head(model)
        
        model.fuse_medusa_heads()
        target_device = model.lm_head.weight.device
        for i in range(len(model.fused_res_weights)):
            model.fused_res_weights[i].data = model.fused_res_weights[i].data.to(dtype=torch.float32, device=target_device)
            model.fused_res_biases[i].data = model.fused_res_biases[i].data.to(dtype=torch.float32, device=target_device)
        model.fused_final_weight.data = model.fused_final_weight.data.to(dtype=torch.float32, device=target_device)
        
        _remove_accelerate_hooks(model)
        return model

class MedusaModel():
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        except Exception:
            config = _load_medusa_config_fallback(pretrained_model_name_or_path)
            
        # Get model_type, ensuring it's not None or empty
        model_type = getattr(config, "model_type", None)
        
        if not model_type:
            # Try to get model_type from base model if Medusa config is minimal or generic
            try:
                base_path = getattr(config, "base_model_name_or_path", pretrained_model_name_or_path)
                if base_path == pretrained_model_name_or_path:
                    # Avoid infinite recursion if they are the same
                    model_type = "llama"
                else:
                    base_config = AutoConfig.from_pretrained(base_path)
                    model_type = base_config.model_type
            except Exception:
                model_type = "llama"

        if model_type == "llama":
            return MedusaModelLlama.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        elif model_type == "mistral":
            return MedusaModelMistral.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        raise ValueError(f"Unsupported model type: '{model_type}'")

try:
    from transformers import GenerationMixin
    for model_class in [MedusaModelLlama, MedusaModelMistral]:
        if not hasattr(model_class, "generate"): model_class.generate = GenerationMixin.generate
except Exception: pass
