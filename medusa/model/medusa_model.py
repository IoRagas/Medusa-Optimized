import torch
import torch.nn as nn
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mistral_kv import MistralForCausalLM as KVMistralForCausalLM
# import transformers

# # monkey patch
# transformers.models.llama.modeling_llama.LlamaForCausalLM = KVLlamaForCausalLM
# transformers.models.mistral.modeling_mistral.MistralForCausalLM = KVMistralForCausalLM

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

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 2.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
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
        config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
        return _ensure_medusa_fields(config, pretrained_model_name_or_path)
    except Exception:
        try:
            config_path = hf_hub_download(
                pretrained_model_name_or_path,
                "config.json",
                local_files_only=True,
            )
        except Exception:
            config_path = hf_hub_download(
                pretrained_model_name_or_path,
                "config.json",
            )
        with open(config_path, "r", encoding="utf-8") as handle:
            config_dict = json.load(handle)
        config = MedusaConfig(**config_dict)
        return _ensure_medusa_fields(config, pretrained_model_name_or_path)

class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        # Cast to the weight's dtype: the 4-bit quantized base model emits
        # bfloat16 hidden states, but medusa_head is kept in float16 (skipped
        # from quantization), so we align dtypes here.
        x = x.to(self.linear.weight.dtype)
        return x + self.act(self.linear(x))


class MedusaModelABC(nn.Module):
    """The Medusa Language Model Head.

    This module creates a series of prediction heads (based on the 'medusa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    # Load the base model
    # base_model_prefix = "model"
    # supports_gradient_checkpointing = True
    # _no_split_modules = ["LlamaDecoderLayer", "MistralDecoderLayer"]
    # _skip_keys_device_placement = "past_key_values"
    # _supports_flash_attn_2 = True

    def __init__(
        self,
        config,
    ):
        """
        Args:
            config (PretrainedConfig): The configuration of the MedusaModel.
        """
        super().__init__(config)
        # For compatibility with the old APIs

        medusa_num_heads = config.medusa_num_heads
        medusa_num_layers = config.medusa_num_layers
        base_model_name_or_path = config._name_or_path
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.medusa = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        # The Medusa heads are dynamically added after base model loading to prevent
        # quantization bugs with `BitsAndBytes` and initialization crashes in `accelerate`.
        self.medusa_head = None
    def _add_medusa_heads(self, config):
        """Dynamically add the Medusa heads after from_pretrained."""
        if getattr(self, "medusa_head", None) is None:
            self.medusa_head = nn.ModuleList(
                [
                    nn.Sequential(
                        *([ResBlock(self.hidden_size)] * getattr(config, "medusa_num_layers", self.medusa_num_layers)),
                        nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                    )
                    for _ in range(getattr(config, "medusa_num_heads", self.medusa))
                ]
            )

    # Add a link named base_model to self
    @property
    def base_model(self):
        return self
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        try:
            if not kwargs.get("load_in_8bit", True):
                kwargs.pop("load_in_8bit", None)
            if not kwargs.get("load_in_4bit", True):
                kwargs.pop("load_in_4bit", None)
            
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            # [MODIFIED] If the config is incomplete (missing model_type), force fallback logic
            if not hasattr(config, "model_type") or config.model_type not in ["llama", "mistral"]:
                raise ValueError("Incomplete config")
            config = _ensure_medusa_fields(config, pretrained_model_name_or_path)
            model = super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
                config=config,
            )
            model._add_medusa_heads(config)
            medusa_head_path = os.path.join(pretrained_model_name_or_path, "medusa_lm_head.pt")
            if os.path.exists(medusa_head_path):
                filename = medusa_head_path
            else:
                return model
        except Exception:
            config = _load_medusa_config_fallback(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            base_model_config.medusa_num_heads = config.medusa_num_heads
            base_model_config.medusa_num_layers = config.medusa_num_layers
            base_model_config.base_model_name_or_path = config.base_model_name_or_path
            model = super().from_pretrained(
                config.base_model_name_or_path,
                *args,
                **kwargs,
                config=base_model_config,
            )
            model._add_medusa_heads(base_model_config)
            medusa_head_path = os.path.join(pretrained_model_name_or_path, "medusa_lm_head.pt")
            if os.path.exists(medusa_head_path):
                filename = medusa_head_path
            else:
                # Try cache first (works offline), then fall back to network download.
                try:
                    filename = hf_hub_download(
                        pretrained_model_name_or_path,
                        "medusa_lm_head.pt",
                        local_files_only=True,
                    )
                except Exception:
                    try:
                        filename = hf_hub_download(
                            pretrained_model_name_or_path, "medusa_lm_head.pt"
                        )
                    except Exception as e:
                        warnings.warn(
                            f"Could not load medusa_lm_head.pt ({e}). "
                            "Medusa heads will be randomly initialized — outputs will be garbage. "
                            "Ensure the file is cached or internet is available."
                        )
                        return model
        # Load to CPU first — safer with quantized models where model.device
        # may not resolve cleanly across all shards.
        medusa_head_state_dict = torch.load(filename, map_location="cpu", weights_only=True)
        try:
            model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False, assign=True)
        except TypeError:
            model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)
            
        # Remove accelerate hooks from medusa_head so it doesn't try to load from an offload index
        def remove_hooks(module):
            if hasattr(module, "_hf_hook"):
                delattr(module, "_hf_hook")
            if hasattr(module, "_old_forward"):
                module.forward = module._old_forward
                delattr(module, "_old_forward")
        model.medusa_head.apply(remove_hooks)
            
        # Move medusa_head to GPU in float16.
        # IMPORTANT: use explicit dtype=torch.float16, not just .to(device).
        # The base model emits bfloat16 activations; if we let .to(device)
        # inherit the model's dtype it would cast to bfloat16 and break the
        # fp16 checkpoint weights that were just loaded.
        model.medusa_head.to(device="cuda:0", dtype=torch.float16)
        
        return model
        

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
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
        """Forward pass of the MedusaModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
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
                # lm_head is fp16 (skipped from quantization); base model may
                # output bfloat16 — cast here to avoid dtype mismatch.
                orig = self.base_model.lm_head(
                    outputs[0].to(self.base_model.lm_head.weight.dtype)
                )
        # Clone the output hidden states
        hidden_states = outputs[0].clone()
        medusa_logits = []
        # TODO: Consider parallelizing this loop for efficiency?
        for i in range(self.medusa):
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

        # Filter: keep only paths whose depth <= medusa_num_heads.
        # vicuna_7b_stage2 assumes 5 heads; this model has self.medusa heads.
        # A path of length k uses head indices 0..k-1. If k > self.medusa,
        # the tree_indices for that path would exceed len(candidates), causing
        # a CUDA index-out-of-bounds crash.
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
        # The hyperparameters below are for the Medusa
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        medusa_choices=None,
        posterior_threshold=0.09,  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
        top_p=0.8, 
        sampling = 'typical', 
        fast = True
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            medusa_choices (list, optional): A list of integers indicating the number of choices for each Medusa head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
            top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
            sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
            fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache medusa buffers (the fixed patterns for tree attention)
        if medusa_choices is None:
            medusa_choices = self.get_medusa_choice(self.base_model_name_or_path)

        if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = self.medusa_buffers
        else:
            # Initialize the medusa buffer
            medusa_buffers = generate_medusa_buffers(
                medusa_choices, device=self.base_model.device
            )
        self.medusa_buffers = medusa_buffers
        self.medusa_choices = medusa_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
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
        # Initialize tree attention mask and process prefill tokens
        medusa_logits, logits = initialize_medusa(
            input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values
        )

        new_token = 0
        last_round_token = 0

        for idx in range(max_steps):
            # Generate candidates with topk predictions from Medusa heads
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

            # Use tree attention to verify the candidates and get predictions
            medusa_logits, logits, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
            )

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast
            )

            # Update the input_ids and logits
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


from transformers.models.llama.modeling_llama import LlamaForCausalLM as StdLlamaForCausalLM
class MedusaModelLlama(MedusaModelABC, KVLlamaForCausalLM):
    """Medusa model for Llama architecture.
    
    Inherits from KVLlamaForCausalLM (custom KV-cache model) which is
    required for the medusa_generate() tree-attention path.
    
    For standard (non-Medusa) autoregressive generation, use a manual
    forward loop WITHOUT use_cache=True, since the KV model's attention
    layers expect KVCache objects (not DynamicCache).
    """
    pass


class MedusaModelMistral(MedusaModelABC, KVMistralForCausalLM):
    pass

# In transformers v4.50+, GenerationMixin is no longer auto-included in
# PreTrainedModel.  Patch it in explicitly so .generate() is available.
try:
    from transformers import GenerationMixin as _GenMixin
    if not hasattr(MedusaModelLlama, "generate"):
        MedusaModelLlama.generate = _GenMixin.generate
    if not hasattr(MedusaModelMistral, "generate"):
        MedusaModelMistral.generate = _GenMixin.generate
except Exception:
    pass


class MedusaModel():
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the medusa_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        except Exception:
            # MEDUSA-v0.1 or legacy config without model_type
            config = _load_medusa_config_fallback(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            config.model_type = base_model_config.model_type

        if config.model_type == "llama":
            return MedusaModelLlama.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        elif config.model_type == "mistral":
            return MedusaModelMistral.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        else:
            raise ValueError("Only support llama and mistral for now!!")
