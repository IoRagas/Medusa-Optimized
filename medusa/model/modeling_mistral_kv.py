import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralMLP,
    MistralRMSNorm,
    MistralDecoderLayer,
    MistralModel,
    MistralForCausalLM,
    apply_rotary_pos_emb,
    repeat_kv,
)

logger = logging.get_logger(__name__)

class KVMistralAttention(MistralAttention):
    def __init__(self, config, layer_idx: Optional[int] = None, model=None):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # Use object.__setattr__ to avoid PyTorch's automatic submodule registration
        object.__setattr__(self, "v_model", model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # [MODIFIED] Check if we're using the Medusa persistent KV cache
        is_medusa_cache = past_key_value is not None and hasattr(past_key_value[0], "current_length")

        if not is_medusa_cache:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = past_key_value[0].shape[-2] + key_states.shape[-2]
        
        # [MODIFIED] Handle missing rotary_emb in layer
        rotary_emb = getattr(self, "rotary_emb", None)
        if rotary_emb is None and hasattr(self, "v_model") and self.v_model is not None:
            rotary_emb = getattr(self.v_model, "rotary_emb", None)
        
        if rotary_emb is not None:
            cos, sin = rotary_emb(value_states, seq_len=kv_seq_len)
        else:
            # Fallback for older versions
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = past_key_value[0].cat(key_states, dim=2)
        value_states = past_key_value[1].cat(value_states, dim=2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        scaling = getattr(self, "scaling", math.sqrt(self.head_dim))
        if isinstance(scaling, float):
            scaling = 1.0 / scaling if scaling != 0 else math.sqrt(self.head_dim)

        attn_weights = torch.matmul(
            query_states.to(torch.float32), 
            key_states.transpose(2, 3).to(torch.float32)
        ) / scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask.to(torch.float32)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states.to(torch.float32)).to(query_states.dtype)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

class KVMistralDecoderLayer(MistralDecoderLayer):
    def __init__(self, config, layer_idx: int, model=None):
        super().__init__(config, layer_idx)
        self.self_attn = KVMistralAttention(config, layer_idx, model=model)

class KVMistralModel(MistralModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([KVMistralDecoderLayer(config, i, model=self) for i in range(config.num_hidden_layers)])

class KVMistralForCausalLM(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = KVMistralModel(config)