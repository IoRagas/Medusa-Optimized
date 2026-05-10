import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    apply_rotary_pos_emb,
    repeat_kv,
)

logger = logging.get_logger(__name__)

class KVLlamaAttention(LlamaAttention):
    def __init__(self, config, layer_idx: Optional[int] = None, model=None):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # Use object.__setattr__ to avoid PyTorch's automatic submodule registration
        # which would cause infinite recursion (Model -> Layer -> Attention -> Model)
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

        # ── Medusa Specific Attention ──────────────────────────────────
        # This part is only executed during Medusa speculative decoding tree verification
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # [MODIFIED] Handle position embeddings for different transformers versions (4.40+)
        kv_seq_len = past_key_value[0].shape[-2] + key_states.shape[-2]
        
        position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # If position_embeddings are passed, they are usually already indexed
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            # Fallback for manual calculation
            try:
                # Newer versions: rotary_emb is often in the model, not the layer
                rotary_emb = getattr(self, "rotary_emb", None)
                if rotary_emb is None and hasattr(self, "v_model") and self.v_model is not None:
                    rotary_emb = getattr(self.v_model, "rotary_emb", None)
                
                if rotary_emb is not None:
                    # Newer versions: rotary_emb(x, position_ids)
                    cos, sin = rotary_emb(value_states, position_ids)
                else:
                    # Very old versions or fallback
                    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            except (TypeError, AttributeError):
                # Older versions: rotary_emb(x, seq_len)
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Update KV cache
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

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_output = torch.matmul(attn_weights, value_states.to(torch.float32)).to(query_states.dtype)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

class KVLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx: int, model=None):
        super().__init__(config, layer_idx)
        self.self_attn = KVLlamaAttention(config, layer_idx, model=model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        is_medusa_cache = past_key_value is not None and hasattr(past_key_value[0], "current_length")

        if not is_medusa_cache:
            # Map past_key_value to past_key_values for newer transformers
            kwargs["past_key_values"] = past_key_value
            # Remove kwargs that are not expected by older versions
            kwargs.pop("past_key_value", None)
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + attn_outputs[0]

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        if use_cache:
            outputs += (attn_outputs[2],)

        return outputs

class KVLlamaModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([KVLlamaDecoderLayer(config, i, model=self) for i in range(config.num_hidden_layers)])
        self.medusa_mask = None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        **kwargs,
    ):
        # [MODIFIED] Inject the medusa_mask if not provided and pad it for past_key_values
        if attention_mask is None and self.medusa_mask is not None:
            medusa_mask = self.medusa_mask
            past_length = past_key_values[0][0].current_length.item() if past_key_values is not None and hasattr(past_key_values[0][0], "current_length") else 0
            if past_length > 0:
                padding = torch.zeros(
                    (medusa_mask.size(0), medusa_mask.size(1), medusa_mask.size(2), past_length),
                    device=medusa_mask.device, dtype=medusa_mask.dtype
                )
                attention_mask = torch.cat([padding, medusa_mask], dim=-1)
            else:
                attention_mask = medusa_mask

        # [MODIFIED] If we're not using the special Medusa KVCache list, 
        # use the standard LlamaModel.forward to ensure compatibility with 
        # newer transformers features (like position_embeddings, cache_position, etc.)
        if past_key_values is None or not isinstance(past_key_values, (list, tuple)):
            return super().forward(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                past_key_values=past_key_values, 
                **kwargs
            )

        # ── Custom Medusa Forward ──────────────────────────────────────
        # This part handles the persistent KV cache used during speculative decoding.
        
        # Extract metadata from kwargs or model config using pop to avoid duplicate argument errors
        output_attentions = kwargs.pop("output_attentions", self.config.output_attentions)
        output_hidden_states = kwargs.pop("output_hidden_states", self.config.output_hidden_states)
        use_cache = kwargs.pop("use_cache", self.config.use_cache)
        return_dict = kwargs.pop("return_dict", self.config.use_return_dict)
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        position_ids = kwargs.pop("position_ids", None)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # [MODIFIED] Generate position_ids if missing in the Medusa path
        if position_ids is None:
            # We assume past_key_values[0][0] is a KVCache object with a current_length
            # Based on kv_cache.py, it has a current_length attribute (tensor)
            past_key_values_length = past_key_values[0][0].current_length.item() if past_key_values is not None else 0
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, 
                dtype=torch.long, device=inputs_embeds.device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            past_key_values_length = past_key_values[0][0].current_length.item() if past_key_values is not None else 0

        # [MODIFIED] Generate causal mask for the prompt
        if attention_mask is None and seq_length > 1:
            attention_mask = torch.full((seq_length, seq_length), torch.finfo(inputs_embeds.dtype).min, device=inputs_embeds.device)
            attention_mask.triu_(diagonal=1)
            attention_mask = attention_mask[None, None, :, :]
            if past_key_values_length > 0:
                padding = torch.zeros((1, 1, seq_length, past_key_values_length), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                attention_mask = torch.cat([padding, attention_mask], dim=-1)

        hidden_states = inputs_embeds

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx]

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs, # Pass along any extra args like position_embeddings
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class KVLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = KVLlamaModel(config)

    def forward(self, input_ids=None, **kwargs):
        # We MUST handle the lm_head upcasting here if we want standard forward to work
        # but LlamaForCausalLM.forward is complex.
        # Instead, we'll patch the lm_head's forward method directly in MedusaModelLlama.
        return super().forward(input_ids=input_ids, **kwargs)
