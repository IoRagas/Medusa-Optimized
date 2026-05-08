import torch
import torch.nn as nn
from transformers import AutoConfig
import sys
import os

# Add the project root to sys.path to import our custom modules
sys.path.append(os.getcwd())

# Mock the Llama model structure slightly for testing
from medusa.model.modeling_llama_kv import LlamaForCausalLM

def test_init_logic():
    print("Testing weight initialization safety logic...")
    
    # Create a tiny config
    config = AutoConfig.from_pretrained("lmsys/vicuna-7b-v1.3")
    config.num_hidden_layers = 1
    config.hidden_size = 128
    config.intermediate_size = 256
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    
    # Initialize the model
    model = LlamaForCausalLM(config)
    print("Model initialized successfully on CPU.")
    
    # Simulate 8-bit quantization on one layer
    # In real bitsandbytes, the weight becomes an Int8Params or a Char tensor
    target_layer = model.model.layers[0].self_attn.q_proj
    target_layer.weight.requires_grad = False
    target_layer.weight.data = torch.randint(-128, 127, target_layer.weight.shape, dtype=torch.int8)
    
    print(f"Simulated quantization on q_proj. Weight dtype: {target_layer.weight.dtype}")
    
    # Now manually call _init_weights to see if it crashes
    try:
        model._init_weights(target_layer)
        print("SUCCESS: _init_weights bypassed the int8 layer without crashing.")
    except Exception as e:
        print(f"FAILURE: _init_weights crashed on int8 layer: {e}")
        return False
        
    # Check a standard float layer
    float_layer = model.model.layers[0].self_attn.k_proj
    print(f"Testing float layer. Weight dtype: {float_layer.weight.dtype}")
    try:
        model._init_weights(float_layer)
        print("SUCCESS: _init_weights worked normally on floating point layer.")
    except Exception as e:
        print(f"FAILURE: _init_weights crashed on float layer: {e}")
        return False
        
    return True

if __name__ == "__main__":
    if test_init_logic():
        print("\nAll initialization tests PASSED!")
    else:
        sys.exit(1)
