import torch
import torch.nn as nn
from transformers import AutoConfig
import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from medusa.model.modeling_llama_kv import LlamaForCausalLM

def test_forward_stability():
    print("Testing forward pass stability (upcasts)...")
    
    # Create a tiny config
    config = AutoConfig.from_pretrained("lmsys/vicuna-7b-v1.3")
    config.num_hidden_layers = 1
    config.hidden_size = 128
    config.intermediate_size = 256
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    
    # Initialize the model on CPU
    model = LlamaForCausalLM(config).half() # Use float16 to test upcasts
    print("Model initialized in fp16.")
    
    # Create inputs
    input_ids = torch.randint(0, 32000, (1, 10))
    
    # Run forward pass
    try:
        with torch.no_grad():
            outputs = model(input_ids)
        print("SUCCESS: Forward pass completed without errors.")
        print(f"Logits shape: {outputs.logits.shape}")
        
        if torch.isnan(outputs.logits).any():
            print("FAILURE: NaNs detected in output!")
            return False
            
    except Exception as e:
        print(f"FAILURE: Forward pass crashed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    if test_forward_stability():
        print("\nStability test PASSED!")
    else:
        sys.exit(1)
