import torch
from medusa.model.medusa_model import MedusaModel
from transformers import BitsAndBytesConfig

def check_weights():
    model_name = "FasterDecoding/medusa-vicuna-7b-v1.3"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    model = MedusaModel.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0}
    )
    
    # Check a few weights in the base model
    first_layer_weight = model.model.layers[0].self_attn.q_proj.weight
    print(f"Layer 0 Q_proj weight sum: {first_layer_weight.float().abs().sum().item()}")
    
    lm_head_weight = model.lm_head.weight
    print(f"LM Head weight sum: {lm_head_weight.float().abs().sum().item()}")
    
    # Standard random weights have a sum of ~N * 0.02
    # For lm_head (32000 * 4096), sum should be ~2.6M
    
if __name__ == "__main__":
    check_weights()
