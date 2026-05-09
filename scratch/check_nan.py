import torch
from medusa.model.medusa_model import MedusaModel
from transformers import BitsAndBytesConfig

def check_model():
    model_name = "FasterDecoding/medusa-vicuna-7b-v1.3"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["medusa_head", "lm_head"],
    )
    print("Loading model...")
    model = MedusaModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        device_map={"": 0}, 
        quantization_config=bnb,
    )
    
    print(f"lm_head weights NaN: {torch.isnan(model.base_model.lm_head.weight).any().item()}")
    print(f"lm_head weights Sum: {model.base_model.lm_head.weight.sum().item():.2f}")
    
    for i, head in enumerate(model.medusa_head):
        # head is Sequential(ResBlock, Linear)
        res_linear = head[0].linear
        final_linear = head[1]
        print(f"Medusa Head {i} | Res Linear NaN: {torch.isnan(res_linear.weight).any().item()} | Final Linear NaN: {torch.isnan(final_linear.weight).any().item()}")

if __name__ == "__main__":
    check_model()
