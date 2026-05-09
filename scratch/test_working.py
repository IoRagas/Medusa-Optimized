import torch
from medusa.model.medusa_model import MedusaModel
from transformers import BitsAndBytesConfig

def test_working():
    model_name = "FasterDecoding/medusa-vicuna-7b-v1.3"
    print(f"Loading {model_name} in 4-bit for fast verification...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    try:
        model = MedusaModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": 0}
        )
        tokenizer = model.get_tokenizer()
        
        # VICUNA PROMPT FORMAT
        prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Explain the importance of healthy eating. ASSISTANT:"
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        
        print("\n--- Testing Standard Generation ---")
        with torch.no_grad():
            cur_ids = input_ids
            for _ in range(50):
                outputs = model(input_ids=cur_ids, medusa_forward=False)
                next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
                cur_ids = torch.cat([cur_ids, next_token_id], dim=-1)
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
        
        output_text = tokenizer.decode(cur_ids[0], skip_special_tokens=True)
        print(f"Output: {output_text.encode('ascii', 'ignore').decode()}")
        
    except Exception as e:
        print(f"FAILED: {str(e)}")

if __name__ == "__main__":
    test_working()
