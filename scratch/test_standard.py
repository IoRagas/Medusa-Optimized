import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def test_standard():
    model_name = "lmsys/vicuna-7b-v1.3"
    print(f"Loading {model_name} in 4-bit...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    prompt = "A chat between a curious user and an artificial intelligence assistant. USER: Explain the importance of healthy eating. ASSISTANT:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    
    print("\n--- Testing Standard Transformers Generation ---")
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=20)
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Output: {output_text.encode('ascii', 'ignore').decode()}")

if __name__ == "__main__":
    test_standard()
