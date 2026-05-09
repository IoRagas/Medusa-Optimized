import torch
from medusa.model.medusa_model import MedusaModel
from transformers import AutoConfig, AutoTokenizer

def check_vocab():
    model_name = "FasterDecoding/medusa-vicuna-7b-v1.3"
    config = AutoConfig.from_pretrained(model_name)
    print(f"Model Config Vocab Size: {config.vocab_size}")
    
    # Load model (use 4-bit to be fast)
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True)
    
    model = MedusaModel.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map={"": 0},
    )
    
    tokenizer = model.get_tokenizer()
    print(f"Tokenizer Vocab Size: {len(tokenizer)}")
    
    print(f"lm_head weight shape: {model.base_model.lm_head.weight.shape}")
    print(f"embed_tokens weight shape: {model.base_model.model.embed_tokens.weight.shape}")
    
    for i, head in enumerate(model.medusa_head):
        print(f"Medusa Head {i} final layer shape: {head[-1].weight.shape}")
        
    print(f"Number of Medusa Heads: {len(model.medusa_head)}")

if __name__ == "__main__":
    check_vocab()
