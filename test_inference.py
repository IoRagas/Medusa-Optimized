"""Diagnostic: check prompt format and actual generated tokens."""
import os, torch
from transformers import BitsAndBytesConfig
from medusa.model.medusa_model import MedusaModel
from fastchat.model.model_adapter import get_conversation_template

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["medusa_head", "lm_head"],
)
print("Loading model...")
model = MedusaModel.from_pretrained(
    "FasterDecoding/medusa-vicuna-7b-v1.3",
    torch_dtype=torch.float16, low_cpu_mem_usage=True,
    device_map={"": 0}, quantization_config=bnb,
)
tokenizer = model.get_tokenizer()
print(f"Head[0] dtype: {model.medusa_head[0][1].weight.dtype}")
print(f"Head[0] weight sum: {model.medusa_head[0][1].weight.sum().item():.2f}")

# Check the conversation template
conv = get_conversation_template("FasterDecoding/medusa-vicuna-7b-v1.3")
print(f"\nConversation template name: {conv.name}")
conv.append_message(conv.roles[0], "hi")
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
print(f"\nPrompt (repr): {repr(prompt[-200:])}")  # last 200 chars

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
print(f"Prompt tokens: {input_ids.shape[1]}")
print(f"Last 5 tokens: {input_ids[0, -5:].tolist()}")
print(f"EOS id: {tokenizer.eos_token_id}")

# Check first predicted token
print("\n--- First 5 generation steps ---")
for i, chunk in enumerate(model.medusa_generate(input_ids, temperature=0.0, max_steps=5)):
    print(f"Step {i}: text={repr(chunk['text'])}")
print("Done.")
