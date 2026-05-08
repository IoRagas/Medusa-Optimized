"""Debug script to capture the real error during model loading."""
import traceback
import sys
import os
import torch

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"VRAM free: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")

import psutil
print(f"System RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
print(f"System RAM free: {psutil.virtual_memory().available / 1024**3:.1f} GB")
print()

try:
    from medusa.model.medusa_model import MedusaModel
    from transformers import BitsAndBytesConfig

    print("=== Attempting to load model with 4-bit quantization (NF4) ===")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,       # saves ~0.4 GB extra
        llm_int8_skip_modules=["medusa_head", "lm_head"],  # keep these in fp16
    )
    
    model = MedusaModel.from_pretrained(
        "FasterDecoding/medusa-vicuna-7b-v1.3",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": 0},  # 4-bit blocks CPU dispatch — force all layers to GPU
        offload_folder=os.path.abspath("offload"),
        quantization_config=quantization_config,
    )
    print("SUCCESS! Model loaded.")
    print(f"Model device: {model.base_model.device}")
    if torch.cuda.is_available():
        used  = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.mem_get_info()[0]) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM used after load: {used:.2f} / {total:.1f} GB")

except Exception as e:
    print(f"\n{'='*60}")
    print(f"ERROR: {type(e).__name__}: {e}")
    print(f"{'='*60}")
    traceback.print_exc()
    print(f"\nFull traceback above ^")
