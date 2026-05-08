"""
Medusa Speculative Decoding Benchmark
======================================
Compares standard autoregressive generation vs Medusa parallel generation.

Usage (Google Colab / any GPU):
    python benchmark_medusa.py                  # fp16 (default, needs ~14GB VRAM — T4/V100/A100)
    python benchmark_medusa.py --load-in-8bit   # 8-bit  (needs ~8GB VRAM + bitsandbytes)
    python benchmark_medusa.py --load-in-4bit   # 4-bit  (needs ~5GB VRAM + bitsandbytes)
"""
import argparse
import torch
import time
import os
from fastchat.model.model_adapter import get_conversation_template
from medusa.model.medusa_model import MedusaModel


def run_benchmark(args):
    model_name = "FasterDecoding/medusa-vicuna-7b-v1.3"
    
    # Ensure offload folder exists for 8-bit loading stability
    os.makedirs("offload", exist_ok=True)
    
    # ── Determine precision mode ──────────────────────────────────────
    if args.load_in_4bit:
        mode = "4-bit"
    elif args.load_in_8bit:
        mode = "8-bit"
    else:
        mode = "fp16"
    
    print(f"Loading {model_name} in {mode} mode...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    print("This will download ~14GB of weights if not already cached.\n")

    # ── Build loading kwargs ──────────────────────────────────────────
    load_kwargs = dict(
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        offload_folder=os.path.abspath("offload"),
    )

    if args.load_in_4bit or args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        skip_modules = ["medusa_head", "lm_head"]
        if args.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=skip_modules,
            )
        else:  # 8-bit
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=skip_modules,
            )
        load_kwargs["device_map"] = {"": 0}
    else:
        # Pure fp16 — force everything onto GPU 0.
        load_kwargs["device_map"] = {"": 0}

    # ── Load model ────────────────────────────────────────────────────
    try:
        model = MedusaModel.from_pretrained(model_name, **load_kwargs)
        print("\nModel loaded successfully!")
    except Exception as e:
        print(f"\nFailed to load model. Error: {e}")
        return

    tokenizer = model.get_tokenizer()
    repetition_penalty = 1.2
    device = next(model.parameters()).device

    if torch.cuda.is_available():
        used = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.mem_get_info()[0]) / 1024**3
        print(f"VRAM used after load: {used:.2f} GB")

    conv = get_conversation_template(model_name)
    conv.append_message(conv.roles[0], "Explain the theory of parallel computing and why it is important for modern systems.")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    max_new = args.max_new_tokens

    # ══════════════════════════════════════════════════════════════════
    # TEST 1: Standard Autoregressive Generation
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 1: Standard Autoregressive Generation")
    print("=" * 60)

    # Manual autoregressive loop (no KV-cache).
    # NOTE: The KV-modified LlamaModel expects KVCache objects for caching,
    # not the standard DynamicCache tuples, so we run without caching here.
    # This is the fair "baseline" — Medusa's speedup is measured against it.
    start_time = time.time()
    generated_ids = []
    with torch.inference_mode():
        cur_ids = input_ids.clone()
        for _ in range(max_new):
            outputs = model(
                input_ids=cur_ids,
                use_cache=False,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :].float()
            
            # Apply repetition penalty
            for token_id in set(generated_ids + input_ids[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

            next_id = logits.argmax(dim=-1)
            tok = next_id.item()
            if tok == tokenizer.eos_token_id:
                break
            generated_ids.append(tok)
            cur_ids = torch.cat([cur_ids, next_id.unsqueeze(0)], dim=-1)

    standard_time = time.time() - start_time

    standard_tokens = len(generated_ids)
    standard_tps = standard_tokens / standard_time if standard_time > 0 else 0

    print(tokenizer.decode(generated_ids, skip_special_tokens=True))
    print(f"\n[Metrics] Generated {standard_tokens} tokens in {standard_time:.2f}s")
    print(f"[Metrics] Standard Speed: {standard_tps:.2f} tokens/second")

    # ══════════════════════════════════════════════════════════════════
    # TEST 2: Medusa Speculative Generation
    # ══════════════════════════════════════════════════════════════════
    # Free GPU memory from the standard test before allocating the KV cache
    try:
        del cur_ids, outputs
    except NameError:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("TEST 2: Medusa Speculative Generation")
    print("=" * 60)

    start_time = time.time()
    medusa_text = ""
    with torch.inference_mode():
        for chunk in model.medusa_generate(
            input_ids,
            temperature=0.0,
            max_steps=max_new,
        ):
            # medusa_generate yields {"text": "full string so far"}
            medusa_text = chunk["text"]
    medusa_time = time.time() - start_time

    # Count tokens from decoded text
    medusa_token_ids = tokenizer.encode(medusa_text, add_special_tokens=False)
    medusa_tokens = len(medusa_token_ids)
    medusa_tps = medusa_tokens / medusa_time if medusa_time > 0 else 0

    print(medusa_text)
    print(f"\n[Metrics] Generated {medusa_tokens} tokens in {medusa_time:.2f}s")
    print(f"[Metrics] Medusa Speed: {medusa_tps:.2f} tokens/second")

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    if standard_tps > 0:
        speedup = medusa_tps / standard_tps
        print(f"SPEEDUP: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} with Medusa!")
    else:
        print("Could not compute speedup (standard generation produced 0 tokens).")
    print(f"Mode: {mode}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medusa speculative decoding benchmark")
    parser.add_argument("--load-in-8bit", action="store_true", help="Use 8-bit quantization (needs bitsandbytes)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit quantization (needs bitsandbytes)")
    parser.add_argument("--max-new-tokens", type=int, default=150, help="Maximum number of new tokens to generate")
    args = parser.parse_args()
    run_benchmark(args)
