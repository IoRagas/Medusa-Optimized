# Adapted from: https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/cli.py
"""
Chat with a model with command line interface.

Usage:
python3 -m medusa.inference.cli --model <model_name_or_path>
Other commands:
- Type "!!exit" or an empty line to exit.
- Type "!!reset" to start a new conversation.
- Type "!!remove" to remove the last prompt.
- Type "!!regen" to regenerate the last message.
- Type "!!save <filename>" to save the conversation history to a json file.
- Type "!!load <filename>" to load a conversation history from a json file.
"""
import argparse
import os
import re
import sys
import torch
from fastchat.serve.cli import SimpleChatIO, RichChatIO, ProgrammaticChatIO
from fastchat.model.model_adapter import get_conversation_template
from fastchat.conversation import get_conv_template
import json
from medusa.model.medusa_model import MedusaModel
# Standard (non-KV) LlamaForCausalLM forward — used for simple autoregressive
# generation to avoid the custom KVCache format that corrupts logits.
import transformers.models.llama.modeling_llama as _llama_module
_StdLlamaForCausalLM = _llama_module.LlamaForCausalLM


def main(args):
    if args.style == "simple":
        chatio = SimpleChatIO(args.multiline)
    elif args.style == "rich":
        chatio = RichChatIO(args.multiline, args.mouse)
    elif args.style == "programmatic":
        chatio = ProgrammaticChatIO()
    else:
        raise ValueError(f"Invalid style for console: {args.style}")
    try:
        if args.load_in_8bit or args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            # medusa_head and lm_head must NOT be quantized:
            # Their weights are loaded as plain fp16 AFTER model init, so
            # landing in LinearFP4 would corrupt the quant state.
            skip_modules = ["medusa_head", "lm_head"]
            if args.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    llm_int8_skip_modules=skip_modules,
                )
                # 4-bit blocks ALL CPU dispatch — force every layer onto GPU.
                # Quantized 7B ≈ 3.5 GB + fp16 heads ≈ 400 MB fits in 6 GB.
                device_map = {"": 0}
            else:  # 8-bit
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    llm_int8_skip_modules=skip_modules,
                )
                device_map = "auto"
            model = MedusaModel.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map=device_map,
                offload_folder=os.path.abspath("offload"),
                quantization_config=quantization_config,
            )
        else:
            model = MedusaModel.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                offload_folder=os.path.abspath("offload"),
            )
        tokenizer = model.get_tokenizer()
        conv = None

        def new_chat():
            return get_conversation_template(args.model)

        def reload_conv(conv):
            """
            Reprints the conversation from the start.
            """
            for message in conv.messages[conv.offset :]:
                chatio.prompt_for_output(message[0])
                chatio.print_output(message[1])

        while True:
            if not conv:
                conv = new_chat()

            try:
                inp = chatio.prompt_for_input(conv.roles[0])
            except EOFError:
                inp = ""

            if inp == "!!exit" or not inp:
                print("exit...")
                break
            elif inp == "!!reset":
                print("resetting...")
                conv = new_chat()
                continue
            elif inp == "!!remove":
                print("removing last message...")
                if len(conv.messages) > conv.offset:
                    # Assistant
                    if conv.messages[-1][0] == conv.roles[1]:
                        conv.messages.pop()
                    # User
                    if conv.messages[-1][0] == conv.roles[0]:
                        conv.messages.pop()
                    reload_conv(conv)
                else:
                    print("No messages to remove.")
                continue
            elif inp == "!!regen":
                print("regenerating last message...")
                if len(conv.messages) > conv.offset:
                    # Assistant
                    if conv.messages[-1][0] == conv.roles[1]:
                        conv.messages.pop()
                    # User
                    if conv.messages[-1][0] == conv.roles[0]:
                        reload_conv(conv)
                        # Set inp to previous message
                        inp = conv.messages.pop()[1]
                    else:
                        # Shouldn't happen in normal circumstances
                        print("No user message to regenerate from.")
                        continue
                else:
                    print("No messages to regenerate.")
                    continue
            elif inp.startswith("!!save"):
                args = inp.split(" ", 1)

                if len(args) != 2:
                    print("usage: !!save <filename>")
                    continue
                else:
                    filename = args[1]

                # Add .json if extension not present
                if not "." in filename:
                    filename += ".json"

                print("saving...", filename)
                with open(filename, "w") as outfile:
                    json.dump(conv.dict(), outfile)
                continue
            elif inp.startswith("!!load"):
                args = inp.split(" ", 1)

                if len(args) != 2:
                    print("usage: !!load <filename>")
                    continue
                else:
                    filename = args[1]

                # Check if file exists and add .json if needed
                if not os.path.exists(filename):
                    if (not filename.endswith(".json")) and os.path.exists(
                        filename + ".json"
                    ):
                        filename += ".json"
                    else:
                        print("file not found:", filename)
                        continue

                print("loading...", filename)
                with open(filename, "r") as infile:
                    new_conv = json.load(infile)

                conv = get_conv_template(new_conv["template_name"])
                conv.set_system_message(new_conv["system_message"])
                conv.messages = new_conv["messages"]
                reload_conv(conv)
                continue

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            try:
                chatio.prompt_for_output(conv.roles[1])
                # Resolve device directly — model.base_model.device can fail on
                # quantized models because base_model is a property returning self.
                device = next(model.parameters()).device
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

                # Standard autoregressive generation loop.
                # We bypass model.generate() entirely because:
                # 1. MedusaModelLlama's custom forward() has extra kwargs
                #    (medusa_forward, output_orig) that confuse GenerationMixin.
                # 2. The KV-cache model expects KVCache objects (not DynamicCache),
                #    so we run without caching and grow the input each step.
                def _standard_stream(input_ids, max_new_tokens=512, temperature=0.7):
                    cur_ids = input_ids.clone()
                    eos_id  = tokenizer.eos_token_id
                    generated = []

                    with torch.inference_mode():
                        for _ in range(max_new_tokens):
                            outputs = model(
                                input_ids=cur_ids,
                                use_cache=False,
                                return_dict=True,
                            )
                            logits = outputs.logits[:, -1, :].float()

                            if temperature <= 0:
                                next_id = logits.argmax(dim=-1)          # greedy
                            else:
                                probs = torch.softmax(logits / temperature, dim=-1)
                                # Guard: replace NaN/0 probs with uniform so
                                # multinomial never gets an all-zero distribution.
                                if not probs.isfinite().all() or probs.sum() == 0:
                                    probs = torch.ones_like(probs) / probs.shape[-1]
                                next_id = torch.multinomial(probs, 1).squeeze(-1)

                            tok = next_id.item()
                            if tok == eos_id:
                                break
                            generated.append(tok)
                            cur_ids = torch.cat([cur_ids, next_id.unsqueeze(0)], dim=-1)

                            # Stream incrementally: decode everything so far.
                            partial = tokenizer.decode(
                                generated,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False,
                            )
                            yield {"text": partial}

                outputs = chatio.stream_output(
                    _standard_stream(
                        input_ids,
                        max_new_tokens=args.max_steps,
                        temperature=args.temperature,
                    )
                )
                conv.update_last_message(outputs.strip())

            except KeyboardInterrupt:
                print("stopped generation.")
                # If generation didn't finish
                if conv.messages[-1][1] is None:
                    conv.messages.pop()
                    # Remove last user message, so there isn't a double up
                    if conv.messages[-1][0] == conv.roles[0]:
                        conv.messages.pop()

                    reload_conv(conv)

    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path.")
    parser.add_argument(
        "--load-in-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true", help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--conv-system-msg", type=str, default=None, help="Conversation system message."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich", "programmatic"],
        help="Display style.",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="[Rich Style]: Enable mouse support for cursor positioning.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )
    args = parser.parse_args()
    main(args)
