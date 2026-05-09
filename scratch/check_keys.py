import torch
import os

def check_keys():
    # Find the downloaded model file
    # Usually in ~/.cache/huggingface/hub/...
    # But we can try to download it specifically
    from huggingface_hub import hf_hub_download
    filename = hf_hub_download("FasterDecoding/medusa-vicuna-7b-v1.3", "medusa_lm_head.pt")
    state_dict = torch.load(filename, map_location="cpu", weights_only=True)
    print(f"State Dict Keys: {list(state_dict.keys())[:20]}")
    print(f"Total Keys: {len(state_dict)}")
    
    # Check head indices
    indices = set()
    for key in state_dict.keys():
        if key.startswith("medusa_head."):
            idx = key.split(".")[1]
            indices.add(idx)
    print(f"Detected Head Indices: {sorted(list(indices))}")
    print(f"Derived medusa_num_heads: {len(indices)}")

if __name__ == "__main__":
    check_keys()
