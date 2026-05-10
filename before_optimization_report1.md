# Medusa Project Summary: Baseline Architecture and State

## 1. Executive Summary
**Medusa** is an advanced inference acceleration framework designed to enhance the generation speed of Large Language Models (LLMs) like Llama and Mistral. Unlike standard speculative decoding which requires a secondary "draft model," Medusa adds multiple decoding heads to a single, frozen base model. These heads predict several future tokens in parallel, which are then validated in a single forward pass using a specialized tree-based attention mechanism.

This report summarizes the project's architecture, core components, and the functional baseline established after resolving critical environment-specific compatibility issues.

---

## 2. Core Technical Architecture

### 2.1 Multiple Decoding Heads
Medusa augments the base LLM with several "Medusa Heads." Each head is a lightweight residual block followed by a linear layer. 
- **Head $k$** is trained to predict the token at position $t+k+1$, given the hidden states at position $t$.
- During inference, these heads generate a "tree" of potential future tokens rather than a single sequence.

### 2.2 Tree-Based Attention
To validate multiple candidate paths simultaneously, Medusa employs a **Tree Attention** mechanism.
- It constructs a sparse tree of candidates.
- A specialized attention mask (the "Medusa Mask") ensures that tokens only attend to their actual ancestors in the speculative tree.
- This allows the model to verify multiple "speculative" paths in the time it would normally take to generate one token.

### 2.3 Persistent KV Cache
The framework implements a custom `KVCache` management system (`medusa/model/kv_cache.py`) that handles the non-linear growth and pruning of the key-value states required by tree-based decoding.

---

## 3. Key Project Components

| Component | Path | Description |
| :--- | :--- | :--- |
| **Model Logic** | `medusa/model/modeling_llama_kv.py` | Specialized Llama implementation with Medusa integration. |
| **Inference Engine** | `medusa/model/medusa_model.py` | Orchestrates the `medusa_generate` loop and head management. |
| **Utility Functions** | `medusa/model/utils.py` | Handles tree construction, candidate generation, and logit processing. |
| **Data Generation** | `data_generation/` | Tools for creating training data for the Medusa heads. |
| **Judge/Evaluation** | `llm_judge/` | A suite for benchmarking speed and quality against baselines. |

---

## 4. Baseline Fixes & Stability Improvements
During the initial setup in a modern environment (Google Colab / Transformers v4.40+), several critical issues were identified and resolved to establish a working baseline:

1. **Quantization Compatibility**: Fixed `AttributeErrors` in `KVLlamaAttention` where `num_heads` and `hidden_size` were missing during 8-bit loading.
2. **Library Version Adaptation**: Refactored `rotary_emb` access to support newer `transformers` versions where embeddings are stored at the model level rather than the layer level.
3. **Recursion Resolution**: Fixed an infinite recursion loop caused by circular PyTorch module references during model initialization.
4. **Precision Alignment**: Resolved `Half` vs `Float` type mismatches in the custom attention matrix multiplications.
5. **Masking Integrity**:
    - Fixed a shape mismatch in attention mask broadcasting during tree verification.
    - Restored **Causal Masking** during prompt initialization to prevent bidirectional attention from corrupting the KV cache.
6. **Device Orchestration**: Ensured Medusa heads are correctly moved to the GPU device during model loading to prevent `RuntimeError` device mismatches.

---

## 5. Current Performance Profile
Based on initial benchmarks (Llama-2 7B in 8-bit mode):
- **Standard Autoregressive Speed**: ~0.80 tokens/sec (Baseline)
- **Medusa Speculative Speed**: ~1.06 tokens/sec (~1.32x Speedup)
- **Memory Usage**: ~6.0 GB VRAM in 8-bit mode.

*Note: Performance varies significantly based on the chosen 'Medusa Tree' (choices) and the complexity of the prompt.*

---
**Report Generated on:** 2026-05-10
**Project State:** Functional Baseline Established.