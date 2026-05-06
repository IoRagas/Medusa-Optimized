# Medusa Optimization Plan

## Executive Summary
This document outlines GPU and runtime optimizations for the Medusa inference pipeline. The current implementation is already GPU-resident but has kernel launch overhead and sequential bottlenecks that can be reduced.

---

## Optimization Opportunities (Priority Order)

### Priority 1: High Impact, Straightforward

#### 1. **Fused Medusa Head Computation**
**Current:** Loop over N heads sequentially
```python
for i in range(self.medusa):
    medusa_logits.append(self.medusa_head[i](hidden_states))
```
**Problem:** N separate forward passes = N kernel launches + memory pressure
**Solution:** 
- Batch heads into a single operation or fuse with a custom CUDA kernel
- Expected speedup: 15-25% on generation loop
**File:** `medusa/model/medusa_model.py` (lines 200-210)
**Effort:** Medium (requires batching or custom kernel)

---

#### 2. **Direct GPU Buffer Initialization**
**Current:** Create buffers on CPU, then `.to(device)` at the end
```python
medusa_attn_mask = torch.eye(medusa_len, medusa_len)  # CPU
# ... later ...
medusa_buffers = { k: v.clone().to(device) ... }  # CPU→GPU transfer
```
**Problem:** Unnecessary CPU allocation and PCI-E bandwidth (one-time, but sloppy)
**Solution:** Build buffers directly on GPU device from start
**File:** `medusa/model/utils.py` (lines 32-125, especially 118-123)
**Effort:** Low (refactor to use `device=device` in tensor creation)
**Expected speedup:** ~1-5ms on model load (one-time)

---

#### 3. **Fused Indexing Operations in `generate_candidates()`**
**Current:** Three separate indexing steps
```python
tree_candidates = candidates[tree_indices]  # GPU indexing
tree_candidates_ext = torch.cat([tree_candidates, ...])  # Separate op
cart_candidates = tree_candidates_ext[retrieve_indices]  # Another indexing
```
**Problem:** Multiple small GPU kernel launches
**Solution:** Combine into single fused kernel or pre-compute index mappings
**File:** `medusa/model/utils.py` (lines 275-305)
**Effort:** Medium (fused kernel; custom CUDA or TorchScript)
**Expected speedup:** 5-10% on candidate generation

---

#### 4. **Softmax + Sampling Fusion**
**Current:** Separate softmax, then multinomial sampling
```python
probs = torch.softmax(logit / temperature, dim=-1)
sampled_tokens = torch.multinomial(probs, 1)
```
**Problem:** Two kernel launches instead of one
**Solution:** Use fused implementations or custom kernel
**File:** `medusa/model/utils.py` (lines 240-280, `get_typical_one_token()` and `get_nucleus_one_token()`)
**Effort:** Low-Medium (can use `torch._C._fused_dropout` style pattern)
**Expected speedup:** 5-8% on sampling operations

---

### Priority 2: Medium Impact, Moderate Effort

#### 5. **Asynchronous GPU Operations**
**Current:** Synchronous CUDA operations block until completion
**Problem:** Missed opportunity for overlapping CPU and GPU work
**Solution:** 
- Use `torch.cuda.stream()` for overlapping KV cache updates with forward passes
- Async H2D transfers where possible
**File:** `medusa/model/utils.py`, `medusa/model/medusa_model.py`
**Effort:** Medium (requires careful synchronization)
**Expected speedup:** 10-15% (depends on CPU-GPU balance)

---

#### 6. **KV Cache Layout Optimization**
**Current:** Pre-allocated dense GPU memory in `initialize_past_key_values()`
**Problem:** May have poor memory layout for access patterns (row-major vs. column-major)
**Solution:** 
- Profile cache access patterns
- Consider using page-table style memory layout or sparse patterns
- Optimize stride/shape for flash attention kernels
**File:** `medusa/model/kv_cache.py` (lines 5-40)
**Effort:** High (requires profiling and CUDA knowledge)
**Expected speedup:** 10-20% (model-dependent)

---

#### 7. **Loop Fusion in `tree_decoding()` Main Generation Loop**
**Current:** Per-iteration: candidates → tree_decoding → posterior eval → input update
**Problem:** Four separate function calls with data deps create stalls
**Solution:** Fuse into single kernel or reduce Python overhead
**File:** `medusa/model/medusa_model.py` (lines 285-370)
**Effort:** High (requires substantial refactoring)
**Expected speedup:** 15-25% (if done well)

---

### Priority 3: Lower Impact, Future Work

#### 8. **Position Embeddings Caching**
**Current:** Recompute rotary embeddings for each forward pass
**Problem:** Redundant computation for cached sequences
**Solution:** Cache position embeddings for known sequence lengths
**File:** `medusa/model/modeling_llama_kv.py`, `medusa/model/modeling_mistral_kv.py`
**Effort:** Low
**Expected speedup:** 2-5%

---

#### 9. **Quantization-Aware Optimization**
**Current:** Supports `--load-in-8bit` and `--load-in-4bit` but no kernel fusion for quant ops
**Solution:** Use TorchAO or custom kernels for fused quantized operations
**File:** Global
**Effort:** Very High
**Expected speedup:** Varies with quant scheme

---

#### 10. **Compile with `torch.compile()` or TorchScript**
**Current:** Eager execution
**Solution:** Wrap `medusa_generate()` with `torch.compile()`
**File:** `medusa/model/medusa_model.py`
**Effort:** Low (try-it-first)
**Expected speedup:** 10-20% (TorchScript/Triton backend dependent)

---

## Recommended Benchmark Flow

### Baseline (Before Optimizations)
```bash
python llm_judge/gen_model_answer_medusa.py \
  --model-path FasterDecoding/medusa-vicuna-7b-v1.3 \
  --model-id medusa-vicuna-7b-v1.3-baseline \
  --question-begin 0 --question-end 80 \
  --answer-file results/baseline_answers.jsonl
```
**Measure:** Tokens/second, wall-time per question, GPU utilization

---

### After Each Optimization
Re-run the same command, save results with new suffix (e.g., `_opt1_fused_heads.jsonl`)

---

## Validation Checklist

- [ ] Correctness: Output tokens match baseline
- [ ] Quality: LLM judge score unchanged or improved
- [ ] Latency: Wall-time reduced
- [ ] Throughput: Tokens/second increased
- [ ] Memory: Peak GPU memory similar or reduced
- [ ] Stability: Multiple runs give consistent timing

---

## Quick Start: Low-Hanging Fruit

Start here for fastest ROI:

1. **Week 1:** Optimization #2 (Direct GPU buffers) + Optimization #10 (torch.compile)
   - Minimal code changes
   - Easy to measure
   - ~10-20ms total speedup

2. **Week 2:** Optimization #1 (Fused heads)
   - Requires batching or simple kernel fusion
   - 15-25% speedup on generation

3. **Week 3:** Optimization #3 + #4 (Fused indexing + sampling)
   - Cumulative effect
   - ~20-30% total speedup

---

## Tools & Profiling

```bash
# Profile with PyTorch profiler
python -m torch.profiler.timeit medusa_generate(...)

# NVIDIA Nsys for GPU timeline
nsys profile python llm_judge/gen_model_answer_medusa.py ...

# Memory profiling
python -m torch.utils.bottleneck llm_judge/gen_model_answer_medusa.py ...
```

---

## Notes

- All optimizations maintain single-GPU, batch-size-1 constraint
- Prioritize benchmarking before/after each change
- Keep baseline code in a branch for regression testing
- Consider upstream contributions for generic optimizations (#1, #3, #4)
