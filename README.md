<div align="center"><h1>&nbsp;Medusa (PDC Optimized): Parallelizing LLM Inference via Fused Kernel Speculative Decoding</h1></div>

<p align="center">
  <b>A Parallel and Distributed Computing (PDC) Project by Sagar Mehmood & Ali Aan (FAST NUCES)</b>
</p>

---

## 🚀 Overview

This repository is a heavily optimized fork of the original [Medusa Framework](https://github.com/FasterDecoding/Medusa). 

While the original Medusa framework successfully breaks the sequential bottleneck of LLM autoregressive decoding by evaluating multiple "speculative heads" in parallel, its implementation suffered from severe **Kernel Launch Overhead** and **Memory Fragmentation** on the GPU.

In this project, we applied rigorous Parallel and Distributed Computing (PDC) principles—specifically **SIMD Vectorization (Kernel Fusion)** and **Memory Locality Optimization**—to achieve up to a **21% additional speedup** over the original Medusa implementation on compute-bound architectures. We also upgraded the codebase to guarantee compatibility with `transformers v4.40+` and `bitsandbytes` 8-bit quantization.

---

## ⚡ Key Optimizations

### 1. Data Parallelism via Kernel Fusion (SIMD)
*   **The Problem:** The original code evaluated $K$ Medusa heads sequentially in a Python `for`-loop. This created immense CPU-dispatch overhead, starving the GPU's Tensor Cores.
*   **The Solution:** We batched the weights of all individual `ResBlocks` and `Linear` layers during model initialization. The loop was replaced with a single Batched Matrix Multiplication (`torch.bmm`), allowing the GPU to evaluate all speculative heads simultaneously in one massive SIMD instruction.

### 2. Memory Locality & Fused Indexing
*   **The Problem:** During the speculative generation loop, the model heavily relied on dynamic `torch.cat` and array slicing to map tree candidates back to flattened cartesian lists, thrashing the GPU cache.
*   **The Solution:** We moved this computation to the initialization phase. By pre-computing a static `mapped_retrieve_indices` pointer array on the GPU, we reduced the dynamic array manipulation down to a single, highly contiguous advanced memory fetch.

### 3. Direct GPU Buffer Initialization
*   **The Problem:** Large structural arrays (like the $64 \times 64$ tree attention mask) were being initialized on CPU RAM and then cloned to the GPU via the PCI-E bus.
*   **The Solution:** We refactored all structural buffers to initialize directly on the GPU (`device=device`), reducing host-to-device synchronization overhead.

---

## 📊 Benchmarking & Bottleneck Analysis (Amdahl's Law)

We conducted a cross-architecture analysis to evaluate our optimizations using the `Vicuna-7B` model in 8-bit precision.

| Target Tokens | RTX 4050 (Compute-Bound) | Nvidia T4 (Memory-Bound) |
| :--- | :--- | :--- |
| **50 Tokens** | **1.59x** Faster | **2.44x** Faster |

### Why the difference? (Amdahl's Law)
*   **RTX 4050 (Ada Lovelace):** This modern GPU has blistering compute speeds. Removing the kernel launch overhead via Kernel Fusion fed the GPU perfectly, yielding a **21.4% relative improvement** over the unoptimized Medusa code (1.31x $\rightarrow$ 1.59x).
*   **Nvidia T4 (Turing):** This older GPU yielded only a **3% - 5% improvement** from our optimizations. Why? The T4 was completely bottlenecked by the memory bandwidth required to de-quantize the 8-bit `bitsandbytes` weights on the fly. Because 95% of the execution time was spent waiting on memory for the base model, making the Medusa heads infinitely faster only affected 5% of the pipeline. **This empirically proves that low-level parallel optimizations are strictly bounded by hardware memory ceilings.**

---

## 🛠️ Usage

### Installation
Ensure you have the required dependencies:
```bash
pip install torch transformers accelerate bitsandbytes sentencepiece
```

### Running the Benchmark
We provide a standardized benchmark script that tests the exact speedup of Medusa against a standard autoregressive baseline. It automatically caps the generation exactly at the target tokens for a perfectly fair, greedy (temperature=0.0) comparison.

```bash
python benchmark_medusa.py --load-in-8bit --max-new-tokens 50
```

*Note: The model will automatically download the `FasterDecoding/medusa-vicuna-7b-v1.3` weights from the HuggingFace Hub on the first run.*

---

## 🔗 Original Acknowledgements
This project is built upon the incredible foundational work by the FasterDecoding team. 
* Original [Medusa GitHub](https://github.com/FasterDecoding/Medusa)
* Original [Medusa Paper](https://arxiv.org/abs/2401.10774)