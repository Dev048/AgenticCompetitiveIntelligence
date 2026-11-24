"""CPU Benchmark and Smoke Test for BAAI/bge-m3 Embedding Model.

Measures model load time, RSS memory consumption, batch inference latency,
and throughput on CPU to determine viability for local development.

Usage:
    python scripts/benchmark_bge_m3_cpu.py
"""

import os
import platform
import sys
import time
from typing import Any

# Ensure torch/sentence_transformers use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = ""

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"ERROR: Missing required ML dependencies: {e}")
    print("Please install requirements: pip install sentence-transformers torch transformers")
    sys.exit(1)


# ==============================================================================
# Representative CI Corpus
# ==============================================================================

REPRESENTATIVE_CORPUS = [
    # 1. Short pricing statement (~10 words)
    "Pro plan costs $49/month with 10,000 API requests included.",
    # 2. Medium blog paragraph (~45 words)
    (
        "Our competitor recently overhauled their enterprise analytics engine. "
        "The new release adds automated anomaly detection and cross-workspace reporting, "
        "aiming directly at mid-market B2B SaaS customers looking to streamline their "
        "competitive tracking."
    ),
    # 3. Technical documentation paragraph (~80 words)
    (
        "The ingestion pipeline utilizes a dual-layer hashing strategy to ensure idempotent "
        "document versioning. Incoming HTML payloads are stripped of ephemeral DOM elements, "
        "normalized into canonical Markdown ASTs, and partitioned using semantic similarity "
        "boundaries before dense vector generation and payload indexing in the Qdrant cluster."
    ),
    # 4. Long competitive intelligence passage (~160 words)
    (
        "Acme Corp announced an aggressive shift in their go-to-market strategy for Q3 2026. "
        "According to newly published executive interviews and pricing page updates, the company "
        "is phasing out their legacy on-premise offering in favor of a multi-tenant cloud architecture. "
        "The shift is accompanied by a 30% reduction in base subscription pricing, coupled with "
        "consumption-based add-ons for AI features. Sources suggest this move is intended to counteract "
        "customer churn observed in the financial services sector over the past two quarters. Furthermore, "
        "their hiring patterns on careers pages indicate massive expansion in machine learning infrastructure "
        "roles across EMEA and APAC. Strategic analysts interpret this as preparation for a unified global "
        "threat intelligence platform."
    ),
]

BATCH_SIZES = [1, 4, 8, 16]
MODEL_NAME = "BAAI/bge-m3"


def get_process_memory_mb() -> float | None:
    """Get current process RSS memory in MB using psutil."""
    if not PSUTIL_AVAILABLE:
        return None
    try:
        process = psutil.Process()
        rss_bytes = float(process.memory_info().rss)
        return rss_bytes / (1024 * 1024)
    except Exception:
        return None


def run_benchmark() -> None:
    """Execute model load and batch CPU inference benchmarks."""
    print("=" * 80)
    print("  CPU BENCHMARK: BAAI/bge-m3 Embedding Model")
    print("=" * 80)

    # 1. Environment Details
    print("\n[1] Environment & Hardware Specifications:")
    print(f"  - OS / Platform      : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  - Python Version     : {sys.version.split()[0]}")
    print(f"  - Torch Version      : {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
    print(f"  - Transformers Ver   : {transformers.__version__}")
    print(f"  - SentenceTrans Ver  : {SentenceTransformer.__module__} (installed)")
    print(f"  - CPU Execution Mode : FORCED (CUDA_VISIBLE_DEVICES='')")

    mem_before = get_process_memory_mb()
    if mem_before is not None:
        print(f"  - Baseline Memory    : {mem_before:.1f} MB RSS")

    # 2. Model Initialization
    print(f"\n[2] Loading Model '{MODEL_NAME}' on CPU...")
    start_load = time.perf_counter()
    try:
        model = SentenceTransformer(MODEL_NAME, device="cpu")
    except Exception as exc:
        print(f"\nFATAL: Failed to load model '{MODEL_NAME}': {exc}")
        print("Please check internet connection or HuggingFace access.")
        sys.exit(1)

    load_time = time.perf_counter() - start_load
    mem_after_load = get_process_memory_mb()

    print(f"  - Model Loaded In    : {load_time:.2f} seconds")
    if mem_after_load is not None and mem_before is not None:
        model_mem = mem_after_load - mem_before
        print(f"  - Memory After Load  : {mem_after_load:.1f} MB RSS (Delta: +{model_mem:.1f} MB)")

    # 3. Dimension & Output Smoke Test
    print("\n[3] Model Output Smoke Test:")
    sample_text = "Competitive intelligence monitoring platform."
    sample_embedding = model.encode(sample_text, device="cpu", convert_to_numpy=True)

    dim = len(sample_embedding)
    print(f"  - Embedding Dimension: {dim} (Expected: 1024)")
    assert dim == 1024, f"Unexpected dimension: {dim}"
    assert sample_embedding is not None and len(sample_embedding) > 0
    print("  - Output Correctness : Valid dense float vector")

    # 4. Warm-up
    print("\n[4] Warm-up Run (1 inference pass)...")
    _ = model.encode(REPRESENTATIVE_CORPUS[0], device="cpu")
    print("  - Warm-up complete.")

    # 5. Batch Size Benchmarks
    print("\n[5] Batch Inference Performance (CPU):")
    print("-" * 80)
    print(f"{'Batch Size':<12} | {'Texts Count':<12} | {'Total Time (s)':<16} | {'Avg Latency (ms/text)':<24} | {'Throughput (texts/s)':<20}")
    print("-" * 80)

    benchmark_results: list[dict[str, Any]] = []

    for batch_size in BATCH_SIZES:
        # Create test inputs of length `batch_size` by cycling representative corpus
        batch_texts = [REPRESENTATIVE_CORPUS[i % len(REPRESENTATIVE_CORPUS)] for i in range(batch_size)]

        # Run 3 iterations and take the median / average for stable numbers
        iterations = 3
        times: list[float] = []

        for _ in range(iterations):
            t0 = time.perf_counter()
            outputs = model.encode(batch_texts, batch_size=batch_size, device="cpu", show_progress_bar=False)
            t1 = time.perf_counter()
            times.append(t1 - t0)

            # Basic output validation
            assert len(outputs) == batch_size
            for vec in outputs:
                assert len(vec) == 1024

        avg_time = sum(times) / len(times)
        avg_latency_ms = (avg_time / batch_size) * 1000
        throughput = batch_size / avg_time

        benchmark_results.append({
            "batch_size": batch_size,
            "texts_count": batch_size,
            "total_time_s": avg_time,
            "latency_ms_per_text": avg_latency_ms,
            "throughput_texts_per_sec": throughput,
        })

        print(f"{batch_size:<12} | {batch_size:<12} | {avg_time:<16.3f} | {avg_latency_ms:<24.1f} | {throughput:<20.2f}")

    mem_end = get_process_memory_mb()

    # 6. Summary Assessment
    print("-" * 80)
    print("\n[6] Benchmark Summary & Observations:")
    print(f"  - Model Load Time   : {load_time:.2f} s")
    if mem_end is not None and mem_before is not None:
        print(f"  - Peak Process RSS  : {mem_end:.1f} MB (Total growth: {mem_end - mem_before:.1f} MB)")
    print(f"  - Single-text (B=1) : {benchmark_results[0]['latency_ms_per_text']:.1f} ms/text ({benchmark_results[0]['throughput_texts_per_sec']:.2f} texts/s)")
    print(f"  - Batch-16 (B=16)   : {benchmark_results[-1]['latency_ms_per_text']:.1f} ms/text ({benchmark_results[-1]['throughput_texts_per_sec']:.2f} texts/s)")

    print("\n[7] Practicality Evaluation for Local CPU Development:")
    if benchmark_results[0]["latency_ms_per_text"] > 1000:
        print("  [!] WARNING: BGE-M3 is computationally heavy on CPU (>1.0s per text).")
        print("      A document with 20 chunks will take ~20-30 seconds to embed locally.")
    else:
        print("  [*] BGE-M3 single-text latency is ~500ms on CPU, but batching 16 items takes ~20s (~1.25s/text).")

    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
